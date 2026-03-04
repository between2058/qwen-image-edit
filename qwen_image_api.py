import os
import io
import gc
import uuid
import shutil
import asyncio
import base64
import random
import torch
import tempfile
import logging
import logging.handlers
import datetime
import numpy as np
from typing import Dict, List, Literal, Optional
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
import traceback

# --- Diffusers Imports ---
from diffusers import QwenImagePipeline, QwenImageEditPlusPipeline


# =============================================================================
# Logging Setup
# =============================================================================
LOG_DIR = "/app/logs"
os.makedirs(LOG_DIR, exist_ok=True)

_TZ_TAIPEI = datetime.timezone(datetime.timedelta(hours=8))

class _TaipeiFormatter(logging.Formatter):
    """Formatter that always outputs UTC+8 (Taiwan) time, regardless of system TZ."""
    def formatTime(self, record, datefmt=None):
        dt = datetime.datetime.fromtimestamp(record.created, tz=_TZ_TAIPEI)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S")

_log_fmt = _TaipeiFormatter(
    "%(asctime)s [%(levelname)-8s] %(message)s"
)

def _make_rotating_handler(filename: str) -> logging.Handler:
    h = logging.handlers.TimedRotatingFileHandler(
        os.path.join(LOG_DIR, filename),
        when="midnight",
        backupCount=14,
        encoding="utf-8",
    )
    h.setFormatter(_log_fmt)
    return h

# App logger — console + rotating file
logger = logging.getLogger("qwen_api")
logger.setLevel(logging.DEBUG)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_log_fmt)
logger.addHandler(_console_handler)
logger.addHandler(_make_rotating_handler("app.log"))
logger.propagate = False

# Uvicorn access log — filter /health, also write to file
class _HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "GET /health" not in record.getMessage()

_uv_access = logging.getLogger("uvicorn.access")
_uv_access.addFilter(_HealthCheckFilter())
_uv_access.addHandler(_make_rotating_handler("access.log"))

# Uvicorn startup / error log → file
logging.getLogger("uvicorn").addHandler(_make_rotating_handler("uvicorn.log"))


# --- Response Schemas ---

class HealthResponse(BaseModel):
    status: str
    device: str = Field(description="'cuda' 或 'cpu'")
    gpu_busy: bool = Field(description="GPU Lock 是否鎖定中（是否有推論正在執行）")

class Text2ImgResponse(BaseModel):
    status: str
    request_id: str
    urls: List[str] = Field(description="生成圖片的下載路徑列表，與 seeds 一一對應")
    seeds: List[int] = Field(description="每張圖片實際使用的 RNG seed，與 urls 一一對應")

class EditResponse(BaseModel):
    status: str
    request_id: str
    input_url: str = Field(description="上傳的原始圖片下載路徑")
    result_urls: List[str] = Field(description="編輯結果圖片路徑列表，與 seeds 一一對應")
    seeds: List[int] = Field(description="每張結果使用的 RNG seed，與 result_urls 一一對應")

class EditMultiResponse(BaseModel):
    status: str
    request_id: str
    count: int = Field(description="上傳的圖片數量")
    inputs: List[str] = Field(description="上傳的原始圖片路徑列表")
    results: List[str] = Field(description="編輯結果路徑列表（目前固定為一張拼接大圖，故 len=1）")

class AngleResponse(BaseModel):
    status: str
    request_id: str
    input_url: str = Field(description="上傳的原始圖片下載路徑")
    results: Dict[str, str] = Field(
        description=(
            "生成的視角圖片 URL map。"
            "mode=custom 時 key 為 'custom'；"
            "mode=multi 時 key 為 'right', 'back', 'left'"
        )
    )

# --- 1. 初始化設定 ---
app = FastAPI(title="Qwen All-in-One API (Text2Img, Edit, Angle)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 建立暫存目錄
OUTPUT_DIR = tempfile.mkdtemp()
logger.info(f"Output directory: {OUTPUT_DIR}")
logger.info(f"Log directory: {LOG_DIR}")

# 硬體設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MAX_SEED = 2147483647

# 全局 GPU 鎖 (確保一次只跑一個模型)
gpu_lock = asyncio.Lock()

# --- 2. 輔助函式與資料結構 ---

def log_gpu_memory(label: str):
    """記錄目前 GPU 記憶體用量，方便事後追查洩漏。"""
    if DEVICE != "cuda":
        return
    alloc = torch.cuda.memory_allocated() / 1024**3
    rsvd  = torch.cuda.memory_reserved()  / 1024**3
    logger.info(f"GPU memory [{label}]: allocated={alloc:.2f} GB  reserved={rsvd:.2f} GB")

def flush_gpu():
    """強制清理 GPU 記憶體"""
    gc.collect()
    torch.cuda.empty_cache()
    log_gpu_memory("after flush")

def classify_exception(e: Exception) -> tuple[int, str, str]:
    """
    將例外分類為對應的 HTTP 狀態碼、錯誤代碼與說明。
    回傳 (status_code, error_code, human_readable_message)

    HTTP status code 語義：
      503 GPU_OOM           — GPU 記憶體不足，稍後可重試（附 Retry-After header）
      503 MODEL_UNAVAILABLE — 模型載入失敗，稍後可重試
      507 DISK_FULL         — 磁碟空間不足，需人工介入
      500 INFERENCE_ERROR   — 未知推論錯誤，不可自動重試
    """
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return 503, "GPU_OOM", "GPU out of memory. Free some VRAM and retry."
    if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
        return 503, "GPU_OOM", "GPU out of memory. Free some VRAM and retry."
    if isinstance(e, RuntimeError) and (
        "Model loading failed" in str(e) or "class not available" in str(e)
    ):
        return 503, "MODEL_UNAVAILABLE", str(e)
    if isinstance(e, OSError) and getattr(e, "errno", None) == 28:
        return 507, "DISK_FULL", "Server disk is full. Contact administrator."
    return 500, "INFERENCE_ERROR", str(e)

# Swagger 文件用的 responses 描述（套用在所有推論 endpoint）
GPU_ERROR_RESPONSES = {
    503: {
        "description": "GPU OOM 或模型載入失敗，稍後可重試（含 `Retry-After: 30` header）",
        "content": {
            "application/json": {
                "examples": {
                    "GPU_OOM": {
                        "summary": "GPU out of memory",
                        "value": {"detail": {"error_code": "GPU_OOM", "message": "GPU out of memory. Free some VRAM and retry."}},
                    },
                    "MODEL_UNAVAILABLE": {
                        "summary": "Model failed to load",
                        "value": {"detail": {"error_code": "MODEL_UNAVAILABLE", "message": "Model loading failed: <reason>"}},
                    },
                }
            }
        },
    },
    507: {
        "description": "Server 磁碟空間不足，需人工介入",
        "content": {
            "application/json": {
                "example": {"detail": {"error_code": "DISK_FULL", "message": "Server disk is full. Contact administrator."}}
            }
        },
    },
    500: {
        "description": "未知推論錯誤，不可自動重試",
        "content": {
            "application/json": {
                "example": {"detail": {"error_code": "INFERENCE_ERROR", "message": "<exception message>"}}
            }
        },
    },
}

def save_image(image: Image.Image, folder: str, filename: str) -> str:
    path = os.path.join(folder, filename)
    image.save(path)
    return path

# --- Angle 相關映射表 ---
AZIMUTH_MAP = {
    0: "front view", 45: "front-right quarter view", 90: "right side view",
    135: "back-right quarter view", 180: "back view", 225: "back-left quarter view",
    270: "left side view", 315: "front-left quarter view"
}
ELEVATION_MAP = {-30: "low-angle shot", 0: "eye-level shot", 30: "elevated shot", 60: "high-angle shot"}
DISTANCE_MAP = {0.6: "close-up", 1.0: "medium shot", 1.8: "wide shot"}

def snap_to_nearest(value: float, options: List[float]) -> float:
    return min(options, key=lambda x: abs(x - value))

def build_angle_prompt(azimuth: float, elevation: float = 0.0, distance: float = 1.0) -> str:
    az_snap = snap_to_nearest(azimuth, list(AZIMUTH_MAP.keys()))
    el_snap = snap_to_nearest(elevation, list(ELEVATION_MAP.keys()))
    dist_snap = snap_to_nearest(distance, list(DISTANCE_MAP.keys()))
    return f"<sks> {AZIMUTH_MAP[az_snap]} {ELEVATION_MAP[el_snap]} {DISTANCE_MAP[dist_snap]}"

SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}

def _validate_image_upload(file: UploadFile, field_name: str = "file"):
    ext = os.path.splitext(file.filename or '')[1].lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported image type '{ext}' for field '{field_name}'. Must be one of: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}",
        )

# Text2Image 的比例設定
ASPECT_RATIOS = {
    "1:1": (1328, 1328), "16:9": (1664, 928), "9:16": (928, 1664),
    "4:3": (1472, 1104), "3:4": (1104, 1472), "3:2": (1584, 1056), "2:3": (1056, 1584),
}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "ok",
        "device": DEVICE,
        "gpu_busy": gpu_lock.locked(),
    }


# ==========================================
# MODEL 1: Text-to-Image (Text -> Image)
# ==========================================
@app.post("/text2img", response_model=Text2ImgResponse, responses=GPU_ERROR_RESPONSES)
async def text_to_image(
    prompt: str = Form(..., description="生成提示詞"),
    negative_prompt: str = Form("low quality, bad anatomy, blurry, distorted", description="負向提示詞"),
    aspect_ratio: Literal["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"] = Form("16:9", description="輸出圖片長寬比"),
    num_steps: int = Form(50, ge=1, le=100, description="擴散步數"),
    cfg_scale: float = Form(4.0, ge=0.0, le=20.0, description="Classifier-free guidance scale"),
    seed: Optional[int] = Form(None, description="Accepted but ignored — each sample uses its own independent random seed"),
    num_samples: int = Form(1, ge=1, le=8, description="一次生成的圖片數量（1–8）"),
):
    """
    [Model 1] Qwen-Image-2512
    輸入: 純文字 Prompt
    輸出: 生成的圖片
    """
    request_id = str(uuid.uuid4())
    req_dir = os.path.join(OUTPUT_DIR, request_id)
    os.makedirs(req_dir, exist_ok=True)

    width, height = ASPECT_RATIOS[aspect_ratio]
    logger.info(f"[Text2Img] id={request_id} prompt={prompt[:60]!r} size={width}x{height} samples={num_samples}")

    async with gpu_lock:
        def run_inference():
            pipe = None
            try:
                log_gpu_memory("before model load")
                logger.info("Loading Qwen-Image-2512...")
                pipe = QwenImagePipeline.from_pretrained(
                    "Qwen/Qwen-Image-2512",
                    torch_dtype=DTYPE
                ).to(DEVICE)
                log_gpu_memory("model loaded")

                seeds = [random.randint(0, MAX_SEED) for _ in range(num_samples)]
                paths = []
                for i, seed_i in enumerate(seeds):
                    generator = torch.Generator(device=DEVICE).manual_seed(seed_i)
                    logger.info(f"Generating image {i+1}/{num_samples} (seed={seed_i})")
                    image = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=num_steps,
                        true_cfg_scale=cfg_scale,
                        generator=generator
                    ).images[0]
                    path = save_image(image, req_dir, f"output_{i}.png")
                    paths.append(path)

                return paths, seeds
            finally:
                if pipe: del pipe
                flush_gpu()

        try:
            output_paths, used_seeds = await run_in_threadpool(run_inference)
            urls = [f"/download/{request_id}/output_{i}.png" for i in range(len(output_paths))]
            logger.info(f"[Text2Img] id={request_id} done, {len(urls)} image(s)")
            return {
                "status": "success",
                "request_id": request_id,
                "urls": urls,
                "seeds": used_seeds
            }
        except Exception as e:
            status, error_code, message = classify_exception(e)
            logger.error(f"[{error_code}] id={request_id} {type(e).__name__}: {e}")
            traceback.print_exc()
            if error_code == "GPU_OOM":
                flush_gpu()
            headers = {"Retry-After": "30"} if status == 503 else {}
            raise HTTPException(
                status_code=status,
                detail={"error_code": error_code, "message": message},
                headers=headers,
            )


# ==========================================
# MODEL 2: Edit (Image + Prompt -> Image)
# ==========================================
@app.post("/edit", response_model=EditResponse, responses=GPU_ERROR_RESPONSES)
async def edit_image(
    file: UploadFile = File(...),
    prompt: str = Form(..., description="編輯指令"),
    steps: int = Form(40, ge=1, le=100, description="擴散步數"),
    cfg_scale: float = Form(4.0, ge=0.0, le=20.0, description="Guidance scale"),
    seed: int = Form(42, description="Accepted but ignored — each sample uses its own independent random seed"),
    num_samples: int = Form(1, ge=1, le=6, description="生成結果數量（1–6），每張使用獨立隨機 seed")
):
    """
    [Model 2] Qwen-Image-Edit-2511 (Base Model)
    輸入: 圖片 + Prompt
    功能: 根據文字指令修改圖片內容
    """
    _validate_image_upload(file, "file")

    request_id = str(uuid.uuid4())
    req_dir = os.path.join(OUTPUT_DIR, request_id)
    os.makedirs(req_dir, exist_ok=True)

    input_path = os.path.join(req_dir, "input.png")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info(f"[Edit] id={request_id} prompt={prompt!r} samples={num_samples}")

    async with gpu_lock:
        def run_inference():
            pipe = None
            try:
                log_gpu_memory("before model load")
                logger.info("Loading Qwen-Image-Edit-2511...")
                pipe = QwenImageEditPlusPipeline.from_pretrained(
                    "Qwen/Qwen-Image-Edit-2511",
                    torch_dtype=DTYPE
                ).to(DEVICE)
                log_gpu_memory("model loaded")

                img_obj = Image.open(input_path).convert("RGB")
                seeds = [random.randint(0, MAX_SEED) for _ in range(num_samples)]
                paths = []
                for i, seed_i in enumerate(seeds):
                    generator = torch.Generator(device=DEVICE).manual_seed(seed_i)
                    logger.info(f"Editing image {i+1}/{num_samples} (seed={seed_i})")
                    output = pipe(
                        image=[img_obj],
                        prompt=prompt,
                        num_inference_steps=steps,
                        true_cfg_scale=cfg_scale,
                        generator=generator,
                        num_images_per_prompt=1
                    ).images[0]
                    path = save_image(output, req_dir, f"result_{i}.png")
                    paths.append(path)

                return paths, seeds
            finally:
                if pipe: del pipe
                flush_gpu()

        try:
            output_paths, used_seeds = await run_in_threadpool(run_inference)
            result_urls = [f"/download/{request_id}/result_{i}.png" for i in range(len(output_paths))]
            logger.info(f"[Edit] id={request_id} done, {len(result_urls)} image(s)")
            return {
                "status": "success",
                "request_id": request_id,
                "input_url": f"/download/{request_id}/input.png",
                "result_urls": result_urls,
                "seeds": used_seeds
            }
        except Exception as e:
            status, error_code, message = classify_exception(e)
            logger.error(f"[{error_code}] id={request_id} {type(e).__name__}: {e}")
            traceback.print_exc()
            if error_code == "GPU_OOM":
                flush_gpu()
            headers = {"Retry-After": "30"} if status == 503 else {}
            raise HTTPException(
                status_code=status,
                detail={"error_code": error_code, "message": message},
                headers=headers,
            )

# ==========================================
# MODEL 2: Edit Multi (Multiple Images + 1 Prompt)
# ==========================================
@app.post("/edit-multi", response_model=EditMultiResponse, responses=GPU_ERROR_RESPONSES)
async def edit_multi_images(
    files: List[UploadFile] = File(..., description="上傳多張圖片"),
    prompt: str = Form(..., description="編輯指令"),
    steps: int = Form(40, ge=1, le=100, description="擴散步數"),
    cfg_scale: float = Form(4.0, ge=0.0, le=20.0, description="Guidance scale"),
    seed: int = Form(42, description="RNG seed")
):
    """
    [Model 2]
    輸入: 多張圖片 + Prompt
    功能: 根據文字指令修改圖片內容
    """
    for i, f in enumerate(files):
        _validate_image_upload(f, f"files[{i}]")

    request_id = str(uuid.uuid4())
    req_dir = os.path.join(OUTPUT_DIR, request_id)
    os.makedirs(req_dir, exist_ok=True)

    logger.info(f"[Edit-Multi] id={request_id} images={len(files)} prompt={prompt!r}")

    input_images_pil = []
    input_urls = []

    for i, file in enumerate(files):
        filename = f"input_{i}.png"
        file_path = os.path.join(req_dir, filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        img = Image.open(file_path).convert("RGB")
        input_images_pil.append(img)
        input_urls.append(f"/download/{request_id}/{filename}")

    async with gpu_lock:
        def run_inference():
            pipe = None
            try:
                log_gpu_memory("before model load")
                logger.info("Loading Qwen-Image-Edit-2511...")
                pipe = QwenImageEditPlusPipeline.from_pretrained(
                    "Qwen/Qwen-Image-Edit-2511",
                    torch_dtype=DTYPE
                ).to(DEVICE)
                log_gpu_memory("model loaded")

                generator = torch.Generator(device=DEVICE).manual_seed(seed)

                logger.info("Editing stitched image...")
                output_stitched = pipe(
                    image=input_images_pil,
                    prompt=prompt,
                    num_inference_steps=steps,
                    true_cfg_scale=cfg_scale,
                    generator=generator,
                    num_images_per_prompt=1
                ).images[0]

                output_filename = "result_stitched.png"
                save_image(output_stitched, req_dir, output_filename)
                return [f"/download/{request_id}/{output_filename}"]

            finally:
                if pipe: del pipe
                flush_gpu()

        try:
            result_urls = await run_in_threadpool(run_inference)
            logger.info(f"[Edit-Multi] id={request_id} done")
            return {
                "status": "success",
                "request_id": request_id,
                "count": len(files),
                "inputs": input_urls,
                "results": result_urls
            }
        except Exception as e:
            status, error_code, message = classify_exception(e)
            logger.error(f"[{error_code}] id={request_id} {type(e).__name__}: {e}")
            traceback.print_exc()
            if error_code == "GPU_OOM":
                flush_gpu()
            headers = {"Retry-After": "30"} if status == 503 else {}
            raise HTTPException(
                status_code=status,
                detail={"error_code": error_code, "message": message},
                headers=headers,
            )


# ==========================================
# MODEL 3: Angle (Image + Angle -> Image)
# ==========================================
@app.post("/angle", response_model=AngleResponse, responses=GPU_ERROR_RESPONSES)
async def change_angle(
    file: UploadFile = File(...),
    mode: Literal["custom", "multi"] = Form("custom", description="'custom' for single angle, 'multi' for 3 views (right/back/left)"),
    azimuth: float = Form(0, ge=0.0, le=360.0, description="Horizontal rotation in degrees (0–360); snapped to nearest supported value"),
    elevation: float = Form(0, ge=-30.0, le=60.0, description="Vertical angle in degrees (-30 to 60); snapped to nearest supported value"),
    distance: float = Form(1.0, ge=0.6, le=1.8, description="Camera distance (0.6, 1.0, or 1.8); snapped to nearest supported value"),
):
    """
    [Model 3] Qwen-Image-Edit-2511 + Angle LoRAs
    輸入: 圖片 + 角度參數
    功能: 旋轉物體視角 (需載入 LoRA)
    """
    _validate_image_upload(file, "file")

    request_id = str(uuid.uuid4())
    req_dir = os.path.join(OUTPUT_DIR, request_id)
    os.makedirs(req_dir, exist_ok=True)

    input_path = os.path.join(req_dir, "input.png")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info(f"[Angle] id={request_id} mode={mode}")

    prompts_map = {}
    if mode == "multi":
        prompts_map["right"] = build_angle_prompt(90, 0, 1.0)
        prompts_map["back"] = build_angle_prompt(180, 0, 1.0)
        prompts_map["left"] = build_angle_prompt(270, 0, 1.0)
    else:
        prompts_map["custom"] = build_angle_prompt(azimuth, elevation, distance)

    async with gpu_lock:
        def run_inference():
            pipe = None
            try:
                log_gpu_memory("before model load")
                logger.info("Loading Qwen-Image-Edit-2511 with LoRAs...")
                pipe = QwenImageEditPlusPipeline.from_pretrained(
                    "Qwen/Qwen-Image-Edit-2511",
                    torch_dtype=DTYPE
                ).to(DEVICE)

                logger.info("Loading adapters (Lightning + Angle)...")
                pipe.load_lora_weights(
                    "lightx2v/Qwen-Image-Edit-2511-Lightning",
                    weight_name="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
                    adapter_name="lightning"
                )
                pipe.load_lora_weights(
                    "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
                    weight_name="qwen-image-edit-2511-multiple-angles-lora.safetensors",
                    adapter_name="angles"
                )
                pipe.set_adapters(["lightning", "angles"], adapter_weights=[1.0, 1.0])
                log_gpu_memory("model + LoRA loaded")

                img_obj = Image.open(input_path).convert("RGB").resize((1024, 1024), Image.LANCZOS)
                results = {}
                seed = random.randint(0, MAX_SEED)

                for key, prompt_str in prompts_map.items():
                    logger.info(f"Generating view: {key} ({prompt_str})")
                    generator = torch.Generator(device=DEVICE).manual_seed(seed)

                    out_img = pipe(
                        image=[img_obj],
                        prompt=prompt_str,
                        height=1024,
                        width=1024,
                        num_inference_steps=4,
                        generator=generator,
                        guidance_scale=1.0,
                        num_images_per_prompt=1,
                    ).images[0]

                    filename = f"output_{key}.png"
                    save_image(out_img, req_dir, filename)
                    results[key] = f"/download/{request_id}/{filename}"

                return results

            finally:
                if pipe: del pipe
                flush_gpu()

        try:
            results = await run_in_threadpool(run_inference)
            logger.info(f"[Angle] id={request_id} done, views={list(results.keys())}")
            return {
                "status": "success",
                "request_id": request_id,
                "input_url": f"/download/{request_id}/input.png",
                "results": results
            }
        except Exception as e:
            status, error_code, message = classify_exception(e)
            logger.error(f"[{error_code}] id={request_id} {type(e).__name__}: {e}")
            traceback.print_exc()
            if error_code == "GPU_OOM":
                flush_gpu()
            headers = {"Retry-After": "30"} if status == 503 else {}
            raise HTTPException(
                status_code=status,
                detail={"error_code": error_code, "message": message},
                headers=headers,
            )


# --- Common: Download & Cleanup ---
@app.get("/download/{request_id}/{file_name}")
async def download_file(request_id: str, file_name: str):
    file_path = os.path.join(OUTPUT_DIR, request_id, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@app.on_event("shutdown")
async def cleanup():
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    logger.info("Temporary files cleaned up")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8190)
