# Qwen Image API — 部署指南

## 系統需求

| 項目 | 規格 |
|------|------|
| GPU | NVIDIA RTX Pro 6000（Blackwell，sm_120）|
| CUDA Driver | ≥ 12.8 |
| 作業系統 | Ubuntu 22.04 |
| Docker | ≥ 24.0 |
| Docker Compose | ≥ 2.20（plugin 版，指令為 `docker compose`）|
| NVIDIA Container Toolkit | 已安裝並設定為 Docker runtime |
| 網路 | 透過 `http://proxy.intra:80` 對外連線 |
| 磁碟空間 | `/pegaai` 至少 **60 GB**（模型快取用）|

---

## 步驟一：確認 Docker GPU 支援

```bash
# 確認 Docker 可存取 GPU
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

正常輸出應顯示 RTX Pro 6000 的 GPU 資訊。若出現錯誤，請先安裝 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)。

---

## 步驟二：準備 HuggingFace 快取目錄

模型（Qwen-Image-2512、Qwen-Image-Edit-2511、Lightning LoRA、Angles LoRA）首次請求時自動下載，統一存放在 host 的 `/pegaai/model_team/huggingface_cache`，掛載至容器內的 `/notebooks/model_team/huggingface_cache`。

```bash
sudo mkdir -p /pegaai/model_team/huggingface_cache
sudo chmod -R 777 /pegaai/model_team/huggingface_cache
```

> **提示：** 如果模型已預先下載，放到此目錄即可，容器啟動後直接使用，不需重新下載。

---

## 步驟三：取得專案原始碼

```bash
# 複製專案至部署機器
git clone <your-repo-url> /opt/qwen-image-api
cd /opt/qwen-image-api
```

確認目錄結構如下：

```
/opt/qwen-image-api/
├── qwen_image_api.py
├── requirements-api.txt
├── Dockerfile
├── docker-compose.yml
└── docker-build.sh
```

---

## 步驟四：賦予腳本執行權限

```bash
chmod +x docker-build.sh
```

---

## 步驟五：建置 Docker Image

### 方案 A — 標準 build（建議首次使用）

編譯多種 CUDA 架構（8.0、8.6、8.9、9.0、10.0、12.0），image 可跨機器使用：

```bash
./docker-build.sh build
```

### 方案 B — 快速 build（只針對 RTX Pro 6000）

只編譯 sm_120，build 速度較快，但 image 只能在 Blackwell GPU 上執行：

```bash
./docker-build.sh build-fast
```

> Build 過程會從 PyTorch 官方 index 下載 ~2 GB wheel，並從 GitHub 安裝 diffusers，全程需透過 proxy。
> 整體 build 時間約 **10–20 分鐘**（視網路速度與 CPU 核心數）。

Build 完成後確認 image 存在：

```bash
docker images | grep qwen-image-api
# qwen-image-api   latest   ...
```

---

## 步驟六：啟動服務

```bash
./docker-build.sh up
```

或直接用 docker compose：

```bash
docker compose up -d
```

服務啟動後輸出：

```
🚀 Container started.
   API docs : http://localhost:8190/docs
   Health   : http://localhost:8190/health
```

---

## 步驟七：確認服務健康狀態

```bash
./docker-build.sh health
```

預期回應（GPU 空閒中）：

```json
{
    "status": "ok",
    "device": "cuda",
    "gpu_busy": false
}
```

> 容器剛啟動時 `device` 就應為 `cuda`。若顯示 `cpu`，代表 GPU 未正確掛載，請回到步驟一排查。

---

## 步驟八：功能驗證

### 測試 Text-to-Image

```bash
./docker-build.sh test-text2img
```

預期回應：

```json
{
    "status": "success",
    "request_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "urls": ["/download/<request_id>/output_0.png"],
    "seeds": [123456789]
}
```

### 測試 Image Edit

準備一張測試圖片：

```bash
cp /path/to/any/image.png ./test.png
./docker-build.sh test-edit
```

### 下載結果圖片

```bash
# 將上方回應的 url 代入
curl -O http://localhost:8190/download/<request_id>/output_0.png
```

---

## API 端點速覽

服務啟動後可開啟 **Swagger UI** 互動測試：

```
http://<host-ip>:8190/docs
```

| 端點 | 方法 | 功能 | 模型 |
|------|------|------|------|
| `/health` | GET | 服務狀態與 GPU 鎖狀態 | — |
| `/text2img` | POST | 文字生成圖片 | Qwen-Image-2512 |
| `/edit` | POST | 單張圖片 + prompt 編輯 | Qwen-Image-Edit-2511 |
| `/edit-multi` | POST | 多張圖片 + prompt 編輯 | Qwen-Image-Edit-2511 |
| `/angle` | POST | 視角轉換（單角度或三視圖）| Qwen-Image-Edit-2511 + LoRA |
| `/download/{request_id}/{file}` | GET | 下載生成結果 | — |

---

## 常用維運指令

```bash
# 查看即時 log
./docker-build.sh logs

# 進入容器 shell（除錯用）
./docker-build.sh shell

# 停止服務
./docker-build.sh down

# 重新 build 並重啟（程式碼更新後）
docker compose up -d --build

# 查看容器 GPU 使用量
docker compose exec qwen-image-api nvidia-smi

# 移除 image 與容器（完整清理）
./docker-build.sh clean
```

---

## 模型首次下載說明

各端點在**第一次被呼叫時**才下載對應模型，不會在容器啟動時一次全部下載：

| 端點 | 下載的模型 | 約佔空間 |
|------|-----------|---------|
| `/text2img` | Qwen-Image-2512 | ~20 GB |
| `/edit`, `/edit-multi` | Qwen-Image-Edit-2511 | ~20 GB |
| `/angle` | Qwen-Image-Edit-2511 + Lightning LoRA + Angles LoRA | ~21 GB |

下載完成後存放在 `/pegaai/model_team/huggingface_cache`，容器重啟或重建後**不需重新下載**。

---

## 常見問題排查

### `device: cpu` — GPU 未掛載

```bash
# 確認 docker compose 有 GPU 設定
grep -A5 "capabilities" docker-compose.yml

# 確認 NVIDIA Container Toolkit 正常
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

### `503 GPU_OOM` — 顯存不足

同一時間只有一個推論任務可執行（由 `asyncio.Lock` 保護）。若同時有其他程序佔用 GPU 記憶體，請先釋放：

```bash
# 查看 GPU 記憶體使用
docker compose exec qwen-image-api nvidia-smi

# 若有其他容器佔用，先停止它們
docker ps
```

### HuggingFace 下載失敗（proxy 問題）

確認 proxy 在容器內生效：

```bash
docker compose exec qwen-image-api curl -v https://huggingface.co
```

若連線失敗，檢查 `/pegaai` 目錄的掛載與 proxy 設定是否正確。

### 查看詳細錯誤 log

```bash
./docker-build.sh logs
# 或
docker compose logs --tail=100 qwen-image-api
```

---

## 指定使用特定 GPU

機器上有多張 GPU 時，可在 `docker-compose.yml` 的 `device_ids` 指定要使用哪一張。

### 先確認 GPU 編號

```bash
nvidia-smi -L
# GPU 0: NVIDIA RTX Pro 6000 (UUID: ...)
# GPU 1: NVIDIA RTX Pro 6000 (UUID: ...)
```

`device_ids` 對應的就是這裡顯示的 index（從 0 起算）。

### 修改 docker-compose.yml

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ["1"]    # ← 改這裡，"0" 第一張、"1" 第二張
          capabilities: [gpu]
```

常見設定範例：

| `device_ids` 值 | 說明 |
|-----------------|------|
| `["0"]` | 只使用第一張 GPU |
| `["1"]` | 只使用第二張 GPU（目前設定）|
| `["0", "1"]` | 同時掛載兩張（服務本身仍一次只用一張）|

> **注意：** `device_ids` 和 `count` 不能同時存在，設定其中一個即可。

修改後重啟服務：

```bash
docker compose up -d
```

進入容器確認 GPU 是否正確：

```bash
docker compose exec qwen-image-api nvidia-smi
# 應只顯示指定的那張 GPU
```

---

## 目錄結構說明

```
Host                                    Container
────────────────────────────────────────────────────────────
/pegaai/                           →    /notebooks/
  model_team/
    huggingface_cache/             →    /notebooks/model_team/huggingface_cache/
      models--Qwen--Qwen-Image-2512/
      models--Qwen--Qwen-Image-Edit-2511/
      models--lightx2v--*/
      models--fal--*/

/opt/qwen-image-api/ (build context)
  qwen_image_api.py               →    /app/qwen_image_api.py
  requirements-api.txt            →    /app/requirements-api.txt (build only)
```
