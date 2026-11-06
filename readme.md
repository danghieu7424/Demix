# Tải mô hình onnx.
```powershell
# Chạy trong thư mục gốc của dự án Rust (tải trực tiếp vào thư mục models/)
mkdir -p models
curl.exe -L -o models/UVR-MDX-NET-Inst_HQ_3.onnx https://huggingface.co/seanghay/uvr_models/resolve/main/UVR-MDX-NET-Inst_HQ_3.onnx
curl.exe -L -o models/mdx_c_configs.json https://huggingface.co/audio-separator/models/resolve/main/mdx_c_configs.json

```