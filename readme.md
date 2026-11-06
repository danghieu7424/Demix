# Tải mô hình onnx.
```powershell
# Chạy trong thư mục gốc của dự án Rust (tải trực tiếp vào thư mục models/)
mkdir -p models

# 1. Mô hình UVR-MDX-NET-Inst_HQ_3 (Tách 2-stem: Vocal & Instrumental, siêu nhanh CPU)
curl.exe -L -o models/UVR-MDX-NET-Inst_HQ_3.onnx https://huggingface.co/seanghay/uvr_models/resolve/main/UVR-MDX-NET-Inst_HQ_3.onnx

# 2. Mô hình HT-Demucs v4 (Tách 4-stem: Drums, Bass, Other, Vocals hoặc gộp 2-stem)
curl.exe -L -o models/htdemucs.onnx https://huggingface.co/StemSplitio/htdemucs-onnx/resolve/main/htdemucs.onnx

```

Nếu muốn giọng sạch nhất thế giới hiện nay: Chọn Mel-Band Roformer (hoặc BS-Roformer).
Nếu muốn bóc tách cả nhạc cụ lẫn hiệu ứng âm thanh (SFX): Chọn HT-Demucs v4.