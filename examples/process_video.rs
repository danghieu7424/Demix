use demux::AudioSeparator;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Start Demux Audio Separator Test ===");

    // 1. Đường dẫn tới file mô hình ONNX
    let model_path = Path::new("models/UVR-MDX-NET-Inst_HQ_3.onnx");
    if !model_path.exists() {
        eprintln!("Lỗi: Không tìm thấy file model tại {:?}", model_path);
        return Ok(());
    }

    // 2. Đường dẫn file video đầu vào và thư mục xuất output
    let video_input = Path::new("video_input/第001集.mp4");
    let output_dir = Path::new("output");
    let export_format = "mp3"; // Có thể thay bằng "wav", "mp3", hoặc "aac"

    if !video_input.exists() {
        eprintln!("Lỗi: Không tìm thấy file video tại {:?}", video_input);
        return Ok(());
    }

    println!("Đang khởi tạo ONNX Session...");
    let separator = AudioSeparator::new(model_path)?;

    println!("Đang xử lý video {:?} ...", video_input);
    let (vocal_path, inst_path) = separator.process_video(video_input, output_dir, export_format)?;

    println!("-> Tách thành công!");
    println!("   - Vocal file: {:?}", vocal_path);
    println!("   - Instrumental file: {:?}", inst_path);

    Ok(())
}