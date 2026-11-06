use demix::{AudioSeparator, DemixStage};
use std::io::Write;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("==================================================");
    println!("     DEMIX - TÁCH VOCAL & INSTRUMENTAL TỪ VIDEO   ");
    println!("==================================================");

    // 1. Đường dẫn mô hình ONNX
    let model_path = Path::new("models/UVR-MDX-NET-Inst_HQ_3.onnx");
    if !model_path.exists() {
        eprintln!("Lỗi: Không tìm thấy file model tại: {:?}", model_path);
        eprintln!("Vui lòng tải model theo hướng dẫn trong readme.md!");
        return Ok(());
    }

    // 2. Đầu vào: Có thể là 1 file video, 1 file audio lẻ, HOẶC 1 thư mục chứa nhiều file
    // Ví dụ:
    //   - 1 file video: Path::new("video_input/第001集.mp4")
    //   - 1 file audio: Path::new("video_input/song.mp3")
    //   - Thư mục nhiều file: Path::new("video_input")
    let input_target = Path::new("video_input");
    let output_dir = Path::new("output");
    let export_format = "mp3"; // "mp3", "wav", hoặc "aac" (hoặc "acc")

    println!("-> Model       : {:?}", model_path);
    println!("-> Đầu vào     : {:?}", input_target);
    println!("-> Định dạng ra: {}", export_format);
    println!("-> Khởi tạo ONNX Session...");

    let separator = AudioSeparator::new(model_path)?;
    println!("-> Bắt đầu tiến trình xử lý:\n");

    // 3. Sử dụng separator.process_path:
    // Tự động nhận diện nếu là 1 file đơn lẻ thì xử lý file đó,
    // nếu là thư mục thì xử lý lần lượt tất cả các file video / audio tìm thấy.
    let results = separator.process_path(
        input_target,
        output_dir,
        export_format,
        |file_path, progress| {
            let file_name = file_path.file_name().and_then(|s| s.to_str()).unwrap_or("");

            match &progress.stage {
                DemixStage::ExtractAudio | DemixStage::LoadAudio => {
                    // Giai đoạn chuẩn bị (chưa vào tính chunk)
                    print!("\r[{}] -> {}", file_name, progress.message);
                    let _ = std::io::stdout().flush();
                }
                DemixStage::Demixing { current_chunk, total_chunks } => {
                    // TIẾN ĐỘ CHỈ TÍNH RIÊNG PHẦN CHUNK CHẠY ĐƯỢC (0% -> 100%)
                    let bar_width: usize = 30;
                    let filled = ((progress.percentage / 100.0) * bar_width as f32) as usize;
                    let bar: String = "█".repeat(filled) + &"-".repeat(bar_width.saturating_sub(filled));

                    print!(
                        "\r[{}] [{}] {:>5.1}% | Chunk {}/{} | {}",
                        file_name, bar, progress.percentage, current_chunk, total_chunks, progress.message
                    );
                    let _ = std::io::stdout().flush();
                }
                DemixStage::ExportAudio { file_name: out_f } => {
                    print!("\r[{}] -> Đang xuất file {}...", file_name, out_f);
                    let _ = std::io::stdout().flush();
                }
                DemixStage::Finished => {
                    println!("\r[{}] [██████████████████████████████] 100.0% | Hoàn thành!", file_name);
                }
            }
        },
    )?;

    println!("\n==================================================");
    println!("-> TỔNG KẾT XỬ LÝ ({} file):", results.len());
    for (idx, (vocal_path, inst_path)) in results.iter().enumerate() {
        println!("   [{}]", idx + 1);
        println!("       - Vocal        : {:?}", vocal_path);
        println!("       - Instrumental : {:?}", inst_path);
    }
    println!("==================================================");

    Ok(())
}
