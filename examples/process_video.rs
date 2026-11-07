use demix::{AudioSeparator, DemixStage, OutputStems};
use std::io::Write;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("==================================================");
    println!("     DEMIX - TÁCH VOCAL & INSTRUMENTAL (MDX / DEMUCS)");
    println!("==================================================");

    // 1. Chọn mô hình AI:
    // - Lựa chọn 1: MDX-Net Inst HQ 3 (mặc định 2-stem: vocal & instrumental, siêu nhanh CPU)
    // - Lựa chọn 2: HT-Demucs v4 (tách 4-stem hoặc gộp 2-stem)
    let model_mdx = Path::new("models/UVR-MDX-NET-Inst_HQ_3.onnx");
    let model_demucs = Path::new("models/htdemucs.onnx");

    // Chọn model cần dùng (ví dụ: model_mdx hoặc model_demucs)
    let chosen_model = if model_demucs.exists() {
        model_demucs
    } else {
        model_mdx
    };

    // Tự động tải model nếu chưa có trên máy theo lệnh hướng dẫn ở readme.md
    if !chosen_model.exists() {
        println!("Chưa tìm thấy model tại {:?}. Đang tự động tải về theo hướng dẫn readme.md...", chosen_model);
        AudioSeparator::ensure_model(chosen_model)?;
    }
    let model_path = chosen_model;

    // 2. Đầu vào & Đầu ra
    let input_target = Path::new("video_input");
    let output_dir = Path::new("output");
    let export_format = "mp3"; // "mp3", "wav", hoặc "aac"

    // 3. Cấu hình số track xuất ra:
    // - OutputStems::TwoStems : Mặc định gộp thành 2 track (vocal & instrumental)
    // - OutputStems::FourStems: Xuất đủ 4 track (drums, bass, other, vocals - chỉ áp dụng cho HT-Demucs)
    let stems_mode = OutputStems::TwoStems; 

    println!("-> Mô hình sử dụng : {:?}", model_path);
    println!("-> Chế độ xuất     : {:?}", stems_mode);
    println!("-> Đầu vào         : {:?}", input_target);
    println!("-> Định dạng xuất  : {}", export_format);
    println!("-> Đang khởi tạo ONNX Session...");

    let separator = AudioSeparator::new(model_path)?
        .with_output_stems(stems_mode);

    println!("   Loại mô hình tự nhận diện: {:?}", separator.model_kind());
    println!("-> Bắt đầu tiến trình xử lý:\n");

    // 4. Xử lý đường dẫn (tự động nhận 1 video, 1 audio lẻ hoặc 1 thư mục chứa nhiều file)
    let results = separator.process_path(
        input_target,
        output_dir,
        export_format,
        |file_path, progress| {
            let file_name = file_path.file_name().and_then(|s| s.to_str()).unwrap_or("");

            match &progress.stage {
                DemixStage::ExtractAudio | DemixStage::LoadAudio => {
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
    println!("-> TỔNG KẾT XỬ LÝ ({} file hoàn thành):", results.len());
    for (idx, file_outputs) in results.iter().enumerate() {
        println!("   [{}] File #{}", idx + 1, idx + 1);
        for out_file in file_outputs {
            println!("       - {:?}", out_file);
        }
    }
    println!("==================================================");

    Ok(())
}
