use demix::AudioSeparator;
use std::io::Write;
use std::path::{Path, PathBuf};

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

    // 2. Thư mục hoặc file video đầu vào
    // Mặc định tìm file video trong thư mục video_input/
    let video_input_folder = Path::new("video_input");
    let target_video: Option<PathBuf> = if video_input_folder.is_file() {
        Some(video_input_folder.to_path_buf())
    } else {
        // Tìm file video đầu tiên trong video_input
        let mut found = None;
        if let Ok(entries) = std::fs::read_dir(video_input_folder) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                    let ext_lower = ext.to_lowercase();
                    if ["mp4", "mkv", "avi", "mov", "flv", "webm"].contains(&ext_lower.as_str()) {
                        found = Some(path);
                        break;
                    }
                }
            }
        }
        found
    };

    let video_path = match target_video {
        Some(path) => path,
        None => {
            eprintln!("Không tìm thấy video nào trong thư mục {:?}", video_input_folder);
            return Ok(());
        }
    };

    // 3. Cấu hình thư mục output và định dạng xuất
    let output_dir = Path::new("output");
    // Hỗ trợ: "mp3", "wav", hoặc "aac" (hoặc "acc")
    let export_format = "mp3";

    println!("-> Model       : {:?}", model_path);
    println!("-> Input video : {:?}", video_path);
    println!("-> Định dạng ra: {}", export_format);
    println!("-> Khởi tạo ONNX Session...");

    let separator = AudioSeparator::new(model_path)?;
    println!("-> Bắt đầu tiến trình xử lý:\n");

    // 4. Gọi hàm process_video_with_progress với callback tiến độ trực quan
    let (vocal_path, inst_path) = separator.process_video_with_progress(
        &video_path,
        output_dir,
        export_format,
        |progress| {
            // Hiển thị thanh tiến trình trực quan ngay trên terminal
            let bar_width: usize = 30;
            let filled = ((progress.percentage / 100.0) * bar_width as f32) as usize;
            let bar: String = "█".repeat(filled) + &"-".repeat(bar_width.saturating_sub(filled));

            print!(
                "\r[{}] {:>5.1}% | {}",
                bar, progress.percentage, progress.message
            );
            let _ = std::io::stdout().flush();

            if progress.percentage >= 100.0 {
                println!();
            }
        },
    )?;

    println!("\n==================================================");
    println!("-> TÁCH THÀNH CÔNG!");
    println!("   - Vocal File        : {:?}", vocal_path);
    println!("   - Instrumental File : {:?}", inst_path);
    println!("==================================================");

    Ok(())
}
