use ndarray::Array4;
use ort::{inputs, GraphOptimizationLevel, Session};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SeparatorError {
    #[error("Lỗi khởi tạo ONNX Session: {0}")]
    OrtError(#[from] ort::Error),
    #[error("Lỗi I/O: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Lỗi xử lý mảng dữ liệu: {0}")]
    ShapeError(String),
    #[error("Lỗi chạy lệnh FFmpeg: {0}")]
    FffmpegError(String),
}

pub struct AudioSeparator {
    session: Session,
}

impl AudioSeparator {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, SeparatorError> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        Ok(Self { session })
    }

    /// Quy trình chạy chính từ file Video Input -> Output folder dạng .tmp/<file_name>/demix/
    pub fn process_video<P: AsRef<Path>>(
        &self,
        video_input: P,
        output_base_dir: P,
        format: &str, // "wav", "mp3", hoặc "aac"
    ) -> Result<(PathBuf, PathBuf), SeparatorError> {
        let video_path = video_input.as_ref();
        
        // 1. Lấy tên file video không chứa phần mở rộng (file_name)
        let file_stem = video_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("media");

        // 2. Tạo đường dẫn đích: output_base_dir/.tmp/<file_name>/demix/
        let target_dir = output_base_dir
            .as_ref()
            .join(".tmp")
            .join(file_stem)
            .join("demix");
        
        fs::create_dir_all(&target_dir)?;

        // 3. Tách Audio từ Video sử dụng FFmpeg ra file tạm PCM 44.1kHz Stereo
        let temp_pcm = target_dir.join("_extracted_temp.wav");
        let ffmpeg_status = Command::new("ffmpeg")
            .args([
                "-y",
                "-i", video_path.to_str().unwrap(),
                "-vn",
                "-ac", "2",
                "-ar", "44100",
                temp_pcm.to_str().unwrap(),
            ])
            .output()?;

        if !ffmpeg_status.status.success() {
            return Err(SeparatorError::FffmpegError(
                String::from_utf8_lossy(&ffmpeg_status.stderr).to_string(),
            ));
        }

        // 4. (Giả định vị trí gọi STFT & Inference model ONNX)
        // ... Code load PCM -> Spectrogram -> ONNX Inference -> iSTFT (như đã định nghĩa ở lib.rs) ...
        
        // Giả lập kết quả mảng PCM âm thanh tách được (Vocals & Instrumental):
        let dummy_vocal_pcm: Vec<f32> = vec![0.0; 44100 * 5]; // 5 giây sample
        let dummy_inst_pcm: Vec<f32> = vec![0.0; 44100 * 5];

        // 5. Xuất file đầu ra theo định dạng được yêu cầu (.wav, .mp3, .aac)
        let vocal_out_path = target_dir.join(format!("vocal.{}", format));
        let inst_out_path = target_dir.join(format!("instrumental.{}", format));

        self.save_audio_file(&dummy_vocal_pcm, 44100, &vocal_out_path, format)?;
        self.save_audio_file(&dummy_inst_pcm, 44100, &inst_out_path, format)?;

        // Dọn dẹp file tạm
        let _ = fs::remove_file(temp_pcm);

        Ok((vocal_out_path, inst_out_path))
    }

    /// Hàm hỗ trợ encode và ghi file âm thanh
    fn save_audio_file(
        &self,
        samples: &[f32],
        sample_rate: u32,
        out_path: &Path,
        format: &str,
    ) -> Result<(), SeparatorError> {
        match format {
            "wav" => {
                let spec = hound::WavSpec {
                    channels: 2,
                    sample_rate,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                };
                let mut writer = hound::WavWriter::create(out_path, spec)
                    .map_err(|e| SeparatorError::IoError(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
                
                for &sample in samples {
                    let amplitude = i16::MAX as f32;
                    writer.write_sample((sample * amplitude) as i16)
                        .map_err(|e| SeparatorError::IoError(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
                }
                writer.finalize().map_err(|e| SeparatorError::IoError(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
            }
            "mp3" | "aac" => {
                // Xuất tạm qua file WAV rồi dùng FFmpeg encode sang MP3/AAC
                let wav_temp = out_path.with_extension("temp.wav");
                self.save_audio_file(samples, sample_rate, &wav_temp, "wav")?;

                Command::new("ffmpeg")
                    .args([
                        "-y",
                        "-i", wav_temp.to_str().unwrap(),
                        out_path.to_str().unwrap(),
                    ])
                    .output()?;

                let _ = fs::remove_file(wav_temp);
            }
            _ => return Err(SeparatorError::ShapeError("Định dạng file không hỗ trợ".into())),
        }
        Ok(())
    }
}