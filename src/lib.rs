use ndarray::Array4;
use ort::{
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use rustfft::{num_complex::Complex, FftPlanner};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SeparatorError {
    #[error("Lỗi ONNX Runtime: {0}")]
    OrtError(String),
    #[error("Lỗi I/O: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Lỗi định dạng dữ liệu: {0}")]
    ShapeError(String),
    #[error("Lỗi chạy FFmpeg: {0}")]
    FffmpegError(String),
    #[error("Lỗi đọc/ghi file WAV: {0}")]
    WavError(String),
}

impl From<ort::Error> for SeparatorError {
    fn from(e: ort::Error) -> Self {
        SeparatorError::OrtError(e.to_string())
    }
}

impl From<hound::Error> for SeparatorError {
    fn from(e: hound::Error) -> Self {
        SeparatorError::WavError(e.to_string())
    }
}

/// Các giai đoạn trong quá trình xử lý tách âm thanh
#[derive(Debug, Clone, PartialEq)]
pub enum DemixStage {
    ExtractAudio,
    LoadAudio,
    Demixing { current_chunk: usize, total_chunks: usize },
    ExportAudio { file_name: String },
    Finished,
}

/// Thông tin tiến độ chuyển giao cho callback
#[derive(Debug, Clone)]
pub struct DemixProgress {
    /// Tiến độ hoàn thành từ 0.0 đến 100.0%
    pub percentage: f32,
    /// Giai đoạn hiện tại
    pub stage: DemixStage,
    /// Mô tả chi tiết tiến độ
    pub message: String,
}

pub struct AudioSeparator {
    session: Mutex<Session>,
}

impl AudioSeparator {
    /// Khởi tạo AudioSeparator với đường dẫn file model ONNX (sử dụng tối đa số luồng CPU có sẵn)
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, SeparatorError> {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self::new_with_threads(model_path, threads)
    }

    /// Khởi tạo AudioSeparator chỉ định số luồng xử lý
    pub fn new_with_threads<P: AsRef<Path>>(
        model_path: P,
        num_threads: usize,
    ) -> Result<Self, SeparatorError> {
        let env = ort::init()
            .build()
            .map_err(|e| SeparatorError::OrtError(e.to_string()))?;

        let session = Session::builder(&env)
            .map_err(|e| SeparatorError::OrtError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap_or_else(|e| e.recover())
            .with_intra_threads(num_threads)
            .unwrap_or_else(|e| e.recover())
            .commit_from_file(model_path)
            .map_err(|e| SeparatorError::OrtError(e.to_string()))?;

        Ok(Self {
            session: Mutex::new(session),
        })
    }

    /// Xử lý video mặc định (không callback tiến độ)
    pub fn process_video<P1: AsRef<Path>, P2: AsRef<Path>>(
        &self,
        video_input: P1,
        output_base_dir: P2,
        format: &str,
    ) -> Result<(PathBuf, PathBuf), SeparatorError> {
        self.process_video_with_progress(video_input, output_base_dir, format, |_| {})
    }

    /// Quy trình chính: Video Input -> output/.tmp/<file_name>/demix/ gồm vocal.<format> và instrumental.<format>
    /// Đồng thời báo cáo tiến độ chi tiết qua closure `on_progress`
    pub fn process_video_with_progress<P1, P2, F>(
        &self,
        video_input: P1,
        output_base_dir: P2,
        format: &str,
        mut on_progress: F,
    ) -> Result<(PathBuf, PathBuf), SeparatorError>
    where
        P1: AsRef<Path>,
        P2: AsRef<Path>,
        F: FnMut(DemixProgress),
    {
        let video_path = video_input.as_ref();
        if !video_path.exists() {
            return Err(SeparatorError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Không tìm thấy file video: {:?}", video_path),
            )));
        }

        // Chuẩn hóa định dạng mở rộng (hỗ trợ cả "acc" thay cho "aac")
        let format_lower = format.to_lowercase();
        let norm_ext = match format_lower.as_str() {
            "acc" | "aac" => "aac",
            "mp3" => "mp3",
            "wav" => "wav",
            other => other,
        };

        // 1. Lấy tên file video không chứa phần mở rộng (file_name / file_stem)
        let file_stem = video_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("media");

        // 2. Tạo đường dẫn đích: output/.tmp/<file_name>/demix/
        let target_dir = output_base_dir
            .as_ref()
            .join(".tmp")
            .join(file_stem)
            .join("demix");
        fs::create_dir_all(&target_dir)?;

        // 3. Trích xuất âm thanh từ Video sang WAV chuẩn (44.1kHz stereo PCM 16-bit)
        on_progress(DemixProgress {
            percentage: 5.0,
            stage: DemixStage::ExtractAudio,
            message: format!("Đang trích xuất audio từ video '{}' qua FFmpeg...", file_stem),
        });

        let temp_pcm = target_dir.join("_extracted_temp.wav");
        let ffmpeg_status = Command::new("ffmpeg")
            .args(["-y", "-i"])
            .arg(video_path)
            .args(["-vn", "-ac", "2", "-ar", "44100"])
            .arg(&temp_pcm)
            .output()?;

        if !ffmpeg_status.status.success() {
            return Err(SeparatorError::FffmpegError(
                String::from_utf8_lossy(&ffmpeg_status.stderr).to_string(),
            ));
        }

        // 4. Đọc dữ liệu audio mẫu từ file PCM WAV tạm thời
        on_progress(DemixProgress {
            percentage: 15.0,
            stage: DemixStage::LoadAudio,
            message: "Đang đọc và phân tích dữ liệu sóng âm...".to_string(),
        });

        let (left_channel, right_channel, sample_rate) = {
            let mut reader = hound::WavReader::open(&temp_pcm)?;
            let spec = reader.spec();
            let num_channels = spec.channels as usize;
            let sample_rate = spec.sample_rate;

            let raw_samples: Vec<f32> = match spec.sample_format {
                hound::SampleFormat::Int => {
                    let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
                    reader
                        .samples::<i32>()
                        .map(|s| s.map(|v| v as f32 / max_val))
                        .collect::<Result<Vec<f32>, _>>()?
                }
                hound::SampleFormat::Float => {
                    reader.samples::<f32>().collect::<Result<Vec<f32>, _>>()?
                }
            };

            let total_frames = raw_samples.len() / num_channels;
            let mut left = Vec::with_capacity(total_frames);
            let mut right = Vec::with_capacity(total_frames);

            if num_channels == 1 {
                for &sample in &raw_samples {
                    left.push(sample);
                    right.push(sample);
                }
            } else {
                for i in 0..total_frames {
                    left.push(raw_samples[i * num_channels]);
                    right.push(raw_samples[i * num_channels + 1]);
                }
            }
            (left, right, sample_rate)
        };

        // 5. Phân tách âm thanh qua MDX-Net ONNX model (STFT -> Model -> ISTFT)
        let (vocal_left, vocal_right, inst_left, inst_right) =
            self.separate_stereo(&left_channel, &right_channel, &mut on_progress)?;

        // Ghép kênh Left - Right lại thành chuỗi interleaved stereo
        let total_frames = vocal_left.len();
        let mut vocal_pcm = Vec::with_capacity(total_frames * 2);
        let mut inst_pcm = Vec::with_capacity(total_frames * 2);

        for i in 0..total_frames {
            vocal_pcm.push(vocal_left[i]);
            vocal_pcm.push(vocal_right[i]);
            inst_pcm.push(inst_left[i]);
            inst_pcm.push(inst_right[i]);
        }

        // 6. Xuất 2 file đầu ra: vocal.<format> và instrumental.<format>
        let vocal_filename = format!("vocal.{}", norm_ext);
        let inst_filename = format!("instrumental.{}", norm_ext);

        let vocal_out_path = target_dir.join(&vocal_filename);
        let inst_out_path = target_dir.join(&inst_filename);

        on_progress(DemixProgress {
            percentage: 88.0,
            stage: DemixStage::ExportAudio {
                file_name: vocal_filename.clone(),
            },
            message: format!("Đang xuất file Vocal ({}) ...", vocal_filename),
        });
        self.save_audio_file(&vocal_pcm, sample_rate, &vocal_out_path, norm_ext)?;

        on_progress(DemixProgress {
            percentage: 95.0,
            stage: DemixStage::ExportAudio {
                file_name: inst_filename.clone(),
            },
            message: format!("Đang xuất file Instrumental ({}) ...", inst_filename),
        });
        self.save_audio_file(&inst_pcm, sample_rate, &inst_out_path, norm_ext)?;

        // Dọn dẹp file tạm
        let _ = fs::remove_file(temp_pcm);

        on_progress(DemixProgress {
            percentage: 100.0,
            stage: DemixStage::Finished,
            message: "Tách âm thanh hoàn tất thành công!".to_string(),
        });

        Ok((vocal_out_path, inst_out_path))
    }

    /// Tách tín hiệu stereo (Left & Right) theo từng chunk qua MDX-Net
    fn separate_stereo<F>(
        &self,
        left: &[f32],
        right: &[f32],
        on_progress: &mut F,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), SeparatorError>
    where
        F: FnMut(DemixProgress),
    {
        let total_samples = left.len();
        let n_fft = 6144;
        let hop = 1024;
        let dim_t = 256;
        let dim_f = 3072;
        let chunk_size = hop * (dim_t - 1); // 261120
        let trim = n_fft / 2; // 3072
        let gen_size = chunk_size - 2 * trim; // 254976

        let num_chunks = (total_samples + gen_size - 1) / gen_size;
        let pad_start = trim;
        let total_padded_len = pad_start + num_chunks * gen_size + trim;

        let mut padded_left = vec![0.0f32; total_padded_len];
        let mut padded_right = vec![0.0f32; total_padded_len];
        padded_left[pad_start..pad_start + total_samples].copy_from_slice(left);
        padded_right[pad_start..pad_start + total_samples].copy_from_slice(right);

        // Chuẩn bị Hann window
        let window: Vec<f32> = (0..n_fft)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n_fft as f32).cos()))
            .collect();

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);
        let ifft = planner.plan_fft_inverse(n_fft);

        let mut vocal_left = Vec::with_capacity(total_samples);
        let mut vocal_right = Vec::with_capacity(total_samples);
        let mut inst_left = Vec::with_capacity(total_samples);
        let mut inst_right = Vec::with_capacity(total_samples);

        for k in 0..num_chunks {
            let chunk_start = k * gen_size;
            let chunk_l = &padded_left[chunk_start..chunk_start + chunk_size];
            let chunk_r = &padded_right[chunk_start..chunk_start + chunk_size];

            // 1. Tính STFT cho từng kênh Left và Right
            let (l_real, l_imag) = Self::stft_chunk(chunk_l, n_fft, hop, dim_t, dim_f, trim, &window, &fft);
            let (r_real, r_imag) = Self::stft_chunk(chunk_r, n_fft, hop, dim_t, dim_f, trim, &window, &fft);

            // 2. Tạo ndarray tensor kích thước [1, 4, 3072, 256]
            let mut input_spec = Array4::<f32>::zeros((1, 4, dim_f, dim_t));
            for f in 0..dim_f {
                for t in 0..dim_t {
                    let idx = f * dim_t + t;
                    input_spec[[0, 0, f, t]] = l_real[idx];
                    input_spec[[0, 1, f, t]] = l_imag[idx];
                    input_spec[[0, 2, f, t]] = r_real[idx];
                    input_spec[[0, 3, f, t]] = r_imag[idx];
                }
            }

            // 3. Chạy model ONNX inference
            let (vocal_spec, inst_spec) = self.separate_spectrogram(input_spec)?;

            // 4. Invert STFT về lại tín hiệu audio thời gian
            let (vl_chunk, vr_chunk) = Self::istft_stereo(&vocal_spec, n_fft, hop, dim_t, dim_f, trim, &window, &ifft);
            let (il_chunk, ir_chunk) = Self::istft_stereo(&inst_spec, n_fft, hop, dim_t, dim_f, trim, &window, &ifft);

            // 5. Cắt phần trung tâm gen_size loại bỏ viền lề
            vocal_left.extend_from_slice(&vl_chunk[trim..trim + gen_size]);
            vocal_right.extend_from_slice(&vr_chunk[trim..trim + gen_size]);
            inst_left.extend_from_slice(&il_chunk[trim..trim + gen_size]);
            inst_right.extend_from_slice(&ir_chunk[trim..trim + gen_size]);

            // Cập nhật tiến độ: giai đoạn Demixing từ 15% đến 85%
            let progress_percent = 15.0 + 70.0 * ((k + 1) as f32 / num_chunks as f32);
            on_progress(DemixProgress {
                percentage: progress_percent,
                stage: DemixStage::Demixing {
                    current_chunk: k + 1,
                    total_chunks: num_chunks,
                },
                message: format!(
                    "Đang phân tách âm thanh qua AI (Chunk {}/{} - {:.1}%)",
                    k + 1,
                    num_chunks,
                    progress_percent
                ),
            });
        }

        vocal_left.truncate(total_samples);
        vocal_right.truncate(total_samples);
        inst_left.truncate(total_samples);
        inst_right.truncate(total_samples);

        Ok((vocal_left, vocal_right, inst_left, inst_right))
    }

    /// Tính STFT cho 1 đoạn audio (chunk)
    fn stft_chunk(
        chunk: &[f32],
        n_fft: usize,
        hop: usize,
        dim_t: usize,
        dim_f: usize,
        trim: usize,
        window: &[f32],
        fft: &std::sync::Arc<dyn rustfft::Fft<f32>>,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut padded = vec![0.0f32; trim];
        padded.extend_from_slice(chunk);
        padded.resize(padded.len() + trim, 0.0);

        let mut real = vec![0.0f32; dim_f * dim_t];
        let mut imag = vec![0.0f32; dim_f * dim_t];

        for t in 0..dim_t {
            let start = t * hop;
            let mut buf: Vec<Complex<f32>> = (0..n_fft)
                .map(|i| Complex::new(padded[start + i] * window[i], 0.0))
                .collect();
            fft.process(&mut buf);
            for f in 0..dim_f {
                let idx = f * dim_t + t;
                real[idx] = buf[f].re;
                imag[idx] = buf[f].im;
            }
        }
        (real, imag)
    }

    /// Biến đổi iSTFT spectrogram 4 kênh (Left Real, Left Imag, Right Real, Right Imag) về audio 2 kênh
    fn istft_stereo(
        spec: &Array4<f32>,
        n_fft: usize,
        hop: usize,
        dim_t: usize,
        dim_f: usize,
        trim: usize,
        window: &[f32],
        ifft: &std::sync::Arc<dyn rustfft::Fft<f32>>,
    ) -> (Vec<f32>, Vec<f32>) {
        let chunk_padded_len = (dim_t - 1) * hop + 2 * trim;
        let mut left_padded = vec![0.0f32; chunk_padded_len];
        let mut right_padded = vec![0.0f32; chunk_padded_len];
        let mut win_sum = vec![0.0f32; chunk_padded_len];

        let norm = 1.0 / (n_fft as f32);

        for t in 0..dim_t {
            let start = t * hop;
            let mut buf_l = vec![Complex::new(0.0f32, 0.0f32); n_fft];
            let mut buf_r = vec![Complex::new(0.0f32, 0.0f32); n_fft];

            for f in 0..dim_f {
                let val_l = Complex::new(spec[[0, 0, f, t]], spec[[0, 1, f, t]]);
                let val_r = Complex::new(spec[[0, 2, f, t]], spec[[0, 3, f, t]]);
                buf_l[f] = val_l;
                buf_r[f] = val_r;
                if f > 0 {
                    buf_l[n_fft - f] = val_l.conj();
                    buf_r[n_fft - f] = val_r.conj();
                }
            }

            ifft.process(&mut buf_l);
            ifft.process(&mut buf_r);

            for i in 0..n_fft {
                left_padded[start + i] += buf_l[i].re * norm * window[i];
                right_padded[start + i] += buf_r[i].re * norm * window[i];
                win_sum[start + i] += window[i] * window[i];
            }
        }

        for i in 0..chunk_padded_len {
            if win_sum[i] > 1e-6 {
                left_padded[i] /= win_sum[i];
                right_padded[i] /= win_sum[i];
            }
        }

        let chunk_len = (dim_t - 1) * hop;
        (
            left_padded[trim..trim + chunk_len].to_vec(),
            right_padded[trim..trim + chunk_len].to_vec(),
        )
    }

    /// Tách spectrogram qua model ONNX (trả về (vocal_spec, instrumental_spec))
    pub fn separate_spectrogram(
        &self,
        input_spec: Array4<f32>,
    ) -> Result<(Array4<f32>, Array4<f32>), SeparatorError> {
        let input_tensor = Value::from_array(input_spec.clone())?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| SeparatorError::ShapeError(e.to_string()))?;
        let outputs = session.run(inputs!["input" => input_tensor])?;

        let inst_extracted = outputs["output"].try_extract_tensor::<f32>()?;
        let (shape, slice) = (inst_extracted.0, inst_extracted.1);

        let shape_vec: Vec<usize> = shape.iter().map(|&v| v as usize).collect();
        let inst_tensor = Array4::from_shape_vec(
            (shape_vec[0], shape_vec[1], shape_vec[2], shape_vec[3]),
            slice.to_vec(),
        )
        .map_err(|e| SeparatorError::ShapeError(e.to_string()))?;

        // Với model UVR-MDX-NET-Inst_HQ_3, model dự đoán Instrumental -> Vocal = Input - Instrumental
        let vocal_tensor = &input_spec - &inst_tensor;

        Ok((vocal_tensor, inst_tensor))
    }

    /// Lưu mảng PCM stereo thành file âm thanh theo định dạng chỉ định (wav, mp3, aac)
    pub fn save_audio_file(
        &self,
        samples: &[f32],
        sample_rate: u32,
        out_path: &Path,
        format: &str,
    ) -> Result<(), SeparatorError> {
        let format_lower = format.to_lowercase();
        let norm_format = match format_lower.as_str() {
            "acc" | "aac" => "aac",
            "mp3" => "mp3",
            "wav" => "wav",
            other => other,
        };

        match norm_format {
            "wav" => {
                let spec = hound::WavSpec {
                    channels: 2,
                    sample_rate,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                };
                let mut writer = hound::WavWriter::create(out_path, spec)?;
                for &sample in samples {
                    let clamped = sample.clamp(-1.0, 1.0);
                    let val = (clamped * i16::MAX as f32) as i16;
                    writer.write_sample(val)?;
                }
                writer.finalize()?;
            }
            "mp3" => {
                let wav_temp = out_path.with_extension("temp.wav");
                self.save_audio_file(samples, sample_rate, &wav_temp, "wav")?;

                let status = Command::new("ffmpeg")
                    .arg("-y")
                    .arg("-i")
                    .arg(&wav_temp)
                    .args(["-vn", "-b:a", "320k"])
                    .arg(out_path)
                    .output()?;

                let _ = fs::remove_file(wav_temp);
                if !status.status.success() {
                    return Err(SeparatorError::FffmpegError(
                        String::from_utf8_lossy(&status.stderr).to_string(),
                    ));
                }
            }
            "aac" => {
                let wav_temp = out_path.with_extension("temp.wav");
                self.save_audio_file(samples, sample_rate, &wav_temp, "wav")?;

                let status = Command::new("ffmpeg")
                    .arg("-y")
                    .arg("-i")
                    .arg(&wav_temp)
                    .args(["-vn", "-c:a", "aac", "-b:a", "256k"])
                    .arg(out_path)
                    .output()?;

                let _ = fs::remove_file(wav_temp);
                if !status.status.success() {
                    return Err(SeparatorError::FffmpegError(
                        String::from_utf8_lossy(&status.stderr).to_string(),
                    ));
                }
            }
            _ => {
                return Err(SeparatorError::ShapeError(format!(
                    "Định dạng '{}' không được hỗ trợ. Vui lòng chọn: wav, mp3, hoặc aac.",
                    format
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_chunk_tiling() {
        let n_fft = 6144;
        let hop = 1024;
        let dim_t = 256;
        let dim_f = 3072;
        let chunk_size = hop * (dim_t - 1); // 261120
        let trim = n_fft / 2; // 3072
        let gen_size = chunk_size - 2 * trim; // 254976

        let total_samples = 600_000;
        let signal: Vec<f32> = (0..total_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let window: Vec<f32> = (0..n_fft)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n_fft as f32).cos()))
            .collect();

        let num_chunks = (total_samples + gen_size - 1) / gen_size;
        let pad_start = trim;
        let total_padded_len = pad_start + num_chunks * gen_size + trim;

        let mut padded = vec![0.0f32; total_padded_len];
        padded[pad_start..pad_start + total_samples].copy_from_slice(&signal);

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);
        let ifft = planner.plan_fft_inverse(n_fft);

        let mut reconstructed = Vec::with_capacity(total_samples);

        for k in 0..num_chunks {
            let start = k * gen_size;
            let chunk = &padded[start..start + chunk_size];

            let (real, imag) =
                AudioSeparator::stft_chunk(chunk, n_fft, hop, dim_t, dim_f, trim, &window, &fft);

            let mut spec = Array4::<f32>::zeros((1, 4, dim_f, dim_t));
            for f in 0..dim_f {
                for t in 0..dim_t {
                    let idx = f * dim_t + t;
                    spec[[0, 0, f, t]] = real[idx];
                    spec[[0, 1, f, t]] = imag[idx];
                }
            }

            let (l_out, _) =
                AudioSeparator::istft_stereo(&spec, n_fft, hop, dim_t, dim_f, trim, &window, &ifft);

            let gen_slice = &l_out[trim..trim + gen_size];
            reconstructed.extend_from_slice(gen_slice);
        }

        reconstructed.truncate(total_samples);

        let mut max_err = 0.0f32;
        for i in 0..total_samples {
            let err = (signal[i] - reconstructed[i]).abs();
            if err > max_err {
                max_err = err;
            }
        }
        println!("Multi-chunk max error: {:e}", max_err);
        assert!(max_err < 1e-4, "Reconstruction error too high: {}", max_err);
    }
}
