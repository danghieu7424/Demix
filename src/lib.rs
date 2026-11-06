use ndarray::{Array3, Array4};
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

/// Loại mô hình phân tách
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// UVR-MDX-NET-Inst_HQ_3 (Tách 2-stem: Vocal & Instrumental, siêu nhanh CPU)
    MdxNet,
    /// HT-Demucs v4 (Mặc định gộp 2-stem hoặc xuất đủ 4-stem)
    HtDemucs,
}

/// Số lượng track đầu ra (áp dụng khi dùng HT-Demucs)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputStems {
    /// Mặc định: Gộp thành 2 file (vocal.<format> và instrumental.<format>)
    #[default]
    TwoStems,
    /// Xuất đủ 4 file riêng biệt: drums, bass, other, vocals (chỉ áp dụng với HT-Demucs)
    FourStems,
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
    /// Chunk hiện tại đang xử lý (0 nếu đang ở khâu trích xuất/đọc file)
    pub current_chunk: usize,
    /// Tổng số chunk cần xử lý
    pub total_chunks: usize,
    /// Tiến độ hoàn thành chỉ tính theo số chunk (0.0 đến 100.0%)
    pub percentage: f32,
    /// Giai đoạn hiện tại
    pub stage: DemixStage,
    /// Mô tả chi tiết tiến độ
    pub message: String,
}

pub struct AudioSeparator {
    session: Mutex<Session>,
    model_kind: ModelKind,
    output_stems: OutputStems,
}

impl AudioSeparator {
    /// Khởi tạo AudioSeparator với đường dẫn file model ONNX (tự động phát hiện loại mô hình và số luồng CPU)
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
            .commit_from_file(model_path.as_ref())
            .map_err(|e| SeparatorError::OrtError(e.to_string()))?;

        // Tự động nhận diện loại mô hình qua tên file hoặc cấu trúc input tensor
        let model_kind = {
            let path_str = model_path.as_ref().to_string_lossy().to_lowercase();
            if path_str.contains("demucs") {
                ModelKind::HtDemucs
            } else if let Some(input) = session.inputs().first() {
                if input.name() == "mix" {
                    ModelKind::HtDemucs
                } else {
                    ModelKind::MdxNet
                }
            } else {
                ModelKind::MdxNet
            }
        };

        Ok(Self {
            session: Mutex::new(session),
            model_kind,
            output_stems: OutputStems::TwoStems,
        })
    }

    /// Cấu hình số track xuất ra (TwoStems mặc định hoặc FourStems)
    pub fn with_output_stems(mut self, stems: OutputStems) -> Self {
        self.output_stems = stems;
        self
    }

    pub fn set_output_stems(&mut self, stems: OutputStems) {
        self.output_stems = stems;
    }

    /// Chỉ định tường minh loại model
    pub fn with_model_kind(mut self, kind: ModelKind) -> Self {
        self.model_kind = kind;
        self
    }

    pub fn set_model_kind(&mut self, kind: ModelKind) {
        self.model_kind = kind;
    }

    pub fn model_kind(&self) -> ModelKind {
        self.model_kind
    }

    pub fn output_stems(&self) -> OutputStems {
        self.output_stems
    }

    /// Xử lý video mặc định (trả về 2 file: vocal và instrumental)
    pub fn process_video<P1: AsRef<Path>, P2: AsRef<Path>>(
        &self,
        video_input: P1,
        output_base_dir: P2,
        format: &str,
    ) -> Result<(PathBuf, PathBuf), SeparatorError> {
        self.process_video_with_progress(video_input, output_base_dir, format, |_| {})
    }

    /// Quy trình chính: Video Input -> output/.tmp/<file_name>/demix/
    /// Mặc định trả về (vocal_path, instrumental_path)
    pub fn process_video_with_progress<P1, P2, F>(
        &self,
        video_input: P1,
        output_base_dir: P2,
        format: &str,
        on_progress: F,
    ) -> Result<(PathBuf, PathBuf), SeparatorError>
    where
        P1: AsRef<Path>,
        P2: AsRef<Path>,
        F: FnMut(DemixProgress),
    {
        let files = self.process_video_stems(video_input, output_base_dir, format, on_progress)?;
        if files.len() >= 2 {
            Ok((files[0].clone(), files[1].clone()))
        } else if let Some(first) = files.first() {
            Ok((first.clone(), first.clone()))
        } else {
            Err(SeparatorError::ShapeError("Không có file đầu ra nào được tạo".to_string()))
        }
    }

    /// Xử lý video và xuất ra danh sách các file (2 file nếu TwoStems, hoặc 4 file nếu FourStems)
    pub fn process_video_stems<P1, P2, F>(
        &self,
        video_input: P1,
        output_base_dir: P2,
        format: &str,
        mut on_progress: F,
    ) -> Result<Vec<PathBuf>, SeparatorError>
    where
        P1: AsRef<Path>,
        P2: AsRef<Path>,
        F: FnMut(DemixProgress),
    {
        let video_path = video_input.as_ref();
        if !video_path.exists() {
            return Err(SeparatorError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Không tìm thấy file: {:?}", video_path),
            )));
        }

        let format_lower = format.to_lowercase();
        let norm_ext = match format_lower.as_str() {
            "acc" | "aac" => "aac",
            "mp3" => "mp3",
            "wav" => "wav",
            other => other,
        };

        let file_stem = video_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("media");

        let target_dir = output_base_dir
            .as_ref()
            .join(".tmp")
            .join(file_stem)
            .join("demix");
        fs::create_dir_all(&target_dir)?;

        on_progress(DemixProgress {
            current_chunk: 0,
            total_chunks: 0,
            percentage: 0.0,
            stage: DemixStage::ExtractAudio,
            message: format!("Đang trích xuất audio từ '{}' qua FFmpeg...", file_stem),
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

        on_progress(DemixProgress {
            current_chunk: 0,
            total_chunks: 0,
            percentage: 0.0,
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

        let mut output_paths = Vec::new();

        match self.model_kind {
            ModelKind::MdxNet => {
                // Tách qua UVR-MDX-NET (STFT -> ONNX -> iSTFT)
                let (vocal_left, vocal_right, inst_left, inst_right) =
                    self.separate_stereo_mdx(&left_channel, &right_channel, &mut on_progress)?;

                let total_frames = vocal_left.len();
                let mut vocal_pcm = Vec::with_capacity(total_frames * 2);
                let mut inst_pcm = Vec::with_capacity(total_frames * 2);

                for i in 0..total_frames {
                    vocal_pcm.push(vocal_left[i]);
                    vocal_pcm.push(vocal_right[i]);
                    inst_pcm.push(inst_left[i]);
                    inst_pcm.push(inst_right[i]);
                }

                let vocal_filename = format!("vocal.{}", norm_ext);
                let inst_filename = format!("instrumental.{}", norm_ext);

                let vocal_out_path = target_dir.join(&vocal_filename);
                let inst_out_path = target_dir.join(&inst_filename);

                on_progress(DemixProgress {
                    current_chunk: 100,
                    total_chunks: 100,
                    percentage: 100.0,
                    stage: DemixStage::ExportAudio {
                        file_name: vocal_filename.clone(),
                    },
                    message: format!("Đang xuất file Vocal ({}) ...", vocal_filename),
                });
                self.save_audio_file(&vocal_pcm, sample_rate, &vocal_out_path, norm_ext)?;

                on_progress(DemixProgress {
                    current_chunk: 100,
                    total_chunks: 100,
                    percentage: 100.0,
                    stage: DemixStage::ExportAudio {
                        file_name: inst_filename.clone(),
                    },
                    message: format!("Đang xuất file Instrumental ({}) ...", inst_filename),
                });
                self.save_audio_file(&inst_pcm, sample_rate, &inst_out_path, norm_ext)?;

                output_paths.push(vocal_out_path);
                output_paths.push(inst_out_path);
            }
            ModelKind::HtDemucs => {
                // Tách qua HT-Demucs v4 (Raw Waveform -> ONNX 4-stems)
                let stems_pcm =
                    self.separate_stereo_demucs(&left_channel, &right_channel, &mut on_progress)?;

                // 4 stems: 0 = drums, 1 = bass, 2 = other, 3 = vocals
                if self.output_stems == OutputStems::FourStems {
                    let stem_names = ["drums", "bass", "other", "vocals"];
                    for (idx, name) in stem_names.iter().enumerate() {
                        let filename = format!("{}.{}", name, norm_ext);
                        let out_path = target_dir.join(&filename);

                        on_progress(DemixProgress {
                            current_chunk: 100,
                            total_chunks: 100,
                            percentage: 100.0,
                            stage: DemixStage::ExportAudio {
                                file_name: filename.clone(),
                            },
                            message: format!("Đang xuất file {} ({}) ...", name, filename),
                        });

                        self.save_audio_file(&stems_pcm[idx], sample_rate, &out_path, norm_ext)?;
                        output_paths.push(out_path);
                    }
                } else {
                    // Mặc định: Gộp 4 track thành 2 track: vocal và instrumental
                    let total_samples = stems_pcm[0].len();
                    let mut inst_pcm = Vec::with_capacity(total_samples);
                    for i in 0..total_samples {
                        // Instrumental = drums + bass + other
                        let sum = stems_pcm[0][i] + stems_pcm[1][i] + stems_pcm[2][i];
                        inst_pcm.push(sum);
                    }
                    let vocal_pcm = &stems_pcm[3];

                    let vocal_filename = format!("vocal.{}", norm_ext);
                    let inst_filename = format!("instrumental.{}", norm_ext);

                    let vocal_out_path = target_dir.join(&vocal_filename);
                    let inst_out_path = target_dir.join(&inst_filename);

                    on_progress(DemixProgress {
                        current_chunk: 100,
                        total_chunks: 100,
                        percentage: 100.0,
                        stage: DemixStage::ExportAudio {
                            file_name: vocal_filename.clone(),
                        },
                        message: format!("Đang xuất file Vocal ({}) ...", vocal_filename),
                    });
                    self.save_audio_file(vocal_pcm, sample_rate, &vocal_out_path, norm_ext)?;

                    on_progress(DemixProgress {
                        current_chunk: 100,
                        total_chunks: 100,
                        percentage: 100.0,
                        stage: DemixStage::ExportAudio {
                            file_name: inst_filename.clone(),
                        },
                        message: format!("Đang xuất file Instrumental gộp ({}) ...", inst_filename),
                    });
                    self.save_audio_file(&inst_pcm, sample_rate, &inst_out_path, norm_ext)?;

                    output_paths.push(vocal_out_path);
                    output_paths.push(inst_out_path);
                }
            }
        }

        let _ = fs::remove_file(temp_pcm);

        on_progress(DemixProgress {
            current_chunk: 100,
            total_chunks: 100,
            percentage: 100.0,
            stage: DemixStage::Finished,
            message: "Tách âm thanh hoàn tất thành công!".to_string(),
        });

        Ok(output_paths)
    }

    /// Xử lý một file hoặc toàn bộ thư mục chứa các file video/audio
    pub fn process_path<P1, P2, F>(
        &self,
        input_path: P1,
        output_base_dir: P2,
        format: &str,
        mut on_progress: F,
    ) -> Result<Vec<Vec<PathBuf>>, SeparatorError>
    where
        P1: AsRef<Path>,
        P2: AsRef<Path>,
        F: FnMut(&Path, DemixProgress),
    {
        let path = input_path.as_ref();
        let mut results = Vec::new();

        let files: Vec<PathBuf> = if path.is_file() {
            vec![path.to_path_buf()]
        } else if path.is_dir() {
            let mut list = Vec::new();
            if let Ok(entries) = fs::read_dir(path) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_file() {
                        if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
                            let ext_lower = ext.to_lowercase();
                            if [
                                "mp4", "mkv", "avi", "mov", "flv", "webm", "ts",
                                "mp3", "wav", "aac", "flac", "m4a", "ogg", "wma",
                            ]
                            .contains(&ext_lower.as_str())
                            {
                                list.push(p);
                            }
                        }
                    }
                }
            }
            list.sort();
            list
        } else {
            return Err(SeparatorError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Đường dẫn không tồn tại: {:?}", path),
            )));
        };

        for file in files {
            let res = self.process_video_stems(
                &file,
                output_base_dir.as_ref(),
                format,
                |prog| on_progress(&file, prog),
            )?;
            results.push(res);
        }

        Ok(results)
    }

    /// Tách tín hiệu stereo (Left & Right) qua mô hình UVR-MDX-NET
    fn separate_stereo_mdx<F>(
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

            let (l_real, l_imag) = Self::stft_chunk(chunk_l, n_fft, hop, dim_t, dim_f, trim, &window, &fft);
            let (r_real, r_imag) = Self::stft_chunk(chunk_r, n_fft, hop, dim_t, dim_f, trim, &window, &fft);

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

            let (vocal_spec, inst_spec) = self.separate_spectrogram(input_spec)?;

            let (vl_chunk, vr_chunk) = Self::istft_stereo(&vocal_spec, n_fft, hop, dim_t, dim_f, trim, &window, &ifft);
            let (il_chunk, ir_chunk) = Self::istft_stereo(&inst_spec, n_fft, hop, dim_t, dim_f, trim, &window, &ifft);

            vocal_left.extend_from_slice(&vl_chunk[trim..trim + gen_size]);
            vocal_right.extend_from_slice(&vr_chunk[trim..trim + gen_size]);
            inst_left.extend_from_slice(&il_chunk[trim..trim + gen_size]);
            inst_right.extend_from_slice(&ir_chunk[trim..trim + gen_size]);

            let progress_percent = ((k + 1) as f32 / num_chunks as f32) * 100.0;
            on_progress(DemixProgress {
                current_chunk: k + 1,
                total_chunks: num_chunks,
                percentage: progress_percent,
                stage: DemixStage::Demixing {
                    current_chunk: k + 1,
                    total_chunks: num_chunks,
                },
                message: format!(
                    "Đang phân tách âm thanh qua MDX-Net (Chunk {}/{} - {:.1}%)",
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

    /// Tách tín hiệu stereo (Left & Right) qua mô hình HT-Demucs v4 (trả về 4 track stereo interleaved)
    fn separate_stereo_demucs<F>(
        &self,
        left: &[f32],
        right: &[f32],
        on_progress: &mut F,
    ) -> Result<Vec<Vec<f32>>, SeparatorError>
    where
        F: FnMut(DemixProgress),
    {
        let total_samples = left.len();
        let segment_len = 343980; // Chuẩn chunk size của HT-Demucs v4 (~7.8s @ 44.1kHz)
        let overlap_ratio = 0.25f32;
        let overlap_len = (segment_len as f32 * overlap_ratio) as usize; // 85995
        let hop = segment_len - overlap_len; // 257985

        // Tạo trọng số crossfade hình thang cho phần chồng lấn
        let mut weight = vec![1.0f32; segment_len];
        for i in 0..overlap_len {
            let w = (i as f32 + 1.0) / (overlap_len as f32 + 1.0);
            weight[i] = w;
            weight[segment_len - 1 - i] = w;
        }

        let num_chunks = if total_samples <= segment_len {
            1
        } else {
            (total_samples - overlap_len + hop - 1) / hop
        };

        // 4 stems: 0 = drums, 1 = bass, 2 = other, 3 = vocals
        let mut out_left = vec![vec![0.0f32; total_samples]; 4];
        let mut out_right = vec![vec![0.0f32; total_samples]; 4];
        let mut weight_sum = vec![0.0f32; total_samples];

        let (input_name, output_name) = {
            let session = self
                .session
                .lock()
                .map_err(|e| SeparatorError::ShapeError(e.to_string()))?;
            let in_n = session
                .inputs()
                .first()
                .map(|inp| inp.name().to_string())
                .unwrap_or_else(|| "mix".to_string());
            let out_n = session
                .outputs()
                .first()
                .map(|out| out.name().to_string())
                .unwrap_or_else(|| "output".to_string());
            (in_n, out_n)
        };

        for k in 0..num_chunks {
            let start = k * hop;
            let end = (start + segment_len).min(total_samples);
            let len = end - start;

            let mut input_tensor_data = Array3::<f32>::zeros((1, 2, segment_len));
            for i in 0..len {
                input_tensor_data[[0, 0, i]] = left[start + i];
                input_tensor_data[[0, 1, i]] = right[start + i];
            }

            let input_tensor = Value::from_array(input_tensor_data)?;

            let mut session = self
                .session
                .lock()
                .map_err(|e| SeparatorError::ShapeError(e.to_string()))?;

            let outputs = session.run(inputs![input_name.as_str() => input_tensor])?;

            let output_extracted = outputs[output_name.as_str()].try_extract_tensor::<f32>()?;
            let slice = output_extracted.1;

            // Phân giải tensor [1, 4, 2, 343980] và cộng dồn trọng số
            for s in 0..4 {
                for i in 0..len {
                    let w = weight[i];
                    let idx_l = s * (2 * segment_len) + i;
                    let idx_r = s * (2 * segment_len) + segment_len + i;

                    out_left[s][start + i] += slice[idx_l] * w;
                    out_right[s][start + i] += slice[idx_r] * w;
                }
            }

            for i in 0..len {
                weight_sum[start + i] += weight[i];
            }

            let progress_percent = ((k + 1) as f32 / num_chunks as f32) * 100.0;
            on_progress(DemixProgress {
                current_chunk: k + 1,
                total_chunks: num_chunks,
                percentage: progress_percent,
                stage: DemixStage::Demixing {
                    current_chunk: k + 1,
                    total_chunks: num_chunks,
                },
                message: format!(
                    "Đang phân tách âm thanh qua HT-Demucs (Chunk {}/{} - {:.1}%)",
                    k + 1,
                    num_chunks,
                    progress_percent
                ),
            });
        }

        // Chuẩn hóa theo tổng trọng số
        for i in 0..total_samples {
            let ws = weight_sum[i].max(1e-6);
            for s in 0..4 {
                out_left[s][i] /= ws;
                out_right[s][i] /= ws;
            }
        }

        // Ghép kênh Left & Right thành mảng interleaved stereo cho từng stem
        let mut stems_interleaved = Vec::with_capacity(4);
        for s in 0..4 {
            let mut pcm = Vec::with_capacity(total_samples * 2);
            for i in 0..total_samples {
                pcm.push(out_left[s][i]);
                pcm.push(out_right[s][i]);
            }
            stems_interleaved.push(pcm);
        }

        Ok(stems_interleaved)
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
        let chunk_size = hop * (dim_t - 1);
        let trim = n_fft / 2;
        let gen_size = chunk_size - 2 * trim;

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
