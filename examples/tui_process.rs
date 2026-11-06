use cli_gui_form::{FormWidget, ProgressStyle, StatusBarWidget, TaskWidget, Theme};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use demix::{AudioSeparator, DemixProgress, DemixStage};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    Terminal,
};
use std::{
    io::stdout,
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Duration,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = Path::new("models/UVR-MDX-NET-Inst_HQ_3.onnx");
    let video_input = Path::new("video_input");
    let output_dir = Path::new("output");

    if !model_path.exists() {
        eprintln!("Lỗi: Không tìm thấy model tại {:?}", model_path);
        return Ok(());
    }

    // Thiết lập terminal với raw mode cho TUI
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Shared state giữa worker thread và TUI rendering thread
    let progress_state = Arc::new(Mutex::new(DemixProgress {
        current_chunk: 0,
        total_chunks: 1,
        percentage: 0.0,
        stage: DemixStage::ExtractAudio,
        message: "Đang khởi tạo...".to_string(),
    }));

    let is_done = Arc::new(AtomicBool::new(false));
    let error_msg = Arc::new(Mutex::new(None::<String>));

    // Khởi chạy tác vụ phân tách âm thanh ở background thread
    let p_state_clone = Arc::clone(&progress_state);
    let done_clone = Arc::clone(&is_done);
    let err_clone = Arc::clone(&error_msg);

    let model_path_buf = model_path.to_path_buf();
    let video_input_buf = video_input.to_path_buf();
    let output_dir_buf = output_dir.to_path_buf();

    thread::spawn(move || {
        let run = || -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            let separator = AudioSeparator::new(&model_path_buf)?;
            separator.process_path(
                &video_input_buf,
                &output_dir_buf,
                "mp3",
                |_file, prog| {
                    if let Ok(mut state) = p_state_clone.lock() {
                        *state = prog;
                    }
                },
            )?;
            Ok(())
        };

        if let Err(e) = run() {
            if let Ok(mut err) = err_clone.lock() {
                *err = Some(e.to_string());
            }
        }
        done_clone.store(true, Ordering::SeqCst);
    });

    // Vòng lặp render TUI
    let status_bar = StatusBarWidget::new()
        .with_left("Demix Audio Separator | UVR-MDX-NET")
        .with_right("Press 'q' or Esc to Quit");

    loop {
        let (current_c, total_c, pct, msg) = {
            let state = progress_state.lock().unwrap();
            (
                state.current_chunk,
                state.total_chunks.max(1),
                state.percentage,
                state.message.clone(),
            )
        };

        let is_completed = is_done.load(Ordering::SeqCst);
        let error_occurred = error_msg.lock().unwrap().clone();

        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(2)
                .constraints([
                    Constraint::Length(3), // Tiêu đề / Header
                    Constraint::Length(5), // TaskWidget Progress Bar
                    Constraint::Min(2),    // Thông báo trạng thái
                    Constraint::Length(1), // StatusBarWidget
                ])
                .split(f.area());

            // 1. TaskWidget từ thư viện cli_gui_form
            let task_color = if error_occurred.is_some() {
                Theme::ERROR
            } else if is_completed {
                Theme::EMERALD
            } else {
                Theme::CYAN
            };

            let status_text = if let Some(err) = &error_occurred {
                format!("Error: {}", err)
            } else if is_completed {
                "Hoàn thành tất cả các file!".to_string()
            } else {
                msg.clone()
            };

            let task_widget = TaskWidget::new_progress(
                "demix_progress",
                "Neural Audio Demixer (UVR-MDX-NET-Inst_HQ_3)",
                current_c,
                total_c,
                "chunks",
                format!("{:.1}%", pct),
                status_text,
                task_color,
            )
            .with_progress_style(ProgressStyle::Line);

            task_widget.render(chunks[1], f);

            // 2. StatusBarWidget từ cli_gui_form ở dưới cùng
            status_bar.render(chunks[3], f);
        })?;

        // Bắt sự kiện bàn phím: ấn 'q' hoặc Esc để thoát
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    if key.code == KeyCode::Char('q') || key.code == KeyCode::Esc {
                        break;
                    }
                }
            }
        }

        if is_completed {
            // Cho hiển thị thêm 1.5s trước khi thoát hoặc chờ user bấm 'q'
            thread::sleep(Duration::from_millis(1500));
            break;
        }
    }

    // Phục hồi lại terminal ban đầu
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    println!("\n[DEMIX TUI] Xử lý hoàn tất. Kiểm tra thư mục: {:?}", output_dir);

    Ok(())
}
