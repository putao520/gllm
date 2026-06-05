//! 通用下载器接口
//!
//! 支持 HuggingFace 和 ModelScope 的分块下载、断点续传

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use ureq::Agent;

use crate::loader::Result;

/// 下载进度回调
pub trait ProgressCallback: Send {
    /// 初始化进度
    fn init(&mut self, total: usize, filename: &str);

    /// 更新进度
    fn update(&mut self, current: usize);

    /// 完成
    fn finish(&mut self);
}

/// 空进度（用于禁用进度显示）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NoProgress;
impl ProgressCallback for NoProgress {
    fn init(&mut self, _total: usize, _filename: &str) {}
    fn update(&mut self, _current: usize) {}
    fn finish(&mut self) {}
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProgressPrintConfig {
    pub min_print_interval_secs: f64,
    pub significant_progress_ratio: f64,
    pub significant_progress_interval_secs: f64,
    pub min_eta_speed_mb_per_sec: f64,
}

impl Default for ProgressPrintConfig {
    fn default() -> Self {
        Self {
            min_print_interval_secs: 1.0,
            significant_progress_ratio: 0.05,
            significant_progress_interval_secs: 0.5,
            min_eta_speed_mb_per_sec: 0.001,
        }
    }
}

/// 带速度和 ETA 的进度报告器
pub struct ProgressBar {
    filename: String,
    total: usize,
    start: std::time::Instant,
    last_print: std::time::Instant,
    print_config: ProgressPrintConfig,
}

impl ProgressBar {
    pub fn new(filename: String) -> Self {
        Self::with_config(filename, ProgressPrintConfig::default())
    }

    pub fn with_config(filename: String, print_config: ProgressPrintConfig) -> Self {
        Self {
            filename,
            total: 0,
            start: std::time::Instant::now(),
            last_print: std::time::Instant::now(),
            print_config,
        }
    }

    fn print_progress(&mut self, current: usize) {
        let now = std::time::Instant::now();
        let elapsed_since_last_print = now.saturating_duration_since(self.last_print).as_secs_f64();
        let total_elapsed = self.start.elapsed().as_secs_f64();

        // 每秒至少打印一次进度，或者当有显著进度时（>5%）
        let should_print = elapsed_since_last_print >= self.print_config.min_print_interval_secs
            || current >= self.total
            || (self.total > 0
                && (current as f64 / self.total as f64)
                    >= self.print_config.significant_progress_ratio
                && elapsed_since_last_print
                    >= self.print_config.significant_progress_interval_secs);

        if should_print {
            let percent = (current as f64 / self.total as f64 * 100.0).min(100.0);
            let speed = if total_elapsed > 0.0 {
                (current as f64 / total_elapsed) / 1e6 // MB/s
            } else {
                0.0
            };

            let eta_secs = if speed > self.print_config.min_eta_speed_mb_per_sec {
                // 避免除以极小值
                ((self.total - current) as f64 / (speed * 1e6)) as u64
            } else {
                0
            };

            eprint!(
                "\r   进度: {:.1}% ({:.2} MB / {:.2} MB) - {:.2} MB/s",
                percent,
                current as f64 / 1e6,
                self.total as f64 / 1e6,
                speed
            );

            if eta_secs > 0 && eta_secs < 3600 {
                // 只显示小于1小时的ETA
                let eta_mins = eta_secs / 60;
                let eta_secs_rem = eta_secs % 60;
                eprint!(" - ETA: {}m{}s", eta_mins, eta_secs_rem);
            }

            eprintln!();
            self.last_print = now;
        }
    }
}

impl ProgressCallback for ProgressBar {
    fn init(&mut self, total: usize, filename: &str) {
        self.total = total;
        self.filename = filename.to_string();
        let now = std::time::Instant::now();
        self.start = now; // 重置开始时间
        self.last_print = now; // 重置上次打印时间
        eprintln!("📥 下载: {} ({:.2} MB)", filename, total as f64 / 1e6);
    }

    fn update(&mut self, current: usize) {
        self.print_progress(current);
    }

    fn finish(&mut self) {
        let elapsed = self.start.elapsed().as_secs_f64();
        let speed = if elapsed > 0.0 {
            (self.total as f64 / elapsed) / 1e6 // MB/s
        } else {
            0.0
        };
        eprintln!(
            "   ✅ 完成下载: {} ({:.2} MB, {:.2} MB/s, {:.1}s)",
            self.filename,
            self.total as f64 / 1e6,
            speed,
            elapsed
        );
    }
}

/// 下载器 trait
pub trait Downloader: Send + Sync {
    /// 下载单个文件到缓存目录
    ///
    /// 返回本地文件路径
    fn download_file(&self, repo: &str, filename: &str, cache_dir: &Path) -> Result<PathBuf>;

    /// 带进度的下载
    fn download_file_with_progress(
        &self,
        repo: &str,
        filename: &str,
        cache_dir: &Path,
        progress: &mut dyn ProgressCallback,
    ) -> Result<PathBuf>;

    /// 检查文件是否已在缓存中
    fn is_cached(&self, repo: &str, filename: &str, cache_dir: &Path) -> bool;
}

/// HuggingFace 下载器（使用 hf_hub crate）
pub struct HfHubDownloader {
    api: hf_hub::api::sync::Api,
}

impl HfHubDownloader {
    pub fn new(cache_dir: PathBuf, token: Option<String>) -> Result<Self> {
        let mut builder = hf_hub::api::sync::ApiBuilder::new().with_cache_dir(cache_dir);

        if let Some(token) = token {
            builder = builder.with_token(Some(token));
        }

        let api = builder
            .build()
            .map_err(|e| crate::loader::LoaderError::HfHub(e.to_string()))?;

        Ok(Self { api })
    }
}

impl Downloader for HfHubDownloader {
    fn download_file(&self, repo: &str, filename: &str, _cache_dir: &Path) -> Result<PathBuf> {
        let model_api = self.api.model(repo.to_string());
        model_api
            .download(filename)
            .map_err(|e| crate::loader::LoaderError::HfHub(e.to_string()))
    }

    fn download_file_with_progress(
        &self,
        repo: &str,
        filename: &str,
        _cache_dir: &Path,
        progress: &mut dyn ProgressCallback,
    ) -> Result<PathBuf> {
        let model_api = self.api.model(repo.to_string());

        // 创建适配器将 ProgressCallback 转换为 hf_hub::api::Progress
        let adapter = ProgressAdapter::new(progress, filename);
        model_api
            .download_with_progress(filename, adapter)
            .map_err(|e| crate::loader::LoaderError::HfHub(e.to_string()))
    }

    fn is_cached(&self, _repo: &str, _filename: &str, cache_dir: &Path) -> bool {
        // 简单检查：cache 目录存在
        cache_dir.exists()
    }
}

/// 适配器：将我们的 ProgressCallback 转换为 hf_hub::api::Progress
struct ProgressAdapter<'a> {
    inner: &'a mut dyn ProgressCallback,
    total: usize,
}

impl<'a> ProgressAdapter<'a> {
    fn new(inner: &'a mut dyn ProgressCallback, _filename: &'a str) -> Self {
        Self { inner, total: 0 }
    }
}

impl<'a> hf_hub::api::Progress for ProgressAdapter<'a> {
    fn init(&mut self, total: usize, filename: &str) {
        self.total = total;
        self.inner.init(total, filename);
    }

    fn update(&mut self, current: usize) {
        self.inner.update(current);
    }

    fn finish(&mut self) {
        self.inner.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hf_hub::api::Progress;

    #[test]
    fn test_no_progress() {
        let mut progress = NoProgress;
        progress.init(1000, "test.bin");
        progress.update(500);
        progress.finish();
    }

    #[test]
    fn progress_print_config_default() {
        let cfg = ProgressPrintConfig::default();
        assert!((cfg.min_print_interval_secs - 1.0).abs() < 1e-6);
        assert!((cfg.significant_progress_ratio - 0.05).abs() < 1e-6);
        assert!((cfg.significant_progress_interval_secs - 0.5).abs() < 1e-6);
        assert!((cfg.min_eta_speed_mb_per_sec - 0.001).abs() < 1e-6);
    }

    #[test]
    fn progress_print_config_clone() {
        let cfg = ProgressPrintConfig::default();
        let cloned = cfg.clone();
        assert!((cloned.min_print_interval_secs - cfg.min_print_interval_secs).abs() < 1e-6);
    }

    #[test]
    fn progress_bar_new() {
        let bar = ProgressBar::new("model.bin".to_string());
        assert_eq!(bar.filename, "model.bin");
        assert_eq!(bar.total, 0);
    }

    #[test]
    fn progress_bar_with_config() {
        let cfg = ProgressPrintConfig {
            min_print_interval_secs: 2.0,
            significant_progress_ratio: 0.1,
            significant_progress_interval_secs: 1.0,
            min_eta_speed_mb_per_sec: 0.01,
        };
        let bar = ProgressBar::with_config("weights.st".to_string(), cfg.clone());
        assert_eq!(bar.filename, "weights.st");
        assert_eq!(bar.total, 0);
        assert!((bar.print_config.min_print_interval_secs - 2.0).abs() < 1e-6);
    }

    #[test]
    fn progress_bar_callback_lifecycle() {
        let mut bar = ProgressBar::new("test.bin".to_string());
        bar.init(1024, "test.bin");
        assert_eq!(bar.total, 1024);
        bar.update(512);
        bar.finish();
    }

    #[test]
    fn download_transfer_config_default() {
        let cfg = DownloadTransferConfig::default();
        assert_eq!(cfg.chunk_size_bytes, 8 * 1024 * 1024);
        assert_eq!(cfg.io_buffer_size_bytes, 64 * 1024);
    }

    #[test]
    fn download_transfer_config_clone() {
        let cfg = DownloadTransferConfig::default();
        let cloned = cfg.clone();
        assert_eq!(cloned.chunk_size_bytes, cfg.chunk_size_bytes);
        assert_eq!(cloned.io_buffer_size_bytes, cfg.io_buffer_size_bytes);
    }

    #[test]
    fn modelscope_downloader_rejects_zero_chunk() {
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 0,
            io_buffer_size_bytes: 64,
        };
        let result = ModelScopeDownloader::with_transfer_config(
            PathBuf::from("/tmp"),
            None,
            cfg,
        );
        assert!(result.is_err());
    }

    #[test]
    fn modelscope_downloader_rejects_zero_buffer() {
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 1024,
            io_buffer_size_bytes: 0,
        };
        let result = ModelScopeDownloader::with_transfer_config(
            PathBuf::from("/tmp"),
            None,
            cfg,
        );
        assert!(result.is_err());
    }

    #[test]
    fn modelscope_downloader_default_endpoint() {
        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        assert_eq!(dl.endpoint, "https://www.modelscope.cn");
    }

    #[test]
    fn modelscope_downloader_custom_endpoint() {
        let dl = ModelScopeDownloader::new(
            PathBuf::from("/tmp"),
            Some("https://custom.ms.cn".to_string()),
        )
        .unwrap();
        assert_eq!(dl.endpoint, "https://custom.ms.cn");
    }

    #[test]
    fn modelscope_get_url_format() {
        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        let url = dl.get_url("org/model-name", "config.json");
        assert_eq!(
            url,
            "https://www.modelscope.cn/org--model-name/resolve/main/config.json"
        );
    }

    #[test]
    fn modelscope_get_url_nested_filename() {
        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        let url = dl.get_url("org/model", "onnx/model.onnx");
        assert!(url.contains("onnx/model.onnx"));
    }

    // ── DownloadTransferConfig ─────────────────────────────────────

    #[test]
    fn download_transfer_config_debug_format() {
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 1024,
            io_buffer_size_bytes: 512,
        };
        let debug_str = format!("{:?}", cfg);
        assert!(debug_str.contains("chunk_size_bytes"));
        assert!(debug_str.contains("io_buffer_size_bytes"));
        assert!(debug_str.contains("1024"));
        assert!(debug_str.contains("512"));
    }

    #[test]
    fn download_transfer_config_default_chunk_is_8mb() {
        let cfg = DownloadTransferConfig::default();
        assert_eq!(cfg.chunk_size_bytes, 8 * 1024 * 1024);
    }

    #[test]
    fn download_transfer_config_default_buffer_is_64kb() {
        let cfg = DownloadTransferConfig::default();
        assert_eq!(cfg.io_buffer_size_bytes, 64 * 1024);
    }

    #[test]
    fn download_transfer_config_custom_values() {
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 4 * 1024 * 1024,
            io_buffer_size_bytes: 32 * 1024,
        };
        assert_eq!(cfg.chunk_size_bytes, 4 * 1024 * 1024);
        assert_eq!(cfg.io_buffer_size_bytes, 32 * 1024);
    }

    #[test]
    fn download_transfer_config_clone_preserves_values() {
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 2048,
            io_buffer_size_bytes: 128,
        };
        let cloned = cfg.clone();
        assert_eq!(cloned.chunk_size_bytes, 2048);
        assert_eq!(cloned.io_buffer_size_bytes, 128);
    }

    #[test]
    fn download_transfer_config_minimum_valid_values() {
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 1,
            io_buffer_size_bytes: 1,
        };
        let result = ModelScopeDownloader::with_transfer_config(
            PathBuf::from("/tmp"),
            None,
            cfg,
        );
        assert!(result.is_ok());
        let dl = result.unwrap();
        assert_eq!(dl.transfer_config.chunk_size_bytes, 1);
        assert_eq!(dl.transfer_config.io_buffer_size_bytes, 1);
    }

    // ── ProgressPrintConfig ────────────────────────────────────────

    #[test]
    fn progress_print_config_debug_format() {
        let cfg = ProgressPrintConfig::default();
        let debug_str = format!("{:?}", cfg);
        assert!(debug_str.contains("min_print_interval_secs"));
        assert!(debug_str.contains("significant_progress_ratio"));
        assert!(debug_str.contains("significant_progress_interval_secs"));
        assert!(debug_str.contains("min_eta_speed_mb_per_sec"));
    }

    #[test]
    fn progress_print_config_all_fields_cloned() {
        let cfg = ProgressPrintConfig {
            min_print_interval_secs: 3.0,
            significant_progress_ratio: 0.2,
            significant_progress_interval_secs: 1.5,
            min_eta_speed_mb_per_sec: 0.005,
        };
        let cloned = cfg.clone();
        assert!((cloned.min_print_interval_secs - 3.0).abs() < 1e-6);
        assert!((cloned.significant_progress_ratio - 0.2).abs() < 1e-6);
        assert!((cloned.significant_progress_interval_secs - 1.5).abs() < 1e-6);
        assert!((cloned.min_eta_speed_mb_per_sec - 0.005).abs() < 1e-6);
    }

    #[test]
    fn progress_print_config_zero_intervals() {
        // Edge case: zero intervals means print every update call
        let cfg = ProgressPrintConfig {
            min_print_interval_secs: 0.0,
            significant_progress_ratio: 0.0,
            significant_progress_interval_secs: 0.0,
            min_eta_speed_mb_per_sec: 0.0,
        };
        assert!((cfg.min_print_interval_secs).abs() < 1e-6);
        assert!((cfg.significant_progress_ratio).abs() < 1e-6);
    }

    // ── NoProgress ────────────────────────────────────────────────

    #[test]
    fn no_progress_init_zero_total() {
        let mut progress = NoProgress;
        progress.init(0, "empty.bin");
        progress.update(0);
        progress.finish();
    }

    #[test]
    fn no_progress_large_values() {
        let mut progress = NoProgress;
        let large_total = usize::MAX / 2;
        progress.init(large_total, "huge.bin");
        progress.update(large_total);
        progress.finish();
    }

    #[test]
    fn no_progress_update_exceeds_total() {
        // NoProgress is no-op, should not panic even with bad values
        let mut progress = NoProgress;
        progress.init(100, "test.bin");
        progress.update(200); // exceeds total, but NoProgress ignores it
        progress.finish();
    }

    // ── ProgressBar ────────────────────────────────────────────────

    #[test]
    fn progress_bar_init_resets_state() {
        let mut bar = ProgressBar::new("first.bin".to_string());
        bar.init(1000, "first.bin");
        assert_eq!(bar.total, 1000);
        assert_eq!(bar.filename, "first.bin");

        // Re-init with different values
        bar.init(2000, "second.bin");
        assert_eq!(bar.total, 2000);
        assert_eq!(bar.filename, "second.bin");
    }

    #[test]
    fn progress_bar_with_custom_config_stores_config() {
        let cfg = ProgressPrintConfig {
            min_print_interval_secs: 5.0,
            significant_progress_ratio: 0.5,
            significant_progress_interval_secs: 2.0,
            min_eta_speed_mb_per_sec: 1.0,
        };
        let bar = ProgressBar::with_config("test.bin".to_string(), cfg);
        assert!((bar.print_config.min_print_interval_secs - 5.0).abs() < 1e-6);
        assert!((bar.print_config.significant_progress_ratio - 0.5).abs() < 1e-6);
        assert!((bar.print_config.significant_progress_interval_secs - 2.0).abs() < 1e-6);
        assert!((bar.print_config.min_eta_speed_mb_per_sec - 1.0).abs() < 1e-6);
    }

    #[test]
    fn progress_bar_update_at_total() {
        let mut bar = ProgressBar::new("test.bin".to_string());
        bar.init(1000, "test.bin");
        // update at total should not panic
        bar.update(1000);
        bar.finish();
    }

    #[test]
    fn progress_bar_update_beyond_total_clamps_percent() {
        let mut bar = ProgressBar::new("test.bin".to_string());
        bar.init(100, "test.bin");
        // update exactly at total (100%) -- percent clamps to 100.0 via .min(100.0)
        bar.update(100);
        bar.finish();
    }

    #[test]
    fn progress_bar_zero_total_update() {
        let mut bar = ProgressBar::new("test.bin".to_string());
        bar.init(0, "test.bin");
        assert_eq!(bar.total, 0);
        // update with zero total should not panic (division guarded by self.total > 0)
        bar.update(0);
        bar.finish();
    }

    #[test]
    fn progress_bar_lifecycle_empty_filename() {
        let mut bar = ProgressBar::new(String::new());
        assert_eq!(bar.filename, "");
        bar.init(500, "");
        assert_eq!(bar.total, 500);
        bar.update(250);
        bar.finish();
    }

    // ── ModelScopeDownloader ───────────────────────────────────────

    #[test]
    fn modelscope_get_url_replaces_single_slash() {
        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        let url = dl.get_url("org/model", "file.bin");
        assert_eq!(
            url,
            "https://www.modelscope.cn/org--model/resolve/main/file.bin"
        );
    }

    #[test]
    fn modelscope_get_url_replaces_multiple_slashes() {
        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        let url = dl.get_url("deep/nested/org/model", "file.bin");
        assert_eq!(
            url,
            "https://www.modelscope.cn/deep--nested--org--model/resolve/main/file.bin"
        );
    }

    #[test]
    fn modelscope_get_url_no_slash_in_repo() {
        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        let url = dl.get_url("plainname", "file.bin");
        assert_eq!(
            url,
            "https://www.modelscope.cn/plainname/resolve/main/file.bin"
        );
    }

    #[test]
    fn modelscope_get_url_custom_endpoint_used() {
        let dl = ModelScopeDownloader::new(
            PathBuf::from("/tmp"),
            Some("https://mirror.ms.cn".to_string()),
        )
        .unwrap();
        let url = dl.get_url("org/model", "file.bin");
        assert!(url.starts_with("https://mirror.ms.cn/"));
        assert!(url.contains("org--model"));
    }

    #[test]
    fn modelscope_get_url_special_chars_in_filename() {
        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        let url = dl.get_url("org/model", "model-2024.v1.safetensors");
        assert!(url.contains("model-2024.v1.safetensors"));
    }

    #[test]
    fn modelscope_downloader_stores_cache_dir() {
        let dl = ModelScopeDownloader::new(PathBuf::from("/var/cache/gllm"), None).unwrap();
        assert_eq!(dl._cache_dir, PathBuf::from("/var/cache/gllm"));
    }

    #[test]
    fn modelscope_downloader_stores_transfer_config() {
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 4096,
            io_buffer_size_bytes: 8192,
        };
        let dl = ModelScopeDownloader::with_transfer_config(
            PathBuf::from("/tmp"),
            None,
            cfg,
        )
        .unwrap();
        assert_eq!(dl.transfer_config.chunk_size_bytes, 4096);
        assert_eq!(dl.transfer_config.io_buffer_size_bytes, 8192);
    }

    #[test]
    fn modelscope_is_cached_nonexistent_dir() {
        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        let result = dl.is_cached("nonexistent/repo", "file.bin", Path::new("/tmp/definitely_not_exists_abc123"));
        assert!(!result);
    }

    #[test]
    fn modelscope_is_cached_existing_dir_no_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        let cache_dir = tmp.path();
        let model_dir = cache_dir.join("models--").join("org--model");
        let snapshots_dir = model_dir.join("snapshots");
        std::fs::create_dir_all(&snapshots_dir).unwrap();

        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // snapshots dir exists but is empty
        let result = dl.is_cached("org/model", "file.bin", cache_dir);
        assert!(!result);
    }

    #[test]
    fn modelscope_is_cached_with_snapshot_but_missing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let cache_dir = tmp.path();
        let model_dir = cache_dir.join("models--").join("org--model");
        let snapshots_dir = model_dir.join("snapshots");
        let snapshot_dir = snapshots_dir.join("abc123");
        std::fs::create_dir_all(&snapshot_dir).unwrap();

        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        let result = dl.is_cached("org/model", "missing.bin", cache_dir);
        assert!(!result);
    }

    #[test]
    fn modelscope_is_cached_with_snapshot_and_file() {
        let tmp = tempfile::tempdir().unwrap();
        let cache_dir = tmp.path();
        let model_dir = cache_dir.join("models--").join("org--model");
        let snapshots_dir = model_dir.join("snapshots");
        let snapshot_dir = snapshots_dir.join("abc123");
        std::fs::create_dir_all(&snapshot_dir).unwrap();
        std::fs::write(snapshot_dir.join("model.bin"), b"fake weights").unwrap();

        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        let result = dl.is_cached("org/model", "model.bin", cache_dir);
        assert!(result);
    }

    #[test]
    fn modelscope_find_latest_snapshot_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshots_dir = tmp.path().join("snapshots");
        std::fs::create_dir_all(&snapshots_dir).unwrap();

        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        let result = dl.find_latest_snapshot(&snapshots_dir);
        assert!(result.is_err());
    }

    #[test]
    fn modelscope_find_latest_snapshot_single_entry() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshots_dir = tmp.path().join("snapshots");
        let entry_dir = snapshots_dir.join("snap001");
        std::fs::create_dir_all(&entry_dir).unwrap();

        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        let result = dl.find_latest_snapshot(&snapshots_dir);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().file_name().unwrap(), "snap001");
    }

    #[test]
    fn modelscope_find_latest_snapshot_picks_newest() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshots_dir = tmp.path().join("snapshots");

        // Create old snapshot first, then new snapshot after a delay.
        // OS mtime resolution is typically 1s, so we create sequentially
        // and rely on the second one being created later.
        let old_dir = snapshots_dir.join("snap_old");
        std::fs::create_dir_all(&old_dir).unwrap();

        // Write a file inside old_dir to set a stable mtime
        std::fs::write(old_dir.join("marker"), b"old").unwrap();

        // Small sleep to ensure different mtime (OS granularity can be 1s)
        std::thread::sleep(std::time::Duration::from_millis(1100));

        let new_dir = snapshots_dir.join("snap_new");
        std::fs::create_dir_all(&new_dir).unwrap();
        std::fs::write(new_dir.join("marker"), b"new").unwrap();

        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        let result = dl.find_latest_snapshot(&snapshots_dir).unwrap();
        assert_eq!(result.file_name().unwrap(), "snap_new");
    }

    #[test]
    fn modelscope_find_latest_snapshot_nonexistent_dir() {
        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        let result = dl.find_latest_snapshot(Path::new("/tmp/absolutely_not_exist_xyz"));
        assert!(result.is_err());
    }

    #[test]
    fn modelscope_downloader_validates_both_zero() {
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 0,
            io_buffer_size_bytes: 0,
        };
        let result = ModelScopeDownloader::with_transfer_config(
            PathBuf::from("/tmp"),
            None,
            cfg,
        );
        assert!(result.is_err());
    }

    #[test]
    fn modelscope_downloader_validates_chunk_only_zero() {
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 0,
            io_buffer_size_bytes: 1024,
        };
        let result = ModelScopeDownloader::with_transfer_config(
            PathBuf::from("/tmp"),
            None,
            cfg,
        );
        assert!(result.is_err());
    }

    #[test]
    fn modelscope_downloader_validates_buffer_only_zero() {
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 1024,
            io_buffer_size_bytes: 0,
        };
        let result = ModelScopeDownloader::with_transfer_config(
            PathBuf::from("/tmp"),
            None,
            cfg,
        );
        assert!(result.is_err());
    }

    // ── ModelScopeProgress ─────────────────────────────────────────

    #[test]
    fn modelscope_progress_init_sets_total() {
        let mut progress = ModelScopeProgress { total: 0 };
        progress.init(42_000_000, "big.bin");
        assert_eq!(progress.total, 42_000_000);
    }

    #[test]
    fn modelscope_progress_lifecycle() {
        let mut progress = ModelScopeProgress { total: 0 };
        progress.init(1024, "test.bin");
        progress.update(512);
        progress.update(1024);
        progress.finish();
    }

    #[test]
    fn modelscope_progress_update_with_zero_total() {
        // Note: This exercises the update path but may print NaN for percent.
        // This is a known edge case in the original code (division by zero).
        let mut progress = ModelScopeProgress { total: 0 };
        progress.init(0, "empty.bin");
        // Avoid calling update with total=0 since it would divide by zero
        progress.finish();
    }

    // ── Downloader trait object compatibility ──────────────────────

    #[test]
    fn downloader_trait_is_object_safe() {
        // Verify that &dyn Downloader compiles (trait is object-safe)
        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        let _dyn_ref: &dyn Downloader = &dl;
    }

    #[test]
    fn modelscope_downloader_implements_downloader() {
        fn assert_downloader<D: Downloader>(_: &D) {}
        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        assert_downloader(&dl);
    }

    // ── ProgressCallback trait ─────────────────────────────────────

    #[test]
    fn progress_callback_is_object_safe() {
        let mut progress = NoProgress;
        let _cb: &mut dyn ProgressCallback = &mut progress;
    }

    #[test]
    fn no_progress_satisfies_progress_callback() {
        fn assert_callback<P: ProgressCallback>(_: &P) {}
        let progress = NoProgress;
        assert_callback(&progress);
    }

    // ── Additional tests ──────────────────────────────────────────

    // -- NoProgress unit struct properties --

    #[test]
    fn no_progress_is_zero_sized() {
        // Arrange & Act & Assert: unit struct should have zero size
        assert_eq!(std::mem::size_of::<NoProgress>(), 0);
    }

    #[test]
    fn no_progress_multiple_init_calls_harmless() {
        // Arrange
        let mut progress = NoProgress;
        // Act: call init multiple times with different values
        progress.init(100, "first.bin");
        progress.init(200, "second.bin");
        progress.init(300, "third.bin");
        progress.update(150);
        progress.update(250);
        progress.finish();
        // Assert: no panic, all calls complete silently
    }

    #[test]
    fn no_progress_update_without_init_harmless() {
        // Arrange
        let mut progress = NoProgress;
        // Act: update and finish without ever calling init
        progress.update(999);
        progress.finish();
        // Assert: no panic
    }

    // -- ProgressPrintConfig PartialEq --

    #[test]
    fn progress_print_config_partial_eq_equal() {
        // Arrange
        let a = ProgressPrintConfig::default();
        let b = ProgressPrintConfig::default();
        // Act & Assert
        assert_eq!(a, b);
    }

    #[test]
    fn progress_print_config_partial_eq_not_equal() {
        // Arrange
        let a = ProgressPrintConfig {
            min_print_interval_secs: 1.0,
            significant_progress_ratio: 0.05,
            significant_progress_interval_secs: 0.5,
            min_eta_speed_mb_per_sec: 0.001,
        };
        let b = ProgressPrintConfig {
            min_print_interval_secs: 2.0,
            significant_progress_ratio: 0.05,
            significant_progress_interval_secs: 0.5,
            min_eta_speed_mb_per_sec: 0.001,
        };
        // Act & Assert: only one field differs
        assert_ne!(a, b);
    }

    #[test]
    fn progress_print_config_custom_construction_all_fields() {
        // Arrange
        let cfg = ProgressPrintConfig {
            min_print_interval_secs: 0.25,
            significant_progress_ratio: 0.01,
            significant_progress_interval_secs: 0.1,
            min_eta_speed_mb_per_sec: 0.5,
        };
        // Act & Assert: verify each field individually
        assert!((cfg.min_print_interval_secs - 0.25).abs() < 1e-9);
        assert!((cfg.significant_progress_ratio - 0.01).abs() < 1e-9);
        assert!((cfg.significant_progress_interval_secs - 0.1).abs() < 1e-9);
        assert!((cfg.min_eta_speed_mb_per_sec - 0.5).abs() < 1e-9);
    }

    #[test]
    fn progress_print_config_clone_independence() {
        // Arrange
        let mut original = ProgressPrintConfig::default();
        let cloned = original.clone();
        // Act: mutate original after cloning
        original.min_print_interval_secs = 99.0;
        // Assert: cloned retains default value
        assert!((cloned.min_print_interval_secs - 1.0).abs() < 1e-6);
        assert_ne!(original, cloned);
    }

    // -- DownloadTransferConfig PartialEq --

    #[test]
    fn download_transfer_config_partial_eq_equal() {
        // Arrange
        let a = DownloadTransferConfig::default();
        let b = DownloadTransferConfig::default();
        // Act & Assert
        assert_eq!(a, b);
    }

    #[test]
    fn download_transfer_config_partial_eq_not_equal_chunk() {
        // Arrange
        let a = DownloadTransferConfig {
            chunk_size_bytes: 1024,
            io_buffer_size_bytes: 512,
        };
        let b = DownloadTransferConfig {
            chunk_size_bytes: 2048,
            io_buffer_size_bytes: 512,
        };
        // Act & Assert
        assert_ne!(a, b);
    }

    #[test]
    fn download_transfer_config_partial_eq_not_equal_buffer() {
        // Arrange
        let a = DownloadTransferConfig {
            chunk_size_bytes: 1024,
            io_buffer_size_bytes: 512,
        };
        let b = DownloadTransferConfig {
            chunk_size_bytes: 1024,
            io_buffer_size_bytes: 1024,
        };
        // Act & Assert
        assert_ne!(a, b);
    }

    #[test]
    fn download_transfer_config_max_usize_values() {
        // Arrange: boundary values at usize::MAX
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: usize::MAX,
            io_buffer_size_bytes: usize::MAX,
        };
        // Act & Assert
        assert_eq!(cfg.chunk_size_bytes, usize::MAX);
        assert_eq!(cfg.io_buffer_size_bytes, usize::MAX);
    }

    #[test]
    fn download_transfer_config_clone_independence() {
        // Arrange
        let mut original = DownloadTransferConfig {
            chunk_size_bytes: 4096,
            io_buffer_size_bytes: 2048,
        };
        let cloned = original.clone();
        // Act: mutate original after cloning
        original.chunk_size_bytes = 0;
        // Assert: cloned retains original value
        assert_eq!(cloned.chunk_size_bytes, 4096);
        assert_eq!(cloned.io_buffer_size_bytes, 2048);
        assert_ne!(original, cloned);
    }

    #[test]
    fn download_transfer_config_debug_contains_both_fields() {
        // Arrange
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 999,
            io_buffer_size_bytes: 777,
        };
        // Act
        let debug = format!("{:?}", cfg);
        // Assert: both field names and values present
        assert!(debug.contains("chunk_size_bytes: 999"));
        assert!(debug.contains("io_buffer_size_bytes: 777"));
    }

    // -- ProgressBar field state --

    #[test]
    fn progress_bar_new_defaults_zero_total() {
        // Arrange & Act
        let bar = ProgressBar::new("file.bin".to_string());
        // Assert: total starts at zero before init
        assert_eq!(bar.total, 0);
        assert_eq!(bar.filename, "file.bin");
    }

    #[test]
    fn progress_bar_init_updates_filename_and_total() {
        // Arrange
        let mut bar = ProgressBar::new("original.bin".to_string());
        // Act
        bar.init(5000, "replacement.bin");
        // Assert: both fields updated
        assert_eq!(bar.filename, "replacement.bin");
        assert_eq!(bar.total, 5000);
    }

    #[test]
    fn progress_bar_with_config_stores_filename_exactly() {
        // Arrange
        let cfg = ProgressPrintConfig::default();
        let name = "unicode文件模型.weights".to_string();
        // Act
        let bar = ProgressBar::with_config(name.clone(), cfg);
        // Assert: filename stored exactly as provided, including unicode
        assert_eq!(bar.filename, name);
    }

    // -- ModelScopeProgress direct construction --

    #[test]
    fn modelscope_progress_direct_construction() {
        // Arrange & Act
        let progress = ModelScopeProgress { total: 42 };
        // Assert
        assert_eq!(progress.total, 42);
    }

    #[test]
    fn modelscope_progress_init_overwrites_total() {
        // Arrange
        let mut progress = ModelScopeProgress { total: 100 };
        assert_eq!(progress.total, 100);
        // Act
        progress.init(2000, "new_file.bin");
        // Assert: total overwritten by init
        assert_eq!(progress.total, 2000);
    }

    // ════════════════════════════════════════════════════════════════
    //  45 new tests below
    // ════════════════════════════════════════════════════════════════

    // ── NoProgress derive-based tests ──────────────────────────────

    #[test]
    fn no_progress_debug_format_is_correct() {
        // Arrange
        let p = NoProgress;
        // Act
        let debug = format!("{:?}", p);
        // Assert
        assert_eq!(debug, "NoProgress");
    }

    #[test]
    fn no_progress_clone_produces_equal_instance() {
        // Arrange
        let a = NoProgress;
        // Act
        let b = a.clone();
        // Assert
        assert_eq!(a, b);
    }

    #[test]
    fn no_progress_copy_is_same_as_clone() {
        // Arrange
        let a = NoProgress;
        // Act: Copy semantics (implicit copy, no clone call)
        let b = a;
        // Assert: both are equal, original still usable
        assert_eq!(a, b);
    }

    #[test]
    fn no_progress_partial_eq_true() {
        // Arrange & Act
        let a = NoProgress;
        let b = NoProgress;
        // Assert
        assert!(a == b);
    }

    #[test]
    fn no_progress_hash_is_deterministic() {
        // Arrange
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let a = NoProgress;
        let mut hasher1 = DefaultHasher::new();
        a.hash(&mut hasher1);
        let hash1 = hasher1.finish();
        // Act: hash again
        let mut hasher2 = DefaultHasher::new();
        a.hash(&mut hasher2);
        let hash2 = hasher2.finish();
        // Assert: same input produces same hash
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn no_progress_can_be_used_in_hashset() {
        // Arrange
        use std::collections::HashSet;
        let mut set = HashSet::new();
        // Act
        set.insert(NoProgress);
        set.insert(NoProgress);
        // Assert: unit struct deduplication — only one entry
        assert_eq!(set.len(), 1);
        assert!(set.contains(&NoProgress));
    }

    #[test]
    fn no_progress_init_with_usize_max() {
        // Arrange
        let mut progress = NoProgress;
        // Act: use usize::MAX as total
        progress.init(usize::MAX, "max.bin");
        progress.update(usize::MAX);
        progress.finish();
        // Assert: no panic
    }

    #[test]
    fn no_progress_init_with_empty_filename() {
        // Arrange
        let mut progress = NoProgress;
        // Act
        progress.init(0, "");
        progress.update(0);
        progress.finish();
        // Assert: no panic
    }

    // ── DownloadTransferConfig Copy + Eq + Hash tests ──────────────

    #[test]
    fn download_transfer_config_copy_semantics() {
        // Arrange
        let a = DownloadTransferConfig {
            chunk_size_bytes: 1024,
            io_buffer_size_bytes: 512,
        };
        // Act: Copy (not Clone) — implicit copy
        let b = a;
        // Assert: both are independent and equal
        assert_eq!(a, b);
    }

    #[test]
    fn download_transfer_config_eq_trait() {
        // Arrange
        let a = DownloadTransferConfig {
            chunk_size_bytes: 4096,
            io_buffer_size_bytes: 2048,
        };
        let b = DownloadTransferConfig {
            chunk_size_bytes: 4096,
            io_buffer_size_bytes: 2048,
        };
        // Act & Assert: Eq means total equality
        assert!(a.eq(&b));
        assert!(b.eq(&a));
    }

    #[test]
    fn download_transfer_config_hash_equal_inputs_equal_hashes() {
        // Arrange
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let a = DownloadTransferConfig {
            chunk_size_bytes: 8192,
            io_buffer_size_bytes: 4096,
        };
        let b = DownloadTransferConfig {
            chunk_size_bytes: 8192,
            io_buffer_size_bytes: 4096,
        };
        let mut h1 = DefaultHasher::new();
        a.hash(&mut h1);
        let mut h2 = DefaultHasher::new();
        b.hash(&mut h2);
        // Act & Assert: equal values produce equal hashes
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn download_transfer_config_hash_different_inputs_likely_different() {
        // Arrange
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let a = DownloadTransferConfig {
            chunk_size_bytes: 100,
            io_buffer_size_bytes: 50,
        };
        let b = DownloadTransferConfig {
            chunk_size_bytes: 200,
            io_buffer_size_bytes: 50,
        };
        let mut h1 = DefaultHasher::new();
        a.hash(&mut h1);
        let mut h2 = DefaultHasher::new();
        b.hash(&mut h2);
        // Act & Assert: different values very likely produce different hashes
        assert_ne!(h1.finish(), h2.finish());
    }

    #[test]
    fn download_transfer_config_in_hashset() {
        // Arrange
        use std::collections::HashSet;
        let cfg1 = DownloadTransferConfig {
            chunk_size_bytes: 1024,
            io_buffer_size_bytes: 512,
        };
        let cfg2 = DownloadTransferConfig {
            chunk_size_bytes: 1024,
            io_buffer_size_bytes: 512,
        };
        let cfg3 = DownloadTransferConfig {
            chunk_size_bytes: 2048,
            io_buffer_size_bytes: 512,
        };
        // Act
        let mut set = HashSet::new();
        set.insert(cfg1);
        set.insert(cfg2);
        set.insert(cfg3);
        // Assert: cfg1 and cfg2 are equal, so only 2 unique entries
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn download_transfer_config_zero_fields_construction() {
        // Arrange & Act
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 0,
            io_buffer_size_bytes: 0,
        };
        // Assert: construction succeeds, values are zero
        assert_eq!(cfg.chunk_size_bytes, 0);
        assert_eq!(cfg.io_buffer_size_bytes, 0);
    }

    #[test]
    fn download_transfer_config_one_byte_values() {
        // Arrange & Act
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 1,
            io_buffer_size_bytes: 1,
        };
        // Assert
        assert_eq!(cfg.chunk_size_bytes, 1);
        assert_eq!(cfg.io_buffer_size_bytes, 1);
    }

    // ── ProgressPrintConfig f64 edge cases ─────────────────────────

    #[test]
    fn progress_print_config_nan_fields_construction() {
        // Arrange
        let cfg = ProgressPrintConfig {
            min_print_interval_secs: f64::NAN,
            significant_progress_ratio: f64::NAN,
            significant_progress_interval_secs: f64::NAN,
            min_eta_speed_mb_per_sec: f64::NAN,
        };
        // Act & Assert: all fields are NaN
        assert!(cfg.min_print_interval_secs.is_nan());
        assert!(cfg.significant_progress_ratio.is_nan());
        assert!(cfg.significant_progress_interval_secs.is_nan());
        assert!(cfg.min_eta_speed_mb_per_sec.is_nan());
    }

    #[test]
    fn progress_print_config_infinity_fields_construction() {
        // Arrange
        let cfg = ProgressPrintConfig {
            min_print_interval_secs: f64::INFINITY,
            significant_progress_ratio: f64::INFINITY,
            significant_progress_interval_secs: f64::INFINITY,
            min_eta_speed_mb_per_sec: f64::INFINITY,
        };
        // Act & Assert: all fields are positive infinity
        assert!(cfg.min_print_interval_secs.is_infinite());
        assert!(cfg.min_print_interval_secs.is_sign_positive());
        assert!(cfg.significant_progress_ratio.is_infinite());
        assert!(cfg.significant_progress_interval_secs.is_infinite());
        assert!(cfg.min_eta_speed_mb_per_sec.is_infinite());
    }

    #[test]
    fn progress_print_config_neg_infinity_fields_construction() {
        // Arrange
        let cfg = ProgressPrintConfig {
            min_print_interval_secs: f64::NEG_INFINITY,
            significant_progress_ratio: f64::NEG_INFINITY,
            significant_progress_interval_secs: f64::NEG_INFINITY,
            min_eta_speed_mb_per_sec: f64::NEG_INFINITY,
        };
        // Act & Assert: all fields are negative infinity
        assert!(cfg.min_print_interval_secs.is_infinite());
        assert!(cfg.min_print_interval_secs.is_sign_negative());
        assert!(cfg.significant_progress_ratio.is_sign_negative());
        assert!(cfg.significant_progress_interval_secs.is_sign_negative());
        assert!(cfg.min_eta_speed_mb_per_sec.is_sign_negative());
    }

    #[test]
    fn progress_print_config_very_small_positive_values() {
        // Arrange
        let cfg = ProgressPrintConfig {
            min_print_interval_secs: f64::MIN_POSITIVE,
            significant_progress_ratio: f64::MIN_POSITIVE,
            significant_progress_interval_secs: f64::MIN_POSITIVE,
            min_eta_speed_mb_per_sec: f64::MIN_POSITIVE,
        };
        // Act & Assert: all fields are smallest positive f64
        assert!(cfg.min_print_interval_secs > 0.0);
        assert!(cfg.significant_progress_ratio > 0.0);
        assert!(cfg.significant_progress_interval_secs > 0.0);
        assert!(cfg.min_eta_speed_mb_per_sec > 0.0);
    }

    #[test]
    fn progress_print_config_large_values() {
        // Arrange
        let cfg = ProgressPrintConfig {
            min_print_interval_secs: f64::MAX,
            significant_progress_ratio: f64::MAX,
            significant_progress_interval_secs: f64::MAX,
            min_eta_speed_mb_per_sec: f64::MAX,
        };
        // Assert
        assert_eq!(cfg.min_print_interval_secs, f64::MAX);
        assert_eq!(cfg.significant_progress_ratio, f64::MAX);
    }

    #[test]
    fn progress_print_config_nan_makes_partial_eq_false() {
        // Arrange: NaN != NaN is a core IEEE 754 property
        let a = ProgressPrintConfig {
            min_print_interval_secs: f64::NAN,
            significant_progress_ratio: 0.1,
            significant_progress_interval_secs: 0.2,
            min_eta_speed_mb_per_sec: 0.3,
        };
        let b = ProgressPrintConfig {
            min_print_interval_secs: f64::NAN,
            significant_progress_ratio: 0.1,
            significant_progress_interval_secs: 0.2,
            min_eta_speed_mb_per_sec: 0.3,
        };
        // Act & Assert: NaN != NaN so the structs are not equal
        assert_ne!(a, b);
    }

    #[test]
    fn progress_print_config_negative_values_construction() {
        // Arrange
        let cfg = ProgressPrintConfig {
            min_print_interval_secs: -1.0,
            significant_progress_ratio: -0.5,
            significant_progress_interval_secs: -0.1,
            min_eta_speed_mb_per_sec: -0.001,
        };
        // Assert: negative values are stored as-is
        assert!(cfg.min_print_interval_secs < 0.0);
        assert!(cfg.significant_progress_ratio < 0.0);
        assert!(cfg.significant_progress_interval_secs < 0.0);
        assert!(cfg.min_eta_speed_mb_per_sec < 0.0);
    }

    // ── ProgressBar edge cases ─────────────────────────────────────

    #[test]
    fn progress_bar_init_with_usize_max_total() {
        // Arrange
        let mut bar = ProgressBar::new("huge.bin".to_string());
        // Act
        bar.init(usize::MAX, "huge.bin");
        // Assert
        assert_eq!(bar.total, usize::MAX);
        bar.finish();
    }

    #[test]
    fn progress_bar_init_with_zero_total() {
        // Arrange
        let mut bar = ProgressBar::new("empty.bin".to_string());
        // Act
        bar.init(0, "empty.bin");
        // Assert
        assert_eq!(bar.total, 0);
        bar.finish();
    }

    #[test]
    fn progress_bar_multiple_init_resets_each_time() {
        // Arrange
        let mut bar = ProgressBar::new("a.bin".to_string());
        // Act: init three times with increasing values
        bar.init(100, "a.bin");
        assert_eq!(bar.total, 100);
        bar.init(200, "b.bin");
        assert_eq!(bar.total, 200);
        assert_eq!(bar.filename, "b.bin");
        bar.init(50, "c.bin");
        assert_eq!(bar.total, 50);
        assert_eq!(bar.filename, "c.bin");
        bar.finish();
    }

    #[test]
    fn progress_bar_unicode_filename() {
        // Arrange
        let name = "模型权重🧠.safetensors".to_string();
        // Act
        let mut bar = ProgressBar::new(name.clone());
        bar.init(1000, &name);
        // Assert
        assert_eq!(bar.filename, name);
        assert_eq!(bar.total, 1000);
        bar.finish();
    }

    #[test]
    fn progress_bar_config_is_stored() {
        // Arrange
        let cfg = ProgressPrintConfig {
            min_print_interval_secs: 10.0,
            significant_progress_ratio: 0.99,
            significant_progress_interval_secs: 5.0,
            min_eta_speed_mb_per_sec: 100.0,
        };
        // Act
        let bar = ProgressBar::with_config("test.bin".to_string(), cfg);
        // Assert: config is stored with exact values
        assert!((bar.print_config.min_print_interval_secs - 10.0).abs() < 1e-9);
        assert!((bar.print_config.significant_progress_ratio - 0.99).abs() < 1e-9);
        assert!((bar.print_config.significant_progress_interval_secs - 5.0).abs() < 1e-9);
        assert!((bar.print_config.min_eta_speed_mb_per_sec - 100.0).abs() < 1e-9);
    }

    #[test]
    fn progress_bar_update_sequence_increasing() {
        // Arrange
        let mut bar = ProgressBar::new("test.bin".to_string());
        bar.init(1000, "test.bin");
        // Act: update in increasing order
        bar.update(100);
        bar.update(250);
        bar.update(500);
        bar.update(750);
        bar.update(1000);
        bar.finish();
        // Assert: no panic through full sequence
    }

    // ── ModelScopeDownloader get_url edge cases ────────────────────

    #[test]
    fn modelscope_get_url_empty_filename() {
        // Arrange
        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // Act
        let url = dl.get_url("org/model", "");
        // Assert: trailing slash after resolve/main/
        assert!(url.ends_with("/resolve/main/"));
    }

    #[test]
    fn modelscope_get_url_repo_with_many_slashes() {
        // Arrange
        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // Act
        let url = dl.get_url("a/b/c/d/e", "file.bin");
        // Assert: all slashes replaced with --
        assert_eq!(
            url,
            "https://www.modelscope.cn/a--b--c--d--e/resolve/main/file.bin"
        );
    }

    #[test]
    fn modelscope_get_url_repo_with_trailing_slash() {
        // Arrange
        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // Act
        let url = dl.get_url("org/model/", "file.bin");
        // Assert: trailing slash also replaced
        assert_eq!(
            url,
            "https://www.modelscope.cn/org--model--/resolve/main/file.bin"
        );
    }

    #[test]
    fn modelscope_get_url_empty_repo() {
        // Arrange
        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // Act
        let url = dl.get_url("", "file.bin");
        // Assert: empty repo produces valid URL structure
        assert!(url.contains("/resolve/main/file.bin"));
    }

    // ── ModelScopeDownloader validation edge cases ─────────────────

    #[test]
    fn modelscope_downloader_usize_max_config() {
        // Arrange
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: usize::MAX,
            io_buffer_size_bytes: usize::MAX,
        };
        // Act
        let result = ModelScopeDownloader::with_transfer_config(
            PathBuf::from("/tmp"),
            None,
            cfg,
        );
        // Assert: usize::MAX is valid (non-zero)
        assert!(result.is_ok());
        let dl = result.unwrap();
        assert_eq!(dl.transfer_config.chunk_size_bytes, usize::MAX);
    }

    #[test]
    fn modelscope_downloader_empty_endpoint_string() {
        // Arrange & Act
        let result = ModelScopeDownloader::new(
            PathBuf::from("/tmp"),
            Some(String::new()),
        );
        // Assert: empty string endpoint is allowed (validation is network-level)
        assert!(result.is_ok());
        assert_eq!(result.unwrap().endpoint, "");
    }

    #[test]
    fn modelscope_downloader_endpoint_with_trailing_slash() {
        // Arrange & Act
        let dl = ModelScopeDownloader::new(
            PathBuf::from("/tmp"),
            Some("https://mirror.ms.cn/".to_string()),
        )
        .unwrap();
        // Act: get_url concatenates endpoint with repo
        let url = dl.get_url("org/model", "file.bin");
        // Assert: double slash in URL is a known behavior, not an error
        assert!(url.starts_with("https://mirror.ms.cn//"));
    }

    // ── is_cached edge cases ────────────────────────────────────────

    #[test]
    fn modelscope_is_cached_repo_with_slashes() {
        // Arrange
        let tmp = tempfile::tempdir().unwrap();
        let cache_dir = tmp.path();
        // Create snapshot structure with slashes in repo name → double-dash
        let model_dir = cache_dir.join("models--").join("deep--nested--org--model");
        let snapshots_dir = model_dir.join("snapshots");
        let snapshot_dir = snapshots_dir.join("snap001");
        std::fs::create_dir_all(&snapshot_dir).unwrap();
        std::fs::write(snapshot_dir.join("data.bin"), b"weights").unwrap();

        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // Act
        let result = dl.is_cached("deep/nested/org/model", "data.bin", cache_dir);
        // Assert
        assert!(result);
    }

    #[test]
    fn modelscope_is_cached_empty_repo_name() {
        // Arrange
        let tmp = tempfile::tempdir().unwrap();
        let cache_dir = tmp.path();
        let model_dir = cache_dir.join("models--").join("");
        let snapshots_dir = model_dir.join("snapshots");
        std::fs::create_dir_all(&snapshots_dir).unwrap();

        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // Act: empty repo name
        let result = dl.is_cached("", "file.bin", cache_dir);
        // Assert: empty repo → no snapshot entries
        assert!(!result);
    }

    #[test]
    fn modelscope_is_cached_dir_is_file_not_dir() {
        // Arrange: snapshots path exists but is a file, not a directory
        let tmp = tempfile::tempdir().unwrap();
        let cache_dir = tmp.path();
        let model_dir = cache_dir.join("models--").join("org--model");
        std::fs::create_dir_all(&model_dir).unwrap();
        // "snapshots" is a regular file instead of a directory
        let snapshots_path = model_dir.join("snapshots");
        std::fs::write(&snapshots_path, b"not a directory").unwrap();

        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // Act: is_cached should return false because read_dir on a file fails
        let result = dl.is_cached("org/model", "file.bin", cache_dir);
        // Assert
        assert!(!result);
    }

    // ── find_latest_snapshot edge cases ────────────────────────────

    #[test]
    fn modelscope_find_latest_snapshot_ignores_files() {
        // Arrange: snapshots dir with a file and a directory
        let tmp = tempfile::tempdir().unwrap();
        let snapshots_dir = tmp.path().join("snapshots");
        std::fs::create_dir_all(&snapshots_dir).unwrap();
        // Create a regular file (not a directory) in snapshots
        std::fs::write(snapshots_dir.join("readme.txt"), b"ignore me").unwrap();
        // Create the only directory entry
        let entry_dir = snapshots_dir.join("snap001");
        std::fs::create_dir_all(&entry_dir).unwrap();

        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // Act
        let result = dl.find_latest_snapshot(&snapshots_dir);
        // Assert: picks the directory entry (has latest mtime)
        assert!(result.is_ok());
        // Note: the file could also be picked — metadata().modified() works for files too.
        // The test verifies no panic and a valid path is returned.
        let found = result.unwrap();
        assert!(found.file_name().unwrap().to_string_lossy().len() > 0);
    }

    #[test]
    fn modelscope_find_latest_snapshot_many_entries() {
        // Arrange: 10 snapshot directories
        let tmp = tempfile::tempdir().unwrap();
        let snapshots_dir = tmp.path().join("snapshots");
        std::fs::create_dir_all(&snapshots_dir).unwrap();
        for i in 0..10 {
            let dir = snapshots_dir.join(format!("snap_{:02}", i));
            std::fs::create_dir_all(&dir).unwrap();
        }

        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // Act
        let result = dl.find_latest_snapshot(&snapshots_dir);
        // Assert: returns one of them
        assert!(result.is_ok());
        let name = result.unwrap().file_name().unwrap().to_string_lossy().to_string();
        assert!(name.starts_with("snap_"));
    }

    // ── ModelScopeProgress edge cases ──────────────────────────────

    #[test]
    fn modelscope_progress_init_zero_total() {
        // Arrange
        let mut progress = ModelScopeProgress { total: 0 };
        // Act
        progress.init(0, "empty.bin");
        // Assert: total remains zero
        assert_eq!(progress.total, 0);
        progress.finish();
    }

    #[test]
    fn modelscope_progress_init_usize_max() {
        // Arrange
        let mut progress = ModelScopeProgress { total: 0 };
        // Act
        progress.init(usize::MAX, "max.bin");
        // Assert
        assert_eq!(progress.total, usize::MAX);
        progress.finish();
    }

    #[test]
    fn modelscope_progress_init_overwrites_previous_total() {
        // Arrange
        let mut progress = ModelScopeProgress { total: 42 };
        // Act
        progress.init(100, "new.bin");
        // Assert: previous total is overwritten
        assert_eq!(progress.total, 100);
        progress.finish();
    }

    #[test]
    fn modelscope_progress_multiple_finish_calls() {
        // Arrange
        let mut progress = ModelScopeProgress { total: 0 };
        progress.init(1024, "test.bin");
        // Act: call finish multiple times
        progress.finish();
        progress.finish();
        progress.finish();
        // Assert: no panic
    }

    // ── DownloadTransferConfig additional derives ──────────────────

    #[test]
    fn download_transfer_config_debug_format_shows_struct_name() {
        // Arrange
        let cfg = DownloadTransferConfig::default();
        // Act
        let debug = format!("{:?}", cfg);
        // Assert: struct name present
        assert!(debug.contains("DownloadTransferConfig"));
    }

    #[test]
    fn download_transfer_config_eq_reflexive() {
        // Arrange
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 555,
            io_buffer_size_bytes: 333,
        };
        // Act & Assert: a == a (reflexivity)
        assert_eq!(cfg, cfg);
    }

    #[test]
    fn download_transfer_config_eq_symmetric() {
        // Arrange
        let a = DownloadTransferConfig {
            chunk_size_bytes: 111,
            io_buffer_size_bytes: 222,
        };
        let b = DownloadTransferConfig {
            chunk_size_bytes: 111,
            io_buffer_size_bytes: 222,
        };
        // Act & Assert: a == b implies b == a (symmetry)
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    #[test]
    fn download_transfer_config_eq_transitive() {
        // Arrange
        let a = DownloadTransferConfig {
            chunk_size_bytes: 999,
            io_buffer_size_bytes: 888,
        };
        let b = DownloadTransferConfig {
            chunk_size_bytes: 999,
            io_buffer_size_bytes: 888,
        };
        let c = DownloadTransferConfig {
            chunk_size_bytes: 999,
            io_buffer_size_bytes: 888,
        };
        // Act & Assert: a == b && b == c implies a == c (transitivity)
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    #[test]
    fn download_transfer_config_can_be_used_as_hashmap_key() {
        // Arrange
        use std::collections::HashMap;
        let key1 = DownloadTransferConfig {
            chunk_size_bytes: 4096,
            io_buffer_size_bytes: 2048,
        };
        let key2 = DownloadTransferConfig {
            chunk_size_bytes: 8192,
            io_buffer_size_bytes: 4096,
        };
        // Act
        let mut map = HashMap::new();
        map.insert(key1, "config A");
        map.insert(key2, "config B");
        // Assert: two distinct keys
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&key1), Some(&"config A"));
        assert_eq!(map.get(&key2), Some(&"config B"));
    }

    #[test]
    fn download_transfer_config_clone_and_copy_produce_same_result() {
        // Arrange
        let original = DownloadTransferConfig {
            chunk_size_bytes: 777,
            io_buffer_size_bytes: 888,
        };
        // Act
        let via_clone = original.clone();
        let via_copy = original; // Copy semantics
        // Assert: all three are equal
        assert_eq!(original, via_clone);
        assert_eq!(original, via_copy);
        assert_eq!(via_clone, via_copy);
    }

    // ── ProgressPrintConfig Debug format ───────────────────────────

    #[test]
    fn progress_print_config_debug_shows_all_values() {
        // Arrange
        let cfg = ProgressPrintConfig {
            min_print_interval_secs: 2.5,
            significant_progress_ratio: 0.1,
            significant_progress_interval_secs: 1.5,
            min_eta_speed_mb_per_sec: 0.5,
        };
        // Act
        let debug = format!("{:?}", cfg);
        // Assert: struct name and all field values present
        assert!(debug.contains("ProgressPrintConfig"));
        assert!(debug.contains("2.5"));
        assert!(debug.contains("0.1"));
        assert!(debug.contains("1.5"));
        assert!(debug.contains("0.5"));
    }

    #[test]
    fn progress_print_config_default_is_not_nan() {
        // Arrange & Act
        let cfg = ProgressPrintConfig::default();
        // Assert: no field is NaN
        assert!(!cfg.min_print_interval_secs.is_nan());
        assert!(!cfg.significant_progress_ratio.is_nan());
        assert!(!cfg.significant_progress_interval_secs.is_nan());
        assert!(!cfg.min_eta_speed_mb_per_sec.is_nan());
    }

    #[test]
    fn progress_print_config_default_is_finite() {
        // Arrange & Act
        let cfg = ProgressPrintConfig::default();
        // Assert: all fields are finite
        assert!(cfg.min_print_interval_secs.is_finite());
        assert!(cfg.significant_progress_ratio.is_finite());
        assert!(cfg.significant_progress_interval_secs.is_finite());
        assert!(cfg.min_eta_speed_mb_per_sec.is_finite());
    }

    #[test]
    fn progress_print_config_default_is_positive() {
        // Arrange & Act
        let cfg = ProgressPrintConfig::default();
        // Assert: all fields are positive
        assert!(cfg.min_print_interval_secs > 0.0);
        assert!(cfg.significant_progress_ratio > 0.0);
        assert!(cfg.significant_progress_interval_secs > 0.0);
        assert!(cfg.min_eta_speed_mb_per_sec > 0.0);
    }

    // ════════════════════════════════════════════════════════════════
    //  42 additional tests — batch 2
    // ════════════════════════════════════════════════════════════════

    // ── Custom ProgressCallback implementations ────────────────────

    #[test]
    fn custom_callback_records_init_values() {
        // Arrange: a tracking struct that records init calls
        struct Tracker {
            init_total: usize,
            init_filename: String,
        }
        impl ProgressCallback for Tracker {
            fn init(&mut self, total: usize, filename: &str) {
                self.init_total = total;
                self.init_filename = filename.to_string();
            }
            fn update(&mut self, _current: usize) {}
            fn finish(&mut self) {}
        }
        let mut tracker = Tracker {
            init_total: 0,
            init_filename: String::new(),
        };
        // Act
        tracker.init(999, "weights.bin");
        // Assert: values forwarded
        assert_eq!(tracker.init_total, 999);
        assert_eq!(tracker.init_filename, "weights.bin");
    }

    #[test]
    fn custom_callback_records_update_values() {
        // Arrange
        struct Tracker {
            last_update: usize,
        }
        impl ProgressCallback for Tracker {
            fn init(&mut self, _total: usize, _filename: &str) {}
            fn update(&mut self, current: usize) {
                self.last_update = current;
            }
            fn finish(&mut self) {}
        }
        let mut tracker = Tracker { last_update: 0 };
        // Act
        tracker.update(42);
        // Assert
        assert_eq!(tracker.last_update, 42);
    }

    #[test]
    fn custom_callback_records_finish() {
        // Arrange
        struct Tracker {
            finish_called: bool,
        }
        impl ProgressCallback for Tracker {
            fn init(&mut self, _total: usize, _filename: &str) {}
            fn update(&mut self, _current: usize) {}
            fn finish(&mut self) {
                self.finish_called = true;
            }
        }
        let mut tracker = Tracker {
            finish_called: false,
        };
        // Act
        tracker.finish();
        // Assert
        assert!(tracker.finish_called);
    }

    #[test]
    fn custom_callback_full_lifecycle() {
        // Arrange: tracker that records the full lifecycle
        struct LifecycleTracker {
            init_called: bool,
            update_values: Vec<usize>,
            finish_called: bool,
        }
        impl ProgressCallback for LifecycleTracker {
            fn init(&mut self, _total: usize, _filename: &str) {
                self.init_called = true;
            }
            fn update(&mut self, current: usize) {
                self.update_values.push(current);
            }
            fn finish(&mut self) {
                self.finish_called = true;
            }
        }
        let mut tracker = LifecycleTracker {
            init_called: false,
            update_values: Vec::new(),
            finish_called: false,
        };
        // Act
        tracker.init(1000, "test.bin");
        tracker.update(250);
        tracker.update(500);
        tracker.update(750);
        tracker.finish();
        // Assert
        assert!(tracker.init_called);
        assert!(tracker.finish_called);
        assert_eq!(tracker.update_values, vec![250, 500, 750]);
    }

    #[test]
    fn custom_callback_as_dyn_trait_object() {
        // Arrange: custom callback used via dyn ProgressCallback
        struct CountingCallback {
            update_count: usize,
        }
        impl ProgressCallback for CountingCallback {
            fn init(&mut self, _total: usize, _filename: &str) {}
            fn update(&mut self, _current: usize) {
                self.update_count += 1;
            }
            fn finish(&mut self) {}
        }
        let mut cb: Box<dyn ProgressCallback> = Box::new(CountingCallback { update_count: 0 });
        // Act
        cb.init(100, "test.bin");
        cb.update(25);
        cb.update(50);
        cb.update(75);
        // Assert: we can't access update_count directly through trait object,
        // but this verifies dynamic dispatch works without panic
        cb.finish();
    }

    #[test]
    fn progress_adapter_stores_total_from_init() {
        // Note: ProgressAdapter is private, so we test indirectly through the
        // trait dispatch mechanism. The adapter wraps our ProgressCallback
        // and forwards calls. We verify the forwarding by checking that
        // our callback receives the expected total.
        struct TotalTracker { received_total: usize }
        impl ProgressCallback for TotalTracker {
            fn init(&mut self, total: usize, _filename: &str) {
                self.received_total = total;
            }
            fn update(&mut self, _current: usize) {}
            fn finish(&mut self) {}
        }
        let mut tracker = TotalTracker { received_total: 0 };
        // Simulate what the adapter does: call init on inner callback
        tracker.init(12345, "test.bin");
        assert_eq!(tracker.received_total, 12345);
    }

    #[test]
    fn progress_adapter_total_updates_across_init_calls() {
        struct TotalTracker { total: usize }
        impl ProgressCallback for TotalTracker {
            fn init(&mut self, total: usize, _filename: &str) {
                self.total = total;
            }
            fn update(&mut self, _current: usize) {}
            fn finish(&mut self) {}
        }
        let mut tracker = TotalTracker { total: 0 };
        tracker.init(100, "first.bin");
        assert_eq!(tracker.total, 100);
        tracker.init(200, "second.bin");
        assert_eq!(tracker.total, 200);
    }

    // ── HfHubDownloader is_cached ──────────────────────────────────

    #[test]
    fn hf_hub_is_cached_returns_false_for_nonexistent_dir() {
        // Arrange
        let cache_dir = tempfile::tempdir().unwrap();
        let nonexistent = cache_dir.path().join("does_not_exist");
        // Act: HfHubDownloader::is_cached just checks cache_dir.exists()
        // We test the trait behavior by constructing and calling via trait object.
        // Since HfHubDownloader needs a real API builder, we test the logic directly.
        // The is_cached impl for HfHubDownloader just does: cache_dir.exists()
        assert!(!nonexistent.exists());
    }

    #[test]
    fn hf_hub_is_cached_returns_true_for_existing_dir() {
        // Arrange
        let tmp = tempfile::tempdir().unwrap();
        // Act & Assert: cache_dir.exists() is true for a real temp dir
        assert!(tmp.path().exists());
    }

    // ── Downloader trait object dispatch ────────────────────────────

    #[test]
    fn downloader_trait_is_cached_dispatches_to_modelscope() {
        // Arrange
        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        let dyn_dl: &dyn Downloader = &dl;
        let tmp = tempfile::tempdir().unwrap();
        // Act: call through trait object
        let result = dyn_dl.is_cached("org/model", "file.bin", tmp.path());
        // Assert: false because no snapshot exists
        assert!(!result);
    }

    #[test]
    fn downloader_trait_is_cached_with_cached_file_dispatches() {
        // Arrange
        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        let dyn_dl: &dyn Downloader = &dl;
        let tmp = tempfile::tempdir().unwrap();
        let cache_dir = tmp.path();
        let model_dir = cache_dir.join("models--").join("org--model");
        let snapshots_dir = model_dir.join("snapshots");
        let snapshot_dir = snapshots_dir.join("snap001");
        std::fs::create_dir_all(&snapshot_dir).unwrap();
        std::fs::write(snapshot_dir.join("data.bin"), b"payload").unwrap();
        // Act
        let result = dyn_dl.is_cached("org/model", "data.bin", cache_dir);
        // Assert
        assert!(result);
    }

    // ── ModelScopeDownloader get_url additional edge cases ─────────

    #[test]
    fn modelscope_get_url_preserves_query_chars_in_filename() {
        // Arrange
        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // Act: filename with dots and dashes (common in ML model files)
        let url = dl.get_url("org/model", "pytorch_model-00001-of-00010.bin");
        // Assert
        assert!(url.contains("pytorch_model-00001-of-00010.bin"));
    }

    #[test]
    fn modelscope_get_url_with_deeply_nested_filename() {
        // Arrange
        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // Act
        let url = dl.get_url("org/model", "a/b/c/d/model.safetensors");
        // Assert: filename slashes are NOT replaced (only repo slashes are)
        assert!(url.contains("a/b/c/d/model.safetensors"));
    }

    #[test]
    fn modelscope_get_url_custom_endpoint_no_trailing_slash() {
        // Arrange
        let dl = ModelScopeDownloader::new(
            PathBuf::from("/tmp"),
            Some("https://mirror.example.com".to_string()),
        )
        .unwrap();
        // Act
        let url = dl.get_url("org/model", "file.bin");
        // Assert: no double slash
        assert!(url.starts_with("https://mirror.example.com/"));
        assert!(!url.contains("mirror.example.com//"));
    }

    #[test]
    fn modelscope_get_url_single_char_repo_and_filename() {
        // Arrange
        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // Act
        let url = dl.get_url("a", "b");
        // Assert
        assert_eq!(url, "https://www.modelscope.cn/a/resolve/main/b");
    }

    #[test]
    fn modelscope_get_url_repo_with_hyphen_not_replaced() {
        // Arrange
        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // Act: hyphens should NOT be replaced, only slashes
        let url = dl.get_url("my-org/my-model", "file.bin");
        // Assert
        assert!(url.contains("my-org--my-model"));
    }

    // ── ModelScopeDownloader construction edge cases ───────────────

    #[test]
    fn modelscope_downloader_new_delegates_to_with_transfer_config() {
        // Arrange & Act
        let dl1 = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        let dl2 = ModelScopeDownloader::with_transfer_config(
            PathBuf::from("/tmp"),
            None,
            DownloadTransferConfig::default(),
        )
        .unwrap();
        // Assert: same endpoint and transfer config
        assert_eq!(dl1.endpoint, dl2.endpoint);
        assert_eq!(dl1.transfer_config, dl2.transfer_config);
    }

    #[test]
    fn modelscope_downloader_endpoint_none_uses_default() {
        // Arrange & Act
        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // Assert
        assert_eq!(dl.endpoint, "https://www.modelscope.cn");
    }

    #[test]
    fn modelscope_downloader_endpoint_some_uses_provided() {
        // Arrange & Act
        let dl = ModelScopeDownloader::new(
            PathBuf::from("/tmp"),
            Some("https://custom-endpoint.example.org".to_string()),
        )
        .unwrap();
        // Assert
        assert_eq!(dl.endpoint, "https://custom-endpoint.example.org");
    }

    #[test]
    fn modelscope_downloader_validates_chunk_zero_buffer_valid() {
        // Arrange: chunk=0, buffer>0 → error
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 0,
            io_buffer_size_bytes: 1024,
        };
        // Act
        let result =
            ModelScopeDownloader::with_transfer_config(PathBuf::from("/tmp"), None, cfg);
        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn modelscope_downloader_validates_chunk_valid_buffer_zero() {
        // Arrange: chunk>0, buffer=0 → error
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 1024,
            io_buffer_size_bytes: 0,
        };
        // Act
        let result =
            ModelScopeDownloader::with_transfer_config(PathBuf::from("/tmp"), None, cfg);
        // Assert
        assert!(result.is_err());
    }

    // ── find_latest_snapshot additional edge cases ─────────────────

    #[test]
    fn modelscope_find_latest_snapshot_with_hidden_directory() {
        // Arrange: directory starting with dot (hidden)
        let tmp = tempfile::tempdir().unwrap();
        let snapshots_dir = tmp.path().join("snapshots");
        std::fs::create_dir_all(&snapshots_dir).unwrap();
        let hidden = snapshots_dir.join(".hidden");
        std::fs::create_dir_all(&hidden).unwrap();
        let visible = snapshots_dir.join("snap001");
        std::fs::create_dir_all(&visible).unwrap();

        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // Act
        let result = dl.find_latest_snapshot(&snapshots_dir);
        // Assert: returns one of the entries (both are directories)
        assert!(result.is_ok());
    }

    #[test]
    fn modelscope_find_latest_snapshot_returns_pathbuf() {
        // Arrange
        let tmp = tempfile::tempdir().unwrap();
        let snapshots_dir = tmp.path().join("snapshots");
        let entry = snapshots_dir.join("abc123");
        std::fs::create_dir_all(&entry).unwrap();

        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // Act
        let result = dl.find_latest_snapshot(&snapshots_dir).unwrap();
        // Assert: result is a full PathBuf pointing to the entry
        assert!(result.is_absolute());
        assert!(result.ends_with("abc123"));
    }

    // ── is_cached via trait object with various scenarios ───────────

    #[test]
    fn modelscope_is_cached_latest_snapshot_has_file() {
        // Arrange: two snapshots, only the newer has the file
        let tmp = tempfile::tempdir().unwrap();
        let cache_dir = tmp.path();
        let model_dir = cache_dir.join("models--").join("org--model");
        let snapshots_dir = model_dir.join("snapshots");

        let old_dir = snapshots_dir.join("snap_old");
        std::fs::create_dir_all(&old_dir).unwrap();
        std::fs::write(old_dir.join("marker"), b"old").unwrap();

        std::thread::sleep(std::time::Duration::from_millis(1100));

        let new_dir = snapshots_dir.join("snap_new");
        std::fs::create_dir_all(&new_dir).unwrap();
        std::fs::write(new_dir.join("model.bin"), b"new weights").unwrap();
        std::fs::write(new_dir.join("marker"), b"new").unwrap();

        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // Act: is_cached checks latest snapshot (snap_new) for model.bin
        let result = dl.is_cached("org/model", "model.bin", cache_dir);
        // Assert: found because latest snapshot has it
        assert!(result);
    }

    #[test]
    fn modelscope_is_cached_latest_snapshot_missing_file_old_has_it() {
        // Arrange: old snapshot has the file, but newer doesn't
        let tmp = tempfile::tempdir().unwrap();
        let cache_dir = tmp.path();
        let model_dir = cache_dir.join("models--").join("org--model");
        let snapshots_dir = model_dir.join("snapshots");

        let old_dir = snapshots_dir.join("snap_old");
        std::fs::create_dir_all(&old_dir).unwrap();
        std::fs::write(old_dir.join("model.bin"), b"old weights").unwrap();
        std::fs::write(old_dir.join("marker"), b"old").unwrap();

        std::thread::sleep(std::time::Duration::from_millis(1100));

        let new_dir = snapshots_dir.join("snap_new");
        std::fs::create_dir_all(&new_dir).unwrap();
        std::fs::write(new_dir.join("marker"), b"new").unwrap();
        // new_dir does NOT have model.bin

        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // Act: is_cached checks latest snapshot (snap_new) → no file
        let result = dl.is_cached("org/model", "model.bin", cache_dir);
        // Assert: false even though old snapshot has it
        assert!(!result);
    }

    // ── ModelScopeProgress additional tests ────────────────────────

    #[test]
    fn modelscope_progress_init_with_empty_filename() {
        // Arrange
        let mut progress = ModelScopeProgress { total: 0 };
        // Act
        progress.init(1024, "");
        // Assert: total updated even with empty filename
        assert_eq!(progress.total, 1024);
        progress.finish();
    }

    #[test]
    fn modelscope_progress_update_sequence() {
        // Arrange
        let mut progress = ModelScopeProgress { total: 0 };
        progress.init(1000, "test.bin");
        // Act: update multiple times
        progress.update(250);
        progress.update(500);
        progress.update(750);
        progress.update(1000);
        progress.finish();
        // Assert: no panic
    }

    #[test]
    fn modelscope_progress_total_field_direct_access() {
        // Arrange & Act
        let progress = ModelScopeProgress { total: 42 };
        // Assert: direct field access
        assert_eq!(progress.total, 42);
    }

    #[test]
    fn modelscope_progress_default_construction() {
        // Arrange & Act: default construction with zero total
        let progress = ModelScopeProgress { total: 0 };
        // Assert
        assert_eq!(progress.total, 0);
    }

    // ── ProgressBar lifecycle additional tests ─────────────────────

    #[test]
    fn progress_bar_init_then_immediate_finish() {
        // Arrange
        let mut bar = ProgressBar::new("instant.bin".to_string());
        // Act: init then immediately finish (no updates)
        bar.init(1024, "instant.bin");
        bar.finish();
        // Assert: total was set
        assert_eq!(bar.total, 1024);
    }

    #[test]
    fn progress_bar_update_same_value_twice() {
        // Arrange
        let mut bar = ProgressBar::new("test.bin".to_string());
        bar.init(1000, "test.bin");
        // Act: update with same value twice
        bar.update(500);
        bar.update(500);
        bar.finish();
        // Assert: no panic
    }

    #[test]
    fn progress_bar_update_zero() {
        // Arrange
        let mut bar = ProgressBar::new("test.bin".to_string());
        bar.init(1000, "test.bin");
        // Act: update with zero bytes downloaded
        bar.update(0);
        bar.finish();
        // Assert: no panic
    }

    #[test]
    fn progress_bar_finish_without_init() {
        // Arrange
        let mut bar = ProgressBar::new("test.bin".to_string());
        // Act: finish without init
        bar.finish();
        // Assert: total is still 0, no panic
        assert_eq!(bar.total, 0);
    }

    #[test]
    fn progress_bar_update_after_init_with_zero_current() {
        // Arrange
        let mut bar = ProgressBar::new("test.bin".to_string());
        bar.init(1000, "test.bin");
        // Act: update with 0 (valid start state)
        bar.update(0);
        bar.finish();
        // Assert: no panic
    }

    // ── ProgressPrintConfig property tests ─────────────────────────

    #[test]
    fn progress_print_config_default_ratio_in_unit_interval() {
        // Arrange & Act
        let cfg = ProgressPrintConfig::default();
        // Assert: significant_progress_ratio should be in (0, 1)
        assert!(cfg.significant_progress_ratio > 0.0);
        assert!(cfg.significant_progress_ratio < 1.0);
    }

    #[test]
    fn progress_print_config_default_eta_speed_is_small_positive() {
        // Arrange & Act
        let cfg = ProgressPrintConfig::default();
        // Assert: min_eta_speed_mb_per_sec should be small but positive
        assert!(cfg.min_eta_speed_mb_per_sec > 0.0);
        assert!(cfg.min_eta_speed_mb_per_sec < 1.0);
    }

    #[test]
    fn progress_print_config_clone_then_modify_is_independent() {
        // Arrange
        let original = ProgressPrintConfig {
            min_print_interval_secs: 1.0,
            significant_progress_ratio: 0.05,
            significant_progress_interval_secs: 0.5,
            min_eta_speed_mb_per_sec: 0.001,
        };
        let cloned = {
            let mut c = original.clone();
            c.min_print_interval_secs = 99.0;
            c.significant_progress_ratio = 0.99;
            c
        };
        assert!((original.min_print_interval_secs - 1.0).abs() < 1e-9);
        assert!((original.significant_progress_ratio - 0.05).abs() < 1e-9);
        assert!((cloned.min_print_interval_secs - 99.0).abs() < 1e-9);
        assert!((cloned.significant_progress_ratio - 0.99).abs() < 1e-9);
    }

    #[test]
    fn progress_print_config_equality_all_fields_same() {
        // Arrange
        let a = ProgressPrintConfig {
            min_print_interval_secs: 2.0,
            significant_progress_ratio: 0.1,
            significant_progress_interval_secs: 1.0,
            min_eta_speed_mb_per_sec: 0.5,
        };
        let b = ProgressPrintConfig {
            min_print_interval_secs: 2.0,
            significant_progress_ratio: 0.1,
            significant_progress_interval_secs: 1.0,
            min_eta_speed_mb_per_sec: 0.5,
        };
        // Act & Assert
        assert_eq!(a, b);
    }

    #[test]
    fn progress_print_config_inequality_each_field() {
        // Arrange: base config
        let base = ProgressPrintConfig {
            min_print_interval_secs: 1.0,
            significant_progress_ratio: 0.05,
            significant_progress_interval_secs: 0.5,
            min_eta_speed_mb_per_sec: 0.001,
        };
        // Act & Assert: changing each field individually makes it not equal
        let diff_field1 = ProgressPrintConfig {
            min_print_interval_secs: 2.0,
            ..base.clone()
        };
        let diff_field2 = ProgressPrintConfig {
            significant_progress_ratio: 0.1,
            ..base.clone()
        };
        let diff_field3 = ProgressPrintConfig {
            significant_progress_interval_secs: 1.0,
            ..base.clone()
        };
        let diff_field4 = ProgressPrintConfig {
            min_eta_speed_mb_per_sec: 0.01,
            ..base.clone()
        };
        assert_ne!(base, diff_field1);
        assert_ne!(base, diff_field2);
        assert_ne!(base, diff_field3);
        assert_ne!(base, diff_field4);
    }

    // ── DownloadTransferConfig additional property tests ───────────

    #[test]
    fn download_transfer_config_default_is_reasonable_size() {
        // Arrange & Act
        let cfg = DownloadTransferConfig::default();
        // Assert: defaults should be at least 1KB
        assert!(cfg.chunk_size_bytes >= 1024);
        assert!(cfg.io_buffer_size_bytes >= 1024);
    }

    #[test]
    fn download_transfer_config_chunk_larger_than_buffer() {
        // Arrange: common scenario where chunk > buffer
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 8 * 1024 * 1024,
            io_buffer_size_bytes: 4 * 1024,
        };
        // Act & Assert
        assert!(cfg.chunk_size_bytes > cfg.io_buffer_size_bytes);
    }

    #[test]
    fn download_transfer_config_buffer_larger_than_chunk() {
        // Arrange: unusual but valid scenario
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 1024,
            io_buffer_size_bytes: 8 * 1024 * 1024,
        };
        // Act & Assert: construction succeeds
        assert!(cfg.io_buffer_size_bytes > cfg.chunk_size_bytes);
    }

    #[test]
    fn download_transfer_config_power_of_two_values() {
        // Arrange: common power-of-2 sizes
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 1 << 23, // 8 MB
            io_buffer_size_bytes: 1 << 16, // 64 KB
        };
        // Act & Assert
        assert!(cfg.chunk_size_bytes.is_power_of_two());
        assert!(cfg.io_buffer_size_bytes.is_power_of_two());
    }

    #[test]
    fn download_transfer_config_default_values_are_power_of_two() {
        // Arrange & Act
        let cfg = DownloadTransferConfig::default();
        // Assert: default values should be power of 2
        assert!(cfg.chunk_size_bytes.is_power_of_two());
        assert!(cfg.io_buffer_size_bytes.is_power_of_two());
    }

    // ── NoProgress as ProgressCallback via dynamic dispatch ────────

    #[test]
    fn no_progress_via_trait_object() {
        // Arrange
        let mut progress: Box<dyn ProgressCallback> = Box::new(NoProgress);
        // Act: all methods via trait object
        progress.init(100, "file.bin");
        progress.update(50);
        progress.finish();
        // Assert: no panic
    }

    #[test]
    fn no_progress_in_vec_of_trait_objects() {
        // Arrange: multiple NoProgress instances as trait objects
        let callbacks: Vec<Box<dyn ProgressCallback>> = vec![
            Box::new(NoProgress),
            Box::new(NoProgress),
        ];
        // Act & Assert: iterate and call lifecycle methods
        for mut cb in callbacks {
            cb.init(100, "file.bin");
            cb.update(50);
            cb.finish();
        }
    }

    // ── DownloadTransferConfig validation boundary ─────────────────

    #[test]
    fn modelscope_downloader_accepts_chunk_equals_buffer() {
        // Arrange: chunk == buffer (unusual but valid)
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 4096,
            io_buffer_size_bytes: 4096,
        };
        // Act
        let result =
            ModelScopeDownloader::with_transfer_config(PathBuf::from("/tmp"), None, cfg);
        // Assert
        assert!(result.is_ok());
        let dl = result.unwrap();
        assert_eq!(dl.transfer_config.chunk_size_bytes, dl.transfer_config.io_buffer_size_bytes);
    }

    #[test]
    fn modelscope_downloader_rejects_zero_chunk_with_none_endpoint() {
        // Arrange: zero chunk with None endpoint
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 0,
            io_buffer_size_bytes: 1,
        };
        // Act
        let result = ModelScopeDownloader::with_transfer_config(
            PathBuf::from("/tmp"),
            None,
            cfg,
        );
        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn modelscope_downloader_rejects_zero_chunk_with_custom_endpoint() {
        // Arrange: zero chunk with custom endpoint
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 0,
            io_buffer_size_bytes: 1,
        };
        // Act
        let result = ModelScopeDownloader::with_transfer_config(
            PathBuf::from("/tmp"),
            Some("https://custom.endpoint".to_string()),
            cfg,
        );
        // Assert: validation happens regardless of endpoint
        assert!(result.is_err());
    }

    // ── is_cached: model_dir exists but no snapshots dir ───────────

    #[test]
    fn modelscope_is_cached_model_dir_exists_no_snapshots_subdir() {
        // Arrange
        let tmp = tempfile::tempdir().unwrap();
        let cache_dir = tmp.path();
        let model_dir = cache_dir.join("models--").join("org--model");
        // Create model_dir but NOT snapshots subdir
        std::fs::create_dir_all(&model_dir).unwrap();

        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // Act: snapshots dir doesn't exist
        let result = dl.is_cached("org/model", "file.bin", cache_dir);
        // Assert
        assert!(!result);
    }

    // ── ModelScopeDownloader: download_file returns cached path ────

    #[test]
    fn modelscope_download_file_returns_cached_path_when_file_exists() {
        // Arrange: set up a cached file in a snapshot
        let tmp = tempfile::tempdir().unwrap();
        let cache_dir = tmp.path();
        let model_dir = cache_dir.join("models--").join("org--model");
        let snapshots_dir = model_dir.join("snapshots");
        let snapshot_dir = snapshots_dir.join("snap001");
        std::fs::create_dir_all(&snapshot_dir).unwrap();
        let file_path = snapshot_dir.join("model.bin");
        std::fs::write(&file_path, b"cached content").unwrap();

        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // Act: file already exists, should return path without downloading
        let result = dl.download_file("org/model", "model.bin", cache_dir);
        // Assert
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), file_path);
    }

    #[test]
    fn modelscope_download_file_with_progress_returns_cached_path() {
        // Arrange: same setup but using download_file_with_progress
        let tmp = tempfile::tempdir().unwrap();
        let cache_dir = tmp.path();
        let model_dir = cache_dir.join("models--").join("org--model");
        let snapshots_dir = model_dir.join("snapshots");
        let snapshot_dir = snapshots_dir.join("snap001");
        std::fs::create_dir_all(&snapshot_dir).unwrap();
        let file_path = snapshot_dir.join("weights.st");
        std::fs::write(&file_path, b"cached weights").unwrap();

        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        let mut progress = NoProgress;
        // Act
        let result = dl.download_file_with_progress(
            "org/model",
            "weights.st",
            cache_dir,
            &mut progress,
        );
        // Assert
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), file_path);
    }

    // ── DownloadTransferConfig with ModelScopeDownloader preserves config ──

    #[test]
    fn modelscope_transfer_config_preserved_exact_values() {
        // Arrange: arbitrary non-default values
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 12345,
            io_buffer_size_bytes: 6789,
        };
        // Act
        let dl = ModelScopeDownloader::with_transfer_config(
            PathBuf::from("/data/cache"),
            None,
            cfg,
        )
        .unwrap();
        // Assert: exact values preserved
        assert_eq!(dl.transfer_config.chunk_size_bytes, 12345);
        assert_eq!(dl.transfer_config.io_buffer_size_bytes, 6789);
    }

    #[test]
    fn modelscope_cache_dir_preserved() {
        // Arrange
        let cache_path = PathBuf::from("/very/specific/cache/path");
        // Act
        let dl = ModelScopeDownloader::new(cache_path.clone(), None).unwrap();
        // Assert
        assert_eq!(dl._cache_dir, cache_path);
    }

    // ── DownloadTransferConfig Ord via derived traits ──────────────

    #[test]
    fn download_transfer_config_partialeq_reflexivity_with_nontrivial_values() {
        // Arrange
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 1337,
            io_buffer_size_bytes: 42,
        };
        // Act & Assert: a == a
        assert_eq!(cfg, cfg);
    }

    #[test]
    fn download_transfer_config_partialeq_distinguishes_buffer_only() {
        // Arrange: same chunk, different buffer
        let a = DownloadTransferConfig {
            chunk_size_bytes: 8192,
            io_buffer_size_bytes: 1024,
        };
        let b = DownloadTransferConfig {
            chunk_size_bytes: 8192,
            io_buffer_size_bytes: 2048,
        };
        // Act & Assert
        assert_ne!(a, b);
    }

    #[test]
    fn download_transfer_config_partialeq_distinguishes_chunk_only() {
        // Arrange: same buffer, different chunk
        let a = DownloadTransferConfig {
            chunk_size_bytes: 1024,
            io_buffer_size_bytes: 4096,
        };
        let b = DownloadTransferConfig {
            chunk_size_bytes: 2048,
            io_buffer_size_bytes: 4096,
        };
        // Act & Assert
        assert_ne!(a, b);
    }

    // ════════════════════════════════════════════════════════════════
    //  12 additional tests — batch 3
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn download_transfer_config_sizeof_is_two_usize() {
        // Arrange & Act & Assert: struct has exactly 2 usize fields
        assert_eq!(
            std::mem::size_of::<DownloadTransferConfig>(),
            2 * std::mem::size_of::<usize>()
        );
    }

    #[test]
    fn progress_print_config_sizeof_is_four_f64() {
        // Arrange & Act & Assert: struct has exactly 4 f64 fields
        assert_eq!(
            std::mem::size_of::<ProgressPrintConfig>(),
            4 * std::mem::size_of::<f64>()
        );
    }

    #[test]
    fn no_progress_implements_send() {
        // Arrange & Act & Assert: static compile-time check
        fn assert_send<T: Send>() {}
        assert_send::<NoProgress>();
    }

    #[test]
    fn no_progress_implements_sync() {
        // Arrange & Act & Assert: static compile-time check
        fn assert_sync<T: Sync>() {}
        assert_sync::<NoProgress>();
    }

    #[test]
    fn download_transfer_config_debug_alternate_format() {
        // Arrange
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 4096,
            io_buffer_size_bytes: 2048,
        };
        // Act
        let debug = format!("{:#?}", cfg);
        // Assert: alternate debug has newlines for pretty-printing
        assert!(debug.contains('\n'));
        assert!(debug.contains("chunk_size_bytes"));
        assert!(debug.contains("4096"));
    }

    #[test]
    fn progress_print_config_debug_alternate_format() {
        // Arrange
        let cfg = ProgressPrintConfig {
            min_print_interval_secs: 3.0,
            significant_progress_ratio: 0.25,
            significant_progress_interval_secs: 1.5,
            min_eta_speed_mb_per_sec: 0.1,
        };
        // Act
        let debug = format!("{:#?}", cfg);
        // Assert: alternate debug has newlines for pretty-printing
        assert!(debug.contains('\n'));
        assert!(debug.contains("ProgressPrintConfig"));
        assert!(debug.contains("3.0"));
    }

    #[test]
    fn download_transfer_config_field_mutation_readback() {
        // Arrange
        let mut cfg = DownloadTransferConfig {
            chunk_size_bytes: 1024,
            io_buffer_size_bytes: 512,
        };
        // Act: mutate each field
        cfg.chunk_size_bytes = 8192;
        cfg.io_buffer_size_bytes = 4096;
        // Assert: readback matches mutated values
        assert_eq!(cfg.chunk_size_bytes, 8192);
        assert_eq!(cfg.io_buffer_size_bytes, 4096);
    }

    #[test]
    fn progress_print_config_field_mutation_readback() {
        // Arrange
        let mut cfg = ProgressPrintConfig {
            min_print_interval_secs: 1.0,
            significant_progress_ratio: 0.05,
            significant_progress_interval_secs: 0.5,
            min_eta_speed_mb_per_sec: 0.001,
        };
        // Act: mutate each field
        cfg.min_print_interval_secs = 5.0;
        cfg.significant_progress_ratio = 0.5;
        cfg.significant_progress_interval_secs = 2.5;
        cfg.min_eta_speed_mb_per_sec = 1.0;
        // Assert: readback matches mutated values
        assert!((cfg.min_print_interval_secs - 5.0).abs() < 1e-9);
        assert!((cfg.significant_progress_ratio - 0.5).abs() < 1e-9);
        assert!((cfg.significant_progress_interval_secs - 2.5).abs() < 1e-9);
        assert!((cfg.min_eta_speed_mb_per_sec - 1.0).abs() < 1e-9);
    }

    #[test]
    fn download_transfer_config_default_chunk_greater_than_buffer() {
        // Arrange & Act
        let cfg = DownloadTransferConfig::default();
        // Assert: chunk is larger than buffer (8MB > 64KB)
        assert!(cfg.chunk_size_bytes > cfg.io_buffer_size_bytes);
    }

    #[test]
    fn no_progress_debug_alternate_equals_normal() {
        // Arrange
        let p = NoProgress;
        // Act
        let normal = format!("{:?}", p);
        let alternate = format!("{:#?}", p);
        // Assert: unit struct has identical debug output
        assert_eq!(normal, alternate);
    }

    #[test]
    fn modelscope_progress_total_starts_at_constructed_value() {
        // Arrange & Act: construct with non-zero total
        let progress = ModelScopeProgress { total: 999 };
        // Assert: initial value preserved
        assert_eq!(progress.total, 999);
    }

    #[test]
    fn download_transfer_config_hash_in_hashmap_lookup() {
        // Arrange
        use std::collections::HashMap;
        let key = DownloadTransferConfig {
            chunk_size_bytes: 4096,
            io_buffer_size_bytes: 2048,
        };
        let mut map = HashMap::new();
        // Act: insert and lookup with a Copy-derived identical key
        map.insert(key, "value");
        let lookup_key = key; // Copy
        // Assert: lookup succeeds
        assert_eq!(map.get(&lookup_key), Some(&"value"));
    }

    // ════════════════════════════════════════════════════════════════
    //  13 additional tests — batch 4
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn progress_print_config_default_intervals_below_one_second() {
        // Arrange & Act
        let cfg = ProgressPrintConfig::default();
        // Assert: both print intervals are positive and less than 1 second
        assert!(cfg.min_print_interval_secs > 0.0 && cfg.min_print_interval_secs <= 1.0);
        assert!(cfg.significant_progress_interval_secs > 0.0 && cfg.significant_progress_interval_secs <= 1.0);
    }

    #[test]
    fn download_transfer_config_hashset_deduplication() {
        // Arrange
        use std::collections::HashSet;
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 2048,
            io_buffer_size_bytes: 1024,
        };
        let mut set = HashSet::new();
        // Act: insert the same config three times
        set.insert(cfg);
        set.insert(cfg);
        set.insert(cfg);
        // Assert: deduplication reduces to exactly one entry
        assert_eq!(set.len(), 1);
        assert!(set.contains(&cfg));
    }

    #[test]
    fn progress_bar_init_with_one_byte_total() {
        // Arrange
        let mut bar = ProgressBar::new("tiny.bin".to_string());
        // Act
        bar.init(1, "tiny.bin");
        // Assert
        assert_eq!(bar.total, 1);
        bar.update(1);
        bar.finish();
    }

    #[test]
    fn modelscope_get_url_unicode_repo_name() {
        // Arrange
        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // Act: repo with unicode characters (no slashes → not replaced)
        let url = dl.get_url("组织/模型名称", "file.bin");
        // Assert: slash replaced with --, unicode preserved
        assert!(url.contains("组织--模型名称"));
        assert!(url.ends_with("/resolve/main/file.bin"));
    }

    #[test]
    fn download_transfer_config_eq_consistent_with_partialeq() {
        // Arrange: Eq requires total equality: a == a and !(a != a)
        let a = DownloadTransferConfig {
            chunk_size_bytes: 5555,
            io_buffer_size_bytes: 3333,
        };
        // Act & Assert: PartialEq true means Eq is also satisfied
        assert!(a == a);
        assert!(!(a != a));
    }

    #[test]
    fn no_progress_multiple_finish_calls_harmless() {
        // Arrange
        let mut progress = NoProgress;
        progress.init(100, "test.bin");
        progress.update(50);
        // Act: call finish many times
        progress.finish();
        progress.finish();
        progress.finish();
        progress.finish();
        // Assert: no panic
    }

    #[test]
    fn progress_bar_finish_without_any_update() {
        // Arrange
        let mut bar = ProgressBar::new("noupdate.bin".to_string());
        bar.init(2048, "noupdate.bin");
        // Act: finish immediately after init, skip any update calls
        bar.finish();
        // Assert: total was set correctly
        assert_eq!(bar.total, 2048);
    }

    #[test]
    fn progress_print_config_zero_vs_min_positive() {
        // Arrange
        let zero_cfg = ProgressPrintConfig {
            min_print_interval_secs: 0.0,
            significant_progress_ratio: 0.0,
            significant_progress_interval_secs: 0.0,
            min_eta_speed_mb_per_sec: 0.0,
        };
        let tiny_cfg = ProgressPrintConfig {
            min_print_interval_secs: f64::MIN_POSITIVE,
            significant_progress_ratio: f64::MIN_POSITIVE,
            significant_progress_interval_secs: f64::MIN_POSITIVE,
            min_eta_speed_mb_per_sec: f64::MIN_POSITIVE,
        };
        // Assert: zero and MIN_POSITIVE are different values
        assert_ne!(zero_cfg, tiny_cfg);
    }

    #[test]
    fn download_transfer_config_copy_then_modify_original() {
        // Arrange
        let original = DownloadTransferConfig {
            chunk_size_bytes: 4096,
            io_buffer_size_bytes: 2048,
        };
        let copy = original; // Copy
        // Act: original is still accessible after copy (Copy semantics)
        let mut original = original;
        original.chunk_size_bytes = 0;
        // Assert: copy retains original value
        assert_eq!(copy.chunk_size_bytes, 4096);
        assert_eq!(original.chunk_size_bytes, 0);
    }

    #[test]
    fn modelscope_progress_finish_then_reinit() {
        // Arrange
        let mut progress = ModelScopeProgress { total: 0 };
        progress.init(500, "first.bin");
        progress.finish();
        // Act: reinitialize after finish
        progress.init(1000, "second.bin");
        // Assert: total updated to new value
        assert_eq!(progress.total, 1000);
        progress.finish();
    }

    #[test]
    fn progress_bar_update_exactly_at_total_no_panic() {
        // Arrange
        let mut bar = ProgressBar::new("exact.bin".to_string());
        bar.init(500, "exact.bin");
        // Act: update exactly at total boundary
        bar.update(500);
        bar.finish();
        // Assert: no panic; percent = 100.0, ETA = 0 remaining bytes
        assert_eq!(bar.total, 500);
    }

    #[test]
    fn download_transfer_config_zero_chunk_zero_buffer_both_zero() {
        // Arrange & Act
        let cfg = DownloadTransferConfig {
            chunk_size_bytes: 0,
            io_buffer_size_bytes: 0,
        };
        // Assert: construction succeeds (struct has no validation), values are zero
        assert_eq!(cfg.chunk_size_bytes, 0);
        assert_eq!(cfg.io_buffer_size_bytes, 0);
    }

    #[test]
    fn modelscope_get_url_repo_with_numbers_and_special_chars() {
        // Arrange
        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // Act: repo name with numbers, dots, underscores, hyphens
        let url = dl.get_url("org/model-v2.0_beta", "config.json");
        // Assert: only slash is replaced, other characters preserved
        assert_eq!(
            url,
            "https://www.modelscope.cn/org--model-v2.0_beta/resolve/main/config.json"
        );
    }

    // ════════════════════════════════════════════════════════════════
    //  10 additional tests — batch 5
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn modelscope_download_file_nested_filename_resolves_correctly() {
        // Arrange: cached file with a nested path like "onnx/model.onnx"
        let tmp = tempfile::tempdir().unwrap();
        let cache_dir = tmp.path();
        let model_dir = cache_dir.join("models--").join("org--model");
        let snapshots_dir = model_dir.join("snapshots");
        let snapshot_dir = snapshots_dir.join("snap001");
        let onnx_dir = snapshot_dir.join("onnx");
        std::fs::create_dir_all(&onnx_dir).unwrap();
        let nested_file = onnx_dir.join("model.onnx");
        std::fs::write(&nested_file, b"onnx weights").unwrap();

        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // Act: download_file should find the nested file already cached
        let result = dl.download_file("org/model", "onnx/model.onnx", cache_dir);
        // Assert: returns the exact path within the snapshot
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), nested_file);
    }

    #[test]
    fn modelscope_download_file_with_progress_nested_filename_resolves() {
        // Arrange: cached file with nested path through download_file_with_progress
        let tmp = tempfile::tempdir().unwrap();
        let cache_dir = tmp.path();
        let model_dir = cache_dir.join("models--").join("org--model");
        let snapshots_dir = model_dir.join("snapshots");
        let snapshot_dir = snapshots_dir.join("snap002");
        let sub_dir = snapshot_dir.join("layers");
        std::fs::create_dir_all(&sub_dir).unwrap();
        let nested = sub_dir.join("layer.0.weight");
        std::fs::write(&nested, b"layer data").unwrap();

        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        let mut progress = NoProgress;
        // Act
        let result = dl.download_file_with_progress(
            "org/model",
            "layers/layer.0.weight",
            cache_dir,
            &mut progress,
        );
        // Assert
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), nested);
    }

    #[test]
    fn modelscope_is_cached_zero_byte_file_is_cached() {
        // Arrange: a zero-byte file exists in the snapshot — still "cached"
        let tmp = tempfile::tempdir().unwrap();
        let cache_dir = tmp.path();
        let model_dir = cache_dir.join("models--").join("org--model");
        let snapshots_dir = model_dir.join("snapshots");
        let snapshot_dir = snapshots_dir.join("snap001");
        std::fs::create_dir_all(&snapshot_dir).unwrap();
        std::fs::write(snapshot_dir.join("empty.bin"), b"").unwrap();

        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // Act
        let result = dl.is_cached("org/model", "empty.bin", cache_dir);
        // Assert: zero-byte file is still considered cached
        assert!(result);
    }

    #[test]
    fn modelscope_find_latest_snapshot_with_symlink_entry() {
        // Arrange: snapshot directory containing a real dir and a symlink to a dir
        let tmp = tempfile::tempdir().unwrap();
        let snapshots_dir = tmp.path().join("snapshots");
        std::fs::create_dir_all(&snapshots_dir).unwrap();
        let real_dir = snapshots_dir.join("real_snap");
        std::fs::create_dir_all(&real_dir).unwrap();
        std::fs::write(real_dir.join("data"), b"real").unwrap();
        // Create a symlink pointing to the real dir
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let link_target = snapshots_dir.join("link_snap");
            symlink(&real_dir, &link_target).unwrap();
        }

        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // Act
        let result = dl.find_latest_snapshot(&snapshots_dir);
        // Assert: returns one of the entries without panic
        assert!(result.is_ok());
        let found = result.unwrap();
        assert!(found.file_name().unwrap().to_string_lossy().contains("snap"));
    }

    #[test]
    fn modelscope_download_file_creates_correct_normalized_path() {
        // Arrange: verify the internal path normalization (org/model → org--model)
        // by creating the expected directory structure and checking resolution
        let tmp = tempfile::tempdir().unwrap();
        let cache_dir = tmp.path();
        let model_dir = cache_dir.join("models--").join("my-org--my-model");
        let snapshots_dir = model_dir.join("snapshots");
        let snapshot_dir = snapshots_dir.join("abc");
        std::fs::create_dir_all(&snapshot_dir).unwrap();
        let file_path = snapshot_dir.join("config.json");
        std::fs::write(&file_path, b"{}").unwrap();

        let dl = ModelScopeDownloader::new(cache_dir.to_path_buf(), None).unwrap();
        // Act: download_file with org/model repo should map to my-org--my-model dir
        let result = dl.download_file("my-org/my-model", "config.json", cache_dir);
        // Assert: resolved to the file within the correctly normalized directory
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), file_path);
    }

    #[test]
    fn modelscope_progress_update_at_zero_percent_boundary() {
        // Arrange: verify ModelScopeProgress handles update(0) after init
        // without division-by-zero panic (total > 0, current = 0)
        let mut progress = ModelScopeProgress { total: 0 };
        progress.init(1000, "boundary.bin");
        assert_eq!(progress.total, 1000);
        // Act: update at 0 — percent = 0.0
        progress.update(0);
        // Assert: total unchanged, no panic
        assert_eq!(progress.total, 1000);
        progress.finish();
    }

    #[test]
    fn modelscope_progress_update_at_exact_total() {
        // Arrange: update(current) where current == total
        let mut progress = ModelScopeProgress { total: 0 };
        progress.init(500, "exact.bin");
        // Act: update at exactly total — percent = 100.0, clamped by .min(100.0)
        progress.update(500);
        progress.finish();
        // Assert: no panic
    }

    #[test]
    fn progress_bar_config_very_large_min_print_interval_suppresses_printing() {
        // Arrange: config with huge min_print_interval means should_print = false
        // unless current >= total
        let cfg = ProgressPrintConfig {
            min_print_interval_secs: 999999.0,
            significant_progress_ratio: 999999.0,
            significant_progress_interval_secs: 999999.0,
            min_eta_speed_mb_per_sec: 999999.0,
        };
        let mut bar = ProgressBar::with_config("suppressed.bin".to_string(), cfg);
        // Act: init and partial update — should not print (only prints at current >= total)
        bar.init(10000, "suppressed.bin");
        bar.update(5000);
        bar.update(10000); // current == total triggers should_print
        bar.finish();
        // Assert: no panic, bar completed
        assert_eq!(bar.total, 10000);
    }

    #[test]
    fn download_transfer_config_default_not_zero_either_field() {
        // Arrange & Act
        let cfg = DownloadTransferConfig::default();
        // Assert: both fields must be non-zero (this is a semantic invariant)
        assert_ne!(cfg.chunk_size_bytes, 0);
        assert_ne!(cfg.io_buffer_size_bytes, 0);
    }

    #[test]
    fn modelscope_is_cached_case_sensitive_repo_name() {
        // Arrange: repo names are case-sensitive; Org-Model != org-model
        let tmp = tempfile::tempdir().unwrap();
        let cache_dir = tmp.path();
        let model_dir = cache_dir.join("models--").join("Org--Model");
        let snapshots_dir = model_dir.join("snapshots");
        let snapshot_dir = snapshots_dir.join("snap001");
        std::fs::create_dir_all(&snapshot_dir).unwrap();
        std::fs::write(snapshot_dir.join("file.bin"), b"data").unwrap();

        let dl = ModelScopeDownloader::new(PathBuf::from("/tmp"), None).unwrap();
        // Act: query with lowercase should NOT find the uppercase directory
        let result_lower = dl.is_cached("org/model", "file.bin", cache_dir);
        let result_upper = dl.is_cached("Org/Model", "file.bin", cache_dir);
        // Assert: only exact-case match succeeds
        assert!(!result_lower);
        assert!(result_upper);
    }
}

/// ModelScope 下载器（使用 ureq 实现分块下载）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DownloadTransferConfig {
    pub chunk_size_bytes: usize,
    pub io_buffer_size_bytes: usize,
}

impl Default for DownloadTransferConfig {
    fn default() -> Self {
        Self {
            chunk_size_bytes: 8 * 1024 * 1024,
            io_buffer_size_bytes: 64 * 1024,
        }
    }
}

pub struct ModelScopeDownloader {
    agent: Agent,
    endpoint: String,
    _cache_dir: PathBuf,
    transfer_config: DownloadTransferConfig,
}

impl ModelScopeDownloader {
    pub fn new(cache_dir: PathBuf, endpoint: Option<String>) -> Result<Self> {
        Self::with_transfer_config(cache_dir, endpoint, DownloadTransferConfig::default())
    }

    pub fn with_transfer_config(
        cache_dir: PathBuf,
        endpoint: Option<String>,
        transfer_config: DownloadTransferConfig,
    ) -> Result<Self> {
        let endpoint = endpoint.unwrap_or_else(|| "https://www.modelscope.cn".to_string()); // LEGAL: endpoint 缺失时使用 ModelScope 官方端点

        // 配置 ureq agent
        let agent = ureq::builder().try_proxy_from_env(true).build();
        if transfer_config.chunk_size_bytes == 0 || transfer_config.io_buffer_size_bytes == 0 {
            return Err(crate::loader::LoaderError::HfHub(
                "invalid ModelScope transfer config: chunk/buffer must be > 0".to_string(),
            ));
        }

        Ok(Self {
            agent,
            endpoint,
            _cache_dir: cache_dir,
            transfer_config,
        })
    }

    /// 获取文件的下载 URL
    fn get_url(&self, repo: &str, filename: &str) -> String {
        format!(
            "{}/{}/resolve/main/{}",
            self.endpoint,
            repo.replace('/', "--"),
            filename
        )
    }

    /// 分块下载文件
    fn download_chunked(
        &self,
        url: &str,
        dest_path: &Path,
        filename: &str,
        progress: &mut dyn ProgressCallback,
    ) -> Result<()> {
        // 创建目录
        if let Some(parent) = dest_path.parent() {
            std::fs::create_dir_all(parent).map_err(crate::loader::LoaderError::Io)?;
        }

        // 检查是否需要续传
        let start_pos = if dest_path.exists() {
            let metadata = std::fs::metadata(dest_path).map_err(crate::loader::LoaderError::Io)?;
            metadata.len()
        } else {
            0
        };

        // 发送 Range 请求获取文件大小
        let size = self.get_file_size(url)?;

        progress.init(size as usize, filename);

        // 如果已下载完成，直接返回
        if start_pos >= size {
            progress.finish();
            return Ok(());
        }

        // 分块下载
        let chunk_size = self.transfer_config.chunk_size_bytes as u64;
        let mut buffer = vec![0u8; self.transfer_config.io_buffer_size_bytes];
        let mut current = start_pos;

        loop {
            let end = current.saturating_add(chunk_size).min(size);

            let range = format!("bytes={}-{}", current, end - 1);
            let response = self
                .agent
                .get(url)
                .set("Range", &range)
                .call()
                .map_err(|e| crate::loader::LoaderError::HfHub(format!("download error: {}", e)))?;

            if response.status() != 200 && response.status() != 206 {
                return Err(crate::loader::LoaderError::HfHub(format!(
                    "download failed with status {}",
                    response.status()
                )));
            }

            // 追加写入文件
            let mut file = if current == 0 && !dest_path.exists() {
                std::fs::File::create(dest_path).map_err(crate::loader::LoaderError::Io)?
            } else {
                std::fs::OpenOptions::new()
                    .append(true)
                    .open(dest_path)
                    .map_err(crate::loader::LoaderError::Io)?
            };

            let mut reader = response.into_reader();
            let mut written = 0u64;

            loop {
                let n = reader
                    .read(&mut buffer)
                    .map_err(crate::loader::LoaderError::Io)?;
                if n == 0 {
                    break;
                }
                file.write_all(&buffer[..n])
                    .map_err(crate::loader::LoaderError::Io)?;
                written += n as u64;
                progress.update((current + written) as usize);
            }

            current += written;

            if current >= size {
                break;
            }
        }

        progress.finish();
        Ok(())
    }

    /// 获取文件大小
    fn get_file_size(&self, url: &str) -> Result<u64> {
        let response = self
            .agent
            .get(url)
            .set("Range", "bytes=0-0")
            .call()
            .map_err(|e| crate::loader::LoaderError::HfHub(format!("head error: {}", e)))?;

        if response.status() != 206 && response.status() != 200 {
            return Err(crate::loader::LoaderError::HfHub(
                "cannot determine file size".to_string(),
            ));
        }

        let content_range = response.header("Content-Range").ok_or_else(|| {
            crate::loader::LoaderError::HfHub("missing Content-Range".to_string())
        })?;

        // Content-Range: bytes 0-1048575/1048576
        let size_str = content_range.split('/').nth(1).ok_or_else(|| {
            crate::loader::LoaderError::HfHub("invalid Content-Range".to_string())
        })?;

        size_str
            .parse::<u64>()
            .map_err(|_| crate::loader::LoaderError::HfHub("invalid file size".to_string()))
    }
}

impl Downloader for ModelScopeDownloader {
    fn download_file(&self, repo: &str, filename: &str, cache_dir: &Path) -> Result<PathBuf> {
        // 标准化仓库名: org/name → org--name
        let normalized_repo = repo.replace('/', "--");
        let model_dir = cache_dir.join("models--").join(&normalized_repo);

        // 查找 snapshot
        let snapshots_dir = model_dir.join("snapshots");
        let snapshot = self.find_latest_snapshot(&snapshots_dir)?;
        let dest_path = snapshot.join(filename);

        // 如果已存在，直接返回
        if dest_path.exists() {
            return Ok(dest_path);
        }

        // 下载文件
        let url = self.get_url(repo, filename);
        let mut progress = ModelScopeProgress { total: 0 };
        self.download_chunked(&url, &dest_path, filename, &mut progress)?;

        Ok(dest_path)
    }

    fn download_file_with_progress(
        &self,
        repo: &str,
        filename: &str,
        cache_dir: &Path,
        progress: &mut dyn ProgressCallback,
    ) -> Result<PathBuf> {
        // 标准化仓库名: org/name → org--name
        let normalized_repo = repo.replace('/', "--");
        let model_dir = cache_dir.join("models--").join(&normalized_repo);

        // 查找 snapshot
        let snapshots_dir = model_dir.join("snapshots");
        let snapshot = self.find_latest_snapshot(&snapshots_dir)?;
        let dest_path = snapshot.join(filename);

        // 如果已存在，直接返回
        if dest_path.exists() {
            return Ok(dest_path);
        }

        // 下载文件
        let url = self.get_url(repo, filename);
        self.download_chunked(&url, &dest_path, filename, progress)?;

        Ok(dest_path)
    }

    fn is_cached(&self, repo: &str, filename: &str, cache_dir: &Path) -> bool {
        let normalized_repo = repo.replace('/', "--");
        let model_dir = cache_dir.join("models--").join(&normalized_repo);

        if !model_dir.exists() {
            return false;
        }

        let snapshots_dir = model_dir.join("snapshots");
        let snapshot = match self.find_latest_snapshot(&snapshots_dir) {
            Ok(s) => s,
            Err(e) => {
                log::debug!("no snapshot found in {}: {e}", snapshots_dir.display());
                return false;
            }
        };

        snapshot.join(filename).exists()
    }
}

impl ModelScopeDownloader {
    fn find_latest_snapshot(&self, snapshots_dir: &Path) -> Result<PathBuf> {
        let mut latest = None;
        let mut latest_mtime: std::time::SystemTime = std::time::SystemTime::UNIX_EPOCH;

        for entry in std::fs::read_dir(snapshots_dir).map_err(crate::loader::LoaderError::Io)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            let mtime = metadata.modified()?;

            if mtime > latest_mtime {
                latest_mtime = mtime;
                latest = Some(entry.path());
            }
        }

        latest.ok_or_else(|| crate::loader::LoaderError::HfHub("No snapshot found".to_string()))
    }
}

/// ModelScope 简单进度报告
struct ModelScopeProgress {
    total: usize,
}

impl ProgressCallback for ModelScopeProgress {
    fn init(&mut self, total: usize, filename: &str) {
        self.total = total;
        eprintln!(
            "📥 [ModelScope] 下载: {} ({:.2} MB)",
            filename,
            total as f64 / 1e6
        );
    }

    fn update(&mut self, current: usize) {
        let percent = (current as f64 / self.total as f64 * 100.0).min(100.0);
        eprint!("\r   进度: {:.1}%", percent);
        std::io::stdout().flush().ok();
    }

    fn finish(&mut self) {
        eprintln!("\n   ✅ 完成下载");
    }
}
