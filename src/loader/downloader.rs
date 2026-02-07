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
pub struct NoProgress;
impl ProgressCallback for NoProgress {
    fn init(&mut self, _total: usize, _filename: &str) {}
    fn update(&mut self, _current: usize) {}
    fn finish(&mut self) {}
}

#[derive(Debug, Clone)]
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

    #[test]
    fn test_no_progress() {
        let mut progress = NoProgress;
        progress.init(1000, "test.bin");
        progress.update(500);
        progress.finish();
    }
}

/// ModelScope 下载器（使用 ureq 实现分块下载）
#[derive(Debug, Clone)]
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
        let endpoint = endpoint.unwrap_or_else(|| "https://www.modelscope.cn".to_string());

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
            Err(_) => return false,
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
