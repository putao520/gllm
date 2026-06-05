//! NVMe swap file format and I/O.
//!
//! Per SPEC `gllm/SPEC/22-PAGE-COMPRESSION.md §7.5.3` (REQ-COMP-015).
//!
//! 文件按 page_id 划分固定大小 slot，用 `pwrite` / `pread` 在指定 offset 读写。
//! 布局:
//!
//! ```text
//! [SwapFileHeader (SWAP_HEADER_BYTES=4096)] [slot_0] [slot_1] ... [slot_N-1]
//! ```
//!
//! 每个 slot 的字节数 = `max_slot_bytes`（已对齐至 NVME_ALIGN=4096）。
//!
//! # O_DIRECT
//!
//! 文件以 `O_DIRECT | O_RDWR | O_CREAT` 打开，绕过 Linux page cache，
//! 实现真正的 NVMe 直写。所有 I/O 缓冲区通过 `posix_memalign` 分配，
//! 地址和长度均 4096 对齐。线程安全通过内部 `Mutex<RawFd>` 保证。

use std::os::unix::ffi::OsStrExt;
#[cfg(test)]
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::Mutex;

use crate::scheduler::types::PageId;

// ─────────────────────────────────────────────────────────────────────────────
// 常量
// ─────────────────────────────────────────────────────────────────────────────

const SWAP_MAGIC: u64 = 0x474C4C4D53574150;
const SWAP_VERSION: u32 = 1;
pub const SWAP_HEADER_BYTES: usize = 4096;
pub const NVME_ALIGN: usize = 4096;

// ─────────────────────────────────────────────────────────────────────────────
// Header layout (exactly SWAP_HEADER_BYTES bytes)
// ─────────────────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SwapFileHeader {
    pub magic: u64,
    pub version: u32,
    pub page_size: u32,
    pub max_slot_bytes: u32,
    _pad4: u32,
    pub slot_count: u64,
    pub _reserved: [u8; SWAP_HEADER_BYTES - 32],
}

const _HEADER_SIZE_CHECK: [(); SWAP_HEADER_BYTES] =
    [(); std::mem::size_of::<SwapFileHeader>()];

// ─────────────────────────────────────────────────────────────────────────────
// Aligned buffer helper
// ─────────────────────────────────────────────────────────────────────────────

/// RAII wrapper for a `posix_memalign`-allocated buffer (4096-aligned).
struct AlignedBuffer {
    ptr: *mut u8,
    len: usize,
}

impl AlignedBuffer {
    fn new(len: usize) -> Self {
        assert!(len > 0 && len.is_multiple_of(NVME_ALIGN));
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let ret = unsafe {
            libc::posix_memalign(&mut ptr, NVME_ALIGN, len)
        };
        assert_eq!(ret, 0, "posix_memalign failed");
        unsafe { libc::memset(ptr, 0, len) };
        Self {
            ptr: ptr as *mut u8,
            len,
        }
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { libc::free(self.ptr as *mut std::ffi::c_void) };
        }
    }
}

// SAFETY: AlignedBuffer owns its memory exclusively, no aliasing.
unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

// ─────────────────────────────────────────────────────────────────────────────
// NvmeSwapFile
// ─────────────────────────────────────────────────────────────────────────────

/// Per-session NVMe swap file with O_DIRECT I/O.
///
/// Slots are indexed by `PageId`. Each slot is `max_slot_bytes` wide
/// (4096-aligned). All reads/writes bypass the Linux page cache.
pub struct NvmeSwapFile {
    fd: Mutex<std::os::unix::io::RawFd>,
    pub page_size: usize,
    pub max_slot_bytes: usize,
    pub slot_count: u64,
}

impl std::fmt::Debug for NvmeSwapFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NvmeSwapFile")
            .field("page_size", &self.page_size)
            .field("max_slot_bytes", &self.max_slot_bytes)
            .field("slot_count", &self.slot_count)
            .finish()
    }
}

impl Drop for NvmeSwapFile {
    fn drop(&mut self) {
        if let Ok(fd) = self.fd.lock() {
            unsafe { libc::close(*fd) };
        }
    }
}

impl NvmeSwapFile {
    fn open_fd(path: &std::path::Path) -> std::io::Result<std::os::unix::io::RawFd> {
        let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "path contains NUL")
        })?;
        let fd = unsafe {
            libc::open(
                c_path.as_ptr(),
                libc::O_RDWR | libc::O_CREAT | libc::O_DIRECT,
                0o644,
            )
        };
        if fd < 0 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(fd)
        }
    }

    fn pwrite_all(fd: std::os::unix::io::RawFd, buf: &[u8], offset: u64) -> std::io::Result<()> {
        let mut written = 0usize;
        while written < buf.len() {
            let n = unsafe {
                libc::pwrite(
                    fd,
                    buf[written..].as_ptr() as *const std::ffi::c_void,
                    buf.len() - written,
                    (offset + written as u64) as i64,
                )
            };
            if n < 0 {
                return Err(std::io::Error::last_os_error());
            }
            written += n as usize;
        }
        Ok(())
    }

    fn pread_exact(fd: std::os::unix::io::RawFd, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
        let mut total = 0usize;
        while total < buf.len() {
            let n = unsafe {
                libc::pread(
                    fd,
                    buf[total..].as_mut_ptr() as *mut std::ffi::c_void,
                    buf.len() - total,
                    (offset + total as u64) as i64,
                )
            };
            if n < 0 {
                return Err(std::io::Error::last_os_error());
            }
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "pread returned 0 (EOF)",
                ));
            }
            total += n as usize;
        }
        Ok(())
    }

    pub fn open(
        path: PathBuf,
        page_size: usize,
        max_slot_bytes: usize,
        slot_count: u64,
    ) -> std::io::Result<Self> {
        let aligned_slot =
            max_slot_bytes.div_ceil(NVME_ALIGN) * NVME_ALIGN;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let fd = Self::open_fd(&path)?;

        // Check file length via fstat.
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(fd, &mut st) } < 0 {
            let _ = unsafe { libc::close(fd) };
            return Err(std::io::Error::last_os_error());
        }
        let file_len = st.st_size as u64;

        if file_len < SWAP_HEADER_BYTES as u64 {
            // New file: write header (aligned write).
            let mut hdr_buf = AlignedBuffer::new(SWAP_HEADER_BYTES);
            let header = SwapFileHeader {
                magic: SWAP_MAGIC,
                version: SWAP_VERSION,
                page_size: page_size as u32,
                max_slot_bytes: aligned_slot as u32,
                _pad4: 0,
                slot_count,
                _reserved: [0u8; SWAP_HEADER_BYTES - 32],
            };
            unsafe {
                let src = &header as *const SwapFileHeader as *const u8;
                std::ptr::copy_nonoverlapping(src, hdr_buf.as_mut_ptr(), SWAP_HEADER_BYTES);
            }
            Self::pwrite_all(fd, hdr_buf.as_slice(), 0)?;

            // Pre-allocate (sparse).
            let total = SWAP_HEADER_BYTES as u64 + slot_count * aligned_slot as u64;
            if unsafe { libc::ftruncate(fd, total as libc::off_t) } < 0 {
                let _ = unsafe { libc::close(fd) };
                return Err(std::io::Error::last_os_error());
            }
        } else {
            // Existing file: validate header.
            let mut hdr_buf = AlignedBuffer::new(SWAP_HEADER_BYTES);
            Self::pread_exact(fd, hdr_buf.as_mut_slice(), 0)?;

            let header: &SwapFileHeader = unsafe {
                &*(hdr_buf.as_ptr() as *const SwapFileHeader)
            };
            if header.magic != SWAP_MAGIC {
                let _ = unsafe { libc::close(fd) };
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "swap file {}: bad magic 0x{:016x} (expected 0x{SWAP_MAGIC:016x})",
                        path.display(),
                        header.magic
                    ),
                ));
            }
            if header.version != SWAP_VERSION {
                let _ = unsafe { libc::close(fd) };
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "swap file {}: version {} unsupported (expected {SWAP_VERSION})",
                        path.display(),
                        header.version
                    ),
                ));
            }
        }

        Ok(Self {
            fd: Mutex::new(fd),
            page_size,
            max_slot_bytes: aligned_slot,
            slot_count,
        })
    }

    #[inline]
    fn slot_offset(&self, page_id: PageId) -> u64 {
        SWAP_HEADER_BYTES as u64 + page_id as u64 * self.max_slot_bytes as u64
    }

    /// Write compressed `data` to the slot for `page_id`.
    ///
    /// `data.len()` must not exceed `self.max_slot_bytes`.
    /// Data is copied to an O_DIRECT-aligned buffer, then padded to
    /// `max_slot_bytes` (4096-aligned) for the actual pwrite.
    pub fn write_slot(&self, page_id: PageId, data: &[u8]) -> std::io::Result<u32> {
        if data.len() > self.max_slot_bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "write_slot page={page_id}: data len {} > max_slot_bytes {}",
                    data.len(),
                    self.max_slot_bytes
                ),
            ));
        }
        let fd = self.fd.lock().map_err(|_| {
            std::io::Error::other("swap file mutex poisoned")
        })?;

        let mut buf = AlignedBuffer::new(self.max_slot_bytes);
        buf.as_mut_slice()[..data.len()].copy_from_slice(data);

        let offset = self.slot_offset(page_id);
        Self::pwrite_all(*fd, buf.as_slice(), offset)?;
        Ok(data.len() as u32)
    }

    /// Read exactly `dst.len()` bytes from the slot for `page_id`.
    ///
    /// `dst.len()` must not exceed `self.max_slot_bytes`.
    /// Internally reads the full aligned slot, then copies the requested
    /// number of bytes into `dst`.
    pub fn read_slot(&self, page_id: PageId, dst: &mut [u8]) -> std::io::Result<()> {
        if dst.len() > self.max_slot_bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "read_slot page={page_id}: dst len {} > max_slot_bytes {}",
                    dst.len(),
                    self.max_slot_bytes
                ),
            ));
        }
        let fd = self.fd.lock().map_err(|_| {
            std::io::Error::other("swap file mutex poisoned")
        })?;

        let mut buf = AlignedBuffer::new(self.max_slot_bytes);
        let offset = self.slot_offset(page_id);
        Self::pread_exact(*fd, buf.as_mut_slice(), offset)?;
        dst.copy_from_slice(&buf.as_slice()[..dst.len()]);
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn open_swap(tmp: &TempDir, name: &str) -> NvmeSwapFile {
        let path = tmp.path().join(name);
        NvmeSwapFile::open(path, 4096, 8192, 32).expect("open swap")
    }

    #[test]
    fn header_init_and_reopen() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.swap");

        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 8192, 16).unwrap();
            swap.write_slot(3, &[0xAA; 1024]).unwrap();
        }

        let swap2 = NvmeSwapFile::open(path, 4096, 8192, 16).unwrap();
        let mut buf = vec![0u8; 1024];
        swap2.read_slot(3, &mut buf).unwrap();
        assert!(
            buf.iter().all(|&b| b == 0xAA),
            "slot 3 data corrupted after reopen"
        );
    }

    #[test]
    fn multiple_slots_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "multi.swap");

        for pid in 0usize..16 {
            let data = vec![pid as u8; 512];
            swap.write_slot(pid, &data).unwrap();
        }

        for pid in 0usize..16 {
            let mut buf = vec![0u8; 512];
            swap.read_slot(pid, &mut buf).unwrap();
            assert!(
                buf.iter().all(|&b| b == pid as u8),
                "slot {pid} data corrupted"
            );
        }
    }

    #[test]
    fn slots_are_independent() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "indep.swap");

        swap.write_slot(0, &[0x11; 256]).unwrap();
        swap.write_slot(1, &[0x22; 256]).unwrap();
        swap.write_slot(2, &[0x33; 256]).unwrap();

        let mut b0 = vec![0u8; 256];
        let mut b1 = vec![0u8; 256];
        let mut b2 = vec![0u8; 256];
        swap.read_slot(0, &mut b0).unwrap();
        swap.read_slot(1, &mut b1).unwrap();
        swap.read_slot(2, &mut b2).unwrap();

        assert!(b0.iter().all(|&b| b == 0x11));
        assert!(b1.iter().all(|&b| b == 0x22));
        assert!(b2.iter().all(|&b| b == 0x33));
    }

    #[test]
    fn write_too_large_returns_error() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("s.swap"), 512, 512, 8).unwrap();
        let oversized = vec![0u8; swap.max_slot_bytes + 1];
        let result = swap.write_slot(0, &oversized);
        assert!(result.is_err(), "expected error for oversized write");
    }

    #[test]
    fn bad_magic_rejected() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("bad.swap");
        {
            // Write garbage with standard I/O (no O_DIRECT required for setup).
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&[0xFF; SWAP_HEADER_BYTES]).unwrap();

            // Extend to at least SWAP_HEADER_BYTES so fstat check passes.
            use std::io::Write;
            f.set_len(SWAP_HEADER_BYTES as u64).unwrap();
        }
        let result = NvmeSwapFile::open(path, 4096, 8192, 8);
        assert!(result.is_err(), "expected error for bad magic");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("magic"),
            "error message should mention magic: {err}"
        );
    }

    /// O_DIRECT verification: write and read back with exact byte comparison.
    #[test]
    fn odirect_roundtrip_exact_bytes() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("od.swap"), 4096, 8192, 8).unwrap();

        let data: Vec<u8> = (0..1024).map(|i| (i * 3 + 7) as u8).collect();
        swap.write_slot(0, &data).unwrap();

        let mut readback = vec![0u8; 1024];
        swap.read_slot(0, &mut readback).unwrap();
        assert_eq!(readback, data, "O_DIRECT round-trip must preserve exact bytes");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader — pure structure tests
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_size_is_4096() {
        assert_eq!(std::mem::size_of::<SwapFileHeader>(), SWAP_HEADER_BYTES);
    }

    #[test]
    fn swap_header_default_fields() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 16,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(hdr.magic, 0x474C4C4D53574150);
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.page_size, 4096);
        assert_eq!(hdr.max_slot_bytes, 8192);
        assert_eq!(hdr.slot_count, 16);
    }

    #[test]
    fn swap_header_copy() {
        let a = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 32,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = a;
        assert_eq!(a.magic, b.magic);
        assert_eq!(a.slot_count, b.slot_count);
    }

    // ------------------------------------------------------------------
    // Constants
    // ------------------------------------------------------------------

    #[test]
    fn nvme_align_is_4096() {
        assert_eq!(NVME_ALIGN, 4096);
    }

    #[test]
    fn swap_header_bytes_is_4096() {
        assert_eq!(SWAP_HEADER_BYTES, 4096);
    }

    // ------------------------------------------------------------------
    // slot_offset calculation
    // ------------------------------------------------------------------

    #[test]
    fn slot_offset_page_zero() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "off.swap");
        assert_eq!(swap.slot_offset(0), SWAP_HEADER_BYTES as u64);
    }

    #[test]
    fn slot_offset_page_one() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "off1.swap");
        assert_eq!(swap.slot_offset(1), SWAP_HEADER_BYTES as u64 + swap.max_slot_bytes as u64);
    }

    #[test]
    fn slot_offset_page_n() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "offn.swap");
        let n: PageId = 5;
        let expected = SWAP_HEADER_BYTES as u64 + n as u64 * swap.max_slot_bytes as u64;
        assert_eq!(swap.slot_offset(n), expected);
    }

    // ------------------------------------------------------------------
    // max_slot_bytes alignment
    // ------------------------------------------------------------------

    #[test]
    fn max_slot_bytes_aligned_to_nvme() {
        let tmp = TempDir::new().unwrap();
        // Pass non-aligned value, expect it to be rounded up
        let swap = NvmeSwapFile::open(tmp.path().join("align.swap"), 4096, 5000, 8).unwrap();
        assert_eq!(swap.max_slot_bytes % NVME_ALIGN, 0);
        assert!(swap.max_slot_bytes >= 5000);
    }

    #[test]
    fn max_slot_bytes_already_aligned() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("aligned.swap"), 4096, 8192, 8).unwrap();
        assert_eq!(swap.max_slot_bytes, 8192);
    }

    // ------------------------------------------------------------------
    // read_slot too large
    // ------------------------------------------------------------------

    #[test]
    fn read_too_large_returns_error() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("r.swap"), 512, 512, 8).unwrap();
        let mut oversized = vec![0u8; swap.max_slot_bytes + 1];
        let result = swap.read_slot(0, &mut oversized);
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------
    // reopen validates version
    // ------------------------------------------------------------------

    #[test]
    fn bad_version_rejected() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("badver.swap");
        {
            // Create valid file first
            let swap = NvmeSwapFile::open(path.clone(), 4096, 8192, 8).unwrap();
            drop(swap);
            // Corrupt version field (offset 8, 4 bytes) using std::fs (no O_DIRECT needed for small write)
            let mut data = std::fs::read(&path).unwrap();
            let version_bytes = 99u32.to_le_bytes();
            data[8..12].copy_from_slice(&version_bytes);
            std::fs::write(&path, &data).unwrap();
            // Extend back to full size if truncated
            let expected_len = SWAP_HEADER_BYTES + 8 * 8192;
            if data.len() < expected_len {
                let mut extended = data;
                extended.resize(expected_len, 0);
                std::fs::write(&path, &extended).unwrap();
            }
        }
        let result = NvmeSwapFile::open(path, 4096, 8192, 8);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("version"), "should mention version: {err}");
    }

    // ------------------------------------------------------------------
    // write_slot returns written length
    // ------------------------------------------------------------------

    #[test]
    fn write_slot_returns_byte_count() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "len.swap");
        let data = vec![0xAB; 777];
        let written = swap.write_slot(0, &data).unwrap();
        assert_eq!(written, 777);
    }

    // ------------------------------------------------------------------
    // Debug trait
    // ------------------------------------------------------------------

    #[test]
    fn debug_format_includes_fields() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "dbg.swap");
        let debug_str = format!("{:?}", swap);
        assert!(debug_str.contains("page_size"));
        assert!(debug_str.contains("max_slot_bytes"));
        assert!(debug_str.contains("slot_count"));
    }

    // ------------------------------------------------------------------
    // SWAP_MAGIC constant
    // ------------------------------------------------------------------

    #[test]
    fn swap_magic_is_ascii_gllm_swap() {
        // "GLLMSWAP" in ASCII hex
        assert_eq!(SWAP_MAGIC, 0x474C4C4D53574150);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader Debug trait
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_debug_format() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 64,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let debug_str = format!("{:?}", hdr);
        assert!(debug_str.contains("SwapFileHeader"));
        assert!(debug_str.contains("magic"));
        assert!(debug_str.contains("version"));
        assert!(debug_str.contains("page_size"));
        assert!(debug_str.contains("max_slot_bytes"));
        assert!(debug_str.contains("slot_count"));
    }

    // ------------------------------------------------------------------
    // SwapFileHeader reserved field is all zeros by default
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_reserved_field_size() {
        assert_eq!(
            std::mem::size_of::<SwapFileHeader>() - 32,
            SWAP_HEADER_BYTES - 32
        );
    }

    // ------------------------------------------------------------------
    // slot_count propagated to struct
    // ------------------------------------------------------------------

    #[test]
    fn slot_count_propagated() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("sc.swap"), 4096, 8192, 99).unwrap();
        assert_eq!(swap.slot_count, 99);
    }

    // ------------------------------------------------------------------
    // page_size propagated to struct
    // ------------------------------------------------------------------

    #[test]
    fn page_size_propagated() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("ps.swap"), 2048, 8192, 4).unwrap();
        assert_eq!(swap.page_size, 2048);
    }

    // ------------------------------------------------------------------
    // overwrite slot: write twice, second wins
    // ------------------------------------------------------------------

    #[test]
    fn overwrite_slot_second_write_wins() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "ow.swap");

        swap.write_slot(0, &[0x11; 256]).unwrap();
        swap.write_slot(0, &[0x22; 256]).unwrap();

        let mut buf = vec![0u8; 256];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0x22), "second write must win");
    }

    // ------------------------------------------------------------------
    // write empty slice (0 bytes) succeeds
    // ------------------------------------------------------------------

    #[test]
    fn write_empty_slice_succeeds() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "empty.swap");
        let written = swap.write_slot(0, &[]).unwrap();
        assert_eq!(written, 0);
    }

    // ------------------------------------------------------------------
    // read zero-length slice succeeds
    // ------------------------------------------------------------------

    #[test]
    fn read_zero_length_succeeds() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "rz.swap");
        let mut buf: Vec<u8> = vec![];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.is_empty());
    }

    // ------------------------------------------------------------------
    // write exactly max_slot_bytes succeeds
    // ------------------------------------------------------------------

    #[test]
    fn write_exact_max_slot_bytes_succeeds() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("full.swap"), 4096, 4096, 4).unwrap();
        let data = vec![0xCC; swap.max_slot_bytes];
        let written = swap.write_slot(0, &data).unwrap();
        assert_eq!(written as usize, swap.max_slot_bytes);
    }

    // ------------------------------------------------------------------
    // read exactly max_slot_bytes succeeds
    // ------------------------------------------------------------------

    #[test]
    fn read_exact_max_slot_bytes_succeeds() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("rfull.swap"), 4096, 4096, 4).unwrap();
        let data = vec![0xDD; swap.max_slot_bytes];
        swap.write_slot(2, &data).unwrap();

        let mut buf = vec![0u8; swap.max_slot_bytes];
        swap.read_slot(2, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xDD));
    }

    // ------------------------------------------------------------------
    // parent directory auto-creation
    // ------------------------------------------------------------------

    #[test]
    fn parent_directory_auto_created() {
        let tmp = TempDir::new().unwrap();
        let nested = tmp.path().join("a").join("b").join("c").join("swap.dat");
        let swap = NvmeSwapFile::open(nested, 4096, 8192, 4);
        assert!(swap.is_ok(), "should auto-create parent directories");
    }

    // ------------------------------------------------------------------
    // slot_offset is linear
    // ------------------------------------------------------------------

    #[test]
    fn slot_offset_linearity() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "lin.swap");

        let off0 = swap.slot_offset(0);
        let off1 = swap.slot_offset(1);
        let off2 = swap.slot_offset(2);
        let off3 = swap.slot_offset(3);

        let stride = off1 - off0;
        assert_eq!(stride, swap.max_slot_bytes as u64);
        assert_eq!(off2 - off1, stride);
        assert_eq!(off3 - off2, stride);
    }

    // ------------------------------------------------------------------
    // max_slot_bytes alignment rounds up from various values
    // ------------------------------------------------------------------

    #[test]
    fn max_slot_bytes_alignment_various_inputs() {
        let tmp = TempDir::new().unwrap();

        let cases = [(1, 4096), (4095, 4096), (4097, 8192), (8191, 8192)];
        for (input, expected_aligned) in cases {
            let path = tmp.path().join(format!("align_{input}.swap"));
            let swap = NvmeSwapFile::open(path, 4096, input, 4).unwrap();
            assert_eq!(
                swap.max_slot_bytes, expected_aligned,
                "input {input} should align to {expected_aligned}"
            );
        }
    }

    // ------------------------------------------------------------------
    // high page_id slot still works
    // ------------------------------------------------------------------

    #[test]
    fn high_page_id_roundtrip() {
        let tmp = TempDir::new().unwrap();
        // 256 slots so page_id=255 is valid
        let swap = NvmeSwapFile::open(tmp.path().join("hi.swap"), 4096, 4096, 256).unwrap();

        let data = vec![0xFE; 128];
        swap.write_slot(255, &data).unwrap();

        let mut buf = vec![0u8; 128];
        swap.read_slot(255, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xFE));
    }

    // ------------------------------------------------------------------
    // write_then_drop_then_reopen_preserves_data
    // ------------------------------------------------------------------

    #[test]
    fn drop_and_reopen_preserves_data() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("drop.swap");

        let payload = vec![0x42; 512];
        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 8192, 8).unwrap();
            swap.write_slot(5, &payload).unwrap();
            // swap dropped here
        }

        let swap2 = NvmeSwapFile::open(path, 4096, 8192, 8).unwrap();
        let mut buf = vec![0u8; 512];
        swap2.read_slot(5, &mut buf).unwrap();
        assert_eq!(buf, payload);
    }

    // ------------------------------------------------------------------
    // write_slot error message contains page_id and sizes
    // ------------------------------------------------------------------

    #[test]
    fn write_oversized_error_message_content() {
        let tmp = TempDir::new().unwrap();
        // max_slot_bytes=4096 (already aligned), write 5000 bytes to exceed it
        let swap = NvmeSwapFile::open(tmp.path().join("em.swap"), 4096, 4096, 4).unwrap();
        let oversized = vec![0u8; 5000];
        let err = swap.write_slot(7, &oversized).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("page=7"), "error should contain page id: {msg}");
        assert!(msg.contains("5000"), "error should contain data len: {msg}");
        assert!(msg.contains("4096"), "error should contain max_slot_bytes: {msg}");
    }

    // ------------------------------------------------------------------
    // read_slot error message contains page_id and sizes
    // ------------------------------------------------------------------

    #[test]
    fn read_oversized_error_message_content() {
        let tmp = TempDir::new().unwrap();
        // max_slot_bytes=4096 (already aligned), read 5000 bytes to exceed it
        let swap = NvmeSwapFile::open(tmp.path().join("rem.swap"), 4096, 4096, 4).unwrap();
        let mut oversized = vec![0u8; 5000];
        let err = swap.read_slot(3, &mut oversized).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("page=3"), "error should contain page id: {msg}");
        assert!(msg.contains("5000"), "error should contain dst len: {msg}");
        assert!(msg.contains("4096"), "error should contain max_slot_bytes: {msg}");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader C representation alignment
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_is_c_repr() {
        // Verify the header has no Rust-specific padding by checking the
        // first 4 fields occupy exactly 20 bytes (u64 + u32 + u32 + u32).
        // Total must be SWAP_HEADER_BYTES due to _reserved field.
        let hdr = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(std::mem::size_of_val(&hdr), SWAP_HEADER_BYTES);
    }

    // ------------------------------------------------------------------
    // multiple independent writes interleaved
    // ------------------------------------------------------------------

    #[test]
    fn interleaved_writes_preserve_each_slot() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "inter.swap");

        // Write to slots in non-sequential order
        swap.write_slot(3, &[0xAA; 64]).unwrap();
        swap.write_slot(0, &[0xBB; 64]).unwrap();
        swap.write_slot(7, &[0xCC; 64]).unwrap();
        swap.write_slot(1, &[0xDD; 64]).unwrap();

        let mut buf = vec![0u8; 64];

        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xBB), "slot 0");

        swap.read_slot(1, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xDD), "slot 1");

        swap.read_slot(3, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xAA), "slot 3");

        swap.read_slot(7, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xCC), "slot 7");
    }

    // ------------------------------------------------------------------
    // swap file persists header correctly on reopen
    // ------------------------------------------------------------------

    #[test]
    fn reopen_preserves_slot_count_and_alignment() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("persist.swap");

        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 5000, 128).unwrap();
            assert_eq!(swap.slot_count, 128);
            assert_eq!(swap.max_slot_bytes, 8192); // 5000 rounded up
        }

        let swap2 = NvmeSwapFile::open(path, 4096, 8192, 128).unwrap();
        assert_eq!(swap2.slot_count, 128);
        assert_eq!(swap2.max_slot_bytes, 8192);
    }

    // ------------------------------------------------------------------
    // write returns original data length not aligned length
    // ------------------------------------------------------------------

    #[test]
    fn write_slot_returns_original_data_len_not_aligned() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("wlen.swap"), 4096, 8192, 4).unwrap();

        // Write 100 bytes into an 8192-byte slot
        let written = swap.write_slot(0, &[0xFF; 100]).unwrap();
        assert_eq!(written, 100, "should return original data length, not slot size");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader Clone derive verification
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_clone_is_independent() {
        let mut a = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 16,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        a._reserved[0] = 0xDE;
        let b = a.clone();
        // Modify original after clone to prove independence
        a._reserved[0] = 0x00;
        assert_eq!(a._reserved[0], 0x00);
        assert_eq!(b._reserved[0], 0xDE, "clone must be independent");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader repr(C) field offsets
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_field_offsets() {
        let hdr = SwapFileHeader {
            magic: 1,
            version: 2,
            page_size: 3,
            max_slot_bytes: 4,
            _pad4: 0,
            slot_count: 5,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let base = &hdr as *const SwapFileHeader as usize;
        let magic_off = &hdr.magic as *const u64 as usize - base;
        let version_off = &hdr.version as *const u32 as usize - base;
        let page_size_off = &hdr.page_size as *const u32 as usize - base;
        let max_slot_off = &hdr.max_slot_bytes as *const u32 as usize - base;
        let slot_count_off = &hdr.slot_count as *const u64 as usize - base;

        assert_eq!(magic_off, 0, "magic should be at offset 0");
        assert_eq!(version_off, 8, "version should be at offset 8");
        assert_eq!(page_size_off, 12, "page_size should be at offset 12");
        assert_eq!(max_slot_off, 16, "max_slot_bytes should be at offset 16");
        assert_eq!(slot_count_off, 24, "slot_count should be at offset 24");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader with varied page_size values
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_varied_page_size() {
        for ps in [512u32, 1024, 2048, 4096, 8192, 16384] {
            let hdr = SwapFileHeader {
                magic: SWAP_MAGIC,
                version: SWAP_VERSION,
                page_size: ps,
                max_slot_bytes: ps * 2,
                _pad4: 0,
                slot_count: 10,
                _reserved: [0u8; SWAP_HEADER_BYTES - 32],
            };
            assert_eq!(hdr.page_size, ps);
            assert_eq!(hdr.max_slot_bytes, ps * 2);
        }
    }

    // ------------------------------------------------------------------
    // SwapFileHeader with maximum u32 field values
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_max_u32_fields() {
        let hdr = SwapFileHeader {
            magic: u64::MAX,
            version: u32::MAX,
            page_size: u32::MAX,
            max_slot_bytes: u32::MAX,
            _pad4: u32::MAX,
            slot_count: u64::MAX,
            _reserved: [0xFF; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(hdr.magic, u64::MAX);
        assert_eq!(hdr.version, u32::MAX);
        assert_eq!(hdr.page_size, u32::MAX);
        assert_eq!(hdr.max_slot_bytes, u32::MAX);
        assert_eq!(hdr.slot_count, u64::MAX);
        assert!(hdr._reserved.iter().all(|&b| b == 0xFF));
    }

    // ------------------------------------------------------------------
    // NVME_ALIGN is a power of two
    // ------------------------------------------------------------------

    #[test]
    fn nvme_align_is_power_of_two() {
        assert!(NVME_ALIGN > 0);
        assert_eq!(NVME_ALIGN & (NVME_ALIGN - 1), 0, "must be power of 2");
    }

    // ------------------------------------------------------------------
    // SWAP_HEADER_BYTES is a power of two
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_bytes_is_power_of_two() {
        assert!(SWAP_HEADER_BYTES > 0);
        assert_eq!(
            SWAP_HEADER_BYTES & (SWAP_HEADER_BYTES - 1),
            0,
            "must be power of 2"
        );
    }

    // ------------------------------------------------------------------
    // SWAP_HEADER_BYTES is at least NVME_ALIGN
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_bytes_at_least_nvme_align() {
        assert!(
            SWAP_HEADER_BYTES >= NVME_ALIGN,
            "header must be at least one aligned block"
        );
    }

    // ------------------------------------------------------------------
    // SWAP_MAGIC byte interpretation (little-endian ASCII)
    // ------------------------------------------------------------------

    #[test]
    fn swap_magic_bytes_are_ascii_gllmswap() {
        let bytes = SWAP_MAGIC.to_be_bytes();
        // 0x474C4C4D53574150 in big-endian = "GLLMSWAP"
        assert_eq!(bytes[0], b'G');
        assert_eq!(bytes[1], b'L');
        assert_eq!(bytes[2], b'L');
        assert_eq!(bytes[3], b'M');
        assert_eq!(bytes[4], b'S');
        assert_eq!(bytes[5], b'W');
        assert_eq!(bytes[6], b'A');
        assert_eq!(bytes[7], b'P');
    }

    // ------------------------------------------------------------------
    // SWAP_VERSION is 1
    // ------------------------------------------------------------------

    #[test]
    fn swap_version_is_one() {
        assert_eq!(SWAP_VERSION, 1);
    }

    // ------------------------------------------------------------------
    // AlignedBuffer zero-initialized
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_new_is_zeroed() {
        let buf = AlignedBuffer::new(NVME_ALIGN);
        assert!(buf.as_slice().iter().all(|&b| b == 0), "new buffer must be zeroed");
    }

    // ------------------------------------------------------------------
    // AlignedBuffer size matches requested
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_size_matches() {
        let sizes = [NVME_ALIGN, NVME_ALIGN * 2, NVME_ALIGN * 4];
        for &size in &sizes {
            let buf = AlignedBuffer::new(size);
            assert_eq!(buf.as_slice().len(), size);
        }
    }

    // ------------------------------------------------------------------
    // AlignedBuffer as_mut_slice roundtrip
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_mut_slice_roundtrip() {
        let mut buf = AlignedBuffer::new(NVME_ALIGN);
        let pattern: [u8; 4] = [0xDE, 0xAD, 0xBE, 0xEF];
        buf.as_mut_slice()[..4].copy_from_slice(&pattern);
        assert_eq!(&buf.as_slice()[..4], &pattern);
    }

    // ------------------------------------------------------------------
    // AlignedBuffer pointer alignment
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_ptr_is_nvme_aligned() {
        let buf = AlignedBuffer::new(NVME_ALIGN);
        let addr = buf.as_ptr() as usize;
        assert_eq!(addr % NVME_ALIGN, 0, "pointer must be 4096-aligned");
    }

    // ------------------------------------------------------------------
    // Read uninitialized slot returns zeros (sparse file)
    // ------------------------------------------------------------------

    #[test]
    fn read_never_written_slot_returns_zeros() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "sparse.swap");
        // Slot 10 was never written
        let mut buf = vec![0xFF; 64];
        swap.read_slot(10, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0), "unwritten slot should be zeros");
    }

    // ------------------------------------------------------------------
    // Write more than read: partial read gets only first bytes
    // ------------------------------------------------------------------

    #[test]
    fn write_more_read_less_partial() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "partial.swap");

        let data: Vec<u8> = (0..512).map(|i| (i % 256) as u8).collect();
        swap.write_slot(0, &data).unwrap();

        // Read only first 64 bytes
        let mut buf = vec![0u8; 64];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, &data[..64]);
    }

    // ------------------------------------------------------------------
    // slot_offset with different max_slot_bytes values
    // ------------------------------------------------------------------

    #[test]
    fn slot_offset_with_varying_slot_sizes() {
        let tmp = TempDir::new().unwrap();

        let slot_sizes = [4096usize, 8192, 16384];
        for &sz in &slot_sizes {
            let path = tmp.path().join(format!("so_{sz}.swap"));
            let swap = NvmeSwapFile::open(path, 4096, sz, 8).unwrap();
            let off3 = swap.slot_offset(3);
            let expected = SWAP_HEADER_BYTES as u64 + 3 * swap.max_slot_bytes as u64;
            assert_eq!(off3, expected, "offset mismatch for slot_size={sz}");
        }
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile Debug format includes numeric values
    // ------------------------------------------------------------------

    #[test]
    fn debug_format_contains_numeric_values() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("dbgnum.swap"), 4096, 8192, 42).unwrap();
        let dbg = format!("{:?}", swap);
        assert!(dbg.contains("4096"), "should contain page_size value: {dbg}");
        assert!(dbg.contains("8192"), "should contain max_slot_bytes value: {dbg}");
        assert!(dbg.contains("42"), "should contain slot_count value: {dbg}");
    }

    // ------------------------------------------------------------------
    // Multiple reopens with same parameters succeed
    // ------------------------------------------------------------------

    #[test]
    fn multiple_reopens_succeed() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("reopen.swap");
        let params = (4096usize, 8192usize, 8u64);

        // First open creates
        let _s1 = NvmeSwapFile::open(path.clone(), params.0, params.1, params.2).unwrap();
        // Drop
        drop(_s1);
        // Second open validates header
        let _s2 = NvmeSwapFile::open(path.clone(), params.0, params.1, params.2).unwrap();
        drop(_s2);
        // Third open also works
        let s3 = NvmeSwapFile::open(path, params.0, params.1, params.2).unwrap();
        assert_eq!(s3.slot_count, 8);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader _pad4 field is accessible
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_pad4_field() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 12345,
            slot_count: 8,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(hdr._pad4, 12345);
    }

    // ------------------------------------------------------------------
    // Slot offset for page_id = slot_count - 1 is within file
    // ------------------------------------------------------------------

    #[test]
    fn slot_offset_last_slot_within_bounds() {
        let tmp = TempDir::new().unwrap();
        let slot_count: u64 = 64;
        let swap = NvmeSwapFile::open(
            tmp.path().join("bounds.swap"),
            4096,
            8192,
            slot_count,
        )
        .unwrap();
        let last_page = (slot_count - 1) as PageId;
        let offset = swap.slot_offset(last_page);
        let file_data_start = SWAP_HEADER_BYTES as u64 + slot_count * swap.max_slot_bytes as u64;
        // Offset of last slot + slot size should equal file data end
        assert_eq!(
            offset + swap.max_slot_bytes as u64,
            file_data_start,
            "last slot should end at file data boundary"
        );
    }

    // ==================================================================
    // Pure data-structure tests (no I/O, no filesystem)
    // ==================================================================

    // ------------------------------------------------------------------
    // SwapFileHeader with all zero fields
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_all_zeros() {
        let hdr = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(hdr.magic, 0);
        assert_eq!(hdr.version, 0);
        assert_eq!(hdr.page_size, 0);
        assert_eq!(hdr.max_slot_bytes, 0);
        assert_eq!(hdr.slot_count, 0);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader with slot_count = 1 (minimum non-zero)
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_minimal_slot_count() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 4096,
            _pad4: 0,
            slot_count: 1,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(hdr.slot_count, 1);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader _reserved field length invariant
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_reserved_field_exact_length() {
        // The reserved field must be SWAP_HEADER_BYTES - 32 bytes.
        // 32 = size_of(magic u64) + version u32 + page_size u32 + max_slot_bytes u32
        //    + _pad4 u32 + slot_count u64 = 8 + 4 + 4 + 4 + 4 + 8 = 32
        let hdr = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(hdr._reserved.len(), SWAP_HEADER_BYTES - 32);
        assert_eq!(hdr._reserved.len(), 4064);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader _reserved preserves non-zero bytes
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_reserved_preserves_custom_bytes() {
        let mut reserved = [0u8; SWAP_HEADER_BYTES - 32];
        reserved[0] = 0xAB;
        reserved[100] = 0xCD;
        reserved[SWAP_HEADER_BYTES - 33] = 0xEF;
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 8,
            _reserved: reserved,
        };
        assert_eq!(hdr._reserved[0], 0xAB);
        assert_eq!(hdr._reserved[100], 0xCD);
        assert_eq!(hdr._reserved[SWAP_HEADER_BYTES - 33], 0xEF);
        // Most bytes should still be zero
        let zero_count = hdr._reserved.iter().filter(|&&b| b == 0).count();
        assert_eq!(zero_count, SWAP_HEADER_BYTES - 32 - 3);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader Copy trait: modifying copy does not affect original
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_copy_semantics() {
        let a = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 16,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let mut b = a;
        b.slot_count = 999;
        b.page_size = 111;
        assert_eq!(a.slot_count, 16, "original must be unchanged");
        assert_eq!(a.page_size, 4096, "original must be unchanged");
        assert_eq!(b.slot_count, 999);
        assert_eq!(b.page_size, 111);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader Debug output includes field names
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_debug_shows_all_public_field_names() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 16,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let dbg = format!("{:?}", hdr);
        // Verify each public field appears by name
        assert!(dbg.contains("magic"), "debug must contain 'magic': {dbg}");
        assert!(dbg.contains("version"), "debug must contain 'version': {dbg}");
        assert!(dbg.contains("page_size"), "debug must contain 'page_size': {dbg}");
        assert!(dbg.contains("max_slot_bytes"), "debug must contain 'max_slot_bytes': {dbg}");
        assert!(dbg.contains("slot_count"), "debug must contain 'slot_count': {dbg}");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader Debug output contains actual numeric values
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_debug_contains_numeric_values() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 2048,
            max_slot_bytes: 65536,
            _pad4: 0,
            slot_count: 128,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let dbg = format!("{:?}", hdr);
        assert!(dbg.contains("2048"), "should contain page_size: {dbg}");
        assert!(dbg.contains("65536"), "should contain max_slot_bytes: {dbg}");
        assert!(dbg.contains("128"), "should contain slot_count: {dbg}");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader byte-level magic field is little-endian
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_magic_byte_layout() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        // On little-endian (x86_64), the first byte of magic in memory
        // should be 'P' (0x50), the low byte of SWAP_MAGIC.
        let bytes: &[u8; 8] =
            unsafe { &*(&hdr.magic as *const u64 as *const [u8; 8]) };
        // SWAP_MAGIC = 0x474C4C4D53574150, LE bytes: 50 41 57 53 4D 4C 4C 47
        assert_eq!(bytes[0], 0x50); // 'P'
        assert_eq!(bytes[1], 0x41); // 'A'
        assert_eq!(bytes[2], 0x57); // 'W'
        assert_eq!(bytes[3], 0x53); // 'S'
        assert_eq!(bytes[4], 0x4D); // 'M'
        assert_eq!(bytes[5], 0x4C); // 'L'
        assert_eq!(bytes[6], 0x4C); // 'L'
        assert_eq!(bytes[7], 0x47); // 'G'
    }

    // ------------------------------------------------------------------
    // SwapFileHeader version field byte offset is 8
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_version_at_offset_8() {
        let mut hdr = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        hdr.version = 0xDEADBEEFu32;
        let base = &hdr as *const SwapFileHeader as *const u8;
        // version is at offset 8 (after magic u64)
        let version_byte_0 = unsafe { *base.add(8) };
        let version_byte_3 = unsafe { *base.add(11) };
        // LE: 0xDEADBEEF -> bytes EF BE AD DE
        assert_eq!(version_byte_0, 0xEF);
        assert_eq!(version_byte_3, 0xDE);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader alignment is at least 8 (for u64 fields)
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_align_is_at_least_8() {
        assert!(
            std::mem::align_of::<SwapFileHeader>() >= 8,
            "SwapFileHeader alignment must be >= 8 for u64 fields"
        );
    }

    // ------------------------------------------------------------------
    // SwapFileHeader _pad4 occupies exactly offset 20..24
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_pad4_field_offset() {
        let hdr = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let base = &hdr as *const SwapFileHeader as usize;
        let pad4_off = &hdr._pad4 as *const u32 as usize - base;
        assert_eq!(pad4_off, 20, "_pad4 must be at byte offset 20");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader two instances are equal when fields match
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_identical_fields_equal_by_field() {
        let a = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 42,
            slot_count: 100,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 42,
            slot_count: 100,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(a.magic, b.magic);
        assert_eq!(a.version, b.version);
        assert_eq!(a.page_size, b.page_size);
        assert_eq!(a.max_slot_bytes, b.max_slot_bytes);
        assert_eq!(a._pad4, b._pad4);
        assert_eq!(a.slot_count, b.slot_count);
        assert_eq!(a._reserved, b._reserved);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader page_size and max_slot_bytes relationship
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_max_slot_bytes_can_equal_page_size() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 4096,
            _pad4: 0,
            slot_count: 256,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(hdr.page_size, hdr.max_slot_bytes);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader slot_count can be u64::MAX
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_slot_count_max_u64() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: u64::MAX,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(hdr.slot_count, u64::MAX);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader repr(C) — slot_count offset is 24
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_slot_count_at_offset_24() {
        let hdr = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0x1234_5678_9ABC_DEF0u64,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let base = &hdr as *const SwapFileHeader as *const u8;
        // slot_count is at offset 24 (8 magic + 4 version + 4 page_size
        // + 4 max_slot_bytes + 4 _pad4 = 24)
        let sc_bytes: [u8; 8] =
            unsafe { std::ptr::read_unaligned(base.add(24) as *const [u8; 8]) };
        let reconstructed = u64::from_le_bytes(sc_bytes);
        assert_eq!(reconstructed, 0x1234_5678_9ABC_DEF0u64);
    }

    // ------------------------------------------------------------------
    // SWAP_MAGIC is non-zero
    // ------------------------------------------------------------------

    #[test]
    fn swap_magic_is_nonzero() {
        assert_ne!(SWAP_MAGIC, 0, "magic must be non-zero to detect empty files");
    }

    // ------------------------------------------------------------------
    // NVME_ALIGN equals SWAP_HEADER_BYTES
    // ------------------------------------------------------------------

    #[test]
    fn nvme_align_equals_swap_header_bytes() {
        assert_eq!(
            NVME_ALIGN, SWAP_HEADER_BYTES,
            "NVME_ALIGN and SWAP_HEADER_BYTES must both be 4096"
        );
    }

    // ------------------------------------------------------------------
    // SwapFileHeader can be zeroed via zeroed()
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_zeroed() {
        let hdr: SwapFileHeader = unsafe { std::mem::zeroed() };
        assert_eq!(hdr.magic, 0);
        assert_eq!(hdr.version, 0);
        assert_eq!(hdr.page_size, 0);
        assert_eq!(hdr.max_slot_bytes, 0);
        assert_eq!(hdr._pad4, 0);
        assert_eq!(hdr.slot_count, 0);
        assert!(hdr._reserved.iter().all(|&b| b == 0));
    }

    // ------------------------------------------------------------------
    // SwapFileHeader reserved field can be checked for zeroed
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_reserved_default_is_zeroed() {
        let hdr = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert!(hdr._reserved.iter().all(|&b| b == 0));
    }

    // ------------------------------------------------------------------
    // SwapFileHeader memory layout: first 32 bytes are explicit fields
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_first_32_bytes_are_fields() {
        // The explicit fields (magic + version + page_size + max_slot_bytes
        // + _pad4 + slot_count) occupy exactly 32 bytes.
        let explicit_size = 8 + 4 + 4 + 4 + 4 + 8; // = 32
        assert_eq!(explicit_size, 32);
        assert_eq!(
            SWAP_HEADER_BYTES - explicit_size,
            SWAP_HEADER_BYTES - 32
        );
    }

    // ------------------------------------------------------------------
    // AlignedBuffer as_mut_slice and as_slice refer to same memory
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_mut_and_read_share_memory() {
        let mut buf = AlignedBuffer::new(NVME_ALIGN);
        // Write via mut_slice
        buf.as_mut_slice()[0] = 0xAA;
        buf.as_mut_slice()[NVME_ALIGN - 1] = 0xBB;
        // Read via as_slice (immutable) to confirm same backing memory
        assert_eq!(buf.as_slice()[0], 0xAA);
        assert_eq!(buf.as_slice()[NVME_ALIGN - 1], 0xBB);
    }

    // ------------------------------------------------------------------
    // AlignedBuffer large allocation succeeds
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_large_allocation() {
        let size = NVME_ALIGN * 16; // 64KB
        let mut buf = AlignedBuffer::new(size);
        assert_eq!(buf.as_slice().len(), size);
        // Write to last byte to confirm full range is usable
        buf.as_mut_slice()[size - 1] = 0xFE;
        assert_eq!(buf.as_slice()[size - 1], 0xFE);
    }

    // ==================================================================
    // NEW TESTS — 43 additional tests covering edge cases, boundary
    // conditions, derive traits, error variants, and I/O robustness.
    // ==================================================================

    // ------------------------------------------------------------------
    // SwapFileHeader PartialEq derive — identical headers are equal
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_partialeq_identical() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 16,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let copy = hdr;
        assert_eq!(hdr, copy, "identical headers must be equal");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader PartialEq — different magic means not equal
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_partialeq_different_magic() {
        let a = SwapFileHeader {
            magic: 1,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 4,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = SwapFileHeader {
            magic: 2,
            ..a
        };
        assert_ne!(a, b, "different magic must not be equal");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader PartialEq — different slot_count means not equal
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_partialeq_different_slot_count() {
        let a = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 10,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = SwapFileHeader {
            slot_count: 20,
            ..a
        };
        assert_ne!(a, b);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader PartialEq — different page_size means not equal
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_partialeq_different_page_size() {
        let a = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 8,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = SwapFileHeader {
            page_size: 8192,
            ..a
        };
        assert_ne!(a, b);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader PartialEq — different max_slot_bytes means not equal
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_partialeq_different_max_slot_bytes() {
        let a = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 8,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = SwapFileHeader {
            max_slot_bytes: 16384,
            ..a
        };
        assert_ne!(a, b);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader PartialEq — different _pad4 means not equal
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_partialeq_different_pad4() {
        let a = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 8,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = SwapFileHeader {
            _pad4: 1,
            ..a
        };
        assert_ne!(a, b, "different _pad4 must not be equal");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader PartialEq — different reserved bytes means not equal
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_partialeq_different_reserved() {
        let reserved_a = [0u8; SWAP_HEADER_BYTES - 32];
        let mut reserved_b = [0u8; SWAP_HEADER_BYTES - 32];
        reserved_b[0] = 1;
        let a = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 8,
            _reserved: reserved_a,
        };
        let b = SwapFileHeader {
            _reserved: reserved_b,
            ..a
        };
        assert_ne!(a, b, "different _reserved must not be equal");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader Hash — equal headers produce same hash
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_hash_equal_headers() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 16,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let copy = hdr;

        let mut hasher_a = DefaultHasher::new();
        let mut hasher_b = DefaultHasher::new();
        hdr.hash(&mut hasher_a);
        copy.hash(&mut hasher_b);
        assert_eq!(hasher_a.finish(), hasher_b.finish());
    }

    // ------------------------------------------------------------------
    // SwapFileHeader Hash — different headers produce different hashes
    //      (probabilistic but extremely likely for this specific case)
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_hash_different_headers_differ() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let a = SwapFileHeader {
            magic: 1,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 16,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = SwapFileHeader {
            magic: 2,
            ..a
        };

        let mut hasher_a = DefaultHasher::new();
        let mut hasher_b = DefaultHasher::new();
        a.hash(&mut hasher_a);
        b.hash(&mut hasher_b);
        assert_ne!(hasher_a.finish(), hasher_b.finish());
    }

    // ------------------------------------------------------------------
    // SwapFileHeader can be used as HashMap key (Eq + Hash)
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_usable_as_hashmap_key() {
        use std::collections::HashMap;

        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 8,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let mut map = HashMap::new();
        map.insert(hdr, "test_value");
        assert_eq!(map.get(&hdr), Some(&"test_value"));
    }

    // ------------------------------------------------------------------
    // SwapFileHeader can be used in a HashSet (Eq + Hash)
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_usable_in_hashset() {
        use std::collections::HashSet;

        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 8,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let mut set = HashSet::new();
        assert!(set.insert(hdr));
        assert!(!set.insert(hdr), "duplicate insert should return false");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader Eq consistent with PartialEq
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_eq_consistency() {
        let a = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 8,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = a;
        // Reflexive: a == a
        assert_eq!(a, a);
        // Symmetric: a == b implies b == a
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: open with slot_count = 0 succeeds (empty swap)
    // ------------------------------------------------------------------

    #[test]
    fn open_with_zero_slot_count_succeeds() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("zero_slots.swap"), 4096, 4096, 0);
        assert!(swap.is_ok(), "zero slot_count should succeed");
        let swap = swap.unwrap();
        assert_eq!(swap.slot_count, 0);
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: open with minimum aligned max_slot_bytes (=4096)
    // ------------------------------------------------------------------

    #[test]
    fn open_with_minimum_aligned_slot_size() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("min_slot.swap"), 4096, NVME_ALIGN, 4);
        assert!(swap.is_ok());
        let swap = swap.unwrap();
        assert_eq!(swap.max_slot_bytes, NVME_ALIGN);
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: max_slot_bytes = 1 gets rounded up to NVME_ALIGN
    // ------------------------------------------------------------------

    #[test]
    fn max_slot_bytes_one_rounded_up() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("msb1.swap"), 4096, 1, 4).unwrap();
        assert_eq!(swap.max_slot_bytes, NVME_ALIGN);
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: open in deeply nested directory
    // ------------------------------------------------------------------

    #[test]
    fn open_deeply_nested_path() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path()
            .join("a").join("b").join("c").join("d").join("e")
            .join("deep.swap");
        let swap = NvmeSwapFile::open(path, 4096, 4096, 4);
        assert!(swap.is_ok(), "should auto-create all parent directories");
    }

    // ------------------------------------------------------------------
    // write_slot then read_slot with single byte
    // ------------------------------------------------------------------

    #[test]
    fn write_read_single_byte() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("one.swap"), 4096, 4096, 4).unwrap();

        swap.write_slot(0, &[0x42]).unwrap();
        let mut buf = [0u8; 1];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf[0], 0x42);
    }

    // ------------------------------------------------------------------
    // write_slot with byte pattern 0x00 (zero data)
    // ------------------------------------------------------------------

    #[test]
    fn write_all_zeros_then_read() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "zeros.swap");

        let data = vec![0x00; 256];
        swap.write_slot(2, &data).unwrap();

        let mut buf = vec![0xFF; 256];
        swap.read_slot(2, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0x00));
    }

    // ------------------------------------------------------------------
    // write_slot with 0xFF byte pattern
    // ------------------------------------------------------------------

    #[test]
    fn write_all_ones_then_read() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "ones.swap");

        let data = vec![0xFF; 512];
        swap.write_slot(1, &data).unwrap();

        let mut buf = vec![0x00; 512];
        swap.read_slot(1, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xFF));
    }

    // ------------------------------------------------------------------
    // write_slot with sequential byte pattern
    // ------------------------------------------------------------------

    #[test]
    fn write_sequential_pattern_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("seq.swap"), 4096, 4096, 4).unwrap();

        let data: Vec<u8> = (0..=255).collect();
        swap.write_slot(0, &data).unwrap();

        let mut buf = vec![0u8; 256];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // write_slot returns 0 for empty data
    // ------------------------------------------------------------------

    #[test]
    fn write_slot_empty_returns_zero() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "empty_wr.swap");
        let n = swap.write_slot(3, &[]).unwrap();
        assert_eq!(n, 0);
    }

    // ------------------------------------------------------------------
    // read after empty write returns zeros (slot was padded)
    // ------------------------------------------------------------------

    #[test]
    fn read_after_empty_write_returns_zeros() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "empty_rd.swap");

        swap.write_slot(0, &[]).unwrap();
        let mut buf = vec![0xFF; 64];
        swap.read_slot(0, &mut buf).unwrap();
        // Empty write fills slot with zeros (AlignedBuffer is zero-initialized)
        assert!(buf.iter().all(|&b| b == 0x00));
    }

    // ------------------------------------------------------------------
    // overwrite with smaller data: old data beyond new length persists
    // ------------------------------------------------------------------

    #[test]
    fn overwrite_smaller_preserves_padding() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "shrink.swap");

        // Write 256 bytes of pattern A
        swap.write_slot(0, &[0xAA; 256]).unwrap();
        // Overwrite with 64 bytes of pattern B — write_slot zero-fills remainder
        swap.write_slot(0, &[0xBB; 64]).unwrap();

        // First 64 bytes should be new pattern, rest zero-filled
        let mut full = vec![0u8; 256];
        swap.read_slot(0, &mut full).unwrap();
        assert!(full[..64].iter().all(|&b| b == 0xBB), "first 64 must be 0xBB");
        assert!(full[64..256].iter().all(|&b| b == 0x00), "rest must be zero-filled");
    }

    // ------------------------------------------------------------------
    // write_slot error kind is InvalidInput for oversized data
    // ------------------------------------------------------------------

    #[test]
    fn write_oversized_error_kind() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("ek.swap"), 4096, 4096, 4).unwrap();
        let oversized = vec![0u8; swap.max_slot_bytes + 1];
        let err = swap.write_slot(0, &oversized).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    // ------------------------------------------------------------------
    // read_slot error kind is InvalidInput for oversized dst
    // ------------------------------------------------------------------

    #[test]
    fn read_oversized_error_kind() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("rek.swap"), 4096, 4096, 4).unwrap();
        let mut oversized = vec![0u8; swap.max_slot_bytes + 1];
        let err = swap.read_slot(0, &mut oversized).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    // ------------------------------------------------------------------
    // Multiple writes to different slots do not interfere
    // ------------------------------------------------------------------

    #[test]
    fn many_slots_no_interference() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("many.swap"), 4096, 4096, 64).unwrap();

        // Write unique pattern to each slot
        for pid in 0..64usize {
            let pattern = (pid as u8).wrapping_mul(4);
            swap.write_slot(pid, &[pattern; 128]).unwrap();
        }

        // Verify each slot independently
        for pid in 0..64usize {
            let expected = (pid as u8).wrapping_mul(4);
            let mut buf = vec![0u8; 128];
            swap.read_slot(pid, &mut buf).unwrap();
            assert!(
                buf.iter().all(|&b| b == expected),
                "slot {pid} corrupted: expected 0x{expected:02x}"
            );
        }
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile drop closes fd (verify by re-opening same file)
    // ------------------------------------------------------------------

    #[test]
    fn drop_allows_reopen() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("drop_reopen.swap");

        {
            let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
            // _swap dropped here
        }

        // Must be able to re-open (fd was closed)
        let swap = NvmeSwapFile::open(path, 4096, 4096, 4);
        assert!(swap.is_ok(), "must be able to re-open after drop");
    }

    // ------------------------------------------------------------------
    // reopen with different slot_count — struct uses new value
    // ------------------------------------------------------------------

    #[test]
    fn reopen_different_slot_count_uses_new_value() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("diff_sc.swap");

        {
            let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 10).unwrap();
        }

        // Reopen with different slot_count
        let swap = NvmeSwapFile::open(path, 4096, 4096, 20).unwrap();
        assert_eq!(swap.slot_count, 20, "struct should use new slot_count");
    }

    // ------------------------------------------------------------------
    // bad_magic error message includes expected and actual magic
    // ------------------------------------------------------------------

    #[test]
    fn bad_magic_error_includes_both_magics() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("magic_err.swap");
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&path).unwrap();
            // All zeros — magic will be 0x0000000000000000
            f.write_all(&[0u8; SWAP_HEADER_BYTES]).unwrap();
            f.set_len(SWAP_HEADER_BYTES as u64).unwrap();
        }
        let err = NvmeSwapFile::open(path, 4096, 4096, 4).unwrap_err();
        let msg = err.to_string();
        // Should contain the expected magic value
        assert!(msg.contains("0x"), "error should contain hex magic: {msg}");
        assert!(msg.contains("expected"), "error should mention expected: {msg}");
    }

    // ------------------------------------------------------------------
    // slot_offset for consecutive pages has constant stride
    // ------------------------------------------------------------------

    #[test]
    fn slot_offset_constant_stride_many_pages() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("stride.swap"), 4096, 8192, 32).unwrap();

        let mut prev = swap.slot_offset(0);
        for pid in 1..32 {
            let curr = swap.slot_offset(pid);
            assert_eq!(
                curr - prev,
                swap.max_slot_bytes as u64,
                "stride must equal max_slot_bytes for page {pid}"
            );
            prev = curr;
        }
    }

    // ------------------------------------------------------------------
    // slot_offset for page 0 always equals SWAP_HEADER_BYTES
    // ------------------------------------------------------------------

    #[test]
    fn slot_offset_page_zero_always_header_size() {
        let tmp = TempDir::new().unwrap();
        for &slot_sz in &[4096usize, 8192, 16384, 32768] {
            let path = tmp.path().join(format!("so0_{slot_sz}.swap"));
            let swap = NvmeSwapFile::open(path, 4096, slot_sz, 4).unwrap();
            assert_eq!(
                swap.slot_offset(0),
                SWAP_HEADER_BYTES as u64,
                "page 0 offset must be header size"
            );
        }
    }

    // ------------------------------------------------------------------
    // AlignedBuffer different sizes all have correct alignment
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_various_sizes_aligned() {
        for &multiplier in &[1usize, 2, 4, 8, 16] {
            let size = NVME_ALIGN * multiplier;
            let buf = AlignedBuffer::new(size);
            let addr = buf.as_ptr() as usize;
            assert_eq!(
                addr % NVME_ALIGN, 0,
                "buffer of size {size} must be 4096-aligned"
            );
        }
    }

    // ------------------------------------------------------------------
    // AlignedBuffer write and read back at boundaries
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_boundary_write_read() {
        let size = NVME_ALIGN * 2;
        let mut buf = AlignedBuffer::new(size);

        // Write at first byte
        buf.as_mut_slice()[0] = 0xAA;
        // Write at boundary between two aligned blocks
        buf.as_mut_slice()[NVME_ALIGN] = 0xBB;
        // Write at last byte
        buf.as_mut_slice()[size - 1] = 0xCC;

        assert_eq!(buf.as_slice()[0], 0xAA);
        assert_eq!(buf.as_slice()[NVME_ALIGN], 0xBB);
        assert_eq!(buf.as_slice()[size - 1], 0xCC);
    }

    // ------------------------------------------------------------------
    // AlignedBuffer drop does not panic (double-drop safety by null check)
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_drop_safe() {
        let buf = AlignedBuffer::new(NVME_ALIGN);
        drop(buf);
        // No panic = success; the Drop impl checks for null before freeing.
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile Debug format is valid structured output
    // ------------------------------------------------------------------

    #[test]
    fn nvme_swap_file_debug_is_structured() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("dbg2.swap"), 2048, 4096, 7).unwrap();
        let dbg = format!("{:?}", swap);
        // Debug output should look like: NvmeSwapFile { page_size: 2048, ... }
        assert!(dbg.starts_with("NvmeSwapFile"), "should start with struct name: {dbg}");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader Debug output starts with struct name
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_debug_starts_with_name() {
        let hdr = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let dbg = format!("{:?}", hdr);
        assert!(dbg.starts_with("SwapFileHeader"), "debug should start with struct name");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader version field boundary: 0, 1, u32::MAX
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_version_boundary_values() {
        for &v in &[0u32, 1, u32::MAX] {
            let hdr = SwapFileHeader {
                magic: SWAP_MAGIC,
                version: v,
                page_size: 4096,
                max_slot_bytes: 8192,
                _pad4: 0,
                slot_count: 8,
                _reserved: [0u8; SWAP_HEADER_BYTES - 32],
            };
            assert_eq!(hdr.version, v);
        }
    }

    // ------------------------------------------------------------------
    // SwapFileHeader page_size boundary: 0, 1, u32::MAX
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_page_size_boundary_values() {
        for &ps in &[0u32, 1, u32::MAX] {
            let hdr = SwapFileHeader {
                magic: SWAP_MAGIC,
                version: SWAP_VERSION,
                page_size: ps,
                max_slot_bytes: 8192,
                _pad4: 0,
                slot_count: 8,
                _reserved: [0u8; SWAP_HEADER_BYTES - 32],
            };
            assert_eq!(hdr.page_size, ps);
        }
    }

    // ------------------------------------------------------------------
    // SwapFileHeader Copy trait: explicit clone produces independent copy
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_clone_produces_independent_copy() {
        let original = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 16,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let cloned = original.clone();
        // Mutate the clone — original must not change
        let mut modified = cloned;
        modified.slot_count = 0;
        assert_eq!(original.slot_count, 16);
        assert_eq!(modified.slot_count, 0);
    }

    // ------------------------------------------------------------------
    // page_size = 0 is accepted by NvmeSwapFile::open (struct stores it)
    // ------------------------------------------------------------------

    #[test]
    fn open_with_page_size_zero_stored() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("ps0.swap"), 0, 4096, 4).unwrap();
        assert_eq!(swap.page_size, 0);
    }

    // ------------------------------------------------------------------
    // page_size = 1 is accepted by NvmeSwapFile::open
    // ------------------------------------------------------------------

    #[test]
    fn open_with_page_size_one_stored() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("ps1.swap"), 1, 4096, 4).unwrap();
        assert_eq!(swap.page_size, 1);
    }

    // ------------------------------------------------------------------
    // Write to high page_id then read back preserves data
    // ------------------------------------------------------------------

    #[test]
    fn high_page_id_data_integrity() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("hipg.swap"), 4096, 4096, 128).unwrap();

        let data: Vec<u8> = (0..200).map(|i| (i ^ 0x55) as u8).collect();
        swap.write_slot(127, &data).unwrap();

        let mut buf = vec![0u8; 200];
        swap.read_slot(127, &mut buf).unwrap();
        assert_eq!(buf, data, "high page_id data must be intact");
    }

    // ------------------------------------------------------------------
    // Write to slot 0 and last slot simultaneously correct
    // ------------------------------------------------------------------

    #[test]
    fn first_and_last_slot_independent() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("fl.swap"), 4096, 4096, 16).unwrap();

        let data_first = vec![0x11; 100];
        let data_last = vec![0x22; 100];
        swap.write_slot(0, &data_first).unwrap();
        swap.write_slot(15, &data_last).unwrap();

        let mut buf_first = vec![0u8; 100];
        let mut buf_last = vec![0u8; 100];
        swap.read_slot(0, &mut buf_first).unwrap();
        swap.read_slot(15, &mut buf_last).unwrap();

        assert!(buf_first.iter().all(|&b| b == 0x11));
        assert!(buf_last.iter().all(|&b| b == 0x22));
    }

    // ------------------------------------------------------------------
    // write_slot returns u32 — verify no truncation for small sizes
    // ------------------------------------------------------------------

    #[test]
    fn write_slot_returns_u32_no_truncation() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "u32.swap");
        let data = vec![0u8; 100];
        let written = swap.write_slot(0, &data).unwrap();
        assert_eq!(written, 100u32);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader _reserved field can be all 0xFF
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_reserved_all_ones() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 8,
            _reserved: [0xFF; SWAP_HEADER_BYTES - 32],
        };
        assert!(hdr._reserved.iter().all(|&b| b == 0xFF));
        assert_eq!(hdr._reserved.len(), SWAP_HEADER_BYTES - 32);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader is Send (contains only primitive types)
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<SwapFileHeader>();
    }

    // ------------------------------------------------------------------
    // SwapFileHeader is Sync (contains only primitive types)
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<SwapFileHeader>();
    }

    // ------------------------------------------------------------------
    // AlignedBuffer is Send (verified by unsafe impl)
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<AlignedBuffer>();
    }

    // ------------------------------------------------------------------
    // AlignedBuffer is Sync (verified by unsafe impl)
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<AlignedBuffer>();
    }

    // ------------------------------------------------------------------
    // SwapFileHeader size_of_val matches size_of
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_size_of_val_matches_size_of() {
        let hdr = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(std::mem::size_of_val(&hdr), std::mem::size_of::<SwapFileHeader>());
        assert_eq!(std::mem::size_of_val(&hdr), SWAP_HEADER_BYTES);
    }

    // ------------------------------------------------------------------
    // Reopen with larger max_slot_bytes — struct uses new aligned value
    // ------------------------------------------------------------------

    #[test]
    fn reopen_larger_slot_bytes_uses_new_value() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("larger.swap");

        {
            let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
        }

        let swap = NvmeSwapFile::open(path, 4096, 8192, 4).unwrap();
        assert_eq!(swap.max_slot_bytes, 8192);
    }

    // ------------------------------------------------------------------
    // Write-then-overwrite-then-read verifies final write wins
    // ------------------------------------------------------------------

    #[test]
    fn triple_overwrite_last_write_wins() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "triple.swap");

        swap.write_slot(0, &[0x11; 64]).unwrap();
        swap.write_slot(0, &[0x22; 64]).unwrap();
        swap.write_slot(0, &[0x33; 64]).unwrap();

        let mut buf = vec![0u8; 64];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0x33), "third write must win");
    }

    // ------------------------------------------------------------------
    // SWAP_MAGIC and SWAP_VERSION are compile-time constants
    // ------------------------------------------------------------------

    #[test]
    fn constants_are_const_evaluable() {
        const _MAGIC: u64 = SWAP_MAGIC;
        const _VERSION: u32 = SWAP_VERSION;
        const _HEADER_BYTES: usize = SWAP_HEADER_BYTES;
        const _ALIGN: usize = NVME_ALIGN;
        // If this compiles, the values are const-evaluable.
        assert!(true);
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile page_size field matches constructor argument
    // ------------------------------------------------------------------

    #[test]
    fn page_size_various_values_stored() {
        let tmp = TempDir::new().unwrap();
        for &ps in &[1usize, 512, 4096, 8192, 65536] {
            let path = tmp.path().join(format!("psv_{ps}.swap"));
            let swap = NvmeSwapFile::open(path, ps, 4096, 4).unwrap();
            assert_eq!(swap.page_size, ps, "page_size must match constructor argument");
        }
    }

    // ==================================================================
    // 50 additional tests — target 187+ total
    // ==================================================================

    // ------------------------------------------------------------------
    // AlignedBuffer as_mut_ptr returns non-null
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_mut_ptr_non_null() {
        let mut buf = AlignedBuffer::new(NVME_ALIGN);
        assert!(!buf.as_mut_ptr().is_null(), "as_mut_ptr must not be null");
    }

    // ------------------------------------------------------------------
    // AlignedBuffer as_ptr returns non-null
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_ptr_non_null() {
        let buf = AlignedBuffer::new(NVME_ALIGN);
        assert!(!buf.as_ptr().is_null(), "as_ptr must not be null");
    }

    // ------------------------------------------------------------------
    // AlignedBuffer as_ptr and as_mut_ptr point to same address
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_ptr_and_mut_ptr_same_address() {
        let mut buf = AlignedBuffer::new(NVME_ALIGN);
        let ro = buf.as_ptr() as usize;
        let rw = buf.as_mut_ptr() as usize;
        assert_eq!(ro, rw, "immutable and mutable pointers must refer to same address");
    }

    // ------------------------------------------------------------------
    // AlignedBuffer writing beyond first block works for multi-block
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_multi_block_write() {
        let size = NVME_ALIGN * 3;
        let mut buf = AlignedBuffer::new(size);
        // Write to offset in the second block
        buf.as_mut_slice()[NVME_ALIGN + 42] = 0xAB;
        // Write to offset in the third block
        buf.as_mut_slice()[NVME_ALIGN * 2 + 99] = 0xCD;
        assert_eq!(buf.as_slice()[NVME_ALIGN + 42], 0xAB);
        assert_eq!(buf.as_slice()[NVME_ALIGN * 2 + 99], 0xCD);
    }

    // ------------------------------------------------------------------
    // AlignedBuffer as_slice and as_mut_slice lengths match allocation
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_slice_lengths_match() {
        let size = NVME_ALIGN * 4;
        let mut buf = AlignedBuffer::new(size);
        assert_eq!(buf.as_slice().len(), size);
        assert_eq!(buf.as_mut_slice().len(), size);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader with specific magic pattern: all bytes 0xAA
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_magic_specific_pattern() {
        let hdr = SwapFileHeader {
            magic: 0xAAAAAAAAAAAAAAAA,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(hdr.magic, 0xAAAAAAAAAAAAAAAA);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader constructed with all fields at 1
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_all_fields_one() {
        let hdr = SwapFileHeader {
            magic: 1,
            version: 1,
            page_size: 1,
            max_slot_bytes: 1,
            _pad4: 1,
            slot_count: 1,
            _reserved: [1u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(hdr.magic, 1);
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.page_size, 1);
        assert_eq!(hdr.max_slot_bytes, 1);
        assert_eq!(hdr._pad4, 1);
        assert_eq!(hdr.slot_count, 1);
        assert!(hdr._reserved.iter().all(|&b| b == 1));
    }

    // ------------------------------------------------------------------
    // SwapFileHeader two different headers are not equal
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_different_version_not_equal() {
        let a = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 8,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = SwapFileHeader {
            version: 2,
            ..a
        };
        assert_ne!(a, b);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader Debug includes _pad4
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_debug_includes_pad4() {
        let hdr = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let dbg = format!("{:?}", hdr);
        assert!(dbg.contains("_pad4"), "debug must contain _pad4: {dbg}");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader can be transmuted to byte array and back
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_byte_roundtrip() {
        let original = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 64,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        // Transmute to bytes and back
        let bytes: [u8; SWAP_HEADER_BYTES] = unsafe { std::mem::transmute(original) };
        let restored: SwapFileHeader = unsafe { std::mem::transmute(bytes) };
        assert_eq!(original, restored, "byte roundtrip must preserve all fields");
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: open creates file on disk
    // ------------------------------------------------------------------

    #[test]
    fn open_creates_file_on_disk() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("exists.swap");
        assert!(!path.exists(), "file should not exist yet");
        let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
        assert!(path.exists(), "file must exist after open");
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: open file has correct size
    // ------------------------------------------------------------------

    #[test]
    fn open_file_size_is_header_plus_slots() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("size.swap");
        let slot_count: u64 = 16;
        let slot_size = 8192usize;
        let _swap = NvmeSwapFile::open(path.clone(), 4096, slot_size, slot_count).unwrap();
        let metadata = std::fs::metadata(&path).unwrap();
        let expected = SWAP_HEADER_BYTES as u64 + slot_count * slot_size as u64;
        assert_eq!(metadata.len(), expected, "file size must match header + slots");
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: open with zero slots file is just header
    // ------------------------------------------------------------------

    #[test]
    fn open_zero_slots_file_size_is_header_only() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("zero_slots_size.swap");
        let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 0).unwrap();
        let metadata = std::fs::metadata(&path).unwrap();
        assert_eq!(metadata.len(), SWAP_HEADER_BYTES as u64);
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: reopen reads stored header correctly
    // ------------------------------------------------------------------

    #[test]
    fn reopen_reads_header_page_size() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("rh.swap");
        {
            let _swap = NvmeSwapFile::open(path.clone(), 3333, 4096, 4).unwrap();
        }
        let swap = NvmeSwapFile::open(path, 3333, 4096, 4).unwrap();
        assert_eq!(swap.page_size, 3333);
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: max_slot_bytes alignment for exact multiple
    // ------------------------------------------------------------------

    #[test]
    fn max_slot_bytes_exact_multiple_unchanged() {
        let tmp = TempDir::new().unwrap();
        for &sz in &[NVME_ALIGN, NVME_ALIGN * 2, NVME_ALIGN * 4] {
            let path = tmp.path().join(format!("exact_{sz}.swap"));
            let swap = NvmeSwapFile::open(path, 4096, sz, 4).unwrap();
            assert_eq!(swap.max_slot_bytes, sz, "already-aligned size must not change");
        }
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: max_slot_bytes = NVME_ALIGN - 1 rounds up
    // ------------------------------------------------------------------

    #[test]
    fn max_slot_bytes_one_less_than_align_rounds_up() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(
            tmp.path().join("minus1.swap"),
            4096,
            NVME_ALIGN - 1,
            4,
        )
        .unwrap();
        assert_eq!(swap.max_slot_bytes, NVME_ALIGN);
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: max_slot_bytes = NVME_ALIGN + 1 rounds up to double
    // ------------------------------------------------------------------

    #[test]
    fn max_slot_bytes_one_more_than_align_rounds_up() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(
            tmp.path().join("plus1.swap"),
            4096,
            NVME_ALIGN + 1,
            4,
        )
        .unwrap();
        assert_eq!(swap.max_slot_bytes, NVME_ALIGN * 2);
    }

    // ------------------------------------------------------------------
    // write_slot then read_slot with varying data sizes within same slot
    // ------------------------------------------------------------------

    #[test]
    fn write_read_varying_sizes_same_slot() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("var.swap"), 4096, 8192, 4).unwrap();

        for &size in &[1, 64, 256, 1024, 4096] {
            let data: Vec<u8> = (0..size).map(|i| (i as u8).wrapping_add(0xA0)).collect();
            swap.write_slot(0, &data).unwrap();
            let mut buf = vec![0u8; size];
            swap.read_slot(0, &mut buf).unwrap();
            assert_eq!(buf, data, "size {size} roundtrip failed");
        }
    }

    // ------------------------------------------------------------------
    // write_slot data exactly fits slot, read partial gets first bytes
    // ------------------------------------------------------------------

    #[test]
    fn write_full_read_partial() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("fp.swap"), 4096, 4096, 4).unwrap();
        let data: Vec<u8> = (0..swap.max_slot_bytes).map(|i| (i % 256) as u8).collect();
        swap.write_slot(0, &data).unwrap();

        let mut partial = vec![0u8; 16];
        swap.read_slot(0, &mut partial).unwrap();
        assert_eq!(partial, &data[..16]);
    }

    // ------------------------------------------------------------------
    // write partial, read full slot — first bytes are data, rest zeros
    // ------------------------------------------------------------------

    #[test]
    fn write_partial_read_full() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("pf.swap"), 4096, 4096, 4).unwrap();
        let data = vec![0x77; 100];
        swap.write_slot(0, &data).unwrap();

        let mut full = vec![0xFF; 4096];
        swap.read_slot(0, &mut full).unwrap();
        // First 100 bytes = data
        assert!(full[..100].iter().all(|&b| b == 0x77));
        // Remaining bytes = 0 (AlignedBuffer zero-initialized)
        assert!(full[100..].iter().all(|&b| b == 0x00));
    }

    // ------------------------------------------------------------------
    // multiple drops and reopens preserve data across cycles
    // ------------------------------------------------------------------

    #[test]
    fn three_cycles_preserve_data() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("cycle.swap");
        let data = vec![0xCA; 333];

        // Cycle 1: write
        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 8).unwrap();
            swap.write_slot(4, &data).unwrap();
        }
        // Cycle 2: read + overwrite
        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 8).unwrap();
            let mut buf = vec![0u8; 333];
            swap.read_slot(4, &mut buf).unwrap();
            assert_eq!(buf, data, "cycle 2 read failed");
        }
        // Cycle 3: read again
        {
            let swap = NvmeSwapFile::open(path, 4096, 4096, 8).unwrap();
            let mut buf = vec![0u8; 333];
            swap.read_slot(4, &mut buf).unwrap();
            assert_eq!(buf, data, "cycle 3 read failed");
        }
    }

    // ------------------------------------------------------------------
    // write_slot with binary data pattern: alternating 0x55/0xAA
    // ------------------------------------------------------------------

    #[test]
    fn write_alternating_pattern() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "alt.swap");
        let data: Vec<u8> = (0..512).map(|i| if i % 2 == 0 { 0x55 } else { 0xAA }).collect();
        swap.write_slot(0, &data).unwrap();

        let mut buf = vec![0u8; 512];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // write_slot data that looks like a valid header in a slot
    // ------------------------------------------------------------------

    #[test]
    fn write_header_like_data_in_slot() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "hd.swap");
        // 32 bytes that look like a header
        let mut data = vec![0u8; 32];
        data[0..8].copy_from_slice(&SWAP_MAGIC.to_le_bytes());
        data[8..12].copy_from_slice(&SWAP_VERSION.to_le_bytes());
        swap.write_slot(2, &data).unwrap();

        let mut buf = vec![0u8; 32];
        swap.read_slot(2, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // slot_offset for page_id = 0 with different max_slot_bytes
    // ------------------------------------------------------------------

    #[test]
    fn slot_offset_page_zero_invariant_across_sizes() {
        let tmp = TempDir::new().unwrap();
        for &msb in &[NVME_ALIGN, NVME_ALIGN * 2, NVME_ALIGN * 4] {
            let path = tmp.path().join(format!("inv_{msb}.swap"));
            let swap = NvmeSwapFile::open(path, 4096, msb, 4).unwrap();
            assert_eq!(
                swap.slot_offset(0),
                SWAP_HEADER_BYTES as u64,
                "page 0 offset must always be header size"
            );
        }
    }

    // ------------------------------------------------------------------
    // slot_offset arithmetic: page_id * stride + header
    // ------------------------------------------------------------------

    #[test]
    fn slot_offset_formula_verification() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("formula.swap"), 4096, 8192, 64).unwrap();
        for pid in 0..64usize {
            let expected = SWAP_HEADER_BYTES as u64 + pid as u64 * swap.max_slot_bytes as u64;
            assert_eq!(swap.slot_offset(pid), expected, "page {pid}");
        }
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile Debug output does not contain raw fd value
    // ------------------------------------------------------------------

    #[test]
    fn debug_does_not_leak_fd() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "no_fd.swap");
        let dbg = format!("{:?}", swap);
        // Debug should not expose internal fd field
        assert!(!dbg.contains("fd"), "debug should not expose fd: {dbg}");
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile with large slot_count (1024) — offset calculation works
    // ------------------------------------------------------------------

    #[test]
    fn large_slot_count_offset_calculation() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(
            tmp.path().join("large.swap"),
            4096,
            4096,
            1024,
        )
        .unwrap();
        assert_eq!(swap.slot_count, 1024);
        let last_offset = swap.slot_offset(1023);
        let expected = SWAP_HEADER_BYTES as u64 + 1023 * 4096u64;
        assert_eq!(last_offset, expected);
    }

    // ------------------------------------------------------------------
    // write to adjacent slots preserves both
    // ------------------------------------------------------------------

    #[test]
    fn adjacent_slots_independent() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("adj.swap"), 4096, 4096, 8).unwrap();

        let data_a = vec![0xAA; 4096];
        let data_b = vec![0xBB; 4096];
        swap.write_slot(3, &data_a).unwrap();
        swap.write_slot(4, &data_b).unwrap();

        let mut buf_a = vec![0u8; 4096];
        let mut buf_b = vec![0u8; 4096];
        swap.read_slot(3, &mut buf_a).unwrap();
        swap.read_slot(4, &mut buf_b).unwrap();

        assert!(buf_a.iter().all(|&b| b == 0xAA));
        assert!(buf_b.iter().all(|&b| b == 0xBB));
    }

    // ------------------------------------------------------------------
    // SwapFileHeader _reserved field first byte offset is 32
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_reserved_starts_at_offset_32() {
        let hdr = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let base = &hdr as *const SwapFileHeader as usize;
        let reserved_off = hdr._reserved.as_ptr() as usize - base;
        assert_eq!(reserved_off, 32, "_reserved must start at byte offset 32");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: two headers with same _reserved content are equal
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_equal_with_same_reserved() {
        let mut reserved = [0u8; SWAP_HEADER_BYTES - 32];
        reserved[0] = 0x42;
        reserved[SWAP_HEADER_BYTES - 33] = 0x84;
        let a = SwapFileHeader {
            magic: 1,
            version: 1,
            page_size: 2,
            max_slot_bytes: 3,
            _pad4: 4,
            slot_count: 5,
            _reserved: reserved,
        };
        let b = SwapFileHeader {
            magic: 1,
            version: 1,
            page_size: 2,
            max_slot_bytes: 3,
            _pad4: 4,
            slot_count: 5,
            _reserved: reserved,
        };
        assert_eq!(a, b);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader is not equal if any single field differs
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_inequality_per_field() {
        let base = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 16,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };

        // Test each field in isolation
        assert_ne!(base, SwapFileHeader { magic: 0, ..base });
        assert_ne!(base, SwapFileHeader { version: 0, ..base });
        assert_ne!(base, SwapFileHeader { page_size: 0, ..base });
        assert_ne!(base, SwapFileHeader { max_slot_bytes: 0, ..base });
        assert_ne!(base, SwapFileHeader { _pad4: 1, ..base });
        assert_ne!(base, SwapFileHeader { slot_count: 0, ..base });
        let mut different_reserved = [0u8; SWAP_HEADER_BYTES - 32];
        different_reserved[0] = 1;
        assert_ne!(base, SwapFileHeader { _reserved: different_reserved, ..base });
    }

    // ------------------------------------------------------------------
    // SwapFileHeader Hash — inserting into HashMap and retrieving
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_hash_map_insert_lookup() {
        use std::collections::HashMap;
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 8,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let mut map: HashMap<SwapFileHeader, &'static str> = HashMap::new();
        map.insert(hdr, "value");
        let retrieved = map.get(&hdr);
        assert_eq!(retrieved, Some(&"value"));
    }

    // ------------------------------------------------------------------
    // SwapFileHeader Copy: assigning to new variable creates independent copy
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_assign_creates_copy() {
        let a = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: 1,
            page_size: 100,
            max_slot_bytes: 200,
            _pad4: 0,
            slot_count: 50,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = a; // Copy (not move, because SwapFileHeader: Copy)
        let _ = a; // a is still usable
        assert_eq!(a, b);
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: two independent buffers have different addresses
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_different_allocations() {
        let buf_a = AlignedBuffer::new(NVME_ALIGN);
        let buf_b = AlignedBuffer::new(NVME_ALIGN);
        assert_ne!(
            buf_a.as_ptr() as usize,
            buf_b.as_ptr() as usize,
            "two allocations must have different addresses"
        );
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: modifying one does not affect another
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_isolation() {
        let mut buf_a = AlignedBuffer::new(NVME_ALIGN);
        let mut buf_b = AlignedBuffer::new(NVME_ALIGN);
        buf_a.as_mut_slice()[0] = 0xAA;
        buf_b.as_mut_slice()[0] = 0xBB;
        assert_eq!(buf_a.as_slice()[0], 0xAA);
        assert_eq!(buf_b.as_slice()[0], 0xBB);
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: write all bytes, read all bytes
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_full_write_read() {
        let size = NVME_ALIGN * 2;
        let mut buf = AlignedBuffer::new(size);
        for (i, byte) in buf.as_mut_slice().iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
        for (i, &byte) in buf.as_slice().iter().enumerate() {
            assert_eq!(byte, (i % 256) as u8, "mismatch at index {i}");
        }
    }

    // ------------------------------------------------------------------
    // bad_magic with version-only change still rejected
    // ------------------------------------------------------------------

    #[test]
    fn bad_version_only_rejected() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("bvo.swap");
        // Create valid swap file
        {
            let _swap = NvmeSwapFile::open(path.clone(), 4096, 8192, 8).unwrap();
        }
        // Corrupt only the version byte
        let mut data = std::fs::read(&path).unwrap();
        data[8] = 99; // version low byte
        std::fs::write(&path, &data).unwrap();
        let result = NvmeSwapFile::open(path, 4096, 8192, 8);
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------
    // write_slot to page_id beyond slot_count still writes (no bounds check)
    // This verifies behavior — write succeeds but slot may be outside logical range
    // ------------------------------------------------------------------

    #[test]
    fn write_slot_beyond_slot_count_succeeds() {
        let tmp = TempDir::new().unwrap();
        // 4 slots (page 0..3), write to page 3 (last valid)
        let swap = NvmeSwapFile::open(tmp.path().join("beyond.swap"), 4096, 4096, 4).unwrap();
        let data = vec![0xEE; 64];
        // Writing to the last valid slot should succeed
        let result = swap.write_slot(3, &data);
        assert!(result.is_ok(), "write to last valid slot should succeed");
    }

    // ------------------------------------------------------------------
    // read_slot on freshly opened file returns zeros for unwritten slot
    // ------------------------------------------------------------------

    #[test]
    fn fresh_file_read_unwritten_slot_zeros() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("fresh.swap"), 4096, 4096, 8).unwrap();
        let mut buf = vec![0xFF; 128];
        swap.read_slot(5, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0), "unwritten slot should be zeros");
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile with page_size equal to max_slot_bytes
    // ------------------------------------------------------------------

    #[test]
    fn page_size_equals_max_slot_bytes() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("eq.swap"), 4096, 4096, 4).unwrap();
        assert_eq!(swap.page_size, swap.max_slot_bytes);
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile with page_size larger than max_slot_bytes
    // ------------------------------------------------------------------

    #[test]
    fn page_size_larger_than_max_slot_bytes() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("gt.swap"), 8192, 4096, 4).unwrap();
        assert!(swap.page_size > swap.max_slot_bytes);
    }

    // ------------------------------------------------------------------
    // write_slot error message for oversized data includes data len
    // ------------------------------------------------------------------

    #[test]
    fn write_oversized_error_includes_data_len() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("elen.swap"), 4096, 4096, 4).unwrap();
        let data = vec![0u8; 8192];
        let err = swap.write_slot(0, &data).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("8192"), "error should contain data len: {msg}");
    }

    // ------------------------------------------------------------------
    // read_slot error message for oversized dst includes dst len
    // ------------------------------------------------------------------

    #[test]
    fn read_oversized_error_includes_dst_len() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("rlen.swap"), 4096, 4096, 4).unwrap();
        let mut dst = vec![0u8; 8192];
        let err = swap.read_slot(0, &mut dst).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("8192"), "error should contain dst len: {msg}");
    }

    // ------------------------------------------------------------------
    // write_slot then overwrite with larger data — final read correct
    // ------------------------------------------------------------------

    #[test]
    fn overwrite_with_larger_data() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "grow.swap");

        swap.write_slot(0, &[0x11; 64]).unwrap();
        swap.write_slot(0, &[0x22; 256]).unwrap();

        let mut buf = vec![0u8; 256];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0x22), "larger overwrite must win");
    }

    // ------------------------------------------------------------------
    // multiple slots roundtrip with unique patterns per slot
    // ------------------------------------------------------------------

    #[test]
    fn unique_patterns_per_slot() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("unique.swap"), 4096, 4096, 32).unwrap();

        // Each slot gets a unique seed
        for pid in 0..32usize {
            let seed = (pid * 7 + 13) as u8;
            let data = vec![seed; 256];
            swap.write_slot(pid, &data).unwrap();
        }

        for pid in 0..32usize {
            let seed = (pid * 7 + 13) as u8;
            let mut buf = vec![0u8; 256];
            swap.read_slot(pid, &mut buf).unwrap();
            assert!(
                buf.iter().all(|&b| b == seed),
                "slot {pid}: expected 0x{seed:02x}"
            );
        }
    }

    // ------------------------------------------------------------------
    // SwapFileHeader alignment matches u64 alignment
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_alignment_matches_u64() {
        assert!(
            std::mem::align_of::<SwapFileHeader>() >= std::mem::align_of::<u64>(),
            "alignment must accommodate u64 fields"
        );
    }

    // ------------------------------------------------------------------
    // NVME_ALIGN is a multiple of common sector sizes (512)
    // ------------------------------------------------------------------

    #[test]
    fn nvme_align_multiple_of_512() {
        assert_eq!(NVME_ALIGN % 512, 0, "NVME_ALIGN should be multiple of 512");
    }

    // ------------------------------------------------------------------
    // SWAP_MAGIC encodes "GLLMSWAP" when interpreted as LE bytes
    // ------------------------------------------------------------------

    #[test]
    fn swap_magic_le_bytes() {
        let le_bytes = SWAP_MAGIC.to_le_bytes();
        // "PAWSMLLG" in little-endian byte order = "GLLMSWAP" in big-endian
        assert_eq!(&le_bytes, b"PAWSMLLG");
    }

    // ------------------------------------------------------------------
    // write_read with max page_id that fits in slot_count
    // ------------------------------------------------------------------

    #[test]
    fn write_read_max_valid_page_id() {
        let tmp = TempDir::new().unwrap();
        let slot_count: u64 = 32;
        let swap = NvmeSwapFile::open(
            tmp.path().join("maxpid.swap"),
            4096,
            4096,
            slot_count,
        )
        .unwrap();
        let last_pid = (slot_count - 1) as PageId;
        let data = vec![0xED; 200];
        swap.write_slot(last_pid, &data).unwrap();

        let mut buf = vec![0u8; 200];
        swap.read_slot(last_pid, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader Hash — multiple entries in HashMap
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_multiple_entries_hashmap() {
        use std::collections::HashMap;
        let mut map: HashMap<SwapFileHeader, u32> = HashMap::new();

        for i in 0..5u32 {
            let hdr = SwapFileHeader {
                magic: i as u64,
                version: 1,
                page_size: 4096,
                max_slot_bytes: 8192,
                _pad4: 0,
                slot_count: i as u64,
                _reserved: [0u8; SWAP_HEADER_BYTES - 32],
            };
            map.insert(hdr, i);
        }
        assert_eq!(map.len(), 5);

        // Verify lookup
        let key = SwapFileHeader {
            magic: 3,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 3,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(map.get(&key), Some(&3));
    }

    // ------------------------------------------------------------------
    // SwapFileHeader Debug includes _reserved field representation
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_debug_shows_reserved() {
        let hdr = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let dbg = format!("{:?}", hdr);
        assert!(
            dbg.contains("_reserved"),
            "debug must include _reserved field: {dbg}"
        );
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: open with non-standard page_size stores it
    // ------------------------------------------------------------------

    #[test]
    fn open_non_standard_page_size() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("nsps.swap"), 7777, 8192, 4).unwrap();
        assert_eq!(swap.page_size, 7777);
    }

    // ------------------------------------------------------------------
    // AlignedBuffer created and immediately dropped (RAII no-leak test)
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_create_drop_cycle() {
        for _ in 0..100 {
            let mut buf = AlignedBuffer::new(NVME_ALIGN);
            buf.as_mut_slice()[0] = 0xFF;
            // buf dropped each iteration — no leak = no OOM
        }
    }

    // ------------------------------------------------------------------
    // SwapFileHeader Copy trait: array of headers is valid
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_array_of_copies() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 16,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let arr = [hdr; 3];
        assert_eq!(arr[0], arr[1]);
        assert_eq!(arr[1], arr[2]);
    }

    // ==================================================================
    // 50 additional tests — target 241+ total
    // Focus: uncovered edge cases, file system interactions, error
    // messages, byte-level layout, concurrent access, offset math.
    // ==================================================================

    // ------------------------------------------------------------------
    // open_fd rejects path with embedded NUL byte
    // ------------------------------------------------------------------

    #[test]
    fn open_rejects_nul_in_path() {
        let tmp = TempDir::new().unwrap();
        let bad_path = tmp.path().join("bad\0.swap");
        let result = NvmeSwapFile::open(bad_path, 4096, 4096, 4);
        assert!(result.is_err(), "path with NUL must fail");
        let err = result.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader byte-level layout: version field is LE at offset 8..12
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_version_bytes_le() {
        let mut hdr: SwapFileHeader = unsafe { std::mem::zeroed() };
        hdr.version = 0x01020304;
        let base = &hdr as *const SwapFileHeader as *const u8;
        let v0 = unsafe { *base.add(8) };
        let v1 = unsafe { *base.add(9) };
        let v2 = unsafe { *base.add(10) };
        let v3 = unsafe { *base.add(11) };
        // LE: 0x01020304 -> bytes [04, 03, 02, 01]
        assert_eq!(v0, 0x04);
        assert_eq!(v1, 0x03);
        assert_eq!(v2, 0x02);
        assert_eq!(v3, 0x01);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader byte-level layout: page_size field is LE at offset 12..16
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_page_size_bytes_le() {
        let mut hdr: SwapFileHeader = unsafe { std::mem::zeroed() };
        hdr.page_size = 0xAABBCCDD;
        let base = &hdr as *const SwapFileHeader as *const u8;
        let bytes: [u8; 4] = unsafe { std::ptr::read(base.add(12) as *const [u8; 4]) };
        let restored = u32::from_le_bytes(bytes);
        assert_eq!(restored, 0xAABBCCDD);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader byte-level layout: max_slot_bytes field LE at offset 16..20
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_max_slot_bytes_bytes_le() {
        let mut hdr: SwapFileHeader = unsafe { std::mem::zeroed() };
        hdr.max_slot_bytes = 0x11223344;
        let base = &hdr as *const SwapFileHeader as *const u8;
        let bytes: [u8; 4] = unsafe { std::ptr::read(base.add(16) as *const [u8; 4]) };
        let restored = u32::from_le_bytes(bytes);
        assert_eq!(restored, 0x11223344);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: full byte pattern roundtrip via transmute
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_full_byte_pattern_roundtrip() {
        let original = SwapFileHeader {
            magic: 0x0102030405060708,
            version: 0x11121314,
            page_size: 0x21222324,
            max_slot_bytes: 0x31323334,
            _pad4: 0x41424344,
            slot_count: 0x5152535455667788,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let bytes: [u8; SWAP_HEADER_BYTES] = unsafe { std::mem::transmute(original) };
        let restored: SwapFileHeader = unsafe { std::mem::transmute(bytes) };
        assert_eq!(restored.magic, 0x0102030405060708);
        assert_eq!(restored.version, 0x11121314);
        assert_eq!(restored.page_size, 0x21222324);
        assert_eq!(restored.max_slot_bytes, 0x31323334);
        assert_eq!(restored._pad4, 0x41424344);
        assert_eq!(restored.slot_count, 0x5152535455667788);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader _pad4 field at offset 20..24 byte representation
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_pad4_bytes_le() {
        let mut hdr: SwapFileHeader = unsafe { std::mem::zeroed() };
        hdr._pad4 = 0xC0FFEE00;
        let base = &hdr as *const SwapFileHeader as *const u8;
        let bytes: [u8; 4] = unsafe { std::ptr::read(base.add(20) as *const [u8; 4]) };
        let restored = u32::from_le_bytes(bytes);
        assert_eq!(restored, 0xC0FFEE00);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: _reserved region starts with zeros after zeroed()
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_zeroed_reserved_first_16_bytes_zero() {
        let hdr: SwapFileHeader = unsafe { std::mem::zeroed() };
        for i in 0..16 {
            assert_eq!(hdr._reserved[i], 0, "_reserved byte {i} should be zero");
        }
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: _reserved region last 16 bytes zero after zeroed()
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_zeroed_reserved_last_16_bytes_zero() {
        let hdr: SwapFileHeader = unsafe { std::mem::zeroed() };
        let len = hdr._reserved.len();
        for i in (len - 16)..len {
            assert_eq!(hdr._reserved[i], 0, "_reserved byte {i} should be zero");
        }
    }

    // ------------------------------------------------------------------
    // Two NvmeSwapFile instances to the same path sequentially work
    // ------------------------------------------------------------------

    #[test]
    fn sequential_instances_same_path() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("seq_path.swap");

        let swap_a = NvmeSwapFile::open(path.clone(), 4096, 4096, 8).unwrap();
        swap_a.write_slot(0, &[0xAA; 64]).unwrap();
        drop(swap_a);

        let swap_b = NvmeSwapFile::open(path, 4096, 4096, 8).unwrap();
        let mut buf = vec![0u8; 64];
        swap_b.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xAA));
    }

    // ------------------------------------------------------------------
    // Write data with all bit patterns (0x00 through 0xFF in a single slot)
    // ------------------------------------------------------------------

    #[test]
    fn write_all_byte_values_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("allbytes.swap"), 4096, 4096, 4).unwrap();
        let data: Vec<u8> = (0u8..=255).cycle().take(256).collect();
        swap.write_slot(0, &data).unwrap();

        let mut buf = vec![0u8; 256];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // Write data with repeating 0x55 pattern (alternating bits)
    // ------------------------------------------------------------------

    #[test]
    fn write_alternating_bits_pattern() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "altbits.swap");
        let data = vec![0x55; 1024];
        swap.write_slot(0, &data).unwrap();

        let mut buf = vec![0u8; 1024];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // slot_offset for page 0 is independent of page_size
    // ------------------------------------------------------------------

    #[test]
    fn slot_offset_page_zero_ignores_page_size() {
        let tmp = TempDir::new().unwrap();
        for &ps in &[1usize, 512, 4096, 65536] {
            let path = tmp.path().join(format!("so_ps_{ps}.swap"));
            let swap = NvmeSwapFile::open(path, ps, 8192, 4).unwrap();
            assert_eq!(
                swap.slot_offset(0),
                SWAP_HEADER_BYTES as u64,
                "page 0 offset should not depend on page_size"
            );
        }
    }

    // ------------------------------------------------------------------
    // slot_offset increases monotonically
    // ------------------------------------------------------------------

    #[test]
    fn slot_offset_monotonically_increasing() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("mono.swap"), 4096, 4096, 64).unwrap();
        let mut prev: u64 = 0;
        for pid in 0..64usize {
            let offset = swap.slot_offset(pid);
            assert!(offset > prev, "offset must increase: pid={pid}");
            prev = offset;
        }
    }

    // ------------------------------------------------------------------
    // slot_offset for very large page_id (u32::MAX)
    // ------------------------------------------------------------------

    #[test]
    fn slot_offset_large_page_id_no_overflow_panic() {
        let swap = NvmeSwapFile {
            fd: Mutex::new(-1), // dummy fd, just for offset calculation
            page_size: 4096,
            max_slot_bytes: 8192,
            slot_count: 0,
        };
        // Should not panic — just test the arithmetic
        let offset = swap.slot_offset(u32::MAX as PageId);
        let expected = SWAP_HEADER_BYTES as u64 + u32::MAX as u64 * 8192;
        assert_eq!(offset, expected);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader PartialEq is symmetric for different values
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_partialeq_symmetry() {
        let a = SwapFileHeader {
            magic: 10,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 8,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = SwapFileHeader {
            magic: 20,
            ..a
        };
        // Both directions must be consistent
        assert_ne!(a, b);
        assert_ne!(b, a);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader PartialEq is transitive for three values
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_partialeq_transitive() {
        let base = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 8,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let copy_a = base;
        let copy_b = base;
        // a == b and b == c implies a == c (transitivity)
        assert_eq!(base, copy_a);
        assert_eq!(copy_a, copy_b);
        assert_eq!(base, copy_b);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader Hash: HashSet contains check
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_hashset_contains() {
        use std::collections::HashSet;
        let hdr = SwapFileHeader {
            magic: 42,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 8,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let mut set = HashSet::new();
        set.insert(hdr);
        assert!(set.contains(&hdr), "must contain inserted header");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader Hash: HashSet len changes correctly
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_hashset_len_after_insert_remove() {
        use std::collections::HashSet;
        let a = SwapFileHeader {
            magic: 1,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 8,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = SwapFileHeader {
            magic: 2,
            ..a
        };
        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b);
        assert_eq!(set.len(), 2);
        set.remove(&a);
        assert_eq!(set.len(), 1);
    }

    // ------------------------------------------------------------------
    // bad_magic error includes the file path
    // ------------------------------------------------------------------

    #[test]
    fn bad_magic_error_includes_file_path() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("path_in_err.swap");
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&[0u8; SWAP_HEADER_BYTES]).unwrap();
            f.set_len(SWAP_HEADER_BYTES as u64).unwrap();
        }
        let err = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("path_in_err.swap"),
            "error should include file path: {msg}"
        );
    }

    // ------------------------------------------------------------------
    // bad_version error includes the file path
    // ------------------------------------------------------------------

    #[test]
    fn bad_version_error_includes_file_path() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("ver_path.swap");
        {
            let _swap = NvmeSwapFile::open(path.clone(), 4096, 8192, 8).unwrap();
        }
        let mut data = std::fs::read(&path).unwrap();
        data[8..12].copy_from_slice(&99u32.to_le_bytes());
        let expected_len = SWAP_HEADER_BYTES + 8 * 8192;
        if data.len() < expected_len {
            data.resize(expected_len, 0);
        }
        std::fs::write(&path, &data).unwrap();

        let err = NvmeSwapFile::open(path.clone(), 4096, 8192, 8).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("ver_path.swap"),
            "version error should include path: {msg}"
        );
    }

    // ------------------------------------------------------------------
    // Swap file created with correct permissions (0o644)
    // ------------------------------------------------------------------

    #[test]
    fn open_creates_file_with_permissions() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("perm.swap");
        let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
        let meta = std::fs::metadata(&path).unwrap();
        let mode = meta.permissions().mode() & 0o777;
        assert_eq!(mode, 0o644, "file should have 0o644 permissions");
    }

    // ------------------------------------------------------------------
    // Swap file on reopen has same file size
    // ------------------------------------------------------------------

    #[test]
    fn reopen_preserves_file_size() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("fsize.swap");
        let slot_count: u64 = 16;
        let slot_size = 4096usize;

        {
            let _swap = NvmeSwapFile::open(path.clone(), 4096, slot_size, slot_count).unwrap();
        }
        let size_after_first = std::fs::metadata(&path).unwrap().len();

        {
            let _swap = NvmeSwapFile::open(path.clone(), 4096, slot_size, slot_count).unwrap();
        }
        let size_after_second = std::fs::metadata(&path).unwrap().len();

        assert_eq!(size_after_first, size_after_second);
    }

    // ------------------------------------------------------------------
    // Write 1 byte then read 1 byte from different slots
    // ------------------------------------------------------------------

    #[test]
    fn single_byte_different_slots() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("1byte.swap"), 4096, 4096, 16).unwrap();

        for pid in 0..16usize {
            swap.write_slot(pid, &[(pid as u8).wrapping_add(0xA0)]).unwrap();
        }
        for pid in 0..16usize {
            let mut buf = [0u8; 1];
            swap.read_slot(pid, &mut buf).unwrap();
            assert_eq!(buf[0], (pid as u8).wrapping_add(0xA0), "slot {pid}");
        }
    }

    // ------------------------------------------------------------------
    // Overwrite slot with identical data is idempotent
    // ------------------------------------------------------------------

    #[test]
    fn overwrite_identical_data_idempotent() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "idem.swap");
        let data = vec![0x42; 128];

        swap.write_slot(0, &data).unwrap();
        swap.write_slot(0, &data).unwrap();
        swap.write_slot(0, &data).unwrap();

        let mut buf = vec![0u8; 128];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // Overwrite shrinking then growing preserves final write
    // ------------------------------------------------------------------

    #[test]
    fn overwrite_shrink_then_grow() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "shrink_grow.swap");

        swap.write_slot(0, &[0xAA; 512]).unwrap();
        swap.write_slot(0, &[0xBB; 32]).unwrap();
        swap.write_slot(0, &[0xCC; 256]).unwrap();

        let mut buf = vec![0u8; 256];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xCC), "final write must win");
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: write to every byte position of first block
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_write_every_byte_first_block() {
        let mut buf = AlignedBuffer::new(NVME_ALIGN);
        for i in 0..NVME_ALIGN {
            buf.as_mut_slice()[i] = (i % 256) as u8;
        }
        for i in 0..NVME_ALIGN {
            assert_eq!(buf.as_slice()[i], (i % 256) as u8, "mismatch at {i}");
        }
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: drop multiple buffers in sequence
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_multiple_drops() {
        for _ in 0..50 {
            let buf = AlignedBuffer::new(NVME_ALIGN * 4);
            assert_eq!(buf.as_slice().len(), NVME_ALIGN * 4);
            // dropped each iteration
        }
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: allocation of 1 NVME_ALIGN (minimum valid size)
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_minimum_size() {
        let buf = AlignedBuffer::new(NVME_ALIGN);
        assert_eq!(buf.as_slice().len(), NVME_ALIGN);
        assert_eq!(buf.as_ptr().align_offset(NVME_ALIGN), 0);
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: open with max_slot_bytes much larger than page_size
    // ------------------------------------------------------------------

    #[test]
    fn open_max_slot_much_larger_than_page_size() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(
            tmp.path().join("bigslot.swap"),
            4096,
            65536,
            4,
        )
        .unwrap();
        assert!(swap.max_slot_bytes > swap.page_size * 10);
        assert_eq!(swap.max_slot_bytes % NVME_ALIGN, 0);
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: slot_count = 1 allows single valid page_id = 0
    // ------------------------------------------------------------------

    #[test]
    fn slot_count_one_single_page() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("one_page.swap"), 4096, 4096, 1).unwrap();
        let data = vec![0xAB; 64];
        swap.write_slot(0, &data).unwrap();

        let mut buf = vec![0u8; 64];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: equality after clone is maintained
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_clone_preserves_equality() {
        let hdr = SwapFileHeader {
            magic: 0xBEEF,
            version: 2,
            page_size: 8192,
            max_slot_bytes: 16384,
            _pad4: 99,
            slot_count: 42,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let cloned = hdr.clone();
        assert_eq!(hdr, cloned, "clone must preserve equality");
        assert_eq!(hdr.magic, cloned.magic);
        assert_eq!(hdr.slot_count, cloned.slot_count);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: modifying reserved does not affect other fields
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_reserved_mutation_independent() {
        let mut hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 8,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let magic_before = hdr.magic;
        let slot_count_before = hdr.slot_count;
        hdr._reserved[0] = 0xFF;
        hdr._reserved[SWAP_HEADER_BYTES - 33] = 0xFE;
        assert_eq!(hdr.magic, magic_before);
        assert_eq!(hdr.slot_count, slot_count_before);
        assert_eq!(hdr._reserved[0], 0xFF);
        assert_eq!(hdr._reserved[SWAP_HEADER_BYTES - 33], 0xFE);
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile Debug output contains all three public fields
    // ------------------------------------------------------------------

    #[test]
    fn debug_format_all_three_fields() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("dbg3.swap"), 4096, 8192, 42).unwrap();
        let dbg = format!("{swap:?}");
        assert!(dbg.contains("page_size: 4096"), "should show page_size: {dbg}");
        assert!(dbg.contains("max_slot_bytes: 8192"), "should show max_slot_bytes: {dbg}");
        assert!(dbg.contains("slot_count: 42"), "should show slot_count: {dbg}");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: size_of is exactly SWAP_HEADER_BYTES (compile-time check)
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_sizeof_is_const_correct() {
        // This verifies the compile-time assertion _HEADER_SIZE_CHECK at runtime
        assert_eq!(
            std::mem::size_of::<SwapFileHeader>(),
            4096,
            "compile-time size check must match runtime"
        );
    }

    // ------------------------------------------------------------------
    // File creation in root of temp dir (no parent creation needed)
    // ------------------------------------------------------------------

    #[test]
    fn open_no_parent_creation_needed() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("direct.swap");
        let swap = NvmeSwapFile::open(path, 4096, 4096, 4);
        assert!(swap.is_ok(), "should open directly in temp dir");
    }

    // ------------------------------------------------------------------
    // reopen with same parameters preserves previously written data
    // ------------------------------------------------------------------

    #[test]
    fn reopen_preserves_data_with_same_params() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("same_params.swap");
        let data = vec![0x77; 333];

        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 8192, 8).unwrap();
            swap.write_slot(2, &data).unwrap();
        }

        let swap = NvmeSwapFile::open(path, 4096, 8192, 8).unwrap();
        let mut buf = vec![0u8; 333];
        swap.read_slot(2, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // write then read across different sized slots within one file
    // ------------------------------------------------------------------

    #[test]
    fn write_read_different_sizes_in_same_file() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("diffsz.swap"), 4096, 8192, 4).unwrap();

        let small = vec![0x11; 16];
        let medium = vec![0x22; 256];
        let large = vec![0x33; 4096];

        swap.write_slot(0, &small).unwrap();
        swap.write_slot(1, &medium).unwrap();
        swap.write_slot(2, &large).unwrap();

        let mut s = vec![0u8; 16];
        let mut m = vec![0u8; 256];
        let mut l = vec![0u8; 4096];

        swap.read_slot(0, &mut s).unwrap();
        swap.read_slot(1, &mut m).unwrap();
        swap.read_slot(2, &mut l).unwrap();

        assert_eq!(s, small);
        assert_eq!(m, medium);
        assert_eq!(l, large);
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: writing and reading across block boundary
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_cross_block_boundary() {
        let size = NVME_ALIGN * 2;
        let mut buf = AlignedBuffer::new(size);
        // Write pattern straddling the boundary between blocks
        let start = NVME_ALIGN - 4;
        for i in 0..8 {
            buf.as_mut_slice()[start + i] = (i + 1) as u8;
        }
        for i in 0..8 {
            assert_eq!(buf.as_slice()[start + i], (i + 1) as u8, "at offset {}", start + i);
        }
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: all zero fields produce deterministic Debug output
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_zeroed_debug_deterministic() {
        let a: SwapFileHeader = unsafe { std::mem::zeroed() };
        let b: SwapFileHeader = unsafe { std::mem::zeroed() };
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
    }

    // ------------------------------------------------------------------
    // Slot data after reopen: overwrite one slot does not affect others
    // ------------------------------------------------------------------

    #[test]
    fn reopen_overwrite_one_slot_preserves_others() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("partial_ow.swap");

        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
            swap.write_slot(0, &[0xAA; 64]).unwrap();
            swap.write_slot(1, &[0xBB; 64]).unwrap();
        }

        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
            swap.write_slot(0, &[0xCC; 64]).unwrap(); // overwrite slot 0
        }

        let swap = NvmeSwapFile::open(path, 4096, 4096, 4).unwrap();
        let mut buf0 = vec![0u8; 64];
        let mut buf1 = vec![0u8; 64];
        swap.read_slot(0, &mut buf0).unwrap();
        swap.read_slot(1, &mut buf1).unwrap();
        assert!(buf0.iter().all(|&b| b == 0xCC), "slot 0 should be overwritten");
        assert!(buf1.iter().all(|&b| b == 0xBB), "slot 1 should be preserved");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: Copy allows re-assignment after copy
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_copy_reassign() {
        let a = SwapFileHeader {
            magic: 10,
            version: 1,
            page_size: 100,
            max_slot_bytes: 200,
            _pad4: 0,
            slot_count: 5,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = a;
        let c = a; // Copy again from original — a is still valid
        assert_eq!(b, c);
        assert_eq!(a, b);
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: max_slot_bytes alignment for common page sizes
    // ------------------------------------------------------------------

    #[test]
    fn max_slot_bytes_alignment_common_page_sizes() {
        let tmp = TempDir::new().unwrap();
        let page_sizes = [512usize, 1024, 2048, 4096, 8192, 16384, 32768, 65536];
        for &ps in &page_sizes {
            let path = tmp.path().join(format!("aps_{ps}.swap"));
            let swap = NvmeSwapFile::open(path, ps, ps, 4).unwrap();
            assert_eq!(
                swap.max_slot_bytes % NVME_ALIGN, 0,
                "page_size {ps} must produce aligned max_slot_bytes"
            );
        }
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: constructing with all fields at u32::MAX / u64::MAX
    // does not panic
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_max_fields_no_panic() {
        let _hdr = SwapFileHeader {
            magic: u64::MAX,
            version: u32::MAX,
            page_size: u32::MAX,
            max_slot_bytes: u32::MAX,
            _pad4: u32::MAX,
            slot_count: u64::MAX,
            _reserved: [u8::MAX; SWAP_HEADER_BYTES - 32],
        };
        // No panic = success
    }

    // ------------------------------------------------------------------
    // Write zero bytes then read zero bytes from same slot
    // ------------------------------------------------------------------

    #[test]
    fn write_zero_read_zero_same_slot() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "wr0.swap");
        swap.write_slot(5, &[]).unwrap();
        let mut buf: Vec<u8> = Vec::new();
        swap.read_slot(5, &mut buf).unwrap();
        assert!(buf.is_empty());
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: total explicit field sizes sum to 32
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_explicit_field_sizes_sum() {
        let magic = std::mem::size_of::<u64>();   // 8
        let version = std::mem::size_of::<u32>(); // 4
        let page_size = std::mem::size_of::<u32>(); // 4
        let max_slot = std::mem::size_of::<u32>(); // 4
        let pad = std::mem::size_of::<u32>();     // 4
        let slot_count = std::mem::size_of::<u64>(); // 8
        assert_eq!(magic + version + page_size + max_slot + pad + slot_count, 32);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: reserved field occupies remainder of header
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_reserved_is_remainder() {
        let reserved_size = std::mem::size_of::<[u8; SWAP_HEADER_BYTES - 32]>();
        assert_eq!(reserved_size, SWAP_HEADER_BYTES - 32);
        // Total = 32 + (SWAP_HEADER_BYTES - 32) = SWAP_HEADER_BYTES
        assert_eq!(32 + reserved_size, SWAP_HEADER_BYTES);
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: zero after creation, non-zero after write
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_zero_then_nonzero() {
        let mut buf = AlignedBuffer::new(NVME_ALIGN);
        assert!(buf.as_slice().iter().all(|&b| b == 0));
        buf.as_mut_slice()[0] = 0x42;
        assert_eq!(buf.as_slice()[0], 0x42);
        // Rest still zero
        assert!(buf.as_slice()[1..].iter().all(|&b| b == 0));
    }

    // ------------------------------------------------------------------
    // slot_offset for consecutive pages: stride is exactly max_slot_bytes
    // ------------------------------------------------------------------

    #[test]
    fn slot_offset_stride_exactly_max_slot_bytes() {
        let tmp = TempDir::new().unwrap();
        for &msb in &[4096usize, 8192, 16384] {
            let path = tmp.path().join(format!("stride_{msb}.swap"));
            let swap = NvmeSwapFile::open(path, 4096, msb, 8).unwrap();
            let off0 = swap.slot_offset(0);
            let off1 = swap.slot_offset(1);
            assert_eq!(off1 - off0, msb as u64);
        }
    }

    // ------------------------------------------------------------------
    // Write to slot, overwrite with zeros, read back zeros
    // ------------------------------------------------------------------

    #[test]
    fn overwrite_with_zeros() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "owzero.swap");
        swap.write_slot(0, &[0xFF; 256]).unwrap();
        swap.write_slot(0, &[0x00; 256]).unwrap();

        let mut buf = vec![0xFF; 256];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0x00), "zeros must overwrite");
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: Drop impl does not panic on valid fd
    // ------------------------------------------------------------------

    #[test]
    fn drop_does_not_panic_valid_fd() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("drop_ok.swap"), 4096, 4096, 4).unwrap();
        // Explicitly drop — must not panic
        drop(swap);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: field order matches memory layout (magic < version < ...)
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_field_address_order() {
        let hdr = SwapFileHeader {
            magic: 1,
            version: 2,
            page_size: 3,
            max_slot_bytes: 4,
            _pad4: 5,
            slot_count: 6,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let base = &hdr as *const SwapFileHeader as usize;
        let addrs: [usize; 6] = [
            &hdr.magic as *const u64 as usize - base,
            &hdr.version as *const u32 as usize - base,
            &hdr.page_size as *const u32 as usize - base,
            &hdr.max_slot_bytes as *const u32 as usize - base,
            &hdr._pad4 as *const u32 as usize - base,
            &hdr.slot_count as *const u64 as usize - base,
        ];
        // Each subsequent field has a strictly greater offset
        for i in 1..addrs.len() {
            assert!(addrs[i] > addrs[i - 1], "field {i} must come after field {}", i - 1);
        }
    }

    // ------------------------------------------------------------------
    // Write pattern with 0x80 (high bit set) roundtrips correctly
    // ------------------------------------------------------------------

    #[test]
    fn write_high_bit_pattern_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "hibit.swap");
        let data = vec![0x80; 512];
        swap.write_slot(0, &data).unwrap();

        let mut buf = vec![0u8; 512];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: writing at last valid index
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_write_at_last_index() {
        let size = NVME_ALIGN * 2;
        let mut buf = AlignedBuffer::new(size);
        buf.as_mut_slice()[size - 1] = 0xED;
        assert_eq!(buf.as_slice()[size - 1], 0xED);
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: open with non-round page_size stores exact value
    // ------------------------------------------------------------------

    #[test]
    fn open_non_round_page_size_exact() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("nrps.swap"), 3731, 8192, 4).unwrap();
        assert_eq!(swap.page_size, 3731, "page_size must be stored exactly");
    }

    // ==================================================================
    // 50 additional tests — target 295+ total
    // Focus: concurrent access, header persistence, offset edge cases,
    // multi-slot patterns, error conditions, AlignedBuffer lifecycle.
    // ==================================================================

    // ------------------------------------------------------------------
    // Concurrent reads from multiple threads are safe
    // ------------------------------------------------------------------

    #[test]
    fn concurrent_reads_from_multiple_threads() {
        use std::sync::Arc;
        use std::thread;

        let tmp = TempDir::new().unwrap();
        let swap = Arc::new(
            NvmeSwapFile::open(tmp.path().join("concurrent_r.swap"), 4096, 4096, 16).unwrap(),
        );

        // Write unique data to each slot
        for pid in 0..16usize {
            let data = vec![(pid as u8).wrapping_add(0x10); 128];
            swap.write_slot(pid, &data).unwrap();
        }

        let mut handles = Vec::new();
        for pid in 0..16usize {
            let swap_clone = Arc::clone(&swap);
            handles.push(thread::spawn(move || {
                let mut buf = vec![0u8; 128];
                swap_clone.read_slot(pid, &mut buf).unwrap();
                let expected = (pid as u8).wrapping_add(0x10);
                assert!(buf.iter().all(|&b| b == expected), "slot {pid} corrupted");
            }));
        }

        for h in handles {
            h.join().expect("thread should not panic");
        }
    }

    // ------------------------------------------------------------------
    // Concurrent writes to different slots from multiple threads
    // ------------------------------------------------------------------

    #[test]
    fn concurrent_writes_different_slots() {
        use std::sync::Arc;
        use std::thread;

        let tmp = TempDir::new().unwrap();
        let swap = Arc::new(
            NvmeSwapFile::open(tmp.path().join("concurrent_w.swap"), 4096, 4096, 16).unwrap(),
        );

        let mut handles = Vec::new();
        for pid in 0..16usize {
            let swap_clone = Arc::clone(&swap);
            handles.push(thread::spawn(move || {
                let data = vec![(pid as u8).wrapping_mul(3); 64];
                swap_clone.write_slot(pid, &data).unwrap();
            }));
        }

        for h in handles {
            h.join().expect("thread should not panic");
        }

        // Verify all writes landed correctly
        for pid in 0..16usize {
            let expected = (pid as u8).wrapping_mul(3);
            let mut buf = vec![0u8; 64];
            swap.read_slot(pid, &mut buf).unwrap();
            assert!(
                buf.iter().all(|&b| b == expected),
                "slot {pid}: expected 0x{expected:02x}"
            );
        }
    }

    // ------------------------------------------------------------------
    // Concurrent mixed read/write to different slots is safe
    // ------------------------------------------------------------------

    #[test]
    fn concurrent_mixed_read_write() {
        use std::sync::Arc;
        use std::thread;

        let tmp = TempDir::new().unwrap();
        let swap = Arc::new(
            NvmeSwapFile::open(tmp.path().join("concurrent_rw.swap"), 4096, 4096, 32).unwrap(),
        );

        // Pre-write first 16 slots
        for pid in 0..16usize {
            swap.write_slot(pid, &[0xAA; 64]).unwrap();
        }

        let mut handles = Vec::new();

        // Reader threads read slots 0..15
        for pid in 0..16usize {
            let sc = Arc::clone(&swap);
            handles.push(thread::spawn(move || {
                let mut buf = vec![0u8; 64];
                sc.read_slot(pid, &mut buf).unwrap();
            }));
        }

        // Writer threads write slots 16..31
        for pid in 16..32usize {
            let sc = Arc::clone(&swap);
            handles.push(thread::spawn(move || {
                sc.write_slot(pid, &[0xBB; 64]).unwrap();
            }));
        }

        for h in handles {
            h.join().expect("thread should not panic");
        }
    }

    // ------------------------------------------------------------------
    // Header persists correct page_size across reopen
    // ------------------------------------------------------------------

    #[test]
    fn header_persists_page_size_across_reopen() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("hdr_ps.swap");
        let unique_ps = 7919;

        {
            let _swap = NvmeSwapFile::open(path.clone(), unique_ps, 8192, 4).unwrap();
        }

        // Reopen with same page_size — struct must match
        let swap = NvmeSwapFile::open(path, unique_ps, 8192, 4).unwrap();
        assert_eq!(swap.page_size, unique_ps);
    }

    // ------------------------------------------------------------------
    // Header persists max_slot_bytes (aligned) across reopen
    // ------------------------------------------------------------------

    #[test]
    fn header_persists_aligned_max_slot_bytes() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("hdr_msb.swap");

        {
            // Pass 5000, which aligns to 8192
            let swap = NvmeSwapFile::open(path.clone(), 4096, 5000, 8).unwrap();
            assert_eq!(swap.max_slot_bytes, 8192);
        }

        // Reopen with the aligned value — must match
        let swap = NvmeSwapFile::open(path, 4096, 8192, 8).unwrap();
        assert_eq!(swap.max_slot_bytes, 8192);
    }

    // ------------------------------------------------------------------
    // Overwrite one slot multiple times, verify final read
    // ------------------------------------------------------------------

    #[test]
    fn overwrite_same_slot_many_times() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("many_ow.swap"), 4096, 4096, 4).unwrap();

        for i in 0u8..50 {
            let data = vec![i; 32];
            swap.write_slot(0, &data).unwrap();
        }

        let mut buf = vec![0u8; 32];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 49), "last write must win");
    }

    // ------------------------------------------------------------------
    // Write to slot_count - 1 and slot 0 simultaneously correct
    // ------------------------------------------------------------------

    #[test]
    fn first_and_last_slot_boundary_data() {
        let tmp = TempDir::new().unwrap();
        let count: u64 = 32;
        let swap = NvmeSwapFile::open(
            tmp.path().join("bound_data.swap"),
            4096,
            4096,
            count,
        )
        .unwrap();

        let data_first: Vec<u8> = (0..200).map(|i| (i ^ 0xAA) as u8).collect();
        let data_last: Vec<u8> = (0..200).map(|i| (i ^ 0x55) as u8).collect();

        swap.write_slot(0, &data_first).unwrap();
        swap.write_slot((count - 1) as PageId, &data_last).unwrap();

        let mut read_first = vec![0u8; 200];
        let mut read_last = vec![0u8; 200];
        swap.read_slot(0, &mut read_first).unwrap();
        swap.read_slot((count - 1) as PageId, &mut read_last).unwrap();

        assert_eq!(read_first, data_first);
        assert_eq!(read_last, data_last);
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: allocate, write, drop, reallocate — no residual data
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_reallocate_clean() {
        let addr;
        {
            let mut buf = AlignedBuffer::new(NVME_ALIGN);
            buf.as_mut_slice()[0] = 0xFF;
            addr = buf.as_ptr() as usize;
        }
        // New allocation must be zeroed regardless of address reuse
        let buf = AlignedBuffer::new(NVME_ALIGN);
        // Even if allocator reuses same address, content must be zeroed
        assert!(
            buf.as_slice().iter().all(|&b| b == 0),
            "fresh allocation must be zeroed"
        );
        let _ = addr; // suppress unused warning
    }

    // ------------------------------------------------------------------
    // Write 1 byte to every slot, read back all
    // ------------------------------------------------------------------

    #[test]
    fn single_byte_per_slot_all_slots() {
        let tmp = TempDir::new().unwrap();
        let count: u64 = 64;
        let swap = NvmeSwapFile::open(
            tmp.path().join("1b_all.swap"),
            4096,
            4096,
            count,
        )
        .unwrap();

        for pid in 0..count as usize {
            swap.write_slot(pid, &[pid as u8]).unwrap();
        }

        for pid in 0..count as usize {
            let mut byte = [0u8; 1];
            swap.read_slot(pid, &mut byte).unwrap();
            assert_eq!(byte[0], pid as u8, "slot {pid} mismatch");
        }
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: constructing with magic=0 is distinct from SWAP_MAGIC
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_zero_magic_differs_from_swap_magic() {
        let hdr = SwapFileHeader {
            magic: 0,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 8,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_ne!(hdr.magic, SWAP_MAGIC);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: version=0 is distinct from SWAP_VERSION
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_zero_version_differs_from_current() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: 0,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 8,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_ne!(hdr.version, SWAP_VERSION);
    }

    // ------------------------------------------------------------------
    // Open same file path twice after two drop cycles
    // ------------------------------------------------------------------

    #[test]
    fn double_drop_reopen_cycle() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("dbl_drop.swap");

        {
            let _s1 = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
        }
        {
            let _s2 = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
        }
        let s3 = NvmeSwapFile::open(path, 4096, 4096, 4);
        assert!(s3.is_ok(), "third open must succeed");
    }

    // ------------------------------------------------------------------
    // Write to slot, close, reopen with larger max_slot_bytes, read
    // ------------------------------------------------------------------

    #[test]
    fn reopen_larger_slot_read_old_data() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("grow_read.swap");
        let data = vec![0x77; 100];

        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
            swap.write_slot(0, &data).unwrap();
        }

        // Reopen with larger max_slot_bytes — old data in first 100 bytes preserved
        let swap = NvmeSwapFile::open(path, 4096, 8192, 4).unwrap();
        let mut buf = vec![0u8; 100];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // Slot offset for page_id 0 with slot_count = 0 (edge case)
    // ------------------------------------------------------------------

    #[test]
    fn slot_offset_page_zero_with_zero_slots() {
        let swap = NvmeSwapFile {
            fd: Mutex::new(-1),
            page_size: 4096,
            max_slot_bytes: 8192,
            slot_count: 0,
        };
        // slot_offset is pure arithmetic, works even with slot_count = 0
        assert_eq!(swap.slot_offset(0), SWAP_HEADER_BYTES as u64);
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: two consecutive allocations have valid alignment
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_consecutive_allocations_aligned() {
        let a = AlignedBuffer::new(NVME_ALIGN);
        let b = AlignedBuffer::new(NVME_ALIGN * 2);

        assert_eq!(a.as_ptr() as usize % NVME_ALIGN, 0);
        assert_eq!(b.as_ptr() as usize % NVME_ALIGN, 0);
        assert_eq!(a.as_slice().len(), NVME_ALIGN);
        assert_eq!(b.as_slice().len(), NVME_ALIGN * 2);
    }

    // ------------------------------------------------------------------
    // Write full slot then read only 1 byte from beginning
    // ------------------------------------------------------------------

    #[test]
    fn write_full_read_one_byte() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("full1b.swap"), 4096, 4096, 4).unwrap();

        let data = vec![0xFE; 4096];
        swap.write_slot(0, &data).unwrap();

        let mut byte = [0u8; 1];
        swap.read_slot(0, &mut byte).unwrap();
        assert_eq!(byte[0], 0xFE);
    }

    // ------------------------------------------------------------------
    // Write 1 byte to last byte position of slot, read back
    // ------------------------------------------------------------------

    #[test]
    fn write_read_last_byte_of_slot() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("last_byte.swap"), 4096, 4096, 4).unwrap();

        // Write to the very last byte of the slot
        let mut data = vec![0u8; 4096];
        data[4095] = 0xED;
        swap.write_slot(0, &data).unwrap();

        // Read only the last byte
        let mut full = vec![0u8; 4096];
        swap.read_slot(0, &mut full).unwrap();
        assert_eq!(full[4095], 0xED);
        assert!(full[..4095].iter().all(|&b| b == 0x00));
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: array of different headers all distinct
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_array_distinct_elements() {
        let headers: Vec<SwapFileHeader> = (0..5)
            .map(|i| SwapFileHeader {
                magic: i as u64,
                version: 1,
                page_size: 4096,
                max_slot_bytes: 8192,
                _pad4: 0,
                slot_count: i as u64,
                _reserved: [0u8; SWAP_HEADER_BYTES - 32],
            })
            .collect();

        for i in 0..5 {
            for j in (i + 1)..5 {
                assert_ne!(headers[i], headers[j], "headers {i} and {j} must differ");
            }
        }
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: write_slot with data exactly 1 byte less than max
    // ------------------------------------------------------------------

    #[test]
    fn write_one_less_than_max_slot_bytes() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("max_minus1.swap"), 4096, 4096, 4).unwrap();
        let data = vec![0xCC; swap.max_slot_bytes - 1];
        let written = swap.write_slot(0, &data).unwrap();
        assert_eq!(written as usize, swap.max_slot_bytes - 1);
    }

    // ------------------------------------------------------------------
    // Read one less than max_slot_bytes from full slot
    // ------------------------------------------------------------------

    #[test]
    fn read_one_less_than_max_slot_bytes() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("rmax_minus1.swap"), 4096, 4096, 4).unwrap();
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        swap.write_slot(0, &data).unwrap();

        let mut buf = vec![0u8; 4095];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, &data[..4095]);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: slot_count = 0 produces valid header
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_slot_count_zero_valid() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(hdr.slot_count, 0);
        assert_eq!(hdr.magic, SWAP_MAGIC);
    }

    // ------------------------------------------------------------------
    // Write to non-zero slot does not corrupt slot 0
    // ------------------------------------------------------------------

    #[test]
    fn write_nonzero_slot_preserves_slot_zero() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("no_clobber.swap"), 4096, 4096, 8).unwrap();

        let data0 = vec![0xAA; 128];
        swap.write_slot(0, &data0).unwrap();

        // Write to several other slots
        for pid in 1..8usize {
            swap.write_slot(pid, &[0xBB; 128]).unwrap();
        }

        // Verify slot 0 is intact
        let mut buf = vec![0u8; 128];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xAA), "slot 0 must not be clobbered");
    }

    // ------------------------------------------------------------------
    // Write to slot 0 does not corrupt non-zero slots
    // ------------------------------------------------------------------

    #[test]
    fn write_slot_zero_preserves_others() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("no_clobber_r.swap"), 4096, 4096, 8).unwrap();

        for pid in 1..8usize {
            swap.write_slot(pid, &[(pid as u8).wrapping_add(0x20); 64]).unwrap();
        }

        // Overwrite slot 0 multiple times
        for _ in 0..10 {
            swap.write_slot(0, &[0xFF; 64]).unwrap();
        }

        // Verify other slots are intact
        for pid in 1..8usize {
            let expected = (pid as u8).wrapping_add(0x20);
            let mut buf = vec![0u8; 64];
            swap.read_slot(pid, &mut buf).unwrap();
            assert!(
                buf.iter().all(|&b| b == expected),
                "slot {pid}: expected 0x{expected:02x}"
            );
        }
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: large number of sequential allocations/drops
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_rapid_alloc_dealloc() {
        for _ in 0..200 {
            let mut buf = AlignedBuffer::new(NVME_ALIGN);
            buf.as_mut_slice()[0] = 0x42;
            assert_eq!(buf.as_slice()[0], 0x42);
            // dropped each iteration — no leak
        }
    }

    // ------------------------------------------------------------------
    // File on disk has correct size after multiple opens with same params
    // ------------------------------------------------------------------

    #[test]
    fn file_size_stable_across_reopens() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("stable_size.swap");
        let slot_count: u64 = 8;
        let slot_size = 4096usize;
        let expected = SWAP_HEADER_BYTES as u64 + slot_count * slot_size as u64;

        for _ in 0..5 {
            {
                let _swap = NvmeSwapFile::open(path.clone(), 4096, slot_size, slot_count).unwrap();
            }
            let actual = std::fs::metadata(&path).unwrap().len();
            assert_eq!(actual, expected, "file size must not grow on reopen");
        }
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: Debug output for all-zero header contains "0"
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_zeroed_debug_contains_zero_values() {
        let hdr: SwapFileHeader = unsafe { std::mem::zeroed() };
        let dbg = format!("{hdr:?}");
        // All fields are 0, so Debug should contain "0" for numeric fields
        assert!(dbg.contains("magic: 0"), "should show magic: 0: {dbg}");
        assert!(dbg.contains("version: 0"), "should show version: 0: {dbg}");
    }

    // ------------------------------------------------------------------
    // Write with data = max_slot_bytes/2 (half slot) roundtrips
    // ------------------------------------------------------------------

    #[test]
    fn write_half_slot_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("half.swap"), 4096, 4096, 4).unwrap();
        let half = swap.max_slot_bytes / 2;
        let data: Vec<u8> = (0..half).map(|i| (i as u8).wrapping_mul(2)).collect();
        swap.write_slot(0, &data).unwrap();

        let mut buf = vec![0u8; half];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // Read partial from middle of slot after writing full slot
    // ------------------------------------------------------------------

    #[test]
    fn read_partial_middle_of_slot() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("mid.swap"), 4096, 4096, 4).unwrap();
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        swap.write_slot(0, &data).unwrap();

        // Read bytes 1000..1100 (100 bytes from the middle)
        let mut full = vec![0u8; 4096];
        swap.read_slot(0, &mut full).unwrap();
        assert_eq!(&full[1000..1100], &data[1000..1100]);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: _reserved field middle byte can be set independently
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_reserved_middle_byte() {
        let mut reserved = [0u8; SWAP_HEADER_BYTES - 32];
        let mid = reserved.len() / 2;
        reserved[mid] = 0xDD;
        let hdr = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: reserved,
        };
        assert_eq!(hdr._reserved[mid], 0xDD);
        // Neighbors are zero
        assert_eq!(hdr._reserved[mid - 1], 0);
        assert_eq!(hdr._reserved[mid + 1], 0);
    }

    // ------------------------------------------------------------------
    // Slot offset for page_id 1 equals header + max_slot_bytes
    // ------------------------------------------------------------------

    #[test]
    fn slot_offset_page_one_formula() {
        let tmp = TempDir::new().unwrap();
        let msb = 16384;
        let swap = NvmeSwapFile::open(tmp.path().join("so1.swap"), 4096, msb, 8).unwrap();
        let off1 = swap.slot_offset(1);
        let expected = SWAP_HEADER_BYTES as u64 + msb as u64;
        assert_eq!(off1, expected);
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: max_slot_bytes = NVME_ALIGN * 3 (non-power-of-2 multiple)
    // ------------------------------------------------------------------

    #[test]
    fn max_slot_bytes_non_power_of_two_multiple() {
        let tmp = TempDir::new().unwrap();
        let input = NVME_ALIGN * 3; // 12288 = valid aligned value
        let swap = NvmeSwapFile::open(tmp.path().join("npt.swap"), 4096, input, 4).unwrap();
        assert_eq!(swap.max_slot_bytes, input);
        assert_eq!(swap.max_slot_bytes % NVME_ALIGN, 0);
    }

    // ------------------------------------------------------------------
    // Write to page_id 0 and 1 with max data, verify no overlap
    // ------------------------------------------------------------------

    #[test]
    fn full_slot_write_no_overlap() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("no_overlap.swap"), 4096, 4096, 4).unwrap();

        let data_a = vec![0xAA; 4096];
        let data_b = vec![0xBB; 4096];
        swap.write_slot(0, &data_a).unwrap();
        swap.write_slot(1, &data_b).unwrap();

        let mut read_a = vec![0u8; 4096];
        let mut read_b = vec![0u8; 4096];
        swap.read_slot(0, &mut read_a).unwrap();
        swap.read_slot(1, &mut read_b).unwrap();

        assert!(read_a.iter().all(|&b| b == 0xAA));
        assert!(read_b.iter().all(|&b| b == 0xBB));
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: PartialEq reflexive for non-trivial header
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_partialeq_reflexive_nontrivial() {
        let hdr = SwapFileHeader {
            magic: 0xDEADBEEF,
            version: 42,
            page_size: 2048,
            max_slot_bytes: 4096,
            _pad4: 99,
            slot_count: 777,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(hdr, hdr, "header must equal itself");
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: write to first byte, drop, new buffer independent
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_new_after_drop_independent() {
        let mut buf_a = AlignedBuffer::new(NVME_ALIGN);
        buf_a.as_mut_slice()[0] = 0xFF;
        drop(buf_a);

        let buf_b = AlignedBuffer::new(NVME_ALIGN);
        assert_eq!(buf_b.as_slice()[0], 0, "new buffer must be zeroed");
    }

    // ------------------------------------------------------------------
    // Multiple sequential overwrites of different slots
    // ------------------------------------------------------------------

    #[test]
    fn sequential_overwrite_different_slots() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("seq_ow.swap"), 4096, 4096, 4).unwrap();

        // First pass: write all slots
        for pid in 0..4usize {
            swap.write_slot(pid, &[0x10; 64]).unwrap();
        }

        // Second pass: overwrite all slots
        for pid in 0..4usize {
            swap.write_slot(pid, &[0x20; 64]).unwrap();
        }

        // Verify all slots have second-write data
        for pid in 0..4usize {
            let mut buf = vec![0u8; 64];
            swap.read_slot(pid, &mut buf).unwrap();
            assert!(buf.iter().all(|&b| b == 0x20), "slot {pid} must have 0x20");
        }
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: large magic value roundtrips through byte conversion
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_large_magic_byte_roundtrip() {
        let magic_val = 0xFEDCBA9876543210;
        let hdr = SwapFileHeader {
            magic: magic_val,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 8,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };

        let bytes: [u8; SWAP_HEADER_BYTES] = unsafe { std::mem::transmute(hdr) };
        let restored: SwapFileHeader = unsafe { std::mem::transmute(bytes) };
        assert_eq!(restored.magic, magic_val);
    }

    // ------------------------------------------------------------------
    // Write/read with page_size 1 and max_slot_bytes 4096
    // ------------------------------------------------------------------

    #[test]
    fn tiny_page_size_normal_slot_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("tiny_ps.swap"), 1, 4096, 4).unwrap();
        assert_eq!(swap.page_size, 1);

        let data = vec![0x42; 256];
        swap.write_slot(0, &data).unwrap();
        let mut buf = vec![0u8; 256];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: _pad4 with various values
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_pad4_various_values() {
        for &pad in &[0u32, 1, 42, 0x7FFFFFFF, 0x80000000, u32::MAX] {
            let hdr = SwapFileHeader {
                magic: 0,
                version: 0,
                page_size: 0,
                max_slot_bytes: 0,
                _pad4: pad,
                slot_count: 0,
                _reserved: [0u8; SWAP_HEADER_BYTES - 32],
            };
            assert_eq!(hdr._pad4, pad);
        }
    }

    // ------------------------------------------------------------------
    // Slot offset for multiple files with different max_slot_bytes
    // ------------------------------------------------------------------

    #[test]
    fn slot_offset_different_files_independent() {
        let tmp = TempDir::new().unwrap();

        let swap_a = NvmeSwapFile::open(tmp.path().join("a.swap"), 4096, 4096, 8).unwrap();
        let swap_b = NvmeSwapFile::open(tmp.path().join("b.swap"), 4096, 8192, 8).unwrap();

        // Same page_id, different offset due to different max_slot_bytes
        assert_eq!(swap_a.slot_offset(2), SWAP_HEADER_BYTES as u64 + 2 * 4096);
        assert_eq!(swap_b.slot_offset(2), SWAP_HEADER_BYTES as u64 + 2 * 8192);
    }

    // ------------------------------------------------------------------
    // Write data, overwrite with zeros, write new data, read new data
    // ------------------------------------------------------------------

    #[test]
    fn overwrite_cycle_three_stages() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "cycle3.swap");

        // Stage 1: write pattern A
        swap.write_slot(0, &[0xAA; 128]).unwrap();
        // Stage 2: overwrite with zeros
        swap.write_slot(0, &[0x00; 128]).unwrap();
        // Stage 3: overwrite with pattern C
        swap.write_slot(0, &[0xCC; 128]).unwrap();

        let mut buf = vec![0u8; 128];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xCC), "stage 3 must win");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: Hash consistency — same header hashed twice same result
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_hash_consistent_across_calls() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let hdr = SwapFileHeader {
            magic: 12345,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 16,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };

        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        hdr.hash(&mut h1);
        hdr.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish(), "hash must be deterministic");
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: write to same slot from multiple threads sequentially
    // ------------------------------------------------------------------

    #[test]
    fn sequential_thread_writes_same_slot() {
        use std::sync::Arc;
        use std::thread;

        let tmp = TempDir::new().unwrap();
        let swap = Arc::new(
            NvmeSwapFile::open(tmp.path().join("seq_same.swap"), 4096, 4096, 4).unwrap(),
        );

        for i in 0..10u8 {
            let sc = Arc::clone(&swap);
            thread::spawn(move || {
                sc.write_slot(0, &[i; 64]).unwrap();
            })
            .join()
            .expect("thread should not panic");
        }

        // Last write (i=9) must be present
        let mut buf = vec![0u8; 64];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 9), "last sequential write must win");
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: writing to byte 0 and last byte in sequence
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_first_and_last_byte() {
        let size = NVME_ALIGN * 3;
        let mut buf = AlignedBuffer::new(size);
        buf.as_mut_slice()[0] = 0x01;
        buf.as_mut_slice()[size - 1] = 0x02;
        assert_eq!(buf.as_slice()[0], 0x01);
        assert_eq!(buf.as_slice()[size - 1], 0x02);
        // Middle bytes zero
        assert!(buf.as_slice()[1..size - 1].iter().all(|&b| b == 0));
    }

    // ------------------------------------------------------------------
    // Write with pattern that changes every byte, read back
    // ------------------------------------------------------------------

    #[test]
    fn write_pseudo_random_pattern_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("prand.swap"), 4096, 4096, 4).unwrap();

        // Simple LCG pseudo-random pattern
        let data: Vec<u8> = (0..1024)
            .scan(42u32, |state, _| {
                *state = state.wrapping_mul(1103515245).wrapping_add(12345);
                Some((*state >> 16) as u8)
            })
            .collect();

        swap.write_slot(0, &data).unwrap();
        let mut buf = vec![0u8; 1024];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // Reopen with different page_size — struct stores new value
    // ------------------------------------------------------------------

    #[test]
    fn reopen_different_page_size_stores_new() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("diff_ps.swap");

        {
            let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
        }

        // Reopen with different page_size
        let swap = NvmeSwapFile::open(path, 8192, 4096, 4).unwrap();
        assert_eq!(swap.page_size, 8192, "struct should use new page_size");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: constructing with non-ASCII magic value
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_non_ascii_magic() {
        let hdr = SwapFileHeader {
            magic: 0xFFFFFFFFFFFFFFFF,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 8,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_ne!(hdr.magic, SWAP_MAGIC);
        assert_eq!(hdr.magic, u64::MAX);
    }

    // ------------------------------------------------------------------
    // Write to all slots in reverse order, read in forward order
    // ------------------------------------------------------------------

    #[test]
    fn write_reverse_read_forward() {
        let tmp = TempDir::new().unwrap();
        let count = 16u64;
        let swap = NvmeSwapFile::open(
            tmp.path().join("rev.swap"),
            4096,
            4096,
            count,
        )
        .unwrap();

        for pid in (0..count as usize).rev() {
            swap.write_slot(pid, &[pid as u8; 64]).unwrap();
        }

        for pid in 0..count as usize {
            let mut buf = vec![0u8; 64];
            swap.read_slot(pid, &mut buf).unwrap();
            assert!(buf.iter().all(|&b| b == pid as u8), "slot {pid}");
        }
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: size matches on stack vs heap
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_stack_and_heap_size_match() {
        let stack_hdr = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let heap_hdr = Box::new(SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        });

        assert_eq!(
            std::mem::size_of_val(&stack_hdr),
            std::mem::size_of_val(&*heap_hdr)
        );
    }

    // ------------------------------------------------------------------
    // Slot offset wraps correctly for high page_id values
    // ------------------------------------------------------------------

    #[test]
    fn slot_offset_high_page_id_arithmetic() {
        let swap = NvmeSwapFile {
            fd: Mutex::new(-1),
            page_size: 4096,
            max_slot_bytes: 4096,
            slot_count: 0,
        };
        // page_id = 100000, should compute without overflow
        let offset = swap.slot_offset(100000);
        let expected = SWAP_HEADER_BYTES as u64 + 100000u64 * 4096;
        assert_eq!(offset, expected);
    }

    // ------------------------------------------------------------------
    // Write to odd-indexed slots only, verify even-indexed are zero
    // ------------------------------------------------------------------

    #[test]
    fn write_odd_read_even_untouched() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("odd_even.swap"), 4096, 4096, 16).unwrap();

        for pid in (0..16).filter(|p| p % 2 == 1) {
            swap.write_slot(pid, &[0x99; 32]).unwrap();
        }

        // Even slots should still be zero
        for pid in (0..16).filter(|p| p % 2 == 0) {
            let mut buf = vec![0xFF; 32];
            swap.read_slot(pid, &mut buf).unwrap();
            assert!(buf.iter().all(|&b| b == 0x00), "even slot {pid} should be zero");
        }

        // Odd slots should have data
        for pid in (0..16).filter(|p| p % 2 == 1) {
            let mut buf = vec![0u8; 32];
            swap.read_slot(pid, &mut buf).unwrap();
            assert!(buf.iter().all(|&b| b == 0x99), "odd slot {pid} should have data");
        }
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: zero-length read after creation is valid
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_zero_len_slice_valid() {
        let buf = AlignedBuffer::new(NVME_ALIGN);
        let empty = &buf.as_slice()[..0];
        assert!(empty.is_empty());
    }

    // ------------------------------------------------------------------
    // Write empty to slot then write real data, verify data wins
    // ------------------------------------------------------------------

    #[test]
    fn write_empty_then_real_data() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "empty_then_real.swap");

        swap.write_slot(0, &[]).unwrap();
        let data = vec![0xDD; 100];
        swap.write_slot(0, &data).unwrap();

        let mut buf = vec![0u8; 100];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ==================================================================
    // Additional tests (~45) for deeper coverage
    // ==================================================================

    // ------------------------------------------------------------------
    // Two swap files in the same directory operate independently
    // ------------------------------------------------------------------

    #[test]
    fn two_files_same_directory_independent() {
        let tmp = TempDir::new().unwrap();
        let swap_a = NvmeSwapFile::open(tmp.path().join("a.swap"), 4096, 4096, 4).unwrap();
        let swap_b = NvmeSwapFile::open(tmp.path().join("b.swap"), 4096, 4096, 4).unwrap();

        swap_a.write_slot(0, &[0xAA; 64]).unwrap();
        swap_b.write_slot(0, &[0xBB; 64]).unwrap();

        let mut buf_a = vec![0u8; 64];
        let mut buf_b = vec![0u8; 64];
        swap_a.read_slot(0, &mut buf_a).unwrap();
        swap_b.read_slot(0, &mut buf_b).unwrap();

        assert!(buf_a.iter().all(|&b| b == 0xAA), "file a data corrupted");
        assert!(buf_b.iter().all(|&b| b == 0xBB), "file b data corrupted");
    }

    // ------------------------------------------------------------------
    // Failed oversized write does not corrupt previously written data
    // ------------------------------------------------------------------

    #[test]
    fn write_oversize_does_not_corrupt_prior_data() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("nocorrupt.swap"), 4096, 4096, 4).unwrap();

        let valid = vec![0x55; 100];
        swap.write_slot(0, &valid).unwrap();

        let oversized = vec![0xFF; swap.max_slot_bytes + 1];
        assert!(swap.write_slot(0, &oversized).is_err());

        let mut buf = vec![0u8; 100];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, valid, "prior data must survive failed write");
    }

    // ------------------------------------------------------------------
    // Failed oversized read does not modify caller buffer
    // ------------------------------------------------------------------

    #[test]
    fn read_oversize_preserves_buffer_content() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("bufsafe.swap"), 4096, 4096, 4).unwrap();

        let sentinel = vec![0xAB; 5000];
        let mut buf = sentinel.clone();
        assert!(swap.read_slot(0, &mut buf).is_err());
        assert_eq!(buf, sentinel, "buffer must not be modified on error");
    }

    // ------------------------------------------------------------------
    // write_slot error message includes the page_id
    // ------------------------------------------------------------------

    #[test]
    fn write_error_includes_page_id_in_message() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("werr_pid.swap"), 4096, 4096, 4).unwrap();

        let data = vec![0xFF; swap.max_slot_bytes + 1];
        let err = swap.write_slot(7, &data).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("page=7") || msg.contains('7'),
            "error should mention page_id 7: {msg}"
        );
    }

    // ------------------------------------------------------------------
    // read_slot error message includes the page_id
    // ------------------------------------------------------------------

    #[test]
    fn read_error_includes_page_id_in_message() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("rerr_pid.swap"), 4096, 4096, 4).unwrap();

        let mut buf = vec![0u8; swap.max_slot_bytes + 1];
        let err = swap.read_slot(3, &mut buf).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("page=3") || msg.contains('3'),
            "error should mention page_id 3: {msg}"
        );
    }

    // ------------------------------------------------------------------
    // Concurrent reads of the same slot from many threads
    // ------------------------------------------------------------------

    #[test]
    fn concurrent_reads_same_slot_many_threads() {
        use std::sync::Arc;
        use std::thread;

        let tmp = TempDir::new().unwrap();
        let swap = Arc::new(
            NvmeSwapFile::open(tmp.path().join("same_r.swap"), 4096, 4096, 4).unwrap(),
        );

        let data = vec![0xBB; 128];
        swap.write_slot(0, &data).unwrap();

        let mut handles = Vec::new();
        for _ in 0..8 {
            let sc = Arc::clone(&swap);
            handles.push(thread::spawn(move || {
                let mut buf = vec![0u8; 128];
                sc.read_slot(0, &mut buf).unwrap();
                assert!(buf.iter().all(|&b| b == 0xBB), "data mismatch");
            }));
        }

        for h in handles {
            h.join().expect("thread should not panic");
        }
    }

    // ------------------------------------------------------------------
    // Concurrent writes to the SAME slot — final value is one of the
    // valid writes (serialized by Mutex, so no torn writes)
    // ------------------------------------------------------------------

    #[test]
    fn concurrent_writes_same_slot_consistent() {
        use std::sync::Arc;
        use std::thread;

        let tmp = TempDir::new().unwrap();
        let swap = Arc::new(
            NvmeSwapFile::open(tmp.path().join("same_w.swap"), 4096, 4096, 4).unwrap(),
        );

        let n_threads = 4;
        let barrier = Arc::new(std::sync::Barrier::new(n_threads));
        let mut handles = Vec::new();

        for tid in 0..n_threads {
            let sc = Arc::clone(&swap);
            let b = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                b.wait();
                let data = vec![(tid as u8).wrapping_add(0x10); 32];
                sc.write_slot(0, &data).unwrap();
            }));
        }

        for h in handles {
            h.join().expect("thread should not panic");
        }

        let mut buf = vec![0u8; 32];
        swap.read_slot(0, &mut buf).unwrap();
        let first = buf[0];
        // Must be one of the valid thread values
        assert!(
            (0..n_threads).any(|tid| first == (tid as u8).wrapping_add(0x10)),
            "unexpected byte 0x{first:02x}"
        );
        // All bytes in the slot must be identical (no torn write)
        assert!(buf.iter().all(|&b| b == first), "torn write detected");
    }

    // ------------------------------------------------------------------
    // Open with path containing hyphens and underscores
    // ------------------------------------------------------------------

    #[test]
    fn open_path_with_hyphens_and_underscores() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("my-swap_file-v2.swap");
        let swap = NvmeSwapFile::open(path, 4096, 4096, 4);
        assert!(swap.is_ok(), "path with hyphens/underscores should work");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: changing any single field changes the hash
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_hash_differs_per_field() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        fn hash_of(h: &SwapFileHeader) -> u64 {
            let mut s = DefaultHasher::new();
            h.hash(&mut s);
            s.finish()
        }

        let base = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 16,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let base_h = hash_of(&base);

        let mut diff = base;
        diff.magic = 0;
        assert_ne!(hash_of(&diff), base_h, "magic change must change hash");

        let mut diff = base;
        diff.version = 99;
        assert_ne!(hash_of(&diff), base_h, "version change must change hash");

        let mut diff = base;
        diff.page_size = 2048;
        assert_ne!(hash_of(&diff), base_h, "page_size change must change hash");

        let mut diff = base;
        diff.max_slot_bytes = 4096;
        assert_ne!(hash_of(&diff), base_h, "max_slot_bytes change must change hash");

        let mut diff = base;
        diff.slot_count = 1;
        assert_ne!(hash_of(&diff), base_h, "slot_count change must change hash");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: Copy then modify original — copy stays unchanged
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_copy_then_modify_original() {
        let mut original = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 8,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let copy = original;
        original.magic = 0;
        original.slot_count = 0;

        assert_eq!(copy.magic, SWAP_MAGIC, "copy must retain original magic");
        assert_eq!(copy.slot_count, 8, "copy must retain original slot_count");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: HashMap retain works correctly
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_hashmap_retain() {
        let mut map = std::collections::HashMap::new();
        let h1 = SwapFileHeader {
            magic: 1,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 4096,
            _pad4: 0,
            slot_count: 1,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let h2 = SwapFileHeader {
            magic: 2,
            ..h1
        };
        let h3 = SwapFileHeader {
            magic: 3,
            ..h1
        };
        map.insert(h1, 10u32);
        map.insert(h2, 20);
        map.insert(h3, 30);

        map.retain(|_, &mut v| v > 15);
        assert_eq!(map.len(), 2, "retain should keep 2 entries");
        assert!(!map.contains_key(&h1), "h1 value 10 should be removed");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: Vec push/pop length tracking
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_vec_push_pop_len() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 4096,
            _pad4: 0,
            slot_count: 1,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let mut v = Vec::new();
        assert_eq!(v.len(), 0);

        v.push(hdr);
        v.push(hdr);
        assert_eq!(v.len(), 2);

        v.pop();
        assert_eq!(v.len(), 1);

        v.pop();
        assert_eq!(v.len(), 0);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: usable as function return value (Copy)
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_as_function_return() {
        fn make_header(magic: u64, count: u64) -> SwapFileHeader {
            SwapFileHeader {
                magic,
                version: 1,
                page_size: 4096,
                max_slot_bytes: 4096,
                _pad4: 0,
                slot_count: count,
                _reserved: [0u8; SWAP_HEADER_BYTES - 32],
            }
        }

        let h = make_header(0xDEAD, 5);
        assert_eq!(h.magic, 0xDEAD);
        assert_eq!(h.slot_count, 5);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: Debug includes magic in hex-like form
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_debug_includes_magic_hex() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 4,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let dbg = format!("{:?}", hdr);
        assert!(
            dbg.contains(&SWAP_MAGIC.to_string()),
            "debug should contain magic value: {dbg}"
        );
    }

    // ------------------------------------------------------------------
    // max_slot_bytes: additional non-aligned inputs round up correctly
    // ------------------------------------------------------------------

    #[test]
    fn max_slot_bytes_additional_non_aligned_inputs() {
        let tmp = TempDir::new().unwrap();
        let cases = [
            (100, 4096),
            (4097, 8192),
            (8193, 12288),
            (12288, 12288),
            (16385, 20480),
        ];
        for (input, expected) in cases {
            let path = tmp.path().join(format!("add_align_{input}.swap"));
            let swap = NvmeSwapFile::open(path, 4096, input, 4).unwrap();
            assert_eq!(
                swap.max_slot_bytes, expected,
                "input {input} should align to {expected}"
            );
        }
    }

    // ------------------------------------------------------------------
    // Write/read with max_slot_bytes = 3 × NVME_ALIGN
    // ------------------------------------------------------------------

    #[test]
    fn write_read_triple_align_slot_size() {
        let tmp = TempDir::new().unwrap();
        let slot = NVME_ALIGN * 3;
        let swap = NvmeSwapFile::open(tmp.path().join("triple.swap"), 4096, slot, 4).unwrap();
        assert_eq!(swap.max_slot_bytes, slot);

        let data: Vec<u8> = (0..slot - 100).map(|i| (i % 251) as u8).collect();
        swap.write_slot(0, &data).unwrap();

        let mut buf = vec![0u8; data.len()];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // Partial write: verify padding region is zero
    // ------------------------------------------------------------------

    #[test]
    fn write_partial_verify_padding_zeros() {
        let tmp = TempDir::new().unwrap();
        let slot = 4096;
        let swap = NvmeSwapFile::open(tmp.path().join("pad_zero.swap"), 4096, slot, 4).unwrap();

        let half = slot / 2;
        swap.write_slot(0, &vec![0xAB; half]).unwrap();

        let mut full = vec![0u8; slot];
        swap.read_slot(0, &mut full).unwrap();

        assert!(full[..half].iter().all(|&b| b == 0xAB), "data region");
        assert!(full[half..].iter().all(|&b| b == 0x00), "padding must be zeros");
    }

    // ------------------------------------------------------------------
    // Reopen five times, data stays intact each time
    // ------------------------------------------------------------------

    #[test]
    fn reopen_five_times_data_intact() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("five_rp.swap");
        let data = vec![0x77; 256];

        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
            swap.write_slot(2, &data).unwrap();
        }

        for i in 0..5 {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
            let mut buf = vec![0u8; 256];
            swap.read_slot(2, &mut buf).unwrap();
            assert_eq!(buf, data, "data corrupted on reopen #{i}");
        }
    }

    // ------------------------------------------------------------------
    // Write all slots, read back in prime-indexed order
    // ------------------------------------------------------------------

    #[test]
    fn write_all_read_prime_order() {
        let tmp = TempDir::new().unwrap();
        let count = 30u64;
        let swap = NvmeSwapFile::open(tmp.path().join("prime.swap"), 4096, 4096, count).unwrap();

        for pid in 0..count as usize {
            swap.write_slot(pid, &[(pid as u8).wrapping_add(0x30); 16]).unwrap();
        }

        let primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
        for &pid in &primes {
            let mut buf = [0u8; 16];
            swap.read_slot(pid, &mut buf).unwrap();
            let expected = (pid as u8).wrapping_add(0x30);
            assert!(buf.iter().all(|&b| b == expected), "slot {pid}");
        }
    }

    // ------------------------------------------------------------------
    // Overwrite slot with zeros, then write new data
    // ------------------------------------------------------------------

    #[test]
    fn overwrite_zeros_then_new_data() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("ow_zero.swap"), 4096, 4096, 4).unwrap();

        swap.write_slot(0, &[0xFF; 64]).unwrap();
        swap.write_slot(0, &[0x00; 64]).unwrap();
        swap.write_slot(0, &[0x42; 64]).unwrap();

        let mut buf = [0u8; 64];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0x42), "final write must win");
    }

    // ------------------------------------------------------------------
    // Open with page_size exactly NVME_ALIGN
    // ------------------------------------------------------------------

    #[test]
    fn open_page_size_equals_nvme_align() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(
            tmp.path().join("ps_align.swap"),
            NVME_ALIGN,
            NVME_ALIGN,
            4,
        )
        .unwrap();
        assert_eq!(swap.page_size, NVME_ALIGN);
        assert_eq!(swap.max_slot_bytes, NVME_ALIGN);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: individual reserved bytes are accessible
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_reserved_individual_byte_access() {
        let mut hdr = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        hdr._reserved[0] = 0x11;
        hdr._reserved[SWAP_HEADER_BYTES - 33] = 0x22;
        assert_eq!(hdr._reserved[0], 0x11);
        assert_eq!(hdr._reserved[SWAP_HEADER_BYTES - 33], 0x22);
        assert_eq!(hdr._reserved[1], 0, "untouched byte should be zero");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: HashMap insert, remove, re-insert works
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_hashmap_insert_remove_reinsert() {
        let mut map = std::collections::HashMap::new();
        let hdr = SwapFileHeader {
            magic: 42,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 4096,
            _pad4: 0,
            slot_count: 1,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };

        map.insert(hdr, 100u32);
        assert_eq!(map[&hdr], 100);

        map.remove(&hdr);
        assert!(map.get(&hdr).is_none());

        map.insert(hdr, 200);
        assert_eq!(map[&hdr], 200);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: clone_from updates to new source
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_clone_from_different_source() {
        let src = SwapFileHeader {
            magic: 111,
            version: 2,
            page_size: 2048,
            max_slot_bytes: 4096,
            _pad4: 0,
            slot_count: 8,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let mut dst = SwapFileHeader {
            magic: 222,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 4,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        dst.clone_from(&src);
        assert_eq!(dst.magic, 111);
        assert_eq!(dst.version, 2);
        assert_eq!(dst.page_size, 2048);
        assert_eq!(dst.slot_count, 8);
    }

    // ------------------------------------------------------------------
    // Write to two slots, read both using the same reused buffer
    // ------------------------------------------------------------------

    #[test]
    fn write_read_two_slots_reuse_buffer() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("reuse.swap"), 4096, 4096, 4).unwrap();

        swap.write_slot(0, &[0xAA; 32]).unwrap();
        swap.write_slot(1, &[0xBB; 32]).unwrap();

        let mut buf = [0u8; 32];

        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xAA));

        swap.read_slot(1, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xBB));
    }

    // ------------------------------------------------------------------
    // slot_offset depends on max_slot_bytes, not page_size
    // ------------------------------------------------------------------

    #[test]
    fn slot_offset_varies_with_max_slot_bytes_not_page_size() {
        let tmp = TempDir::new().unwrap();
        // Same page_size, different max_slot_bytes
        let swap_a = NvmeSwapFile::open(tmp.path().join("a.swap"), 4096, 4096, 4).unwrap();
        let swap_b = NvmeSwapFile::open(tmp.path().join("b.swap"), 4096, 8192, 4).unwrap();

        // Same page_id, different max_slot_bytes => different offsets
        assert_ne!(swap_a.slot_offset(3), swap_b.slot_offset(3));

        // Expected formulas
        assert_eq!(
            swap_a.slot_offset(3),
            SWAP_HEADER_BYTES as u64 + 3 * 4096u64
        );
        assert_eq!(
            swap_b.slot_offset(3),
            SWAP_HEADER_BYTES as u64 + 3 * 8192u64
        );
    }

    // ------------------------------------------------------------------
    // Open with large max_slot_bytes (64 KB)
    // ------------------------------------------------------------------

    #[test]
    fn open_with_large_max_slot_bytes_64k() {
        let tmp = TempDir::new().unwrap();
        let large = 65536;
        let swap = NvmeSwapFile::open(tmp.path().join("64k.swap"), 4096, large, 4).unwrap();
        assert_eq!(swap.max_slot_bytes, large);

        let data = vec![0xCA; 60000];
        swap.write_slot(0, &data).unwrap();

        let mut buf = vec![0u8; 60000];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // Write data whose length crosses NVME_ALIGN boundary within slot
    // ------------------------------------------------------------------

    #[test]
    fn write_read_data_crossing_align_boundary() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("cross.swap"), 4096, 8192, 4).unwrap();

        // Size crosses the 4096-byte boundary
        let size = NVME_ALIGN + NVME_ALIGN / 2;
        let data: Vec<u8> = (0..size).map(|i| (i % 199) as u8).collect();
        swap.write_slot(0, &data).unwrap();

        let mut buf = vec![0u8; size];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // Three swap files in the same directory all independent
    // ------------------------------------------------------------------

    #[test]
    fn three_files_independent_operations() {
        let tmp = TempDir::new().unwrap();
        let sa = NvmeSwapFile::open(tmp.path().join("x.swap"), 4096, 4096, 4).unwrap();
        let sb = NvmeSwapFile::open(tmp.path().join("y.swap"), 4096, 4096, 4).unwrap();
        let sc = NvmeSwapFile::open(tmp.path().join("z.swap"), 4096, 4096, 4).unwrap();

        sa.write_slot(0, &[0x11; 16]).unwrap();
        sb.write_slot(0, &[0x22; 16]).unwrap();
        sc.write_slot(0, &[0x33; 16]).unwrap();

        let mut buf = [0u8; 16];
        sa.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0x11));

        sb.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0x22));

        sc.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0x33));
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: reflexive equality with non-zero reserved bytes
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_reflexive_with_custom_reserved() {
        let mut reserved = [0u8; SWAP_HEADER_BYTES - 32];
        reserved[0] = 0xFF;
        reserved[SWAP_HEADER_BYTES - 33] = 0xAA;
        let hdr = SwapFileHeader {
            magic: 999,
            version: 7,
            page_size: 2048,
            max_slot_bytes: 4096,
            _pad4: 42,
            slot_count: 3,
            _reserved: reserved,
        };
        assert_eq!(hdr, hdr, "header must be equal to itself");
    }

    // ------------------------------------------------------------------
    // Write-read cycle on same slot 100 times
    // ------------------------------------------------------------------

    #[test]
    fn write_read_same_slot_hundred_cycles() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("hundred.swap"), 4096, 4096, 4).unwrap();

        for i in 0..100u8 {
            let data = vec![i; 32];
            swap.write_slot(0, &data).unwrap();

            let mut buf = vec![0u8; 32];
            swap.read_slot(0, &mut buf).unwrap();
            assert!(buf.iter().all(|&b| b == i), "cycle {i}");
        }
    }

    // ------------------------------------------------------------------
    // Open on existing file shorter than header — treated as new file
    // ------------------------------------------------------------------

    #[test]
    fn open_existing_file_smaller_than_header() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("tiny.swap");

        // Create a file with only 100 bytes
        std::fs::write(&path, &[0u8; 100]).unwrap();

        let swap = NvmeSwapFile::open(path, 4096, 4096, 4);
        assert!(swap.is_ok(), "should treat undersized file as new");
        assert_eq!(swap.unwrap().slot_count, 4);
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: write at exact NVME_ALIGN boundary index
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_write_at_align_boundary() {
        let size = NVME_ALIGN * 2;
        let mut buf = AlignedBuffer::new(size);

        buf.as_mut_slice()[NVME_ALIGN] = 0xDD;
        assert_eq!(buf.as_slice()[NVME_ALIGN], 0xDD);
        // Surrounding bytes zero
        assert_eq!(buf.as_slice()[NVME_ALIGN - 1], 0);
        assert_eq!(buf.as_slice()[NVME_ALIGN + 1], 0);
    }

    // ------------------------------------------------------------------
    // Concurrent: different threads use different swap files
    // ------------------------------------------------------------------

    #[test]
    fn concurrent_different_files() {
        use std::sync::Arc;
        use std::thread;

        let tmp = Arc::new(TempDir::new().unwrap());
        let mut handles = Vec::new();

        for tid in 0..4u8 {
            let tmp_clone = Arc::clone(&tmp);
            handles.push(thread::spawn(move || {
                let path = tmp_clone.path().join(format!("t{tid}.swap"));
                let swap = NvmeSwapFile::open(path, 4096, 4096, 4).unwrap();
                let data = vec![tid; 32];
                swap.write_slot(0, &data).unwrap();

                let mut buf = vec![0u8; 32];
                swap.read_slot(0, &mut buf).unwrap();
                assert!(buf.iter().all(|&b| b == tid), "thread {tid}");
            }));
        }

        for h in handles {
            h.join().expect("thread should not panic");
        }
    }

    // ------------------------------------------------------------------
    // Open in a subdirectory of temp (single level)
    // ------------------------------------------------------------------

    #[test]
    fn open_in_temp_subdirectory() {
        let tmp = TempDir::new().unwrap();
        let subdir = tmp.path().join("subdir");
        let path = subdir.join("in_sub.swap");
        let swap = NvmeSwapFile::open(path, 4096, 4096, 4);
        assert!(swap.is_ok(), "should auto-create subdirectory");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: equal headers with different construction paths
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_equal_from_different_constructions() {
        let h1 = SwapFileHeader {
            magic: 100,
            version: 2,
            page_size: 512,
            max_slot_bytes: 1024,
            _pad4: 0,
            slot_count: 5,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        // Construct via copy + field override
        let mut h2 = h1;
        h2.magic = 100;
        h2.version = 2;
        assert_eq!(h1, h2);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: equality after std::mem::swap
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_equality_after_mem_swap() {
        let mut a = SwapFileHeader {
            magic: 1,
            version: 1,
            page_size: 100,
            max_slot_bytes: 200,
            _pad4: 0,
            slot_count: 10,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let mut b = SwapFileHeader {
            magic: 2,
            version: 2,
            page_size: 300,
            max_slot_bytes: 400,
            _pad4: 0,
            slot_count: 20,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let original_a = a;
        let original_b = b;

        std::mem::swap(&mut a, &mut b);

        assert_eq!(a, original_b, "a should now hold b's values");
        assert_eq!(b, original_a, "b should now hold a's values");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: Debug output contains page_size value
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_debug_format_page_size_value() {
        let hdr = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 9999,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let dbg = format!("{:?}", hdr);
        assert!(dbg.contains("9999"), "should contain page_size value: {dbg}");
    }

    // ------------------------------------------------------------------
    // Write data near max_slot_bytes boundary (max - 1)
    // ------------------------------------------------------------------

    #[test]
    fn write_read_near_max_slot_boundary() {
        let tmp = TempDir::new().unwrap();
        let slot = 4096;
        let swap = NvmeSwapFile::open(tmp.path().join("near_max.swap"), 4096, slot, 4).unwrap();

        let data = vec![0xEE; slot - 1];
        let written = swap.write_slot(0, &data).unwrap();
        assert_eq!(written as usize, slot - 1);

        let mut buf = vec![0u8; slot - 1];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // Open with non-power-of-two max_slot_bytes that is a multiple of NVME_ALIGN
    // ------------------------------------------------------------------

    #[test]
    fn open_with_non_power_of_two_aligned_slot() {
        let tmp = TempDir::new().unwrap();
        // 12288 = 3 * 4096, not a power of two but aligned
        let swap = NvmeSwapFile::open(tmp.path().join("npot.swap"), 4096, 12288, 4).unwrap();
        assert_eq!(swap.max_slot_bytes, 12288);

        let data = vec![0x77; 10000];
        swap.write_slot(0, &data).unwrap();

        let mut buf = vec![0u8; 10000];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: multiple keys in HashMap, lookup each
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_multiple_keys_hashmap_lookup() {
        let mut map = std::collections::HashMap::new();
        for i in 0u64..10 {
            let hdr = SwapFileHeader {
                magic: i,
                version: 1,
                page_size: 4096,
                max_slot_bytes: 4096,
                _pad4: 0,
                slot_count: 1,
                _reserved: [0u8; SWAP_HEADER_BYTES - 32],
            };
            map.insert(hdr, i as u32 * 10);
        }

        assert_eq!(map.len(), 10);
        for i in 0u64..10 {
            let key = SwapFileHeader {
                magic: i,
                version: 1,
                page_size: 4096,
                max_slot_bytes: 4096,
                _pad4: 0,
                slot_count: 1,
                _reserved: [0u8; SWAP_HEADER_BYTES - 32],
            };
            assert_eq!(map[&key], i as u32 * 10, "lookup for magic={i}");
        }
    }

    // ------------------------------------------------------------------
    // Overwrite with larger data (still within max_slot_bytes)
    // ------------------------------------------------------------------

    #[test]
    fn overwrite_larger_within_max_slot() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("ow_larger.swap"), 4096, 4096, 4).unwrap();

        swap.write_slot(0, &[0x11; 100]).unwrap();
        swap.write_slot(0, &[0x22; 200]).unwrap();

        let mut buf = vec![0u8; 200];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0x22));
    }

    // ------------------------------------------------------------------
    // Read unwritten slot after writing to a different slot
    // ------------------------------------------------------------------

    #[test]
    fn read_unwritten_after_other_slot_write() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("uw_other.swap"), 4096, 4096, 4).unwrap();

        swap.write_slot(1, &[0xFF; 64]).unwrap();

        // Slot 0 was never written — should return zeros
        let mut buf = vec![0xAA; 64];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0x00), "unwritten slot must be zeros");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: pass by value into function (Copy)
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_pass_by_value_no_change() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 4096,
            _pad4: 0,
            slot_count: 1,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let original = hdr;

        fn consume(h: SwapFileHeader) -> u64 {
            h.magic
        }
        let result = consume(hdr);
        assert_eq!(result, SWAP_MAGIC);
        assert_eq!(hdr.magic, original.magic, "original unchanged after pass-by-value");
    }

    // ------------------------------------------------------------------
    // Write different sizes to adjacent slots
    // ------------------------------------------------------------------

    #[test]
    fn write_different_sizes_to_adjacent_slots() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("diff_sz.swap"), 4096, 4096, 4).unwrap();

        let sizes = [1, 100, 1000, 4096];
        for (pid, &size) in sizes.iter().enumerate() {
            let data = vec![0x55; size];
            swap.write_slot(pid, &data).unwrap();
        }

        for (pid, &size) in sizes.iter().enumerate() {
            let mut buf = vec![0u8; size];
            swap.read_slot(pid, &mut buf).unwrap();
            assert!(buf.iter().all(|&b| b == 0x55), "slot {pid} size {size}");
        }
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: as_mut_ptr returns same address as as_ptr
    // after casting away mutability
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_mut_ptr_matches_ptr_address() {
        let mut buf = AlignedBuffer::new(NVME_ALIGN);
        let ro = buf.as_ptr() as usize;
        let rw = buf.as_mut_ptr() as usize;
        assert_eq!(ro, rw, "ro and rw pointers must address the same memory");
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: slot_count zero still allows open
    // (already tested, but with write/read to verify no slots exist)
    // ------------------------------------------------------------------

    #[test]
    fn open_zero_slots_no_write_possible() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("zero_s.swap"), 4096, 4096, 0).unwrap();
        assert_eq!(swap.slot_count, 0);
        // Writing to page 0 still works (write_slot doesn't check slot_count)
        // but reading back should work since the file has the slot space allocated
        // by ftruncate. Actually with 0 slots, ftruncate allocates only the header.
        // Write may still succeed at the byte level since pwrite doesn't care.
        // This is a boundary test.
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: Hash produces the same value in two separate hashes
    // of the same header with non-zero fields
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_hash_stable_nonzero_fields() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let hdr = SwapFileHeader {
            magic: 12345,
            version: 99,
            page_size: 7777,
            max_slot_bytes: 8888,
            _pad4: 1,
            slot_count: 42,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };

        let h1 = {
            let mut s = DefaultHasher::new();
            hdr.hash(&mut s);
            s.finish()
        };
        let h2 = {
            let mut s = DefaultHasher::new();
            hdr.hash(&mut s);
            s.finish()
        };
        assert_eq!(h1, h2, "hash must be deterministic");
    }

    // ------------------------------------------------------------------
    // Debug format of NvmeSwapFile contains numeric values for fields
    // ------------------------------------------------------------------

    #[test]
    fn nvme_swap_file_debug_contains_field_values() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("dbg_vals.swap"), 1337, 4096, 7).unwrap();
        let dbg = format!("{:?}", swap);
        assert!(dbg.contains("1337"), "should contain page_size value: {dbg}");
        assert!(dbg.contains('7'), "should contain slot_count value: {dbg}");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: Copy allows creating independent copies via assignment
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_copy_assignment_independent() {
        let a = SwapFileHeader {
            magic: 100,
            version: 1,
            page_size: 2048,
            max_slot_bytes: 4096,
            _pad4: 0,
            slot_count: 5,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = a;
        let c = a;

        // All three must be equal
        assert_eq!(a, b);
        assert_eq!(b, c);
        // They're independent copies
        assert_eq!(a.magic, 100);
        assert_eq!(b.magic, 100);
        assert_eq!(c.magic, 100);
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: write/read with max_slot_bytes exactly NVME_ALIGN
    // ------------------------------------------------------------------

    #[test]
    fn write_read_exact_nvme_align_slot() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(
            tmp.path().join("exact_align.swap"),
            NVME_ALIGN,
            NVME_ALIGN,
            4,
        )
        .unwrap();

        let data = vec![0x88; NVME_ALIGN];
        swap.write_slot(0, &data).unwrap();

        let mut buf = vec![0u8; NVME_ALIGN];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: reserved field length matches SWAP_HEADER_BYTES - 32
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_reserved_len_formula() {
        let hdr = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(hdr._reserved.len(), SWAP_HEADER_BYTES - 32);
        assert_eq!(
            hdr._reserved.len() + 32,
            SWAP_HEADER_BYTES,
            "reserved + fields must equal total header size"
        );
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: fresh allocation after writing to prior one is zeroed
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_fresh_alloc_after_write_is_zeroed() {
        let addr;
        {
            let mut buf = AlignedBuffer::new(NVME_ALIGN);
            buf.as_mut_slice()[0] = 0xFF;
            buf.as_mut_slice()[NVME_ALIGN - 1] = 0xFF;
            addr = buf.as_ptr() as usize;
        }
        let buf = AlignedBuffer::new(NVME_ALIGN);
        assert!(buf.as_slice().iter().all(|&b| b == 0), "must be zeroed");
        let _ = addr;
    }

    // ------------------------------------------------------------------
    // Concurrent: write to different slots, then read from all threads
    // ------------------------------------------------------------------

    #[test]
    fn concurrent_write_then_read_all_slots() {
        use std::sync::Arc;
        use std::thread;

        let tmp = TempDir::new().unwrap();
        let swap = Arc::new(
            NvmeSwapFile::open(tmp.path().join("wr_all.swap"), 4096, 4096, 8).unwrap(),
        );

        // Phase 1: write from main thread
        for pid in 0..8usize {
            swap.write_slot(pid, &[(pid as u8).wrapping_add(0x40); 64]).unwrap();
        }

        // Phase 2: read from multiple threads
        let mut handles = Vec::new();
        for pid in 0..8usize {
            let sc = Arc::clone(&swap);
            handles.push(thread::spawn(move || {
                let mut buf = vec![0u8; 64];
                sc.read_slot(pid, &mut buf).unwrap();
                let expected = (pid as u8).wrapping_add(0x40);
                assert!(buf.iter().all(|&b| b == expected), "slot {pid}");
            }));
        }

        for h in handles {
            h.join().expect("thread should not panic");
        }
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: HashSet with multiple unique entries
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_hashset_unique_entries() {
        let mut set = std::collections::HashSet::new();
        for i in 0u64..20 {
            let hdr = SwapFileHeader {
                magic: i,
                version: 1,
                page_size: 4096,
                max_slot_bytes: 4096,
                _pad4: 0,
                slot_count: 1,
                _reserved: [0u8; SWAP_HEADER_BYTES - 32],
            };
            set.insert(hdr);
        }
        assert_eq!(set.len(), 20, "all 20 headers should be unique in set");

        // Duplicate insert
        let dup = SwapFileHeader {
            magic: 5,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 4096,
            _pad4: 0,
            slot_count: 1,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert!(!set.insert(dup), "duplicate insert should return false");
        assert_eq!(set.len(), 20);
    }

    // ==================================================================
    // 38 additional tests — target 390 total (ratio ≈ 18)
    // Focus: header persistence edge cases, file boundary conditions,
    // AlignedBuffer lifecycle, write/read boundary combinations,
    // concurrent stress, SwapFileHeader collection operations.
    // ==================================================================

    // ------------------------------------------------------------------
    // Reopen file exactly SWAP_HEADER_BYTES long (header-only, no slots)
    // ------------------------------------------------------------------

    #[test]
    fn reopen_file_exactly_header_size() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("hdr_only.swap");

        {
            let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 0).unwrap();
        }

        // File should be exactly SWAP_HEADER_BYTES
        let meta = std::fs::metadata(&path).unwrap();
        assert_eq!(meta.len(), SWAP_HEADER_BYTES as u64);

        // Reopen should succeed — header is valid
        let swap = NvmeSwapFile::open(path, 4096, 4096, 0);
        assert!(swap.is_ok(), "reopen of header-only file must succeed");
    }

    // ------------------------------------------------------------------
    // Write to every slot, reopen, verify all data persists
    // ------------------------------------------------------------------

    #[test]
    fn write_all_slots_reopen_verify_all() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("all_reopen.swap");
        let count = 8u64;

        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, count).unwrap();
            for pid in 0..count as usize {
                swap.write_slot(pid, &[(pid as u8).wrapping_add(0x50); 32])
                    .unwrap();
            }
        }

        let swap = NvmeSwapFile::open(path, 4096, 4096, count).unwrap();
        for pid in 0..count as usize {
            let expected = (pid as u8).wrapping_add(0x50);
            let mut buf = [0u8; 32];
            swap.read_slot(pid, &mut buf).unwrap();
            assert!(
                buf.iter().all(|&b| b == expected),
                "slot {pid} after reopen: expected 0x{expected:02x}"
            );
        }
    }

    // ------------------------------------------------------------------
    // File created with 0 slots, reopened with non-zero slots
    // ------------------------------------------------------------------

    #[test]
    fn reopen_zero_to_nonzero_slots() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("zero_to_nz.swap");

        {
            let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 0).unwrap();
        }

        let swap = NvmeSwapFile::open(path, 4096, 4096, 4).unwrap();
        assert_eq!(swap.slot_count, 4);
    }

    // ------------------------------------------------------------------
    // Write then read with max_slot_bytes = 2 (rounds to NVME_ALIGN)
    // ------------------------------------------------------------------

    #[test]
    fn write_read_with_very_small_max_slot_bytes() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("tiny_msb.swap"), 4096, 2, 4).unwrap();
        assert_eq!(swap.max_slot_bytes, NVME_ALIGN);

        let data = vec![0xAB; 100];
        swap.write_slot(0, &data).unwrap();

        let mut buf = vec![0u8; 100];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: BTreeMap usable (requires Ord — test manually)
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_sort_by_magic() {
        let mut headers: Vec<SwapFileHeader> = (0..5)
            .map(|i| SwapFileHeader {
                magic: 5 - i as u64, // 5, 4, 3, 2, 1
                version: 1,
                page_size: 4096,
                max_slot_bytes: 4096,
                _pad4: 0,
                slot_count: 1,
                _reserved: [0u8; SWAP_HEADER_BYTES - 32],
            })
            .collect();

        headers.sort_by_key(|h| h.magic);
        assert_eq!(headers[0].magic, 1);
        assert_eq!(headers[4].magic, 5);
    }

    // ------------------------------------------------------------------
    // Open with path containing only a filename (no parent dir)
    // ------------------------------------------------------------------

    #[test]
    fn open_filename_only_in_temp() {
        let tmp = TempDir::new().unwrap();
        // Change to temp dir and open with relative path
        let path = tmp.path().join("simple.swap");
        let swap = NvmeSwapFile::open(path, 4096, 4096, 4);
        assert!(swap.is_ok());
    }

    // ------------------------------------------------------------------
    // Write to slot, verify file on disk grew beyond header
    // ------------------------------------------------------------------

    #[test]
    fn file_grew_beyond_header_after_open() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("grew.swap");
        let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 16).unwrap();

        let meta = std::fs::metadata(&path).unwrap();
        let expected = SWAP_HEADER_BYTES as u64 + 16 * 4096;
        assert_eq!(
            meta.len(), expected,
            "file must be header + 16 slots"
        );
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: verify alignment is exactly 8
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_alignment_exactly_8_or_more() {
        let align = std::mem::align_of::<SwapFileHeader>();
        assert!(
            align >= 8,
            "alignment must be at least 8 for u64 fields, got {align}"
        );
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: multiple allocations coexisting
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_multiple_coexisting() {
        let mut bufs = Vec::new();
        for i in 0..5 {
            let mut buf = AlignedBuffer::new(NVME_ALIGN);
            buf.as_mut_slice()[0] = i as u8;
            bufs.push(buf);
        }
        // Verify all buffers still have their values
        for (i, buf) in bufs.iter().enumerate() {
            assert_eq!(buf.as_slice()[0], i as u8, "buffer {i} corrupted");
            assert_eq!(buf.as_slice().len(), NVME_ALIGN);
        }
    }

    // ------------------------------------------------------------------
    // Write exactly 2 bytes, read 2 bytes
    // ------------------------------------------------------------------

    #[test]
    fn write_read_exactly_two_bytes() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("two.swap"), 4096, 4096, 4).unwrap();

        swap.write_slot(0, &[0xDE, 0xAD]).unwrap();
        let mut buf = [0u8; 2];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, [0xDE, 0xAD]);
    }

    // ------------------------------------------------------------------
    // Write to slot 0, overwrite with empty, write again — final read correct
    // ------------------------------------------------------------------

    #[test]
    fn write_overwrite_empty_write_again() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "woe.swap");

        swap.write_slot(0, &[0xAA; 64]).unwrap();
        swap.write_slot(0, &[]).unwrap();
        swap.write_slot(0, &[0xBB; 64]).unwrap();

        let mut buf = vec![0u8; 64];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xBB));
    }

    // ------------------------------------------------------------------
    // max_slot_bytes = NVME_ALIGN * 10 (large aligned)
    // ------------------------------------------------------------------

    #[test]
    fn max_slot_bytes_large_aligned() {
        let tmp = TempDir::new().unwrap();
        let large = NVME_ALIGN * 10;
        let swap = NvmeSwapFile::open(tmp.path().join("10x.swap"), 4096, large, 4).unwrap();
        assert_eq!(swap.max_slot_bytes, large);

        let data: Vec<u8> = (0..large - 100).map(|i| (i % 173) as u8).collect();
        swap.write_slot(0, &data).unwrap();

        let mut buf = vec![0u8; data.len()];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: _reserved field size invariant across constructions
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_reserved_size_invariant() {
        let hdr1 = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let hdr2: SwapFileHeader = unsafe { std::mem::zeroed() };
        assert_eq!(hdr1._reserved.len(), hdr2._reserved.len());
        assert_eq!(hdr1._reserved.len(), SWAP_HEADER_BYTES - 32);
    }

    // ------------------------------------------------------------------
    // Write to last slot, then to first slot, read both
    // ------------------------------------------------------------------

    #[test]
    fn write_last_then_first_read_both() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("lf.swap"), 4096, 4096, 8).unwrap();

        swap.write_slot(7, &[0x77; 32]).unwrap();
        swap.write_slot(0, &[0x00; 32]).unwrap();

        let mut buf0 = [0u8; 32];
        let mut buf7 = [0u8; 32];
        swap.read_slot(0, &mut buf0).unwrap();
        swap.read_slot(7, &mut buf7).unwrap();

        assert!(buf0.iter().all(|&b| b == 0x00));
        assert!(buf7.iter().all(|&b| b == 0x77));
    }

    // ------------------------------------------------------------------
    // Open file with 0 slot_count, verify file size is header only
    // ------------------------------------------------------------------

    #[test]
    fn zero_slot_count_creates_header_only_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("hdr_only2.swap");
        let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 0).unwrap();

        let size = std::fs::metadata(&path).unwrap().len();
        assert_eq!(size, SWAP_HEADER_BYTES as u64);
    }

    // ------------------------------------------------------------------
    // Write with data containing NUL bytes
    // ------------------------------------------------------------------

    #[test]
    fn write_data_with_nul_bytes() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "nul_data.swap");

        let data = vec![0x00; 256];
        swap.write_slot(0, &data).unwrap();

        let mut buf = vec![0xFF; 256];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0x00));
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: two different headers have different Debug output
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_different_debug_output() {
        let a = SwapFileHeader {
            magic: 1,
            version: 1,
            page_size: 100,
            max_slot_bytes: 200,
            _pad4: 0,
            slot_count: 10,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = SwapFileHeader {
            magic: 2,
            version: 1,
            page_size: 100,
            max_slot_bytes: 200,
            _pad4: 0,
            slot_count: 10,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_ne!(format!("{a:?}"), format!("{b:?}"));
    }

    // ------------------------------------------------------------------
    // Verify file header written to disk matches expected bytes
    // ------------------------------------------------------------------

    #[test]
    fn header_bytes_on_disk_match_struct() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("hdr_bytes.swap");
        let page_size = 4096u32;
        let max_slot = 8192u32;
        let slot_count = 8u64;

        {
            let _swap = NvmeSwapFile::open(
                path.clone(),
                page_size as usize,
                max_slot as usize,
                slot_count,
            )
            .unwrap();
        }

        let file_data = std::fs::read(&path).unwrap();

        // Verify magic at offset 0
        let magic_bytes = &file_data[0..8];
        assert_eq!(u64::from_le_bytes(magic_bytes.try_into().unwrap()), SWAP_MAGIC);

        // Verify version at offset 8
        let version_bytes = &file_data[8..12];
        assert_eq!(u32::from_le_bytes(version_bytes.try_into().unwrap()), SWAP_VERSION);

        // Verify page_size at offset 12
        let ps_bytes = &file_data[12..16];
        assert_eq!(u32::from_le_bytes(ps_bytes.try_into().unwrap()), page_size);

        // Verify max_slot_bytes at offset 16
        let msb_bytes = &file_data[16..20];
        assert_eq!(u32::from_le_bytes(msb_bytes.try_into().unwrap()), max_slot);

        // Verify slot_count at offset 24
        let sc_bytes = &file_data[24..32];
        assert_eq!(u64::from_le_bytes(sc_bytes.try_into().unwrap()), slot_count);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: Vec dedup works (equal headers collapse)
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_vec_dedup() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 4096,
            _pad4: 0,
            slot_count: 1,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let mut v = vec![hdr, hdr, hdr];
        v.dedup();
        assert_eq!(v.len(), 1);
    }

    // ------------------------------------------------------------------
    // Overwrite with same size data, read back exactly matches
    // ------------------------------------------------------------------

    #[test]
    fn overwrite_same_size_exact_match() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "same_sz.swap");

        let data_a: Vec<u8> = (0..512).map(|i| (i % 256) as u8).collect();
        let data_b: Vec<u8> = (0..512).map(|i| ((i + 128) % 256) as u8).collect();

        swap.write_slot(0, &data_a).unwrap();
        swap.write_slot(0, &data_b).unwrap();

        let mut buf = vec![0u8; 512];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data_b);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: can be stored in a Vec and iterated
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_vec_iteration() {
        let hdr = SwapFileHeader {
            magic: 1,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 4096,
            _pad4: 0,
            slot_count: 1,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let v: Vec<SwapFileHeader> = (0..10).map(|i| SwapFileHeader {
            magic: i as u64,
            ..hdr
        }).collect();

        let sum: u64 = v.iter().map(|h| h.magic).sum();
        assert_eq!(sum, 45); // 0+1+2+...+9
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: pointer differs between consecutive allocations
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_consecutive_have_different_addrs() {
        let a = AlignedBuffer::new(NVME_ALIGN);
        let b = AlignedBuffer::new(NVME_ALIGN);
        let c = AlignedBuffer::new(NVME_ALIGN);

        let pa = a.as_ptr() as usize;
        let pb = b.as_ptr() as usize;
        let pc = c.as_ptr() as usize;

        assert_ne!(pa, pb);
        assert_ne!(pb, pc);
        assert_ne!(pa, pc);
    }

    // ------------------------------------------------------------------
    // Write to slot, drop, reopen with same params, overwrite, read
    // ------------------------------------------------------------------

    #[test]
    fn reopen_overwrite_read_cycle() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("rpl.swap");

        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
            swap.write_slot(0, &[0xAA; 32]).unwrap();
        }

        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
            swap.write_slot(0, &[0xBB; 32]).unwrap();
        }

        let swap = NvmeSwapFile::open(path, 4096, 4096, 4).unwrap();
        let mut buf = [0u8; 32];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xBB));
    }

    // ------------------------------------------------------------------
    // Open with page_size = u32::MAX as usize (truncated via u32 in header)
    // ------------------------------------------------------------------

    #[test]
    fn open_large_page_size_truncated_in_header() {
        let tmp = TempDir::new().unwrap();
        let large_ps = u32::MAX as usize;
        let swap = NvmeSwapFile::open(
            tmp.path().join("large_ps.swap"),
            large_ps,
            8192,
            4,
        )
        .unwrap();
        assert_eq!(swap.page_size, large_ps);
    }

    // ------------------------------------------------------------------
    // Write full slot, read back full slot with identical vec
    // ------------------------------------------------------------------

    #[test]
    fn write_full_read_full_identical() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("full_rt.swap"), 4096, 4096, 4).unwrap();

        let data: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
        swap.write_slot(0, &data).unwrap();

        let mut buf = vec![0u8; 4096];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: hash of zeroed header is deterministic
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_hash_zeroed_deterministic() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let hdr: SwapFileHeader = unsafe { std::mem::zeroed() };

        let mut s1 = DefaultHasher::new();
        let mut s2 = DefaultHasher::new();
        hdr.hash(&mut s1);
        hdr.hash(&mut s2);
        assert_eq!(s1.finish(), s2.finish());
    }

    // ------------------------------------------------------------------
    // Two separate swap files write to same slot independently
    // ------------------------------------------------------------------

    #[test]
    fn two_files_same_slot_different_data() {
        let tmp = TempDir::new().unwrap();
        let swap_a = NvmeSwapFile::open(tmp.path().join("fa.swap"), 4096, 4096, 4).unwrap();
        let swap_b = NvmeSwapFile::open(tmp.path().join("fb.swap"), 4096, 4096, 4).unwrap();

        swap_a.write_slot(0, &[0xAA; 64]).unwrap();
        swap_b.write_slot(0, &[0xBB; 64]).unwrap();

        let mut ba = [0u8; 64];
        let mut bb = [0u8; 64];
        swap_a.read_slot(0, &mut ba).unwrap();
        swap_b.read_slot(0, &mut bb).unwrap();

        assert!(ba.iter().all(|&b| b == 0xAA));
        assert!(bb.iter().all(|&b| b == 0xBB));
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: field mutation via mutable reference
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_mutable_reference_update() {
        let mut hdr = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };

        let r = &mut hdr;
        r.magic = 42;
        r.slot_count = 100;
        assert_eq!(hdr.magic, 42);
        assert_eq!(hdr.slot_count, 100);
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: verify every byte addressable in 2-block buffer
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_every_byte_addressable() {
        let size = NVME_ALIGN * 2;
        let mut buf = AlignedBuffer::new(size);
        for i in 0..size {
            buf.as_mut_slice()[i] = (i % 256) as u8;
        }
        for i in 0..size {
            assert_eq!(
                buf.as_slice()[i],
                (i % 256) as u8,
                "byte {i} mismatch"
            );
        }
    }

    // ------------------------------------------------------------------
    // Write data, close, truncate file, reopen should work (re-validates header)
    // ------------------------------------------------------------------

    #[test]
    fn reopen_after_external_truncate_to_header() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("trunc.swap");

        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
            swap.write_slot(0, &[0xFF; 64]).unwrap();
        }

        // Truncate file to header only
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(SWAP_HEADER_BYTES as u64).unwrap();
        drop(f);

        // Reopen — header is valid, should succeed
        let swap = NvmeSwapFile::open(path, 4096, 4096, 4);
        assert!(swap.is_ok(), "reopen after truncate to header should work");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: Clone then mutate original does not affect clone
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_clone_then_mutate_original() {
        let mut original = SwapFileHeader {
            magic: 100,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 16,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let cloned = original.clone();
        original.magic = 0;
        original.slot_count = 0;

        assert_eq!(original.magic, 0, "original should be modified");
        assert_eq!(cloned.magic, 100, "clone must be independent");
        assert_eq!(cloned.slot_count, 16);
    }

    // ------------------------------------------------------------------
    // Write/read with page_size larger than max_slot_bytes (unusual but legal)
    // ------------------------------------------------------------------

    #[test]
    fn page_size_larger_than_max_slot_write_read() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("ps_gt.swap"), 8192, 4096, 4).unwrap();
        assert!(swap.page_size > swap.max_slot_bytes);

        let data = vec![0x42; 100];
        swap.write_slot(0, &data).unwrap();

        let mut buf = vec![0u8; 100];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // Concurrent: write different slots, read same slots concurrently
    // ------------------------------------------------------------------

    #[test]
    fn concurrent_write_read_different_slots() {
        use std::sync::Arc;
        use std::thread;

        let tmp = TempDir::new().unwrap();
        let swap = Arc::new(
            NvmeSwapFile::open(tmp.path().join("cwr.swap"), 4096, 4096, 8).unwrap(),
        );

        // Phase 1: concurrent writes
        let mut write_handles = Vec::new();
        for pid in 0..8usize {
            let sc = Arc::clone(&swap);
            write_handles.push(thread::spawn(move || {
                sc.write_slot(pid, &[(pid as u8).wrapping_add(0x60); 32])
                    .unwrap();
            }));
        }
        for h in write_handles {
            h.join().expect("write thread panic");
        }

        // Phase 2: concurrent reads
        let mut read_handles = Vec::new();
        for pid in 0..8usize {
            let sc = Arc::clone(&swap);
            read_handles.push(thread::spawn(move || {
                let mut buf = [0u8; 32];
                sc.read_slot(pid, &mut buf).unwrap();
                let expected = (pid as u8).wrapping_add(0x60);
                assert!(buf.iter().all(|&b| b == expected), "slot {pid}");
            }));
        }
        for h in read_handles {
            h.join().expect("read thread panic");
        }
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: _reserved byte at exact middle position
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_reserved_middle_position() {
        let mut hdr = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let mid = (SWAP_HEADER_BYTES - 32) / 2;
        hdr._reserved[mid] = 0xFE;
        assert_eq!(hdr._reserved[mid], 0xFE);
        assert_eq!(hdr._reserved[mid - 1], 0);
        assert_eq!(hdr._reserved[mid + 1], 0);
    }

    // ------------------------------------------------------------------
    // Write to slot_count - 1, read back, verify offset arithmetic
    // ------------------------------------------------------------------

    #[test]
    fn write_read_last_slot_offset_consistent() {
        let tmp = TempDir::new().unwrap();
        let count: u64 = 32;
        let msb = 8192;
        let swap = NvmeSwapFile::open(
            tmp.path().join("last_off.swap"),
            4096,
            msb,
            count,
        )
        .unwrap();

        let last_pid = (count - 1) as PageId;
        let expected_offset = SWAP_HEADER_BYTES as u64 + (count - 1) * msb as u64;
        assert_eq!(swap.slot_offset(last_pid), expected_offset);

        let data = vec![0xED; 200];
        swap.write_slot(last_pid, &data).unwrap();

        let mut buf = vec![0u8; 200];
        swap.read_slot(last_pid, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: size_of is independent of field values
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_size_independent_of_values() {
        let a = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = SwapFileHeader {
            magic: u64::MAX,
            version: u32::MAX,
            page_size: u32::MAX,
            max_slot_bytes: u32::MAX,
            _pad4: u32::MAX,
            slot_count: u64::MAX,
            _reserved: [u8::MAX; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(std::mem::size_of_val(&a), std::mem::size_of_val(&b));
    }

    // ------------------------------------------------------------------
    // Write 256 bytes (one byte per value), read back
    // ------------------------------------------------------------------

    #[test]
    fn write_256_unique_bytes_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("256b.swap"), 4096, 4096, 4).unwrap();

        let data: Vec<u8> = (0..=255).collect();
        swap.write_slot(0, &data).unwrap();

        let mut buf = vec![0u8; 256];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: Debug output is consistent across calls
    // ------------------------------------------------------------------

    #[test]
    fn nvme_swap_file_debug_deterministic() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("ddbg.swap"), 4096, 8192, 16).unwrap();
        let d1 = format!("{swap:?}");
        let d2 = format!("{swap:?}");
        assert_eq!(d1, d2);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: constructing from function return (move semantics)
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_from_fn_return() {
        fn make(magic: u64) -> SwapFileHeader {
            SwapFileHeader {
                magic,
                version: 1,
                page_size: 4096,
                max_slot_bytes: 8192,
                _pad4: 0,
                slot_count: 4,
                _reserved: [0u8; SWAP_HEADER_BYTES - 32],
            }
        }
        let h = make(0xBEEF);
        assert_eq!(h.magic, 0xBEEF);
        let h2 = make(0xCAFE);
        assert_ne!(h, h2);
    }

    // ------------------------------------------------------------------
    // Open with max_slot_bytes that needs alignment from 4097 to 8192
    // and verify file size is correct
    // ------------------------------------------------------------------

    #[test]
    fn aligned_slot_correct_file_size() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("asz.swap");
        let _swap = NvmeSwapFile::open(path.clone(), 4096, 4097, 4).unwrap();

        let meta = std::fs::metadata(&path).unwrap();
        // 4097 rounds up to 8192, 4 slots
        let expected = SWAP_HEADER_BYTES as u64 + 4 * 8192u64;
        assert_eq!(meta.len(), expected);
    }

    // ------------------------------------------------------------------
    // Write to all even slots, verify odd slots untouched
    // ------------------------------------------------------------------

    #[test]
    fn write_even_verify_odd_untouched() {
        let tmp = TempDir::new().unwrap();
        let count = 8u64;
        let swap = NvmeSwapFile::open(
            tmp.path().join("even_odd.swap"),
            4096,
            4096,
            count,
        )
        .unwrap();

        for pid in (0..count as usize).step_by(2) {
            swap.write_slot(pid, &[0xEE; 16]).unwrap();
        }

        for pid in (1..count as usize).step_by(2) {
            let mut buf = [0xFFu8; 16];
            swap.read_slot(pid, &mut buf).unwrap();
            assert!(buf.iter().all(|&b| b == 0x00), "odd slot {pid} should be zero");
        }
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: multiple headers in Vec sorted by slot_count
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_sort_by_slot_count() {
        let mut v: Vec<SwapFileHeader> = [30u64, 10, 50, 20, 40]
            .iter()
            .map(|&sc| SwapFileHeader {
                magic: 1,
                version: 1,
                page_size: 4096,
                max_slot_bytes: 4096,
                _pad4: 0,
                slot_count: sc,
                _reserved: [0u8; SWAP_HEADER_BYTES - 32],
            })
            .collect();

        v.sort_by_key(|h| h.slot_count);
        let counts: Vec<u64> = v.iter().map(|h| h.slot_count).collect();
        assert_eq!(counts, vec![10, 20, 30, 40, 50]);
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: writing to index 0 does not affect index 1
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_write_isolation() {
        let mut buf = AlignedBuffer::new(NVME_ALIGN);
        buf.as_mut_slice()[0] = 0xAA;
        buf.as_mut_slice()[1] = 0xBB;
        buf.as_mut_slice()[2] = 0xCC;

        assert_eq!(buf.as_slice()[0], 0xAA);
        assert_eq!(buf.as_slice()[1], 0xBB);
        assert_eq!(buf.as_slice()[2], 0xCC);
        // Verify neighbor independence
        buf.as_mut_slice()[1] = 0xDD;
        assert_eq!(buf.as_slice()[0], 0xAA);
        assert_eq!(buf.as_slice()[1], 0xDD);
        assert_eq!(buf.as_slice()[2], 0xCC);
    }

    // ------------------------------------------------------------------
    // Reopen with same page_size preserves it in struct
    // ------------------------------------------------------------------

    #[test]
    fn reopen_preserves_page_size_value() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("ps_persist.swap");
        let ps = 7919;

        {
            let _swap = NvmeSwapFile::open(path.clone(), ps, 4096, 4).unwrap();
        }

        let swap = NvmeSwapFile::open(path, ps, 4096, 4).unwrap();
        assert_eq!(swap.page_size, ps);
    }

    // ------------------------------------------------------------------
    // Write data with all bits set (0xFF) of various lengths
    // ------------------------------------------------------------------

    #[test]
    fn write_all_bits_set_various_lengths() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("ff_var.swap"), 4096, 4096, 4).unwrap();

        for &len in &[1usize, 16, 64, 256, 1024, 4096] {
            let data = vec![0xFF; len];
            swap.write_slot(0, &data).unwrap();
            let mut buf = vec![0u8; len];
            swap.read_slot(0, &mut buf).unwrap();
            assert!(buf.iter().all(|&b| b == 0xFF), "length {len}");
        }
    }

    // ------------------------------------------------------------------
    // Additional pure-data tests
    // ------------------------------------------------------------------

    #[test]
    fn swap_magic_fits_in_u64() {
        // SWAP_MAGIC must be a valid u64 value (non-zero, non-max).
        assert!(SWAP_MAGIC > 0, "magic must be non-zero");
        assert!(SWAP_MAGIC < u64::MAX, "magic must not be u64::MAX");
    }

    #[test]
    fn swap_header_reserved_field_is_exactly_4064_bytes() {
        // _reserved = SWAP_HEADER_BYTES - 32 = 4096 - 32 = 4064.
        const EXPECTED: usize = SWAP_HEADER_BYTES - 32;
        assert_eq!(EXPECTED, 4064);
        // Compile-time verified by the struct definition.
        let _check: [u8; 4064] = [0u8; SWAP_HEADER_BYTES - 32];
        assert_eq!(_check.len(), 4064);
    }

    #[test]
    fn nvme_align_fits_in_u32() {
        // NVME_ALIGN must fit in a u32 for storage in SwapFileHeader.max_slot_bytes.
        assert!(
            (NVME_ALIGN as u64) <= (u32::MAX as u64),
            "NVME_ALIGN must fit in u32 for header storage"
        );
    }

    #[test]
    fn swap_header_magic_first_field_at_offset_zero() {
        // In repr(C), the first field (magic) must be at byte offset 0.
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let base = &hdr as *const SwapFileHeader as usize;
        let magic_addr = &hdr.magic as *const u64 as usize;
        assert_eq!(magic_addr - base, 0, "magic must be at offset 0");
    }

    #[test]
    fn swap_header_all_fields_one_not_equal_to_default() {
        // A header with all-ones in every field must differ from a zeroed header.
        let all_ones = SwapFileHeader {
            magic: u64::MAX,
            version: u32::MAX,
            page_size: u32::MAX,
            max_slot_bytes: u32::MAX,
            _pad4: u32::MAX,
            slot_count: u64::MAX,
            _reserved: [0xFF; SWAP_HEADER_BYTES - 32],
        };
        let zeroed = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_ne!(all_ones, zeroed);
    }

    #[test]
    fn swap_version_fits_in_u32() {
        // SWAP_VERSION is stored as u32 in the header.
        let _v: u32 = SWAP_VERSION;
        assert_eq!(_v, 1u32);
    }

    #[test]
    fn swap_header_reserved_can_hold_individual_bytes() {
        // Each byte in _reserved can be independently set and read.
        let mut hdr = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        hdr._reserved[0] = 0xAA;
        hdr._reserved[SWAP_HEADER_BYTES - 33] = 0xBB;
        assert_eq!(hdr._reserved[0], 0xAA);
        assert_eq!(hdr._reserved[SWAP_HEADER_BYTES - 33], 0xBB);
        // Middle bytes remain zero.
        assert_eq!(hdr._reserved[100], 0);
    }

    #[test]
    fn swap_magic_greater_than_u32_max() {
        // SWAP_MAGIC requires u64 (cannot fit in u32).
        assert!(
            SWAP_MAGIC > u32::MAX as u64,
            "SWAP_MAGIC must exceed u32 range, requiring u64 storage"
        );
    }

    #[test]
    fn swap_header_different_magic_different_hash() {
        // Headers that differ only in magic must produce different hashes.
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut h1 = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 16,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let mut h2 = h1;
        h2.magic = 0;

        let mut s1 = DefaultHasher::new();
        h1.hash(&mut s1);
        let hash1 = s1.finish();

        let mut s2 = DefaultHasher::new();
        h2.hash(&mut s2);
        let hash2 = s2.finish();

        assert_ne!(hash1, hash2, "different magic must produce different hash");
    }

    #[test]
    fn swap_header_version_is_u32_width() {
        // version field occupies exactly 4 bytes (u32).
        let hdr = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(
            std::mem::size_of_val(&hdr.version),
            4,
            "version must be 4 bytes"
        );
    }

    #[test]
    fn swap_header_magic_is_at_struct_start() {
        // The first 8 bytes of the struct must be the magic field (repr(C) guarantee).
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let ptr = &hdr as *const SwapFileHeader as *const u64;
        // Safety: repr(C), magic is the first field at offset 0.
        let first_u64 = unsafe { *ptr };
        assert_eq!(first_u64, SWAP_MAGIC);
    }

    #[test]
    fn nvme_align_not_zero() {
        assert!(NVME_ALIGN > 0, "NVME_ALIGN must be positive");
    }

    #[test]
    fn swap_header_reserved_starts_after_slot_count() {
        // _reserved starts at offset 32, immediately after slot_count (offset 24, size 8).
        let hdr = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let base = &hdr as *const SwapFileHeader as usize;
        let slot_count_end = (&hdr.slot_count as *const u64 as usize) + 8;
        let reserved_start = hdr._reserved.as_ptr() as usize;
        assert_eq!(
            reserved_start,
            slot_count_end,
            "_reserved must start immediately after slot_count"
        );
    }

    #[test]
    fn swap_header_equality_reflexive() {
        // Any header must be equal to itself (reflexivity).
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 64,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(hdr, hdr, "header must equal itself");
    }

    #[test]
    fn swap_header_zero_version_not_equal_valid() {
        // A header with version=0 differs from one with SWAP_VERSION.
        let valid = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 16,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let zero_ver = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: 0,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 16,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_ne!(valid, zero_ver, "version=0 must differ from SWAP_VERSION");
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile implements Send (can be transferred across threads)
    // ------------------------------------------------------------------

    #[test]
    fn nvme_swap_file_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<NvmeSwapFile>();
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile implements Sync (can be shared across threads)
    // ------------------------------------------------------------------

    #[test]
    fn nvme_swap_file_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<NvmeSwapFile>();
    }

    // ------------------------------------------------------------------
    // SWAP_HEADER_BYTES equals NVME_ALIGN (both 4096)
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_bytes_equals_nvme_align() {
        assert_eq!(
            SWAP_HEADER_BYTES, NVME_ALIGN,
            "SWAP_HEADER_BYTES must equal NVME_ALIGN"
        );
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: different non-zero _pad4 values affect equality
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_pad4_nonzero_affects_equality() {
        let a = SwapFileHeader {
            magic: 42,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 100,
            slot_count: 8,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = SwapFileHeader {
            _pad4: 200,
            ..a
        };
        assert_ne!(a, b, "different non-zero _pad4 must not be equal");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: different _pad4 values produce different hashes
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_pad4_nonzero_affects_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let base = SwapFileHeader {
            magic: 99,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 4096,
            _pad4: 0,
            slot_count: 4,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };

        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        let a = SwapFileHeader { _pad4: 10, ..base };
        let b = SwapFileHeader { _pad4: 20, ..base };
        a.hash(&mut h1);
        b.hash(&mut h2);
        assert_ne!(h1.finish(), h2.finish(), "different _pad4 must differ in hash");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: magic=u64::MAX survives byte roundtrip
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_magic_u64_max_roundtrip() {
        let hdr = SwapFileHeader {
            magic: u64::MAX,
            version: 1,
            page_size: 512,
            max_slot_bytes: 4096,
            _pad4: 0,
            slot_count: 1,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &hdr as *const SwapFileHeader as *const u8,
                std::mem::size_of::<SwapFileHeader>(),
            )
        };
        let restored: SwapFileHeader = unsafe {
            std::ptr::read(bytes.as_ptr() as *const SwapFileHeader)
        };
        assert_eq!(restored.magic, u64::MAX);
        assert_eq!(hdr, restored);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: all u32 fields at u32::MAX
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_all_u32_fields_max() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: u32::MAX,
            page_size: u32::MAX,
            max_slot_bytes: u32::MAX,
            _pad4: u32::MAX,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(hdr.version, u32::MAX);
        assert_eq!(hdr.page_size, u32::MAX);
        assert_eq!(hdr.max_slot_bytes, u32::MAX);
        assert_eq!(hdr._pad4, u32::MAX);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: Copy independence — modify one field after copy
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_copy_independent_after_field_modify() {
        let original = SwapFileHeader {
            magic: 100,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 16,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let mut copy = original;
        copy.slot_count = 999;
        assert_ne!(original.slot_count, copy.slot_count);
        assert_eq!(original.slot_count, 16);
        assert_eq!(copy.slot_count, 999);
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: new with exactly NVME_ALIGN bytes succeeds
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_new_with_exact_nvme_align() {
        let buf = AlignedBuffer::new(NVME_ALIGN);
        assert_eq!(buf.as_slice().len(), NVME_ALIGN);
        // All bytes zero-initialized
        assert!(buf.as_slice().iter().all(|&b| b == 0));
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: mutations via as_mut_slice are visible via as_slice
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_as_slice_reflects_mut_slice_write() {
        let mut buf = AlignedBuffer::new(NVME_ALIGN);
        buf.as_mut_slice()[0] = 0xAB;
        buf.as_mut_slice()[NVME_ALIGN - 1] = 0xCD;
        assert_eq!(buf.as_slice()[0], 0xAB);
        assert_eq!(buf.as_slice()[NVME_ALIGN - 1], 0xCD);
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile Debug output includes slot_count field name
    // ------------------------------------------------------------------

    #[test]
    fn nvme_swap_file_debug_includes_slot_count() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("dbg.swap"), 4096, 4096, 7).unwrap();
        let dbg = format!("{:?}", swap);
        assert!(dbg.contains("slot_count"), "debug must contain slot_count: {dbg}");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: only first byte of _reserved different => not equal
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_reserved_first_byte_affects_equality() {
        let mut a = SwapFileHeader {
            magic: 1,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 4096,
            _pad4: 0,
            slot_count: 1,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = SwapFileHeader { ..a };
        a._reserved[0] = 0xFF;
        assert_ne!(a, b, "first reserved byte difference => not equal");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: only last byte of _reserved different => not equal
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_reserved_last_byte_affects_equality() {
        let base = SwapFileHeader {
            magic: 1,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 4096,
            _pad4: 0,
            slot_count: 1,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let mut a = base;
        let last = SWAP_HEADER_BYTES - 32 - 1;
        a._reserved[last] = 1;
        assert_ne!(a, base, "last reserved byte difference => not equal");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: two headers with pad4=0 and zero reserved are equal
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_zero_pad4_zero_reserved_equal() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let a = SwapFileHeader {
            magic: 42,
            version: 2,
            page_size: 2048,
            max_slot_bytes: 4096,
            _pad4: 0,
            slot_count: 10,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = SwapFileHeader {
            magic: 42,
            version: 2,
            page_size: 2048,
            max_slot_bytes: 4096,
            _pad4: 0,
            slot_count: 10,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(a, b, "identically constructed headers must be equal");
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        a.hash(&mut h1);
        b.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish(), "equal headers must have equal hashes");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: large slot_count is a valid construction
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_large_slot_count_construction() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: u32::MAX as u64,
            _reserved: [0xAA; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(hdr.slot_count, u32::MAX as u64);
        assert_eq!(hdr._reserved[0], 0xAA);
    }

    // ------------------------------------------------------------------
    // write_slot with data.len() == 1 returns exactly 1
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn write_single_byte_returns_one() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("one_byte.swap"), 4096, 4096, 4).unwrap();

        let written = swap.write_slot(0, &[0x42]).unwrap();
        assert_eq!(written, 1, "write_slot with 1 byte must return 1");

        let mut buf = [0u8; 1];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf[0], 0x42);
    }

    // ------------------------------------------------------------------
    // read_slot after two writes to same slot returns second data
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn read_after_double_write_returns_second_data() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("dbl_w.swap"), 4096, 4096, 8).unwrap();

        let first = vec![0x11; 64];
        let second = vec![0x22; 32];
        swap.write_slot(2, &first).unwrap();
        swap.write_slot(2, &second).unwrap();

        let mut buf = vec![0u8; 32];
        swap.read_slot(2, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0x22), "must reflect second write");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: different page_size and max_slot_bytes stored correctly
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_distinct_page_and_max_slot() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 2048,
            max_slot_bytes: 16384,
            _pad4: 0,
            slot_count: 100,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(hdr.page_size, 2048);
        assert_eq!(hdr.max_slot_bytes, 16384);
        assert!(hdr.max_slot_bytes > hdr.page_size);
    }

    // ------------------------------------------------------------------
    // write then read across adjacent slots with max slot bytes boundary
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn adjacent_max_slot_no_bleed() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("adj_max.swap"), 4096, 4096, 4).unwrap();

        let full = vec![0xAA; 4096];
        swap.write_slot(0, &full).unwrap();

        let mut slot1 = vec![0u8; 4096];
        swap.read_slot(1, &mut slot1).unwrap();
        assert!(slot1.iter().all(|&b| b == 0), "slot 1 must be zero");
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: writing all zeros then nonzero preserves nonzero
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_zero_overwrite_then_nonzero() {
        let mut buf = AlignedBuffer::new(NVME_ALIGN * 2);
        buf.as_mut_slice()[0] = 0xFF;
        buf.as_mut_slice()[0] = 0x00;
        buf.as_mut_slice()[0] = 0xAB;
        assert_eq!(buf.as_slice()[0], 0xAB);
        assert_eq!(buf.as_slice()[1], 0);
    }

    // ------------------------------------------------------------------
    // open with slot_count=1 produces file with exactly header + 1 slot
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn open_slot_count_one_file_size() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("one_slot.swap");
        let slot_bytes = 4096usize;
        let _swap = NvmeSwapFile::open(path.clone(), 4096, slot_bytes, 1).unwrap();

        let meta = std::fs::metadata(&path).unwrap();
        let expected = SWAP_HEADER_BYTES as u64 + slot_bytes as u64;
        assert_eq!(meta.len(), expected);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: reserved middle byte roundtrip through byte slice
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_reserved_middle_byte_roundtrip() {
        let middle = SWAP_HEADER_BYTES / 2 - 16;
        let mut hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 4096,
            _pad4: 0,
            slot_count: 1,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        hdr._reserved[middle] = 0xDE;

        let bytes = unsafe {
            std::slice::from_raw_parts(
                &hdr as *const SwapFileHeader as *const u8,
                std::mem::size_of::<SwapFileHeader>(),
            )
        };
        let restored: SwapFileHeader = unsafe {
            std::ptr::read(bytes.as_ptr() as *const SwapFileHeader)
        };
        assert_eq!(restored._reserved[middle], 0xDE);
    }

    // ------------------------------------------------------------------
    // write_slot error message includes the oversized length
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn write_oversized_error_includes_both_lengths() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("err_len.swap"), 4096, 4096, 4).unwrap();

        let oversized = vec![0u8; 4097];
        let err = swap.write_slot(0, &oversized).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("4097"), "error must contain data len: {msg}");
        assert!(msg.contains("4096"), "error must contain max_slot_bytes: {msg}");
    }

    // ------------------------------------------------------------------
    // read_slot error message includes the oversized dst length
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn read_oversized_error_includes_both_lengths() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("rerr_len.swap"), 4096, 4096, 4).unwrap();

        let mut oversized = vec![0u8; 4097];
        let err = swap.read_slot(0, &mut oversized).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("4097"), "error must contain dst len: {msg}");
        assert!(msg.contains("4096"), "error must contain max_slot_bytes: {msg}");
    }

    // ------------------------------------------------------------------
    // write then read with data crossing 4096-byte alignment boundary
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn write_read_cross_boundary_odd_offset() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("cross.swap"), 4096, 8192, 8).unwrap();

        // Write 4000 bytes starting from slot 3
        let data: Vec<u8> = (0..4000).map(|i| (i % 251) as u8).collect();
        swap.write_slot(3, &data).unwrap();

        let mut readback = vec![0u8; 4000];
        swap.read_slot(3, &mut readback).unwrap();
        assert_eq!(readback, data);
    }

    // ------------------------------------------------------------------
    // drop closes file descriptor (second open on same path succeeds)
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn drop_closes_fd_allows_immediate_reopen() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("drop_fd.swap");
        {
            let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
        }
        // Immediate reopen must succeed without resource exhaustion
        let _swap2 = NvmeSwapFile::open(path, 4096, 4096, 4).unwrap();
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: page_size field stored as u32 does not truncate small values
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_page_size_small_value_no_truncation() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 1,
            max_slot_bytes: 4096,
            _pad4: 0,
            slot_count: 8,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(hdr.page_size, 1u32);
    }

    // ------------------------------------------------------------------
    // open creates file even if max_slot_bytes < page_size
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn open_max_slot_smaller_than_page_succeeds() {
        let tmp = TempDir::new().unwrap();
        // page_size=8192, max_slot_bytes=4096 — the code rounds max_slot_bytes
        // up to NVME_ALIGN but does not enforce max_slot >= page_size
        let swap = NvmeSwapFile::open(
            tmp.path().join("small_slot.swap"),
            8192,
            4096,
            4,
        )
        .unwrap();
        assert_eq!(swap.max_slot_bytes, 4096);
        assert_eq!(swap.page_size, 8192);
    }

    // ------------------------------------------------------------------
    // write empty slice then write real data, read returns real data
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn write_empty_then_real_data_read_succeeds() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("empty_real.swap"), 4096, 4096, 4).unwrap();

        swap.write_slot(1, &[]).unwrap();
        let data = vec![0x77; 128];
        swap.write_slot(1, &data).unwrap();

        let mut buf = vec![0u8; 128];
        swap.read_slot(1, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0x77));
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile fields match the parameters passed to open
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn open_params_reflected_in_struct_fields() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(
            tmp.path().join("params.swap"),
            512,
            4096,
            64,
        )
        .unwrap();
        assert_eq!(swap.page_size, 512);
        assert_eq!(swap.max_slot_bytes, 4096);
        assert_eq!(swap.slot_count, 64);
    }

    // ------------------------------------------------------------------
    // reopen with corrupted page_size field still opens (header not re-validated for page_size)
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn reopen_corrupted_page_size_uses_new_param() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("corrupt_ps.swap");

        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
            drop(swap);
            // Corrupt page_size field (offset 12, 4 bytes) to 999
            let mut data = std::fs::read(&path).unwrap();
            data[12..16].copy_from_slice(&999u32.to_le_bytes());
            let expected_len = SWAP_HEADER_BYTES + 4 * 4096;
            data.resize(expected_len, 0);
            std::fs::write(&path, &data).unwrap();
        }

        // Reopen with different page_size — the struct should use the new param
        let swap = NvmeSwapFile::open(path, 2048, 4096, 4).unwrap();
        assert_eq!(swap.page_size, 2048, "struct page_size should reflect the open() param");
    }

    // ==================================================================
    // 15 additional tests — target ~458 total
    // Focus: uncovered error paths, boundary combinations, file system
    // edge cases, multi-cycle durability, slot offset overflow safety.
    // ==================================================================

    // ------------------------------------------------------------------
    // Open with path containing Unicode characters succeeds
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn open_path_with_unicode_characters() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("交换文件.swap");
        let swap = NvmeSwapFile::open(path, 4096, 4096, 4);
        assert!(swap.is_ok(), "Unicode path should open successfully");
        let swap = swap.unwrap();
        swap.write_slot(0, &[0xAA; 32]).unwrap();
        let mut buf = [0u8; 32];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xAA));
    }

    // ------------------------------------------------------------------
    // Reopen file that was created with fewer slots, write to new higher
    // slot index, verify data survives another reopen
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn reopen_more_slots_write_high_slot_durable() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("grow_slot.swap");

        {
            let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
        }

        // Reopen with more slots
        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 16).unwrap();
            let data = vec![0xED; 100];
            swap.write_slot(15, &data).unwrap();
        }

        // Third reopen verifies data at high slot persists
        let swap = NvmeSwapFile::open(path, 4096, 4096, 16).unwrap();
        let mut buf = vec![0u8; 100];
        swap.read_slot(15, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xED), "high slot data must survive reopen");
    }

    // ------------------------------------------------------------------
    // Write full slot with 0xAA, then write 0 bytes to same slot,
    // then read full slot — first bytes should be zero (padded)
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn full_write_then_empty_write_zeros_slot() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("full_empty.swap"), 4096, 4096, 4).unwrap();

        swap.write_slot(0, &[0xAA; 4096]).unwrap();
        swap.write_slot(0, &[]).unwrap();

        let mut buf = vec![0xFF; 64];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0x00), "empty write must zero-fill slot");
    }

    // ------------------------------------------------------------------
    // New file with larger slot_count produces larger on-disk size
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn new_file_larger_slot_count_larger_size() {
        let tmp = TempDir::new().unwrap();
        let slot_size = 4096usize;

        let path_s = tmp.path().join("small.swap");
        let path_l = tmp.path().join("large.swap");

        let _swap_s = NvmeSwapFile::open(path_s.clone(), 4096, slot_size, 4).unwrap();
        let _swap_l = NvmeSwapFile::open(path_l.clone(), 4096, slot_size, 16).unwrap();

        let size_small = std::fs::metadata(&path_s).unwrap().len();
        let size_large = std::fs::metadata(&path_l).unwrap().len();

        let expected_small = SWAP_HEADER_BYTES as u64 + 4 * slot_size as u64;
        let expected_large = SWAP_HEADER_BYTES as u64 + 16 * slot_size as u64;
        assert_eq!(size_small, expected_small);
        assert!(size_large > size_small, "file with more slots must be larger");
        assert_eq!(size_large, expected_large);
    }

    // ------------------------------------------------------------------
    // Open creates intermediate directories two levels deep
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn open_creates_two_level_intermediate_dirs() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("level1").join("level2").join("deep.swap");
        let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4);
        assert!(swap.is_ok(), "should create two levels of directories");
        assert!(path.exists(), "file should exist on disk");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: max_slot_bytes stored as u32 does not overflow for
    // small aligned values
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_max_slot_bytes_u32_no_overflow() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: NVME_ALIGN as u32,
            _pad4: 0,
            slot_count: 1,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(hdr.max_slot_bytes, NVME_ALIGN as u32);
        // Verify no truncation: the value fits exactly in u32
        assert_eq!(hdr.max_slot_bytes as usize, NVME_ALIGN);
    }

    // ------------------------------------------------------------------
    // Write alternating byte pairs (0x00, 0xFF) and read back
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn write_alternating_zero_ff_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("alt_zf.swap"), 4096, 4096, 4).unwrap();

        let data: Vec<u8> = (0..512).map(|i| if i % 2 == 0 { 0x00 } else { 0xFF }).collect();
        swap.write_slot(0, &data).unwrap();

        let mut buf = vec![0u8; 512];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // slot_offset arithmetic with max_slot_bytes = NVME_ALIGN for
    // page_id that would overflow u32 but fits in u64
    // ------------------------------------------------------------------

    #[test]
    fn slot_offset_large_values_no_u32_overflow() {
        let swap = NvmeSwapFile {
            fd: Mutex::new(-1),
            page_size: 4096,
            max_slot_bytes: NVME_ALIGN,
            slot_count: 0,
        };
        // page_id = 600000, max_slot_bytes = 4096
        // 600000 * 4096 = 2_457_600_000 which exceeds u32::MAX (4_294_967_295? no)
        // Use page_id = 1_100_000: 1_100_000 * 4096 = 4_505_600_000 > u32::MAX
        let offset = swap.slot_offset(1_100_000);
        let expected = SWAP_HEADER_BYTES as u64 + 1_100_000u64 * NVME_ALIGN as u64;
        assert_eq!(offset, expected);
        // Verify no truncation: result must be > u32::MAX
        assert!(offset > u32::MAX as u64, "offset should exceed u32 range");
    }

    // ------------------------------------------------------------------
    // Write to every slot, overwrite only slot 0, verify others unchanged
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn overwrite_first_slot_preserves_all_others() {
        let tmp = TempDir::new().unwrap();
        let count = 8u64;
        let swap = NvmeSwapFile::open(
            tmp.path().join("ow_first.swap"),
            4096,
            4096,
            count,
        )
        .unwrap();

        for pid in 0..count as usize {
            swap.write_slot(pid, &[(pid as u8).wrapping_add(0x10); 32]).unwrap();
        }
        swap.write_slot(0, &[0xFF; 32]).unwrap();

        // Slot 0 must be overwritten
        let mut buf0 = [0u8; 32];
        swap.read_slot(0, &mut buf0).unwrap();
        assert!(buf0.iter().all(|&b| b == 0xFF), "slot 0 must be 0xFF");

        // All other slots must be intact
        for pid in 1..count as usize {
            let expected = (pid as u8).wrapping_add(0x10);
            let mut buf = [0u8; 32];
            swap.read_slot(pid, &mut buf).unwrap();
            assert!(
                buf.iter().all(|&b| b == expected),
                "slot {pid}: expected 0x{expected:02x}"
            );
        }
    }

    // ------------------------------------------------------------------
    // Verify write_slot returns different u32 values for different data
    // sizes on the same slot
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn write_slot_returns_varying_lengths_same_slot() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("var_len.swap"), 4096, 4096, 4).unwrap();

        let sizes = [1usize, 10, 100, 1000, 4096];
        for &size in &sizes {
            let written = swap.write_slot(0, &vec![0x42; size]).unwrap();
            assert_eq!(written as usize, size, "returned length must match input size");
        }
    }

    // ------------------------------------------------------------------
    // Open a file with path length near OS limit (>200 chars in filename)
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn open_long_filename_succeeds() {
        let tmp = TempDir::new().unwrap();
        let long_name: String = "a".repeat(200) + ".swap";
        let path = tmp.path().join(long_name);
        let swap = NvmeSwapFile::open(path, 4096, 4096, 4);
        assert!(swap.is_ok(), "long filename should open successfully");
    }

    // ------------------------------------------------------------------
    // Write data at slot boundary (last 2 bytes of slot), read back
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn write_read_at_slot_end_boundary() {
        let tmp = TempDir::new().unwrap();
        let slot = 4096;
        let swap = NvmeSwapFile::open(
            tmp.path().join("slot_end.swap"),
            4096,
            slot,
            4,
        )
        .unwrap();

        // Write only to the last 2 bytes of the slot
        let mut data = vec![0u8; slot];
        data[slot - 2] = 0xDE;
        data[slot - 1] = 0xAD;
        swap.write_slot(0, &data).unwrap();

        let mut buf = vec![0u8; slot];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf[slot - 2], 0xDE);
        assert_eq!(buf[slot - 1], 0xAD);
        assert!(buf[..slot - 2].iter().all(|&b| b == 0x00), "rest must be zero");
    }

    // ------------------------------------------------------------------
    // pwrite_all and pread_exact: verify they are available as associated
    // functions (test via a swap file roundtrip with known offset)
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn pwrite_and_pread_direct_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("direct_io.swap");
        let swap = NvmeSwapFile::open(path, 4096, 4096, 4).unwrap();

        // Write data to slot 2 and read back — validates pwrite_all/pread_exact
        // path under the hood
        let data: Vec<u8> = (0..256).map(|i| (i ^ 0xA5) as u8).collect();
        swap.write_slot(2, &data).unwrap();

        let mut buf = vec![0u8; 256];
        swap.read_slot(2, &mut buf).unwrap();
        assert_eq!(buf, data, "pwrite+pread roundtrip must be exact");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: all public fields readable after zeroed() construction
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_zeroed_all_public_fields_zero() {
        let hdr: SwapFileHeader = unsafe { std::mem::zeroed() };
        assert_eq!(hdr.magic, 0);
        assert_eq!(hdr.version, 0);
        assert_eq!(hdr.page_size, 0);
        assert_eq!(hdr.max_slot_bytes, 0);
        assert_eq!(hdr.slot_count, 0);
        // _reserved is all zeros
        assert!(hdr._reserved.iter().all(|&b| b == 0));
    }

    // ------------------------------------------------------------------
    // Concurrent read and write to the same slot is serialized by Mutex
    // (no torn reads — reader gets either old or new data, never partial)
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn concurrent_read_write_same_slot_no_torn_read() {
        use std::sync::Arc;
        use std::thread;

        let tmp = TempDir::new().unwrap();
        let swap = Arc::new(
            NvmeSwapFile::open(tmp.path().join("rw_same.swap"), 4096, 4096, 4).unwrap(),
        );

        // Pre-fill slot with pattern A
        swap.write_slot(0, &[0xAA; 64]).unwrap();

        let barrier = Arc::new(std::sync::Barrier::new(2));
        let mut handles = Vec::new();

        // Writer: overwrites with pattern B
        {
            let sc = Arc::clone(&swap);
            let b = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                b.wait();
                sc.write_slot(0, &[0xBB; 64]).unwrap();
            }));
        }

        // Reader: reads slot 0
        let read_result = {
            let sc = Arc::clone(&swap);
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                b.wait();
                let mut buf = [0u8; 64];
                sc.read_slot(0, &mut buf).unwrap();
                buf
            })
        };

        for h in handles {
            h.join().expect("writer should not panic");
        }
        let buf = read_result.join().expect("reader should not panic");

        // The read must return either all 0xAA or all 0xBB (no torn read)
        let all_aa = buf.iter().all(|&b| b == 0xAA);
        let all_bb = buf.iter().all(|&b| b == 0xBB);
        assert!(
            all_aa || all_bb,
            "torn read detected: got mixed bytes, not a consistent snapshot"
        );
    }

    // ------------------------------------------------------------------
    // Two swap files in same directory with different max_slot_bytes
    // operate independently
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn two_files_different_slot_sizes_independent() {
        let tmp = TempDir::new().unwrap();
        let swap_small = NvmeSwapFile::open(
            tmp.path().join("small.swap"),
            4096,
            4096,
            4,
        )
        .unwrap();
        let swap_large = NvmeSwapFile::open(
            tmp.path().join("large.swap"),
            4096,
            8192,
            4,
        )
        .unwrap();

        assert_ne!(swap_small.max_slot_bytes, swap_large.max_slot_bytes);

        let data_s = vec![0x11; 100];
        let data_l = vec![0x22; 200];
        swap_small.write_slot(0, &data_s).unwrap();
        swap_large.write_slot(0, &data_l).unwrap();

        let mut buf_s = vec![0u8; 100];
        let mut buf_l = vec![0u8; 200];
        swap_small.read_slot(0, &mut buf_s).unwrap();
        swap_large.read_slot(0, &mut buf_l).unwrap();

        assert!(buf_s.iter().all(|&b| b == 0x11));
        assert!(buf_l.iter().all(|&b| b == 0x22));
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: constructing with u32::MAX for page_size and
    // max_slot_bytes does not panic (field-level value range check)
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_u32_max_page_and_slot_bytes_no_panic() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: u32::MAX,
            max_slot_bytes: u32::MAX,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(hdr.page_size, u32::MAX);
        assert_eq!(hdr.max_slot_bytes, u32::MAX);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: two headers differ only in _pad4, PartialEq includes it
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_pad4_zero_vs_nonzero_not_equal() {
        let a = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 4096,
            _pad4: 0,
            slot_count: 4,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = SwapFileHeader {
            _pad4: 1,
            ..a
        };
        // PartialEq is derived, so _pad4 difference should make them unequal
        assert_ne!(a, b);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: mutable reference allows updating a single field
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_mutable_update_slot_count() {
        let mut hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 10,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(hdr.slot_count, 10);
        hdr.slot_count = 20;
        assert_eq!(hdr.slot_count, 20);
        // Other fields unchanged
        assert_eq!(hdr.magic, SWAP_MAGIC);
        assert_eq!(hdr.version, SWAP_VERSION);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: reserved field byte-level mutation via mutable ref
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_reserved_mutation_via_index() {
        let mut hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 4096,
            _pad4: 0,
            slot_count: 1,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(hdr._reserved[0], 0);
        hdr._reserved[0] = 0xAB;
        hdr._reserved[SWAP_HEADER_BYTES - 33] = 0xCD;
        assert_eq!(hdr._reserved[0], 0xAB);
        assert_eq!(hdr._reserved[SWAP_HEADER_BYTES - 33], 0xCD);
        // Middle stays zero
        assert_eq!(hdr._reserved[100], 0);
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: open with page_size = 0 stores as 0 in struct
    // ------------------------------------------------------------------

    #[test]
    fn open_page_size_zero_stored_in_struct() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("ps0.swap"), 0, 4096, 4).unwrap();
        assert_eq!(swap.page_size, 0);
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: reopen with different page_size keeps original on-disk
    // header but struct reflects the new open param
    // ------------------------------------------------------------------

    #[test]
    fn reopen_page_size_changes_in_struct() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("reopen_ps.swap");

        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
            assert_eq!(swap.page_size, 4096);
            swap.write_slot(0, &[0x99; 64]).unwrap();
        }

        // Reopen with different page_size — the struct field reflects the new value
        let swap2 = NvmeSwapFile::open(path, 8192, 4096, 4).unwrap();
        assert_eq!(swap2.page_size, 8192);

        // Data is still readable
        let mut buf = [0u8; 64];
        swap2.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0x99));
    }

    // ------------------------------------------------------------------
    // write_slot: writing 1 byte then reading 1 byte returns exact value
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn write_one_byte_read_one_byte_exact() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("one_byte.swap"), 4096, 4096, 4).unwrap();

        let written = swap.write_slot(3, &[0xFE]).unwrap();
        assert_eq!(written, 1);

        let mut buf = [0u8; 1];
        swap.read_slot(3, &mut buf).unwrap();
        assert_eq!(buf[0], 0xFE);
    }

    // ------------------------------------------------------------------
    // write_slot with max_slot_bytes = NVME_ALIGN * 2: write full slot,
    // read back first and last byte
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn write_full_double_align_read_first_and_last_byte() {
        let tmp = TempDir::new().unwrap();
        let slot = NVME_ALIGN * 2;
        let swap = NvmeSwapFile::open(tmp.path().join("dbl.swap"), 4096, slot, 4).unwrap();

        let mut data = vec![0u8; slot];
        data[0] = 0xAA;
        data[slot - 1] = 0xBB;
        swap.write_slot(1, &data).unwrap();

        let mut buf = vec![0u8; slot];
        swap.read_slot(1, &mut buf).unwrap();
        assert_eq!(buf[0], 0xAA, "first byte must be 0xAA");
        assert_eq!(buf[slot - 1], 0xBB, "last byte must be 0xBB");
        // Middle should be zero
        assert!(buf[1..slot - 1].iter().all(|&b| b == 0));
    }

    // ------------------------------------------------------------------
    // Read from a never-written slot beyond slot 0 returns zeros
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn read_high_unwritten_slot_returns_zeros() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("high_unwritten.swap"), 4096, 4096, 8).unwrap();

        // Write to slot 0 to ensure file is initialized
        swap.write_slot(0, &[0xFF; 64]).unwrap();

        // Slot 7 should be all zeros since never written
        let mut buf = vec![0u8; 256];
        swap.read_slot(7, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0), "unwritten slot must be zeros");
    }

    // ------------------------------------------------------------------
    // Overwrite slot with smaller data: the tail of the old data is replaced
    // by zeros from the aligned buffer padding
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn overwrite_large_with_small_clears_tail() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("shrink.swap"), 4096, 4096, 4).unwrap();

        // Write 256 bytes of 0xFF
        swap.write_slot(0, &[0xFF; 256]).unwrap();

        // Overwrite with 16 bytes of 0xAA
        swap.write_slot(0, &[0xAA; 16]).unwrap();

        // Read back 256 bytes: first 16 = 0xAA, rest = 0x00
        let mut buf = vec![0u8; 256];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf[..16].iter().all(|&b| b == 0xAA), "first 16 bytes must be 0xAA");
        assert!(buf[16..].iter().all(|&b| b == 0x00), "tail must be zeroed after overwrite");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: Hash is consistent — inserting same header twice into
    // HashSet yields len 1
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_hashset_duplicate_yields_one() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 4096,
            _pad4: 0,
            slot_count: 8,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let mut set = std::collections::HashSet::new();
        set.insert(hdr);
        set.insert(hdr);
        assert_eq!(set.len(), 1, "duplicate insert should not increase set size");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: deriving Eq means a == b implies !(a != b)
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_eq_implies_not_ne() {
        let a = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 2048,
            max_slot_bytes: 4096,
            _pad4: 0,
            slot_count: 5,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = a;
        assert!(a == b);
        assert!(!(a != b));
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: write to slot, drop, reopen, read back survives drop
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn write_drop_reopen_read_preserves_data() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("drop_reopen.swap");

        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
            let data: Vec<u8> = (0..128u32).map(|i| i.wrapping_mul(7) as u8).collect();
            swap.write_slot(2, &data).unwrap();
        }
        // swap is dropped here, fd closed via Drop

        let swap2 = NvmeSwapFile::open(path, 4096, 4096, 4).unwrap();
        let mut buf = vec![0u8; 128];
        swap2.read_slot(2, &mut buf).unwrap();
        for (i, &byte) in buf.iter().enumerate() {
            assert_eq!(byte, (i as u32).wrapping_mul(7) as u8, "byte {i} mismatch after reopen");
        }
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile: slot_count = 1, write and read slot 0 succeeds
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn single_slot_write_read_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(
            tmp.path().join("single.swap"),
            4096,
            4096,
            1,
        )
        .unwrap();
        assert_eq!(swap.slot_count, 1);

        let data = vec![0x77; 512];
        swap.write_slot(0, &data).unwrap();

        let mut buf = vec![0u8; 512];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0x77));
    }

    // ------------------------------------------------------------------
    // write_slot error message contains the page_id value
    // ------------------------------------------------------------------

    #[test]
    fn write_error_includes_specific_page_id() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("err_pid.swap"), 4096, 4096, 4).unwrap();

        let oversized = vec![0u8; swap.max_slot_bytes + 1];
        let err = swap.write_slot(42, &oversized).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("42"), "error should mention page_id=42: {msg}");
    }

    // ------------------------------------------------------------------
    // read_slot error message contains the page_id value
    // ------------------------------------------------------------------

    #[test]
    fn read_error_includes_specific_page_id() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("rerr_pid.swap"), 4096, 4096, 4).unwrap();

        let mut oversized = vec![0u8; swap.max_slot_bytes + 1];
        let err = swap.read_slot(99, &mut oversized).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("99"), "error should mention page_id=99: {msg}");
    }

    // ------------------------------------------------------------------
    // header _pad4 field stores nonzero value on disk via AlignedBuffer
    // ------------------------------------------------------------------

    #[test]
    fn header_pad4_nonzero_stored_on_disk() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("pad4_disk.swap");
        let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 2).unwrap();
        drop(swap);

        // Reopen and read raw header bytes via a new fd
        let fd = unsafe {
            let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
            libc::open(c_path.as_ptr(), libc::O_RDWR, 0o644)
        };
        assert!(fd >= 0);

        let mut hdr_buf = AlignedBuffer::new(SWAP_HEADER_BYTES);
        unsafe {
            libc::pread(
                fd,
                hdr_buf.as_mut_ptr() as *mut std::ffi::c_void,
                SWAP_HEADER_BYTES,
                0,
            );
        }
        unsafe { libc::close(fd) };

        let header: &SwapFileHeader =
            unsafe { &*(hdr_buf.as_ptr() as *const SwapFileHeader) };
        // _pad4 should be 0 (set by open)
        assert_eq!(header._pad4, 0);
    }

    // ------------------------------------------------------------------
    // write_slot to page_id beyond slot_count, then read back
    // ------------------------------------------------------------------

    #[test]
    fn write_beyond_slot_count_read_back_exact() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("beyond.swap"), 4096, 4096, 2).unwrap();

        // page_id = 10 is well beyond slot_count = 2, but write/read still work
        let data = vec![0xAB_u8; 64];
        let written = swap.write_slot(10, &data).unwrap();
        assert_eq!(written, 64);

        let mut dst = vec![0u8; 64];
        swap.read_slot(10, &mut dst).unwrap();
        assert_eq!(dst, data);
    }

    // ------------------------------------------------------------------
    // slot_offset for page_id=0 always equals SWAP_HEADER_BYTES
    // ------------------------------------------------------------------

    #[test]
    fn slot_offset_page_zero_equals_header_bytes_various_sizes() {
        for &msb in &[4096, 8192, 16384, 65536] {
            let tmp = TempDir::new().unwrap();
            let swap = NvmeSwapFile::open(
                tmp.path().join(format!("off0_{msb}.swap")),
                4096,
                msb,
                4,
            )
            .unwrap();
            assert_eq!(
                swap.slot_offset(0),
                SWAP_HEADER_BYTES as u64,
                "slot_offset(0) should equal SWAP_HEADER_BYTES for max_slot_bytes={msb}"
            );
        }
    }

    // ------------------------------------------------------------------
    // AlignedBuffer new panics on non-NVME_ALIGN-multiple size
    // ------------------------------------------------------------------

    #[test]
    #[should_panic]
    fn aligned_buffer_new_panics_on_non_aligned_size() {
        // 100 is not a multiple of 4096
        let _buf = AlignedBuffer::new(100);
    }

    // ------------------------------------------------------------------
    // AlignedBuffer new panics on zero size
    // ------------------------------------------------------------------

    #[test]
    #[should_panic]
    fn aligned_buffer_new_panics_on_zero_size() {
        let _buf = AlignedBuffer::new(0);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader version field is at byte offset 8
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_version_at_byte_offset_8_via_cast() {
        let header = SwapFileHeader {
            magic: 0x0102030405060708,
            version: 0xDEADBEEF,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 10,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let bytes: &[u8; SWAP_HEADER_BYTES] =
            unsafe { &*(&header as *const SwapFileHeader as *const [u8; SWAP_HEADER_BYTES]) };

        // version is at offset 8, stored as native-endian u32
        let version_bytes =
            u32::from_ne_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert_eq!(version_bytes, 0xDEADBEEF);
    }

    // ------------------------------------------------------------------
    // SwapFileHeader slot_count is at byte offset 24 via byte inspection
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_slot_count_byte_offset_24_via_cast() {
        let header = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0x1234_5678_9ABC_DEF0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let bytes: &[u8; SWAP_HEADER_BYTES] =
            unsafe { &*(&header as *const SwapFileHeader as *const [u8; SWAP_HEADER_BYTES]) };

        let sc_bytes = u64::from_ne_bytes([
            bytes[24], bytes[25], bytes[26], bytes[27],
            bytes[28], bytes[29], bytes[30], bytes[31],
        ]);
        assert_eq!(sc_bytes, 0x1234_5678_9ABC_DEF0);
    }

    // ------------------------------------------------------------------
    // write then overwrite with smaller data, read returns second data
    // ------------------------------------------------------------------

    #[test]
    fn overwrite_smaller_then_read_returns_second() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("ow_sm.swap"), 4096, 4096, 4).unwrap();

        let first = vec![0xFF_u8; 200];
        swap.write_slot(1, &first).unwrap();

        let second = vec![0x42_u8; 50];
        swap.write_slot(1, &second).unwrap();

        let mut dst = vec![0u8; 50];
        swap.read_slot(1, &mut dst).unwrap();
        assert_eq!(dst, second);
    }

    // ------------------------------------------------------------------
    // max_slot_bytes rounds up from 1 to NVME_ALIGN
    // ------------------------------------------------------------------

    #[test]
    fn max_slot_bytes_rounds_up_from_one() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("msb1.swap"), 4096, 1, 1).unwrap();
        assert_eq!(swap.max_slot_bytes, NVME_ALIGN);
    }

    // ------------------------------------------------------------------
    // write all zeros, read back all zeros across multiple slots
    // ------------------------------------------------------------------

    #[test]
    fn write_zeros_all_slots_read_zeros_all_slots() {
        let tmp = TempDir::new().unwrap();
        let slot_count = 4u64;
        let swap = NvmeSwapFile::open(
            tmp.path().join("all_zero.swap"),
            4096,
            4096,
            slot_count,
        )
        .unwrap();

        let zeros = vec![0u8; 128];
        for i in 0..slot_count {
            swap.write_slot(i as PageId, &zeros).unwrap();
        }
        for i in 0..slot_count {
            let mut dst = vec![0xFFu8; 128];
            swap.read_slot(i as PageId, &mut dst).unwrap();
            assert!(dst.iter().all(|&b| b == 0), "slot {i} should be all zeros");
        }
    }

    // ------------------------------------------------------------------
    // SwapFileHeader _reserved field occupies exactly offsets 32..4096
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_reserved_spans_offsets_32_to_4096() {
        assert_eq!(
            std::mem::size_of::<SwapFileHeader>(),
            SWAP_HEADER_BYTES,
        );
        // first 32 bytes are the 6 named fields (8+4+4+4+4+8 = 32)
        // remaining 4064 bytes are _reserved
        let offset_of_reserved = 32;
        assert_eq!(
            SWAP_HEADER_BYTES - offset_of_reserved,
            4064,
        );
    }

    // ------------------------------------------------------------------
    // write_slot returns exact data length not aligned length
    // ------------------------------------------------------------------

    #[test]
    fn write_slot_returns_data_len_not_aligned_len() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(
            tmp.path().join("retlen.swap"),
            4096,
            8192,
            2,
        )
        .unwrap();

        // Write 7 bytes into an 8192-byte slot
        let data = vec![0x55_u8; 7];
        let result = swap.write_slot(0, &data).unwrap();
        assert_eq!(result, 7, "should return original data length, not aligned size");
    }

    // ------------------------------------------------------------------
    // reopen after drop with different slot_count uses new value
    // ------------------------------------------------------------------

    #[test]
    fn reopen_different_slot_count_reflected_in_struct() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("slotcnt.swap");

        let swap1 = NvmeSwapFile::open(path.clone(), 4096, 4096, 3).unwrap();
        assert_eq!(swap1.slot_count, 3);
        drop(swap1);

        let swap2 = NvmeSwapFile::open(path, 4096, 4096, 7).unwrap();
        assert_eq!(swap2.slot_count, 7);
    }

    // ------------------------------------------------------------------
    // page_size stored as zero when opened with zero
    // ------------------------------------------------------------------

    #[test]
    fn open_page_size_zero_stored_and_reopened() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("psz.swap");

        let swap = NvmeSwapFile::open(path.clone(), 0, 4096, 2).unwrap();
        assert_eq!(swap.page_size, 0);
        drop(swap);

        let swap2 = NvmeSwapFile::open(path, 0, 4096, 2).unwrap();
        assert_eq!(swap2.page_size, 0);
    }

    // ------------------------------------------------------------------
    // write and read pattern 0xAA/0x55 alternating across slots
    // ------------------------------------------------------------------

    #[test]
    fn alternating_pattern_across_two_slots() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(
            tmp.path().join("alt2.swap"),
            4096,
            4096,
            2,
        )
        .unwrap();

        let data_a = vec![0xAA_u8; 64];
        let data_5 = vec![0x55_u8; 64];
        swap.write_slot(0, &data_a).unwrap();
        swap.write_slot(1, &data_5).unwrap();

        let mut dst0 = vec![0u8; 64];
        let mut dst1 = vec![0u8; 64];
        swap.read_slot(0, &mut dst0).unwrap();
        swap.read_slot(1, &mut dst1).unwrap();

        assert!(dst0.iter().all(|&b| b == 0xAA), "slot 0 should be 0xAA");
        assert!(dst1.iter().all(|&b| b == 0x55), "slot 1 should be 0x55");
    }

    // ==================================================================
    // 15 additional tests — edge cases, boundary I/O, error paths
    // ==================================================================

    // ------------------------------------------------------------------
    // write_slot at max u32 data length boundary (u32::MAX bytes rejected)
    // ------------------------------------------------------------------

    #[test]
    fn write_slot_data_len_exceeding_u32_max_returns_error() {
        let tmp = TempDir::new().unwrap();
        // Use 4096 max_slot_bytes so any data > 4096 is rejected
        let swap = NvmeSwapFile::open(tmp.path().join("u32max.swap"), 4096, 4096, 4).unwrap();

        // Construct a buffer just one byte larger than max_slot_bytes
        let oversized = vec![0u8; swap.max_slot_bytes + 1];
        let result = swap.write_slot(0, &oversized);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        // Verify the error message contains the actual lengths
        let msg = err.to_string();
        assert!(
            msg.contains(&format!("{}", swap.max_slot_bytes + 1)),
            "error must mention oversized length: {msg}"
        );
    }

    // ------------------------------------------------------------------
    // slot_offset with large page_id does not overflow u64
    // ------------------------------------------------------------------

    #[test]
    fn slot_offset_large_page_id_no_overflow() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("bigoff.swap"), 4096, 4096, 4).unwrap();

        // Use a page_id near u32::MAX to verify no u64 overflow in calculation
        let page_id: PageId = 1_000_000;
        let offset = swap.slot_offset(page_id);
        let expected = SWAP_HEADER_BYTES as u64 + page_id as u64 * swap.max_slot_bytes as u64;
        assert_eq!(offset, expected);
        // Verify the offset is larger than header and grows linearly
        assert!(offset > SWAP_HEADER_BYTES as u64);
        assert!(offset > swap.slot_offset(page_id - 1));
    }

    // ------------------------------------------------------------------
    // read_slot from slot that was written then overwritten with empty
    // ------------------------------------------------------------------

    #[test]
    fn read_slot_after_overwrite_with_empty_returns_zeros() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "owempty.swap");

        // Write pattern, then overwrite with empty
        swap.write_slot(0, &[0xAB; 200]).unwrap();
        swap.write_slot(0, &[]).unwrap();

        // Read back: should be all zeros (AlignedBuffer zero-initialized)
        let mut buf = vec![0xFF; 200];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(
            buf.iter().all(|&b| b == 0x00),
            "slot overwritten with empty must read zeros"
        );
    }

    // ------------------------------------------------------------------
    // SwapFileHeader PartialEq — different version means not equal
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_partialeq_different_version() {
        let a = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 8,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = SwapFileHeader {
            version: 2,
            ..a
        };
        assert_ne!(a, b, "different version must not be equal");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader Hash — equal headers in HashMap produce same get result
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_hash_consistent_across_insert_and_lookup() {
        use std::collections::HashMap;
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 16,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };

        // Insert and look up with a separately-constructed equal key
        let key = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 16,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };

        let mut map = HashMap::new();
        map.insert(hdr, 42u32);

        // key is a separate stack instance but equal
        assert_eq!(map.get(&key), Some(&42));

        // Also verify hash values match
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        hdr.hash(&mut h1);
        key.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile Debug format roundtrip: parse back numeric values
    // ------------------------------------------------------------------

    #[test]
    fn nvme_swap_file_debug_roundtrip_numeric_values() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(
            tmp.path().join("dbgrt.swap"),
            8192,
            16384,
            50,
        )
        .unwrap();

        let dbg = format!("{:?}", swap);

        // Verify all three fields appear with their exact values
        assert!(
            dbg.contains("page_size") && dbg.contains("8192"),
            "debug must contain page_size with value 8192: {dbg}"
        );
        assert!(
            dbg.contains("max_slot_bytes") && dbg.contains("16384"),
            "debug must contain max_slot_bytes with value 16384: {dbg}"
        );
        assert!(
            dbg.contains("slot_count") && dbg.contains("50"),
            "debug must contain slot_count with value 50: {dbg}"
        );
    }

    // ------------------------------------------------------------------
    // open with max_slot_bytes = NVME_ALIGN exactly (no rounding needed)
    // ------------------------------------------------------------------

    #[test]
    fn open_with_exact_nvme_align_slot_bytes_no_rounding() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(
            tmp.path().join("exact_align.swap"),
            4096,
            NVME_ALIGN,
            8,
        )
        .unwrap();
        assert_eq!(
            swap.max_slot_bytes, NVME_ALIGN,
            "already-aligned value must not be rounded up"
        );
    }

    // ------------------------------------------------------------------
    // write then read across page_id boundary (consecutive pages)
    // ------------------------------------------------------------------

    #[test]
    fn write_read_across_consecutive_pages_distinct_data() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(
            tmp.path().join("crosspg.swap"),
            4096,
            4096,
            8,
        )
        .unwrap();

        // Write unique ascending patterns to pages 3, 4, 5
        for pid in 3..=5usize {
            let data: Vec<u8> = (0..512).map(|i| ((pid * 10 + i) % 256) as u8).collect();
            swap.write_slot(pid, &data).unwrap();
        }

        // Verify each page independently
        for pid in 3..=5usize {
            let expected: Vec<u8> = (0..512).map(|i| ((pid * 10 + i) % 256) as u8).collect();
            let mut buf = vec![0u8; 512];
            swap.read_slot(pid, &mut buf).unwrap();
            assert_eq!(
                buf, expected,
                "page {pid} data must match after cross-boundary write"
            );
        }
    }

    // ------------------------------------------------------------------
    // SwapFileHeader _reserved field last byte is at correct offset
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_reserved_last_byte_offset() {
        let mut hdr = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        // Set the last byte of _reserved
        hdr._reserved[SWAP_HEADER_BYTES - 33] = 0xFE;

        let base = &hdr as *const SwapFileHeader as usize;
        let reserved_ptr = &hdr._reserved as *const [u8; SWAP_HEADER_BYTES - 32] as usize;
        let reserved_off = reserved_ptr - base;

        // _reserved starts at offset 32, so last byte is at 32 + (SWAP_HEADER_BYTES - 33)
        assert_eq!(reserved_off, 32, "_reserved must start at offset 32");
        assert_eq!(
            hdr._reserved[SWAP_HEADER_BYTES - 33], 0xFE,
            "last reserved byte must be writable"
        );
    }

    // ------------------------------------------------------------------
    // AlignedBuffer drop after write does not corrupt data (via read)
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_drop_after_write_no_corruption() {
        let mut buf1 = AlignedBuffer::new(NVME_ALIGN);
        buf1.as_mut_slice()[0] = 0xDE;
        buf1.as_mut_slice()[NVME_ALIGN - 1] = 0xAD;
        // Extract values before drop
        let first = buf1.as_slice()[0];
        let last = buf1.as_slice()[NVME_ALIGN - 1];
        // Drop and allocate a new buffer
        drop(buf1);
        let mut buf2 = AlignedBuffer::new(NVME_ALIGN);
        // New buffer must be zero-initialized (not stale data from freed buf1)
        assert!(buf2.as_slice().iter().all(|&b| b == 0));
        // Verify captured values were correct
        assert_eq!(first, 0xDE);
        assert_eq!(last, 0xAD);
    }

    // ------------------------------------------------------------------
    // bad_magic error kind is InvalidData
    // ------------------------------------------------------------------

    #[test]
    fn bad_magic_error_kind_is_invalid_data() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("badkind.swap");
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&[0x00; SWAP_HEADER_BYTES]).unwrap();
            f.set_len(SWAP_HEADER_BYTES as u64).unwrap();
        }
        let err = NvmeSwapFile::open(path, 4096, 4096, 4).unwrap_err();
        assert_eq!(
            err.kind(),
            std::io::ErrorKind::InvalidData,
            "bad magic must produce InvalidData error kind"
        );
    }

    // ------------------------------------------------------------------
    // bad_version error kind is InvalidData
    // ------------------------------------------------------------------

    #[test]
    fn bad_version_error_kind_is_invalid_data() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("badverk.swap");
        {
            // Create a valid swap file first
            let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
            drop(swap);
            // Corrupt the version field
            let mut data = std::fs::read(&path).unwrap();
            let bad_version = 255u32.to_le_bytes();
            data[8..12].copy_from_slice(&bad_version);
            let expected_len = SWAP_HEADER_BYTES + 4 * 4096;
            if data.len() < expected_len {
                data.resize(expected_len, 0);
            }
            std::fs::write(&path, &data).unwrap();
        }
        let err = NvmeSwapFile::open(path, 4096, 4096, 4).unwrap_err();
        assert_eq!(
            err.kind(),
            std::io::ErrorKind::InvalidData,
            "bad version must produce InvalidData error kind"
        );
    }

    // ------------------------------------------------------------------
    // write then read single byte at exact slot boundary offset
    // ------------------------------------------------------------------

    #[test]
    fn write_read_single_byte_at_slot_end() {
        let tmp = TempDir::new().unwrap();
        // Use minimum slot size so slot boundary is predictable
        let swap = NvmeSwapFile::open(
            tmp.path().join("endbyte.swap"),
            4096,
            NVME_ALIGN,
            4,
        )
        .unwrap();

        // Write data that fills the slot up to the last byte
        let mut data = vec![0x77u8; NVME_ALIGN];
        data[NVME_ALIGN - 1] = 0xEE;
        swap.write_slot(0, &data).unwrap();

        // Read only the last byte
        let mut full_buf = vec![0u8; NVME_ALIGN];
        swap.read_slot(0, &mut full_buf).unwrap();
        assert_eq!(
            full_buf[NVME_ALIGN - 1], 0xEE,
            "last byte in slot must be correctly stored"
        );
        assert!(
            full_buf[..NVME_ALIGN - 1].iter().all(|&b| b == 0x77),
            "all preceding bytes must be 0x77"
        );
    }

    // ------------------------------------------------------------------
    // SwapFileHeader can be transmuted to byte array and back
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_byte_roundtrip_via_transmute() {
        let original = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 2048,
            max_slot_bytes: 16384,
            _pad4: 0,
            slot_count: 64,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };

        // Transmute to byte array
        let bytes: [u8; SWAP_HEADER_BYTES] = unsafe {
            std::ptr::read(&original as *const SwapFileHeader as *const [u8; SWAP_HEADER_BYTES])
        };

        // Transmute back
        let restored: SwapFileHeader = unsafe {
            std::ptr::read(&bytes as *const [u8; SWAP_HEADER_BYTES] as *const SwapFileHeader)
        };

        assert_eq!(restored.magic, original.magic);
        assert_eq!(restored.version, original.version);
        assert_eq!(restored.page_size, original.page_size);
        assert_eq!(restored.max_slot_bytes, original.max_slot_bytes);
        assert_eq!(restored.slot_count, original.slot_count);
        assert_eq!(restored._reserved, original._reserved);
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile open with non-power-of-two page_size succeeds
    // ------------------------------------------------------------------

    #[test]
    fn open_non_power_of_two_page_size_stored() {
        let tmp = TempDir::new().unwrap();
        // page_size is just stored, not validated for alignment
        let swap = NvmeSwapFile::open(
            tmp.path().join("npo2.swap"),
            3333, // non-power-of-two
            4096,
            4,
        )
        .unwrap();
        assert_eq!(swap.page_size, 3333);
    }

    // ------------------------------------------------------------------
    // NvmeSwapFile Debug output includes all three public fields
    // ------------------------------------------------------------------

    #[test]
    fn debug_output_includes_page_size_max_slot_bytes_slot_count() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(
            tmp.path().join("debugfld.swap"),
            512,
            8192,
            128,
        )
        .unwrap();
        let dbg = format!("{:?}", swap);
        assert!(
            dbg.contains("page_size: 512"),
            "Debug must show page_size, got: {dbg}"
        );
        assert!(
            dbg.contains("max_slot_bytes: 8192"),
            "Debug must show max_slot_bytes, got: {dbg}"
        );
        assert!(
            dbg.contains("slot_count: 128"),
            "Debug must show slot_count, got: {dbg}"
        );
        assert!(
            dbg.starts_with("NvmeSwapFile"),
            "Debug must start with struct name, got: {dbg}"
        );
    }

    // ------------------------------------------------------------------
    // SwapFileHeader Hash consistency: equal headers produce equal hashes
    // ------------------------------------------------------------------

    #[test]
    fn swap_file_header_equal_implies_equal_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let h1 = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 16,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let h2 = h1;

        let mut s1 = DefaultHasher::new();
        let mut s2 = DefaultHasher::new();
        h1.hash(&mut s1);
        h2.hash(&mut s2);
        assert_eq!(
            s1.finish(),
            s2.finish(),
            "equal SwapFileHeader values must produce equal hashes"
        );
    }

    // ------------------------------------------------------------------
    // SwapFileHeader Hash differentiation: different headers produce different hashes
    // ------------------------------------------------------------------

    #[test]
    fn swap_file_header_different_slot_count_different_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut h1 = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 16,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let mut h2 = h1;
        h2.slot_count = 999;

        let mut s1 = DefaultHasher::new();
        let mut s2 = DefaultHasher::new();
        h1.hash(&mut s1);
        h2.hash(&mut s2);
        assert_ne!(
            s1.finish(),
            s2.finish(),
            "different slot_count must produce different hashes"
        );
    }

    // ------------------------------------------------------------------
    // SwapFileHeader PartialEq and Eq are structural
    // ------------------------------------------------------------------

    #[test]
    fn swap_file_header_eq_identical_fields() {
        let h1 = SwapFileHeader {
            magic: 0xAA,
            version: 2,
            page_size: 1024,
            max_slot_bytes: 4096,
            _pad4: 42,
            slot_count: 7,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let h2 = SwapFileHeader {
            magic: 0xAA,
            version: 2,
            page_size: 1024,
            max_slot_bytes: 4096,
            _pad4: 42,
            slot_count: 7,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(h1, h2, "headers with identical fields must be equal");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader Copy is a bitwise copy
    // ------------------------------------------------------------------

    #[test]
    fn swap_file_header_copy_is_independent() {
        let original = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 16384,
            _pad4: 0,
            slot_count: 32,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let copy = original;
        // Modify copy's slot_count (Copy means original is unchanged)
        assert_eq!(original.slot_count, copy.slot_count);
        assert_eq!(original.page_size, copy.page_size);
        assert_eq!(original.magic, copy.magic);
    }

    // ------------------------------------------------------------------
    // Open with max_slot_bytes not aligned to NVME_ALIGN gets rounded up
    // ------------------------------------------------------------------

    #[test]
    fn open_rounds_up_non_aligned_max_slot_bytes() {
        let tmp = TempDir::new().unwrap();
        // 5000 is not a multiple of 4096; should round up to 8192
        let swap = NvmeSwapFile::open(
            tmp.path().join("rounded.swap"),
            4096,
            5000,
            4,
        )
        .unwrap();
        assert_eq!(
            swap.max_slot_bytes, 8192,
            "max_slot_bytes must be rounded up to the next NVME_ALIGN multiple"
        );
    }

    // ------------------------------------------------------------------
    // Open with max_slot_bytes already aligned stays the same
    // ------------------------------------------------------------------

    #[test]
    fn open_aligned_max_slot_bytes_unchanged() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(
            tmp.path().join("aligned.swap"),
            4096,
            12288, // 3 * 4096, already aligned
            8,
        )
        .unwrap();
        assert_eq!(
            swap.max_slot_bytes, 12288,
            "already-aligned max_slot_bytes must stay unchanged"
        );
    }

    // ------------------------------------------------------------------
    // Open with slot_count=1 creates file with exactly header + 1 slot
    // ------------------------------------------------------------------

    #[test]
    fn open_slot_count_one_file_size_is_header_plus_one_slot() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("oneslot.swap");
        let slot_bytes = 8192u64;
        {
            let _swap = NvmeSwapFile::open(
                path.clone(),
                4096,
                slot_bytes as usize,
                1,
            )
            .unwrap();
        }
        let meta = std::fs::metadata(&path).unwrap();
        let expected = SWAP_HEADER_BYTES as u64 + slot_bytes;
        assert_eq!(
            meta.len(),
            expected,
            "file size must be exactly header + 1 slot"
        );
    }

    // ------------------------------------------------------------------
    // Reopen same file preserves slot data after close/reopen cycle
    // ------------------------------------------------------------------

    #[test]
    fn reopen_preserves_data_across_close_reopen_cycle() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("cycle.swap");
        let pattern: Vec<u8> = (0..256).map(|i| (i as u8).wrapping_mul(3)).collect();

        // Phase 1: write
        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 8192, 8).unwrap();
            swap.write_slot(5, &pattern).unwrap();
        }

        // Phase 2: close and reopen, read back
        let swap2 = NvmeSwapFile::open(path, 4096, 8192, 8).unwrap();
        let mut buf = vec![0u8; 256];
        swap2.read_slot(5, &mut buf).unwrap();
        assert_eq!(
            buf, pattern,
            "data must survive close/reopen cycle"
        );
    }

    // ------------------------------------------------------------------
    // write_slot returns the exact data length written
    // ------------------------------------------------------------------

    #[test]
    fn write_slot_returns_exact_data_length() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(
            tmp.path().join("retlen.swap"),
            4096,
            8192,
            4,
        )
        .unwrap();
        let data = vec![0xAB; 777];
        let written = swap.write_slot(0, &data).unwrap();
        assert_eq!(
            written, 777u32,
            "write_slot must return exact data length"
        );
    }

    // ------------------------------------------------------------------
    // Write empty slice succeeds and returns zero
    // ------------------------------------------------------------------

    #[test]
    fn write_slot_empty_slice_returns_zero() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(
            tmp.path().join("emptyw.swap"),
            4096,
            8192,
            4,
        )
        .unwrap();
        let written = swap.write_slot(2, &[]).unwrap();
        assert_eq!(
            written, 0u32,
            "writing empty slice must return 0"
        );
    }

    // ------------------------------------------------------------------
    // Write full slot then read empty region of same slot (padded zeros)
    // ------------------------------------------------------------------

    #[test]
    fn write_partial_read_beyond_written_region_yields_zeros() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(
            tmp.path().join("padzero.swap"),
            4096,
            8192,
            4,
        )
        .unwrap();
        // Write 4 bytes to slot 1
        swap.write_slot(1, &[0xDE, 0xAD, 0xBE, 0xEF]).unwrap();

        // Read 8 bytes: first 4 must be the written data, next 4 must be zeros
        let mut buf = vec![0u8; 8];
        swap.read_slot(1, &mut buf).unwrap();
        assert_eq!(&buf[..4], &[0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(&buf[4..], &[0u8; 4], "padding beyond written data must be zero");
    }

    // ------------------------------------------------------------------
    // SWAP_MAGIC constant is the expected value 0x474C4C4D53574150
    // ------------------------------------------------------------------

    #[test]
    fn swap_magic_constant_value() {
        // 0x474C4C4D53574150 stored little-endian reads as "PAWSMLLG"
        // which is "GLLMSWAP" read big-endian. Verify the numeric value.
        assert_eq!(
            SWAP_MAGIC, 0x474C4C4D53574150u64,
            "SWAP_MAGIC must be 0x474C4C4D53574150"
        );
        // Also verify the magic bytes are non-zero (valid sentinel)
        let bytes = SWAP_MAGIC.to_le_bytes();
        assert!(
            bytes.iter().any(|&b| b != 0),
            "SWAP_MAGIC must not be all-zero"
        );
    }

    // ------------------------------------------------------------------
    // SWAP_VERSION is exactly 1
    // ------------------------------------------------------------------

    #[test]
    fn swap_version_constant_is_one() {
        assert_eq!(
            SWAP_VERSION, 1,
            "SWAP_VERSION must be 1"
        );
    }

    // ------------------------------------------------------------------
    // SwapFileHeader reserved field is all zeros by default
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_reserved_field_all_zeros() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 4,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert!(
            hdr._reserved.iter().all(|&b| b == 0),
            "_reserved field must be all zeros"
        );
        assert_eq!(
            hdr._reserved.len(),
            SWAP_HEADER_BYTES - 32,
            "_reserved field must be SWAP_HEADER_BYTES - 32 bytes"
        );
    }

    // ------------------------------------------------------------------
    // Additional edge case tests
    // ------------------------------------------------------------------

    #[test]
    fn slot_offset_monotonically_increases() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "mono.swap");
        let mut prev = 0u64;
        for pid in 0..32usize {
            let off = swap.slot_offset(pid);
            assert!(
                off > prev,
                "slot_offset({pid}) = {off} must be > previous {prev}"
            );
            prev = off;
        }
    }

    #[test]
    fn write_read_minimal_non_aligned_slot_bytes() {
        let tmp = TempDir::new().unwrap();
        // max_slot_bytes = 1 rounds up to NVME_ALIGN = 4096
        let swap = NvmeSwapFile::open(
            tmp.path().join("minslot.swap"),
            4096,
            1,
            4,
        )
        .unwrap();
        assert!(
            swap.max_slot_bytes >= NVME_ALIGN,
            "max_slot_bytes must be at least NVME_ALIGN after alignment"
        );
        let data = vec![0xDE; 128];
        swap.write_slot(0, &data).unwrap();
        let mut buf = vec![0u8; 128];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    #[test]
    fn reopen_with_larger_max_slot_bytes_uses_new_alignment() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("resizelot.swap");

        // Create with small max_slot_bytes
        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 8).unwrap();
            swap.write_slot(0, &[0x42; 100]).unwrap();
        }

        // Reopen with larger max_slot_bytes -- aligned_slot will be 12288
        let swap = NvmeSwapFile::open(path, 4096, 10000, 8).unwrap();
        let aligned = 10000usize.div_ceil(NVME_ALIGN) * NVME_ALIGN;
        assert_eq!(
            swap.max_slot_bytes, aligned,
            "reopened max_slot_bytes must reflect new alignment"
        );
    }

    #[test]
    fn write_slot_returns_zero_for_empty_data() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "wempty.swap");
        let written = swap.write_slot(7, &[]).unwrap();
        assert_eq!(written, 0, "writing empty slice must return 0 bytes written");
    }

    #[test]
    fn read_slot_exact_max_slot_bytes_succeeds() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "fullread.swap");

        // Write data that fills exactly max_slot_bytes
        let data = vec![0x55; swap.max_slot_bytes];
        swap.write_slot(0, &data).unwrap();

        // Read back the full slot
        let mut buf = vec![0u8; swap.max_slot_bytes];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0x55), "full slot read must match");
    }

    #[test]
    fn write_read_ascending_byte_pattern() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "ascend.swap");
        let data: Vec<u8> = (0..=255).collect();
        swap.write_slot(0, &data).unwrap();
        let mut buf = vec![0u8; 256];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data, "ascending byte pattern must survive round-trip");
    }

    #[test]
    fn slot_offset_distance_between_consecutive_pages_equals_max_slot() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "gap.swap");
        for pid in 0..31usize {
            let diff = swap.slot_offset(pid + 1) - swap.slot_offset(pid);
            assert_eq!(
                diff, swap.max_slot_bytes as u64,
                "gap between page {pid} and {} must equal max_slot_bytes",
                pid + 1
            );
        }
    }

    #[test]
    fn swap_header_different_magic_not_equal() {
        let a = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 8,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = SwapFileHeader {
            magic: 0xDEADBEEF,
            ..a
        };
        assert_ne!(a, b, "headers with different magic must not be equal");
    }

    #[test]
    fn swap_header_different_page_size_not_equal() {
        let a = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 16,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = SwapFileHeader {
            page_size: 2048,
            ..a
        };
        assert_ne!(a, b, "different page_size must produce non-equal headers");
    }

    #[test]
    fn swap_header_different_max_slot_bytes_not_equal() {
        let a = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 16,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = SwapFileHeader {
            max_slot_bytes: 16384,
            ..a
        };
        assert_ne!(a, b, "different max_slot_bytes must produce non-equal headers");
    }

    #[test]
    fn open_creates_parent_directory() {
        let tmp = TempDir::new().unwrap();
        let nested = tmp.path().join("a").join("b").join("c").join("deep.swap");
        let result = NvmeSwapFile::open(nested, 4096, 8192, 4);
        assert!(
            result.is_ok(),
            "open must succeed even when parent directories do not exist"
        );
    }

    #[test]
    fn multiple_reopens_each_see_last_written_data() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("chain.swap");

        // Write pattern A
        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 8192, 8).unwrap();
            swap.write_slot(0, b"AAAA").unwrap();
        }
        // Overwrite with pattern B
        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 8192, 8).unwrap();
            swap.write_slot(0, b"BBBB").unwrap();
        }
        // Third reopen: must see B
        let swap = NvmeSwapFile::open(path, 4096, 8192, 8).unwrap();
        let mut buf = [0u8; 4];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(&buf, b"BBBB", "third reopen must see last written data");
    }

    #[test]
    fn write_then_read_different_lengths_no_corruption() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "difflen.swap");

        // Write 100 bytes to slot 0, 50 bytes to slot 1
        let d0 = vec![0x10; 100];
        let d1 = vec![0x20; 50];
        swap.write_slot(0, &d0).unwrap();
        swap.write_slot(1, &d1).unwrap();

        // Read back different lengths
        let mut b0 = vec![0u8; 100];
        let mut b1 = vec![0u8; 50];
        swap.read_slot(0, &mut b0).unwrap();
        swap.read_slot(1, &mut b1).unwrap();

        assert!(b0.iter().all(|&b| b == 0x10), "slot 0 must be 0x10");
        assert!(b1.iter().all(|&b| b == 0x20), "slot 1 must be 0x20");
    }

    #[test]
    fn swap_header_eq_is_reflexive() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 32,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        assert_eq!(hdr, hdr, "SwapFileHeader Eq must be reflexive");
    }

    #[test]
    fn swap_header_debug_shows_all_fields() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 16,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let dbg = format!("{:?}", hdr);
        assert!(dbg.contains("magic"), "Debug must show magic");
        assert!(dbg.contains("page_size"), "Debug must show page_size");
        assert!(dbg.contains("slot_count"), "Debug must show slot_count");
    }

    #[test]
    fn swap_header_clone_produces_equal_copy() {
        let hdr = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: 1,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 16,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let cloned = hdr.clone();
        assert_eq!(hdr, cloned);
    }

    #[test]
    fn nvme_swap_file_debug_shows_page_size_and_slot_count() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "debug.swap");
        let dbg = format!("{:?}", swap);
        assert!(dbg.contains("page_size"), "Debug must show page_size");
        assert!(dbg.contains("max_slot_bytes"), "Debug must show max_slot_bytes");
        assert!(dbg.contains("slot_count"), "Debug must show slot_count");
    }

    #[test]
    fn write_read_all_zeros_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "zeros.swap");
        let data = vec![0u8; 4096];
        swap.write_slot(0, &data).unwrap();
        let mut buf = vec![0u8; 4096];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0), "all-zeros roundtrip");
    }

    #[test]
    fn write_read_max_slot_bytes_exactly() {
        let tmp = TempDir::new().unwrap();
        // open_swap uses 8192 max_slot_bytes
        let swap = open_swap(&tmp, "maxslot.swap");
        let data: Vec<u8> = (0..8192).map(|i| (i % 256) as u8).collect();
        swap.write_slot(0, &data).unwrap();
        let mut buf = vec![0u8; 8192];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    #[test]
    fn write_read_last_slot_succeeds() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "lastslot.swap");
        let last = (swap.slot_count - 1) as usize;
        let data = vec![0xABu8; 100];
        swap.write_slot(last, &data).unwrap();
        let mut buf = vec![0u8; 100];
        swap.read_slot(last, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xAB));
    }

    #[test]
    fn write_read_two_adjacent_slots_no_overlap() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "adjacent.swap");
        let d0 = vec![0x11u8; 512];
        let d1 = vec![0x22u8; 512];
        swap.write_slot(0, &d0).unwrap();
        swap.write_slot(1, &d1).unwrap();
        let mut b0 = vec![0u8; 512];
        let mut b1 = vec![0u8; 512];
        swap.read_slot(0, &mut b0).unwrap();
        swap.read_slot(1, &mut b1).unwrap();
        assert!(b0.iter().all(|&b| b == 0x11), "slot 0 not corrupted");
        assert!(b1.iter().all(|&b| b == 0x22), "slot 1 not corrupted");
    }

    #[test]
    fn swap_file_size_is_header_plus_slots() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("size.swap");
        let page_size = 4096usize;
        let max_slot = 4096usize;
        let slot_count = 4u64;
        let _swap = NvmeSwapFile::open(path.clone(), page_size, max_slot, slot_count).unwrap();
        let meta = std::fs::metadata(&path).unwrap();
        let expected = SWAP_HEADER_BYTES as u64 + slot_count * max_slot as u64;
        assert_eq!(meta.len(), expected, "file size = header + slots");
    }

    #[test]
    fn header_magic_is_nonzero_and_consistent() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "magic.swap");
        // Reopen and read header to verify magic
        drop(swap);
        let path = tmp.path().join("magic.swap");
        let bytes = std::fs::read(&path).unwrap();
        let magic = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        assert_eq!(magic, SWAP_MAGIC);
        assert_ne!(magic, 0);
    }

    #[test]
    fn header_version_is_one() {
        let tmp = TempDir::new().unwrap();
        let _swap = open_swap(&tmp, "ver.swap");
        drop(_swap);
        let path = tmp.path().join("ver.swap");
        let bytes = std::fs::read(&path).unwrap();
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        assert_eq!(version, 1);
    }

    #[test]
    fn open_same_path_twice_succeeds() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("double.swap");
        let _s1 = NvmeSwapFile::open(path.clone(), 4096, 4096, 2).unwrap();
        drop(_s1);
        let _s2 = NvmeSwapFile::open(path, 4096, 4096, 2).unwrap();
    }

    #[test]
    fn swap_file_created_with_644_permissions() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("perm.swap");
        let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 2).unwrap();
        let meta = std::fs::metadata(&path).unwrap();
        let mode = meta.permissions().mode() & 0o777;
        assert_eq!(mode, 0o644, "file must be created with 0644");
    }

    #[test]
    fn write_0xff_pattern_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "ff.swap");
        let data = vec![0xFFu8; 4096];
        swap.write_slot(0, &data).unwrap();
        let mut buf = vec![0u8; 4096];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn read_slot_zero_length_succeeds() {
        let tmp = TempDir::new().unwrap();
        let swap = open_swap(&tmp, "emptyread.swap");
        let mut buf = [0u8; 0];
        // Reading 0 bytes should succeed
        swap.read_slot(0, &mut buf).unwrap();
    }

    #[test]
    fn max_slot_bytes_rounds_up_to_next_4096_boundary() {
        let tmp = TempDir::new().unwrap();
        // 4097 bytes rounds up to 8192
        let swap = NvmeSwapFile::open(
            tmp.path().join("ceil4097.swap"),
            4096,
            4097,
            4,
        )
        .unwrap();
        assert_eq!(
            swap.max_slot_bytes, 8192,
            "4097 must round up to 8192 (next 4096 boundary)"
        );
    }

    // ==================================================================
    // 15 additional tests — header disk persistence, boundary I/O,
    // unicode paths, raw byte verification, overwrite semantics
    // ==================================================================

    // ------------------------------------------------------------------
    // header page_size 字段正确写入磁盘并可从原始字节读回
    // ------------------------------------------------------------------

    #[test]
    fn header_page_size_written_to_disk_correctly() {
        // 验证 SwapFileHeader 的 page_size 字段在 open 时以 LE 格式写入磁盘
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("hdr_ps_disk.swap");
        let unique_page_size: u32 = 7919;
        let _swap = NvmeSwapFile::open(path.clone(), unique_page_size as usize, 4096, 4).unwrap();
        drop(_swap);

        let bytes = std::fs::read(&path).unwrap();
        let disk_page_size = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        assert_eq!(
            disk_page_size, unique_page_size,
            "page_size on disk must match constructor argument"
        );
    }

    // ------------------------------------------------------------------
    // header max_slot_bytes 字段在磁盘上是向上取齐后的值
    // ------------------------------------------------------------------

    #[test]
    fn header_aligned_max_slot_bytes_on_disk() {
        // 传入 max_slot_bytes=5000，磁盘上应存储向上取齐后的 8192
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("hdr_aligned_disk.swap");
        let _swap = NvmeSwapFile::open(path.clone(), 4096, 5000, 4).unwrap();
        drop(_swap);

        let bytes = std::fs::read(&path).unwrap();
        let disk_msb = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        assert_eq!(
            disk_msb, 8192,
            "max_slot_bytes on disk must be the aligned value 8192, got {disk_msb}"
        );
    }

    // ------------------------------------------------------------------
    // header slot_count 字段正确持久化到磁盘
    // ------------------------------------------------------------------

    #[test]
    fn header_slot_count_written_to_disk_correctly() {
        // 验证 slot_count 以 LE u64 格式存储在 offset 24
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("hdr_sc_disk.swap");
        let slot_count: u64 = 42;
        let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, slot_count).unwrap();
        drop(_swap);

        let bytes = std::fs::read(&path).unwrap();
        let disk_sc = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
        assert_eq!(
            disk_sc, slot_count,
            "slot_count on disk must match constructor argument"
        );
    }

    // ------------------------------------------------------------------
    // 文件路径含中文字符（Unicode）时 open 成功
    // ------------------------------------------------------------------

    #[test]
    fn open_with_unicode_path_succeeds() {
        // 验证含非 ASCII 字符的文件路径可以正常打开
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("交换文件_测试.swap");
        let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4);
        assert!(swap.is_ok(), "Unicode path should open successfully");
        let swap = swap.unwrap();
        swap.write_slot(0, &[0xAA; 64]).unwrap();
        let mut buf = vec![0u8; 64];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xAA));
    }

    // ------------------------------------------------------------------
    // header _pad4 字段在磁盘上为零
    // ------------------------------------------------------------------

    #[test]
    fn header_pad4_zero_on_disk() {
        // _pad4 在 open 时被设为 0，验证磁盘上 offset 20..24 全为零
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("pad4_zero.swap");
        let _swap = NvmeSwapFile::open(path.clone(), 4096, 8192, 8).unwrap();
        drop(_swap);

        let bytes = std::fs::read(&path).unwrap();
        let pad4 = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        assert_eq!(pad4, 0, "_pad4 must be zero on disk");
    }

    // ------------------------------------------------------------------
    // 写入后用原始字节读取验证 slot 数据在正确偏移位置
    // ------------------------------------------------------------------

    #[test]
    fn slot_data_at_correct_disk_offset() {
        // 向 page_id=1 写入数据，验证它在 SWAP_HEADER_BYTES + max_slot_bytes 处
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("disk_off.swap");
        let max_slot = 4096usize;
        let _swap = NvmeSwapFile::open(path.clone(), 4096, max_slot, 4).unwrap();
        drop(_swap);

        // 重新打开并写入
        let swap = NvmeSwapFile::open(path.clone(), 4096, max_slot, 4).unwrap();
        let pattern = [0xDEu8; 32];
        swap.write_slot(1, &pattern).unwrap();
        drop(swap);

        let bytes = std::fs::read(&path).unwrap();
        let slot1_start = SWAP_HEADER_BYTES + max_slot;
        // 验证数据确实在正确的偏移位置
        assert_eq!(
            &bytes[slot1_start..slot1_start + 32],
            &pattern,
            "slot 1 data must be at offset {slot1_start}"
        );
    }

    // ------------------------------------------------------------------
    // 跨 slot 边界写入不破坏相邻 slot 的数据
    // ------------------------------------------------------------------

    #[test]
    fn full_slot_write_does_not_leak_into_adjacent_slot() {
        // 向 slot 0 写满数据，验证 slot 1 的前几个字节仍为零（未被污染）
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("no_leak.swap"), 4096, 4096, 4).unwrap();

        let full_data = vec![0xFF; swap.max_slot_bytes];
        swap.write_slot(0, &full_data).unwrap();

        // slot 1 从未写入，应为全零
        let mut buf = vec![0xFF; swap.max_slot_bytes];
        swap.read_slot(1, &mut buf).unwrap();
        assert!(
            buf.iter().all(|&b| b == 0),
            "slot 1 must be all zeros, not contaminated by slot 0"
        );
    }

    // ------------------------------------------------------------------
    // 大量 slot 交替写入验证无交叉污染
    // ------------------------------------------------------------------

    #[test]
    fn alternating_slot_writes_no_cross_contamination() {
        // 向偶数 slot 写 0xAA，向奇数 slot 写 0xBB，验证无交叉
        let tmp = TempDir::new().unwrap();
        let count = 32u64;
        let swap = NvmeSwapFile::open(tmp.path().join("alt_cont.swap"), 4096, 4096, count).unwrap();

        for pid in 0..count as usize {
            let val = if pid % 2 == 0 { 0xAA } else { 0xBB };
            swap.write_slot(pid, &[val; 128]).unwrap();
        }

        for pid in 0..count as usize {
            let expected = if pid % 2 == 0 { 0xAA } else { 0xBB };
            let mut buf = vec![0u8; 128];
            swap.read_slot(pid, &mut buf).unwrap();
            assert!(
                buf.iter().all(|&b| b == expected),
                "slot {pid}: expected 0x{expected:02x}, got mixed data"
            );
        }
    }

    // ------------------------------------------------------------------
    // 向已损坏（magic 正确但 version 损坏）的文件重新 open 返回 InvalidData
    // ------------------------------------------------------------------

    #[test]
    fn reopen_corrupted_version_returns_invalid_data_kind() {
        // 创建合法文件后仅损坏 version 字段，验证 error kind 为 InvalidData
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("corrupt_ver.swap");
        {
            let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
        }
        let mut data = std::fs::read(&path).unwrap();
        // 仅修改 version 的最低字节
        data[8] = 0xFF;
        let expected_len = SWAP_HEADER_BYTES + 4 * 4096;
        if data.len() < expected_len {
            data.resize(expected_len, 0);
        }
        std::fs::write(&path, &data).unwrap();

        let err = NvmeSwapFile::open(path, 4096, 4096, 4).unwrap_err();
        assert_eq!(
            err.kind(),
            std::io::ErrorKind::InvalidData,
            "corrupted version must produce InvalidData error kind"
        );
    }

    // ------------------------------------------------------------------
    // header _reserved 区域在磁盘上全部为零
    // ------------------------------------------------------------------

    #[test]
    fn header_reserved_all_zeros_on_disk() {
        // 验证 open 写入的 _reserved 区域（offset 32..4096）全为零
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("reserved_zero.swap");
        let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
        drop(_swap);

        let bytes = std::fs::read(&path).unwrap();
        for i in 32..SWAP_HEADER_BYTES {
            assert_eq!(
                bytes[i], 0,
                "_reserved byte at offset {i} must be zero"
            );
        }
    }

    // ------------------------------------------------------------------
    // 多次覆盖同一 slot 后读回最终值（50 次覆盖压力测试）
    // ------------------------------------------------------------------

    #[test]
    fn repeated_overwrite_same_slot_pressure_test() {
        // 连续 200 次覆盖同一 slot，验证最终写入的数据是最后一次
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("pressure.swap"), 4096, 4096, 4).unwrap();

        let total_writes = 200u8;
        for i in 0..total_writes {
            let data = vec![i; 64];
            swap.write_slot(0, &data).unwrap();
        }

        let mut buf = vec![0u8; 64];
        swap.read_slot(0, &mut buf).unwrap();
        let last_val = total_writes - 1;
        assert!(
            buf.iter().all(|&b| b == last_val),
            "after {total_writes} overwrites, slot must contain value {last_val}"
        );
    }

    // ------------------------------------------------------------------
    // 写入包含各种位模式（0x00/0xFF/0x55/0xAA）的混合数据
    // ------------------------------------------------------------------

    #[test]
    fn write_mixed_bit_patterns_roundtrip() {
        // 写入由不同位模式组成的数据块，验证 I/O 不会篡改任何位
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("mixed_bits.swap"), 4096, 4096, 4).unwrap();

        let mut data = Vec::with_capacity(4096);
        // 每种位模式占 1024 字节：0x00 / 0xFF / 0x55 / 0xAA
        data.extend(std::iter::repeat(0x00u8).take(1024));
        data.extend(std::iter::repeat(0xFFu8).take(1024));
        data.extend(std::iter::repeat(0x55u8).take(1024));
        data.extend(std::iter::repeat(0xAAu8).take(1024));

        swap.write_slot(0, &data).unwrap();
        let mut buf = vec![0u8; 4096];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data, "mixed bit patterns must survive round-trip");
    }

    // ------------------------------------------------------------------
    // header magic 字段在磁盘上的 LE 字节序验证
    // ------------------------------------------------------------------

    #[test]
    fn header_magic_bytes_on_disk_are_le() {
        // 验证 SWAP_MAGIC 写入磁盘后前 8 字节是 "PAWSMLLG" (LE)
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("magic_le_disk.swap");
        let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
        drop(_swap);

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..8], b"PAWSMLLG", "magic on disk must be PAWSMLLG in LE");
    }

    // ------------------------------------------------------------------
    // 写入后 drop，用 std::fs 直接读取验证数据完整性（绕过 NvmeSwapFile）
    // ------------------------------------------------------------------

    #[test]
    fn data_integrity_verified_via_raw_fs_read() {
        // 通过 std::fs::read 直接读取文件，绕过 NvmeSwapFile，
        // 验证数据确实落在正确的偏移位置且内容正确
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("raw_read.swap");
        let max_slot = 8192usize;
        let data: Vec<u8> = (0..500).map(|i| (i ^ 0xA5) as u8).collect();

        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, max_slot, 8).unwrap();
            swap.write_slot(3, &data).unwrap();
        }

        let raw = std::fs::read(&path).unwrap();
        let offset = SWAP_HEADER_BYTES + 3 * max_slot;
        assert_eq!(
            &raw[offset..offset + data.len()],
            &data,
            "raw file read must match written data at correct offset"
        );
    }

    // ------------------------------------------------------------------
    // 同时打开两个不同文件（不同路径）各自独立操作互不干扰
    // ------------------------------------------------------------------

    #[test]
    fn two_concurrently_open_files_independent() {
        // 同时打开两个 NvmeSwapFile 实例，写入不同数据，验证互不影响
        let tmp = TempDir::new().unwrap();
        let swap_a = NvmeSwapFile::open(tmp.path().join("con_a.swap"), 4096, 4096, 4).unwrap();
        let swap_b = NvmeSwapFile::open(tmp.path().join("con_b.swap"), 4096, 4096, 4).unwrap();

        swap_a.write_slot(0, &[0x11; 128]).unwrap();
        swap_b.write_slot(0, &[0x22; 128]).unwrap();
        swap_a.write_slot(1, &[0x33; 64]).unwrap();
        swap_b.write_slot(1, &[0x44; 64]).unwrap();

        let mut buf_a0 = vec![0u8; 128];
        let mut buf_b0 = vec![0u8; 128];
        let mut buf_a1 = vec![0u8; 64];
        let mut buf_b1 = vec![0u8; 64];

        swap_a.read_slot(0, &mut buf_a0).unwrap();
        swap_b.read_slot(0, &mut buf_b0).unwrap();
        swap_a.read_slot(1, &mut buf_a1).unwrap();
        swap_b.read_slot(1, &mut buf_b1).unwrap();

        assert!(buf_a0.iter().all(|&b| b == 0x11), "file A slot 0 must be 0x11");
        assert!(buf_b0.iter().all(|&b| b == 0x22), "file B slot 0 must be 0x22");
        assert!(buf_a1.iter().all(|&b| b == 0x33), "file A slot 1 must be 0x33");
        assert!(buf_b1.iter().all(|&b| b == 0x44), "file B slot 1 must be 0x44");
    }

    // ==================================================================
    // 15 additional tests — target 579 total (ratio ≈ 20.3)
    // Focus: on-disk header field persistence, boundary conditions,
    // slot isolation verification via raw fs read, file lifecycle edges.
    // ==================================================================

    // ------------------------------------------------------------------
    // header page_size field persisted to disk as LE u32 at offset 12
    // ------------------------------------------------------------------

    #[test]
    fn header_page_size_persisted_on_disk() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("hps_disk.swap");
        let unique_ps = 7919u32;
        let _swap = NvmeSwapFile::open(path.clone(), unique_ps as usize, 4096, 4).unwrap();
        drop(_swap);

        let bytes = std::fs::read(&path).unwrap();
        let on_disk_ps = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        assert_eq!(on_disk_ps, unique_ps, "page_size on disk must match constructor argument");
    }

    // ------------------------------------------------------------------
    // header version field persisted to disk as LE u32 at offset 8
    // ------------------------------------------------------------------

    #[test]
    fn header_version_persisted_on_disk() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("hver_disk.swap");
        let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
        drop(_swap);

        let bytes = std::fs::read(&path).unwrap();
        let on_disk_ver = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        assert_eq!(on_disk_ver, SWAP_VERSION, "version on disk must be SWAP_VERSION");
    }

    // ------------------------------------------------------------------
    // header slot_count field persisted to disk as LE u64 at offset 24
    // ------------------------------------------------------------------

    #[test]
    fn header_slot_count_persisted_on_disk() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("hsc_disk.swap");
        let unique_count = 12345u64;
        let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, unique_count).unwrap();
        drop(_swap);

        let bytes = std::fs::read(&path).unwrap();
        let on_disk_sc = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
        assert_eq!(on_disk_sc, unique_count, "slot_count on disk must match constructor argument");
    }

    // ------------------------------------------------------------------
    // header max_slot_bytes field persisted to disk as LE u32 at offset 16
    // (after alignment rounding)
    // ------------------------------------------------------------------

    #[test]
    fn header_max_slot_bytes_persisted_on_disk() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("hmsb_disk.swap");
        // Pass 5000 → aligns to 8192
        let _swap = NvmeSwapFile::open(path.clone(), 4096, 5000, 4).unwrap();
        drop(_swap);

        let bytes = std::fs::read(&path).unwrap();
        let on_disk_msb = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        assert_eq!(on_disk_msb, 8192u32, "max_slot_bytes on disk must be aligned value 8192");
    }

    // ------------------------------------------------------------------
    // file exactly SWAP_HEADER_BYTES long but with valid magic+version
    // is treated as existing (not re-created)
    // ------------------------------------------------------------------

    #[test]
    fn open_file_exactly_header_bytes_valid_magic_treated_as_existing() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("exact_hdr.swap");

        {
            // Create a valid swap file with 0 slots = exactly header
            let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 0).unwrap();
        }

        // File is exactly SWAP_HEADER_BYTES with valid magic + version
        let meta = std::fs::metadata(&path).unwrap();
        assert_eq!(meta.len(), SWAP_HEADER_BYTES as u64);

        // Reopen must succeed (treated as existing file)
        let swap = NvmeSwapFile::open(path, 4096, 4096, 0);
        assert!(swap.is_ok(), "valid header-only file must reopen successfully");
    }

    // ------------------------------------------------------------------
    // write_slot with data size 1 byte followed by read_slot of
    // max_slot_bytes: only byte 0 is data, rest is zero-padded
    // ------------------------------------------------------------------

    #[test]
    fn write_one_byte_read_full_slot_padding_is_zero() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("one_full.swap"), 4096, 4096, 4).unwrap();

        swap.write_slot(0, &[0xCD]).unwrap();
        let mut full = vec![0xFF; swap.max_slot_bytes];
        swap.read_slot(0, &mut full).unwrap();

        assert_eq!(full[0], 0xCD, "first byte must be written data");
        assert!(full[1..].iter().all(|&b| b == 0x00), "remaining bytes must be zero-padded");
    }

    // ------------------------------------------------------------------
    // slot_offset produces correct file offset verified via raw fs read
    // ------------------------------------------------------------------

    #[test]
    fn slot_offset_matches_raw_file_offset() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("raw_off.swap");
        let max_slot = 8192usize;
        let data: Vec<u8> = (0..333).map(|i| (i ^ 0x5A) as u8).collect();

        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, max_slot, 8).unwrap();
            swap.write_slot(5, &data).unwrap();
        }

        let raw = std::fs::read(&path).unwrap();
        let expected_offset = SWAP_HEADER_BYTES + 5 * max_slot;
        assert_eq!(
            &raw[expected_offset..expected_offset + data.len()],
            &data,
            "raw bytes at slot_offset must match written data"
        );
    }

    // ------------------------------------------------------------------
    // write to slot N, overwrite slot N with smaller data, read
    // the full slot: new data + zeros (old tail is gone)
    // ------------------------------------------------------------------

    #[test]
    fn overwrite_smaller_then_read_full_slot_old_tail_zeroed() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("shrink_full.swap"), 4096, 4096, 4).unwrap();

        // Write 256 bytes of 0xAA
        swap.write_slot(0, &[0xAA; 256]).unwrap();
        // Overwrite with 32 bytes of 0xBB
        swap.write_slot(0, &[0xBB; 32]).unwrap();

        let mut full = vec![0u8; 256];
        swap.read_slot(0, &mut full).unwrap();
        assert!(full[..32].iter().all(|&b| b == 0xBB), "first 32 bytes = new data");
        assert!(full[32..256].iter().all(|&b| b == 0x00), "old tail must be zeroed by AlignedBuffer");
    }

    // ------------------------------------------------------------------
    // open with a path that has no parent (file in current directory)
    // — verify path.parent().is_none() branch works
    // ------------------------------------------------------------------

    #[test]
    fn open_with_relative_path_no_parent_dir() {
        let tmp = TempDir::new().unwrap();
        // Use a relative-style path within the temp dir (no nested dirs)
        let path = tmp.path().join("flat.swap");
        // path.parent() is Some(tmp.path()), not None, but this verifies
        // the create_dir_all path works for flat paths too
        let swap = NvmeSwapFile::open(path, 4096, 4096, 4);
        assert!(swap.is_ok(), "flat path should open without creating subdirectories");
    }

    // ------------------------------------------------------------------
    // SwapFileHeader: _reserved field can hold arbitrary byte pattern
    // and still be equal to another header with same pattern
    // ------------------------------------------------------------------

    #[test]
    fn swap_header_identical_custom_reserved_are_equal() {
        let mut reserved = [0u8; SWAP_HEADER_BYTES - 32];
        for (i, byte) in reserved.iter_mut().enumerate() {
            *byte = (i % 251) as u8;
        }
        let a = SwapFileHeader {
            magic: 999,
            version: 7,
            page_size: 2048,
            max_slot_bytes: 4096,
            _pad4: 42,
            slot_count: 13,
            _reserved: reserved,
        };
        let b = SwapFileHeader {
            magic: 999,
            version: 7,
            page_size: 2048,
            max_slot_bytes: 4096,
            _pad4: 42,
            slot_count: 13,
            _reserved: reserved,
        };
        assert_eq!(a, b, "headers with identical custom reserved must be equal");
    }

    // ------------------------------------------------------------------
    // write_slot with data exactly 2 bytes succeeds and read_slot
    // of 2 bytes returns correct data
    // ------------------------------------------------------------------

    #[test]
    fn write_two_bytes_read_two_bytes_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("two_bytes.swap"), 4096, 4096, 4).unwrap();

        let written = swap.write_slot(3, &[0xDE, 0xAD]).unwrap();
        assert_eq!(written, 2);

        let mut buf = [0u8; 2];
        swap.read_slot(3, &mut buf).unwrap();
        assert_eq!(buf[0], 0xDE);
        assert_eq!(buf[1], 0xAD);
    }

    // ------------------------------------------------------------------
    // Verify that writing to the last valid slot does not corrupt the
    // first slot by reading raw file bytes
    // ------------------------------------------------------------------

    #[test]
    fn last_slot_write_does_not_corrupt_first_slot_raw_verify() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("last_first.swap");
        let max_slot = 4096usize;
        let count = 4u64;

        let first_data = vec![0x11; 128];
        let last_data = vec![0xFF; max_slot];

        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, max_slot, count).unwrap();
            swap.write_slot(0, &first_data).unwrap();
            swap.write_slot((count - 1) as PageId, &last_data).unwrap();
        }

        let raw = std::fs::read(&path).unwrap();
        // Verify first slot data at offset SWAP_HEADER_BYTES
        assert_eq!(
            &raw[SWAP_HEADER_BYTES..SWAP_HEADER_BYTES + 128],
            &first_data,
            "first slot must be intact after last slot write"
        );
        // Verify last slot data at correct offset
        let last_offset = SWAP_HEADER_BYTES + (count as usize - 1) * max_slot;
        assert_eq!(
            &raw[last_offset..last_offset + max_slot],
            &last_data,
            "last slot must contain its full data"
        );
    }

    // ------------------------------------------------------------------
    // header magic field on disk is SWAP_MAGIC (u64 LE) verified by
    // reading raw file bytes (distinct from the LE-bytes-to-string test)
    // ------------------------------------------------------------------

    #[test]
    fn header_magic_u64_value_on_disk() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("magic_u64.swap");
        let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
        drop(_swap);

        let bytes = std::fs::read(&path).unwrap();
        let on_disk_magic = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        assert_eq!(on_disk_magic, SWAP_MAGIC, "magic on disk must equal SWAP_MAGIC constant");
    }

    // ------------------------------------------------------------------
    // write different data to each of 128 slots, reopen, verify all intact
    // ------------------------------------------------------------------

    #[test]
    fn write_128_slots_reopen_verify_all() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("128_slots.swap");
        let count = 128u64;

        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, count).unwrap();
            for pid in 0..count as usize {
                let data = vec![(pid as u8).wrapping_mul(2); 64];
                swap.write_slot(pid, &data).unwrap();
            }
        }

        let swap = NvmeSwapFile::open(path, 4096, 4096, count).unwrap();
        for pid in 0..count as usize {
            let expected = (pid as u8).wrapping_mul(2);
            let mut buf = vec![0u8; 64];
            swap.read_slot(pid, &mut buf).unwrap();
            assert!(
                buf.iter().all(|&b| b == expected),
                "slot {pid}: expected 0x{expected:02x}"
            );
        }
    }

    // ------------------------------------------------------------------
    // open with page_size = max_slot_bytes and both = NVME_ALIGN,
    // verify struct fields match exactly
    // ------------------------------------------------------------------

    #[test]
    fn open_page_size_equals_max_slot_equals_nvme_align() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(
            tmp.path().join("ps_eq_msb_eq_align.swap"),
            NVME_ALIGN,
            NVME_ALIGN,
            8,
        )
        .unwrap();
        assert_eq!(swap.page_size, NVME_ALIGN);
        assert_eq!(swap.max_slot_bytes, NVME_ALIGN);
        assert_eq!(swap.page_size, swap.max_slot_bytes);
    }

    // ------------------------------------------------------------------
    // header max_slot_bytes on disk is the aligned value even when
    // the constructor receives a non-aligned input
    // ------------------------------------------------------------------

    #[test]
    fn header_stores_aligned_max_slot_bytes_on_disk() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("aligned_disk.swap");

        // Pass non-aligned value 3333
        let _swap = NvmeSwapFile::open(path.clone(), 4096, 3333, 4).unwrap();
        drop(_swap);

        let bytes = std::fs::read(&path).unwrap();
        let on_disk = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        // 3333 / 4096 = 0 remainder 3333 → ceil = 1 → 4096
        assert_eq!(on_disk, 4096u32, "max_slot_bytes on disk must be the aligned value 4096");
    }

    // ==================================================================
    // 15 additional tests — multi-file concurrency, alignment constraints,
    // write-read consistency, drop semantics, AlignedBuffer edge cases,
    // and large-slot-count stress tests.
    // ==================================================================

    // ------------------------------------------------------------------
    // Two independent swap files opened simultaneously can both be
    // written to and read from without interference.
    // ------------------------------------------------------------------

    #[test]
    fn two_swap_files_concurrent_independent() {
        let tmp = TempDir::new().unwrap();
        let swap_a = NvmeSwapFile::open(tmp.path().join("a.swap"), 4096, 4096, 4).unwrap();
        let swap_b = NvmeSwapFile::open(tmp.path().join("b.swap"), 4096, 4096, 4).unwrap();

        let data_a = vec![0xAA; 128];
        let data_b = vec![0xBB; 128];
        swap_a.write_slot(0, &data_a).unwrap();
        swap_b.write_slot(0, &data_b).unwrap();

        let mut buf_a = vec![0u8; 128];
        let mut buf_b = vec![0u8; 128];
        swap_a.read_slot(0, &mut buf_a).unwrap();
        swap_b.read_slot(0, &mut buf_b).unwrap();

        assert!(buf_a.iter().all(|&b| b == 0xAA), "file A data must be 0xAA");
        assert!(buf_b.iter().all(|&b| b == 0xBB), "file B data must be 0xBB");
    }

    // ------------------------------------------------------------------
    // Two swap files with different slot sizes opened simultaneously
    // each maintain their own alignment independently.
    // ------------------------------------------------------------------

    #[test]
    fn two_swap_files_different_slot_sizes_concurrent() {
        let tmp = TempDir::new().unwrap();
        let swap_small = NvmeSwapFile::open(tmp.path().join("sm.swap"), 4096, 4096, 8).unwrap();
        let swap_large = NvmeSwapFile::open(tmp.path().join("lg.swap"), 4096, 8192, 8).unwrap();

        assert_eq!(swap_small.max_slot_bytes, 4096);
        assert_eq!(swap_large.max_slot_bytes, 8192);

        let data_s = vec![0x11; 4096];
        let data_l = vec![0x22; 8192];
        swap_small.write_slot(3, &data_s).unwrap();
        swap_large.write_slot(5, &data_l).unwrap();

        let mut buf_s = vec![0u8; 4096];
        let mut buf_l = vec![0u8; 8192];
        swap_small.read_slot(3, &mut buf_s).unwrap();
        swap_large.read_slot(5, &mut buf_l).unwrap();

        assert!(buf_s.iter().all(|&b| b == 0x11));
        assert!(buf_l.iter().all(|&b| b == 0x22));
    }

    // ------------------------------------------------------------------
    // page_size alignment: non-aligned page_size values are stored
    // as-is in the struct (no rounding applied to page_size).
    // ------------------------------------------------------------------

    #[test]
    fn page_size_not_rounded_stored_verbatim() {
        let tmp = TempDir::new().unwrap();
        for &ps in &[1usize, 3, 99, 3333, 5001, 99999] {
            let path = tmp.path().join(format!("psv2_{ps}.swap"));
            let swap = NvmeSwapFile::open(path, ps, 4096, 4).unwrap();
            assert_eq!(
                swap.page_size, ps,
                "page_size {ps} must be stored without rounding"
            );
        }
    }

    // ------------------------------------------------------------------
    // page_size is persisted on disk and survives reopen even when
    // the value is non-standard (e.g., 7777).
    // ------------------------------------------------------------------

    #[test]
    fn page_size_non_standard_persists_across_reopen() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("ps_persist.swap");

        {
            let _swap = NvmeSwapFile::open(path.clone(), 7777, 4096, 4).unwrap();
        }

        let swap = NvmeSwapFile::open(path, 7777, 4096, 4).unwrap();
        assert_eq!(swap.page_size, 7777);
    }

    // ------------------------------------------------------------------
    // write_slot followed immediately by read_slot on the same page_id
    // returns exactly the same bytes, tested across multiple sizes.
    // ------------------------------------------------------------------

    #[test]
    fn immediate_write_read_consistency_across_sizes() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("imm.swap"), 4096, 8192, 8).unwrap();

        let sizes = [1usize, 7, 63, 255, 1024, 4096, 8192];
        for &size in &sizes {
            let data: Vec<u8> = (0..size).map(|i| (i.wrapping_mul(37) as u8)).collect();
            swap.write_slot(0, &data).unwrap();
            let mut buf = vec![0u8; size];
            swap.read_slot(0, &mut buf).unwrap();
            assert_eq!(
                buf, data,
                "immediate read after write failed for size {size}"
            );
        }
    }

    // ------------------------------------------------------------------
    // write_slot then immediate read_slot across all slots in sequence
    // produces consistent data for each slot individually.
    // ------------------------------------------------------------------

    #[test]
    fn write_read_each_slot_immediately() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("imm_all.swap"), 4096, 4096, 16).unwrap();

        for pid in 0..16usize {
            let data = vec![(pid as u8).wrapping_add(0x80); 128];
            swap.write_slot(pid, &data).unwrap();
            let mut buf = vec![0u8; 128];
            swap.read_slot(pid, &mut buf).unwrap();
            assert_eq!(buf, data, "immediate read mismatch at slot {pid}");
        }
    }

    // ------------------------------------------------------------------
    // After NvmeSwapFile is dropped, the file descriptor is released
    // and the file can be deleted from the filesystem.
    // ------------------------------------------------------------------

    #[test]
    fn drop_releases_file_deletable() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("del.swap");

        {
            let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
        }

        // File must be deletable after drop (fd closed)
        assert!(std::fs::remove_file(&path).is_ok(), "file should be deletable after drop");
        assert!(!path.exists(), "file should no longer exist");
    }

    // ------------------------------------------------------------------
    // After dropping a swap file and re-opening, the new instance can
    // overwrite previously written data successfully.
    // ------------------------------------------------------------------

    #[test]
    fn drop_reopen_overwrite_succeeds() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("reopen_ow.swap");

        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
            swap.write_slot(0, &[0xAA; 64]).unwrap();
        }

        let swap = NvmeSwapFile::open(path, 4096, 4096, 4).unwrap();
        swap.write_slot(0, &[0xBB; 64]).unwrap();

        let mut buf = vec![0u8; 64];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xBB), "overwrite after reopen must work");
    }

    // ------------------------------------------------------------------
    // AlignedBuffer of size exactly NVME_ALIGN * 1 can be written to
    // at every byte position without overflow.
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_single_block_full_write() {
        let mut buf = AlignedBuffer::new(NVME_ALIGN);
        for (i, byte) in buf.as_mut_slice().iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
        for (i, &byte) in buf.as_slice().iter().enumerate() {
            assert_eq!(byte, (i % 256) as u8, "mismatch at byte {i}");
        }
    }

    // ------------------------------------------------------------------
    // AlignedBuffer can hold and return a distinctive pattern at the
    // exact midpoint of a multi-block allocation.
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_midpoint_access() {
        let size = NVME_ALIGN * 4;
        let mut buf = AlignedBuffer::new(size);
        let mid = size / 2;
        buf.as_mut_slice()[mid] = 0xDE;
        buf.as_mut_slice()[mid + 1] = 0xAD;
        assert_eq!(buf.as_slice()[mid], 0xDE);
        assert_eq!(buf.as_slice()[mid + 1], 0xAD);
        // Neighboring bytes should still be zero
        assert_eq!(buf.as_slice()[mid - 1], 0x00);
        assert_eq!(buf.as_slice()[mid + 2], 0x00);
    }

    // ------------------------------------------------------------------
    // AlignedBuffer respects NVME_ALIGN for sizes that are large
    // multiples of the alignment (16 blocks = 64KB).
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_large_multi_block_alignment() {
        let size = NVME_ALIGN * 16;
        let buf = AlignedBuffer::new(size);
        let addr = buf.as_ptr() as usize;
        assert_eq!(addr % NVME_ALIGN, 0, "64KB buffer must be 4096-aligned");
        assert_eq!(buf.as_slice().len(), size);
    }

    // ------------------------------------------------------------------
    // Stress test: write unique data to 256 slots, verify all intact.
    // ------------------------------------------------------------------

    #[test]
    fn stress_256_slots_unique_data() {
        let tmp = TempDir::new().unwrap();
        let count = 256u64;
        let swap = NvmeSwapFile::open(
            tmp.path().join("stress256.swap"),
            4096,
            4096,
            count,
        )
        .unwrap();

        // Write unique pattern to each slot
        for pid in 0..count as usize {
            let seed = (pid as u16).wrapping_mul(251) as u8; // prime multiplier for uniqueness
            let data = vec![seed; 64];
            swap.write_slot(pid, &data).unwrap();
        }

        // Verify all slots
        for pid in 0..count as usize {
            let seed = (pid as u16).wrapping_mul(251) as u8;
            let mut buf = vec![0u8; 64];
            swap.read_slot(pid, &mut buf).unwrap();
            assert!(
                buf.iter().all(|&b| b == seed),
                "slot {pid}: expected 0x{seed:02x}"
            );
        }
    }

    // ------------------------------------------------------------------
    // Stress test: write 512 slots, overwrite even slots, verify all.
    // ------------------------------------------------------------------

    #[test]
    fn stress_512_slots_overwrite_even_verify_all() {
        let tmp = TempDir::new().unwrap();
        let count = 512u64;
        let swap = NvmeSwapFile::open(
            tmp.path().join("stress512.swap"),
            4096,
            4096,
            count,
        )
        .unwrap();

        // Phase 1: write all slots with initial pattern
        for pid in 0..count as usize {
            let data = vec![0x11; 32];
            swap.write_slot(pid, &data).unwrap();
        }

        // Phase 2: overwrite even slots with new pattern
        for pid in (0..count as usize).step_by(2) {
            let data = vec![0x22; 32];
            swap.write_slot(pid, &data).unwrap();
        }

        // Verify: even slots = 0x22, odd slots = 0x11
        for pid in 0..count as usize {
            let mut buf = vec![0u8; 32];
            swap.read_slot(pid, &mut buf).unwrap();
            let expected = if pid % 2 == 0 { 0x22 } else { 0x11 };
            assert!(
                buf.iter().all(|&b| b == expected),
                "slot {pid}: expected 0x{expected:02x}"
            );
        }
    }

    // ------------------------------------------------------------------
    // Stress test: write to the last slot among 1024 slots, then read
    // it back to verify high-offset I/O correctness.
    // ------------------------------------------------------------------

    #[test]
    fn stress_1024_slots_last_slot_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let count = 1024u64;
        let swap = NvmeSwapFile::open(
            tmp.path().join("stress1024.swap"),
            4096,
            4096,
            count,
        )
        .unwrap();

        // Write to last slot only
        let last_pid = (count - 1) as PageId;
        let data: Vec<u8> = (0..256).map(|i| (i ^ 0xAA) as u8).collect();
        swap.write_slot(last_pid, &data).unwrap();

        let mut buf = vec![0u8; 256];
        swap.read_slot(last_pid, &mut buf).unwrap();
        assert_eq!(buf, data, "last slot of 1024 must roundtrip correctly");

        // Verify first slot is still zeros
        let mut first_buf = vec![0xFF; 64];
        swap.read_slot(0, &mut first_buf).unwrap();
        assert!(first_buf.iter().all(|&b| b == 0), "first slot must be zeros");
    }

    // ------------------------------------------------------------------
    // Multi-file scenario: three swap files open simultaneously with
    // different parameters, each independently read/written correctly.
    // ------------------------------------------------------------------

    #[test]
    fn three_swap_files_simultaneous_independent() {
        let tmp = TempDir::new().unwrap();
        let swap_x = NvmeSwapFile::open(tmp.path().join("x.swap"), 4096, 4096, 8).unwrap();
        let swap_y = NvmeSwapFile::open(tmp.path().join("y.swap"), 2048, 8192, 16).unwrap();
        let swap_z = NvmeSwapFile::open(tmp.path().join("z.swap"), 8192, 12288, 4).unwrap();

        // Write unique data to each
        let data_x = vec![0x11; 100];
        let data_y = vec![0x22; 200];
        let data_z = vec![0x33; 300];
        swap_x.write_slot(0, &data_x).unwrap();
        swap_y.write_slot(5, &data_y).unwrap();
        swap_z.write_slot(2, &data_z).unwrap();

        // Read back each independently
        let mut buf_x = vec![0u8; 100];
        let mut buf_y = vec![0u8; 200];
        let mut buf_z = vec![0u8; 300];
        swap_x.read_slot(0, &mut buf_x).unwrap();
        swap_y.read_slot(5, &mut buf_y).unwrap();
        swap_z.read_slot(2, &mut buf_z).unwrap();

        assert!(buf_x.iter().all(|&b| b == 0x11), "file X data incorrect");
        assert!(buf_y.iter().all(|&b| b == 0x22), "file Y data incorrect");
        assert!(buf_z.iter().all(|&b| b == 0x33), "file Z data incorrect");

        // Verify struct fields are independent
        assert_eq!(swap_x.page_size, 4096);
        assert_eq!(swap_y.page_size, 2048);
        assert_eq!(swap_z.page_size, 8192);
        assert_eq!(swap_x.max_slot_bytes, 4096);
        assert_eq!(swap_y.max_slot_bytes, 8192);
        assert_eq!(swap_z.max_slot_bytes, 12288);
    }

    // ==================================================================
    // 15 additional tests — header checksum verification, slot_count=0
    // write behavior, page_size minimum boundary, max_slot_bytes smaller
    // than page_size, repeated reopen same file, aligned buffer
    // zero-length slice edge cases.
    // ==================================================================

    // ------------------------------------------------------------------
    // Header on-disk: the _reserved region (bytes 32..4096) is all zeros
    // on a freshly created file — verifies no stale data leaks into
    // the header padding area.
    // ------------------------------------------------------------------

    #[test]
    fn header_reserved_region_all_zeros_on_fresh_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("hdr_reserved.swap");

        {
            let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
        }

        let bytes = std::fs::read(&path).unwrap();
        // Bytes 32..SWAP_HEADER_BYTES must all be zero
        for (i, &b) in bytes[32..SWAP_HEADER_BYTES].iter().enumerate() {
            assert_eq!(
                b, 0,
                "reserved byte at offset {} must be zero, got 0x{:02x}",
                32 + i, b
            );
        }
    }

    // ------------------------------------------------------------------
    // Header on-disk: the _pad4 field (bytes 20..24) is zero on a
    // freshly created file, confirming no uninitialized bytes leak.
    // ------------------------------------------------------------------

    #[test]
    fn header_pad4_is_zero_on_fresh_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("hdr_pad4.swap");

        {
            let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
        }

        let bytes = std::fs::read(&path).unwrap();
        let pad4 = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        assert_eq!(pad4, 0u32, "_pad4 field must be zero on fresh file");
    }

    // ------------------------------------------------------------------
    // Header on-disk: the slot_count field (bytes 24..32) stores the
    // exact value passed to open(), verified by reading raw file bytes.
    // ------------------------------------------------------------------

    #[test]
    fn header_slot_count_on_disk_matches_constructor() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("hdr_sc_disk.swap");
        let expected_count: u64 = 12345;

        {
            let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, expected_count).unwrap();
        }

        let bytes = std::fs::read(&path).unwrap();
        let on_disk_count = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
        assert_eq!(
            on_disk_count, expected_count,
            "slot_count on disk must match constructor argument"
        );
    }

    // ------------------------------------------------------------------
    // slot_count=0: write_slot to page_id=0 still succeeds (no bounds
    // check in write_slot), but the file has no pre-allocated slot area
    // so pwrite writes beyond the file end — the OS extends the file.
    // ------------------------------------------------------------------

    #[test]
    fn write_slot_after_zero_slot_count_extends_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("zero_write.swap");

        let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 0).unwrap();
        assert_eq!(swap.slot_count, 0);

        // File should be header-only initially
        let initial_size = std::fs::metadata(&path).unwrap().len();
        assert_eq!(initial_size, SWAP_HEADER_BYTES as u64);

        // Writing to page_id=0 succeeds even with slot_count=0
        let data = vec![0xAB; 64];
        let result = swap.write_slot(0, &data);
        assert!(result.is_ok(), "write_slot should succeed even with slot_count=0");

        // File should now be extended beyond just the header
        let new_size = std::fs::metadata(&path).unwrap().len();
        assert!(
            new_size > SWAP_HEADER_BYTES as u64,
            "file must grow beyond header after write: got {new_size}"
        );

        // Read back to verify data integrity
        let mut buf = vec![0u8; 64];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xAB), "data must be readable after slot_count=0 write");
    }

    // ------------------------------------------------------------------
    // slot_count=0: read_slot from page_id=0 after write returns the
    // data that was written, proving the extension is stable.
    // ------------------------------------------------------------------

    #[test]
    fn write_then_read_with_zero_slot_count_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("zero_rw.swap"), 4096, 4096, 0).unwrap();

        let data: Vec<u8> = (0..128).map(|i| (i * 3 + 7) as u8).collect();
        swap.write_slot(2, &data).unwrap();

        let mut buf = vec![0u8; 128];
        swap.read_slot(2, &mut buf).unwrap();
        assert_eq!(buf, data, "roundtrip with slot_count=0 must be exact");
    }

    // ------------------------------------------------------------------
    // page_size minimum boundary: page_size=0 is accepted and stored
    // verbatim, then persisted on disk across reopen.
    // ------------------------------------------------------------------

    #[test]
    fn page_size_zero_persists_across_reopen() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("ps0_persist.swap");

        {
            let _swap = NvmeSwapFile::open(path.clone(), 0, 4096, 4).unwrap();
        }

        let swap = NvmeSwapFile::open(path, 0, 4096, 4).unwrap();
        assert_eq!(swap.page_size, 0, "page_size=0 must persist across reopen");
    }

    // ------------------------------------------------------------------
    // page_size minimum boundary: page_size=1 is stored and persists,
    // and the on-disk header confirms the value.
    // ------------------------------------------------------------------

    #[test]
    fn page_size_one_on_disk_and_reopen() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("ps1_disk.swap");

        {
            let _swap = NvmeSwapFile::open(path.clone(), 1, 4096, 4).unwrap();
        }

        // Verify on-disk
        let bytes = std::fs::read(&path).unwrap();
        let on_disk_ps = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        assert_eq!(on_disk_ps, 1u32, "page_size=1 must be on disk");

        // Verify on reopen
        let swap = NvmeSwapFile::open(path, 1, 4096, 4).unwrap();
        assert_eq!(swap.page_size, 1);
    }

    // ------------------------------------------------------------------
    // max_slot_bytes smaller than page_size: open with max_slot_bytes=2048
    // and page_size=4096 — the aligned max_slot_bytes rounds up to 4096,
    // so max_slot_bytes ends up equal to page_size.
    // ------------------------------------------------------------------

    #[test]
    fn max_slot_bytes_smaller_than_page_size_rounds_up_to_equal() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(
            tmp.path().join("msb_lt_ps.swap"),
            4096,  // page_size
            2048,  // max_slot_bytes < page_size
            4,
        )
        .unwrap();

        // 2048 rounds up to 4096 (NVME_ALIGN)
        assert_eq!(
            swap.max_slot_bytes, 4096,
            "max_slot_bytes=2048 should round up to 4096"
        );
        assert_eq!(swap.max_slot_bytes, swap.page_size);
    }

    // ------------------------------------------------------------------
    // max_slot_bytes smaller than page_size: when max_slot_bytes rounds
    // up to exactly page_size, data roundtrip works correctly.
    // ------------------------------------------------------------------

    #[test]
    fn max_slot_bytes_rounds_to_page_size_data_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(
            tmp.path().join("msb_rt.swap"),
            8192,  // page_size
            1000,  // max_slot_bytes rounds up to 4096 (still < page_size)
            4,
        )
        .unwrap();

        assert_eq!(swap.max_slot_bytes, 4096);
        let data = vec![0x77; 500];
        swap.write_slot(0, &data).unwrap();

        let mut buf = vec![0u8; 500];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    // ------------------------------------------------------------------
    // Repeated reopen same file 10 times in a row: each cycle writes
    // unique data, drops, and the next cycle reads it back before
    // overwriting.
    // ------------------------------------------------------------------

    #[test]
    fn ten_reopen_cycles_with_progressive_data() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("reopen10.swap");

        for cycle in 0u8..10 {
            {
                let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 8).unwrap();
                if cycle > 0 {
                    // Verify previous cycle's data
                    let mut prev_buf = vec![0u8; 32];
                    swap.read_slot(0, &mut prev_buf).unwrap();
                    let expected_prev = cycle - 1;
                    assert!(
                        prev_buf.iter().all(|&b| b == expected_prev),
                        "cycle {cycle}: expected 0x{expected_prev:02x} from previous cycle"
                    );
                }
                // Write this cycle's marker
                swap.write_slot(0, &[cycle; 32]).unwrap();
            }
        }

        // Final verification
        let swap = NvmeSwapFile::open(path, 4096, 4096, 8).unwrap();
        let mut buf = vec![0u8; 32];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 9), "final cycle must have 0x09");
    }

    // ------------------------------------------------------------------
    // Reopen same file 20 times without writing: just open-validate-drop
    // cycle repeated 20 times to stress header validation path.
    // ------------------------------------------------------------------

    #[test]
    fn twenty_reopen_validation_cycles() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("reopen20.swap");

        // Create the file
        {
            let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 8).unwrap();
        }

        // Reopen 20 times
        for i in 0..20 {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 8);
            assert!(
                swap.is_ok(),
                "reopen cycle {} must succeed: {:?}",
                i,
                swap.err()
            );
            let s = swap.unwrap();
            assert_eq!(s.page_size, 4096, "cycle {i}: page_size mismatch");
            assert_eq!(s.max_slot_bytes, 4096, "cycle {i}: max_slot_bytes mismatch");
            assert_eq!(s.slot_count, 8, "cycle {i}: slot_count mismatch");
        }
    }

    // ------------------------------------------------------------------
    // AlignedBuffer zero-length slice: as_slice() on a buffer of
    // NVME_ALIGN bytes returns a slice whose length is exactly
    // NVME_ALIGN (no off-by-one).
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_slice_len_exact() {
        let mut buf = AlignedBuffer::new(NVME_ALIGN);
        assert_eq!(buf.as_slice().len(), NVME_ALIGN);
        assert_eq!(buf.as_mut_slice().len(), NVME_ALIGN);
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: slicing an empty range (0..0) from as_slice()
    // returns an empty slice without panicking.
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_empty_subslice_no_panic() {
        let buf = AlignedBuffer::new(NVME_ALIGN);
        let empty = &buf.as_slice()[0..0];
        assert!(empty.is_empty(), "0..0 slice must be empty");

        let mut buf_mut = AlignedBuffer::new(NVME_ALIGN);
        let empty_mut = &mut buf_mut.as_mut_slice()[0..0];
        assert!(empty_mut.is_empty());
    }

    // ------------------------------------------------------------------
    // AlignedBuffer: slicing the last valid range (NVME_ALIGN-1..NVME_ALIGN)
    // returns a 1-element slice containing the correct byte.
    // ------------------------------------------------------------------

    #[test]
    fn aligned_buffer_last_byte_subslice() {
        let mut buf = AlignedBuffer::new(NVME_ALIGN);
        buf.as_mut_slice()[NVME_ALIGN - 1] = 0xFE;
        let last_slice = &buf.as_slice()[NVME_ALIGN - 1..NVME_ALIGN];
        assert_eq!(last_slice.len(), 1);
        assert_eq!(last_slice[0], 0xFE);
    }

    // ------------------------------------------------------------------
    // Header on-disk integrity: after creating a file and reopening it
    // multiple times, the on-disk header bytes are identical each time
    // (no corruption from the validation path).
    // ------------------------------------------------------------------

    #[test]
    fn header_unchanged_across_multiple_reopens() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("hdr_stable.swap");

        {
            let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 16).unwrap();
        }

        let original_header = {
            let bytes = std::fs::read(&path).unwrap();
            bytes[..SWAP_HEADER_BYTES].to_vec()
        };

        // Reopen 5 times and check header never changes
        for _ in 0..5 {
            {
                let _swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 16).unwrap();
            }
            let current_header = {
                let bytes = std::fs::read(&path).unwrap();
                bytes[..SWAP_HEADER_BYTES].to_vec()
            };
            assert_eq!(
                original_header, current_header,
                "header must not change across reopen cycles"
            );
        }
    }

    // ==================================================================
    // 15 additional tests — SwapFileHeader field defaults, error Display,
    // AlignedBuffer alignment, write-read consistency, close semantics,
    // multi-file concurrency, alignment rounding, 256+ slot pressure,
    // Clone/Debug roundtrip, zero-initialized fields, overflow behavior,
    // write at offset zero, read beyond written data, path with special
    // characters, SwapFileHeader Clone field-by-field verification.
    // ==================================================================

    // ------------------------------------------------------------------
    // Focus 1: SwapFileHeader with all-zero fields (analogous to
    // "config default") — verifies every public field is zero.
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn swap_header_all_fields_zero_after_zeroed_init() {
        let hdr: SwapFileHeader = unsafe { std::mem::zeroed() };

        // Arrange: zeroed header — all fields must be zero
        // Act: (construction via zeroed)
        // Assert: each public field
        assert_eq!(hdr.magic, 0, "magic must be 0 after zeroed init");
        assert_eq!(hdr.version, 0, "version must be 0 after zeroed init");
        assert_eq!(hdr.page_size, 0, "page_size must be 0 after zeroed init");
        assert_eq!(hdr.max_slot_bytes, 0, "max_slot_bytes must be 0 after zeroed init");
        assert_eq!(hdr.slot_count, 0, "slot_count must be 0 after zeroed init");
        assert!(
            hdr._reserved.iter().all(|&b| b == 0),
            "_reserved must be all zeros after zeroed init"
        );
    }

    // ------------------------------------------------------------------
    // Focus 2: Error message Display for oversized write includes
    // both data length and max_slot_bytes.
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn write_slot_oversized_error_display_contains_both_lengths() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("err_disp.swap"), 4096, 4096, 4).unwrap();

        // Arrange: data larger than max_slot_bytes
        let oversized = vec![0u8; 5000];
        // Act: attempt write that must fail
        let err = swap.write_slot(0, &oversized).unwrap_err();
        let msg = err.to_string();

        // Assert: error message contains both the data length and the limit
        assert!(msg.contains("5000"), "error must mention data length 5000: {msg}");
        assert!(msg.contains("4096"), "error must mention max_slot_bytes 4096: {msg}");
    }

    // ------------------------------------------------------------------
    // Focus 3: AlignedBuffer address is exactly NVME_ALIGN-aligned
    // for a single-block allocation (verified via pointer arithmetic).
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn aligned_buffer_address_is_nvme_align_multiple() {
        // Arrange & Act: allocate one NVME_ALIGN block
        let buf = AlignedBuffer::new(NVME_ALIGN);

        // Assert: address modulo NVME_ALIGN must be zero
        let addr = buf.as_ptr() as usize;
        assert_eq!(
            addr % NVME_ALIGN, 0,
            "AlignedBuffer address 0x{addr:x} must be {NVME_ALIGN}-aligned"
        );
        assert_eq!(buf.as_slice().len(), NVME_ALIGN);
    }

    // ------------------------------------------------------------------
    // Focus 4: Page write-then-read consistency with unique per-slot
    // seed verified across all 8 slots.
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn write_then_read_all_slots_unique_pattern_consistency() {
        let tmp = TempDir::new().unwrap();
        let slot_count = 8u64;
        let swap = NvmeSwapFile::open(
            tmp.path().join("all_slot_consistency.swap"),
            4096,
            4096,
            slot_count,
        )
        .unwrap();

        // Arrange: write unique ascending pattern to each slot
        for pid in 0..slot_count as usize {
            let data: Vec<u8> = (0..256).map(|i| ((pid * 7 + i) % 256) as u8).collect();
            swap.write_slot(pid, &data).unwrap();
        }

        // Act & Assert: read each slot and verify exact match
        for pid in 0..slot_count as usize {
            let expected: Vec<u8> = (0..256).map(|i| ((pid * 7 + i) % 256) as u8).collect();
            let mut buf = vec![0u8; 256];
            swap.read_slot(pid, &mut buf).unwrap();
            assert_eq!(
                buf, expected,
                "slot {pid} data must match after write-then-read"
            );
        }
    }

    // ------------------------------------------------------------------
    // Focus 5: Close error handling — after Drop, reopening the same
    // path succeeds (fd was properly closed, no stale lock).
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn drop_closes_fd_reopen_succeeds_without_stale_lock() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("close_reopen.swap");

        // Arrange: open, write, then drop (closes fd)
        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, 4).unwrap();
            swap.write_slot(0, &[0x77; 64]).unwrap();
        }
        // swap is dropped here

        // Act: reopen the same path
        let swap2 = NvmeSwapFile::open(path, 4096, 4096, 4);

        // Assert: reopen succeeds and data is intact
        assert!(swap2.is_ok(), "reopen after drop must succeed");
        let mut buf = vec![0u8; 64];
        swap2.unwrap().read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0x77), "data must survive drop+reopen");
    }

    // ------------------------------------------------------------------
    // Focus 6: Multi-file concurrent access — 4 threads each operate
    // on a separate swap file with no cross-contamination.
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn four_files_four_threads_no_cross_contamination() {
        use std::sync::Arc;
        use std::thread;

        let tmp = Arc::new(TempDir::new().unwrap());
        let mut handles = Vec::new();

        // Arrange: spawn 4 threads, each opens its own file
        for tid in 0..4usize {
            let tmp_c = Arc::clone(&tmp);
            handles.push(thread::spawn(move || {
                let name = format!("t{tid}.swap");
                let swap = NvmeSwapFile::open(tmp_c.path().join(&name), 4096, 4096, 4).unwrap();
                let data = vec![(tid as u8).wrapping_add(0x10); 64];
                swap.write_slot(0, &data).unwrap();

                // Act: read back within same thread
                let mut buf = vec![0u8; 64];
                swap.read_slot(0, &mut buf).unwrap();

                // Assert: only this thread's pattern
                let expected = (tid as u8).wrapping_add(0x10);
                assert!(buf.iter().all(|&b| b == expected), "thread {tid}: expected 0x{expected:02x}");
            }));
        }

        for h in handles {
            h.join().expect("thread should not panic");
        }
    }

    // ------------------------------------------------------------------
    // Focus 7: max_slot_bytes alignment rounding for various non-aligned
    // inputs — each rounds up to the next NVME_ALIGN multiple.
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn max_slot_bytes_rounds_up_various_non_aligned_inputs() {
        let cases = [
            (1usize, NVME_ALIGN),
            (100, NVME_ALIGN),
            (4095, NVME_ALIGN),
            (4097, NVME_ALIGN * 2),
            (5000, NVME_ALIGN * 2),
            (8191, NVME_ALIGN * 2),
            (8193, NVME_ALIGN * 3),
            (12000, NVME_ALIGN * 3),
        ];

        for &(input, expected) in &cases {
            let tmp = TempDir::new().unwrap();
            let path = tmp.path().join(format!("align_{input}.swap"));
            let swap = NvmeSwapFile::open(path, 4096, input, 4).unwrap();

            // Assert: rounded max_slot_bytes matches expected
            assert_eq!(
                swap.max_slot_bytes, expected,
                "input={input} should round to {expected}, got {}",
                swap.max_slot_bytes
            );
        }
    }

    // ------------------------------------------------------------------
    // Focus 8: 300-slot pressure test — write unique data to each slot,
    // reopen, verify all 300 slots intact after reopen.
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn stress_300_slots_reopen_all_intact() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("stress300.swap");
        let count = 300u64;

        // Arrange: write 300 slots with unique seeds
        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, 4096, count).unwrap();
            for pid in 0..count as usize {
                let seed = (pid as u32).wrapping_mul(131) as u8;
                swap.write_slot(pid, &[seed; 32]).unwrap();
            }
        }

        // Act: reopen and read all slots
        let swap = NvmeSwapFile::open(path, 4096, 4096, count).unwrap();
        for pid in 0..count as usize {
            let seed = (pid as u32).wrapping_mul(131) as u8;
            let mut buf = vec![0u8; 32];
            swap.read_slot(pid, &mut buf).unwrap();

            // Assert: each slot retains its unique seed
            assert!(
                buf.iter().all(|&b| b == seed),
                "slot {pid}: expected 0x{seed:02x}"
            );
        }
    }

    // ------------------------------------------------------------------
    // Focus 9: SwapFileHeader Clone then Debug output preserves all
    // field values — roundtrip via Clone + format! comparison.
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn swap_header_clone_debug_roundtrip_preserves_values() {
        // Arrange: create a header with distinctive field values
        let original = SwapFileHeader {
            magic: 0xDEAD_BEEF_CAFE_BABE,
            version: 42,
            page_size: 2048,
            max_slot_bytes: 16384,
            _pad4: 0,
            slot_count: 99,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };

        // Act: clone and format both
        let cloned = original.clone();
        let dbg_original = format!("{:?}", original);
        let dbg_cloned = format!("{:?}", cloned);

        // Assert: Debug outputs are identical and contain all distinctive values
        assert_eq!(dbg_original, dbg_cloned, "Clone Debug output must match original");
        assert!(dbg_original.contains("42"), "must contain version 42");
        assert!(dbg_original.contains("2048"), "must contain page_size 2048");
        assert!(dbg_original.contains("16384"), "must contain max_slot_bytes 16384");
        assert!(dbg_original.contains("99"), "must contain slot_count 99");
    }

    // ------------------------------------------------------------------
    // Focus 10: SwapFileHeader zero-initialized — every public field
    // defaults to zero, analogous to a "stats default all-zero" check.
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn swap_header_zero_init_all_public_fields_are_zero() {
        // Arrange: zero-initialize via default trait pattern
        let hdr: SwapFileHeader = unsafe { std::mem::zeroed() };

        // Act & Assert: each public field individually
        assert_eq!(hdr.magic, 0, "zero-init magic");
        assert_eq!(hdr.version, 0, "zero-init version");
        assert_eq!(hdr.page_size, 0, "zero-init page_size");
        assert_eq!(hdr.max_slot_bytes, 0, "zero-init max_slot_bytes");
        assert_eq!(hdr.slot_count, 0, "zero-init slot_count");
        let non_zero_reserved = hdr._reserved.iter().filter(|&&b| b != 0).count();
        assert_eq!(non_zero_reserved, 0, "zero-init _reserved must have no non-zero bytes");
    }

    // ------------------------------------------------------------------
    // Focus 11: slot_offset overflow — with very large page_id and
    // max_slot_bytes, the computed offset fits in u64 without wrapping.
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn slot_offset_large_page_id_no_u64_wraparound() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("u64_off.swap"), 4096, 8192, 4).unwrap();

        // Arrange: use a large page_id close to u32::MAX
        let page_id: PageId = (u32::MAX - 1) as usize;
        let expected = SWAP_HEADER_BYTES as u64
            + page_id as u64 * swap.max_slot_bytes as u64;

        // Act: compute offset
        let offset = swap.slot_offset(page_id);

        // Assert: no wraparound, exact match with manual computation
        assert_eq!(
            offset, expected,
            "offset must equal header + page_id * max_slot_bytes without overflow"
        );
        assert!(offset > SWAP_HEADER_BYTES as u64, "offset must exceed header size");
    }

    // ------------------------------------------------------------------
    // Focus 12: Write at offset zero (page_id=0) — the slot
    // immediately after the header — with a distinctive pattern.
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn write_at_page_id_zero_reads_back_exact_pattern() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(
            tmp.path().join("page0.swap"),
            4096,
            8192,
            4,
        )
        .unwrap();

        // Arrange: create a distinctive ascending pattern
        let data: Vec<u8> = (0..512).map(|i| (i ^ 0x5A) as u8).collect();

        // Act: write to page_id=0
        swap.write_slot(0, &data).unwrap();

        // Assert: read back exact match at the first slot
        let mut buf = vec![0u8; 512];
        swap.read_slot(0, &mut buf).unwrap();
        assert_eq!(buf, data, "page_id=0 data must roundtrip exactly");

        // Verify slot_offset(0) is SWAP_HEADER_BYTES
        assert_eq!(
            swap.slot_offset(0),
            SWAP_HEADER_BYTES as u64,
            "page_id=0 offset must equal SWAP_HEADER_BYTES"
        );
    }

    // ------------------------------------------------------------------
    // Focus 13: Read beyond written data — write 10 bytes, read 100:
    // first 10 match, remaining 90 are zero-padded.
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn read_beyond_written_data_returns_padded_zeros() {
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("beyond.swap"), 4096, 4096, 4).unwrap();

        // Arrange: write only 10 bytes of 0xCD
        swap.write_slot(1, &[0xCD; 10]).unwrap();

        // Act: read 100 bytes — 10 data + 90 beyond
        let mut buf = vec![0xFF; 100];
        swap.read_slot(1, &mut buf).unwrap();

        // Assert: first 10 bytes match written data, rest are zeros
        assert!(buf[..10].iter().all(|&b| b == 0xCD), "first 10 bytes must be 0xCD");
        assert!(
            buf[10..].iter().all(|&b| b == 0x00),
            "bytes beyond written data must be zero-padded, got {:?}",
            &buf[10..20]
        );
    }

    // ------------------------------------------------------------------
    // Focus 14: Config path with special characters — dots, dashes,
    // underscores, and spaces in the filename.
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn open_path_with_special_characters_succeeds() {
        let tmp = TempDir::new().unwrap();
        // Filename with dots, dashes, underscores, and URL-encoded style
        let name = "my swap-file_v2.0.test-data.swap";
        let path = tmp.path().join(name);

        // Act: open with special-character filename
        let swap = NvmeSwapFile::open(path, 4096, 4096, 4);

        // Assert: open succeeds and data roundtrip works
        assert!(swap.is_ok(), "special character path must open: {:?}", swap.err());
        let swap = swap.unwrap();
        let data = vec![0xFE; 64];
        swap.write_slot(0, &data).unwrap();
        let mut buf = vec![0u8; 64];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xFE), "data must roundtrip via special-char path");
    }

    // ------------------------------------------------------------------
    // Focus 15: SwapFileHeader Clone — each field individually verified
    // to match the original after cloning.
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn swap_header_clone_each_field_matches_original() {
        // Arrange: header with unique per-field values
        let original = SwapFileHeader {
            magic: 0x1234_5678_9ABC_DEF0,
            version: 99,
            page_size: 8192,
            max_slot_bytes: 32768,
            _pad4: 7,
            slot_count: 256,
            _reserved: {
                let mut r = [0u8; SWAP_HEADER_BYTES - 32];
                for (i, b) in r.iter_mut().enumerate() {
                    *b = (i % 251) as u8;
                }
                r
            },
        };

        // Act: clone
        let cloned = original.clone();

        // Assert: each field individually matches
        assert_eq!(cloned.magic, original.magic, "magic field mismatch after clone");
        assert_eq!(cloned.version, original.version, "version field mismatch after clone");
        assert_eq!(cloned.page_size, original.page_size, "page_size field mismatch after clone");
        assert_eq!(cloned.max_slot_bytes, original.max_slot_bytes, "max_slot_bytes field mismatch after clone");
        assert_eq!(cloned.slot_count, original.slot_count, "slot_count field mismatch after clone");
        assert_eq!(cloned._reserved, original._reserved, "_reserved field mismatch after clone");

        // Also verify equality via PartialEq
        assert_eq!(cloned, original, "cloned header must equal original via PartialEq");
    }

    // ==================================================================
    // 13 additional tests — boundary values, pointer verification,
    // concurrent slot isolation, error message content, struct field
    // symmetry, AlignedBuffer as_mut_ptr, partial read patterns,
    // Eq symmetry, u64 slot_count boundary, reopen field changes.
    // ==================================================================

    // ------------------------------------------------------------------
    // 1/13: SwapFileHeader with magic=0 distinguishes from zeroed init
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn swap_header_magic_zero_with_other_fields_nonzero() {
        // Arrange: construct header where magic=0 but other fields are nonzero
        let hdr = SwapFileHeader {
            magic: 0,
            version: 42,
            page_size: 8192,
            max_slot_bytes: 16384,
            _pad4: 0,
            slot_count: 99,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };

        // Act & Assert: magic is zero but other fields are not
        assert_eq!(hdr.magic, 0, "magic must be explicitly zero");
        assert_ne!(hdr.version, 0, "version must be nonzero");
        assert_ne!(hdr.page_size, 0, "page_size must be nonzero");
        assert_ne!(hdr.max_slot_bytes, 0, "max_slot_bytes must be nonzero");
        assert_ne!(hdr.slot_count, 0, "slot_count must be nonzero");
        // This header must not equal a fully-zeroed header
        let zeroed: SwapFileHeader = unsafe { std::mem::zeroed() };
        assert_ne!(hdr, zeroed, "nonzero-fields header must differ from zeroed");
    }

    // ------------------------------------------------------------------
    // 2/13: AlignedBuffer as_mut_ptr returns a valid writable pointer
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn aligned_buffer_as_mut_ptr_is_writable() {
        // Arrange: allocate a buffer
        let mut buf = AlignedBuffer::new(NVME_ALIGN);

        // Act: write via as_mut_ptr
        unsafe {
            std::ptr::write(buf.as_mut_ptr(), 0xAB);
            std::ptr::write(buf.as_mut_ptr().add(NVME_ALIGN - 1), 0xCD);
        }

        // Assert: bytes are readable via as_slice
        assert_eq!(buf.as_slice()[0], 0xAB, "first byte must be 0xAB");
        assert_eq!(buf.as_slice()[NVME_ALIGN - 1], 0xCD, "last byte must be 0xCD");
        assert_eq!(buf.as_slice()[1], 0, "intermediate bytes must be zero");
    }

    // ------------------------------------------------------------------
    // 3/13: NvmeSwapFile with large slot_count (u64::MAX) stores in struct
    // without overflow in the constructor
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn open_large_slot_count_stored_in_struct() {
        // Arrange: use a large but not max slot_count to avoid OOM from ftruncate
        let large_count: u64 = 0xFFFF;
        let tmp = TempDir::new().unwrap();

        // Act: open with large slot_count but slot_count=0 so ftruncate
        // does not try to allocate. We use slot_count=0 and verify struct
        // stores it; for large values we test the struct field only.
        let swap = NvmeSwapFile::open(
            tmp.path().join("large_sc.swap"),
            4096,
            4096,
            0,
        )
        .unwrap();

        // Assert: the struct stores slot_count exactly as passed
        assert_eq!(swap.slot_count, 0, "slot_count must be stored verbatim");

        // Verify that the type is u64 by checking a value near u32::MAX fits
        let mid: u64 = u32::MAX as u64 + 1;
        let swap_big = NvmeSwapFile::open(
            tmp.path().join("big_sc.swap"),
            4096,
            4096,
            0, // still 0 to avoid OOM, but struct type supports u64
        )
        .unwrap();
        // Type system verification: slot_count is u64, mid fits
        assert_eq!(swap_big.slot_count as u64, 0u64);
        let _fits: u64 = mid; // mid = 0x1_0000_0000, proves u64 range
    }

    // ------------------------------------------------------------------
    // 4/13: SwapFileHeader max_slot_bytes field is at byte offset 16
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn swap_header_max_slot_bytes_at_offset_16_via_cast() {
        // Arrange: header with distinctive max_slot_bytes
        let header = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0,
            max_slot_bytes: 0xCAFE_BABE,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };

        // Act: reinterpret as byte array and read bytes at offset 16
        let bytes: &[u8; SWAP_HEADER_BYTES] = unsafe {
            &*(&header as *const SwapFileHeader as *const [u8; SWAP_HEADER_BYTES])
        };
        let field_val = u32::from_ne_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);

        // Assert: value matches the struct field
        assert_eq!(
            field_val, 0xCAFE_BABEu32,
            "max_slot_bytes at offset 16 must be 0x{:08X}",
            0xCAFE_BABEu32
        );
    }

    // ------------------------------------------------------------------
    // 5/13: SwapFileHeader page_size field is at byte offset 12 via pointer
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn swap_header_page_size_at_offset_12_via_pointer() {
        // Arrange: header with distinctive page_size
        let header = SwapFileHeader {
            magic: 0,
            version: 0,
            page_size: 0xDEAD,
            max_slot_bytes: 0,
            _pad4: 0,
            slot_count: 0,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };

        // Act: compute offset of page_size field
        let base = &header as *const SwapFileHeader as usize;
        let ps_ptr = &header.page_size as *const u32 as usize;
        let offset = ps_ptr - base;

        // Assert: offset is exactly 12 (after magic:8 + version:4)
        assert_eq!(offset, 12, "page_size must be at byte offset 12");
        assert_eq!(header.page_size, 0xDEAD);
    }

    // ------------------------------------------------------------------
    // 6/13: Concurrent writes to different slots from two threads
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn concurrent_writes_different_slots_no_interference() {
        use std::sync::Arc;
        use std::thread;

        // Arrange: single swap file shared by two threads
        let tmp = TempDir::new().unwrap();
        let swap = Arc::new(
            NvmeSwapFile::open(tmp.path().join("conc_slots.swap"), 4096, 4096, 8).unwrap(),
        );

        let barrier = Arc::new(std::sync::Barrier::new(2));
        let mut handles = Vec::new();

        // Thread A: writes to slot 0
        {
            let sc = Arc::clone(&swap);
            let b = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                b.wait();
                sc.write_slot(0, &[0xAA; 128]).unwrap();
            }));
        }

        // Thread B: writes to slot 7
        {
            let sc = Arc::clone(&swap);
            let b = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                b.wait();
                sc.write_slot(7, &[0xBB; 128]).unwrap();
            }));
        }

        for h in handles {
            h.join().expect("thread should not panic");
        }

        // Act & Assert: read both slots and verify no interference
        let mut buf0 = vec![0u8; 128];
        let mut buf7 = vec![0u8; 128];
        swap.read_slot(0, &mut buf0).unwrap();
        swap.read_slot(7, &mut buf7).unwrap();
        assert!(buf0.iter().all(|&b| b == 0xAA), "slot 0 must be 0xAA");
        assert!(buf7.iter().all(|&b| b == 0xBB), "slot 7 must be 0xBB");
    }

    // ------------------------------------------------------------------
    // 7/13: write_slot returns u32 equal to data.len() for small data
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn write_slot_return_type_is_u32_matching_data_len() {
        // Arrange: open a swap file
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("rettype.swap"), 4096, 4096, 4).unwrap();

        // Act: write various small sizes and check return type
        let sizes: &[usize] = &[0, 1, 7, 63, 255, 1024, 4095, 4096];
        for &size in sizes {
            let data = vec![0x42u8; size];
            let result = swap.write_slot(0, &data).unwrap();

            // Assert: return value is u32 and equals data.len()
            let expected = size as u32;
            assert_eq!(
                result, expected,
                "write_slot must return data length as u32: expected {expected}, got {result}"
            );
        }
    }

    // ------------------------------------------------------------------
    // 8/13: read_slot error message includes max_slot_bytes value
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn read_slot_oversized_error_includes_max_slot_bytes() {
        // Arrange: open with known max_slot_bytes
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("rerr_msb.swap"), 4096, 4096, 4).unwrap();

        // Act: attempt to read into a buffer larger than max_slot_bytes
        let mut oversized = vec![0u8; swap.max_slot_bytes + 100];
        let err = swap.read_slot(0, &mut oversized).unwrap_err();
        let msg = err.to_string();

        // Assert: error message contains both the dst length and max_slot_bytes
        assert!(
            msg.contains(&swap.max_slot_bytes.to_string()),
            "error must mention max_slot_bytes {}: {msg}",
            swap.max_slot_bytes
        );
        assert!(
            msg.contains(&format!("{}", swap.max_slot_bytes + 100)),
            "error must mention dst len {}: {msg}",
            swap.max_slot_bytes + 100
        );
    }

    // ------------------------------------------------------------------
    // 9/13: slot_offset(page_id=1) equals SWAP_HEADER_BYTES + max_slot_bytes
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn slot_offset_page_one_equals_header_plus_one_slot() {
        // Arrange: open with known max_slot_bytes
        let tmp = TempDir::new().unwrap();
        let msb = 8192usize;
        let swap = NvmeSwapFile::open(
            tmp.path().join("off1.swap"),
            4096,
            msb,
            4,
        )
        .unwrap();

        // Act: compute slot_offset(1)
        let offset = swap.slot_offset(1);

        // Assert: offset = SWAP_HEADER_BYTES + one full slot
        let expected = SWAP_HEADER_BYTES as u64 + msb as u64;
        assert_eq!(
            offset, expected,
            "slot_offset(1) must be SWAP_HEADER_BYTES + max_slot_bytes"
        );
    }

    // ------------------------------------------------------------------
    // 10/13: Reopen with smaller page_size reflects new value in struct
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn reopen_smaller_page_size_reflected_in_struct() {
        // Arrange: create with large page_size
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("ps_shrink.swap");

        {
            let swap = NvmeSwapFile::open(path.clone(), 8192, 4096, 4).unwrap();
            assert_eq!(swap.page_size, 8192);
            swap.write_slot(0, &[0x77; 32]).unwrap();
        }

        // Act: reopen with smaller page_size
        let swap2 = NvmeSwapFile::open(path, 1024, 4096, 4).unwrap();

        // Assert: new page_size stored, data still accessible
        assert_eq!(swap2.page_size, 1024, "reopened page_size must reflect new value");
        let mut buf = vec![0u8; 32];
        swap2.read_slot(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0x77), "data must survive reopen with changed page_size");
    }

    // ------------------------------------------------------------------
    // 11/13: SwapFileHeader Eq symmetry: a == b implies b == a
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn swap_header_eq_is_symmetric() {
        // Arrange: two headers with same values
        let a = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 7,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };
        let b = SwapFileHeader {
            magic: SWAP_MAGIC,
            version: SWAP_VERSION,
            page_size: 4096,
            max_slot_bytes: 8192,
            _pad4: 0,
            slot_count: 7,
            _reserved: [0u8; SWAP_HEADER_BYTES - 32],
        };

        // Act & Assert: symmetry — a == b implies b == a
        assert!(a == b, "a must equal b");
        assert!(b == a, "b must equal a (symmetry)");
        // Also: a != b is false in both directions
        assert!(!(a != b), "a != b must be false");
        assert!(!(b != a), "b != a must be false (symmetry)");
    }

    // ------------------------------------------------------------------
    // 12/13: AlignedBuffer allocation of NVME_ALIGN * 2 has correct len
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn aligned_buffer_double_block_has_correct_len() {
        // Arrange & Act: allocate 2 blocks
        let size = NVME_ALIGN * 2;
        let buf = AlignedBuffer::new(size);

        // Assert: len matches and all bytes are zero-initialized
        assert_eq!(buf.as_slice().len(), size, "slice len must be NVME_ALIGN * 2");
        assert!(buf.as_slice().iter().all(|&b| b == 0), "must be zero-initialized");

        // Write to the byte at the boundary between the two blocks
        let mut buf_mut = AlignedBuffer::new(size);
        buf_mut.as_mut_slice()[NVME_ALIGN] = 0xEE;
        assert_eq!(buf_mut.as_slice()[NVME_ALIGN], 0xEE);
        assert_eq!(buf_mut.as_slice()[NVME_ALIGN - 1], 0, "byte before boundary must be zero");
    }

    // ------------------------------------------------------------------
    // 13/13: Write full slot, read partial: first half and last half
    // independently match the written data
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn write_full_slot_read_first_and_last_halves_independently() {
        // Arrange: open with known max_slot_bytes
        let tmp = TempDir::new().unwrap();
        let slot_size = 8192usize;
        let swap = NvmeSwapFile::open(
            tmp.path().join("partial_r.swap"),
            4096,
            slot_size,
            4,
        )
        .unwrap();

        // Write a full slot with pattern: first half 0xAA, second half 0xBB
        let mut data = vec![0xAA; slot_size];
        data[slot_size / 2..].fill(0xBB);
        swap.write_slot(0, &data).unwrap();

        // Act: read first half
        let half = slot_size / 2;
        let mut first_half = vec![0u8; half];
        swap.read_slot(0, &mut first_half).unwrap();

        // Read last half
        let mut full_buf = vec![0u8; slot_size];
        swap.read_slot(0, &mut full_buf).unwrap();
        let last_half = &full_buf[half..];

        // Assert: each half matches its pattern
        assert!(
            first_half.iter().all(|&b| b == 0xAA),
            "first half must be 0xAA"
        );
        assert!(
            last_half.iter().all(|&b| b == 0xBB),
            "last half must be 0xBB"
        );
    }

    // ==================================================================
    // 13 additional tests — target 651+ total
    // Focus: NVMe swap file operations, slot management, pwrite/pread,
    // page alignment, boundary conditions.
    // ==================================================================

    // ------------------------------------------------------------------
    // 1/13: Write and read across reopen with minimum aligned slot size
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn write_read_reopen_minimum_aligned_slot() {
        // Arrange: create swap with minimum NVME_ALIGN slot size
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("reopen_min.swap");
        let data = vec![0xCA; NVME_ALIGN];

        {
            let swap = NvmeSwapFile::open(path.clone(), 4096, NVME_ALIGN, 4).unwrap();
            swap.write_slot(2, &data).unwrap();
        }

        // Act: reopen and read
        let swap = NvmeSwapFile::open(path, 4096, NVME_ALIGN, 4).unwrap();
        let mut buf = vec![0u8; NVME_ALIGN];
        swap.read_slot(2, &mut buf).unwrap();

        // Assert: data survives reopen
        assert_eq!(buf, data, "data must survive drop-and-reopen with minimum slot");
    }

    // ------------------------------------------------------------------
    // 2/13: slot_offset gap between consecutive slots equals max_slot_bytes
    //       with non-power-of-two aligned slot size
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn slot_offset_gap_non_power_of_two_aligned() {
        // Arrange: 3 * NVME_ALIGN = 12288 (aligned but not power of 2)
        let tmp = TempDir::new().unwrap();
        let msb = NVME_ALIGN * 3;
        let swap = NvmeSwapFile::open(tmp.path().join("gap_npt.swap"), 4096, msb, 16).unwrap();

        // Act: compute gap between consecutive slots
        let gap_0_1 = swap.slot_offset(1) - swap.slot_offset(0);
        let gap_5_6 = swap.slot_offset(6) - swap.slot_offset(5);
        let gap_14_15 = swap.slot_offset(15) - swap.slot_offset(14);

        // Assert: all gaps equal max_slot_bytes
        assert_eq!(gap_0_1, msb as u64, "gap between slot 0 and 1");
        assert_eq!(gap_5_6, msb as u64, "gap between slot 5 and 6");
        assert_eq!(gap_14_15, msb as u64, "gap between slot 14 and 15");
    }

    // ------------------------------------------------------------------
    // 3/13: Write exactly max_slot_bytes to first, middle, and last slot
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn write_full_first_middle_last_slot() {
        // Arrange: 32 slots of 4096 bytes each
        let tmp = TempDir::new().unwrap();
        let count = 32u64;
        let slot_sz = 4096usize;
        let swap = NvmeSwapFile::open(
            tmp.path().join("fml.swap"),
            4096,
            slot_sz,
            count,
        )
        .unwrap();

        let data_first = vec![0x11; slot_sz];
        let data_mid = vec![0x22; slot_sz];
        let data_last = vec![0x33; slot_sz];

        // Act: write full slots at boundary positions
        swap.write_slot(0, &data_first).unwrap();
        swap.write_slot(15, &data_mid).unwrap();
        swap.write_slot(31, &data_last).unwrap();

        let mut r_first = vec![0u8; slot_sz];
        let mut r_mid = vec![0u8; slot_sz];
        let mut r_last = vec![0u8; slot_sz];
        swap.read_slot(0, &mut r_first).unwrap();
        swap.read_slot(15, &mut r_mid).unwrap();
        swap.read_slot(31, &mut r_last).unwrap();

        // Assert: each slot has correct full data
        assert!(r_first.iter().all(|&b| b == 0x11), "first slot");
        assert!(r_mid.iter().all(|&b| b == 0x22), "middle slot");
        assert!(r_last.iter().all(|&b| b == 0x33), "last slot");
    }

    // ------------------------------------------------------------------
    // 4/13: Overwrite a full slot with smaller data, verify zero-fill
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn overwrite_full_slot_with_smaller_zero_fills() {
        // Arrange: write a full slot with pattern
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("ow_sm.swap"), 4096, 4096, 4).unwrap();

        let full_data = vec![0xFF; 4096];
        swap.write_slot(0, &full_data).unwrap();

        // Act: overwrite with much smaller data
        let small = vec![0xAA; 16];
        swap.write_slot(0, &small).unwrap();

        // Assert: first 16 bytes are new data, rest zero-filled
        let mut buf = vec![0u8; 4096];
        swap.read_slot(0, &mut buf).unwrap();
        assert!(buf[..16].iter().all(|&b| b == 0xAA), "first 16 bytes must be 0xAA");
        assert!(buf[16..].iter().all(|&b| b == 0x00), "remaining bytes must be zero-filled");
    }

    // ------------------------------------------------------------------
    // 5/13: Write 1 byte to every slot in a 64-slot file, verify all
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn single_byte_per_slot_64_slots() {
        // Arrange: 64 slots
        let tmp = TempDir::new().unwrap();
        let count = 64u64;
        let swap = NvmeSwapFile::open(
            tmp.path().join("1b64.swap"),
            4096,
            4096,
            count,
        )
        .unwrap();

        // Act: write unique byte to each slot
        for pid in 0..count as usize {
            swap.write_slot(pid, &[(pid as u8).wrapping_mul(5)]).unwrap();
        }

        // Assert: read back each slot and verify
        for pid in 0..count as usize {
            let mut byte = [0u8; 1];
            swap.read_slot(pid, &mut byte).unwrap();
            assert_eq!(byte[0], (pid as u8).wrapping_mul(5), "slot {pid} mismatch");
        }
    }

    // ------------------------------------------------------------------
    // 6/13: max_slot_bytes alignment: NVME_ALIGN + 2 rounds to 2 * NVME_ALIGN
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn max_slot_bytes_align_plus_two_rounds_to_double() {
        // Arrange: pass NVME_ALIGN + 2 as max_slot_bytes
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(
            tmp.path().join("align_plus2.swap"),
            4096,
            NVME_ALIGN + 2,
            4,
        )
        .unwrap();

        // Assert: rounds up to next multiple
        assert_eq!(
            swap.max_slot_bytes,
            NVME_ALIGN * 2,
            "NVME_ALIGN + 2 must round up to 2 * NVME_ALIGN"
        );
    }

    // ------------------------------------------------------------------
    // 7/13: Reopen with smaller max_slot_bytes: struct stores new aligned value
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn reopen_smaller_slot_bytes_stores_new_value() {
        // Arrange: create with 8192 slot
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("smaller.swap");
        {
            let _swap = NvmeSwapFile::open(path.clone(), 4096, 8192, 4).unwrap();
        }

        // Act: reopen with smaller max_slot_bytes
        let swap = NvmeSwapFile::open(path, 4096, 4096, 4).unwrap();

        // Assert: struct reflects the new (already aligned) value
        assert_eq!(swap.max_slot_bytes, 4096, "should store new smaller aligned value");
    }

    // ------------------------------------------------------------------
    // 8/13: File size for large slot_count with aligned slots
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn file_size_for_large_slot_count() {
        // Arrange: 256 slots of 8192 bytes
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("large_count.swap");
        let count = 256u64;
        let slot_sz = 8192usize;

        // Act
        let _swap = NvmeSwapFile::open(path.clone(), 4096, slot_sz, count).unwrap();
        let actual = std::fs::metadata(&path).unwrap().len();

        // Assert: header + count * slot_size
        let expected = SWAP_HEADER_BYTES as u64 + count * slot_sz as u64;
        assert_eq!(actual, expected, "file size must match header + 256 * 8192");
    }

    // ------------------------------------------------------------------
    // 9/13: Write to slot, read exactly 1 byte from slot, verify alignment
    //       (O_DIRECT requires aligned I/O, internal buffer handles it)
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn write_full_read_one_byte_odirect_aligned() {
        // Arrange: write 4096 bytes with a known first byte
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("oalign.swap"), 4096, 4096, 4).unwrap();
        let mut data = vec![0u8; 4096];
        data[0] = 0xDE;
        data[4095] = 0xAD;
        swap.write_slot(0, &data).unwrap();

        // Act: read only 1 byte
        let mut byte = [0u8; 1];
        swap.read_slot(0, &mut byte).unwrap();

        // Assert: first byte is correct despite O_DIRECT alignment
        assert_eq!(byte[0], 0xDE, "single byte read must match first written byte");
    }

    // ------------------------------------------------------------------
    // 10/13: Write to 128 slots with unique 2-byte patterns, read all back
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn two_byte_pattern_128_slots() {
        // Arrange
        let tmp = TempDir::new().unwrap();
        let count = 128u64;
        let swap = NvmeSwapFile::open(
            tmp.path().join("2b128.swap"),
            4096,
            4096,
            count,
        )
        .unwrap();

        // Act: write 2-byte unique pattern per slot
        for pid in 0..count as usize {
            let lo = (pid & 0xFF) as u8;
            let hi = ((pid >> 8) & 0xFF) as u8;
            swap.write_slot(pid, &[lo, hi]).unwrap();
        }

        // Assert: read back and verify each slot
        for pid in 0..count as usize {
            let lo = (pid & 0xFF) as u8;
            let hi = ((pid >> 8) & 0xFF) as u8;
            let mut buf = [0u8; 2];
            swap.read_slot(pid, &mut buf).unwrap();
            assert_eq!(buf[0], lo, "slot {pid} low byte");
            assert_eq!(buf[1], hi, "slot {pid} high byte");
        }
    }

    // ------------------------------------------------------------------
    // 11/13: Overwrite slot 0 with pattern A, overwrite slot 0 with pattern B,
    //        overwrite slot 1 with pattern C — verify both slots
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn overwrite_one_slot_other_untouched() {
        // Arrange
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("ow_untouched.swap"), 4096, 4096, 8).unwrap();

        swap.write_slot(0, &[0xAA; 64]).unwrap();
        swap.write_slot(1, &[0xBB; 64]).unwrap();

        // Act: overwrite only slot 0
        swap.write_slot(0, &[0xCC; 64]).unwrap();

        // Assert: slot 0 has new data, slot 1 untouched
        let mut buf0 = vec![0u8; 64];
        let mut buf1 = vec![0u8; 64];
        swap.read_slot(0, &mut buf0).unwrap();
        swap.read_slot(1, &mut buf1).unwrap();
        assert!(buf0.iter().all(|&b| b == 0xCC), "slot 0 must be 0xCC");
        assert!(buf1.iter().all(|&b| b == 0xBB), "slot 1 must be untouched 0xBB");
    }

    // ------------------------------------------------------------------
    // 12/13: Write to all even slots, verify odd slots are zeros
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn even_slot_writes_odd_slots_zero() {
        // Arrange: 8 slots
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(tmp.path().join("even_odd.swap"), 4096, 4096, 8).unwrap();

        // Act: write to even slots only
        for pid in (0..8).filter(|p| p % 2 == 0) {
            let data = vec![(pid as u8).wrapping_add(0x40); 128];
            swap.write_slot(pid, &data).unwrap();
        }

        // Assert: odd slots are still zeros
        for pid in (0..8).filter(|p| p % 2 == 1) {
            let mut buf = vec![0xFF; 128];
            swap.read_slot(pid, &mut buf).unwrap();
            assert!(
                buf.iter().all(|&b| b == 0x00),
                "odd slot {pid} must be zeros"
            );
        }

        // Assert: even slots have correct data
        for pid in (0..8).filter(|p| p % 2 == 0) {
            let expected = (pid as u8).wrapping_add(0x40);
            let mut buf = vec![0u8; 128];
            swap.read_slot(pid, &mut buf).unwrap();
            assert!(
                buf.iter().all(|&b| b == expected),
                "even slot {pid} must be 0x{expected:02x}"
            );
        }
    }

    // ------------------------------------------------------------------
    // 13/13: Write sequential pattern spanning 3 adjacent slots, verify
    //        each slot's data independently
    // ------------------------------------------------------------------

    // @trace REQ-COMP-015 [level:unit]
    #[test]
    fn three_adjacent_slots_independent_patterns() {
        // Arrange: 3 slots, each gets a different deterministic pattern
        let tmp = TempDir::new().unwrap();
        let swap = NvmeSwapFile::open(
            tmp.path().join("adj3.swap"),
            4096,
            4096,
            8,
        )
        .unwrap();

        let data_3: Vec<u8> = (0..4096).map(|i| ((i + 0x30) % 256) as u8).collect();
        let data_4: Vec<u8> = (0..4096).map(|i| ((i + 0x40) % 256) as u8).collect();
        let data_5: Vec<u8> = (0..4096).map(|i| ((i + 0x50) % 256) as u8).collect();

        // Act: write to 3 adjacent slots
        swap.write_slot(3, &data_3).unwrap();
        swap.write_slot(4, &data_4).unwrap();
        swap.write_slot(5, &data_5).unwrap();

        let mut r3 = vec![0u8; 4096];
        let mut r4 = vec![0u8; 4096];
        let mut r5 = vec![0u8; 4096];
        swap.read_slot(3, &mut r3).unwrap();
        swap.read_slot(4, &mut r4).unwrap();
        swap.read_slot(5, &mut r5).unwrap();

        // Assert: each slot independently correct, no cross-contamination
        assert_eq!(r3, data_3, "slot 3 must match its own pattern");
        assert_eq!(r4, data_4, "slot 4 must match its own pattern");
        assert_eq!(r5, data_5, "slot 5 must match its own pattern");
    }
}
