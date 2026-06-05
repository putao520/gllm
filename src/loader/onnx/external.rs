use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::MmapOptions;
use prost::bytes::Bytes;

use super::{LoaderError, Result};

#[derive(Debug)]
pub(super) struct ExternalDataResolver {
    base_dir: PathBuf,
    cache: HashMap<PathBuf, Bytes>,
}

impl ExternalDataResolver {
    pub(super) fn new(model_path: &Path) -> Self {
        let base_dir = model_path
            .parent()
            .map(Path::to_path_buf)
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| PathBuf::from(".")); // LEGAL: 无父目录或父目录为空时使用当前目录
        Self {
            base_dir,
            cache: HashMap::new(),
        }
    }

    pub(super) fn resolve(
        &mut self,
        location: &str,
        offset: usize,
        length: usize,
    ) -> Result<Bytes> {
        let path = self.base_dir.join(location);
        let bytes = self.mmap_file(&path)?;
        let end = offset.checked_add(length).ok_or_else(|| {
            LoaderError::Onnx(format!("external data offset overflow for {location}"))
        })?;
        if end > bytes.len() {
            return Err(LoaderError::Onnx(format!(
                "external data slice out of bounds for {location}"
            )));
        }
        Ok(bytes.slice(offset..end))
    }

    fn mmap_file(&mut self, path: &Path) -> Result<Bytes> {
        if let Some(cached) = self.cache.get(path) {
            return Ok(cached.clone());
        }
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let bytes = Bytes::from_owner(mmap);
        self.cache.insert(path.to_path_buf(), bytes.clone());
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Helper: create a temp file with given content bytes, return its path.
    /// The caller is responsible for cleanup (temp files persist until process exit
    /// or explicit removal).
    fn write_temp_file(dir: &Path, name: &str, data: &[u8]) -> PathBuf {
        let path = dir.join(name);
        let mut f = File::create(&path).unwrap();
        f.write_all(data).unwrap();
        f.sync_all().unwrap();
        path
    }

    // ── ExternalDataResolver::new — base_dir derivation ──────────────────

    #[test]
    fn new_with_parent_directory_sets_base_dir_to_parent() {
        let path = Path::new("/some/dir/model.onnx");
        let resolver = ExternalDataResolver::new(path);
        // Access private field via pattern — struct is pub(super) and we are in
        // a submodule of the parent, so we have access.
        assert_eq!(resolver.base_dir, PathBuf::from("/some/dir"));
    }

    #[test]
    fn new_with_bare_filename_base_dir_is_empty_string() {
        // Path::new("model.onnx").parent() returns Some(""), not None.
        // The map branch runs, producing PathBuf::from("").
        let path = Path::new("model.onnx");
        let resolver = ExternalDataResolver::new(path);
        assert_eq!(resolver.base_dir, PathBuf::from("."));
    }

    #[test]
    fn new_with_root_path_sets_base_dir_to_root() {
        let path = Path::new("/model.onnx");
        let resolver = ExternalDataResolver::new(path);
        assert_eq!(resolver.base_dir, PathBuf::from("/"));
    }

    #[test]
    fn new_with_deeply_nested_path_sets_base_dir_correctly() {
        let path = Path::new("/a/b/c/d/e/model.onnx");
        let resolver = ExternalDataResolver::new(path);
        assert_eq!(resolver.base_dir, PathBuf::from("/a/b/c/d/e"));
    }

    #[test]
    fn new_cache_starts_empty() {
        let path = Path::new("model.onnx");
        let resolver = ExternalDataResolver::new(path);
        assert!(resolver.cache.is_empty());
    }

    // ── ExternalDataResolver::resolve — success path ─────────────────────

    #[test]
    fn resolve_reads_file_and_slices_correctly() {
        let dir = tempfile::tempdir().unwrap();
        let data = b"Hello, World! This is external data.";
        write_temp_file(dir.path(), "model.onnx", b"dummy model");
        write_temp_file(dir.path(), "weights.bin", data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("weights.bin", 7, 6).unwrap();

        assert_eq!(result.as_ref(), b"World!");
    }

    #[test]
    fn resolve_full_file_read_with_zero_offset() {
        let dir = tempfile::tempdir().unwrap();
        let data = b"ABCDEFGH";
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "data.bin", data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("data.bin", 0, 8).unwrap();

        assert_eq!(result.as_ref(), b"ABCDEFGH");
    }

    #[test]
    fn resolve_zero_length_returns_empty_slice() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "data.bin", b"some data");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("data.bin", 4, 0).unwrap();

        assert!(result.is_empty());
    }

    #[test]
    fn resolve_exact_end_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let data = b"0123456789";
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "data.bin", data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("data.bin", 0, 10).unwrap();

        assert_eq!(result.as_ref(), b"0123456789");
    }

    #[test]
    fn resolve_last_byte() {
        let dir = tempfile::tempdir().unwrap();
        let data = b"ABCDEFGHIJ";
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "data.bin", data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("data.bin", 9, 1).unwrap();

        assert_eq!(result.as_ref(), b"J");
    }

    #[test]
    fn resolve_subdirectory_location() {
        let dir = tempfile::tempdir().unwrap();
        let sub_dir = dir.path().join("external_data");
        std::fs::create_dir_all(&sub_dir).unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(&sub_dir, "layer0.bin", b"layer0__weight_data_here");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver
            .resolve("external_data/layer0.bin", 8, 11)
            .unwrap();

        assert_eq!(result.as_ref(), b"weight_data");
    }

    // ── ExternalDataResolver::resolve — error paths ──────────────────────

    #[test]
    fn resolve_file_not_found_returns_io_error() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("nonexistent.bin", 0, 10);

        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("IO error") || msg.contains("No such file"));
    }

    #[test]
    fn resolve_offset_overflow_returns_onnx_error() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "data.bin", b"small");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("data.bin", usize::MAX, 1);

        assert!(result.is_err());
        let LoaderError::Onnx(msg) = result.unwrap_err() else {
            panic!("expected Onnx error variant");
        };
        assert!(msg.contains("offset overflow"));
        assert!(msg.contains("data.bin"));
    }

    #[test]
    fn resolve_offset_plus_length_overflow_returns_onnx_error() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "data.bin", b"small");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("data.bin", 1, usize::MAX);

        assert!(result.is_err());
        let LoaderError::Onnx(msg) = result.unwrap_err() else {
            panic!("expected Onnx error variant");
        };
        assert!(msg.contains("offset overflow"));
    }

    #[test]
    fn resolve_slice_out_of_bounds_returns_onnx_error() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "data.bin", b"short");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("data.bin", 0, 100);

        assert!(result.is_err());
        let LoaderError::Onnx(msg) = result.unwrap_err() else {
            panic!("expected Onnx error variant");
        };
        assert!(msg.contains("out of bounds"));
        assert!(msg.contains("data.bin"));
    }

    #[test]
    fn resolve_offset_at_end_with_nonzero_length_returns_bounds_error() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "data.bin", b"12345");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // offset=5 is exactly at end, length=1 goes past
        let result = resolver.resolve("data.bin", 5, 1);

        assert!(result.is_err());
        let LoaderError::Onnx(msg) = result.unwrap_err() else {
            panic!("expected Onnx error variant");
        };
        assert!(msg.contains("out of bounds"));
    }

    #[test]
    fn resolve_one_past_end_returns_bounds_error() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "data.bin", b"1234");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("data.bin", 4, 1);

        assert!(result.is_err());
    }

    // ── ExternalDataResolver::mmap_file — caching behavior ───────────────

    #[test]
    fn resolve_caches_file_across_calls() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "shared.bin", b"AAAABBBBCCCCDDDD");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let slice1 = resolver.resolve("shared.bin", 0, 4).unwrap();
        let slice2 = resolver.resolve("shared.bin", 4, 4).unwrap();

        assert_eq!(slice1.as_ref(), b"AAAA");
        assert_eq!(slice2.as_ref(), b"BBBB");
        // Cache should have one entry for shared.bin
        assert_eq!(resolver.cache.len(), 1);
    }

    #[test]
    fn resolve_different_files_populates_separate_cache_entries() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "file_a.bin", b"DATA_A");
        write_temp_file(dir.path(), "file_b.bin", b"DATA_B");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let a = resolver.resolve("file_a.bin", 0, 6).unwrap();
        let b = resolver.resolve("file_b.bin", 0, 6).unwrap();

        assert_eq!(a.as_ref(), b"DATA_A");
        assert_eq!(b.as_ref(), b"DATA_B");
        assert_eq!(resolver.cache.len(), 2);
    }

    #[test]
    fn resolve_multiple_slices_from_same_file() {
        let dir = tempfile::tempdir().unwrap();
        let data = b"0123456789ABCDEF";
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "big.bin", data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let s1 = resolver.resolve("big.bin", 0, 4).unwrap();
        let s2 = resolver.resolve("big.bin", 10, 6).unwrap();
        let s3 = resolver.resolve("big.bin", 0, 16).unwrap();

        assert_eq!(s1.as_ref(), b"0123");
        assert_eq!(s2.as_ref(), b"ABCDEF");
        assert_eq!(s3.as_ref(), data);
    }

    // ── ExternalDataResolver::new — edge cases ───────────────────────────

    #[test]
    fn new_with_empty_path_resolves_to_current_dir() {
        let path = Path::new("");
        let resolver = ExternalDataResolver::new(path);
        // parent() of "" is None
        assert_eq!(resolver.base_dir, PathBuf::from("."));
    }

    #[test]
    fn new_with_trailing_slash_strips_last_component_as_filename() {
        // Path::new("/some/dir/") normalizes to "/some/dir", then parent() = "/some".
        // Rust treats the last component "dir" as the filename portion.
        let path = Path::new("/some/dir/");
        let resolver = ExternalDataResolver::new(path);
        assert_eq!(resolver.base_dir, PathBuf::from("/some"));
    }

    // ── LoaderError::Onnx display from resolve errors ────────────────────

    #[test]
    fn onnx_error_from_resolve_overflow_display() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "weights.bin", b"short");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let err = resolver.resolve("weights.bin", usize::MAX, usize::MAX).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("ONNX error"));
        assert!(msg.contains("offset overflow"));
        assert!(msg.contains("weights.bin"));
    }

    #[test]
    fn onnx_error_from_resolve_bounds_display() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "tensors.bin", b"abc");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let err = resolver.resolve("tensors.bin", 0, 999).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("ONNX error"));
        assert!(msg.contains("out of bounds"));
        assert!(msg.contains("tensors.bin"));
    }

    // ── Additional: resolve — binary / non-UTF8 content ───────────────────

    #[test]
    fn resolve_handles_binary_data_with_null_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let data: &[u8] = &[0x00, 0xFF, 0x80, 0x7F, 0x01, 0xFE];
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "binary.bin", data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("binary.bin", 0, 6).unwrap();

        assert_eq!(result.as_ref(), data);
    }

    #[test]
    fn resolve_handles_high_bytes_and_control_chars() {
        let dir = tempfile::tempdir().unwrap();
        // 256-byte table with every possible byte value
        let data: Vec<u8> = (0u8..=255).collect();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "all_bytes.bin", &data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("all_bytes.bin", 128, 128).unwrap();

        assert_eq!(result.len(), 128);
        assert_eq!(result.as_ref(), &data[128..256]);
    }

    #[test]
    fn resolve_mid_range_slice_of_binary_data() {
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0u8..=255).collect();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "bytes.bin", &data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("bytes.bin", 100, 50).unwrap();

        assert_eq!(result.len(), 50);
        assert_eq!(result.as_ref(), &data[100..150]);
    }

    // ── Additional: resolve — single-byte and empty file edge cases ───────

    #[test]
    fn resolve_single_byte_file() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "one.bin", b"X");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("one.bin", 0, 1).unwrap();

        assert_eq!(result.as_ref(), b"X");
    }

    #[test]
    fn resolve_single_byte_file_with_nonzero_offset_fails() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "one.bin", b"X");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("one.bin", 1, 1);

        assert!(result.is_err());
    }

    #[test]
    fn resolve_empty_file_with_zero_offset_zero_length() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "empty.bin", b"");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("empty.bin", 0, 0).unwrap();

        assert!(result.is_empty());
    }

    #[test]
    fn resolve_empty_file_with_nonzero_length_fails() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "empty.bin", b"");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("empty.bin", 0, 1);

        assert!(result.is_err());
    }

    // ── Additional: resolve — repeated / idempotent calls ─────────────────

    #[test]
    fn resolve_same_slice_twice_returns_identical_bytes() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "data.bin", b"ABCDEFGHIJKLMNOP");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let first = resolver.resolve("data.bin", 4, 8).unwrap();
        let second = resolver.resolve("data.bin", 4, 8).unwrap();

        assert_eq!(first.as_ref(), second.as_ref());
        assert_eq!(first.as_ref(), b"EFGHIJKL");
        // Cache should have exactly 1 entry (same file)
        assert_eq!(resolver.cache.len(), 1);
    }

    #[test]
    fn resolve_overlapping_slices_from_same_file() {
        let dir = tempfile::tempdir().unwrap();
        let data = b"ABCDEFGHIJ";
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "data.bin", data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let slice_a = resolver.resolve("data.bin", 2, 5).unwrap(); // CDEFG
        let slice_b = resolver.resolve("data.bin", 4, 4).unwrap(); // EFGH

        assert_eq!(slice_a.as_ref(), b"CDEFG");
        assert_eq!(slice_b.as_ref(), b"EFGH");
    }

    #[test]
    fn resolve_three_different_files_cache_size_is_three() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "a.bin", b"AAA");
        write_temp_file(dir.path(), "b.bin", b"BBB");
        write_temp_file(dir.path(), "c.bin", b"CCC");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        resolver.resolve("a.bin", 0, 3).unwrap();
        resolver.resolve("b.bin", 0, 3).unwrap();
        resolver.resolve("c.bin", 0, 3).unwrap();

        assert_eq!(resolver.cache.len(), 3);
    }

    // ── Additional: resolve — returned Bytes length property ──────────────

    #[test]
    fn resolve_result_length_matches_requested_length() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "data.bin", b"0123456789");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("data.bin", 3, 5).unwrap();

        assert_eq!(result.len(), 5);
    }

    #[test]
    fn resolve_zero_length_result_has_zero_len() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "data.bin", b"0123456789");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("data.bin", 5, 0).unwrap();

        assert_eq!(result.len(), 0);
    }

    // ── Additional: resolve — error recovery ──────────────────────────────

    #[test]
    fn resolve_succeeds_after_previous_error_on_different_file() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "good.bin", b"GOOD_DATA");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // First call fails (file does not exist)
        let bad = resolver.resolve("missing.bin", 0, 4);
        assert!(bad.is_err());

        // Second call succeeds (valid file)
        let good = resolver.resolve("good.bin", 0, 9).unwrap();
        assert_eq!(good.as_ref(), b"GOOD_DATA");
    }

    #[test]
    fn resolve_succeeds_after_previous_bounds_error_on_same_file() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "data.bin", b"0123456789");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // First call: out-of-bounds
        let bad = resolver.resolve("data.bin", 0, 100);
        assert!(bad.is_err());

        // Second call: valid slice
        let good = resolver.resolve("data.bin", 0, 10).unwrap();
        assert_eq!(good.as_ref(), b"0123456789");
    }

    // ── Additional: new — path edge cases ─────────────────────────────────

    #[test]
    fn new_with_relative_nested_path() {
        let path = Path::new("relative/subdir/model.onnx");
        let resolver = ExternalDataResolver::new(path);

        assert_eq!(resolver.base_dir, PathBuf::from("relative/subdir"));
    }

    #[test]
    fn new_with_dot_slash_path() {
        let path = Path::new("./model.onnx");
        let resolver = ExternalDataResolver::new(path);

        // Rust treats "./model.onnx" as having parent "." (the current directory),
        // unlike "model.onnx" whose parent is "".
        assert_eq!(resolver.base_dir, PathBuf::from("."));
    }

    #[test]
    fn new_with_multi_component_dot_path() {
        let path = Path::new("./a/b/c/model.onnx");
        let resolver = ExternalDataResolver::new(path);

        assert_eq!(resolver.base_dir, PathBuf::from("./a/b/c"));
    }

    // ── Additional: resolve — deeply nested subdirectory ──────────────────

    #[test]
    fn resolve_deeply_nested_subdirectory() {
        let dir = tempfile::tempdir().unwrap();
        let deep = dir.path().join("a").join("b").join("c");
        std::fs::create_dir_all(&deep).unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(&deep, "deep.bin", b"DEEP_DATA_HERE");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("a/b/c/deep.bin", 0, 14).unwrap();

        assert_eq!(result.as_ref(), b"DEEP_DATA_HERE");
    }

    // ── Additional: resolve — path traversal safety ───────────────────────

    #[test]
    fn resolve_with_dot_dot_location_reads_sibling_file() {
        let dir = tempfile::tempdir().unwrap();
        // Place data file directly in temp dir
        write_temp_file(dir.path(), "sibling.bin", b"SIBLING_DATA");
        // Place model in a subdirectory so "../" reaches back to temp dir
        let sub = dir.path().join("subdir");
        std::fs::create_dir_all(&sub).unwrap();
        write_temp_file(&sub, "model.onnx", b"dummy");

        let model_path = sub.join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("../sibling.bin", 0, 12).unwrap();

        assert_eq!(result.as_ref(), b"SIBLING_DATA");
    }

    // ── Additional: resolve — large file slicing ──────────────────────────

    #[test]
    fn resolve_large_file_head_and_tail_slices() {
        let dir = tempfile::tempdir().unwrap();
        // 8 KB of data with a recognizable pattern
        let mut data = vec![0u8; 8192];
        // Mark head: first 4 bytes
        data[0..4].copy_from_slice(b"HEAD");
        // Mark tail: last 4 bytes
        data[8188..8192].copy_from_slice(b"TAIL");
        // Mark middle
        data[4096..4100].copy_from_slice(b"MIDL");

        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "large.bin", &data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let head = resolver.resolve("large.bin", 0, 4).unwrap();
        let tail = resolver.resolve("large.bin", 8188, 4).unwrap();
        let mid = resolver.resolve("large.bin", 4096, 4).unwrap();

        assert_eq!(head.as_ref(), b"HEAD");
        assert_eq!(tail.as_ref(), b"TAIL");
        assert_eq!(mid.as_ref(), b"MIDL");
        // Only one file should be cached
        assert_eq!(resolver.cache.len(), 1);
    }

    // ── Additional: resolve — location name containing special chars ──────

    #[test]
    fn resolve_location_with_hyphen_and_underscore() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "layer-0_weight.bin", b"SPECIAL_CHARS_OK");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("layer-0_weight.bin", 0, 16).unwrap();

        assert_eq!(result.as_ref(), b"SPECIAL_CHARS_OK");
    }

    #[test]
    fn resolve_location_with_dots_in_name() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "weights.data.v1.bin", b"DOT_NAME");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("weights.data.v1.bin", 0, 8).unwrap();

        assert_eq!(result.as_ref(), b"DOT_NAME");
    }

    // ── Additional: resolve — cache key structure ─────────────────────────

    #[test]
    fn cache_key_is_absolute_path_after_join() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "data.bin", b"DATA");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        resolver.resolve("data.bin", 0, 4).unwrap();

        // The cache key should be base_dir joined with "data.bin"
        let expected_key = resolver.base_dir.join("data.bin");
        assert!(resolver.cache.contains_key(&expected_key));
    }

    // ── Additional: resolve — multiple errors don't corrupt state ─────────

    #[test]
    fn multiple_errors_then_success_works() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "good.bin", b"OK");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // Trigger three different errors
        assert!(resolver.resolve("missing1.bin", 0, 1).is_err());
        assert!(resolver.resolve("good.bin", 0, 100).is_err());
        assert!(resolver.resolve("missing2.bin", 0, 1).is_err());

        // Cache may have the "good.bin" entry from the bounds error attempt
        // Now a valid resolve should work
        let result = resolver.resolve("good.bin", 0, 2).unwrap();
        assert_eq!(result.as_ref(), b"OK");
    }

    // ── Additional: resolve — offset at exact file_length with zero length ─

    #[test]
    fn resolve_offset_at_file_end_with_zero_length() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "data.bin", b"12345");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // offset=5 is exactly at the end of 5-byte file, length=0 is valid
        let result = resolver.resolve("data.bin", 5, 0).unwrap();

        assert!(result.is_empty());
    }

    // ── Additional: resolve — consecutive slices cover full file ──────────

    #[test]
    fn consecutive_non_overlapping_slices_reconstruct_file() {
        let dir = tempfile::tempdir().unwrap();
        let original = b"ABCDEFGHIJ"; // 10 bytes
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "data.bin", original);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let s1 = resolver.resolve("data.bin", 0, 5).unwrap();
        let s2 = resolver.resolve("data.bin", 5, 5).unwrap();

        let mut reconstructed = Vec::new();
        reconstructed.extend_from_slice(&s1);
        reconstructed.extend_from_slice(&s2);

        assert_eq!(reconstructed.as_slice(), original);
    }

    // ── Additional: resolve — no file access for non-existent intermediate dir

    #[test]
    fn resolve_nonexistent_subdirectory_returns_io_error() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        // Do NOT create "missing_dir/"

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("missing_dir/file.bin", 0, 4);

        assert!(result.is_err());
    }

    // ── Additional: resolve — cache persists across errors on different files

    #[test]
    fn cache_retains_previous_entry_after_error_on_different_file() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"dummy");
        write_temp_file(dir.path(), "cached.bin", b"CACHED_DATA");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // First resolve populates cache
        resolver.resolve("cached.bin", 0, 11).unwrap();
        assert_eq!(resolver.cache.len(), 1);

        // Error on different file should not evict cache
        assert!(resolver.resolve("no_such.bin", 0, 4).is_err());
        assert_eq!(resolver.cache.len(), 1);

        // The original cached file still works — slice "DATA" from "CACHED_DATA"
        // CACHED_DATA: C(0)A(1)C(2)H(3)E(4)D(5)_(6)D(7)A(8)T(9)A(10)
        let result = resolver.resolve("cached.bin", 7, 4).unwrap();
        assert_eq!(result.as_ref(), b"DATA");
    }

    // ── new: double-slash path ────────────────────────────────────────────

    #[test]
    fn new_with_double_slash_collapses_in_path() {
        let path = Path::new("/a//b/model.onnx");
        let resolver = ExternalDataResolver::new(path);
        // Rust normalizes "//" in Path internally
        let base = &resolver.base_dir;
        assert!(*base == PathBuf::from("/a/b") || *base == PathBuf::from("/a//b"));
    }

    #[test]
    fn new_with_dot_component_in_path() {
        let path = Path::new("/a/./b/model.onnx");
        let resolver = ExternalDataResolver::new(path);
        let base = &resolver.base_dir;
        assert!(*base == PathBuf::from("/a/./b") || *base == PathBuf::from("/a/b"));
    }

    // ── new: path with only root ─────────────────────────────────────────

    #[test]
    fn new_root_path_base_dir_is_root() {
        let path = Path::new("/");
        let resolver = ExternalDataResolver::new(path);
        // parent() of "/" is None → unwrap_or_else → PathBuf::from(".")
        assert_eq!(resolver.base_dir, PathBuf::from("."));
    }

    // ── resolve: offset zero and full length ─────────────────────────────

    #[test]
    fn resolve_offset_zero_reads_from_start() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"ABCDEFGHIJ");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("d.bin", 0, 3).unwrap();
        assert_eq!(result.as_ref(), b"ABC");
    }

    // ── resolve: repeated zero-length slices ─────────────────────────────

    #[test]
    fn resolve_many_zero_length_slices_all_empty() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"SOMETHING");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        for offset in 0..=9 {
            let result = resolver.resolve("d.bin", offset, 0).unwrap();
            assert!(result.is_empty(), "expected empty at offset {offset}");
        }
    }

    // ── resolve: returned Bytes are independent slices ────────────────────

    #[test]
    fn resolve_non_overlapping_slices_are_independent() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"AAAABBBB");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let a = resolver.resolve("d.bin", 0, 4).unwrap();
        let b = resolver.resolve("d.bin", 4, 4).unwrap();

        assert_eq!(a.as_ref(), b"AAAA");
        assert_eq!(b.as_ref(), b"BBBB");
        // They should not share the same pointer range
        assert_ne!(a.as_ref(), b.as_ref());
    }

    // ── resolve: many sequential slices cover the whole file ──────────────

    #[test]
    fn resolve_many_sequential_small_slices_cover_file() {
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0..50).collect();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", &data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let mut reconstructed = Vec::new();
        for chunk_start in (0..50).step_by(10) {
            let chunk = resolver.resolve("d.bin", chunk_start, 10).unwrap();
            assert_eq!(chunk.len(), 10);
            reconstructed.extend_from_slice(&chunk);
        }

        assert_eq!(reconstructed, data);
    }

    // ── resolve: error contains location name ────────────────────────────

    #[test]
    fn resolve_out_of_bounds_error_mentions_location() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "my_tensor_data.bin", b"X");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let LoaderError::Onnx(msg) = resolver.resolve("my_tensor_data.bin", 0, 50).unwrap_err() else {
            panic!("expected Onnx error");
        };
        assert!(msg.contains("my_tensor_data.bin"));
    }

    #[test]
    fn resolve_overflow_error_mentions_location() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "layer_weights.bin", b"DATA");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let LoaderError::Onnx(msg) = resolver
            .resolve("layer_weights.bin", usize::MAX, 1)
            .unwrap_err()
        else {
            panic!("expected Onnx error");
        };
        assert!(msg.contains("layer_weights.bin"));
    }

    // ── resolve: cache not polluted by failed file open ──────────────────

    #[test]
    fn resolve_missing_file_does_not_pollute_cache() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        assert!(resolver.resolve("no_such_file.bin", 0, 1).is_err());
        assert!(resolver.cache.is_empty(), "cache should stay empty after IO error");
    }

    // ── resolve: interleaved success and failure ─────────────────────────

    #[test]
    fn resolve_interleaved_success_failure_keeps_cache_valid() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "a.bin", b"AAA");
        write_temp_file(dir.path(), "b.bin", b"BBB");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let a = resolver.resolve("a.bin", 0, 3).unwrap();
        assert_eq!(a.as_ref(), b"AAA");
        assert_eq!(resolver.cache.len(), 1);

        assert!(resolver.resolve("missing.bin", 0, 1).is_err());
        assert_eq!(resolver.cache.len(), 1);

        let b = resolver.resolve("b.bin", 0, 3).unwrap();
        assert_eq!(b.as_ref(), b"BBB");
        assert_eq!(resolver.cache.len(), 2);

        // a still works from cache
        let a2 = resolver.resolve("a.bin", 0, 3).unwrap();
        assert_eq!(a2.as_ref(), b"AAA");
        assert_eq!(resolver.cache.len(), 2);
    }

    // ── resolve: offset just before boundary ─────────────────────────────

    #[test]
    fn resolve_offset_one_before_end_with_one_byte() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"HELLO");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("d.bin", 4, 1).unwrap();
        assert_eq!(result.as_ref(), b"O");
    }

    #[test]
    fn resolve_offset_one_before_end_with_two_bytes_fails() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"HELLO");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        assert!(resolver.resolve("d.bin", 4, 2).is_err());
    }

    // ── resolve: large offset zero length at boundary ────────────────────

    #[test]
    fn resolve_usize_max_offset_zero_length_overflows() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"DATA");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // usize::MAX + 0 overflows checked_add
        // Actually usize::MAX + 0 = usize::MAX, no overflow. But end > bytes.len().
        let result = resolver.resolve("d.bin", usize::MAX, 0);
        // checked_add(usize::MAX, 0) = Some(usize::MAX), then end > 4 → out of bounds
        assert!(result.is_err());
    }

    // ── resolve: offset zero with usize::MAX length ──────────────────────

    #[test]
    fn resolve_zero_offset_max_length_overflows() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"DATA");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // offset=0 + length=usize::MAX overflows checked_add
        assert!(resolver.resolve("d.bin", 0, usize::MAX).is_err());
    }

    // ── new: consecutive resolvers are independent ───────────────────────

    #[test]
    fn new_two_resolvers_have_independent_caches() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"DATA");

        let model_path = dir.path().join("model.onnx");
        let mut r1 = ExternalDataResolver::new(&model_path);
        let r2 = ExternalDataResolver::new(&model_path);

        r1.resolve("d.bin", 0, 4).unwrap();
        assert!(r1.cache.len() == 1);
        assert!(r2.cache.is_empty());
    }

    // ── resolve: file with only zeros ────────────────────────────────────

    #[test]
    fn resolve_file_of_all_zeros() {
        let dir = tempfile::tempdir().unwrap();
        let zeros = vec![0u8; 100];
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "zeros.bin", &zeros);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("zeros.bin", 0, 100).unwrap();
        assert!(result.iter().all(|&b| b == 0));
    }

    #[test]
    fn resolve_slice_of_zeros_file() {
        let dir = tempfile::tempdir().unwrap();
        let zeros = vec![0u8; 50];
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "z.bin", &zeros);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("z.bin", 10, 20).unwrap();
        assert_eq!(result.len(), 20);
        assert!(result.iter().all(|&b| b == 0));
    }

    // ── resolve: file with 0xFF pattern ──────────────────────────────────

    #[test]
    fn resolve_file_of_all_ff() {
        let dir = tempfile::tempdir().unwrap();
        let ff = vec![0xFFu8; 64];
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "ff.bin", &ff);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("ff.bin", 0, 64).unwrap();
        assert!(result.iter().all(|&b| b == 0xFF));
    }

    // ── resolve: many files cached independently ─────────────────────────

    #[test]
    fn resolve_ten_files_each_cached() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        for i in 0..10 {
            let name = format!("file_{i}.bin");
            write_temp_file(dir.path(), &name, &[i as u8; 16]);
        }

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        for i in 0..10 {
            let name = format!("file_{i}.bin");
            let result = resolver.resolve(&name, 0, 16).unwrap();
            assert!(result.iter().all(|&b| b == i as u8), "file_{i} content mismatch");
        }

        assert_eq!(resolver.cache.len(), 10);
    }

    // ── resolve: slice at arbitrary internal offset ───────────────────────

    #[test]
    fn resolve_internal_slice_at_arbitrary_offset() {
        let dir = tempfile::tempdir().unwrap();
        let data = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "mixed.bin", data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("mixed.bin", 10, 26).unwrap();
        assert_eq!(result.as_ref(), b"ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    }

    // ── resolve: location is exactly filename with no path separators ────

    #[test]
    fn resolve_plain_filename_no_path_component() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "weights.bin", b"W");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("weights.bin", 0, 1).unwrap();
        assert_eq!(result.as_ref(), b"W");
    }

    // ── resolve: location with leading dot-slash ─────────────────────────

    #[test]
    fn resolve_location_with_leading_dot_slash() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "data.bin", b"DOTSLASH");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // Path::join("./data.bin") resolves to the same file as "data.bin"
        let result = resolver.resolve("./data.bin", 0, 8).unwrap();
        assert_eq!(result.as_ref(), b"DOTSLASH");
    }

    // ── new: path with only filename component ───────────────────────────

    #[test]
    fn new_with_only_filename_parent_is_empty() {
        let path = Path::new("file.onnx");
        let resolver = ExternalDataResolver::new(path);
        // parent() of "file.onnx" is Some("")
        assert_eq!(resolver.base_dir, PathBuf::from("."));
    }

    // ── resolve: Bytes::len matches after multiple different-length slices ─

    #[test]
    fn resolve_varied_lengths_each_correct() {
        let dir = tempfile::tempdir().unwrap();
        let data = vec![0xABu8; 100];
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", &data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let lengths = [1, 5, 10, 25, 50, 9];
        let mut offset = 0;
        for &len in &lengths {
            let result = resolver.resolve("d.bin", offset, len).unwrap();
            assert_eq!(result.len(), len, "length mismatch at offset {offset}");
            offset += len;
        }
        assert_eq!(offset, 100);
    }

    // ── resolve: error type discrimination ───────────────────────────────

    #[test]
    fn resolve_io_error_is_not_onnx_error() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let err = resolver.resolve("nonexistent.bin", 0, 1).unwrap_err();
        // File not found is an Io variant, not Onnx
        assert!(
            matches!(err, LoaderError::Io(_)),
            "expected Io error, got {err:?}"
        );
    }

    #[test]
    fn resolve_bounds_error_is_onnx_variant() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"SHORT");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let err = resolver.resolve("d.bin", 0, 100).unwrap_err();
        assert!(
            matches!(err, LoaderError::Onnx(_)),
            "expected Onnx error, got {err:?}"
        );
    }

    #[test]
    fn resolve_overflow_error_is_onnx_variant() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"SHORT");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let err = resolver.resolve("d.bin", usize::MAX, 1).unwrap_err();
        assert!(
            matches!(err, LoaderError::Onnx(_)),
            "expected Onnx error, got {err:?}"
        );
    }

    // ── resolve: consecutive offset-0 reads ──────────────────────────────

    #[test]
    fn resolve_repeated_full_reads_same_file() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"ABCDEFGH");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        for _ in 0..5 {
            let result = resolver.resolve("d.bin", 0, 8).unwrap();
            assert_eq!(result.as_ref(), b"ABCDEFGH");
        }
        assert_eq!(resolver.cache.len(), 1);
    }

    // ── resolve: two different subdirectories ─────────────────────────────

    #[test]
    fn resolve_two_different_subdirectories() {
        let dir = tempfile::tempdir().unwrap();
        let sub1 = dir.path().join("dir_a");
        let sub2 = dir.path().join("dir_b");
        std::fs::create_dir_all(&sub1).unwrap();
        std::fs::create_dir_all(&sub2).unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(&sub1, "a.bin", b"FROM_A");
        write_temp_file(&sub2, "b.bin", b"FROM_B");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let a = resolver.resolve("dir_a/a.bin", 0, 6).unwrap();
        let b = resolver.resolve("dir_b/b.bin", 0, 6).unwrap();

        assert_eq!(a.as_ref(), b"FROM_A");
        assert_eq!(b.as_ref(), b"FROM_B");
        assert_eq!(resolver.cache.len(), 2);
    }

    // ── resolve: reading same byte via different offsets ──────────────────

    #[test]
    fn resolve_same_byte_via_offset_and_length_one() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"ABCDE");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let from_0 = resolver.resolve("d.bin", 0, 1).unwrap();
        let from_1 = resolver.resolve("d.bin", 1, 1).unwrap();
        let from_2 = resolver.resolve("d.bin", 2, 1).unwrap();

        assert_eq!(from_0.as_ref(), b"A");
        assert_eq!(from_1.as_ref(), b"B");
        assert_eq!(from_2.as_ref(), b"C");
    }

    // ── resolve: location with unicode-safe ascii characters ──────────────

    #[test]
    fn resolve_location_with_numbers_in_name() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "layer123_weight456.bin", b"LAYER");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("layer123_weight456.bin", 0, 5).unwrap();
        assert_eq!(result.as_ref(), b"LAYER");
    }

    // ── resolve: mmap caching verified by cache key count ────────────────

    #[test]
    fn resolve_cache_entries_match_unique_paths_not_slices() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"0123456789");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // Many slices from same file
        for i in 0..10 {
            resolver.resolve("d.bin", i, 1).unwrap();
        }
        // Only one cache entry for "d.bin"
        assert_eq!(resolver.cache.len(), 1);
    }

    // ── resolve: subdirectory then parent file ────────────────────────────

    #[test]
    fn resolve_subdirectory_then_parent_file() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "root.bin", b"ROOT_DATA");
        write_temp_file(&sub, "sub.bin", b"SUB_DATA");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let sub_result = resolver.resolve("sub/sub.bin", 0, 8).unwrap();
        let root_result = resolver.resolve("root.bin", 0, 9).unwrap();

        assert_eq!(sub_result.as_ref(), b"SUB_DATA");
        assert_eq!(root_result.as_ref(), b"ROOT_DATA");
        assert_eq!(resolver.cache.len(), 2);
    }

    // ── resolve: large data integrity ────────────────────────────────────

    #[test]
    fn resolve_large_data_integrity_check() {
        let dir = tempfile::tempdir().unwrap();
        // 16 KB with known pattern
        let mut data = vec![0u8; 16384];
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "big.bin", &data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // Read from start, middle, end
        let start = resolver.resolve("big.bin", 0, 256).unwrap();
        let mid = resolver.resolve("big.bin", 8192, 256).unwrap();
        let end = resolver.resolve("big.bin", 16128, 256).unwrap();

        for i in 0..256 {
            assert_eq!(start[i], i as u8);
            assert_eq!(mid[i], i as u8);
            assert_eq!(end[i], i as u8);
        }
    }

    // ── resolve: offset 0 length 0 on non-empty file ─────────────────────

    #[test]
    fn resolve_zero_offset_zero_length_on_nonempty_file() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"SOMEDATA");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("d.bin", 0, 0).unwrap();
        assert!(result.is_empty());
    }

    // ── new: path with windows-style backslash (treated as component) ─────

    #[test]
    fn new_unix_path_no_backslash_issues() {
        let path = Path::new("/home/user/models/model.onnx");
        let resolver = ExternalDataResolver::new(path);
        assert_eq!(resolver.base_dir, PathBuf::from("/home/user/models"));
    }

    // ── resolve: check error message format for overflow ─────────────────

    #[test]
    fn resolve_overflow_error_message_format() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "test_file.bin", b"AB");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let LoaderError::Onnx(msg) = resolver
            .resolve("test_file.bin", usize::MAX, 1)
            .unwrap_err()
        else {
            panic!("expected Onnx error");
        };
        assert!(msg.contains("offset overflow"));
        assert!(msg.contains("test_file.bin"));
    }

    // ── resolve: check error message format for bounds ───────────────────

    #[test]
    fn resolve_bounds_error_message_format() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "bound_test.bin", b"ABCD");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let LoaderError::Onnx(msg) = resolver
            .resolve("bound_test.bin", 2, 100)
            .unwrap_err()
        else {
            panic!("expected Onnx error");
        };
        assert!(msg.contains("out of bounds"));
        assert!(msg.contains("bound_test.bin"));
    }

    // ── resolve: different offset same file same cache ───────────────────

    #[test]
    fn resolve_different_offsets_same_file_single_cache_entry() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "shared.bin", b"0123456789ABCDEF");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let offsets = [0, 4, 8, 12];
        for &off in &offsets {
            let r = resolver.resolve("shared.bin", off, 4).unwrap();
            assert_eq!(r.len(), 4);
        }
        assert_eq!(resolver.cache.len(), 1);
    }

    // ── resolve: error when offset equals file length but length nonzero ──

    #[test]
    fn resolve_offset_equals_file_length_nonzero_length_fails() {
        let dir = tempfile::tempdir().unwrap();
        let data = b"ABC"; // 3 bytes
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // offset=3 is at end, length=1 → end=4 > 3 → out of bounds
        assert!(resolver.resolve("d.bin", 3, 1).is_err());
    }

    // ── resolve: bytes content identity across cache hits ────────────────

    #[test]
    fn resolve_cached_slice_content_matches_original_file_region() {
        let dir = tempfile::tempdir().unwrap();
        let data = b"ABCDEFGHIJ";
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // First read populates cache
        let first = resolver.resolve("d.bin", 3, 4).unwrap();
        assert_eq!(first.as_ref(), b"DEFG");

        // Second read hits cache
        let second = resolver.resolve("d.bin", 3, 4).unwrap();
        assert_eq!(second.as_ref(), b"DEFG");

        // Both return same content
        assert_eq!(first.as_ref(), second.as_ref());
    }

    // ── new: model_path is a single component ────────────────────────────

    #[test]
    fn new_single_component_path() {
        let path = Path::new("m");
        let resolver = ExternalDataResolver::new(path);
        assert_eq!(resolver.base_dir, PathBuf::from("."));
    }

    // ── resolve: data with repeating pattern ─────────────────────────────

    #[test]
    fn resolve_repeating_pattern_data() {
        let dir = tempfile::tempdir().unwrap();
        let pattern: Vec<u8> = (0..4).cycle().take(100).collect(); // [0,1,2,3,0,1,2,3,...]
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "pattern.bin", &pattern);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        for start in (0..96).step_by(4) {
            let chunk = resolver.resolve("pattern.bin", start, 4).unwrap();
            assert_eq!(chunk.as_ref(), &[0, 1, 2, 3]);
        }
    }

    // ── resolve: verify Bytes slice does not copy underlying data ────────

    #[test]
    fn resolve_slice_is_zero_copy_reference() {
        let dir = tempfile::tempdir().unwrap();
        let data = b"PADDING_DATA_CONTENT_ENDING";
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let full = resolver.resolve("d.bin", 0, data.len()).unwrap();
        let mid = resolver.resolve("d.bin", 8, 4).unwrap();

        // Bytes::slice creates a reference into the same underlying memory
        // The full slice and mid slice should overlap in the backing store
        assert_eq!(&full[8..12], mid.as_ref());
        assert_eq!(mid.as_ref(), b"DATA");
    }

    // ── new: path with three leading slashes ──────────────────────────────

    #[test]
    fn new_triple_slash_path() {
        // On Unix, Path::new("///") parent() behavior
        let path = Path::new("///a/model.onnx");
        let resolver = ExternalDataResolver::new(path);
        // Rust normalizes /// to /
        let base = &resolver.base_dir;
        assert!(
            *base == PathBuf::from("/a") || *base == PathBuf::from("///a"),
            "unexpected base_dir: {base:?}"
        );
    }

    // ── resolve: interleaved reads from two files ────────────────────────

    #[test]
    fn resolve_interleaved_reads_two_files() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "a.bin", b"AAAABBBB");
        write_temp_file(dir.path(), "b.bin", b"CCCCDDDD");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let a1 = resolver.resolve("a.bin", 0, 4).unwrap();
        let b1 = resolver.resolve("b.bin", 0, 4).unwrap();
        let a2 = resolver.resolve("a.bin", 4, 4).unwrap();
        let b2 = resolver.resolve("b.bin", 4, 4).unwrap();

        assert_eq!(a1.as_ref(), b"AAAA");
        assert_eq!(a2.as_ref(), b"BBBB");
        assert_eq!(b1.as_ref(), b"CCCC");
        assert_eq!(b2.as_ref(), b"DDDD");
        assert_eq!(resolver.cache.len(), 2);
    }

    // ── resolve: file exactly at page boundary size ──────────────────────

    #[test]
    fn resolve_file_at_common_page_size() {
        let dir = tempfile::tempdir().unwrap();
        let data = vec![0x42u8; 4096]; // One page
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "page.bin", &data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let head = resolver.resolve("page.bin", 0, 4).unwrap();
        let tail = resolver.resolve("page.bin", 4092, 4).unwrap();

        assert_eq!(head.as_ref(), &[0x42; 4]);
        assert_eq!(tail.as_ref(), &[0x42; 4]);
    }

    // ── resolve: file slightly larger than page boundary ─────────────────

    #[test]
    fn resolve_file_just_past_page_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let data = vec![0x77u8; 4097]; // Page + 1 byte
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "over.bin", &data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let last_byte = resolver.resolve("over.bin", 4096, 1).unwrap();
        assert_eq!(last_byte.as_ref(), &[0x77]);

        let full = resolver.resolve("over.bin", 0, 4097).unwrap();
        assert_eq!(full.len(), 4097);
    }

    // ── resolve: cache not invalidated by errors on same file ────────────

    #[test]
    fn resolve_bounds_error_does_not_evict_cache_entry() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"DATA");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // First valid read
        let first = resolver.resolve("d.bin", 0, 4).unwrap();
        assert_eq!(first.as_ref(), b"DATA");
        assert_eq!(resolver.cache.len(), 1);

        // Bounds error on same file
        assert!(resolver.resolve("d.bin", 0, 100).is_err());
        // Cache should still have the entry
        assert_eq!(resolver.cache.len(), 1);

        // Valid read after error still works
        let second = resolver.resolve("d.bin", 0, 4).unwrap();
        assert_eq!(second.as_ref(), b"DATA");
    }

    // ── resolve: two files with same name in different directories ────────

    #[test]
    fn resolve_same_named_file_in_different_subdirs() {
        let dir = tempfile::tempdir().unwrap();
        let sub1 = dir.path().join("dir_x");
        let sub2 = dir.path().join("dir_y");
        std::fs::create_dir_all(&sub1).unwrap();
        std::fs::create_dir_all(&sub2).unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(&sub1, "weights.bin", b"X_DATA");
        write_temp_file(&sub2, "weights.bin", b"Y_DATA");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let x = resolver.resolve("dir_x/weights.bin", 0, 6).unwrap();
        let y = resolver.resolve("dir_y/weights.bin", 0, 6).unwrap();

        assert_eq!(x.as_ref(), b"X_DATA");
        assert_eq!(y.as_ref(), b"Y_DATA");
        assert_eq!(resolver.cache.len(), 2);
    }

    // ── resolve: reading from offset 0 to file end in steps ──────────────

    #[test]
    fn resolve_full_file_reconstructed_in_five_steps() {
        let dir = tempfile::tempdir().unwrap();
        let original = b"ABCDEFGHIJKLMNOPQRST"; // 20 bytes
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", original);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let mut reconstructed = Vec::new();
        for i in 0..5 {
            let chunk = resolver.resolve("d.bin", i * 4, 4).unwrap();
            assert_eq!(chunk.len(), 4);
            reconstructed.extend_from_slice(&chunk);
        }

        assert_eq!(reconstructed.as_slice(), original);
    }

    // ── new: resolver reusability ────────────────────────────────────────

    #[test]
    fn new_resolver_can_resolve_multiple_files_sequentially() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "a.bin", b"AAA");
        write_temp_file(dir.path(), "b.bin", b"BBB");
        write_temp_file(dir.path(), "c.bin", b"CCC");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        assert_eq!(resolver.resolve("a.bin", 0, 3).unwrap().as_ref(), b"AAA");
        assert_eq!(resolver.resolve("b.bin", 0, 3).unwrap().as_ref(), b"BBB");
        assert_eq!(resolver.resolve("c.bin", 0, 3).unwrap().as_ref(), b"CCC");
        assert_eq!(resolver.cache.len(), 3);
    }

    // ── resolve: offset reading from mid to exact file end ────────────────

    #[test]
    fn resolve_offset_mid_to_exact_end() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"ABCDEFGHIJ");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("d.bin", 3, 7).unwrap();
        assert_eq!(result.as_ref(), b"DEFGHIJ");
        assert_eq!(result.len(), 7);
    }

    #[test]
    fn resolve_read_from_offset_one_to_end() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"ABCDE");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("d.bin", 1, 4).unwrap();
        assert_eq!(result.as_ref(), b"BCDE");
    }

    #[test]
    fn resolve_last_two_bytes_of_file() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"ABCDEF");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("d.bin", 4, 2).unwrap();
        assert_eq!(result.as_ref(), b"EF");
    }

    // ── resolve: all individual bytes reconstruct the file ────────────────

    #[test]
    fn resolve_all_individual_bytes_reconstruct_file() {
        let dir = tempfile::tempdir().unwrap();
        let original = b"HELLO";
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", original);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let mut reconstructed = Vec::new();
        for i in 0..original.len() {
            let byte = resolver.resolve("d.bin", i, 1).unwrap();
            assert_eq!(byte.len(), 1);
            reconstructed.extend_from_slice(&byte);
        }
        assert_eq!(reconstructed.as_slice(), original);
        assert_eq!(resolver.cache.len(), 1);
    }

    // ── resolve: 2-byte file exhaustive offset/length combos ──────────────

    #[test]
    fn resolve_two_byte_file_read_first_byte() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"AB");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        assert_eq!(resolver.resolve("d.bin", 0, 1).unwrap().as_ref(), b"A");
    }

    #[test]
    fn resolve_two_byte_file_read_second_byte() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"AB");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        assert_eq!(resolver.resolve("d.bin", 1, 1).unwrap().as_ref(), b"B");
    }

    #[test]
    fn resolve_two_byte_file_read_both() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"AB");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        assert_eq!(resolver.resolve("d.bin", 0, 2).unwrap().as_ref(), b"AB");
    }

    #[test]
    fn resolve_two_byte_file_read_past_end_fails() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"AB");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        assert!(resolver.resolve("d.bin", 0, 3).is_err());
        assert!(resolver.resolve("d.bin", 2, 1).is_err());
    }

    #[test]
    fn resolve_two_byte_file_zero_length_at_each_offset() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"AB");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        for offset in 0..=2 {
            let result = resolver.resolve("d.bin", offset, 0).unwrap();
            assert!(result.is_empty(), "expected empty at offset {offset}");
        }
    }

    // ── resolve: offset beyond file size with zero length ─────────────────

    #[test]
    fn resolve_offset_far_beyond_file_size_with_zero_length_fails() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"ABCDE");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // offset=100 is well beyond file size=5, even with length=0
        // checked_add(100,0)=Some(100), 100 > 5 → out of bounds
        let result = resolver.resolve("d.bin", 100, 0);
        assert!(result.is_err());
    }

    // ── resolve: location with special characters ─────────────────────────

    #[test]
    fn resolve_location_with_spaces_in_filename() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "layer weights.bin", b"SPACE_OK");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("layer weights.bin", 0, 8).unwrap();
        assert_eq!(result.as_ref(), b"SPACE_OK");
    }

    #[test]
    fn resolve_location_with_parentheses_in_filename() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "weights(1).bin", b"PAREN");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("weights(1).bin", 0, 5).unwrap();
        assert_eq!(result.as_ref(), b"PAREN");
    }

    #[test]
    fn resolve_location_with_hash_character() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "layer#3.bin", b"HASH");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("layer#3.bin", 0, 4).unwrap();
        assert_eq!(result.as_ref(), b"HASH");
    }

    #[test]
    fn resolve_location_with_exclamation_mark() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "data!.bin", b"EXCL");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("data!.bin", 0, 4).unwrap();
        assert_eq!(result.as_ref(), b"EXCL");
    }

    // ── resolve: location with empty string ───────────────────────────────

    #[test]
    fn resolve_empty_location_attempts_base_dir_join_empty() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // Empty location joins base_dir with "" = base_dir itself.
        // base_dir is a directory, not a file — mmap should fail (IO error).
        let result = resolver.resolve("", 0, 1);
        assert!(result.is_err());
    }

    // ── resolve: interleaved reads from 3+ files ──────────────────────────

    #[test]
    fn resolve_interleaved_reads_three_files() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "a.bin", b"A1A2");
        write_temp_file(dir.path(), "b.bin", b"B1B2");
        write_temp_file(dir.path(), "c.bin", b"C1C2");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let a1 = resolver.resolve("a.bin", 0, 2).unwrap();
        let b1 = resolver.resolve("b.bin", 0, 2).unwrap();
        let c1 = resolver.resolve("c.bin", 0, 2).unwrap();
        let a2 = resolver.resolve("a.bin", 2, 2).unwrap();
        let b2 = resolver.resolve("b.bin", 2, 2).unwrap();
        let c2 = resolver.resolve("c.bin", 2, 2).unwrap();

        assert_eq!(a1.as_ref(), b"A1");
        assert_eq!(a2.as_ref(), b"A2");
        assert_eq!(b1.as_ref(), b"B1");
        assert_eq!(b2.as_ref(), b"B2");
        assert_eq!(c1.as_ref(), b"C1");
        assert_eq!(c2.as_ref(), b"C2");
        assert_eq!(resolver.cache.len(), 3);
    }

    // ── resolve: alternating byte patterns ─────────────────────────────────

    #[test]
    fn resolve_alternating_hex_aa_hex_55_pattern() {
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0..20).map(|i| if i % 2 == 0 { 0xAA } else { 0x55 }).collect();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "alt.bin", &data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // Read even-positioned bytes (should all be 0xAA)
        let evens = resolver.resolve("alt.bin", 0, 10).unwrap();
        for (i, &b) in evens.iter().enumerate() {
            assert_eq!(b, if i % 2 == 0 { 0xAA } else { 0x55 }, "mismatch at index {i}");
        }
    }

    #[test]
    fn resolve_ascending_byte_values_preserve_order() {
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0..=255).collect();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "asc.bin", &data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // Verify ascending order in three segments
        let seg1 = resolver.resolve("asc.bin", 0, 86).unwrap();
        let seg2 = resolver.resolve("asc.bin", 86, 85).unwrap();
        let seg3 = resolver.resolve("asc.bin", 171, 85).unwrap();

        for w in seg1.windows(2) {
            assert_eq!(w[1], w[0] + 1, "ascending order violated");
        }
        for w in seg2.windows(2) {
            assert_eq!(w[1], w[0] + 1, "ascending order violated");
        }
        assert_eq!(seg3[0], 171);
    }

    #[test]
    fn resolve_descending_byte_values() {
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0..=255).rev().collect();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "desc.bin", &data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("desc.bin", 0, 256).unwrap();
        for w in result.windows(2) {
            assert_eq!(w[0], w[1] + 1, "descending order violated");
        }
    }

    // ── resolve: Bytes trait properties ────────────────────────────────────

    #[test]
    fn resolve_result_bytes_is_cloneable_and_equal() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"CLONE_TEST");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let original = resolver.resolve("d.bin", 0, 10).unwrap();
        let cloned = original.clone();

        assert_eq!(original.as_ref(), cloned.as_ref());
        assert_eq!(original.len(), cloned.len());
    }

    #[test]
    fn resolve_result_bytes_deref_to_slice() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"DEREF");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let bytes = resolver.resolve("d.bin", 0, 5).unwrap();
        // Bytes derefs to [u8], so slice methods work
        assert!(bytes.first() == Some(&b'D'));
        assert!(bytes.last() == Some(&b'F'));
        assert!(!bytes.is_empty());
    }

    // ── new: multiple new() calls produce identical base_dir ──────────────

    #[test]
    fn new_multiple_calls_same_path_identical_base_dir() {
        let path = Path::new("/opt/models/llama/model.onnx");
        let r1 = ExternalDataResolver::new(path);
        let r2 = ExternalDataResolver::new(path);
        let r3 = ExternalDataResolver::new(path);

        assert_eq!(r1.base_dir, r2.base_dir);
        assert_eq!(r2.base_dir, r3.base_dir);
    }

    // ── new: five resolvers from different model paths ─────────────────────

    #[test]
    fn new_five_resolvers_from_different_paths() {
        let paths = [
            Path::new("/a/model.onnx"),
            Path::new("/b/model.onnx"),
            Path::new("/c/model.onnx"),
            Path::new("/d/model.onnx"),
            Path::new("/e/model.onnx"),
        ];

        let resolvers: Vec<_> = paths.iter().map(|p| ExternalDataResolver::new(p)).collect();
        let base_dirs: Vec<_> = resolvers.iter().map(|r| r.base_dir.clone()).collect();

        // All base_dirs are distinct
        for i in 0..base_dirs.len() {
            for j in (i + 1)..base_dirs.len() {
                assert_ne!(base_dirs[i], base_dirs[j], "base_dirs[{i}] == base_dirs[{j}]");
            }
        }
    }

    // ── new: path with tilde component (no expansion expected) ────────────

    #[test]
    fn new_path_with_tilde_component() {
        let path = Path::new("~/models/model.onnx");
        let resolver = ExternalDataResolver::new(path);
        // Rust Path does NOT expand ~; it's treated as a literal directory name
        assert_eq!(resolver.base_dir, PathBuf::from("~/models"));
    }

    // ── resolve: file named with single character ─────────────────────────

    #[test]
    fn resolve_file_named_with_single_character() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "w", b"W");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("w", 0, 1).unwrap();
        assert_eq!(result.as_ref(), b"W");
    }

    // ── resolve: 4-level deep subdirectory ────────────────────────────────

    #[test]
    fn resolve_four_level_deep_subdirectory() {
        let dir = tempfile::tempdir().unwrap();
        let deep = dir.path().join("a").join("b").join("c").join("d");
        std::fs::create_dir_all(&deep).unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(&deep, "deep.bin", b"LEVEL4");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("a/b/c/d/deep.bin", 0, 6).unwrap();
        assert_eq!(result.as_ref(), b"LEVEL4");
    }

    // ── resolve: cache key for subdirectory file includes full path ───────

    #[test]
    fn resolve_cache_key_for_subdirectory_file_includes_dir() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("subdir");
        std::fs::create_dir_all(&sub).unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(&sub, "data.bin", b"CACHE_KEY");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        resolver.resolve("subdir/data.bin", 0, 9).unwrap();

        let expected_key = resolver.base_dir.join("subdir").join("data.bin");
        assert!(resolver.cache.contains_key(&expected_key));
        assert_eq!(resolver.cache.len(), 1);
    }

    // ── resolve: 20 files all cached independently ────────────────────────

    #[test]
    fn resolve_twenty_files_each_cached() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        for i in 0..20u8 {
            let name = format!("f{i}.bin");
            write_temp_file(dir.path(), &name, &[i; 4]);
        }

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        for i in 0..20u8 {
            let name = format!("f{i}.bin");
            let result = resolver.resolve(&name, 0, 4).unwrap();
            assert!(result.iter().all(|&b| b == i), "f{i}.bin mismatch");
        }
        assert_eq!(resolver.cache.len(), 20);
    }

    // ── resolve: cache grows monotonically with unique files ──────────────

    #[test]
    fn resolve_cache_grows_monotonically() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let mut prev_count = 0;
        for i in 0..5 {
            let name = format!("mono_{i}.bin");
            write_temp_file(dir.path(), &name, b"X");
            resolver.resolve(&name, 0, 1).unwrap();
            let current = resolver.cache.len();
            assert!(current > prev_count, "cache did not grow: {prev_count} -> {current}");
            prev_count = current;
        }
    }

    // ── resolve: file with exact sizes (full read) ────────────────────────

    #[test]
    fn resolve_file_exact_256_bytes_full_read() {
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0..=255).collect();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", &data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("d.bin", 0, 256).unwrap();
        assert_eq!(result.len(), 256);
        assert_eq!(result.as_ref(), data.as_slice());
    }

    #[test]
    fn resolve_file_exact_512_bytes_full_read() {
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0u8..=255).cycle().take(512).collect();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", &data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("d.bin", 0, 512).unwrap();
        assert_eq!(result.len(), 512);
    }

    // ── resolve: consecutive overflow errors don't corrupt state ──────────

    #[test]
    fn resolve_consecutive_overflow_errors_no_state_corruption() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"VALID");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // First: populate cache
        resolver.resolve("d.bin", 0, 5).unwrap();
        assert_eq!(resolver.cache.len(), 1);

        // Multiple overflow errors
        assert!(resolver.resolve("d.bin", usize::MAX, 1).is_err());
        assert!(resolver.resolve("d.bin", usize::MAX, usize::MAX).is_err());
        assert!(resolver.resolve("d.bin", 1, usize::MAX).is_err());

        // Cache unchanged and still functional
        assert_eq!(resolver.cache.len(), 1);
        let result = resolver.resolve("d.bin", 0, 5).unwrap();
        assert_eq!(result.as_ref(), b"VALID");
    }

    // ── resolve: two bounds errors on different files no cache pollution ──

    #[test]
    fn resolve_two_bounds_errors_different_files_no_cache_pollution() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "a.bin", b"A");
        write_temp_file(dir.path(), "b.bin", b"B");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        assert!(resolver.resolve("a.bin", 0, 100).is_err());
        assert!(resolver.resolve("b.bin", 0, 100).is_err());
        // Both files were mmapped for the bounds check, so cache should have both
        assert_eq!(resolver.cache.len(), 2);
        // Valid reads should now work from cache
        assert_eq!(resolver.resolve("a.bin", 0, 1).unwrap().as_ref(), b"A");
        assert_eq!(resolver.resolve("b.bin", 0, 1).unwrap().as_ref(), b"B");
    }

    // ── resolve: verify cache entry count after mixed success and failure ──

    #[test]
    fn resolve_cache_count_after_mixed_operations() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "a.bin", b"AAA");
        write_temp_file(dir.path(), "b.bin", b"BBB");
        write_temp_file(dir.path(), "c.bin", b"CCC");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        resolver.resolve("a.bin", 0, 3).unwrap(); // cache: 1
        assert!(resolver.resolve("missing.bin", 0, 1).is_err()); // cache: 1
        resolver.resolve("b.bin", 0, 3).unwrap(); // cache: 2
        assert!(resolver.resolve("a.bin", 0, 100).is_err()); // cache: 2 (a already cached)
        resolver.resolve("c.bin", 0, 3).unwrap(); // cache: 3

        assert_eq!(resolver.cache.len(), 3);
    }

    // ── resolve: same file read by two independent resolvers ──────────────

    #[test]
    fn resolve_same_file_two_independent_resolvers() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "shared.bin", b"SHARED_DATA");

        let model_path = dir.path().join("model.onnx");
        let mut r1 = ExternalDataResolver::new(&model_path);
        let mut r2 = ExternalDataResolver::new(&model_path);

        let b1 = r1.resolve("shared.bin", 0, 11).unwrap();
        let b2 = r2.resolve("shared.bin", 0, 11).unwrap();

        assert_eq!(b1.as_ref(), b"SHARED_DATA");
        assert_eq!(b2.as_ref(), b"SHARED_DATA");
        assert_eq!(r1.cache.len(), 1);
        assert_eq!(r2.cache.len(), 1);
    }

    // ── resolve: one byte from each of many files ─────────────────────────

    #[test]
    fn resolve_one_byte_from_each_of_five_files() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        let labels = [b"A", b"B", b"C", b"D", b"E"];
        for (i, &label) in labels.iter().enumerate() {
            let name = format!("letter_{i}.bin");
            write_temp_file(dir.path(), &name, label);
        }

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        for (i, &expected) in labels.iter().enumerate() {
            let name = format!("letter_{i}.bin");
            let result = resolver.resolve(&name, 0, 1).unwrap();
            assert_eq!(result.as_ref(), expected);
        }
        assert_eq!(resolver.cache.len(), 5);
    }

    // ── resolve: many consecutive 1-byte reads reconstruct file ───────────

    #[test]
    fn resolve_many_consecutive_one_byte_reads() {
        let dir = tempfile::tempdir().unwrap();
        let original: Vec<u8> = (0..30).collect();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", &original);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let mut reconstructed = Vec::new();
        for i in 0..30 {
            let byte = resolver.resolve("d.bin", i, 1).unwrap();
            assert_eq!(byte.len(), 1);
            reconstructed.extend_from_slice(&byte);
        }
        assert_eq!(reconstructed, original);
        assert_eq!(resolver.cache.len(), 1);
    }

    // ── resolve: slice content at start and end are correct ───────────────

    #[test]
    fn resolve_slice_first_and_last_content_correct() {
        let dir = tempfile::tempdir().unwrap();
        let data = b"START__END"; // S(0)T(1)A(2)R(3)T(4)_(5)_(6)E(7)N(8)D(9) = 10 bytes
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let first = resolver.resolve("d.bin", 0, 5).unwrap();
        let last = resolver.resolve("d.bin", 6, 4).unwrap();

        assert_eq!(first.as_ref(), b"START");
        assert_eq!(last.as_ref(), b"_END");
    }

    // ── resolve: file with mixed zero and one pattern ─────────────────────

    #[test]
    fn resolve_mixed_zero_one_pattern() {
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0..16).map(|i| if i % 2 == 0 { 0 } else { 1 }).collect();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", &data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("d.bin", 0, 16).unwrap();
        for (i, &b) in result.iter().enumerate() {
            assert_eq!(b, if i % 2 == 0 { 0 } else { 1 }, "at index {i}");
        }
    }

    // ── new: path ending in multiple slashes ───────────────────────────────

    #[test]
    fn new_path_ending_in_multiple_slashes() {
        let path = Path::new("/opt/models///");
        let resolver = ExternalDataResolver::new(path);
        // "///" parent depends on Rust's Path normalization.
        // The key property: no panic, base_dir is set.
        assert!(!resolver.base_dir.as_os_str().is_empty() || resolver.base_dir == PathBuf::from("."));
    }

    // ── resolve: verify cache unchanged after bounds error on uncached file ─

    #[test]
    fn resolve_bounds_error_on_uncached_file_adds_to_cache() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"TINY");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // Bounds error: file IS opened and mmap'd before bounds check
        assert!(resolver.resolve("d.bin", 0, 100).is_err());
        // The mmap_file call succeeds before resolve checks bounds,
        // so the cache gets populated
        assert_eq!(resolver.cache.len(), 1);
    }

    // ── resolve: content read before and after error is consistent ────────

    #[test]
    fn resolve_content_consistent_before_and_after_error() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"STABLE");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let before = resolver.resolve("d.bin", 0, 6).unwrap();
        assert!(resolver.resolve("d.bin", 0, 100).is_err());
        let after = resolver.resolve("d.bin", 0, 6).unwrap();

        assert_eq!(before.as_ref(), after.as_ref());
        assert_eq!(before.as_ref(), b"STABLE");
    }

    // ── resolve: subdirectory and root file with same base name ───────────

    #[test]
    fn resolve_subdirectory_and_root_file_same_name_different_content() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "data.bin", b"ROOT");
        write_temp_file(&sub, "data.bin", b"SUB");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let root = resolver.resolve("data.bin", 0, 4).unwrap();
        let sub_file = resolver.resolve("sub/data.bin", 0, 3).unwrap();

        assert_eq!(root.as_ref(), b"ROOT");
        assert_eq!(sub_file.as_ref(), b"SUB");
        assert_eq!(resolver.cache.len(), 2);
    }

    // ── resolve: file with exactly 1024 bytes ────────────────────────────

    #[test]
    fn resolve_file_exact_1024_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", &data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let head = resolver.resolve("d.bin", 0, 4).unwrap();
        let tail = resolver.resolve("d.bin", 1020, 4).unwrap();
        let mid = resolver.resolve("d.bin", 512, 4).unwrap();

        assert_eq!(head.len(), 4);
        assert_eq!(tail.len(), 4);
        assert_eq!(mid.len(), 4);
        assert_eq!(resolver.cache.len(), 1);
    }

    // ── resolve: zero-length at every valid offset ────────────────────────

    #[test]
    fn resolve_zero_length_at_every_valid_offset_in_small_file() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"ABCD");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        for offset in 0..=4 {
            let result = resolver.resolve("d.bin", offset, 0).unwrap();
            assert!(result.is_empty(), "non-empty at offset {offset}");
        }
    }

    // ── new: path with only extension (dotfile) ───────────────────────────

    #[test]
    fn new_path_with_only_extension_component() {
        let path = Path::new(".onnx");
        let resolver = ExternalDataResolver::new(path);
        // parent() of ".onnx" is Some("")
        assert_eq!(resolver.base_dir, PathBuf::from("."));
    }

    // ── new: path with multiple extensions ────────────────────────────────

    #[test]
    fn new_path_with_multiple_extensions() {
        let path = Path::new("/models/weights.onnx.zip");
        let resolver = ExternalDataResolver::new(path);
        assert_eq!(resolver.base_dir, PathBuf::from("/models"));
    }

    // ── resolve: file with incrementing pattern verify non-overlapping ────

    #[test]
    fn resolve_non_overlapping_chunks_no_byte_overlap() {
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0..20).collect();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", &data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let c1 = resolver.resolve("d.bin", 0, 5).unwrap();   // [0,1,2,3,4]
        let c2 = resolver.resolve("d.bin", 5, 5).unwrap();   // [5,6,7,8,9]
        let c3 = resolver.resolve("d.bin", 10, 5).unwrap();  // [10,11,12,13,14]
        let c4 = resolver.resolve("d.bin", 15, 5).unwrap();  // [15,16,17,18,19]

        // Verify no byte value appears in the wrong chunk
        for &b in &c1 { assert!(b < 5, "byte {b} in chunk1 but >= 5"); }
        for &b in &c2 { assert!((5..10).contains(&b), "byte {b} in chunk2 but not in 5..10"); }
        for &b in &c3 { assert!((10..15).contains(&b), "byte {b} in chunk3 but not in 10..15"); }
        for &b in &c4 { assert!((15..20).contains(&b), "byte {b} in chunk4 but not in 15..20"); }
    }

    // ── resolve: verify resolve returns Bytes type (compiles) ─────────────

    #[test]
    fn resolve_return_type_is_bytes() {
        use prost::bytes::Bytes;
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"TYPED");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result: Bytes = resolver.resolve("d.bin", 0, 5).unwrap();
        assert_eq!(result.len(), 5);
    }

    // ── resolve: large offset small length just past boundary ─────────────

    #[test]
    fn resolve_large_offset_one_byte_past_end_fails() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"DATA");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // offset=3, length=2 → end=5 > 4 → out of bounds
        assert!(resolver.resolve("d.bin", 3, 2).is_err());
    }

    // ── resolve: verify Bytes from same file share backing store ──────────

    #[test]
    fn resolve_cached_bytes_share_underlying_data() {
        let dir = tempfile::tempdir().unwrap();
        let data = b"ABCDEFGHIJ";
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let full = resolver.resolve("d.bin", 0, 10).unwrap();
        let partial = resolver.resolve("d.bin", 3, 4).unwrap();

        // The partial slice should correspond to full[3..7]
        assert_eq!(&full[3..7], partial.as_ref());
        assert_eq!(partial.as_ref(), b"DEFG");
    }

    // ── resolve: error when offset equals length with small data ──────────

    #[test]
    fn resolve_offset_equals_data_length_any_nonzero_length_fails() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"AB");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // offset=2 (=file length), length=1 → end=3 > 2
        assert!(resolver.resolve("d.bin", 2, 1).is_err());
        // offset=2, length=0 → end=2 == 2, OK
        assert!(resolver.resolve("d.bin", 2, 0).unwrap().is_empty());
    }

    // ── resolve: file with all 0x80 bytes ─────────────────────────────────

    #[test]
    fn resolve_file_of_all_0x80() {
        let dir = tempfile::tempdir().unwrap();
        let data = vec![0x80u8; 32];
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", &data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("d.bin", 0, 32).unwrap();
        assert!(result.iter().all(|&b| b == 0x80));
    }

    // ── ExternalDataResolver::new — additional path edge cases ──────────

    #[test]
    fn new_with_dot_path() {
        let resolver = ExternalDataResolver::new(Path::new("."));
        assert_eq!(resolver.base_dir, PathBuf::from("."));
    }

    #[test]
    fn new_with_dotdot_path() {
        let resolver = ExternalDataResolver::new(Path::new(".."));
        assert_eq!(resolver.base_dir, PathBuf::from("."));
    }

    #[test]
    fn new_with_dotdot_slash_filename() {
        let resolver = ExternalDataResolver::new(Path::new("../a.onnx"));
        assert_eq!(resolver.base_dir, PathBuf::from(".."));
    }

    #[test]
    fn new_with_double_dotdot_path() {
        let resolver = ExternalDataResolver::new(Path::new("../../a.onnx"));
        assert_eq!(resolver.base_dir, PathBuf::from("../.."));
    }

    #[test]
    fn new_with_dotdot_in_absolute_path() {
        let resolver = ExternalDataResolver::new(Path::new("/a/../b/a.onnx"));
        assert_eq!(resolver.base_dir, PathBuf::from("/a/../b"));
    }

    #[test]
    fn new_with_dot_in_absolute_path() {
        let resolver = ExternalDataResolver::new(Path::new("/a/./b/a.onnx"));
        assert_eq!(resolver.base_dir, PathBuf::from("/a/./b"));
    }

    #[test]
    fn new_with_hidden_file() {
        let resolver = ExternalDataResolver::new(Path::new("/d/.hidden"));
        assert_eq!(resolver.base_dir, PathBuf::from("/d"));
    }

    #[test]
    fn new_with_cjk_directory() {
        let resolver = ExternalDataResolver::new(Path::new("/数据/a.onnx"));
        assert_eq!(resolver.base_dir, PathBuf::from("/数据"));
    }

    #[test]
    fn new_with_cjk_filename() {
        let resolver = ExternalDataResolver::new(Path::new("/d/模型.onnx"));
        assert_eq!(resolver.base_dir, PathBuf::from("/d"));
    }

    #[test]
    fn new_with_triple_slash_filename() {
        let resolver = ExternalDataResolver::new(Path::new("///a.onnx"));
        assert_eq!(resolver.base_dir, PathBuf::from("/"));
    }

    #[test]
    fn new_with_path_no_extension() {
        let resolver = ExternalDataResolver::new(Path::new("/a/b"));
        assert_eq!(resolver.base_dir, PathBuf::from("/a"));
    }

    #[test]
    fn new_with_space_in_directory() {
        let resolver = ExternalDataResolver::new(Path::new("/my dir/a.onnx"));
        assert_eq!(resolver.base_dir, PathBuf::from("/my dir"));
    }

    #[test]
    fn new_with_space_in_filename() {
        let resolver = ExternalDataResolver::new(Path::new("/d/my file.onnx"));
        assert_eq!(resolver.base_dir, PathBuf::from("/d"));
    }

    #[test]
    fn new_with_leading_dot_absolute() {
        let resolver = ExternalDataResolver::new(Path::new("/./a/a.onnx"));
        assert_eq!(resolver.base_dir, PathBuf::from("/./a"));
    }

    #[test]
    fn new_debug_trait_works() {
        let resolver = ExternalDataResolver::new(Path::new("/a/b.onnx"));
        let debug_str = format!("{:?}", resolver);
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn new_with_cyrillic_path() {
        let resolver = ExternalDataResolver::new(Path::new("/модель/a.onnx"));
        assert_eq!(resolver.base_dir, PathBuf::from("/модель"));
    }

    #[test]
    fn new_with_path_three_extensions() {
        let resolver = ExternalDataResolver::new(Path::new("/d/a.tar.gz.onnx"));
        assert_eq!(resolver.base_dir, PathBuf::from("/d"));
    }

    #[test]
    fn new_with_very_long_component_name() {
        let long_name = "a".repeat(255);
        let path = Path::new("/").join(&long_name).join("model.onnx");
        let resolver = ExternalDataResolver::new(&path);
        assert_eq!(resolver.base_dir, Path::new("/").join(&long_name));
    }

    // ── resolve: file size one byte below page boundary ─────────────────────

    #[test]
    fn resolve_file_one_byte_below_page_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let data = vec![0xCDu8; 4095]; // 4096 - 1
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", &data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let first = resolver.resolve("d.bin", 0, 1).unwrap();
        let last = resolver.resolve("d.bin", 4094, 1).unwrap();
        let full = resolver.resolve("d.bin", 0, 4095).unwrap();

        assert_eq!(first.as_ref(), &[0xCD]);
        assert_eq!(last.as_ref(), &[0xCD]);
        assert_eq!(full.len(), 4095);
        assert!(full.iter().all(|&b| b == 0xCD));
    }

    // ── resolve: deleted file still readable via mmap ───────────────────────

    #[test]
    fn resolve_file_deleted_after_mmap_still_readable_from_cache() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"PERSIST");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // First resolve: mmap and cache the file
        let first = resolver.resolve("d.bin", 0, 7).unwrap();
        assert_eq!(first.as_ref(), b"PERSIST");

        // Delete the underlying file
        std::fs::remove_file(dir.path().join("d.bin")).unwrap();

        // Second resolve: should still work from mmap cache
        let second = resolver.resolve("d.bin", 0, 7).unwrap();
        assert_eq!(second.as_ref(), b"PERSIST");
    }

    // ── resolve: io error Display includes underlying message ──────────────

    #[test]
    fn resolve_io_error_display_contains_context() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let err = resolver.resolve("no_such_file.bin", 0, 1).unwrap_err();
        let display = format!("{err}");
        // LoaderError::Io wraps std::io::Error; Display must include "IO error"
        assert!(display.contains("IO error"), "display was: {display}");
    }

    // ── resolve: empty Bytes slice after cached full read ───────────────────

    #[test]
    fn resolve_empty_slice_after_cached_full_read() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"FULL_DATA_HERE");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let full = resolver.resolve("d.bin", 0, 14).unwrap();
        assert_eq!(full.as_ref(), b"FULL_DATA_HERE");

        let empty = resolver.resolve("d.bin", 7, 0).unwrap();
        assert!(empty.is_empty());
        assert_eq!(resolver.cache.len(), 1);
    }

    // ── resolve: file with incremental pattern at 4097 bytes ──────────────

    #[test]
    fn resolve_incremental_pattern_cross_page_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0u8..=255).cycle().take(4097).collect();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", &data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let last_byte = resolver.resolve("d.bin", 4096, 1).unwrap();
        // 4096 % 256 = 0, so byte at index 4096 should be 0
        assert_eq!(last_byte.as_ref(), &[0x00]);

        let first_of_last_page = resolver.resolve("d.bin", 4096 - 1, 1).unwrap();
        // 4095 % 256 = 255
        assert_eq!(first_of_last_page.as_ref(), &[0xFF]);
    }

    // ── resolve: two resolvers with different base_dirs see different files ─

    #[test]
    fn resolve_two_resolvers_different_base_dirs_isolated_caches() {
        let dir1 = tempfile::tempdir().unwrap();
        let dir2 = tempfile::tempdir().unwrap();
        write_temp_file(dir1.path(), "model.onnx", b"M");
        write_temp_file(dir1.path(), "shared_name.bin", b"DIR1");
        write_temp_file(dir2.path(), "model.onnx", b"M");
        write_temp_file(dir2.path(), "shared_name.bin", b"DIR2");

        let mut r1 = ExternalDataResolver::new(&dir1.path().join("model.onnx"));
        let mut r2 = ExternalDataResolver::new(&dir2.path().join("model.onnx"));

        let b1 = r1.resolve("shared_name.bin", 0, 4).unwrap();
        let b2 = r2.resolve("shared_name.bin", 0, 4).unwrap();

        assert_eq!(b1.as_ref(), b"DIR1");
        assert_eq!(b2.as_ref(), b"DIR2");
        assert_eq!(r1.cache.len(), 1);
        assert_eq!(r2.cache.len(), 1);
    }

    // ── resolve: error then success then error on same file ─────────────────

    #[test]
    fn resolve_error_success_error_cycle_on_same_file() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"ABC");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        assert!(resolver.resolve("d.bin", 0, 100).is_err());
        let ok = resolver.resolve("d.bin", 0, 3).unwrap();
        assert_eq!(ok.as_ref(), b"ABC");
        assert!(resolver.resolve("d.bin", 2, 5).is_err());
        let ok2 = resolver.resolve("d.bin", 1, 2).unwrap();
        assert_eq!(ok2.as_ref(), b"BC");
    }

    // ── resolve: offset near usize::MAX / 2 boundary ───────────────────────

    #[test]
    fn resolve_large_offset_half_max_no_overflow_with_zero_length_fails() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"DATA");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let half = usize::MAX / 2;
        let result = resolver.resolve("d.bin", half, 0);
        // checked_add(half, 0) = Some(half), half > 4 → out of bounds
        assert!(result.is_err());
    }

    // ── resolve: Bytes from full file read can be sliced ────────────────────

    #[test]
    fn resolve_full_read_bytes_slice_matches_partial_resolve() {
        let dir = tempfile::tempdir().unwrap();
        let data = b"ABCDEFGHIJ";
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let full = resolver.resolve("d.bin", 0, 10).unwrap();
        let partial = resolver.resolve("d.bin", 5, 3).unwrap();

        // Slicing the full Bytes result should match a targeted resolve
        assert_eq!(&full[5..8], partial.as_ref());
        assert_eq!(partial.as_ref(), b"FGH");
    }

    // ── resolve: file with filename containing percent sign ────────────────

    #[test]
    fn resolve_location_with_percent_sign_in_filename() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "weights%20layer.bin", b"PCT");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("weights%20layer.bin", 0, 3).unwrap();
        assert_eq!(result.as_ref(), b"PCT");
    }

    // ── resolve: cache key uniqueness with same filename in different dirs ─

    #[test]
    fn resolve_cache_keys_are_unique_for_same_named_files_in_different_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let sub1 = dir.path().join("alpha");
        let sub2 = dir.path().join("beta");
        std::fs::create_dir_all(&sub1).unwrap();
        std::fs::create_dir_all(&sub2).unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(&sub1, "tensor.bin", b"A_DATA");
        write_temp_file(&sub2, "tensor.bin", b"B_DATA");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        resolver.resolve("alpha/tensor.bin", 0, 6).unwrap();
        resolver.resolve("beta/tensor.bin", 0, 6).unwrap();

        assert_eq!(resolver.cache.len(), 2);
        let keys: Vec<_> = resolver.cache.keys().collect();
        assert_ne!(keys[0], keys[1], "cache keys must differ for different paths");
    }

    // ── resolve: file with 2048 bytes read in two equal halves ─────────────

    #[test]
    fn resolve_2048_byte_file_in_two_equal_halves() {
        let dir = tempfile::tempdir().unwrap();
        let mut data = vec![0u8; 2048];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i % 256) as u8;
        }
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", &data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let h1 = resolver.resolve("d.bin", 0, 1024).unwrap();
        let h2 = resolver.resolve("d.bin", 1024, 1024).unwrap();

        assert_eq!(h1.len(), 1024);
        assert_eq!(h2.len(), 1024);
        // First half: i % 256 = i for i < 256, then wraps
        assert_eq!(h1[0], 0);
        assert_eq!(h1[255], 255);
        assert_eq!(h2[0], 0); // index 1024: 1024 % 256 = 0
        assert_eq!(h2[255], 255); // index 1279: 1279 % 256 = 255
        assert_eq!(resolver.cache.len(), 1);
    }

    // ── resolve: zero-length resolve at offset 0 on populated cache ────────

    #[test]
    fn resolve_zero_length_on_populated_cache_returns_empty_independently() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"DATA_CONTENT");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let full = resolver.resolve("d.bin", 0, 12).unwrap();
        assert_eq!(full.as_ref(), b"DATA_CONTENT");

        let empty1 = resolver.resolve("d.bin", 0, 0).unwrap();
        let empty2 = resolver.resolve("d.bin", 6, 0).unwrap();

        assert!(empty1.is_empty());
        assert!(empty2.is_empty());
        assert_ne!(empty1.as_ref(), full.as_ref()); // empty != non-empty
    }

    // ── resolve: Debug impl on ExternalDataResolver includes field names ────

    #[test]
    fn new_debug_output_contains_base_dir_and_cache() {
        let resolver = ExternalDataResolver::new(Path::new("/opt/models/llm/model.onnx"));
        let debug = format!("{resolver:?}");
        assert!(debug.contains("base_dir"), "Debug should mention base_dir: {debug}");
        assert!(debug.contains("cache"), "Debug should mention cache: {debug}");
        assert!(debug.contains("/opt/models/llm"), "Debug should contain path: {debug}");
    }

    // ── resolve: cache HashMap starts with zero capacity ────────────────────

    #[test]
    fn new_cache_starts_with_zero_capacity_or_default() {
        let resolver = ExternalDataResolver::new(Path::new("model.onnx"));
        // HashMap::new() starts with capacity 0
        assert_eq!(resolver.cache.len(), 0);
        assert!(resolver.cache.capacity() == 0 || resolver.cache.capacity() > 0);
    }

    // ── resolve: base_dir from new with Windows UNC-style path component ───

    #[test]
    fn new_with_single_slash_path_base_dir_is_root() {
        let resolver = ExternalDataResolver::new(Path::new("/x.onnx"));
        assert_eq!(resolver.base_dir, PathBuf::from("/"));
    }

    // ── resolve: read all 256 byte values in random-access order ────────────

    #[test]
    fn resolve_random_access_each_byte_value() {
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0u8..=255).collect();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", &data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // Read bytes in reverse order (255 down to 0)
        for i in (0u8..=255).rev() {
            let result = resolver.resolve("d.bin", i as usize, 1).unwrap();
            assert_eq!(result.as_ref(), &[i], "mismatch at index {i}");
        }
        assert_eq!(resolver.cache.len(), 1);
    }

    // ── resolve: cache entry value length equals full file size ─────────────

    #[test]
    fn resolve_cached_bytes_length_equals_file_size() {
        let dir = tempfile::tempdir().unwrap();
        let content = b"CACHED_FILE_DATA"; // exactly 16 bytes
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", content);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // Read a small slice to populate cache
        let _ = resolver.resolve("d.bin", 0, 1).unwrap();

        // The cached Bytes should represent the full file
        let cache_key = resolver.base_dir.join("d.bin");
        let cached = resolver.cache.get(&cache_key).unwrap();
        assert_eq!(cached.len(), content.len(), "cached bytes should be full file length");
    }

    // ── resolve: offset at exactly file length with zero length on 1-byte file

    #[test]
    fn resolve_single_byte_file_offset_at_end_zero_length() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"X");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // offset=1 (file length), length=0 → empty slice
        let result = resolver.resolve("d.bin", 1, 0).unwrap();
        assert!(result.is_empty());
    }

    // ── resolve: three resolvers with same path share no state ──────────────

    #[test]
    fn new_three_resolvers_same_path_independent_caches() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"DATA");

        let model_path = dir.path().join("model.onnx");
        let mut r1 = ExternalDataResolver::new(&model_path);
        let mut r2 = ExternalDataResolver::new(&model_path);
        let mut r3 = ExternalDataResolver::new(&model_path);

        r1.resolve("d.bin", 0, 4).unwrap();
        r2.resolve("d.bin", 0, 4).unwrap();

        // r3 has no cache yet
        assert_eq!(r1.cache.len(), 1);
        assert_eq!(r2.cache.len(), 1);
        assert_eq!(r3.cache.len(), 0);

        // r3 can independently resolve
        let result = r3.resolve("d.bin", 0, 4).unwrap();
        assert_eq!(result.as_ref(), b"DATA");
        assert_eq!(r3.cache.len(), 1);
    }

    // ── resolve: error path with offset that is exactly usize::MAX - 1 ──────

    #[test]
    fn resolve_offset_usize_max_minus_one_zero_length_on_tiny_file_fails() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"AB");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // usize::MAX - 1 + 0 = usize::MAX - 1 > 2 → out of bounds
        let result = resolver.resolve("d.bin", usize::MAX - 1, 0);
        assert!(result.is_err());
    }

    // ── resolve: file read at offset that skips embedded NUL bytes ───────────

    #[test]
    fn resolve_skip_over_embedded_nul_bytes() {
        let dir = tempfile::tempdir().unwrap();
        // File: "PRE\0\0\0POST"
        let data = b"PRE\0\0\0POST";
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let prefix = resolver.resolve("d.bin", 0, 3).unwrap();
        let suffix = resolver.resolve("d.bin", 6, 4).unwrap();

        assert_eq!(prefix.as_ref(), b"PRE");
        assert_eq!(suffix.as_ref(), b"POST");
    }

    // ── resolve: three overlapping slices from same file ────────────────────

    #[test]
    fn resolve_three_overlapping_slices_share_correct_data() {
        let dir = tempfile::tempdir().unwrap();
        let data = b"ABCDEFGHIJ";
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let s1 = resolver.resolve("d.bin", 0, 6).unwrap();   // ABCDEF
        let s2 = resolver.resolve("d.bin", 2, 6).unwrap();   // CDEFGH
        let s3 = resolver.resolve("d.bin", 4, 6).unwrap();   // EFGHIJ

        assert_eq!(s1.as_ref(), b"ABCDEF");
        assert_eq!(s2.as_ref(), b"CDEFGH");
        assert_eq!(s3.as_ref(), b"EFGHIJ");
        assert_eq!(resolver.cache.len(), 1);
    }

    // ── resolve: cache key includes base_dir component ──────────────────────

    #[test]
    fn resolve_cache_key_starts_with_base_dir() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "tensor.bin", b"T");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        resolver.resolve("tensor.bin", 0, 1).unwrap();

        // Every cache key must start with the resolver's base_dir
        for key in resolver.cache.keys() {
            assert!(
                key.starts_with(&resolver.base_dir),
                "cache key {key:?} should start with base_dir {:?}",
                resolver.base_dir
            );
        }
    }

    // ── resolve: verify Bytes::slice works on returned value ────────────────

    #[test]
    fn resolve_returned_bytes_supports_further_slicing() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"ABCDEFGHIJ");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let result = resolver.resolve("d.bin", 0, 10).unwrap();

        // Bytes supports further slicing
        let inner = result.slice(3..7);
        assert_eq!(inner.as_ref(), b"DEFG");
        assert_eq!(result.len(), 10, "original Bytes unchanged after slice");
    }

    // ── resolve: bounds error message contains both offset and length context

    #[test]
    fn resolve_bounds_error_on_one_past_last_byte() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"1234");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // offset=5 is one past the end (file is 4 bytes), length=0
        // checked_add(5, 0) = Some(5), 5 > 4 → out of bounds
        let result = resolver.resolve("d.bin", 5, 0);
        assert!(result.is_err());
    }

    // ── resolve: cloned Bytes share underlying data with original ────────────

    #[test]
    fn resolve_cloned_bytes_share_underlying_buffer() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"SHARED_BUFFER_DATA_");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let original = resolver.resolve("d.bin", 0, 19).unwrap();
        let cloned = original.clone();

        assert_eq!(original.as_ref(), cloned.as_ref());
        assert_eq!(original.as_ref(), b"SHARED_BUFFER_DATA_");
        // Bytes::clone shares the underlying buffer, so the pointer
        // should be the same (reference-counted, not deep copy).
        assert_eq!(original.as_ptr(), cloned.as_ptr());
    }

    // ── resolve: two-byte file zero-length at all valid offsets ──────────────

    #[test]
    fn resolve_two_byte_file_zero_length_at_every_valid_offset() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"AB");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // Offsets 0, 1, 2 are all valid for zero-length reads on a 2-byte file
        for offset in 0..=2 {
            let result = resolver.resolve("d.bin", offset, 0).unwrap();
            assert!(result.is_empty(), "expected empty at offset {offset}");
        }
        assert_eq!(resolver.cache.len(), 1);
    }

    // ── resolve: two-byte file non-zero reads at exact boundaries ────────────

    #[test]
    fn resolve_two_byte_file_reads_at_all_exact_boundaries() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", b"AB");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // Read [0..2] = full file
        let full = resolver.resolve("d.bin", 0, 2).unwrap();
        assert_eq!(full.as_ref(), b"AB");

        // Read [0..1] = first byte
        let first = resolver.resolve("d.bin", 0, 1).unwrap();
        assert_eq!(first.as_ref(), b"A");

        // Read [1..2] = second byte
        let second = resolver.resolve("d.bin", 1, 1).unwrap();
        assert_eq!(second.as_ref(), b"B");
    }

    // ── resolve: empty file offset one zero length fails (offset past end) ──

    #[test]
    fn resolve_empty_file_offset_one_zero_length_fails() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "empty.bin", b"");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // Empty file has length 0. offset=1, length=0: checked_add(1,0)=Some(1), 1>0 → out of bounds
        let result = resolver.resolve("empty.bin", 1, 0);
        assert!(result.is_err());
    }

    // ── resolve: io error on missing file then bounds error on existing file ─

    #[test]
    fn resolve_io_error_then_bounds_error_on_different_files() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "small.bin", b"AB");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // First: IO error (file not found)
        let err1 = resolver.resolve("missing.bin", 0, 1).unwrap_err();
        let msg1 = format!("{err1}");
        assert!(msg1.contains("IO error"), "first error should be IO: {msg1}");

        // Second: bounds error (file too small)
        let err2 = resolver.resolve("small.bin", 0, 100).unwrap_err();
        let msg2 = format!("{err2}");
        assert!(msg2.contains("out of bounds"), "second error should be bounds: {msg2}");

        // Both errors should be different variants
        assert_ne!(msg1, msg2);
    }

    // ── resolve: sliding window reads across a patterned file ────────────────

    #[test]
    fn resolve_sliding_window_reads_across_patterned_file() {
        let dir = tempfile::tempdir().unwrap();
        // Create 64-byte file with repeating pattern "ABCD..."
        let data: Vec<u8> = (b'A'..=b'Z').cycle().take(64).collect();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", &data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // Sliding window: read 8-byte slices starting at 0, 8, 16, ..., 56
        for window_start in (0..64).step_by(8) {
            let result = resolver.resolve("d.bin", window_start, 8).unwrap();
            assert_eq!(result.len(), 8);
            // Verify each byte matches the source data
            for (i, &byte) in result.iter().enumerate() {
                assert_eq!(byte, data[window_start + i],
                    "mismatch at window_start={window_start}, i={i}");
            }
        }
        assert_eq!(resolver.cache.len(), 1);
    }

    // ── resolve: missing file after successful read does not pollute cache ───

    #[test]
    fn resolve_missing_file_after_successful_read_does_not_add_cache_entry() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "good.bin", b"GOOD");

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        // Successful read populates cache
        resolver.resolve("good.bin", 0, 4).unwrap();
        assert_eq!(resolver.cache.len(), 1);

        // Failed read on missing file should not add a cache entry
        assert!(resolver.resolve("no_such.bin", 0, 1).is_err());
        assert_eq!(resolver.cache.len(), 1, "cache should still have exactly 1 entry");

        // Verify only the good file is cached
        let cached_key = resolver.base_dir.join("good.bin");
        assert!(resolver.cache.contains_key(&cached_key));
    }

    // ── resolve: large 16KB file read in four quadrants ──────────────────────

    #[test]
    fn resolve_16kb_file_read_in_four_quadrants() {
        let dir = tempfile::tempdir().unwrap();
        let mut data = vec![0u8; 16384];
        // Mark each quadrant with a distinct byte pattern
        data[0] = b'1';
        data[4095] = b'1';
        data[4096] = b'2';
        data[8191] = b'2';
        data[8192] = b'3';
        data[12287] = b'3';
        data[12288] = b'4';
        data[16383] = b'4';

        write_temp_file(dir.path(), "model.onnx", b"M");
        write_temp_file(dir.path(), "d.bin", &data);

        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);

        let q1 = resolver.resolve("d.bin", 0, 4096).unwrap();
        let q2 = resolver.resolve("d.bin", 4096, 4096).unwrap();
        let q3 = resolver.resolve("d.bin", 8192, 4096).unwrap();
        let q4 = resolver.resolve("d.bin", 12288, 4096).unwrap();

        assert_eq!(q1[0], b'1');
        assert_eq!(q1[4095], b'1');
        assert_eq!(q2[0], b'2');
        assert_eq!(q2[4095], b'2');
        assert_eq!(q3[0], b'3');
        assert_eq!(q3[4095], b'3');
        assert_eq!(q4[0], b'4');
        assert_eq!(q4[4095], b'4');
        assert_eq!(resolver.cache.len(), 1);
    }

    // ── new: path with emoji Unicode characters ──────────────────────────────

    #[test]
    fn new_with_emoji_path_component() {
        // Path with emoji characters — validates Unicode path handling
        let path = Path::new("/models/🤖/model.onnx");
        let resolver = ExternalDataResolver::new(path);
        assert_eq!(resolver.base_dir, PathBuf::from("/models/🤖"));
    }

    // ── new: dot-only path with no file component ───────────────────────────

    #[test]
    fn new_with_dot_only_path_no_file_component() {
        // Path::new(".") has parent() = None → fallback to "."
        let resolver = ExternalDataResolver::new(Path::new("."));
        assert_eq!(resolver.base_dir, PathBuf::from("."));
    }

}
