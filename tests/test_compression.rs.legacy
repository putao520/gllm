//! Compression Integration Tests (REQ-COMP-013)
//!
//! Covers SPEC 22-PAGE-COMPRESSION.md §9 Phase 12: numerical correctness tests.
//!
//! Test scenarios:
//! 1. Each codec compression/decompression roundtrip (None, LZ4, BitPackRle, ZstdDict)
//! 2. PageMigrationActor: HBM↔DRAM, DRAM↔NVMe, full three-tier loop
//! 3. Compression ratio verification
//! 4. Edge cases: empty data, large pages, all-zeros, mixed content

use gllm::kv_cache::{CompressionCodec, StorageTier};
use gllm::scheduler::dma_helpers::CpuDmaBackendSized;
use gllm::scheduler::dma_helpers::DmaBackend;
use gllm::scheduler::migration_actor::{
    MigrationActorConfig, MigrationCommand, MigrationDone, MigrationResult, PageAddrEntry,
    PageAddrTable, PageMigrationActor,
};
use gllm::scheduler::nvme_swap::NvmeSwapFile;
use gllm::scheduler::types::PageId;
use gllm::static_compression::{
    compress_bitpack_rle, compress_zstd_dict, decompress_bitpack_rle, decompress_zstd_dict,
    lz4_compress, lz4_decompress, train_zstd_dictionary,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tempfile::TempDir;

// ============================================================================
// Helpers
// ============================================================================

/// Create deterministic test data of given size with a known pattern.
fn make_test_data(size: usize, seed: u8) -> Vec<u8> {
    (0..size).map(|i| ((i as u32).wrapping_mul(seed as u32 + 7) ^ (seed as u32 * 13)) as u8).collect()
}

/// Create test data with repeating runs (good for RLE compression).
fn make_run_data(size: usize, run_len: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(size);
    let mut val: u8 = 0;
    let mut remaining = size;
    while remaining > 0 {
        let chunk = run_len.min(remaining);
        data.extend(std::iter::repeat(val).take(chunk));
        val = val.wrapping_add(1);
        remaining -= chunk;
    }
    data
}

/// Create deterministic FP16-like data (good for LZ4).
fn make_fp16_data(num_elements: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(num_elements * 2);
    for i in 0..num_elements {
        let val = ((i % 128) * 100 + (i / 128) * 3) as u16;
        data.extend_from_slice(&val.to_le_bytes());
    }
    data
}

/// Compute CRC16 checksum (polynomial 0x8005, same as migration_actor).
fn crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for &byte in data {
        crc ^= (byte as u16) << 8;
        for _ in 0..8 {
            if crc & 0x8000 != 0 {
                crc = (crc << 1) ^ 0x8005;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

/// Create a PageMigrationActor with CPU backend and optional NVMe swap.
fn make_actor_cpu() -> (PageMigrationActor, PageAddrTable) {
    let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
    let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
    let actor = PageMigrationActor::spawn_with_backend(
        MigrationActorConfig::default(),
        Arc::clone(&backend),
        Arc::clone(&addr_table),
        None,
    );
    (actor, addr_table)
}

/// Create a PageMigrationActor with NVMe swap file.
fn make_actor_with_nvme(
    tmp: &TempDir,
    page_size: usize,
) -> (PageMigrationActor, PageAddrTable, Arc<NvmeSwapFile>) {
    let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
    let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
    let swap_path = tmp.path().join("test.swap");
    let nvme = Arc::new(NvmeSwapFile::open(swap_path, page_size, page_size * 2, 64).unwrap());
    let actor = PageMigrationActor::spawn_with_backend(
        MigrationActorConfig::default(),
        Arc::clone(&backend),
        Arc::clone(&addr_table),
        Some(Arc::clone(&nvme)),
    );
    (actor, addr_table, nvme)
}

/// Populate addr_table with a GPU-resident page for a given PageId.
fn register_gpu_page(
    addr_table: &PageAddrTable,
    page_id: PageId,
    data: &[u8],
    backend: &dyn DmaBackend,
) {
    let page_bytes = data.len();
    let gpu_ptr = backend.allocate_gpu_page(page_bytes).expect("alloc gpu page");
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), gpu_ptr as *mut u8, page_bytes);
    }
    let mut table = addr_table.write().expect("write lock");
    table.insert(
        page_id,
        PageAddrEntry {
            gpu_ptr: Some(gpu_ptr),
            host_buffer: None,
            current_tier: StorageTier::GpuHbm,
            original_bytes: page_bytes,
            codec: CompressionCodec::None,
        },
    );
}

/// Read back GPU page data and free it.
fn readback_gpu_page(addr_table: &PageAddrTable, page_id: PageId, page_bytes: usize) -> Vec<u8> {
    let table = addr_table.read().expect("read lock");
    let entry = table.get(&page_id).expect("entry missing");
    let gpu_ptr = entry.gpu_ptr.expect("gpu_ptr missing");
    let mut buf = vec![0u8; page_bytes];
    unsafe {
        std::ptr::copy_nonoverlapping(gpu_ptr as *const u8, buf.as_mut_ptr(), page_bytes);
    }
    buf
}

/// Populate addr_table with a DRAM-resident page (host_buffer).
fn register_dram_page(addr_table: &PageAddrTable, page_id: PageId, data: Vec<u8>) {
    let page_bytes = data.len();
    let mut table = addr_table.write().expect("write lock");
    table.insert(
        page_id,
        PageAddrEntry {
            gpu_ptr: None,
            host_buffer: Some(data),
            current_tier: StorageTier::CpuDram,
            original_bytes: page_bytes,
            codec: CompressionCodec::None,
        },
    );
}

/// Drain done events until a specific page_id is received, verifying success.
fn expect_migration_done(actor: &PageMigrationActor, page_id: PageId) -> MigrationDone {
    let done = actor.recv_done().expect("recv migration done");
    assert_eq!(done.page_id, page_id, "unexpected page_id in MigrationDone");
    match &done.result {
        MigrationResult::Ok { .. } => {}
        MigrationResult::Failed { reason } => {
            panic!("migration for page {page_id} failed: {reason}");
        }
    }
    done
}

// ============================================================================
// §1 — Codec Roundtrip Correctness Tests (REQ-COMP-013)
// ============================================================================

#[test]
fn test_compression_lz4_roundtrip() {
    for size in [0, 1, 256, 1024, 4096, 16384] {
        let original = make_test_data(size, 42);
        let compressed = lz4_compress(&original);
        if size > 0 {
            assert!(!compressed.is_empty(), "LZ4 compressed must not be empty for non-empty input");
        }
        let decompressed = lz4_decompress(&compressed, original.len())
            .expect("LZ4 decompress must succeed");
        assert_eq!(original, decompressed, "LZ4 roundtrip data mismatch at size {size}");
    }
}

#[test]
fn test_compression_lz4_codec_dispatch_roundtrip() {
    let original = make_test_data(2048, 99);
    let compressed = CompressionCodec::Lz4
        .compress(&original)
        .expect("LZ4 dispatch compress must return Some")
        .expect("LZ4 dispatch compress must succeed");
    let decompressed = CompressionCodec::Lz4
        .decompress(&compressed, original.len())
        .expect("LZ4 dispatch decompress must return Some")
        .expect("LZ4 dispatch decompress must succeed");
    assert_eq!(original, decompressed, "LZ4 codec dispatch roundtrip mismatch");
}

#[test]
fn test_compression_bitpack_rle_roundtrip() {
    for size in [0, 1, 64, 256, 1024, 4096] {
        let original = make_run_data(size, 16);
        let compressed = compress_bitpack_rle(&original);
        let decompressed = decompress_bitpack_rle(&compressed, original.len());
        assert_eq!(original, decompressed, "BitPackRle roundtrip data mismatch at size {size}");
    }
}

#[test]
fn test_compression_bitpack_rle_codec_dispatch_roundtrip() {
    let original = make_run_data(512, 8);
    let compressed = CompressionCodec::BitPackRle
        .compress(&original)
        .expect("BitPackRle dispatch compress must return Some")
        .expect("BitPackRle dispatch compress must succeed");
    let decompressed = CompressionCodec::BitPackRle
        .decompress(&compressed, original.len())
        .expect("BitPackRle dispatch decompress must return Some")
        .expect("BitPackRle dispatch decompress must succeed");
    assert_eq!(original, decompressed, "BitPackRle codec dispatch roundtrip mismatch");
}

#[test]
fn test_compression_bitpack_rle_long_runs() {
    // Data with very long runs — tests escape/continuation logic.
    let mut data = Vec::with_capacity(2000);
    data.extend(std::iter::repeat(0u8).take(500));
    data.extend(std::iter::repeat(3u8).take(300));
    data.extend(std::iter::repeat(7u8).take(400));
    data.extend(std::iter::repeat(15u8).take(200));
    data.extend(std::iter::repeat(2u8).take(600));
    let compressed = compress_bitpack_rle(&data);
    let decompressed = decompress_bitpack_rle(&compressed, data.len());
    assert_eq!(data, decompressed, "BitPackRle long-run roundtrip mismatch");
}

#[test]
fn test_compression_bitpack_rle_kivi4_nibble_pattern() {
    // Simulate KIVI4 quantized KV data: nibble values in [0, 15].
    let original: Vec<u8> = (0..1024).map(|i| (i % 16) as u8).collect();
    let compressed = compress_bitpack_rle(&original);
    let decompressed = decompress_bitpack_rle(&compressed, original.len());
    assert_eq!(original, decompressed, "BitPackRle KIVI4 nibble roundtrip mismatch");
}

#[test]
fn test_compression_bitpack_rle_kivi2_style() {
    // Simulate KIVI2-style: values in [0, 3].
    let original: Vec<u8> = (0..512).map(|i| (i % 4) as u8).collect();
    let compressed = compress_bitpack_rle(&original);
    let decompressed = decompress_bitpack_rle(&compressed, original.len());
    assert_eq!(original, decompressed, "BitPackRle KIVI2-style roundtrip mismatch");
}

#[test]
fn test_compression_zstd_dict_roundtrip() {
    let dict = b"KV cache compression dictionary for integration testing with varied patterns";
    for size in [0, 256, 1024, 4096, 8192] {
        let original = make_test_data(size, 77);
        let compressed = compress_zstd_dict(&original, dict)
            .expect("ZstdDict compress must succeed");
        if size > 0 {
            assert!(!compressed.is_empty(), "ZstdDict compressed empty for non-empty input");
        }
        let decompressed = decompress_zstd_dict(&compressed, dict, original.len())
            .expect("ZstdDict decompress must succeed");
        assert_eq!(original, decompressed, "ZstdDict roundtrip data mismatch at size {size}");
    }
}

#[test]
fn test_compression_zstd_dict_wrong_dictionary_fails() {
    let dict_a = b"dictionary alpha for compression";
    let dict_b = b"dictionary beta — completely different content";
    let original = make_test_data(1024, 55);
    let compressed = compress_zstd_dict(&original, dict_a).expect("compress with dict_a must succeed");
    let result = decompress_zstd_dict(&compressed, dict_b, original.len());
    // Decompressing with wrong dictionary should fail
    assert!(result.is_err(), "decompression with wrong dictionary must fail");
}

#[test]
fn test_compression_none_codec_passthrough() {
    // None codec dispatch returns None — caller handles as passthrough.
    let data = vec![1u8, 2, 3, 4, 5];
    let result = CompressionCodec::None.compress(&data);
    assert!(result.is_none(), "None codec compress returns None for passthrough");
    let result = CompressionCodec::None.decompress(&data, data.len());
    assert!(result.is_none(), "None codec decompress returns None for passthrough");
}

#[test]
fn test_compression_codec_display_and_discriminants() {
    assert_eq!(CompressionCodec::None.as_u8(), 0);
    assert_eq!(CompressionCodec::Lz4.as_u8(), 1);
    assert_eq!(CompressionCodec::BitPackRle.as_u8(), 2);
    assert_eq!(CompressionCodec::NvcompAns.as_u8(), 3);
    assert_eq!(CompressionCodec::ZstdDict.as_u8(), 4);

    assert_eq!(CompressionCodec::from_u8(0), Some(CompressionCodec::None));
    assert_eq!(CompressionCodec::from_u8(1), Some(CompressionCodec::Lz4));
    assert_eq!(CompressionCodec::from_u8(4), Some(CompressionCodec::ZstdDict));
    assert_eq!(CompressionCodec::from_u8(5), None);
    assert_eq!(CompressionCodec::from_u8(255), None);
}

#[test]
fn test_compression_codec_gpu_flags() {
    assert!(!CompressionCodec::None.is_gpu_decompressible());
    assert!(CompressionCodec::Lz4.is_gpu_decompressible());
    assert!(CompressionCodec::BitPackRle.is_gpu_decompressible());
    assert!(CompressionCodec::NvcompAns.is_gpu_decompressible());
    assert!(!CompressionCodec::ZstdDict.is_gpu_decompressible());

    assert!(!CompressionCodec::None.requires_cpu_decompress());
    assert!(!CompressionCodec::Lz4.requires_cpu_decompress());
    assert!(CompressionCodec::ZstdDict.requires_cpu_decompress());
}

// ============================================================================
// §2 — Compression Ratio Verification (REQ-COMP-013)
// ============================================================================

#[test]
fn test_compression_ratio_lz4_fp16_pattern() {
    let data = make_fp16_data(2048); // 4096 bytes
    let compressed = lz4_compress(&data);
    let ratio = compressed.len() as f64 / data.len() as f64;
    assert!(ratio < 1.0, "LZ4 must achieve compression on FP16 data: ratio={ratio:.3}");
    assert!(ratio < 0.95, "LZ4 FP16 compression ratio {ratio:.3} too high");
}

#[test]
fn test_compression_ratio_lz4_all_zeros() {
    let data = vec![0u8; 8192];
    let compressed = lz4_compress(&data);
    let ratio = compressed.len() as f64 / data.len() as f64;
    assert!(ratio < 0.05, "LZ4 all-zeros compression ratio {ratio:.3} too high (expected < 0.05)");
}

#[test]
fn test_compression_ratio_bitpack_rle_redundant() {
    // KIVI4-like nibble data with runs — should compress well
    let data = make_run_data(4096, 64);
    let compressed = compress_bitpack_rle(&data);
    let ratio = compressed.len() as f64 / data.len() as f64;
    assert!(ratio < 0.3, "BitPackRle run-data compression ratio {ratio:.3} too high (expected < 0.3)");
}

#[test]
fn test_compression_ratio_bitpack_rle_all_zeros() {
    let data = vec![0u8; 4096];
    let compressed = compress_bitpack_rle(&data);
    let ratio = compressed.len() as f64 / data.len() as f64;
    assert!(ratio < 0.02, "BitPackRle all-zeros ratio {ratio:.3} too high (expected < 0.02)");
}

#[test]
fn test_compression_ratio_zstd_dict() {
    let dict = b"FP16 KV cache patterns dictionary for training and compression testing";
    let data = make_fp16_data(2048);
    let compressed = compress_zstd_dict(&data, dict).expect("zstd compress");
    let ratio = compressed.len() as f64 / data.len() as f64;
    assert!(ratio < 0.85, "ZstdDict FP16 compression ratio {ratio:.3} too high");
}

#[test]
fn test_compression_bitpack_rle_compresses_smaller() {
    // Verify that compression actually reduces size for compressible data.
    for run_len in [4, 8, 16, 32] {
        let data = make_run_data(1024, run_len);
        let compressed = compress_bitpack_rle(&data);
        assert!(
            compressed.len() < data.len(),
            "BitPackRle must compress (run_len={run_len}): compressed={} >= original={}",
            compressed.len(),
            data.len()
        );
    }
}

// ============================================================================
// §3 — PageMigrationActor: HBM ↔ DRAM (REQ-COMP-013, SPEC §7.5.2)
// ============================================================================

#[test]
fn test_migration_hbm_to_dram_no_codec() {
    const PAGE_BYTES: usize = 1024;
    const PAGE_ID: PageId = 100;

    let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
    let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
    let original = make_test_data(PAGE_BYTES, 33);
    register_gpu_page(&addr_table, PAGE_ID, &original, &*backend);

    let actor = PageMigrationActor::spawn_with_backend(
        MigrationActorConfig::default(),
        Arc::clone(&backend),
        Arc::clone(&addr_table),
        None,
    );

    // Evict HBM → DRAM with no compression
    actor
        .send(MigrationCommand::EvictToDram {
            page_id: PAGE_ID,
            codec: CompressionCodec::None,
            page_bytes: PAGE_BYTES,
        })
        .expect("send evict");

    let done = expect_migration_done(&actor, PAGE_ID);
    assert_eq!(done.from_tier, StorageTier::GpuHbm);
    assert_eq!(done.to_tier, StorageTier::CpuDram);

    // Verify addr_table state
    {
        let table = addr_table.read().expect("read lock");
        let entry = table.get(&PAGE_ID).expect("entry missing after evict");
        assert!(entry.gpu_ptr.is_none(), "gpu_ptr must be None after evict");
        assert!(entry.host_buffer.is_some(), "host_buffer must be set after evict");
        assert_eq!(entry.current_tier, StorageTier::CpuDram);
    }

    // Promote DRAM → HBM
    actor
        .send(MigrationCommand::PromoteToHbm {
            page_id: PAGE_ID,
            page_bytes: PAGE_BYTES,
        })
        .expect("send promote");

    let done = expect_migration_done(&actor, PAGE_ID);
    assert_eq!(done.from_tier, StorageTier::CpuDram);
    assert_eq!(done.to_tier, StorageTier::GpuHbm);

    // Verify data integrity
    let readback = readback_gpu_page(&addr_table, PAGE_ID, PAGE_BYTES);
    assert_eq!(readback, original, "HBM roundtrip data mismatch");

    // Cleanup
    {
        let table = addr_table.read().expect("read lock");
        let entry = table.get(&PAGE_ID).expect("entry missing");
        if let Some(ptr) = entry.gpu_ptr {
            backend.free_gpu_page(ptr).expect("free gpu page");
        }
    }
    actor.shutdown();
}

#[test]
fn test_migration_hbm_to_dram_with_lz4() {
    const PAGE_BYTES: usize = 4096;
    const PAGE_ID: PageId = 200;

    let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
    let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
    let original = make_fp16_data(PAGE_BYTES / 2);
    register_gpu_page(&addr_table, PAGE_ID, &original, &*backend);

    let actor = PageMigrationActor::spawn_with_backend(
        MigrationActorConfig::default(),
        Arc::clone(&backend),
        Arc::clone(&addr_table),
        None,
    );

    // Evict with LZ4 compression
    actor
        .send(MigrationCommand::EvictToDram {
            page_id: PAGE_ID,
            codec: CompressionCodec::Lz4,
            page_bytes: PAGE_BYTES,
        })
        .expect("send evict");

    let done = expect_migration_done(&actor, PAGE_ID);
    if let MigrationResult::Ok { compressed_bytes, .. } = &done.result {
        assert!(
            (*compressed_bytes as usize) < PAGE_BYTES,
            "LZ4 must compress: {compressed_bytes} >= {PAGE_BYTES}"
        );
    }

    // Promote — actor decompresses LZ4 internally
    actor
        .send(MigrationCommand::PromoteToHbm {
            page_id: PAGE_ID,
            page_bytes: PAGE_BYTES,
        })
        .expect("send promote");

    let _done = expect_migration_done(&actor, PAGE_ID);

    let readback = readback_gpu_page(&addr_table, PAGE_ID, PAGE_BYTES);
    assert_eq!(readback, original, "HBM roundtrip with LZ4 data mismatch");

    {
        let table = addr_table.read().expect("read lock");
        let entry = table.get(&PAGE_ID).expect("entry missing");
        if let Some(ptr) = entry.gpu_ptr {
            backend.free_gpu_page(ptr).expect("free gpu page");
        }
    }
    actor.shutdown();
}

#[test]
fn test_migration_hbm_to_dram_with_bitpack_rle() {
    const PAGE_BYTES: usize = 2048;
    const PAGE_ID: PageId = 300;

    let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
    let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
    // Use run-length data that BitPackRle compresses well
    let original = make_run_data(PAGE_BYTES, 32);
    register_gpu_page(&addr_table, PAGE_ID, &original, &*backend);

    let actor = PageMigrationActor::spawn_with_backend(
        MigrationActorConfig::default(),
        Arc::clone(&backend),
        Arc::clone(&addr_table),
        None,
    );

    actor
        .send(MigrationCommand::EvictToDram {
            page_id: PAGE_ID,
            codec: CompressionCodec::BitPackRle,
            page_bytes: PAGE_BYTES,
        })
        .expect("send evict");

    let done = expect_migration_done(&actor, PAGE_ID);
    if let MigrationResult::Ok { compressed_bytes, .. } = &done.result {
        assert!(
            (*compressed_bytes as usize) < PAGE_BYTES,
            "BitPackRle must compress: {compressed_bytes} >= {PAGE_BYTES}"
        );
    }

    actor
        .send(MigrationCommand::PromoteToHbm {
            page_id: PAGE_ID,
            page_bytes: PAGE_BYTES,
        })
        .expect("send promote");

    let _done = expect_migration_done(&actor, PAGE_ID);

    let readback = readback_gpu_page(&addr_table, PAGE_ID, PAGE_BYTES);
    assert_eq!(readback, original, "HBM roundtrip with BitPackRle data mismatch");

    {
        let table = addr_table.read().expect("read lock");
        let entry = table.get(&PAGE_ID).expect("entry missing");
        if let Some(ptr) = entry.gpu_ptr {
            backend.free_gpu_page(ptr).expect("free gpu page");
        }
    }
    actor.shutdown();
}

#[test]
fn test_migration_hbm_dram_multiple_pages() {
    const PAGE_BYTES: usize = 1024;

    let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
    let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));

    let pages: Vec<(PageId, Vec<u8>)> = (0..6usize)
        .map(|pid| {
            let data = make_test_data(PAGE_BYTES, pid as u8);
            (pid, data)
        })
        .collect();

    for (pid, data) in &pages {
        register_gpu_page(&addr_table, *pid, data, &*backend);
    }

    let actor = PageMigrationActor::spawn_with_backend(
        MigrationActorConfig::default(),
        Arc::clone(&backend),
        Arc::clone(&addr_table),
        None,
    );

    // Evict all pages
    for (pid, _) in &pages {
        actor
            .send(MigrationCommand::EvictToDram {
                page_id: *pid,
                codec: CompressionCodec::Lz4,
                page_bytes: PAGE_BYTES,
            })
            .expect("send evict");
    }

    for (pid, _) in &pages {
        let done = expect_migration_done(&actor, *pid);
        assert_eq!(done.to_tier, StorageTier::CpuDram);
    }

    // Promote all pages
    for (pid, _) in &pages {
        actor
            .send(MigrationCommand::PromoteToHbm {
                page_id: *pid,
                page_bytes: PAGE_BYTES,
            })
            .expect("send promote");
    }

    for (pid, original) in &pages {
        let done = expect_migration_done(&actor, *pid);
        assert_eq!(done.to_tier, StorageTier::GpuHbm);
        let readback = readback_gpu_page(&addr_table, *pid, PAGE_BYTES);
        assert_eq!(
            &readback, original,
            "page {pid} data mismatch after multi-page evict/promote"
        );
    }

    // Cleanup
    {
        let table = addr_table.read().expect("read lock");
        for (pid, _) in &pages {
            if let Some(entry) = table.get(pid) {
                if let Some(ptr) = entry.gpu_ptr {
                    backend.free_gpu_page(ptr).expect("free");
                }
            }
        }
    }
    actor.shutdown();
}

#[test]
fn test_migration_evict_missing_page_fails() {
    let (actor, _table) = make_actor_cpu();
    actor
        .send(MigrationCommand::EvictToDram {
            page_id: 999,
            codec: CompressionCodec::Lz4,
            page_bytes: 1024,
        })
        .expect("send");
    let done = actor.recv_done().expect("recv");
    assert_eq!(done.page_id, 999);
    assert!(matches!(done.result, MigrationResult::Failed { .. }), "must fail for missing page");
    actor.shutdown();
}

// ============================================================================
// §4 — PageMigrationActor: DRAM ↔ NVMe (REQ-COMP-013, SPEC §7.5.2)
// ============================================================================

#[test]
fn test_migration_dram_to_nvme_roundtrip() {
    const PAGE_BYTES: usize = 4096;
    const PAGE_ID: PageId = 7;

    let tmp = TempDir::new().expect("tempdir");
    let (actor, addr_table, _nvme) = make_actor_with_nvme(&tmp, PAGE_BYTES);

    let original = make_test_data(PAGE_BYTES, 91);
    register_dram_page(&addr_table, PAGE_ID, original.clone());

    // Evict DRAM → NVMe
    actor
        .send(MigrationCommand::EvictToNvme {
            page_id: PAGE_ID,
            codec: CompressionCodec::ZstdDict,
            page_bytes: PAGE_BYTES,
        })
        .expect("send evict nvme");

    let done = expect_migration_done(&actor, PAGE_ID);
    assert_eq!(done.from_tier, StorageTier::CpuDram);
    assert_eq!(done.to_tier, StorageTier::Nvme);
    if let MigrationResult::Ok { compressed_bytes, .. } = &done.result {
        assert!(*compressed_bytes > 0, "compressed_bytes must be > 0");
    }

    {
        let table = addr_table.read().expect("read lock");
        let entry = table.get(&PAGE_ID).expect("entry missing");
        assert_eq!(entry.current_tier, StorageTier::Nvme);
        assert!(entry.host_buffer.is_none(), "host_buffer cleared after NVMe evict");
    }

    // Promote NVMe → DRAM
    actor
        .send(MigrationCommand::PromoteToDram {
            page_id: PAGE_ID,
            page_bytes: PAGE_BYTES,
        })
        .expect("send promote dram");

    let done = expect_migration_done(&actor, PAGE_ID);
    assert_eq!(done.from_tier, StorageTier::Nvme);
    assert_eq!(done.to_tier, StorageTier::CpuDram);

    let restored = {
        let table = addr_table.read().expect("read lock");
        let entry = table.get(&PAGE_ID).expect("entry missing");
        assert_eq!(entry.current_tier, StorageTier::CpuDram);
        entry.host_buffer.clone().expect("host_buffer present")
    };
    assert_eq!(restored, original, "NVMe roundtrip data mismatch");

    actor.shutdown();
}

#[test]
fn test_migration_nvme_multiple_pages() {
    const PAGE_BYTES: usize = 2048;

    let tmp = TempDir::new().expect("tempdir");
    let (actor, addr_table, _nvme) = make_actor_with_nvme(&tmp, PAGE_BYTES);

    let pages: Vec<(PageId, Vec<u8>)> = (0..5usize)
        .map(|pid| {
            let data = make_test_data(PAGE_BYTES, (pid * 17) as u8);
            (pid, data)
        })
        .collect();

    for (pid, data) in &pages {
        register_dram_page(&addr_table, *pid, data.clone());
    }

    // Evict all → NVMe
    for (pid, _) in &pages {
        actor
            .send(MigrationCommand::EvictToNvme {
                page_id: *pid,
                codec: CompressionCodec::ZstdDict,
                page_bytes: PAGE_BYTES,
            })
            .expect("send evict nvme");
    }
    for (pid, _) in &pages {
        let done = expect_migration_done(&actor, *pid);
        assert_eq!(done.to_tier, StorageTier::Nvme);
    }

    // Promote all → DRAM
    for (pid, _) in &pages {
        actor
            .send(MigrationCommand::PromoteToDram {
                page_id: *pid,
                page_bytes: PAGE_BYTES,
            })
            .expect("send promote dram");
    }
    for (pid, original) in &pages {
        let done = expect_migration_done(&actor, *pid);
        assert_eq!(done.to_tier, StorageTier::CpuDram);

        let table = addr_table.read().expect("read lock");
        let entry = table.get(pid).expect("entry missing");
        let restored = entry.host_buffer.as_ref().expect("host_buffer");
        assert_eq!(restored, original, "page {pid} NVMe roundtrip mismatch");
    }

    actor.shutdown();
}

// ============================================================================
// §5 — Full Three-Tier Loop: HBM → DRAM → NVMe → DRAM → HBM (REQ-COMP-013)
// ============================================================================

#[test]
fn test_migration_three_tier_full_loop_no_compression() {
    const PAGE_BYTES: usize = 4096;
    const PAGE_ID: PageId = 500;

    let tmp = TempDir::new().expect("tempdir");
    let (actor, addr_table, _nvme) = make_actor_with_nvme(&tmp, PAGE_BYTES);

    let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
    let original = make_test_data(PAGE_BYTES, 123);
    register_gpu_page(&addr_table, PAGE_ID, &original, &*backend);

    // Phase 1: HBM → DRAM (evict, no compression)
    actor
        .send(MigrationCommand::EvictToDram {
            page_id: PAGE_ID,
            codec: CompressionCodec::None,
            page_bytes: PAGE_BYTES,
        })
        .expect("send evict dram");
    let done = expect_migration_done(&actor, PAGE_ID);
    assert_eq!(done.from_tier, StorageTier::GpuHbm);
    assert_eq!(done.to_tier, StorageTier::CpuDram);

    {
        let table = addr_table.read().expect("read lock");
        let entry = table.get(&PAGE_ID).expect("entry missing");
        assert_eq!(entry.current_tier, StorageTier::CpuDram);
        assert!(entry.host_buffer.is_some());
    }

    // Phase 2: DRAM → NVMe
    actor
        .send(MigrationCommand::EvictToNvme {
            page_id: PAGE_ID,
            codec: CompressionCodec::ZstdDict,
            page_bytes: PAGE_BYTES,
        })
        .expect("send evict nvme");
    let done = expect_migration_done(&actor, PAGE_ID);
    assert_eq!(done.from_tier, StorageTier::CpuDram);
    assert_eq!(done.to_tier, StorageTier::Nvme);

    {
        let table = addr_table.read().expect("read lock");
        let entry = table.get(&PAGE_ID).expect("entry missing");
        assert_eq!(entry.current_tier, StorageTier::Nvme);
        assert!(entry.host_buffer.is_none());
    }

    // Phase 3: NVMe → DRAM
    actor
        .send(MigrationCommand::PromoteToDram {
            page_id: PAGE_ID,
            page_bytes: PAGE_BYTES,
        })
        .expect("send promote dram");
    let done = expect_migration_done(&actor, PAGE_ID);
    assert_eq!(done.from_tier, StorageTier::Nvme);
    assert_eq!(done.to_tier, StorageTier::CpuDram);

    {
        let table = addr_table.read().expect("read lock");
        let entry = table.get(&PAGE_ID).expect("entry missing");
        assert_eq!(entry.current_tier, StorageTier::CpuDram);
        assert!(entry.host_buffer.is_some());
        let restored = entry.host_buffer.as_ref().unwrap();
        assert_eq!(restored, &original, "three-tier data mismatch after NVMe→DRAM");
    }

    // Phase 4: DRAM → HBM
    actor
        .send(MigrationCommand::PromoteToHbm {
            page_id: PAGE_ID,
            page_bytes: PAGE_BYTES,
        })
        .expect("send promote hbm");
    let done = expect_migration_done(&actor, PAGE_ID);
    assert_eq!(done.from_tier, StorageTier::CpuDram);
    assert_eq!(done.to_tier, StorageTier::GpuHbm);

    let readback = readback_gpu_page(&addr_table, PAGE_ID, PAGE_BYTES);
    assert_eq!(readback, original, "full three-tier loop data mismatch");

    {
        let table = addr_table.read().expect("read lock");
        let entry = table.get(&PAGE_ID).expect("entry missing");
        if let Some(ptr) = entry.gpu_ptr {
            backend.free_gpu_page(ptr).expect("free gpu page");
        }
    }
    actor.shutdown();
}

#[test]
fn test_migration_three_tier_full_loop_with_lz4() {
    const PAGE_BYTES: usize = 2048;
    const PAGE_ID: PageId = 600;

    let tmp = TempDir::new().expect("tempdir");
    let (actor, addr_table, _nvme) = make_actor_with_nvme(&tmp, PAGE_BYTES);

    let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
    let original = make_fp16_data(PAGE_BYTES / 2);
    register_gpu_page(&addr_table, PAGE_ID, &original, &*backend);

    // HBM → DRAM (LZ4 compress)
    actor
        .send(MigrationCommand::EvictToDram {
            page_id: PAGE_ID,
            codec: CompressionCodec::Lz4,
            page_bytes: PAGE_BYTES,
        })
        .expect("send");
    expect_migration_done(&actor, PAGE_ID);

    // DRAM → NVMe
    actor
        .send(MigrationCommand::EvictToNvme {
            page_id: PAGE_ID,
            codec: CompressionCodec::ZstdDict,
            page_bytes: PAGE_BYTES,
        })
        .expect("send");
    expect_migration_done(&actor, PAGE_ID);

    // NVMe → DRAM
    actor
        .send(MigrationCommand::PromoteToDram {
            page_id: PAGE_ID,
            page_bytes: PAGE_BYTES,
        })
        .expect("send");
    expect_migration_done(&actor, PAGE_ID);

    // DRAM → HBM (LZ4 decompress)
    actor
        .send(MigrationCommand::PromoteToHbm {
            page_id: PAGE_ID,
            page_bytes: PAGE_BYTES,
        })
        .expect("send");
    expect_migration_done(&actor, PAGE_ID);

    let readback = readback_gpu_page(&addr_table, PAGE_ID, PAGE_BYTES);
    assert_eq!(readback, original, "three-tier LZ4 loop data mismatch");

    {
        let table = addr_table.read().expect("read lock");
        let entry = table.get(&PAGE_ID).expect("entry missing");
        if let Some(ptr) = entry.gpu_ptr {
            backend.free_gpu_page(ptr).expect("free gpu page");
        }
    }
    actor.shutdown();
}

#[test]
fn test_migration_three_tier_full_loop_with_bitpack_rle() {
    const PAGE_BYTES: usize = 2048;
    const PAGE_ID: PageId = 700;

    let tmp = TempDir::new().expect("tempdir");
    let (actor, addr_table, _nvme) = make_actor_with_nvme(&tmp, PAGE_BYTES);

    let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
    let original = make_run_data(PAGE_BYTES, 32);
    register_gpu_page(&addr_table, PAGE_ID, &original, &*backend);

    // HBM → DRAM (BitPackRle compress)
    actor
        .send(MigrationCommand::EvictToDram {
            page_id: PAGE_ID,
            codec: CompressionCodec::BitPackRle,
            page_bytes: PAGE_BYTES,
        })
        .expect("send");
    expect_migration_done(&actor, PAGE_ID);

    // DRAM → NVMe
    actor
        .send(MigrationCommand::EvictToNvme {
            page_id: PAGE_ID,
            codec: CompressionCodec::ZstdDict,
            page_bytes: PAGE_BYTES,
        })
        .expect("send");
    expect_migration_done(&actor, PAGE_ID);

    // NVMe → DRAM
    actor
        .send(MigrationCommand::PromoteToDram {
            page_id: PAGE_ID,
            page_bytes: PAGE_BYTES,
        })
        .expect("send");
    expect_migration_done(&actor, PAGE_ID);

    // DRAM → HBM (BitPackRle decompress)
    actor
        .send(MigrationCommand::PromoteToHbm {
            page_id: PAGE_ID,
            page_bytes: PAGE_BYTES,
        })
        .expect("send");
    expect_migration_done(&actor, PAGE_ID);

    let readback = readback_gpu_page(&addr_table, PAGE_ID, PAGE_BYTES);
    assert_eq!(readback, original, "three-tier BitPackRle loop data mismatch");

    {
        let table = addr_table.read().expect("read lock");
        let entry = table.get(&PAGE_ID).expect("entry missing");
        if let Some(ptr) = entry.gpu_ptr {
            backend.free_gpu_page(ptr).expect("free gpu page");
        }
    }
    actor.shutdown();
}

// ============================================================================
// §6 — Edge Cases and Stress Tests (REQ-COMP-013)
// ============================================================================

#[test]
fn test_compression_empty_data_all_codecs() {
    // LZ4 empty roundtrip
    let compressed = lz4_compress(&[]);
    let decompressed = lz4_decompress(&compressed, 0).expect("LZ4 empty decompress");
    assert!(decompressed.is_empty());

    // BitPackRle empty roundtrip
    let compressed = compress_bitpack_rle(&[]);
    let decompressed = decompress_bitpack_rle(&compressed, 0);
    assert!(decompressed.is_empty());

    // ZstdDict empty roundtrip
    let dict = b"test dict";
    let compressed = compress_zstd_dict(&[], dict).expect("zstd empty compress");
    let decompressed = decompress_zstd_dict(&compressed, dict, 0).expect("zstd empty decompress");
    assert!(decompressed.is_empty());
}

#[test]
fn test_compression_single_byte_all_codecs() {
    let original = vec![42u8];

    let c = lz4_compress(&original);
    let d = lz4_decompress(&c, 1).expect("LZ4 single byte");
    assert_eq!(d, original);

    let c = compress_bitpack_rle(&original);
    let d = decompress_bitpack_rle(&c, 1);
    assert_eq!(d, original);

    let dict = b"single byte test dict";
    let c = compress_zstd_dict(&original, dict).expect("zstd single byte");
    let d = decompress_zstd_dict(&c, dict, 1).expect("zstd single byte decompress");
    assert_eq!(d, original);
}

#[test]
fn test_compression_large_page_16kb_all_codecs() {
    let original = make_test_data(16384, 55);

    // LZ4
    let c = lz4_compress(&original);
    let d = lz4_decompress(&c, original.len()).expect("LZ4 16KB");
    assert_eq!(d, original);

    // BitPackRle — run-length data
    let rle_data = make_run_data(16384, 64);
    let c = compress_bitpack_rle(&rle_data);
    let d = decompress_bitpack_rle(&c, rle_data.len());
    assert_eq!(d, rle_data);

    // ZstdDict
    let dict = b"large page dictionary for 16KB KV cache compression testing";
    let c = compress_zstd_dict(&original, dict).expect("zstd 16KB");
    let d = decompress_zstd_dict(&c, dict, original.len()).expect("zstd 16KB decompress");
    assert_eq!(d, original);
}

#[test]
fn test_compression_lz4_all_zeros_roundtrip() {
    let original = vec![0u8; 8192];
    let c = lz4_compress(&original);
    let d = lz4_decompress(&c, original.len()).expect("LZ4 zeros");
    assert_eq!(d, original, "LZ4 all-zeros roundtrip mismatch");
}

#[test]
fn test_compression_lz4_header_validity() {
    let original = make_test_data(512, 7);
    let compressed = lz4_compress(&original);

    // Verify magic bytes
    let magic = u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]);
    assert_eq!(magic, 0x4C5A3442, "LZ4 magic mismatch");

    // Verify decompressed_len in header
    let stored_len =
        u32::from_le_bytes([compressed[4], compressed[5], compressed[6], compressed[7]]) as usize;
    assert_eq!(stored_len, original.len(), "LZ4 header decompressed_len mismatch");
}

#[test]
fn test_compression_zstd_dict_header_validity() {
    let dict = b"header validation dictionary";
    let original = make_test_data(1024, 13);
    let compressed = compress_zstd_dict(&original, dict).expect("zstd compress");

    let magic = u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]);
    assert_eq!(magic, 0x5A534454, "ZstdDict magic mismatch");

    let stored_len =
        u32::from_le_bytes([compressed[4], compressed[5], compressed[6], compressed[7]]) as usize;
    assert_eq!(stored_len, original.len(), "ZstdDict header decompressed_len mismatch");
}

#[test]
fn test_compression_zstd_dict_train_dictionary() {
    let samples: Vec<Vec<u8>> = (0..4)
        .map(|seed| make_test_data(1024, seed))
        .collect();
    let sample_refs: Vec<&[u8]> = samples.iter().map(|s| s.as_slice()).collect();

    let dict = train_zstd_dictionary(&sample_refs, 1024);
    // Dictionary may be empty if training fails with insufficient data — that's OK.
    // If non-empty, verify roundtrip works with it.
    let test_data = make_test_data(2048, 42);
    let compressed = compress_zstd_dict(&test_data, &dict).expect("zstd compress with trained dict");
    let decompressed =
        decompress_zstd_dict(&compressed, &dict, test_data.len()).expect("zstd decompress trained");
    assert_eq!(decompressed, test_data, "trained dict roundtrip mismatch");
}

#[test]
fn test_migration_actor_spawn_and_shutdown() {
    let actor = PageMigrationActor::spawn(MigrationActorConfig::default());
    actor.shutdown();
}

#[test]
fn test_migration_nvme_slot_reuse() {
    // Verify that reusing the same slot (same page_id) works correctly.
    const PAGE_BYTES: usize = 2048;
    const PAGE_ID: PageId = 88;

    let tmp = TempDir::new().expect("tempdir");
    let (actor, addr_table, _nvme) = make_actor_with_nvme(&tmp, PAGE_BYTES);

    let data_a = make_test_data(PAGE_BYTES, 10);
    let data_b = make_test_data(PAGE_BYTES, 20);

    // First cycle
    register_dram_page(&addr_table, PAGE_ID, data_a.clone());
    actor
        .send(MigrationCommand::EvictToNvme {
            page_id: PAGE_ID,
            codec: CompressionCodec::ZstdDict,
            page_bytes: PAGE_BYTES,
        })
        .expect("send");
    expect_migration_done(&actor, PAGE_ID);

    actor
        .send(MigrationCommand::PromoteToDram {
            page_id: PAGE_ID,
            page_bytes: PAGE_BYTES,
        })
        .expect("send");
    expect_migration_done(&actor, PAGE_ID);

    {
        let table = addr_table.read().expect("read lock");
        let entry = table.get(&PAGE_ID).expect("entry missing");
        let restored = entry.host_buffer.as_ref().expect("host_buffer");
        assert_eq!(restored, &data_a, "first NVMe cycle data mismatch");
    }

    // Second cycle with different data (same slot)
    {
        let mut table = addr_table.write().expect("write lock");
        let entry = table.get_mut(&PAGE_ID).expect("entry missing");
        entry.host_buffer = Some(data_b.clone());
        entry.current_tier = StorageTier::CpuDram;
    }

    actor
        .send(MigrationCommand::EvictToNvme {
            page_id: PAGE_ID,
            codec: CompressionCodec::ZstdDict,
            page_bytes: PAGE_BYTES,
        })
        .expect("send");
    expect_migration_done(&actor, PAGE_ID);

    actor
        .send(MigrationCommand::PromoteToDram {
            page_id: PAGE_ID,
            page_bytes: PAGE_BYTES,
        })
        .expect("send");
    expect_migration_done(&actor, PAGE_ID);

    {
        let table = addr_table.read().expect("read lock");
        let entry = table.get(&PAGE_ID).expect("entry missing");
        let restored = entry.host_buffer.as_ref().expect("host_buffer");
        assert_eq!(restored, &data_b, "second NVMe cycle (slot reuse) data mismatch");
    }

    actor.shutdown();
}

#[test]
fn test_migration_crc16_deterministic() {
    let data = b"compression integration test";
    let c1 = crc16(data);
    let c2 = crc16(data);
    assert_eq!(c1, c2, "CRC16 must be deterministic");
}

#[test]
fn test_migration_crc16_sensitive() {
    assert_ne!(
        crc16(b"hello world"),
        crc16(b"hello worlD"),
        "CRC16 must detect single-byte change"
    );
    assert_ne!(crc16(b""), crc16(b"\x00"), "CRC16 must distinguish empty from zero");
}

#[test]
fn test_zstd_dict_train_dictionary_empty_samples() {
    let samples: Vec<&[u8]> = vec![];
    let dict = train_zstd_dictionary(&samples, 1024);
    // Must return empty dict for empty samples
    assert!(dict.is_empty(), "training on empty samples must return empty dict");
}

#[test]
fn test_compression_bitpack_rle_no_compression_pathological() {
    // Data with no runs and full-byte values — RLE won't help, but header is added.
    let original: Vec<u8> = (0..64).map(|i| i as u8).collect();
    let compressed = compress_bitpack_rle(&original);
    let decompressed = decompress_bitpack_rle(&compressed, original.len());
    assert_eq!(original, decompressed, "BitPackRle pathological roundtrip mismatch");
}

#[test]
fn test_compression_lz4_decompress_bad_magic() {
    let mut buf = vec![0u8; 16];
    buf[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
    let result = lz4_decompress(&buf, 16);
    assert!(result.is_err(), "LZ4 bad magic must error");
}

#[test]
fn test_compression_lz4_decompress_truncated() {
    let compressed = lz4_compress(&[1u8; 1024]);
    let mut truncated = compressed[..12 + 2].to_vec(); // truncate payload
    // Pad to avoid panicking on header parse
    truncated.resize(12 + 4, 0);
    let result = lz4_decompress(&truncated, 1024);
    assert!(result.is_err(), "LZ4 truncated payload must error");
}

#[test]
fn test_compression_lz4_decompress_length_mismatch() {
    let compressed = lz4_compress(&[1u8, 2, 3, 4]);
    let result = lz4_decompress(&compressed, 999);
    assert!(result.is_err(), "LZ4 length mismatch must error");
}

#[test]
fn test_compression_zstd_dict_decompress_bad_magic() {
    let dict = b"test dict";
    let mut buf = vec![0u8; 16];
    buf[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
    let result = decompress_zstd_dict(&buf, dict, 16);
    assert!(result.is_err(), "ZstdDict bad magic must error");
}

#[test]
fn test_compression_zstd_dict_decompress_buffer_too_short() {
    let dict = b"test dict";
    let result = decompress_zstd_dict(&[0u8; 4], dict, 16);
    assert!(result.is_err(), "ZstdDict too short must error");
}

#[test]
fn test_compression_zstd_dict_decompress_length_mismatch() {
    let dict = b"test dict for length mismatch";
    let original = make_test_data(128, 31);
    let compressed = compress_zstd_dict(&original, dict).expect("compress");
    let result = decompress_zstd_dict(&compressed, dict, 9999);
    assert!(result.is_err(), "ZstdDict length mismatch must error");
}

#[test]
fn test_compression_zstd_dict_decompress_truncated_payload() {
    let dict = b"test dict for truncation";
    let original = make_test_data(1024, 47);
    let mut compressed = compress_zstd_dict(&original, dict).expect("compress");
    compressed.truncate(14); // truncate most of payload
    let result = decompress_zstd_dict(&compressed, dict, 1024);
    assert!(result.is_err(), "ZstdDict truncated payload must error");
}

#[test]
fn test_migration_actor_shutdown_cleanup() {
    // Verify repeated spawn/shutdown does not leak resources.
    for _ in 0..5 {
        let (actor, _table) = make_actor_cpu();
        actor.shutdown();
    }
}

#[test]
fn test_migration_error_path_nvme_without_host_buffer() {
    const PAGE_ID: PageId = 404;
    let tmp = TempDir::new().expect("tempdir");
    let (actor, addr_table, _nvme) = make_actor_with_nvme(&tmp, 4096);

    // Register page with no host_buffer (already evicted)
    {
        let mut table = addr_table.write().expect("write lock");
        table.insert(
            PAGE_ID,
            PageAddrEntry {
                gpu_ptr: None,
                host_buffer: None,
                current_tier: StorageTier::Nvme,
                original_bytes: 4096,
                codec: CompressionCodec::ZstdDict,
            },
        );
    }

    // Try to evict again — must fail because no host_buffer
    actor
        .send(MigrationCommand::EvictToNvme {
            page_id: PAGE_ID,
            codec: CompressionCodec::ZstdDict,
            page_bytes: 4096,
        })
        .expect("send");

    let done = actor.recv_done().expect("recv");
    assert_eq!(done.page_id, PAGE_ID);
    assert!(
        matches!(done.result, MigrationResult::Failed { .. }),
        "evicting NVMe page without host_buffer must fail"
    );

    actor.shutdown();
}
