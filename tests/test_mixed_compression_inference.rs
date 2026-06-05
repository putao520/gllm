//! Mega-Kernel Mixed Compression Integration Test (REQ-COMP-013)
//!
//! Validates that KV pages with mixed compression codecs (LZ4, BitPackRle, uncompressed)
//! can coexist in the same batch, with correct KvPageHeader metadata and lossless
//! roundtrip through compression/decompression.
//!
//! This test simulates what the mega-kernel attention path does at runtime:
//! 1. Read KvPageHeader.codec (offset 0x28) and KvPageHeader.compressed_size (offset 0x2C)
//! 2. Branch on codec: 0=skip (uncompressed), 1=LZ4 decode, 2=BitPackRle decode
//! 3. Decompress into scratch buffer
//! 4. Compute attention using decompressed data
//!
//! The JIT-side VmInstr sequence is verified in gllm-kernels (attention_emit.rs).
//! This file tests the Rust-side data contract that the JIT code depends on.

use gllm::kv_cache::{CompressionCodec, KvPageHeader, StorageTier};
use gllm::static_compression::{
    compress_bitpack_rle, decompress_bitpack_rle, lz4_compress, lz4_decompress,
};

/// Page size: 56B header + 4040B data = 4096B total (typical KV page).
const PAGE_SIZE: usize = 4096;
const HEADER_SIZE: usize = 56;
const DATA_SIZE: usize = PAGE_SIZE - HEADER_SIZE;

/// Simulated FP16 KV data (little-endian u16 pairs).
fn make_fp16_kv_data(size: usize, seed: u8) -> Vec<u8> {
    (0..size).map(|i| ((i as u32).wrapping_mul(seed as u32 + 7) ^ (seed as u32 * 13)) as u8).collect()
}

/// Simulated KIVI4 nibble stream data (values 0-15 with runs).
fn make_nibble_kv_data(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 16) as u8).collect()
}

/// Build a complete KV page: [KvPageHeader(56B) | payload].
fn build_kv_page(page_id: u32, codec: CompressionCodec, raw_data: &[u8]) -> Vec<u8> {
    assert_eq!(raw_data.len(), DATA_SIZE, "raw data must be exactly DATA_SIZE bytes");

    let (payload, compressed_size) = match codec {
        CompressionCodec::None => (raw_data.to_vec(), 0u32),
        CompressionCodec::Lz4 => {
            let c = lz4_compress(raw_data);
            let csz = c.len() as u32;
            (c, csz)
        }
        CompressionCodec::BitPackRle => {
            let c = compress_bitpack_rle(raw_data);
            let csz = c.len() as u32;
            (c, csz)
        }
        _ => panic!("unsupported codec for this test: {:?}", codec),
    };

    let mut header = KvPageHeader::new(page_id);
    header.codec = codec;
    header.storage_tier = StorageTier::GpuHbm;
    header.compressed_size = compressed_size;

    let mut page = Vec::with_capacity(PAGE_SIZE);
    let header_bytes = unsafe {
        std::slice::from_raw_parts(&header as *const KvPageHeader as *const u8, HEADER_SIZE)
    };
    page.extend_from_slice(header_bytes);

    if codec == CompressionCodec::None {
        page.extend_from_slice(&payload);
        assert_eq!(page.len(), PAGE_SIZE);
    } else {
        // Compressed payload may be shorter — pad to DATA_SIZE for consistent layout
        page.extend_from_slice(&payload);
        page.resize(PAGE_SIZE, 0);
    }

    page
}

/// Read KvPageHeader from the first 56 bytes of a page.
fn read_header(page: &[u8]) -> KvPageHeader {
    assert!(page.len() >= HEADER_SIZE, "page too short for header");
    unsafe {
        let ptr = page.as_ptr() as *const KvPageHeader;
        std::ptr::read(ptr)
    }
}

/// Decompress page payload based on codec field in KvPageHeader.
/// Returns the decompressed data (DATA_SIZE bytes).
fn decompress_page(page: &[u8]) -> Vec<u8> {
    let header = read_header(page);
    let payload = &page[HEADER_SIZE..];

    match header.codec {
        CompressionCodec::None => {
            // Uncompressed: data starts right after header
            assert!(payload.len() >= DATA_SIZE, "uncompressed page too short");
            payload[..DATA_SIZE].to_vec()
        }
        CompressionCodec::Lz4 => {
            let csz = header.compressed_size as usize;
            assert!(payload.len() >= csz, "LZ4 compressed data truncated");
            lz4_decompress(&payload[..csz], DATA_SIZE)
                .expect("LZ4 decompress must succeed")
        }
        CompressionCodec::BitPackRle => {
            let csz = header.compressed_size as usize;
            assert!(payload.len() >= csz, "BitPackRle compressed data truncated");
            decompress_bitpack_rle(&payload[..csz], DATA_SIZE)
        }
        _ => panic!("unexpected codec: {:?}", header.codec),
    }
}

// ============================================================================
// §1 — KvPageHeader Layout Contract (offset 0x28 = codec, 0x2C = compressed_size)
// ============================================================================

#[test]
fn test_kv_page_header_codec_offset() {
    let header = KvPageHeader::new(42);
    let bytes = unsafe {
        std::slice::from_raw_parts(&header as *const KvPageHeader as *const u8, HEADER_SIZE)
    };
    // codec field must be at offset 0x28 (40)
    assert_eq!(
        bytes[0x28],
        CompressionCodec::None.as_u8(),
        "codec at offset 0x28 must be None(0)"
    );
}

#[test]
fn test_kv_page_header_codec_offset_lz4() {
    let mut header = KvPageHeader::new(1);
    header.codec = CompressionCodec::Lz4;
    let bytes = unsafe {
        std::slice::from_raw_parts(&header as *const KvPageHeader as *const u8, HEADER_SIZE)
    };
    assert_eq!(bytes[0x28], 1, "LZ4 codec must be 1 at offset 0x28");
}

#[test]
fn test_kv_page_header_codec_offset_bitpack_rle() {
    let mut header = KvPageHeader::new(2);
    header.codec = CompressionCodec::BitPackRle;
    let bytes = unsafe {
        std::slice::from_raw_parts(&header as *const KvPageHeader as *const u8, HEADER_SIZE)
    };
    assert_eq!(bytes[0x28], 2, "BitPackRle codec must be 2 at offset 0x28");
}

#[test]
fn test_kv_page_header_compressed_size_offset() {
    let mut header = KvPageHeader::new(3);
    header.compressed_size = 0xDEADBEEF;
    let bytes = unsafe {
        std::slice::from_raw_parts(&header as *const KvPageHeader as *const u8, HEADER_SIZE)
    };
    let stored = u32::from_le_bytes([bytes[0x2C], bytes[0x2D], bytes[0x2E], bytes[0x2F]]);
    assert_eq!(stored, 0xDEADBEEF, "compressed_size at offset 0x2C must match");
}

#[test]
fn test_kv_page_header_size_is_56() {
    assert_eq!(
        std::mem::size_of::<KvPageHeader>(),
        56,
        "KvPageHeader must be exactly 56 bytes"
    );
}

// ============================================================================
// §2 — Single-Codec Page Roundtrips
// ============================================================================

#[test]
fn test_single_page_uncompressed_roundtrip() {
    let raw = make_fp16_kv_data(DATA_SIZE, 42);
    let page = build_kv_page(0, CompressionCodec::None, &raw);

    let header = read_header(&page);
    assert_eq!(header.codec, CompressionCodec::None);
    assert_eq!(header.compressed_size, 0);

    let decompressed = decompress_page(&page);
    assert_eq!(decompressed, raw, "uncompressed page roundtrip mismatch");
}

#[test]
fn test_single_page_lz4_roundtrip() {
    let raw = make_fp16_kv_data(DATA_SIZE, 77);
    let page = build_kv_page(1, CompressionCodec::Lz4, &raw);

    let header = read_header(&page);
    assert_eq!(header.codec, CompressionCodec::Lz4);
    assert!(header.compressed_size > 0, "LZ4 compressed_size must be > 0");

    let decompressed = decompress_page(&page);
    assert_eq!(decompressed, raw, "LZ4 page roundtrip mismatch");
}

#[test]
fn test_single_page_bitpack_rle_roundtrip() {
    let raw = make_nibble_kv_data(DATA_SIZE);
    let page = build_kv_page(2, CompressionCodec::BitPackRle, &raw);

    let header = read_header(&page);
    assert_eq!(header.codec, CompressionCodec::BitPackRle);
    assert!(header.compressed_size > 0, "BitPackRle compressed_size must be > 0");

    let decompressed = decompress_page(&page);
    assert_eq!(decompressed, raw, "BitPackRle page roundtrip mismatch");
}

// ============================================================================
// §3 — Mixed Codec Batch (REQ-COMP-013 core scenario)
// ============================================================================

#[test]
fn test_mixed_codec_batch_all_roundtrip() {
    // Simulate a batch of 6 KV pages with different codecs:
    //   Page 0: uncompressed (hot, FP16)
    //   Page 1: LZ4 compressed (warm, FP16)
    //   Page 2: BitPackRle compressed (KIVI4 nibble)
    //   Page 3: uncompressed (hot, FP16)
    //   Page 4: LZ4 compressed (warm, FP16)
    //   Page 5: BitPackRle compressed (KIVI4 nibble)
    let codecs = [
        CompressionCodec::None,
        CompressionCodec::Lz4,
        CompressionCodec::BitPackRle,
        CompressionCodec::None,
        CompressionCodec::Lz4,
        CompressionCodec::BitPackRle,
    ];

    let originals: Vec<Vec<u8>> = codecs
        .iter()
        .enumerate()
        .map(|(i, codec)| {
            match codec {
                CompressionCodec::BitPackRle => make_nibble_kv_data(DATA_SIZE),
                _ => make_fp16_kv_data(DATA_SIZE, (i * 37 + 13) as u8),
            }
        })
        .collect();

    let pages: Vec<Vec<u8>> = codecs
        .iter()
        .enumerate()
        .map(|(i, codec)| build_kv_page(i as u32, *codec, &originals[i]))
        .collect();

    // Verify each page roundtrips independently
    for (i, (page, original)) in pages.iter().zip(originals.iter()).enumerate() {
        let header = read_header(page);
        assert_eq!(header.codec, codecs[i], "page {} codec mismatch", i);
        assert_eq!(header.page_id, i as u32, "page {} id mismatch", i);

        let decompressed = decompress_page(page);
        assert_eq!(
            &decompressed, original,
            "page {} ({:?}) roundtrip mismatch",
            i, codecs[i]
        );
    }
}

#[test]
fn test_mixed_codec_header_fields_independent() {
    // Verify that codec/compressed_size in one page doesn't leak into another.
    let fp16_data = make_fp16_kv_data(DATA_SIZE, 55);
    let nibble_data = make_nibble_kv_data(DATA_SIZE);

    let page_none = build_kv_page(10, CompressionCodec::None, &fp16_data);
    let page_lz4 = build_kv_page(11, CompressionCodec::Lz4, &fp16_data);
    let page_bpr = build_kv_page(12, CompressionCodec::BitPackRle, &nibble_data);

    let h_none = read_header(&page_none);
    let h_lz4 = read_header(&page_lz4);
    let h_bpr = read_header(&page_bpr);

    assert_eq!(h_none.codec, CompressionCodec::None);
    assert_eq!(h_none.compressed_size, 0);

    assert_eq!(h_lz4.codec, CompressionCodec::Lz4);
    assert!(h_lz4.compressed_size > 0);

    assert_eq!(h_bpr.codec, CompressionCodec::BitPackRle);
    assert!(h_bpr.compressed_size > 0);

    // LZ4 and BitPackRle compressed sizes should differ
    assert_ne!(h_lz4.compressed_size, h_bpr.compressed_size);
}

// ============================================================================
// §4 — Compression Ratio Verification in Page Context
// ============================================================================

#[test]
fn test_lz4_page_compressed_smaller_than_uncompressed() {
    // FP16-like data with patterns should compress
    let raw = make_fp16_kv_data(DATA_SIZE, 42);
    let page = build_kv_page(0, CompressionCodec::Lz4, &raw);
    let header = read_header(&page);

    assert!(
        (header.compressed_size as usize) < DATA_SIZE,
        "LZ4 compressed ({}) must be < DATA_SIZE ({})",
        header.compressed_size,
        DATA_SIZE
    );
}

#[test]
fn test_bitpack_rle_page_compressed_smaller_for_run_data() {
    // BitPackRle compresses runs well; use run-heavy data
    let mut raw = Vec::with_capacity(DATA_SIZE);
    let mut val: u8 = 0;
    let mut remaining = DATA_SIZE;
    while remaining > 0 {
        let chunk = 64.min(remaining);
        raw.extend(std::iter::repeat(val & 0x0F).take(chunk));
        val = val.wrapping_add(1);
        remaining -= chunk;
    }
    let page = build_kv_page(0, CompressionCodec::BitPackRle, &raw);
    let header = read_header(&page);

    assert!(
        (header.compressed_size as usize) < DATA_SIZE,
        "BitPackRle compressed ({}) must be < DATA_SIZE ({})",
        header.compressed_size,
        DATA_SIZE
    );
    let decompressed = decompress_page(&page);
    assert_eq!(decompressed, raw, "roundtrip mismatch");
}

#[test]
fn test_all_zeros_page_extreme_compression() {
    // All-zero page should compress extremely well with both codecs
    let zeros = vec![0u8; DATA_SIZE];

    let page_lz4 = build_kv_page(0, CompressionCodec::Lz4, &zeros);
    let h_lz4 = read_header(&page_lz4);
    let ratio_lz4 = h_lz4.compressed_size as f64 / DATA_SIZE as f64;
    assert!(ratio_lz4 < 0.1, "LZ4 all-zeros ratio {ratio_lz4:.3} too high");

    let page_bpr = build_kv_page(1, CompressionCodec::BitPackRle, &zeros);
    let h_bpr = read_header(&page_bpr);
    let ratio_bpr = h_bpr.compressed_size as f64 / DATA_SIZE as f64;
    assert!(ratio_bpr < 0.1, "BitPackRle all-zeros ratio {ratio_bpr:.3} too high");

    // Both roundtrip
    assert_eq!(decompress_page(&page_lz4), zeros);
    assert_eq!(decompress_page(&page_bpr), zeros);
}

// ============================================================================
// §5 — Simulated Mega-Kernel Attention Path (Branch-on-Codec Logic)
// ============================================================================

#[test]
fn test_simulated_attention_branch_on_codec() {
    // Simulates what the mega-kernel JIT code does:
    // For each page in the batch:
    //   1. Read codec from header byte at offset 0x28
    //   2. Read compressed_size from u32 at offset 0x2C
    //   3. Branch: codec=0 → skip, codec=1 → LZ4 decode, codec=2 → BitPackRle decode
    //   4. Feed decompressed data to attention computation
    let fp16_data = make_fp16_kv_data(DATA_SIZE, 88);
    let nibble_data = make_nibble_kv_data(DATA_SIZE);

    let pages = vec![
        build_kv_page(0, CompressionCodec::None, &fp16_data),
        build_kv_page(1, CompressionCodec::Lz4, &fp16_data),
        build_kv_page(2, CompressionCodec::BitPackRle, &nibble_data),
    ];
    let originals = vec![fp16_data.clone(), fp16_data.clone(), nibble_data.clone()];

    for (page, original) in pages.iter().zip(originals.iter()) {
        // Step 1: Read codec byte at offset 0x28
        let codec_byte = page[0x28];
        let codec = CompressionCodec::from_u8(codec_byte)
            .expect("codec byte must be valid");

        // Step 2: Read compressed_size u32 at offset 0x2C
        let csz = u32::from_le_bytes([
            page[0x2C],
            page[0x2D],
            page[0x2E],
            page[0x2F],
        ]);

        // Step 3: Branch on codec
        let decompressed = match codec {
            CompressionCodec::None => {
                assert_eq!(csz, 0, "uncompressed page must have compressed_size=0");
                // Skip past 56-byte header
                page[HEADER_SIZE..HEADER_SIZE + DATA_SIZE].to_vec()
            }
            CompressionCodec::Lz4 => {
                assert!(csz > 0, "LZ4 page must have compressed_size > 0");
                let payload = &page[HEADER_SIZE..HEADER_SIZE + csz as usize];
                lz4_decompress(payload, DATA_SIZE).expect("LZ4 decode must succeed")
            }
            CompressionCodec::BitPackRle => {
                assert!(csz > 0, "BitPackRle page must have compressed_size > 0");
                let payload = &page[HEADER_SIZE..HEADER_SIZE + csz as usize];
                decompress_bitpack_rle(payload, DATA_SIZE)
            }
            _ => panic!("unexpected codec: {:?}", codec),
        };

        // Step 4: Feed to "attention" — just verify data matches
        assert_eq!(
            decompressed, *original,
            "decompressed data must match original for codec {:?}",
            codec
        );
    }
}

#[test]
fn test_simulated_attention_mixed_batch_numerical_equivalence() {
    // Build a "batch" of 9 pages: 3 uncompressed, 3 LZ4, 3 BitPackRle
    // All using the same underlying FP16 data to verify numerical equivalence.
    let fp16_data = make_fp16_kv_data(DATA_SIZE, 42);
    let nibble_data = make_nibble_kv_data(DATA_SIZE);

    let mut pages = Vec::new();
    let mut expected = Vec::new();

    // 3 uncompressed FP16 pages
    for i in 0..3u32 {
        pages.push(build_kv_page(i, CompressionCodec::None, &fp16_data));
        expected.push(fp16_data.clone());
    }
    // 3 LZ4-compressed FP16 pages
    for i in 3..6u32 {
        pages.push(build_kv_page(i, CompressionCodec::Lz4, &fp16_data));
        expected.push(fp16_data.clone());
    }
    // 3 BitPackRle-compressed nibble pages
    for i in 6..9u32 {
        pages.push(build_kv_page(i, CompressionCodec::BitPackRle, &nibble_data));
        expected.push(nibble_data.clone());
    }

    // Decompress all pages and verify numerical equivalence
    for (i, (page, expected_data)) in pages.iter().zip(expected.iter()).enumerate() {
        let decompressed = decompress_page(page);
        assert_eq!(
            decompressed, *expected_data,
            "page {} numerical mismatch: compressed and decompressed data must be identical",
            i
        );
    }

    // Cross-check: all FP16 pages (compressed and uncompressed) produce identical data
    let fp16_results: Vec<Vec<u8>> = (0..6)
        .map(|i| decompress_page(&pages[i]))
        .collect();

    for i in 1..6 {
        assert_eq!(
            fp16_results[i], fp16_results[0],
            "FP16 page {} must be numerically identical to page 0 (uncompressed reference)",
            i
        );
    }

    // Free leaked boxes (avoid warning; data is 'static from Box::leak)
    drop(fp16_results);

    // Same for nibble pages
    for i in 7..9 {
        let d_i = decompress_page(&pages[i]);
        let d_6 = decompress_page(&pages[6]);
        assert_eq!(
            d_i, d_6,
            "nibble page {} must be numerically identical to page 6 (uncompressed nibble reference)",
            i
        );
    }
}

// ============================================================================
// §6 — Edge Cases
// ============================================================================

#[test]
fn test_page_with_all_same_byte_lz4() {
    let raw = vec![0xABu8; DATA_SIZE];
    let page = build_kv_page(0, CompressionCodec::Lz4, &raw);
    let decompressed = decompress_page(&page);
    assert_eq!(decompressed, raw);
}

#[test]
fn test_page_with_all_same_byte_bitpack_rle() {
    // BitPackRle only keeps low nibble, so 0x0F → output 0x0F
    let raw = vec![0x0Fu8; DATA_SIZE];
    let page = build_kv_page(0, CompressionCodec::BitPackRle, &raw);
    let decompressed = decompress_page(&page);
    assert_eq!(decompressed, raw);
}

#[test]
fn test_page_id_preserved_across_compression() {
    for (pid, codec) in [
        (0u32, CompressionCodec::None),
        (12345u32, CompressionCodec::Lz4),
        (u32::MAX, CompressionCodec::BitPackRle),
    ] {
        let raw = make_fp16_kv_data(DATA_SIZE, pid as u8);
        let page = build_kv_page(pid, codec, &raw);
        let header = read_header(&page);
        assert_eq!(header.page_id, pid, "page_id {} lost with codec {:?}", pid, codec);
    }
}

#[test]
fn test_header_read_write_consistency() {
    let raw = make_fp16_kv_data(DATA_SIZE, 99);
    let page = build_kv_page(42, CompressionCodec::Lz4, &raw);

    // Read header, modify, re-read
    let mut header = read_header(&page);
    assert_eq!(header.page_id, 42);
    assert_eq!(header.codec, CompressionCodec::Lz4);

    header.importance_score = 200;
    header.storage_tier = StorageTier::CpuDram;

    // Write back
    let mut page_mut = page.clone();
    unsafe {
        let ptr = page_mut.as_mut_ptr() as *mut KvPageHeader;
        std::ptr::write(ptr, header);
    }

    let header2 = read_header(&page_mut);
    assert_eq!(header2.importance_score, 200);
    assert_eq!(header2.storage_tier, StorageTier::CpuDram);
    assert_eq!(header2.page_id, 42);
    assert_eq!(header2.codec, CompressionCodec::Lz4);
}

#[test]
fn test_storage_tier_preserved() {
    for tier in [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme] {
        let mut header = KvPageHeader::new(1);
        header.storage_tier = tier;
        assert_eq!(header.storage_tier, tier);
        assert_eq!(StorageTier::from_u8(tier.as_u8()), Some(tier));
    }
}
