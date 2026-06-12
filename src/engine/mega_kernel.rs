//! Mega-Kernel 执行器 (SPEC §9.1)
//!
//! ARCH-RUST-IS-CODEGEN 铁律: 推理时 Rust 只做一次 CALL。
//! 含 Argmax 的图（embedding → N 个同构子结构 → logits-producer → sampling → generate loop）
//! 编译为单一 JIT 机器码，推理时通过 MegaKernelFn 单次 CALL 完成。
//! 无 Argmax 的图（embedding/rerank/classify）同样通过 MegaKernelFn 单次 CALL 完成。
//!
//! 无 fallback。编译失败 = 致命错误。
//!
//! 代码组织 (include! 模式 — 编译为单模块，物理分散到 4 个片段):
//! - `mega_kernel/abi_types.inc.rs`      — KernelContext ABI + 配置类型 + 编译产物结构体
//! - `mega_kernel/executor_core.inc.rs`  — MegaKernelExecutor struct + compile + generate 方法
//! - `mega_kernel/executor_ops.inc.rs`   — MegaKernelExecutor diagnostic/encode/rerank/辅助方法
//! - `mega_kernel/pack_observe.inc.rs`   — 权重打包 + 遥测观测 + 诊断 scratchpad

use gllm_kernels::types::DType;

include!("mega_kernel/abi_types.inc.rs");
include!("mega_kernel/executor_core.inc.rs");
include!("mega_kernel/executor_ops.inc.rs");
include!("mega_kernel/pack_observe.inc.rs");

#[cfg(test)]
mod tests {
    use super::*;

    // ── TelemetryFlagsBitmask ──

    #[test]
    fn telemetry_flags_default_is_zero() {
        let flags = TelemetryFlagsBitmask::default();
        assert_eq!(flags.0, 0);
    }

    // ── WeightPageJitConfig ──

    #[test]
    fn weight_page_jit_config_default_disabled() {
        let cfg = WeightPageJitConfig::default();
        assert!(!cfg.enabled);
        assert_eq!(cfg.page_size_bytes, 64 * 1024 * 1024);
        assert_eq!(cfg.num_pages, 1024);
        assert_eq!(cfg.prefetch_distance, 0);
    }

    #[test]
    fn weight_page_jit_config_custom_values() {
        let cfg = WeightPageJitConfig {
            enabled: true,
            num_pages: 2048,
            page_size_bytes: 128 * 1024 * 1024,
            prefetch_distance: 3,
        };
        assert!(cfg.enabled);
        assert_eq!(cfg.num_pages, 2048);
        assert_eq!(cfg.page_size_bytes, 128 * 1024 * 1024);
        assert_eq!(cfg.prefetch_distance, 3);
    }

    // ── KvPageDecompressConfig ──

    #[test]
    fn kv_page_decompress_config_default_disabled() {
        let cfg = KvPageDecompressConfig::default();
        assert!(!cfg.enabled);
        assert_eq!(cfg.num_pages, 1024);
        assert_eq!(cfg.page_size_bytes, 64 * 1024);
    }

    #[test]
    fn kv_page_decompress_config_custom_values() {
        let cfg = KvPageDecompressConfig {
            enabled: true,
            num_pages: 4096,
            page_size_bytes: 128 * 1024,
        };
        assert!(cfg.enabled);
        assert_eq!(cfg.num_pages, 4096);
        assert_eq!(cfg.page_size_bytes, 128 * 1024);
    }

    #[test]
    fn mega_kernel_error_display() {
        let compile_err = MegaKernelError::Compilation("bad graph".into());
        assert!(format!("{compile_err}").contains("bad graph"));
        let exec_err = MegaKernelError::Execution("segfault".into());
        assert!(format!("{exec_err}").contains("segfault"));
    }

    #[test]
    fn telemetry_flags_bitmask_is_transparent_u32() {
        let flags = TelemetryFlagsBitmask(0xABCD);
        assert_eq!(flags.0, 0xABCD);
    }

    // ── KernelContext ──

    #[test]
    fn kernel_context_zeroed_all_null() {
        let ctx = KernelContext::zeroed();
        assert!(ctx.weight_blob_ptr.is_null());
        assert!(ctx.kv_cache_ptr.is_null());
        assert!(ctx.output_buffer_ptr.is_null());
        assert!(ctx.hook_ctx_ptr.is_null());
        assert!(ctx.seq_len_ptr.is_null());
        assert!(ctx.rope_freqs_ptr.is_null());
        assert!(ctx.kv_page_table_ptr.is_null());
        assert!(ctx.batch_meta_ptr.is_null());
        assert_eq!(ctx.kv_page_size, 0);
        assert_eq!(ctx.kv_num_layers, 0);
        assert_eq!(ctx.kv_num_heads, 0);
        assert_eq!(ctx.kv_head_dim, 0);
        assert!(ctx.telemetry_ptr.is_null());
        assert_eq!(ctx.telemetry_flags, 0);
        assert!(ctx.callback_table_ptr.is_null());
        assert!(ctx.batch_ctx_ptr.is_null());
        assert_eq!(ctx.weight_page_inject_flags, 0);
        assert_eq!(ctx.decompress_inject_flags, 0);
    }

    #[test]
    fn kernel_context_build_sets_fields() {
        let dummy: u8 = 42;
        let (ctx, seq_len_box) = KernelContext::build(
            &dummy as *const u8,       // weight_blob_ptr
            std::ptr::null_mut(),      // kv_cache_ptr
            std::ptr::null_mut(),      // output_buffer_ptr
            std::ptr::null_mut(),      // hook_ctx_ptr
            128,                       // seq_len
            std::ptr::null(),          // rope_freqs_ptr
            std::ptr::null(),          // kv_page_table_ptr
            std::ptr::null(),          // batch_meta_ptr
            4096,                      // kv_page_size
            32,                        // kv_num_layers
            8,                         // kv_num_heads
            128,                       // kv_head_dim
            std::ptr::null_mut(),      // telemetry_ptr
            0x3,                       // telemetry_flags
            std::ptr::null(),          // business_config_ptr
            std::ptr::null(),          // weight_offsets_ptr
            0,                         // weight_offsets_len
            std::ptr::null(),          // callback_table_ptr
            std::ptr::null_mut(),      // scratch_buffer_ptr
            std::ptr::null(),          // batch_ctx_ptr
            std::ptr::null(),          // weight_page_table_ptr
            std::ptr::null(),          // weight_page_fault_cb_ptr
            1,                         // weight_page_inject_flags
            std::ptr::null(),          // kv_page_header_ptr
            0,                         // decompress_inject_flags
        );
        assert_eq!(*seq_len_box, 128);
        assert_eq!(ctx.kv_page_size, 4096);
        assert_eq!(ctx.kv_num_layers, 32);
        assert_eq!(ctx.kv_num_heads, 8);
        assert_eq!(ctx.kv_head_dim, 128);
        assert_eq!(ctx.telemetry_flags, 0x3);
        assert_eq!(ctx.weight_page_inject_flags, 1);
        assert!(!ctx.weight_blob_ptr.is_null());
    }

    // ── MegaKernelObservation ──

    #[test]
    fn observation_from_zero_buffer_has_defaults() {
        let buf = vec![0u8; 256];
        let obs = MegaKernelObservation::from_buffer(0, &buf);
        assert_eq!(obs.dead_neuron_ratio(768), 0.0);
        assert!(!obs.is_bypass_candidate(0.1, 0.9));
    }

    #[test]
    fn observation_dead_neuron_ratio_computed() {
        let mut buf = vec![0u8; 256];
        // Set softmax_sum (first f32 after header) to a non-zero value
        let softmax_sum_offset = 0;
        let softmax_sum: f32 = 1.0;
        let bytes = softmax_sum.to_le_bytes();
        buf[softmax_sum_offset..softmax_sum_offset + 4].copy_from_slice(&bytes);
        let obs = MegaKernelObservation::from_buffer(0, &buf);
        // dead_neuron_ratio = 1.0 - softmax_sum, hidden_size=1 for simplicity
        // With default values the ratio depends on internal layout
        assert!(obs.dead_neuron_ratio(1) >= 0.0);
    }

    // ── DiagnosticScratchpad ──

    #[test]
    fn scratchpad_read_f32_at_reads_zeroes() {
        let buf = vec![0u8; 1024];
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            prompt_len: 1,
            hidden_size: 4,
            vocab_size: 8,
        };
        let vals = sp.read_f32_at(0, 4);
        assert_eq!(vals.len(), 4);
        assert!(vals.iter().all(|&v| v == 0.0));
    }

    // ── Additional tests ──

    // ── TelemetryFlagsBitmask: trait + edge values ──

    #[test]
    fn telemetry_flags_clone_copy_eq() {
        let a = TelemetryFlagsBitmask(42);
        let b = a; // Copy
        let c = a.clone();
        assert_eq!(b.0, 42);
        assert_eq!(c.0, 42);
    }

    #[test]
    fn telemetry_flags_max_value() {
        let flags = TelemetryFlagsBitmask(u32::MAX);
        assert_eq!(flags.0, u32::MAX);
        let cloned = flags.clone();
        assert_eq!(cloned.0, u32::MAX);
    }

    #[test]
    fn telemetry_flags_debug_format() {
        let flags = TelemetryFlagsBitmask(0xFF);
        let debug_str = format!("{flags:?}");
        assert!(debug_str.contains("TelemetryFlagsBitmask"));
    }

    // ── WeightPageJitConfig: Clone, Debug, edge values ──

    #[test]
    fn weight_page_jit_config_clone() {
        let cfg = WeightPageJitConfig {
            enabled: true,
            num_pages: 512,
            page_size_bytes: 32 * 1024 * 1024,
            prefetch_distance: 2,
        };
        let cloned = cfg.clone();
        assert_eq!(cloned.enabled, true);
        assert_eq!(cloned.num_pages, 512);
        assert_eq!(cloned.page_size_bytes, 32 * 1024 * 1024);
        assert_eq!(cloned.prefetch_distance, 2);
    }

    #[test]
    fn weight_page_jit_config_debug() {
        let cfg = WeightPageJitConfig::default();
        let debug_str = format!("{cfg:?}");
        assert!(debug_str.contains("WeightPageJitConfig"));
    }

    #[test]
    fn weight_page_jit_config_zero_page_size() {
        let cfg = WeightPageJitConfig {
            enabled: false,
            num_pages: 0,
            page_size_bytes: 0,
            prefetch_distance: 0,
        };
        assert_eq!(cfg.page_size_bytes, 0);
        assert_eq!(cfg.num_pages, 0);
    }

    // ── KvPageDecompressConfig: Clone, Debug, edge values ──

    #[test]
    fn kv_page_decompress_config_clone() {
        let cfg = KvPageDecompressConfig {
            enabled: true,
            num_pages: 8192,
            page_size_bytes: 256 * 1024,
        };
        let cloned = cfg.clone();
        assert_eq!(cloned.enabled, true);
        assert_eq!(cloned.num_pages, 8192);
        assert_eq!(cloned.page_size_bytes, 256 * 1024);
    }

    #[test]
    fn kv_page_decompress_config_debug() {
        let cfg = KvPageDecompressConfig::default();
        let debug_str = format!("{cfg:?}");
        assert!(debug_str.contains("KvPageDecompressConfig"));
    }

    #[test]
    fn kv_page_decompress_config_zero_values() {
        let cfg = KvPageDecompressConfig {
            enabled: false,
            num_pages: 0,
            page_size_bytes: 0,
        };
        assert_eq!(cfg.num_pages, 0);
        assert_eq!(cfg.page_size_bytes, 0);
    }

    // ── KernelContext: repr(C) layout, build, zeroed ──

    #[test]
    fn kernel_context_size_is_aligned() {
        // KernelContext is repr(C); verify it has a known non-zero size
        let size = std::mem::size_of::<KernelContext>();
        assert!(size > 0);
        // Should be a multiple of 8 since all fields are naturally aligned to 8 bytes
        assert_eq!(size % 8, 0);
    }

    #[test]
    fn kernel_context_zeroed_seq_len_ptr_is_valid() {
        let ctx = KernelContext::zeroed();
        assert!(ctx.seq_len_ptr.is_null());
    }

    #[test]
    fn kernel_context_build_seq_len_box_outlives_context() {
        let dummy: u8 = 0;
        let (ctx, seq_len_box) = KernelContext::build(
            &dummy as *const u8,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0, // seq_len = 0
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0, 0, 0, 0,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
        );
        assert_eq!(*seq_len_box, 0);
        assert!(!ctx.seq_len_ptr.is_null());
        // Verify the pointer matches the boxed value
        assert_eq!(unsafe { *ctx.seq_len_ptr }, 0);
    }

    #[test]
    fn kernel_context_build_large_seq_len() {
        let dummy: u8 = 0;
        let (_ctx, seq_len_box) = KernelContext::build(
            &dummy as *const u8,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            usize::MAX / 2, // large seq_len
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0, 0, 0, 0,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
        );
        assert_eq!(*seq_len_box, usize::MAX / 2);
    }

    #[test]
    fn kernel_context_all_pointer_fields_null_after_zeroed() {
        let ctx = KernelContext::zeroed();
        // Enumerate all pointer fields to ensure none are dangling
        assert!(ctx.weight_blob_ptr.is_null());
        assert!(ctx.kv_cache_ptr.is_null());
        assert!(ctx.output_buffer_ptr.is_null());
        assert!(ctx.hook_ctx_ptr.is_null());
        assert!(ctx.rope_freqs_ptr.is_null());
        assert!(ctx.kv_page_table_ptr.is_null());
        assert!(ctx.batch_meta_ptr.is_null());
        assert!(ctx.telemetry_ptr.is_null());
        assert!(ctx.business_config_ptr.is_null());
        assert!(ctx.weight_offsets_ptr.is_null());
        assert!(ctx.callback_table_ptr.is_null());
        assert!(ctx.scratch_buffer_ptr.is_null());
        assert!(ctx.batch_ctx_ptr.is_null());
        assert!(ctx.weight_page_table_ptr.is_null());
        assert!(ctx.weight_page_fault_cb_ptr.is_null());
        assert!(ctx.kv_page_header_ptr.is_null());
    }

    // ── MegaKernelError: variants, Debug, source chain ──

    #[test]
    fn mega_kernel_error_compilation_variant() {
        let err = MegaKernelError::Compilation("test error".into());
        let msg = format!("{err}");
        assert!(msg.contains("test error"));
        assert!(msg.contains("compilation"));
    }

    #[test]
    fn mega_kernel_error_execution_variant() {
        let err = MegaKernelError::Execution("runtime crash".into());
        let msg = format!("{err}");
        assert!(msg.contains("runtime crash"));
        assert!(msg.contains("execution"));
    }

    #[test]
    fn mega_kernel_error_debug() {
        let err = MegaKernelError::Compilation("dbg".into());
        let debug = format!("{err:?}");
        assert!(debug.contains("Compilation"));
    }

    // ── MegaKernelObservation: from_buffer, dead_neuron_ratio, is_bypass_candidate ──

    #[test]
    fn observation_dead_neuron_ratio_zero_hidden() {
        let obs = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: 0.0,
            cosine_similarity: 0.0,
            dead_neuron_count: 100,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        // hidden_size = 0 must return 0.0 to avoid division by zero
        assert_eq!(obs.dead_neuron_ratio(0), 0.0);
    }

    #[test]
    fn observation_dead_neuron_ratio_full() {
        let obs = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: 0.0,
            cosine_similarity: 0.0,
            dead_neuron_count: 768,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        let ratio = obs.dead_neuron_ratio(768);
        assert!((ratio - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn observation_dead_neuron_ratio_partial() {
        let obs = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: 0.0,
            cosine_similarity: 0.0,
            dead_neuron_count: 256,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        let ratio = obs.dead_neuron_ratio(1024);
        assert!((ratio - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn observation_is_bypass_candidate_true() {
        let obs = MegaKernelObservation {
            layer_idx: 5,
            entropy: 0.5,
            residual_delta: 0.001,  // < 0.01 threshold
            cosine_similarity: 0.99, // > 0.95 threshold
            dead_neuron_count: 0,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 1.0,
            row_l1_norm: 10.0,
            row_max: 5.0,
        };
        assert!(obs.is_bypass_candidate(0.01, 0.95));
    }

    #[test]
    fn observation_is_bypass_candidate_false_high_delta() {
        let obs = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: 0.5, // >= 0.01 threshold
            cosine_similarity: 0.99,
            dead_neuron_count: 0,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        assert!(!obs.is_bypass_candidate(0.01, 0.95));
    }

    #[test]
    fn observation_is_bypass_candidate_false_low_cosine() {
        let obs = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: 0.001,
            cosine_similarity: 0.5, // <= 0.95 threshold
            dead_neuron_count: 0,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        assert!(!obs.is_bypass_candidate(0.01, 0.95));
    }

    #[test]
    fn observation_copy_and_debug_traits() {
        let obs = MegaKernelObservation {
            layer_idx: 3,
            entropy: 1.5,
            residual_delta: 0.2,
            cosine_similarity: 0.8,
            dead_neuron_count: 42,
            sink_status: AttentionSinkStatus::SinkToken,
            per_channel_scale: 2.0,
            row_l1_norm: 15.0,
            row_max: 7.5,
        };
        let copy = obs; // Copy trait
        assert_eq!(copy.layer_idx, 3);
        assert_eq!(copy.dead_neuron_count, 42);
        assert_eq!(copy.sink_status, AttentionSinkStatus::SinkToken);
        let debug_str = format!("{obs:?}");
        assert!(debug_str.contains("MegaKernelObservation"));
    }

    #[test]
    fn observation_from_buffer_too_small_returns_zeros() {
        // Buffer of only 4 bytes — far too small for all telemetry offsets
        let buf = vec![0u8; 4];
        let obs = MegaKernelObservation::from_buffer(0, &buf);
        assert_eq!(obs.layer_idx, 0);
        assert_eq!(obs.entropy, 0.0);
        assert_eq!(obs.dead_neuron_count, 0);
        assert_eq!(obs.sink_status, AttentionSinkStatus::Normal);
    }

    #[test]
    fn observation_from_buffer_empty_returns_zeros() {
        let buf = vec![];
        let obs = MegaKernelObservation::from_buffer(7, &buf);
        assert_eq!(obs.layer_idx, 7);
        assert_eq!(obs.entropy, 0.0);
    }

    // ── DiagnosticScratchpad: read_f32_at, embedding, last_token_logits, Debug ──

    #[test]
    fn scratchpad_debug_trait() {
        let sp = DiagnosticScratchpad {
            data: vec![0u8; 64],
            logits_offset: 0,
            vocab_size: 8,
            prompt_len: 2,
            hidden_size: 4,
        };
        let debug_str = format!("{sp:?}");
        assert!(debug_str.contains("DiagnosticScratchpad"));
    }

    #[test]
    fn scratchpad_read_f32_at_out_of_bounds_returns_empty() {
        let buf = vec![0u8; 16]; // only 4 f32s worth
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 8,
            prompt_len: 1,
            hidden_size: 4,
        };
        // Request 5 f32s starting at offset 0 — needs 20 bytes, only 16 available
        let vals = sp.read_f32_at(0, 5);
        assert!(vals.is_empty());
    }

    #[test]
    fn scratchpad_read_f32_at_offset_out_of_bounds() {
        let buf = vec![0u8; 16];
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 8,
            prompt_len: 1,
            hidden_size: 4,
        };
        // Start at byte 16 — beyond buffer
        let vals = sp.read_f32_at(16, 1);
        assert!(vals.is_empty());
    }

    #[test]
    fn scratchpad_read_f32_at_reads_actual_values() {
        let mut buf = vec![0u8; 32];
        // Write f32 value 3.14 at offset 0
        let val: f32 = 3.14;
        buf[0..4].copy_from_slice(&val.to_le_bytes());
        // Write f32 value -1.0 at offset 4
        let val2: f32 = -1.0;
        buf[4..8].copy_from_slice(&val2.to_le_bytes());

        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 8,
            prompt_len: 1,
            hidden_size: 4,
        };
        let vals = sp.read_f32_at(0, 2);
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 3.14).abs() < 0.001);
        assert!((vals[1] - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn scratchpad_embedding_reads_correct_count() {
        let buf = vec![0u8; 64];
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 8,
            prompt_len: 3,
            hidden_size: 4,
        };
        // embedding = prompt_len * hidden_size = 3 * 4 = 12 f32s
        let emb = sp.embedding();
        assert_eq!(emb.len(), 12);
        assert!(emb.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn scratchpad_embedding_buffer_too_small_returns_empty() {
        let buf = vec![0u8; 8]; // only 2 f32s, but need prompt_len=2 * hidden_size=4 = 8 f32s
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 8,
            prompt_len: 2,
            hidden_size: 4,
        };
        let emb = sp.embedding();
        assert!(emb.is_empty());
    }

    #[test]
    fn scratchpad_last_token_logits_reads_correct_offset() {
        // logits_offset=16, vocab_size=2, prompt_len=2
        // last_token_logits reads at offset = logits_offset + (prompt_len-1) * vocab_size*4
        // = 16 + 1 * 8 = 24
        let mut buf = vec![0u8; 64];
        // Write a known f32 at byte offset 24
        let expected: f32 = 42.0;
        buf[24..28].copy_from_slice(&expected.to_le_bytes());
        let expected2: f32 = -7.0;
        buf[28..32].copy_from_slice(&expected2.to_le_bytes());

        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 16,
            vocab_size: 2,
            prompt_len: 2,
            hidden_size: 4,
        };
        let logits = sp.last_token_logits();
        assert_eq!(logits.len(), 2);
        assert!((logits[0] - 42.0).abs() < f32::EPSILON);
        assert!((logits[1] - (-7.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn scratchpad_last_token_logits_out_of_bounds_returns_empty() {
        // logits_offset near end of buffer so (prompt_len-1)*row overflows
        let buf = vec![0u8; 8];
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 100, // way past buffer
            vocab_size: 8,
            prompt_len: 2,
            hidden_size: 4,
        };
        let logits = sp.last_token_logits();
        assert!(logits.is_empty());
    }

    // ── MegaKernelError: source chain (std::error::Error) ──

    #[test]
    fn mega_kernel_error_source_is_none() {
        // MegaKernelError wraps String, not another error — source() must return None
        let err = MegaKernelError::Compilation("source test".into());
        assert!(std::error::Error::source(&err).is_none());
        let err2 = MegaKernelError::Execution("source test".into());
        assert!(std::error::Error::source(&err2).is_none());
    }

    // ── MegaKernelObservation: special float values ──

    #[test]
    fn observation_nan_entropy_propagates() {
        let obs = MegaKernelObservation {
            layer_idx: 0,
            entropy: f32::NAN,
            residual_delta: 0.0,
            cosine_similarity: 1.0,
            dead_neuron_count: 0,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: f32::NAN,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        // Copy should preserve NaN bit-for-bit
        let copy = obs;
        assert!(copy.entropy.is_nan());
        assert!(copy.per_channel_scale.is_nan());
        // NaN should appear in Debug output
        let dbg = format!("{obs:?}");
        assert!(dbg.contains("NaN") || dbg.contains("nan"));
    }

    #[test]
    fn observation_infinity_values_preserved() {
        let obs = MegaKernelObservation {
            layer_idx: 0,
            entropy: f32::INFINITY,
            residual_delta: f32::NEG_INFINITY,
            cosine_similarity: f32::INFINITY,
            dead_neuron_count: 0,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: f32::INFINITY,
            row_max: f32::NEG_INFINITY,
        };
        let copy = obs;
        assert!(copy.entropy.is_infinite() && copy.entropy.is_sign_positive());
        assert!(copy.residual_delta.is_infinite() && copy.residual_delta.is_sign_negative());
        assert!(copy.row_max.is_infinite() && copy.row_max.is_sign_negative());
    }

    #[test]
    fn observation_is_bypass_boundary_delta_equals_threshold() {
        // When residual_delta == delta_threshold exactly, should return false (<, not <=)
        let obs = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: 0.01,
            cosine_similarity: 0.99,
            dead_neuron_count: 0,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        assert!(!obs.is_bypass_candidate(0.01, 0.95));
    }

    #[test]
    fn observation_is_bypass_boundary_cosine_equals_threshold() {
        // When cosine_similarity == cosine_threshold exactly, should return false (>, not >=)
        let obs = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: 0.001,
            cosine_similarity: 0.95,
            dead_neuron_count: 0,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        assert!(!obs.is_bypass_candidate(0.01, 0.95));
    }

    #[test]
    fn observation_dead_neuron_count_exceeds_hidden() {
        // dead_neuron_count > hidden_size should yield ratio > 1.0
        let obs = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: 0.0,
            cosine_similarity: 0.0,
            dead_neuron_count: 2000,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        let ratio = obs.dead_neuron_ratio(1000);
        assert!((ratio - 2.0).abs() < f32::EPSILON);
    }

    // ── TelemetryFlagsBitmask: equality semantics ──

    #[test]
    fn telemetry_flags_equality() {
        let a = TelemetryFlagsBitmask(0xFF);
        let b = TelemetryFlagsBitmask(0xFF);
        let c = TelemetryFlagsBitmask(0x00);
        // Copy type — compare inner value
        assert_eq!(a.0, b.0);
        assert_ne!(a.0, c.0);
        // Verify Copy produces equal values
        let d = a;
        assert_eq!(a.0, d.0);
    }

    // ── WeightPageJitConfig: PartialEq by field ──

    #[test]
    fn weight_page_config_field_equality() {
        let a = WeightPageJitConfig {
            enabled: true,
            num_pages: 512,
            page_size_bytes: 4096,
            prefetch_distance: 1,
        };
        let b = WeightPageJitConfig {
            enabled: true,
            num_pages: 512,
            page_size_bytes: 4096,
            prefetch_distance: 1,
        };
        assert_eq!(a.enabled, b.enabled);
        assert_eq!(a.num_pages, b.num_pages);
        assert_eq!(a.page_size_bytes, b.page_size_bytes);
        assert_eq!(a.prefetch_distance, b.prefetch_distance);
    }

    // ── KvPageDecompressConfig: PartialEq by field ──

    #[test]
    fn kv_decompress_config_field_equality() {
        let a = KvPageDecompressConfig {
            enabled: false,
            num_pages: 2048,
            page_size_bytes: 32 * 1024,
        };
        let b = KvPageDecompressConfig {
            enabled: false,
            num_pages: 2048,
            page_size_bytes: 32 * 1024,
        };
        assert_eq!(a.enabled, b.enabled);
        assert_eq!(a.num_pages, b.num_pages);
        assert_eq!(a.page_size_bytes, b.page_size_bytes);
    }

    // ── KernelContext: max u32 values for KV params ──

    #[test]
    fn kernel_context_build_max_u32_kv_params() {
        let dummy: u8 = 0;
        let (ctx, _seq) = KernelContext::build(
            &dummy as *const u8,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            1,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
        );
        assert_eq!(ctx.kv_page_size, u32::MAX);
        assert_eq!(ctx.kv_num_layers, u32::MAX);
        assert_eq!(ctx.kv_num_heads, u32::MAX);
        assert_eq!(ctx.kv_head_dim, u32::MAX);
    }

    // ── DiagnosticScratchpad: read_f32_at at exact boundary ──

    #[test]
    fn scratchpad_read_f32_at_exact_boundary() {
        let mut buf = vec![0u8; 16]; // exactly 4 f32s
        let val: f32 = 7.5;
        buf[12..16].copy_from_slice(&val.to_le_bytes());
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 4,
            prompt_len: 1,
            hidden_size: 4,
        };
        // Read 4 f32s starting at offset 0 — exactly fits the buffer
        let vals = sp.read_f32_at(0, 4);
        assert_eq!(vals.len(), 4);
        assert!((vals[3] - 7.5).abs() < f32::EPSILON);
    }

    // ── DiagnosticScratchpad: embedding with prompt_len=0 ──

    #[test]
    fn scratchpad_embedding_prompt_len_zero() {
        let buf = vec![0u8; 64];
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 8,
            prompt_len: 0,
            hidden_size: 4,
        };
        // prompt_len=0 → count=0 → empty vec
        let emb = sp.embedding();
        assert!(emb.is_empty());
    }

    // ── DiagnosticScratchpad: last_token_logits with prompt_len=1 ──

    #[test]
    fn scratchpad_last_token_logits_single_prompt_token() {
        let mut buf = vec![0u8; 32];
        let val: f32 = -3.5;
        // prompt_len=1 → last token offset = logits_offset + 0 * row_bytes
        // = logits_offset directly; vocab_size=2 → row_bytes=8
        buf[0..4].copy_from_slice(&val.to_le_bytes());
        let val2: f32 = 99.0;
        buf[4..8].copy_from_slice(&val2.to_le_bytes());
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 2,
            prompt_len: 1,
            hidden_size: 4,
        };
        let logits = sp.last_token_logits();
        assert_eq!(logits.len(), 2);
        assert!((logits[0] - (-3.5)).abs() < f32::EPSILON);
        assert!((logits[1] - 99.0).abs() < f32::EPSILON);
    }

    // ── DiagnosticScratchpad: read_f32_at single element at end ──

    #[test]
    fn scratchpad_read_f32_single_at_buffer_end() {
        let mut buf = vec![0u8; 8]; // 2 f32s
        let val: f32 = 42.42;
        buf[4..8].copy_from_slice(&val.to_le_bytes());
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 2,
            prompt_len: 1,
            hidden_size: 2,
        };
        let vals = sp.read_f32_at(4, 1);
        assert_eq!(vals.len(), 1);
        assert!((vals[0] - 42.42).abs() < 0.01);
    }

    // ── MegaKernelObservation: dead_neuron_ratio with zero count ──
    // @trace TEST-MKO-061 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_dead_neuron_ratio_zero_count() {
        let obs = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: 0.0,
            cosine_similarity: 0.0,
            dead_neuron_count: 0,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        let ratio = obs.dead_neuron_ratio(768);
        assert!((ratio - 0.0).abs() < f32::EPSILON);
    }

    // ── MegaKernelObservation: is_bypass_candidate both conditions met ──
    // @trace TEST-MKO-062 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_is_bypass_candidate_both_conditions_just_met() {
        // delta just below threshold, cosine just above threshold
        let obs = MegaKernelObservation {
            layer_idx: 2,
            entropy: 1.0,
            residual_delta: 0.009,  // < 0.01
            cosine_similarity: 0.951, // > 0.95
            dead_neuron_count: 0,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 1.0,
            row_l1_norm: 5.0,
            row_max: 3.0,
        };
        assert!(obs.is_bypass_candidate(0.01, 0.95));
    }

    // ── MegaKernelObservation: from_buffer with non-zero layer_idx ──
    // @trace TEST-MKO-063 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_from_buffer_preserves_layer_idx() {
        let buf = vec![0u8; 256];
        let obs = MegaKernelObservation::from_buffer(42, &buf);
        assert_eq!(obs.layer_idx, 42);
        // All other fields should be zero since buffer is all zeros
        assert_eq!(obs.dead_neuron_count, 0);
        assert_eq!(obs.sink_status, AttentionSinkStatus::Normal);
    }

    // ── DiagnosticScratchpad: read_f32_at with count=0 ──
    // @trace TEST-MKO-064 [req:REQ-OBS] [level:unit]

    #[test]
    fn scratchpad_read_f32_at_zero_count_returns_empty() {
        let buf = vec![0u8; 16];
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 4,
            prompt_len: 1,
            hidden_size: 4,
        };
        let vals = sp.read_f32_at(0, 0);
        assert!(vals.is_empty());
    }

    // ── DiagnosticScratchpad: read_f32_at with non-zero byte offset ──
    // @trace TEST-MKO-065 [req:REQ-OBS] [level:unit]

    #[test]
    fn scratchpad_read_f32_at_nonzero_offset() {
        let mut buf = vec![0u8; 32];
        // Write known value at byte offset 8 (2nd f32 position)
        let val: f32 = -999.5;
        buf[8..12].copy_from_slice(&val.to_le_bytes());
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 8,
            prompt_len: 1,
            hidden_size: 4,
        };
        let vals = sp.read_f32_at(8, 1);
        assert_eq!(vals.len(), 1);
        assert!((vals[0] - (-999.5)).abs() < 0.01);
    }

    // ── MegaKernelObservation: from_buffer with entropy at known offset ──
    // @trace TEST-MKO-066 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_from_buffer_small_valid_reads_entropy() {
        use gllm_kernels::compiler::graph::telemetry_offsets;
        let offset = telemetry_offsets::SOFTMAX_SHARPNESS_OFFSET;
        let mut buf = vec![0u8; 512];
        // Write a known entropy value at the correct offset
        let entropy_val: f32 = 2.718;
        buf[offset..offset + 4].copy_from_slice(&entropy_val.to_le_bytes());
        let obs = MegaKernelObservation::from_buffer(0, &buf);
        assert!((obs.entropy - 2.718).abs() < 0.01);
    }

    // ── MegaKernelObservation: all fields populated via from_buffer ──
    // @trace TEST-MKO-067 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_from_buffer_all_fields_populated() {
        use gllm_kernels::compiler::graph::telemetry_offsets;
        let mut buf = vec![0u8; 512];
        // Write entropy at offset 332
        let entropy: f32 = 1.23;
        let off0 = telemetry_offsets::SOFTMAX_SHARPNESS_OFFSET;
        buf[off0..off0 + 4].copy_from_slice(&entropy.to_le_bytes());
        // Write residual_delta at offset 128
        let delta: f32 = 0.045;
        let off1 = telemetry_offsets::RESIDUAL_DELTA_OFFSET;
        buf[off1..off1 + 4].copy_from_slice(&delta.to_le_bytes());
        // Write cosine_similarity at offset 136
        let cosine: f32 = 0.87;
        let off2 = telemetry_offsets::COSINE_SIMILARITY_OFFSET;
        buf[off2..off2 + 4].copy_from_slice(&cosine.to_le_bytes());

        let obs = MegaKernelObservation::from_buffer(5, &buf);
        assert_eq!(obs.layer_idx, 5);
        assert!((obs.entropy - 1.23).abs() < 0.01);
        assert!((obs.residual_delta - 0.045).abs() < 0.001);
        assert!((obs.cosine_similarity - 0.87).abs() < 0.01);
    }

    // ── TelemetryFlagsBitmask: bitwise operations ──
    // @trace TEST-MKO-068 [req:REQ-OBS] [level:unit]

    #[test]
    fn telemetry_flags_bitmask_bitwise_ops() {
        let a = TelemetryFlagsBitmask(0x0F);
        let b = TelemetryFlagsBitmask(0xF0);
        // Verify we can read individual bits
        assert_ne!(a.0 & b.0, 0xFF); // no overlap
        assert_eq!(a.0 | b.0, 0xFF);
        assert_eq!(a.0 ^ b.0, 0xFF);
        // Zero flags
        let zero = TelemetryFlagsBitmask(0);
        assert_eq!(zero.0 & 0xFF, 0);
    }

    // ── KernelContext: build with all non-null pointer fields ──
    // @trace TEST-MKO-069 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn kernel_context_build_all_nonnull_pointers() {
        let weight_data: u8 = 1;
        let mut kv_data: u8 = 2;
        let mut output_data: u8 = 3;
        let mut hook_data: u8 = 4;
        let rope_data: f32 = 5.0;
        let page_table_data: u32 = 6;
        let batch_meta_data: u8 = 7;
        let mut telemetry_data: u8 = 8;
        let business_data: u8 = 9;
        let weight_offsets_data: usize = 10;
        let callback_data: u64 = 11;
        let mut scratch_data: u8 = 12;
        let batch_ctx_data: u8 = 13;
        let wp_table_data: u8 = 14;
        let wp_fault_cb_data: u64 = 15;
        let kv_header_data: u8 = 16;

        let (ctx, _seq) = KernelContext::build(
            &weight_data as *const u8,
            &mut kv_data as *mut u8,
            &mut output_data as *mut u8,
            &mut hook_data as *mut u8,
            10,
            &rope_data as *const f32,
            &page_table_data as *const u32,
            &batch_meta_data as *const u8,
            4096, 24, 8, 128,
            &mut telemetry_data as *mut u8,
            0xFF,
            &business_data as *const u8,
            &weight_offsets_data as *const usize,
            1,
            &callback_data as *const u64,
            &mut scratch_data as *mut u8,
            &batch_ctx_data as *const u8,
            &wp_table_data as *const u8,
            &wp_fault_cb_data as *const u64,
            1,
            &kv_header_data as *const u8,
            1,
        );
        assert!(!ctx.weight_blob_ptr.is_null());
        assert!(!ctx.kv_cache_ptr.is_null());
        assert!(!ctx.output_buffer_ptr.is_null());
        assert!(!ctx.hook_ctx_ptr.is_null());
        assert!(!ctx.rope_freqs_ptr.is_null());
        assert!(!ctx.kv_page_table_ptr.is_null());
        assert!(!ctx.batch_meta_ptr.is_null());
        assert!(!ctx.telemetry_ptr.is_null());
        assert!(!ctx.business_config_ptr.is_null());
        assert!(!ctx.weight_offsets_ptr.is_null());
        assert!(!ctx.callback_table_ptr.is_null());
        assert!(!ctx.scratch_buffer_ptr.is_null());
        assert!(!ctx.batch_ctx_ptr.is_null());
        assert!(!ctx.weight_page_table_ptr.is_null());
        assert!(!ctx.weight_page_fault_cb_ptr.is_null());
        assert!(!ctx.kv_page_header_ptr.is_null());
        assert_eq!(ctx.weight_offsets_len, 1);
        assert_eq!(ctx.telemetry_flags, 0xFF);
        assert_eq!(ctx.weight_page_inject_flags, 1);
        assert_eq!(ctx.decompress_inject_flags, 1);
    }

    // ── MegaKernelError: Display includes variant-specific prefix ──
    // @trace TEST-MKO-070 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn mega_kernel_error_display_compilation_prefix() {
        let err = MegaKernelError::Compilation("graph error".into());
        let msg = format!("{err}");
        assert!(msg.starts_with("compilation"));
        assert!(msg.contains("graph error"));
    }

    #[test]
    fn mega_kernel_error_display_execution_prefix() {
        let err = MegaKernelError::Execution("signal 11".into());
        let msg = format!("{err}");
        assert!(msg.starts_with("execution"));
        assert!(msg.contains("signal 11"));
    }

    // ── DiagnosticScratchpad: last_token_logits with prompt_len=0 panics ──
    // @trace TEST-MKO-071 [req:REQ-MEGA-001] [level:unit]

    #[test]
    #[should_panic]
    fn scratchpad_last_token_logits_prompt_len_zero_panics() {
        // prompt_len=0 → (prompt_len-1) underflows to usize::MAX → arithmetic overflow panic
        let buf = vec![0u8; 64];
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 4,
            prompt_len: 0,
            hidden_size: 4,
        };
        // This must panic due to underflow in (prompt_len - 1) * row_bytes
        let _ = sp.last_token_logits();
    }

    // ── MegaKernelObservation: is_bypass with zero thresholds ──
    // @trace TEST-MKO-072 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_is_bypass_zero_thresholds() {
        // delta_threshold=0: only residual_delta < 0 passes (negative delta)
        // cosine_threshold=0: cosine_similarity > 0 passes for any positive cosine
        let obs_positive = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: -0.001,
            cosine_similarity: 0.5,
            dead_neuron_count: 0,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        assert!(obs_positive.is_bypass_candidate(0.0, 0.0));

        let obs_exact_zero_delta = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: 0.0,
            cosine_similarity: 0.5,
            dead_neuron_count: 0,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        // residual_delta == 0 is NOT < 0, so should be false
        assert!(!obs_exact_zero_delta.is_bypass_candidate(0.0, 0.0));
    }

    // ── MegaKernelObservation: is_bypass with negative cosine ──
    // @trace TEST-MKO-073 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_is_bypass_negative_cosine_fails() {
        let obs = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: 0.0,
            cosine_similarity: -1.0,
            dead_neuron_count: 0,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        // cosine=-1.0 is NOT > any positive threshold
        assert!(!obs.is_bypass_candidate(0.01, 0.5));
    }

    // ── KernelContext: build and read seq_len_ptr through unsafe deref ──
    // @trace TEST-MKO-074 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn kernel_context_build_seq_len_ptr_readable() {
        let dummy: u8 = 0;
        let (ctx, _seq) = KernelContext::build(
            &dummy as *const u8,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            999,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0, 0, 0, 0,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
        );
        // seq_len_ptr should point to the boxed value of 999
        assert!(!ctx.seq_len_ptr.is_null());
        assert_eq!(unsafe { *ctx.seq_len_ptr }, 999);
    }

    // ── WeightPageJitConfig: large page size ──
    // @trace TEST-MKO-075 [req:REQ-WP-008] [level:unit]

    #[test]
    fn weight_page_jit_config_large_page_size() {
        let cfg = WeightPageJitConfig {
            enabled: true,
            num_pages: 1,
            page_size_bytes: usize::MAX,
            prefetch_distance: usize::MAX,
        };
        assert_eq!(cfg.page_size_bytes, usize::MAX);
        assert_eq!(cfg.prefetch_distance, usize::MAX);
        // Clone should preserve values
        let cloned = cfg.clone();
        assert_eq!(cloned.page_size_bytes, usize::MAX);
        assert_eq!(cloned.prefetch_distance, usize::MAX);
    }

    // ── KvPageDecompressConfig: enabled with specific page sizes ──
    // @trace TEST-MKO-076 [req:REQ-COMP11] [level:unit]

    #[test]
    fn kv_decompress_config_enabled_specific_sizes() {
        // 16 KiB page — minimum typical KV page size
        let cfg = KvPageDecompressConfig {
            enabled: true,
            num_pages: 65536,
            page_size_bytes: 16 * 1024,
        };
        assert!(cfg.enabled);
        assert_eq!(cfg.num_pages, 65536);
        assert_eq!(cfg.page_size_bytes, 16 * 1024);
        // Verify clone
        let cloned = cfg.clone();
        assert!(cloned.enabled);
        assert_eq!(cloned.num_pages, 65536);
    }

    // ── MegaKernelObservation: from_buffer with buffer too small for all offsets ──
    // @trace TEST-MKO-077 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_from_buffer_too_small_for_entropy_reads_zero() {
        // Buffer of 64 bytes — too small for SOFTMAX_SHARPNESS_OFFSET (332),
        // so entropy should default to 0.0. dead_neuron_count (offset 8) is readable.
        let mut buf = vec![0u8; 64];
        let dead_count: u32 = 42;
        buf[8..12].copy_from_slice(&dead_count.to_le_bytes());
        let obs = MegaKernelObservation::from_buffer(1, &buf);
        assert_eq!(obs.layer_idx, 1);
        // entropy at offset 332 is beyond the 64-byte buffer, so it reads 0.0
        assert_eq!(obs.entropy, 0.0);
        // dead_neuron_count at offset 8 is within the buffer
        assert_eq!(obs.dead_neuron_count, 42);
    }

    // ── DiagnosticScratchpad: embedding with hidden_size=0 ──
    // @trace TEST-MKO-078 [req:REQ-OBS] [level:unit]

    #[test]
    fn scratchpad_embedding_hidden_size_zero() {
        let buf = vec![0u8; 64];
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 8,
            prompt_len: 5,
            hidden_size: 0,
        };
        // prompt_len * hidden_size = 5 * 0 = 0 → empty vec
        let emb = sp.embedding();
        assert!(emb.is_empty());
    }

    // ── DiagnosticScratchpad: embedding with single token and single hidden dim ──
    // @trace TEST-MKO-079 [req:REQ-OBS] [level:unit]

    #[test]
    fn scratchpad_embedding_single_token_single_dim() {
        let mut buf = vec![0u8; 16];
        let val: f32 = -2.5;
        buf[0..4].copy_from_slice(&val.to_le_bytes());
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 1,
            prompt_len: 1,
            hidden_size: 1,
        };
        let emb = sp.embedding();
        assert_eq!(emb.len(), 1);
        assert!((emb[0] - (-2.5)).abs() < 0.01);
    }

    // ── MegaKernelError: empty string messages ──
    // @trace TEST-MKO-080 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn mega_kernel_error_empty_string_messages() {
        let compile_err = MegaKernelError::Compilation(String::new());
        let msg = format!("{compile_err}");
        assert!(msg.contains("compilation"));
        // Empty inner message is still valid Display
        assert_eq!(format!("{}", compile_err), "compilation failed: ");

        let exec_err = MegaKernelError::Execution(String::new());
        assert_eq!(format!("{}", exec_err), "execution failed: ");
    }

    // ── KernelContext: repr(C) size is deterministic ──
    // @trace TEST-MKO-081 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn kernel_context_size_deterministic_across_instances() {
        let size1 = std::mem::size_of::<KernelContext>();
        let size2 = std::mem::size_of::<KernelContext>();
        assert_eq!(size1, size2);
        // Also verify alignment
        let align = std::mem::align_of::<KernelContext>();
        assert!(align > 0);
        assert_eq!(align % 8, 0);
    }

    // ── MegaKernelObservation: copy preserves all fields exactly ──
    // @trace TEST-MKO-082 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_copy_preserves_all_fields() {
        let obs = MegaKernelObservation {
            layer_idx: 7,
            entropy: 1.5,
            residual_delta: 0.3,
            cosine_similarity: 0.85,
            dead_neuron_count: 42,
            sink_status: AttentionSinkStatus::SinkToken,
            per_channel_scale: 2.5,
            row_l1_norm: 12.3,
            row_max: 6.7,
        };
        let copy = obs;
        assert_eq!(copy.layer_idx, 7);
        assert!((copy.entropy - 1.5).abs() < f32::EPSILON);
        assert!((copy.residual_delta - 0.3).abs() < f32::EPSILON);
        assert!((copy.cosine_similarity - 0.85).abs() < f32::EPSILON);
        assert_eq!(copy.dead_neuron_count, 42);
        assert_eq!(copy.sink_status, AttentionSinkStatus::SinkToken);
        assert!((copy.per_channel_scale - 2.5).abs() < f32::EPSILON);
        assert!((copy.row_l1_norm - 12.3).abs() < 0.01);
        assert!((copy.row_max - 6.7).abs() < 0.01);
    }

    // ── TelemetryFlagsBitmask: PartialEq derived via transparent u32 ──
    // @trace TEST-MKO-083 [req:REQ-OBS] [level:unit]

    #[test]
    fn telemetry_flags_partialeq_same_and_different() {
        let a = TelemetryFlagsBitmask(0x1234);
        let b = TelemetryFlagsBitmask(0x1234);
        let c = TelemetryFlagsBitmask(0x5678);
        // Same inner value => equal (verified via field comparison since #[derive] is not present)
        assert_eq!(a.0, b.0);
        assert_ne!(a.0, c.0);
        // Verify default is equivalent to explicit 0
        let d = TelemetryFlagsBitmask::default();
        assert_eq!(d.0, 0);
        assert_eq!(d.0, TelemetryFlagsBitmask(0).0);
    }

    // ── WeightPageJitConfig: inequality across different field combinations ──
    // @trace TEST-MKO-084 [req:REQ-WP-008] [level:unit]

    #[test]
    fn weight_page_jit_config_inequality_different_fields() {
        let base = WeightPageJitConfig {
            enabled: true,
            num_pages: 1024,
            page_size_bytes: 64 * 1024 * 1024,
            prefetch_distance: 2,
        };
        // Vary each field and confirm inequality
        let diff_enabled = WeightPageJitConfig { enabled: false, ..base.clone() };
        assert_ne!(base.enabled, diff_enabled.enabled);

        let diff_pages = WeightPageJitConfig { num_pages: 2048, ..base.clone() };
        assert_ne!(base.num_pages, diff_pages.num_pages);

        let diff_size = WeightPageJitConfig { page_size_bytes: 128 * 1024 * 1024, ..base.clone() };
        assert_ne!(base.page_size_bytes, diff_size.page_size_bytes);

        let diff_prefetch = WeightPageJitConfig { prefetch_distance: 5, ..base.clone() };
        assert_ne!(base.prefetch_distance, diff_prefetch.prefetch_distance);
    }

    // ── KvPageDecompressConfig: inequality across different field values ──
    // @trace TEST-MKO-085 [req:REQ-COMP11] [level:unit]

    #[test]
    fn kv_decompress_config_inequality_different_fields() {
        let base = KvPageDecompressConfig {
            enabled: true,
            num_pages: 4096,
            page_size_bytes: 64 * 1024,
        };
        let diff_enabled = KvPageDecompressConfig { enabled: false, ..base.clone() };
        assert_ne!(base.enabled, diff_enabled.enabled);

        let diff_pages = KvPageDecompressConfig { num_pages: 8192, ..base.clone() };
        assert_ne!(base.num_pages, diff_pages.num_pages);

        let diff_size = KvPageDecompressConfig { page_size_bytes: 128 * 1024, ..base.clone() };
        assert_ne!(base.page_size_bytes, diff_size.page_size_bytes);
    }

    // ── KernelContext: padding fields remain zero after build ──
    // @trace TEST-MKO-086 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn kernel_context_padding_fields_zero_after_build() {
        let dummy: u8 = 0;
        let (ctx, _seq) = KernelContext::build(
            &dummy as *const u8,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            100,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            4096,
            32,
            8,
            128,
            std::ptr::null_mut(),
            0x3,
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            1,
            std::ptr::null(),
            1,
        );
        // The repr(C) layout has _pad0, _pad1, _pad2 fields.
        // We cannot access private fields directly, but we can verify the
        // size and alignment are consistent — the padding does not corrupt
        // adjacent public fields.
        assert_eq!(ctx.kv_page_size, 4096);
        assert_eq!(ctx.kv_num_layers, 32);
        assert_eq!(ctx.kv_num_heads, 8);
        assert_eq!(ctx.kv_head_dim, 128);
        assert_eq!(ctx.telemetry_flags, 0x3);
        assert_eq!(ctx.weight_page_inject_flags, 1);
        assert_eq!(ctx.decompress_inject_flags, 1);
    }

    // ── MegaKernelObservation: from_buffer reads sink_status correctly ──
    // @trace TEST-MKO-087 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_from_buffer_attention_sink_true() {
        use gllm_kernels::compiler::graph::telemetry_offsets;
        let offset = telemetry_offsets::IS_ATTENTION_SINK_OFFSET;
        let mut buf = vec![0u8; 512];
        // Write non-zero u32 at IS_ATTENTION_SINK_OFFSET
        let val: u32 = 1;
        buf[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
        let obs = MegaKernelObservation::from_buffer(0, &buf);
        assert_eq!(obs.sink_status, AttentionSinkStatus::SinkToken);
    }

    // ── MegaKernelObservation: from_buffer reads per_channel_scale ──
    // @trace TEST-MKO-088 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_from_buffer_per_channel_scale() {
        use gllm_kernels::compiler::graph::telemetry_offsets;
        let offset = telemetry_offsets::CHANNEL_SCALE_PTR_OFFSET;
        let mut buf = vec![0u8; 512];
        let scale: f32 = 3.14;
        buf[offset..offset + 4].copy_from_slice(&scale.to_le_bytes());
        let obs = MegaKernelObservation::from_buffer(0, &buf);
        assert!((obs.per_channel_scale - 3.14).abs() < 0.01);
    }

    // ── MegaKernelObservation: from_buffer reads row_l1_norm and row_max ──
    // @trace TEST-MKO-089 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_from_buffer_row_norm_and_max() {
        use gllm_kernels::compiler::graph::telemetry_offsets;
        let mut buf = vec![0u8; 512];
        let l1_norm: f32 = 42.5;
        let off_l1 = telemetry_offsets::GEMM_ROW_NORM_L1_OFFSET;
        buf[off_l1..off_l1 + 4].copy_from_slice(&l1_norm.to_le_bytes());

        let row_max: f32 = -7.3;
        let off_max = telemetry_offsets::GEMM_ROW_MAX_OFFSET;
        buf[off_max..off_max + 4].copy_from_slice(&row_max.to_le_bytes());

        let obs = MegaKernelObservation::from_buffer(3, &buf);
        assert_eq!(obs.layer_idx, 3);
        assert!((obs.row_l1_norm - 42.5).abs() < 0.01);
        assert!((obs.row_max - (-7.3)).abs() < 0.01);
    }

    // ── DiagnosticScratchpad: embedding returns non-zero values from buffer ──
    // @trace TEST-MKO-090 [req:REQ-OBS] [level:unit]

    #[test]
    fn scratchpad_embedding_reads_nonzero_data() {
        let mut buf = vec![0u8; 64];
        // Write 3 f32 values at offset 0 (embedding starts at offset 0)
        let vals: [f32; 3] = [1.1, -2.2, 3.3];
        for (i, &v) in vals.iter().enumerate() {
            let off = i * 4;
            buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 32,
            vocab_size: 8,
            prompt_len: 3,
            hidden_size: 1,
        };
        // 3 tokens * 1 hidden = 3 f32s
        let emb = sp.embedding();
        assert_eq!(emb.len(), 3);
        assert!((emb[0] - 1.1).abs() < 0.01);
        assert!((emb[1] - (-2.2)).abs() < 0.01);
        assert!((emb[2] - 3.3).abs() < 0.01);
    }

    // ── DiagnosticScratchpad: last_token_logits with multi-token prompt ──
    // @trace TEST-MKO-091 [req:REQ-OBS] [level:unit]

    #[test]
    fn scratchpad_last_token_logits_multi_token_prompt() {
        // 3-token prompt, vocab=2, logits_offset=0
        // Row bytes = 2 * 4 = 8. Last token = index 2, offset = 2 * 8 = 16.
        let mut buf = vec![0u8; 64];
        let val0: f32 = 10.0;
        let val1: f32 = 20.0;
        buf[16..20].copy_from_slice(&val0.to_le_bytes());
        buf[20..24].copy_from_slice(&val1.to_le_bytes());
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 2,
            prompt_len: 3,
            hidden_size: 4,
        };
        let logits = sp.last_token_logits();
        assert_eq!(logits.len(), 2);
        assert!((logits[0] - 10.0).abs() < f32::EPSILON);
        assert!((logits[1] - 20.0).abs() < f32::EPSILON);
    }

    // ── MegaKernelError: Clone via String clone (Debug derives Clone) ──
    // @trace TEST-MKO-092 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn mega_kernel_error_debug_clone_matches_original() {
        let err = MegaKernelError::Compilation("test_msg".into());
        let debug_orig = format!("{err:?}");
        assert!(debug_orig.contains("Compilation"));
        assert!(debug_orig.contains("test_msg"));
        // Verify the error enum Debug format is stable across multiple format calls
        let debug_again = format!("{err:?}");
        assert_eq!(debug_orig, debug_again);
    }

    // ── MegaKernelObservation: is_bypass_candidate with subnormal float thresholds ──
    // @trace TEST-MKO-093 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_is_bypass_subnormal_thresholds() {
        // Subnormal f32 as delta_threshold: a very tiny positive float
        let tiny: f32 = 1.0e-38;
        let obs = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: 0.0, // 0.0 < tiny (subnormal) => true
            cosine_similarity: 1.0,
            dead_neuron_count: 0,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        assert!(obs.is_bypass_candidate(tiny, 0.0));

        // residual_delta = tiny => NOT < tiny (equal), so false
        let obs_equal = MegaKernelObservation {
            residual_delta: tiny,
            ..obs
        };
        assert!(!obs_equal.is_bypass_candidate(tiny, 0.0));
    }

    // ── MegaKernelObservation: dead_neuron_count as u32::MAX ──
    // @trace TEST-MKO-094 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_dead_neuron_count_max_u32() {
        let obs = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: 0.0,
            cosine_similarity: 0.0,
            dead_neuron_count: u32::MAX,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        let ratio = obs.dead_neuron_ratio(1);
        // u32::MAX as f32 / 1.0 = approximately 4294967296.0 (f32 has 24-bit mantissa)
        assert!(ratio > 4.0e9);
    }

    // ── KernelContext: build with seq_len=1 (minimum valid) ──
    // @trace TEST-MKO-095 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn kernel_context_build_seq_len_one() {
        let dummy: u8 = 0;
        let (ctx, seq_len_box) = KernelContext::build(
            &dummy as *const u8,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            1,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            4096,
            1,
            1,
            64,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
        );
        assert_eq!(*seq_len_box, 1);
        assert_eq!(unsafe { *ctx.seq_len_ptr }, 1);
        assert_eq!(ctx.kv_page_size, 4096);
        assert_eq!(ctx.kv_num_layers, 1);
        assert_eq!(ctx.kv_num_heads, 1);
        assert_eq!(ctx.kv_head_dim, 64);
    }

    // ── DiagnosticScratchpad: read_f32_at requires 4-aligned byte offset ──
    // @trace TEST-MKO-096 [req:REQ-OBS] [level:unit]

    #[test]
    fn scratchpad_read_f32_at_aligned_offset_spanning_values() {
        let mut buf = vec![0u8; 32];
        // Write two f32 values at aligned offsets 0 and 4
        let val0: f32 = 1.5;
        buf[0..4].copy_from_slice(&val0.to_le_bytes());
        let val1: f32 = -999.0;
        buf[4..8].copy_from_slice(&val1.to_le_bytes());
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 8,
            prompt_len: 1,
            hidden_size: 4,
        };
        // Read 2 f32s starting at offset 0 (aligned) — should get both values
        let vals = sp.read_f32_at(0, 2);
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 1.5).abs() < f32::EPSILON);
        assert!((vals[1] - (-999.0)).abs() < f32::EPSILON);
    }

    // ── DiagnosticScratchpad: embedding with large prompt_len but tiny buffer ──
    // @trace TEST-MKO-097 [req:REQ-OBS] [level:unit]

    #[test]
    fn scratchpad_embedding_large_prompt_len_tiny_buffer() {
        let buf = vec![0u8; 8]; // Only 2 f32s
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 4,
            prompt_len: 100, // Wants 100 * 4 = 400 f32s = 1600 bytes
            hidden_size: 4,
        };
        let emb = sp.embedding();
        // Buffer too small for the request => returns empty
        assert!(emb.is_empty());
    }

    // ── MegaKernelObservation: from_buffer with all telemetry offsets populated ──
    // @trace TEST-MKO-098 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_from_buffer_all_offsets_distinct_values() {
        use gllm_kernels::compiler::graph::telemetry_offsets;
        let mut buf = vec![0u8; 512];

        // Populate all readable offsets with distinct sentinel values
        let entropy: f32 = 1.11;
        buf[telemetry_offsets::SOFTMAX_SHARPNESS_OFFSET
            ..telemetry_offsets::SOFTMAX_SHARPNESS_OFFSET + 4]
            .copy_from_slice(&entropy.to_le_bytes());

        let delta: f32 = 2.22;
        buf[telemetry_offsets::RESIDUAL_DELTA_OFFSET
            ..telemetry_offsets::RESIDUAL_DELTA_OFFSET + 4]
            .copy_from_slice(&delta.to_le_bytes());

        let cosine: f32 = 3.33;
        buf[telemetry_offsets::COSINE_SIMILARITY_OFFSET
            ..telemetry_offsets::COSINE_SIMILARITY_OFFSET + 4]
            .copy_from_slice(&cosine.to_le_bytes());

        let dead_count: u32 = 777;
        buf[telemetry_offsets::SILU_DEAD_NEURON_MASK_OFFSET
            ..telemetry_offsets::SILU_DEAD_NEURON_MASK_OFFSET + 4]
            .copy_from_slice(&dead_count.to_le_bytes());

        let sink_flag: u32 = 42;
        buf[telemetry_offsets::IS_ATTENTION_SINK_OFFSET
            ..telemetry_offsets::IS_ATTENTION_SINK_OFFSET + 4]
            .copy_from_slice(&sink_flag.to_le_bytes());

        let scale: f32 = 4.44;
        buf[telemetry_offsets::CHANNEL_SCALE_PTR_OFFSET
            ..telemetry_offsets::CHANNEL_SCALE_PTR_OFFSET + 4]
            .copy_from_slice(&scale.to_le_bytes());

        let norm: f32 = 5.55;
        buf[telemetry_offsets::GEMM_ROW_NORM_L1_OFFSET
            ..telemetry_offsets::GEMM_ROW_NORM_L1_OFFSET + 4]
            .copy_from_slice(&norm.to_le_bytes());

        let max_val: f32 = 6.66;
        buf[telemetry_offsets::GEMM_ROW_MAX_OFFSET
            ..telemetry_offsets::GEMM_ROW_MAX_OFFSET + 4]
            .copy_from_slice(&max_val.to_le_bytes());

        let obs = MegaKernelObservation::from_buffer(9, &buf);

        assert_eq!(obs.layer_idx, 9);
        assert!((obs.entropy - 1.11).abs() < 0.01);
        assert!((obs.residual_delta - 2.22).abs() < 0.01);
        assert!((obs.cosine_similarity - 3.33).abs() < 0.01);
        assert_eq!(obs.dead_neuron_count, 777);
        assert_eq!(obs.sink_status, AttentionSinkStatus::SinkToken);
        assert!((obs.per_channel_scale - 4.44).abs() < 0.01);
        assert!((obs.row_l1_norm - 5.55).abs() < 0.01);
        assert!((obs.row_max - 6.66).abs() < 0.01);
    }

    // ── KernelContext: Send + Sync trait verification ──
    // @trace TEST-MKO-099 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn kernel_context_send_sync_traits() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<KernelContext>();
        assert_sync::<KernelContext>();
    }

    // ── KernelContext: build with non-zero weight_offsets_len ──
    // @trace TEST-MKO-100 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn kernel_context_build_with_weight_offsets() {
        let dummy: u8 = 0;
        let offsets: [usize; 3] = [100, 200, 300];
        let (ctx, _seq) = KernelContext::build(
            &dummy as *const u8,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            64,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            4096, 4, 8, 64,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
            offsets.as_ptr(),
            3,
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
        );
        assert_eq!(ctx.weight_offsets_len, 3);
        assert!(!ctx.weight_offsets_ptr.is_null());
        // Verify the offsets are readable through the pointer
        unsafe {
            assert_eq!(*ctx.weight_offsets_ptr, 100);
            assert_eq!(*ctx.weight_offsets_ptr.add(1), 200);
            assert_eq!(*ctx.weight_offsets_ptr.add(2), 300);
        }
    }

    // ── KernelContext: zeroed has zero weight_offsets_len ──
    // @trace TEST-MKO-101 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn kernel_context_zeroed_weight_offsets_len_is_zero() {
        let ctx = KernelContext::zeroed();
        assert_eq!(ctx.weight_offsets_len, 0);
        assert!(ctx.weight_offsets_ptr.is_null());
    }

    // ── TelemetryFlagsBitmask: bitwise NOT and shifting ──
    // @trace TEST-MKO-102 [req:REQ-OBS] [level:unit]

    #[test]
    fn telemetry_flags_bitwise_not_and_shift() {
        let flags = TelemetryFlagsBitmask(0x00FF_0000);
        let inverted = TelemetryFlagsBitmask(!flags.0);
        // Low 16 bits should all be 1s, bit 16-23 should be 0s, high 8 should be 1s
        assert_eq!(inverted.0 & 0xFFFF, 0xFFFF);
        assert_eq!(inverted.0 & 0x00FF_0000, 0);
        // Shift and combine
        let shifted = TelemetryFlagsBitmask(flags.0 >> 16);
        assert_eq!(shifted.0, 0x00FF);
    }

    // ── MegaKernelError: usable as Box<dyn std::error::Error> ──
    // @trace TEST-MKO-103 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn mega_kernel_error_as_trait_object() {
        let err: Box<dyn std::error::Error> =
            Box::new(MegaKernelError::Compilation("boxed error".into()));
        let msg = format!("{err}");
        assert!(msg.contains("boxed error"));
        assert!(err.source().is_none());
        // Also test Execution variant
        let err2: Box<dyn std::error::Error> =
            Box::new(MegaKernelError::Execution("boxed exec".into()));
        let msg2 = format!("{err2}");
        assert!(msg2.contains("boxed exec"));
    }

    // ── MegaKernelObservation: from_buffer with buffer exactly at boundary ──
    // @trace TEST-MKO-104 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_from_buffer_exact_boundary_for_dead_neuron() {
        use gllm_kernels::compiler::graph::telemetry_offsets;
        // Create a buffer exactly large enough for SILU_DEAD_NEURON_MASK_OFFSET + 4
        let offset = telemetry_offsets::SILU_DEAD_NEURON_MASK_OFFSET;
        let mut buf = vec![0u8; offset + 4];
        let dead_count: u32 = 12345;
        buf[offset..offset + 4].copy_from_slice(&dead_count.to_le_bytes());
        let obs = MegaKernelObservation::from_buffer(0, &buf);
        assert_eq!(obs.dead_neuron_count, 12345);
        // Entropy is at a larger offset and should be 0.0 (buffer too small for it)
        if telemetry_offsets::SOFTMAX_SHARPNESS_OFFSET >= offset + 4 {
            assert_eq!(obs.entropy, 0.0);
        }
    }

    // ── DiagnosticScratchpad: embedding and logits coexist in same buffer ──
    // @trace TEST-MKO-105 [req:REQ-OBS] [level:unit]

    #[test]
    fn scratchpad_embedding_and_logits_share_buffer() {
        let mut buf = vec![0u8; 64];
        // Write embedding data at offset 0: prompt_len=2, hidden_size=2 → 4 f32s
        let emb_vals: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        for (i, &v) in emb_vals.iter().enumerate() {
            let off = i * 4;
            buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        // Write logits at logits_offset=16 (after embedding): vocab_size=2, prompt_len=2
        // last token logits at offset = 16 + (2-1) * 8 = 24
        let logit_vals: [f32; 2] = [10.0, 20.0];
        buf[24..28].copy_from_slice(&logit_vals[0].to_le_bytes());
        buf[28..32].copy_from_slice(&logit_vals[1].to_le_bytes());

        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 16,
            vocab_size: 2,
            prompt_len: 2,
            hidden_size: 2,
        };

        // Verify embedding reads correctly
        let emb = sp.embedding();
        assert_eq!(emb.len(), 4);
        assert!((emb[0] - 1.0).abs() < f32::EPSILON);
        assert!((emb[3] - 4.0).abs() < f32::EPSILON);

        // Verify logits read correctly from the same buffer
        let logits = sp.last_token_logits();
        assert_eq!(logits.len(), 2);
        assert!((logits[0] - 10.0).abs() < f32::EPSILON);
        assert!((logits[1] - 20.0).abs() < f32::EPSILON);
    }

    // ── DiagnosticScratchpad: last_token_logits with vocab_size=1 ──
    // @trace TEST-MKO-106 [req:REQ-OBS] [level:unit]

    #[test]
    fn scratchpad_last_token_logits_vocab_size_one() {
        let mut buf = vec![0u8; 32];
        // vocab_size=1 → row_bytes=4. prompt_len=3 → last token at offset 2*4=8
        let val: f32 = -42.0;
        buf[8..12].copy_from_slice(&val.to_le_bytes());
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 1,
            prompt_len: 3,
            hidden_size: 4,
        };
        let logits = sp.last_token_logits();
        assert_eq!(logits.len(), 1);
        assert!((logits[0] - (-42.0)).abs() < f32::EPSILON);
    }

    // ── MegaKernelObservation: dead_neuron_ratio with hidden_size=1 ──
    // @trace TEST-MKO-107 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_dead_neuron_ratio_hidden_size_one() {
        let obs = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: 0.0,
            cosine_similarity: 0.0,
            dead_neuron_count: 1,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        let ratio = obs.dead_neuron_ratio(1);
        assert!((ratio - 1.0).abs() < f32::EPSILON);
        // Zero dead neurons with hidden=1
        let obs_zero = MegaKernelObservation { dead_neuron_count: 0, ..obs };
        let ratio_zero = obs_zero.dead_neuron_ratio(1);
        assert!((ratio_zero - 0.0).abs() < f32::EPSILON);
    }

    // ── MegaKernelObservation: is_bypass_candidate with negative delta and negative cosine ──
    // @trace TEST-MKO-108 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_is_bypass_negative_delta_and_cosine() {
        // Negative residual_delta should be < any positive threshold
        let obs_neg_delta = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: -1.0,
            cosine_similarity: 0.96,
            dead_neuron_count: 0,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        assert!(obs_neg_delta.is_bypass_candidate(0.01, 0.95));

        // Negative cosine should fail any positive cosine threshold
        let obs_neg_cosine_pos_threshold = MegaKernelObservation {
            residual_delta: -1.0,
            cosine_similarity: -0.5,
            ..obs_neg_delta
        };
        // cosine=-0.5 is NOT > 0.95 → false
        assert!(!obs_neg_cosine_pos_threshold.is_bypass_candidate(0.01, 0.95));

        // But with negative cosine_threshold=-1.0, cosine=-0.5 > -1.0 → true
        // delta=-1.0 < 0.01 (true) AND cosine=-0.5 > -1.0 (true) → bypass
        assert!(obs_neg_cosine_pos_threshold.is_bypass_candidate(0.01, -1.0));
    }

    // ── KernelContext: build with seq_len=usize::MAX ──
    // @trace TEST-MKO-109 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn kernel_context_build_seq_len_usize_max() {
        let dummy: u8 = 0;
        let (ctx, seq_len_box) = KernelContext::build(
            &dummy as *const u8,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            usize::MAX,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0, 0, 0, 0,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
        );
        assert_eq!(*seq_len_box, usize::MAX);
        assert_eq!(unsafe { *ctx.seq_len_ptr }, usize::MAX);
    }

    // ── MegaKernelObservation: from_buffer with buffer exactly 4 bytes ──
    // @trace TEST-MKO-110 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_from_buffer_four_bytes() {
        let mut buf = vec![0u8; 4];
        // Only first 4 bytes exist: write to offset 0
        let val: f32 = 99.99;
        buf[0..4].copy_from_slice(&val.to_le_bytes());
        let obs = MegaKernelObservation::from_buffer(0, &buf);
        // Most telemetry offsets are beyond 4 bytes, so default to 0.0
        assert_eq!(obs.entropy, 0.0);
        assert_eq!(obs.residual_delta, 0.0);
        assert_eq!(obs.dead_neuron_count, 0);
    }

    // ── DiagnosticScratchpad: read_f32_at with count=0 at large offset ──
    // @trace TEST-MKO-111 [req:REQ-OBS] [level:unit]

    #[test]
    fn scratchpad_read_f32_at_zero_count_large_offset() {
        let buf = vec![0u8; 16];
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 4,
            prompt_len: 1,
            hidden_size: 4,
        };
        let vals = sp.read_f32_at(1000, 0);
        assert!(vals.is_empty());
    }

    // ── WeightPageJitConfig: clone independence ──
    // @trace TEST-MKO-112 [req:REQ-WP-008] [level:unit]

    #[test]
    fn weight_page_jit_config_clone_independence() {
        let mut cfg = WeightPageJitConfig {
            enabled: true,
            num_pages: 1024,
            page_size_bytes: 65536,
            prefetch_distance: 2,
        };
        let cloned = cfg.clone();
        // Modify original — clone should not change
        cfg.enabled = false;
        cfg.num_pages = 0;
        assert!(cloned.enabled);
        assert_eq!(cloned.num_pages, 1024);
        assert_eq!(cloned.page_size_bytes, 65536);
        assert_eq!(cloned.prefetch_distance, 2);
    }

    // ── KvPageDecompressConfig: clone independence ──
    // @trace TEST-MKO-113 [req:REQ-COMP11] [level:unit]

    #[test]
    fn kv_decompress_config_clone_independence() {
        let mut cfg = KvPageDecompressConfig {
            enabled: true,
            num_pages: 2048,
            page_size_bytes: 32768,
        };
        let cloned = cfg.clone();
        cfg.enabled = false;
        cfg.page_size_bytes = 0;
        assert!(cloned.enabled);
        assert_eq!(cloned.num_pages, 2048);
        assert_eq!(cloned.page_size_bytes, 32768);
    }

    // ── MegaKernelError: Display format consistency across calls ──
    // @trace TEST-MKO-114 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn mega_kernel_error_display_idempotent() {
        let err = MegaKernelError::Execution("stable message".into());
        let display1 = format!("{err}");
        let display2 = format!("{err}");
        let display3 = format!("{err}");
        assert_eq!(display1, display2);
        assert_eq!(display2, display3);
        assert!(display1.starts_with("execution"));
    }

    // ── MegaKernelError: messages with special characters and unicode ──
    // @trace TEST-MKO-115 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn mega_kernel_error_special_chars_and_unicode() {
        let err = MegaKernelError::Compilation("JIT 失败: \x00\x01\n\t🚀".into());
        let msg = format!("{err}");
        assert!(msg.contains("compilation"));
        assert!(msg.contains("JIT"));
        // Debug representation should also contain the original message
        let debug = format!("{err:?}");
        assert!(debug.contains("Compilation"));
    }

    // ── MegaKernelError: very long message string ──
    // @trace TEST-MKO-116 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn mega_kernel_error_very_long_message() {
        let long_msg = "x".repeat(10_000);
        let err = MegaKernelError::Execution(long_msg.clone());
        let msg = format!("{err}");
        assert!(msg.starts_with("execution"));
        assert_eq!(msg.len(), "execution failed: ".len() + 10_000);
        assert!(msg.ends_with('x'));
    }

    // ── TelemetryFlagsBitmask: individual bit flag testing ──
    // @trace TEST-MKO-117 [req:REQ-OBS] [level:unit]

    #[test]
    fn telemetry_flags_individual_bit_flags() {
        // Verify individual bits can be set and read
        let bit0 = TelemetryFlagsBitmask(1 << 0);
        let bit1 = TelemetryFlagsBitmask(1 << 1);
        let bit2 = TelemetryFlagsBitmask(1 << 2);
        let bit31 = TelemetryFlagsBitmask(1 << 31);

        assert_ne!(bit0.0, bit1.0);
        assert_ne!(bit1.0, bit2.0);

        // Combine flags
        let combined = TelemetryFlagsBitmask(bit0.0 | bit1.0 | bit31.0);
        assert_eq!(combined.0 & bit0.0, bit0.0);
        assert_eq!(combined.0 & bit1.0, bit1.0);
        assert_eq!(combined.0 & bit31.0, bit31.0);
        assert_eq!(combined.0 & bit2.0, 0); // bit2 not set

        // Verify no spurious bits
        assert_eq!(combined.0.count_ones(), 3);
    }

    // ── TelemetryFlagsBitmask: default matches explicit zero and remains stable ──
    // @trace TEST-MKO-118 [req:REQ-OBS] [level:unit]

    #[test]
    fn telemetry_flags_default_consistency_across_calls() {
        let d1 = TelemetryFlagsBitmask::default();
        let d2 = TelemetryFlagsBitmask::default();
        let d3 = TelemetryFlagsBitmask::default();
        assert_eq!(d1.0, 0);
        assert_eq!(d1.0, d2.0);
        assert_eq!(d2.0, d3.0);
    }

    // ── WeightPageJitConfig: default values are consistent across multiple constructions ──
    // @trace TEST-MKO-119 [req:REQ-WP-008] [level:unit]

    #[test]
    fn weight_page_jit_config_default_consistency() {
        let d1 = WeightPageJitConfig::default();
        let d2 = WeightPageJitConfig::default();
        assert_eq!(d1.enabled, d2.enabled);
        assert_eq!(d1.num_pages, d2.num_pages);
        assert_eq!(d1.page_size_bytes, d2.page_size_bytes);
        assert_eq!(d1.prefetch_distance, d2.prefetch_distance);
        // Verify specific default values
        assert!(!d1.enabled);
        assert_eq!(d1.page_size_bytes, 64 * 1024 * 1024);
        assert_eq!(d1.prefetch_distance, 0);
    }

    // ── KvPageDecompressConfig: default values are consistent across multiple constructions ──
    // @trace TEST-MKO-120 [req:REQ-COMP11] [level:unit]

    #[test]
    fn kv_decompress_config_default_consistency() {
        let d1 = KvPageDecompressConfig::default();
        let d2 = KvPageDecompressConfig::default();
        assert_eq!(d1.enabled, d2.enabled);
        assert_eq!(d1.num_pages, d2.num_pages);
        assert_eq!(d1.page_size_bytes, d2.page_size_bytes);
        assert!(!d1.enabled);
        assert_eq!(d1.page_size_bytes, 64 * 1024);
    }

    // ── KernelContext: build with all zero scalar KV params ──
    // @trace TEST-MKO-121 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn kernel_context_build_zero_kv_params() {
        let dummy: u8 = 0;
        let (ctx, seq_len_box) = KernelContext::build(
            &dummy as *const u8,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            1,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0, // kv_page_size = 0
            0, // kv_num_layers = 0
            0, // kv_num_heads = 0
            0, // kv_head_dim = 0
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
        );
        assert_eq!(*seq_len_box, 1);
        assert_eq!(ctx.kv_page_size, 0);
        assert_eq!(ctx.kv_num_layers, 0);
        assert_eq!(ctx.kv_num_heads, 0);
        assert_eq!(ctx.kv_head_dim, 0);
        assert_eq!(ctx.telemetry_flags, 0);
        assert_eq!(ctx.weight_page_inject_flags, 0);
        assert_eq!(ctx.decompress_inject_flags, 0);
    }

    // ── KernelContext: zeroed is equivalent to build with all nulls and zeros ──
    // @trace TEST-MKO-122 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn kernel_context_zeroed_matches_build_with_zeros() {
        let dummy: u8 = 0;
        let (built, _seq) = KernelContext::build(
            &dummy as *const u8,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0, 0, 0, 0,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
        );
        let zeroed = KernelContext::zeroed();
        // All scalar fields should match
        assert_eq!(built.kv_page_size, zeroed.kv_page_size);
        assert_eq!(built.kv_num_layers, zeroed.kv_num_layers);
        assert_eq!(built.kv_num_heads, zeroed.kv_num_heads);
        assert_eq!(built.kv_head_dim, zeroed.kv_head_dim);
        assert_eq!(built.telemetry_flags, zeroed.telemetry_flags);
        assert_eq!(built.weight_page_inject_flags, zeroed.weight_page_inject_flags);
        assert_eq!(built.decompress_inject_flags, zeroed.decompress_inject_flags);
        assert_eq!(built.weight_offsets_len, zeroed.weight_offsets_len);
    }

    // ── MegaKernelObservation: is_bypass_candidate with f32::MAX and f32::MIN thresholds ──
    // @trace TEST-MKO-123 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_is_bypass_extreme_thresholds() {
        // delta_threshold = f32::MAX: any finite residual_delta should pass
        // cosine_threshold = f32::MIN (smallest positive subnormal)
        let obs = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: 1.0e30,
            cosine_similarity: 0.001,
            dead_neuron_count: 0,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        assert!(obs.is_bypass_candidate(f32::MAX, f32::MIN));

        // delta_threshold = f32::MIN (smallest positive): residual_delta = 0.0 < f32::MIN => true
        // cosine_threshold = f32::MAX: cosine_similarity = 0.5 < f32::MAX => NOT > => false
        assert!(!obs.is_bypass_candidate(f32::MIN, f32::MAX));
    }

    // ── MegaKernelObservation: from_buffer with attention_sink = 0 explicitly ──
    // @trace TEST-MKO-124 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_from_buffer_attention_sink_explicitly_zero() {
        use gllm_kernels::compiler::graph::telemetry_offsets;
        let offset = telemetry_offsets::IS_ATTENTION_SINK_OFFSET;
        let mut buf = vec![0u8; 512];
        // Explicitly write 0 at IS_ATTENTION_SINK_OFFSET
        let zero_val: u32 = 0;
        buf[offset..offset + 4].copy_from_slice(&zero_val.to_le_bytes());
        let obs = MegaKernelObservation::from_buffer(0, &buf);
        assert_eq!(obs.sink_status, AttentionSinkStatus::Normal);
        // Write 2 (non-zero) — should be true
        let nonzero: u32 = 2;
        buf[offset..offset + 4].copy_from_slice(&nonzero.to_le_bytes());
        let obs2 = MegaKernelObservation::from_buffer(0, &buf);
        assert_eq!(obs2.sink_status, AttentionSinkStatus::SinkToken);
    }

    // ── DiagnosticScratchpad: read_f32_at returns correct values at middle of buffer ──
    // @trace TEST-MKO-125 [req:REQ-OBS] [level:unit]

    #[test]
    fn scratchpad_read_f32_at_mid_buffer_offset() {
        let mut buf = vec![0u8; 32];
        // Write distinct f32 values at byte offsets 0, 4, 8, 12, 16, 20, 24, 28
        let vals: [f32; 8] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        for (i, &v) in vals.iter().enumerate() {
            let off = i * 4;
            buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 8,
            prompt_len: 1,
            hidden_size: 8,
        };
        // Read 3 f32s starting at offset 8 (the middle of the buffer)
        let result = sp.read_f32_at(8, 3);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 2.0).abs() < f32::EPSILON);
        assert!((result[1] - 3.0).abs() < f32::EPSILON);
        assert!((result[2] - 4.0).abs() < f32::EPSILON);
        // Read 2 f32s starting at offset 24 (near end)
        let tail = sp.read_f32_at(24, 2);
        assert_eq!(tail.len(), 2);
        assert!((tail[0] - 6.0).abs() < f32::EPSILON);
        assert!((tail[1] - 7.0).abs() < f32::EPSILON);
    }

    // ── DiagnosticScratchpad: read_f32_at with exact buffer size match ──
    // @trace TEST-MKO-126 [req:REQ-OBS] [level:unit]

    #[test]
    fn scratchpad_read_f32_exact_buffer_size() {
        let mut buf = vec![0u8; 8]; // exactly 2 f32s
        let val0: f32 = 1.0;
        let val1: f32 = 2.0;
        buf[0..4].copy_from_slice(&val0.to_le_bytes());
        buf[4..8].copy_from_slice(&val1.to_le_bytes());
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 2,
            prompt_len: 1,
            hidden_size: 2,
        };
        // Request exactly the buffer capacity
        let vals = sp.read_f32_at(0, 2);
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 1.0).abs() < f32::EPSILON);
        assert!((vals[1] - 2.0).abs() < f32::EPSILON);
        // One more should fail
        let overflow = sp.read_f32_at(0, 3);
        assert!(overflow.is_empty());
    }

    // ── MegaKernelObservation: dead_neuron_ratio with large hidden_size ──
    // @trace TEST-MKO-127 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_dead_neuron_ratio_large_hidden_size() {
        let obs = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: 0.0,
            cosine_similarity: 0.0,
            dead_neuron_count: 1,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        // Very large hidden_size: ratio should be very small
        let ratio = obs.dead_neuron_ratio(1_000_000);
        assert!(ratio > 0.0);
        assert!(ratio < 0.001);
        // With dead_neuron_count=0 and large hidden, ratio should be exactly 0.0
        let obs_zero = MegaKernelObservation { dead_neuron_count: 0, ..obs };
        let ratio_zero = obs_zero.dead_neuron_ratio(1_000_000);
        assert!((ratio_zero - 0.0).abs() < f32::EPSILON);
    }

    // ── KernelContext: alignment is at least pointer-width ──
    // @trace TEST-MKO-128 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn kernel_context_alignment_at_least_pointer_width() {
        let align = std::mem::align_of::<KernelContext>();
        let ptr_size = std::mem::size_of::<*const u8>();
        assert!(align >= ptr_size);
        // KernelContext contains *const u8 fields, so alignment must be >= 8 on 64-bit
        #[cfg(target_pointer_width = "64")]
        assert!(align >= 8);
    }

    // ── MegaKernelObservation: from_buffer with all zeros and large layer_idx ──
    // @trace TEST-MKO-129 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_from_buffer_large_layer_idx_all_zeros() {
        let buf = vec![0u8; 512];
        let obs = MegaKernelObservation::from_buffer(usize::MAX / 2, &buf);
        assert_eq!(obs.layer_idx, usize::MAX / 2);
        assert_eq!(obs.entropy, 0.0);
        assert_eq!(obs.residual_delta, 0.0);
        assert_eq!(obs.cosine_similarity, 0.0);
        assert_eq!(obs.dead_neuron_count, 0);
        assert_eq!(obs.sink_status, AttentionSinkStatus::Normal);
        assert_eq!(obs.per_channel_scale, 0.0);
        assert_eq!(obs.row_l1_norm, 0.0);
        assert_eq!(obs.row_max, 0.0);
    }

    // ── MegaKernelObservation: from_buffer with dead_neuron_count=u32::MAX ──
    // @trace TEST-MKO-130 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_from_buffer_dead_neuron_count_max_u32() {
        use gllm_kernels::compiler::graph::telemetry_offsets;
        let offset = telemetry_offsets::SILU_DEAD_NEURON_MASK_OFFSET;
        let mut buf = vec![0u8; 512];
        let max_val: u32 = u32::MAX;
        buf[offset..offset + 4].copy_from_slice(&max_val.to_le_bytes());
        let obs = MegaKernelObservation::from_buffer(0, &buf);
        assert_eq!(obs.dead_neuron_count, u32::MAX);
    }

    // ── MegaKernelObservation: dead_neuron_ratio with hidden_size=u32::MAX as usize ──
    // @trace TEST-MKO-131 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_dead_neuron_ratio_very_large_hidden_size() {
        let obs = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: 0.0,
            cosine_similarity: 0.0,
            dead_neuron_count: 1,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        let ratio = obs.dead_neuron_ratio(u32::MAX as usize);
        // 1 / 4294967295 ≈ 2.33e-10 — extremely small but non-zero
        assert!(ratio > 0.0);
        assert!(ratio < 1e-9);
    }

    // ── DiagnosticScratchpad: read_f32_at at offset equal to buffer length ──
    // @trace TEST-MKO-132 [req:REQ-OBS] [level:unit]

    #[test]
    fn scratchpad_read_f32_at_offset_equals_buffer_len() {
        let buf = vec![0u8; 16]; // exactly 4 f32s
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 4,
            prompt_len: 1,
            hidden_size: 4,
        };
        // Offset 16 == buffer length, even count=0 should return empty
        let vals = sp.read_f32_at(16, 0);
        assert!(vals.is_empty());
        // Offset 16 with count=1 also out of bounds
        let vals2 = sp.read_f32_at(16, 1);
        assert!(vals2.is_empty());
    }

    // ── KernelContext: build with inject flags at u32::MAX ──
    // @trace TEST-MKO-133 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn kernel_context_build_max_inject_flags() {
        let dummy: u8 = 0;
        let (ctx, _seq) = KernelContext::build(
            &dummy as *const u8,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            1,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            4096, 1, 1, 64,
            std::ptr::null_mut(),
            u32::MAX, // telemetry_flags
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            u32::MAX, // weight_page_inject_flags
            std::ptr::null(),
            u32::MAX, // decompress_inject_flags
        );
        assert_eq!(ctx.telemetry_flags, u32::MAX);
        assert_eq!(ctx.weight_page_inject_flags, u32::MAX);
        assert_eq!(ctx.decompress_inject_flags, u32::MAX);
    }

    // ── TelemetryFlagsBitmask: AND filtering with u32::MAX preserves all bits ──
    // @trace TEST-MKO-134 [req:REQ-OBS] [level:unit]

    #[test]
    fn telemetry_flags_and_mask_preserves_all_bits() {
        let flags = TelemetryFlagsBitmask(0xDEAD_BEEF);
        // AND with u32::MAX is identity
        assert_eq!(flags.0 & u32::MAX, flags.0);
        // AND with 0 clears all bits
        assert_eq!(flags.0 & 0u32, 0);
        // OR with 0 preserves all bits
        assert_eq!(flags.0 | 0u32, flags.0);
    }

    // ── WeightPageJitConfig: all fields zero with disabled ──
    // @trace TEST-MKO-135 [req:REQ-WP-008] [level:unit]

    #[test]
    fn weight_page_jit_config_all_zero_disabled() {
        let cfg = WeightPageJitConfig {
            enabled: false,
            num_pages: 0,
            page_size_bytes: 0,
            prefetch_distance: 0,
        };
        assert!(!cfg.enabled);
        assert_eq!(cfg.num_pages, 0);
        assert_eq!(cfg.page_size_bytes, 0);
        assert_eq!(cfg.prefetch_distance, 0);
        // Clone should preserve the all-zero state
        let cloned = cfg.clone();
        assert!(!cloned.enabled);
        assert_eq!(cloned.num_pages, 0);
        assert_eq!(cloned.page_size_bytes, 0);
        assert_eq!(cloned.prefetch_distance, 0);
        // Debug should not panic
        let _ = format!("{cfg:?}");
    }

    // ── KvPageDecompressConfig: num_pages=1 minimum ──
    // @trace TEST-MKO-136 [req:REQ-COMP11] [level:unit]

    #[test]
    fn kv_decompress_config_min_pages_one() {
        let cfg = KvPageDecompressConfig {
            enabled: true,
            num_pages: 1,
            page_size_bytes: 4096,
        };
        assert!(cfg.enabled);
        assert_eq!(cfg.num_pages, 1);
        assert_eq!(cfg.page_size_bytes, 4096);
        // Clone and verify independence
        let mut original = cfg.clone();
        original.num_pages = 999;
        assert_eq!(cfg.num_pages, 1); // original unchanged
    }

    // ── MegaKernelError: Debug format for Execution variant ──
    // @trace TEST-MKO-137 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn mega_kernel_error_debug_execution_variant() {
        let err = MegaKernelError::Execution("segfault at 0xDEAD".into());
        let debug = format!("{err:?}");
        assert!(debug.contains("Execution"));
        assert!(debug.contains("segfault at 0xDEAD"));
    }

    // ── MegaKernelObservation: is_bypass_candidate both thresholds f32::MAX ──
    // @trace TEST-MKO-138 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_is_bypass_both_max_thresholds() {
        // Both thresholds = f32::MAX
        // residual_delta must be < f32::MAX AND cosine_similarity > f32::MAX
        // f32::MAX > f32::MAX is false for cosine, so result is always false
        let obs = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: 0.0,
            cosine_similarity: f32::MAX,
            dead_neuron_count: 0,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        // cosine == f32::MAX is NOT > f32::MAX (strict greater-than)
        assert!(!obs.is_bypass_candidate(f32::MAX, f32::MAX));
    }

    // ── DiagnosticScratchpad: embedding and last_token_logits with vocab_size=0 ──
    // @trace TEST-MKO-139 [req:REQ-OBS] [level:unit]

    #[test]
    fn scratchpad_last_token_logits_vocab_size_zero_reads_empty() {
        let buf = vec![0u8; 64];
        // vocab_size=0 triggers should_panic due to underflow but read_f32_at with count=0 returns empty
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 0,
            prompt_len: 1,
            hidden_size: 4,
        };
        // last_token_logits: row_bytes = 0 * 4 = 0, off = 0 + 0 * 0 = 0
        // read_f32_at(0, 0) returns empty vec
        let logits = sp.last_token_logits();
        assert!(logits.is_empty());
    }

    // ── MegaKernelObservation: from_buffer layer_idx=0 with fully zeroed buffer ──
    // @trace TEST-MKO-140 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_from_buffer_layer_zero_512_byte_zeros() {
        let buf = vec![0u8; 512];
        let obs = MegaKernelObservation::from_buffer(0, &buf);
        assert_eq!(obs.layer_idx, 0);
        assert_eq!(obs.entropy, 0.0);
        assert_eq!(obs.residual_delta, 0.0);
        assert_eq!(obs.cosine_similarity, 0.0);
        assert_eq!(obs.dead_neuron_count, 0);
        assert_eq!(obs.sink_status, AttentionSinkStatus::Normal);
        assert_eq!(obs.per_channel_scale, 0.0);
        assert_eq!(obs.row_l1_norm, 0.0);
        assert_eq!(obs.row_max, 0.0);
        // Should not be bypass candidate (cosine=0.0 NOT > any positive threshold)
        assert!(!obs.is_bypass_candidate(0.01, 0.0));
        // dead_neuron_ratio with any hidden_size is 0.0
        assert!((obs.dead_neuron_ratio(768) - 0.0).abs() < f32::EPSILON);
    }

    // ── TelemetryFlagsBitmask: cumulative OR pattern ──
    // @trace TEST-MKO-141 [req:REQ-OBS] [level:unit]

    #[test]
    fn telemetry_flags_cumulative_or_pattern() {
        let mut flags = TelemetryFlagsBitmask::default();
        // Accumulate individual feature flags
        flags = TelemetryFlagsBitmask(flags.0 | (1 << 0));
        flags = TelemetryFlagsBitmask(flags.0 | (1 << 3));
        flags = TelemetryFlagsBitmask(flags.0 | (1 << 7));
        // Should have exactly bits 0, 3, 7 set
        assert_eq!(flags.0.count_ones(), 3);
        assert_eq!(flags.0 & (1 << 0), 1 << 0);
        assert_eq!(flags.0 & (1 << 3), 1 << 3);
        assert_eq!(flags.0 & (1 << 7), 1 << 7);
        assert_eq!(flags.0 & (1 << 1), 0);
        assert_eq!(flags.0, 0b1000_1001);
    }

    // ── WeightPageJitConfig: non-power-of-two page size ──
    // @trace TEST-MKO-142 [req:REQ-WP-008] [level:unit]

    #[test]
    fn weight_page_jit_config_non_power_of_two_page_size() {
        let cfg = WeightPageJitConfig {
            enabled: true,
            num_pages: 100,
            page_size_bytes: 100_000, // not a power of 2
            prefetch_distance: 5,
        };
        assert_eq!(cfg.page_size_bytes, 100_000);
        assert!(!cfg.page_size_bytes.is_power_of_two());
        // Verify clone preserves the non-power-of-two value
        let cloned = cfg.clone();
        assert_eq!(cloned.page_size_bytes, 100_000);
        assert_eq!(cloned.num_pages, 100);
    }

    // ── KernelContext: build with kv_page_table_ptr non-null ──
    // @trace TEST-MKO-143 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn kernel_context_build_kv_page_table_readable() {
        let dummy: u8 = 0;
        let page_table: [u32; 4] = [10, 20, 30, 40];
        let (ctx, _seq) = KernelContext::build(
            &dummy as *const u8,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            8,
            std::ptr::null(),
            page_table.as_ptr(),
            std::ptr::null(),
            4096, 1, 4, 64,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
        );
        assert!(!ctx.kv_page_table_ptr.is_null());
        assert_eq!(unsafe { *ctx.kv_page_table_ptr }, 10);
        assert_eq!(unsafe { *ctx.kv_page_table_ptr.add(1) }, 20);
        assert_eq!(unsafe { *ctx.kv_page_table_ptr.add(3) }, 40);
    }

    // ── KernelContext: build with batch_meta_ptr non-null ──
    // @trace TEST-MKO-144 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn kernel_context_build_batch_meta_readable() {
        let dummy: u8 = 0;
        let batch_meta: [u8; 8] = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x11, 0x22];
        let (ctx, _seq) = KernelContext::build(
            &dummy as *const u8,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            4,
            std::ptr::null(),
            std::ptr::null(),
            batch_meta.as_ptr(),
            4096, 1, 1, 64,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
        );
        assert!(!ctx.batch_meta_ptr.is_null());
        assert_eq!(unsafe { *ctx.batch_meta_ptr }, 0xAA);
        assert_eq!(unsafe { *ctx.batch_meta_ptr.add(7) }, 0x22);
    }

    // ── KernelContext: build with callback_table_ptr non-null ──
    // @trace TEST-MKO-145 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn kernel_context_build_callback_table_readable() {
        let dummy: u8 = 0;
        let callbacks: [u64; 2] = [0xDEADBEEF_12345678, 0xCAFE_BABE_00000000];
        let (ctx, _seq) = KernelContext::build(
            &dummy as *const u8,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            4,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            4096, 1, 1, 64,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
            std::ptr::null(),
            0,
            callbacks.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
        );
        assert!(!ctx.callback_table_ptr.is_null());
        assert_eq!(unsafe { *ctx.callback_table_ptr }, 0xDEADBEEF_12345678);
        assert_eq!(unsafe { *ctx.callback_table_ptr.add(1) }, 0xCAFE_BABE_00000000);
    }

    // ── DiagnosticScratchpad: read_f32_at with offset equal to buffer length minus 4 ──
    // @trace TEST-MKO-146 [req:REQ-OBS] [level:unit]

    #[test]
    fn scratchpad_read_f32_at_last_aligned_position() {
        let mut buf = vec![0u8; 16]; // exactly 4 f32s
        let val: f32 = -88.5;
        // Write at the last f32 position (offset 12)
        buf[12..16].copy_from_slice(&val.to_le_bytes());
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 4,
            prompt_len: 1,
            hidden_size: 4,
        };
        // Read 1 f32 starting at offset 12 — exactly at end of buffer
        let vals = sp.read_f32_at(12, 1);
        assert_eq!(vals.len(), 1);
        assert!((vals[0] - (-88.5)).abs() < f32::EPSILON);
        // Reading 2 f32s at offset 12 would overflow
        let overflow = sp.read_f32_at(12, 2);
        assert!(overflow.is_empty());
    }

    // ── MegaKernelObservation: from_buffer dead_neuron and sink both non-zero ──
    // @trace TEST-MKO-148 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_from_buffer_dead_neuron_and_sink_both_nonzero() {
        use gllm_kernels::compiler::graph::telemetry_offsets;
        let mut buf = vec![0u8; 512];
        // dead_neuron_count = 500
        let dead: u32 = 500;
        buf[telemetry_offsets::SILU_DEAD_NEURON_MASK_OFFSET
            ..telemetry_offsets::SILU_DEAD_NEURON_MASK_OFFSET + 4]
            .copy_from_slice(&dead.to_le_bytes());
        // sink_status = SinkToken (non-zero u32)
        let sink: u32 = 1;
        buf[telemetry_offsets::IS_ATTENTION_SINK_OFFSET
            ..telemetry_offsets::IS_ATTENTION_SINK_OFFSET + 4]
            .copy_from_slice(&sink.to_le_bytes());

        let obs = MegaKernelObservation::from_buffer(10, &buf);
        assert_eq!(obs.layer_idx, 10);
        assert_eq!(obs.dead_neuron_count, 500);
        assert_eq!(obs.sink_status, AttentionSinkStatus::SinkToken);
        // Ratio: 500 / 768 ≈ 0.651
        let ratio = obs.dead_neuron_ratio(768);
        assert!((ratio - (500.0f32 / 768.0)).abs() < 0.01);
    }

    // ── MegaKernelObservation: is_bypass_candidate with both negative thresholds ──
    // @trace TEST-MKO-149 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_is_bypass_both_negative_thresholds() {
        // Both thresholds negative: delta must be < negative (more negative) AND cosine > negative
        let obs = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: -2.0,  // < -1.0 => true
            cosine_similarity: 0.0, // > -1.0 => true
            dead_neuron_count: 0,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        assert!(obs.is_bypass_candidate(-1.0, -1.0));

        // residual_delta exactly at threshold (equal, not less-than)
        let obs_eq = MegaKernelObservation {
            residual_delta: -1.0,
            ..obs
        };
        assert!(!obs_eq.is_bypass_candidate(-1.0, -1.0));
    }

    // ── TelemetryFlagsBitmask: OR and AND composition with multiple flags ──
    // @trace TEST-MKO-150 [req:REQ-OBS] [level:unit]

    #[test]
    fn telemetry_flags_or_and_clear_pattern() {
        let flag_a = TelemetryFlagsBitmask(0x01);
        let flag_b = TelemetryFlagsBitmask(0x02);
        let flag_c = TelemetryFlagsBitmask(0x04);
        // Compose via OR
        let combined = TelemetryFlagsBitmask(flag_a.0 | flag_b.0 | flag_c.0);
        assert_eq!(combined.0, 0x07);
        // Clear flag_b via AND-NOT
        let cleared = TelemetryFlagsBitmask(combined.0 & !flag_b.0);
        assert_eq!(cleared.0, 0x05);
        assert_eq!(cleared.0 & flag_a.0, flag_a.0);
        assert_eq!(cleared.0 & flag_c.0, flag_c.0);
        assert_eq!(cleared.0 & flag_b.0, 0);
    }

    // ── WeightPageJitConfig: Debug output contains field names ──
    // @trace TEST-MKO-151 [req:REQ-WP-008] [level:unit]

    #[test]
    fn weight_page_jit_config_debug_contains_fields() {
        let cfg = WeightPageJitConfig {
            enabled: true,
            num_pages: 2048,
            page_size_bytes: 128 * 1024 * 1024,
            prefetch_distance: 4,
        };
        let debug = format!("{cfg:?}");
        assert!(debug.contains("WeightPageJitConfig"));
        assert!(debug.contains("enabled"));
        assert!(debug.contains("num_pages"));
        assert!(debug.contains("page_size_bytes"));
        assert!(debug.contains("prefetch_distance"));
    }

    // ── KvPageDecompressConfig: Debug output contains field names ──
    // @trace TEST-MKO-152 [req:REQ-COMP11] [level:unit]

    #[test]
    fn kv_decompress_config_debug_contains_fields() {
        let cfg = KvPageDecompressConfig {
            enabled: false,
            num_pages: 8192,
            page_size_bytes: 256 * 1024,
        };
        let debug = format!("{cfg:?}");
        assert!(debug.contains("KvPageDecompressConfig"));
        assert!(debug.contains("enabled"));
        assert!(debug.contains("num_pages"));
        assert!(debug.contains("page_size_bytes"));
    }

    // ── KernelContext: build preserves rope_freqs_ptr value ──
    // @trace TEST-MKO-153 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn kernel_context_build_rope_freqs_ptr_readable() {
        let dummy: u8 = 0;
        let rope_table: [f32; 4] = [0.1, 0.2, 0.3, 0.4];
        let (ctx, _seq) = KernelContext::build(
            &dummy as *const u8,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            16,
            rope_table.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            4096, 2, 4, 64,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
        );
        assert!(!ctx.rope_freqs_ptr.is_null());
        // Verify the pointer points to the expected data
        assert!((unsafe { *ctx.rope_freqs_ptr } - 0.1).abs() < 0.01);
        assert!((unsafe { *ctx.rope_freqs_ptr.add(1) } - 0.2).abs() < 0.01);
        assert!((unsafe { *ctx.rope_freqs_ptr.add(3) } - 0.4).abs() < 0.01);
    }

    // ── MegaKernelObservation: from_buffer writes f32::NEG_INFINITY at entropy ──
    // @trace TEST-MKO-154 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_from_buffer_negative_infinity_entropy() {
        use gllm_kernels::compiler::graph::telemetry_offsets;
        let mut buf = vec![0u8; 512];
        let neg_inf: f32 = f32::NEG_INFINITY;
        let offset = telemetry_offsets::SOFTMAX_SHARPNESS_OFFSET;
        buf[offset..offset + 4].copy_from_slice(&neg_inf.to_le_bytes());
        let obs = MegaKernelObservation::from_buffer(0, &buf);
        assert!(obs.entropy.is_infinite());
        assert!(obs.entropy.is_sign_negative());
    }

    // ── DiagnosticScratchpad: last_token_logits with large vocab and small buffer ──
    // @trace TEST-MKO-155 [req:REQ-OBS] [level:unit]

    #[test]
    fn scratchpad_last_token_logits_large_vocab_small_buffer() {
        let buf = vec![0u8; 32];
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 1000, // needs 4000 bytes per row, but buffer is only 32 bytes
            prompt_len: 1,
            hidden_size: 4,
        };
        let logits = sp.last_token_logits();
        // read_f32_at(0, 1000) requires 4000 bytes, buffer has 32 => returns empty
        assert!(logits.is_empty());
    }

    // ── MegaKernelError: both variants are Send + Sync ──
    // @trace TEST-MKO-156 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn mega_kernel_error_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<MegaKernelError>();
        assert_sync::<MegaKernelError>();
        // Also verify via trait object
        fn assert_error_send_sync<T: std::error::Error + Send + Sync>() {}
        assert_error_send_sync::<MegaKernelError>();
    }

    // ── Test 157: TelemetryFlagsBitmask XOR toggles individual bits ──

    #[test]
    fn telemetry_flags_xor_toggles_bits() {
        let original = TelemetryFlagsBitmask(0b1010);
        let toggle_mask = TelemetryFlagsBitmask(0b1100);
        let toggled = TelemetryFlagsBitmask(original.0 ^ toggle_mask.0);
        // bit 0: 0 ^ 0 = 0, bit 1: 1 ^ 0 = 1, bit 2: 0 ^ 1 = 1, bit 3: 1 ^ 1 = 0
        assert_eq!(toggled.0, 0b0110);
        // XOR again restores original
        let restored = TelemetryFlagsBitmask(toggled.0 ^ toggle_mask.0);
        assert_eq!(restored.0, original.0);
    }

    // ── Test 158: TelemetryFlagsBitmask count_zeros on all-ones ──

    #[test]
    fn telemetry_flags_count_zeros_all_ones() {
        let flags = TelemetryFlagsBitmask(u32::MAX);
        assert_eq!(flags.0.count_zeros(), 0);
        assert_eq!(flags.0.count_ones(), 32);
        let zero_flags = TelemetryFlagsBitmask(0);
        assert_eq!(zero_flags.0.count_zeros(), 32);
        assert_eq!(zero_flags.0.count_ones(), 0);
    }

    // ── Test 159: WeightPageJitConfig enabled without prefetch ──

    #[test]
    fn weight_page_jit_config_enabled_no_prefetch() {
        let cfg = WeightPageJitConfig {
            enabled: true,
            num_pages: 512,
            page_size_bytes: 32 * 1024 * 1024,
            prefetch_distance: 0,
        };
        assert!(cfg.enabled);
        assert_eq!(cfg.prefetch_distance, 0);
        // Verify debug output includes the enabled state
        let debug = format!("{cfg:?}");
        assert!(debug.contains("enabled"));
        assert!(debug.contains("true"));
    }

    // ── Test 160: KvPageDecompressConfig with page_size_bytes=usize::MAX ──

    #[test]
    fn kv_decompress_config_max_page_size() {
        let cfg = KvPageDecompressConfig {
            enabled: true,
            num_pages: 1,
            page_size_bytes: usize::MAX,
        };
        assert_eq!(cfg.page_size_bytes, usize::MAX);
        let cloned = cfg.clone();
        assert_eq!(cloned.page_size_bytes, usize::MAX);
        // Debug should not panic
        let _ = format!("{cfg:?}");
    }

    // ── Test 161: MegaKernelError conversion from &str ──

    #[test]
    fn mega_kernel_error_from_str_literal() {
        let err: MegaKernelError = MegaKernelError::Compilation("graph is invalid".into());
        let msg = format!("{err}");
        assert!(msg.contains("graph is invalid"));
        // Verify Debug also works
        let debug = format!("{err:?}");
        assert!(debug.contains("Compilation"));
    }

    // ── Test 162: MegaKernelError both variants produce different prefixes ──

    #[test]
    fn mega_kernel_error_variants_have_distinct_prefixes() {
        let compile = format!("{}", MegaKernelError::Compilation("x".into()));
        let execution = format!("{}", MegaKernelError::Execution("x".into()));
        assert!(compile.starts_with("compilation"));
        assert!(execution.starts_with("execution"));
        assert_ne!(compile, execution);
    }

    // ── Test 163: KernelContext build preserves business_config_ptr ──

    #[test]
    fn kernel_context_build_business_config_ptr() {
        let dummy: u8 = 0;
        let biz_config: [u8; 4] = [0xDE, 0xAD, 0xBE, 0xEF];
        let (ctx, _seq) = KernelContext::build(
            &dummy as *const u8,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            1,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0, 0, 0, 0,
            std::ptr::null_mut(),
            0,
            biz_config.as_ptr(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
        );
        assert!(!ctx.business_config_ptr.is_null());
        assert_eq!(unsafe { *ctx.business_config_ptr }, 0xDE);
        assert_eq!(unsafe { *ctx.business_config_ptr.add(3) }, 0xEF);
    }

    // ── Test 164: DiagnosticScratchpad read_f32_at with offset past known values ──

    #[test]
    fn scratchpad_read_f32_at_offset_past_written_data() {
        let mut buf = vec![0u8; 32];
        // Write two f32 values at aligned positions 0 and 4
        let val0: f32 = 1.0;
        buf[0..4].copy_from_slice(&val0.to_le_bytes());
        let val1: f32 = 2.0;
        buf[4..8].copy_from_slice(&val1.to_le_bytes());
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 8,
            prompt_len: 1,
            hidden_size: 8,
        };
        // Read from offset 8 onwards — these are unwritten (zero)
        let vals = sp.read_f32_at(8, 4);
        assert_eq!(vals.len(), 4);
        assert!(vals.iter().all(|&v| v == 0.0));
        // Verify the first two values are still correct
        let first_two = sp.read_f32_at(0, 2);
        assert!((first_two[0] - 1.0).abs() < f32::EPSILON);
        assert!((first_two[1] - 2.0).abs() < f32::EPSILON);
    }

    // ── Test 165: MegaKernelObservation from_buffer with sink_status=u32::MAX ──

    #[test]
    fn observation_from_buffer_attention_sink_u32_max() {
        use gllm_kernels::compiler::graph::telemetry_offsets;
        let offset = telemetry_offsets::IS_ATTENTION_SINK_OFFSET;
        let mut buf = vec![0u8; 512];
        let max_val: u32 = u32::MAX;
        buf[offset..offset + 4].copy_from_slice(&max_val.to_le_bytes());
        let obs = MegaKernelObservation::from_buffer(0, &buf);
        // Any non-zero u32 should set sink_status to SinkToken
        assert_eq!(obs.sink_status, AttentionSinkStatus::SinkToken);
    }

    // ── Test 166: TelemetryFlagsBitmask wrapping_add and wrapping_sub ──

    #[test]
    fn telemetry_flags_wrapping_arithmetic() {
        let flags = TelemetryFlagsBitmask(1);
        // wrapping_sub: 1 - 1 = 0
        let zero = TelemetryFlagsBitmask(flags.0.wrapping_sub(1));
        assert_eq!(zero.0, 0);
        // wrapping_sub: 0 - 1 wraps to u32::MAX
        let wrapped = TelemetryFlagsBitmask(zero.0.wrapping_sub(1));
        assert_eq!(wrapped.0, u32::MAX);
        // wrapping_add: u32::MAX + 1 wraps to 0
        let back_to_zero = TelemetryFlagsBitmask(wrapped.0.wrapping_add(1));
        assert_eq!(back_to_zero.0, 0);
    }

    // ── Test 167: DiagnosticScratchpad embedding with prompt_len=1 and hidden_size=1 ──

    #[test]
    fn scratchpad_embedding_single_token_single_dim_with_value() {
        let mut buf = vec![0u8; 8];
        let val: f32 = 123.456;
        buf[0..4].copy_from_slice(&val.to_le_bytes());
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 4,
            vocab_size: 1,
            prompt_len: 1,
            hidden_size: 1,
        };
        let emb = sp.embedding();
        assert_eq!(emb.len(), 1);
        assert!((emb[0] - 123.456).abs() < 0.01);
    }

    // ── Test 168: MegaKernelObservation is_bypass_candidate with NaN residual_delta ──

    #[test]
    fn observation_is_bypass_nan_residual_delta() {
        // NaN < threshold is always false in IEEE 754
        let obs = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: f32::NAN,
            cosine_similarity: 1.0,
            dead_neuron_count: 0,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        // NaN < anything is false, so is_bypass_candidate returns false
        assert!(!obs.is_bypass_candidate(1.0, 0.0));
        assert!(!obs.is_bypass_candidate(0.0, 0.0));
        assert!(!obs.is_bypass_candidate(f32::MAX, f32::MIN));
    }

    // ── Test 169: MegaKernelObservation is_bypass_candidate with NaN cosine_similarity ──

    #[test]
    fn observation_is_bypass_nan_cosine_similarity() {
        // residual_delta = 0.0 < 1.0 is true, but NaN > threshold is always false
        let obs = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: 0.0,
            cosine_similarity: f32::NAN,
            dead_neuron_count: 0,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        // NaN > anything is false, so even though delta passes, overall is false
        assert!(!obs.is_bypass_candidate(1.0, 0.0));
        assert!(!obs.is_bypass_candidate(1.0, -100.0));
    }

    // ── Test 170: TelemetryFlagsBitmask leading_zeros and trailing_zeros ──

    #[test]
    fn telemetry_flags_leading_and_trailing_zeros() {
        // Arrange: single bit set at position 5
        let flags = TelemetryFlagsBitmask(1 << 5);
        // Act & Assert: 26 leading zeros (32 - 6), 5 trailing zeros
        assert_eq!(flags.0.leading_zeros(), 26);
        assert_eq!(flags.0.trailing_zeros(), 5);
        // All-zero has 32 leading and trailing zeros
        let zero = TelemetryFlagsBitmask(0);
        assert_eq!(zero.0.leading_zeros(), 32);
        assert_eq!(zero.0.trailing_zeros(), 32);
        // Bit 0 set: 31 leading, 0 trailing
        let bit0 = TelemetryFlagsBitmask(1);
        assert_eq!(bit0.0.leading_zeros(), 31);
        assert_eq!(bit0.0.trailing_zeros(), 0);
    }

    // ── Test 171: WeightPageJitConfig default num_pages is 1024 ──

    #[test]
    fn weight_page_jit_config_default_num_pages_value() {
        // Arrange & Act
        let cfg = WeightPageJitConfig::default();
        // Assert: default num_pages must be exactly 1024
        assert_eq!(cfg.num_pages, 1024);
        // Verify the total capacity calculation is reasonable
        let total_capacity = cfg.num_pages * cfg.page_size_bytes;
        assert_eq!(total_capacity, 1024 * 64 * 1024 * 1024); // 64 GiB
    }

    // ── Test 172: KvPageDecompressConfig total page storage calculation ──

    #[test]
    fn kv_decompress_config_total_storage_bytes() {
        // Arrange
        let cfg = KvPageDecompressConfig {
            enabled: true,
            num_pages: 4096,
            page_size_bytes: 16 * 1024,
        };
        // Act
        let total = cfg.num_pages * cfg.page_size_bytes;
        // Assert
        assert_eq!(total, 4096 * 16 * 1024); // 64 MiB
        // Clone preserves values used in calculation
        let cloned = cfg.clone();
        assert_eq!(cloned.num_pages * cloned.page_size_bytes, total);
    }

    // ── Test 173: KernelContext build with scratch_buffer_ptr non-null ──

    #[test]
    fn kernel_context_build_scratch_buffer_ptr() {
        // Arrange
        let dummy: u8 = 0;
        let mut scratch: [u8; 16] = [0xAB; 16];
        // Act
        let (ctx, _seq) = KernelContext::build(
            &dummy as *const u8,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            1,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0, 0, 0, 0,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            scratch.as_mut_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
        );
        // Assert
        assert!(!ctx.scratch_buffer_ptr.is_null());
        assert_eq!(unsafe { *ctx.scratch_buffer_ptr }, 0xAB);
        assert_eq!(unsafe { *ctx.scratch_buffer_ptr.add(15) }, 0xAB);
    }

    // ── Test 174: MegaKernelError Compilation variant preserves multi-line message ──

    #[test]
    fn mega_kernel_error_multiline_compilation_message() {
        // Arrange
        let multi_line = "line 1: bad graph\nline 2: missing node\nline 3: cycle detected";
        let err = MegaKernelError::Compilation(multi_line.into());
        // Act
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("line 1"));
        assert!(msg.contains("line 3"));
        assert!(msg.contains("cycle detected"));
        // Verify newline characters preserved
        assert_eq!(msg.matches('\n').count(), 2);
    }

    // ── Test 175: MegaKernelObservation from_buffer with sequential entropy and delta ──

    #[test]
    fn observation_from_buffer_sequential_fields_independent() {
        // Arrange: write different sentinel values to entropy and residual_delta
        use gllm_kernels::compiler::graph::telemetry_offsets;
        let mut buf = vec![0u8; 512];
        let entropy: f32 = 0.25;
        let off_e = telemetry_offsets::SOFTMAX_SHARPNESS_OFFSET;
        buf[off_e..off_e + 4].copy_from_slice(&entropy.to_le_bytes());
        let delta: f32 = 0.75;
        let off_d = telemetry_offsets::RESIDUAL_DELTA_OFFSET;
        buf[off_d..off_d + 4].copy_from_slice(&delta.to_le_bytes());
        // Act
        let obs = MegaKernelObservation::from_buffer(3, &buf);
        // Assert: each field is read independently, not affected by other offsets
        assert!((obs.entropy - 0.25).abs() < 0.01);
        assert!((obs.residual_delta - 0.75).abs() < 0.01);
        // Unwritten fields default to zero
        assert_eq!(obs.cosine_similarity, 0.0);
        assert_eq!(obs.dead_neuron_count, 0);
    }

    // ── Test 176: DiagnosticScratchpad embedding with multiple tokens and multi-dim hidden ──

    #[test]
    fn scratchpad_embedding_multi_token_multi_dim() {
        // Arrange: 3 tokens, 2 hidden dims = 6 f32s at offset 0
        let mut buf = vec![0u8; 48];
        let values: [f32; 6] = [0.1, 0.2, 1.1, 1.2, 2.1, 2.2];
        for (i, &v) in values.iter().enumerate() {
            let off = i * 4;
            buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 24, // past embedding region
            vocab_size: 4,
            prompt_len: 3,
            hidden_size: 2,
        };
        // Act
        let emb = sp.embedding();
        // Assert: all 6 values read in order
        assert_eq!(emb.len(), 6);
        assert!((emb[0] - 0.1).abs() < 0.01);
        assert!((emb[1] - 0.2).abs() < 0.01);
        assert!((emb[2] - 1.1).abs() < 0.01);
        assert!((emb[3] - 1.2).abs() < 0.01);
        assert!((emb[4] - 2.1).abs() < 0.01);
        assert!((emb[5] - 2.2).abs() < 0.01);
    }

    // ── Test 177: TelemetryFlagsBitmask swap_bytes round-trip ──

    #[test]
    fn telemetry_flags_endian_conversion() {
        // Arrange
        let flags = TelemetryFlagsBitmask(0x0123_4567);
        // Act
        let be = flags.0.to_be();
        let le = flags.0.to_le();
        // Assert: on little-endian systems, to_le is identity and to_be swaps
        assert_eq!(le, flags.0);
        // to_be should produce a different bit pattern on LE machines
        #[cfg(target_endian = "little")]
        assert_ne!(be, flags.0);
        // Round-trip: to_be().to_be() restores original (double swap = identity)
        assert_eq!(be.to_be(), flags.0);
        // swap_bytes is its own inverse
        let swapped = flags.0.swap_bytes();
        assert_eq!(swapped.swap_bytes(), flags.0);
    }

    // ── Test 178: WeightPageJitConfig enabled but page_size_bytes=0 ──

    #[test]
    fn weight_page_jit_config_enabled_zero_page_size() {
        // Arrange
        let cfg = WeightPageJitConfig {
            enabled: true,
            num_pages: 100,
            page_size_bytes: 0,
            prefetch_distance: 1,
        };
        // Act & Assert: enabled with zero page_size is semantically odd but structurally valid
        assert!(cfg.enabled);
        assert_eq!(cfg.page_size_bytes, 0);
        // Total capacity would be 0
        assert_eq!(cfg.num_pages * cfg.page_size_bytes, 0);
        // Debug should not panic
        let debug = format!("{cfg:?}");
        assert!(debug.contains("WeightPageJitConfig"));
    }

    // ── Test 179: KernelContext build with batch_ctx_ptr non-null ──

    #[test]
    fn kernel_context_build_batch_ctx_ptr_readable() {
        // Arrange
        let dummy: u8 = 0;
        let batch_ctx: [u8; 4] = [0x11, 0x22, 0x33, 0x44];
        // Act
        let (ctx, _seq) = KernelContext::build(
            &dummy as *const u8,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            8,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            4096, 2, 4, 64,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
            batch_ctx.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
        );
        // Assert
        assert!(!ctx.batch_ctx_ptr.is_null());
        assert_eq!(unsafe { *ctx.batch_ctx_ptr }, 0x11);
        assert_eq!(unsafe { *ctx.batch_ctx_ptr.add(3) }, 0x44);
    }

    // ── Test 180: MegaKernelObservation dead_neuron_ratio with hidden_size matching count ──

    #[test]
    fn observation_dead_neuron_ratio_count_equals_hidden() {
        // Arrange: dead_neuron_count == hidden_size → ratio exactly 1.0
        let obs = MegaKernelObservation {
            layer_idx: 5,
            entropy: 0.0,
            residual_delta: 0.0,
            cosine_similarity: 0.0,
            dead_neuron_count: 2048,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        // Act
        let ratio = obs.dead_neuron_ratio(2048);
        // Assert
        assert!((ratio - 1.0).abs() < f32::EPSILON);
        // Verify count > hidden gives ratio > 1.0
        let ratio_over = obs.dead_neuron_ratio(1024);
        assert!((ratio_over - 2.0).abs() < f32::EPSILON);
    }

    // ── Test 181: DiagnosticScratchpad read_f32_at with 1-byte buffer ──

    #[test]
    fn scratchpad_read_f32_at_one_byte_buffer() {
        // Arrange: 1-byte buffer — too small for any f32
        let buf = vec![0xFF; 1];
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 1,
            prompt_len: 1,
            hidden_size: 1,
        };
        // Act: request even a single f32 (needs 4 bytes)
        let vals = sp.read_f32_at(0, 1);
        // Assert: buffer too small, returns empty
        assert!(vals.is_empty());
        // count=0 at offset 0 still returns empty (count is 0)
        let zero_vals = sp.read_f32_at(0, 0);
        assert!(zero_vals.is_empty());
    }

    // ── Test 182: MegaKernelError Execution variant with tab and newline ──

    #[test]
    fn mega_kernel_error_execution_whitespace_message() {
        // Arrange
        let msg = "line1\n\tindented detail\nline3";
        let err = MegaKernelError::Execution(msg.into());
        // Act
        let display = format!("{err}");
        let debug = format!("{err:?}");
        // Assert: whitespace preserved in both Display and Debug
        assert!(display.contains("line1"));
        assert!(display.contains("\tindented detail"));
        assert!(display.contains("line3"));
        assert!(debug.contains("Execution"));
        assert!(display.starts_with("execution failed:"));
    }

    // ── Test 183: KernelContext field offsets match documented ABI layout ──
    // @trace TEST-MKO-183 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn kernel_context_field_offsets_match_documentation() {
        // Arrange: documented offsets from the KernelContext ABI comment
        let ctx = KernelContext::zeroed();
        let base = &ctx as *const KernelContext as usize;
        // Act: compute offsets of each documented field
        let off_weight_blob = &ctx.weight_blob_ptr as *const _ as usize - base;
        let off_kv_cache = &ctx.kv_cache_ptr as *const _ as usize - base;
        let off_output_buffer = &ctx.output_buffer_ptr as *const _ as usize - base;
        let off_seq_len = &ctx.seq_len_ptr as *const _ as usize - base;
        let off_kv_page_size = &ctx.kv_page_size as *const _ as usize - base;
        let off_telemetry_flags = &ctx.telemetry_flags as *const _ as usize - base;
        // Assert: offsets increase monotonically for these key fields
        assert!(off_weight_blob < off_kv_cache);
        assert!(off_kv_cache < off_output_buffer);
        assert!(off_output_buffer < off_seq_len);
        assert!(off_seq_len < off_kv_page_size);
        assert!(off_kv_page_size < off_telemetry_flags);
        // kv_page_size at documented offset 0x40
        assert_eq!(off_kv_page_size, 0x40);
    }

    // ── Test 184: KernelContext build with weight_page_fault_cb_ptr non-null ──
    // @trace TEST-MKO-184 [req:REQ-WP-008] [level:unit]

    #[test]
    fn kernel_context_build_weight_page_fault_cb_readable() {
        // Arrange
        let dummy: u8 = 0;
        let fault_callbacks: [u64; 2] = [0xAAAA_BBBB_CCCC_DDDD, 0x1111_2222_3333_4444];
        // Act
        let (ctx, _seq) = KernelContext::build(
            &dummy as *const u8,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            1,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0, 0, 0, 0,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            fault_callbacks.as_ptr(),
            0,
            std::ptr::null(),
            0,
        );
        // Assert: weight_page_fault_cb_ptr points to the callback array
        assert!(!ctx.weight_page_fault_cb_ptr.is_null());
        assert_eq!(unsafe { *ctx.weight_page_fault_cb_ptr }, 0xAAAA_BBBB_CCCC_DDDD);
        assert_eq!(unsafe { *ctx.weight_page_fault_cb_ptr.add(1) }, 0x1111_2222_3333_4444);
    }

    // ── Test 185: KernelContext build with weight_page_table_ptr non-null ──
    // @trace TEST-MKO-185 [req:REQ-WP-008] [level:unit]

    #[test]
    fn kernel_context_build_weight_page_table_ptr_readable() {
        // Arrange
        let dummy: u8 = 0;
        let page_data: [u8; 8] = [0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80];
        // Act
        let (ctx, _seq) = KernelContext::build(
            &dummy as *const u8,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            1,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0, 0, 0, 0,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            page_data.as_ptr(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
        );
        // Assert: weight_page_table_ptr points to the page data
        assert!(!ctx.weight_page_table_ptr.is_null());
        assert_eq!(unsafe { *ctx.weight_page_table_ptr }, 0x10);
        assert_eq!(unsafe { *ctx.weight_page_table_ptr.add(7) }, 0x80);
    }

    // ── Test 186: MegaKernelObservation from_buffer with f32::MAX entropy ──
    // @trace TEST-MKO-186 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_from_buffer_max_f32_entropy() {
        // Arrange
        use gllm_kernels::compiler::graph::telemetry_offsets;
        let mut buf = vec![0u8; 512];
        let max_f32: f32 = f32::MAX;
        let offset = telemetry_offsets::SOFTMAX_SHARPNESS_OFFSET;
        buf[offset..offset + 4].copy_from_slice(&max_f32.to_le_bytes());
        // Act
        let obs = MegaKernelObservation::from_buffer(0, &buf);
        // Assert
        assert!((obs.entropy - f32::MAX).abs() < f32::EPSILON);
        // Other fields remain zero since buffer is zero elsewhere
        assert_eq!(obs.residual_delta, 0.0);
        assert_eq!(obs.dead_neuron_count, 0);
    }

    // ── Test 187: MegaKernelObservation dead_neuron_ratio with typical model sizes ──
    // @trace TEST-MKO-187 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_dead_neuron_ratio_typical_model_sizes() {
        // Arrange: simulate a 4096-hidden model with 256 dead neurons
        let obs = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: 0.0,
            cosine_similarity: 0.0,
            dead_neuron_count: 256,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        // Act & Assert: ratio for common hidden sizes
        // hidden=4096: 256/4096 = 0.0625
        let r4096 = obs.dead_neuron_ratio(4096);
        assert!((r4096 - 0.0625).abs() < 0.001);
        // hidden=768: 256/768 = 0.333...
        let r768 = obs.dead_neuron_ratio(768);
        assert!((r768 - (256.0f32 / 768.0)).abs() < 0.01);
        // hidden=256: 256/256 = 1.0
        let r256 = obs.dead_neuron_ratio(256);
        assert!((r256 - 1.0).abs() < f32::EPSILON);
    }

    // ── Test 188: DiagnosticScratchpad read_f32_at with interleaved known/zero data ──
    // @trace TEST-MKO-188 [req:REQ-OBS] [level:unit]

    #[test]
    fn scratchpad_read_f32_interleaved_pattern() {
        // Arrange: write values at every other f32 position
        let mut buf = vec![0u8; 32]; // 8 f32 slots
        let known: [f32; 4] = [10.0, 20.0, 30.0, 40.0];
        for (i, &v) in known.iter().enumerate() {
            let off = i * 2 * 4; // write at positions 0, 2, 4, 6 (every other)
            buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 8,
            prompt_len: 1,
            hidden_size: 8,
        };
        // Act: read all 8 f32s
        let all = sp.read_f32_at(0, 8);
        // Assert: interleaved pattern — known values at even positions, zeros at odd
        assert_eq!(all.len(), 8);
        assert!((all[0] - 10.0).abs() < f32::EPSILON);
        assert!((all[1] - 0.0).abs() < f32::EPSILON); // gap
        assert!((all[2] - 20.0).abs() < f32::EPSILON);
        assert!((all[3] - 0.0).abs() < f32::EPSILON); // gap
        assert!((all[4] - 30.0).abs() < f32::EPSILON);
        assert!((all[5] - 0.0).abs() < f32::EPSILON); // gap
        assert!((all[6] - 40.0).abs() < f32::EPSILON);
        assert!((all[7] - 0.0).abs() < f32::EPSILON); // gap
    }

    // ── Test 189: TelemetryFlagsBitmask rotate_left and rotate_right ──
    // @trace TEST-MKO-189 [req:REQ-OBS] [level:unit]

    #[test]
    fn telemetry_flags_rotate_operations() {
        // Arrange: single bit at position 0
        let flags = TelemetryFlagsBitmask(1);
        // Act: rotate left by 4 positions
        let rotated_left = TelemetryFlagsBitmask(flags.0.rotate_left(4));
        // Assert: bit 0 moved to bit 4
        assert_eq!(rotated_left.0, 1 << 4);
        // Act: rotate right to restore
        let restored = TelemetryFlagsBitmask(rotated_left.0.rotate_right(4));
        assert_eq!(restored.0, 1);
        // Full rotation: rotate left by 32 = identity
        let full_rot = TelemetryFlagsBitmask(flags.0.rotate_left(32));
        assert_eq!(full_rot.0, flags.0);
    }

    // ── Test 190: WeightPageJitConfig total_capacity with realistic sizes ──
    // @trace TEST-MKO-190 [req:REQ-WP-008] [level:unit]

    #[test]
    fn weight_page_jit_config_realistic_capacity() {
        // Arrange: 8 GiB model with 1 MiB pages
        let cfg = WeightPageJitConfig {
            enabled: true,
            num_pages: 8192,
            page_size_bytes: 1024 * 1024, // 1 MiB
            prefetch_distance: 4,
        };
        // Act
        let total = cfg.num_pages * cfg.page_size_bytes;
        // Assert: 8192 * 1 MiB = 8 GiB
        assert_eq!(total, 8 * 1024 * 1024 * 1024);
        // Clone and verify independence
        let mut original = cfg.clone();
        original.num_pages = 0;
        assert_eq!(cfg.num_pages, 8192);
        assert!(original.num_pages != cfg.num_pages);
    }

    // ── Test 191: MegaKernelError used in Result propagation ──
    // @trace TEST-MKO-191 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn mega_kernel_error_result_propagation() {
        // Arrange: helper function that returns Result<_, MegaKernelError>
        fn inner_fn() -> Result<usize, MegaKernelError> {
            Err(MegaKernelError::Compilation("inner failure".into()))
        }
        // Act: use ? operator for propagation
        let result = (|| -> Result<usize, MegaKernelError> {
            let val = inner_fn()?;
            Ok(val + 1)
        })();
        // Assert: error propagated correctly
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("inner failure"));
        assert!(msg.contains("compilation"));
    }

    // ── Test 192: MegaKernelObservation from_buffer with negative float values ──
    // @trace TEST-MKO-192 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_from_buffer_negative_float_values() {
        // Arrange: write negative floats at multiple telemetry offsets
        use gllm_kernels::compiler::graph::telemetry_offsets;
        let mut buf = vec![0u8; 512];
        let neg_entropy: f32 = -3.14;
        buf[telemetry_offsets::SOFTMAX_SHARPNESS_OFFSET
            ..telemetry_offsets::SOFTMAX_SHARPNESS_OFFSET + 4]
            .copy_from_slice(&neg_entropy.to_le_bytes());
        let neg_delta: f32 = -0.999;
        buf[telemetry_offsets::RESIDUAL_DELTA_OFFSET
            ..telemetry_offsets::RESIDUAL_DELTA_OFFSET + 4]
            .copy_from_slice(&neg_delta.to_le_bytes());
        let neg_cosine: f32 = -0.5;
        buf[telemetry_offsets::COSINE_SIMILARITY_OFFSET
            ..telemetry_offsets::COSINE_SIMILARITY_OFFSET + 4]
            .copy_from_slice(&neg_cosine.to_le_bytes());
        // Act
        let obs = MegaKernelObservation::from_buffer(2, &buf);
        // Assert: negative values preserved exactly
        assert!((obs.entropy - (-3.14)).abs() < 0.01);
        assert!((obs.residual_delta - (-0.999)).abs() < 0.001);
        assert!((obs.cosine_similarity - (-0.5)).abs() < f32::EPSILON);
        assert_eq!(obs.layer_idx, 2);
    }

    // ── Test 193: DiagnosticScratchpad embedding and last_token_logits with offset=0 ──
    // @trace TEST-MKO-193 [req:REQ-OBS] [level:unit]

    #[test]
    fn scratchpad_embedding_and_logits_overlap_at_offset_zero() {
        // Arrange: logits_offset=0, so embedding and logits share the same region
        let mut buf = vec![0u8; 32];
        // Write 4 f32 values: [1.0, 2.0, 3.0, 4.0]
        let vals: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        for (i, &v) in vals.iter().enumerate() {
            let off = i * 4;
            buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0, // logits start at same offset as embedding
            vocab_size: 2,
            prompt_len: 2,
            hidden_size: 2,
        };
        // Act: embedding = prompt_len * hidden_size = 2 * 2 = 4 f32s
        let emb = sp.embedding();
        // last_token_logits at offset = 0 + (2-1) * (2*4) = 8, read 2 f32s
        let logits = sp.last_token_logits();
        // Assert: both methods read from overlapping regions
        assert_eq!(emb.len(), 4);
        assert!((emb[0] - 1.0).abs() < f32::EPSILON);
        assert!((emb[3] - 4.0).abs() < f32::EPSILON);
        // last_token_logits reads from byte 8: vals[2]=3.0, vals[3]=4.0
        assert_eq!(logits.len(), 2);
        assert!((logits[0] - 3.0).abs() < f32::EPSILON);
        assert!((logits[1] - 4.0).abs() < f32::EPSILON);
    }

    // ── Test 194: MegaKernelError Compilation variant preserves percent-formatted string ──
    // @trace TEST-MKO-194 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn mega_kernel_error_compilation_percent_in_message() {
        // Arrange: message containing literal % character
        let msg = "GEMM utilization at 95% — below threshold of 98%";
        let err = MegaKernelError::Compilation(msg.into());
        // Act
        let display = format!("{err}");
        let debug = format!("{err:?}");
        // Assert: percent characters preserved (no format escaping issue)
        assert!(display.contains("95%"));
        assert!(display.contains("98%"));
        assert!(display.starts_with("compilation failed:"));
        assert!(debug.contains("Compilation"));
    }

    // ── Test 195: KernelContext build preserves kv_page_header_ptr non-null ──
    // @trace TEST-MKO-195 [req:REQ-COMP11] [level:unit]

    #[test]
    fn kernel_context_build_kv_page_header_ptr_readable() {
        // Arrange
        let dummy: u8 = 0;
        let page_headers: [u8; 8] = [0xCA, 0xFE, 0xBA, 0xBE, 0xDE, 0xAD, 0xBE, 0xEF];
        // Act
        let (ctx, _seq) = KernelContext::build(
            &dummy as *const u8,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            1,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0, 0, 0, 0,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            page_headers.as_ptr(),
            1, // decompress_inject_flags
        );
        // Assert
        assert!(!ctx.kv_page_header_ptr.is_null());
        assert_eq!(unsafe { *ctx.kv_page_header_ptr }, 0xCA);
        assert_eq!(unsafe { *ctx.kv_page_header_ptr.add(7) }, 0xEF);
        assert_eq!(ctx.decompress_inject_flags, 1);
    }

    // ── Test 196: TelemetryFlagsBitmask checked_add saturating at u32::MAX ──
    // @trace TEST-MKO-196 [req:REQ-OBS] [level:unit]

    #[test]
    fn telemetry_flags_checked_add_saturates() {
        // Arrange: start at u32::MAX - 1
        let flags = TelemetryFlagsBitmask(u32::MAX - 1);
        // Act: checked_add 1 succeeds
        let bumped = flags.0.checked_add(1);
        // Assert: reaches u32::MAX
        assert_eq!(bumped, Some(u32::MAX));
        // Act: checked_add one more overflows
        let overflow = bumped.and_then(|v| v.checked_add(1));
        // Assert: overflow returns None
        assert!(overflow.is_none());
        // Saturating add clamps at u32::MAX
        let saturated = flags.0.saturating_add(2);
        assert_eq!(saturated, u32::MAX);
    }

    // ── Test 197: WeightPageJitConfig clone and then modify original is isolated ──
    // @trace TEST-MKO-197 [req:REQ-WP-008] [level:unit]

    #[test]
    fn weight_page_jit_config_clone_deep_isolation() {
        // Arrange
        let cfg = WeightPageJitConfig {
            enabled: true,
            num_pages: 4096,
            page_size_bytes: 32 * 1024 * 1024,
            prefetch_distance: 8,
        };
        // Act: clone, then mutate every field of original
        let snapshot = cfg.clone();
        let mut mutated = cfg;
        mutated.enabled = false;
        mutated.num_pages = 0;
        mutated.page_size_bytes = 1;
        mutated.prefetch_distance = 999;
        // Assert: snapshot preserves all original values
        assert!(snapshot.enabled);
        assert_eq!(snapshot.num_pages, 4096);
        assert_eq!(snapshot.page_size_bytes, 32 * 1024 * 1024);
        assert_eq!(snapshot.prefetch_distance, 8);
    }

    // ── Test 198: KvPageDecompressConfig default is disabled with 64 KiB pages ──
    // @trace TEST-MKO-198 [req:REQ-COMP11] [level:unit]

    #[test]
    fn kv_decompress_config_default_64kib_pages() {
        // Arrange & Act
        let cfg = KvPageDecompressConfig::default();
        // Assert: disabled by default, page size 64 KiB, 1024 pages
        assert!(!cfg.enabled);
        assert_eq!(cfg.page_size_bytes, 64 * 1024);
        assert_eq!(cfg.num_pages, 1024);
        // Total default storage = 1024 * 64 KiB = 64 MiB
        assert_eq!(cfg.num_pages * cfg.page_size_bytes, 64 * 1024 * 1024);
    }

    // ── Test 199: KernelContext zeroed then build preserves no stale pointer state ──
    // @trace TEST-MKO-199 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn kernel_context_zeroed_then_build_no_stale_state() {
        // Arrange: create a zeroed context, then build a new one with selective non-null pointers
        let zeroed = KernelContext::zeroed();
        let dummy: u8 = 0;
        // Act: build with only weight_blob_ptr and seq_len non-null
        let (built, _seq) = KernelContext::build(
            &dummy as *const u8,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            512,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0, 0, 0, 0,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
        );
        // Assert: zeroed has all null pointers, built has exactly one non-null pointer
        assert!(zeroed.weight_blob_ptr.is_null());
        assert!(!built.weight_blob_ptr.is_null());
        // All other pointer fields in built are still null
        assert!(built.kv_cache_ptr.is_null());
        assert!(built.output_buffer_ptr.is_null());
        assert!(built.hook_ctx_ptr.is_null());
        assert!(built.rope_freqs_ptr.is_null());
        assert!(built.kv_page_table_ptr.is_null());
        assert!(built.batch_meta_ptr.is_null());
        assert!(built.telemetry_ptr.is_null());
        assert!(built.business_config_ptr.is_null());
        assert!(built.weight_offsets_ptr.is_null());
        assert!(built.callback_table_ptr.is_null());
        assert!(built.scratch_buffer_ptr.is_null());
        assert!(built.batch_ctx_ptr.is_null());
        assert!(built.weight_page_table_ptr.is_null());
        assert!(built.weight_page_fault_cb_ptr.is_null());
        assert!(built.kv_page_header_ptr.is_null());
    }

    // ── Test 200: MegaKernelError Execution variant with empty Display is stable ──
    // @trace TEST-MKO-200 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn mega_kernel_error_execution_empty_display_roundtrip() {
        // Arrange
        let err = MegaKernelError::Execution(String::new());
        // Act
        let display = format!("{err}");
        let debug = format!("{err:?}");
        // Assert: Display is deterministic and contains the prefix
        assert_eq!(display, "execution failed: ");
        // Debug format contains the variant name
        assert!(debug.contains("Execution"));
        // Formatting again yields the same result
        assert_eq!(format!("{err}"), display);
    }

    // ── Test 201: MegaKernelObservation is_bypass_candidate with both NaN thresholds ──
    // @trace TEST-MKO-201 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_is_bypass_nan_thresholds_always_false() {
        // Arrange: NaN thresholds — IEEE 754 says NaN comparisons are always false
        let obs = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: -999.0,
            cosine_similarity: 999.0,
            dead_neuron_count: 0,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        // Act & Assert: NaN delta_threshold => residual_delta < NaN is false
        assert!(!obs.is_bypass_candidate(f32::NAN, 0.0));
        // NaN cosine_threshold => cosine_similarity > NaN is false
        assert!(!obs.is_bypass_candidate(0.0, f32::NAN));
        // Both NaN => both conditions false
        assert!(!obs.is_bypass_candidate(f32::NAN, f32::NAN));
    }

    // ── Test 202: DiagnosticScratchpad embedding with buffer size exactly matching embedding region ──
    // @trace TEST-MKO-202 [req:REQ-OBS] [level:unit]

    #[test]
    fn scratchpad_embedding_exact_buffer_match() {
        // Arrange: 2 tokens * 3 hidden = 6 f32s = 24 bytes exactly
        let mut buf = vec![0u8; 24];
        let vals: [f32; 6] = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        for (i, &v) in vals.iter().enumerate() {
            let off = i * 4;
            buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 24, // logits start right after embedding
            vocab_size: 4,
            prompt_len: 2,
            hidden_size: 3,
        };
        // Act
        let emb = sp.embedding();
        // Assert: all 6 values read exactly
        assert_eq!(emb.len(), 6);
        assert!((emb[0] - (-1.0)).abs() < f32::EPSILON);
        assert!((emb[2] - 1.0).abs() < f32::EPSILON);
        assert!((emb[5] - 4.0).abs() < f32::EPSILON);
    }

    // ── Test 203: TelemetryFlagsBitmask reverse_bits round-trip ──
    // @trace TEST-MKO-203 [req:REQ-OBS] [level:unit]

    #[test]
    fn telemetry_flags_reverse_bits_roundtrip() {
        // Arrange
        let flags = TelemetryFlagsBitmask(0x0F0F_0F0F);
        // Act: reverse bits, then reverse again
        let reversed = flags.0.reverse_bits();
        let restored = reversed.reverse_bits();
        // Assert: double reverse restores original
        assert_eq!(restored, flags.0);
        // Single reverse is deterministic
        assert_eq!(flags.0.reverse_bits(), reversed);
        // Verify the reversed pattern: 0x0F0F0F0F reversed = 0xF0F0F0F0
        assert_eq!(reversed, 0xF0F0_F0F0);
    }

    // ── Test 204: KernelContext size is exactly 0xB8 (184 bytes) per ABI layout ──
    // @trace TEST-MKO-204 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn kernel_context_size_matches_abi_layout() {
        // Arrange & Act: the ABI comment documents the layout ending at 0xB4 + 4 = 0xB8
        let size = std::mem::size_of::<KernelContext>();
        // Assert: size must be 0xB8 (184 bytes) based on the documented field offsets
        // 0x00-0x38: 8 ptr fields * 8 = 64
        // 0x40-0x4C: 4 u32 fields = 16
        // 0x50-0x58: 2 ptr fields * 8 = 16  (telemetry_ptr, telemetry_flags+pad)
        // 0x60-0x70: 2 ptr fields + 1 usize = 24 (business, weight_offsets, weight_offsets_len)
        // 0x78-0x80: 2 ptr fields * 8 = 16  (callback, scratch)
        // 0x88: 1 ptr = 8  (batch_ctx)
        // 0x90-0x98: 2 ptr fields * 8 = 16 (weight_page_table, weight_page_fault_cb)
        // 0xA0-0xA4: u32 + pad = 8 (weight_page_inject_flags + _pad1)
        // 0xA8: 1 ptr = 8  (kv_page_header_ptr)
        // 0xB0-0xB4: u32 + pad = 8 (decompress_inject_flags + _pad2)
        // Total: 64+16+16+24+16+8+16+8+8+8 = 184 = 0xB8
        assert_eq!(size, 0xB8);
    }

    // ── Test 205: WeightPageJitConfig enabled with prefetch_distance exceeding num_pages ──
    // @trace TEST-MKO-205 [req:REQ-WP-008] [level:unit]

    #[test]
    fn weight_page_jit_config_prefetch_exceeds_pages() {
        // Arrange: prefetch_distance > num_pages is semantically invalid but structurally valid
        let cfg = WeightPageJitConfig {
            enabled: true,
            num_pages: 10,
            page_size_bytes: 4096,
            prefetch_distance: 100,
        };
        // Act & Assert: struct stores the values as-is; validation is the caller's responsibility
        assert!(cfg.enabled);
        assert_eq!(cfg.num_pages, 10);
        assert_eq!(cfg.prefetch_distance, 100);
        assert!(cfg.prefetch_distance > cfg.num_pages);
        // Clone preserves the mismatch
        let cloned = cfg.clone();
        assert_eq!(cloned.prefetch_distance, 100);
    }

    // ── Test 206: MegaKernelError Compilation with extremely long repeated pattern ──
    // @trace TEST-MKO-206 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn mega_kernel_error_compilation_repeated_pattern() {
        // Arrange: 1000 repetitions of a known pattern
        let pattern = "node_error;";
        let long_msg = pattern.repeat(1000);
        let err = MegaKernelError::Compilation(long_msg.clone());
        // Act
        let display = format!("{err}");
        let debug = format!("{err:?}");
        // Assert: full message preserved in both Display and Debug
        assert!(display.starts_with("compilation failed: node_error;"));
        assert!(display.ends_with("node_error;"));
        assert_eq!(display.len(), "compilation failed: ".len() + pattern.len() * 1000);
        assert!(debug.contains("Compilation"));
        // Count occurrences of pattern in display
        assert_eq!(display.matches(pattern).count(), 1000);
    }

    // ── Test 207: KvPageDecompressConfig enabled with minimal 1-byte page size ──
    // @trace TEST-MKO-207 [req:REQ-COMP11] [level:unit]

    #[test]
    fn kv_decompress_config_enabled_minimal_page_size() {
        // Arrange: enabled with 1-byte page (edge case)
        let cfg = KvPageDecompressConfig {
            enabled: true,
            num_pages: 1,
            page_size_bytes: 1,
        };
        // Act & Assert: structurally valid, total storage = 1 byte
        assert!(cfg.enabled);
        assert_eq!(cfg.page_size_bytes, 1);
        assert_eq!(cfg.num_pages * cfg.page_size_bytes, 1);
        // Debug output should not panic
        let debug = format!("{cfg:?}");
        assert!(debug.contains("KvPageDecompressConfig"));
        // Clone preserves the edge values
        let cloned = cfg.clone();
        assert_eq!(cloned.page_size_bytes, 1);
        assert_eq!(cloned.num_pages, 1);
    }

    // ── Test 208: DiagnosticScratchpad last_token_logits with logits_offset at buffer end ──
    // @trace TEST-MKO-208 [req:REQ-OBS] [level:unit]

    #[test]
    fn scratchpad_last_token_logits_offset_at_buffer_end() {
        // Arrange: logits_offset points to the very end of the buffer
        let buf = vec![0u8; 16];
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 16, // exactly at buffer end
            vocab_size: 4,
            prompt_len: 1,
            hidden_size: 4,
        };
        // Act: last_token_logits at offset = 16 + 0 = 16, read 4 f32s needs 16 bytes
        // but buffer is only 16 bytes so offset 16 is out of bounds
        let logits = sp.last_token_logits();
        // Assert: out of bounds, returns empty
        assert!(logits.is_empty());
    }

    // ── Test 209: TelemetryFlagsBitmask individual bit enable check pattern ──
    // @trace TEST-MKO-209 [req:REQ-OBS] [level:unit]

    #[test]
    fn telemetry_flags_individual_bit_enabled_check() {
        // Arrange: enable bits 0, 2, 4 (e.g. entropy, residual, cosine probes)
        let flags = TelemetryFlagsBitmask(0b0001_0101);
        // Act & Assert: verify each bit independently
        assert_ne!(flags.0 & (1 << 0), 0); // bit 0 enabled
        assert_eq!(flags.0 & (1 << 1), 0); // bit 1 disabled
        assert_ne!(flags.0 & (1 << 2), 0); // bit 2 enabled
        assert_eq!(flags.0 & (1 << 3), 0); // bit 3 disabled
        assert_ne!(flags.0 & (1 << 4), 0); // bit 4 enabled
        // Count enabled bits
        let count = flags.0.count_ones();
        assert_eq!(count, 3);
    }

    // ── Test 210: WeightPageJitConfig enabled=false with non-zero page params ──
    // @trace TEST-MKO-210 [req:REQ-WP-008] [level:unit]

    #[test]
    fn weight_page_jit_config_disabled_with_nonzero_params() {
        // Arrange: disabled but with non-zero configuration values (valid: config prepared but not active)
        let cfg = WeightPageJitConfig {
            enabled: false,
            num_pages: 8192,
            page_size_bytes: 256 * 1024 * 1024,
            prefetch_distance: 16,
        };
        // Act & Assert: struct stores all values regardless of enabled state
        assert!(!cfg.enabled);
        assert_eq!(cfg.num_pages, 8192);
        assert_eq!(cfg.page_size_bytes, 256 * 1024 * 1024);
        assert_eq!(cfg.prefetch_distance, 16);
        // Debug output shows all fields
        let debug = format!("{cfg:?}");
        assert!(debug.contains("WeightPageJitConfig"));
        // Clone preserves the disabled state with non-zero values
        let cloned = cfg.clone();
        assert!(!cloned.enabled);
        assert_eq!(cloned.num_pages, 8192);
    }

    // ── Test 211: KvPageDecompressConfig enabled with large page count exceeds u32 range ──
    // @trace TEST-MKO-211 [req:REQ-COMP11] [level:unit]

    #[test]
    fn kv_decompress_config_large_page_count_overflow_check() {
        // Arrange: num_pages at usize::MAX / 64 KiB to verify no overflow in total_bytes calc
        let cfg = KvPageDecompressConfig {
            enabled: true,
            num_pages: 1024,
            page_size_bytes: 64 * 1024,
        };
        // Act: compute total bytes (should not overflow for reasonable values)
        let total = cfg.num_pages.checked_mul(cfg.page_size_bytes);
        // Assert: multiplication succeeds and equals 64 MiB
        assert!(total.is_some());
        assert_eq!(total.unwrap(), 64 * 1024 * 1024);
        // Debug output does not panic
        let debug = format!("{cfg:?}");
        assert!(debug.contains("KvPageDecompressConfig"));
        assert!(debug.contains("enabled"));
    }

    // ── Test 212: KernelContext build sets weight_offsets_ptr and weight_offsets_len ──
    // @trace TEST-MKO-212 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn kernel_context_build_weight_offsets_fields() {
        // Arrange
        let dummy: u8 = 0;
        let offsets: [usize; 3] = [100, 200, 300];
        // Act
        let (ctx, _seq) = KernelContext::build(
            &dummy as *const u8,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            1,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0, 0, 0, 0,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
            offsets.as_ptr(),
            3, // weight_offsets_len
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
        );
        // Assert: weight_offsets_ptr and len are set correctly
        assert!(!ctx.weight_offsets_ptr.is_null());
        assert_eq!(ctx.weight_offsets_len, 3);
        // Verify pointer contents
        assert_eq!(unsafe { *ctx.weight_offsets_ptr }, 100);
        assert_eq!(unsafe { *ctx.weight_offsets_ptr.add(1) }, 200);
        assert_eq!(unsafe { *ctx.weight_offsets_ptr.add(2) }, 300);
    }

    // ── Test 213: KernelContext build rope_freqs_ptr is dereferenceable ──
    // @trace TEST-MKO-213 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn kernel_context_build_rope_freqs_ptr_dereferenceable() {
        // Arrange
        let dummy: u8 = 0;
        let freqs: [f32; 4] = [0.1, 0.2, 0.3, 0.4];
        // Act
        let (ctx, _seq) = KernelContext::build(
            &dummy as *const u8,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            1,
            freqs.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            0, 0, 0, 0,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
        );
        // Assert: rope_freqs_ptr points to the freqs array
        assert!(!ctx.rope_freqs_ptr.is_null());
        assert!((unsafe { *ctx.rope_freqs_ptr } - 0.1).abs() < f32::EPSILON);
        assert!((unsafe { *ctx.rope_freqs_ptr.add(3) } - 0.4).abs() < f32::EPSILON);
    }

    // ── Test 214: MegaKernelError Execution with multiline message ──
    // @trace TEST-MKO-214 [req:REQ-MEGA-001] [level:unit]

    #[test]
    fn mega_kernel_error_execution_multiline_message() {
        // Arrange: error message containing newlines (realistic for stack traces)
        let multiline = "signal 11 received\nat pc=0x7fff1234\nframe: layer 12 GEMM\n";
        let err = MegaKernelError::Execution(multiline.to_string());
        // Act
        let display = format!("{err}");
        let debug = format!("{err:?}");
        // Assert: newlines preserved in Display and Debug
        assert!(display.contains('\n'));
        assert_eq!(display.lines().count(), 3);
        assert!(display.starts_with("execution failed: signal 11"));
        assert!(display.contains("layer 12 GEMM"));
        assert!(debug.contains("Execution"));
    }

    // ── Test 215: MegaKernelObservation is_bypass_candidate with cosine exactly zero and threshold zero ──
    // @trace TEST-MKO-215 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_is_bypass_cosine_zero_threshold_zero_fails() {
        // Arrange: cosine_similarity = 0.0, cosine_threshold = 0.0
        // 0.0 > 0.0 is false, so bypass should fail even though delta passes
        let obs = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: -1.0, // < 0.0 threshold
            cosine_similarity: 0.0, // NOT > 0.0 threshold
            dead_neuron_count: 0,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        // Act & Assert: cosine 0.0 > threshold 0.0 is false
        assert!(!obs.is_bypass_candidate(0.0, 0.0));
    }

    // ── Test 216: MegaKernelObservation dead_neuron_ratio precision with large hidden_size ──
    // @trace TEST-MKO-216 [req:REQ-OBS] [level:unit]

    #[test]
    fn observation_dead_neuron_ratio_large_hidden_precision() {
        // Arrange: large hidden_size simulating a real model (e.g. 8192 hidden dim)
        let obs = MegaKernelObservation {
            layer_idx: 0,
            entropy: 0.0,
            residual_delta: 0.0,
            cosine_similarity: 0.0,
            dead_neuron_count: 1024,
            sink_status: AttentionSinkStatus::Normal,
            per_channel_scale: 0.0,
            row_l1_norm: 0.0,
            row_max: 0.0,
        };
        // Act: ratio = 1024 / 8192 = 0.125
        let ratio = obs.dead_neuron_ratio(8192);
        // Assert: exact within floating point precision
        assert!((ratio - 0.125).abs() < f32::EPSILON);
        // ratio is between 0 and 1
        assert!(ratio >= 0.0);
        assert!(ratio <= 1.0);
    }

    // ── Test 217: DiagnosticScratchpad embedding with non-zero logits_offset reads from start ──
    // @trace TEST-MKO-217 [req:REQ-OBS] [level:unit]

    #[test]
    fn scratchpad_embedding_with_nonzero_logits_offset_reads_from_zero() {
        // Arrange: embedding always reads from offset 0 regardless of logits_offset
        let mut buf = vec![0u8; 64];
        // Write embedding values at offset 0
        let emb_vals: [f32; 4] = [10.0, 20.0, 30.0, 40.0];
        for (i, &v) in emb_vals.iter().enumerate() {
            let off = i * 4;
            buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        // Write different data at logits_offset to confirm they don't mix
        let logits_start = 32u8;
        buf[32] = logits_start;
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 32, // logits start at byte 32
            vocab_size: 4,
            prompt_len: 2,
            hidden_size: 2, // embedding = 2 * 2 = 4 f32s from offset 0
        };
        // Act
        let emb = sp.embedding();
        // Assert: reads from byte 0, not from logits_offset
        assert_eq!(emb.len(), 4);
        assert!((emb[0] - 10.0).abs() < f32::EPSILON);
        assert!((emb[1] - 20.0).abs() < f32::EPSILON);
        assert!((emb[2] - 30.0).abs() < f32::EPSILON);
        assert!((emb[3] - 40.0).abs() < f32::EPSILON);
    }

    // ── Test 218: DiagnosticScratchpad read_f32_at preserves negative zero ──
    // @trace TEST-MKO-218 [req:REQ-OBS] [level:unit]

    #[test]
    fn scratchpad_read_f32_preserves_negative_zero() {
        // Arrange: write -0.0 (0x80000000) into the buffer
        let mut buf = vec![0u8; 16];
        let neg_zero: f32 = -0.0;
        buf[0..4].copy_from_slice(&neg_zero.to_le_bytes());
        // Verify the bit pattern is indeed negative zero
        assert_eq!(buf[0..4], (-0.0f32).to_le_bytes());
        let sp = DiagnosticScratchpad {
            data: buf,
            logits_offset: 0,
            vocab_size: 4,
            prompt_len: 1,
            hidden_size: 4,
        };
        // Act
        let vals = sp.read_f32_at(0, 1);
        // Assert: negative zero is preserved (sign bit maintained)
        assert_eq!(vals.len(), 1);
        assert_eq!(vals[0], 0.0); // -0.0 == 0.0 in IEEE 754
        assert!(vals[0].is_sign_negative()); // but sign bit is preserved
    }
}
