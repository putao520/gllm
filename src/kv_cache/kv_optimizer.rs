//! KIVI Key/Val mixed-precision strategy (per SPEC 19-KV-CACHE-OPTIMIZATION.md §3)
//!
//! KIVI asymmetric quantization exploits the fundamentally different outlier
//! distributions of key and value caches:
//!
//! - **K cache**: outliers concentrated in specific *channels* (stable across tokens)
//!   → Per-Channel quantization with higher precision (FP16/FP8)
//! - **V cache**: outliers concentrated in specific *tokens* (stable across channels)
//!   → Per-Token quantization with lower precision (KIVI4/KIVI2)
//!
//! ## Write Path
//! 1. K data → per-channel scale computation → FP16/FP8 write (keep high precision)
//! 2. V data → per-token scale computation → INT4/INT2 write (aggressive compression)
//!
//! ## Read Path
//! 1. K data → direct FP16/FP8 read (no dequant needed)
//! 2. V data → INT4/INT2 dequant + per-token scale → FP32 compute precision
//!
//! ## Integration
//! KiviStrategy is consulted by the KV cache write path to determine the
//! quantization precision for each page. The per-page PrecisionTier in
//! KvPageHeader reflects the *page-level* tier, while KiviStrategy provides
//! the *key vs value* asymmetry within a page.
//!
//! 代码组织 (include! 模式 — 编译为单模块，物理分散到 5 个片段):
//! - `kv_optimizer/kivi_mustafar.inc.rs`  — KIVI + Mustafar 策略 (SPEC §3, §4)
//! - `kv_optimizer/kvtuner_chunk.inc.rs`  — KvTuner + Chunk 策略 (SPEC §5, §6)
//! - `kv_optimizer/decision_matrix.inc.rs` — DecisionVariant + CrossDecisionMatrix + VariantMatrix (SPEC §7)
//! - `kv_optimizer/epilogue_config.inc.rs` — EpilogueSparse + KvOptimization Config/Status (SPEC §8, §9)
//! - `kv_optimizer/tests.inc.rs`          — 测试模块

use super::{KvPageHeader, PrecisionTier};

include!("kv_optimizer/kivi_mustafar.inc.rs");
include!("kv_optimizer/kvtuner_chunk.inc.rs");
include!("kv_optimizer/decision_matrix.inc.rs");
include!("kv_optimizer/epilogue_config.inc.rs");

include!("kv_optimizer/tests.inc.rs");
