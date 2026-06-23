//! TEST-DIST-001~017 — 分布式推理专项测试 (REQ-DIST-001~017)
//!
//! 覆盖 SPEC 43-DISTRIBUTED-IMPLEMENTATION.html 中 REQ-DIST-001~017 的核心逻辑。
//! 优先测试编译时逻辑（枚举/决策/配置），不依赖运行时 NCCL 通信。
//!
//! 运行方式:
//! ```bash
//! # 非 nccl 环境（仅测试非 nccl-gated 逻辑）
//! cargo test --test test_dist -- --test-threads=2
//!
//! # nccl 环境（完整测试）
//! cargo test --test test_dist --features nccl -- --test-threads=2
//! ```

#[path = "dist/mod.rs"]
mod dist;
