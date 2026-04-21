//! SemanticGatekeeperCallback — LayerCallback 实现 (SPEC §2.1 + §7).
//!
//! Phase A 只提供 struct 定义 + 构造器 + 共享状态字段.
//! LayerCallback trait impl 和 Q-tap 读取 / 层级路由 / 稳定性追踪 /
//! 残差注入的完整逻辑见 Phase D (本文件的 impl block).

use std::sync::{Arc, RwLock};

use super::{
    active_state::ActiveState, level_keys::LevelKeysCache, ring_buffer::GatekeeperRingBuffer,
    AstSentinel, KnowledgeProvider,
};

/// Priority for Semantic Gatekeeper in the CallbackChain.
///
/// 高于 RAG Inject(80),低于 Prefetch(100).
/// 按 `SPEC/05-OPTIMIZATIONS.md §8` 执行优先级表.
pub const SEMANTIC_GATEKEEPER_PRIORITY: u32 = 90;

/// SG 运行时主体. 持有配置、预计算缓存、ring buffer、状态机.
///
/// 注册到 `CallbackChain` 后由 `FusedGraphExecutor::run_with_kv_cache_with_callbacks`
/// 在每个检测层 `pre_node` 触发.
pub struct SemanticGatekeeperCallback {
    /// 预计算的层级键. `target_layers()` 返回其 `detection_layers()`.
    pub(super) level_keys: Arc<LevelKeysCache>,

    /// Q-tap ring buffer. `FusedAttentionLayerConfig.q_tap.sink_ptr` 指向此处.
    pub(super) ring_buffer: Arc<GatekeeperRingBuffer>,

    /// 跨 decode step 的活跃注入状态.
    pub(super) active_state: RwLock<ActiveState>,

    /// 用户实现的知识源.
    pub(super) provider: Arc<dyn KnowledgeProvider>,

    /// 可选的 AST 哨兵.
    pub(super) ast_sentinel: Option<Arc<dyn AstSentinel>>,

    /// 门控阈值 τ.
    pub(super) gate_threshold: f32,
    /// 稳定性阈值.
    pub(super) stability_threshold: f32,
    /// 残差相加强度 α.
    pub(super) alpha: f32,

    /// 主模型 hidden_size (用于从 hidden_state 字节切片最后一个 token).
    pub(super) hidden_size: usize,
}

impl SemanticGatekeeperCallback {
    /// 从预计算的 LevelKeysCache + ring buffer + provider 构造.
    ///
    /// 调用方: `Client::register_semantic_gatekeeper` (Phase E).
    pub fn new(
        level_keys: Arc<LevelKeysCache>,
        ring_buffer: Arc<GatekeeperRingBuffer>,
        provider: Arc<dyn KnowledgeProvider>,
        ast_sentinel: Option<Arc<dyn AstSentinel>>,
        gate_threshold: f32,
        stability_threshold: f32,
        alpha: f32,
        hidden_size: usize,
    ) -> Self {
        Self {
            level_keys,
            ring_buffer,
            active_state: RwLock::new(ActiveState::default()),
            provider,
            ast_sentinel,
            gate_threshold,
            stability_threshold,
            alpha,
            hidden_size,
        }
    }

    /// 清空 ActiveState (SPEC §5.3 刷新触发器 3: 用户显式重置).
    ///
    /// 不影响 LevelKeysCache.
    pub fn reset_state(&self) {
        if let Ok(mut guard) = self.active_state.write() {
            guard.clear();
        }
    }

    /// 当前注册的检测层物理索引集合.
    pub fn detection_layers(&self) -> &[usize] {
        self.level_keys.detection_layers()
    }
}
