//! ActiveState — 稳定性追踪状态机 (SPEC §5).
//!
//! 跨 decode step 维护当前活跃的知识注入状态. 若 hidden 与锚点相似度
//! 高于阈值且 AST 节点未变则复用 `v_knowledge`,否则 FullCompute.

use crate::scheduler::types::RequestId;

use super::SemanticLevel;

/// 活跃注入状态 (per-session).
#[derive(Debug, Default, Clone)]
pub struct ActiveState {
    /// 当前活跃的知识层级.
    pub level: Option<SemanticLevel>,
    /// 当前知识条目的文本 hash (避免重复检索同一条目).
    pub key_hash: Option<u64>,
    /// 注入时的 hidden_state 最后 token 向量 (锚点).
    pub anchor_hidden: Option<Vec<f32>>,
    /// 当前注入的知识向量 (confidence 已编码入向量模值).
    pub v_knowledge: Option<Vec<f32>>,
    /// 注入时的 AST 节点 kind (AST 变化强制刷新).
    pub ast_node_kind: Option<String>,
    /// 最近一次更新的 decode step 号.
    pub last_step: u64,
    /// 最近一次更新的请求 ID (跨请求自动刷新).
    pub last_request: Option<RequestId>,
}

impl ActiveState {
    /// 清空所有字段,保留到初始状态.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// 判断是否需要对新请求触发刷新 (SPEC §5.3 刷新触发器 2).
    pub fn needs_request_boundary_refresh(&self, new_request: RequestId) -> bool {
        matches!(self.last_request, Some(r) if r != new_request)
    }
}
