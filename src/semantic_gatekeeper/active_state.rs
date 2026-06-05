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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_all_none() {
        let s = ActiveState::default();
        assert!(s.level.is_none());
        assert!(s.key_hash.is_none());
        assert!(s.anchor_hidden.is_none());
        assert!(s.v_knowledge.is_none());
        assert!(s.ast_node_kind.is_none());
        assert_eq!(s.last_step, 0);
        assert!(s.last_request.is_none());
    }

    #[test]
    fn clear_resets_to_default() {
        let mut s = ActiveState::default();
        s.level = Some(SemanticLevel::L2);
        s.key_hash = Some(42);
        s.last_step = 100;
        s.last_request = Some(1);
        s.clear();
        assert!(s.level.is_none());
        assert!(s.key_hash.is_none());
        assert_eq!(s.last_step, 0);
    }

    #[test]
    fn needs_refresh_on_different_request() {
        let mut s = ActiveState::default();
        s.last_request = Some(1);
        assert!(s.needs_request_boundary_refresh(2));
    }

    #[test]
    fn no_refresh_on_same_request() {
        let mut s = ActiveState::default();
        s.last_request = Some(1);
        assert!(!s.needs_request_boundary_refresh(1));
    }

    #[test]
    fn no_refresh_when_no_prior_request() {
        let s = ActiveState::default();
        assert!(!s.needs_request_boundary_refresh(99));
    }

    #[test]
    fn clone_produces_equal_instance() {
        let mut s = ActiveState::default();
        s.level = Some(SemanticLevel::L3);
        s.key_hash = Some(0xDEADBEEF);
        s.anchor_hidden = Some(vec![1.0, 2.0, 3.0]);
        s.last_step = 42;
        let c = s.clone();
        assert_eq!(c.level, s.level);
        assert_eq!(c.key_hash, s.key_hash);
        assert_eq!(c.anchor_hidden, s.anchor_hidden);
        assert_eq!(c.last_step, s.last_step);
    }

    #[test]
    fn clear_with_empty_anchor_hidden_and_v_knowledge() {
        let mut s = ActiveState::default();
        s.anchor_hidden = Some(vec![]);
        s.v_knowledge = Some(vec![]);
        s.clear();
        assert!(s.anchor_hidden.is_none());
        assert!(s.v_knowledge.is_none());
    }

    #[test]
    fn debug_format_contains_field_names() {
        let s = ActiveState::default();
        let dbg = format!("{:?}", s);
        assert!(dbg.contains("level"), "Debug output missing 'level'");
        assert!(dbg.contains("key_hash"), "Debug output missing 'key_hash'");
        assert!(dbg.contains("last_step"), "Debug output missing 'last_step'");
    }

    #[test]
    fn anchor_hidden_with_special_floats() {
        let mut s = ActiveState::default();
        s.anchor_hidden = Some(vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0f32]);
        let cloned = s.clone();
        assert!(cloned.anchor_hidden.as_ref().unwrap()[0].is_nan());
        assert_eq!(cloned.anchor_hidden.as_ref().unwrap()[1], f32::INFINITY);
        assert_eq!(cloned.anchor_hidden.as_ref().unwrap()[2], f32::NEG_INFINITY);
        assert!(cloned.anchor_hidden.as_ref().unwrap()[3].is_sign_negative());
    }

    #[test]
    fn v_knowledge_with_subnormal_floats() {
        let mut s = ActiveState::default();
        let subnormal = f32::from_bits(1);
        s.v_knowledge = Some(vec![subnormal]);
        assert!(s.v_knowledge.as_ref().unwrap()[0].is_subnormal());
        s.clear();
        assert!(s.v_knowledge.is_none());
    }

    #[test]
    fn needs_refresh_boundary_request_id_zero() {
        let mut s = ActiveState::default();
        s.last_request = Some(0);
        assert!(s.needs_request_boundary_refresh(1));
        assert!(!s.needs_request_boundary_refresh(0));
    }

    #[test]
    fn needs_refresh_max_request_id() {
        let mut s = ActiveState::default();
        s.last_request = Some(u64::MAX);
        assert!(!s.needs_request_boundary_refresh(u64::MAX));
        assert!(s.needs_request_boundary_refresh(u64::MAX - 1));
    }

    #[test]
    fn key_hash_zero_is_distinct_from_none() {
        let mut s = ActiveState::default();
        assert!(s.key_hash.is_none());
        s.key_hash = Some(0);
        assert!(s.key_hash.is_some());
        assert_eq!(s.key_hash, Some(0));
    }

    #[test]
    fn ast_node_kind_empty_string_vs_none() {
        let mut s = ActiveState::default();
        assert!(s.ast_node_kind.is_none());
        s.ast_node_kind = Some(String::new());
        assert!(s.ast_node_kind.is_some());
        assert_eq!(s.ast_node_kind.as_ref().unwrap().len(), 0);
    }

    #[test]
    fn last_step_max_u64() {
        let mut s = ActiveState::default();
        s.last_step = u64::MAX;
        assert_eq!(s.last_step, u64::MAX);
        s.clear();
        assert_eq!(s.last_step, 0);
    }

    #[test]
    fn all_semantic_levels_assignable() {
        for level in SemanticLevel::ORDER {
            let mut s = ActiveState::default();
            s.level = Some(level);
            assert_eq!(s.level, Some(level));
        }
    }

    #[test]
    fn clear_does_not_affect_independent_instance() {
        let mut a = ActiveState::default();
        a.level = Some(SemanticLevel::L1);
        a.last_step = 999;
        let b = a.clone();
        a.clear();
        assert_eq!(b.level, Some(SemanticLevel::L1));
        assert_eq!(b.last_step, 999);
    }

    #[test]
    fn anchor_hidden_large_vector() {
        let mut s = ActiveState::default();
        let large: Vec<f32> = (0..10_000).map(|i| i as f32).collect();
        s.anchor_hidden = Some(large.clone());
        assert_eq!(s.anchor_hidden.as_ref().unwrap().len(), 10_000);
        assert_eq!(s.anchor_hidden.as_ref().unwrap()[0], 0.0);
        assert_eq!(s.anchor_hidden.as_ref().unwrap()[9999], 9999.0);
    }

    #[test]
    fn sequential_clears_are_idempotent() {
        let mut s = ActiveState::default();
        s.level = Some(SemanticLevel::L2);
        s.last_step = 50;
        s.clear();
        let first = s.clone();
        s.clear();
        assert_eq!(format!("{:?}", first), format!("{:?}", s));
    }

    // --- 15 new tests below ---

    #[test]
    fn v_knowledge_preserves_values_after_assign() {
        // Arrange
        let mut s = ActiveState::default();
        let knowledge = vec![0.1, -0.5, 0.9, 1.0];

        // Act
        s.v_knowledge = Some(knowledge.clone());

        // Assert
        assert_eq!(s.v_knowledge.as_ref().unwrap().len(), 4);
        assert_eq!(s.v_knowledge.as_ref().unwrap()[0], 0.1);
        assert_eq!(s.v_knowledge.as_ref().unwrap()[1], -0.5);
        assert_eq!(s.v_knowledge.as_ref().unwrap()[2], 0.9);
        assert_eq!(s.v_knowledge.as_ref().unwrap()[3], 1.0);
    }

    #[test]
    fn all_fields_populated_then_clear() {
        // Arrange
        let mut s = ActiveState::default();
        s.level = Some(SemanticLevel::L1);
        s.key_hash = Some(12345);
        s.anchor_hidden = Some(vec![0.25, 0.75]);
        s.v_knowledge = Some(vec![-1.0, 1.0]);
        s.ast_node_kind = Some("FunctionDecl".to_string());
        s.last_step = 777;
        s.last_request = Some(42);

        // Act
        s.clear();

        // Assert
        assert!(s.level.is_none());
        assert!(s.key_hash.is_none());
        assert!(s.anchor_hidden.is_none());
        assert!(s.v_knowledge.is_none());
        assert!(s.ast_node_kind.is_none());
        assert_eq!(s.last_step, 0);
        assert!(s.last_request.is_none());
    }

    #[test]
    fn needs_refresh_after_clear_then_new_request() {
        // Arrange: set a request, clear, then check against a different request
        let mut s = ActiveState::default();
        s.last_request = Some(10);
        s.clear();

        // Act & Assert
        // After clear, last_request is None, so no refresh needed for any request
        assert!(!s.needs_request_boundary_refresh(10));
        assert!(!s.needs_request_boundary_refresh(20));
    }

    #[test]
    fn last_request_transitions_across_multiple_requests() {
        // Arrange
        let mut s = ActiveState::default();
        s.last_request = Some(1);

        // Act & Assert: transition through several request IDs
        assert!(s.needs_request_boundary_refresh(2));
        s.last_request = Some(2);
        assert!(s.needs_request_boundary_refresh(3));
        assert!(!s.needs_request_boundary_refresh(2));
        s.last_request = Some(3);
        assert!(!s.needs_request_boundary_refresh(3));
    }

    #[test]
    fn clone_independence_after_mutation() {
        // Arrange
        let mut s = ActiveState::default();
        s.level = Some(SemanticLevel::L2);
        s.key_hash = Some(999);
        s.last_step = 100;
        let c = s.clone();

        // Act: mutate original
        s.level = Some(SemanticLevel::L3);
        s.key_hash = Some(0);
        s.last_step = 200;

        // Assert: clone retains original values
        assert_eq!(c.level, Some(SemanticLevel::L2));
        assert_eq!(c.key_hash, Some(999));
        assert_eq!(c.last_step, 100);
    }

    #[test]
    fn anchor_hidden_reassignment_replaces_previous() {
        // Arrange
        let mut s = ActiveState::default();
        s.anchor_hidden = Some(vec![1.0, 2.0, 3.0]);

        // Act: replace with different dimensionality
        s.anchor_hidden = Some(vec![4.0, 5.0]);

        // Assert
        assert_eq!(s.anchor_hidden.as_ref().unwrap().len(), 2);
        assert_eq!(s.anchor_hidden.as_ref().unwrap()[0], 4.0);
        assert_eq!(s.anchor_hidden.as_ref().unwrap()[1], 5.0);
    }

    #[test]
    fn ast_node_kind_various_strings() {
        // Arrange
        let mut s = ActiveState::default();
        let kinds = ["IfStmt", "BinaryExpr", "CallExpr", "VarDecl", "ReturnStmt"];

        // Act & Assert: each string round-trips correctly
        for kind in &kinds {
            s.ast_node_kind = Some(kind.to_string());
            assert_eq!(s.ast_node_kind.as_ref().unwrap(), kind);
        }
    }

    #[test]
    fn debug_output_includes_all_seven_fields() {
        // Arrange
        let s = ActiveState {
            level: Some(SemanticLevel::L3),
            key_hash: Some(42),
            anchor_hidden: Some(vec![1.0]),
            v_knowledge: Some(vec![2.0]),
            ast_node_kind: Some("Test".to_string()),
            last_step: 5,
            last_request: Some(99),
        };

        // Act
        let dbg = format!("{:?}", s);

        // Assert: all seven field names appear
        assert!(dbg.contains("level"), "Debug missing 'level'");
        assert!(dbg.contains("key_hash"), "Debug missing 'key_hash'");
        assert!(dbg.contains("anchor_hidden"), "Debug missing 'anchor_hidden'");
        assert!(dbg.contains("v_knowledge"), "Debug missing 'v_knowledge'");
        assert!(dbg.contains("ast_node_kind"), "Debug missing 'ast_node_kind'");
        assert!(dbg.contains("last_step"), "Debug missing 'last_step'");
        assert!(dbg.contains("last_request"), "Debug missing 'last_request'");
    }

    #[test]
    fn v_knowledge_large_vector_roundtrip() {
        // Arrange
        let mut s = ActiveState::default();
        let large: Vec<f32> = (0..8192).map(|i| (i as f32) * 0.001).collect();

        // Act
        s.v_knowledge = Some(large.clone());
        let cloned = s.clone();

        // Assert
        assert_eq!(cloned.v_knowledge.as_ref().unwrap().len(), 8192);
        assert_eq!(cloned.v_knowledge.as_ref().unwrap()[0], 0.0);
        assert_eq!(
            cloned.v_knowledge.as_ref().unwrap()[8191],
            8191.0f32 * 0.001
        );
    }

    #[test]
    fn key_hash_collision_resistance_semantic() {
        // Arrange: two distinct knowledge entries should have different hashes
        let mut s = ActiveState::default();
        s.key_hash = Some(0xABCD);
        let first = s.clone();
        s.key_hash = Some(0xABCE); // differ by 1
        let second = s.clone();

        // Assert
        assert_ne!(first.key_hash, second.key_hash);
        assert_eq!(first.key_hash, Some(0xABCD));
        assert_eq!(second.key_hash, Some(0xABCE));
    }

    #[test]
    fn level_transition_l1_to_l2_to_l3() {
        // Arrange
        let mut s = ActiveState::default();

        // Act & Assert: sequential level escalation
        s.level = Some(SemanticLevel::L1);
        assert_eq!(s.level, Some(SemanticLevel::L1));

        s.level = Some(SemanticLevel::L2);
        assert_eq!(s.level, Some(SemanticLevel::L2));

        s.level = Some(SemanticLevel::L3);
        assert_eq!(s.level, Some(SemanticLevel::L3));
    }

    #[test]
    fn level_transition_downward() {
        // Arrange
        let mut s = ActiveState::default();
        s.level = Some(SemanticLevel::L3);

        // Act: transition downward
        s.level = Some(SemanticLevel::L1);

        // Assert
        assert_eq!(s.level, Some(SemanticLevel::L1));
    }

    #[test]
    fn anchor_hidden_and_v_knowledge_different_dimensions() {
        // Arrange: anchor is 4-dim, knowledge is 2-dim (asymmetric)
        let mut s = ActiveState::default();
        s.anchor_hidden = Some(vec![0.1, 0.2, 0.3, 0.4]);
        s.v_knowledge = Some(vec![0.5, 0.6]);

        // Assert: dimensions are independent
        assert_eq!(s.anchor_hidden.as_ref().unwrap().len(), 4);
        assert_eq!(s.v_knowledge.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn multiple_clear_cycles_with_population() {
        // Arrange
        let mut s = ActiveState::default();

        // Act: cycle through populate-clear three times
        for cycle in 0..3 {
            s.level = Some(SemanticLevel::L2);
            s.key_hash = Some(cycle as u64 * 100);
            s.last_step = (cycle as u64 + 1) * 10;
            s.last_request = Some(cycle as u64 + 1);

            // Assert populated
            assert_eq!(s.level, Some(SemanticLevel::L2));
            assert_eq!(s.key_hash, Some(cycle as u64 * 100));

            s.clear();

            // Assert cleared
            assert!(s.level.is_none());
            assert!(s.key_hash.is_none());
            assert_eq!(s.last_step, 0);
            assert!(s.last_request.is_none());
        }
    }

    #[test]
    fn default_equality_is_consistent() {
        // Arrange
        let a = ActiveState::default();
        let b = ActiveState::default();

        // Act: clear an already-default instance
        let mut c = ActiveState::default();
        c.clear();

        // Assert: all three are equivalent
        assert_eq!(format!("{:?}", a), format!("{:?}", b));
        assert_eq!(format!("{:?}", a), format!("{:?}", c));
    }

    // --- 13 additional tests ---

    #[test]
    fn set_all_fields_manually_verify_each() {
        // Arrange
        let mut s = ActiveState::default();

        // Act
        s.level = Some(SemanticLevel::L2);
        s.key_hash = Some(0xCAFE);
        s.anchor_hidden = Some(vec![0.1, 0.2, 0.3]);
        s.v_knowledge = Some(vec![0.4, 0.5]);
        s.ast_node_kind = Some("CallExpr".to_string());
        s.last_step = 42;
        s.last_request = Some(7);

        // Assert — every field holds the exact value assigned
        assert_eq!(s.level, Some(SemanticLevel::L2));
        assert_eq!(s.key_hash, Some(0xCAFE));
        assert_eq!(s.anchor_hidden.as_deref(), Some([0.1, 0.2, 0.3].as_slice()));
        assert_eq!(s.v_knowledge.as_deref(), Some([0.4, 0.5].as_slice()));
        assert_eq!(s.ast_node_kind.as_deref(), Some("CallExpr"));
        assert_eq!(s.last_step, 42);
        assert_eq!(s.last_request, Some(7));
    }

    #[test]
    fn clear_after_full_population_resets_every_field_individually() {
        // Arrange — populate every field with a non-default value
        let mut s = ActiveState {
            level: Some(SemanticLevel::L3),
            key_hash: Some(9999),
            anchor_hidden: Some(vec![1.0]),
            v_knowledge: Some(vec![2.0]),
            ast_node_kind: Some("Test".to_string()),
            last_step: 1234,
            last_request: Some(5678),
        };

        // Act
        s.clear();

        // Assert — each field individually reset
        assert!(s.level.is_none(), "level must be None after clear");
        assert!(s.key_hash.is_none(), "key_hash must be None after clear");
        assert!(s.anchor_hidden.is_none(), "anchor_hidden must be None after clear");
        assert!(s.v_knowledge.is_none(), "v_knowledge must be None after clear");
        assert!(s.ast_node_kind.is_none(), "ast_node_kind must be None after clear");
        assert_eq!(s.last_step, 0, "last_step must be 0 after clear");
        assert!(s.last_request.is_none(), "last_request must be None after clear");
    }

    #[test]
    fn clone_vec_fields_are_deep_copies() {
        // Arrange
        let mut s = ActiveState::default();
        s.anchor_hidden = Some(vec![1.0, 2.0]);
        s.v_knowledge = Some(vec![3.0, 4.0]);
        s.ast_node_kind = Some("Init".to_string());

        // Act
        let c = s.clone();
        // Mutate original vec fields
        s.anchor_hidden.as_mut().unwrap()[0] = 99.0;
        s.v_knowledge.as_mut().unwrap()[1] = -99.0;
        s.ast_node_kind.as_mut().unwrap().push_str("X");

        // Assert — clone's vecs are independent (deep copy)
        assert_eq!(c.anchor_hidden.as_deref(), Some([1.0, 2.0].as_slice()));
        assert_eq!(c.v_knowledge.as_deref(), Some([3.0, 4.0].as_slice()));
        assert_eq!(c.ast_node_kind.as_deref(), Some("Init"));
    }

    #[test]
    fn debug_format_shows_level_variant_name() {
        // Arrange
        let s = ActiveState {
            level: Some(SemanticLevel::L2),
            key_hash: None,
            anchor_hidden: None,
            v_knowledge: None,
            ast_node_kind: None,
            last_step: 0,
            last_request: None,
        };

        // Act
        let dbg = format!("{:?}", s);

        // Assert — the L2 variant name is present in the Debug output
        assert!(dbg.contains("L2"), "Debug output must contain 'L2' variant");
    }

    #[test]
    fn last_step_preserved_across_level_changes() {
        // Arrange
        let mut s = ActiveState::default();
        s.last_step = 500;
        s.level = Some(SemanticLevel::L1);

        // Act — change level, leave last_step untouched
        s.level = Some(SemanticLevel::L3);

        // Assert — last_step unchanged by level mutation
        assert_eq!(s.last_step, 500);
        assert_eq!(s.level, Some(SemanticLevel::L3));
    }

    #[test]
    fn needs_refresh_none_last_request_always_false() {
        // Arrange — default state has last_request = None
        let s = ActiveState::default();

        // Act & Assert — for any request ID, no refresh is needed
        assert!(!s.needs_request_boundary_refresh(0));
        assert!(!s.needs_request_boundary_refresh(1));
        assert!(!s.needs_request_boundary_refresh(u64::MAX));
    }

    #[test]
    fn needs_refresh_consecutive_same_id_no_refresh() {
        // Arrange
        let mut s = ActiveState::default();
        s.last_request = Some(42);

        // Act & Assert — repeated calls with the same ID all return false
        assert!(!s.needs_request_boundary_refresh(42));
        assert!(!s.needs_request_boundary_refresh(42));
        assert!(!s.needs_request_boundary_refresh(42));
    }

    #[test]
    fn anchor_hidden_single_value_preserved() {
        // Arrange
        let mut s = ActiveState::default();

        // Act
        s.anchor_hidden = Some(vec![std::f32::consts::PI]);

        // Assert
        assert_eq!(s.anchor_hidden.as_ref().unwrap().len(), 1);
        assert!((s.anchor_hidden.as_ref().unwrap()[0] - std::f32::consts::PI).abs() < f32::EPSILON);
    }

    #[test]
    fn v_knowledge_negative_values_preserved() {
        // Arrange
        let mut s = ActiveState::default();
        let negatives = vec![-0.001, -100.0, -f32::MAX, -f32::MIN_POSITIVE];

        // Act
        s.v_knowledge = Some(negatives.clone());

        // Assert — all negative values round-trip exactly
        let stored = s.v_knowledge.as_ref().unwrap();
        assert_eq!(stored.len(), 4);
        assert_eq!(stored[0], -0.001);
        assert_eq!(stored[1], -100.0);
        assert_eq!(stored[2], -f32::MAX);
        assert_eq!(stored[3], -f32::MIN_POSITIVE);
    }

    #[test]
    fn ast_node_kind_unicode_preserved() {
        // Arrange
        let mut s = ActiveState::default();
        let unicode_name = "関数宣言\u{1F600}";

        // Act
        s.ast_node_kind = Some(unicode_name.to_string());

        // Assert — Unicode string preserved exactly
        assert_eq!(s.ast_node_kind.as_deref(), Some(unicode_name));
        assert_eq!(s.ast_node_kind.as_ref().unwrap().chars().count(), unicode_name.chars().count());
    }

    #[test]
    fn key_hash_max_u64_preserved() {
        // Arrange
        let mut s = ActiveState::default();

        // Act
        s.key_hash = Some(u64::MAX);

        // Assert
        assert_eq!(s.key_hash, Some(u64::MAX));
    }

    #[test]
    fn triple_clear_produces_same_default() {
        // Arrange
        let mut s = ActiveState {
            level: Some(SemanticLevel::L1),
            key_hash: Some(100),
            anchor_hidden: Some(vec![5.0]),
            v_knowledge: Some(vec![6.0]),
            ast_node_kind: Some("A".to_string()),
            last_step: 999,
            last_request: Some(1),
        };

        // Act — three consecutive clears
        s.clear();
        let after_first = format!("{:?}", s);
        s.clear();
        let after_second = format!("{:?}", s);
        s.clear();
        let after_third = format!("{:?}", s);

        // Assert — all three produce identical output
        assert_eq!(after_first, after_second);
        assert_eq!(after_second, after_third);
        assert_eq!(after_first, format!("{:?}", ActiveState::default()));
    }

    #[test]
    fn level_set_clear_reset_cycle() {
        // Arrange
        let mut s = ActiveState::default();

        // Act — set level to L1
        s.level = Some(SemanticLevel::L1);
        assert_eq!(s.level, Some(SemanticLevel::L1));

        // Act — clear resets it
        s.clear();
        assert!(s.level.is_none());

        // Act — set level to L3 after clear
        s.level = Some(SemanticLevel::L3);

        // Assert — level holds the new value, not the old one
        assert_eq!(s.level, Some(SemanticLevel::L3));
        assert_ne!(s.level, Some(SemanticLevel::L1));
    }

    // --- 13 additional edge-case tests (wave 2) ---

    #[test]
    fn ast_node_kind_with_whitespace_and_control_chars() {
        // Arrange
        let mut s = ActiveState::default();
        let weird = "node\twith\nspaces and\u{0000}null";

        // Act
        s.ast_node_kind = Some(weird.to_string());

        // Assert — string with tabs, newlines, embedded NUL preserved exactly
        assert_eq!(s.ast_node_kind.as_deref(), Some(weird));
        assert_eq!(s.ast_node_kind.as_ref().unwrap().len(), weird.len());
    }

    #[test]
    fn last_step_overflow_wraps_saturating() {
        // Arrange
        let mut s = ActiveState::default();
        s.last_step = u64::MAX;

        // Act — manually wrapping add (simulates decode step increment)
        let next = s.last_step.wrapping_add(1);

        // Assert — wrapping semantics produce 0
        assert_eq!(next, 0);
        // The field itself is unchanged; the caller manages overflow policy
        assert_eq!(s.last_step, u64::MAX);
    }

    #[test]
    fn last_step_sequential_increment_from_zero() {
        // Arrange
        let mut s = ActiveState::default();
        assert_eq!(s.last_step, 0);

        // Act — simulate 5 decode step increments
        for i in 1..=5 {
            s.last_step = i;
        }

        // Assert
        assert_eq!(s.last_step, 5);
    }

    #[test]
    fn clear_then_repopulate_then_clear_again() {
        // Arrange — first population
        let mut s = ActiveState::default();
        s.level = Some(SemanticLevel::L1);
        s.key_hash = Some(100);
        s.last_step = 10;
        s.clear();

        // Act — second population with different values
        s.level = Some(SemanticLevel::L3);
        s.key_hash = Some(200);
        s.anchor_hidden = Some(vec![9.0]);
        s.v_knowledge = Some(vec![8.0]);
        s.ast_node_kind = Some("Second".to_string());
        s.last_step = 20;
        s.last_request = Some(5);

        // Assert — second values held
        assert_eq!(s.level, Some(SemanticLevel::L3));
        assert_eq!(s.key_hash, Some(200));
        assert_eq!(s.anchor_hidden.as_deref(), Some([9.0].as_slice()));
        assert_eq!(s.last_step, 20);
        assert_eq!(s.last_request, Some(5));

        // Act — second clear
        s.clear();

        // Assert — back to default again
        assert!(s.level.is_none());
        assert!(s.key_hash.is_none());
        assert_eq!(s.last_step, 0);
    }

    #[test]
    fn field_independence_changing_level_does_not_affect_others() {
        // Arrange — set all fields
        let mut s = ActiveState::default();
        s.level = Some(SemanticLevel::L2);
        s.key_hash = Some(555);
        s.anchor_hidden = Some(vec![1.0, 2.0]);
        s.v_knowledge = Some(vec![3.0]);
        s.ast_node_kind = Some("Fn".to_string());
        s.last_step = 99;
        s.last_request = Some(10);

        // Act — mutate only level
        s.level = Some(SemanticLevel::L1);

        // Assert — all other fields unchanged
        assert_eq!(s.key_hash, Some(555));
        assert_eq!(s.anchor_hidden.as_deref(), Some([1.0, 2.0].as_slice()));
        assert_eq!(s.v_knowledge.as_deref(), Some([3.0].as_slice()));
        assert_eq!(s.ast_node_kind.as_deref(), Some("Fn"));
        assert_eq!(s.last_step, 99);
        assert_eq!(s.last_request, Some(10));
    }

    #[test]
    fn field_independence_changing_last_step_does_not_affect_vecs() {
        // Arrange
        let mut s = ActiveState::default();
        s.anchor_hidden = Some(vec![4.0, 5.0]);
        s.v_knowledge = Some(vec![6.0]);
        s.last_step = 10;

        // Act — increment last_step
        s.last_step = 11;

        // Assert — vec fields untouched
        assert_eq!(s.anchor_hidden.as_deref(), Some([4.0, 5.0].as_slice()));
        assert_eq!(s.v_knowledge.as_deref(), Some([6.0].as_slice()));
    }

    #[test]
    fn individual_field_reset_to_none() {
        // Arrange
        let mut s = ActiveState::default();
        s.level = Some(SemanticLevel::L3);
        s.key_hash = Some(777);
        s.anchor_hidden = Some(vec![1.0]);
        s.v_knowledge = Some(vec![2.0]);
        s.ast_node_kind = Some("Expr".to_string());
        s.last_request = Some(3);

        // Act — reset individual fields to None without clear()
        s.level = None;
        s.key_hash = None;
        s.anchor_hidden = None;

        // Assert — only those fields are None, others untouched
        assert!(s.level.is_none());
        assert!(s.key_hash.is_none());
        assert!(s.anchor_hidden.is_none());
        assert_eq!(s.v_knowledge.as_deref(), Some([2.0].as_slice()));
        assert_eq!(s.ast_node_kind.as_deref(), Some("Expr"));
        assert_eq!(s.last_request, Some(3));
    }

    #[test]
    fn needs_refresh_alternating_request_ids() {
        // Arrange — simulate alternating between two concurrent requests
        let mut s = ActiveState::default();
        s.last_request = Some(1);

        // Act & Assert — switch to request 2
        assert!(s.needs_request_boundary_refresh(2));
        s.last_request = Some(2);

        // Switch back to request 1
        assert!(s.needs_request_boundary_refresh(1));
        s.last_request = Some(1);

        // Same request again — no refresh
        assert!(!s.needs_request_boundary_refresh(1));
    }

    #[test]
    fn ast_node_kind_overwrite_multiple_times() {
        // Arrange
        let mut s = ActiveState::default();
        let values = ["A", "BB", "CCC", "DDDD", "EEEEE"];

        // Act — overwrite 5 times
        for v in &values {
            s.ast_node_kind = Some(v.to_string());
        }

        // Assert — only the last value is retained
        assert_eq!(s.ast_node_kind.as_deref(), Some("EEEEE"));
    }

    #[test]
    fn debug_format_contains_specific_key_hash_value() {
        // Arrange
        let s = ActiveState {
            level: None,
            key_hash: Some(12345),
            anchor_hidden: None,
            v_knowledge: None,
            ast_node_kind: None,
            last_step: 0,
            last_request: None,
        };

        // Act
        let dbg = format!("{:?}", s);

        // Assert — the numeric value appears in the Debug output
        assert!(dbg.contains("12345"), "Debug output must contain key_hash value 12345");
    }

    #[test]
    fn debug_format_contains_last_step_value() {
        // Arrange
        let s = ActiveState {
            level: None,
            key_hash: None,
            anchor_hidden: None,
            v_knowledge: None,
            ast_node_kind: None,
            last_step: 999,
            last_request: None,
        };

        // Act
        let dbg = format!("{:?}", s);

        // Assert
        assert!(dbg.contains("999"), "Debug output must contain last_step value 999");
    }

    #[test]
    fn anchor_hidden_and_v_knowledge_cleared_independently() {
        // Arrange
        let mut s = ActiveState::default();
        s.anchor_hidden = Some(vec![1.0]);
        s.v_knowledge = Some(vec![2.0]);

        // Act — clear only anchor_hidden
        s.anchor_hidden = None;

        // Assert — v_knowledge still present
        assert!(s.anchor_hidden.is_none());
        assert_eq!(s.v_knowledge.as_deref(), Some([2.0].as_slice()));

        // Act — clear v_knowledge too
        s.v_knowledge = None;

        // Assert — both now None
        assert!(s.v_knowledge.is_none());
    }

    #[test]
    fn clone_deep_copy_push_does_not_affect_original() {
        // Arrange
        let mut s = ActiveState::default();
        s.anchor_hidden = Some(vec![1.0]);
        s.v_knowledge = Some(vec![2.0]);
        let c = s.clone();

        // Act — push into original's vecs
        s.anchor_hidden.as_mut().unwrap().push(3.0);
        s.v_knowledge.as_mut().unwrap().push(4.0);

        // Assert — clone has original length
        assert_eq!(c.anchor_hidden.as_ref().unwrap().len(), 1);
        assert_eq!(c.v_knowledge.as_ref().unwrap().len(), 1);
        assert_eq!(c.anchor_hidden.as_deref(), Some([1.0].as_slice()));
        assert_eq!(c.v_knowledge.as_deref(), Some([2.0].as_slice()));
    }

    // --- 13 additional tests (wave 3) ---

    #[test]
    fn level_as_idx_roundtrip_via_semantic_level() {
        // Arrange — assign levels via from_idx and verify via as_idx
        let mut s = ActiveState::default();

        for i in 0..3 {
            let level = SemanticLevel::from_idx(i);
            // Act
            s.level = level;
            // Assert — as_idx round-trips
            assert_eq!(s.level.unwrap().as_idx(), i);
        }
    }

    #[test]
    fn partial_struct_construction_only_level() {
        // Arrange — construct with only level set, rest default
        let s = ActiveState {
            level: Some(SemanticLevel::L2),
            key_hash: None,
            anchor_hidden: None,
            v_knowledge: None,
            ast_node_kind: None,
            last_step: 0,
            last_request: None,
        };

        // Assert — only level is non-default
        assert_eq!(s.level, Some(SemanticLevel::L2));
        assert!(s.key_hash.is_none());
        assert!(s.anchor_hidden.is_none());
        assert!(s.v_knowledge.is_none());
        assert!(s.ast_node_kind.is_none());
        assert_eq!(s.last_step, 0);
        assert!(s.last_request.is_none());
    }

    #[test]
    fn needs_refresh_request_id_zero_vs_none() {
        // Arrange — two states: one with last_request=Some(0), one with None
        let mut with_zero = ActiveState::default();
        with_zero.last_request = Some(0);
        let with_none = ActiveState::default();

        // Act & Assert — Some(0) vs request_id 0: no refresh (same)
        assert!(!with_zero.needs_request_boundary_refresh(0));
        // None vs request_id 0: no refresh (no prior)
        assert!(!with_none.needs_request_boundary_refresh(0));
        // Some(0) vs request_id 1: refresh needed
        assert!(with_zero.needs_request_boundary_refresh(1));
    }

    #[test]
    fn last_request_direct_equality_after_assign() {
        // Arrange
        let mut s = ActiveState::default();
        s.last_request = Some(42);

        // Act — reassign same value
        s.last_request = Some(42);

        // Assert — still equal
        assert_eq!(s.last_request, Some(42));
        assert!(!s.needs_request_boundary_refresh(42));
    }

    #[test]
    fn anchor_hidden_extreme_float_max_and_epsilon() {
        // Arrange
        let mut s = ActiveState::default();
        let extremes = vec![f32::MAX, f32::MIN_POSITIVE, f32::EPSILON, 0.0f32];

        // Act
        s.anchor_hidden = Some(extremes.clone());

        // Assert — exact round-trip
        let stored = s.anchor_hidden.as_ref().unwrap();
        assert_eq!(stored[0], f32::MAX);
        assert_eq!(stored[1], f32::MIN_POSITIVE);
        assert_eq!(stored[2], f32::EPSILON);
        assert_eq!(stored[3], 0.0f32);
    }

    #[test]
    fn v_knowledge_empty_vec_replaced_with_values() {
        // Arrange — start with empty vec
        let mut s = ActiveState::default();
        s.v_knowledge = Some(vec![]);
        assert_eq!(s.v_knowledge.as_ref().unwrap().len(), 0);

        // Act — replace with actual values
        s.v_knowledge = Some(vec![1.0, 2.0, 3.0]);

        // Assert
        assert_eq!(s.v_knowledge.as_ref().unwrap().len(), 3);
        assert_eq!(s.v_knowledge.as_deref(), Some([1.0, 2.0, 3.0].as_slice()));
    }

    #[test]
    fn ast_node_kind_very_long_string() {
        // Arrange — 10,000 character string
        let mut s = ActiveState::default();
        let long: String = "A".repeat(10_000);

        // Act
        s.ast_node_kind = Some(long.clone());

        // Assert
        assert_eq!(s.ast_node_kind.as_ref().unwrap().len(), 10_000);
        assert_eq!(s.ast_node_kind.as_deref(), Some(long.as_str()));

        // Clear removes it
        s.clear();
        assert!(s.ast_node_kind.is_none());
    }

    #[test]
    fn level_change_does_not_affect_needs_refresh() {
        // Arrange — set last_request and level
        let mut s = ActiveState::default();
        s.last_request = Some(1);
        s.level = Some(SemanticLevel::L1);

        // Act — change level only
        s.level = Some(SemanticLevel::L3);

        // Assert — refresh decision based solely on last_request, not level
        assert!(s.needs_request_boundary_refresh(2));
        assert!(!s.needs_request_boundary_refresh(1));
    }

    #[test]
    fn key_hash_alternating_bits_pattern() {
        // Arrange — 0x5555...5555 (alternating bits)
        let mut s = ActiveState::default();
        let pattern = 0x5555_5555_5555_5555u64;

        // Act
        s.key_hash = Some(pattern);

        // Assert
        assert_eq!(s.key_hash, Some(pattern));

        // Clear removes it
        s.clear();
        assert!(s.key_hash.is_none());
    }

    #[test]
    fn anchor_hidden_mutation_via_as_mut() {
        // Arrange
        let mut s = ActiveState::default();
        s.anchor_hidden = Some(vec![1.0, 2.0, 3.0]);

        // Act — mutate in-place
        let hidden = s.anchor_hidden.as_mut().unwrap();
        hidden[0] = 10.0;
        hidden.push(4.0);

        // Assert — mutations reflected
        assert_eq!(s.anchor_hidden.as_deref(), Some([10.0, 2.0, 3.0, 4.0].as_slice()));
    }

    #[test]
    fn last_request_reassigned_after_clear() {
        // Arrange
        let mut s = ActiveState::default();
        s.last_request = Some(100);
        s.clear();

        // Act — assign a new request after clear
        s.last_request = Some(200);

        // Assert — holds new value, refresh logic works
        assert_eq!(s.last_request, Some(200));
        assert!(!s.needs_request_boundary_refresh(200));
        assert!(s.needs_request_boundary_refresh(100));
    }

    #[test]
    fn multiple_clones_chain_independence() {
        // Arrange
        let mut s = ActiveState::default();
        s.level = Some(SemanticLevel::L3);
        s.key_hash = Some(42);
        s.last_step = 100;

        // Act — chain of clones, mutate original each time
        let c1 = s.clone();
        s.level = Some(SemanticLevel::L1);
        let c2 = s.clone();
        s.key_hash = Some(99);
        let c3 = s.clone();

        // Assert — each clone captured the state at clone time
        assert_eq!(c1.level, Some(SemanticLevel::L3));
        assert_eq!(c1.key_hash, Some(42));
        assert_eq!(c2.level, Some(SemanticLevel::L1));
        assert_eq!(c2.key_hash, Some(42));
        assert_eq!(c3.level, Some(SemanticLevel::L1));
        assert_eq!(c3.key_hash, Some(99));
    }

    #[test]
    fn from_idx_out_of_bounds_yields_none_in_level() {
        // Arrange — SemanticLevel::from_idx returns None for out-of-bounds
        assert!(SemanticLevel::from_idx(3).is_none());
        assert!(SemanticLevel::from_idx(100).is_none());

        // Act — assign None to level (simulating failed from_idx)
        let mut s = ActiveState::default();
        s.level = SemanticLevel::from_idx(3);

        // Assert — level stays None
        assert!(s.level.is_none());
    }

    // --- 13 new tests (wave 4) ---

    // @trace TEST-AS-74 [req:REQ-SG] [level:unit]
    #[test]
    fn needs_refresh_adjacent_request_ids_ascending() {
        // Arrange — simulate sequential request processing
        let mut s = ActiveState::default();
        s.last_request = Some(0);

        // Act & Assert — each adjacent request triggers refresh
        assert!(s.needs_request_boundary_refresh(1));
        s.last_request = Some(1);
        assert!(s.needs_request_boundary_refresh(2));
        s.last_request = Some(2);
        assert!(!s.needs_request_boundary_refresh(2));
    }

    // @trace TEST-AS-75 [req:REQ-SG] [level:unit]
    #[test]
    fn level_explicit_none_assignment_after_population() {
        // Arrange — populate level
        let mut s = ActiveState::default();
        s.level = Some(SemanticLevel::L2);

        // Act — explicitly set to None (not via clear)
        s.level = None;

        // Assert — level is None but other fields remain at default
        assert!(s.level.is_none());
        assert_eq!(s.last_step, 0);
    }

    // @trace TEST-AS-76 [req:REQ-SG] [level:unit]
    #[test]
    fn v_knowledge_replaced_multiple_times_without_clear() {
        // Arrange
        let mut s = ActiveState::default();

        // Act — three successive reassignments
        s.v_knowledge = Some(vec![1.0]);
        assert_eq!(s.v_knowledge.as_ref().unwrap().len(), 1);

        s.v_knowledge = Some(vec![2.0, 3.0]);
        assert_eq!(s.v_knowledge.as_ref().unwrap().len(), 2);

        s.v_knowledge = Some(vec![4.0, 5.0, 6.0]);

        // Assert — only the last value is retained
        assert_eq!(s.v_knowledge.as_deref(), Some([4.0, 5.0, 6.0].as_slice()));
    }

    // @trace TEST-AS-77 [req:REQ-SG] [level:unit]
    #[test]
    fn key_hash_all_ones_bit_pattern() {
        // Arrange — 0xFFFF...FFFF (all bits set)
        let mut s = ActiveState::default();
        let all_ones = !0u64;

        // Act
        s.key_hash = Some(all_ones);

        // Assert
        assert_eq!(s.key_hash, Some(u64::MAX));
        assert_ne!(s.key_hash, Some(0));
    }

    // @trace TEST-AS-78 [req:REQ-SG] [level:unit]
    #[test]
    fn last_step_set_nonzero_to_another_nonzero() {
        // Arrange
        let mut s = ActiveState::default();
        s.last_step = 100;

        // Act — overwrite with a different nonzero value
        s.last_step = 200;

        // Assert — holds the latest value
        assert_eq!(s.last_step, 200);
        assert_ne!(s.last_step, 100);
    }

    // @trace TEST-AS-79 [req:REQ-SG] [level:unit]
    #[test]
    fn ast_node_kind_whitespace_only_string() {
        // Arrange
        let mut s = ActiveState::default();
        let spaces = "   \t  \n  ";

        // Act
        s.ast_node_kind = Some(spaces.to_string());

        // Assert — whitespace-only string is preserved exactly (not trimmed)
        assert_eq!(s.ast_node_kind.as_deref(), Some(spaces));
        assert_eq!(s.ast_node_kind.as_ref().unwrap().len(), spaces.len());
    }

    // @trace TEST-AS-80 [req:REQ-SG] [level:unit]
    #[test]
    fn clone_fully_populated_then_clear_clone_only() {
        // Arrange — fully populated state
        let mut s = ActiveState {
            level: Some(SemanticLevel::L3),
            key_hash: Some(42),
            anchor_hidden: Some(vec![1.0, 2.0]),
            v_knowledge: Some(vec![3.0]),
            ast_node_kind: Some("Test".to_string()),
            last_step: 10,
            last_request: Some(5),
        };

        // Act — clone then clear the clone
        let mut c = s.clone();
        c.clear();

        // Assert — original is untouched, clone is default
        assert_eq!(s.level, Some(SemanticLevel::L3));
        assert_eq!(s.key_hash, Some(42));
        assert_eq!(s.last_step, 10);

        assert!(c.level.is_none());
        assert!(c.key_hash.is_none());
        assert_eq!(c.last_step, 0);
    }

    // @trace TEST-AS-81 [req:REQ-SG] [level:unit]
    #[test]
    fn needs_refresh_minimal_difference_request_id_1_vs_2() {
        // Arrange
        let mut s = ActiveState::default();
        s.last_request = Some(1);

        // Act & Assert — request ID 2 triggers refresh (minimum numeric difference)
        assert!(s.needs_request_boundary_refresh(2));
        // Same ID does not
        assert!(!s.needs_request_boundary_refresh(1));
    }

    // @trace TEST-AS-82 [req:REQ-SG] [level:unit]
    #[test]
    fn anchor_hidden_set_to_none_after_having_values() {
        // Arrange — populate anchor_hidden
        let mut s = ActiveState::default();
        s.anchor_hidden = Some(vec![1.0, 2.0, 3.0]);
        assert!(s.anchor_hidden.is_some());

        // Act — explicitly set back to None
        s.anchor_hidden = None;

        // Assert
        assert!(s.anchor_hidden.is_none());
    }

    // @trace TEST-AS-83 [req:REQ-SG] [level:unit]
    #[test]
    fn level_reassigned_per_order_index() {
        // Arrange
        let mut s = ActiveState::default();

        // Act — assign each level by ORDER index
        for (i, &level) in SemanticLevel::ORDER.iter().enumerate() {
            s.level = Some(level);

            // Assert — level matches the index
            assert_eq!(s.level.unwrap().as_idx(), i);
        }
    }

    // @trace TEST-AS-84 [req:REQ-SG] [level:unit]
    #[test]
    fn debug_format_none_fields_show_none() {
        // Arrange — all fields are None/default
        let s = ActiveState::default();

        // Act
        let dbg = format!("{:?}", s);

        // Assert — None fields should render as "None" in Debug output
        assert!(dbg.contains("None"), "Debug output must contain 'None' for unset fields");
    }

    // @trace TEST-AS-85 [req:REQ-SG] [level:unit]
    #[test]
    fn last_request_wraparound_from_max_to_zero() {
        // Arrange — set last_request to u64::MAX
        let mut s = ActiveState::default();
        s.last_request = Some(u64::MAX);

        // Act & Assert — refresh needed for request ID 0 (wraparound scenario)
        assert!(s.needs_request_boundary_refresh(0));
        // No refresh for same max ID
        assert!(!s.needs_request_boundary_refresh(u64::MAX));
    }

    // @trace TEST-AS-86 [req:REQ-SG] [level:unit]
    #[test]
    fn clear_idempotent_after_partial_population() {
        // Arrange — only set some fields (partial population)
        let mut s = ActiveState::default();
        s.level = Some(SemanticLevel::L1);
        s.last_step = 50;
        // key_hash, anchor_hidden, v_knowledge, ast_node_kind, last_request all remain None/default

        // Act — clear twice
        s.clear();
        let after_first = format!("{:?}", s);
        s.clear();
        let after_second = format!("{:?}", s);

        // Assert — both clears produce identical default state
        assert_eq!(after_first, after_second);
        assert_eq!(after_first, format!("{:?}", ActiveState::default()));
    }
}
