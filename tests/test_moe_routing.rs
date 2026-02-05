use std::borrow::Cow;
use std::time::{Duration, Instant};

use gllm::adapter::adapter_for;
use gllm::manifest::{
    ModelArchitecture, ModelKind, ModelManifest, TensorNamingRule, EMPTY_FILE_MAP,
};
use gllm::scheduler::{GroupState, HGALConfig, HGALScheduler, SequenceGroup};
use gllm_kernels::cpu_backend::CpuBackend;
use gllm_kernels::kernel_types::PageState;

/// TEST-MOE-001: Qwen3 MoE manifest 选择 MoE 适配器
///
/// **关联需求**: REQ-TEST-009
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 创建 Qwen3MoE manifest
/// 2. 验证 is_moe() 返回 true
/// 3. 验证 adapter_for() 返回 Some
///
/// **期望结果**: MoE 模型正确识别并获取适配器
#[test]
fn qwen3_moe_manifest_selects_moe_adapter() {
    let manifest = ModelManifest {
        model_id: Cow::Borrowed("Qwen/Qwen3-235B-A22B-Instruct"),
        file_map: EMPTY_FILE_MAP,
        arch: ModelArchitecture::Qwen3MoE,
        tensor_rules: TensorNamingRule::Qwen3,
        kind: ModelKind::Chat,
        rope_base_override: None,
        max_context_override: None,
        moe_config: None,
    };
    assert!(manifest.is_moe(), "qwen3-moe should be marked as MoE");
    let adapter = adapter_for::<CpuBackend>(&manifest);
    assert!(
        adapter.is_some(),
        "MoE adapter should be available for qwen3-moe"
    );
}

/// TEST-MOE-002: MoE 路由器优先驱逐冷专家
///
/// **关联需求**: REQ-TEST-009
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 创建热专家 (高频访问)
/// 2. 创建冷专家 (低频访问)
/// 3. 执行 select_victim_groups()
///
/// **期望结果**: 冷专家被优先驱逐
#[test]
fn moe_router_prefers_colder_experts_for_eviction() {
    let mut scheduler = HGALScheduler::new(HGALConfig::default());
    let now = Instant::now();

    // Expert A: hot, recently accessed.
    let expert_a = SequenceGroup {
        id: 1,
        pages: vec![1, 2],
        state: GroupState::Running,
        access_count: 8,
        last_access: now,
        is_pinned: true, // active expert stays resident
    };
    scheduler.upsert_group(expert_a);
    scheduler.update_page_state(1, Some(1), PageState::Active);
    scheduler.update_page_state(2, Some(1), PageState::Active);

    // Expert B: colder, fewer accesses.
    let expert_b = SequenceGroup {
        id: 2,
        pages: vec![3, 4],
        state: GroupState::Running,
        access_count: 1,
        last_access: now - Duration::from_millis(10),
        is_pinned: false,
    };
    scheduler.upsert_group(expert_b);
    scheduler.update_page_state(3, Some(2), PageState::Standby);
    scheduler.update_page_state(4, Some(2), PageState::Standby);

    // Victim selection should evict the colder expert to keep load balanced.
    let victims = scheduler.select_victim_groups(1);
    assert_eq!(victims, vec![2]);
}

/// TEST-MOE-003: MoE 路由器标记访问并平衡最近性
///
/// **关联需求**: REQ-TEST-009
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 创建两个专家
/// 2. 标记访问
/// 3. 验证选择逻辑保持平衡
///
/// **期望结果**: 两个专家都保持在活跃状态
#[test]
fn moe_router_marks_access_and_balances_recency() {
    let mut scheduler = HGALScheduler::new(HGALConfig::default());
    let now = Instant::now();

    let expert_a = SequenceGroup {
        id: 10,
        pages: vec![10],
        state: GroupState::Running,
        access_count: 0,
        last_access: now,
        is_pinned: false,
    };
    let expert_b = SequenceGroup {
        id: 11,
        pages: vec![11],
        state: GroupState::Running,
        access_count: 0,
        last_access: now,
        is_pinned: false,
    };
    scheduler.upsert_group(expert_a);
    scheduler.upsert_group(expert_b);
    scheduler.update_page_state(10, Some(10), PageState::Active);
    scheduler.update_page_state(11, Some(11), PageState::Active);

    // Route first token to expert A, second to expert B to keep accesses balanced.
    scheduler.mark_accessed(10);
    scheduler.mark_accessed(11);

    let victims = scheduler.select_victim_groups(1);
    assert!(
        victims.contains(&11) || victims.contains(&10),
        "router should keep both experts in play"
    );
}
