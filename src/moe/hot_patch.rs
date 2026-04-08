//! Hot JMP Patching 框架 (SPEC §14.4)
//!
//! ## 核心职责
//! 管理 JIT 代码段的热修补决策:
//! - 场景 A: MoE 冷板凳专家持续零命中 → NOP/JMP 物理抹平
//! - 场景 B: 前缀复用塌缩 → 共享直读存储指令
//!
//! ## §14.4 安全性约束
//! Hot JMP Patching **仅用于绝对的全局物理共识**，严禁个体行为触发:
//! - ❌ Request A 可以跳过某层 → 不能 patch（会破坏 Request B）
//! - ✅ 全部请求 100 万步从未使用某专家 → 可以 patch
//! - ❌ 单个请求的低方差 → 不能 patch
//! - ✅ 所有并发请求共享同一 System Prompt 前缀 → 可以 patch

use super::thermal::{ExpertHeatLevel, ExpertThermalManager};
use super::routing::ExpertRouteConfig;

/// 热修补操作类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchOperation {
    /// NOP 替换: 将专家代码段替换为 NOP (无操作)
    NopReplace,
    /// JMP 跳转: 将专家代码段替换为 JMP 到 Deopt Handler
    DeoptJump,
    /// 前缀塌缩: 将 Prefill 图塌缩为共享直读指令
    PrefixCollapse,
    /// 恢复: 回写原始专家代码
    Restore,
}

/// 热修补目标
#[derive(Debug, Clone)]
pub enum PatchTarget {
    /// MoE 专家代码段
    ExpertCode {
        expert_idx: usize,
        layer_idx: usize,
    },
    /// Prefill 前缀图
    PrefixGraph {
        prefix_hash: u64,
        shared_length: usize,
    },
}

/// 单个热修补指令
#[derive(Debug, Clone)]
pub struct PatchInstruction {
    /// 补丁目标
    pub target: PatchTarget,
    /// 操作类型
    pub operation: PatchOperation,
    /// 全局共识步数 (必须达到阈值才能执行)
    pub consensus_steps: u64,
    /// 补丁原因
    pub reason: String,
    /// 优先级 (0 = 最高)
    pub priority: u32,
}

/// 热修补执行结果
#[derive(Debug, Clone)]
pub struct PatchResult {
    /// 被补丁的目标
    pub target: PatchTarget,
    /// 执行的操作
    pub operation: PatchOperation,
    /// 是否成功
    pub success: bool,
    /// 失败原因
    pub failure_reason: Option<String>,
    /// 补丁大小 (bytes)
    pub patch_size: usize,
}

/// 热修补安全检查结果
#[derive(Debug, Clone)]
pub enum PatchSafetyCheck {
    /// 安全: 全局物理共识已达成
    Safe {
        consensus_steps: u64,
        zero_hit_count: u64,
    },
    /// 不安全: 存在活跃请求依赖该目标
    UnsafeActiveRequests {
        active_request_count: usize,
    },
    /// 不安全: 共识步数不足
    UnsafeInsufficientConsensus {
        current_steps: u64,
        required_steps: u64,
    },
    /// 已修补: 目标已经被修补过
    AlreadyPatched,
}

/// Hot JMP Patching 管理器
///
/// §14.4: 管理全局物理共识级别的 JIT 代码热修补。
/// 只处理"绝对的全局物理共识"场景，个体级别的优化交给 §9.1 块级内嵌路由。
pub struct HotPatchManager {
    /// MoE 专家路由配置
    route_config: ExpertRouteConfig,
    /// 已修补的专家集合 (expert_idx, layer_idx)
    patched_experts: Vec<(usize, usize)>,
    /// 已塌缩的前缀集合 (prefix_hash)
    collapsed_prefixes: Vec<u64>,
    /// 全局共识步数阈值 (必须连续零命中多少步才允许 patch)
    consensus_threshold: u64,
    /// 最小并发请求数 (并发请求少时允许更激进 patch)
    min_requests_for_safety: usize,
    /// 累计 patch 执行次数
    total_patches_applied: u64,
    /// 累计 patch 回滚次数
    total_patches_rolled_back: u64,
}

impl HotPatchManager {
    /// 创建新的 Hot JMP Patching 管理器
    pub fn new(route_config: ExpertRouteConfig) -> Self {
        Self {
            route_config,
            patched_experts: Vec::new(),
            collapsed_prefixes: Vec::new(),
            consensus_threshold: 1_000_000, // 100 万步零命中
            min_requests_for_safety: 1,
            total_patches_applied: 0,
            total_patches_rolled_back: 0,
        }
    }

    /// 配置共识阈值
    pub fn with_consensus_threshold(mut self, threshold: u64) -> Self {
        self.consensus_threshold = threshold;
        self
    }

    /// 安全检查: 是否可以对指定专家执行热修补
    ///
    /// §14.4 安全性铁律:
    /// - 只有全局物理共识才能触发热修补
    /// - 个体行为（如 Request A 想跳过）不能触发
    pub fn check_expert_patch_safety(
        &self,
        expert_idx: usize,
        layer_idx: usize,
        thermal_manager: &ExpertThermalManager,
        active_request_count: usize,
    ) -> PatchSafetyCheck {
        // 检查是否已修补
        if self.patched_experts.contains(&(expert_idx, layer_idx)) {
            return PatchSafetyCheck::AlreadyPatched;
        }

        // 检查热度状态
        let state = match thermal_manager.state(expert_idx) {
            Some(s) => s,
            None => {
                return PatchSafetyCheck::UnsafeInsufficientConsensus {
                    current_steps: 0,
                    required_steps: self.consensus_threshold,
                };
            }
        };

        // 只有 Evicted 状态的专家才能 patch
        if state.heat_level != ExpertHeatLevel::Evicted {
            return PatchSafetyCheck::UnsafeInsufficientConsensus {
                current_steps: state.consecutive_zero_streak,
                required_steps: self.consensus_threshold,
            };
        }

        // 检查共识步数
        if state.consecutive_zero_streak < self.consensus_threshold {
            return PatchSafetyCheck::UnsafeInsufficientConsensus {
                current_steps: state.consecutive_zero_streak,
                required_steps: self.consensus_threshold,
            };
        }

        // 检查活跃请求 (如果有活跃请求在用该专家，不能 patch)
        if state.reactivation_count > 0 && active_request_count > 0 {
            return PatchSafetyCheck::UnsafeActiveRequests {
                active_request_count,
            };
        }

        PatchSafetyCheck::Safe {
            consensus_steps: state.consecutive_zero_streak,
            zero_hit_count: state.route_count - state.hit_count,
        }
    }

    /// 生成专家 NOP/Deopt 修补指令
    ///
    /// §14.4 场景 A: MoE 冷板凳专家 → NOP/JMP 物理抹平
    pub fn generate_expert_patch_instructions(
        &self,
        thermal_manager: &ExpertThermalManager,
        num_layers: usize,
        active_request_count: usize,
    ) -> Vec<PatchInstruction> {
        let mut instructions = Vec::new();

        for expert_idx in 0..self.route_config.num_experts {
            for layer_idx in 0..num_layers {
                let safety = self.check_expert_patch_safety(
                    expert_idx,
                    layer_idx,
                    thermal_manager,
                    active_request_count,
                );

                match safety {
                    PatchSafetyCheck::Safe { consensus_steps, .. } => {
                        // DeoptJump: 比 NOP 更安全，保留恢复路径
                        instructions.push(PatchInstruction {
                            target: PatchTarget::ExpertCode { expert_idx, layer_idx },
                            operation: PatchOperation::DeoptJump,
                            consensus_steps,
                            reason: format!(
                                "Expert {} at layer {} evicted after {} consecutive zero-hit steps",
                                expert_idx, layer_idx, consensus_steps
                            ),
                            priority: expert_idx as u32,
                        });
                    }
                    PatchSafetyCheck::UnsafeActiveRequests { .. } => {
                        // 有活跃请求，不能 patch，记录但跳过
                    }
                    PatchSafetyCheck::UnsafeInsufficientConsensus { .. }
                    | PatchSafetyCheck::AlreadyPatched => {}
                }
            }
        }

        // 按优先级排序
        instructions.sort_by_key(|i| i.priority);
        instructions
    }

    /// 生成前缀塌缩修补指令
    ///
    /// §14.4 场景 B: 所有并发请求共享同一前缀 → 塌缩为共享直读指令
    pub fn generate_prefix_collapse_instruction(
        &self,
        prefix_hash: u64,
        shared_length: usize,
        active_request_count: usize,
    ) -> Option<PatchInstruction> {
        // 已塌缩
        if self.collapsed_prefixes.contains(&prefix_hash) {
            return None;
        }

        // 至少需要 min_requests_for_safety 个请求共享同一前缀
        if active_request_count < self.min_requests_for_safety {
            return None;
        }

        Some(PatchInstruction {
            target: PatchTarget::PrefixGraph {
                prefix_hash,
                shared_length,
            },
            operation: PatchOperation::PrefixCollapse,
            consensus_steps: 0, // 前缀塌缩不需要共识步数
            reason: format!(
                "Prefix hash {:016x} shared by {} requests ({} tokens)",
                prefix_hash, active_request_count, shared_length
            ),
            priority: 0,
        })
    }

    /// 执行补丁 (标记为已修补，实际 .text 回写由 JIT Director 完成)
    ///
    /// 返回 PatchResult 供 JIT Director 消费
    pub fn apply_patch(&mut self, instruction: &PatchInstruction) -> PatchResult {
        let success = match &instruction.target {
            PatchTarget::ExpertCode { expert_idx, layer_idx } => {
                let key = (*expert_idx, *layer_idx);
                if self.patched_experts.contains(&key) {
                    false // 已修补
                } else {
                    self.patched_experts.push(key);
                    self.total_patches_applied += 1;
                    true
                }
            }
            PatchTarget::PrefixGraph { prefix_hash, .. } => {
                if self.collapsed_prefixes.contains(prefix_hash) {
                    false
                } else {
                    self.collapsed_prefixes.push(*prefix_hash);
                    self.total_patches_applied += 1;
                    true
                }
            }
        };

        PatchResult {
            target: instruction.target.clone(),
            operation: instruction.operation,
            success,
            failure_reason: if success {
                None
            } else {
                Some("target already patched".to_string())
            },
            patch_size: 0, // 由 JIT Director 填充实际大小
        }
    }

    /// 回滚补丁 (OSR Bailout 恢复专家时调用)
    ///
    /// §15.4: 当 Uncommon Trap 触发时，JIT Director 需要恢复专家代码
    pub fn rollback_patch(&mut self, expert_idx: usize, layer_idx: usize) -> bool {
        let key = (expert_idx, layer_idx);
        if let Some(pos) = self.patched_experts.iter().position(|&k| k == key) {
            self.patched_experts.remove(pos);
            self.total_patches_rolled_back += 1;
            true
        } else {
            false
        }
    }

    /// 回滚前缀塌缩
    pub fn rollback_prefix_collapse(&mut self, prefix_hash: u64) -> bool {
        if let Some(pos) = self.collapsed_prefixes.iter().position(|&h| h == prefix_hash) {
            self.collapsed_prefixes.remove(pos);
            self.total_patches_rolled_back += 1;
            true
        } else {
            false
        }
    }

    /// 检查指定专家是否已被修补
    pub fn is_expert_patched(&self, expert_idx: usize, layer_idx: usize) -> bool {
        self.patched_experts.contains(&(expert_idx, layer_idx))
    }

    /// 检查指定前缀是否已塌缩
    pub fn is_prefix_collapsed(&self, prefix_hash: u64) -> bool {
        self.collapsed_prefixes.contains(&prefix_hash)
    }

    /// 获取统计摘要
    pub fn summary(&self) -> HotPatchSummary {
        HotPatchSummary {
            total_patches_applied: self.total_patches_applied,
            total_patches_rolled_back: self.total_patches_rolled_back,
            patched_expert_count: self.patched_experts.len(),
            collapsed_prefix_count: self.collapsed_prefixes.len(),
            consensus_threshold: self.consensus_threshold,
        }
    }

    /// 获取路由配置
    pub fn route_config(&self) -> &ExpertRouteConfig {
        &self.route_config
    }
}

/// 热修补统计摘要
#[derive(Debug, Clone)]
pub struct HotPatchSummary {
    pub total_patches_applied: u64,
    pub total_patches_rolled_back: u64,
    pub patched_expert_count: usize,
    pub collapsed_prefix_count: usize,
    pub consensus_threshold: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_patch_safety_evicted() {
        let config = ExpertRouteConfig::new(4, 2);
        let thermal = ExpertThermalManager::new(4)
            .with_eviction_threshold(10);

        let manager = HotPatchManager::new(config)
            .with_consensus_threshold(5);

        // 模拟: expert 1 连续 10 步零命中 → 封杀
        let mut thermal = thermal;
        for _ in 0..11 {
            thermal.step(&[10, 0, 5, 3]);
        }
        thermal.evict_expert(1);

        // 安全检查
        let safety = manager.check_expert_patch_safety(1, 0, &thermal, 0);
        match safety {
            PatchSafetyCheck::Safe { consensus_steps, .. } => {
                assert!(consensus_steps >= 5);
            }
            _ => panic!("Expected Safe, got {:?}", safety),
        }
    }

    #[test]
    fn test_expert_patch_safety_not_evicted() {
        let config = ExpertRouteConfig::new(4, 2);
        let thermal = ExpertThermalManager::new(4);
        let manager = HotPatchManager::new(config);

        // Expert 0 活跃使用中，不应被 patch
        let safety = manager.check_expert_patch_safety(0, 0, &thermal, 0);
        assert!(matches!(safety, PatchSafetyCheck::UnsafeInsufficientConsensus { .. }));
    }

    #[test]
    fn test_expert_patch_safety_already_patched() {
        let config = ExpertRouteConfig::new(4, 2);
        let mut manager = HotPatchManager::new(config);
        manager.patched_experts.push((2, 0));

        let thermal = ExpertThermalManager::new(4);
        let safety = manager.check_expert_patch_safety(2, 0, &thermal, 0);
        assert!(matches!(safety, PatchSafetyCheck::AlreadyPatched));
    }

    #[test]
    fn test_generate_and_apply_patch() {
        let config = ExpertRouteConfig::new(4, 2);
        let mut thermal = ExpertThermalManager::new(4)
            .with_eviction_threshold(5);

        for _ in 0..6 {
            thermal.step(&[10, 0, 0, 0]);
        }
        thermal.evict_expert(1);
        thermal.evict_expert(2);

        let mut manager = HotPatchManager::new(config)
            .with_consensus_threshold(5);

        let instructions = manager.generate_expert_patch_instructions(&thermal, 2, 0);
        assert!(!instructions.is_empty());

        // 执行第一个补丁
        let result = manager.apply_patch(&instructions[0]);
        assert!(result.success);
        assert_eq!(manager.total_patches_applied, 1);
    }

    #[test]
    fn test_rollback_patch() {
        let config = ExpertRouteConfig::new(4, 2);
        let mut manager = HotPatchManager::new(config);
        manager.patched_experts.push((1, 0));

        assert!(manager.is_expert_patched(1, 0));
        assert!(manager.rollback_patch(1, 0));
        assert!(!manager.is_expert_patched(1, 0));
        assert_eq!(manager.total_patches_rolled_back, 1);
    }

    #[test]
    fn test_prefix_collapse() {
        let config = ExpertRouteConfig::new(4, 2);
        let mut manager = HotPatchManager::new(config);

        let hash = 0xABCD1234;
        let instruction = manager.generate_prefix_collapse_instruction(hash, 10000, 8);
        assert!(instruction.is_some());

        let instr = instruction.unwrap();
        assert_eq!(instr.operation, PatchOperation::PrefixCollapse);

        // 执行
        let result = manager.apply_patch(&instr);
        assert!(result.success);

        // 已塌缩
        assert!(manager.is_prefix_collapsed(hash));

        // 重复请求返回 None
        let dup = manager.generate_prefix_collapse_instruction(hash, 10000, 8);
        assert!(dup.is_none());

        // 回滚
        assert!(manager.rollback_prefix_collapse(hash));
        assert!(!manager.is_prefix_collapsed(hash));
    }

    #[test]
    fn test_safety_insufficient_consensus() {
        let config = ExpertRouteConfig::new(4, 2);
        let mut thermal = ExpertThermalManager::new(4)
            .with_eviction_threshold(5);

        // 只有 6 步，不够共识阈值 100
        for _ in 0..6 {
            thermal.step(&[10, 0, 0, 0]);
        }
        thermal.evict_expert(1);

        let manager = HotPatchManager::new(config)
            .with_consensus_threshold(100); // 高阈值

        let safety = manager.check_expert_patch_safety(1, 0, &thermal, 0);
        match safety {
            PatchSafetyCheck::UnsafeInsufficientConsensus {
                current_steps,
                required_steps,
            } => {
                assert!(current_steps < required_steps);
                assert_eq!(required_steps, 100);
            }
            _ => panic!("Expected UnsafeInsufficientConsensus, got {:?}", safety),
        }
    }

    #[test]
    fn test_safety_safe_when_consensus_reached() {
        // 验证: 达到共识阈值且无活跃冲突时可以 patch
        let config = ExpertRouteConfig::new(4, 2);
        let mut thermal = ExpertThermalManager::new(4)
            .with_eviction_threshold(5);

        for _ in 0..6 {
            thermal.step(&[10, 0, 0, 0]);
        }
        thermal.evict_expert(1);

        let manager = HotPatchManager::new(config)
            .with_consensus_threshold(5);

        let safety = manager.check_expert_patch_safety(1, 0, &thermal, 0);
        match safety {
            PatchSafetyCheck::Safe { consensus_steps, .. } => {
                assert!(consensus_steps >= 5);
            }
            _ => panic!("Expected Safe, got {:?}", safety),
        }
    }

    #[test]
    fn test_summary() {
        let config = ExpertRouteConfig::new(4, 2);
        let mut manager = HotPatchManager::new(config);
        manager.total_patches_applied = 5;
        manager.total_patches_rolled_back = 1;
        manager.patched_experts.push((0, 0));
        manager.collapsed_prefixes.push(0x1234);

        let summary = manager.summary();
        assert_eq!(summary.total_patches_applied, 5);
        assert_eq!(summary.total_patches_rolled_back, 1);
        assert_eq!(summary.patched_expert_count, 1);
        assert_eq!(summary.collapsed_prefix_count, 1);
    }
}
