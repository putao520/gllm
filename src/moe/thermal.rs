//! 专家热度追踪与封杀管理 (SPEC §15.4)
//!
//! ## 核心职责
//! 追踪 MoE 专家的运行时热度状态，实现冷专家封杀与复活:
//! - 基于 ExpertThermalTracker (§13.6) 的热度数据
//! - 冷专家封杀决策 (Deopt / Uncommon Trap)
//! - OSR Bailout 机制 (On-Stack Replacement)
//! - JIT Director 热修补决策
//!
//! ## §15.4 冷板凳专家的全域封杀与复活陷阱
//! 1. 门控 (Gate Router) 计算永远保留 (开销 < 1%)
//! 2. 冷专家权重被 NOP/Deopt 跳转替换
//! 3. 请求触发冷专家时 → Uncommon Trap → DEOPT_REQUEST → 微冷冻 → 回写 .text → 重算

use super::routing::ExpertRouteConfig;

/// 专家热度级别
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExpertHeatLevel {
    /// 热专家: 活跃使用中，权重常驻 GPU L2
    Hot,
    /// 温专家: 偶尔使用，权重在 CPU RAM
    Warm,
    /// 冷专家: 长时间未使用，权重可能被封杀
    Cold,
    /// 封杀: 已被 NOP/Deopt 替换，需要 OSR Bailout 才能恢复
    Evicted,
}

impl ExpertHeatLevel {
    /// 从命中率推导热度级别
    pub fn from_hit_rate(rate: f64, hot_threshold: f64, cold_threshold: f64) -> Self {
        if rate >= hot_threshold {
            ExpertHeatLevel::Hot
        } else if rate >= cold_threshold {
            ExpertHeatLevel::Warm
        } else if rate > 0.0 {
            ExpertHeatLevel::Cold
        } else {
            ExpertHeatLevel::Evicted
        }
    }
}

/// 单个专家的热度状态
#[derive(Debug, Clone)]
pub struct ExpertHeatState {
    /// 专家索引
    pub expert_idx: usize,
    /// 历史命中率 (0.0-1.0)
    pub hit_rate: f64,
    /// 累计命中次数
    pub hit_count: u64,
    /// 累计路由次数
    pub route_count: u64,
    /// 当前热度级别
    pub heat_level: ExpertHeatLevel,
    /// 连续零命中次数 (用于封杀决策)
    pub consecutive_zero_streak: u64,
    /// 最近一次命中时间 (单调递增计数器)
    pub last_hit_step: u64,
    /// 是否已被封杀 (Deopt 跳转已写入)
    pub is_evicted: bool,
    /// 封杀后又被触发的次数 (用于复活统计)
    pub reactivation_count: u64,
}

impl ExpertHeatState {
    fn new(expert_idx: usize) -> Self {
        Self {
            expert_idx,
            hit_rate: 0.0,
            hit_count: 0,
            route_count: 0,
            heat_level: ExpertHeatLevel::Warm,
            consecutive_zero_streak: 0,
            last_hit_step: 0,
            is_evicted: false,
            reactivation_count: 0,
        }
    }
}

/// 封杀决策
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionDecision {
    /// 保持当前状态
    Keep,
    /// 封杀: 用 NOP/Deopt 替换专家权重
    Evict,
    /// 恢复: 回写专家权重
    Reactivate,
}

/// Deopt 请求 (Uncommon Trap 触发时写入)
#[derive(Debug, Clone)]
pub struct DeoptRequest {
    /// 触发 Deopt 的请求 ID
    pub request_id: u64,
    /// 触发 Deopt 的专家索引
    pub expert_idx: usize,
    /// 触发时的层索引
    pub layer_idx: usize,
    /// 触发时间步
    pub step: u64,
}

/// 专家热度管理器
///
/// §15.4: 管理专家的热度状态，做出封杀/恢复决策。
/// 与 JIT Director 联动实现 Hot JMP Patching (§14.4)。
pub struct ExpertThermalManager {
    /// 专家数量
    num_experts: usize,
    /// 每个专家的热度状态
    states: Vec<ExpertHeatState>,
    /// 封杀阈值: 连续零命中次数超过此值则封杀
    eviction_streak_threshold: u64,
    /// 热专家命中率阈值
    hot_threshold: f64,
    /// 冷专家命中率阈值
    cold_threshold: f64,
    /// 当前时间步 (单调递增)
    current_step: u64,
    /// 累计封杀次数
    total_evictions: u64,
    /// 累计恢复次数
    total_reactivations: u64,
    /// 待处理的 Deopt 请求队列
    pending_deopt_requests: Vec<DeoptRequest>,
}

impl ExpertThermalManager {
    /// 创建新的专家热度管理器
    pub fn new(num_experts: usize) -> Self {
        let states = (0..num_experts).map(ExpertHeatState::new).collect();
        Self {
            num_experts,
            states,
            eviction_streak_threshold: 1_000_000, // 100 万次零命中后封杀
            hot_threshold: 0.1,                    // > 10% 命中率 → Hot
            cold_threshold: 0.001,                 // < 0.1% 命中率 → Cold
            current_step: 0,
            total_evictions: 0,
            total_reactivations: 0,
            pending_deopt_requests: Vec::new(),
        }
    }

    /// 配置封杀阈值
    pub fn with_eviction_threshold(mut self, streak_threshold: u64) -> Self {
        self.eviction_streak_threshold = streak_threshold;
        self
    }

    /// 配置热度阈值
    pub fn with_heat_thresholds(mut self, hot: f64, cold: f64) -> Self {
        self.hot_threshold = hot;
        self.cold_threshold = cold;
        self
    }

    /// 推进一步，更新所有专家的热度状态
    ///
    /// # Arguments
    /// * `route_counts` - 本步中每个专家被路由的 token 数
    pub fn step(&mut self, route_counts: &[usize]) {
        self.current_step += 1;

        for (idx, &count) in route_counts.iter().enumerate() {
            if idx >= self.num_experts {
                break;
            }
            let state = &mut self.states[idx];
            state.route_count += 1;

            if count > 0 {
                state.hit_count += 1;
                state.last_hit_step = self.current_step;
                state.consecutive_zero_streak = 0;
            } else {
                state.consecutive_zero_streak += 1;
            }

            // 更新命中率 (滑动窗口近似)
            if state.route_count > 0 {
                state.hit_rate = state.hit_count as f64 / state.route_count as f64;
            }

            // 更新热度级别
            state.heat_level = ExpertHeatLevel::from_hit_rate(
                state.hit_rate,
                self.hot_threshold,
                self.cold_threshold,
            );
            // 已封杀的专家保持 Evicted
            if state.is_evicted {
                state.heat_level = ExpertHeatLevel::Evicted;
            }
        }
    }

    /// 为单个专家做出封杀决策
    pub fn eviction_decision(&self, expert_idx: usize) -> EvictionDecision {
        if expert_idx >= self.num_experts {
            return EvictionDecision::Keep;
        }

        let state = &self.states[expert_idx];

        if state.is_evicted {
            // 已封杀: 检查是否需要恢复
            // 如果该专家在封杀后被触发过 Deopt，说明需要恢复
            if state.reactivation_count > 0 {
                return EvictionDecision::Reactivate;
            }
            return EvictionDecision::Keep;
        }

        // 未封杀: 检查是否需要封杀
        if state.consecutive_zero_streak >= self.eviction_streak_threshold {
            return EvictionDecision::Evict;
        }

        EvictionDecision::Keep
    }

    /// 执行封杀: 将专家标记为 Evicted
    ///
    /// §14.4 / §15.4: JIT Director 用 NOP/Deopt 替换冷专家的访存分支。
    /// 此方法更新 gllm 侧的状态标记，实际的 .text 回写由 JIT Director 完成。
    pub fn evict_expert(&mut self, expert_idx: usize) -> bool {
        if expert_idx >= self.num_experts {
            return false;
        }

        let state = &mut self.states[expert_idx];
        if state.is_evicted {
            return false; // 已经封杀
        }

        state.is_evicted = true;
        state.heat_level = ExpertHeatLevel::Evicted;
        state.reactivation_count = 0;
        self.total_evictions += 1;

        log::info!(
            "Expert {} evicted after {} consecutive zero-hit steps (hit_rate={:.4})",
            expert_idx,
            state.consecutive_zero_streak,
            state.hit_rate,
        );

        true
    }

    /// 恢复专家: 将专家从 Evicted 状态恢复
    ///
    /// §15.4 OSR Bailout:
    /// 1. Thread Block 撞进 Uncommon Trap → 写下 DEOPT_REQUEST
    /// 2. 引擎主循环发现 DEOPT_REQUEST → 调用此方法
    /// 3. JIT Director 回写 .text 恢复专家代码
    /// 4. 异步唤回主存的 4-bit 权重
    /// 5. 挂起的 Request 走一遍回炉重造 (Re-evaluate)
    pub fn reactivate_expert(&mut self, expert_idx: usize) -> bool {
        if expert_idx >= self.num_experts {
            return false;
        }

        let state = &mut self.states[expert_idx];
        if !state.is_evicted {
            return false; // 未被封杀
        }

        state.is_evicted = false;
        state.heat_level = ExpertHeatLevel::Cold;
        state.reactivation_count += 1;
        state.consecutive_zero_streak = 0;
        self.total_reactivations += 1;

        log::info!(
            "Expert {} reactivated (reactivation #{})",
            expert_idx,
            state.reactivation_count,
        );

        true
    }

    /// 处理 Deopt 请求: 记录并触发恢复
    ///
    /// §15.4: 当 Thread Block 撞进 Uncommon Trap 时，
    /// 它写下 DEOPT_REQUEST 并挂起。引擎调用此方法处理。
    pub fn handle_deopt_request(&mut self, request: DeoptRequest) -> DeoptHandlingResult {
        // 记录请求
        self.pending_deopt_requests.push(request.clone());

        let expert_idx = request.expert_idx;
        let state = &mut self.states[expert_idx];

        // 标记需要恢复
        if state.is_evicted {
            state.reactivation_count += 1;

            // 执行恢复
            self.reactivate_expert(expert_idx);

            DeoptHandlingResult::ReactivateAndRerun {
                expert_idx,
                request_id: request.request_id,
            }
        } else {
            // 专家未被封杀，Deopt 可能是错误的
            DeoptHandlingResult::SpuriousDeopt {
                expert_idx,
                request_id: request.request_id,
            }
        }
    }

    /// 获取所有需要封杀的专家 (供 JIT Director 批量处理)
    pub fn experts_to_evict(&self) -> Vec<usize> {
        let mut to_evict = Vec::new();
        for state in &self.states {
            if !state.is_evicted && state.consecutive_zero_streak >= self.eviction_streak_threshold {
                to_evict.push(state.expert_idx);
            }
        }
        to_evict
    }

    /// 获取所有需要恢复的专家 (供 JIT Director 批量处理)
    pub fn experts_to_reactivate(&self) -> Vec<usize> {
        let mut to_reactivate = Vec::new();
        for state in &self.states {
            if state.is_evicted && state.reactivation_count > 0 {
                to_reactivate.push(state.expert_idx);
            }
        }
        to_reactivate
    }

    /// 获取专家热度状态
    pub fn state(&self, expert_idx: usize) -> Option<&ExpertHeatState> {
        self.states.get(expert_idx)
    }

    /// 获取所有热度状态
    pub fn states(&self) -> &[ExpertHeatState] {
        &self.states
    }

    /// 获取热专家列表
    pub fn hot_experts(&self) -> Vec<usize> {
        self.states
            .iter()
            .filter(|s| s.heat_level == ExpertHeatLevel::Hot)
            .map(|s| s.expert_idx)
            .collect()
    }

    /// 获取冷/封杀专家列表
    pub fn cold_or_evicted_experts(&self) -> Vec<usize> {
        self.states
            .iter()
            .filter(|s| matches!(s.heat_level, ExpertHeatLevel::Cold | ExpertHeatLevel::Evicted))
            .map(|s| s.expert_idx)
            .collect()
    }

    /// 获取统计摘要
    pub fn summary(&self) -> ThermalSummary {
        let hot_count = self.states.iter().filter(|s| s.heat_level == ExpertHeatLevel::Hot).count();
        let warm_count = self.states.iter().filter(|s| s.heat_level == ExpertHeatLevel::Warm).count();
        let cold_count = self.states.iter().filter(|s| s.heat_level == ExpertHeatLevel::Cold).count();
        let evicted_count = self.states.iter().filter(|s| s.heat_level == ExpertHeatLevel::Evicted).count();

        ThermalSummary {
            num_experts: self.num_experts,
            hot_count,
            warm_count,
            cold_count,
            evicted_count,
            total_evictions: self.total_evictions,
            total_reactivations: self.total_reactivations,
            current_step: self.current_step,
            pending_deopt_count: self.pending_deopt_requests.len(),
        }
    }

    /// 获取待处理的 Deopt 请求
    pub fn pending_deopt_requests(&self) -> &[DeoptRequest] {
        &self.pending_deopt_requests
    }

    /// 清空已处理的 Deopt 请求
    pub fn clear_deopt_requests(&mut self) {
        self.pending_deopt_requests.clear();
    }

    /// 获取专家数量
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }
}

/// Deopt 处理结果
#[derive(Debug, Clone)]
pub enum DeoptHandlingResult {
    /// 需要恢复专家并重新执行请求
    ReactivateAndRerun {
        expert_idx: usize,
        request_id: u64,
    },
    /// 虚假 Deopt (专家未被封杀)
    SpuriousDeopt {
        expert_idx: usize,
        request_id: u64,
    },
}

/// 热度管理统计摘要
#[derive(Debug, Clone)]
pub struct ThermalSummary {
    pub num_experts: usize,
    pub hot_count: usize,
    pub warm_count: usize,
    pub cold_count: usize,
    pub evicted_count: usize,
    pub total_evictions: u64,
    pub total_reactivations: u64,
    pub current_step: u64,
    pub pending_deopt_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heat_level_classification() {
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.5, 0.1, 0.001), ExpertHeatLevel::Hot);
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.05, 0.1, 0.001), ExpertHeatLevel::Warm);
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.0005, 0.1, 0.001), ExpertHeatLevel::Cold);
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.0, 0.1, 0.001), ExpertHeatLevel::Evicted);
    }

    #[test]
    fn test_expert_thermal_manager_eviction() {
        let mut manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(100);

        // 模拟 100 步: expert 0 一直被使用，其他专家从不被使用
        for _ in 0..100 {
            manager.step(&[10, 0, 0, 0]); // 只有 expert 0 有路由
        }

        // Expert 0 应该是热专家
        assert_eq!(manager.state(0).unwrap().heat_level, ExpertHeatLevel::Hot);

        // Experts 1-3 应该需要封杀
        let to_evict = manager.experts_to_evict();
        assert_eq!(to_evict.len(), 3);
        assert!(to_evict.contains(&1));
        assert!(to_evict.contains(&2));
        assert!(to_evict.contains(&3));

        // 执行封杀
        assert!(manager.evict_expert(1));
        assert!(manager.state(1).unwrap().is_evicted);
        assert_eq!(manager.state(1).unwrap().heat_level, ExpertHeatLevel::Evicted);
    }

    #[test]
    fn test_expert_thermal_manager_reactivation() {
        let mut manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(10);

        // 封杀 expert 2
        for _ in 0..11 {
            manager.step(&[10, 5, 0, 3]);
        }
        manager.evict_expert(2);

        // 触发 Deopt
        let deopt = DeoptRequest {
            request_id: 42,
            expert_idx: 2,
            layer_idx: 5,
            step: manager.current_step,
        };

        let result = manager.handle_deopt_request(deopt);
        match result {
            DeoptHandlingResult::ReactivateAndRerun { expert_idx, request_id } => {
                assert_eq!(expert_idx, 2);
                assert_eq!(request_id, 42);
            }
            DeoptHandlingResult::SpuriousDeopt { .. } => {
                panic!("Expected ReactivateAndRerun for evicted expert");
            }
        }

        // Expert 2 应该已恢复
        assert!(!manager.state(2).unwrap().is_evicted);
        assert_eq!(manager.state(2).unwrap().heat_level, ExpertHeatLevel::Cold);
    }

    #[test]
    fn test_thermal_summary() {
        let mut manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(5);

        for _ in 0..6 {
            manager.step(&[10, 5, 0, 3]);
        }
        manager.evict_expert(2);

        let summary = manager.summary();
        assert_eq!(summary.num_experts, 4);
        assert!(summary.hot_count >= 1);
        assert_eq!(summary.evicted_count, 1);
        assert_eq!(summary.total_evictions, 1);
    }

    #[test]
    fn test_hot_experts_list() {
        let mut manager = ExpertThermalManager::new(4);

        // Expert 0 和 1 频繁使用
        for _ in 0..100 {
            manager.step(&[50, 50, 0, 0]);
        }

        let hot = manager.hot_experts();
        assert_eq!(hot.len(), 2);
        assert!(hot.contains(&0));
        assert!(hot.contains(&1));
    }

    #[test]
    fn test_cold_or_evicted_experts() {
        let mut manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(5);

        for _ in 0..6 {
            manager.step(&[50, 0, 0, 0]);
        }
        manager.evict_expert(2);

        let cold = manager.cold_or_evicted_experts();
        assert!(cold.contains(&1)); // cold
        assert!(cold.contains(&2)); // evicted
        assert!(cold.contains(&3)); // cold
    }

    #[test]
    fn test_spurious_deopt() {
        let mut manager = ExpertThermalManager::new(4);

        // Expert 1 从未被封杀
        let deopt = DeoptRequest {
            request_id: 1,
            expert_idx: 1,
            layer_idx: 0,
            step: 0,
        };

        let result = manager.handle_deopt_request(deopt);
        assert!(matches!(result, DeoptHandlingResult::SpuriousDeopt { expert_idx: 1, .. }));
    }

    #[test]
    fn test_double_eviction_prevention() {
        let mut manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(5);

        for _ in 0..6 {
            manager.step(&[10, 0, 0, 0]);
        }
        manager.evict_expert(1);

        // 再次封杀应该失败
        assert!(!manager.evict_expert(1));
    }
}
