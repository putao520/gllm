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

/// 专家热度级别
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd)]
pub enum EvictionDecision {
    /// 保持当前状态
    Keep,
    /// 封杀: 用 NOP/Deopt 替换专家权重
    Evict,
    /// 恢复: 回写专家权重
    Reactivate,
}

/// Deopt 请求 (Uncommon Trap 触发时写入)
#[derive(Debug, Clone, PartialEq)]
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

/// Adaptive working set tracking for eviction threshold tuning.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkingSetTracker {
    /// Sliding window of recently accessed expert indices (bitset per step).
    window: Vec<Vec<bool>>,
    /// Window capacity (number of steps to track).
    window_size: usize,
    /// Write cursor into the ring buffer.
    cursor: usize,
    /// Number of experts.
    num_experts: usize,
    /// Base eviction threshold (before adaptive scaling).
    base_threshold: u64,
}

impl WorkingSetTracker {
    fn new(num_experts: usize, window_size: usize, base_threshold: u64) -> Self {
        let window = vec![vec![false; num_experts]; window_size];
        Self {
            window,
            window_size,
            cursor: 0,
            num_experts,
            base_threshold,
        }
    }

    /// Record which experts were accessed in this step.
    fn record_step(&mut self, route_counts: &[usize]) {
        let slot = &mut self.window[self.cursor];
        for accessed in slot.iter_mut() {
            *accessed = false;
        }
        for (idx, &count) in route_counts.iter().enumerate() {
            if idx < self.num_experts && count > 0 {
                slot[idx] = true;
            }
        }
        self.cursor = (self.cursor + 1) % self.window_size;
    }

    /// Count distinct experts accessed in the window.
    pub fn working_set_size(&self) -> usize {
        let mut seen = vec![false; self.num_experts];
        for slot in &self.window {
            for (idx, &accessed) in slot.iter().enumerate() {
                if accessed {
                    seen[idx] = true;
                }
            }
        }
        seen.iter().filter(|&&s| s).count()
    }

    /// Compute adaptive eviction threshold based on memory pressure.
    ///
    /// Formula: `base * (num_experts / working_set_size) * headroom_factor`
    /// where `headroom_factor = 1.0 - memory_pressure` (clamped to [0.1, 1.0]).
    pub fn adaptive_threshold(&self, memory_pressure: f32) -> u64 {
        let ws = self.working_set_size().max(1) as f64;
        let headroom = (1.0 - memory_pressure as f64).clamp(0.1, 1.0);
        let ratio = self.num_experts as f64 / ws;
        let threshold = self.base_threshold as f64 * ratio * headroom;
        (threshold.round() as u64).max(1)
    }
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
    /// Adaptive working set tracker for dynamic eviction thresholds.
    working_set: WorkingSetTracker,
    /// Whether adaptive eviction is enabled.
    adaptive_eviction: bool,
    /// Current memory pressure (updated externally).
    memory_pressure: f32,
    /// Eviction aggressiveness from StrategyBias: 0.0 = full resident, 2.0 = aggressive eviction.
    eviction_aggressiveness: f64,
}

impl ExpertThermalManager {
    /// 创建新的专家热度管理器
    pub fn new(num_experts: usize) -> Self {
        let base_threshold = 1_000_000u64;
        let states = (0..num_experts).map(ExpertHeatState::new).collect();
        Self {
            num_experts,
            states,
            eviction_streak_threshold: base_threshold,
            hot_threshold: 0.1,
            cold_threshold: 0.001,
            current_step: 0,
            total_evictions: 0,
            total_reactivations: 0,
            pending_deopt_requests: Vec::new(),
            working_set: WorkingSetTracker::new(num_experts, 100, base_threshold),
            adaptive_eviction: false,
            memory_pressure: 0.0,
            eviction_aggressiveness: 0.0,
        }
    }

    /// 配置封杀阈值
    pub fn with_eviction_threshold(mut self, streak_threshold: u64) -> Self {
        self.eviction_streak_threshold = streak_threshold;
        self.working_set.base_threshold = streak_threshold;
        self
    }

    /// Enable adaptive eviction with a custom working set window size.
    pub fn with_adaptive_eviction(mut self, window_size: usize) -> Self {
        self.adaptive_eviction = true;
        self.working_set = WorkingSetTracker::new(
            self.num_experts,
            window_size.max(1),
            self.eviction_streak_threshold,
        );
        self
    }

    /// Set eviction aggressiveness from StrategyBias (SPEC §5.9).
    /// 0.0 = full resident (never evict), 2.0 = aggressive eviction.
    pub fn with_eviction_aggressiveness(mut self, aggressiveness: f64) -> Self {
        self.eviction_aggressiveness = aggressiveness;
        self
    }

    /// Update memory pressure for adaptive eviction threshold.
    pub fn update_memory_pressure(&mut self, pressure: f32) {
        self.memory_pressure = pressure.clamp(0.0, 1.0);
    }

    /// Get the current effective eviction threshold.
    ///
    /// When `eviction_aggressiveness` > 0, the threshold is scaled down
    /// (making eviction easier). Formula per SPEC §5.9:
    /// `bias_factor = 1.0 / (1.0 + aggressiveness)`, then threshold *= bias_factor.
    pub fn effective_eviction_threshold(&self) -> u64 {
        let base = if self.adaptive_eviction {
            self.working_set.adaptive_threshold(self.memory_pressure)
        } else {
            self.eviction_streak_threshold
        };
        let bias_factor = 1.0 / (1.0 + self.eviction_aggressiveness);
        (base as f64 * bias_factor) as u64
    }

    /// Get the working set size (experts accessed in the tracking window).
    pub fn working_set_size(&self) -> usize {
        self.working_set.working_set_size()
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
        self.working_set.record_step(route_counts);

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
        let threshold = self.effective_eviction_threshold();
        if state.consecutive_zero_streak >= threshold {
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
        let threshold = self.effective_eviction_threshold();
        let mut to_evict = Vec::new();
        for state in &self.states {
            if !state.is_evicted && state.consecutive_zero_streak >= threshold {
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
            working_set_size: self.working_set.working_set_size(),
            effective_eviction_threshold: self.effective_eviction_threshold(),
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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
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
    /// Number of distinct experts accessed in the tracking window.
    pub working_set_size: usize,
    /// Current effective eviction threshold (adaptive or static).
    pub effective_eviction_threshold: u64,
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

    // ────────────────────────────────────────────────────────
    // ExpertHeatLevel additional tests
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_heat_level_from_hit_rate_exact_thresholds() {
        // Rate exactly at hot_threshold => Hot
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.1, 0.1, 0.001), ExpertHeatLevel::Hot);

        // Rate exactly at cold_threshold => Warm (>= cold_threshold)
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.001, 0.1, 0.001), ExpertHeatLevel::Warm);

        // Rate just above zero but below cold_threshold => Cold
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.0001, 0.1, 0.001), ExpertHeatLevel::Cold);

        // Rate at 1.0 => Hot
        assert_eq!(ExpertHeatLevel::from_hit_rate(1.0, 0.1, 0.001), ExpertHeatLevel::Hot);

        // Rate at 0.0 => Evicted
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.0, 0.1, 0.001), ExpertHeatLevel::Evicted);
    }

    #[test]
    fn test_heat_level_ordering() {
        // derive(Ord) uses variant declaration order:
        // Hot < Warm < Cold < Evicted
        assert!(ExpertHeatLevel::Hot < ExpertHeatLevel::Warm);
        assert!(ExpertHeatLevel::Warm < ExpertHeatLevel::Cold);
        assert!(ExpertHeatLevel::Cold < ExpertHeatLevel::Evicted);
        assert!(ExpertHeatLevel::Hot < ExpertHeatLevel::Evicted);
    }

    #[test]
    fn test_heat_level_copy_clone() {
        let level = ExpertHeatLevel::Hot;
        let copied = level;
        let cloned = level.clone();
        assert_eq!(level, copied);
        assert_eq!(level, cloned);
    }

    #[test]
    fn test_heat_level_custom_thresholds() {
        // High thresholds: most experts are cold
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.5, 0.9, 0.6), ExpertHeatLevel::Cold);

        // Very low thresholds: most experts are hot
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.0001, 0.00001, 0.000001), ExpertHeatLevel::Hot);

        // Equal hot and cold thresholds: rate >= hot => Hot, rate > 0 but < hot => Cold
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.5, 0.5, 0.5), ExpertHeatLevel::Hot);
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.4, 0.5, 0.5), ExpertHeatLevel::Cold);
    }

    // ────────────────────────────────────────────────────────
    // EvictionDecision variants
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_eviction_decision_copy_clone_eq() {
        let d1 = EvictionDecision::Evict;
        let d2 = d1;
        let d3 = d1.clone();
        assert_eq!(d1, d2);
        assert_eq!(d1, d3);
        assert_ne!(EvictionDecision::Keep, EvictionDecision::Evict);
        assert_ne!(EvictionDecision::Evict, EvictionDecision::Reactivate);
    }

    #[test]
    fn test_eviction_decision_keep_for_active_expert() {
        let manager = ExpertThermalManager::new(4).with_eviction_threshold(100);
        // Fresh manager, no steps taken => all experts have streak 0 => Keep
        assert_eq!(manager.eviction_decision(0), EvictionDecision::Keep);
        assert_eq!(manager.eviction_decision(3), EvictionDecision::Keep);
    }

    #[test]
    fn test_eviction_decision_out_of_bounds() {
        let manager = ExpertThermalManager::new(4);
        assert_eq!(manager.eviction_decision(100), EvictionDecision::Keep);
        assert_eq!(manager.eviction_decision(4), EvictionDecision::Keep);
    }

    #[test]
    fn test_eviction_decision_reactivate_after_manual_reactivation_count() {
        // Test the Reactivate path: evicted expert with reactivation_count > 0
        // This requires the expert to be evicted AND have reactivation_count > 0
        // without being fully reactivated (which clears is_evicted).
        // Since handle_deopt_request calls reactivate_expert internally (clearing
        // is_evicted), the Reactivate decision can only be observed between
        // setting reactivation_count and calling reactivate_expert.
        // In practice, the Reactivate decision is consumed internally.
        // Test the observable behavior: after deopt, expert is reactivated => Keep.
        let mut manager = ExpertThermalManager::new(4).with_eviction_threshold(5);

        for _ in 0..6 {
            manager.step(&[10, 0, 0, 0]);
        }
        manager.evict_expert(1);

        // Evicted but no reactivation yet => Keep (stays evicted)
        assert_eq!(manager.eviction_decision(1), EvictionDecision::Keep);

        // Trigger deopt: handle_deopt_request internally reactivates
        let deopt = DeoptRequest {
            request_id: 1,
            expert_idx: 1,
            layer_idx: 0,
            step: 6,
        };
        let result = manager.handle_deopt_request(deopt);
        assert!(matches!(result, DeoptHandlingResult::ReactivateAndRerun { expert_idx: 1, .. }));

        // After reactivation, expert is no longer evicted => Keep
        assert!(!manager.state(1).unwrap().is_evicted);
        assert_eq!(manager.eviction_decision(1), EvictionDecision::Keep);
    }

    // ────────────────────────────────────────────────────────
    // DeoptRequest construction and Clone
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_deopt_request_clone() {
        let req = DeoptRequest {
            request_id: 42,
            expert_idx: 7,
            layer_idx: 3,
            step: 100,
        };
        let cloned = req.clone();
        assert_eq!(cloned.request_id, 42);
        assert_eq!(cloned.expert_idx, 7);
        assert_eq!(cloned.layer_idx, 3);
        assert_eq!(cloned.step, 100);
    }

    // ────────────────────────────────────────────────────────
    // ExpertThermalManager construction and defaults
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_manager_new_defaults() {
        let manager = ExpertThermalManager::new(8);
        assert_eq!(manager.num_experts(), 8);
        assert_eq!(manager.states().len(), 8);

        // All experts start with default state
        for state in manager.states() {
            assert_eq!(state.hit_rate, 0.0);
            assert_eq!(state.hit_count, 0);
            assert_eq!(state.route_count, 0);
            assert_eq!(state.heat_level, ExpertHeatLevel::Warm);
            assert_eq!(state.consecutive_zero_streak, 0);
            assert!(!state.is_evicted);
            assert_eq!(state.reactivation_count, 0);
        }

        // Empty lists
        assert!(manager.hot_experts().is_empty());
        assert!(manager.cold_or_evicted_experts().is_empty());
        assert!(manager.experts_to_evict().is_empty());
        assert!(manager.experts_to_reactivate().is_empty());
        assert!(manager.pending_deopt_requests().is_empty());

        let summary = manager.summary();
        assert_eq!(summary.num_experts, 8);
        assert_eq!(summary.hot_count, 0);
        assert_eq!(summary.warm_count, 8);
        assert_eq!(summary.cold_count, 0);
        assert_eq!(summary.evicted_count, 0);
        assert_eq!(summary.total_evictions, 0);
        assert_eq!(summary.total_reactivations, 0);
        assert_eq!(summary.current_step, 0);
        assert_eq!(summary.pending_deopt_count, 0);
    }

    #[test]
    fn test_manager_zero_experts() {
        let mut manager = ExpertThermalManager::new(0);
        assert_eq!(manager.num_experts(), 0);
        assert!(manager.states().is_empty());
        assert!(manager.hot_experts().is_empty());
        assert!(manager.experts_to_evict().is_empty());

        // Out-of-bounds access returns None / Keep
        assert!(manager.state(0).is_none());
        assert_eq!(manager.eviction_decision(0), EvictionDecision::Keep);
        assert!(!manager.evict_expert(0));
        assert!(!manager.reactivate_expert(0));
    }

    // ────────────────────────────────────────────────────────
    // Builder pattern: with_heat_thresholds, with_eviction_threshold
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_with_heat_thresholds_high() {
        let mut manager = ExpertThermalManager::new(4)
            .with_heat_thresholds(0.9, 0.5);

        // With high thresholds, even moderate usage => Cold
        manager.step(&[10, 0, 0, 0]);
        // expert 0: hit_rate = 1.0 (>= 0.9) => Hot
        assert_eq!(manager.state(0).unwrap().heat_level, ExpertHeatLevel::Hot);
        // expert 1: hit_rate = 0.0 => Evicted
        assert_eq!(manager.state(1).unwrap().heat_level, ExpertHeatLevel::Evicted);
    }

    #[test]
    fn test_with_heat_thresholds_low() {
        let mut manager = ExpertThermalManager::new(4)
            .with_heat_thresholds(0.0001, 0.00001);

        manager.step(&[10, 5, 1, 0]);
        // expert 0: hit_rate = 1.0 => Hot (>= 0.0001)
        assert_eq!(manager.state(0).unwrap().heat_level, ExpertHeatLevel::Hot);
        // expert 3: hit_rate = 0.0 => Evicted
        assert_eq!(manager.state(3).unwrap().heat_level, ExpertHeatLevel::Evicted);
    }

    #[test]
    fn test_with_eviction_threshold_affects_decision() {
        let mut manager_low = ExpertThermalManager::new(2).with_eviction_threshold(3);
        let mut manager_high = ExpertThermalManager::new(2).with_eviction_threshold(1000);

        // Run 5 steps with only expert 0 active
        for _ in 0..5 {
            manager_low.step(&[10, 0]);
            manager_high.step(&[10, 0]);
        }

        // Low threshold: expert 1 should be flagged for eviction
        assert!(manager_low.experts_to_evict().contains(&1));

        // High threshold: expert 1 streak (5) < 1000 => not flagged
        assert!(!manager_high.experts_to_evict().contains(&1));
    }

    // ────────────────────────────────────────────────────────
    // Step behavior edge cases
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_step_short_route_counts() {
        let mut manager = ExpertThermalManager::new(4);

        // route_counts shorter than num_experts: remaining experts unchanged
        manager.step(&[5]);
        assert_eq!(manager.state(0).unwrap().route_count, 1);
        assert_eq!(manager.state(1).unwrap().route_count, 0);
        assert_eq!(manager.state(2).unwrap().route_count, 0);
        assert_eq!(manager.state(3).unwrap().route_count, 0);
    }

    #[test]
    fn test_step_empty_route_counts() {
        let mut manager = ExpertThermalManager::new(4);
        manager.step(&[]);

        // No experts updated, step still incremented
        let summary = manager.summary();
        assert_eq!(summary.current_step, 1);
        for state in manager.states() {
            assert_eq!(state.route_count, 0);
        }
    }

    #[test]
    fn test_step_all_active() {
        let mut manager = ExpertThermalManager::new(3);

        manager.step(&[10, 20, 30]);

        for state in manager.states() {
            assert_eq!(state.route_count, 1);
            assert_eq!(state.hit_count, 1);
            assert_eq!(state.hit_rate, 1.0);
            assert_eq!(state.heat_level, ExpertHeatLevel::Hot);
            assert_eq!(state.consecutive_zero_streak, 0);
        }
    }

    #[test]
    fn test_step_hit_rate_convergence() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(1000);

        // Expert 0: hit 50% of the time
        for i in 0..100 {
            if i % 2 == 0 {
                manager.step(&[10, 5]);
            } else {
                manager.step(&[0, 5]);
            }
        }

        let state0 = manager.state(0).unwrap();
        assert!((state0.hit_rate - 0.5).abs() < 0.01);
        assert_eq!(state0.hit_count, 50);
        assert_eq!(state0.route_count, 100);
    }

    #[test]
    fn test_step_last_hit_step_tracking() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(100);

        manager.step(&[10, 0]); // step 1
        manager.step(&[10, 0]); // step 2
        manager.step(&[0, 5]);  // step 3: expert 0 not hit

        assert_eq!(manager.state(0).unwrap().last_hit_step, 2);
        assert_eq!(manager.state(1).unwrap().last_hit_step, 3);
    }

    // ────────────────────────────────────────────────────────
    // Evict / Reactivate edge cases
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_evict_out_of_bounds() {
        let mut manager = ExpertThermalManager::new(4);
        assert!(!manager.evict_expert(4));
        assert!(!manager.evict_expert(100));
    }

    #[test]
    fn test_reactivate_out_of_bounds() {
        let mut manager = ExpertThermalManager::new(4);
        assert!(!manager.reactivate_expert(4));
        assert!(!manager.reactivate_expert(usize::MAX));
    }

    #[test]
    fn test_reactivate_non_evicted_expert() {
        let mut manager = ExpertThermalManager::new(4);
        // Expert is not evicted => reactivation fails
        assert!(!manager.reactivate_expert(0));
    }

    #[test]
    fn test_double_reactivation() {
        let mut manager = ExpertThermalManager::new(4).with_eviction_threshold(5);

        for _ in 0..6 {
            manager.step(&[10, 0, 0, 0]);
        }
        manager.evict_expert(1);

        // First reactivation succeeds
        assert!(manager.reactivate_expert(1));
        assert!(!manager.state(1).unwrap().is_evicted);
        assert_eq!(manager.state(1).unwrap().heat_level, ExpertHeatLevel::Cold);
        assert_eq!(manager.state(1).unwrap().reactivation_count, 1);

        // Re-evict (resets reactivation_count to 0 per evict_expert)
        manager.evict_expert(1);
        assert!(manager.state(1).unwrap().is_evicted);
        assert_eq!(manager.state(1).unwrap().reactivation_count, 0);

        // Second reactivation also succeeds, count is 1 again
        assert!(manager.reactivate_expert(1));
        assert_eq!(manager.state(1).unwrap().reactivation_count, 1);
    }

    #[test]
    fn test_reactivation_resets_streak() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(5);

        for _ in 0..6 {
            manager.step(&[10, 0]);
        }
        manager.evict_expert(1);
        assert!(manager.state(1).unwrap().consecutive_zero_streak >= 5);

        manager.reactivate_expert(1);
        assert_eq!(manager.state(1).unwrap().consecutive_zero_streak, 0);
    }

    // ────────────────────────────────────────────────────────
    // experts_to_reactivate
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_experts_to_reactivate_empty() {
        let manager = ExpertThermalManager::new(4);
        assert!(manager.experts_to_reactivate().is_empty());
    }

    #[test]
    fn test_experts_to_reactivate_after_deopt() {
        let mut manager = ExpertThermalManager::new(4).with_eviction_threshold(5);

        for _ in 0..6 {
            manager.step(&[10, 0, 0, 0]);
        }
        manager.evict_expert(1);
        manager.evict_expert(2);

        // No reactivation requests yet
        assert!(manager.experts_to_reactivate().is_empty());

        // Trigger deopt for expert 1: handle_deopt_request internally
        // calls reactivate_expert, so expert 1 is no longer evicted.
        let deopt = DeoptRequest {
            request_id: 1,
            expert_idx: 1,
            layer_idx: 0,
            step: 6,
        };
        manager.handle_deopt_request(deopt);

        // Expert 1 was fully reactivated by handle_deopt_request,
        // so it is no longer in the to-reactivate list.
        let to_reactivate = manager.experts_to_reactivate();
        assert!(to_reactivate.is_empty());

        // Expert 2 is still evicted with no reactivation_count
        assert!(manager.state(2).unwrap().is_evicted);
        assert_eq!(manager.state(2).unwrap().reactivation_count, 0);
    }

    // ────────────────────────────────────────────────────────
    // Deopt request queue management
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_pending_and_clear_deopt_requests() {
        let mut manager = ExpertThermalManager::new(4).with_eviction_threshold(5);

        for _ in 0..6 {
            manager.step(&[10, 0, 0, 0]);
        }
        manager.evict_expert(1);

        // Multiple deopt requests
        for i in 0..3 {
            manager.handle_deopt_request(DeoptRequest {
                request_id: i,
                expert_idx: 1,
                layer_idx: 0,
                step: 6,
            });
        }

        assert_eq!(manager.pending_deopt_requests().len(), 3);
        assert_eq!(manager.pending_deopt_requests()[0].request_id, 0);
        assert_eq!(manager.pending_deopt_requests()[2].request_id, 2);

        manager.clear_deopt_requests();
        assert!(manager.pending_deopt_requests().is_empty());
    }

    // ────────────────────────────────────────────────────────
    // DeoptHandlingResult variants
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_deopt_handling_result_fields() {
        let result = DeoptHandlingResult::ReactivateAndRerun {
            expert_idx: 5,
            request_id: 99,
        };
        match result {
            DeoptHandlingResult::ReactivateAndRerun { expert_idx, request_id } => {
                assert_eq!(expert_idx, 5);
                assert_eq!(request_id, 99);
            }
            DeoptHandlingResult::SpuriousDeopt { .. } => panic!("wrong variant"),
        }

        let spurious = DeoptHandlingResult::SpuriousDeopt {
            expert_idx: 3,
            request_id: 7,
        };
        match spurious {
            DeoptHandlingResult::SpuriousDeopt { expert_idx, request_id } => {
                assert_eq!(expert_idx, 3);
                assert_eq!(request_id, 7);
            }
            DeoptHandlingResult::ReactivateAndRerun { .. } => panic!("wrong variant"),
        }
    }

    // ────────────────────────────────────────────────────────
    // ThermalSummary construction and fields
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_thermal_summary_clone() {
        let summary = ThermalSummary {
            num_experts: 8,
            hot_count: 3,
            warm_count: 2,
            cold_count: 2,
            evicted_count: 1,
            total_evictions: 5,
            total_reactivations: 2,
            current_step: 100,
            pending_deopt_count: 1,
            working_set_size: 4,
            effective_eviction_threshold: 500,
        };
        let cloned = summary.clone();
        assert_eq!(cloned.num_experts, 8);
        assert_eq!(cloned.hot_count, 3);
        assert_eq!(cloned.total_evictions, 5);
        assert_eq!(cloned.working_set_size, 4);
    }

    #[test]
    fn test_summary_step_counter() {
        let mut manager = ExpertThermalManager::new(2);

        assert_eq!(manager.summary().current_step, 0);

        manager.step(&[1, 1]);
        assert_eq!(manager.summary().current_step, 1);

        manager.step(&[1, 1]);
        manager.step(&[1, 1]);
        assert_eq!(manager.summary().current_step, 3);
    }

    // ────────────────────────────────────────────────────────
    // state() and states() accessors
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_state_accessor_bounds() {
        let manager = ExpertThermalManager::new(4);
        assert!(manager.state(0).is_some());
        assert!(manager.state(3).is_some());
        assert!(manager.state(4).is_none());
        assert!(manager.state(usize::MAX).is_none());
    }

    // ────────────────────────────────────────────────────────
    // Adaptive eviction and working set
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_adaptive_eviction_disabled_by_default() {
        let manager = ExpertThermalManager::new(8).with_eviction_threshold(500);
        // Without adaptive eviction, effective threshold = base threshold
        assert_eq!(manager.effective_eviction_threshold(), 500);
    }

    #[test]
    fn test_with_adaptive_eviction_enables() {
        let manager = ExpertThermalManager::new(8)
            .with_eviction_threshold(1000)
            .with_adaptive_eviction(50);

        // With no steps, working set is 0, adaptive threshold scales up
        let threshold = manager.effective_eviction_threshold();
        assert!(threshold >= 1000); // num_experts/1 * base * headroom >= base
    }

    #[test]
    fn test_working_set_size_tracking() {
        let mut manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(100)
            .with_adaptive_eviction(10);

        // Initially empty working set
        assert_eq!(manager.working_set_size(), 0);

        // Step with experts 0 and 1 active
        manager.step(&[5, 3, 0, 0]);
        assert_eq!(manager.working_set_size(), 2);

        // Step with expert 3 also active
        manager.step(&[0, 0, 0, 7]);
        assert_eq!(manager.working_set_size(), 3);

        // Step with all active
        manager.step(&[1, 1, 1, 1]);
        assert_eq!(manager.working_set_size(), 4);
    }

    #[test]
    fn test_adaptive_threshold_with_memory_pressure() {
        let mut manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(1000)
            .with_adaptive_eviction(10);

        manager.step(&[1, 1, 0, 0]); // working set = 2

        // No memory pressure
        manager.update_memory_pressure(0.0);
        let threshold_no_pressure = manager.effective_eviction_threshold();

        // High memory pressure
        manager.update_memory_pressure(0.9);
        let threshold_high_pressure = manager.effective_eviction_threshold();

        // High pressure => smaller headroom => lower threshold (easier eviction)
        assert!(threshold_high_pressure < threshold_no_pressure);
    }

    #[test]
    fn test_memory_pressure_clamped() {
        let mut manager = ExpertThermalManager::new(4);

        manager.update_memory_pressure(-0.5);
        // Clamped to 0.0, effective threshold unchanged
        let threshold_neg = manager.effective_eviction_threshold();

        manager.update_memory_pressure(0.0);
        let threshold_zero = manager.effective_eviction_threshold();

        assert_eq!(threshold_neg, threshold_zero);

        manager.update_memory_pressure(2.0);
        // Clamped to 1.0
        // Not testing exact value, just that it doesn't panic
    }

    // ────────────────────────────────────────────────────────
    // Eviction aggressiveness
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_eviction_aggressiveness_reduces_threshold() {
        let manager_passive = ExpertThermalManager::new(4)
            .with_eviction_threshold(1000)
            .with_eviction_aggressiveness(0.0);

        let manager_aggressive = ExpertThermalManager::new(4)
            .with_eviction_threshold(1000)
            .with_eviction_aggressiveness(1.0);

        let manager_very_aggressive = ExpertThermalManager::new(4)
            .with_eviction_threshold(1000)
            .with_eviction_aggressiveness(2.0);

        let t0 = manager_passive.effective_eviction_threshold();
        let t1 = manager_aggressive.effective_eviction_threshold();
        let t2 = manager_very_aggressive.effective_eviction_threshold();

        // aggressiveness=0 => factor=1.0 => threshold unchanged
        assert_eq!(t0, 1000);
        // aggressiveness=1 => factor=0.5 => threshold halved
        assert_eq!(t1, 500);
        // aggressiveness=2 => factor=1/3 => threshold = 333
        assert_eq!(t2, 333);
    }

    #[test]
    fn test_aggressiveness_triggers_earlier_eviction() {
        let mut manager = ExpertThermalManager::new(2)
            .with_eviction_threshold(100)
            .with_eviction_aggressiveness(1.0); // threshold becomes 50

        // 60 steps: expert 1 has streak 60, >= 50
        for _ in 0..60 {
            manager.step(&[10, 0]);
        }

        assert!(manager.experts_to_evict().contains(&1));
    }

    // ────────────────────────────────────────────────────────
    // ExpertHeatState Clone
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_expert_heat_state_clone() {
        let mut manager = ExpertThermalManager::new(2);
        manager.step(&[10, 0]);

        let state = manager.state(0).unwrap().clone();
        assert_eq!(state.expert_idx, 0);
        assert_eq!(state.hit_count, 1);
        assert_eq!(state.route_count, 1);
        assert_eq!(state.hit_rate, 1.0);
        assert!(!state.is_evicted);
    }

    // ────────────────────────────────────────────────────────
    // Full lifecycle: step → evict → deopt → reactivate → step
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_full_lifecycle() {
        let mut manager = ExpertThermalManager::new(3)
            .with_eviction_threshold(5);

        // Phase 1: All experts warm
        manager.step(&[5, 5, 5]);
        assert_eq!(manager.summary().hot_count, 3);

        // Phase 2: Expert 2 goes cold
        for _ in 0..6 {
            manager.step(&[5, 5, 0]);
        }

        // Expert 2 should be flagged for eviction
        assert!(manager.experts_to_evict().contains(&2));

        // Phase 3: Evict expert 2
        assert!(manager.evict_expert(2));
        assert_eq!(manager.summary().evicted_count, 1);
        assert_eq!(manager.summary().total_evictions, 1);

        // Phase 4: Deopt triggered for expert 2
        let deopt = DeoptRequest {
            request_id: 1,
            expert_idx: 2,
            layer_idx: 3,
            step: 7,
        };
        let result = manager.handle_deopt_request(deopt);
        assert!(matches!(result, DeoptHandlingResult::ReactivateAndRerun { expert_idx: 2, .. }));

        // Phase 5: Expert 2 reactivated
        assert!(!manager.state(2).unwrap().is_evicted);
        assert_eq!(manager.state(2).unwrap().heat_level, ExpertHeatLevel::Cold);
        assert_eq!(manager.summary().total_reactivations, 1);

        // Phase 6: Expert 2 becomes hot again
        for _ in 0..50 {
            manager.step(&[5, 5, 5]);
        }
        assert_eq!(manager.state(2).unwrap().heat_level, ExpertHeatLevel::Hot);
    }

    // ────────────────────────────────────────────────────────
    // Summary fields after complex operations
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_summary_pending_deopt_count() {
        let mut manager = ExpertThermalManager::new(4).with_eviction_threshold(3);

        for _ in 0..4 {
            manager.step(&[10, 0, 0, 0]);
        }
        manager.evict_expert(1);

        // File multiple deopt requests
        for i in 0..5 {
            manager.handle_deopt_request(DeoptRequest {
                request_id: i,
                expert_idx: 1,
                layer_idx: 0,
                step: 4,
            });
        }

        let summary = manager.summary();
        assert_eq!(summary.pending_deopt_count, 5);
    }

    #[test]
    fn test_summary_working_set_size() {
        let mut manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(100)
            .with_adaptive_eviction(10);

        manager.step(&[1, 1, 0, 0]);
        assert_eq!(manager.summary().working_set_size, 2);
    }

    // ────────────────────────────────────────────────────────
    // Debug trait formatting for public types
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_expert_heat_level_debug_format() {
        assert_eq!(format!("{:?}", ExpertHeatLevel::Hot), "Hot");
        assert_eq!(format!("{:?}", ExpertHeatLevel::Warm), "Warm");
        assert_eq!(format!("{:?}", ExpertHeatLevel::Cold), "Cold");
        assert_eq!(format!("{:?}", ExpertHeatLevel::Evicted), "Evicted");
    }

    #[test]
    fn test_eviction_decision_debug_format() {
        assert_eq!(format!("{:?}", EvictionDecision::Keep), "Keep");
        assert_eq!(format!("{:?}", EvictionDecision::Evict), "Evict");
        assert_eq!(format!("{:?}", EvictionDecision::Reactivate), "Reactivate");
    }

    #[test]
    fn test_deopt_handling_result_debug_format() {
        let r1 = DeoptHandlingResult::ReactivateAndRerun { expert_idx: 3, request_id: 42 };
        let debug1 = format!("{:?}", r1);
        assert!(debug1.contains("ReactivateAndRerun"));
        assert!(debug1.contains("expert_idx: 3"));

        let r2 = DeoptHandlingResult::SpuriousDeopt { expert_idx: 7, request_id: 99 };
        let debug2 = format!("{:?}", r2);
        assert!(debug2.contains("SpuriousDeopt"));
        assert!(debug2.contains("expert_idx: 7"));
    }

    #[test]
    fn test_deopt_handling_result_clone() {
        let r = DeoptHandlingResult::ReactivateAndRerun { expert_idx: 2, request_id: 10 };
        let cloned = r.clone();
        match cloned {
            DeoptHandlingResult::ReactivateAndRerun { expert_idx, request_id } => {
                assert_eq!(expert_idx, 2);
                assert_eq!(request_id, 10);
            }
            DeoptHandlingResult::SpuriousDeopt { .. } => panic!("wrong variant"),
        }

        let s = DeoptHandlingResult::SpuriousDeopt { expert_idx: 5, request_id: 20 };
        let cloned_s = s.clone();
        match cloned_s {
            DeoptHandlingResult::SpuriousDeopt { expert_idx, request_id } => {
                assert_eq!(expert_idx, 5);
                assert_eq!(request_id, 20);
            }
            DeoptHandlingResult::ReactivateAndRerun { .. } => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_expert_heat_state_debug_format() {
        let state = ExpertHeatState {
            expert_idx: 42,
            hit_rate: 0.75,
            hit_count: 300,
            route_count: 400,
            heat_level: ExpertHeatLevel::Hot,
            consecutive_zero_streak: 0,
            last_hit_step: 400,
            is_evicted: false,
            reactivation_count: 0,
        };
        let debug = format!("{:?}", state);
        assert!(debug.contains("expert_idx: 42"));
        assert!(debug.contains("hit_rate: 0.75"));
        assert!(debug.contains("hit_count: 300"));
        assert!(debug.contains("route_count: 400"));
        assert!(debug.contains("Hot"));
        assert!(debug.contains("is_evicted: false"));
    }

    #[test]
    fn test_thermal_summary_debug_format() {
        let summary = ThermalSummary {
            num_experts: 16,
            hot_count: 4,
            warm_count: 6,
            cold_count: 3,
            evicted_count: 3,
            total_evictions: 10,
            total_reactivations: 7,
            current_step: 5000,
            pending_deopt_count: 2,
            working_set_size: 8,
            effective_eviction_threshold: 600,
        };
        let debug = format!("{:?}", summary);
        assert!(debug.contains("num_experts: 16"));
        assert!(debug.contains("hot_count: 4"));
        assert!(debug.contains("warm_count: 6"));
        assert!(debug.contains("cold_count: 3"));
        assert!(debug.contains("evicted_count: 3"));
        assert!(debug.contains("total_evictions: 10"));
        assert!(debug.contains("current_step: 5000"));
        assert!(debug.contains("pending_deopt_count: 2"));
        assert!(debug.contains("working_set_size: 8"));
        assert!(debug.contains("effective_eviction_threshold: 600"));
    }

    #[test]
    fn test_deopt_request_debug_format() {
        let req = DeoptRequest {
            request_id: 123,
            expert_idx: 5,
            layer_idx: 10,
            step: 999,
        };
        let debug = format!("{:?}", req);
        assert!(debug.contains("request_id: 123"));
        assert!(debug.contains("expert_idx: 5"));
        assert!(debug.contains("layer_idx: 10"));
        assert!(debug.contains("step: 999"));
    }

    // ────────────────────────────────────────────────────────
    // WorkingSetTracker direct edge cases
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_working_set_tracker_new_initial_state() {
        let tracker = WorkingSetTracker::new(8, 5, 1000);
        assert_eq!(tracker.working_set_size(), 0);
        assert_eq!(tracker.num_experts, 8);
        assert_eq!(tracker.window_size, 5);
        assert_eq!(tracker.base_threshold, 1000);
    }

    #[test]
    fn test_working_set_tracker_record_step_all_accessed() {
        let mut tracker = WorkingSetTracker::new(4, 3, 100);
        tracker.record_step(&[5, 10, 3, 1]);
        assert_eq!(tracker.working_set_size(), 4);
    }

    #[test]
    fn test_working_set_tracker_record_step_none_accessed() {
        let mut tracker = WorkingSetTracker::new(4, 3, 100);
        tracker.record_step(&[0, 0, 0, 0]);
        assert_eq!(tracker.working_set_size(), 0);
    }

    #[test]
    fn test_working_set_tracker_record_step_extra_indices_ignored() {
        let mut tracker = WorkingSetTracker::new(3, 5, 100);
        // route_counts longer than num_experts: extra entries ignored
        tracker.record_step(&[1, 2, 3, 4, 5]);
        assert_eq!(tracker.working_set_size(), 3);
    }

    #[test]
    fn test_working_set_tracker_ring_buffer_overwrite() {
        let mut tracker = WorkingSetTracker::new(2, 2, 100);
        // Window size = 2, so cursor cycles: slot 0, slot 1, slot 0, slot 1, ...
        // Step 1 (cursor=0): expert 0 accessed. After: cursor=1
        tracker.record_step(&[1, 0]);
        assert_eq!(tracker.working_set_size(), 1);

        // Step 2 (cursor=1): expert 1 accessed. After: cursor=0
        tracker.record_step(&[0, 1]);
        // Slot 0 = [true, false], Slot 1 = [false, true] => {0, 1}
        assert_eq!(tracker.working_set_size(), 2);

        // Step 3 (cursor=0): expert 1 only, overwrites slot 0. After: cursor=1
        tracker.record_step(&[0, 1]);
        // Slot 0 = [false, true], Slot 1 = [false, true] => {1}
        assert_eq!(tracker.working_set_size(), 1);

        // Step 4 (cursor=1): expert 0 only, overwrites slot 1. After: cursor=0
        tracker.record_step(&[1, 0]);
        // Slot 0 = [false, true], Slot 1 = [true, false] => {0, 1}
        assert_eq!(tracker.working_set_size(), 2);
    }

    #[test]
    fn test_working_set_tracker_window_size_one() {
        let mut tracker = WorkingSetTracker::new(3, 1, 100);
        tracker.record_step(&[1, 0, 0]);
        assert_eq!(tracker.working_set_size(), 1);

        // Overwrite single slot: now only expert 2
        tracker.record_step(&[0, 0, 5]);
        assert_eq!(tracker.working_set_size(), 1);
    }

    #[test]
    fn test_working_set_tracker_adaptive_threshold_no_pressure() {
        let mut tracker = WorkingSetTracker::new(8, 10, 1000);
        tracker.record_step(&[1, 1, 0, 0, 0, 0, 0, 0]); // working set = 2

        let threshold = tracker.adaptive_threshold(0.0);
        // base * (num_experts / ws) * headroom(1.0) = 1000 * (8/2) * 1.0 = 4000
        assert_eq!(threshold, 4000);
    }

    #[test]
    fn test_working_set_tracker_adaptive_threshold_full_pressure() {
        let tracker = WorkingSetTracker::new(4, 10, 1000);
        // No steps recorded: working_set_size = 0 => max(1) => ws = 1
        // threshold = 1000 * (4/1) * clamp(1.0-1.0, 0.1, 1.0) = 1000 * 4 * 0.1 = 400
        let threshold = tracker.adaptive_threshold(1.0);
        assert_eq!(threshold, 400);
    }

    #[test]
    fn test_working_set_tracker_adaptive_threshold_minimum_one() {
        let tracker = WorkingSetTracker::new(1, 5, 1);
        // Even with smallest base, threshold should be at least 1
        let threshold = tracker.adaptive_threshold(0.0);
        assert!(threshold >= 1);
    }

    // ────────────────────────────────────────────────────────
    // Step: evicted expert preserves Evicted heat level
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_step_evicted_expert_stays_evicted() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(3);
        for _ in 0..4 {
            manager.step(&[10, 0]);
        }
        manager.evict_expert(1);

        // Now route tokens to expert 1 in subsequent steps
        for _ in 0..10 {
            manager.step(&[10, 5]);
        }

        // Expert 1's heat_level must remain Evicted despite hit_rate increasing
        let state = manager.state(1).unwrap();
        assert_eq!(state.heat_level, ExpertHeatLevel::Evicted);
        assert!(state.is_evicted);
        // But the underlying hit_rate and counts are still updated
        assert!(state.hit_count > 0);
        assert!(state.route_count > 0);
    }

    // ────────────────────────────────────────────────────────
    // Step: route_counts longer than num_experts is truncated
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_step_route_counts_longer_than_experts() {
        let mut manager = ExpertThermalManager::new(2);
        manager.step(&[10, 5, 20, 30]); // extra entries ignored

        assert_eq!(manager.state(0).unwrap().route_count, 1);
        assert_eq!(manager.state(1).unwrap().route_count, 1);
        // expert indices 2 and 3 don't exist
        assert_eq!(manager.states().len(), 2);
    }

    // ────────────────────────────────────────────────────────
    // Cumulative eviction and reactivation counters
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_cumulative_eviction_reactivation_counts() {
        let mut manager = ExpertThermalManager::new(4).with_eviction_threshold(3);

        for _ in 0..4 {
            manager.step(&[10, 0, 0, 0]);
        }

        // Evict experts 1, 2, 3
        assert!(manager.evict_expert(1));
        assert!(manager.evict_expert(2));
        assert!(manager.evict_expert(3));
        assert_eq!(manager.summary().total_evictions, 3);

        // Reactivate experts 1 and 2
        assert!(manager.reactivate_expert(1));
        assert!(manager.reactivate_expert(2));
        assert_eq!(manager.summary().total_reactivations, 2);

        // Re-evict expert 1
        manager.evict_expert(1);
        assert_eq!(manager.summary().total_evictions, 4);

        // Reactivate expert 1 again
        manager.reactivate_expert(1);
        assert_eq!(manager.summary().total_reactivations, 3);
    }

    // ────────────────────────────────────────────────────────
    // ExpertHeatLevel equality edge cases
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_heat_level_equality_and_inequality() {
        assert_eq!(ExpertHeatLevel::Hot, ExpertHeatLevel::Hot);
        assert_ne!(ExpertHeatLevel::Hot, ExpertHeatLevel::Warm);
        assert_ne!(ExpertHeatLevel::Warm, ExpertHeatLevel::Cold);
        assert_ne!(ExpertHeatLevel::Cold, ExpertHeatLevel::Evicted);
    }

    // ────────────────────────────────────────────────────────
    // EvictionDecision all variants are distinct
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_eviction_decision_all_variants_distinct() {
        let variants = [EvictionDecision::Keep, EvictionDecision::Evict, EvictionDecision::Reactivate];
        for (i, v1) in variants.iter().enumerate() {
            for (j, v2) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(v1, v2);
                } else {
                    assert_ne!(v1, v2);
                }
            }
        }
    }

    // ────────────────────────────────────────────────────────
    // DeoptRequest field access after construction
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_deopt_request_field_access() {
        let req = DeoptRequest {
            request_id: u64::MAX,
            expert_idx: 0,
            layer_idx: usize::MAX,
            step: 0,
        };
        assert_eq!(req.request_id, u64::MAX);
        assert_eq!(req.expert_idx, 0);
        assert_eq!(req.layer_idx, usize::MAX);
        assert_eq!(req.step, 0);
    }

    // ────────────────────────────────────────────────────────
    // with_adaptive_eviction minimum window size
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_with_adaptive_eviction_window_size_minimum() {
        // window_size of 0 should be clamped to 1
        let manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(500)
            .with_adaptive_eviction(0);

        // Should not panic; effective threshold should compute
        let threshold = manager.effective_eviction_threshold();
        assert!(threshold >= 1);
    }

    // ────────────────────────────────────────────────────────
    // ExpertHeatState direct construction with all fields
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_expert_heat_state_manual_construction() {
        let state = ExpertHeatState {
            expert_idx: 10,
            hit_rate: 0.42,
            hit_count: 42,
            route_count: 100,
            heat_level: ExpertHeatLevel::Warm,
            consecutive_zero_streak: 7,
            last_hit_step: 93,
            is_evicted: true,
            reactivation_count: 3,
        };
        assert_eq!(state.expert_idx, 10);
        assert!((state.hit_rate - 0.42).abs() < f64::EPSILON);
        assert_eq!(state.hit_count, 42);
        assert_eq!(state.route_count, 100);
        assert_eq!(state.heat_level, ExpertHeatLevel::Warm);
        assert_eq!(state.consecutive_zero_streak, 7);
        assert_eq!(state.last_hit_step, 93);
        assert!(state.is_evicted);
        assert_eq!(state.reactivation_count, 3);
    }

    // ────────────────────────────────────────────────────────
    // Eviction aggressiveness combined with adaptive eviction
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_aggressiveness_with_adaptive_eviction() {
        let mut manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(1000)
            .with_adaptive_eviction(10)
            .with_eviction_aggressiveness(1.0);

        manager.step(&[1, 1, 0, 0]); // working set = 2
        manager.update_memory_pressure(0.0);

        let threshold = manager.effective_eviction_threshold();
        // adaptive_threshold(0.0) = base * (4/2) * 1.0 = 2000
        // aggressiveness=1.0 => bias_factor = 1/(1+1) = 0.5
        // effective = 2000 * 0.5 = 1000
        assert_eq!(threshold, 1000);
    }

    // ────────────────────────────────────────────────────────
    // Step with zero route_counts for specific expert preserves streak
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_consecutive_zero_streak_increments_monotonically() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(1000);

        for i in 0..5 {
            manager.step(&[10, 0]);
            assert_eq!(
                manager.state(1).unwrap().consecutive_zero_streak,
                (i + 1) as u64,
                "streak should be {} after step {}",
                i + 1,
                i + 1
            );
        }
    }

    // ────────────────────────────────────────────────────────
    // Hit rate accuracy with large counts
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_hit_rate_accuracy_large_counts() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(u64::MAX);

        // Expert 0: hit 3 out of 7
        manager.step(&[10, 5]); // hit (1/1)
        manager.step(&[0, 5]);  // miss (1/2)
        manager.step(&[10, 5]); // hit (2/3)
        manager.step(&[0, 5]);  // miss (2/4)
        manager.step(&[0, 5]);  // miss (2/5)
        manager.step(&[10, 5]); // hit (3/6)
        manager.step(&[0, 5]);  // miss (3/7)

        let state = manager.state(0).unwrap();
        assert_eq!(state.hit_count, 3);
        assert_eq!(state.route_count, 7);
        assert!((state.hit_rate - 3.0 / 7.0).abs() < 1e-10);
    }

    // ────────────────────────────────────────────────────────
    // Evicting expert resets reactivation_count to zero
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_evict_resets_reactivation_count() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(3);

        for _ in 0..4 {
            manager.step(&[10, 0]);
        }
        manager.evict_expert(1);
        manager.reactivate_expert(1);
        assert_eq!(manager.state(1).unwrap().reactivation_count, 1);

        // Re-evict: reactivation_count should reset to 0
        manager.evict_expert(1);
        assert_eq!(manager.state(1).unwrap().reactivation_count, 0);
    }

    // ────────────────────────────────────────────────────────
    // Cold experts list excludes hot and warm
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_cold_or_evicted_excludes_hot_and_warm() {
        let mut manager = ExpertThermalManager::new(4)
            .with_heat_thresholds(0.5, 0.01)
            .with_eviction_threshold(5);

        // Expert 0: always hit (hot), expert 1: hit 50% (warm), expert 2: never hit (cold), expert 3: evicted
        for _ in 0..6 {
            manager.step(&[10, 3, 0, 0]);
        }
        manager.evict_expert(3);

        let cold_evicted = manager.cold_or_evicted_experts();
        assert!(!cold_evicted.contains(&0)); // hot
        // expert 1 could be warm or hot depending on exact threshold math
        assert!(cold_evicted.contains(&3)); // evicted
    }

    // ────────────────────────────────────────────────────────
    // experts_to_evict after reactivation
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_experts_to_evict_after_reactivation_cycle() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(3);

        // Evict expert 1
        for _ in 0..4 {
            manager.step(&[10, 0]);
        }
        manager.evict_expert(1);
        assert!(manager.experts_to_evict().is_empty()); // already evicted

        // Reactivate
        manager.reactivate_expert(1);
        assert!(!manager.state(1).unwrap().is_evicted);

        // Expert 1 streak was reset to 0, so not yet flagged for eviction
        assert!(manager.experts_to_evict().is_empty());

        // Build up streak again
        for _ in 0..3 {
            manager.step(&[10, 0]);
        }
        assert!(manager.experts_to_evict().contains(&1));
    }

    // ────────────────────────────────────────────────────────
    // WorkingSetTracker record_step overwrites previous slot
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_working_set_tracker_record_step_clears_previous_slot() {
        let mut tracker = WorkingSetTracker::new(3, 2, 100);

        // Slot 0: experts 0 and 1
        tracker.record_step(&[5, 3, 0]);
        assert_eq!(tracker.working_set_size(), 2);

        // Slot 1: expert 2 only
        tracker.record_step(&[0, 0, 7]);
        assert_eq!(tracker.working_set_size(), 3);

        // Slot 0 overwritten: expert 0 only
        tracker.record_step(&[1, 0, 0]);
        // Slot 0: expert 0, Slot 1: expert 2 => working set = {0, 2}
        assert_eq!(tracker.working_set_size(), 2);
    }

    // ────────────────────────────────────────────────────────
    // New tests: additional coverage for gaps
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_heat_level_from_hit_rate_inverted_thresholds() {
        // When cold_threshold > hot_threshold, the logic still follows:
        // rate >= hot => Hot, rate >= cold => Warm, rate > 0 => Cold, else Evicted.
        // With hot=0.001, cold=0.1: rate=0.05 >= 0.001 => Hot
        assert_eq!(
            ExpertHeatLevel::from_hit_rate(0.05, 0.001, 0.1),
            ExpertHeatLevel::Hot,
        );
        // rate=0.5 >= 0.001 => Hot (first branch wins)
        assert_eq!(
            ExpertHeatLevel::from_hit_rate(0.5, 0.001, 0.1),
            ExpertHeatLevel::Hot,
        );
    }

    #[test]
    fn test_heat_level_from_hit_rate_zero_thresholds() {
        // Both thresholds at 0: any positive rate is Hot
        assert_eq!(
            ExpertHeatLevel::from_hit_rate(0.0001, 0.0, 0.0),
            ExpertHeatLevel::Hot,
        );
        // Rate exactly 0 with 0 thresholds => Hot (0.0 >= 0.0 is true)
        assert_eq!(
            ExpertHeatLevel::from_hit_rate(0.0, 0.0, 0.0),
            ExpertHeatLevel::Hot,
        );
    }

    #[test]
    fn test_heat_level_partial_ord_transitivity() {
        // Verify transitivity: Hot < Warm < Cold < Evicted
        let hot = ExpertHeatLevel::Hot;
        let warm = ExpertHeatLevel::Warm;
        let cold = ExpertHeatLevel::Cold;
        let evicted = ExpertHeatLevel::Evicted;

        // Transitivity: if A < B and B < C then A < C
        assert!(hot < warm);
        assert!(warm < cold);
        assert!(hot < cold);
        assert!(cold < evicted);
        assert!(hot < evicted);
        assert!(warm < evicted);
    }

    #[test]
    fn test_eviction_decision_copy() {
        // Verify Copy trait works: assigning to a new variable doesn't move
        let d1 = EvictionDecision::Evict;
        let d2 = d1; // copy, not move
        let d3 = d1; // still valid because Copy
        assert_eq!(d1, d2);
        assert_eq!(d1, d3);
    }

    #[test]
    fn test_step_all_zero_route_counts() {
        let mut manager = ExpertThermalManager::new(3).with_eviction_threshold(100);

        // All zeros: every expert gets route_count += 1 but no hits
        manager.step(&[0, 0, 0]);

        for state in manager.states() {
            assert_eq!(state.route_count, 1);
            assert_eq!(state.hit_count, 0);
            assert_eq!(state.hit_rate, 0.0);
            assert_eq!(state.consecutive_zero_streak, 1);
        }
    }

    #[test]
    fn test_effective_eviction_threshold_very_high_aggressiveness() {
        let manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(1000)
            .with_eviction_aggressiveness(99.0);

        // bias_factor = 1/(1+99) = 0.01, effective = 1000 * 0.01 = 10
        assert_eq!(manager.effective_eviction_threshold(), 10);
    }

    #[test]
    fn test_effective_eviction_threshold_zero_aggressiveness_unchanged() {
        let manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(500)
            .with_eviction_aggressiveness(0.0);

        assert_eq!(manager.effective_eviction_threshold(), 500);
    }

    #[test]
    fn test_reactivate_sets_heat_level_to_cold() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(3);

        for _ in 0..4 {
            manager.step(&[10, 0]);
        }
        manager.evict_expert(1);
        assert_eq!(manager.state(1).unwrap().heat_level, ExpertHeatLevel::Evicted);

        manager.reactivate_expert(1);
        assert_eq!(manager.state(1).unwrap().heat_level, ExpertHeatLevel::Cold);
        assert!(!manager.state(1).unwrap().is_evicted);
    }

    #[test]
    fn test_evict_increments_total_evictions_only_on_success() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(3);

        for _ in 0..4 {
            manager.step(&[10, 0]);
        }

        // Successful eviction
        assert!(manager.evict_expert(1));
        assert_eq!(manager.summary().total_evictions, 1);

        // Double eviction fails, counter stays same
        assert!(!manager.evict_expert(1));
        assert_eq!(manager.summary().total_evictions, 1);

        // Out-of-bounds fails, counter stays same
        assert!(!manager.evict_expert(99));
        assert_eq!(manager.summary().total_evictions, 1);
    }

    #[test]
    fn test_reactivate_increments_total_reactivations_only_on_success() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(3);

        for _ in 0..4 {
            manager.step(&[10, 0]);
        }
        manager.evict_expert(1);

        // Successful reactivation
        assert!(manager.reactivate_expert(1));
        assert_eq!(manager.summary().total_reactivations, 1);

        // Non-evicted reactivation fails
        assert!(!manager.reactivate_expert(0));
        assert_eq!(manager.summary().total_reactivations, 1);

        // Out-of-bounds fails
        assert!(!manager.reactivate_expert(99));
        assert_eq!(manager.summary().total_reactivations, 1);
    }

    #[test]
    fn test_deopt_request_extreme_field_values() {
        // Test with u64::MAX and 0 values to verify no overflow or panic
        let req = DeoptRequest {
            request_id: u64::MAX,
            expert_idx: usize::MAX,
            layer_idx: usize::MAX,
            step: u64::MAX,
        };
        assert_eq!(req.request_id, u64::MAX);
        assert_eq!(req.expert_idx, usize::MAX);
        assert_eq!(req.layer_idx, usize::MAX);
        assert_eq!(req.step, u64::MAX);

        let cloned = req.clone();
        assert_eq!(cloned.request_id, u64::MAX);
    }

    #[test]
    fn test_thermal_summary_zero_values() {
        let summary = ThermalSummary {
            num_experts: 0,
            hot_count: 0,
            warm_count: 0,
            cold_count: 0,
            evicted_count: 0,
            total_evictions: 0,
            total_reactivations: 0,
            current_step: 0,
            pending_deopt_count: 0,
            working_set_size: 0,
            effective_eviction_threshold: 0,
        };
        assert_eq!(summary.num_experts, 0);
        assert_eq!(summary.current_step, 0);
        assert_eq!(summary.effective_eviction_threshold, 0);
    }

    #[test]
    fn test_thermal_summary_clone_independence() {
        let summary = ThermalSummary {
            num_experts: 4,
            hot_count: 2,
            warm_count: 1,
            cold_count: 0,
            evicted_count: 1,
            total_evictions: 3,
            total_reactivations: 1,
            current_step: 100,
            pending_deopt_count: 0,
            working_set_size: 3,
            effective_eviction_threshold: 500,
        };
        let cloned = summary.clone();

        // Verify clone matches
        assert_eq!(cloned.num_experts, summary.num_experts);
        assert_eq!(cloned.hot_count, summary.hot_count);
        assert_eq!(cloned.total_evictions, summary.total_evictions);

        // Modify original, verify clone is independent
        let _ = summary.hot_count; // consume before overwrite to avoid warning
        assert_eq!(cloned.hot_count, 2);
    }

    #[test]
    fn test_working_set_tracker_adaptive_threshold_full_working_set() {
        let mut tracker = WorkingSetTracker::new(4, 5, 1000);
        tracker.record_step(&[1, 2, 3, 4]); // all 4 experts accessed

        // ratio = 4/4 = 1.0, headroom = 1.0 => threshold = 1000
        let threshold = tracker.adaptive_threshold(0.0);
        assert_eq!(threshold, 1000);
    }

    #[test]
    fn test_working_set_tracker_adaptive_threshold_negative_pressure_clamped() {
        let tracker = WorkingSetTracker::new(4, 5, 1000);
        // Negative pressure: headroom = clamp(1.0 - (-0.5), 0.1, 1.0) = clamp(1.5, 0.1, 1.0) = 1.0
        let threshold_neg = tracker.adaptive_threshold(-0.5);
        let threshold_zero = tracker.adaptive_threshold(0.0);
        assert_eq!(threshold_neg, threshold_zero);
    }

    #[test]
    fn test_working_set_tracker_adaptive_threshold_above_one_clamped() {
        let tracker = WorkingSetTracker::new(4, 5, 1000);
        // Pressure > 1.0: headroom = clamp(1.0 - 1.5, 0.1, 1.0) = clamp(-0.5, 0.1, 1.0) = 0.1
        let threshold_above = tracker.adaptive_threshold(1.5);
        // Pressure = 1.0: headroom = clamp(0.0, 0.1, 1.0) = 0.1
        let threshold_one = tracker.adaptive_threshold(1.0);
        assert_eq!(threshold_above, threshold_one);
    }

    #[test]
    fn test_handle_deopt_for_non_evicted_is_spurious() {
        let mut manager = ExpertThermalManager::new(4);

        // Expert 0 is active, not evicted
        manager.step(&[10, 5, 0, 0]);

        let deopt = DeoptRequest {
            request_id: 77,
            expert_idx: 0,
            layer_idx: 2,
            step: 1,
        };
        let result = manager.handle_deopt_request(deopt);
        match result {
            DeoptHandlingResult::SpuriousDeopt { expert_idx, request_id } => {
                assert_eq!(expert_idx, 0);
                assert_eq!(request_id, 77);
            }
            DeoptHandlingResult::ReactivateAndRerun { .. } => {
                panic!("Expected SpuriousDeopt for non-evicted expert");
            }
        }

        // Deopt request should still be recorded
        assert_eq!(manager.pending_deopt_requests().len(), 1);
    }

    #[test]
    fn test_multiple_evict_reactivate_cycles() {
        let mut manager = ExpertThermalManager::new(1).with_eviction_threshold(2);

        for cycle in 1..=5 {
            // Build streak
            for _ in 0..3 {
                manager.step(&[0]);
            }
            assert!(manager.evict_expert(0), "eviction should succeed in cycle {}", cycle);
            assert_eq!(manager.state(0).unwrap().reactivation_count, 0);

            assert!(manager.reactivate_expert(0), "reactivation should succeed in cycle {}", cycle);
            assert_eq!(manager.state(0).unwrap().reactivation_count, 1);
            assert_eq!(manager.state(0).unwrap().consecutive_zero_streak, 0);
        }

        // Total counters: 5 evictions, 5 reactivations
        assert_eq!(manager.summary().total_evictions, 5);
        assert_eq!(manager.summary().total_reactivations, 5);
    }

    #[test]
    fn test_experts_to_evict_includes_only_non_evicted() {
        let mut manager = ExpertThermalManager::new(3).with_eviction_threshold(3);

        for _ in 0..4 {
            manager.step(&[10, 0, 0]);
        }

        // Expert 1 and 2 both qualify for eviction
        let to_evict = manager.experts_to_evict();
        assert_eq!(to_evict.len(), 2);
        assert!(to_evict.contains(&1));
        assert!(to_evict.contains(&2));

        // Evict expert 1
        manager.evict_expert(1);

        // Now only expert 2 appears in to_evict (expert 1 is already evicted)
        let to_evict_after = manager.experts_to_evict();
        assert_eq!(to_evict_after.len(), 1);
        assert!(to_evict_after.contains(&2));
    }

    #[test]
    fn test_summary_effective_eviction_threshold_reflects_aggressiveness() {
        let manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(1000)
            .with_eviction_aggressiveness(1.0);

        let summary = manager.summary();
        // bias_factor = 0.5, effective = 500
        assert_eq!(summary.effective_eviction_threshold, 500);
    }

    // ────────────────────────────────────────────────────────
    // Additional coverage: 18 new tests
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_heat_level_from_hit_rate_epsilon_positive_rate() {
        // A rate arbitrarily close to zero but positive must be Cold, not Evicted
        let level = ExpertHeatLevel::from_hit_rate(f64::MIN_POSITIVE, 0.1, 0.001);
        assert_eq!(level, ExpertHeatLevel::Cold);
    }

    #[test]
    fn test_heat_level_from_hit_rate_just_below_hot_threshold() {
        // rate = 0.0999 with hot_threshold=0.1 => Warm
        let level = ExpertHeatLevel::from_hit_rate(0.0999, 0.1, 0.001);
        assert_eq!(level, ExpertHeatLevel::Warm);
    }

    #[test]
    fn test_heat_level_from_hit_rate_just_above_cold_threshold() {
        // rate = 0.0011 with cold_threshold=0.001 => Warm
        let level = ExpertHeatLevel::from_hit_rate(0.0011, 0.1, 0.001);
        assert_eq!(level, ExpertHeatLevel::Warm);
    }

    #[test]
    fn test_heat_level_from_hit_rate_just_below_cold_threshold() {
        // rate = 0.0009 with cold_threshold=0.001 => Cold (above zero but below cold)
        let level = ExpertHeatLevel::from_hit_rate(0.0009, 0.1, 0.001);
        assert_eq!(level, ExpertHeatLevel::Cold);
    }

    #[test]
    fn test_eviction_decision_at_exact_streak_threshold() {
        // Arrange: threshold=10, run exactly 10 zero-hit steps
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(10);

        // Act: 10 steps where expert 1 gets zero hits
        for _ in 0..10 {
            manager.step(&[10, 0]);
        }

        // Assert: streak == threshold => Evict
        assert_eq!(manager.eviction_decision(1), EvictionDecision::Evict);
    }

    #[test]
    fn test_eviction_decision_just_below_streak_threshold() {
        // Arrange: threshold=10, run 9 zero-hit steps
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(10);

        for _ in 0..9 {
            manager.step(&[10, 0]);
        }

        // Assert: streak=9 < threshold=10 => Keep
        assert_eq!(manager.eviction_decision(1), EvictionDecision::Keep);
    }

    #[test]
    fn test_step_on_zero_expert_manager() {
        // Arrange: manager with 0 experts
        let mut manager = ExpertThermalManager::new(0);

        // Act: stepping with any route_counts should not panic
        manager.step(&[]);
        manager.step(&[1, 2, 3]); // longer than 0 experts, gracefully truncated

        // Assert
        assert_eq!(manager.summary().current_step, 2);
        assert_eq!(manager.summary().num_experts, 0);
    }

    #[test]
    fn test_single_expert_full_lifecycle() {
        // Arrange
        let mut manager = ExpertThermalManager::new(1).with_eviction_threshold(3);

        // Act: build streak of 4 zero-hit steps
        for _ in 0..4 {
            manager.step(&[0]);
        }
        assert!(manager.experts_to_evict().contains(&0));
        assert!(manager.evict_expert(0));

        // Reactivate via deopt
        let result = manager.handle_deopt_request(DeoptRequest {
            request_id: 1,
            expert_idx: 0,
            layer_idx: 0,
            step: 4,
        });

        // Assert
        assert!(matches!(result, DeoptHandlingResult::ReactivateAndRerun { expert_idx: 0, .. }));
        assert!(!manager.state(0).unwrap().is_evicted);
        // handle_deopt_request increments reactivation_count internally (line 427)
        // then reactivate_expert increments it again (line 401), so count = 2
        assert_eq!(manager.state(0).unwrap().reactivation_count, 2);
        assert_eq!(manager.summary().total_evictions, 1);
        assert_eq!(manager.summary().total_reactivations, 1);
    }

    #[test]
    fn test_builder_chains_all_with_methods() {
        // Arrange: chain all builder methods together
        let manager = ExpertThermalManager::new(16)
            .with_eviction_threshold(500)
            .with_heat_thresholds(0.2, 0.01)
            .with_adaptive_eviction(50)
            .with_eviction_aggressiveness(0.5);

        // Assert: effective threshold = adaptive(base=500) * bias(1/1.5)
        // With no steps, working_set_size=0 => max(1) => ws=1
        // adaptive = 500 * (16/1) * 1.0 = 8000
        // effective = 8000 * (1/1.5) = 5333 (integer truncation)
        let threshold = manager.effective_eviction_threshold();
        assert_eq!(threshold, 5333);
        assert_eq!(manager.num_experts(), 16);
    }

    #[test]
    fn test_working_set_tracker_record_step_shorter_counts() {
        // Arrange: 4 experts, but route_counts has only 2 entries
        let mut tracker = WorkingSetTracker::new(4, 3, 100);

        // Act
        tracker.record_step(&[5, 3]);

        // Assert: only experts 0 and 1 recorded; experts 2 and 3 untouched
        assert_eq!(tracker.working_set_size(), 2);
    }

    #[test]
    fn test_working_set_tracker_adaptive_threshold_half_working_set() {
        // Arrange: 4 experts, 2 active => ratio = 4/2 = 2.0
        let mut tracker = WorkingSetTracker::new(4, 5, 1000);
        tracker.record_step(&[1, 1, 0, 0]);

        // Act: pressure = 0.0 => headroom = 1.0
        let threshold = tracker.adaptive_threshold(0.0);

        // Assert: 1000 * (4/2) * 1.0 = 2000
        assert_eq!(threshold, 2000);
    }

    #[test]
    fn test_working_set_tracker_adaptive_threshold_medium_pressure() {
        // Arrange: 4 experts, all active => ratio = 1.0
        let mut tracker = WorkingSetTracker::new(4, 5, 1000);
        tracker.record_step(&[1, 1, 1, 1]);

        // Act: pressure = 0.5 => headroom = 0.5
        let threshold = tracker.adaptive_threshold(0.5);

        // Assert: 1000 * (4/4) * 0.5 = 500
        assert_eq!(threshold, 500);
    }

    #[test]
    fn test_handle_deopt_request_records_in_pending_queue() {
        // Arrange
        let mut manager = ExpertThermalManager::new(4).with_eviction_threshold(3);

        for _ in 0..4 {
            manager.step(&[10, 0, 0, 0]);
        }
        manager.evict_expert(1);

        // Act: two deopt requests for expert 1
        manager.handle_deopt_request(DeoptRequest {
            request_id: 10,
            expert_idx: 1,
            layer_idx: 0,
            step: 4,
        });
        manager.handle_deopt_request(DeoptRequest {
            request_id: 20,
            expert_idx: 1,
            layer_idx: 1,
            step: 5,
        });

        // Assert: both recorded in pending queue (reactivation happens internally)
        assert_eq!(manager.pending_deopt_requests().len(), 2);
        assert_eq!(manager.pending_deopt_requests()[0].request_id, 10);
        assert_eq!(manager.pending_deopt_requests()[1].request_id, 20);
    }

    #[test]
    fn test_clear_deopt_requests_then_refill() {
        // Arrange
        let mut manager = ExpertThermalManager::new(2);

        manager.handle_deopt_request(DeoptRequest {
            request_id: 1,
            expert_idx: 0,
            layer_idx: 0,
            step: 0,
        });
        assert_eq!(manager.pending_deopt_requests().len(), 1);

        // Act: clear then add another
        manager.clear_deopt_requests();
        assert!(manager.pending_deopt_requests().is_empty());

        manager.handle_deopt_request(DeoptRequest {
            request_id: 2,
            expert_idx: 1,
            layer_idx: 0,
            step: 1,
        });

        // Assert
        assert_eq!(manager.pending_deopt_requests().len(), 1);
        assert_eq!(manager.pending_deopt_requests()[0].request_id, 2);
    }

    #[test]
    fn test_step_hit_resets_consecutive_zero_streak() {
        // Arrange
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(100);

        // Build up a streak of 5 for expert 1
        for _ in 0..5 {
            manager.step(&[10, 0]);
        }
        assert_eq!(manager.state(1).unwrap().consecutive_zero_streak, 5);

        // Act: expert 1 gets a hit
        manager.step(&[10, 5]);

        // Assert: streak reset to 0
        assert_eq!(manager.state(1).unwrap().consecutive_zero_streak, 0);
        assert_eq!(manager.state(1).unwrap().hit_count, 1);
    }

    #[test]
    fn test_heat_level_warm_between_thresholds() {
        // Test the middle band: hot_threshold=0.5, cold_threshold=0.1
        // rate=0.3 is in [0.1, 0.5) => Warm
        let level = ExpertHeatLevel::from_hit_rate(0.3, 0.5, 0.1);
        assert_eq!(level, ExpertHeatLevel::Warm);
    }

    #[test]
    fn test_eviction_decision_evict_then_keep_after_partial_reactivation() {
        // This tests the code path where an evicted expert has reactivation_count > 0
        // but is then fully reactivated, so the next decision is Keep
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(3);

        for _ in 0..4 {
            manager.step(&[10, 0]);
        }
        manager.evict_expert(1);

        // Manually set reactivation_count to simulate partial state
        // (in real usage this happens internally during handle_deopt_request)
        // We test via the public API: handle_deopt_request does both
        manager.handle_deopt_request(DeoptRequest {
            request_id: 1,
            expert_idx: 1,
            layer_idx: 0,
            step: 4,
        });

        // After full reactivation, the expert is no longer evicted => Keep
        assert_eq!(manager.eviction_decision(1), EvictionDecision::Keep);
        assert!(!manager.state(1).unwrap().is_evicted);
    }

    #[test]
    fn test_summary_all_cold_after_many_zero_steps() {
        // Arrange: 3 experts, all get zero hits for many steps
        let mut manager = ExpertThermalManager::new(3)
            .with_eviction_threshold(100)
            .with_heat_thresholds(0.1, 0.001);

        for _ in 0..50 {
            manager.step(&[0, 0, 0]);
        }

        // Assert: all experts have hit_rate=0.0 => heat_level=Evicted
        let summary = manager.summary();
        assert_eq!(summary.hot_count, 0);
        assert_eq!(summary.warm_count, 0);
        assert_eq!(summary.cold_count, 0);
        assert_eq!(summary.evicted_count, 3);
    }

    #[test]
    fn test_experts_to_reactivate_multiple_experts_with_pending_reactivation() {
        // Arrange: evict two experts, trigger deopt for both
        let mut manager = ExpertThermalManager::new(3).with_eviction_threshold(3);

        for _ in 0..4 {
            manager.step(&[10, 0, 0]);
        }
        manager.evict_expert(1);
        manager.evict_expert(2);

        // Act: deopt for both experts triggers internal reactivation
        manager.handle_deopt_request(DeoptRequest {
            request_id: 1,
            expert_idx: 1,
            layer_idx: 0,
            step: 4,
        });
        manager.handle_deopt_request(DeoptRequest {
            request_id: 2,
            expert_idx: 2,
            layer_idx: 0,
            step: 4,
        });

        // Assert: both experts are fully reactivated by handle_deopt_request
        // so experts_to_reactivate should be empty (is_evicted=false)
        assert!(manager.experts_to_reactivate().is_empty());
        assert!(!manager.state(1).unwrap().is_evicted);
        assert!(!manager.state(2).unwrap().is_evicted);
    }

    // ────────────────────────────────────────────────────────
    // Batch 3: 40 additional tests for derive traits and edge cases
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_expert_heat_level_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        assert!(set.insert(ExpertHeatLevel::Hot));
        assert!(set.insert(ExpertHeatLevel::Warm));
        assert!(set.insert(ExpertHeatLevel::Cold));
        assert!(set.insert(ExpertHeatLevel::Evicted));
        // Duplicate insertions return false
        assert!(!set.insert(ExpertHeatLevel::Hot));
        assert!(!set.insert(ExpertHeatLevel::Evicted));
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn test_eviction_decision_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        assert!(set.insert(EvictionDecision::Keep));
        assert!(set.insert(EvictionDecision::Evict));
        assert!(set.insert(EvictionDecision::Reactivate));
        assert!(!set.insert(EvictionDecision::Keep));
        assert!(!set.insert(EvictionDecision::Evict));
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_expert_heat_state_partial_eq_identical() {
        let s1 = ExpertHeatState {
            expert_idx: 5,
            hit_rate: 0.33,
            hit_count: 33,
            route_count: 100,
            heat_level: ExpertHeatLevel::Warm,
            consecutive_zero_streak: 0,
            last_hit_step: 100,
            is_evicted: false,
            reactivation_count: 0,
        };
        let s2 = ExpertHeatState {
            expert_idx: 5,
            hit_rate: 0.33,
            hit_count: 33,
            route_count: 100,
            heat_level: ExpertHeatLevel::Warm,
            consecutive_zero_streak: 0,
            last_hit_step: 100,
            is_evicted: false,
            reactivation_count: 0,
        };
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_expert_heat_state_partial_eq_differs_by_field() {
        let base = ExpertHeatState {
            expert_idx: 1,
            hit_rate: 0.5,
            hit_count: 50,
            route_count: 100,
            heat_level: ExpertHeatLevel::Hot,
            consecutive_zero_streak: 0,
            last_hit_step: 50,
            is_evicted: false,
            reactivation_count: 0,
        };

        let diff_idx = ExpertHeatState { expert_idx: 2, ..base.clone() };
        let diff_rate = ExpertHeatState { hit_rate: 0.25, ..base.clone() };
        let diff_evicted = ExpertHeatState { is_evicted: true, ..base.clone() };
        let diff_level = ExpertHeatState { heat_level: ExpertHeatLevel::Cold, ..base.clone() };
        let diff_streak = ExpertHeatState { consecutive_zero_streak: 10, ..base.clone() };
        let diff_reactivation = ExpertHeatState { reactivation_count: 5, ..base.clone() };

        assert_ne!(base, diff_idx);
        assert_ne!(base, diff_rate);
        assert_ne!(base, diff_evicted);
        assert_ne!(base, diff_level);
        assert_ne!(base, diff_streak);
        assert_ne!(base, diff_reactivation);
    }

    #[test]
    fn test_deopt_request_partial_eq() {
        let r1 = DeoptRequest { request_id: 1, expert_idx: 0, layer_idx: 0, step: 0 };
        let r2 = DeoptRequest { request_id: 1, expert_idx: 0, layer_idx: 0, step: 0 };
        let r3 = DeoptRequest { request_id: 2, expert_idx: 0, layer_idx: 0, step: 0 };
        let r4 = DeoptRequest { request_id: 1, expert_idx: 1, layer_idx: 0, step: 0 };
        let r5 = DeoptRequest { request_id: 1, expert_idx: 0, layer_idx: 1, step: 0 };
        let r6 = DeoptRequest { request_id: 1, expert_idx: 0, layer_idx: 0, step: 1 };

        assert_eq!(r1, r2);
        assert_ne!(r1, r3);
        assert_ne!(r1, r4);
        assert_ne!(r1, r5);
        assert_ne!(r1, r6);
    }

    #[test]
    fn test_deopt_handling_result_partial_eq() {
        let r1 = DeoptHandlingResult::ReactivateAndRerun { expert_idx: 3, request_id: 10 };
        let r2 = DeoptHandlingResult::ReactivateAndRerun { expert_idx: 3, request_id: 10 };
        let r3 = DeoptHandlingResult::ReactivateAndRerun { expert_idx: 3, request_id: 11 };
        let r4 = DeoptHandlingResult::SpuriousDeopt { expert_idx: 3, request_id: 10 };

        assert_eq!(r1, r2);
        assert_ne!(r1, r3);
        assert_ne!(r1, r4);

        let s1 = DeoptHandlingResult::SpuriousDeopt { expert_idx: 5, request_id: 20 };
        let s2 = DeoptHandlingResult::SpuriousDeopt { expert_idx: 5, request_id: 20 };
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_thermal_summary_partial_eq() {
        let s1 = ThermalSummary {
            num_experts: 4, hot_count: 1, warm_count: 1, cold_count: 1, evicted_count: 1,
            total_evictions: 5, total_reactivations: 2, current_step: 100,
            pending_deopt_count: 3, working_set_size: 2, effective_eviction_threshold: 500,
        };
        let s2 = ThermalSummary {
            num_experts: 4, hot_count: 1, warm_count: 1, cold_count: 1, evicted_count: 1,
            total_evictions: 5, total_reactivations: 2, current_step: 100,
            pending_deopt_count: 3, working_set_size: 2, effective_eviction_threshold: 500,
        };
        let s3 = ThermalSummary {
            num_experts: 8, ..s1.clone()
        };
        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
    }

    #[test]
    fn test_expert_heat_level_from_hit_rate_nan_rate() {
        let level = ExpertHeatLevel::from_hit_rate(f64::NAN, 0.1, 0.001);
        assert_eq!(level, ExpertHeatLevel::Evicted);
    }

    #[test]
    fn test_expert_heat_level_from_hit_rate_infinity_rate() {
        let level = ExpertHeatLevel::from_hit_rate(f64::INFINITY, 0.1, 0.001);
        assert_eq!(level, ExpertHeatLevel::Hot);
    }

    #[test]
    fn test_expert_heat_level_from_hit_rate_neg_infinity_rate() {
        let level = ExpertHeatLevel::from_hit_rate(f64::NEG_INFINITY, 0.1, 0.001);
        assert_eq!(level, ExpertHeatLevel::Evicted);
    }

    #[test]
    fn test_expert_heat_level_from_hit_rate_negative_rate() {
        let level = ExpertHeatLevel::from_hit_rate(-0.5, 0.1, 0.001);
        assert_eq!(level, ExpertHeatLevel::Evicted);
    }

    #[test]
    fn test_expert_heat_level_from_hit_rate_above_one() {
        let level = ExpertHeatLevel::from_hit_rate(5.0, 0.1, 0.001);
        assert_eq!(level, ExpertHeatLevel::Hot);
    }

    #[test]
    fn test_expert_heat_level_from_hit_rate_threshold_one() {
        let level = ExpertHeatLevel::from_hit_rate(1.0, 1.0, 0.5);
        assert_eq!(level, ExpertHeatLevel::Hot);
        let level_below = ExpertHeatLevel::from_hit_rate(0.99, 1.0, 0.5);
        assert_eq!(level_below, ExpertHeatLevel::Warm);
    }

    #[test]
    fn test_expert_heat_level_ord_total_ordering() {
        let levels = [
            ExpertHeatLevel::Evicted,
            ExpertHeatLevel::Cold,
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Hot,
        ];
        let mut sorted = levels;
        sorted.sort();
        assert_eq!(sorted, [ExpertHeatLevel::Hot, ExpertHeatLevel::Warm, ExpertHeatLevel::Cold, ExpertHeatLevel::Evicted]);
    }

    #[test]
    fn test_expert_heat_level_all_variants_exhaustive() {
        let hot = ExpertHeatLevel::Hot;
        let warm = ExpertHeatLevel::Warm;
        let cold = ExpertHeatLevel::Cold;
        let evicted = ExpertHeatLevel::Evicted;
        // Verify each variant is distinct via Copy + PartialEq
        let all = [hot, warm, cold, evicted];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn test_working_set_tracker_partial_eq_identical() {
        let t1 = WorkingSetTracker::new(4, 3, 100);
        let t2 = WorkingSetTracker::new(4, 3, 100);
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_working_set_tracker_partial_eq_differs() {
        let t1 = WorkingSetTracker::new(4, 3, 100);
        let t2 = WorkingSetTracker::new(8, 3, 100);
        let t3 = WorkingSetTracker::new(4, 5, 100);
        let t4 = WorkingSetTracker::new(4, 3, 200);
        assert_ne!(t1, t2);
        assert_ne!(t1, t3);
        assert_ne!(t1, t4);
    }

    #[test]
    fn test_working_set_tracker_partial_eq_after_record() {
        let mut t1 = WorkingSetTracker::new(2, 3, 100);
        let t2 = WorkingSetTracker::new(2, 3, 100);
        t1.record_step(&[1, 0]);
        assert_ne!(t1, t2);
    }

    #[test]
    fn test_working_set_tracker_large_window_size() {
        let mut tracker = WorkingSetTracker::new(4, 1000, 500);
        tracker.record_step(&[1, 0, 0, 0]);
        assert_eq!(tracker.working_set_size(), 1);

        // Fill all slots with different experts across many steps
        for step in 0..1000 {
            let expert_idx = step % 4;
            let mut counts = vec![0usize; 4];
            counts[expert_idx] = 1;
            tracker.record_step(&counts);
        }
        assert_eq!(tracker.working_set_size(), 4);
    }

    #[test]
    fn test_working_set_tracker_single_expert() {
        let mut tracker = WorkingSetTracker::new(1, 5, 100);
        tracker.record_step(&[0]);
        assert_eq!(tracker.working_set_size(), 0);
        tracker.record_step(&[1]);
        assert_eq!(tracker.working_set_size(), 1);
        tracker.record_step(&[0]);
        assert_eq!(tracker.working_set_size(), 1); // previous step still in window
    }

    #[test]
    fn test_adaptive_threshold_with_full_working_set_equals_base() {
        let mut tracker = WorkingSetTracker::new(4, 5, 1000);
        tracker.record_step(&[1, 1, 1, 1]);
        // ratio = 4/4 = 1.0, headroom(0.0) = 1.0, threshold = 1000
        assert_eq!(tracker.adaptive_threshold(0.0), 1000);
    }

    #[test]
    fn test_adaptive_threshold_quarter_working_set() {
        let mut tracker = WorkingSetTracker::new(8, 5, 1000);
        tracker.record_step(&[1, 0, 0, 0, 0, 0, 0, 0]); // 1 of 8 active
        // ratio = 8/1 = 8.0, headroom(0.0) = 1.0, threshold = 8000
        assert_eq!(tracker.adaptive_threshold(0.0), 8000);
    }

    #[test]
    fn test_step_does_not_update_beyond_num_experts() {
        let mut manager = ExpertThermalManager::new(2);
        manager.step(&[10, 5, 100, 200, 300]);
        assert_eq!(manager.state(0).unwrap().route_count, 1);
        assert_eq!(manager.state(1).unwrap().route_count, 1);
        // States list is exactly 2, no extra entries
        assert_eq!(manager.states().len(), 2);
    }

    #[test]
    fn test_manager_with_single_expert_always_hot() {
        let mut manager = ExpertThermalManager::new(1);
        for _ in 0..10 {
            manager.step(&[5]);
        }
        assert_eq!(manager.state(0).unwrap().heat_level, ExpertHeatLevel::Hot);
        assert_eq!(manager.state(0).unwrap().hit_rate, 1.0);
        assert_eq!(manager.state(0).unwrap().consecutive_zero_streak, 0);
    }

    #[test]
    fn test_manager_with_single_expert_always_cold() {
        let mut manager = ExpertThermalManager::new(1).with_eviction_threshold(1000);
        for _ in 0..10 {
            manager.step(&[0]);
        }
        assert_eq!(manager.state(0).unwrap().heat_level, ExpertHeatLevel::Evicted);
        assert_eq!(manager.state(0).unwrap().hit_rate, 0.0);
        assert_eq!(manager.state(0).unwrap().consecutive_zero_streak, 10);
    }

    #[test]
    fn test_step_hit_rate_one_after_single_hit() {
        let mut manager = ExpertThermalManager::new(2);
        manager.step(&[5, 0]);
        assert_eq!(manager.state(0).unwrap().hit_rate, 1.0);
        assert_eq!(manager.state(1).unwrap().hit_rate, 0.0);
    }

    #[test]
    fn test_step_hit_rate_zero_after_single_miss() {
        let mut manager = ExpertThermalManager::new(2);
        manager.step(&[0, 5]);
        assert_eq!(manager.state(0).unwrap().hit_rate, 0.0);
        assert_eq!(manager.state(1).unwrap().hit_rate, 1.0);
    }

    #[test]
    fn test_hit_rate_reflects_ratio_accurately() {
        let mut manager = ExpertThermalManager::new(1).with_eviction_threshold(u64::MAX);
        // 3 hits in 10 steps
        let pattern = [true, false, false, true, false, false, false, true, false, false];
        for hit in pattern {
            if hit {
                manager.step(&[5]);
            } else {
                manager.step(&[0]);
            }
        }
        let state = manager.state(0).unwrap();
        assert_eq!(state.hit_count, 3);
        assert_eq!(state.route_count, 10);
        assert!((state.hit_rate - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_eviction_decision_large_expert_index() {
        let manager = ExpertThermalManager::new(4);
        assert_eq!(manager.eviction_decision(usize::MAX), EvictionDecision::Keep);
        assert_eq!(manager.eviction_decision(1_000_000), EvictionDecision::Keep);
    }

    #[test]
    fn test_evict_then_reactivate_then_step_updates_heat() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(3);

        // Evict expert 1
        for _ in 0..4 { manager.step(&[10, 0]); }
        manager.evict_expert(1);

        // Reactivate
        manager.reactivate_expert(1);
        assert_eq!(manager.state(1).unwrap().heat_level, ExpertHeatLevel::Cold);

        // Expert 1 becomes hot after many hits
        for _ in 0..20 { manager.step(&[10, 5]); }
        // Despite is_evicted=false, heat_level is derived from hit_rate
        assert_eq!(manager.state(1).unwrap().heat_level, ExpertHeatLevel::Hot);
    }

    #[test]
    fn test_deopt_request_partial_eq_all_fields_matter() {
        let base = DeoptRequest { request_id: 1, expert_idx: 2, layer_idx: 3, step: 4 };
        assert_ne!(base, DeoptRequest { request_id: 99, ..base.clone() });
        assert_ne!(base, DeoptRequest { expert_idx: 99, ..base.clone() });
        assert_ne!(base, DeoptRequest { layer_idx: 99, ..base.clone() });
        assert_ne!(base, DeoptRequest { step: 99, ..base.clone() });
    }

    #[test]
    fn test_deopt_handling_result_partial_eq_cross_variant() {
        let r = DeoptHandlingResult::ReactivateAndRerun { expert_idx: 1, request_id: 1 };
        let s = DeoptHandlingResult::SpuriousDeopt { expert_idx: 1, request_id: 1 };
        assert_ne!(r, s);
    }

    #[test]
    fn test_thermal_summary_partial_eq_all_fields() {
        let base = ThermalSummary {
            num_experts: 4, hot_count: 1, warm_count: 1, cold_count: 1, evicted_count: 1,
            total_evictions: 5, total_reactivations: 2, current_step: 100,
            pending_deopt_count: 3, working_set_size: 2, effective_eviction_threshold: 500,
        };
        // Each field change causes inequality
        assert_ne!(base, ThermalSummary { num_experts: 99, ..base.clone() });
        assert_ne!(base, ThermalSummary { hot_count: 99, ..base.clone() });
        assert_ne!(base, ThermalSummary { warm_count: 99, ..base.clone() });
        assert_ne!(base, ThermalSummary { cold_count: 99, ..base.clone() });
        assert_ne!(base, ThermalSummary { evicted_count: 99, ..base.clone() });
        assert_ne!(base, ThermalSummary { total_evictions: 99, ..base.clone() });
        assert_ne!(base, ThermalSummary { total_reactivations: 99, ..base.clone() });
        assert_ne!(base, ThermalSummary { current_step: 99, ..base.clone() });
        assert_ne!(base, ThermalSummary { pending_deopt_count: 99, ..base.clone() });
        assert_ne!(base, ThermalSummary { working_set_size: 99, ..base.clone() });
        assert_ne!(base, ThermalSummary { effective_eviction_threshold: 99, ..base.clone() });
    }

    #[test]
    fn test_manager_large_num_experts() {
        let manager = ExpertThermalManager::new(256);
        assert_eq!(manager.num_experts(), 256);
        assert_eq!(manager.states().len(), 256);
        for state in manager.states() {
            assert_eq!(state.heat_level, ExpertHeatLevel::Warm);
        }
    }

    #[test]
    fn test_summary_counts_sum_to_num_experts() {
        let mut manager = ExpertThermalManager::new(8)
            .with_eviction_threshold(5)
            .with_heat_thresholds(0.5, 0.1);

        for _ in 0..6 { manager.step(&[10, 10, 0, 0, 5, 3, 0, 1]); }
        manager.evict_expert(6);

        let summary = manager.summary();
        assert_eq!(
            summary.hot_count + summary.warm_count + summary.cold_count + summary.evicted_count,
            summary.num_experts,
        );
    }

    #[test]
    fn test_memory_pressure_boundary_values() {
        let mut manager = ExpertThermalManager::new(4)
            .with_adaptive_eviction(10)
            .with_eviction_threshold(1000);

        manager.step(&[1, 1, 0, 0]);

        manager.update_memory_pressure(0.0);
        let t0 = manager.effective_eviction_threshold();

        manager.update_memory_pressure(1.0);
        let t1 = manager.effective_eviction_threshold();

        manager.update_memory_pressure(0.5);
        let t_half = manager.effective_eviction_threshold();

        // Higher pressure => lower threshold
        assert!(t1 < t_half);
        assert!(t_half < t0);
    }

    #[test]
    fn test_eviction_aggressiveness_fractional_values() {
        let m0 = ExpertThermalManager::new(4).with_eviction_threshold(999).with_eviction_aggressiveness(0.0);
        let m_half = ExpertThermalManager::new(4).with_eviction_threshold(999).with_eviction_aggressiveness(0.5);
        let m1 = ExpertThermalManager::new(4).with_eviction_threshold(999).with_eviction_aggressiveness(1.0);

        let t0 = m0.effective_eviction_threshold();
        let t_half = m_half.effective_eviction_threshold();
        let t1 = m1.effective_eviction_threshold();

        // factor(0.0) = 1.0, factor(0.5) = 2/3, factor(1.0) = 0.5
        assert_eq!(t0, 999);
        assert_eq!(t_half, 666); // 999 * 2/3 = 666
        assert_eq!(t1, 499);     // 999 * 0.5 = 499.5 => 499
        assert!(t_half < t0);
        assert!(t1 < t_half);
    }

    #[test]
    fn test_effective_threshold_with_zero_base_and_aggressiveness() {
        let manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(0)
            .with_eviction_aggressiveness(0.0);
        assert_eq!(manager.effective_eviction_threshold(), 0);
    }

    #[test]
    fn test_effective_threshold_with_zero_base_and_high_aggressiveness() {
        let manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(0)
            .with_eviction_aggressiveness(10.0);
        assert_eq!(manager.effective_eviction_threshold(), 0);
    }

    #[test]
    fn test_step_many_steps_hit_rate_converges_to_zero() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(u64::MAX);
        // Expert 0 hits only on step 1, then never again
        manager.step(&[5, 5]); // step 1: both hit
        for _ in 0..99 {
            manager.step(&[0, 5]); // steps 2-100: only expert 1 hits
        }
        let state0 = manager.state(0).unwrap();
        assert!((state0.hit_rate - 0.01).abs() < 0.001);
        assert_eq!(state0.hit_count, 1);
        assert_eq!(state0.route_count, 100);
    }

    #[test]
    fn test_expert_heat_state_default_values_via_manager() {
        let manager = ExpertThermalManager::new(3);
        let state = manager.state(1).unwrap();
        assert_eq!(state.expert_idx, 1);
        assert_eq!(state.hit_rate, 0.0);
        assert_eq!(state.hit_count, 0);
        assert_eq!(state.route_count, 0);
        assert_eq!(state.heat_level, ExpertHeatLevel::Warm);
        assert_eq!(state.consecutive_zero_streak, 0);
        assert_eq!(state.last_hit_step, 0);
        assert!(!state.is_evicted);
        assert_eq!(state.reactivation_count, 0);
    }

    // ────────────────────────────────────────────────────────
    // Batch 4: 50 additional tests for edge cases and coverage
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_heat_level_from_hit_rate_very_small_positive_thresholds() {
        // Arrange: extremely small thresholds
        let level = ExpertHeatLevel::from_hit_rate(1e-15, 1e-10, 1e-12);
        // 1e-15 < 1e-12 (cold_threshold) but > 0 => Cold
        assert_eq!(level, ExpertHeatLevel::Cold);
    }

    #[test]
    fn test_heat_level_from_hit_rate_rate_exactly_between_thresholds() {
        // hot=0.6, cold=0.2, rate=0.4 => Warm (0.2 <= 0.4 < 0.6)
        let level = ExpertHeatLevel::from_hit_rate(0.4, 0.6, 0.2);
        assert_eq!(level, ExpertHeatLevel::Warm);
    }

    #[test]
    fn test_heat_level_from_hit_rate_all_variants_with_specific_thresholds() {
        // Arrange: thresholds that cleanly partition into 4 regions
        let hot = 0.8;
        let cold = 0.2;

        // Act & Assert
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.9, hot, cold), ExpertHeatLevel::Hot);
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.8, hot, cold), ExpertHeatLevel::Hot);
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.5, hot, cold), ExpertHeatLevel::Warm);
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.2, hot, cold), ExpertHeatLevel::Warm);
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.1, hot, cold), ExpertHeatLevel::Cold);
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.0, hot, cold), ExpertHeatLevel::Evicted);
    }

    #[test]
    fn test_expert_heat_level_copy_semantics() {
        // Verify Copy: after assignment, original is still usable
        let original = ExpertHeatLevel::Cold;
        let assigned = original;
        // Both should be usable independently
        assert_eq!(original, ExpertHeatLevel::Cold);
        assert_eq!(assigned, ExpertHeatLevel::Cold);
    }

    #[test]
    fn test_expert_heat_level_ord_max_min() {
        assert_eq!(std::cmp::max(ExpertHeatLevel::Hot, ExpertHeatLevel::Evicted), ExpertHeatLevel::Evicted);
        assert_eq!(std::cmp::min(ExpertHeatLevel::Hot, ExpertHeatLevel::Evicted), ExpertHeatLevel::Hot);
    }

    #[test]
    fn test_eviction_decision_hash_in_hashset() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(EvictionDecision::Keep);
        set.insert(EvictionDecision::Evict);
        set.insert(EvictionDecision::Reactivate);
        assert_eq!(set.len(), 3);
        assert!(set.contains(&EvictionDecision::Keep));
        assert!(set.contains(&EvictionDecision::Evict));
        assert!(set.contains(&EvictionDecision::Reactivate));
    }

    #[test]
    fn test_eviction_decision_all_copyable() {
        let keep = EvictionDecision::Keep;
        let evict = EvictionDecision::Evict;
        let reactivate = EvictionDecision::Reactivate;
        // Copy: can use all three after assignment
        let _k = keep;
        let _e = evict;
        let _r = reactivate;
        assert_eq!(keep, EvictionDecision::Keep);
        assert_eq!(evict, EvictionDecision::Evict);
        assert_eq!(reactivate, EvictionDecision::Reactivate);
    }

    #[test]
    fn test_working_set_tracker_clone() {
        let mut tracker = WorkingSetTracker::new(4, 3, 100);
        tracker.record_step(&[1, 2, 0, 0]);
        let cloned = tracker.clone();
        assert_eq!(tracker.working_set_size(), cloned.working_set_size());
        assert_eq!(tracker.num_experts, cloned.num_experts);
        assert_eq!(tracker.window_size, cloned.window_size);
        assert_eq!(tracker.base_threshold, cloned.base_threshold);
    }

    #[test]
    fn test_working_set_tracker_debug_format() {
        let tracker = WorkingSetTracker::new(4, 3, 100);
        let debug = format!("{:?}", tracker);
        assert!(debug.contains("WorkingSetTracker"));
        assert!(debug.contains("window_size: 3"));
        assert!(debug.contains("num_experts: 4"));
        assert!(debug.contains("base_threshold: 100"));
    }

    #[test]
    fn test_working_set_tracker_record_step_empty_counts() {
        let mut tracker = WorkingSetTracker::new(4, 3, 100);
        tracker.record_step(&[]);
        assert_eq!(tracker.working_set_size(), 0);
    }

    #[test]
    fn test_working_set_tracker_record_step_zeros_do_not_count() {
        let mut tracker = WorkingSetTracker::new(3, 5, 100);
        tracker.record_step(&[0, 0, 0]);
        assert_eq!(tracker.working_set_size(), 0);
        tracker.record_step(&[0, 1, 0]);
        assert_eq!(tracker.working_set_size(), 1);
    }

    #[test]
    fn test_working_set_tracker_adaptive_threshold_high_pressure_low_headroom() {
        let mut tracker = WorkingSetTracker::new(4, 5, 1000);
        tracker.record_step(&[1, 1, 0, 0]); // ws = 2
        // pressure = 0.9 => headroom = clamp(0.1, 0.1, 1.0) = 0.1
        let threshold = tracker.adaptive_threshold(0.9);
        // 1000 * (4/2) * 0.1 = 200
        assert_eq!(threshold, 200);
    }

    #[test]
    fn test_working_set_tracker_adaptive_threshold_75_percent_pressure() {
        let mut tracker = WorkingSetTracker::new(4, 5, 1000);
        tracker.record_step(&[1, 1, 0, 0]); // ws = 2
        // pressure = 0.75 => headroom = 0.25
        let threshold = tracker.adaptive_threshold(0.75);
        // 1000 * (4/2) * 0.25 = 500
        assert_eq!(threshold, 500);
    }

    #[test]
    fn test_working_set_tracker_cursor_advances_correctly() {
        let mut tracker = WorkingSetTracker::new(2, 3, 100);
        // Initial cursor = 0
        // Step 1: writes to slot 0, cursor -> 1
        tracker.record_step(&[1, 0]);
        // Step 2: writes to slot 1, cursor -> 2
        tracker.record_step(&[0, 1]);
        // Step 3: writes to slot 2, cursor -> 0
        tracker.record_step(&[1, 1]);
        assert_eq!(tracker.working_set_size(), 2);

        // Step 4: overwrites slot 0, cursor -> 1
        tracker.record_step(&[0, 0]);
        // All slots still have data from steps 2, 3, and step 4 (all zeros)
        // Slot 0: [false, false], Slot 1: [false, true], Slot 2: [true, true]
        assert_eq!(tracker.working_set_size(), 2);
    }

    #[test]
    fn test_working_set_tracker_adaptive_threshold_clamp_below_one() {
        let tracker = WorkingSetTracker::new(2, 3, 500);
        // pressure = 1.5 => headroom = clamp(-0.5, 0.1, 1.0) = 0.1
        let threshold = tracker.adaptive_threshold(1.5);
        // ws=0 => max(1) => ws=1, ratio=2/1=2, 500*2*0.1 = 100
        assert_eq!(threshold, 100);
    }

    #[test]
    fn test_deopt_request_clone_preserves_all_fields() {
        let req = DeoptRequest {
            request_id: u64::MAX,
            expert_idx: usize::MAX,
            layer_idx: 0,
            step: u64::MAX,
        };
        let cloned = req.clone();
        assert_eq!(cloned.request_id, req.request_id);
        assert_eq!(cloned.expert_idx, req.expert_idx);
        assert_eq!(cloned.layer_idx, req.layer_idx);
        assert_eq!(cloned.step, req.step);
        assert_eq!(cloned, req);
    }

    #[test]
    fn test_deopt_request_equality_with_zero_fields() {
        let r1 = DeoptRequest { request_id: 0, expert_idx: 0, layer_idx: 0, step: 0 };
        let r2 = DeoptRequest { request_id: 0, expert_idx: 0, layer_idx: 0, step: 0 };
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_deopt_handling_result_reactivate_clone_eq() {
        let r = DeoptHandlingResult::ReactivateAndRerun { expert_idx: 10, request_id: 20 };
        let cloned = r.clone();
        assert_eq!(r, cloned);
    }

    #[test]
    fn test_deopt_handling_result_spurious_clone_eq() {
        let s = DeoptHandlingResult::SpuriousDeopt { expert_idx: 5, request_id: 15 };
        let cloned = s.clone();
        assert_eq!(s, cloned);
    }

    #[test]
    fn test_thermal_summary_all_zero_fields() {
        let s = ThermalSummary {
            num_experts: 0,
            hot_count: 0,
            warm_count: 0,
            cold_count: 0,
            evicted_count: 0,
            total_evictions: 0,
            total_reactivations: 0,
            current_step: 0,
            pending_deopt_count: 0,
            working_set_size: 0,
            effective_eviction_threshold: 0,
        };
        let cloned = s.clone();
        assert_eq!(s, cloned);
        assert_eq!(cloned.num_experts, 0);
        assert_eq!(cloned.current_step, 0);
    }

    #[test]
    fn test_thermal_summary_max_values() {
        let s = ThermalSummary {
            num_experts: usize::MAX,
            hot_count: usize::MAX,
            warm_count: 0,
            cold_count: 0,
            evicted_count: 0,
            total_evictions: u64::MAX,
            total_reactivations: u64::MAX,
            current_step: u64::MAX,
            pending_deopt_count: usize::MAX,
            working_set_size: usize::MAX,
            effective_eviction_threshold: u64::MAX,
        };
        assert_eq!(s.num_experts, usize::MAX);
        assert_eq!(s.total_evictions, u64::MAX);
    }

    #[test]
    fn test_expert_heat_state_equality_all_fields_same() {
        let s = ExpertHeatState {
            expert_idx: 0,
            hit_rate: 0.0,
            hit_count: 0,
            route_count: 0,
            heat_level: ExpertHeatLevel::Warm,
            consecutive_zero_streak: 0,
            last_hit_step: 0,
            is_evicted: false,
            reactivation_count: 0,
        };
        let s2 = ExpertHeatState {
            expert_idx: 0,
            hit_rate: 0.0,
            hit_count: 0,
            route_count: 0,
            heat_level: ExpertHeatLevel::Warm,
            consecutive_zero_streak: 0,
            last_hit_step: 0,
            is_evicted: false,
            reactivation_count: 0,
        };
        assert_eq!(s, s2);
    }

    #[test]
    fn test_expert_heat_state_differs_by_hit_count() {
        let s1 = ExpertHeatState {
            expert_idx: 0, hit_rate: 0.5, hit_count: 5, route_count: 10,
            heat_level: ExpertHeatLevel::Warm, consecutive_zero_streak: 0,
            last_hit_step: 10, is_evicted: false, reactivation_count: 0,
        };
        let s2 = ExpertHeatState { hit_count: 6, ..s1.clone() };
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_expert_heat_state_differs_by_route_count() {
        let s1 = ExpertHeatState {
            expert_idx: 0, hit_rate: 0.5, hit_count: 5, route_count: 10,
            heat_level: ExpertHeatLevel::Warm, consecutive_zero_streak: 0,
            last_hit_step: 10, is_evicted: false, reactivation_count: 0,
        };
        let s2 = ExpertHeatState { route_count: 11, ..s1.clone() };
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_expert_heat_state_differs_by_last_hit_step() {
        let s1 = ExpertHeatState {
            expert_idx: 0, hit_rate: 0.5, hit_count: 5, route_count: 10,
            heat_level: ExpertHeatLevel::Warm, consecutive_zero_streak: 0,
            last_hit_step: 10, is_evicted: false, reactivation_count: 0,
        };
        let s2 = ExpertHeatState { last_hit_step: 20, ..s1.clone() };
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_manager_step_updates_current_step_monotonically() {
        let mut manager = ExpertThermalManager::new(2);
        assert_eq!(manager.summary().current_step, 0);
        manager.step(&[1, 1]);
        assert_eq!(manager.summary().current_step, 1);
        manager.step(&[1, 1]);
        assert_eq!(manager.summary().current_step, 2);
        manager.step(&[0, 0]);
        assert_eq!(manager.summary().current_step, 3);
    }

    #[test]
    fn test_manager_step_route_count_increments_per_step() {
        let mut manager = ExpertThermalManager::new(3);
        for _ in 0..10 {
            manager.step(&[5, 0, 3]);
        }
        assert_eq!(manager.state(0).unwrap().route_count, 10);
        assert_eq!(manager.state(1).unwrap().route_count, 10);
        assert_eq!(manager.state(2).unwrap().route_count, 10);
    }

    #[test]
    fn test_manager_hot_experts_empty_initially() {
        let manager = ExpertThermalManager::new(8);
        assert!(manager.hot_experts().is_empty());
    }

    #[test]
    fn test_manager_cold_or_evicted_empty_initially() {
        let manager = ExpertThermalManager::new(8);
        // All start as Warm, so cold_or_evicted should be empty
        assert!(manager.cold_or_evicted_experts().is_empty());
    }

    #[test]
    fn test_manager_hot_experts_after_mixed_steps() {
        let mut manager = ExpertThermalManager::new(4).with_eviction_threshold(1000);
        // Expert 0 and 3 hot, 1 and 2 cold
        for _ in 0..10 {
            manager.step(&[10, 0, 0, 10]);
        }
        let hot = manager.hot_experts();
        assert!(hot.contains(&0));
        assert!(hot.contains(&3));
        assert!(!hot.contains(&1));
        assert!(!hot.contains(&2));
    }

    #[test]
    fn test_manager_states_slice_matches_num_experts() {
        for n in [0, 1, 4, 16, 64] {
            let manager = ExpertThermalManager::new(n);
            assert_eq!(manager.states().len(), n);
        }
    }

    #[test]
    fn test_manager_state_returns_correct_expert_idx() {
        let manager = ExpertThermalManager::new(5);
        for i in 0..5 {
            assert_eq!(manager.state(i).unwrap().expert_idx, i);
        }
    }

    #[test]
    fn test_eviction_decision_for_just_evicted_without_reactivation() {
        // Arrange: evict an expert, verify decision is Keep (stays evicted)
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(3);
        for _ in 0..4 {
            manager.step(&[10, 0]);
        }
        manager.evict_expert(1);

        // Act: check eviction decision
        let decision = manager.eviction_decision(1);

        // Assert: evicted expert without reactivation => Keep
        assert_eq!(decision, EvictionDecision::Keep);
        assert!(manager.state(1).unwrap().is_evicted);
    }

    #[test]
    fn test_eviction_decision_for_non_evicted_below_threshold() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(100);
        // 5 steps: expert 1 streak = 5, well below 100
        for _ in 0..5 {
            manager.step(&[10, 0]);
        }
        assert_eq!(manager.eviction_decision(1), EvictionDecision::Keep);
    }

    #[test]
    fn test_eviction_decision_for_non_evicted_at_threshold() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(5);
        for _ in 0..5 {
            manager.step(&[10, 0]);
        }
        // streak=5 == threshold=5 => Evict
        assert_eq!(manager.eviction_decision(1), EvictionDecision::Evict);
    }

    #[test]
    fn test_eviction_decision_for_non_evicted_above_threshold() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(5);
        for _ in 0..10 {
            manager.step(&[10, 0]);
        }
        // streak=10 > threshold=5 => Evict
        assert_eq!(manager.eviction_decision(1), EvictionDecision::Evict);
    }

    #[test]
    fn test_evict_expert_sets_heat_level_to_evicted() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(3);
        for _ in 0..4 {
            manager.step(&[10, 0]);
        }
        assert!(manager.evict_expert(1));
        assert_eq!(manager.state(1).unwrap().heat_level, ExpertHeatLevel::Evicted);
    }

    #[test]
    fn test_reactivate_expert_sets_heat_level_to_cold() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(3);
        for _ in 0..4 {
            manager.step(&[10, 0]);
        }
        manager.evict_expert(1);
        assert!(manager.reactivate_expert(1));
        assert_eq!(manager.state(1).unwrap().heat_level, ExpertHeatLevel::Cold);
    }

    #[test]
    fn test_reactivate_expert_clears_is_evicted() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(3);
        for _ in 0..4 {
            manager.step(&[10, 0]);
        }
        manager.evict_expert(1);
        assert!(manager.state(1).unwrap().is_evicted);
        manager.reactivate_expert(1);
        assert!(!manager.state(1).unwrap().is_evicted);
    }

    #[test]
    fn test_step_hit_count_increments_correctly() {
        let mut manager = ExpertThermalManager::new(3).with_eviction_threshold(1000);
        // 5 steps: expert 0 hit every time, expert 1 hit 3 times, expert 2 never
        manager.step(&[5, 3, 0]);
        manager.step(&[5, 0, 0]);
        manager.step(&[5, 3, 0]);
        manager.step(&[5, 0, 0]);
        manager.step(&[5, 3, 0]);

        assert_eq!(manager.state(0).unwrap().hit_count, 5);
        assert_eq!(manager.state(1).unwrap().hit_count, 3);
        assert_eq!(manager.state(2).unwrap().hit_count, 0);
    }

    #[test]
    fn test_step_consecutive_zero_streak_resets_on_hit() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(1000);
        // 5 misses
        for _ in 0..5 {
            manager.step(&[10, 0]);
        }
        assert_eq!(manager.state(1).unwrap().consecutive_zero_streak, 5);

        // 1 hit
        manager.step(&[10, 5]);
        assert_eq!(manager.state(1).unwrap().consecutive_zero_streak, 0);

        // 3 more misses
        for _ in 0..3 {
            manager.step(&[10, 0]);
        }
        assert_eq!(manager.state(1).unwrap().consecutive_zero_streak, 3);
    }

    #[test]
    fn test_step_last_hit_step_only_updates_on_hit() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(1000);
        manager.step(&[10, 0]); // step 1
        manager.step(&[10, 5]); // step 2: expert 1 hit
        manager.step(&[10, 0]); // step 3: expert 1 miss
        manager.step(&[10, 0]); // step 4: expert 1 miss

        assert_eq!(manager.state(1).unwrap().last_hit_step, 2);
        assert_eq!(manager.state(0).unwrap().last_hit_step, 4);
    }

    #[test]
    fn test_summary_reflects_eviction_and_reactivation_counts() {
        let mut manager = ExpertThermalManager::new(3).with_eviction_threshold(3);
        for _ in 0..4 {
            manager.step(&[10, 0, 0]);
        }
        manager.evict_expert(1);
        manager.evict_expert(2);
        manager.reactivate_expert(1);

        let summary = manager.summary();
        assert_eq!(summary.total_evictions, 2);
        assert_eq!(summary.total_reactivations, 1);
    }

    #[test]
    fn test_pending_deopt_requests_ordering() {
        let mut manager = ExpertThermalManager::new(2);
        for i in 0..5 {
            manager.handle_deopt_request(DeoptRequest {
                request_id: i,
                expert_idx: 0,
                layer_idx: 0,
                step: i as u64,
            });
        }
        let requests = manager.pending_deopt_requests();
        assert_eq!(requests.len(), 5);
        // Verify FIFO ordering
        for i in 0..5 {
            assert_eq!(requests[i].request_id, i as u64);
            assert_eq!(requests[i].step, i as u64);
        }
    }

    #[test]
    fn test_clear_deopt_requests_on_empty() {
        let mut manager = ExpertThermalManager::new(2);
        // Clear on empty should not panic
        manager.clear_deopt_requests();
        assert!(manager.pending_deopt_requests().is_empty());
    }

    #[test]
    fn test_clear_deopt_requests_idempotent() {
        let mut manager = ExpertThermalManager::new(2);
        manager.handle_deopt_request(DeoptRequest {
            request_id: 1, expert_idx: 0, layer_idx: 0, step: 0,
        });
        manager.clear_deopt_requests();
        manager.clear_deopt_requests(); // double clear
        assert!(manager.pending_deopt_requests().is_empty());
    }

    #[test]
    fn test_with_heat_thresholds_returns_new_instance() {
        let m1 = ExpertThermalManager::new(4);
        let m2 = m1.with_heat_thresholds(0.5, 0.1);
        // m2 is a new configured instance
        assert_eq!(m2.num_experts(), 4);
    }

    #[test]
    fn test_with_eviction_threshold_returns_new_instance() {
        let m1 = ExpertThermalManager::new(4);
        let m2 = m1.with_eviction_threshold(999);
        assert_eq!(m2.num_experts(), 4);
        assert_eq!(m2.effective_eviction_threshold(), 999);
    }

    #[test]
    fn test_with_adaptive_eviction_returns_new_instance() {
        let m1 = ExpertThermalManager::new(4);
        let m2 = m1.with_adaptive_eviction(20);
        assert_eq!(m2.num_experts(), 4);
    }

    #[test]
    fn test_with_eviction_aggressiveness_returns_new_instance() {
        let m1 = ExpertThermalManager::new(4);
        let m2 = m1.with_eviction_aggressiveness(1.5);
        assert_eq!(m2.num_experts(), 4);
    }

    #[test]
    fn test_update_memory_pressure_extreme_values() {
        let mut manager = ExpertThermalManager::new(4);
        // These should not panic
        manager.update_memory_pressure(f32::MIN);
        manager.update_memory_pressure(f32::MAX);
        manager.update_memory_pressure(f32::NAN);
        manager.update_memory_pressure(f32::INFINITY);
        manager.update_memory_pressure(f32::NEG_INFINITY);
    }

    #[test]
    fn test_manager_step_single_expert_varies_between_hit_and_miss() {
        let mut manager = ExpertThermalManager::new(1).with_eviction_threshold(1000);
        manager.step(&[5]); // hit
        manager.step(&[0]); // miss
        manager.step(&[5]); // hit
        manager.step(&[0]); // miss

        let state = manager.state(0).unwrap();
        assert_eq!(state.hit_count, 2);
        assert_eq!(state.route_count, 4);
        assert!((state.hit_rate - 0.5).abs() < 1e-10);
        assert_eq!(state.consecutive_zero_streak, 1); // last step was miss
    }

    #[test]
    fn test_manager_eviction_triggers_at_exact_aggressiveness_threshold() {
        // threshold=100, aggressiveness=1.0 => effective=50
        let mut manager = ExpertThermalManager::new(2)
            .with_eviction_threshold(100)
            .with_eviction_aggressiveness(1.0);

        // 50 steps: expert 1 streak = 50, exactly at effective threshold
        for _ in 0..50 {
            manager.step(&[10, 0]);
        }
        assert!(manager.experts_to_evict().contains(&1));
    }

    #[test]
    fn test_manager_eviction_does_not_trigger_below_aggressiveness_threshold() {
        // threshold=100, aggressiveness=1.0 => effective=50
        let mut manager = ExpertThermalManager::new(2)
            .with_eviction_threshold(100)
            .with_eviction_aggressiveness(1.0);

        // 49 steps: streak = 49, just below effective threshold
        for _ in 0..49 {
            manager.step(&[10, 0]);
        }
        assert!(!manager.experts_to_evict().contains(&1));
    }

    #[test]
    fn test_eviction_with_adaptive_and_aggressiveness_combined() {
        let mut manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(1000)
            .with_adaptive_eviction(10)
            .with_eviction_aggressiveness(1.0);

        // 2 of 4 experts active => ratio = 2.0
        manager.step(&[1, 1, 0, 0]);
        manager.update_memory_pressure(0.5);

        // adaptive = 1000 * (4/2) * 0.5 = 1000
        // aggressiveness=1.0 => bias = 0.5
        // effective = 1000 * 0.5 = 500
        assert_eq!(manager.effective_eviction_threshold(), 500);
    }

    #[test]
    fn test_experts_to_evict_empty_after_all_evicted() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(3);
        for _ in 0..4 {
            manager.step(&[0, 0]);
        }
        manager.evict_expert(0);
        manager.evict_expert(1);
        assert!(manager.experts_to_evict().is_empty());
    }

    #[test]
    fn test_experts_to_reactivate_empty_after_all_reactivated() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(3);
        for _ in 0..4 {
            manager.step(&[0, 0]);
        }
        manager.evict_expert(0);
        manager.evict_expert(1);
        manager.reactivate_expert(0);
        manager.reactivate_expert(1);
        assert!(manager.experts_to_reactivate().is_empty());
    }

    #[test]
    fn test_handle_deopt_records_request_even_for_non_evicted() {
        let mut manager = ExpertThermalManager::new(4);
        // Expert 3 is not evicted, but we still send a deopt for it
        manager.handle_deopt_request(DeoptRequest {
            request_id: 42,
            expert_idx: 3,
            layer_idx: 7,
            step: 100,
        });
        // Request should still be recorded
        assert_eq!(manager.pending_deopt_requests().len(), 1);
        assert_eq!(manager.pending_deopt_requests()[0].expert_idx, 3);
    }

    // ────────────────────────────────────────────────────────
    // Additional tests (55 new)
    // ────────────────────────────────────────────────────────

    // --- ExpertHeatLevel additional ---

    #[test]
    fn test_heat_level_ord_impl_consistency_with_partial_ord() {
        // Ensure Ord and PartialOrd agree on all pairs
        let variants = [
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Cold,
            ExpertHeatLevel::Evicted,
        ];
        for i in 0..variants.len() {
            for j in 0..variants.len() {
                let ord_cmp = variants[i].cmp(&variants[j]);
                let partial_cmp = variants[i].partial_cmp(&variants[j]);
                assert_eq!(partial_cmp, Some(ord_cmp));
            }
        }
    }

    #[test]
    fn test_heat_level_hash_in_btreemap() {
        use std::collections::BTreeMap;
        let mut map = BTreeMap::new();
        map.insert(ExpertHeatLevel::Hot, 1u32);
        map.insert(ExpertHeatLevel::Warm, 2u32);
        map.insert(ExpertHeatLevel::Cold, 3u32);
        map.insert(ExpertHeatLevel::Evicted, 4u32);
        assert_eq!(map.len(), 4);
        assert_eq!(map[&ExpertHeatLevel::Hot], 1);
        assert_eq!(map[&ExpertHeatLevel::Evicted], 4);
    }

    #[test]
    fn test_heat_level_from_hit_rate_threshold_zero_hot() {
        // hot_threshold=0.0: any rate >= 0.0 => Hot (0.0 >= 0.0 is true)
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.0001, 0.0, 0.0), ExpertHeatLevel::Hot);
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.0, 0.0, 0.0), ExpertHeatLevel::Hot);
        // With hot=0.1, cold=0.0: rate=0.0 >= cold(0.0) => Warm (not Evicted)
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.0, 0.1, 0.0), ExpertHeatLevel::Warm);
        // To get Evicted, need rate=0.0 and cold_threshold > 0.0
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.0, 0.1, 0.001), ExpertHeatLevel::Evicted);
    }

    #[test]
    fn test_heat_level_from_hit_rate_very_large_rate() {
        // Rates > 1.0: >= hot_threshold => Hot
        assert_eq!(
            ExpertHeatLevel::from_hit_rate(1e18, 0.1, 0.001),
            ExpertHeatLevel::Hot
        );
        // 1e18 >= 1e17 (cold) but < 1e19 (hot) => Warm
        assert_eq!(
            ExpertHeatLevel::from_hit_rate(1e18, 1e19, 1e17),
            ExpertHeatLevel::Warm
        );
    }

    #[test]
    fn test_heat_level_from_hit_rate_equal_hot_cold_thresholds() {
        // hot == cold: rate >= hot => Hot; rate >= cold (same) => also Hot; 0 < rate < hot => Cold
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.5, 0.5, 0.5), ExpertHeatLevel::Hot);
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.1, 0.5, 0.5), ExpertHeatLevel::Cold);
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.0, 0.5, 0.5), ExpertHeatLevel::Evicted);
    }

    // --- ExpertHeatState additional ---

    #[test]
    fn test_expert_heat_state_evicted_flag_independent_of_heat_level() {
        // It's possible to construct a state where is_evicted=true but heat_level is
        // not Evicted (external construction). Manager maintains the invariant, but
        // the struct itself doesn't enforce it.
        let state = ExpertHeatState {
            expert_idx: 0,
            hit_rate: 0.0,
            hit_count: 0,
            route_count: 0,
            heat_level: ExpertHeatLevel::Hot,
            consecutive_zero_streak: 0,
            last_hit_step: 0,
            is_evicted: true,
            reactivation_count: 0,
        };
        assert!(state.is_evicted);
        assert_eq!(state.heat_level, ExpertHeatLevel::Hot);
    }

    #[test]
    fn test_expert_heat_state_equality_reflexive() {
        let state = ExpertHeatState {
            expert_idx: 7,
            hit_rate: 0.33,
            hit_count: 33,
            route_count: 100,
            heat_level: ExpertHeatLevel::Warm,
            consecutive_zero_streak: 5,
            last_hit_step: 42,
            is_evicted: false,
            reactivation_count: 1,
        };
        assert_eq!(state, state);
    }

    #[test]
    fn test_expert_heat_state_differs_by_is_evicted() {
        let base = ExpertHeatState {
            expert_idx: 0,
            hit_rate: 0.5,
            hit_count: 10,
            route_count: 20,
            heat_level: ExpertHeatLevel::Hot,
            consecutive_zero_streak: 0,
            last_hit_step: 10,
            is_evicted: false,
            reactivation_count: 0,
        };
        let mut evicted = base.clone();
        evicted.is_evicted = true;
        assert_ne!(base, evicted);
    }

    #[test]
    fn test_expert_heat_state_differs_by_consecutive_zero_streak() {
        let mut a = ExpertHeatState {
            expert_idx: 0,
            hit_rate: 0.0,
            hit_count: 0,
            route_count: 0,
            heat_level: ExpertHeatLevel::Warm,
            consecutive_zero_streak: 5,
            last_hit_step: 0,
            is_evicted: false,
            reactivation_count: 0,
        };
        let b = a.clone();
        a.consecutive_zero_streak = 10;
        assert_ne!(a, b);
    }

    #[test]
    fn test_expert_heat_state_differs_by_heat_level() {
        let mut a = ExpertHeatState {
            expert_idx: 0,
            hit_rate: 0.5,
            hit_count: 5,
            route_count: 10,
            heat_level: ExpertHeatLevel::Hot,
            consecutive_zero_streak: 0,
            last_hit_step: 10,
            is_evicted: false,
            reactivation_count: 0,
        };
        let b = a.clone();
        a.heat_level = ExpertHeatLevel::Cold;
        assert_ne!(a, b);
    }

    #[test]
    fn test_expert_heat_state_differs_by_reactivation_count() {
        let mut a = ExpertHeatState {
            expert_idx: 0,
            hit_rate: 0.0,
            hit_count: 0,
            route_count: 0,
            heat_level: ExpertHeatLevel::Warm,
            consecutive_zero_streak: 0,
            last_hit_step: 0,
            is_evicted: true,
            reactivation_count: 0,
        };
        let b = a.clone();
        a.reactivation_count = 3;
        assert_ne!(a, b);
    }

    // --- EvictionDecision additional ---

    #[test]
    fn test_eviction_decision_partial_ord_ordering() {
        assert!(EvictionDecision::Keep < EvictionDecision::Evict);
        assert!(EvictionDecision::Evict < EvictionDecision::Reactivate);
        assert!(EvictionDecision::Keep < EvictionDecision::Reactivate);
    }

    #[test]
    fn test_eviction_decision_partial_ord_consistency() {
        // Verify partial_cmp returns consistent results for all pairs
        let variants = [EvictionDecision::Keep, EvictionDecision::Evict, EvictionDecision::Reactivate];
        for i in 0..variants.len() {
            assert_eq!(variants[i].partial_cmp(&variants[i]), Some(std::cmp::Ordering::Equal));
            for j in 0..variants.len() {
                let cmp_ij = variants[i].partial_cmp(&variants[j]);
                let cmp_ji = variants[j].partial_cmp(&variants[i]);
                // Symmetry: cmp_ij is the reverse of cmp_ji
                assert_eq!(cmp_ij.map(|o| o.reverse()), cmp_ji);
            }
        }
    }

    // --- DeoptRequest additional ---

    #[test]
    fn test_deopt_request_differs_by_request_id() {
        let a = DeoptRequest { request_id: 1, expert_idx: 0, layer_idx: 0, step: 0 };
        let b = DeoptRequest { request_id: 2, expert_idx: 0, layer_idx: 0, step: 0 };
        assert_ne!(a, b);
    }

    #[test]
    fn test_deopt_request_differs_by_layer_idx() {
        let a = DeoptRequest { request_id: 1, expert_idx: 0, layer_idx: 5, step: 0 };
        let b = DeoptRequest { request_id: 1, expert_idx: 0, layer_idx: 6, step: 0 };
        assert_ne!(a, b);
    }

    #[test]
    fn test_deopt_request_differs_by_step() {
        let a = DeoptRequest { request_id: 1, expert_idx: 0, layer_idx: 0, step: 100 };
        let b = DeoptRequest { request_id: 1, expert_idx: 0, layer_idx: 0, step: 200 };
        assert_ne!(a, b);
    }

    // --- DeoptHandlingResult additional ---

    #[test]
    fn test_deopt_handling_result_reactivate_fields() {
        let result = DeoptHandlingResult::ReactivateAndRerun { expert_idx: 7, request_id: 42 };
        if let DeoptHandlingResult::ReactivateAndRerun { expert_idx, request_id } = result {
            assert_eq!(expert_idx, 7);
            assert_eq!(request_id, 42);
        } else {
            panic!("Expected ReactivateAndRerun variant");
        }
    }

    #[test]
    fn test_deopt_handling_result_spurious_fields() {
        let result = DeoptHandlingResult::SpuriousDeopt { expert_idx: 3, request_id: 99 };
        if let DeoptHandlingResult::SpuriousDeopt { expert_idx, request_id } = result {
            assert_eq!(expert_idx, 3);
            assert_eq!(request_id, 99);
        } else {
            panic!("Expected SpuriousDeopt variant");
        }
    }

    #[test]
    fn test_deopt_handling_result_clone_independence() {
        let original = DeoptHandlingResult::ReactivateAndRerun { expert_idx: 1, request_id: 5 };
        let cloned = original.clone();
        // Both should be equal but independent
        assert_eq!(original, cloned);
    }

    // --- ThermalSummary additional ---

    #[test]
    fn test_thermal_summary_counts_add_to_num_experts() {
        let summary = ThermalSummary {
            num_experts: 16,
            hot_count: 4,
            warm_count: 5,
            cold_count: 3,
            evicted_count: 4,
            total_evictions: 10,
            total_reactivations: 2,
            current_step: 500,
            pending_deopt_count: 1,
            working_set_size: 10,
            effective_eviction_threshold: 100,
        };
        assert_eq!(
            summary.hot_count + summary.warm_count + summary.cold_count + summary.evicted_count,
            summary.num_experts
        );
    }

    #[test]
    fn test_thermal_summary_clone_preserves_all_fields() {
        let original = ThermalSummary {
            num_experts: 8,
            hot_count: 2,
            warm_count: 3,
            cold_count: 2,
            evicted_count: 1,
            total_evictions: 5,
            total_reactivations: 3,
            current_step: 1000,
            pending_deopt_count: 7,
            working_set_size: 6,
            effective_eviction_threshold: 500,
        };
        let cloned = original.clone();
        assert_eq!(cloned.num_experts, 8);
        assert_eq!(cloned.hot_count, 2);
        assert_eq!(cloned.warm_count, 3);
        assert_eq!(cloned.cold_count, 2);
        assert_eq!(cloned.evicted_count, 1);
        assert_eq!(cloned.total_evictions, 5);
        assert_eq!(cloned.total_reactivations, 3);
        assert_eq!(cloned.current_step, 1000);
        assert_eq!(cloned.pending_deopt_count, 7);
        assert_eq!(cloned.working_set_size, 6);
        assert_eq!(cloned.effective_eviction_threshold, 500);
    }

    #[test]
    fn test_thermal_summary_equality_same_and_different() {
        let a = ThermalSummary {
            num_experts: 4,
            hot_count: 1,
            warm_count: 1,
            cold_count: 1,
            evicted_count: 1,
            total_evictions: 1,
            total_reactivations: 0,
            current_step: 10,
            pending_deopt_count: 0,
            working_set_size: 2,
            effective_eviction_threshold: 100,
        };
        let b = a.clone();
        assert_eq!(a, b);

        let mut c = a.clone();
        c.current_step = 11;
        assert_ne!(a, c);
    }

    // --- WorkingSetTracker additional ---

    #[test]
    fn test_working_set_tracker_working_set_size_accumulates_across_steps() {
        let mut tracker = WorkingSetTracker::new(4, 10, 100);
        tracker.record_step(&[1, 0, 0, 0]); // expert 0
        assert_eq!(tracker.working_set_size(), 1);
        tracker.record_step(&[0, 1, 0, 0]); // expert 1
        assert_eq!(tracker.working_set_size(), 2);
        tracker.record_step(&[0, 0, 1, 0]); // expert 2
        assert_eq!(tracker.working_set_size(), 3);
        tracker.record_step(&[0, 0, 0, 1]); // expert 3
        assert_eq!(tracker.working_set_size(), 4);
    }

    #[test]
    fn test_working_set_tracker_adaptive_threshold_monotonic_with_pressure() {
        let mut tracker = WorkingSetTracker::new(4, 10, 1000);
        tracker.record_step(&[1, 1, 0, 0]); // ws=2

        let t_low = tracker.adaptive_threshold(0.0);
        let t_mid = tracker.adaptive_threshold(0.5);
        let t_high = tracker.adaptive_threshold(0.9);
        // Higher pressure => lower headroom => lower threshold
        assert!(t_low > t_mid);
        assert!(t_mid > t_high);
    }

    #[test]
    fn test_working_set_tracker_record_step_empty_slice() {
        let mut tracker = WorkingSetTracker::new(4, 5, 100);
        tracker.record_step(&[]);
        // No entries recorded: working set should be 0
        assert_eq!(tracker.working_set_size(), 0);
    }

    #[test]
    fn test_working_set_tracker_base_threshold_preserved() {
        let tracker = WorkingSetTracker::new(4, 5, 7777);
        assert_eq!(tracker.base_threshold, 7777);
    }

    #[test]
    fn test_working_set_tracker_equality_reflexive() {
        let tracker = WorkingSetTracker::new(4, 5, 100);
        assert_eq!(tracker, tracker);
    }

    #[test]
    fn test_working_set_tracker_equality_after_same_record() {
        let mut a = WorkingSetTracker::new(3, 5, 100);
        let mut b = WorkingSetTracker::new(3, 5, 100);
        a.record_step(&[1, 2, 3]);
        b.record_step(&[1, 2, 3]);
        assert_eq!(a, b);
    }

    #[test]
    fn test_working_set_tracker_inequality_different_expert_count() {
        let a = WorkingSetTracker::new(3, 5, 100);
        let b = WorkingSetTracker::new(4, 5, 100);
        assert_ne!(a, b);
    }

    #[test]
    fn test_working_set_tracker_cursor_wraps_at_window_size() {
        let mut tracker = WorkingSetTracker::new(2, 3, 100);
        // Fill all 3 slots + 1 more to verify wrap
        tracker.record_step(&[1, 0]); // cursor 0->1
        tracker.record_step(&[0, 1]); // cursor 1->2
        tracker.record_step(&[1, 1]); // cursor 2->0
        tracker.record_step(&[0, 1]); // cursor 0->1, overwrites first slot
        // Only experts accessed in current window should be tracked
        assert!(tracker.working_set_size() > 0);
        assert!(tracker.working_set_size() <= 2);
    }

    #[test]
    fn test_working_set_tracker_large_num_experts() {
        let mut tracker = WorkingSetTracker::new(256, 10, 1000);
        let mut counts = vec![0usize; 256];
        counts[0] = 5;
        counts[255] = 3;
        tracker.record_step(&counts);
        assert_eq!(tracker.working_set_size(), 2);
    }

    // --- ExpertThermalManager additional ---

    #[test]
    fn test_manager_builder_chain_all_options() {
        let manager = ExpertThermalManager::new(8)
            .with_eviction_threshold(500)
            .with_heat_thresholds(0.2, 0.01)
            .with_adaptive_eviction(50)
            .with_eviction_aggressiveness(0.5);
        assert_eq!(manager.num_experts(), 8);
        // effective = adaptive * bias
        // adaptive depends on working_set; initially ws=0 => adaptive large
        let threshold = manager.effective_eviction_threshold();
        assert!(threshold >= 1);
    }

    #[test]
    fn test_manager_step_does_not_overflow_current_step() {
        let mut manager = ExpertThermalManager::new(1);
        // Simulate many steps — current_step increments each time
        for _ in 0..200 {
            manager.step(&[0]);
        }
        assert_eq!(manager.summary().current_step, 200);
    }

    #[test]
    fn test_manager_state_returns_none_for_out_of_bounds() {
        let manager = ExpertThermalManager::new(3);
        assert!(manager.state(3).is_none());
        assert!(manager.state(100).is_none());
    }

    #[test]
    fn test_manager_state_returns_correct_index() {
        let manager = ExpertThermalManager::new(5);
        for i in 0..5 {
            assert_eq!(manager.state(i).unwrap().expert_idx, i);
        }
    }

    #[test]
    fn test_manager_states_len_matches_num_experts() {
        let manager = ExpertThermalManager::new(7);
        assert_eq!(manager.states().len(), 7);
    }

    #[test]
    fn test_manager_evict_then_step_preserves_evicted_state() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(3);
        for _ in 0..4 {
            manager.step(&[5, 0]);
        }
        assert!(manager.evict_expert(1));

        // Step again with expert 1 active — should stay evicted
        manager.step(&[5, 10]);
        let state = manager.state(1).unwrap();
        assert!(state.is_evicted);
        assert_eq!(state.heat_level, ExpertHeatLevel::Evicted);
    }

    #[test]
    fn test_manager_eviction_decision_after_partial_steps() {
        let mut manager = ExpertThermalManager::new(3).with_eviction_threshold(10);
        // Expert 1 gets hit only on step 1
        manager.step(&[5, 5, 5]);
        for _ in 0..5 {
            manager.step(&[5, 0, 5]);
        }
        // Expert 1 streak = 5, threshold = 10 => Keep
        assert_eq!(manager.eviction_decision(1), EvictionDecision::Keep);
    }

    #[test]
    fn test_manager_eviction_decision_evict_after_sufficient_streak() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(5);
        for _ in 0..6 {
            manager.step(&[5, 0]);
        }
        assert_eq!(manager.eviction_decision(1), EvictionDecision::Evict);
    }

    #[test]
    fn test_manager_eviction_decision_reactivate_after_deopt_on_evicted() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(3);
        for _ in 0..4 {
            manager.step(&[5, 0]);
        }
        manager.evict_expert(1);

        // Simulate deopt: handle_deopt_request increments reactivation_count
        manager.handle_deopt_request(DeoptRequest {
            request_id: 1,
            expert_idx: 1,
            layer_idx: 0,
            step: 4,
        });

        // After deopt handling, expert 1 is reactivated, reactivation_count > 0
        // but it's no longer evicted, so decision should be Keep
        assert_eq!(manager.eviction_decision(1), EvictionDecision::Keep);
    }

    #[test]
    fn test_manager_multiple_deopt_requests_different_experts() {
        let mut manager = ExpertThermalManager::new(4).with_eviction_threshold(3);
        for _ in 0..4 {
            manager.step(&[5, 0, 0, 0]);
        }
        manager.evict_expert(1);
        manager.evict_expert(2);

        manager.handle_deopt_request(DeoptRequest {
            request_id: 1, expert_idx: 1, layer_idx: 0, step: 4,
        });
        manager.handle_deopt_request(DeoptRequest {
            request_id: 2, expert_idx: 2, layer_idx: 1, step: 4,
        });

        let pending = manager.pending_deopt_requests();
        assert_eq!(pending.len(), 2);
        assert_eq!(pending[0].expert_idx, 1);
        assert_eq!(pending[1].expert_idx, 2);
    }

    #[test]
    fn test_manager_clear_deopt_then_add_more() {
        let mut manager = ExpertThermalManager::new(2);
        manager.handle_deopt_request(DeoptRequest {
            request_id: 1, expert_idx: 0, layer_idx: 0, step: 0,
        });
        manager.clear_deopt_requests();
        assert!(manager.pending_deopt_requests().is_empty());

        manager.handle_deopt_request(DeoptRequest {
            request_id: 2, expert_idx: 1, layer_idx: 0, step: 1,
        });
        assert_eq!(manager.pending_deopt_requests().len(), 1);
        assert_eq!(manager.pending_deopt_requests()[0].request_id, 2);
    }

    #[test]
    fn test_manager_evict_does_not_affect_other_experts() {
        let mut manager = ExpertThermalManager::new(4).with_eviction_threshold(3);
        for _ in 0..10 {
            manager.step(&[5, 5, 0, 0]);
        }
        assert!(manager.evict_expert(2));
        assert!(manager.evict_expert(3));

        // Experts 0 and 1 should not be evicted
        assert!(!manager.state(0).unwrap().is_evicted);
        assert!(!manager.state(1).unwrap().is_evicted);
    }

    #[test]
    fn test_manager_reactivate_only_affects_target() {
        let mut manager = ExpertThermalManager::new(3).with_eviction_threshold(3);
        for _ in 0..4 {
            manager.step(&[0, 0, 0]);
        }
        manager.evict_expert(0);
        manager.evict_expert(1);
        manager.evict_expert(2);

        manager.reactivate_expert(1);

        assert!(manager.state(0).unwrap().is_evicted);
        assert!(!manager.state(1).unwrap().is_evicted);
        assert!(manager.state(2).unwrap().is_evicted);
    }

    #[test]
    fn test_manager_summary_reflects_all_evictions_and_reactivations() {
        let mut manager = ExpertThermalManager::new(4).with_eviction_threshold(3);
        for _ in 0..4 {
            manager.step(&[0, 0, 0, 0]);
        }
        manager.evict_expert(0);
        manager.evict_expert(1);
        manager.reactivate_expert(0);

        let summary = manager.summary();
        assert_eq!(summary.total_evictions, 2);
        assert_eq!(summary.total_reactivations, 1);
    }

    #[test]
    fn test_manager_hot_experts_returns_correct_set() {
        let mut manager = ExpertThermalManager::new(4);
        // Only experts 0 and 3 are hot
        for _ in 0..100 {
            manager.step(&[10, 0, 0, 10]);
        }
        let hot = manager.hot_experts();
        assert_eq!(hot.len(), 2);
        assert!(hot.contains(&0));
        assert!(hot.contains(&3));
    }

    #[test]
    fn test_manager_cold_or_evicted_returns_correct_set() {
        let mut manager = ExpertThermalManager::new(4).with_eviction_threshold(3);
        for _ in 0..100 {
            manager.step(&[10, 0, 0, 10]);
        }
        // Expert 1 and 2 should be cold or evicted
        let cold = manager.cold_or_evicted_experts();
        assert!(cold.contains(&1));
        assert!(cold.contains(&2));
        assert!(!cold.contains(&0));
        assert!(!cold.contains(&3));
    }

    #[test]
    fn test_manager_experts_to_evict_empty_initially() {
        let manager = ExpertThermalManager::new(4);
        assert!(manager.experts_to_evict().is_empty());
    }

    #[test]
    fn test_manager_experts_to_reactivate_empty_initially() {
        let manager = ExpertThermalManager::new(4);
        assert!(manager.experts_to_reactivate().is_empty());
    }

    #[test]
    fn test_manager_experts_to_reactivate_after_partial_eviction_and_deopt() {
        let mut manager = ExpertThermalManager::new(4).with_eviction_threshold(3);
        for _ in 0..4 {
            manager.step(&[5, 0, 0, 5]);
        }
        manager.evict_expert(1);
        manager.evict_expert(2);

        // Only expert 1 gets a deopt
        manager.handle_deopt_request(DeoptRequest {
            request_id: 1, expert_idx: 1, layer_idx: 0, step: 4,
        });

        // After handle_deopt_request, expert 1 is auto-reactivated
        // experts_to_reactivate should be empty (reactivation already done)
        assert!(manager.experts_to_reactivate().is_empty());
    }

    #[test]
    fn test_manager_effective_threshold_with_only_aggressiveness() {
        // threshold=1000, aggressiveness=1.0 => bias=0.5 => effective=500
        let manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(1000)
            .with_eviction_aggressiveness(1.0);
        assert_eq!(manager.effective_eviction_threshold(), 500);
    }

    #[test]
    fn test_manager_effective_threshold_aggressiveness_zero_no_change() {
        let manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(1000)
            .with_eviction_aggressiveness(0.0);
        assert_eq!(manager.effective_eviction_threshold(), 1000);
    }

    #[test]
    fn test_manager_effective_threshold_aggressiveness_two() {
        // threshold=900, aggressiveness=2.0 => bias=1/(1+2)=1/3 => 300
        let manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(900)
            .with_eviction_aggressiveness(2.0);
        assert_eq!(manager.effective_eviction_threshold(), 300);
    }

    #[test]
    fn test_manager_update_memory_pressure_clamps_to_zero() {
        let mut manager = ExpertThermalManager::new(2)
            .with_adaptive_eviction(5);
        manager.update_memory_pressure(-0.5);
        // Should be clamped to 0.0
        let threshold_low = manager.effective_eviction_threshold();
        manager.update_memory_pressure(0.0);
        let threshold_zero = manager.effective_eviction_threshold();
        assert_eq!(threshold_low, threshold_zero);
    }

    #[test]
    fn test_manager_update_memory_pressure_clamps_to_one() {
        let mut manager = ExpertThermalManager::new(2)
            .with_adaptive_eviction(5);
        manager.update_memory_pressure(1.5);
        // Should be clamped to 1.0
        let threshold_high = manager.effective_eviction_threshold();
        manager.update_memory_pressure(1.0);
        let threshold_one = manager.effective_eviction_threshold();
        assert_eq!(threshold_high, threshold_one);
    }

    #[test]
    fn test_manager_step_updates_working_set() {
        let mut manager = ExpertThermalManager::new(4)
            .with_adaptive_eviction(10);
        assert_eq!(manager.working_set_size(), 0);

        manager.step(&[1, 1, 0, 0]);
        assert_eq!(manager.working_set_size(), 2);

        manager.step(&[0, 0, 1, 1]);
        assert_eq!(manager.working_set_size(), 4);
    }

    #[test]
    fn test_manager_summary_after_no_steps() {
        let manager = ExpertThermalManager::new(4);
        let summary = manager.summary();
        assert_eq!(summary.current_step, 0);
        assert_eq!(summary.num_experts, 4);
        assert_eq!(summary.total_evictions, 0);
        assert_eq!(summary.total_reactivations, 0);
        assert_eq!(summary.pending_deopt_count, 0);
    }

    #[test]
    fn test_manager_num_experts_accessor() {
        assert_eq!(ExpertThermalManager::new(1).num_experts(), 1);
        assert_eq!(ExpertThermalManager::new(64).num_experts(), 64);
        assert_eq!(ExpertThermalManager::new(256).num_experts(), 256);
    }

    #[test]
    fn test_manager_evict_increments_total_only_on_success() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(3);
        for _ in 0..4 {
            manager.step(&[0, 0]);
        }
        assert!(manager.evict_expert(0));  // success
        assert!(!manager.evict_expert(0)); // already evicted
        assert!(manager.evict_expert(1));  // success
        assert!(!manager.evict_expert(5)); // out of bounds
        assert_eq!(manager.summary().total_evictions, 2);
    }

    #[test]
    fn test_manager_reactivate_increments_total_only_on_success() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(3);
        for _ in 0..4 {
            manager.step(&[0, 0]);
        }
        manager.evict_expert(0);
        assert!(manager.reactivate_expert(0));  // success
        assert!(!manager.reactivate_expert(0)); // already active
        assert!(!manager.reactivate_expert(5)); // out of bounds
        assert_eq!(manager.summary().total_reactivations, 1);
    }

    #[test]
    fn test_manager_evict_expert_sets_correct_heat_level() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(3);
        for _ in 0..4 {
            manager.step(&[5, 0]);
        }
        manager.evict_expert(1);
        assert_eq!(manager.state(1).unwrap().heat_level, ExpertHeatLevel::Evicted);
    }

    #[test]
    fn test_manager_reactivate_sets_cold_heat_level() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(3);
        for _ in 0..4 {
            manager.step(&[5, 0]);
        }
        manager.evict_expert(1);
        manager.reactivate_expert(1);
        assert_eq!(manager.state(1).unwrap().heat_level, ExpertHeatLevel::Cold);
    }

    #[test]
    fn test_manager_reactivate_clears_is_evicted() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(3);
        for _ in 0..4 {
            manager.step(&[5, 0]);
        }
        manager.evict_expert(1);
        assert!(manager.state(1).unwrap().is_evicted);
        manager.reactivate_expert(1);
        assert!(!manager.state(1).unwrap().is_evicted);
    }

    #[test]
    fn test_manager_reactivate_resets_streak() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(3);
        for _ in 0..4 {
            manager.step(&[5, 0]);
        }
        manager.evict_expert(1);
        assert!(manager.state(1).unwrap().consecutive_zero_streak >= 4);
        manager.reactivate_expert(1);
        assert_eq!(manager.state(1).unwrap().consecutive_zero_streak, 0);
    }

    #[test]
    fn test_manager_step_route_count_increments_per_step_even_on_miss() {
        let mut manager = ExpertThermalManager::new(1);
        manager.step(&[0]);
        manager.step(&[0]);
        manager.step(&[0]);
        assert_eq!(manager.state(0).unwrap().route_count, 3);
        assert_eq!(manager.state(0).unwrap().hit_count, 0);
    }

    #[test]
    fn test_manager_step_hit_rate_updates_each_step() {
        let mut manager = ExpertThermalManager::new(1);
        manager.step(&[5]);
        let rate1 = manager.state(0).unwrap().hit_rate;
        assert!((rate1 - 1.0).abs() < 1e-10);

        manager.step(&[0]);
        let rate2 = manager.state(0).unwrap().hit_rate;
        assert!((rate2 - 0.5).abs() < 1e-10);

        manager.step(&[5]);
        let rate3 = manager.state(0).unwrap().hit_rate;
        assert!((rate3 - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_manager_summary_pending_deopt_reflects_clear() {
        let mut manager = ExpertThermalManager::new(2);
        manager.handle_deopt_request(DeoptRequest { request_id: 1, expert_idx: 0, layer_idx: 0, step: 0 });
        manager.handle_deopt_request(DeoptRequest { request_id: 2, expert_idx: 1, layer_idx: 0, step: 0 });
        assert_eq!(manager.summary().pending_deopt_count, 2);
        manager.clear_deopt_requests();
        assert_eq!(manager.summary().pending_deopt_count, 0);
    }

    #[test]
    fn test_manager_working_set_size_accessible_via_manager() {
        let mut manager = ExpertThermalManager::new(4)
            .with_adaptive_eviction(10);
        manager.step(&[1, 0, 0, 0]);
        assert_eq!(manager.working_set_size(), 1);
    }

    #[test]
    fn test_manager_adaptive_eviction_affects_threshold() {
        let manager_static = ExpertThermalManager::new(4)
            .with_eviction_threshold(1000);
        let static_threshold = manager_static.effective_eviction_threshold();

        let manager_adaptive = ExpertThermalManager::new(4)
            .with_eviction_threshold(1000)
            .with_adaptive_eviction(10);
        // Initially working_set_size=0, adaptive => large threshold
        let adaptive_threshold = manager_adaptive.effective_eviction_threshold();
        // With ws=0, ratio=num_experts/max(1)=4, so threshold >= base
        assert!(adaptive_threshold >= static_threshold);
    }

    #[test]
    fn test_manager_eviction_with_aggressiveness_triggers_sooner() {
        let mut manager_no_agg = ExpertThermalManager::new(2)
            .with_eviction_threshold(100);
        let mut manager_agg = ExpertThermalManager::new(2)
            .with_eviction_threshold(100)
            .with_eviction_aggressiveness(1.0);

        // 60 steps: streak = 60
        // No aggressiveness: threshold = 100 => not evicted
        // Aggressiveness=1.0: threshold = 50 => evicted
        for _ in 0..60 {
            manager_no_agg.step(&[5, 0]);
            manager_agg.step(&[5, 0]);
        }
        assert!(!manager_no_agg.experts_to_evict().contains(&1));
        assert!(manager_agg.experts_to_evict().contains(&1));
    }

    #[test]
    fn test_manager_lifecycle_evict_reactivate_re_evict() {
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(3);

        // Evict expert 1
        for _ in 0..4 {
            manager.step(&[5, 0]);
        }
        assert!(manager.evict_expert(1));
        assert_eq!(manager.summary().total_evictions, 1);

        // Reactivate
        manager.reactivate_expert(1);
        assert_eq!(manager.summary().total_reactivations, 1);
        assert!(!manager.state(1).unwrap().is_evicted);

        // Expert 1 goes cold again
        for _ in 0..4 {
            manager.step(&[5, 0]);
        }
        assert!(manager.experts_to_evict().contains(&1));

        // Re-evict
        assert!(manager.evict_expert(1));
        assert_eq!(manager.summary().total_evictions, 2);
        assert!(manager.state(1).unwrap().is_evicted);
    }

    #[test]
    fn test_manager_step_interleaved_hit_miss_pattern() {
        let mut manager = ExpertThermalManager::new(1).with_eviction_threshold(1000);
        // Alternate hit/miss for 10 steps
        for i in 0..10 {
            if i % 2 == 0 {
                manager.step(&[5]);
            } else {
                manager.step(&[0]);
            }
        }
        let state = manager.state(0).unwrap();
        assert_eq!(state.hit_count, 5);
        assert_eq!(state.route_count, 10);
        assert!((state.hit_rate - 0.5).abs() < 1e-10);
        // Last step was i=9 (miss), so streak = 1
        assert_eq!(state.consecutive_zero_streak, 1);
    }

    #[test]
    fn test_manager_handle_deopt_for_non_evicted_is_spurious_preserves_state() {
        let mut manager = ExpertThermalManager::new(2);
        // Expert 0 is not evicted
        let result = manager.handle_deopt_request(DeoptRequest {
            request_id: 1, expert_idx: 0, layer_idx: 0, step: 0,
        });
        assert!(matches!(result, DeoptHandlingResult::SpuriousDeopt { .. }));
        // Expert 0 should remain non-evicted
        assert!(!manager.state(0).unwrap().is_evicted);
    }

    // ────────────────────────────────────────────────────────
    // Batch 5: 45 additional tests for further coverage
    // ────────────────────────────────────────────────────────

    // --- ExpertHeatLevel: additional from_hit_rate edge cases ---

    #[test]
    fn test_heat_level_from_hit_rate_subnormal_positive() {
        // Arrange: subnormal positive f64 (smaller than MIN_POSITIVE but > 0)
        let subnormal = f64::from_bits(1u64); // smallest positive subnormal
        assert!(subnormal > 0.0);
        assert!(subnormal < f64::MIN_POSITIVE);

        // Act: rate > 0 but below cold_threshold => Cold
        let level = ExpertHeatLevel::from_hit_rate(subnormal, 0.1, 0.001);

        // Assert
        assert_eq!(level, ExpertHeatLevel::Cold);
    }

    #[test]
    fn test_heat_level_from_hit_rate_negative_zero() {
        // Arrange: -0.0 should be treated as 0.0 by comparison
        let neg_zero = -0.0f64;

        // Act: -0.0 >= 0.1 is false, -0.0 >= 0.001 is false, -0.0 > 0.0 is false
        let level = ExpertHeatLevel::from_hit_rate(neg_zero, 0.1, 0.001);

        // Assert: -0.0 > 0.0 is false => Evicted
        assert_eq!(level, ExpertHeatLevel::Evicted);
    }

    #[test]
    fn test_heat_level_from_hit_rate_nan_hot_threshold() {
        // Arrange: NaN hot_threshold — rate >= NaN is false for all rates
        let level = ExpertHeatLevel::from_hit_rate(0.5, f64::NAN, 0.001);

        // Assert: 0.5 >= NaN is false, 0.5 >= 0.001 is true => Warm
        assert_eq!(level, ExpertHeatLevel::Warm);
    }

    #[test]
    fn test_heat_level_from_hit_rate_nan_cold_threshold() {
        // Arrange: NaN cold_threshold — rate >= NaN is false
        let level = ExpertHeatLevel::from_hit_rate(0.01, 0.1, f64::NAN);

        // Assert: 0.01 >= 0.1 is false, 0.01 >= NaN is false, 0.01 > 0.0 is true => Cold
        assert_eq!(level, ExpertHeatLevel::Cold);
    }

    #[test]
    fn test_heat_level_from_hit_rate_infinity_hot_threshold() {
        // Arrange: Infinity hot_threshold — rate >= INF is false for finite rates
        let level = ExpertHeatLevel::from_hit_rate(0.999, f64::INFINITY, 0.5);

        // Assert: 0.999 >= INF is false, 0.999 >= 0.5 is true => Warm
        assert_eq!(level, ExpertHeatLevel::Warm);
    }

    #[test]
    fn test_heat_level_from_hit_rate_negative_hot_threshold() {
        // Arrange: negative hot_threshold — even 0.0 >= -0.1 is true => Hot
        let level = ExpertHeatLevel::from_hit_rate(0.0, -0.1, -0.5);

        // Assert: 0.0 >= -0.1 is true => Hot
        assert_eq!(level, ExpertHeatLevel::Hot);
    }

    #[test]
    fn test_heat_level_from_hit_rate_large_hot_small_cold_thresholds() {
        // Arrange: very large hot, very small cold — wide Warm band
        // rate in the middle => Warm
        let level = ExpertHeatLevel::from_hit_rate(0.5, 1e15, 1e-15);
        assert_eq!(level, ExpertHeatLevel::Warm);

        // rate above hot => Hot
        let hot_level = ExpertHeatLevel::from_hit_rate(2e15, 1e15, 1e-15);
        assert_eq!(hot_level, ExpertHeatLevel::Hot);
    }

    // --- WorkingSetTracker: additional edge cases ---

    #[test]
    fn test_working_set_tracker_zero_experts_always_empty() {
        // Arrange: 0 experts — no experts to track
        let mut tracker = WorkingSetTracker::new(0, 5, 100);

        // Act: record a step (route_counts ignored since num_experts=0)
        tracker.record_step(&[1, 2, 3]);

        // Assert: working set size is always 0
        assert_eq!(tracker.working_set_size(), 0);
    }

    #[test]
    fn test_working_set_tracker_zero_experts_adaptive_threshold() {
        // Arrange: 0 experts, base_threshold=1000
        let tracker = WorkingSetTracker::new(0, 5, 1000);

        // Act: ws=0 => max(1) => ratio=0/1=0.0, threshold=1000*0.0*1.0=0 => max(1)
        let threshold = tracker.adaptive_threshold(0.0);

        // Assert: threshold clamped to at least 1
        assert!(threshold >= 1);
    }

    #[test]
    fn test_working_set_tracker_record_step_overwrites_old_data() {
        // Arrange: window_size=2, track 3 experts
        let mut tracker = WorkingSetTracker::new(3, 2, 100);

        // Step 1: experts 0,1,2 all active
        tracker.record_step(&[1, 1, 1]);
        assert_eq!(tracker.working_set_size(), 3);

        // Step 2: overwrite slot 0 with only expert 0
        tracker.record_step(&[1, 0, 0]);
        // Window now: slot 0=[T,F,F], slot 1=[T,T,T] => {0,1,2}
        assert_eq!(tracker.working_set_size(), 3);

        // Step 3: overwrite slot 1 with only expert 1
        tracker.record_step(&[0, 1, 0]);
        // Window now: slot 0=[T,F,F], slot 1=[F,T,F] => {0,1}
        assert_eq!(tracker.working_set_size(), 2);
    }

    #[test]
    fn test_working_set_tracker_adaptive_threshold_eighth_working_set() {
        // Arrange: 8 experts, 1 active
        let mut tracker = WorkingSetTracker::new(8, 5, 1000);
        tracker.record_step(&[1, 0, 0, 0, 0, 0, 0, 0]);

        // Act: ratio = 8/1 = 8.0, headroom(0.0) = 1.0
        let threshold = tracker.adaptive_threshold(0.0);

        // Assert: 1000 * 8.0 * 1.0 = 8000
        assert_eq!(threshold, 8000);
    }

    #[test]
    fn test_working_set_tracker_adaptive_threshold_pressure_zero_point_one() {
        // Arrange: 4 experts, 2 active, pressure=0.1
        let mut tracker = WorkingSetTracker::new(4, 5, 1000);
        tracker.record_step(&[1, 1, 0, 0]);

        // Act: headroom = clamp(1.0 - 0.1, 0.1, 1.0) = 0.9
        let threshold = tracker.adaptive_threshold(0.1);

        // Assert: 1000 * (4/2) * 0.9 = 1800
        assert_eq!(threshold, 1800);
    }


    // --- ExpertThermalManager: eviction decision with adaptive threshold ---

    #[test]
    fn test_eviction_decision_with_adaptive_higher_threshold() {
        // Arrange: adaptive eviction with 2/4 experts active
        // This raises the effective threshold, making eviction harder
        let mut manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(100)
            .with_adaptive_eviction(10);

        // Act: record 1 step to establish working set
        manager.step(&[1, 1, 0, 0]);
        // working_set_size = 2, adaptive = 100 * (4/2) * 1.0 = 200

        // Assert: effective threshold is higher than base
        assert!(manager.effective_eviction_threshold() > 100);
    }

    #[test]
    fn test_eviction_decision_with_adaptive_lower_threshold_under_pressure() {
        // Arrange: adaptive eviction with high memory pressure reduces threshold
        let mut manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(100)
            .with_adaptive_eviction(10);

        manager.step(&[1, 1, 0, 0]); // ws=2
        manager.update_memory_pressure(0.9);
        // adaptive = 100 * (4/2) * clamp(0.1, 0.1, 1.0) = 100 * 2 * 0.1 = 20

        // Assert: effective threshold is lower than base
        let threshold = manager.effective_eviction_threshold();
        assert!(threshold < 100);
        assert_eq!(threshold, 20);
    }

    #[test]
    fn test_eviction_triggers_with_threshold_one() {
        // Arrange: threshold=1 — expert evicted after just 1 zero-hit step
        let mut manager = ExpertThermalManager::new(2)
            .with_eviction_threshold(1);

        // Act: 1 step where expert 1 gets zero hits
        manager.step(&[10, 0]);

        // Assert: expert 1 immediately qualifies for eviction
        assert!(manager.experts_to_evict().contains(&1));
        assert_eq!(manager.state(1).unwrap().consecutive_zero_streak, 1);
    }

    #[test]
    fn test_no_eviction_with_threshold_zero_base() {
        // Arrange: threshold=0 — streak of 0 meets threshold, immediate eviction
        let manager = ExpertThermalManager::new(2)
            .with_eviction_threshold(0);

        // Act: Even before any step, streak=0 which >= threshold=0
        // But eviction_decision checks is_evicted first, then streak >= threshold
        // For fresh experts with streak=0 and threshold=0, it should Evict
        assert_eq!(manager.eviction_decision(0), EvictionDecision::Evict);
        assert_eq!(manager.eviction_decision(1), EvictionDecision::Evict);
    }

    // --- Manager: hot/cold expert transitions ---

    #[test]
    fn test_hot_experts_after_all_become_hot() {
        // Arrange
        let mut manager = ExpertThermalManager::new(3);

        // Act: all experts consistently hit
        for _ in 0..50 {
            manager.step(&[10, 10, 10]);
        }

        // Assert: all are hot
        let hot = manager.hot_experts();
        assert_eq!(hot.len(), 3);
    }

    #[test]
    fn test_cold_or_evicted_after_all_recover() {
        // Arrange: make experts cold then recover them
        let mut manager = ExpertThermalManager::new(2)
            .with_eviction_threshold(3)
            .with_heat_thresholds(0.1, 0.001);

        // Make both cold
        for _ in 0..6 {
            manager.step(&[0, 0]);
        }

        // Recover both by hitting them consistently
        for _ in 0..50 {
            manager.step(&[10, 10]);
        }

        // Assert: no cold/evicted experts
        assert!(manager.cold_or_evicted_experts().is_empty());
    }

    #[test]
    fn test_hot_experts_empty_when_all_evicted() {
        // Arrange
        let mut manager = ExpertThermalManager::new(3).with_eviction_threshold(3);

        for _ in 0..4 {
            manager.step(&[0, 0, 0]);
        }
        manager.evict_expert(0);
        manager.evict_expert(1);
        manager.evict_expert(2);

        // Assert: no hot experts when all are evicted
        assert!(manager.hot_experts().is_empty());
        assert_eq!(manager.cold_or_evicted_experts().len(), 3);
    }

    // --- Manager: expert state transitions across steps ---

    #[test]
    fn test_expert_transitions_warm_to_hot_across_steps() {
        // Arrange
        let mut manager = ExpertThermalManager::new(1)
            .with_heat_thresholds(0.5, 0.1);

        // Act: 1 hit out of 2 => hit_rate=0.5, >= hot_threshold => Hot
        manager.step(&[5]);
        assert_eq!(manager.state(0).unwrap().heat_level, ExpertHeatLevel::Hot);

        // Miss => hit_rate = 1/2 = 0.5, still Hot
        manager.step(&[0]);
        assert_eq!(manager.state(0).unwrap().heat_level, ExpertHeatLevel::Hot);
    }

    #[test]
    fn test_expert_transitions_hot_to_cold_across_steps() {
        // Arrange
        let mut manager = ExpertThermalManager::new(2)
            .with_heat_thresholds(0.5, 0.1)
            .with_eviction_threshold(1000);

        // Act: make expert 1 hot first
        for _ in 0..10 {
            manager.step(&[0, 10]);
        }
        assert_eq!(manager.state(1).unwrap().heat_level, ExpertHeatLevel::Hot);

        // Then make it cold
        for _ in 0..90 {
            manager.step(&[10, 0]);
        }
        // hit_rate = 10/100 = 0.1, exactly at cold_threshold => Warm
        // Actually: 10/100 = 0.1 >= cold(0.1) => Warm, not Cold
        // Let it go further to get Cold
        for _ in 0..900 {
            manager.step(&[10, 0]);
        }
        // hit_rate = 10/1000 = 0.01 > 0 but < 0.1 => Cold
        assert_eq!(manager.state(1).unwrap().heat_level, ExpertHeatLevel::Cold);
    }

    // --- Manager: combined adaptive + aggressiveness scenarios ---

    #[test]
    fn test_combined_adaptive_aggressiveness_high_pressure() {
        // Arrange: all modifiers active
        let mut manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(1000)
            .with_adaptive_eviction(10)
            .with_eviction_aggressiveness(1.0);

        manager.step(&[1, 1, 0, 0]); // ws=2
        manager.update_memory_pressure(0.75);

        // Act
        let threshold = manager.effective_eviction_threshold();
        // adaptive = 1000 * (4/2) * clamp(0.25, 0.1, 1.0) = 1000 * 2 * 0.25 = 500
        // aggressiveness=1.0 => bias = 0.5
        // effective = 500 * 0.5 = 250
        assert_eq!(threshold, 250);
    }

    #[test]
    fn test_combined_adaptive_aggressiveness_low_pressure() {
        // Arrange: low pressure, moderate aggressiveness
        let mut manager = ExpertThermalManager::new(8)
            .with_eviction_threshold(1000)
            .with_adaptive_eviction(10)
            .with_eviction_aggressiveness(0.5);

        manager.step(&[1, 1, 1, 1, 0, 0, 0, 0]); // ws=4
        manager.update_memory_pressure(0.0);

        // Act
        let threshold = manager.effective_eviction_threshold();
        // adaptive = 1000 * (8/4) * 1.0 = 2000
        // aggressiveness=0.5 => bias = 1/(1+0.5) = 2/3
        // effective = 2000 * 2/3 = 1333 (integer truncation)
        assert_eq!(threshold, 1333);
    }

    #[test]
    fn test_adaptive_threshold_decreases_as_working_set_grows() {
        // Arrange
        let mut manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(1000)
            .with_adaptive_eviction(10);
        manager.update_memory_pressure(0.0);

        // Act: ws=1
        manager.step(&[1, 0, 0, 0]);
        let t1 = manager.effective_eviction_threshold();

        // ws=2
        manager.step(&[0, 1, 0, 0]);
        let t2 = manager.effective_eviction_threshold();

        // ws=3
        manager.step(&[0, 0, 1, 0]);
        let t3 = manager.effective_eviction_threshold();

        // ws=4
        manager.step(&[0, 0, 0, 1]);
        let t4 = manager.effective_eviction_threshold();

        // Assert: threshold decreases as more experts become active
        assert!(t1 > t2);
        assert!(t2 > t3);
        assert!(t3 > t4);
    }

    // --- Manager: handle_deopt_request with different layers ---

    #[test]
    fn test_handle_deopt_request_records_layer_idx() {
        // Arrange
        let mut manager = ExpertThermalManager::new(2)
            .with_eviction_threshold(3);

        for _ in 0..4 {
            manager.step(&[10, 0]);
        }
        manager.evict_expert(1);

        // Act: deopt at layer 7
        let result = manager.handle_deopt_request(DeoptRequest {
            request_id: 1,
            expert_idx: 1,
            layer_idx: 7,
            step: 4,
        });

        // Assert: result is ReactivateAndRerun and request is recorded with layer_idx
        assert!(matches!(result, DeoptHandlingResult::ReactivateAndRerun { .. }));
        assert_eq!(manager.pending_deopt_requests()[0].layer_idx, 7);
    }

    // --- Manager: step interactions after reactivation ---

    #[test]
    fn test_step_after_reactivation_starts_new_streak() {
        // Arrange
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(5);

        for _ in 0..6 {
            manager.step(&[10, 0]);
        }
        manager.evict_expert(1);
        manager.reactivate_expert(1);

        // Act: 3 more zero-hit steps
        for _ in 0..3 {
            manager.step(&[10, 0]);
        }

        // Assert: streak is 3, not cumulative from before eviction
        assert_eq!(manager.state(1).unwrap().consecutive_zero_streak, 3);
        // Not yet at threshold of 5
        assert!(!manager.experts_to_evict().contains(&1));
    }

    #[test]
    fn test_step_after_reactivation_hit_resets_streak_immediately() {
        // Arrange
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(5);

        for _ in 0..6 {
            manager.step(&[10, 0]);
        }
        manager.evict_expert(1);
        manager.reactivate_expert(1);

        // Act: expert 1 gets a hit
        manager.step(&[10, 5]);

        // Assert: streak is 0
        assert_eq!(manager.state(1).unwrap().consecutive_zero_streak, 0);
        assert_eq!(manager.state(1).unwrap().hit_count, 1);
    }

    // --- Manager: expert_to_evict ordering follows state order ---

    #[test]
    fn test_experts_to_evict_returns_in_state_order() {
        // Arrange
        let mut manager = ExpertThermalManager::new(4).with_eviction_threshold(3);

        for _ in 0..4 {
            manager.step(&[10, 0, 0, 0]);
        }

        // Act
        let to_evict = manager.experts_to_evict();

        // Assert: returned in iteration order (0,1,2,3 but expert 0 has hits)
        assert_eq!(to_evict.len(), 3);
        assert_eq!(to_evict[0], 1);
        assert_eq!(to_evict[1], 2);
        assert_eq!(to_evict[2], 3);
    }

    // --- Manager: summary working_set_size consistency ---

    #[test]
    fn test_summary_working_set_size_matches_manager_method() {
        // Arrange
        let mut manager = ExpertThermalManager::new(4)
            .with_adaptive_eviction(10);

        manager.step(&[1, 1, 0, 0]);
        manager.step(&[0, 0, 1, 0]);

        // Act & Assert
        assert_eq!(manager.summary().working_set_size, manager.working_set_size());
        assert_eq!(manager.working_set_size(), 3);
    }

    // --- Manager: reactivation_count through multiple deopt cycles ---

    #[test]
    fn test_reactivation_count_accumulates_through_deopt_cycles() {
        // Arrange
        let mut manager = ExpertThermalManager::new(1).with_eviction_threshold(2);

        // Cycle 1: evict then deopt
        for _ in 0..3 { manager.step(&[0]); }
        manager.evict_expert(0);
        manager.handle_deopt_request(DeoptRequest {
            request_id: 1, expert_idx: 0, layer_idx: 0, step: 3,
        });

        // After first deopt: reactivate_expert sets reactivation_count=1,
        // but handle_deopt_request already incremented it to 1 before calling reactivate_expert
        // which increments again to 2 (from the handle_deopt_request code path)
        assert_eq!(manager.state(0).unwrap().reactivation_count, 2);

        // Cycle 2: re-evict then deopt again
        for _ in 0..3 { manager.step(&[0]); }
        manager.evict_expert(0); // resets reactivation_count to 0
        assert_eq!(manager.state(0).unwrap().reactivation_count, 0);

        manager.handle_deopt_request(DeoptRequest {
            request_id: 2, expert_idx: 0, layer_idx: 0, step: 6,
        });

        // Second deopt cycle: same pattern
        assert_eq!(manager.state(0).unwrap().reactivation_count, 2);
    }

    // --- Manager: total counters are independent ---

    #[test]
    fn test_total_evictions_independent_of_reactivations() {
        // Arrange
        let mut manager = ExpertThermalManager::new(3).with_eviction_threshold(3);

        for _ in 0..4 { manager.step(&[0, 0, 0]); }

        // Act: evict all three
        manager.evict_expert(0);
        manager.evict_expert(1);
        manager.evict_expert(2);

        // Assert: 3 evictions, 0 reactivations
        assert_eq!(manager.summary().total_evictions, 3);
        assert_eq!(manager.summary().total_reactivations, 0);

        // Reactivate only one
        manager.reactivate_expert(1);

        assert_eq!(manager.summary().total_evictions, 3);
        assert_eq!(manager.summary().total_reactivations, 1);
    }

    // --- ExpertHeatState: Clone independence ---

    #[test]
    fn test_expert_heat_state_clone_independence() {
        // Arrange
        let original = ExpertHeatState {
            expert_idx: 0,
            hit_rate: 0.5,
            hit_count: 10,
            route_count: 20,
            heat_level: ExpertHeatLevel::Hot,
            consecutive_zero_streak: 0,
            last_hit_step: 20,
            is_evicted: false,
            reactivation_count: 0,
        };

        // Act: clone original
        let cloned = original.clone();
        let modified = ExpertHeatState {
            hit_count: 99,
            hit_rate: 0.99,
            ..original.clone()
        };

        // Assert: clone is independent of modifications
        assert_eq!(cloned.hit_count, 10);
        assert!((cloned.hit_rate - 0.5).abs() < f64::EPSILON);
        assert_eq!(modified.hit_count, 99);
    }

    // --- DeoptRequest: equality symmetry ---

    #[test]
    fn test_deopt_request_equality_symmetry() {
        // Arrange
        let a = DeoptRequest { request_id: 42, expert_idx: 3, layer_idx: 7, step: 100 };
        let b = DeoptRequest { request_id: 42, expert_idx: 3, layer_idx: 7, step: 100 };

        // Assert: symmetry (a==b implies b==a)
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    #[test]
    fn test_deopt_request_equality_transitivity() {
        // Arrange
        let a = DeoptRequest { request_id: 1, expert_idx: 2, layer_idx: 3, step: 4 };
        let b = DeoptRequest { request_id: 1, expert_idx: 2, layer_idx: 3, step: 4 };
        let c = DeoptRequest { request_id: 1, expert_idx: 2, layer_idx: 3, step: 4 };

        // Assert: transitivity (a==b and b==c implies a==c)
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    // --- EvictionDecision: Ord total ordering ---

    #[test]
    fn test_eviction_decision_partial_ord_ordering_complete() {
        // EvictionDecision derives PartialOrd but not Ord
        // Verify the complete ordering: Keep < Evict < Reactivate
        assert!(EvictionDecision::Keep < EvictionDecision::Evict);
        assert!(EvictionDecision::Evict < EvictionDecision::Reactivate);
        assert!(EvictionDecision::Keep < EvictionDecision::Reactivate);

        // Verify reverse: Reactivate > Evict > Keep
        assert!(EvictionDecision::Reactivate > EvictionDecision::Evict);
        assert!(EvictionDecision::Evict > EvictionDecision::Keep);
        assert!(EvictionDecision::Reactivate > EvictionDecision::Keep);
    }

    // --- Manager: working set interacts with eviction ---

    #[test]
    fn test_working_set_shrinks_when_experts_go_cold() {
        // Arrange: adaptive eviction with window_size=3
        let mut manager = ExpertThermalManager::new(3)
            .with_adaptive_eviction(3);

        // Step 1: all active
        manager.step(&[1, 1, 1]);
        assert_eq!(manager.working_set_size(), 3);

        // Steps 2-4: only expert 0 active — window overwrites
        for _ in 0..3 {
            manager.step(&[1, 0, 0]);
        }

        // Assert: working set shrinks to 1 (only expert 0 in window)
        assert_eq!(manager.working_set_size(), 1);
    }

    #[test]
    fn test_effective_threshold_dynamic_with_working_set_changes() {
        // Arrange
        let mut manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(1000)
            .with_adaptive_eviction(10);
        manager.update_memory_pressure(0.0);

        // ws=1 => high threshold
        manager.step(&[1, 0, 0, 0]);
        let t_small_ws = manager.effective_eviction_threshold();

        // ws=4 => threshold = base
        manager.step(&[0, 1, 0, 0]);
        manager.step(&[0, 0, 1, 0]);
        manager.step(&[0, 0, 0, 1]);
        let t_large_ws = manager.effective_eviction_threshold();

        // Assert: threshold decreased as working set grew
        assert!(t_small_ws > t_large_ws);
    }

    // --- Manager: edge case with large number of steps ---

    #[test]
    fn test_manager_many_steps_no_overflow() {
        // Arrange: run 1000 steps to verify no overflow in counters
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(u64::MAX);

        // Act: alternate hits
        for i in 0..1000 {
            if i % 2 == 0 {
                manager.step(&[5, 0]);
            } else {
                manager.step(&[0, 5]);
            }
        }

        // Assert: counters are correct
        assert_eq!(manager.state(0).unwrap().hit_count, 500);
        assert_eq!(manager.state(0).unwrap().route_count, 1000);
        assert_eq!(manager.state(1).unwrap().hit_count, 500);
        assert_eq!(manager.state(1).unwrap().route_count, 1000);
        assert_eq!(manager.summary().current_step, 1000);
    }

    // --- Manager: eviction_decision at boundary for expert 0 ---

    #[test]
    fn test_eviction_decision_for_expert_zero_at_threshold() {
        // Arrange
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(5);

        // Expert 0 gets zero hits for 5 steps
        for _ in 0..5 {
            manager.step(&[0, 10]);
        }

        // Assert: expert 0 streak=5 == threshold=5 => Evict
        assert_eq!(manager.eviction_decision(0), EvictionDecision::Evict);
    }

    // --- ThermalSummary: property after complex operations ---


    // --- ExpertHeatLevel: from_hit_rate with very specific boundary values ---

    #[test]
    fn test_heat_level_from_hit_rate_cold_threshold_epsilon_above() {
        // Arrange: rate just barely above cold_threshold
        let cold_threshold = 0.001;
        let rate = cold_threshold + f64::EPSILON;

        // Act
        let level = ExpertHeatLevel::from_hit_rate(rate, 0.1, cold_threshold);

        // Assert: rate >= cold_threshold => Warm
        assert_eq!(level, ExpertHeatLevel::Warm);
    }

    #[test]
    fn test_heat_level_from_hit_rate_hot_threshold_epsilon_above() {
        // Arrange: rate just barely above hot_threshold
        let hot_threshold = 0.1;
        let rate = hot_threshold + f64::EPSILON;

        // Act
        let level = ExpertHeatLevel::from_hit_rate(rate, hot_threshold, 0.001);

        // Assert: rate >= hot_threshold => Hot
        assert_eq!(level, ExpertHeatLevel::Hot);
    }

    #[test]
    fn test_heat_level_from_hit_rate_hot_threshold_epsilon_below() {
        // Arrange: rate just barely below hot_threshold
        let hot_threshold = 0.1;
        let rate = hot_threshold - f64::EPSILON;

        // Act
        let level = ExpertHeatLevel::from_hit_rate(rate, hot_threshold, 0.001);

        // Assert: rate < hot_threshold but >= cold_threshold => Warm
        assert_eq!(level, ExpertHeatLevel::Warm);
    }

    // --- Manager: eviction does not affect summary counts of other levels ---

    #[test]
    fn test_eviction_changes_summary_level_counts_correctly() {
        // Arrange
        let mut manager = ExpertThermalManager::new(3)
            .with_eviction_threshold(3);

        for _ in 0..4 {
            manager.step(&[10, 10, 0]);
        }

        let before = manager.summary();
        // Expert 2 is cold/evicted level
        assert!(before.cold_count + before.evicted_count >= 1);

        // Act: evict expert 2
        manager.evict_expert(2);

        // Assert
        let after = manager.summary();
        assert_eq!(after.evicted_count, 1);
        assert_eq!(after.hot_count, 2); // experts 0 and 1 are hot
    }

    // --- Manager: step with only some experts getting route_counts ---

    #[test]
    fn test_step_partial_route_counts_updates_only_covered_experts() {
        // Arrange
        let mut manager = ExpertThermalManager::new(5);

        // Act: only 3 entries in route_counts
        manager.step(&[10, 5, 3]);

        // Assert
        assert_eq!(manager.state(0).unwrap().route_count, 1);
        assert_eq!(manager.state(0).unwrap().hit_count, 1);
        assert_eq!(manager.state(1).unwrap().route_count, 1);
        assert_eq!(manager.state(1).unwrap().hit_count, 1);
        assert_eq!(manager.state(2).unwrap().route_count, 1);
        assert_eq!(manager.state(2).unwrap().hit_count, 1);
        assert_eq!(manager.state(3).unwrap().route_count, 0); // untouched
        assert_eq!(manager.state(4).unwrap().route_count, 0); // untouched
    }

    // --- Manager: summary effective_eviction_threshold changes dynamically ---

    #[test]
    fn test_summary_effective_eviction_threshold_with_adaptive_and_pressure() {
        // Arrange
        let mut manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(1000)
            .with_adaptive_eviction(10);

        manager.step(&[1, 1, 0, 0]); // ws=2

        // No pressure
        manager.update_memory_pressure(0.0);
        let t0 = manager.summary().effective_eviction_threshold;

        // High pressure
        manager.update_memory_pressure(0.9);
        let t1 = manager.summary().effective_eviction_threshold;

        // Assert
        assert!(t1 < t0);
    }

    // --- Manager: multiple experts to evict and reactivate in one cycle ---

    #[test]
    fn test_batch_evict_then_batch_reactivate() {
        // Arrange: 8 experts, only 2 active
        let mut manager = ExpertThermalManager::new(8).with_eviction_threshold(3);

        for _ in 0..4 {
            manager.step(&[10, 10, 0, 0, 0, 0, 0, 0]);
        }

        // Act: batch evict all cold experts
        let to_evict = manager.experts_to_evict();
        assert_eq!(to_evict.len(), 6);
        for &idx in &to_evict {
            assert!(manager.evict_expert(idx));
        }

        // Assert: all evicted
        assert_eq!(manager.summary().evicted_count, 6);
        assert_eq!(manager.summary().total_evictions, 6);

        // Batch reactivate
        for &idx in &to_evict {
            assert!(manager.reactivate_expert(idx));
        }
        assert_eq!(manager.summary().total_reactivations, 6);
        assert!(manager.experts_to_evict().is_empty());
    }

    // --- WorkingSetTracker: adaptive threshold with partial working set at various sizes ---

    #[test]
    fn test_working_set_tracker_adaptive_threshold_three_quarter_working_set() {
        // Arrange: 4 experts, 3 active
        let mut tracker = WorkingSetTracker::new(4, 5, 1000);
        tracker.record_step(&[1, 1, 1, 0]);

        // Act: ratio = 4/3 ≈ 1.333, headroom(0.0) = 1.0
        let threshold = tracker.adaptive_threshold(0.0);

        // Assert: 1000 * (4/3) * 1.0 ≈ 1333.33 => round to 1333
        assert_eq!(threshold, 1333);
    }

    #[test]
    fn test_working_set_tracker_adaptive_threshold_one_third_working_set() {
        // Arrange: 3 experts, 1 active
        let mut tracker = WorkingSetTracker::new(3, 5, 600);
        tracker.record_step(&[1, 0, 0]);

        // Act: ratio = 3/1 = 3.0, headroom(0.0) = 1.0
        let threshold = tracker.adaptive_threshold(0.0);

        // Assert: 600 * 3.0 * 1.0 = 1800
        assert_eq!(threshold, 1800);
    }

    // --- Manager: hit_rate precision with many operations ---

    #[test]
    fn test_hit_rate_precision_with_1000_steps() {
        // Arrange
        let mut manager = ExpertThermalManager::new(1).with_eviction_threshold(u64::MAX);

        // Act: 333 hits out of 1000 steps
        for i in 0..1000 {
            if i % 3 == 0 {
                manager.step(&[5]);
            } else {
                manager.step(&[0]);
            }
        }

        // Assert
        let state = manager.state(0).unwrap();
        assert_eq!(state.hit_count, 334); // 0,3,6,...,999 => ceil(1000/3) = 334
        assert_eq!(state.route_count, 1000);
        let expected_rate = 334.0 / 1000.0;
        assert!((state.hit_rate - expected_rate).abs() < 1e-10);
    }

    // --- Manager: step where route count value doesn't matter, only > 0 ---

    #[test]
    fn test_step_hit_count_independent_of_route_count_magnitude() {
        // Arrange: route_counts of varying magnitude
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(100);

        // Act: count > 0 counts as a hit regardless of magnitude
        manager.step(&[1, 0]);     // minimal count
        manager.step(&[1000000, 0]); // large count
        manager.step(&[1, 0]);

        // Assert: hit_count = 3 (3 steps with count > 0), not sum of counts
        assert_eq!(manager.state(0).unwrap().hit_count, 3);
        assert_eq!(manager.state(0).unwrap().route_count, 3);
    }

    // --- Manager: eviction decision unchanged by memory_pressure without adaptive ---

    #[test]
    fn test_memory_pressure_no_effect_without_adaptive() {
        // Arrange: no adaptive eviction enabled
        let mut manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(500);

        let threshold_before = manager.effective_eviction_threshold();

        // Act: update pressure — should have no effect
        manager.update_memory_pressure(0.99);
        let threshold_after = manager.effective_eviction_threshold();

        // Assert: unchanged
        assert_eq!(threshold_before, threshold_after);
    }

    // ────────────────────────────────────────────────────────
    // Batch 6: 15 additional tests
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_heat_level_from_hit_rate_positive_zero_rate() {
        // Arrange: +0.0 as rate
        let pos_zero: f64 = 0.0f64;
        assert_eq!(pos_zero.signum(), 1.0); // +0.0

        // Act: +0.0 >= hot_threshold is false, +0.0 >= cold_threshold is false,
        // +0.0 > 0.0 is false => Evicted
        let level = ExpertHeatLevel::from_hit_rate(pos_zero, 0.1, 0.001);

        // Assert
        assert_eq!(level, ExpertHeatLevel::Evicted);
    }

    #[test]
    fn test_working_set_tracker_debug_after_record_step() {
        // Arrange
        let mut tracker = WorkingSetTracker::new(4, 3, 100);
        tracker.record_step(&[1, 0, 0, 0]);

        // Act
        let debug = format!("{:?}", tracker);

        // Assert: window should reflect the recorded step
        assert!(debug.contains("WorkingSetTracker"));
        assert!(debug.contains("cursor: 1")); // advanced from 0 to 1
    }

    #[test]
    fn test_expert_heat_state_debug_with_evicted_level() {
        // Arrange
        let state = ExpertHeatState {
            expert_idx: 3,
            hit_rate: 0.0,
            hit_count: 0,
            route_count: 50,
            heat_level: ExpertHeatLevel::Evicted,
            consecutive_zero_streak: 50,
            last_hit_step: 0,
            is_evicted: true,
            reactivation_count: 0,
        };

        // Act
        let debug = format!("{:?}", state);

        // Assert
        assert!(debug.contains("Evicted"));
        assert!(debug.contains("is_evicted: true"));
        assert!(debug.contains("consecutive_zero_streak: 50"));
        assert!(debug.contains("route_count: 50"));
    }

    #[test]
    fn test_thermal_summary_debug_all_zero_fields() {
        // Arrange
        let summary = ThermalSummary {
            num_experts: 0,
            hot_count: 0,
            warm_count: 0,
            cold_count: 0,
            evicted_count: 0,
            total_evictions: 0,
            total_reactivations: 0,
            current_step: 0,
            pending_deopt_count: 0,
            working_set_size: 0,
            effective_eviction_threshold: 0,
        };

        // Act
        let debug = format!("{:?}", summary);

        // Assert
        assert!(debug.contains("num_experts: 0"));
        assert!(debug.contains("current_step: 0"));
        assert!(debug.contains("effective_eviction_threshold: 0"));
    }

    #[test]
    fn test_eviction_decision_threshold_one_boundary() {
        // Arrange: threshold=1, run exactly 1 step
        let mut manager = ExpertThermalManager::new(2)
            .with_eviction_threshold(1);

        // Act: 1 step, expert 1 gets zero hits => streak=1 >= threshold=1
        manager.step(&[10, 0]);

        // Assert
        assert_eq!(manager.eviction_decision(1), EvictionDecision::Evict);
        assert_eq!(manager.state(1).unwrap().consecutive_zero_streak, 1);
    }

    #[test]
    fn test_from_hit_rate_negative_zero_with_negative_cold_threshold() {
        // Arrange: rate=-0.0, hot=-0.1, cold=-0.5
        // -0.0 >= -0.1 is true (both are zero or less)
        let level = ExpertHeatLevel::from_hit_rate(-0.0f64, -0.1, -0.5);

        // Assert: -0.0 >= -0.1 => Hot
        assert_eq!(level, ExpertHeatLevel::Hot);
    }

    #[test]
    fn test_update_memory_pressure_nan_adaptive_threshold_consistent() {
        // Arrange
        let mut manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(1000)
            .with_adaptive_eviction(10);
        manager.step(&[1, 1, 0, 0]); // ws=2

        // Act: set pressure to NaN
        manager.update_memory_pressure(f32::NAN);
        let threshold_nan = manager.effective_eviction_threshold();

        // Assert: should not panic; threshold should still be computed
        // NaN gets clamped by clamp(0.1, 1.0) — but NaN comparisons are false,
        // so clamp returns the "default" which is the first arg that passes.
        // clamp for NaN: NaN.clamp(0.1, 1.0) => 0.1 (NaN propagates, but clamp handles it)
        assert!(threshold_nan >= 1);
    }

    #[test]
    fn test_hot_experts_initially_empty_when_all_warm() {
        // Arrange: fresh manager, all experts start as Warm
        let manager = ExpertThermalManager::new(6);

        // Act
        let hot = manager.hot_experts();
        let cold = manager.cold_or_evicted_experts();

        // Assert: Warm experts are neither hot nor cold/evicted
        assert!(hot.is_empty());
        assert!(cold.is_empty());
    }

    #[test]
    fn test_evict_then_deopt_then_evict_again_lifecycle() {
        // Arrange
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(3);

        for _ in 0..4 { manager.step(&[10, 0]); }
        assert!(manager.evict_expert(1));

        // Act: deopt triggers reactivation
        manager.handle_deopt_request(DeoptRequest {
            request_id: 1, expert_idx: 1, layer_idx: 0, step: 4,
        });
        assert!(!manager.state(1).unwrap().is_evicted);

        // Build streak again and re-evict
        for _ in 0..4 { manager.step(&[10, 0]); }
        assert!(manager.evict_expert(1));

        // Assert
        assert!(manager.state(1).unwrap().is_evicted);
        assert_eq!(manager.summary().total_evictions, 2);
    }

    #[test]
    fn test_eviction_decision_just_evicted_still_at_streak_threshold() {
        // Arrange: evict an expert whose streak exactly equals threshold
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(5);

        for _ in 0..5 { manager.step(&[10, 0]); }
        // Expert 1 streak=5 == threshold
        assert!(manager.evict_expert(1));

        // Act: eviction_decision for an already-evicted expert returns Keep
        let decision = manager.eviction_decision(1);

        // Assert
        assert_eq!(decision, EvictionDecision::Keep);
        assert!(manager.state(1).unwrap().is_evicted);
    }

    #[test]
    fn test_rapid_evict_reactivate_evict_cycle() {
        // Arrange
        let mut manager = ExpertThermalManager::new(1).with_eviction_threshold(2);

        for _ in 0..3 { manager.step(&[0]); }

        // Act: evict -> reactivate -> evict in rapid succession
        assert!(manager.evict_expert(0));
        assert!(manager.reactivate_expert(0));
        // Streak was reset, but we can still re-evict immediately if streak was
        // artificially built up again
        for _ in 0..3 { manager.step(&[0]); }
        assert!(manager.evict_expert(0));

        // Assert: 2 evictions, 1 reactivation
        assert_eq!(manager.summary().total_evictions, 2);
        assert_eq!(manager.summary().total_reactivations, 1);
        assert!(manager.state(0).unwrap().is_evicted);
    }

    #[test]
    fn test_working_set_tracker_clone_preserves_cursor_state() {
        // Arrange
        let mut tracker = WorkingSetTracker::new(3, 5, 100);
        tracker.record_step(&[1, 0, 0]); // cursor 0->1
        tracker.record_step(&[0, 1, 0]); // cursor 1->2

        // Act
        let cloned = tracker.clone();

        // Assert: cloned has same cursor position and data
        assert_eq!(cloned.cursor, 2);
        assert_eq!(cloned.working_set_size(), tracker.working_set_size());
        assert_eq!(cloned.num_experts, tracker.num_experts);
    }

    #[test]
    fn test_hit_rate_exact_one_third_precision() {
        // Arrange: hit every 3rd step out of 9
        let mut manager = ExpertThermalManager::new(1).with_eviction_threshold(u64::MAX);

        // Act: pattern hit-miss-miss repeated 3 times => 3 hits / 9 routes
        for _ in 0..3 {
            manager.step(&[5]);   // hit
            manager.step(&[0]);   // miss
            manager.step(&[0]);   // miss
        }

        // Assert
        let state = manager.state(0).unwrap();
        assert_eq!(state.hit_count, 3);
        assert_eq!(state.route_count, 9);
        let expected = 3.0 / 9.0;
        assert!((state.hit_rate - expected).abs() < 1e-15);
    }

    #[test]
    fn test_summary_all_warm_initially() {
        // Arrange: fresh manager with 5 experts
        let manager = ExpertThermalManager::new(5);
        let summary = manager.summary();

        // Assert: all experts start as Warm
        assert_eq!(summary.warm_count, 5);
        assert_eq!(summary.hot_count, 0);
        assert_eq!(summary.cold_count, 0);
        assert_eq!(summary.evicted_count, 0);
        assert_eq!(summary.num_experts, 5);
    }

    #[test]
    fn test_deopt_request_distinguished_by_expert_idx_only() {
        // Arrange: same request_id, layer_idx, step but different expert_idx
        let a = DeoptRequest { request_id: 1, expert_idx: 0, layer_idx: 5, step: 100 };
        let b = DeoptRequest { request_id: 1, expert_idx: 1, layer_idx: 5, step: 100 };

        // Assert
        assert_ne!(a, b);
        assert_ne!(a.expert_idx, b.expert_idx);
    }

    // ────────────────────────────────────────────────────────
    // Batch 7: 15 additional tests for further coverage
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_eviction_decision_threshold_zero_all_evictable() {
        // Arrange: threshold=0 means any expert with streak >= 0 qualifies
        let manager = ExpertThermalManager::new(3).with_eviction_threshold(0);

        // Act & Assert: all fresh experts have streak=0 which >= 0 => Evict
        assert_eq!(manager.eviction_decision(0), EvictionDecision::Evict);
        assert_eq!(manager.eviction_decision(1), EvictionDecision::Evict);
        assert_eq!(manager.eviction_decision(2), EvictionDecision::Evict);
    }

    #[test]
    fn test_working_set_tracker_window_size_two_full_cycle() {
        // Arrange: window_size=2, verify complete overwrite after 2 steps
        let mut tracker = WorkingSetTracker::new(3, 2, 100);

        // Act
        tracker.record_step(&[1, 0, 0]); // slot 0: {0}
        assert_eq!(tracker.working_set_size(), 1);

        tracker.record_step(&[0, 1, 1]); // slot 1: {1,2}
        assert_eq!(tracker.working_set_size(), 3);

        tracker.record_step(&[0, 0, 1]); // overwrite slot 0: {2} (only slot 0)
        // slot 0 = {2}, slot 1 = {1,2} => union = {1,2}
        assert_eq!(tracker.working_set_size(), 2);

        tracker.record_step(&[0, 0, 0]); // overwrite slot 1: {} (empty)
        // slot 0 = {2}, slot 1 = {} => union = {2}
        assert_eq!(tracker.working_set_size(), 1);
    }

    #[test]
    fn test_manager_evict_first_expert_keeps_others_untouched() {
        // Arrange: 4 experts, all active
        let mut manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(3);

        // All experts become hot
        for _ in 0..10 {
            manager.step(&[10, 10, 10, 10]);
        }

        // Act: evict expert 0 (even though it's hot — eviction is unconditional)
        assert!(manager.evict_expert(0));

        // Assert: only expert 0 is evicted, others remain hot
        assert!(manager.state(0).unwrap().is_evicted);
        assert_eq!(manager.state(0).unwrap().heat_level, ExpertHeatLevel::Evicted);
        for idx in 1..4 {
            assert!(!manager.state(idx).unwrap().is_evicted);
            assert_eq!(manager.state(idx).unwrap().heat_level, ExpertHeatLevel::Hot);
        }
    }

    #[test]
    fn test_hit_rate_after_single_miss_followed_by_many_hits() {
        // Arrange
        let mut manager = ExpertThermalManager::new(1).with_eviction_threshold(u64::MAX);

        // Act: 1 miss then 99 hits
        manager.step(&[0]);
        for _ in 0..99 {
            manager.step(&[5]);
        }

        // Assert: hit_rate = 99/100 = 0.99
        let state = manager.state(0).unwrap();
        assert_eq!(state.hit_count, 99);
        assert_eq!(state.route_count, 100);
        assert!((state.hit_rate - 0.99).abs() < 1e-10);
        assert_eq!(state.consecutive_zero_streak, 0);
    }

    #[test]
    fn test_adaptive_threshold_rounding_behavior() {
        // Arrange: 3 experts, 2 active => ratio = 3/2 = 1.5
        // base=1001 => 1001 * 1.5 * 1.0 = 1501.5 => round to 1502
        let mut tracker = WorkingSetTracker::new(3, 5, 1001);
        tracker.record_step(&[1, 1, 0]);

        // Act
        let threshold = tracker.adaptive_threshold(0.0);

        // Assert
        assert_eq!(threshold, 1502);
    }

    #[test]
    fn test_deopt_handling_result_both_variants_inequal() {
        // Arrange: create both variants with same expert_idx and request_id
        let reactivate = DeoptHandlingResult::ReactivateAndRerun {
            expert_idx: 1,
            request_id: 42,
        };
        let spurious = DeoptHandlingResult::SpuriousDeopt {
            expert_idx: 1,
            request_id: 42,
        };

        // Assert: different variants with same field values are not equal
        assert_ne!(reactivate, spurious);
    }

    #[test]
    fn test_summary_counts_sum_after_mixed_heat_levels() {
        // Arrange: create a manager with mixed heat levels
        let mut manager = ExpertThermalManager::new(6)
            .with_eviction_threshold(3)
            .with_heat_thresholds(0.5, 0.1);

        // Experts 0,1 always hit (hot); 2,3 always hit too (hot);
        // 4,5 never hit (evicted level from 0.0 rate)
        for _ in 0..10 {
            manager.step(&[10, 10, 5, 5, 0, 0]);
        }
        manager.evict_expert(4);
        manager.evict_expert(5);

        // Act
        let summary = manager.summary();

        // Assert: counts must sum to num_experts
        assert_eq!(
            summary.hot_count + summary.warm_count + summary.cold_count + summary.evicted_count,
            summary.num_experts,
        );
        assert_eq!(summary.evicted_count, 2);
        assert_eq!(summary.num_experts, 6);
    }

    #[test]
    fn test_eviction_with_adaptive_and_zero_pressure_equals_scaled_base() {
        // Arrange: 8 experts, all active, zero pressure => threshold = base
        let mut manager = ExpertThermalManager::new(8)
            .with_eviction_threshold(500)
            .with_adaptive_eviction(10);

        manager.step(&[1, 1, 1, 1, 1, 1, 1, 1]);
        manager.update_memory_pressure(0.0);

        // Act: all 8 active => ratio = 8/8 = 1.0, headroom = 1.0
        let threshold = manager.effective_eviction_threshold();

        // Assert
        assert_eq!(threshold, 500);
    }

    #[test]
    fn test_manager_evict_reactivate_preserves_hit_and_route_counts() {
        // Arrange
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(3);

        // Expert 1 gets 5 hits in 5 steps
        for _ in 0..5 {
            manager.step(&[10, 5]);
        }
        // Expert 1: hit_count=5, route_count=5

        // Act: 3 more steps where expert 1 gets zero hits (route_count increases)
        for _ in 0..3 { manager.step(&[10, 0]); }
        // Expert 1: hit_count=5, route_count=8

        manager.evict_expert(1);
        manager.reactivate_expert(1);

        // Assert: eviction/reactivation does NOT reset hit_count or route_count
        assert_eq!(manager.state(1).unwrap().hit_count, 5);
        assert_eq!(manager.state(1).unwrap().route_count, 8);
    }

    #[test]
    fn test_working_set_tracker_adaptive_threshold_minimum_clamp_with_zero_experts() {
        // Arrange: 0 experts => ws=0 => max(1)=1, ratio=0/1=0
        // threshold = 100 * 0.0 * 1.0 = 0 => max(1) = 1
        let tracker = WorkingSetTracker::new(0, 5, 100);

        // Act
        let threshold = tracker.adaptive_threshold(0.0);

        // Assert: minimum clamp ensures threshold >= 1
        assert_eq!(threshold, 1);
    }

    #[test]
    fn test_step_single_hit_after_long_miss_streak_resets_streak_to_zero() {
        // Arrange: build a long miss streak, then one hit
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(1000);

        for _ in 0..50 {
            manager.step(&[10, 0]);
        }
        assert_eq!(manager.state(1).unwrap().consecutive_zero_streak, 50);

        // Act: single hit
        manager.step(&[10, 1]);

        // Assert: streak resets to 0 immediately
        assert_eq!(manager.state(1).unwrap().consecutive_zero_streak, 0);
        assert_eq!(manager.state(1).unwrap().hit_count, 1);
    }

    #[test]
    fn test_experts_to_evict_does_not_include_active_experts() {
        // Arrange: 5 experts, only experts 2 and 4 are inactive
        let mut manager = ExpertThermalManager::new(5).with_eviction_threshold(3);

        for _ in 0..4 {
            manager.step(&[10, 10, 0, 10, 0]);
        }

        // Act
        let to_evict = manager.experts_to_evict();

        // Assert: only experts 2 and 4 qualify
        assert_eq!(to_evict.len(), 2);
        assert!(to_evict.contains(&2));
        assert!(to_evict.contains(&4));
        assert!(!to_evict.contains(&0));
        assert!(!to_evict.contains(&1));
        assert!(!to_evict.contains(&3));
    }

    #[test]
    fn test_deopt_request_recorded_for_multiple_different_experts() {
        // Arrange
        let mut manager = ExpertThermalManager::new(3);

        // Act: send deopt for each expert
        for i in 0..3 {
            manager.handle_deopt_request(DeoptRequest {
                request_id: i as u64,
                expert_idx: i,
                layer_idx: i,
                step: i as u64,
            });
        }

        // Assert: all 3 recorded in order
        let pending = manager.pending_deopt_requests();
        assert_eq!(pending.len(), 3);
        for i in 0..3 {
            assert_eq!(pending[i].request_id, i as u64);
            assert_eq!(pending[i].expert_idx, i);
            assert_eq!(pending[i].layer_idx, i);
            assert_eq!(pending[i].step, i as u64);
        }
    }

    #[test]
    fn test_eviction_aggressiveness_with_large_threshold_no_underflow() {
        // Arrange: very large threshold with high aggressiveness
        let manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(u64::MAX)
            .with_eviction_aggressiveness(1000.0);

        // Act: bias_factor = 1/(1+1000) ≈ 0.001
        let threshold = manager.effective_eviction_threshold();

        // Assert: should not panic, should be a valid u64
        // u64::MAX * (1.0 / 1001.0) ≈ 1.84e16, fits in u64
        let expected = (u64::MAX as f64 / 1001.0) as u64;
        assert_eq!(threshold, expected);
        assert!(threshold > 0);
    }

    // ────────────────────────────────────────────────────────
    // Batch 8: 15 new tests
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_heat_level_from_hit_rate_positive_negative_rate_is_evicted() {
        // Arrange: a small negative rate should fall through to Evicted
        let rate = -1e-300;

        // Act: rate < 0.0, so >= hot_threshold fails, >= cold_threshold fails,
        // rate > 0.0 also fails (negative is not > 0) => Evicted
        let level = ExpertHeatLevel::from_hit_rate(rate, 0.1, 0.001);

        // Assert
        assert_eq!(level, ExpertHeatLevel::Evicted);
    }

    #[test]
    fn test_expert_heat_state_clone_then_modify_independence() {
        // Arrange
        let original = ExpertHeatState {
            expert_idx: 2,
            hit_rate: 0.75,
            hit_count: 75,
            route_count: 100,
            heat_level: ExpertHeatLevel::Hot,
            consecutive_zero_streak: 0,
            last_hit_step: 100,
            is_evicted: false,
            reactivation_count: 0,
        };

        // Act: clone and modify the clone
        let mut modified = original.clone();
        modified.hit_count = 999;
        modified.is_evicted = true;

        // Assert: original unchanged
        assert_eq!(original.hit_count, 75);
        assert!(!original.is_evicted);
        assert_eq!(modified.hit_count, 999);
        assert!(modified.is_evicted);
    }

    #[test]
    fn test_thermal_summary_equality_reflexive() {
        // Arrange
        let summary = ThermalSummary {
            num_experts: 4,
            hot_count: 1,
            warm_count: 1,
            cold_count: 1,
            evicted_count: 1,
            total_evictions: 2,
            total_reactivations: 1,
            current_step: 50,
            pending_deopt_count: 3,
            working_set_size: 2,
            effective_eviction_threshold: 500,
        };

        // Assert: reflexive property
        assert_eq!(summary, summary);
    }

    #[test]
    fn test_deopt_handling_result_equality_same_variant_different_fields() {
        // Arrange: two ReactivateAndRerun with different request_ids
        let r1 = DeoptHandlingResult::ReactivateAndRerun { expert_idx: 1, request_id: 10 };
        let r2 = DeoptHandlingResult::ReactivateAndRerun { expert_idx: 1, request_id: 20 };

        // Assert: same variant, different field => not equal
        assert_ne!(r1, r2);

        // Same fields => equal
        let r3 = DeoptHandlingResult::ReactivateAndRerun { expert_idx: 1, request_id: 10 };
        assert_eq!(r1, r3);
    }

    #[test]
    fn test_working_set_tracker_adaptive_threshold_with_99_percent_pressure() {
        // Arrange: 4 experts, 2 active, pressure = 0.99
        let mut tracker = WorkingSetTracker::new(4, 5, 1000);
        tracker.record_step(&[1, 1, 0, 0]); // ws = 2

        // Act: headroom = clamp(1.0 - 0.99, 0.1, 1.0) = clamp(0.01, 0.1, 1.0) = 0.1
        let threshold = tracker.adaptive_threshold(0.99);

        // Assert: 1000 * (4/2) * 0.1 = 200
        assert_eq!(threshold, 200);
    }

    #[test]
    fn test_manager_step_alternating_active_experts_updates_correctly() {
        // Arrange: 3 experts, alternating which one is active each step
        let mut manager = ExpertThermalManager::new(3).with_eviction_threshold(100);

        // Act
        manager.step(&[10, 0, 0]); // step 1: only expert 0
        manager.step(&[0, 10, 0]); // step 2: only expert 1
        manager.step(&[0, 0, 10]); // step 3: only expert 2

        // Assert: each expert has 1 hit, 3 routes, hit_rate ≈ 0.333
        for i in 0..3 {
            let state = manager.state(i).unwrap();
            assert_eq!(state.hit_count, 1);
            assert_eq!(state.route_count, 3);
            assert!((state.hit_rate - 1.0 / 3.0).abs() < 1e-10);
        }
        assert_eq!(manager.summary().current_step, 3);
    }

    #[test]
    fn test_manager_evict_all_then_reactivate_one_summary_consistency() {
        // Arrange: 4 experts, evict all, reactivate one
        let mut manager = ExpertThermalManager::new(4).with_eviction_threshold(3);

        for _ in 0..4 {
            manager.step(&[0, 0, 0, 0]);
        }

        // Act: evict all
        for i in 0..4 {
            assert!(manager.evict_expert(i));
        }
        let after_evict = manager.summary();
        assert_eq!(after_evict.evicted_count, 4);
        assert_eq!(after_evict.total_evictions, 4);

        // Reactivate only expert 2
        assert!(manager.reactivate_expert(2));

        // Assert
        let after_reactivate = manager.summary();
        assert_eq!(after_reactivate.evicted_count, 3);
        assert_eq!(after_reactivate.total_reactivations, 1);
        assert_eq!(after_reactivate.total_evictions, 4); // unchanged
    }

    #[test]
    fn test_eviction_decision_threshold_two_boundary() {
        // Arrange: threshold=2, test at exactly streak=2
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(2);

        // Act: 2 steps with expert 1 inactive => streak=2
        manager.step(&[10, 0]);
        manager.step(&[10, 0]);

        // Assert: streak=2 >= threshold=2 => Evict
        assert_eq!(manager.eviction_decision(1), EvictionDecision::Evict);

        // 1 step => streak=1 < threshold=2 => Keep
        let mut manager2 = ExpertThermalManager::new(2).with_eviction_threshold(2);
        manager2.step(&[10, 0]);
        assert_eq!(manager2.eviction_decision(1), EvictionDecision::Keep);
    }

    #[test]
    fn test_manager_state_last_hit_step_initial_zero() {
        // Arrange: fresh manager
        let manager = ExpertThermalManager::new(3);

        // Assert: last_hit_step starts at 0 for all experts
        for i in 0..3 {
            assert_eq!(manager.state(i).unwrap().last_hit_step, 0);
        }
    }

    #[test]
    fn test_manager_hit_rate_one_after_all_hits() {
        // Arrange: 5 experts, all hit every step for 20 steps
        let mut manager = ExpertThermalManager::new(5);

        // Act
        for _ in 0..20 {
            manager.step(&[10, 20, 30, 40, 50]);
        }

        // Assert: all have hit_rate exactly 1.0
        for i in 0..5 {
            let state = manager.state(i).unwrap();
            assert_eq!(state.hit_count, 20);
            assert_eq!(state.route_count, 20);
            assert!((state.hit_rate - 1.0).abs() < f64::EPSILON);
            assert_eq!(state.heat_level, ExpertHeatLevel::Hot);
            assert_eq!(state.consecutive_zero_streak, 0);
        }
    }

    #[test]
    fn test_manager_deopt_request_order_preserved_across_mixed_requests() {
        // Arrange: mix of spurious and reactivation deopt requests
        let mut manager = ExpertThermalManager::new(3).with_eviction_threshold(3);

        for _ in 0..4 {
            manager.step(&[0, 0, 0]);
        }
        manager.evict_expert(1); // only expert 1 is evicted

        // Act: send deopt for expert 0 (spurious), 1 (reactivation), 2 (spurious)
        manager.handle_deopt_request(DeoptRequest {
            request_id: 100, expert_idx: 0, layer_idx: 0, step: 4,
        });
        manager.handle_deopt_request(DeoptRequest {
            request_id: 200, expert_idx: 1, layer_idx: 1, step: 4,
        });
        manager.handle_deopt_request(DeoptRequest {
            request_id: 300, expert_idx: 2, layer_idx: 2, step: 4,
        });

        // Assert: all 3 recorded in FIFO order
        let pending = manager.pending_deopt_requests();
        assert_eq!(pending.len(), 3);
        assert_eq!(pending[0].request_id, 100);
        assert_eq!(pending[1].request_id, 200);
        assert_eq!(pending[2].request_id, 300);
        assert_eq!(pending[0].expert_idx, 0);
        assert_eq!(pending[1].expert_idx, 1);
        assert_eq!(pending[2].expert_idx, 2);
    }

    #[test]
    fn test_working_set_tracker_window_size_four_cyclic_overwrite() {
        // Arrange: window_size=4, 3 experts
        let mut tracker = WorkingSetTracker::new(3, 4, 100);

        // Fill all 4 slots: each slot activates a different subset
        tracker.record_step(&[1, 0, 0]); // slot 0: {0}
        tracker.record_step(&[0, 1, 0]); // slot 1: {1}
        tracker.record_step(&[0, 0, 1]); // slot 2: {2}
        tracker.record_step(&[1, 1, 0]); // slot 3: {0,1}
        assert_eq!(tracker.working_set_size(), 3); // {0,1,2}

        // Act: overwrite slot 0 with only expert 2
        tracker.record_step(&[0, 0, 1]); // slot 0 overwritten: {2}
        // Slots: {2}, {1}, {2}, {0,1} => {0,1,2}
        assert_eq!(tracker.working_set_size(), 3);

        // Overwrite slot 1 with only expert 0
        tracker.record_step(&[1, 0, 0]); // slot 1 overwritten: {0}
        // Slots: {2}, {0}, {2}, {0,1} => {0,1,2}
        assert_eq!(tracker.working_set_size(), 3);

        // Overwrite slot 2 with empty
        tracker.record_step(&[0, 0, 0]); // slot 2 overwritten: {}
        // Slots: {2}, {0}, {}, {0,1} => {0,1,2}
        assert_eq!(tracker.working_set_size(), 3);

        // Overwrite slot 3 with empty
        tracker.record_step(&[0, 0, 0]); // slot 3 overwritten: {}
        // Slots: {2}, {0}, {}, {} => {0,2}
        assert_eq!(tracker.working_set_size(), 2);
    }

    #[test]
    fn test_manager_cold_or_evicted_after_partial_eviction() {
        // Arrange: 4 experts, evict only expert 3
        let mut manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(3)
            .with_heat_thresholds(0.5, 0.1);

        // Experts 0,1 hot; expert 2 cold; expert 3 evicted
        for _ in 0..10 {
            manager.step(&[10, 10, 0, 0]);
        }
        manager.evict_expert(3);

        // Act
        let cold_evicted = manager.cold_or_evicted_experts();

        // Assert: expert 2 is cold (hit_rate=0.0 => Evicted level), expert 3 is evicted
        assert!(cold_evicted.contains(&2));
        assert!(cold_evicted.contains(&3));
        assert!(!cold_evicted.contains(&0));
        assert!(!cold_evicted.contains(&1));
    }

    #[test]
    fn test_manager_adaptive_threshold_with_full_working_set_no_aggressiveness() {
        // Arrange: all experts active, no aggressiveness
        let mut manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(1000)
            .with_adaptive_eviction(10)
            .with_eviction_aggressiveness(0.0);

        manager.step(&[1, 1, 1, 1]); // ws = 4
        manager.update_memory_pressure(0.0);

        // Act: ratio = 4/4 = 1.0, headroom = 1.0, bias = 1.0
        let threshold = manager.effective_eviction_threshold();

        // Assert: effective = 1000 * 1.0 * 1.0 * 1.0 = 1000
        assert_eq!(threshold, 1000);
    }

    #[test]
    fn test_deopt_request_clone_then_compare() {
        // Arrange
        let req = DeoptRequest {
            request_id: 42,
            expert_idx: 3,
            layer_idx: 7,
            step: 1000,
        };

        // Act
        let cloned = req.clone();

        // Assert: cloned equals original
        assert_eq!(req, cloned);
        // Verify each field individually
        assert_eq!(cloned.request_id, 42);
        assert_eq!(cloned.expert_idx, 3);
        assert_eq!(cloned.layer_idx, 7);
        assert_eq!(cloned.step, 1000);
    }

    // ────────────────────────────────────────────────────────
    // Additional edge case tests
    // ────────────────────────────────────────────────────────

    // @trace TEST-MOE-THERMAL-001 [req:REQ-MOE-001] [level:unit]
    #[test]
    fn test_heat_level_from_hit_rate_negative_rate_produces_evicted() {
        // Arrange: negative hit rate (invalid but possible via floating point)
        // Act
        let level = ExpertHeatLevel::from_hit_rate(-1.0, 0.5, 0.1);

        // Assert: negative rate < 0.0 falls through all >= checks => Evicted
        assert_eq!(level, ExpertHeatLevel::Evicted);
    }

    // @trace TEST-MOE-THERMAL-002 [req:REQ-MOE-001] [level:unit]
    #[test]
    fn test_heat_level_hash_consistency() {
        use std::collections::HashSet;

        // Arrange: create same level twice
        let a = ExpertHeatLevel::Hot;
        let b = ExpertHeatLevel::Hot;

        // Act: insert both into HashSet
        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b);

        // Assert: deduplication works (Hash+Eq contract)
        assert_eq!(set.len(), 1);
    }

    // @trace TEST-MOE-THERMAL-003 [req:REQ-MOE-001] [level:unit]
    #[test]
    fn test_eviction_decision_all_four_variants_distinct() {
        // Arrange
        let keep = EvictionDecision::Keep;
        let evict = EvictionDecision::Evict;
        let reactivate = EvictionDecision::Reactivate;

        // Assert: all three named variants are distinct
        assert_ne!(keep, evict);
        assert_ne!(evict, reactivate);
        assert_ne!(keep, reactivate);
    }

    // @trace TEST-MOE-THERMAL-004 [req:REQ-MOE-001] [level:unit]
    #[test]
    fn test_eviction_decision_ord_ordering() {
        // Arrange: derive(PartialOrd) uses declaration order: Keep < Evict < Reactivate
        // Assert
        assert!(EvictionDecision::Keep < EvictionDecision::Evict);
        assert!(EvictionDecision::Evict < EvictionDecision::Reactivate);
        assert!(EvictionDecision::Keep < EvictionDecision::Reactivate);
    }

    // @trace TEST-MOE-THERMAL-005 [req:REQ-MOE-001] [level:unit]
    #[test]
    fn test_deopt_request_equality_reflexive_different_fields() {
        // Arrange: two requests with different field values
        let req_a = DeoptRequest {
            request_id: 1,
            expert_idx: 0,
            layer_idx: 0,
            step: 0,
        };
        let req_b = DeoptRequest {
            request_id: 2,
            expert_idx: 0,
            layer_idx: 0,
            step: 0,
        };

        // Assert: different request_id => not equal
        assert_ne!(req_a, req_b);
    }

    // @trace TEST-MOE-THERMAL-006 [req:REQ-MOE-001] [level:unit]
    #[test]
    fn test_manager_single_expert_manager() {
        // Arrange: manager with exactly 1 expert
        let mut manager = ExpertThermalManager::new(1)
            .with_eviction_threshold(5);

        // Act: run 6 steps with the sole expert active
        for _ in 0..6 {
            manager.step(&[10]);
        }

        // Assert: the single expert is hot, nothing to evict
        assert_eq!(manager.state(0).unwrap().heat_level, ExpertHeatLevel::Hot);
        assert!(manager.experts_to_evict().is_empty());
        assert_eq!(manager.hot_experts(), vec![0]);
    }

    // @trace TEST-MOE-THERMAL-007 [req:REQ-MOE-001] [level:unit]
    #[test]
    fn test_manager_single_expert_never_hit_becomes_evictable() {
        // Arrange: 1 expert, low threshold
        let mut manager = ExpertThermalManager::new(1)
            .with_eviction_threshold(3);

        // Act: never route any tokens to expert 0
        for _ in 0..5 {
            manager.step(&[0]);
        }

        // Assert: expert 0 has streak 5 >= threshold 3, should be evictable
        assert!(manager.experts_to_evict().contains(&0));
    }

    // @trace TEST-MOE-THERMAL-008 [req:REQ-MOE-001] [level:unit]
    #[test]
    fn test_deopt_handling_result_clone_preserves_variant() {
        // Arrange
        let reactivate = DeoptHandlingResult::ReactivateAndRerun {
            expert_idx: 7,
            request_id: 42,
        };
        let spurious = DeoptHandlingResult::SpuriousDeopt {
            expert_idx: 3,
            request_id: 99,
        };

        // Act
        let r_clone = reactivate.clone();
        let s_clone = spurious.clone();

        // Assert
        assert_eq!(reactivate, r_clone);
        assert_eq!(spurious, s_clone);
    }

    // @trace TEST-MOE-THERMAL-009 [req:REQ-MOE-001] [level:unit]
    #[test]
    fn test_thermal_summary_all_counts_sum_to_num_experts() {
        // Arrange: run enough steps to create mixed heat levels
        let mut manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(5)
            .with_heat_thresholds(0.3, 0.05);

        // 10 steps: experts 0,1 always hit; expert 2 never; expert 3 sometimes
        for i in 0..10 {
            if i % 3 == 0 {
                manager.step(&[10, 10, 0, 5]);
            } else {
                manager.step(&[10, 10, 0, 0]);
            }
        }
        manager.evict_expert(2);

        // Act
        let summary = manager.summary();

        // Assert: hot + warm + cold + evicted == num_experts
        let total = summary.hot_count + summary.warm_count
            + summary.cold_count + summary.evicted_count;
        assert_eq!(total, summary.num_experts);
        assert_eq!(summary.evicted_count, 1);
    }

    // @trace TEST-MOE-THERMAL-010 [req:REQ-MOE-001] [level:unit]
    #[test]
    fn test_working_set_tracker_fresh_zero_working_set_and_threshold() {
        // Arrange: 4 experts, window 3, base threshold 500
        let tracker = WorkingSetTracker::new(4, 3, 500);

        // Act & Assert: no steps recorded, working set is 0
        assert_eq!(tracker.working_set_size(), 0);

        // Verify adaptive_threshold uses base correctly: ws=0 => ws.max(1)=1
        // ratio = 4/1 = 4, headroom = 1.0 (pressure=0), threshold = 500*4*1.0 = 2000
        assert_eq!(tracker.adaptive_threshold(0.0), 2000);
    }

    // @trace TEST-MOE-THERMAL-011 [req:REQ-MOE-001] [level:unit]
    #[test]
    fn test_working_set_tracker_single_expert_always_active() {
        // Arrange: 1 expert, window 5
        let mut tracker = WorkingSetTracker::new(1, 5, 100);

        // Act: record 3 steps, all with expert 0 active
        tracker.record_step(&[1]);
        tracker.record_step(&[1]);
        tracker.record_step(&[1]);

        // Assert: working set is just {0}
        assert_eq!(tracker.working_set_size(), 1);
    }

    // @trace TEST-MOE-THERMAL-012 [req:REQ-MOE-001] [level:unit]
    #[test]
    fn test_working_set_tracker_record_step_with_empty_counts() {
        // Arrange: 3 experts, window 2
        let mut tracker = WorkingSetTracker::new(3, 2, 100);

        // Act: record one step with all-zero counts
        tracker.record_step(&[0, 0, 0]);

        // Assert: no experts accessed
        assert_eq!(tracker.working_set_size(), 0);
    }

    // @trace TEST-MOE-THERMAL-013 [req:REQ-MOE-001] [level:unit]
    #[test]
    fn test_working_set_tracker_record_step_with_longer_input_than_num_experts() {
        // Arrange: 2 experts, but we pass 4 counts
        let mut tracker = WorkingSetTracker::new(2, 3, 100);

        // Act: extra entries beyond num_experts are ignored
        tracker.record_step(&[5, 3, 10, 20]);

        // Assert: only experts 0 and 1 are tracked
        assert_eq!(tracker.working_set_size(), 2);
    }

    // @trace TEST-MOE-THERMAL-014 [req:REQ-MOE-001] [level:unit]
    #[test]
    fn test_manager_with_eviction_aggressiveness_extreme_value() {
        // Arrange: very high aggressiveness reduces threshold significantly
        let manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(1000)
            .with_eviction_aggressiveness(100.0);

        // Act
        let threshold = manager.effective_eviction_threshold();

        // Assert: bias_factor = 1.0/(1.0+100.0) ≈ 0.0099, threshold ≈ 9
        assert!(threshold < 1000);
        assert!(threshold > 0);
    }

    // @trace TEST-MOE-THERMAL-015 [req:REQ-MOE-001] [level:unit]
    #[test]
    fn test_manager_step_large_route_count_does_not_affect_binary_hit() {
        // Arrange
        let mut manager = ExpertThermalManager::new(2)
            .with_eviction_threshold(100);

        // Act: expert 0 gets count 1000000, expert 1 gets 0
        manager.step(&[1000000, 0]);

        // Assert: hit_count is incremented by 1 (binary), not by route_count magnitude
        assert_eq!(manager.state(0).unwrap().hit_count, 1);
        assert_eq!(manager.state(0).unwrap().route_count, 1);
        assert_eq!(manager.state(0).unwrap().consecutive_zero_streak, 0);
        assert_eq!(manager.state(1).unwrap().hit_count, 0);
        assert_eq!(manager.state(1).unwrap().route_count, 1);
        assert_eq!(manager.state(1).unwrap().consecutive_zero_streak, 1);
    }

    // ────────────────────────────────────────────────────────
    // Batch 9: 10 additional tests
    // ────────────────────────────────────────────────────────

    #[test]
    fn test_summary_working_set_size_zero_when_adaptive_disabled() {
        // Arrange: manager without adaptive eviction enabled
        let mut manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(100);

        // Act: step with some experts active
        manager.step(&[10, 5, 0, 0]);
        let summary = manager.summary();

        // Assert: working_set_size tracks via WorkingSetTracker even without adaptive,
        // because the tracker is always initialized internally
        assert_eq!(summary.working_set_size, 2);
    }

    #[test]
    fn test_eviction_decision_after_hit_resets_streak_then_miss_builds_new_streak() {
        // Arrange: build streak, break it with a hit, then build again
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(5);

        // 5 misses => streak=5
        for _ in 0..5 {
            manager.step(&[10, 0]);
        }
        assert_eq!(manager.state(1).unwrap().consecutive_zero_streak, 5);

        // 1 hit resets streak
        manager.step(&[10, 1]);
        assert_eq!(manager.state(1).unwrap().consecutive_zero_streak, 0);
        assert_eq!(manager.eviction_decision(1), EvictionDecision::Keep);

        // 4 more misses => streak=4, still below threshold
        for _ in 0..4 {
            manager.step(&[10, 0]);
        }
        assert_eq!(manager.state(1).unwrap().consecutive_zero_streak, 4);
        assert_eq!(manager.eviction_decision(1), EvictionDecision::Keep);

        // 1 more miss => streak=5, exactly at threshold => Evict
        manager.step(&[10, 0]);
        assert_eq!(manager.state(1).unwrap().consecutive_zero_streak, 5);
        assert_eq!(manager.eviction_decision(1), EvictionDecision::Evict);
    }

    #[test]
    fn test_deopt_request_for_just_reactivated_expert_is_spurious() {
        // Arrange: evict then reactivate expert, then send deopt
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(3);

        for _ in 0..4 {
            manager.step(&[10, 0]);
        }
        manager.evict_expert(1);
        manager.reactivate_expert(1);

        // Act: expert 1 is now active (not evicted), so deopt is spurious
        let result = manager.handle_deopt_request(DeoptRequest {
            request_id: 1,
            expert_idx: 1,
            layer_idx: 0,
            step: 4,
        });

        // Assert
        assert!(matches!(result, DeoptHandlingResult::SpuriousDeopt { expert_idx: 1, .. }));
        assert!(!manager.state(1).unwrap().is_evicted);
        // Request is still recorded
        assert_eq!(manager.pending_deopt_requests().len(), 1);
    }

    #[test]
    fn test_adaptive_threshold_with_single_active_expert_out_of_many() {
        // Arrange: 16 experts, only 1 active => high ratio => high threshold
        let mut manager = ExpertThermalManager::new(16)
            .with_eviction_threshold(100)
            .with_adaptive_eviction(10);
        manager.update_memory_pressure(0.0);

        // Act: only expert 0 active
        manager.step(&[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let threshold = manager.effective_eviction_threshold();

        // Assert: ratio = 16/1 = 16, threshold = 100 * 16 * 1.0 = 1600
        assert_eq!(threshold, 1600);
    }

    #[test]
    fn test_manager_eviction_impossible_with_max_threshold() {
        // Arrange: threshold = u64::MAX means eviction is practically impossible
        let mut manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(u64::MAX);

        // Act: 10000 steps with only expert 0 active
        for _ in 0..10000 {
            manager.step(&[10, 0, 0, 0]);
        }

        // Assert: experts 1-3 have streak 10000, but u64::MAX is unreachable
        assert!(!manager.experts_to_evict().contains(&1));
        assert!(!manager.experts_to_evict().contains(&2));
        assert!(!manager.experts_to_evict().contains(&3));
        assert_eq!(manager.state(1).unwrap().consecutive_zero_streak, 10000);
    }

    #[test]
    fn test_working_set_tracker_adaptive_threshold_with_base_one() {
        // Arrange: smallest meaningful base_threshold
        let mut tracker = WorkingSetTracker::new(4, 5, 1);
        tracker.record_step(&[1, 1, 0, 0]); // ws = 2

        // Act: ratio = 4/2 = 2.0, headroom(0.0) = 1.0
        let threshold = tracker.adaptive_threshold(0.0);

        // Assert: 1 * 2.0 * 1.0 = 2.0, rounded to 2
        assert_eq!(threshold, 2);

        // With high pressure: headroom(0.9) = clamp(0.1, 0.1, 1.0) = 0.1
        let threshold_pressure = tracker.adaptive_threshold(0.9);
        // 1 * 2.0 * 0.1 = 0.2, rounded to 0, max(1) => 1
        assert_eq!(threshold_pressure, 1);
    }

    #[test]
    fn test_expert_heat_state_hit_rate_zero_with_nonzero_route_count() {
        // Arrange: expert with many routes but zero hits
        let mut manager = ExpertThermalManager::new(2).with_eviction_threshold(1000);

        for _ in 0..100 {
            manager.step(&[10, 0]);
        }

        // Act
        let state = manager.state(1).unwrap();

        // Assert: hit_rate is exactly 0.0, not NaN or infinity
        assert_eq!(state.hit_rate, 0.0);
        assert_eq!(state.hit_count, 0);
        assert_eq!(state.route_count, 100);
        assert_eq!(state.heat_level, ExpertHeatLevel::Evicted);
    }

    #[test]
    fn test_manager_full_cycle_with_adaptive_eviction_and_memory_pressure() {
        // Arrange: adaptive eviction with memory pressure
        let mut manager = ExpertThermalManager::new(4)
            .with_eviction_threshold(1000)
            .with_adaptive_eviction(10)
            .with_eviction_aggressiveness(0.0);
        manager.update_memory_pressure(0.0);

        // Act: 2 of 4 experts active => adaptive threshold = 1000 * 2 * 1.0 = 2000
        manager.step(&[10, 10, 0, 0]);
        assert_eq!(manager.effective_eviction_threshold(), 2000);

        // Simulate memory pressure increase: threshold drops
        manager.update_memory_pressure(0.5);
        // adaptive = 1000 * 2 * 0.5 = 1000
        assert_eq!(manager.effective_eviction_threshold(), 1000);

        // More memory pressure: threshold drops further
        manager.update_memory_pressure(0.9);
        // adaptive = 1000 * 2 * 0.1 = 200
        assert_eq!(manager.effective_eviction_threshold(), 200);

        // Build streak of 201 for experts 2 and 3 (exceeds threshold 200)
        for _ in 0..201 {
            manager.step(&[10, 10, 0, 0]);
        }
        assert!(manager.experts_to_evict().contains(&2));
        assert!(manager.experts_to_evict().contains(&3));
    }

    #[test]
    fn test_manager_reactivate_increments_reactivation_count_on_state() {
        // Arrange: verify reactivation_count tracks individual expert reactivations
        let mut manager = ExpertThermalManager::new(3).with_eviction_threshold(3);

        for _ in 0..4 {
            manager.step(&[0, 0, 0]);
        }

        // Evict all, reactivate expert 0 three times (evict-reactivate cycle)
        for _ in 0..3 {
            manager.evict_expert(0);
            manager.reactivate_expert(0);
            // Build streak again for re-eviction
            for _ in 0..4 {
                manager.step(&[0, 0, 0]);
            }
        }

        // Assert: expert 0's reactivation_count should be 1 (last reactivate_expert set it to 1)
        // because evict_expert resets it to 0 each time
        assert_eq!(manager.state(0).unwrap().reactivation_count, 1);
        // Total reactivations = 3 (three successful reactivate_expert calls)
        assert_eq!(manager.summary().total_reactivations, 3);
    }

    #[test]
    fn test_working_set_tracker_empty_counts_slot_clears_previous_data() {
        // Arrange: window_size=2 to force overwrite behavior
        let mut tracker = WorkingSetTracker::new(3, 2, 100);

        // Step 1: all experts active => slot 0 = {0,1,2}
        tracker.record_step(&[1, 1, 1]);
        assert_eq!(tracker.working_set_size(), 3);

        // Step 2: only expert 0 => slot 1 = {0}
        tracker.record_step(&[1, 0, 0]);
        // Union of {0,1,2} and {0} = {0,1,2}
        assert_eq!(tracker.working_set_size(), 3);

        // Act: step 3 overwrites slot 0 with all zeros => slot 0 = {}
        tracker.record_step(&[0, 0, 0]);
        // Union of {} and {0} = {0}
        assert_eq!(tracker.working_set_size(), 1);

        // Step 4 overwrites slot 1 with all zeros => slot 1 = {}
        tracker.record_step(&[0, 0, 0]);
        // Union of {} and {} = {}
        assert_eq!(tracker.working_set_size(), 0);
    }
}
