//! MoE 生产级专家路由与分发 (SPEC §15.1)
//!
//! ## 核心职责
//! 实现 MoE (Mixture-of-Experts) 的生产级路由:
//! - Top-K 专家选择 + 容量因子 (Capacity Factor)
//! - 负载均衡 (Load Balancing) — 防止专家过载/空闲
//! - 路由表生成 — 供 Mega-Kernel 内核分发使用
//! - 专家权重分页管理
//!
//! ## 数据流
//! ```
//! Gate Logits → TopK Selection → Capacity Filtering → Route Table
//!                                                         ↓
//!                                               Mega-Kernel Dispatch
//! ```
//!
//! ## §15.1 核内分发零启动开销
//! MoE 层只启动 1 个 Mega-Kernel，Thread Block 拿到 Softmax(Gate) 后
//! 利用内置字典读出目标 Expert，Kernel 内部利用汇编 jmp 直接跃迁。

/// 专家路由配置
#[derive(Debug, Clone)]
pub struct ExpertRouteConfig {
    /// 总专家数
    pub num_experts: usize,
    /// Top-K 选择数
    pub top_k: usize,
    /// 容量因子 (每个专家最大 token 数 = capacity_factor * total_tokens / num_experts)
    pub capacity_factor: f32,
    /// 是否启用负载均衡辅助损失
    pub load_balance_loss: bool,
    /// 负载均衡辅助损失权重 (λ)
    pub load_balance_lambda: f32,
    /// 是否启用专家噪声注入 (训练用，推理时为 0.0)
    pub noise_sigma: f32,
}

impl ExpertRouteConfig {
    /// 创建新的专家路由配置
    pub fn new(num_experts: usize, top_k: usize) -> Self {
        Self {
            num_experts,
            top_k,
            capacity_factor: 1.25,
            load_balance_loss: false,
            load_balance_lambda: 0.01,
            noise_sigma: 0.0,
        }
    }

    /// 每个专家的最大 token 容量
    pub fn expert_capacity(&self, total_tokens: usize) -> usize {
        (self.capacity_factor * total_tokens as f32 / self.num_experts as f32).ceil() as usize
    }
}

impl Default for ExpertRouteConfig {
    fn default() -> Self {
        Self::new(8, 2)
    }
}

/// 单个 token 的路由结果
#[derive(Debug, Clone)]
pub struct TokenRoute {
    /// 被选中的专家索引列表 (长度 = top_k)
    pub expert_indices: Vec<usize>,
    /// 对应的专家权重 (softmax 后的概率)
    pub expert_weights: Vec<f32>,
    /// 该 token 在各专家中的位置 (用于 scatter/gather)
    pub expert_positions: Vec<usize>,
}

/// 一层的完整路由表
///
/// §15.1: 供 Mega-Kernel 内核分发使用的路由表。
/// 包含每个 token 的专家分配和位置信息。
#[derive(Debug, Clone)]
pub struct ExpertRouteTable {
    /// 路由配置
    pub config: ExpertRouteConfig,
    /// 每个 token 的路由结果
    pub token_routes: Vec<TokenRoute>,
    /// 每个专家已分配的 token 数 (用于容量检查)
    pub expert_token_counts: Vec<usize>,
    /// 溢出的 token 数 (超出专家容量的 token)
    pub overflow_count: usize,
}

impl ExpertRouteTable {
    /// 从 gate logits 计算路由表
    ///
    /// §15.1 核内分发: Top-K 选择 + 容量过滤
    pub fn from_gate_logits(
        config: ExpertRouteConfig,
        gate_logits: &[Vec<f32>], // [num_tokens][num_experts]
    ) -> Self {
        let num_tokens = gate_logits.len();
        let capacity = config.expert_capacity(num_tokens);
        let mut expert_token_counts = vec![0usize; config.num_experts];
        let mut token_routes = Vec::with_capacity(num_tokens);
        let mut overflow_count = 0;

        for token_logits in gate_logits {
            // 1. Top-K 选择
            let topk = topk_with_weights(token_logits, config.top_k);

            // 2. 容量过滤: 检查每个专家是否还有容量
            let mut selected_indices = Vec::with_capacity(config.top_k);
            let mut selected_weights = Vec::with_capacity(config.top_k);
            let mut selected_positions = Vec::with_capacity(config.top_k);

            for (expert_idx, weight) in topk {
                if expert_token_counts[expert_idx] < capacity {
                    let position = expert_token_counts[expert_idx];
                    expert_token_counts[expert_idx] += 1;
                    selected_indices.push(expert_idx);
                    selected_weights.push(weight);
                    selected_positions.push(position);
                } else {
                    // 专家已满，该 token 溢出
                    overflow_count += 1;
                }
            }

            // 如果所有 top-k 专家都满了，分配到负载最低的专家
            if selected_indices.is_empty() {
                let (least_loaded, _) = expert_token_counts
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, &count)| count)
                    .expect("at least one expert");
                let position = expert_token_counts[least_loaded];
                expert_token_counts[least_loaded] += 1;
                // 权重设为 1.0 (单个专家全权负责)
                selected_indices.push(least_loaded);
                selected_weights.push(1.0);
                selected_positions.push(position);
            }

            token_routes.push(TokenRoute {
                expert_indices: selected_indices,
                expert_weights: selected_weights,
                expert_positions: selected_positions,
            });
        }

        Self {
            config,
            token_routes,
            expert_token_counts,
            overflow_count,
        }
    }

    /// 获取路由到指定专家的所有 token 索引
    pub fn tokens_for_expert(&self, expert_idx: usize) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        for (token_idx, route) in self.token_routes.iter().enumerate() {
            for (i, &ei) in route.expert_indices.iter().enumerate() {
                if ei == expert_idx {
                    result.push((token_idx, route.expert_positions[i]));
                }
            }
        }
        result
    }

    /// 计算负载均衡辅助损失 (Load Balance Loss)
    ///
    /// L_balance = λ * N * Σ(f_i * P_i)
    /// 其中 f_i = 路由到专家 i 的 token 比例
    ///      P_i = 专家 i 的平均 gate 概率
    pub fn load_balance_loss(&self, gate_logits: &[Vec<f32>]) -> f32 {
        if !self.config.load_balance_loss {
            return 0.0;
        }

        let n = self.config.num_experts as f32;
        let num_tokens = self.token_routes.len() as f32;
        let lambda = self.config.load_balance_lambda;

        // f_i: 路由到专家 i 的 token 比例
        let f: Vec<f32> = self
            .expert_token_counts
            .iter()
            .map(|&count| count as f32 / num_tokens)
            .collect();

        // P_i: 专家 i 的平均 gate 概率
        let mut p = vec![0.0f32; self.config.num_experts];
        for token_logits in gate_logits {
            let softmax = softmax(token_logits);
            for (i, &prob) in softmax.iter().enumerate() {
                p[i] += prob;
            }
        }
        for prob in p.iter_mut() {
            *prob /= num_tokens;
        }

        // L_balance = λ * N * Σ(f_i * P_i)
        let sum: f32 = f.iter().zip(p.iter()).map(|(&fi, &pi)| fi * pi).sum();
        lambda * n * sum
    }

    /// 获取专家利用率统计
    pub fn utilization_stats(&self) -> ExpertUtilizationStats {
        let total_slots: usize = self.expert_token_counts.iter().sum();
        let max_count = *self.expert_token_counts.iter().max().unwrap_or(&0);
        let min_count = *self.expert_token_counts.iter().min().unwrap_or(&0);
        let mean = if self.config.num_experts > 0 {
            total_slots as f32 / self.config.num_experts as f32
        } else {
            0.0
        };

        // 均衡度: 1.0 = 完美均衡, 0.0 = 完全不均衡
        let balance = if max_count > 0 {
            1.0 - (max_count - min_count) as f32 / max_count as f32
        } else {
            1.0
        };

        ExpertUtilizationStats {
            total_tokens: self.token_routes.len(),
            total_expert_assignments: total_slots,
            overflow_count: self.overflow_count,
            max_expert_load: max_count,
            min_expert_load: min_count,
            mean_expert_load: mean,
            balance_score: balance,
        }
    }
}

/// 专家利用率统计
#[derive(Debug, Clone, Copy)]
pub struct ExpertUtilizationStats {
    /// 总 token 数
    pub total_tokens: usize,
    /// 总专家分配次数 (含多专家分配)
    pub total_expert_assignments: usize,
    /// 溢出 token 数
    pub overflow_count: usize,
    /// 最繁忙专家的 token 数
    pub max_expert_load: usize,
    /// 最空闲专家的 token 数
    pub min_expert_load: usize,
    /// 平均专家 token 数
    pub mean_expert_load: f32,
    /// 均衡度评分 (0.0-1.0)
    pub balance_score: f32,
}

/// Top-K 选择结果 (专家索引 + 权重)
pub fn topk_with_weights(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let softmax_probs = softmax(logits);

    let mut indexed: Vec<(usize, f32)> = softmax_probs.into_iter().enumerate().collect();
    // 按概率降序排列，取前 K 个
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);

    // 重新归一化权重 (使权重和为 1.0)
    let sum: f32 = indexed.iter().map(|(_, w)| *w).sum();
    if sum > 0.0 {
        for (_, w) in indexed.iter_mut() {
            *w /= sum;
        }
    }

    indexed
}

/// Top-K 索引选择 (无权重)
pub fn topk_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(k).map(|(idx, _)| *idx).collect()
}

/// Softmax 概率计算
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        exps.iter().map(|&e| e / sum).collect()
    } else {
        vec![1.0 / logits.len() as f32; logits.len()]
    }
}

/// MoE dispatch: 加权混合专家输出 (生产级)
///
/// §15.1: 输入 gate_logits 计算路由权重，对 expert_outputs 进行加权混合。
pub fn moe_dispatch(
    input: &[f32],
    gate_logits: &[f32],
    expert_outputs: &[Vec<f32>],
    top_k: usize,
) -> Vec<f32> {
    let topk = topk_with_weights(gate_logits, top_k);
    let mut output = vec![0.0f32; input.len()];

    for (expert_idx, weight) in topk {
        if expert_idx < expert_outputs.len() {
            for (i, &val) in expert_outputs[expert_idx].iter().enumerate() {
                if i < output.len() {
                    output[i] += val * weight;
                }
            }
        }
    }

    output
}

/// 专家负载均衡器
///
/// §15.1: 跟踪专家负载历史，用于动态容量调整。
/// 周期性地收集专家命中率统计，为 Hot JMP Patching (§14.4) 提供决策依据。
#[derive(Debug, Clone)]
pub struct ExpertLoadBalancer {
    /// 配置
    config: ExpertRouteConfig,
    /// 历史负载统计 (expert → hit_count)
    hit_history: Vec<u64>,
    /// 总路由次数
    total_routes: u64,
    /// 冷专家阈值 (命中率 < 此值视为冷专家)
    cold_threshold: f64,
}

impl ExpertLoadBalancer {
    /// 创建新的负载均衡器
    pub fn new(config: ExpertRouteConfig) -> Self {
        let num_experts = config.num_experts;
        Self {
            config,
            hit_history: vec![0; num_experts],
            total_routes: 0,
            cold_threshold: 0.001, // < 0.1% 命中率
        }
    }

    /// 记录一次路由结果
    pub fn record_route(&mut self, route_table: &ExpertRouteTable) {
        for route in &route_table.token_routes {
            for &expert_idx in &route.expert_indices {
                if expert_idx < self.hit_history.len() {
                    self.hit_history[expert_idx] += 1;
                }
            }
            self.total_routes += 1;
        }
    }

    /// 获取冷专家列表 (命中率低于阈值)
    ///
    /// §14.4 / §15.4: 供 Hot JMP Patching 和专家封杀决策使用
    pub fn cold_experts(&self) -> Vec<(usize, f64)> {
        if self.total_routes == 0 {
            return Vec::new();
        }

        self.hit_history
            .iter()
            .enumerate()
            .filter_map(|(idx, &hits)| {
                let rate = hits as f64 / self.total_routes as f64;
                if rate < self.cold_threshold {
                    Some((idx, rate))
                } else {
                    None
                }
            })
            .collect()
    }

    /// 获取热专家列表 (命中率最高)
    pub fn hot_experts(&self, top_n: usize) -> Vec<(usize, f64)> {
        let mut experts: Vec<(usize, f64)> = self
            .hit_history
            .iter()
            .enumerate()
            .map(|(idx, &hits)| {
                let rate = if self.total_routes > 0 {
                    hits as f64 / self.total_routes as f64
                } else {
                    0.0
                };
                (idx, rate)
            })
            .collect();

        experts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        experts.truncate(top_n);
        experts
    }

    /// 建议容量调整: 根据历史负载动态调整各专家容量
    pub fn suggest_capacity_adjustment(&self, base_capacity: usize) -> Vec<usize> {
        if self.total_routes == 0 {
            return vec![base_capacity; self.config.num_experts];
        }

        let total_hits: u64 = self.hit_history.iter().sum();
        if total_hits == 0 {
            return vec![base_capacity; self.config.num_experts];
        }

        self.hit_history
            .iter()
            .map(|&hits| {
                let ratio = hits as f64 / total_hits as f64;
                let adjusted = (base_capacity as f64 * ratio * self.config.num_experts as f64).ceil() as usize;
                // 保证最小容量为 1
                adjusted.max(1)
            })
            .collect()
    }

    /// 重置统计
    pub fn reset(&mut self) {
        self.hit_history.fill(0);
        self.total_routes = 0;
    }

    /// 获取各专家命中率
    pub fn hit_rates(&self) -> Vec<f64> {
        if self.total_routes == 0 {
            return vec![0.0; self.config.num_experts];
        }
        self.hit_history
            .iter()
            .map(|&hits| hits as f64 / self.total_routes as f64)
            .collect()
    }

    /// 获取配置引用
    pub fn config(&self) -> &ExpertRouteConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topk_indices() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let indices = topk_indices(&logits, 2);
        assert_eq!(indices, vec![3, 1]);
    }

    #[test]
    fn test_topk_with_weights_normalization() {
        let logits = vec![1.0, 2.0, 3.0];
        let topk = topk_with_weights(&logits, 2);
        assert_eq!(topk.len(), 2);
        // 权重和应接近 1.0 (归一化后)
        let sum: f32 = topk.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 0.01, "weights should sum to ~1.0, got {}", sum);
        // 最高概率的应该是 index 2
        assert_eq!(topk[0].0, 2);
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        assert_eq!(probs.len(), 3);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax should sum to 1.0, got {}", sum);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_empty() {
        let probs = softmax(&[]);
        assert!(probs.is_empty());
    }

    #[test]
    fn test_expert_route_table_basic() {
        let config = ExpertRouteConfig::new(4, 2);
        let gate_logits = vec![
            vec![0.1, 0.2, 0.7, 0.0], // token 0: experts 2, 1
            vec![0.5, 0.0, 0.0, 0.5], // token 1: experts 0, 3 (tie)
            vec![0.0, 0.0, 0.0, 1.0], // token 2: expert 3
        ];

        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);

        assert_eq!(route_table.token_routes.len(), 3);
        // 每个 token 应有 2 个专家 (top_k=2)
        for route in &route_table.token_routes {
            assert!(!route.expert_indices.is_empty());
        }
    }

    #[test]
    fn test_expert_route_table_capacity() {
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 0.5, // 每个 expert 容量 = ceil(0.5 * 4 / 2) = 1
            ..ExpertRouteConfig::default()
        };

        // 4 个 token 都想用 expert 0
        let gate_logits = vec![
            vec![10.0, 0.0], // → expert 0
            vec![10.0, 0.0], // → expert 0 (overflow → expert 1)
            vec![10.0, 0.0], // → expert 0 (overflow → expert 1)
            vec![10.0, 0.0], // → expert 0 (overflow → expert 1)
        ];

        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);

        // 应该有溢出
        assert!(route_table.overflow_count > 0, "expected overflow with tight capacity");
        // 但所有 token 都应被分配
        for route in &route_table.token_routes {
            assert!(!route.expert_indices.is_empty());
        }
    }

    #[test]
    fn test_expert_capacity_calculation() {
        let config = ExpertRouteConfig::new(8, 2);
        // capacity = ceil(1.25 * 128 / 8) = 20
        assert_eq!(config.expert_capacity(128), 20);
    }

    #[test]
    fn test_tokens_for_expert() {
        let config = ExpertRouteConfig::new(4, 1);
        // 使用极端 logits 确保只路由到目标专家
        let gate_logits = vec![
            vec![0.0, 0.0, 100.0, 0.0], // token 0 → expert 2 (压倒性优势)
            vec![0.0, 100.0, 0.0, 0.0],  // token 1 → expert 1
            vec![0.0, 0.0, 100.0, 0.0],  // token 2 → expert 2
        ];

        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let tokens_for_2 = route_table.tokens_for_expert(2);

        // top_k=1, 每个 token 选 1 个专家
        // token 0 → expert 2, token 2 → expert 2
        assert!(tokens_for_2.len() >= 1, "expected at least 1 token for expert 2, got {}", tokens_for_2.len());
        // 验证 token 0 在列表中
        let token_indices: Vec<usize> = tokens_for_2.iter().map(|(t, _)| *t).collect();
        assert!(token_indices.contains(&0), "token 0 should route to expert 2");
    }

    #[test]
    fn test_utilization_stats() {
        let config = ExpertRouteConfig::new(4, 2);
        let gate_logits = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];

        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let stats = route_table.utilization_stats();

        assert_eq!(stats.total_tokens, 2);
        assert!(stats.total_expert_assignments >= 2);
        assert!(stats.balance_score >= 0.0 && stats.balance_score <= 1.0);
    }

    #[test]
    fn test_load_balancer() {
        // top_k=1, 高 capacity_factor 确保所有 token 能路由到同一专家
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 10.0, // 大容量，允许所有 token 路由到同一专家
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config);

        // 模拟多次路由: 所有 token 路由到 expert 0
        for _ in 0..10 {
            let gate_logits = vec![
                vec![100.0, 0.0, 0.0, 0.0], // expert 0 压倒性
                vec![100.0, 0.0, 0.0, 0.0],
            ];
            let config = balancer.config().clone();
            let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
            balancer.record_route(&route_table);
        }

        // Expert 0 应该是热专家
        let hot = balancer.hot_experts(1);
        assert_eq!(hot[0].0, 0);
        assert!(hot[0].1 > 0.5, "expert 0 rate = {:.4}, expected > 0.5", hot[0].1);

        // Experts 2-3 应该是冷专家
        let cold = balancer.cold_experts();
        assert!(!cold.is_empty());
    }

    #[test]
    fn test_load_balancer_capacity_suggestion() {
        // top_k=1, 高 capacity_factor 确保负载不均匀
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 10.0, // 大容量，允许所有 token 路由到同一专家
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config);

        // 模拟不均匀负载: 所有 token 路由到 expert 0
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0, 0.0],
        ];
        let route_config = balancer.config().clone();
        let route_table = ExpertRouteTable::from_gate_logits(route_config, &gate_logits);
        balancer.record_route(&route_table);

        let suggested = balancer.suggest_capacity_adjustment(10);
        assert_eq!(suggested.len(), 4);
        // Expert 0 应该获得更大容量
        assert!(suggested[0] > suggested[1], "expert 0 capacity ({}) should > expert 1 ({})", suggested[0], suggested[1]);
    }

    #[test]
    fn test_moe_dispatch() {
        let input = vec![1.0, 2.0];
        let gate_logits = vec![0.0, 1.0, 0.5];
        let expert_outputs = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![1.5, 1.5],
        ];

        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 2);
        assert_eq!(output.len(), 2);
        // Expert 1 权重最高 (softmax(1.0) > softmax(0.5))
        assert!(output[0] > 1.5 && output[0] < 2.0);
    }

    #[test]
    fn test_load_balance_loss() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 2,
            load_balance_loss: true,
            load_balance_lambda: 0.01,
            ..ExpertRouteConfig::default()
        };

        let gate_logits = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];

        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let loss = route_table.load_balance_loss(&gate_logits);

        assert!(loss >= 0.0, "loss should be non-negative");
    }

    #[test]
    fn test_load_balancer_reset() {
        let config = ExpertRouteConfig::new(4, 2);
        let mut balancer = ExpertLoadBalancer::new(config);

        let gate_logits = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let route_config = balancer.config().clone();
        let route_table = ExpertRouteTable::from_gate_logits(route_config, &gate_logits);
        balancer.record_route(&route_table);
        assert!(balancer.total_routes > 0);

        balancer.reset();
        assert_eq!(balancer.total_routes, 0);
        let rates = balancer.hit_rates();
        assert!(rates.iter().all(|&r| r == 0.0));
    }
}
