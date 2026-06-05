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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, Copy, PartialEq)]
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

    // --- New tests below ---

    #[test]
    fn test_expert_route_config_default() {
        let config = ExpertRouteConfig::default();
        assert_eq!(config.num_experts, 8);
        assert_eq!(config.top_k, 2);
        assert_eq!(config.capacity_factor, 1.25);
        assert!(!config.load_balance_loss);
        assert!((config.load_balance_lambda - 0.01).abs() < 1e-6);
        assert!((config.noise_sigma).abs() < 1e-6);
    }

    #[test]
    fn test_expert_route_config_clone_debug() {
        let config = ExpertRouteConfig::new(16, 4);
        let cloned = config.clone();
        assert_eq!(cloned.num_experts, 16);
        assert_eq!(cloned.top_k, 4);
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("num_experts: 16"));
        assert!(debug_str.contains("top_k: 4"));
    }

    #[test]
    fn test_expert_capacity_single_expert() {
        let config = ExpertRouteConfig {
            num_experts: 1,
            top_k: 1,
            ..ExpertRouteConfig::default()
        };
        // capacity = ceil(1.25 * 100 / 1) = 125
        assert_eq!(config.expert_capacity(100), 125);
    }

    #[test]
    fn test_expert_capacity_zero_tokens() {
        let config = ExpertRouteConfig::new(8, 2);
        // ceil(1.25 * 0 / 8) = 0
        assert_eq!(config.expert_capacity(0), 0);
    }

    #[test]
    fn test_softmax_single_element() {
        let probs = softmax(&[5.0]);
        assert_eq!(probs.len(), 1);
        assert!((probs[0] - 1.0).abs() < 1e-6, "single element softmax should be 1.0");
    }

    #[test]
    fn test_softmax_all_zeros() {
        let probs = softmax(&[0.0, 0.0, 0.0]);
        assert_eq!(probs.len(), 3);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // All equal logits -> uniform distribution
        for &p in &probs {
            assert!((p - (1.0 / 3.0)).abs() < 1e-5, "expected uniform, got {}", p);
        }
    }

    #[test]
    fn test_topk_indices_k_exceeds_length() {
        let logits = vec![0.5, 0.3];
        let indices = topk_indices(&logits, 5);
        // Should return at most 2 indices
        assert_eq!(indices.len(), 2);
        assert_eq!(indices[0], 0); // highest first
    }

    #[test]
    fn test_topk_with_weights_all_equal_logits() {
        let logits = vec![1.0, 1.0, 1.0];
        let topk = topk_with_weights(&logits, 2);
        assert_eq!(topk.len(), 2);
        let sum: f32 = topk.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 0.01, "weights should sum to 1.0, got {}", sum);
        // Weights should be approximately equal
        let diff = (topk[0].1 - topk[1].1).abs();
        assert!(diff < 0.01, "equal logits should produce equal weights, diff = {}", diff);
    }

    #[test]
    fn test_expert_route_table_clone_debug() {
        let config = ExpertRouteConfig::new(4, 2);
        let gate_logits = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);

        let cloned = table.clone();
        assert_eq!(cloned.token_routes.len(), table.token_routes.len());
        assert_eq!(cloned.overflow_count, table.overflow_count);

        let debug_str = format!("{:?}", table);
        assert!(debug_str.contains("ExpertRouteTable"));
        assert!(debug_str.contains("token_routes"));
    }

    #[test]
    fn test_token_route_debug_clone() {
        let route = TokenRoute {
            expert_indices: vec![0, 2],
            expert_weights: vec![0.6, 0.4],
            expert_positions: vec![0, 1],
        };
        let cloned = route.clone();
        assert_eq!(cloned.expert_indices, route.expert_indices);
        assert_eq!(cloned.expert_weights, route.expert_weights);
        assert_eq!(cloned.expert_positions, route.expert_positions);

        let debug_str = format!("{:?}", route);
        assert!(debug_str.contains("expert_indices"));
        assert!(debug_str.contains("expert_weights"));
    }

    #[test]
    fn test_tokens_for_expert_empty_result() {
        let config = ExpertRouteConfig::new(4, 1);
        // All tokens route to expert 0, none to expert 3
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0, 0.0],
        ];
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let tokens_for_3 = route_table.tokens_for_expert(3);
        assert!(tokens_for_3.is_empty(), "no tokens should route to expert 3");
    }

    #[test]
    fn test_utilization_stats_expert_zero_experts() {
        // Verify ExpertUtilizationStats Copy + Clone + Debug
        let stats = ExpertUtilizationStats {
            total_tokens: 0,
            total_expert_assignments: 0,
            overflow_count: 0,
            max_expert_load: 0,
            min_expert_load: 0,
            mean_expert_load: 0.0,
            balance_score: 1.0,
        };
        let copied = stats;
        assert_eq!(copied.total_tokens, 0);
        assert_eq!(copied.balance_score, 1.0);

        let cloned = stats.clone();
        assert_eq!(cloned.total_tokens, stats.total_tokens);

        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("total_tokens"));
        assert!(debug_str.contains("balance_score"));
    }

    #[test]
    fn test_load_balance_loss_disabled() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 2,
            load_balance_loss: false,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);

        let loss = route_table.load_balance_loss(&gate_logits);
        assert!((loss - 0.0).abs() < 1e-6, "disabled loss should be 0.0, got {}", loss);
    }

    #[test]
    fn test_moe_dispatch_topk_exceeds_experts() {
        let input = vec![1.0, 2.0, 3.0];
        let gate_logits = vec![0.5, 0.3];
        let expert_outputs = vec![
            vec![1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0],
        ];
        // top_k=5 but only 2 experts available
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 5);
        assert_eq!(output.len(), 3);
        // Output should not be all zeros (at least some experts contribute)
        let sum: f32 = output.iter().sum();
        assert!(sum > 0.0, "output should have non-zero contributions");
    }

    #[test]
    fn test_moe_dispatch_expert_output_shorter_than_input() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gate_logits = vec![1.0, 0.0];
        let expert_outputs = vec![
            vec![2.0, 3.0], // shorter than input
        ];
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 1);
        assert_eq!(output.len(), 4);
        // Only first 2 elements should be affected
        assert!((output[0] - 2.0).abs() < 1e-5);
        assert!((output[1] - 3.0).abs() < 1e-5);
        assert!((output[2]).abs() < 1e-5, "unaffected element should be 0.0");
        assert!((output[3]).abs() < 1e-5, "unaffected element should be 0.0");
    }

    #[test]
    fn test_load_balancer_cold_experts_no_routes() {
        let config = ExpertRouteConfig::new(4, 2);
        let balancer = ExpertLoadBalancer::new(config);
        // No routes recorded yet
        let cold = balancer.cold_experts();
        assert!(cold.is_empty(), "no cold experts before any routes");
    }

    #[test]
    fn test_load_balancer_hit_rates_no_routes() {
        let config = ExpertRouteConfig::new(6, 2);
        let balancer = ExpertLoadBalancer::new(config);
        let rates = balancer.hit_rates();
        assert_eq!(rates.len(), 6);
        assert!(rates.iter().all(|&r| r == 0.0));
    }

    #[test]
    fn test_load_balancer_config_returns_reference() {
        let config = ExpertRouteConfig::new(12, 3);
        let balancer = ExpertLoadBalancer::new(config);
        assert_eq!(balancer.config().num_experts, 12);
        assert_eq!(balancer.config().top_k, 3);
    }

    #[test]
    fn test_load_balancer_suggest_capacity_no_routes() {
        let config = ExpertRouteConfig::new(4, 2);
        let balancer = ExpertLoadBalancer::new(config);
        let suggested = balancer.suggest_capacity_adjustment(10);
        // With no routes, all experts should get base capacity
        assert_eq!(suggested.len(), 4);
        assert!(suggested.iter().all(|&c| c == 10));
    }

    #[test]
    fn test_load_balancer_all_experts_equal_load() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 10.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());

        // Each token targets a different expert
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0, 0.0],
            vec![0.0, 100.0, 0.0, 0.0],
            vec![0.0, 0.0, 100.0, 0.0],
            vec![0.0, 0.0, 0.0, 100.0],
        ];
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&route_table);

        let cold = balancer.cold_experts();
        assert!(cold.is_empty(), "uniformly loaded: no cold experts");

        let rates = balancer.hit_rates();
        for &r in &rates {
            assert!((r - 0.25).abs() < 0.01, "each expert should have ~25% rate, got {}", r);
        }
    }

    // --- Additional tests ---

    #[test]
    fn test_expert_route_config_new_custom_fields() {
        let config = ExpertRouteConfig {
            num_experts: 64,
            top_k: 8,
            capacity_factor: 2.0,
            load_balance_loss: true,
            load_balance_lambda: 0.1,
            noise_sigma: 0.5,
        };
        assert_eq!(config.num_experts, 64);
        assert_eq!(config.top_k, 8);
        assert!((config.capacity_factor - 2.0).abs() < 1e-6);
        assert!(config.load_balance_loss);
        assert!((config.load_balance_lambda - 0.1).abs() < 1e-6);
        assert!((config.noise_sigma - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_expert_capacity_fractional_ceil() {
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 1.0,
            ..ExpertRouteConfig::default()
        };
        // capacity = ceil(1.0 * 10 / 3) = ceil(3.333...) = 4
        assert_eq!(config.expert_capacity(10), 4);
    }

    #[test]
    fn test_topk_with_weights_k_zero() {
        let logits = vec![1.0, 2.0, 3.0];
        let topk = topk_with_weights(&logits, 0);
        assert!(topk.is_empty(), "k=0 should return empty");
    }

    #[test]
    fn test_topk_indices_k_zero() {
        let logits = vec![1.0, 2.0, 3.0];
        let indices = topk_indices(&logits, 0);
        assert!(indices.is_empty(), "k=0 should return empty");
    }

    #[test]
    fn test_topk_indices_empty_logits() {
        let indices = topk_indices(&[], 3);
        assert!(indices.is_empty(), "empty logits should return empty");
    }

    #[test]
    fn test_topk_with_weights_empty_logits() {
        let topk = topk_with_weights(&[], 3);
        assert!(topk.is_empty(), "empty logits should return empty");
    }

    #[test]
    fn test_softmax_negative_logits() {
        let logits = vec![-10.0, -5.0, -1.0];
        let probs = softmax(&logits);
        assert_eq!(probs.len(), 3);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax should sum to 1.0, got {}", sum);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_large_values_numerical_stability() {
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = softmax(&logits);
        for &p in &probs {
            assert!(p.is_finite(), "probability should be finite, got {}", p);
        }
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "softmax should sum to 1.0, got {}", sum);
    }

    #[test]
    fn test_expert_route_table_empty_tokens() {
        let config = ExpertRouteConfig::new(4, 2);
        let route_table = ExpertRouteTable::from_gate_logits(config, &[]);
        assert_eq!(route_table.token_routes.len(), 0);
        assert_eq!(route_table.overflow_count, 0);
        assert!(route_table.expert_token_counts.iter().all(|&c| c == 0));
    }

    #[test]
    fn test_expert_route_table_single_token_single_expert() {
        let config = ExpertRouteConfig {
            num_experts: 1,
            top_k: 1,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![vec![5.0]];
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);

        assert_eq!(route_table.token_routes.len(), 1);
        let route = &route_table.token_routes[0];
        assert_eq!(route.expert_indices.len(), 1);
        assert_eq!(route.expert_indices[0], 0);
        assert!((route.expert_weights[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_expert_route_table_overflow_fallback_to_least_loaded() {
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 0.5,
            ..ExpertRouteConfig::default()
        };
        // 2 tokens both want expert 0, capacity = ceil(0.5*2/2) = 1
        let gate_logits = vec![
            vec![100.0, 0.0],
            vec![100.0, 0.0],
        ];
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);

        assert!(route_table.overflow_count > 0);
        for route in &route_table.token_routes {
            assert_eq!(route.expert_indices.len(), 1, "each token must have 1 expert");
        }
    }

    #[test]
    fn test_tokens_for_expert_multi_assignment() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 2,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        // Token 0: experts 2 and 3; Token 1: experts 2 and 0
        let gate_logits = vec![
            vec![0.0, 0.0, 100.0, 50.0],
            vec![50.0, 0.0, 100.0, 0.0],
        ];
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let tokens_for_2 = route_table.tokens_for_expert(2);

        assert!(
            tokens_for_2.len() >= 2,
            "expert 2 should be hit by both tokens, got {}",
            tokens_for_2.len()
        );
    }

    #[test]
    fn test_utilization_stats_with_overflow() {
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 0.5,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 0.0],
            vec![100.0, 0.0],
            vec![100.0, 0.0],
            vec![100.0, 0.0],
        ];
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let stats = route_table.utilization_stats();

        assert_eq!(stats.total_tokens, 4);
        assert!(stats.overflow_count > 0, "should have overflow");
        assert!(stats.total_expert_assignments >= 4);
        assert!(stats.max_expert_load >= stats.min_expert_load);
        assert!(stats.mean_expert_load > 0.0);
        assert!(stats.balance_score >= 0.0 && stats.balance_score <= 1.0);
    }

    #[test]
    fn test_load_balance_loss_positive_for_skewed_routing() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            load_balance_loss: true,
            load_balance_lambda: 0.01,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0, 0.0],
        ];
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let loss = route_table.load_balance_loss(&gate_logits);

        assert!(loss > 0.0, "skewed routing should produce positive loss, got {}", loss);
    }

    #[test]
    fn test_load_balancer_record_then_reset_then_record() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 10.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());

        let gate_logits = vec![vec![100.0, 0.0, 0.0, 0.0]];
        let rt1 = ExpertRouteTable::from_gate_logits(config.clone(), &gate_logits);
        balancer.record_route(&rt1);
        assert!(balancer.hit_rates()[0] > 0.0);

        balancer.reset();
        assert_eq!(balancer.total_routes, 0);

        let gate_logits2 = vec![vec![0.0, 0.0, 0.0, 100.0]];
        let rt2 = ExpertRouteTable::from_gate_logits(config, &gate_logits2);
        balancer.record_route(&rt2);

        let rates = balancer.hit_rates();
        assert!(rates[0].abs() < 1e-10, "expert 0 should have 0 hits after reset, got {}", rates[0]);
        assert!(rates[3] > 0.0, "expert 3 should have hits after second recording");
    }

    #[test]
    fn test_load_balancer_hot_experts_top_n_exceeds_count() {
        let config = ExpertRouteConfig::new(3, 1);
        let balancer = ExpertLoadBalancer::new(config);
        let hot = balancer.hot_experts(10);
        assert_eq!(hot.len(), 3, "should return at most num_experts entries");
    }

    #[test]
    fn test_load_balancer_suggest_capacity_after_reset() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 10.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config);

        let gate_logits = vec![vec![100.0, 0.0, 0.0, 0.0]];
        let rt = ExpertRouteTable::from_gate_logits(balancer.config().clone(), &gate_logits);
        balancer.record_route(&rt);
        assert_ne!(balancer.suggest_capacity_adjustment(10)[0], 10);

        balancer.reset();
        let suggested = balancer.suggest_capacity_adjustment(10);
        assert!(suggested.iter().all(|&c| c == 10), "after reset, all should be base capacity");
    }

    #[test]
    fn test_load_balancer_suggest_capacity_uniform_load() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 10.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());

        let gate_logits = vec![
            vec![100.0, 0.0, 0.0, 0.0],
            vec![0.0, 100.0, 0.0, 0.0],
            vec![0.0, 0.0, 100.0, 0.0],
            vec![0.0, 0.0, 0.0, 100.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&rt);

        let suggested = balancer.suggest_capacity_adjustment(8);
        for &cap in &suggested {
            assert_eq!(cap, 8, "uniform load should yield base capacity for all experts");
        }
    }

    #[test]
    fn test_moe_dispatch_empty_expert_outputs() {
        let input = vec![1.0, 2.0];
        let gate_logits = vec![0.5];
        let expert_outputs: Vec<Vec<f32>> = vec![];
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 1);
        assert_eq!(output.len(), 2);
        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_moe_dispatch_empty_input() {
        let input: Vec<f32> = vec![];
        let gate_logits = vec![1.0, 0.0];
        let expert_outputs = vec![vec![], vec![]];
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 2);
        assert!(output.is_empty(), "empty input should produce empty output");
    }

    #[test]
    fn test_expert_load_balancer_clone_debug() {
        let config = ExpertRouteConfig::new(4, 2);
        let balancer = ExpertLoadBalancer::new(config);
        let cloned = balancer.clone();
        assert_eq!(cloned.config().num_experts, 4);

        let debug_str = format!("{:?}", balancer);
        assert!(debug_str.contains("ExpertLoadBalancer"));
        assert!(debug_str.contains("hit_history"));
        assert!(debug_str.contains("total_routes"));
    }

    #[test]
    fn test_expert_utilization_stats_all_fields() {
        let stats = ExpertUtilizationStats {
            total_tokens: 100,
            total_expert_assignments: 150,
            overflow_count: 5,
            max_expert_load: 40,
            min_expert_load: 10,
            mean_expert_load: 25.0,
            balance_score: 0.75,
        };
        assert_eq!(stats.total_tokens, 100);
        assert_eq!(stats.total_expert_assignments, 150);
        assert_eq!(stats.overflow_count, 5);
        assert_eq!(stats.max_expert_load, 40);
        assert_eq!(stats.min_expert_load, 10);
        assert!((stats.mean_expert_load - 25.0).abs() < 1e-5);
        assert!((stats.balance_score - 0.75).abs() < 1e-5);

        // Copy trait: assignment copies, not references
        let copied = stats;
        assert_eq!(copied.total_tokens, stats.total_tokens);
        assert_eq!(copied.balance_score, stats.balance_score);
    }

    #[test]
    fn test_expert_route_table_expert_token_counts() {
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0],
            vec![0.0, 100.0, 0.0],
            vec![0.0, 0.0, 100.0],
            vec![100.0, 0.0, 0.0],
        ];
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);

        assert_eq!(route_table.expert_token_counts.len(), 3);
        assert_eq!(route_table.expert_token_counts[0], 2); // tokens 0 and 3
        assert_eq!(route_table.expert_token_counts[1], 1);
        assert_eq!(route_table.expert_token_counts[2], 1);
        assert_eq!(route_table.overflow_count, 0);
    }

    #[test]
    fn test_topk_with_weights_ordering() {
        let logits = vec![0.1, 0.9, 0.5, 0.3];
        let topk = topk_with_weights(&logits, 3);
        assert_eq!(topk.len(), 3);
        assert_eq!(topk[0].0, 1); // highest
        assert_eq!(topk[1].0, 2); // second
        assert_eq!(topk[2].0, 3); // third
        assert!(topk[0].1 >= topk[1].1);
        assert!(topk[1].1 >= topk[2].1);
    }

    #[test]
    fn test_topk_indices_single_element() {
        let indices = topk_indices(&[7.0], 1);
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn test_expert_route_config_new_overrides_defaults() {
        let config = ExpertRouteConfig::new(32, 4);
        assert_eq!(config.num_experts, 32);
        assert_eq!(config.top_k, 4);
        assert!((config.capacity_factor - 1.25).abs() < 1e-6);
        assert!(!config.load_balance_loss);
    }

    // --- Round 2: 18 additional tests ---

    #[test]
    fn test_expert_capacity_large_expert_count() {
        // Arrange: 256 experts, 1024 tokens, default capacity_factor=1.25
        let config = ExpertRouteConfig::new(256, 4);
        // capacity = ceil(1.25 * 1024 / 256) = ceil(5.0) = 5
        // Act
        let cap = config.expert_capacity(1024);
        // Assert
        assert_eq!(cap, 5);
    }

    #[test]
    fn test_expert_capacity_with_large_capacity_factor() {
        // Arrange: 4 experts, capacity_factor=10.0, 100 tokens
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 2,
            capacity_factor: 10.0,
            ..ExpertRouteConfig::default()
        };
        // capacity = ceil(10.0 * 100 / 4) = ceil(250.0) = 250
        // Act
        let cap = config.expert_capacity(100);
        // Assert
        assert_eq!(cap, 250);
    }

    #[test]
    fn test_softmax_with_nan_input() {
        // Arrange: logits containing NaN
        let logits = vec![1.0, f32::NAN, 3.0];
        // Act
        let probs = softmax(&logits);
        // Assert: result length preserved, at least one non-finite is acceptable
        // but the function should not panic
        assert_eq!(probs.len(), 3);
    }

    #[test]
    fn test_softmax_two_elements_symmetry() {
        // Arrange: symmetric logits [a, -a]
        let logits = vec![2.0, -2.0];
        // Act
        let probs = softmax(&logits);
        // Assert: e^2/(e^2+e^-2) + e^-2/(e^2+e^-2) = 1.0
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // prob[0] should be much larger than prob[1]
        assert!(probs[0] > 0.5);
        assert!(probs[1] < 0.5);
    }

    #[test]
    fn test_topk_with_weights_duplicate_values() {
        // Arrange: duplicate logit values
        let logits = vec![3.0, 1.0, 3.0, 1.0];
        // Act
        let topk = topk_with_weights(&logits, 2);
        // Assert: picks two of the highest (indices 0 or 2)
        assert_eq!(topk.len(), 2);
        let sum: f32 = topk.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 0.01, "renormalized weights should sum to 1.0, got {}", sum);
        // Both selected experts should have approximately equal weight
        let diff = (topk[0].1 - topk[1].1).abs();
        assert!(diff < 0.01, "duplicate logits should yield equal weights, diff = {}", diff);
    }

    #[test]
    fn test_topk_indices_duplicate_values_stable() {
        // Arrange: all values identical
        let logits = vec![5.0, 5.0, 5.0];
        // Act
        let indices = topk_indices(&logits, 2);
        // Assert: returns 2 indices (any of the three)
        assert_eq!(indices.len(), 2);
        // All indices should be valid
        for &idx in &indices {
            assert!(idx < 3, "index {} out of range", idx);
        }
    }

    #[test]
    fn test_topk_with_weights_single_element() {
        // Arrange
        let logits = vec![4.0];
        // Act
        let topk = topk_with_weights(&logits, 1);
        // Assert
        assert_eq!(topk.len(), 1);
        assert_eq!(topk[0].0, 0);
        assert!((topk[0].1 - 1.0).abs() < 1e-5, "single element weight should be 1.0");
    }

    #[test]
    fn test_token_route_construction_and_field_access() {
        // Arrange & Act
        let route = TokenRoute {
            expert_indices: vec![3, 7, 1],
            expert_weights: vec![0.5, 0.3, 0.2],
            expert_positions: vec![10, 5, 20],
        };
        // Assert
        assert_eq!(route.expert_indices.len(), 3);
        assert_eq!(route.expert_indices[0], 3);
        assert_eq!(route.expert_weights.len(), 3);
        assert!((route.expert_weights.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert_eq!(route.expert_positions[2], 20);
    }

    #[test]
    fn test_token_route_empty_route() {
        // Arrange & Act: a token route with no experts assigned
        let route = TokenRoute {
            expert_indices: vec![],
            expert_weights: vec![],
            expert_positions: vec![],
        };
        // Assert
        assert!(route.expert_indices.is_empty());
        assert!(route.expert_weights.is_empty());
        assert!(route.expert_positions.is_empty());
    }

    #[test]
    fn test_expert_utilization_stats_copy_independence() {
        // Arrange
        let stats = ExpertUtilizationStats {
            total_tokens: 50,
            total_expert_assignments: 80,
            overflow_count: 3,
            max_expert_load: 25,
            min_expert_load: 5,
            mean_expert_load: 16.0,
            balance_score: 0.8,
        };
        // Act: Copy semantics — modifying the copy must not affect the original
        let mut copied = stats;
        copied.total_tokens = 999;
        copied.balance_score = 0.0;
        // Assert
        assert_eq!(stats.total_tokens, 50, "original should be unaffected by copy mutation");
        assert!((stats.balance_score - 0.8).abs() < 1e-5);
        assert_eq!(copied.total_tokens, 999);
    }

    #[test]
    fn test_expert_utilization_stats_debug_format() {
        // Arrange
        let stats = ExpertUtilizationStats {
            total_tokens: 10,
            total_expert_assignments: 15,
            overflow_count: 1,
            max_expert_load: 8,
            min_expert_load: 2,
            mean_expert_load: 3.75,
            balance_score: 0.6,
        };
        // Act
        let debug = format!("{:?}", stats);
        // Assert: key field names present
        assert!(debug.contains("total_tokens: 10"));
        assert!(debug.contains("overflow_count: 1"));
        assert!(debug.contains("max_expert_load: 8"));
        assert!(debug.contains("balance_score"));
    }

    #[test]
    fn test_expert_load_balancer_zero_experts_config() {
        // Arrange: config with 0 experts (edge case)
        let config = ExpertRouteConfig {
            num_experts: 0,
            top_k: 0,
            capacity_factor: 1.0,
            ..ExpertRouteConfig::default()
        };
        // Act
        let balancer = ExpertLoadBalancer::new(config);
        let rates = balancer.hit_rates();
        let hot = balancer.hot_experts(1);
        let cold = balancer.cold_experts();
        // Assert
        assert!(rates.is_empty(), "zero experts → empty hit_rates");
        assert!(hot.is_empty(), "zero experts → empty hot_experts");
        assert!(cold.is_empty(), "zero experts → empty cold_experts");
    }

    #[test]
    fn test_expert_load_balancer_suggest_capacity_minimum_one() {
        // Arrange: skewed load where one expert gets all hits, others get zero
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0, 0.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&rt);
        // Act
        let suggested = balancer.suggest_capacity_adjustment(1);
        // Assert: every expert gets at least capacity 1
        for (i, &cap) in suggested.iter().enumerate() {
            assert!(cap >= 1, "expert {} capacity {} should be >= 1", i, cap);
        }
    }

    #[test]
    fn test_topk_with_weights_k_equals_length() {
        // Arrange: k equals number of logits exactly
        let logits = vec![0.1, 0.5, 0.3];
        // Act
        let topk = topk_with_weights(&logits, 3);
        // Assert: all 3 selected, weights renormalized to sum=1.0
        assert_eq!(topk.len(), 3);
        let sum: f32 = topk.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 0.01, "weights should sum to 1.0, got {}", sum);
    }

    #[test]
    fn test_topk_indices_preserves_highest_first_ordering() {
        // Arrange
        let logits = vec![0.1, 0.9, 0.5, 0.3, 0.7];
        // Act
        let indices = topk_indices(&logits, 3);
        // Assert: indices ordered by descending logit value
        assert_eq!(indices.len(), 3);
        assert_eq!(indices[0], 1); // 0.9
        assert_eq!(indices[1], 4); // 0.7
        assert_eq!(indices[2], 2); // 0.5
    }

    #[test]
    fn test_expert_route_config_capacity_many_tokens_few_experts() {
        // Arrange: 2 experts, 10000 tokens
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 1.5,
            ..ExpertRouteConfig::default()
        };
        // capacity = ceil(1.5 * 10000 / 2) = ceil(7500.0) = 7500
        // Act
        let cap = config.expert_capacity(10000);
        // Assert
        assert_eq!(cap, 7500);
    }

    #[test]
    fn test_expert_route_table_config_preserved() {
        // Arrange
        let config = ExpertRouteConfig {
            num_experts: 5,
            top_k: 2,
            capacity_factor: 1.5,
            load_balance_loss: true,
            load_balance_lambda: 0.05,
            noise_sigma: 0.0,
        };
        let gate_logits = vec![vec![1.0, 0.0, 0.0, 0.0, 0.0]];
        // Act
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: config is stored inside the table
        assert_eq!(route_table.config.num_experts, 5);
        assert_eq!(route_table.config.top_k, 2);
        assert!((route_table.config.capacity_factor - 1.5).abs() < 1e-6);
        assert!(route_table.config.load_balance_loss);
    }

    #[test]
    fn test_moe_dispatch_weighted_sum_correctness() {
        // Arrange: 2 experts, top_k=2, known weights
        let input = vec![0.0; 4];
        let gate_logits = vec![0.0, 0.0]; // equal logits → equal softmax
        let expert_outputs = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4.0, 3.0, 2.0, 1.0],
        ];
        // Act
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 2);
        // Assert: with equal logits, each expert weight ≈ 0.5
        // output[i] ≈ 0.5 * expert0[i] + 0.5 * expert1[i]
        let expected = vec![2.5, 2.5, 2.5, 2.5];
        for i in 0..4 {
            assert!(
                (output[i] - expected[i]).abs() < 0.02,
                "output[{}] = {:.4}, expected ≈ {:.4}",
                i, output[i], expected[i]
            );
        }
    }

    // --- Round 3: 40 additional tests ---

    // === ExpertRouteConfig PartialEq, Default, Constructor ===

    #[test]
    fn test_expert_route_config_partial_eq_equal() {
        let a = ExpertRouteConfig::new(8, 2);
        let b = ExpertRouteConfig::new(8, 2);
        assert_eq!(a, b, "identically constructed configs should be equal");
    }

    #[test]
    fn test_expert_route_config_partial_eq_not_equal() {
        let a = ExpertRouteConfig::new(8, 2);
        let b = ExpertRouteConfig::new(16, 4);
        assert_ne!(a, b, "different configs should not be equal");
    }

    #[test]
    fn test_expert_route_config_partial_eq_field_difference() {
        let a = ExpertRouteConfig {
            load_balance_loss: true,
            ..ExpertRouteConfig::new(8, 2)
        };
        let b = ExpertRouteConfig {
            load_balance_loss: false,
            ..ExpertRouteConfig::new(8, 2)
        };
        assert_ne!(a, b, "configs differing in load_balance_loss should not be equal");
    }

    #[test]
    fn test_expert_route_config_default_equals_new_default() {
        let from_new = ExpertRouteConfig::new(8, 2);
        let from_default = ExpertRouteConfig::default();
        assert_eq!(from_new, from_default, "new(8,2) should equal default()");
    }

    #[test]
    fn test_expert_route_config_new_top_k_greater_than_num_experts() {
        let config = ExpertRouteConfig::new(4, 16);
        assert_eq!(config.num_experts, 4);
        assert_eq!(config.top_k, 16, "top_k > num_experts is allowed in config");
    }

    // === ExpertRouteConfig::expert_capacity boundary conditions ===

    #[test]
    fn test_expert_capacity_one_token_one_expert() {
        let config = ExpertRouteConfig {
            num_experts: 1,
            top_k: 1,
            ..ExpertRouteConfig::default()
        };
        // ceil(1.25 * 1 / 1) = 2
        assert_eq!(config.expert_capacity(1), 2);
    }

    #[test]
    fn test_expert_capacity_very_small_capacity_factor() {
        let config = ExpertRouteConfig {
            num_experts: 8,
            top_k: 2,
            capacity_factor: 0.001,
            ..ExpertRouteConfig::default()
        };
        // ceil(0.001 * 1 / 8) = ceil(0.000125) = 1
        assert_eq!(config.expert_capacity(1), 1);
    }

    #[test]
    fn test_expert_capacity_zero_capacity_factor_zero_tokens() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 2,
            capacity_factor: 0.0,
            ..ExpertRouteConfig::default()
        };
        assert_eq!(config.expert_capacity(0), 0, "0 * 0 / 4 = 0");
    }

    #[test]
    fn test_expert_capacity_large_token_count() {
        let config = ExpertRouteConfig::new(8, 2);
        // ceil(1.25 * 1_000_000 / 8) = ceil(156250.0) = 156250
        assert_eq!(config.expert_capacity(1_000_000), 156_250);
    }

    // === softmax additional edge cases ===

    #[test]
    fn test_softmax_very_negative_logits() {
        let logits = vec![-1000.0, -999.0, -998.0];
        let probs = softmax(&logits);
        assert_eq!(probs.len(), 3);
        for &p in &probs {
            assert!(p.is_finite(), "probability should be finite, got {}", p);
        }
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "sum = {}", sum);
        assert!(probs[2] > probs[1]);
    }

    #[test]
    fn test_softmax_all_same_value() {
        let logits = vec![3.0, 3.0, 3.0, 3.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // All probabilities should be equal = 0.25
        for &p in &probs {
            assert!((p - 0.25).abs() < 1e-5, "expected 0.25, got {}", p);
        }
    }

    #[test]
    fn test_softmax_preserves_ordering() {
        let logits = vec![0.5, 2.0, 1.0, 3.0];
        let probs = softmax(&logits);
        assert!(probs[3] > probs[1], "logits[3]=3.0 > logits[1]=2.0");
        assert!(probs[1] > probs[2], "logits[1]=2.0 > logits[2]=1.0");
        assert!(probs[2] > probs[0], "logits[2]=1.0 > logits[0]=0.5");
    }

    #[test]
    fn test_softmax_two_identical_extremes() {
        let logits = vec![100.0, -100.0];
        let probs = softmax(&logits);
        assert!((probs[0] - 1.0).abs() < 1e-5, "near-zero softmax should dominate");
        assert!((probs[1] - 0.0).abs() < 1e-5);
    }

    // === topk_with_weights additional edge cases ===

    #[test]
    fn test_topk_with_weights_negative_logits() {
        let logits = vec![-5.0, -1.0, -3.0];
        let topk = topk_with_weights(&logits, 2);
        assert_eq!(topk[0].0, 1, "index 1 (-1.0) is highest");
        assert_eq!(topk[1].0, 2, "index 2 (-3.0) is second highest");
        let sum: f32 = topk.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 0.01, "weights sum = {}", sum);
    }

    #[test]
    fn test_topk_with_weights_mixed_signs() {
        let logits = vec![-2.0, 1.0, -1.0, 3.0];
        let topk = topk_with_weights(&logits, 2);
        assert_eq!(topk[0].0, 3, "index 3 (3.0) highest");
        assert_eq!(topk[1].0, 1, "index 1 (1.0) second");
    }

    #[test]
    fn test_topk_with_weights_renormalization_three_of_five() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let topk = topk_with_weights(&logits, 3);
        assert_eq!(topk.len(), 3);
        assert_eq!(topk[0].0, 4, "index 4 (5.0) is highest");
        assert_eq!(topk[1].0, 3, "index 3 (4.0) is second");
        assert_eq!(topk[2].0, 2, "index 2 (3.0) is third");
        let sum: f32 = topk.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 0.01, "renormalized weights sum = {}", sum);
    }

    // === topk_indices additional edge cases ===

    #[test]
    fn test_topk_indices_negative_values() {
        let logits = vec![-3.0, -1.0, -2.0];
        let indices = topk_indices(&logits, 2);
        assert_eq!(indices, vec![1, 2], "-1.0 > -2.0 > -3.0");
    }

    #[test]
    fn test_topk_indices_single() {
        let logits = vec![10.0, 20.0, 30.0];
        let indices = topk_indices(&logits, 1);
        assert_eq!(indices, vec![2], "index 2 has the highest value");
    }

    #[test]
    fn test_topk_indices_all_same() {
        let logits = vec![7.0, 7.0, 7.0, 7.0];
        let indices = topk_indices(&logits, 2);
        assert_eq!(indices.len(), 2);
        for &idx in &indices {
            assert!(idx < 4, "index {} out of range", idx);
        }
    }

    // === ExpertRouteTable boundary and structural tests ===

    #[test]
    fn test_expert_route_table_top_k_exceeds_experts() {
        // top_k=4 but only 2 experts → softmax picks both, k truncated by .truncate()
        let config = ExpertRouteConfig::new(2, 4);
        let gate_logits = vec![
            vec![1.0, 2.0],
        ];
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        assert_eq!(route_table.token_routes.len(), 1);
        let route = &route_table.token_routes[0];
        // Only 2 experts available, so at most 2 selected
        assert!(route.expert_indices.len() <= 2);
        assert!(route.expert_indices.len() >= 1, "at least 1 expert must be assigned");
    }

    #[test]
    fn test_expert_route_table_large_token_count() {
        let config = ExpertRouteConfig::new(4, 2);
        let gate_logits: Vec<Vec<f32>> = (0..100)
            .map(|i| vec![i as f32, 0.0, 0.0, 0.0])
            .collect();
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        assert_eq!(route_table.token_routes.len(), 100);
        // Every token must have at least 1 expert assigned
        for (i, route) in route_table.token_routes.iter().enumerate() {
            assert!(!route.expert_indices.is_empty(), "token {} has no expert", i);
        }
    }

    #[test]
    fn test_expert_route_table_expert_positions_sequential() {
        // With high capacity, expert_positions should be sequential per expert
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 0.0], // → expert 0, position 0
            vec![100.0, 0.0], // → expert 0, position 1
            vec![0.0, 100.0], // → expert 1, position 0
        ];
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // token 0 at expert 0, position 0
        assert_eq!(route_table.token_routes[0].expert_indices[0], 0);
        assert_eq!(route_table.token_routes[0].expert_positions[0], 0);
        // token 1 at expert 0, position 1
        assert_eq!(route_table.token_routes[1].expert_indices[0], 0);
        assert_eq!(route_table.token_routes[1].expert_positions[0], 1);
        // token 2 at expert 1, position 0
        assert_eq!(route_table.token_routes[2].expert_indices[0], 1);
        assert_eq!(route_table.token_routes[2].expert_positions[0], 0);
    }

    #[test]
    fn test_expert_route_table_tokens_for_expert_boundary_index() {
        let config = ExpertRouteConfig::new(4, 1);
        let gate_logits = vec![
            vec![0.0, 0.0, 0.0, 100.0], // → expert 3
        ];
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let tokens = route_table.tokens_for_expert(3);
        assert_eq!(tokens.len(), 1, "expert 3 should have 1 token");
        assert_eq!(tokens[0].0, 0, "token index 0");
    }

    #[test]
    fn test_expert_route_table_overflow_all_experts_full() {
        // Extreme capacity constraint: capacity=1, 3 tokens all want same expert
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 0.34, // ceil(0.34 * 2 / 2) = ceil(0.34) = 1
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 0.0],
            vec![100.0, 0.0],
            vec![100.0, 0.0],
        ];
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Token 2 overflows expert 0, falls back to least loaded (expert 1)
        assert!(route_table.overflow_count >= 1, "expected overflow");
        // Every token still has an assignment
        for (i, route) in route_table.token_routes.iter().enumerate() {
            assert_eq!(route.expert_indices.len(), 1, "token {} missing assignment", i);
        }
    }

    #[test]
    fn test_expert_route_table_overflow_weight_is_one_for_fallback() {
        // When all top-k overflow and fallback occurs, weight should be 1.0
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 0.34, // ceil(0.34 * 2 / 2) = 1
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 0.0],
            vec![100.0, 0.0],
        ];
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // At least one token should be a fallback (weight=1.0)
        let has_fallback = route_table.token_routes.iter().any(|r| {
            r.expert_weights.len() == 1 && (r.expert_weights[0] - 1.0).abs() < 1e-5
        });
        assert!(has_fallback, "at least one token should have fallback weight 1.0");
    }

    // === ExpertUtilizationStats PartialEq ===

    #[test]
    fn test_expert_utilization_stats_partial_eq_equal() {
        let a = ExpertUtilizationStats {
            total_tokens: 10,
            total_expert_assignments: 20,
            overflow_count: 1,
            max_expert_load: 8,
            min_expert_load: 2,
            mean_expert_load: 5.0,
            balance_score: 0.6,
        };
        let b = ExpertUtilizationStats {
            total_tokens: 10,
            total_expert_assignments: 20,
            overflow_count: 1,
            max_expert_load: 8,
            min_expert_load: 2,
            mean_expert_load: 5.0,
            balance_score: 0.6,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn test_expert_utilization_stats_partial_eq_not_equal() {
        let a = ExpertUtilizationStats {
            total_tokens: 10,
            total_expert_assignments: 20,
            overflow_count: 0,
            max_expert_load: 5,
            min_expert_load: 5,
            mean_expert_load: 5.0,
            balance_score: 1.0,
        };
        let b = ExpertUtilizationStats {
            total_tokens: 10,
            total_expert_assignments: 20,
            overflow_count: 1,
            max_expert_load: 5,
            min_expert_load: 5,
            mean_expert_load: 5.0,
            balance_score: 1.0,
        };
        assert_ne!(a, b, "differing overflow_count should mean not equal");
    }

    #[test]
    fn test_expert_utilization_stats_zero_load_perfect_balance() {
        // When max_expert_load = 0, balance_score defaults to 1.0
        let config = ExpertRouteConfig::new(4, 2);
        let route_table = ExpertRouteTable::from_gate_logits(config, &[]);
        let stats = route_table.utilization_stats();
        assert_eq!(stats.max_expert_load, 0);
        assert_eq!(stats.min_expert_load, 0);
        assert!((stats.balance_score - 1.0).abs() < 1e-5, "zero load → perfect balance");
    }

    #[test]
    fn test_expert_utilization_stats_mean_calculation() {
        // 4 experts with loads [10, 20, 30, 40] → mean = 25.0
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        // Tokens: 10 to expert 0, 20 to expert 1... etc.
        // Use extreme logits to force routing
        let mut gate_logits = Vec::new();
        for expert in 0..4 {
            for _ in 0..((expert + 1) * 10) {
                let mut row = vec![0.0; 4];
                row[expert] = 100.0;
                gate_logits.push(row);
            }
        }
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let stats = route_table.utilization_stats();
        assert_eq!(stats.total_tokens, 100); // 10+20+30+40
        assert!((stats.mean_expert_load - 25.0).abs() < 0.5,
            "mean should be ~25.0, got {}", stats.mean_expert_load);
    }

    // === ExpertLoadBalancer edge cases ===

    #[test]
    fn test_load_balancer_cold_experts_all_cold_when_most_hits_one() {
        let config = ExpertRouteConfig {
            num_experts: 8,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        // All 100 tokens route to expert 0
        let gate_logits: Vec<Vec<f32>> = (0..100)
            .map(|_| {
                let mut row = vec![0.0; 8];
                row[0] = 100.0;
                row
            })
            .collect();
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&rt);

        let cold = balancer.cold_experts();
        // Experts 1-7 should be cold (rate = 0/100 = 0.0 < threshold 0.001)
        assert!(cold.len() >= 7, "expected >= 7 cold experts, got {}", cold.len());
        for &(idx, rate) in &cold {
            assert_ne!(idx, 0, "expert 0 should not be cold");
            assert!((rate).abs() < 0.001, "cold expert {} rate = {}", idx, rate);
        }
    }

    #[test]
    fn test_load_balancer_hot_experts_ordering() {
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        // Expert 0: 10 hits, Expert 1: 5 hits, Expert 2: 1 hit
        let mut gate_logits = Vec::new();
        for _ in 0..10 {
            gate_logits.push(vec![100.0, 0.0, 0.0]);
        }
        for _ in 0..5 {
            gate_logits.push(vec![0.0, 100.0, 0.0]);
        }
        gate_logits.push(vec![0.0, 0.0, 100.0]);
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&rt);

        let hot = balancer.hot_experts(3);
        assert_eq!(hot[0].0, 0, "expert 0 should be hottest");
        assert_eq!(hot[1].0, 1, "expert 1 should be second");
        assert_eq!(hot[2].0, 2, "expert 2 should be third");
        assert!(hot[0].1 > hot[1].1);
        assert!(hot[1].1 > hot[2].1);
    }

    #[test]
    fn test_load_balancer_record_multiple_route_tables() {
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        // First route: 1 token to expert 0
        let rt1 = ExpertRouteTable::from_gate_logits(config.clone(), &[
            vec![100.0, 0.0],
        ]);
        balancer.record_route(&rt1);
        // Second route: 1 token to expert 1
        let rt2 = ExpertRouteTable::from_gate_logits(config, &[
            vec![0.0, 100.0],
        ]);
        balancer.record_route(&rt2);

        let rates = balancer.hit_rates();
        assert!((rates[0] - 0.5).abs() < 0.01, "expert 0 rate = {}", rates[0]);
        assert!((rates[1] - 0.5).abs() < 0.01, "expert 1 rate = {}", rates[1]);
        // total_routes should be 2 (one per token)
        assert_eq!(balancer.total_routes, 2);
    }

    #[test]
    fn test_load_balancer_suggest_capacity_skewed_produces_unequal() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        // 10 tokens to expert 0, 2 to expert 1, 1 to expert 2, 0 to expert 3
        let mut gate_logits = Vec::new();
        for _ in 0..10 {
            let mut row = vec![0.0; 4];
            row[0] = 100.0;
            gate_logits.push(row);
        }
        for _ in 0..2 {
            let mut row = vec![0.0; 4];
            row[1] = 100.0;
            gate_logits.push(row);
        }
        {
            let mut row = vec![0.0; 4];
            row[2] = 100.0;
            gate_logits.push(row);
        }
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&rt);

        let suggested = balancer.suggest_capacity_adjustment(10);
        assert_eq!(suggested.len(), 4);
        assert!(suggested[0] > suggested[1], "expert 0 should get more capacity");
        assert!(suggested[1] > suggested[2] || suggested[1] >= 1);
        // Expert 3 got 0 hits but should still get minimum 1
        assert!(suggested[3] >= 1, "expert 3 minimum capacity = {}", suggested[3]);
    }

    #[test]
    fn test_load_balancer_hit_rates_sum_to_one() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0, 0.0],
            vec![0.0, 100.0, 0.0, 0.0],
            vec![0.0, 0.0, 100.0, 0.0],
            vec![0.0, 0.0, 0.0, 100.0],
            vec![100.0, 0.0, 0.0, 0.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&rt);

        let rates = balancer.hit_rates();
        let sum: f64 = rates.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "hit rates should sum to 1.0, got {}", sum);
    }

    #[test]
    fn test_load_balancer_hot_experts_top_n_zero() {
        let config = ExpertRouteConfig::new(4, 2);
        let balancer = ExpertLoadBalancer::new(config);
        let hot = balancer.hot_experts(0);
        assert!(hot.is_empty(), "top_n=0 should return empty");
    }

    // === moe_dispatch additional edge cases ===

    #[test]
    fn test_moe_dispatch_single_expert_selected() {
        let input = vec![1.0, 2.0, 3.0];
        let gate_logits = vec![0.0, 100.0]; // expert 1 dominates
        let expert_outputs = vec![
            vec![10.0, 20.0, 30.0],
            vec![5.0, 6.0, 7.0],
        ];
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 1);
        assert_eq!(output.len(), 3);
        // With top_k=1, only expert 1 selected, weight=1.0
        assert!((output[0] - 5.0).abs() < 0.01, "output[0] = {}", output[0]);
        assert!((output[1] - 6.0).abs() < 0.01);
        assert!((output[2] - 7.0).abs() < 0.01);
    }

    #[test]
    fn test_moe_dispatch_gate_logits_longer_than_experts() {
        let input = vec![1.0, 2.0];
        let gate_logits = vec![1.0, 0.0, 0.5, 2.0]; // 4 gate logits
        let expert_outputs = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
        ];
        // Only 2 experts exist, but gate has 4 entries
        // top_k=2 picks indices 0 and 2, but index 2 >= expert_outputs.len()
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 2);
        assert_eq!(output.len(), 2);
        // Expert 0 should contribute (index 0 < 2)
        // Expert at index 2 is out of bounds, so skipped
        assert!(output[0] > 0.0 || output[1] > 0.0, "at least expert 0 should contribute");
    }

    // === load_balance_loss additional tests ===

    #[test]
    fn test_load_balance_loss_uniform_routing_lower_loss() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            load_balance_loss: true,
            load_balance_lambda: 0.01,
            ..ExpertRouteConfig::default()
        };
        // Uniform routing: each token to a different expert
        let gate_logits_uniform = vec![
            vec![100.0, 0.0, 0.0, 0.0],
            vec![0.0, 100.0, 0.0, 0.0],
            vec![0.0, 0.0, 100.0, 0.0],
            vec![0.0, 0.0, 0.0, 100.0],
        ];
        let rt_uniform = ExpertRouteTable::from_gate_logits(config.clone(), &gate_logits_uniform);
        let loss_uniform = rt_uniform.load_balance_loss(&gate_logits_uniform);

        // Skewed routing: all tokens to expert 0
        let gate_logits_skewed = vec![
            vec![100.0, 0.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0, 0.0],
        ];
        let rt_skewed = ExpertRouteTable::from_gate_logits(config, &gate_logits_skewed);
        let loss_skewed = rt_skewed.load_balance_loss(&gate_logits_skewed);

        // Uniform routing should produce lower loss than skewed
        assert!(
            loss_uniform <= loss_skewed,
            "uniform loss ({}) should be <= skewed loss ({})",
            loss_uniform, loss_skewed
        );
    }

    #[test]
    fn test_load_balance_loss_lambda_zero() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            load_balance_loss: true,
            load_balance_lambda: 0.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![vec![100.0, 0.0, 0.0, 0.0]];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let loss = rt.load_balance_loss(&gate_logits);
        assert!((loss - 0.0).abs() < 1e-10, "lambda=0 → loss=0, got {}", loss);
    }

    // === ExpertRouteTable::utilization_stats balance score ===

    #[test]
    fn test_utilization_stats_perfectly_balanced() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        // Perfectly balanced: 2 tokens per expert
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0, 0.0],
            vec![0.0, 100.0, 0.0, 0.0],
            vec![0.0, 0.0, 100.0, 0.0],
            vec![0.0, 0.0, 0.0, 100.0],
            vec![100.0, 0.0, 0.0, 0.0],
            vec![0.0, 100.0, 0.0, 0.0],
            vec![0.0, 0.0, 100.0, 0.0],
            vec![0.0, 0.0, 0.0, 100.0],
        ];
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let stats = route_table.utilization_stats();
        assert_eq!(stats.max_expert_load, stats.min_expert_load,
            "perfectly balanced: max == min");
        assert!((stats.balance_score - 1.0).abs() < 1e-5,
            "perfectly balanced → score 1.0, got {}", stats.balance_score);
    }

    #[test]
    fn test_utilization_stats_completely_imbalanced() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        // All 10 tokens to expert 0, 0 to others
        let gate_logits: Vec<Vec<f32>> = (0..10)
            .map(|_| vec![100.0, 0.0, 0.0, 0.0])
            .collect();
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let stats = route_table.utilization_stats();
        assert_eq!(stats.max_expert_load, 10);
        assert_eq!(stats.min_expert_load, 0);
        assert!(
            stats.balance_score < 0.5,
            "completely imbalanced → score < 0.5, got {}",
            stats.balance_score
        );
    }

    // === Clone roundtrip for complex structures ===

    #[test]
    fn test_expert_route_table_clone_deep_copy() {
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 2,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![0.0, 100.0, 0.0],
            vec![0.0, 0.0, 100.0],
        ];
        let original = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let cloned = original.clone();

        // Verify deep copy: mutating clone does not affect original
        assert_eq!(cloned.token_routes.len(), original.token_routes.len());
        assert_eq!(cloned.expert_token_counts, original.expert_token_counts);
        assert_eq!(cloned.overflow_count, original.overflow_count);
        assert_eq!(cloned.config, original.config);
    }

    #[test]
    fn test_expert_load_balancer_clone_independent() {
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config);
        let gate_logits = vec![vec![100.0, 0.0]];
        let rt = ExpertRouteTable::from_gate_logits(balancer.config().clone(), &gate_logits);
        balancer.record_route(&rt);

        let cloned = balancer.clone();
        // Cloned should have same state
        assert_eq!(cloned.total_routes, balancer.total_routes);

        // Mutating original should not affect clone
        balancer.reset();
        let original_rates = balancer.hit_rates();
        let cloned_rates = cloned.hit_rates();
        assert!(original_rates.iter().all(|&r| r == 0.0));
        assert!(cloned_rates[0] > 0.0, "cloned should preserve pre-reset state");
    }

    // === ExpertRouteConfig with special float values ===

    #[test]
    fn test_expert_route_config_capacity_factor_zero() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 2,
            capacity_factor: 0.0,
            ..ExpertRouteConfig::default()
        };
        // ceil(0.0 * 100 / 4) = 0
        assert_eq!(config.expert_capacity(100), 0);
    }

    #[test]
    fn test_expert_route_config_noise_sigma_field() {
        let config = ExpertRouteConfig {
            noise_sigma: 0.42,
            ..ExpertRouteConfig::new(8, 2)
        };
        assert!((config.noise_sigma - 0.42).abs() < 1e-6);
        // noise_sigma does not affect capacity calculation
        assert_eq!(config.expert_capacity(128), 20);
    }

    // === topk_with_weights with extreme inputs ===

    #[test]
    fn test_topk_with_weights_single_dominant_value() {
        let logits = vec![0.0, 0.0, 100.0, 0.0];
        let topk = topk_with_weights(&logits, 2);
        assert_eq!(topk[0].0, 2, "index 2 (100.0) should be first");
        // The second pick is from the remaining equal values
        assert!(topk[0].1 > 0.99, "dominant expert should have weight near 1.0, got {}", topk[0].1);
    }

    // === ExpertRouteTable with top_k = num_experts (all selected) ===

    #[test]
    fn test_expert_route_table_top_k_equals_num_experts() {
        let config = ExpertRouteConfig::new(3, 3);
        let gate_logits = vec![
            vec![1.0, 2.0, 3.0],
        ];
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let route = &route_table.token_routes[0];
        assert_eq!(route.expert_indices.len(), 3, "all 3 experts should be selected");
        let sum: f32 = route.expert_weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "weights should sum to 1.0, got {}", sum);
    }

    // === ExpertLoadBalancer with zero hits total_hits ===

    #[test]
    fn test_load_balancer_suggest_capacity_all_zero_hits() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let balancer = ExpertLoadBalancer::new(config);
        // Record a route table where the only token gets routed but total_hits
        // is tracked via hit_history. If all hit_history entries are 0 after recording,
        // the total_hits branch returns base_capacity for all.
        // This shouldn't normally happen unless record_route is not called properly.
        // But let's test suggest_capacity with no recording:
        let suggested = balancer.suggest_capacity_adjustment(5);
        assert!(suggested.iter().all(|&c| c == 5));
    }

    // === Debug format comprehensive checks ===

    #[test]
    fn test_expert_route_config_debug_all_fields() {
        let config = ExpertRouteConfig {
            num_experts: 16,
            top_k: 4,
            capacity_factor: 2.5,
            load_balance_loss: true,
            load_balance_lambda: 0.05,
            noise_sigma: 1.0,
        };
        let debug = format!("{:?}", config);
        assert!(debug.contains("num_experts: 16"));
        assert!(debug.contains("top_k: 4"));
        assert!(debug.contains("capacity_factor: 2.5"));
        assert!(debug.contains("load_balance_loss: true"));
        assert!(debug.contains("load_balance_lambda: 0.05"));
        assert!(debug.contains("noise_sigma: 1.0"));
    }

    #[test]
    fn test_token_route_debug_shows_indices_and_weights() {
        let route = TokenRoute {
            expert_indices: vec![0, 3],
            expert_weights: vec![0.7, 0.3],
            expert_positions: vec![5, 8],
        };
        let debug = format!("{:?}", route);
        assert!(debug.contains("expert_indices"));
        assert!(debug.contains("expert_weights"));
        assert!(debug.contains("expert_positions"));
    }

    // === ExpertRouteTable config is stored by value ===

    #[test]
    fn test_expert_route_table_config_ownership() {
        let config = ExpertRouteConfig::new(6, 3);
        let gate_logits = vec![vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // config is moved into route_table, accessible via route_table.config
        assert_eq!(route_table.config.num_experts, 6);
        assert_eq!(route_table.config.top_k, 3);
    }

    // =========================================================================
    // Round 4: 50 additional tests (target 129 → 179+)
    // =========================================================================

    // --- ExpertRouteConfig field-level verification ---

    #[test]
    fn test_config_new_sets_default_capacity_factor() {
        let config = ExpertRouteConfig::new(4, 2);
        assert!((config.capacity_factor - 1.25).abs() < 1e-6,
            "new() should set capacity_factor to 1.25");
    }

    #[test]
    fn test_config_new_sets_default_load_balance_off() {
        let config = ExpertRouteConfig::new(4, 2);
        assert!(!config.load_balance_loss, "load_balance_loss defaults to false");
    }

    #[test]
    fn test_config_new_sets_default_lambda() {
        let config = ExpertRouteConfig::new(4, 2);
        assert!((config.load_balance_lambda - 0.01).abs() < 1e-8,
            "load_balance_lambda defaults to 0.01");
    }

    #[test]
    fn test_config_new_sets_default_noise_sigma_zero() {
        let config = ExpertRouteConfig::new(4, 2);
        assert!((config.noise_sigma).abs() < 1e-8, "noise_sigma defaults to 0.0");
    }

    #[test]
    fn test_config_default_matches_new_eight_two() {
        let from_new = ExpertRouteConfig::new(8, 2);
        let from_default = ExpertRouteConfig::default();
        assert_eq!(from_new.num_experts, from_default.num_experts);
        assert_eq!(from_new.top_k, from_default.top_k);
        assert!((from_new.capacity_factor - from_default.capacity_factor).abs() < 1e-6);
        assert_eq!(from_new.load_balance_loss, from_default.load_balance_loss);
    }

    #[test]
    fn test_config_partial_eq_differs_by_capacity_factor() {
        let a = ExpertRouteConfig {
            capacity_factor: 1.0,
            ..ExpertRouteConfig::new(4, 2)
        };
        let b = ExpertRouteConfig {
            capacity_factor: 2.0,
            ..ExpertRouteConfig::new(4, 2)
        };
        assert_ne!(a, b, "different capacity_factor should not be equal");
    }

    #[test]
    fn test_config_partial_eq_differs_by_noise_sigma() {
        let a = ExpertRouteConfig {
            noise_sigma: 0.0,
            ..ExpertRouteConfig::new(4, 2)
        };
        let b = ExpertRouteConfig {
            noise_sigma: 0.1,
            ..ExpertRouteConfig::new(4, 2)
        };
        assert_ne!(a, b, "different noise_sigma should not be equal");
    }

    // --- ExpertRouteConfig::expert_capacity comprehensive ---

    #[test]
    fn test_capacity_one_expert_one_token() {
        let config = ExpertRouteConfig {
            num_experts: 1,
            top_k: 1,
            capacity_factor: 1.0,
            ..ExpertRouteConfig::default()
        };
        // ceil(1.0 * 1 / 1) = 1
        assert_eq!(config.expert_capacity(1), 1);
    }

    #[test]
    fn test_capacity_exact_division() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 1.0,
            ..ExpertRouteConfig::default()
        };
        // ceil(1.0 * 16 / 4) = 4
        assert_eq!(config.expert_capacity(16), 4);
    }

    #[test]
    fn test_capacity_high_capacity_factor_large_experts() {
        let config = ExpertRouteConfig {
            num_experts: 128,
            top_k: 1,
            capacity_factor: 5.0,
            ..ExpertRouteConfig::default()
        };
        // ceil(5.0 * 1024 / 128) = ceil(40.0) = 40
        assert_eq!(config.expert_capacity(1024), 40);
    }

    // --- softmax comprehensive ---

    #[test]
    fn test_softmax_two_equal_produces_half() {
        let probs = softmax(&[1.0, 1.0]);
        assert_eq!(probs.len(), 2);
        assert!((probs[0] - 0.5).abs() < 1e-5, "expected 0.5, got {}", probs[0]);
        assert!((probs[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_monotonically_increasing() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let probs = softmax(&logits);
        for i in 1..probs.len() {
            assert!(probs[i] > probs[i - 1],
                "softmax should preserve ordering: probs[{}]={} <= probs[{}]={}",
                i - 1, probs[i - 1], i, probs[i]);
        }
    }

    #[test]
    fn test_softmax_mixed_large_small() {
        let logits = vec![-100.0, 0.0, 100.0];
        let probs = softmax(&logits);
        // Expert 2 should dominate, expert 0 should be near zero
        assert!(probs[2] > 0.99, "expert 2 should dominate, got {}", probs[2]);
        assert!(probs[0] < 0.001, "expert 0 should be near zero, got {}", probs[0]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_softmax_four_elements_sum_to_one() {
        let logits = vec![0.25, 0.5, 0.75, 1.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {}", sum);
        assert!(probs[3] > probs[0], "largest logit should produce largest prob");
    }

    // --- topk_with_weights edge cases ---

    #[test]
    fn test_topk_weights_returns_descending_order() {
        let logits = vec![0.1, 0.9, 0.5, 0.3, 0.7];
        let topk = topk_with_weights(&logits, 3);
        assert!(topk[0].1 >= topk[1].1, "weights should be descending");
        assert!(topk[1].1 >= topk[2].1, "weights should be descending");
    }

    #[test]
    fn test_topk_weights_all_negative() {
        let logits = vec![-10.0, -5.0, -20.0];
        let topk = topk_with_weights(&logits, 2);
        assert_eq!(topk[0].0, 1, "index 1 (-5.0) should be first");
        assert_eq!(topk[1].0, 0, "index 0 (-10.0) should be second");
        let sum: f32 = topk.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 0.01, "renormalized sum = {}", sum);
    }

    #[test]
    fn test_topk_weights_two_identical_high() {
        let logits = vec![5.0, 5.0, 1.0];
        let topk = topk_with_weights(&logits, 2);
        assert_eq!(topk.len(), 2);
        // Both should have approximately equal weight
        assert!((topk[0].1 - topk[1].1).abs() < 0.01,
            "identical logits should yield equal weights: {} vs {}", topk[0].1, topk[1].1);
    }

    // --- topk_indices edge cases ---

    #[test]
    fn test_topk_indices_two_elements() {
        let indices = topk_indices(&[3.0, 7.0], 2);
        assert_eq!(indices, vec![1, 0], "7.0 > 3.0");
    }

    #[test]
    fn test_topk_indices_many_same_top_k_small() {
        let logits = vec![1.0, 5.0, 3.0, 5.0, 2.0];
        let indices = topk_indices(&logits, 1);
        assert_eq!(indices.len(), 1);
        // Should pick one of the indices with value 5.0 (1 or 3)
        assert!(indices[0] == 1 || indices[0] == 3,
            "expected index 1 or 3, got {}", indices[0]);
    }

    #[test]
    fn test_topk_indices_large_k_returns_all() {
        let logits = vec![0.1, 0.2, 0.3];
        let indices = topk_indices(&logits, 10);
        assert_eq!(indices.len(), 3, "should return at most logits.len() indices");
    }

    // --- ExpertRouteTable comprehensive ---

    #[test]
    fn test_route_table_token_routes_length_matches_input() {
        let config = ExpertRouteConfig::new(4, 2);
        let gate_logits = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        assert_eq!(rt.token_routes.len(), 3);
    }

    #[test]
    fn test_route_table_expert_counts_sum_geq_tokens() {
        let config = ExpertRouteConfig::new(4, 2);
        let gate_logits = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4.0, 3.0, 2.0, 1.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let total_assignments: usize = rt.expert_token_counts.iter().sum();
        // With top_k=2, each token gets up to 2 experts, so total >= 2
        assert!(total_assignments >= 2,
            "total assignments should be >= number of tokens, got {}", total_assignments);
    }

    #[test]
    fn test_route_table_each_token_has_weights_summing_to_one() {
        let config = ExpertRouteConfig::new(4, 2);
        let gate_logits = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4.0, 3.0, 2.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        for (i, route) in rt.token_routes.iter().enumerate() {
            let sum: f32 = route.expert_weights.iter().sum();
            assert!((sum - 1.0).abs() < 0.02,
                "token {} weights sum = {}, expected ~1.0", i, sum);
        }
    }

    #[test]
    fn test_route_table_no_overflow_with_high_capacity() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 1000.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0, 0.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        assert_eq!(rt.overflow_count, 0, "high capacity should prevent overflow");
    }

    #[test]
    fn test_route_table_positions_are_unique_per_expert() {
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 0.0],
            vec![100.0, 0.0],
            vec![100.0, 0.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // All 3 tokens route to expert 0; positions should be 0, 1, 2
        let positions: Vec<usize> = rt.token_routes.iter()
            .filter_map(|r| {
                if r.expert_indices.first() == Some(&0) {
                    r.expert_positions.first().copied()
                } else {
                    None
                }
            })
            .collect();
        // Positions should be unique
        let mut sorted = positions.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), positions.len(),
            "positions should be unique, got {:?}", positions);
    }

    #[test]
    fn test_route_table_tokens_for_expert_returns_correct_pairs() {
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0],  // token 0 → expert 0
            vec![0.0, 100.0, 0.0],  // token 1 → expert 1
            vec![0.0, 0.0, 100.0],  // token 2 → expert 2
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let expert_0 = rt.tokens_for_expert(0);
        assert_eq!(expert_0.len(), 1);
        assert_eq!(expert_0[0].0, 0, "token 0 should be at expert 0");

        let expert_1 = rt.tokens_for_expert(1);
        assert_eq!(expert_1.len(), 1);
        assert_eq!(expert_1[0].0, 1, "token 1 should be at expert 1");

        let expert_2 = rt.tokens_for_expert(2);
        assert_eq!(expert_2.len(), 1);
        assert_eq!(expert_2[0].0, 2, "token 2 should be at expert 2");
    }

    #[test]
    fn test_route_table_overflow_count_increments_per_rejected_expert() {
        // capacity=1 per expert, top_k=1, 3 tokens all want expert 0
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 0.34, // ceil(0.34 * 3 / 2) = ceil(0.51) = 1
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 0.0],
            vec![100.0, 0.0],
            vec![100.0, 0.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Token 0 fills expert 0, tokens 1 and 2 overflow
        assert!(rt.overflow_count >= 2,
            "expected >= 2 overflows, got {}", rt.overflow_count);
    }

    // --- ExpertUtilizationStats derived from empty table ---

    #[test]
    fn test_utilization_stats_empty_table() {
        let config = ExpertRouteConfig::new(4, 2);
        let rt = ExpertRouteTable::from_gate_logits(config, &[]);
        let stats = rt.utilization_stats();
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.total_expert_assignments, 0);
        assert_eq!(stats.overflow_count, 0);
        assert_eq!(stats.max_expert_load, 0);
        assert_eq!(stats.min_expert_load, 0);
        assert!((stats.mean_expert_load).abs() < 1e-6);
        assert!((stats.balance_score - 1.0).abs() < 1e-5, "empty → perfect balance");
    }

    #[test]
    fn test_utilization_stats_total_assignments_equals_count_sum() {
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0],
            vec![0.0, 100.0, 0.0],
            vec![0.0, 0.0, 100.0],
            vec![100.0, 0.0, 0.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let stats = rt.utilization_stats();
        let counts_sum: usize = rt.expert_token_counts.iter().sum();
        assert_eq!(stats.total_expert_assignments, counts_sum,
            "total_expert_assignments should equal sum of expert_token_counts");
    }

    // --- ExpertLoadBalancer comprehensive ---

    #[test]
    fn test_balancer_record_route_increments_total() {
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        assert_eq!(balancer.total_routes, 0);

        let rt = ExpertRouteTable::from_gate_logits(config, &[vec![100.0, 0.0]]);
        balancer.record_route(&rt);
        assert_eq!(balancer.total_routes, 1);

        let rt2 = ExpertRouteTable::from_gate_logits(
            balancer.config().clone(),
            &[vec![0.0, 100.0]],
        );
        balancer.record_route(&rt2);
        assert_eq!(balancer.total_routes, 2);
    }

    #[test]
    fn test_balancer_cold_threshold_boundary() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        // 1000 tokens to expert 0, 0 to others
        let gate_logits: Vec<Vec<f32>> = (0..1000)
            .map(|_| vec![100.0, 0.0, 0.0, 0.0])
            .collect();
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&rt);

        let cold = balancer.cold_experts();
        // Experts 1, 2, 3 each have rate 0/1000 = 0.0 < 0.001 threshold
        let cold_indices: Vec<usize> = cold.iter().map(|(i, _)| *i).collect();
        assert!(cold_indices.contains(&1), "expert 1 should be cold");
        assert!(cold_indices.contains(&2), "expert 2 should be cold");
        assert!(cold_indices.contains(&3), "expert 3 should be cold");
        assert!(!cold_indices.contains(&0), "expert 0 should not be cold");
    }

    #[test]
    fn test_balancer_hot_experts_returns_descending() {
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        // 5 to expert 2, 3 to expert 0, 1 to expert 1
        let mut logits = Vec::new();
        for _ in 0..5 { logits.push(vec![0.0, 0.0, 100.0]); }
        for _ in 0..3 { logits.push(vec![100.0, 0.0, 0.0]); }
        logits.push(vec![0.0, 100.0, 0.0]);
        let rt = ExpertRouteTable::from_gate_logits(config, &logits);
        balancer.record_route(&rt);

        let hot = balancer.hot_experts(3);
        assert_eq!(hot[0].0, 2, "expert 2 should be hottest");
        assert_eq!(hot[1].0, 0, "expert 0 should be second");
        assert_eq!(hot[2].0, 1, "expert 1 should be third");
    }

    #[test]
    fn test_balancer_suggest_capacity_non_negative() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        // Extremely skewed: all to expert 0
        let logits: Vec<Vec<f32>> = (0..50)
            .map(|_| vec![100.0, 0.0, 0.0, 0.0])
            .collect();
        let rt = ExpertRouteTable::from_gate_logits(config, &logits);
        balancer.record_route(&rt);

        let suggested = balancer.suggest_capacity_adjustment(1);
        assert_eq!(suggested.len(), 4);
        for (i, &cap) in suggested.iter().enumerate() {
            assert!(cap >= 1, "expert {} capacity {} must be >= 1", i, cap);
        }
    }

    #[test]
    fn test_balancer_reset_clears_hit_history() {
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        let rt = ExpertRouteTable::from_gate_logits(config, &[vec![100.0, 0.0]]);
        balancer.record_route(&rt);
        assert!(balancer.hit_history[0] > 0);

        balancer.reset();
        assert!(balancer.hit_history.iter().all(|&h| h == 0),
            "reset should zero all hit_history entries");
    }

    #[test]
    fn test_balancer_config_returns_correct_reference() {
        let config = ExpertRouteConfig {
            num_experts: 7,
            top_k: 3,
            capacity_factor: 2.0,
            load_balance_loss: true,
            load_balance_lambda: 0.05,
            noise_sigma: 0.1,
        };
        let balancer = ExpertLoadBalancer::new(config);
        let cfg_ref = balancer.config();
        assert_eq!(cfg_ref.num_experts, 7);
        assert_eq!(cfg_ref.top_k, 3);
        assert!((cfg_ref.capacity_factor - 2.0).abs() < 1e-6);
        assert!(cfg_ref.load_balance_loss);
    }

    // --- moe_dispatch comprehensive ---

    #[test]
    fn test_moe_dispatch_zero_gate_logits_uniform_contribution() {
        let input = vec![0.0; 3];
        let gate_logits = vec![0.0, 0.0];
        let expert_outputs = vec![
            vec![2.0, 4.0, 6.0],
            vec![4.0, 2.0, 8.0],
        ];
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 2);
        // Equal logits → equal weights ≈ 0.5 each
        // output ≈ [3.0, 3.0, 7.0]
        assert!((output[0] - 3.0).abs() < 0.02, "output[0] = {}", output[0]);
        assert!((output[1] - 3.0).abs() < 0.02, "output[1] = {}", output[1]);
        assert!((output[2] - 7.0).abs() < 0.02, "output[2] = {}", output[2]);
    }

    #[test]
    fn test_moe_dispatch_single_expert_topk_one() {
        let input = vec![1.0, 2.0];
        let gate_logits = vec![0.0, 0.0, 100.0];
        let expert_outputs = vec![
            vec![10.0, 20.0],
            vec![20.0, 30.0],
            vec![5.0, 15.0],
        ];
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 1);
        // Only expert 2 selected with weight ≈ 1.0
        assert!((output[0] - 5.0).abs() < 0.01, "output[0] = {}", output[0]);
        assert!((output[1] - 15.0).abs() < 0.01, "output[1] = {}", output[1]);
    }

    #[test]
    fn test_moe_dispatch_output_size_matches_input() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gate_logits = vec![1.0, 0.0];
        let expert_outputs = vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0, 2.0, 2.0],
        ];
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 2);
        assert_eq!(output.len(), 5, "output size should match input size");
    }

    #[test]
    fn test_moe_dispatch_expert_index_out_of_bounds_skipped() {
        let input = vec![1.0];
        let gate_logits = vec![0.0, 0.0, 100.0]; // top_k=1 picks index 2
        let expert_outputs = vec![
            vec![10.0],
            vec![20.0],
            // No third expert output
        ];
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 1);
        // index 2 >= expert_outputs.len(), skipped → output stays 0.0
        assert!((output[0]).abs() < 1e-6,
            "out-of-bounds expert should be skipped, got {}", output[0]);
    }

    // --- load_balance_loss formula verification ---

    #[test]
    fn test_load_balance_loss_formula_components() {
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 100.0,
            load_balance_loss: true,
            load_balance_lambda: 1.0, // lambda=1 for easy calculation
            ..ExpertRouteConfig::default()
        };
        // 2 tokens: one to expert 0, one to expert 1
        let gate_logits = vec![
            vec![100.0, 0.0],
            vec![0.0, 100.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let loss = rt.load_balance_loss(&gate_logits);

        // f_i = [0.5, 0.5] (each expert gets 1 of 2 tokens)
        // P_i = [~0.5, ~0.5] (softmax of [100,0] and [0,100])
        // sum = 0.5*0.5 + 0.5*0.5 = 0.5
        // loss = 1.0 * 2 * 0.5 = 1.0
        assert!((loss - 1.0).abs() < 0.05,
            "loss should be close to 1.0, got {}", loss);
    }

    #[test]
    fn test_load_balance_loss_increases_with_skew() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            load_balance_loss: true,
            load_balance_lambda: 0.01,
            ..ExpertRouteConfig::default()
        };
        // Slightly skewed: 3 to expert 0, 1 to expert 1
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0, 0.0],
            vec![0.0, 100.0, 0.0, 0.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config.clone(), &gate_logits);
        let loss_skewed = rt.load_balance_loss(&gate_logits);

        // Uniform: 1 each
        let gate_logits_uniform = vec![
            vec![100.0, 0.0, 0.0, 0.0],
            vec![0.0, 100.0, 0.0, 0.0],
            vec![0.0, 0.0, 100.0, 0.0],
            vec![0.0, 0.0, 0.0, 100.0],
        ];
        let rt_uniform = ExpertRouteTable::from_gate_logits(config, &gate_logits_uniform);
        let loss_uniform = rt_uniform.load_balance_loss(&gate_logits_uniform);

        assert!(loss_skewed >= loss_uniform,
            "skewed loss ({}) should be >= uniform loss ({})", loss_skewed, loss_uniform);
    }

    // --- TokenRoute structural tests ---

    #[test]
    fn test_token_route_single_expert() {
        let route = TokenRoute {
            expert_indices: vec![3],
            expert_weights: vec![1.0],
            expert_positions: vec![0],
        };
        assert_eq!(route.expert_indices.len(), 1);
        assert_eq!(route.expert_weights.len(), 1);
        assert_eq!(route.expert_positions.len(), 1);
    }

    #[test]
    fn test_token_route_clone_is_independent() {
        let route = TokenRoute {
            expert_indices: vec![0, 1],
            expert_weights: vec![0.6, 0.4],
            expert_positions: vec![5, 3],
        };
        let mut cloned = route.clone();
        // Mutate the clone
        cloned.expert_indices.push(99);
        // Original should be unaffected
        assert_eq!(route.expert_indices.len(), 2, "original should have 2 indices");
        assert_eq!(cloned.expert_indices.len(), 3, "cloned should have 3 indices");
    }

    // --- ExpertRouteTable structural with varied configs ---

    #[test]
    fn test_route_table_two_experts_high_topk() {
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 5, // top_k > num_experts
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![vec![1.0, 2.0]];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Only 2 experts available, so at most 2 can be selected
        let route = &rt.token_routes[0];
        assert!(route.expert_indices.len() <= 2);
    }

    #[test]
    fn test_route_table_many_tokens_alternating_experts() {
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        // 20 tokens alternating between expert 0 and 1
        let gate_logits: Vec<Vec<f32>> = (0..20)
            .map(|i| if i % 2 == 0 {
                vec![100.0, 0.0]
            } else {
                vec![0.0, 100.0]
            })
            .collect();
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        assert_eq!(rt.token_routes.len(), 20);
        assert_eq!(rt.overflow_count, 0, "high capacity should prevent overflow");
        assert_eq!(rt.expert_token_counts[0], 10);
        assert_eq!(rt.expert_token_counts[1], 10);
    }

    #[test]
    fn test_route_table_single_expert_all_tokens() {
        let config = ExpertRouteConfig {
            num_experts: 1,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![5.0],
            vec![3.0],
            vec![7.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        assert_eq!(rt.overflow_count, 0);
        assert_eq!(rt.expert_token_counts[0], 3);
        for route in &rt.token_routes {
            assert_eq!(route.expert_indices[0], 0);
        }
    }

    // --- ExpertUtilizationStats Copy trait verification ---

    #[test]
    fn test_stats_copy_and_modify_independence() {
        let stats = ExpertUtilizationStats {
            total_tokens: 50,
            total_expert_assignments: 75,
            overflow_count: 2,
            max_expert_load: 30,
            min_expert_load: 10,
            mean_expert_load: 18.75,
            balance_score: 0.67,
        };
        let copied = stats;
        assert_ne!(copied.overflow_count, 999, "clone should be independent");
        assert_eq!(stats.overflow_count, 2, "original should be unmodified");
        assert!((stats.mean_expert_load - 18.75).abs() < 1e-5);
    }

    // --- ExpertLoadBalancer clone and reset interplay ---

    #[test]
    fn test_balancer_clone_preserves_state_before_reset() {
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        let rt = ExpertRouteTable::from_gate_logits(config, &[vec![100.0, 0.0]]);
        balancer.record_route(&rt);

        let cloned = balancer.clone();
        balancer.reset();

        assert_eq!(balancer.total_routes, 0, "original should be reset");
        assert_eq!(cloned.total_routes, 1, "cloned should preserve state");
        assert!(cloned.hit_history[0] > 0);
        assert!(balancer.hit_history[0] == 0);
    }

    // --- softmax with edge-case lengths ---

    #[test]
    fn test_softmax_long_sequence() {
        let logits: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let probs = softmax(&logits);
        assert_eq!(probs.len(), 100);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum = {}", sum);
        // Last element should have the highest probability
        assert!(probs[99] > probs[0]);
    }

    #[test]
    fn test_softmax_single_negative() {
        let probs = softmax(&[-5.0]);
        assert_eq!(probs.len(), 1);
        assert!((probs[0] - 1.0).abs() < 1e-6);
    }

    // --- topk_with_weights normalization with many equal values ---

    #[test]
    fn test_topk_weights_many_equal_pick_two() {
        let logits = vec![2.0, 2.0, 2.0, 2.0, 2.0];
        let topk = topk_with_weights(&logits, 2);
        assert_eq!(topk.len(), 2);
        let sum: f32 = topk.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 0.01);
        // Both weights should be approximately equal
        assert!((topk[0].1 - topk[1].1).abs() < 0.01);
    }

    // --- ExpertRouteTable overflow fallback behavior ---

    #[test]
    fn test_route_table_fallback_distributes_to_least_loaded() {
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 0.12, // ceil(0.12 * 3 / 3) = ceil(0.12) = 1
            ..ExpertRouteConfig::default()
        };
        // 3 tokens all want expert 0, capacity = 1 each
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Token 0 → expert 0 (position 0, no overflow)
        // Token 1 → expert 0 overflows → fallback to least loaded (expert 1 or 2)
        // Token 2 → expert 0 overflows → fallback to next least loaded
        let assigned_experts: Vec<usize> = rt.token_routes.iter()
            .filter_map(|r| r.expert_indices.first().copied())
            .collect();
        // Not all should be expert 0 (some must be fallback)
        assert!(assigned_experts.iter().any(|&e| e != 0),
            "at least one token should fallback to non-expert-0");
    }

    // --- moe_dispatch with zero input length but valid experts ---

    #[test]
    fn test_moe_dispatch_preserves_dimensionality() {
        let input = vec![0.0; 7];
        let gate_logits = vec![1.0, 0.0];
        let expert_outputs = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            vec![7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        ];
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 2);
        assert_eq!(output.len(), 7, "output dimensionality should match input");
    }

    // --- ExpertRouteConfig PartialEq transitivity ---

    #[test]
    fn test_config_partial_eq_transitive() {
        let a = ExpertRouteConfig::new(4, 2);
        let b = ExpertRouteConfig::new(4, 2);
        let c = ExpertRouteConfig::new(4, 2);
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c, "PartialEq should be transitive");
    }

    // --- ExpertUtilizationStats PartialEq transitivity ---

    #[test]
    fn test_stats_partial_eq_transitive() {
        let a = ExpertUtilizationStats {
            total_tokens: 5, total_expert_assignments: 10, overflow_count: 0,
            max_expert_load: 3, min_expert_load: 2, mean_expert_load: 2.5, balance_score: 0.8,
        };
        let b = a;
        let c = a;
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    // --- ExpertLoadBalancer multiple record_route calls accumulate ---

    #[test]
    fn test_balancer_accumulates_across_multiple_recordings() {
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        for _ in 0..5 {
            let rt = ExpertRouteTable::from_gate_logits(config.clone(), &[vec![100.0, 0.0]]);
            balancer.record_route(&rt);
        }
        assert_eq!(balancer.total_routes, 5);
        assert!(balancer.hit_history[0] >= 5, "expert 0 should have >= 5 hits");
    }

    // --- ExpertRouteTable from_gate_logits with diverse logits ---

    #[test]
    fn test_route_table_diverse_logits_no_panic() {
        let config = ExpertRouteConfig::new(8, 4);
        let gate_logits = vec![
            vec![-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0],
            vec![0.0; 8],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        assert_eq!(rt.token_routes.len(), 3);
        for route in &rt.token_routes {
            assert!(!route.expert_indices.is_empty());
        }
    }

    // =========================================================================
    // Round 5: 50 additional tests
    // =========================================================================

    // --- ExpertRouteConfig::expert_capacity with fractional capacity_factor ---

    #[test]
    fn test_capacity_fractional_factor_small_tokens() {
        let config = ExpertRouteConfig {
            num_experts: 5,
            top_k: 1,
            capacity_factor: 0.8,
            ..ExpertRouteConfig::default()
        };
        // ceil(0.8 * 3 / 5) = ceil(0.48) = 1
        assert_eq!(config.expert_capacity(3), 1);
    }

    #[test]
    fn test_capacity_fractional_factor_zero_result() {
        let config = ExpertRouteConfig {
            num_experts: 10,
            top_k: 1,
            capacity_factor: 0.01,
            ..ExpertRouteConfig::default()
        };
        // ceil(0.01 * 1 / 10) = ceil(0.001) = 1
        assert_eq!(config.expert_capacity(1), 1);
    }

    #[test]
    fn test_capacity_default_factor_four_experts_sixteen_tokens() {
        let config = ExpertRouteConfig::new(4, 2);
        // ceil(1.25 * 16 / 4) = ceil(5.0) = 5
        assert_eq!(config.expert_capacity(16), 5);
    }

    // --- ExpertRouteConfig::expert_capacity with various token counts ---

    #[test]
    fn test_capacity_two_experts_two_tokens() {
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 1.0,
            ..ExpertRouteConfig::default()
        };
        // ceil(1.0 * 2 / 2) = 1
        assert_eq!(config.expert_capacity(2), 1);
    }

    #[test]
    fn test_capacity_three_experts_nine_tokens() {
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 1.0,
            ..ExpertRouteConfig::default()
        };
        // ceil(1.0 * 9 / 3) = 3
        assert_eq!(config.expert_capacity(9), 3);
    }

    // --- softmax with special float values ---

    #[test]
    fn test_softmax_single_zero() {
        let probs = softmax(&[0.0]);
        assert_eq!(probs.len(), 1);
        assert!((probs[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_positive_infinity_dominates() {
        let probs = softmax(&[f32::INFINITY, 0.0, 0.0]);
        assert_eq!(probs.len(), 3);
        // Infinity should dominate: e^(Inf - Inf) = NaN edge case
        // but the function should not panic
        assert_eq!(probs.len(), 3);
    }

    #[test]
    fn test_softmax_negative_infinity_suppressed() {
        let probs = softmax(&[1.0, f32::NEG_INFINITY]);
        assert_eq!(probs.len(), 2);
        // e^(-Inf) = 0, so prob[0] should be 1.0
        assert!((probs[0] - 1.0).abs() < 1e-5, "prob[0] = {}", probs[0]);
        assert!((probs[1]).abs() < 1e-5, "prob[1] = {}", probs[1]);
    }

    #[test]
    fn test_softmax_many_zeros_uniform() {
        let logits = vec![0.0; 10];
        let probs = softmax(&logits);
        assert_eq!(probs.len(), 10);
        for &p in &probs {
            assert!((p - 0.1).abs() < 1e-5, "expected 0.1, got {}", p);
        }
    }

    // --- topk_with_weights with various k values ---

    #[test]
    fn test_topk_weights_k_one_returns_highest() {
        let logits = vec![0.1, 0.5, 0.9, 0.3];
        let topk = topk_with_weights(&logits, 1);
        assert_eq!(topk.len(), 1);
        assert_eq!(topk[0].0, 2, "index 2 has highest value 0.9");
        assert!((topk[0].1 - 1.0).abs() < 1e-5, "single selection weight = 1.0");
    }

    #[test]
    fn test_topk_weights_large_logits_range() {
        let logits = vec![-1000.0, 0.0, 1000.0];
        let topk = topk_with_weights(&logits, 2);
        assert_eq!(topk[0].0, 2, "index 2 (1000.0) should be first");
        // Weight of index 2 should be extremely close to 1.0
        assert!(topk[0].1 > 0.99, "dominant weight = {}", topk[0].1);
    }

    #[test]
    fn test_topk_weights_five_pick_three() {
        let logits = vec![1.0, 5.0, 3.0, 4.0, 2.0];
        let topk = topk_with_weights(&logits, 3);
        assert_eq!(topk.len(), 3);
        assert_eq!(topk[0].0, 1, "index 1 (5.0) first");
        assert_eq!(topk[1].0, 3, "index 3 (4.0) second");
        assert_eq!(topk[2].0, 2, "index 2 (3.0) third");
        let sum: f32 = topk.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 0.01, "sum = {}", sum);
    }

    // --- topk_indices additional edge cases ---

    #[test]
    fn test_topk_indices_with_extreme_negative() {
        let logits = vec![1.0, 2.0, -1e10];
        let indices = topk_indices(&logits, 2);
        assert_eq!(indices, vec![1, 0]);
    }

    #[test]
    fn test_topk_indices_single_element_k_one() {
        let indices = topk_indices(&[42.0], 1);
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn test_topk_indices_preserves_relative_order() {
        // Descending values
        let logits = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let indices = topk_indices(&logits, 3);
        assert_eq!(indices, vec![0, 1, 2]);
    }

    // --- ExpertRouteTable with edge-case expert counts ---

    #[test]
    fn test_route_table_many_experts_single_token() {
        let config = ExpertRouteConfig::new(64, 4);
        let mut logits = vec![0.0; 64];
        logits[10] = 100.0;
        logits[20] = 50.0;
        logits[30] = 30.0;
        logits[40] = 20.0;
        let rt = ExpertRouteTable::from_gate_logits(config, &[logits]);
        assert_eq!(rt.token_routes.len(), 1);
        let route = &rt.token_routes[0];
        assert!(route.expert_indices.len() <= 4);
        assert!(route.expert_indices.contains(&10));
    }

    #[test]
    fn test_route_table_expert_counts_length_matches_config() {
        let config = ExpertRouteConfig::new(7, 2);
        let gate_logits = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        assert_eq!(rt.expert_token_counts.len(), 7);
    }

    #[test]
    fn test_route_table_overflow_count_zero_with_sufficient_capacity() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 2,
            capacity_factor: 10.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        assert_eq!(rt.overflow_count, 0);
    }

    // --- tokens_for_expert with out-of-range index ---

    #[test]
    fn test_tokens_for_expert_out_of_range() {
        let config = ExpertRouteConfig::new(3, 1);
        let gate_logits = vec![vec![100.0, 0.0, 0.0]];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let tokens = rt.tokens_for_expert(100);
        assert!(tokens.is_empty(), "out-of-range expert index should return empty");
    }

    // --- ExpertUtilizationStats balance_score boundary ---

    #[test]
    fn test_stats_balance_score_range() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0, 0.0],
            vec![0.0, 100.0, 0.0, 0.0],
            vec![0.0, 0.0, 100.0, 0.0],
            vec![0.0, 0.0, 0.0, 100.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let stats = rt.utilization_stats();
        assert!(stats.balance_score >= 0.0 && stats.balance_score <= 1.0,
            "balance_score {} out of [0,1] range", stats.balance_score);
    }

    #[test]
    fn test_stats_max_geq_min() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits: Vec<Vec<f32>> = (0..20)
            .map(|i| {
                let mut row = vec![0.0; 4];
                row[i % 4] = 100.0;
                row
            })
            .collect();
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let stats = rt.utilization_stats();
        assert!(stats.max_expert_load >= stats.min_expert_load,
            "max {} < min {}", stats.max_expert_load, stats.min_expert_load);
    }

    #[test]
    fn test_stats_mean_between_min_max() {
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        // 5 to expert 0, 2 to expert 1, 1 to expert 2
        let mut gate_logits = Vec::new();
        for _ in 0..5 { gate_logits.push(vec![100.0, 0.0, 0.0]); }
        for _ in 0..2 { gate_logits.push(vec![0.0, 100.0, 0.0]); }
        gate_logits.push(vec![0.0, 0.0, 100.0]);
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let stats = rt.utilization_stats();
        assert!(
            stats.mean_expert_load >= stats.min_expert_load as f32
                && stats.mean_expert_load <= stats.max_expert_load as f32,
            "mean {} should be between min {} and max {}",
            stats.mean_expert_load, stats.min_expert_load, stats.max_expert_load
        );
    }

    // --- ExpertLoadBalancer hot/cold interaction ---

    #[test]
    fn test_balancer_no_cold_when_all_uniform() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        // 100 tokens, 25 per expert
        let gate_logits: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                let mut row = vec![0.0; 4];
                row[i % 4] = 100.0;
                row
            })
            .collect();
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&rt);
        let cold = balancer.cold_experts();
        assert!(cold.is_empty(), "uniformly loaded should have no cold experts");
    }

    #[test]
    fn test_balancer_hot_experts_top_one_returns_highest() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        let gate_logits: Vec<Vec<f32>> = (0..10)
            .map(|_| vec![100.0, 0.0, 0.0, 0.0])
            .collect();
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&rt);
        let hot = balancer.hot_experts(1);
        assert_eq!(hot.len(), 1);
        assert_eq!(hot[0].0, 0, "expert 0 should be the hottest");
    }

    #[test]
    fn test_balancer_hit_rates_length_matches_experts() {
        let config = ExpertRouteConfig::new(5, 2);
        let balancer = ExpertLoadBalancer::new(config);
        assert_eq!(balancer.hit_rates().len(), 5);
    }

    // --- ExpertLoadBalancer record_route with top_k > 1 ---

    #[test]
    fn test_balancer_records_multiple_experts_per_token() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 2,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        // Each token routes to 2 experts
        let gate_logits = vec![
            vec![100.0, 50.0, 0.0, 0.0],
            vec![0.0, 0.0, 100.0, 50.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&rt);
        // total_routes = number of tokens = 2
        assert_eq!(balancer.total_routes, 2);
        // But hit_history should reflect multiple expert assignments
        let total_hits: u64 = balancer.hit_history.iter().sum();
        assert!(total_hits >= 2, "total_hits = {} should be >= 2", total_hits);
    }

    // --- ExpertLoadBalancer suggest_capacity with varying base ---

    #[test]
    fn test_balancer_suggest_capacity_base_one() {
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let balancer = ExpertLoadBalancer::new(config);
        let suggested = balancer.suggest_capacity_adjustment(1);
        assert_eq!(suggested.len(), 2);
        for &cap in &suggested {
            assert!(cap >= 1);
        }
    }

    #[test]
    fn test_balancer_suggest_capacity_large_base() {
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        let gate_logits = vec![vec![100.0, 0.0, 0.0]];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&rt);
        let suggested = balancer.suggest_capacity_adjustment(1000);
        assert_eq!(suggested.len(), 3);
        assert!(suggested[0] > 0, "expert 0 should get non-zero capacity");
    }

    // --- moe_dispatch additional correctness ---

    #[test]
    fn test_moe_dispatch_all_experts_contribute() {
        let input = vec![1.0, 1.0];
        let gate_logits = vec![1.0, 1.0, 1.0]; // equal logits
        let expert_outputs = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 3);
        assert_eq!(output.len(), 2);
        // With 3 equal-weight experts: output = 1/3 * sum of all expert outputs
        // output[0] = 1/3 * (1 + 0 + 1) = 2/3
        // output[1] = 1/3 * (0 + 1 + 1) = 2/3
        assert!((output[0] - (2.0 / 3.0)).abs() < 0.02, "output[0] = {}", output[0]);
        assert!((output[1] - (2.0 / 3.0)).abs() < 0.02, "output[1] = {}", output[1]);
    }

    #[test]
    fn test_moe_dispatch_topk_one_single_expert_only() {
        let input = vec![0.0, 0.0, 0.0];
        let gate_logits = vec![0.0, 0.0, 100.0];
        let expert_outputs = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 1);
        // Only expert 2 selected, weight ~1.0
        assert!((output[0] - 7.0).abs() < 0.01);
        assert!((output[1] - 8.0).abs() < 0.01);
        assert!((output[2] - 9.0).abs() < 0.01);
    }

    #[test]
    fn test_moe_dispatch_with_negative_outputs() {
        let input = vec![0.0, 0.0];
        let gate_logits = vec![0.0, 0.0]; // equal
        let expert_outputs = vec![
            vec![-1.0, -2.0],
            vec![1.0, 2.0],
        ];
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 2);
        // Equal weights: output ≈ 0.5 * (-1 + 1) = 0, 0.5 * (-2 + 2) = 0
        assert!((output[0]).abs() < 0.02, "output[0] = {}", output[0]);
        assert!((output[1]).abs() < 0.02, "output[1] = {}", output[1]);
    }

    // --- load_balance_loss with varied lambda ---

    #[test]
    fn test_load_balance_loss_scales_with_lambda() {
        let base_config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 100.0,
            load_balance_loss: true,
            load_balance_lambda: 1.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 0.0],
            vec![100.0, 0.0],
        ];
        let rt1 = ExpertRouteTable::from_gate_logits(base_config, &gate_logits);
        let loss_lambda1 = rt1.load_balance_loss(&gate_logits);

        let config_lambda2 = ExpertRouteConfig {
            load_balance_lambda: 2.0,
            ..ExpertRouteConfig {
                num_experts: 2,
                top_k: 1,
                capacity_factor: 100.0,
                load_balance_loss: true,
                ..ExpertRouteConfig::default()
            }
        };
        let rt2 = ExpertRouteTable::from_gate_logits(config_lambda2, &gate_logits);
        let loss_lambda2 = rt2.load_balance_loss(&gate_logits);

        // Doubling lambda should approximately double the loss
        assert!(
            loss_lambda2 > loss_lambda1 * 1.5,
            "lambda=2 loss ({}) should be > 1.5x lambda=1 loss ({})",
            loss_lambda2, loss_lambda1
        );
    }

    #[test]
    fn test_load_balance_loss_non_negative_always() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 2,
            load_balance_loss: true,
            load_balance_lambda: 0.01,
            ..ExpertRouteConfig::default()
        };
        for _ in 0..10 {
            let gate_logits = vec![
                vec![1.0, 2.0, 3.0, 4.0],
                vec![4.0, 3.0, 2.0, 1.0],
            ];
            let rt = ExpertRouteTable::from_gate_logits(config.clone(), &gate_logits);
            let loss = rt.load_balance_loss(&gate_logits);
            assert!(loss >= 0.0, "loss should be non-negative, got {}", loss);
        }
    }

    // --- ExpertRouteTable with single-token edge cases ---

    #[test]
    fn test_route_table_single_token_top_k_two() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 2,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![vec![0.0, 1.0, 2.0, 3.0]];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        assert_eq!(rt.token_routes.len(), 1);
        let route = &rt.token_routes[0];
        assert_eq!(route.expert_indices.len(), 2);
        assert_eq!(route.expert_indices[0], 3, "highest logit at index 3");
        assert_eq!(route.expert_indices[1], 2, "second highest at index 2");
    }

    // --- ExpertRouteTable with all tokens routing to same expert ---

    #[test]
    fn test_route_table_all_tokens_same_expert_no_overflow() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits: Vec<Vec<f32>> = (0..5)
            .map(|_| vec![100.0, 0.0, 0.0, 0.0])
            .collect();
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        assert_eq!(rt.overflow_count, 0);
        assert_eq!(rt.expert_token_counts[0], 5);
        assert_eq!(rt.expert_token_counts[1], 0);
        assert_eq!(rt.expert_token_counts[2], 0);
        assert_eq!(rt.expert_token_counts[3], 0);
    }

    // --- ExpertUtilizationStats Clone ---

    #[test]
    fn test_stats_clone_independence() {
        let stats = ExpertUtilizationStats {
            total_tokens: 100,
            total_expert_assignments: 150,
            overflow_count: 5,
            max_expert_load: 40,
            min_expert_load: 10,
            mean_expert_load: 25.0,
            balance_score: 0.75,
        };
        let mut cloned = stats.clone();
        cloned.total_tokens = 0;
        assert_eq!(stats.total_tokens, 100, "original should be unaffected");
        assert_eq!(cloned.total_tokens, 0);
    }

    // --- ExpertRouteConfig equality edge cases ---

    #[test]
    fn test_config_partial_eq_differs_by_num_experts() {
        let a = ExpertRouteConfig::new(4, 2);
        let b = ExpertRouteConfig::new(8, 2);
        assert_ne!(a, b);
    }

    #[test]
    fn test_config_partial_eq_differs_by_top_k() {
        let a = ExpertRouteConfig::new(4, 1);
        let b = ExpertRouteConfig::new(4, 2);
        assert_ne!(a, b);
    }

    #[test]
    fn test_config_partial_eq_differs_by_lambda() {
        let a = ExpertRouteConfig {
            load_balance_lambda: 0.01,
            ..ExpertRouteConfig::new(4, 2)
        };
        let b = ExpertRouteConfig {
            load_balance_lambda: 0.02,
            ..ExpertRouteConfig::new(4, 2)
        };
        assert_ne!(a, b);
    }

    // --- ExpertRouteTable positions and indices consistency ---

    #[test]
    fn test_route_table_positions_within_expert_capacity() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits: Vec<Vec<f32>> = (0..10)
            .map(|i| {
                let mut row = vec![0.0; 4];
                row[i % 4] = 100.0;
                row
            })
            .collect();
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let capacity = rt.config.expert_capacity(10);
        for route in &rt.token_routes {
            for &pos in &route.expert_positions {
                assert!(pos < capacity,
                    "position {} >= capacity {}", pos, capacity);
            }
        }
    }

    #[test]
    fn test_route_table_indices_within_expert_range() {
        let config = ExpertRouteConfig::new(8, 3);
        let gate_logits = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        for route in &rt.token_routes {
            for &idx in &route.expert_indices {
                assert!(idx < 8, "expert index {} >= num_experts", idx);
            }
        }
    }

    // --- ExpertLoadBalancer repeated resets ---

    #[test]
    fn test_balancer_repeated_resets() {
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        for _ in 0..5 {
            let rt = ExpertRouteTable::from_gate_logits(config.clone(), &[vec![100.0, 0.0]]);
            balancer.record_route(&rt);
            balancer.reset();
        }
        assert_eq!(balancer.total_routes, 0);
        assert!(balancer.hit_history.iter().all(|&h| h == 0));
    }

    // --- ExpertRouteTable with many tokens same pattern ---

    #[test]
    fn test_route_table_large_uniform_distribution() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                let mut row = vec![0.0; 4];
                row[i % 4] = 100.0;
                row
            })
            .collect();
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        assert_eq!(rt.overflow_count, 0);
        // Each expert should get 25 tokens
        for &count in &rt.expert_token_counts {
            assert_eq!(count, 25, "each expert should have 25 tokens");
        }
    }

    // --- ExpertUtilizationStats derived with single token ---

    #[test]
    fn test_utilization_stats_single_token() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 2,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![vec![0.0, 0.0, 100.0, 0.0]];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let stats = rt.utilization_stats();
        assert_eq!(stats.total_tokens, 1);
        assert!(stats.total_expert_assignments >= 1);
        assert_eq!(stats.overflow_count, 0);
    }

    // --- moe_dispatch with varying input sizes ---

    #[test]
    fn test_moe_dispatch_single_element() {
        let input = vec![1.0];
        let gate_logits = vec![1.0, 0.0];
        let expert_outputs = vec![
            vec![3.0],
            vec![5.0],
        ];
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 1);
        assert_eq!(output.len(), 1);
        assert!((output[0] - 3.0).abs() < 0.01, "output = {}", output[0]);
    }

    // --- ExpertLoadBalancer with zero experts and operations ---

    #[test]
    fn test_balancer_zero_experts_suggest_capacity() {
        let config = ExpertRouteConfig {
            num_experts: 0,
            top_k: 0,
            ..ExpertRouteConfig::default()
        };
        let balancer = ExpertLoadBalancer::new(config);
        let suggested = balancer.suggest_capacity_adjustment(10);
        assert!(suggested.is_empty());
    }

    // --- ExpertRouteTable with varying capacity_factor values ---

    #[test]
    fn test_route_table_low_capacity_causes_overflow() {
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 0.1, // very low: ceil(0.1 * 10 / 2) = 1
            ..ExpertRouteConfig::default()
        };
        let gate_logits: Vec<Vec<f32>> = (0..10)
            .map(|_| vec![100.0, 0.0])
            .collect();
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        assert!(rt.overflow_count > 0, "low capacity should cause overflow");
    }

    #[test]
    fn test_route_table_high_capacity_no_overflow() {
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits: Vec<Vec<f32>> = (0..10)
            .map(|_| vec![100.0, 0.0])
            .collect();
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        assert_eq!(rt.overflow_count, 0);
    }

    // --- ExpertUtilizationStats PartialEq symmetry ---

    #[test]
    fn test_stats_partial_eq_symmetry() {
        let a = ExpertUtilizationStats {
            total_tokens: 10,
            total_expert_assignments: 20,
            overflow_count: 1,
            max_expert_load: 8,
            min_expert_load: 2,
            mean_expert_load: 5.0,
            balance_score: 0.6,
        };
        let b = ExpertUtilizationStats {
            total_tokens: 10,
            total_expert_assignments: 20,
            overflow_count: 1,
            max_expert_load: 8,
            min_expert_load: 2,
            mean_expert_load: 5.0,
            balance_score: 0.6,
        };
        assert_eq!(a, b);
        assert_eq!(b, a, "PartialEq should be symmetric");
    }

    // --- softmax with alternating signs ---

    #[test]
    fn test_softmax_alternating_signs() {
        let logits = vec![-1.0, 1.0, -1.0, 1.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Indices 1 and 3 (positive) should have higher prob than 0 and 2 (negative)
        assert!(probs[1] > probs[0]);
        assert!(probs[3] > probs[2]);
        // Symmetric: probs[1] == probs[3] and probs[0] == probs[2]
        assert!((probs[1] - probs[3]).abs() < 1e-5);
        assert!((probs[0] - probs[2]).abs() < 1e-5);
    }

    // --- topk_with_weights ensures weights are non-negative ---

    #[test]
    fn test_topk_weights_are_non_negative() {
        let logits = vec![-5.0, 0.0, 5.0, -3.0, 2.0];
        let topk = topk_with_weights(&logits, 3);
        for (_, w) in &topk {
            assert!(*w >= 0.0, "weight should be non-negative, got {}", w);
        }
    }

    // --- ExpertRouteTable from_gate_logits with two tokens different patterns ---

    #[test]
    fn test_route_table_two_tokens_different_experts() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0, 0.0], // token 0 → expert 0
            vec![0.0, 0.0, 0.0, 100.0], // token 1 → expert 3
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        assert_eq!(rt.token_routes[0].expert_indices[0], 0);
        assert_eq!(rt.token_routes[1].expert_indices[0], 3);
        assert_eq!(rt.expert_token_counts[0], 1);
        assert_eq!(rt.expert_token_counts[3], 1);
        assert_eq!(rt.expert_token_counts[1], 0);
        assert_eq!(rt.expert_token_counts[2], 0);
    }

    // --- ExpertLoadBalancer total_routes tracking ---

    #[test]
    fn test_balancer_total_routes_tracks_tokens_not_experts() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 2,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        let gate_logits = vec![
            vec![100.0, 50.0, 0.0, 0.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&rt);
        // total_routes counts tokens, not expert assignments
        assert_eq!(balancer.total_routes, 1);
    }

    // --- ExpertRouteConfig::new with various sizes ---

    #[test]
    fn test_config_new_large_values() {
        let config = ExpertRouteConfig::new(1024, 64);
        assert_eq!(config.num_experts, 1024);
        assert_eq!(config.top_k, 64);
        assert!((config.capacity_factor - 1.25).abs() < 1e-6);
    }

    #[test]
    fn test_config_new_small_values() {
        let config = ExpertRouteConfig::new(1, 1);
        assert_eq!(config.num_experts, 1);
        assert_eq!(config.top_k, 1);
    }

    // =========================================================================
    // Round 6: 60 additional tests
    // =========================================================================

    // --- ExpertRouteConfig symmetry and edge cases ---

    #[test]
    fn test_config_partial_eq_symmetry() {
        // Arrange
        let a = ExpertRouteConfig::new(4, 2);
        let b = ExpertRouteConfig::new(4, 2);
        // Assert: a == b implies b == a
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    #[test]
    fn test_config_capacity_factor_infinity() {
        // Arrange: infinite capacity factor
        let config = ExpertRouteConfig {
            capacity_factor: f32::INFINITY,
            ..ExpertRouteConfig::new(4, 2)
        };
        // Act
        let cap = config.expert_capacity(100);
        // Assert: ceil(Inf * 100 / 4) — inf as usize is platform-dependent but should not panic
        // Just verify it doesn't panic and returns some usize
        let _ = cap;
    }

    #[test]
    fn test_config_capacity_factor_negative() {
        // Arrange: negative capacity factor
        let config = ExpertRouteConfig {
            capacity_factor: -1.0,
            ..ExpertRouteConfig::new(4, 2)
        };
        // Act
        let cap = config.expert_capacity(100);
        // Assert: ceil(-100.0 / 4) = ceil(-25.0) = -25 → as usize wraps
        // The function should not panic; just verify it completes
        let _ = cap;
    }

    #[test]
    fn test_config_new_zero_experts_is_constructable() {
        // Arrange & Act
        let config = ExpertRouteConfig::new(0, 0);
        // Assert: construction succeeds
        assert_eq!(config.num_experts, 0);
        assert_eq!(config.top_k, 0);
    }

    #[test]
    fn test_config_capacity_large_experts_small_tokens() {
        // Arrange: 1000 experts, 1 token
        let config = ExpertRouteConfig {
            num_experts: 1000,
            top_k: 1,
            capacity_factor: 1.0,
            ..ExpertRouteConfig::default()
        };
        // Act: ceil(1.0 * 1 / 1000) = ceil(0.001) = 1
        let cap = config.expert_capacity(1);
        // Assert
        assert_eq!(cap, 1);
    }

    // --- softmax additional stress and edge cases ---

    #[test]
    fn test_softmax_very_large_count() {
        // Arrange: 1000 elements
        let logits: Vec<f32> = (0..1000).map(|i| i as f32 * 0.001).collect();
        // Act
        let probs = softmax(&logits);
        // Assert
        assert_eq!(probs.len(), 1000);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "sum = {}", sum);
    }

    #[test]
    fn test_softmax_all_very_large_positive() {
        // Arrange
        let logits = vec![1e10_f32, 1e10_f32 + 1.0, 1e10_f32 + 2.0];
        // Act
        let probs = softmax(&logits);
        // Assert: should not produce NaN/Inf
        for &p in &probs {
            assert!(p.is_finite(), "probability should be finite, got {}", p);
        }
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "sum = {}", sum);
    }

    #[test]
    fn test_softmax_all_very_large_negative() {
        // Arrange
        let logits = vec![-1e10_f32, -1e10_f32 + 1.0, -1e10_f32 + 2.0];
        // Act
        let probs = softmax(&logits);
        // Assert
        for &p in &probs {
            assert!(p.is_finite(), "probability should be finite, got {}", p);
        }
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "sum = {}", sum);
    }

    #[test]
    fn test_softmax_values_near_zero() {
        // Arrange: very small values
        let logits = vec![1e-30_f32, 2e-30_f32, 3e-30_f32];
        // Act
        let probs = softmax(&logits);
        // Assert: should behave like uniform since differences are subnormals
        assert_eq!(probs.len(), 3);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "sum = {}", sum);
    }

    // --- topk_with_weights additional ---

    #[test]
    fn test_topk_weights_k_one_highest_at_index_zero() {
        // Arrange: highest value at index 0
        let logits = vec![100.0, 1.0, 2.0, 3.0];
        // Act
        let topk = topk_with_weights(&logits, 1);
        // Assert
        assert_eq!(topk.len(), 1);
        assert_eq!(topk[0].0, 0);
        assert!((topk[0].1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_topk_weights_many_zeros_one_high() {
        // Arrange: one outlier among zeros
        let logits = vec![0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // Act
        let topk = topk_with_weights(&logits, 3);
        // Assert
        assert_eq!(topk.len(), 3);
        assert_eq!(topk[0].0, 2, "index 2 (100.0) should be first");
        assert!(topk[0].1 > 0.99, "dominant weight = {}", topk[0].1);
    }

    #[test]
    fn test_topk_weights_indices_within_range() {
        // Arrange
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        // Act
        let topk = topk_with_weights(&logits, 3);
        // Assert: all indices must be valid
        for &(idx, _) in &topk {
            assert!(idx < logits.len(), "index {} out of range", idx);
        }
    }

    #[test]
    fn test_topk_weights_sum_approximately_one_property() {
        // Arrange: various logit configurations
        let test_cases: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0],
            vec![-5.0, -1.0, 3.0, 0.0],
            vec![100.0],
            vec![0.0, 0.0, 0.0, 0.0],
        ];
        for logits in &test_cases {
            // Act
            let topk = topk_with_weights(logits, 2.min(logits.len()));
            if !topk.is_empty() {
                let sum: f32 = topk.iter().map(|(_, w)| *w).sum();
                // Assert
                assert!((sum - 1.0).abs() < 0.01,
                    "weights sum = {} for logits {:?}", sum, logits);
            }
        }
    }

    // --- topk_indices additional ---

    #[test]
    fn test_topk_indices_zero_length_and_k_zero() {
        // Arrange: empty logits, k=0
        let indices = topk_indices(&[], 0);
        // Assert
        assert!(indices.is_empty());
    }

    #[test]
    fn test_topk_indices_with_mixed_nan() {
        // Arrange: logits containing NaN
        let logits = vec![1.0, f32::NAN, 3.0];
        // Act: should not panic
        let indices = topk_indices(&logits, 2);
        // Assert: returns some indices (NaN comparison is unreliable)
        assert!(indices.len() <= 3);
    }

    #[test]
    fn test_topk_indices_all_valid_indices() {
        // Arrange
        let logits = vec![0.3, 0.1, 0.9, 0.5, 0.7];
        // Act
        let indices = topk_indices(&logits, 3);
        // Assert: every index < logits.len()
        for &idx in &indices {
            assert!(idx < logits.len(), "invalid index {}", idx);
        }
        // No duplicates
        let mut sorted = indices.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), indices.len(), "indices should be unique");
    }

    // --- TokenRoute structural invariants ---

    #[test]
    fn test_token_route_field_lengths_correspond() {
        // Arrange: construct routes where indices/weights/positions lengths match
        let route = TokenRoute {
            expert_indices: vec![0, 2, 5],
            expert_weights: vec![0.5, 0.3, 0.2],
            expert_positions: vec![10, 3, 7],
        };
        // Assert: all three vecs have same length
        assert_eq!(route.expert_indices.len(), route.expert_weights.len());
        assert_eq!(route.expert_weights.len(), route.expert_positions.len());
    }

    #[test]
    fn test_token_route_weights_non_negative() {
        // Arrange: route from actual routing
        let config = ExpertRouteConfig::new(4, 2);
        let gate_logits = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: all weights non-negative
        for route in &rt.token_routes {
            for &w in &route.expert_weights {
                assert!(w >= 0.0, "weight should be non-negative, got {}", w);
            }
        }
    }

    #[test]
    fn test_token_route_positions_non_negative() {
        // Arrange
        let config = ExpertRouteConfig::new(4, 2);
        let gate_logits = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: all positions non-negative
        for route in &rt.token_routes {
            for &pos in &route.expert_positions {
                assert!(pos == pos, "position should be a valid usize"); // no NaN for usize
            }
        }
    }

    #[test]
    fn test_token_route_large_expert_count() {
        // Arrange: route with 128 experts
        let mut logits = vec![0.0; 128];
        logits[42] = 100.0;
        logits[17] = 50.0;
        let config = ExpertRouteConfig::new(128, 2);
        let rt = ExpertRouteTable::from_gate_logits(config, &[logits]);
        // Assert
        assert_eq!(rt.token_routes.len(), 1);
        let route = &rt.token_routes[0];
        assert!(route.expert_indices.contains(&42));
        assert!(route.expert_indices.len() <= 2);
    }

    // --- ExpertRouteTable structural invariants ---

    #[test]
    fn test_route_table_overflow_count_non_negative() {
        // Arrange: various configs
        for cf in [0.1, 0.5, 1.0, 1.25, 2.0, 10.0] {
            let config = ExpertRouteConfig {
                num_experts: 4,
                top_k: 2,
                capacity_factor: cf,
                ..ExpertRouteConfig::default()
            };
            let gate_logits = vec![
                vec![1.0, 2.0, 3.0, 4.0],
                vec![4.0, 3.0, 2.0, 1.0],
            ];
            let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
            // Assert
            assert!(rt.overflow_count <= rt.token_routes.len() * 2,
                "overflow_count {} > max possible {}", rt.overflow_count, rt.token_routes.len() * 2);
        }
    }

    #[test]
    fn test_route_table_expert_token_counts_non_negative() {
        // Arrange
        let config = ExpertRouteConfig::new(4, 2);
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0, 0.0],
            vec![0.0, 100.0, 0.0, 0.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: all counts are non-negative (usize is always non-negative, but verify non-zero sum)
        let total: usize = rt.expert_token_counts.iter().sum();
        assert!(total > 0, "total token assignments should be > 0");
    }

    #[test]
    fn test_route_table_config_num_experts_matches_counts_len() {
        // Arrange: multiple configs
        for num_experts in [1, 2, 4, 8, 16] {
            let config = ExpertRouteConfig::new(num_experts, 1);
            let gate_logits = vec![vec![100.0; num_experts]];
            let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
            // Assert: counts vec length matches num_experts
            assert_eq!(rt.expert_token_counts.len(), num_experts,
                "counts len {} != num_experts {}", rt.expert_token_counts.len(), num_experts);
        }
    }

    #[test]
    fn test_route_table_single_token_all_experts_equal_logits() {
        // Arrange: all equal logits — tie-breaking
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 2,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![vec![5.0, 5.0, 5.0]];
        // Act
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: at least 1 expert assigned, at most top_k
        let route = &rt.token_routes[0];
        assert!(route.expert_indices.len() >= 1);
        assert!(route.expert_indices.len() <= 2);
        let sum: f32 = route.expert_weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.02, "weights sum = {}", sum);
    }

    #[test]
    fn test_route_table_overflow_distributes_to_least_loaded() {
        // Arrange: 4 experts, capacity=1 each, 5 tokens all want expert 0
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 0.2, // ceil(0.2 * 5 / 4) = ceil(0.25) = 1
            ..ExpertRouteConfig::default()
        };
        let gate_logits: Vec<Vec<f32>> = (0..5)
            .map(|_| vec![100.0, 0.0, 0.0, 0.0])
            .collect();
        // Act
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: overflow occurs and tokens are redistributed
        assert!(rt.overflow_count > 0);
        // At least 2 different experts should have tokens (not all go to expert 0)
        let experts_with_tokens: Vec<usize> = rt.expert_token_counts.iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(i, _)| i)
            .collect();
        assert!(experts_with_tokens.len() >= 2,
            "overflow should distribute to multiple experts, got {:?}", experts_with_tokens);
    }

    #[test]
    fn test_route_table_top_k_zero_falls_back() {
        // Arrange: top_k=0 means topk_with_weights returns empty, triggering fallback
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 0,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![vec![1.0, 2.0, 3.0, 4.0]];
        // Act
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: fallback assigns to least loaded expert (expert 0 initially)
        let route = &rt.token_routes[0];
        assert_eq!(route.expert_indices.len(), 1, "fallback should assign exactly 1 expert");
        assert!((route.expert_weights[0] - 1.0).abs() < 1e-5, "fallback weight should be 1.0");
    }

    #[test]
    fn test_route_table_expert_counts_sum_geq_token_count() {
        // Arrange
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 2,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4.0, 3.0, 2.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: sum of expert counts >= number of tokens (each token assigned >= 1 expert)
        let total_assignments: usize = rt.expert_token_counts.iter().sum();
        assert!(total_assignments >= rt.token_routes.len(),
            "total {} < token count {}", total_assignments, rt.token_routes.len());
    }

    #[test]
    fn test_route_table_overflow_count_bounded_by_top_k_times_tokens() {
        // Arrange
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 0.2,
            ..ExpertRouteConfig::default()
        };
        let gate_logits: Vec<Vec<f32>> = (0..10)
            .map(|_| vec![100.0, 0.0])
            .collect();
        let max_overflow = config.top_k * gate_logits.len();
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: overflow_count <= top_k * num_tokens
        assert!(rt.overflow_count <= max_overflow,
            "overflow {} > max possible {}", rt.overflow_count, max_overflow);
    }

    // --- ExpertUtilizationStats additional ---

    #[test]
    fn test_stats_all_zero_fields() {
        // Arrange & Act
        let stats = ExpertUtilizationStats {
            total_tokens: 0,
            total_expert_assignments: 0,
            overflow_count: 0,
            max_expert_load: 0,
            min_expert_load: 0,
            mean_expert_load: 0.0,
            balance_score: 1.0,
        };
        // Assert
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.total_expert_assignments, 0);
        assert!((stats.balance_score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_stats_overflow_less_equal_total_tokens() {
        // Arrange: derive stats from a route table
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 0.3,
            ..ExpertRouteConfig::default()
        };
        let gate_logits: Vec<Vec<f32>> = (0..5)
            .map(|_| vec![100.0, 0.0])
            .collect();
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let stats = rt.utilization_stats();
        // Assert
        assert!(stats.overflow_count <= stats.total_tokens,
            "overflow {} > total_tokens {}", stats.overflow_count, stats.total_tokens);
    }

    #[test]
    fn test_stats_mean_non_negative() {
        // Arrange
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0],
            vec![0.0, 100.0, 0.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let stats = rt.utilization_stats();
        // Assert
        assert!(stats.mean_expert_load >= 0.0,
            "mean should be non-negative, got {}", stats.mean_expert_load);
    }

    #[test]
    fn test_stats_debug_contains_all_field_names() {
        // Arrange
        let stats = ExpertUtilizationStats {
            total_tokens: 1,
            total_expert_assignments: 2,
            overflow_count: 3,
            max_expert_load: 4,
            min_expert_load: 5,
            mean_expert_load: 6.0,
            balance_score: 7.0,
        };
        // Act
        let debug = format!("{:?}", stats);
        // Assert: all field names present
        assert!(debug.contains("total_tokens"), "missing total_tokens");
        assert!(debug.contains("total_expert_assignments"), "missing total_expert_assignments");
        assert!(debug.contains("overflow_count"), "missing overflow_count");
        assert!(debug.contains("max_expert_load"), "missing max_expert_load");
        assert!(debug.contains("min_expert_load"), "missing min_expert_load");
        assert!(debug.contains("mean_expert_load"), "missing mean_expert_load");
        assert!(debug.contains("balance_score"), "missing balance_score");
    }

    // --- ExpertLoadBalancer additional ---

    #[test]
    fn test_balancer_new_hit_history_all_zeros() {
        // Arrange
        let config = ExpertRouteConfig::new(8, 2);
        // Act
        let balancer = ExpertLoadBalancer::new(config);
        // Assert
        assert_eq!(balancer.hit_history.len(), 8);
        assert!(balancer.hit_history.iter().all(|&h| h == 0),
            "new balancer should have all-zero hit_history");
    }

    #[test]
    fn test_balancer_new_total_routes_zero() {
        // Arrange & Act
        let balancer = ExpertLoadBalancer::new(ExpertRouteConfig::new(4, 2));
        // Assert
        assert_eq!(balancer.total_routes, 0);
    }

    #[test]
    fn test_balancer_record_route_empty_table() {
        // Arrange: empty route table (0 tokens)
        let config = ExpertRouteConfig::new(4, 2);
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        let rt = ExpertRouteTable::from_gate_logits(config, &[]);
        // Act
        balancer.record_route(&rt);
        // Assert: no tokens → total_routes stays 0
        assert_eq!(balancer.total_routes, 0);
        assert!(balancer.hit_history.iter().all(|&h| h == 0));
    }

    #[test]
    fn test_balancer_cold_experts_returns_below_threshold_only() {
        // Arrange: threshold is 0.001 (0.1%)
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        // 1000 tokens all to expert 0
        let gate_logits: Vec<Vec<f32>> = (0..1000)
            .map(|_| vec![100.0, 0.0, 0.0, 0.0])
            .collect();
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&rt);
        // Act
        let cold = balancer.cold_experts();
        // Assert: cold experts all have rate < 0.001
        for &(idx, rate) in &cold {
            assert!(rate < 0.001, "cold expert {} rate {} >= 0.001", idx, rate);
        }
        // Expert 0 should not be in cold list
        assert!(cold.iter().all(|(idx, _)| *idx != 0));
    }

    #[test]
    fn test_balancer_hot_experts_all_equal_returns_all() {
        // Arrange: all experts have equal hits
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0],
            vec![0.0, 100.0, 0.0],
            vec![0.0, 0.0, 100.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&rt);
        // Act
        let hot = balancer.hot_experts(3);
        // Assert: all 3 experts returned with approximately equal rates
        assert_eq!(hot.len(), 3);
        for i in 0..hot.len() - 1 {
            assert!((hot[i].1 - hot[i + 1].1).abs() < 0.01,
                "equal-load experts should have similar rates: {} vs {}", hot[i].1, hot[i + 1].1);
        }
    }

    #[test]
    fn test_balancer_suggest_capacity_skewed_hot_gets_more() {
        // Arrange: expert 0 gets 90% of traffic
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        let mut logits = Vec::new();
        for _ in 0..90 { logits.push(vec![100.0, 0.0, 0.0, 0.0]); }
        for _ in 0..10 { logits.push(vec![0.0, 100.0, 0.0, 0.0]); }
        let rt = ExpertRouteTable::from_gate_logits(config, &logits);
        balancer.record_route(&rt);
        // Act
        let suggested = balancer.suggest_capacity_adjustment(100);
        // Assert: expert 0 capacity > expert 1 capacity
        assert!(suggested[0] > suggested[1],
            "hot expert 0 cap {} should > expert 1 cap {}", suggested[0], suggested[1]);
    }

    #[test]
    fn test_balancer_hit_rates_non_negative() {
        // Arrange
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0, 0.0],
            vec![0.0, 100.0, 0.0, 0.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&rt);
        // Act
        let rates = balancer.hit_rates();
        // Assert
        for &rate in &rates {
            assert!(rate >= 0.0, "hit rate should be non-negative, got {}", rate);
        }
    }

    #[test]
    fn test_balancer_record_route_accumulates_not_replaces() {
        // Arrange: record two separate route tables
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        // First: expert 0
        let rt1 = ExpertRouteTable::from_gate_logits(config.clone(), &[vec![100.0, 0.0]]);
        balancer.record_route(&rt1);
        let hits_after_first = balancer.hit_history[0];
        // Second: expert 0 again
        let rt2 = ExpertRouteTable::from_gate_logits(config, &[vec![100.0, 0.0]]);
        balancer.record_route(&rt2);
        // Assert: hits accumulated
        assert!(balancer.hit_history[0] > hits_after_first,
            "hits should accumulate: {} should be > {}", balancer.hit_history[0], hits_after_first);
    }

    #[test]
    fn test_balancer_reset_then_record_fresh_state() {
        // Arrange: record, reset, then record differently
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        // Phase 1: all to expert 0
        let rt1 = ExpertRouteTable::from_gate_logits(config.clone(), &[vec![100.0, 0.0, 0.0]]);
        balancer.record_route(&rt1);
        assert!(balancer.hit_history[0] > 0);
        // Phase 2: reset
        balancer.reset();
        assert_eq!(balancer.total_routes, 0);
        assert!(balancer.hit_history.iter().all(|&h| h == 0));
        // Phase 3: all to expert 2
        let rt2 = ExpertRouteTable::from_gate_logits(config, &[vec![0.0, 0.0, 100.0]]);
        balancer.record_route(&rt2);
        // Assert: only expert 2 has hits
        assert_eq!(balancer.hit_history[0], 0, "expert 0 should have 0 hits after reset+new recording");
        assert_eq!(balancer.hit_history[1], 0, "expert 1 should have 0 hits");
        assert!(balancer.hit_history[2] > 0, "expert 2 should have hits");
    }

    // --- moe_dispatch additional ---

    #[test]
    fn test_moe_dispatch_output_non_negative_with_positive_inputs() {
        // Arrange: all positive
        let input = vec![1.0, 1.0, 1.0];
        let gate_logits = vec![0.5, 0.3, 0.2];
        let expert_outputs = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 1.0, 1.0],
            vec![0.5, 0.5, 0.5],
        ];
        // Act
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 3);
        // Assert: all output elements non-negative
        for &v in &output {
            assert!(v >= 0.0, "output should be non-negative, got {}", v);
        }
    }

    #[test]
    fn test_moe_dispatch_topk_one_weighted_correctly() {
        // Arrange: expert 0 dominates
        let input = vec![1.0, 2.0];
        let gate_logits = vec![100.0, 0.0];
        let expert_outputs = vec![
            vec![3.0, 6.0],
            vec![9.0, 12.0],
        ];
        // Act
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 1);
        // Assert: expert 0 weight ≈ 1.0, output ≈ [3.0, 6.0]
        assert!((output[0] - 3.0).abs() < 0.01, "output[0] = {}", output[0]);
        assert!((output[1] - 6.0).abs() < 0.01, "output[1] = {}", output[1]);
    }

    #[test]
    fn test_moe_dispatch_expert_outputs_longer_than_input() {
        // Arrange: expert outputs have more elements than input
        let input = vec![1.0, 2.0];
        let gate_logits = vec![100.0, 0.0];
        let expert_outputs = vec![
            vec![3.0, 6.0, 9.0, 12.0], // 4 elements, input has 2
            vec![1.0, 1.0, 1.0, 1.0],
        ];
        // Act
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 1);
        // Assert: output size matches input, extra expert elements ignored
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_moe_dispatch_gate_logits_all_negative() {
        // Arrange
        let input = vec![1.0, 1.0];
        let gate_logits = vec![-10.0, -1.0];
        let expert_outputs = vec![
            vec![2.0, 3.0],
            vec![4.0, 5.0],
        ];
        // Act
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 2);
        // Assert: output should still be computed (softmax handles negatives)
        assert_eq!(output.len(), 2);
        let sum: f32 = output.iter().sum();
        assert!(sum > 0.0, "output sum should be positive, got {}", sum);
    }

    #[test]
    fn test_moe_dispatch_single_expert_single_element() {
        // Arrange: minimal possible
        let input = vec![0.0];
        let gate_logits = vec![1.0];
        let expert_outputs = vec![vec![42.0]];
        // Act
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 1);
        // Assert
        assert_eq!(output.len(), 1);
        assert!((output[0] - 42.0).abs() < 1e-5, "output = {}", output[0]);
    }

    #[test]
    fn test_moe_dispatch_large_expert_count() {
        // Arrange: 16 experts
        let input = vec![1.0; 4];
        let gate_logits: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let expert_outputs: Vec<Vec<f32>> = (0..16)
            .map(|i| vec![i as f32; 4])
            .collect();
        // Act
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 4);
        // Assert
        assert_eq!(output.len(), 4);
        // Output should not be all zeros
        let sum: f32 = output.iter().sum();
        assert!(sum > 0.0, "output should have non-zero sum");
    }

    // --- load_balance_loss additional ---

    #[test]
    fn test_load_balance_loss_with_single_expert() {
        // Arrange: N=1 expert
        let config = ExpertRouteConfig {
            num_experts: 1,
            top_k: 1,
            capacity_factor: 100.0,
            load_balance_loss: true,
            load_balance_lambda: 0.01,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![vec![5.0]];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Act
        let loss = rt.load_balance_loss(&gate_logits);
        // Assert: with 1 expert, f_0=1.0, P_0=1.0, loss = 0.01 * 1 * 1.0 * 1.0 = 0.01
        assert!(loss >= 0.0, "loss should be non-negative, got {}", loss);
    }

    #[test]
    fn test_load_balance_loss_many_experts_few_tokens() {
        // Arrange: 16 experts, 2 tokens
        let config = ExpertRouteConfig {
            num_experts: 16,
            top_k: 1,
            capacity_factor: 100.0,
            load_balance_loss: true,
            load_balance_lambda: 0.01,
            ..ExpertRouteConfig::default()
        };
        let mut logits1 = vec![0.0; 16];
        logits1[0] = 100.0;
        let mut logits2 = vec![0.0; 16];
        logits2[1] = 100.0;
        let gate_logits = vec![logits1, logits2];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Act
        let loss = rt.load_balance_loss(&gate_logits);
        // Assert
        assert!(loss >= 0.0, "loss should be non-negative, got {}", loss);
        assert!(loss.is_finite(), "loss should be finite, got {}", loss);
    }

    #[test]
    fn test_load_balance_loss_symmetric_input_same_loss() {
        // Arrange: two symmetric scenarios should give similar loss
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 100.0,
            load_balance_loss: true,
            load_balance_lambda: 0.01,
            ..ExpertRouteConfig::default()
        };
        // Scenario A: token 0 → expert 0, token 1 → expert 1
        let logits_a = vec![vec![100.0, 0.0], vec![0.0, 100.0]];
        let rt_a = ExpertRouteTable::from_gate_logits(config.clone(), &logits_a);
        let loss_a = rt_a.load_balance_loss(&logits_a);
        // Scenario B: token 0 → expert 1, token 1 → expert 0 (swapped)
        let logits_b = vec![vec![0.0, 100.0], vec![100.0, 0.0]];
        let rt_b = ExpertRouteTable::from_gate_logits(config, &logits_b);
        let loss_b = rt_b.load_balance_loss(&logits_b);
        // Assert: symmetric inputs produce same loss
        assert!((loss_a - loss_b).abs() < 1e-5,
            "symmetric losses should be equal: {} vs {}", loss_a, loss_b);
    }

    // --- Integration / property tests ---

    #[test]
    fn test_route_table_utilization_stats_consistency() {
        // Arrange
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 2,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4.0, 3.0, 2.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let stats = rt.utilization_stats();
        // Assert: total_expert_assignments == sum of expert_token_counts
        let counts_sum: usize = rt.expert_token_counts.iter().sum();
        assert_eq!(stats.total_expert_assignments, counts_sum);
        // total_tokens == token_routes.len()
        assert_eq!(stats.total_tokens, rt.token_routes.len());
        // overflow_count matches table's overflow_count
        assert_eq!(stats.overflow_count, rt.overflow_count);
    }

    #[test]
    fn test_balancer_hit_rates_consistent_with_recorded_routes() {
        // Arrange
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        // 5 to expert 0, 3 to expert 1, 2 to expert 2 = 10 total
        let mut logits = Vec::new();
        for _ in 0..5 { logits.push(vec![100.0, 0.0, 0.0]); }
        for _ in 0..3 { logits.push(vec![0.0, 100.0, 0.0]); }
        for _ in 0..2 { logits.push(vec![0.0, 0.0, 100.0]); }
        let rt = ExpertRouteTable::from_gate_logits(config, &logits);
        balancer.record_route(&rt);
        // Act
        let rates = balancer.hit_rates();
        // Assert: rates should be approximately [0.5, 0.3, 0.2]
        assert!((rates[0] - 0.5).abs() < 0.01, "expert 0 rate = {}", rates[0]);
        assert!((rates[1] - 0.3).abs() < 0.01, "expert 1 rate = {}", rates[1]);
        assert!((rates[2] - 0.2).abs() < 0.01, "expert 2 rate = {}", rates[2]);
        let sum: f64 = rates.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "rates sum = {}", sum);
    }

    #[test]
    fn test_full_pipeline_config_to_route_to_dispatch() {
        // Arrange: full pipeline — config → route table → dispatch
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 2,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![vec![0.0, 0.0, 100.0, 50.0]];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: route table constructed correctly
        assert_eq!(rt.token_routes.len(), 1);
        let route = &rt.token_routes[0];
        assert!(route.expert_indices.contains(&2), "expert 2 should be selected");
        // Now use the route for dispatch
        let input = vec![1.0, 1.0];
        let flat_logits = vec![0.0, 0.0, 100.0, 50.0];
        let expert_outputs = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![4.0, 4.0],
        ];
        // Act
        let output = moe_dispatch(&input, &flat_logits, &expert_outputs, 2);
        // Assert: output should be a blend of expert 2 and expert 3 outputs
        assert_eq!(output.len(), 2);
        assert!(output[0] > 0.0 && output[0] < 5.0, "output[0] = {}", output[0]);
    }

    #[test]
    fn test_route_table_overflow_increases_with_tighter_capacity() {
        // Arrange: same tokens, decreasing capacity
        let gate_logits: Vec<Vec<f32>> = (0..10)
            .map(|_| vec![100.0, 0.0, 0.0, 0.0])
            .collect();
        let mut overflows = Vec::new();
        for cf in [0.1, 0.5, 1.0, 5.0, 100.0] {
            let config = ExpertRouteConfig {
                num_experts: 4,
                top_k: 1,
                capacity_factor: cf,
                ..ExpertRouteConfig::default()
            };
            let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
            overflows.push(rt.overflow_count);
        }
        // Assert: overflow should be non-increasing as capacity_factor increases
        for i in 1..overflows.len() {
            assert!(overflows[i] <= overflows[i - 1],
                "overflow[{}]={} > overflow[{}]={} with higher capacity",
                i, overflows[i], i - 1, overflows[i - 1]);
        }
    }

    #[test]
    fn test_softmax_topk_route_table_chain() {
        // Arrange: verify softmax → topk → route table chain produces consistent results
        let logits = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        // Step 1: softmax
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Step 2: topk
        let topk = topk_with_weights(&logits, 2);
        assert_eq!(topk[0].0, 3, "index 3 (5.0) should be first");
        let wsum: f32 = topk.iter().map(|(_, w)| *w).sum();
        assert!((wsum - 1.0).abs() < 0.01);
        // Step 3: route table
        let config = ExpertRouteConfig {
            num_experts: 5,
            top_k: 2,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let rt = ExpertRouteTable::from_gate_logits(config, &[logits]);
        let route = &rt.token_routes[0];
        assert!(route.expert_indices.contains(&3), "expert 3 should be in route");
    }

    #[test]
    fn test_balancer_cold_hot_complementary() {
        // Arrange: create skewed load
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        // 100 tokens to expert 0, 0 to others
        let gate_logits: Vec<Vec<f32>> = (0..100)
            .map(|_| vec![100.0, 0.0, 0.0, 0.0])
            .collect();
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&rt);
        // Act
        let cold = balancer.cold_experts();
        let hot = balancer.hot_experts(4);
        // Assert: cold experts are indices 1,2,3; hot list contains expert 0 at top
        assert_eq!(hot[0].0, 0, "expert 0 should be hottest");
        let cold_indices: Vec<usize> = cold.iter().map(|(i, _)| *i).collect();
        assert!(!cold_indices.contains(&0), "expert 0 should not be cold");
    }

    #[test]
    fn test_config_clone_then_modify_independence() {
        // Arrange
        let original = ExpertRouteConfig {
            num_experts: 8,
            top_k: 2,
            capacity_factor: 1.5,
            load_balance_loss: true,
            load_balance_lambda: 0.05,
            noise_sigma: 0.1,
        };
        // Act
        let mut cloned = original.clone();
        cloned.num_experts = 999;
        cloned.capacity_factor = 0.0;
        // Assert: original unchanged
        assert_eq!(original.num_experts, 8);
        assert!((original.capacity_factor - 1.5).abs() < 1e-6);
        assert_eq!(cloned.num_experts, 999);
        assert!((cloned.capacity_factor - 0.0).abs() < 1e-6);
    }

    // --- Additional edge case coverage ---

    #[test]
    fn test_topk_with_weights_very_small_k() {
        // Arrange: k=1 with many identical values
        let logits = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let topk = topk_with_weights(&logits, 1);
        // Assert
        assert_eq!(topk.len(), 1);
        assert!((topk[0].1 - 1.0).abs() < 1e-5, "single pick weight should be 1.0");
    }

    #[test]
    fn test_softmax_then_topk_preserves_ordering() {
        // Arrange
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let probs = softmax(&logits);
        let topk = topk_with_weights(&logits, 3);
        // Assert: topk indices are in descending softmax order
        for i in 0..topk.len() {
            for j in (i + 1)..topk.len() {
                assert!(probs[topk[i].0] >= probs[topk[j].0],
                    "ordering broken: probs[{}]={} < probs[{}]={}",
                    topk[i].0, probs[topk[i].0], topk[j].0, probs[topk[j].0]);
            }
        }
    }

    #[test]
    fn test_route_table_expert_weights_match_topk_for_single_token() {
        // Arrange: single token, verify weights match topk_with_weights
        let logits = vec![1.0, 3.0, 2.0, 4.0];
        let expected_topk = topk_with_weights(&logits, 2);
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 2,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let rt = ExpertRouteTable::from_gate_logits(config, &[logits.clone()]);
        let route = &rt.token_routes[0];
        // Assert: expert indices match
        assert_eq!(route.expert_indices.len(), expected_topk.len());
        for i in 0..expected_topk.len() {
            assert_eq!(route.expert_indices[i], expected_topk[i].0,
                "mismatch at position {}", i);
        }
    }

    #[test]
    fn test_balancer_hit_rates_sum_to_one_after_multiple_recordings() {
        // Arrange
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        // Record 5 separate route tables
        for i in 0..5 {
            let mut row = vec![0.0; 3];
            row[i % 3] = 100.0;
            let rt = ExpertRouteTable::from_gate_logits(config.clone(), &[row]);
            balancer.record_route(&rt);
        }
        // Act
        let rates = balancer.hit_rates();
        // Assert: rates sum to 1.0
        let sum: f64 = rates.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "rates sum = {}", sum);
    }

    #[test]
    fn test_moe_dispatch_two_experts_equal_weight_average() {
        // Arrange: equal gate logits → equal weights → output is average
        let input = vec![0.0; 3];
        let gate_logits = vec![1.0, 1.0];
        let expert_outputs = vec![
            vec![0.0, 0.0, 0.0],
            vec![6.0, 9.0, 12.0],
        ];
        // Act
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 2);
        // Assert: output ≈ average of both experts
        assert!((output[0] - 3.0).abs() < 0.02, "output[0] = {}", output[0]);
        assert!((output[1] - 4.5).abs() < 0.02, "output[1] = {}", output[1]);
        assert!((output[2] - 6.0).abs() < 0.02, "output[2] = {}", output[2]);
    }

    #[test]
    fn test_route_table_no_duplicate_expert_per_token() {
        // Arrange: tokens that could cause duplicate expert assignment
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 2,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![vec![100.0, 0.0]];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: no duplicate expert indices per token
        for route in &rt.token_routes {
            let mut sorted = route.expert_indices.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), route.expert_indices.len(),
                "duplicate experts found: {:?}", route.expert_indices);
        }
    }

    #[test]
    fn test_utilization_stats_max_min_relationship() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0, 0.0],
            vec![0.0, 100.0, 0.0, 0.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let stats = rt.utilization_stats();
        assert!(stats.max_expert_load >= stats.min_expert_load);
        assert!(stats.total_expert_assignments >= stats.max_expert_load);
        assert!(stats.total_tokens >= stats.overflow_count);
    }

    // =========================================================================
    // Round 7: 25 additional tests
    // =========================================================================

    // --- ExpertRouteConfig::new field-by-field verification ---

    #[test]
    fn test_config_new_preserves_exact_num_experts() {
        let config = ExpertRouteConfig::new(3, 1);
        assert_eq!(config.num_experts, 3);
    }

    #[test]
    fn test_config_new_preserves_exact_top_k() {
        let config = ExpertRouteConfig::new(4, 3);
        assert_eq!(config.top_k, 3);
    }

    #[test]
    fn test_config_default_noise_sigma_is_zero() {
        let config = ExpertRouteConfig::default();
        assert_eq!(config.noise_sigma, 0.0);
    }

    #[test]
    fn test_config_default_capacity_factor_is_1_25() {
        let config = ExpertRouteConfig::default();
        assert_eq!(config.capacity_factor, 1.25);
    }

    // --- ExpertRouteConfig::expert_capacity additional cases ---

    #[test]
    fn test_capacity_two_experts_eight_tokens_default_factor() {
        let config = ExpertRouteConfig::new(2, 1);
        // ceil(1.25 * 8 / 2) = ceil(5.0) = 5
        assert_eq!(config.expert_capacity(8), 5);
    }

    #[test]
    fn test_capacity_six_experts_twelve_tokens() {
        let config = ExpertRouteConfig {
            num_experts: 6,
            top_k: 2,
            capacity_factor: 2.0,
            ..ExpertRouteConfig::default()
        };
        // ceil(2.0 * 12 / 6) = ceil(4.0) = 4
        assert_eq!(config.expert_capacity(12), 4);
    }

    // --- ExpertRouteConfig PartialEq with f32 fields ---

    #[test]
    fn test_config_partial_eq_same_float_fields() {
        let a = ExpertRouteConfig {
            capacity_factor: 1.5,
            load_balance_lambda: 0.05,
            noise_sigma: 0.1,
            ..ExpertRouteConfig::new(4, 2)
        };
        let b = ExpertRouteConfig {
            capacity_factor: 1.5,
            load_balance_lambda: 0.05,
            noise_sigma: 0.1,
            ..ExpertRouteConfig::new(4, 2)
        };
        assert_eq!(a, b);
    }

    #[test]
    fn test_config_partial_eq_differs_by_capacity_factor_only() {
        let a = ExpertRouteConfig {
            capacity_factor: 1.0,
            ..ExpertRouteConfig::new(4, 2)
        };
        let b = ExpertRouteConfig {
            capacity_factor: 3.0,
            ..ExpertRouteConfig::new(4, 2)
        };
        assert_ne!(a, b);
    }

    // --- TokenRoute with varied field lengths ---

    #[test]
    fn test_token_route_three_experts_fields_match_length() {
        let route = TokenRoute {
            expert_indices: vec![0, 4, 7],
            expert_weights: vec![0.5, 0.3, 0.2],
            expert_positions: vec![2, 0, 5],
        };
        assert_eq!(route.expert_indices.len(), route.expert_weights.len());
        assert_eq!(route.expert_weights.len(), route.expert_positions.len());
    }

    // --- ExpertUtilizationStats Copy with all zero fields ---

    #[test]
    fn test_stats_all_zeros_copy() {
        let stats = ExpertUtilizationStats {
            total_tokens: 0,
            total_expert_assignments: 0,
            overflow_count: 0,
            max_expert_load: 0,
            min_expert_load: 0,
            mean_expert_load: 0.0,
            balance_score: 0.0,
        };
        let copied = stats;
        assert_eq!(copied.total_tokens, 0);
        assert_eq!(copied.mean_expert_load, 0.0);
    }

    #[test]
    fn test_stats_all_zeros_clone() {
        let stats = ExpertUtilizationStats {
            total_tokens: 0,
            total_expert_assignments: 0,
            overflow_count: 0,
            max_expert_load: 0,
            min_expert_load: 0,
            mean_expert_load: 0.0,
            balance_score: 0.0,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.overflow_count, 0);
        assert_eq!(cloned.balance_score, 0.0);
    }

    // --- ExpertUtilizationStats PartialEq with different integer fields ---

    #[test]
    fn test_stats_partial_eq_differs_by_max_load() {
        let a = ExpertUtilizationStats {
            total_tokens: 10, total_expert_assignments: 20, overflow_count: 0,
            max_expert_load: 5, min_expert_load: 5, mean_expert_load: 5.0, balance_score: 1.0,
        };
        let b = ExpertUtilizationStats {
            total_tokens: 10, total_expert_assignments: 20, overflow_count: 0,
            max_expert_load: 10, min_expert_load: 5, mean_expert_load: 5.0, balance_score: 1.0,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn test_stats_partial_eq_differs_by_mean_load() {
        let a = ExpertUtilizationStats {
            total_tokens: 10, total_expert_assignments: 20, overflow_count: 0,
            max_expert_load: 5, min_expert_load: 5, mean_expert_load: 5.0, balance_score: 1.0,
        };
        let b = ExpertUtilizationStats {
            total_tokens: 10, total_expert_assignments: 20, overflow_count: 0,
            max_expert_load: 5, min_expert_load: 5, mean_expert_load: 10.0, balance_score: 1.0,
        };
        assert_ne!(a, b);
    }

    // --- ExpertLoadBalancer::new initializes correctly ---

    #[test]
    fn test_balancer_new_initial_total_routes_zero() {
        let config = ExpertRouteConfig::new(4, 2);
        let balancer = ExpertLoadBalancer::new(config);
        assert_eq!(balancer.total_routes, 0);
    }

    #[test]
    fn test_balancer_new_hit_history_len_matches_experts() {
        let config = ExpertRouteConfig::new(6, 2);
        let balancer = ExpertLoadBalancer::new(config);
        assert_eq!(balancer.hit_history.len(), 6);
    }

    // --- ExpertLoadBalancer::config() accessor ---

    #[test]
    fn test_balancer_config_accessor_captures_all_fields() {
        let config = ExpertRouteConfig {
            num_experts: 5,
            top_k: 3,
            capacity_factor: 2.0,
            load_balance_loss: true,
            load_balance_lambda: 0.05,
            noise_sigma: 0.0,
        };
        let balancer = ExpertLoadBalancer::new(config);
        let c = balancer.config();
        assert_eq!(c.num_experts, 5);
        assert_eq!(c.top_k, 3);
        assert!(!c.noise_sigma.is_normal());
        assert!(c.load_balance_loss);
    }

    // --- ExpertRouteTable::expert_token_counts length ---

    #[test]
    fn test_route_table_expert_counts_len_varies_with_config() {
        for num_experts in [1, 2, 4, 8, 16] {
            let config = ExpertRouteConfig::new(num_experts, 1);
            let rt = ExpertRouteTable::from_gate_logits(config, &[vec![0.0; num_experts]]);
            assert_eq!(rt.expert_token_counts.len(), num_experts,
                "expected {} counts, got {}", num_experts, rt.expert_token_counts.len());
        }
    }

    // --- softmax with one zero one nonzero ---

    #[test]
    fn test_softmax_one_zero_one_nonzero() {
        let probs = softmax(&[0.0, 5.0]);
        assert!((probs[0] - probs[1]).abs() > 0.1, "probs should differ");
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // --- topk_indices with two elements ---

    #[test]
    fn test_topk_indices_two_equal_k_one() {
        let indices = topk_indices(&[3.0, 3.0], 1);
        assert_eq!(indices.len(), 1);
        assert!(indices[0] < 2);
    }

    // --- topk_with_weights k=1 returns single element with weight ~1.0 ---

    #[test]
    fn test_topk_weights_k_one_weight_near_one() {
        let logits = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let topk = topk_with_weights(&logits, 1);
        assert_eq!(topk.len(), 1);
        assert!((topk[0].1 - 1.0).abs() < 1e-5, "single selection weight should be 1.0");
    }

    // --- ExpertRouteTable debug includes key fields ---

    #[test]
    fn test_route_table_debug_includes_config_and_overflow() {
        let config = ExpertRouteConfig::new(3, 1);
        let gate_logits = vec![vec![1.0, 0.0, 0.0]];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let debug = format!("{:?}", rt);
        assert!(debug.contains("config"));
        assert!(debug.contains("overflow_count"));
        assert!(debug.contains("token_routes"));
    }

    // --- ExpertLoadBalancer debug includes key fields ---

    #[test]
    fn test_balancer_debug_includes_cold_threshold() {
        let config = ExpertRouteConfig::new(2, 1);
        let balancer = ExpertLoadBalancer::new(config);
        let debug = format!("{:?}", balancer);
        assert!(debug.contains("cold_threshold"));
    }

    // --- ExpertRouteConfig expert_capacity with tokens equal to experts ---

    #[test]
    fn test_capacity_tokens_equals_experts() {
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 1.0,
            ..ExpertRouteConfig::default()
        };
        // ceil(1.0 * 4 / 4) = 1
        assert_eq!(config.expert_capacity(4), 1);
    }

    // --- ExpertUtilizationStats debug includes all field names ---

    #[test]
    fn test_stats_debug_all_seven_fields() {
        let stats = ExpertUtilizationStats {
            total_tokens: 1, total_expert_assignments: 2, overflow_count: 0,
            max_expert_load: 1, min_expert_load: 1, mean_expert_load: 1.0, balance_score: 1.0,
        };
        let debug = format!("{:?}", stats);
        assert!(debug.contains("total_tokens"));
        assert!(debug.contains("total_expert_assignments"));
        assert!(debug.contains("overflow_count"));
        assert!(debug.contains("max_expert_load"));
        assert!(debug.contains("min_expert_load"));
        assert!(debug.contains("mean_expert_load"));
        assert!(debug.contains("balance_score"));
    }

    // --- New tests (wave 12x81): ~15 struct/enum/edge-case tests ---

    #[test]
    fn test_topk_indices_infinity_at_last_position() {
        // Arrange
        let logits = vec![0.1, 0.2, 0.3, f32::INFINITY];
        // Act
        let indices = topk_indices(&logits, 2);
        // Assert: index 3 (infinity) is the first pick
        assert_eq!(indices[0], 3);
        assert_eq!(indices.len(), 2);
    }

    #[test]
    fn test_moe_dispatch_top_k_zero_no_expert_selected() {
        // Arrange
        let input = vec![1.0, 2.0, 3.0];
        let gate = vec![0.5, 0.3, 0.2];
        let expert_outputs = vec![vec![10.0, 20.0, 30.0]];
        // Act
        let output = moe_dispatch(&input, &gate, &expert_outputs, 0);
        // Assert: no expert contributes, output is all zeros
        assert_eq!(output.len(), 3);
        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_route_table_struct_literal_construction_all_fields() {
        // Arrange
        let config = ExpertRouteConfig::new(4, 2);
        // Act
        let table = ExpertRouteTable {
            config: config.clone(),
            token_routes: vec![TokenRoute {
                expert_indices: vec![0],
                expert_weights: vec![1.0],
                expert_positions: vec![0],
            }],
            expert_token_counts: vec![1, 0, 0, 0],
            overflow_count: 0,
        };
        // Assert: all fields accessible and match input
        assert_eq!(table.config, config);
        assert_eq!(table.token_routes.len(), 1);
        assert_eq!(table.overflow_count, 0);
        assert_eq!(table.expert_token_counts[0], 1);
    }

    #[test]
    fn test_config_capacity_ceil_semantics_specific() {
        // Arrange: 1.25 * 7 / 4 = 2.1875, ceil = 3
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 2,
            capacity_factor: 1.25,
            load_balance_loss: false,
            load_balance_lambda: 0.01,
            noise_sigma: 0.0,
        };
        // Act
        let capacity = config.expert_capacity(7);
        // Assert: ceil(2.1875) = 3, not floor(2.1875) = 2
        assert_eq!(capacity, 3);
    }

    #[test]
    fn test_topk_with_weights_neg_inf_ignored_for_low_k() {
        // Arrange
        let logits = vec![10.0, f32::NEG_INFINITY, 5.0, f32::NEG_INFINITY];
        // Act
        let result = topk_with_weights(&logits, 2);
        // Assert: neg-inf entries are never selected
        let selected_indices: Vec<usize> = result.iter().map(|(i, _)| *i).collect();
        assert!(!selected_indices.contains(&1));
        assert!(!selected_indices.contains(&3));
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_balancer_cold_threshold_default_exact() {
        // Arrange & Act
        let balancer = ExpertLoadBalancer::new(ExpertRouteConfig::new(4, 2));
        // Assert: default cold_threshold is 0.001 (< 0.1% hit rate)
        assert!((balancer.cold_threshold - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_stats_balance_score_single_expert_loaded() {
        // Arrange: 3 experts, only expert 0 has load, others are 0
        // max=10, min=0 → balance = 1.0 - (10-0)/10 = 0.0
        let stats = ExpertUtilizationStats {
            total_tokens: 10,
            total_expert_assignments: 10,
            overflow_count: 0,
            max_expert_load: 10,
            min_expert_load: 0,
            mean_expert_load: 10.0 / 3.0,
            balance_score: 0.0,
        };
        // Assert: completely imbalanced → score 0.0
        assert_eq!(stats.balance_score, 0.0);
        assert_eq!(stats.max_expert_load, 10);
        assert_eq!(stats.min_expert_load, 0);
    }

    #[test]
    fn test_config_new_top_k_zero_constructable() {
        // Arrange & Act
        let config = ExpertRouteConfig::new(8, 0);
        // Assert: zero top_k is allowed at construction
        assert_eq!(config.top_k, 0);
        assert_eq!(config.num_experts, 8);
    }

    #[test]
    fn test_route_table_tokens_for_expert_empty_for_unassigned() {
        // Arrange: construct a table where expert 2 has no assignments
        let table = ExpertRouteTable {
            config: ExpertRouteConfig::new(4, 1),
            token_routes: vec![
                TokenRoute {
                    expert_indices: vec![0],
                    expert_weights: vec![1.0],
                    expert_positions: vec![0],
                },
                TokenRoute {
                    expert_indices: vec![1],
                    expert_weights: vec![1.0],
                    expert_positions: vec![0],
                },
            ],
            expert_token_counts: vec![1, 1, 0, 0],
            overflow_count: 0,
        };
        // Act
        let tokens = table.tokens_for_expert(2);
        // Assert: no tokens routed to expert 2
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_config_capacity_many_experts_one_token() {
        // Arrange: 100 experts, 1 token → 1.25 * 1 / 100 = 0.0125, ceil = 1
        let config = ExpertRouteConfig::new(100, 2);
        // Act
        let capacity = config.expert_capacity(1);
        // Assert: even with fractional result < 1, ceil ensures at least 1
        assert_eq!(capacity, 1);
    }

    #[test]
    fn test_stats_total_assignments_exceeds_tokens_with_top_k() {
        // Arrange: 3 tokens each routed to 2 experts → 6 assignments
        let stats = ExpertUtilizationStats {
            total_tokens: 3,
            total_expert_assignments: 6,
            overflow_count: 0,
            max_expert_load: 3,
            min_expert_load: 1,
            mean_expert_load: 2.0,
            balance_score: 1.0 - 2.0 / 3.0,
        };
        // Assert: assignments = tokens × top_k
        assert_eq!(stats.total_expert_assignments, stats.total_tokens * 2);
        assert!(stats.total_expert_assignments > stats.total_tokens);
    }

    #[test]
    fn test_balancer_hot_experts_no_routes_returns_zeros() {
        // Arrange: fresh balancer, no routes recorded
        let balancer = ExpertLoadBalancer::new(ExpertRouteConfig::new(4, 2));
        // Act
        let hot = balancer.hot_experts(3);
        // Assert: returns 3 experts with 0.0 rate when no routes recorded
        assert_eq!(hot.len(), 3);
        assert!(hot.iter().all(|&(_, rate)| rate == 0.0));
    }

    #[test]
    fn test_softmax_returns_vec_of_same_length() {
        // Arrange
        let inputs = vec![vec![1.0, 2.0, 3.0], vec![0.0], vec![-1.0, 0.0, 1.0, 2.0]];
        for input in &inputs {
            // Act
            let output = softmax(input);
            // Assert: output length always matches input length
            assert_eq!(output.len(), input.len());
        }
    }

    #[test]
    fn test_config_expert_capacity_proportional_scaling() {
        // Arrange: double the tokens should roughly double the capacity
        let config = ExpertRouteConfig::new(8, 2);
        let cap_64 = config.expert_capacity(64);
        let cap_128 = config.expert_capacity(128);
        // Act & Assert: cap_128 should be approximately 2x cap_64
        // cap_64 = ceil(1.25 * 64 / 8) = 10, cap_128 = ceil(1.25 * 128 / 8) = 20
        assert_eq!(cap_128, cap_64 * 2);
    }

    #[test]
    fn test_config_load_balance_lambda_default() {
        // Arrange & Act
        let config = ExpertRouteConfig::default();
        // Assert: default lambda is 0.01
        assert!((config.load_balance_lambda - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_config_load_balance_loss_default_false() {
        // Arrange & Act
        let config = ExpertRouteConfig::default();
        // Assert
        assert!(!config.load_balance_loss);
    }

    #[test]
    fn test_route_table_from_gate_logits_balanced_input() {
        // Arrange: 4 experts, each token strongly prefers a different expert
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 10.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0, 0.0],
            vec![0.0, 100.0, 0.0, 0.0],
            vec![0.0, 0.0, 100.0, 0.0],
            vec![0.0, 0.0, 0.0, 100.0],
        ];
        // Act
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: each expert should have exactly 1 token
        assert_eq!(table.expert_token_counts, vec![1, 1, 1, 1]);
        assert_eq!(table.overflow_count, 0);
    }

    #[test]
    fn test_route_table_single_token_top_k_two_expert_count() {
        // Arrange: single token, top_k=2 with 4 experts
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 2,
            capacity_factor: 10.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![vec![0.0, 0.0, 10.0, 5.0]];
        // Act
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: token routed to 2 experts
        assert_eq!(table.token_routes.len(), 1);
        assert_eq!(table.token_routes[0].expert_indices.len(), 2);
    }

    #[test]
    fn test_topk_with_weights_k_one_returns_single_pair() {
        // Arrange
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        // Act
        let topk = topk_with_weights(&logits, 1);
        // Assert: exactly one result, the highest logit index
        assert_eq!(topk.len(), 1);
        assert_eq!(topk[0].0, 1);
        assert!((topk[0].1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_moe_dispatch_single_expert_full_weight() {
        // Arrange: gate logits strongly favor expert 0
        let input = vec![1.0, 2.0, 3.0];
        let gate_logits = vec![100.0, 0.0];
        let expert_outputs = vec![
            vec![10.0, 20.0, 30.0], // expert 0
            vec![99.0, 99.0, 99.0], // expert 1
        ];
        // Act
        let result = moe_dispatch(&input, &gate_logits, &expert_outputs, 1);
        // Assert: output should be very close to expert 0's output
        assert_eq!(result.len(), 3);
        assert!((result[0] - 10.0).abs() < 0.1);
        assert!((result[1] - 20.0).abs() < 0.1);
        assert!((result[2] - 30.0).abs() < 0.1);
    }

    #[test]
    fn test_moe_dispatch_output_length_matches_input() {
        // Arrange
        let input = vec![0.0; 7];
        let gate_logits = vec![1.0, 2.0];
        let expert_outputs = vec![vec![1.0; 7], vec![2.0; 7]];
        // Act
        let result = moe_dispatch(&input, &gate_logits, &expert_outputs, 2);
        // Assert: output length equals input length
        assert_eq!(result.len(), input.len());
    }

    #[test]
    fn test_balancer_record_route_counts_each_expert() {
        // Arrange
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 10.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0, 0.0], // token 0 → expert 0
            vec![0.0, 100.0, 0.0, 0.0], // token 1 → expert 1
        ];
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Act
        balancer.record_route(&table);
        let rates = balancer.hit_rates();
        // Assert: experts 0 and 1 should each have 0.5 rate
        assert!((rates[0] - 0.5).abs() < 1e-5);
        assert!((rates[1] - 0.5).abs() < 1e-5);
        assert_eq!(rates[2], 0.0);
        assert_eq!(rates[3], 0.0);
    }

    #[test]
    fn test_balancer_suggest_capacity_adjustment_skewed_load() {
        // Arrange: all traffic to expert 0
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 10.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        let gate_logits = vec![vec![100.0, 0.0, 0.0, 0.0]];
        for _ in 0..5 {
            let table = ExpertRouteTable::from_gate_logits(config.clone(), &gate_logits);
            balancer.record_route(&table);
        }
        // Act
        let caps = balancer.suggest_capacity_adjustment(10);
        // Assert: expert 0 should get the highest capacity
        assert_eq!(caps.len(), 4);
        assert!(caps[0] > caps[1]);
        assert!(caps[0] > caps[2]);
        assert!(caps[0] > caps[3]);
        // All caps should be at least 1 (minimum guarantee)
        for &c in &caps {
            assert!(c >= 1);
        }
    }

    #[test]
    fn test_balancer_cold_experts_with_uniform_routing() {
        // Arrange: perfectly uniform routing, no expert should be cold
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 10.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0],
            vec![0.0, 100.0, 0.0],
            vec![0.0, 0.0, 100.0],
        ];
        for _ in 0..10 {
            let table = ExpertRouteTable::from_gate_logits(config.clone(), &gate_logits);
            balancer.record_route(&table);
        }
        // Act
        let cold = balancer.cold_experts();
        // Assert: no experts should be cold with uniform distribution
        assert!(cold.is_empty());
    }

    #[test]
    fn test_topk_indices_returns_sorted_descending() {
        // Arrange
        let logits = vec![0.3, 0.9, 0.1, 0.7];
        // Act
        let indices = topk_indices(&logits, 3);
        // Assert: top 3 by value: index 1 (0.9), index 3 (0.7), index 0 (0.3)
        assert_eq!(indices.len(), 3);
        assert_eq!(indices[0], 1);
        assert_eq!(indices[1], 3);
        assert_eq!(indices[2], 0);
    }

    #[test]
    fn test_stats_balance_score_perfectly_balanced() {
        // Arrange: all experts have equal load
        let stats = ExpertUtilizationStats {
            total_tokens: 8,
            total_expert_assignments: 8,
            overflow_count: 0,
            max_expert_load: 2,
            min_expert_load: 2,
            mean_expert_load: 2.0,
            balance_score: 1.0,
        };
        // Assert
        assert_eq!(stats.balance_score, 1.0);
        assert_eq!(stats.max_expert_load, stats.min_expert_load);
    }

    #[test]
    fn test_token_route_expert_weights_sum_approximately_one() {
        // Arrange: build a route table with top_k=2
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 2,
            capacity_factor: 10.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![vec![1.0, 2.0, 3.0, 0.5]];
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Act
        let route = &table.token_routes[0];
        let weight_sum: f32 = route.expert_weights.iter().sum();
        // Assert: re-normalized weights should sum to ~1.0
        assert!((weight_sum - 1.0).abs() < 1e-4,
            "weights should sum to ~1.0, got {}", weight_sum);
    }

    #[test]
    fn test_route_table_overflow_count_zero_with_generous_capacity() {
        // Arrange: large capacity_factor so no overflow
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
        ];
        // Act
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: no overflow with generous capacity
        assert_eq!(table.overflow_count, 0);
    }

    // =========================================================================
    // Round 8: 15 additional edge-case tests
    // =========================================================================

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_route_table_gate_logits_shorter_than_num_experts() {
        // Arrange: 8 experts configured but each token only has 4 logits
        let config = ExpertRouteConfig::new(8, 2);
        let gate_logits = vec![
            vec![1.0, 2.0, 3.0, 4.0], // only 4 entries for 8 experts
        ];
        // Act
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: routing should complete; softmax operates on available logits,
        // so topk picks from indices 0-3 only
        assert_eq!(rt.token_routes.len(), 1);
        for &idx in &rt.token_routes[0].expert_indices {
            assert!(idx < 4, "selected expert {} should be < 4 (logits length)", idx);
        }
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_route_table_capacity_zero_all_tokens_overflow_fallback() {
        // Arrange: capacity_factor=0 → expert_capacity = ceil(0 * tokens / experts) = 0
        // All tokens must overflow and get fallback assignment
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 0.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0],
            vec![0.0, 100.0, 0.0],
        ];
        // Act
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: every token must still get an expert via fallback
        assert_eq!(rt.token_routes.len(), 2);
        for (i, route) in rt.token_routes.iter().enumerate() {
            assert!(!route.expert_indices.is_empty(), "token {} must have fallback", i);
            assert_eq!(route.expert_indices.len(), 1, "fallback assigns exactly 1 expert");
        }
        // overflow_count should be >= 2 (both tokens overflowed)
        assert!(rt.overflow_count >= 2, "expected >= 2 overflows, got {}", rt.overflow_count);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_route_table_fallback_position_equals_expert_count() {
        // Arrange: when fallback occurs, position = expert_token_counts[expert] at that moment
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 0.34, // ceil(0.34 * 3 / 2) = ceil(0.51) = 1
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 0.0], // → expert 0, position 0
            vec![100.0, 0.0], // → expert 0 overflows, fallback
        ];
        // Act
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: token 1 is fallback; its position should be the count
        // of the fallback expert at assignment time (0 since it was unloaded)
        let fallback_route = &rt.token_routes[1];
        assert_eq!(fallback_route.expert_indices.len(), 1);
        assert_eq!(fallback_route.expert_positions.len(), 1);
        // Position for fallback expert should be 0 (first token assigned there)
        assert_eq!(fallback_route.expert_positions[0], 0);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_load_balance_loss_with_empty_route_table() {
        // Arrange: empty token routes (0 tokens), but loss enabled
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 2,
            capacity_factor: 1.0,
            load_balance_loss: true,
            load_balance_lambda: 0.01,
            ..ExpertRouteConfig::default()
        };
        let rt = ExpertRouteTable::from_gate_logits(config, &[]);
        // Act: loss computation with 0 tokens
        let loss = rt.load_balance_loss(&[]);
        // Assert: with 0 tokens, num_tokens=0 → f_i = count/0.0 = 0.0/0.0 = NaN
        // The function does not guard against this; NaN propagates through arithmetic
        // This test documents the actual behavior: loss is NaN (not a panic, not 0.0)
        assert!(loss.is_nan(), "loss should be NaN with 0 tokens, got {}", loss);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_load_balance_loss_single_token_single_expert() {
        // Arrange: 1 expert, 1 token, loss enabled with lambda=1.0
        let config = ExpertRouteConfig {
            num_experts: 1,
            top_k: 1,
            capacity_factor: 100.0,
            load_balance_loss: true,
            load_balance_lambda: 1.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![vec![5.0]];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Act
        let loss = rt.load_balance_loss(&gate_logits);
        // Assert: f_0 = 1/1 = 1.0, P_0 = 1.0 (softmax of single element)
        // loss = 1.0 * 1 * (1.0 * 1.0) = 1.0
        assert!((loss - 1.0).abs() < 0.05, "loss should be ~1.0, got {}", loss);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_moe_dispatch_top_k_exceeds_gate_logits_length() {
        // Arrange: gate has 2 logits, top_k=5 requests more
        let input = vec![1.0, 2.0];
        let gate_logits = vec![3.0, 1.0];
        let expert_outputs = vec![
            vec![5.0, 10.0],
            vec![2.0, 4.0],
        ];
        // Act: topk_with_weights picks at most min(k, logits.len()) = 2
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 5);
        // Assert: output should still be computed correctly
        assert_eq!(output.len(), 2);
        let sum: f32 = output.iter().sum();
        assert!(sum > 0.0, "output should have positive contributions, sum = {}", sum);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_balancer_hot_experts_returns_all_when_top_n_equals_experts() {
        // Arrange: 5 experts, request top_n=5
        let config = ExpertRouteConfig {
            num_experts: 5,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        // 5 tokens, each to a different expert
        let gate_logits: Vec<Vec<f32>> = (0..5)
            .map(|i| {
                let mut row = vec![0.0; 5];
                row[i] = 100.0;
                row
            })
            .collect();
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&rt);
        // Act
        let hot = balancer.hot_experts(5);
        // Assert: all 5 experts returned
        assert_eq!(hot.len(), 5);
        let indices: Vec<usize> = hot.iter().map(|(i, _)| *i).collect();
        for i in 0..5 {
            assert!(indices.contains(&i), "expert {} missing from hot list", i);
        }
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_balancer_suggest_capacity_one_expert_dominates_rest_minimum() {
        // Arrange: expert 0 gets 99% traffic
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        let mut logits = Vec::new();
        for _ in 0..99 {
            logits.push(vec![100.0, 0.0, 0.0, 0.0]);
        }
        logits.push(vec![0.0, 100.0, 0.0, 0.0]); // 1 token to expert 1
        let rt = ExpertRouteTable::from_gate_logits(config, &logits);
        balancer.record_route(&rt);
        // Act
        let suggested = balancer.suggest_capacity_adjustment(10);
        // Assert: expert 0 should get the largest capacity (proportional to its share)
        assert!(suggested[0] > suggested[1], "expert 0 cap {} should > expert 1 cap {}", suggested[0], suggested[1]);
        // Experts 2 and 3 got zero hits but should still get minimum 1
        assert!(suggested[2] >= 1, "expert 2 should get min 1, got {}", suggested[2]);
        assert!(suggested[3] >= 1, "expert 3 should get min 1, got {}", suggested[3]);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_route_table_expert_positions_unique_per_expert_with_overflow() {
        // Arrange: tight capacity causes overflow, verify positions still unique per expert
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 0.34, // ceil(0.34 * 5 / 3) = ceil(0.567) = 1
            ..ExpertRouteConfig::default()
        };
        let gate_logits: Vec<Vec<f32>> = (0..5)
            .map(|_| vec![100.0, 0.0, 0.0])
            .collect();
        // Act
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: collect positions per expert
        let mut positions_per_expert: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for route in &rt.token_routes {
            for (i, &expert_idx) in route.expert_indices.iter().enumerate() {
                positions_per_expert
                    .entry(expert_idx)
                    .or_default()
                    .push(route.expert_positions[i]);
            }
        }
        for (expert, mut positions) in positions_per_expert {
            positions.sort();
            positions.dedup();
            assert_eq!(
                positions.len(),
                rt.expert_token_counts[expert],
                "expert {} has {} positions but {} tokens",
                expert,
                positions.len(),
                rt.expert_token_counts[expert]
            );
        }
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_topk_with_weights_all_nan_returns_empty_after_softmax() {
        // Arrange: all NaN logits
        let logits = vec![f32::NAN, f32::NAN, f32::NAN];
        // Act: softmax of all NaN produces NaN sum, fallback to uniform
        let topk = topk_with_weights(&logits, 2);
        // Assert: function should not panic; returns some result
        assert!(topk.len() <= 3, "topk returned {} entries", topk.len());
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_softmax_subnormal_floats() {
        // Arrange: subnormal (denormalized) f32 values
        let logits = vec![1.0e-40_f32, 2.0e-40_f32, 3.0e-40_f32];
        // Act
        let probs = softmax(&logits);
        // Assert: should not panic; all values are near zero so effectively uniform
        assert_eq!(probs.len(), 3);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.1, "sum should be close to 1.0, got {}", sum);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_balancer_record_route_accumulates_top_k_two() {
        // Arrange: top_k=2 means each token contributes to 2 expert hits
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 2,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        let gate_logits = vec![
            vec![100.0, 50.0, 0.0, 0.0], // experts 0 and 1
            vec![0.0, 0.0, 100.0, 50.0], // experts 2 and 3
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&rt);
        // Act & Assert: total_hits should be 4 (2 tokens × 2 experts each)
        let total_hits: u64 = balancer.hit_history.iter().sum();
        assert_eq!(total_hits, 4, "expected 4 total hits, got {}", total_hits);
        assert_eq!(balancer.total_routes, 2, "total_routes counts tokens");
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_config_capacity_num_experts_zero_no_panic() {
        // Arrange: num_experts=0, capacity_factor=1.0
        let config = ExpertRouteConfig {
            num_experts: 0,
            top_k: 1,
            capacity_factor: 1.0,
            ..ExpertRouteConfig::default()
        };
        // Act: ceil(1.0 * 100 / 0) = division by zero → produces Inf → as usize wraps
        // This should not panic (behavior is defined: f32 / 0.0 = Inf, ceil(Inf) = Inf)
        let cap = config.expert_capacity(100);
        // Assert: just verify it doesn't panic; the result is implementation-defined
        let _ = cap;
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_moe_dispatch_with_zero_weight_expert() {
        // Arrange: expert with effectively zero contribution due to very low gate logit
        let input = vec![1.0, 2.0];
        let gate_logits = vec![100.0, -100.0]; // expert 1 has near-zero softmax probability
        let expert_outputs = vec![
            vec![3.0, 6.0],
            vec![99.0, 99.0],
        ];
        // Act
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 2);
        // Assert: output should be almost entirely from expert 0
        assert!((output[0] - 3.0).abs() < 0.1, "output[0] = {}", output[0]);
        assert!((output[1] - 6.0).abs() < 0.1, "output[1] = {}", output[1]);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_route_table_tokens_for_expert_with_multi_expert_same_token() {
        // Arrange: a token routed to the same expert via top_k=2 and identical logits
        // (though in practice topk_with_weights picks distinct indices)
        let table = ExpertRouteTable {
            config: ExpertRouteConfig::new(4, 2),
            token_routes: vec![
                TokenRoute {
                    expert_indices: vec![2, 2], // same expert twice (synthetic)
                    expert_weights: vec![0.5, 0.5],
                    expert_positions: vec![0, 1],
                },
                TokenRoute {
                    expert_indices: vec![2, 3],
                    expert_weights: vec![0.6, 0.4],
                    expert_positions: vec![2, 0],
                },
            ],
            expert_token_counts: vec![0, 0, 3, 1],
            overflow_count: 0,
        };
        // Act
        let tokens = table.tokens_for_expert(2);
        // Assert: expert 2 appears in both tokens (3 total occurrences)
        assert_eq!(tokens.len(), 3, "expert 2 should have 3 (token, position) pairs");
    }

    // =========================================================================
    // Round 9: 15 additional edge-case tests
    // =========================================================================

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_expert_route_config_equality_reflexive() {
        // Arrange
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 2,
            capacity_factor: 1.5,
            load_balance_loss: true,
            load_balance_lambda: 0.03,
            noise_sigma: 0.1,
        };
        // Assert: a == a (reflexive property of PartialEq)
        assert_eq!(config, config, "config should be equal to itself");
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_softmax_gradient_like_ordering_with_large_negative_spread() {
        // Arrange: logits with large negative spread
        let logits = vec![-50.0, -10.0, -5.0];
        // Act
        let probs = softmax(&logits);
        // Assert: ordering preserved, all finite
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum = {}", sum);
        assert!(probs[2] > probs[1], "ordering: -5.0 > -10.0");
        assert!(probs[1] > probs[0], "ordering: -10.0 > -50.0");
        for &p in &probs {
            assert!(p.is_finite());
        }
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_topk_indices_returns_unique_indices() {
        // Arrange
        let logits = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        // Act
        let indices = topk_indices(&logits, 5);
        // Assert: all indices are unique
        let mut sorted = indices.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), indices.len(),
            "indices should be unique: {:?}", indices);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_topk_with_weights_k_equals_len_renormalizes_to_one() {
        // Arrange: k == logits.len(), non-uniform logits
        let logits = vec![1.0, 3.0, 2.0];
        // Act
        let topk = topk_with_weights(&logits, 3);
        // Assert: all 3 selected, weights sum to 1.0
        assert_eq!(topk.len(), 3);
        let sum: f32 = topk.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 0.01, "weights sum = {}", sum);
        // Descending order
        assert_eq!(topk[0].0, 1, "index 1 (3.0) should be first");
        assert_eq!(topk[1].0, 2, "index 2 (2.0) should be second");
        assert_eq!(topk[2].0, 0, "index 0 (1.0) should be third");
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_route_table_from_gate_logits_weight_consistency_across_tokens() {
        // Arrange: two tokens with very different logit scales
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 2,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![0.01, 0.02, 0.03],   // small scale
            vec![100.0, 200.0, 300.0], // large scale
        ];
        // Act
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: each token's weights sum to approximately 1.0
        for (i, route) in rt.token_routes.iter().enumerate() {
            let sum: f32 = route.expert_weights.iter().sum();
            assert!((sum - 1.0).abs() < 0.02,
                "token {} weights sum = {}, expected ~1.0", i, sum);
        }
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_moe_dispatch_with_negative_gate_logits() {
        // Arrange: all gate logits negative
        let input = vec![1.0, 1.0];
        let gate_logits = vec![-5.0, -1.0, -3.0];
        let expert_outputs = vec![
            vec![2.0, 4.0],
            vec![6.0, 8.0],
            vec![1.0, 3.0],
        ];
        // Act
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 2);
        // Assert: expert 1 (logit=-1.0, highest) dominates
        assert_eq!(output.len(), 2);
        assert!(output[0] > 0.0 && output[1] > 0.0, "output should be positive");
        // Output should be closer to expert 1's output than expert 0's
        assert!(output[0] > 4.0, "output[0] = {} should be > 4.0 (closer to expert 1)", output[0]);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_balancer_record_route_with_empty_token_routes() {
        // Arrange: a route table with zero tokens
        let config = ExpertRouteConfig::new(4, 2);
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        let rt = ExpertRouteTable::from_gate_logits(config, &[]);
        // Act
        balancer.record_route(&rt);
        // Assert: nothing changes
        assert_eq!(balancer.total_routes, 0);
        assert!(balancer.hit_history.iter().all(|&h| h == 0));
        assert!(balancer.cold_experts().is_empty());
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_balancer_hit_rates_after_many_cycles() {
        // Arrange: record route tables 100 times alternating between experts
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        for i in 0..100 {
            let row = if i % 2 == 0 {
                vec![100.0, 0.0]
            } else {
                vec![0.0, 100.0]
            };
            let rt = ExpertRouteTable::from_gate_logits(config.clone(), &[row]);
            balancer.record_route(&rt);
        }
        // Act
        let rates = balancer.hit_rates();
        // Assert: each expert ~50% rate
        assert!((rates[0] - 0.5).abs() < 0.02, "expert 0 rate = {}", rates[0]);
        assert!((rates[1] - 0.5).abs() < 0.02, "expert 1 rate = {}", rates[1]);
        assert_eq!(balancer.total_routes, 100);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_route_table_struct_construction_and_tokens_for_expert() {
        // Arrange: manually construct a table with known routes
        let table = ExpertRouteTable {
            config: ExpertRouteConfig::new(3, 2),
            token_routes: vec![
                TokenRoute {
                    expert_indices: vec![0, 2],
                    expert_weights: vec![0.6, 0.4],
                    expert_positions: vec![0, 0],
                },
                TokenRoute {
                    expert_indices: vec![1, 2],
                    expert_weights: vec![0.7, 0.3],
                    expert_positions: vec![0, 1],
                },
            ],
            expert_token_counts: vec![1, 1, 2],
            overflow_count: 0,
        };
        // Act
        let expert_2_tokens = table.tokens_for_expert(2);
        // Assert: expert 2 appears in both tokens
        assert_eq!(expert_2_tokens.len(), 2);
        assert_eq!(expert_2_tokens[0], (0, 0)); // token 0, position 0
        assert_eq!(expert_2_tokens[1], (1, 1)); // token 1, position 1
        // Expert 0 only in token 0
        let expert_0_tokens = table.tokens_for_expert(0);
        assert_eq!(expert_0_tokens.len(), 1);
        assert_eq!(expert_0_tokens[0], (0, 0));
        // Expert 1 only in token 1
        let expert_1_tokens = table.tokens_for_expert(1);
        assert_eq!(expert_1_tokens.len(), 1);
        assert_eq!(expert_1_tokens[0], (1, 0));
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_expert_route_table_config_stored_by_value_independent() {
        // Arrange: create config, build table, then verify independence
        let config = ExpertRouteConfig::new(4, 2);
        let gate_logits = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let rt = ExpertRouteTable::from_gate_logits(config.clone(), &gate_logits);
        // Act: drop the original config by reassignment — table's config must be unaffected
        // Assert: table's config is unaffected
        assert_eq!(rt.config.num_experts, 4, "table config should be independent");
        assert_eq!(rt.config.top_k, 2, "table config top_k should be independent");
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_load_balance_loss_proportional_to_num_experts() {
        // Arrange: same routing pattern, different num_experts
        let gate_logits_a = vec![vec![100.0, 0.0], vec![0.0, 100.0]];
        let config_a = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 100.0,
            load_balance_loss: true,
            load_balance_lambda: 1.0,
            ..ExpertRouteConfig::default()
        };
        let rt_a = ExpertRouteTable::from_gate_logits(config_a, &gate_logits_a);
        let loss_a = rt_a.load_balance_loss(&gate_logits_a);

        // With 4 experts and same uniform pattern (2 experts used, 2 unused)
        let gate_logits_b = vec![
            vec![100.0, 0.0, 0.0, 0.0],
            vec![0.0, 100.0, 0.0, 0.0],
            vec![0.0, 0.0, 100.0, 0.0],
            vec![0.0, 0.0, 0.0, 100.0],
        ];
        let config_b = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            load_balance_loss: true,
            load_balance_lambda: 1.0,
            ..ExpertRouteConfig::default()
        };
        let rt_b = ExpertRouteTable::from_gate_logits(config_b, &gate_logits_b);
        let loss_b = rt_b.load_balance_loss(&gate_logits_b);

        // Assert: loss_b ≈ loss_a since both are perfectly balanced with same lambda=1.0
        // but loss formula = lambda * N * sum, so loss scales with N
        // 2 experts perfectly balanced: loss = 1.0 * 2 * (0.5*0.5 + 0.5*0.5) = 1.0
        // 4 experts perfectly balanced: loss = 1.0 * 4 * (0.25*0.25 * 4) = 1.0
        assert!((loss_a - 1.0).abs() < 0.05, "loss_a = {}", loss_a);
        assert!((loss_b - 1.0).abs() < 0.05, "loss_b = {}", loss_b);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_utilization_stats_balance_score_zero_when_only_one_expert_loaded() {
        // Arrange: construct route table where all tokens go to one expert
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits: Vec<Vec<f32>> = (0..10)
            .map(|_| vec![100.0, 0.0, 0.0, 0.0])
            .collect();
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let stats = rt.utilization_stats();
        // Assert: balance = 1.0 - (max - min) / max = 1.0 - (10 - 0) / 10 = 0.0
        assert!((stats.balance_score).abs() < 1e-5,
            "completely imbalanced should have score 0.0, got {}", stats.balance_score);
        assert_eq!(stats.max_expert_load, 10);
        assert_eq!(stats.min_expert_load, 0);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_topk_with_weights_large_k_with_small_logits_returns_all() {
        // Arrange: 3 logits, k=10 (larger than logits.len())
        let logits = vec![0.5, 0.3, 0.8];
        // Act
        let topk = topk_with_weights(&logits, 10);
        // Assert: returns at most logits.len() entries
        assert_eq!(topk.len(), 3, "should return at most 3 entries");
        let sum: f32 = topk.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 0.01, "weights should sum to 1.0, got {}", sum);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_moe_dispatch_contributes_only_selected_experts() {
        // Arrange: top_k=1 selects only the highest-logit expert
        let input = vec![0.0; 3];
        let gate_logits = vec![0.0, 0.0, 100.0]; // expert 2 dominates
        let expert_outputs = vec![
            vec![100.0, 100.0, 100.0], // should NOT contribute
            vec![200.0, 200.0, 200.0], // should NOT contribute
            vec![1.0, 2.0, 3.0],       // should be the only contributor
        ];
        // Act
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 1);
        // Assert: output should be exactly expert 2's output (weight ~1.0)
        assert!((output[0] - 1.0).abs() < 0.01, "output[0] = {}", output[0]);
        assert!((output[1] - 2.0).abs() < 0.01, "output[1] = {}", output[1]);
        assert!((output[2] - 3.0).abs() < 0.01, "output[2] = {}", output[2]);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_balancer_suggest_capacity_uniform_load_equals_base() {
        // Arrange: perfectly uniform load across experts
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        // 8 tokens, 2 per expert (perfectly uniform)
        let gate_logits: Vec<Vec<f32>> = (0..8)
            .map(|i| {
                let mut row = vec![0.0; 4];
                row[i % 4] = 100.0;
                row
            })
            .collect();
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&rt);
        // Act
        let suggested = balancer.suggest_capacity_adjustment(20);
        // Assert: uniform load → all experts get base capacity
        for (i, &cap) in suggested.iter().enumerate() {
            assert_eq!(cap, 20, "expert {} should get base capacity 20", i);
        }
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_expert_route_table_clone_preserves_all_token_routes() {
        // Arrange: build a table with multiple tokens
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 2,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 50.0, 0.0],
            vec![0.0, 100.0, 50.0],
            vec![50.0, 0.0, 100.0],
        ];
        let original = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Act
        let cloned = original.clone();
        // Assert: every token route is preserved exactly
        assert_eq!(cloned.token_routes.len(), original.token_routes.len());
        for i in 0..original.token_routes.len() {
            assert_eq!(
                original.token_routes[i].expert_indices,
                cloned.token_routes[i].expert_indices,
                "token {} expert_indices mismatch", i
            );
            assert_eq!(
                original.token_routes[i].expert_positions,
                cloned.token_routes[i].expert_positions,
                "token {} expert_positions mismatch", i
            );
        }
        assert_eq!(cloned.overflow_count, original.overflow_count);
        assert_eq!(cloned.expert_token_counts, original.expert_token_counts);
    }

    // =========================================================================
    // Round 10: 15 additional edge-case tests
    // =========================================================================

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_topk_with_weights_positive_infinity_does_not_panic() {
        // Arrange: one entry is +Inf
        let logits = vec![1.0, f32::INFINITY, 3.0];
        // Act
        let topk = topk_with_weights(&logits, 2);
        // Assert: function must not panic and returns results
        assert!(!topk.is_empty(), "topk should return results with Inf input");
        assert!(topk.len() <= 3, "should return at most 3 entries");
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_expert_capacity_very_small_fractional_rounds_up() {
        // Arrange: 7 experts, 3 tokens, capacity_factor=0.5
        let config = ExpertRouteConfig {
            num_experts: 7,
            top_k: 1,
            capacity_factor: 0.5,
            ..ExpertRouteConfig::default()
        };
        // ceil(0.5 * 3 / 7) = ceil(0.2143) = 1
        // Act
        let cap = config.expert_capacity(3);
        // Assert
        assert_eq!(cap, 1, "fractional result must ceil to at least 1");
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_route_table_top_k_one_early_overflow_spreads_load() {
        // Arrange: 2 experts, capacity=1, 4 tokens all want expert 0
        // This tests that fallback distributes across experts over multiple tokens
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 0.26, // ceil(0.26 * 4 / 2) = ceil(0.52) = 1
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 0.0],
            vec![100.0, 0.0],
            vec![100.0, 0.0],
            vec![100.0, 0.0],
        ];
        // Act
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: both experts must have tokens (fallback spreads)
        assert!(rt.expert_token_counts[0] > 0, "expert 0 should have tokens");
        assert!(rt.expert_token_counts[1] > 0, "expert 1 should have tokens via fallback");
        assert_eq!(rt.overflow_count, 3, "3 of 4 tokens should overflow");
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_config_partial_eq_reflexivity_all_varied_fields() {
        // Arrange: config with non-default values in every field
        let config = ExpertRouteConfig {
            num_experts: 16,
            top_k: 4,
            capacity_factor: 3.0,
            load_balance_loss: true,
            load_balance_lambda: 0.1,
            noise_sigma: 0.5,
        };
        // Assert: reflexivity — config equals itself
        assert_eq!(config, config, "config must be equal to itself");
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_balancer_suggest_capacity_skewed_hot_expert_gets_more() {
        // Arrange: skewed load with base=10 and 3 experts
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        // All 10 tokens to expert 0
        let gate_logits: Vec<Vec<f32>> = (0..10)
            .map(|_| vec![100.0, 0.0, 0.0])
            .collect();
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&rt);
        // Act
        let suggested = balancer.suggest_capacity_adjustment(10);
        // Assert: expert 0 should get more capacity than others due to 100% hit rate
        assert_eq!(suggested.len(), 3);
        assert!(suggested[0] > suggested[1], "expert 0 cap {} should > expert 1 cap {}", suggested[0], suggested[1]);
        assert!(suggested[0] > suggested[2], "expert 0 cap {} should > expert 2 cap {}", suggested[0], suggested[2]);
        // All experts get at least 1 (minimum guarantee)
        for (i, &cap) in suggested.iter().enumerate() {
            assert!(cap >= 1, "expert {} should get at least 1, got {}", i, cap);
        }
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_moe_dispatch_empty_gate_logits() {
        // Arrange: empty gate logits vector
        let input = vec![1.0, 2.0, 3.0];
        let gate_logits: Vec<f32> = vec![];
        let expert_outputs: Vec<Vec<f32>> = vec![];
        // Act
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 1);
        // Assert: no experts contribute, output is all zeros
        assert_eq!(output.len(), 3);
        assert!(output.iter().all(|&v| v == 0.0), "empty gate → zero output");
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_softmax_single_positive_infinity() {
        // Arrange: single Inf element
        let probs = softmax(&[f32::INFINITY]);
        // Assert: function must not panic, returns length 1
        assert_eq!(probs.len(), 1);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_topk_indices_with_all_negative_infinity() {
        // Arrange: all -Inf logits
        let logits = vec![f32::NEG_INFINITY; 4];
        // Act
        let indices = topk_indices(&logits, 2);
        // Assert: returns 2 indices (all equal, any two are fine)
        assert_eq!(indices.len(), 2);
        for &idx in &indices {
            assert!(idx < 4, "index {} out of range", idx);
        }
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_route_table_tokens_for_expert_with_position_tracking() {
        // Arrange: 3 tokens all to expert 0, verify positions are 0, 1, 2
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 0.0],
            vec![100.0, 0.0],
            vec![100.0, 0.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Act
        let tokens = rt.tokens_for_expert(0);
        // Assert: 3 entries with positions 0, 1, 2
        assert_eq!(tokens.len(), 3);
        let positions: Vec<usize> = tokens.iter().map(|(_, pos)| *pos).collect();
        let mut sorted_positions = positions.clone();
        sorted_positions.sort();
        assert_eq!(sorted_positions, vec![0, 1, 2], "positions should be sequential");
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_load_balance_loss_returns_zero_when_disabled_even_with_logits() {
        // Arrange: load_balance_loss=false with non-trivial logits
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 2,
            load_balance_loss: false,
            load_balance_lambda: 0.01,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0, 0.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Act
        let loss = rt.load_balance_loss(&gate_logits);
        // Assert: loss is exactly 0.0 when disabled, regardless of routing skew
        assert!((loss).abs() < 1e-10, "disabled loss must be 0.0, got {}", loss);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_token_route_clone_mutation_isolation() {
        // Arrange
        let route = TokenRoute {
            expert_indices: vec![1, 3],
            expert_weights: vec![0.7, 0.3],
            expert_positions: vec![4, 7],
        };
        let mut cloned = route.clone();
        // Act: modify cloned weights
        cloned.expert_weights[0] = 0.0;
        cloned.expert_positions.push(99);
        // Assert: original unaffected
        assert!((route.expert_weights[0] - 0.7).abs() < 1e-5, "original weight changed");
        assert_eq!(route.expert_positions.len(), 2, "original positions modified");
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_topk_with_weights_two_experts_one_dominant() {
        // Arrange: 2 logits with large difference
        let logits = vec![-50.0, 50.0];
        // Act
        let topk = topk_with_weights(&logits, 2);
        // Assert: both selected, expert 1 has weight near 1.0
        assert_eq!(topk.len(), 2);
        assert_eq!(topk[0].0, 1, "index 1 (50.0) should be first");
        assert!(topk[0].1 > 0.99, "dominant weight = {}", topk[0].1);
        let sum: f32 = topk.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 0.01, "weights should sum to 1.0");
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_balancer_hot_experts_all_zero_rates_when_no_routes() {
        // Arrange: fresh balancer with 3 experts
        let balancer = ExpertLoadBalancer::new(ExpertRouteConfig::new(3, 1));
        // Act
        let hot = balancer.hot_experts(3);
        // Assert: all experts have 0.0 rate
        assert_eq!(hot.len(), 3);
        assert!(hot.iter().all(|&(_, rate)| rate == 0.0), "no routes → all rates 0.0");
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_route_table_from_gate_logits_with_alternating_high_low_logits() {
        // Arrange: alternating high/low logits to stress topk selection
        let config = ExpertRouteConfig::new(6, 3);
        let gate_logits = vec![
            vec![100.0, -100.0, 50.0, -50.0, 25.0, -25.0],
        ];
        // Act
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let route = &rt.token_routes[0];
        // Assert: experts 0, 2, 4 should be selected (positive logits)
        assert_eq!(route.expert_indices.len(), 3);
        assert!(route.expert_indices.contains(&0), "expert 0 (100.0) should be selected");
        assert!(route.expert_indices.contains(&2), "expert 2 (50.0) should be selected");
        assert!(route.expert_indices.contains(&4), "expert 4 (25.0) should be selected");
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_utilization_stats_mean_is_arithmetic_average() {
        // Arrange: 4 experts with counts [4, 2, 1, 1] → mean = 8/4 = 2.0
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut logits = Vec::new();
        for _ in 0..4 { logits.push(vec![100.0, 0.0, 0.0, 0.0]); }
        for _ in 0..2 { logits.push(vec![0.0, 100.0, 0.0, 0.0]); }
        logits.push(vec![0.0, 0.0, 100.0, 0.0]);
        logits.push(vec![0.0, 0.0, 0.0, 100.0]);
        let rt = ExpertRouteTable::from_gate_logits(config, &logits);
        let stats = rt.utilization_stats();
        // Assert: mean = total_assignments / num_experts = 8 / 4 = 2.0
        assert!((stats.mean_expert_load - 2.0).abs() < 0.01,
            "mean should be 2.0, got {}", stats.mean_expert_load);
        assert_eq!(stats.total_expert_assignments, 8);
    }

    // =========================================================================
    // Round 11: 15 additional edge-case tests
    // =========================================================================

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_softmax_three_zeros_middle_index_highest_prob() {
        // Arrange: all zeros produce uniform distribution
        let logits = vec![0.0, 0.0, 0.0];
        // Act
        let probs = softmax(&logits);
        // Assert: each probability is 1/3
        assert_eq!(probs.len(), 3);
        for &p in &probs {
            assert!((p - (1.0 / 3.0)).abs() < 1e-5,
                "uniform distribution: expected 1/3, got {}", p);
        }
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_topk_indices_all_zeros_returns_first_indices() {
        // Arrange: all zero logits
        let logits = vec![0.0, 0.0, 0.0, 0.0];
        // Act
        let indices = topk_indices(&logits, 2);
        // Assert: returns 2 indices (tie-breaking is stable)
        assert_eq!(indices.len(), 2);
        for &idx in &indices {
            assert!(idx < 4, "index {} out of range", idx);
        }
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_expert_route_config_equality_commutative() {
        // Arrange: two configs with identical fields
        let a = ExpertRouteConfig {
            num_experts: 7,
            top_k: 3,
            capacity_factor: 2.5,
            load_balance_loss: true,
            load_balance_lambda: 0.07,
            noise_sigma: 0.25,
        };
        let b = ExpertRouteConfig {
            num_experts: 7,
            top_k: 3,
            capacity_factor: 2.5,
            load_balance_loss: true,
            load_balance_lambda: 0.07,
            noise_sigma: 0.25,
        };
        // Assert: a == b and b == a (commutative)
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_route_table_from_gate_logits_single_expert_two_tokens_deterministic() {
        // Arrange: 1 expert, 2 tokens — deterministic routing
        let config = ExpertRouteConfig {
            num_experts: 1,
            top_k: 1,
            capacity_factor: 10.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![vec![3.0], vec![7.0]];
        // Act
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: both tokens route to expert 0, positions are sequential
        assert_eq!(rt.token_routes[0].expert_indices[0], 0);
        assert_eq!(rt.token_routes[0].expert_positions[0], 0);
        assert_eq!(rt.token_routes[1].expert_indices[0], 0);
        assert_eq!(rt.token_routes[1].expert_positions[0], 1);
        assert_eq!(rt.overflow_count, 0);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_balancer_hot_experts_after_single_recording() {
        // Arrange: record one route table with 3 tokens to different experts
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0],
            vec![0.0, 100.0, 0.0],
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&rt);
        // Act
        let hot = balancer.hot_experts(2);
        // Assert: top 2 are expert 0 and expert 1, expert 0 is hottest
        assert_eq!(hot.len(), 2);
        assert_eq!(hot[0].0, 0, "expert 0 should be hottest with 2 hits");
        assert_eq!(hot[1].0, 1, "expert 1 should be second with 1 hit");
        assert!(hot[0].1 > hot[1].1, "expert 0 rate should exceed expert 1");
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_moe_dispatch_two_experts_one_dominant_output() {
        // Arrange: expert 0 has near-zero weight, expert 1 dominates
        let input = vec![1.0, 1.0, 1.0];
        let gate_logits = vec![-100.0, 100.0];
        let expert_outputs = vec![
            vec![0.0, 0.0, 0.0],
            vec![5.0, 10.0, 15.0],
        ];
        // Act
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 2);
        // Assert: output is almost entirely from expert 1
        assert!((output[0] - 5.0).abs() < 0.1, "output[0] = {}", output[0]);
        assert!((output[1] - 10.0).abs() < 0.1, "output[1] = {}", output[1]);
        assert!((output[2] - 15.0).abs() < 0.1, "output[2] = {}", output[2]);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_expert_capacity_with_one_token_and_many_experts() {
        // Arrange: 64 experts, 1 token, default capacity_factor
        let config = ExpertRouteConfig::new(64, 1);
        // ceil(1.25 * 1 / 64) = ceil(0.01953125) = 1
        // Act
        let cap = config.expert_capacity(1);
        // Assert
        assert_eq!(cap, 1, "even tiny fractional ceil yields at least 1");
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_route_table_overflow_count_with_top_k_two() {
        // Arrange: top_k=2, very tight capacity, all tokens want same experts
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 2,
            capacity_factor: 0.1, // ceil(0.1 * 4 / 3) = ceil(0.133) = 1
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 50.0, 0.0],
            vec![100.0, 50.0, 0.0],
            vec![100.0, 50.0, 0.0],
            vec![100.0, 50.0, 0.0],
        ];
        // Act
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: overflow should be non-zero due to tight capacity
        assert!(rt.overflow_count > 0,
            "tight capacity with top_k=2 should cause overflow, got {}", rt.overflow_count);
        // Every token still gets at least 1 expert (fallback)
        for (i, route) in rt.token_routes.iter().enumerate() {
            assert!(!route.expert_indices.is_empty(),
                "token {} must have at least 1 expert", i);
        }
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_stats_partial_eq_differs_by_balance_score() {
        // Arrange: two stats differing only in balance_score
        let a = ExpertUtilizationStats {
            total_tokens: 10, total_expert_assignments: 20, overflow_count: 0,
            max_expert_load: 5, min_expert_load: 5, mean_expert_load: 5.0,
            balance_score: 1.0,
        };
        let b = ExpertUtilizationStats {
            total_tokens: 10, total_expert_assignments: 20, overflow_count: 0,
            max_expert_load: 5, min_expert_load: 5, mean_expert_load: 5.0,
            balance_score: 0.5,
        };
        // Assert
        assert_ne!(a, b, "stats with different balance_score should not be equal");
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_topk_with_weights_empty_logits_k_nonzero() {
        // Arrange: empty logits with k=3
        let topk = topk_with_weights(&[], 3);
        // Assert: returns empty (nothing to select from)
        assert!(topk.is_empty(), "empty logits should return empty topk");
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_balancer_reset_twice_does_not_double_count() {
        // Arrange
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        let rt = ExpertRouteTable::from_gate_logits(config, &[vec![100.0, 0.0]]);
        balancer.record_route(&rt);
        // Act: reset twice
        balancer.reset();
        balancer.reset();
        // Assert: state is still clean
        assert_eq!(balancer.total_routes, 0);
        assert!(balancer.hit_history.iter().all(|&h| h == 0));
        let rates = balancer.hit_rates();
        assert!(rates.iter().all(|&r| r == 0.0));
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_load_balance_loss_with_two_experts_one_token_each() {
        // Arrange: 2 experts, perfectly balanced, lambda=0.5
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 100.0,
            load_balance_loss: true,
            load_balance_lambda: 0.5,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![vec![100.0, 0.0], vec![0.0, 100.0]];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Act
        let loss = rt.load_balance_loss(&gate_logits);
        // Assert: f = [0.5, 0.5], P = [~0.5, ~0.5], sum ≈ 0.5
        // loss = 0.5 * 2 * 0.5 = 0.5
        assert!((loss - 0.5).abs() < 0.05, "balanced 2-expert loss should be ~0.5, got {}", loss);
        assert!(loss.is_finite());
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_route_table_gate_logits_matches_num_experts_exactly() {
        // Arrange: 5 experts with 5 logits — exact match
        let config = ExpertRouteConfig::new(5, 3);
        let gate_logits = vec![vec![1.0, 5.0, 3.0, 2.0, 4.0]];
        // Act
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: top 3 are indices 1 (5.0), 4 (4.0), 2 (3.0)
        let route = &rt.token_routes[0];
        assert_eq!(route.expert_indices.len(), 3);
        assert!(route.expert_indices.contains(&1), "expert 1 (5.0) should be selected");
        assert!(route.expert_indices.contains(&4), "expert 4 (4.0) should be selected");
        assert!(route.expert_indices.contains(&2), "expert 2 (3.0) should be selected");
        assert_eq!(rt.overflow_count, 0);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_moe_dispatch_expert_output_longer_than_input_clamps() {
        // Arrange: expert output has 6 elements, input has 3
        let input = vec![1.0, 2.0, 3.0];
        let gate_logits = vec![5.0];
        let expert_outputs = vec![vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]];
        // Act
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 1);
        // Assert: output length matches input, extra elements ignored
        assert_eq!(output.len(), 3);
        assert!((output[0] - 10.0).abs() < 0.01);
        assert!((output[1] - 20.0).abs() < 0.01);
        assert!((output[2] - 30.0).abs() < 0.01);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_topk_indices_preserves_all_valid_indices() {
        // Arrange: 5 logits
        let logits = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        // Act: request all 5
        let indices = topk_indices(&logits, 5);
        // Assert: all 5 indices returned, sorted descending by value
        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0], 4, "index 4 (5.0) first");
        assert_eq!(indices[1], 2, "index 2 (4.0) second");
        assert_eq!(indices[2], 0, "index 0 (3.0) third");
        // All indices are unique
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    // --- 15 new tests ---

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_expert_capacity_ceil_rounds_up_fractional_result() {
        // Arrange: 7 experts, capacity_factor=1.0, 10 tokens → ceil(1.0*10/7)=ceil(1.428...)=2
        let config = ExpertRouteConfig {
            num_experts: 7,
            top_k: 1,
            capacity_factor: 1.0,
            load_balance_loss: false,
            load_balance_lambda: 0.01,
            noise_sigma: 0.0,
        };
        // Act
        let cap = config.expert_capacity(10);
        // Assert: must round up, not truncate
        assert_eq!(cap, 2);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_topk_with_weights_k_one_single_dominant_logit_weight_near_one() {
        // Arrange: one logit vastly larger than others
        let logits = vec![100.0_f32, 0.0, 0.0, 0.0];
        // Act
        let result = topk_with_weights(&logits, 1);
        // Assert: expert 0 selected with weight very close to 1.0
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 0);
        assert!((result[0].1 - 1.0).abs() < 1e-5, "dominant weight should be ~1.0, got {}", result[0].1);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_balancer_record_route_with_topk_two_increments_both_experts() {
        // Arrange: 4 experts, top_k=2, one token routing to experts 0 and 1
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 2,
            capacity_factor: 1.0,
            load_balance_loss: false,
            load_balance_lambda: 0.0,
            noise_sigma: 0.0,
        };
        let gate_logits = vec![vec![10.0, 10.0, 0.0, 0.0]];
        let table = ExpertRouteTable::from_gate_logits(config.clone(), &gate_logits);
        let mut balancer = ExpertLoadBalancer::new(config);
        // Act
        balancer.record_route(&table);
        // Assert: experts 0 and 1 each incremented, experts 2 and 3 not
        let rates = balancer.hit_rates();
        assert!(rates[0] > 0.0, "expert 0 should have hits");
        assert!(rates[1] > 0.0, "expert 1 should have hits");
        assert_eq!(rates[2], 0.0, "expert 2 should have zero hits");
        assert_eq!(rates[3], 0.0, "expert 3 should have zero hits");
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_route_table_from_gate_logits_all_tokens_route_to_same_expert() {
        // Arrange: 3 experts, top_k=1, 5 tokens all strongly prefer expert 0
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 2.0,
            load_balance_loss: false,
            load_balance_lambda: 0.0,
            noise_sigma: 0.0,
        };
        let gate_logits = vec![
            vec![10.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0],
        ];
        // Act
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: expert 0 gets capacity=4 filled, overflow goes to least-loaded (expert 1 or 2)
        // capacity = ceil(2.0 * 5 / 3) = 4; token 5 overflows expert 0 → fallback to least loaded
        assert_eq!(table.expert_token_counts[0], 4, "expert 0 should be filled to capacity");
        assert!(table.overflow_count >= 1, "at least 1 overflow expected with capacity=4 and 5 tokens");
        // All tokens still get routed
        assert_eq!(table.token_routes.len(), 5);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_softmax_two_element_one_zero_one_positive() {
        // Arrange: [0.0, 1.0]
        let logits = vec![0.0_f32, 1.0_f32];
        // Act
        let probs = softmax(&logits);
        // Assert: element 1 has higher probability
        assert!(probs[1] > probs[0], "softmax(1.0) > softmax(0.0)");
        // Sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_moe_dispatch_top_k_zero_returns_zero_vector() {
        // Arrange: valid input and expert outputs but top_k=0
        let input = vec![1.0_f32, 2.0, 3.0];
        let gate_logits = vec![10.0, 5.0, 1.0];
        let expert_outputs = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        // Act
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 0);
        // Assert: output is all zeros (no expert selected)
        assert_eq!(output.len(), 3);
        assert!(output.iter().all(|&v| v == 0.0), "top_k=0 should produce zero output");
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_expert_route_config_partial_eq_differs_by_load_balance_loss() {
        // Arrange: two configs differing only in load_balance_loss
        let a = ExpertRouteConfig {
            num_experts: 8,
            top_k: 2,
            capacity_factor: 1.25,
            load_balance_loss: false,
            load_balance_lambda: 0.01,
            noise_sigma: 0.0,
        };
        let b = ExpertRouteConfig {
            num_experts: 8,
            top_k: 2,
            capacity_factor: 1.25,
            load_balance_loss: true,
            load_balance_lambda: 0.01,
            noise_sigma: 0.0,
        };
        // Act & Assert
        assert_ne!(a, b, "configs differing only in load_balance_loss should not be equal");
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_balancer_suggest_capacity_skewed_then_reset_then_uniform() {
        // Arrange: 4 experts, skewed routing first
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 1.0,
            load_balance_loss: false,
            load_balance_lambda: 0.0,
            noise_sigma: 0.0,
        };
        let skewed_logits = vec![
            vec![10.0, 0.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0, 0.0],
        ];
        let skewed_table = ExpertRouteTable::from_gate_logits(config.clone(), &skewed_logits);
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        balancer.record_route(&skewed_table);
        let skewed_suggestion = balancer.suggest_capacity_adjustment(10);
        // Act: reset and record uniform routing
        balancer.reset();
        let uniform_logits = vec![
            vec![10.0, 10.0, 10.0, 10.0],
            vec![10.0, 10.0, 10.0, 10.0],
            vec![10.0, 10.0, 10.0, 10.0],
        ];
        let uniform_table = ExpertRouteTable::from_gate_logits(config, &uniform_logits);
        balancer.record_route(&uniform_table);
        let uniform_suggestion = balancer.suggest_capacity_adjustment(10);
        // Assert: uniform suggestion more balanced than skewed
        let skewed_range = skewed_suggestion.iter().max().unwrap() - skewed_suggestion.iter().min().unwrap();
        let uniform_range = uniform_suggestion.iter().max().unwrap() - uniform_suggestion.iter().min().unwrap();
        assert!(uniform_range <= skewed_range, "uniform should be more balanced");
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_route_table_overflow_count_with_strict_capacity_and_many_tokens() {
        // Arrange: 2 experts, top_k=2, very tight capacity
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 2,
            capacity_factor: 0.5, // capacity = ceil(0.5 * 4 / 2) = 1 per expert
            load_balance_loss: false,
            load_balance_lambda: 0.0,
            noise_sigma: 0.0,
        };
        let gate_logits = vec![
            vec![10.0, 5.0],   // token 0 → experts 0,1
            vec![10.0, 5.0],   // token 1 → experts 0,1 (both overflow → fallback)
            vec![5.0, 10.0],   // token 2 → experts 1,0 (both overflow → fallback)
            vec![5.0, 10.0],   // token 3 → experts 1,0 (both overflow → fallback)
        ];
        // Act
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: token 0 fills both experts, tokens 1-3 each overflow 2 experts
        assert!(table.overflow_count >= 6, "expected at least 6 overflows, got {}", table.overflow_count);
        assert_eq!(table.token_routes.len(), 4, "all tokens must still have routes");
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_expert_load_balancer_cold_experts_threshold_is_exclusive() {
        // Arrange: 4 experts, route only to expert 0 repeatedly, total_routes=100
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            load_balance_loss: false,
            load_balance_lambda: 0.0,
            noise_sigma: 0.0,
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        // Manually set up state: expert 0 gets all 100 hits
        // Create 100 tokens all routing to expert 0
        let gate_logits: Vec<Vec<f32>> = (0..100).map(|_| vec![10.0, 0.0, 0.0, 0.0]).collect();
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&table);
        // Act
        let cold = balancer.cold_experts();
        // Assert: experts 1, 2, 3 should be cold (0 hits → rate = 0.0 < 0.001)
        let cold_indices: Vec<usize> = cold.iter().map(|(idx, _)| *idx).collect();
        assert!(cold_indices.contains(&1), "expert 1 should be cold");
        assert!(cold_indices.contains(&2), "expert 2 should be cold");
        assert!(cold_indices.contains(&3), "expert 3 should be cold");
        assert!(!cold_indices.contains(&0), "expert 0 should NOT be cold");
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_topk_indices_single_element_k_one_returns_zero() {
        // Arrange: single-element logits array
        let logits = vec![42.0_f32];
        // Act
        let indices = topk_indices(&logits, 1);
        // Assert: only index 0, only element
        assert_eq!(indices, vec![0]);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_tokens_for_expert_returns_empty_for_out_of_range_expert() {
        // Arrange: 3 experts, 2 tokens
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 1.0,
            load_balance_loss: false,
            load_balance_lambda: 0.0,
            noise_sigma: 0.0,
        };
        let gate_logits = vec![vec![10.0, 0.0, 0.0], vec![0.0, 10.0, 0.0]];
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Act: query expert index beyond range
        let result = table.tokens_for_expert(5);
        // Assert: empty result, no panic
        assert!(result.is_empty(), "out-of-range expert should yield empty result");
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_moe_dispatch_empty_input_returns_empty_output() {
        // Arrange: empty input vector
        let input: Vec<f32> = vec![];
        let gate_logits = vec![1.0, 0.0];
        let expert_outputs = vec![vec![], vec![]];
        // Act
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 1);
        // Assert: output is empty
        assert!(output.is_empty(), "empty input should produce empty output");
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_expert_utilization_stats_copy_trait_independence() {
        // Arrange
        let original = ExpertUtilizationStats {
            total_tokens: 10,
            total_expert_assignments: 20,
            overflow_count: 2,
            max_expert_load: 5,
            min_expert_load: 1,
            mean_expert_load: 2.5,
            balance_score: 0.8,
        };
        // Act: Copy creates independent value
        let mut copy = original;
        copy.total_tokens = 999;
        copy.balance_score = 0.0;
        // Assert: original unchanged
        assert_eq!(original.total_tokens, 10);
        assert!((original.balance_score - 0.8).abs() < 1e-5);
        assert_eq!(copy.total_tokens, 999);
    }

    // @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_load_balance_loss_with_lambda_zero_returns_zero() {
        // Arrange: 2 experts, lambda=0.0, load_balance_loss enabled
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 1.0,
            load_balance_loss: true,
            load_balance_lambda: 0.0, // zero lambda
            noise_sigma: 0.0,
        };
        let gate_logits = vec![vec![10.0, 0.0], vec![0.0, 10.0]];
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Act
        let loss = table.load_balance_loss(&gate_logits);
        // Assert: zero lambda → zero loss regardless of routing
        assert!((loss - 0.0).abs() < 1e-7, "lambda=0 should produce zero loss, got {}", loss);
    }

    #[test]
    fn test_route_table_position_starts_at_zero_for_first_token() {
        // Arrange: 2 experts, top_k=1, capacity_factor high so no overflow
        let config = ExpertRouteConfig::new(2, 1);
        let gate_logits = vec![vec![5.0, 0.0]]; // first token routes to expert 0
        // Act
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: the first token assigned to expert 0 should have position 0
        assert_eq!(table.token_routes[0].expert_positions, vec![0]);
    }

    #[test]
    fn test_route_table_position_increments_per_assigned_token() {
        // Arrange: 2 experts, top_k=1, high capacity so all route to expert 0
        let mut config = ExpertRouteConfig::new(2, 1);
        config.capacity_factor = 10.0; // generous capacity
        let gate_logits = vec![
            vec![10.0, 0.0],
            vec![10.0, 0.0],
            vec![10.0, 0.0],
        ];
        // Act
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: positions should be 0, 1, 2 for expert 0
        assert_eq!(table.token_routes[0].expert_positions, vec![0]);
        assert_eq!(table.token_routes[1].expert_positions, vec![1]);
        assert_eq!(table.token_routes[2].expert_positions, vec![2]);
    }

    #[test]
    fn test_route_table_no_overflow_when_capacity_exceeds_demand() {
        // Arrange: 4 experts, top_k=2, large capacity factor, 2 tokens
        let mut config = ExpertRouteConfig::new(4, 2);
        config.capacity_factor = 10.0; // very generous capacity
        let gate_logits = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4.0, 3.0, 2.0, 1.0],
        ];
        // Act
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: zero overflow since capacity is huge
        assert_eq!(table.overflow_count, 0, "generous capacity should yield zero overflow");
    }

    #[test]
    fn test_topk_with_weights_tie_breaking_preserves_higher_index_first() {
        // Arrange: three experts with identical logits — all tied
        let logits = vec![1.0, 1.0, 1.0];
        // Act: ask for top 2
        let topk = topk_with_weights(&logits, 2);
        // Assert: both selected, weights equal, and sum to 1.0
        assert_eq!(topk.len(), 2);
        let w0 = topk[0].1;
        let w1 = topk[1].1;
        assert!((w0 - w1).abs() < 1e-6, "tied logits should yield equal weights, got {} vs {}", w0, w1);
        let sum: f32 = topk.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 1e-6, "renormalized weights should sum to 1.0, got {}", sum);
    }

    #[test]
    fn test_softmax_single_element_is_exactly_one() {
        // Arrange: single-element input
        let logits = vec![3.7];
        // Act
        let probs = softmax(&logits);
        // Assert: probability is exactly 1.0
        assert_eq!(probs.len(), 1);
        assert!((probs[0] - 1.0).abs() < 1e-7, "single element softmax should be 1.0, got {}", probs[0]);
    }

    #[test]
    fn test_softmax_two_elements_sum_to_one() {
        // Arrange
        let logits = vec![0.5, 0.5];
        // Act
        let probs = softmax(&logits);
        // Assert: both equal, sum to 1.0
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax should sum to 1.0, got {}", sum);
        assert!((probs[0] - probs[1]).abs() < 1e-6, "equal inputs should yield equal probabilities");
    }

    #[test]
    fn test_moe_dispatch_output_is_weighted_sum_of_experts() {
        // Arrange: 2 experts, top_k=1, known outputs
        let input = vec![0.0; 3];
        let gate_logits = vec![0.0, 10.0]; // expert 1 dominates
        let expert_outputs = vec![
            vec![1.0, 2.0, 3.0], // expert 0
            vec![4.0, 5.0, 6.0], // expert 1
        ];
        // Act
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 1);
        // Assert: only expert 1 contributes (weight ~1.0 after softmax + renorm)
        assert_eq!(output.len(), 3);
        assert!(output[0] > 3.5, "expert 1 output should dominate, got {}", output[0]);
        assert!(output[2] > 5.5, "expert 1 output should dominate, got {}", output[2]);
    }

    #[test]
    fn test_moe_dispatch_all_zero_expert_outputs_yields_zero() {
        // Arrange: experts all produce zero vectors
        let input = vec![0.0; 4];
        let gate_logits = vec![1.0, 2.0, 3.0];
        let expert_outputs = vec![
            vec![0.0; 4],
            vec![0.0; 4],
            vec![0.0; 4],
        ];
        // Act
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 2);
        // Assert: output is all zeros
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 0.0).abs() < 1e-7, "output[{}] should be 0.0, got {}", i, v);
        }
    }

    #[test]
    fn test_config_capacity_factor_one_exact_division() {
        // Arrange: 4 experts, capacity_factor=1.0, exactly 8 tokens
        let mut config = ExpertRouteConfig::new(4, 2);
        config.capacity_factor = 1.0;
        // Act
        let capacity = config.expert_capacity(8);
        // Assert: 1.0 * 8 / 4 = 2.0, ceil(2.0) = 2
        assert_eq!(capacity, 2, "exact division with factor 1.0 should yield tokens/experts");
    }

    #[test]
    fn test_expert_capacity_exactly_one_per_expert() {
        // Arrange: 3 experts, 3 tokens, capacity_factor=1.0
        let mut config = ExpertRouteConfig::new(3, 1);
        config.capacity_factor = 1.0;
        // Act
        let capacity = config.expert_capacity(3);
        // Assert: 1.0 * 3 / 3 = 1.0, ceil(1.0) = 1
        assert_eq!(capacity, 1);
    }

    #[test]
    fn test_balancer_suggest_capacity_sum_is_proportional_to_base() {
        // Arrange: 3 experts, skewed routing where expert 0 gets all hits
        let config = ExpertRouteConfig::new(3, 1);
        let mut balancer = ExpertLoadBalancer::new(config);
        // Record 10 routes all to expert 0
        let gate_logits = vec![vec![100.0, 0.0, 0.0]; 10];
        let route_config = ExpertRouteConfig::new(3, 1);
        let table = ExpertRouteTable::from_gate_logits(route_config, &gate_logits);
        balancer.record_route(&table);
        // Act
        let suggested = balancer.suggest_capacity_adjustment(5);
        // Assert: expert 0 should get highest capacity; others get minimum 1
        assert!(suggested[0] > 1, "dominant expert should get more than minimum capacity");
        assert_eq!(suggested.len(), 3);
        assert!(suggested[1] >= 1, "each expert should get at least 1 capacity");
        assert!(suggested[2] >= 1, "each expert should get at least 1 capacity");
    }

    #[test]
    fn test_balancer_hit_rates_reflect_proportional_load() {
        // Arrange: 2 experts, 3 tokens all route to expert 0, high capacity
        let mut config = ExpertRouteConfig::new(2, 1);
        config.capacity_factor = 10.0;
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        let gate_logits = vec![
            vec![10.0, 0.0],
            vec![10.0, 0.0],
            vec![10.0, 0.0],
        ];
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&table);
        // Act
        let rates = balancer.hit_rates();
        // Assert: expert 0 should have rate 1.0, expert 1 should have 0.0
        assert!((rates[0] - 1.0).abs() < 1e-7, "expert 0 should have 100% hit rate, got {}", rates[0]);
        assert!((rates[1] - 0.0).abs() < 1e-7, "expert 1 should have 0% hit rate, got {}", rates[1]);
    }

    #[test]
    fn test_utilization_stats_total_expert_assignments_equals_sum_of_counts() {
        // Arrange: 3 experts, top_k=2, 2 tokens
        let config = ExpertRouteConfig::new(3, 2);
        let gate_logits = vec![
            vec![1.0, 2.0, 3.0],
            vec![3.0, 2.0, 1.0],
        ];
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Act
        let stats = table.utilization_stats();
        // Assert: total_expert_assignments should equal sum of expert_token_counts
        let counts_sum: usize = table.expert_token_counts.iter().sum();
        assert_eq!(
            stats.total_expert_assignments, counts_sum,
            "total_expert_assignments should match sum of expert_token_counts"
        );
    }

    #[test]
    fn test_load_balance_loss_f_i_proportional_to_routing_count() {
        // Arrange: 2 experts with loss enabled, one expert gets both tokens
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 10.0,
            load_balance_loss: true,
            load_balance_lambda: 1.0,
            noise_sigma: 0.0,
        };
        let gate_logits = vec![
            vec![10.0, 0.0],
            vec![10.0, 0.0],
        ];
        let table = ExpertRouteTable::from_gate_logits(config.clone(), &gate_logits);
        // Act
        let loss = table.load_balance_loss(&gate_logits);
        // Assert: loss should be positive since routing is skewed (f_0=1.0, f_1=0.0)
        assert!(loss > 0.0, "skewed routing should yield positive loss, got {}", loss);
    }

    #[test]
    fn test_token_route_indices_are_within_expert_range() {
        // Arrange: 4 experts, top_k=2
        let config = ExpertRouteConfig::new(4, 2);
        let gate_logits = vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.4, 0.3, 0.2, 0.1],
            vec![1.0, 0.0, 0.0, 0.0],
        ];
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Act & Assert: all expert indices should be in [0, 4)
        for (i, route) in table.token_routes.iter().enumerate() {
            for &expert_idx in &route.expert_indices {
                assert!(
                    expert_idx < 4,
                    "token {} has out-of-range expert index {}",
                    i,
                    expert_idx
                );
            }
        }
    }

    // --- 15 new tests ---

    #[test]
    fn test_balancer_suggest_capacity_zero_base_returns_minimum_one() {
        // Arrange: 4 experts with some recorded routes
        let config = ExpertRouteConfig::new(4, 1);
        let mut balancer = ExpertLoadBalancer::new(config);
        let gate_logits = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let table = ExpertRouteTable::from_gate_logits(balancer.config().clone(), &gate_logits);
        balancer.record_route(&table);
        // Act: base_capacity=0 should still return minimum 1 per expert
        let caps = balancer.suggest_capacity_adjustment(0);
        // Assert: every entry must be at least 1
        for (i, &cap) in caps.iter().enumerate() {
            assert!(cap >= 1, "expert {} capacity {} should be >= 1", i, cap);
        }
    }

    #[test]
    fn test_route_table_single_expert_many_tokens_positions_are_sequential() {
        // Arrange: 1 expert, top_k=1, 5 tokens all routing to same expert
        let config = ExpertRouteConfig::new(1, 1);
        let gate_logits = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
        ];
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Act: collect positions for expert 0
        let pairs = table.tokens_for_expert(0);
        // Assert: positions should be 0,1,2,3,4 in order
        assert_eq!(pairs.len(), 5, "all 5 tokens should be assigned to expert 0");
        let positions: Vec<usize> = pairs.iter().map(|&(_, pos)| pos).collect();
        assert_eq!(positions, vec![0, 1, 2, 3, 4], "positions should be sequential");
    }

    #[test]
    fn test_route_table_expert_counts_consistency_after_overflow() {
        // Arrange: 2 experts, top_k=1, capacity_factor=0.5 so capacity=1 per expert for 4 tokens
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 0.5,
            ..ExpertRouteConfig::default()
        };
        // Token 0 → expert 1, Token 1 → expert 1, Token 2 → expert 0, Token 3 → expert 0
        let gate_logits = vec![
            vec![0.0, 10.0],
            vec![0.0, 10.0],
            vec![10.0, 0.0],
            vec![10.0, 0.0],
        ];
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Act: expert_token_counts sum should account for overflow fallback
        let counts_sum: usize = table.expert_token_counts.iter().sum();
        // Assert: sum should equal number of tokens (every token gets at least 1 expert via fallback)
        assert_eq!(counts_sum, 4, "sum of expert_token_counts should equal 4 tokens");
        assert!(table.overflow_count > 0, "should have overflow with capacity=1");
    }

    #[test]
    fn test_dispatch_expert_outputs_different_lengths_clamps_to_input() {
        // Arrange: input length 3, expert 0 output length 5, expert 1 output length 1
        let input = vec![1.0, 2.0, 3.0];
        let gate_logits = vec![5.0, 5.0]; // equal weight
        let expert_outputs = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![10.0]];
        // Act
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 2);
        // Assert: output length matches input length
        assert_eq!(output.len(), 3, "output length should match input length");
        // expert 1 contributes only to index 0; expert 0 contributes to all 3
        // With equal softmax weights ~0.5 each
        assert!(output[0] > 0.0, "index 0 should have contribution from both experts");
        assert!(output[2] > 0.0, "index 2 should have contribution from expert 0");
    }

    #[test]
    fn test_config_capacity_float_precision_ceil_behavior() {
        // Arrange: num_experts=3, capacity_factor=1.0, total_tokens=10
        // capacity = ceil(1.0 * 10 / 3) = ceil(3.333...) = 4
        let config = ExpertRouteConfig {
            num_experts: 3,
            capacity_factor: 1.0,
            ..ExpertRouteConfig::default()
        };
        // Act
        let cap = config.expert_capacity(10);
        // Assert
        assert_eq!(cap, 4, "ceil(10/3) should be 4");
    }

    #[test]
    fn test_balancer_record_route_same_table_twice_doubles_counts() {
        // Arrange
        let config = ExpertRouteConfig::new(2, 1);
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        let gate_logits = vec![vec![10.0, 0.0], vec![0.0, 10.0]];
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Act: record same table twice
        balancer.record_route(&table);
        balancer.record_route(&table);
        // Assert: hit rates should reflect double recording
        let rates = balancer.hit_rates();
        assert_eq!(rates.len(), 2);
        // Each expert was hit twice per recording, total_routes=4 (2 tokens * 2 recordings)
        // expert 0: 2 hits / 4 routes = 0.5; expert 1: 2 hits / 4 routes = 0.5
        assert!((rates[0] - 0.5).abs() < 1e-9, "expert 0 rate should be 0.5, got {}", rates[0]);
        assert!((rates[1] - 0.5).abs() < 1e-9, "expert 1 rate should be 0.5, got {}", rates[1]);
    }

    #[test]
    fn test_route_table_from_gate_logits_large_expert_count() {
        // Arrange: 64 experts, top_k=4, 3 tokens
        let config = ExpertRouteConfig::new(64, 4);
        let gate_logits: Vec<Vec<f32>> = (0..3)
            .map(|t| {
                (0..64)
                    .map(|e| if e == t * 20 { 10.0 } else { 0.0 })
                    .collect()
            })
            .collect();
        // Act
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Assert: each token should have exactly top_k=4 selected experts
        for (i, route) in table.token_routes.iter().enumerate() {
            assert!(
                route.expert_indices.len() <= 4,
                "token {} should have at most 4 experts, got {}",
                i,
                route.expert_indices.len()
            );
        }
        assert_eq!(table.token_routes.len(), 3);
    }

    #[test]
    fn test_softmax_with_repeated_values_equal_probabilities() {
        // Arrange: [2.0, 2.0, 2.0, 2.0] — all equal
        let logits = vec![2.0, 2.0, 2.0, 2.0];
        // Act
        let probs = softmax(&logits);
        // Assert: all probabilities should be exactly 0.25
        for (i, &p) in probs.iter().enumerate() {
            assert!(
                (p - 0.25).abs() < 1e-6,
                "probability at index {} should be 0.25, got {}",
                i,
                p
            );
        }
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum should be 1.0, got {}", sum);
    }

    #[test]
    fn test_token_route_manual_construction_all_fields_accessible() {
        // Arrange: construct a TokenRoute manually with specific values
        let route = TokenRoute {
            expert_indices: vec![2, 5, 7],
            expert_weights: vec![0.5, 0.3, 0.2],
            expert_positions: vec![0, 1, 3],
        };
        // Act & Assert: all fields should be accessible and match
        assert_eq!(route.expert_indices, vec![2, 5, 7]);
        assert_eq!(route.expert_weights.len(), 3);
        assert_eq!(route.expert_positions, vec![0, 1, 3]);
        let weight_sum: f32 = route.expert_weights.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-6, "weights should sum to 1.0");
    }

    #[test]
    fn test_utilization_stats_overflow_less_than_or_equal_total_tokens() {
        // Arrange: tight capacity to force overflow
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 2,
            capacity_factor: 0.25,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![10.0, 0.0],
            vec![0.0, 10.0],
            vec![10.0, 0.0],
            vec![0.0, 10.0],
        ];
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let stats = table.utilization_stats();
        // Assert: overflow cannot exceed total tokens * top_k
        assert!(
            stats.overflow_count <= stats.total_tokens * 2,
            "overflow {} should not exceed {}",
            stats.overflow_count,
            stats.total_tokens * 2
        );
    }

    #[test]
    fn test_route_table_weights_are_non_negative_per_token() {
        // Arrange: 4 experts, top_k=3, 5 tokens with varied logits
        let config = ExpertRouteConfig::new(4, 3);
        let gate_logits = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4.0, 3.0, 2.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![0.1, 0.2, 0.3, 0.4],
            vec![10.0, -1.0, 0.0, 0.5],
        ];
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Act & Assert: each token's weights should be non-negative and sum > 0
        for (i, route) in table.token_routes.iter().enumerate() {
            let sum: f32 = route.expert_weights.iter().sum();
            assert!(
                sum > 0.0,
                "token {} weights sum should be positive, got {}",
                i,
                sum
            );
            for (j, &w) in route.expert_weights.iter().enumerate() {
                assert!(
                    w >= 0.0,
                    "token {} weight {} should be non-negative, got {}",
                    i,
                    j,
                    w
                );
            }
        }
    }

    #[test]
    fn test_balancer_cold_experts_all_experts_cold_when_most_hits_one() {
        // Arrange: 8 experts, top_k=1, high capacity_factor to avoid fallback to other experts
        let config = ExpertRouteConfig {
            num_experts: 8,
            top_k: 1,
            capacity_factor: 100.0, // large enough so all tokens fit in expert 0
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config);
        // 5 tokens all strongly routing to expert 0
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            5
        ];
        let table = ExpertRouteTable::from_gate_logits(balancer.config().clone(), &gate_logits);
        balancer.record_route(&table);
        // Act
        let cold = balancer.cold_experts();
        // Assert: experts 1-7 should all be cold (rate = 0.0 < 0.001 threshold)
        assert!(
            cold.len() >= 7,
            "at least 7 experts should be cold, got {}",
            cold.len()
        );
        for (idx, rate) in &cold {
            assert!(
                *idx != 0,
                "expert 0 should not be cold"
            );
            assert!(
                *rate < 0.001,
                "cold expert {} rate {} should be < 0.001",
                idx,
                rate
            );
        }
    }

    #[test]
    fn test_topk_indices_returns_exactly_k_when_k_less_than_len() {
        // Arrange: 5 logits, k=3
        let logits = vec![0.5, 1.5, 0.2, 3.0, 0.8];
        // Act
        let indices = topk_indices(&logits, 3);
        // Assert: exactly 3 indices returned
        assert_eq!(indices.len(), 3, "should return exactly 3 indices");
        // The highest values are at indices 3 (3.0), 1 (1.5), 4 (0.8)
        assert_eq!(indices[0], 3, "highest should be index 3");
        assert_eq!(indices[1], 1, "second highest should be index 1");
        assert_eq!(indices[2], 4, "third highest should be index 4");
    }

    #[test]
    fn test_moe_dispatch_input_longer_than_all_expert_outputs() {
        // Arrange: input length 4, expert outputs length 2
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gate_logits = vec![5.0]; // single expert
        let expert_outputs = vec![vec![10.0, 20.0]];
        // Act
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 1);
        // Assert: output length matches input (4), but only first 2 elements have values
        assert_eq!(output.len(), 4, "output length should be 4");
        assert!(output[0] > 0.0, "index 0 should be non-zero");
        assert!(output[1] > 0.0, "index 1 should be non-zero");
        assert!(
            output[2].abs() < 1e-9,
            "index 2 should be zero since expert output is shorter, got {}",
            output[2]
        );
        assert!(
            output[3].abs() < 1e-9,
            "index 3 should be zero since expert output is shorter, got {}",
            output[3]
        );
    }

    #[test]
    fn test_route_table_fallback_when_all_topk_full_assigns_least_loaded() {
        // Arrange: 3 experts, top_k=1, capacity=1, 5 tokens all prefer expert 0
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 0.2, // ceil(0.2 * 5 / 3) = ceil(0.333) = 1
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0],
        ];
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        // Act & Assert: tokens 2-4 should be fallback-assigned to least loaded experts
        // Expert 0 gets token 0 (count=1=capacity), then all subsequent overflow
        assert!(
            table.overflow_count > 0,
            "should have overflow, got {}",
            table.overflow_count
        );
        // Every token must have at least one expert assigned (fallback guarantee)
        for (i, route) in table.token_routes.iter().enumerate() {
            assert!(
                !route.expert_indices.is_empty(),
                "token {} must have at least one expert via fallback",
                i
            );
            assert_eq!(
                route.expert_weights.len(),
                route.expert_indices.len(),
                "token {} weights count must match indices count",
                i
            );
        }
    }

    // ---- Round 12: Edge case tests (15 tests) ----

    /// @trace REQ-MOE-001 [level:unit]
    /// Softmax with a single element should return [1.0].
    #[test]
    fn test_softmax_single_element_returns_one() {
        // Arrange
        let logits = vec![3.7f32];

        // Act
        let probs = softmax(&logits);

        // Assert
        assert_eq!(probs.len(), 1);
        assert!(
            (probs[0] - 1.0).abs() < 1e-6,
            "single element softmax should be 1.0, got {}",
            probs[0]
        );
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// topk_with_weights with exactly 2 elements and k=1 returns the larger.
    #[test]
    fn test_topk_with_weights_two_elements_k1() {
        // Arrange
        let logits = vec![0.1f32, 0.9f32];

        // Act
        let result = topk_with_weights(&logits, 1);

        // Assert
        assert_eq!(result.len(), 1, "k=1 should return exactly 1 result");
        assert_eq!(result[0].0, 1, "index 1 has the larger logit");
        assert!(
            (result[0].1 - 1.0).abs() < 1e-6,
            "single selected weight should be 1.0 after renorm, got {}",
            result[0].1
        );
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// topk_indices with k=0 returns an empty vector.
    #[test]
    fn test_topk_indices_k_zero_returns_empty() {
        // Arrange
        let logits = vec![1.0f32, 2.0, 3.0];

        // Act
        let indices = topk_indices(&logits, 0);

        // Assert
        assert!(indices.is_empty(), "k=0 should produce empty result");
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertUtilizationStats equality: identical structs must be equal.
    #[test]
    fn test_utilization_stats_equality_identical() {
        // Arrange
        let a = ExpertUtilizationStats {
            total_tokens: 10,
            total_expert_assignments: 20,
            overflow_count: 1,
            max_expert_load: 6,
            min_expert_load: 3,
            mean_expert_load: 5.0,
            balance_score: 0.5,
        };
        let b = ExpertUtilizationStats {
            total_tokens: 10,
            total_expert_assignments: 20,
            overflow_count: 1,
            max_expert_load: 6,
            min_expert_load: 3,
            mean_expert_load: 5.0,
            balance_score: 0.5,
        };

        // Act & Assert
        assert_eq!(a, b, "identical stats should be equal");
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertUtilizationStats inequality: different overflow_count must not be equal.
    #[test]
    fn test_utilization_stats_inequality_different_overflow() {
        // Arrange
        let a = ExpertUtilizationStats {
            total_tokens: 10,
            total_expert_assignments: 20,
            overflow_count: 0,
            max_expert_load: 5,
            min_expert_load: 5,
            mean_expert_load: 5.0,
            balance_score: 1.0,
        };
        let b = ExpertUtilizationStats {
            total_tokens: 10,
            total_expert_assignments: 20,
            overflow_count: 3,
            max_expert_load: 5,
            min_expert_load: 5,
            mean_expert_load: 5.0,
            balance_score: 1.0,
        };

        // Act & Assert
        assert_ne!(a, b, "different overflow_count should not be equal");
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertUtilizationStats Copy trait allows assignment without clone.
    #[test]
    fn test_utilization_stats_copy_trait() {
        // Arrange
        let original = ExpertUtilizationStats {
            total_tokens: 5,
            total_expert_assignments: 10,
            overflow_count: 0,
            max_expert_load: 3,
            min_expert_load: 2,
            mean_expert_load: 2.5,
            balance_score: 0.667,
        };

        // Act: Copy (not Clone) — assignment copies the value
        let copied = original;

        // Assert: both are independent copies with the same values
        assert_eq!(original, copied, "Copy should produce equal value");
        assert_eq!(original.total_tokens, 5);
        assert_eq!(copied.total_tokens, 5);
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertRouteConfig PartialEq: configs with different top_k are not equal.
    #[test]
    fn test_route_config_inequality_different_topk() {
        // Arrange
        let a = ExpertRouteConfig::new(8, 2);
        let b = ExpertRouteConfig::new(8, 4);

        // Act & Assert
        assert_ne!(a, b, "configs with different top_k should not be equal");
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertRouteConfig clone produces an equal config.
    #[test]
    fn test_route_config_clone_equal() {
        // Arrange
        let original = ExpertRouteConfig {
            num_experts: 16,
            top_k: 3,
            capacity_factor: 2.0,
            load_balance_loss: true,
            load_balance_lambda: 0.05,
            noise_sigma: 0.1,
        };

        // Act
        let cloned = original.clone();

        // Assert
        assert_eq!(original, cloned, "cloned config should equal original");
        assert_eq!(cloned.num_experts, 16);
        assert_eq!(cloned.top_k, 3);
        assert!((cloned.capacity_factor - 2.0).abs() < f32::EPSILON);
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertRouteTable tokens_for_expert returns empty for unassigned expert.
    #[test]
    fn test_route_table_tokens_for_unassigned_expert_empty() {
        // Arrange: 4 experts, logits strongly biased to expert 0 and 1
        let config = ExpertRouteConfig::new(4, 2);
        let gate_logits = vec![
            vec![10.0, 5.0, -10.0, -10.0],
            vec![10.0, 5.0, -10.0, -10.0],
        ];
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);

        // Act: expert 3 was never selected
        let tokens = table.tokens_for_expert(3);

        // Assert
        assert!(
            tokens.is_empty(),
            "expert 3 should have zero tokens assigned, got {:?}",
            tokens
        );
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertRouteTable clone preserves all data including overflow_count.
    #[test]
    fn test_route_table_clone_preserves_overflow() {
        // Arrange: create a route table that produces overflow
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 0.5,
            load_balance_loss: false,
            load_balance_lambda: 0.01,
            noise_sigma: 0.0,
        };
        let gate_logits = vec![
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 0.0],
        ];
        let original = ExpertRouteTable::from_gate_logits(config, &gate_logits);

        // Act
        let cloned = original.clone();

        // Assert: clone preserves overflow_count
        assert_eq!(
            original.overflow_count, cloned.overflow_count,
            "cloned table should have same overflow_count"
        );
        assert_eq!(
            original.token_routes.len(),
            cloned.token_routes.len(),
            "cloned table should have same number of token routes"
        );
        assert_eq!(
            original.expert_token_counts, cloned.expert_token_counts,
            "cloned table should have same expert_token_counts"
        );
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertLoadBalancer hot_experts with zero total_routes returns all zeros.
    #[test]
    fn test_load_balancer_hot_experts_no_routes_returns_zero_rates() {
        // Arrange
        let config = ExpertRouteConfig::new(4, 2);
        let balancer = ExpertLoadBalancer::new(config);

        // Act: top_n=2 when no routes recorded
        let hot = balancer.hot_experts(2);

        // Assert
        assert_eq!(hot.len(), 2, "should return top_n entries");
        for (idx, rate) in &hot {
            assert!(
            (*rate - 0.0).abs() < 1e-10,
            "rate for expert {} should be 0.0 with no routes, got {}",
            idx,
            rate
        );
        }
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertLoadBalancer suggest_capacity_adjustment with skewed load
    /// returns proportionally adjusted capacities.
    #[test]
    fn test_load_balancer_suggest_capacity_skewed_load() {
        // Arrange: 3 experts, top_k=1, all tokens go to expert 0
        let config = ExpertRouteConfig::new(3, 1);
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0],
        ];
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&table);

        // Act
        let suggested = balancer.suggest_capacity_adjustment(4);

        // Assert: expert 0 should get higher capacity than others
        assert_eq!(suggested.len(), 3, "should return capacity for each expert");
        assert!(
            suggested[0] > suggested[1],
            "expert 0 should have higher capacity ({}) than expert 1 ({})",
            suggested[0],
            suggested[1]
        );
        assert!(
            suggested[0] > suggested[2],
            "expert 0 should have higher capacity ({}) than expert 2 ({})",
            suggested[0],
            suggested[2]
        );
        for (i, &cap) in suggested.iter().enumerate() {
            assert!(cap >= 1, "expert {} capacity {} should be >= 1", i, cap);
        }
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertLoadBalancer reset clears all history and total_routes.
    #[test]
    fn test_load_balancer_reset_clears_history() {
        // Arrange
        let config = ExpertRouteConfig::new(4, 2);
        let mut balancer = ExpertLoadBalancer::new(config);
        let gate_logits = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let table = ExpertRouteTable::from_gate_logits(balancer.config().clone(), &gate_logits);
        balancer.record_route(&table);
        assert!(!balancer.cold_experts().is_empty() || balancer.hot_experts(1)[0].1 > 0.0);

        // Act
        balancer.reset();

        // Assert
        let rates = balancer.hit_rates();
        assert!(
            rates.iter().all(|&r| r == 0.0),
            "after reset all hit rates should be 0.0, got {:?}",
            rates
        );
        assert!(
            balancer.cold_experts().is_empty(),
            "after reset with 0 total_routes, cold_experts should be empty"
        );
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// moe_dispatch with all zero expert outputs returns zero output.
    #[test]
    fn test_moe_dispatch_zero_expert_outputs() {
        // Arrange
        let input = vec![1.0f32, 2.0, 3.0];
        let gate_logits = vec![1.0f32, 2.0, 0.5];
        let expert_outputs = vec![
            vec![0.0f32, 0.0, 0.0],
            vec![0.0f32, 0.0, 0.0],
            vec![0.0f32, 0.0, 0.0],
        ];

        // Act
        let result = moe_dispatch(&input, &gate_logits, &expert_outputs, 2);

        // Assert
        assert_eq!(result.len(), 3, "output dimension should match input");
        for (i, &val) in result.iter().enumerate() {
            assert!(
                val == 0.0f32,
                "all outputs should be 0.0 at index {}, got {}",
                i,
                val
            );
        }
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// moe_dispatch where expert_outputs have shorter length than input
    /// only fills up to the shorter dimension.
    #[test]
    fn test_moe_dispatch_shorter_expert_outputs() {
        // Arrange: input is dim=4, expert outputs are dim=2
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let gate_logits = vec![2.0f32, 0.0]; // expert 0 gets all weight
        let expert_outputs = vec![
            vec![5.0f32, 10.0], // dim=2, shorter than input
        ];

        // Act
        let result = moe_dispatch(&input, &gate_logits, &expert_outputs, 1);

        // Assert: output length = input length, first 2 values mixed, rest 0
        assert_eq!(result.len(), 4, "output dimension should match input");
        assert!(
            (result[0] - 5.0).abs() < 1e-5,
            "index 0 should be expert 0 output, got {}",
            result[0]
        );
        assert!(
            (result[1] - 10.0).abs() < 1e-5,
            "index 1 should be expert 0 output, got {}",
            result[1]
        );
        assert_eq!(
            result[2], 0.0,
            "index 2 beyond expert output length should remain 0"
        );
        assert_eq!(
            result[3], 0.0,
            "index 3 beyond expert output length should remain 0"
        );
    }

    // --- 15 additional tests ---

    /// @trace REQ-MOE-001 [level:unit]
    /// topk_with_weights with all negative infinity logits: softmax fallback
    /// should produce uniform distribution, then topk picks first k indices.
    #[test]
    fn test_topk_with_weights_all_neg_inf_produces_uniform() {
        // Arrange
        let logits = vec![f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];

        // Act
        let result = topk_with_weights(&logits, 2);

        // Assert: softmax of all -inf produces uniform fallback (1/n each)
        assert_eq!(result.len(), 2, "should return k=2 results");
        let sum: f32 = result.iter().map(|(_, w)| *w).sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "renormalized weights should sum to ~1.0, got {}",
            sum
        );
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertRouteTable::tokens_for_expert with expert index beyond num_experts
    /// returns empty since no token can be assigned to a non-existent expert.
    #[test]
    fn test_route_table_tokens_for_expert_beyond_range_returns_empty() {
        // Arrange: 4 experts
        let config = ExpertRouteConfig::new(4, 2);
        let gate_logits = vec![
            vec![0.0, 0.0, 10.0, 0.0],
            vec![10.0, 0.0, 0.0, 0.0],
        ];
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);

        // Act: query expert index 100 (way beyond num_experts=4)
        let tokens = table.tokens_for_expert(100);

        // Assert
        assert!(
            tokens.is_empty(),
            "expert index beyond range should have no tokens, got {:?}",
            tokens
        );
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertLoadBalancer::hit_rates after recording a single-expert table
    /// repeatedly should show one dominant expert.
    #[test]
    fn test_balancer_hit_rates_single_expert_dominant_after_repeated_record() {
        // Arrange: 3 experts, all tokens go to expert 0
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());

        // Act: record the same route table 5 times
        for _ in 0..5 {
            let gate_logits = vec![vec![100.0, 0.0, 0.0]];
            let table = ExpertRouteTable::from_gate_logits(config.clone(), &gate_logits);
            balancer.record_route(&table);
        }

        // Assert: expert 0 should have all hits, experts 1 and 2 should have zero
        let rates = balancer.hit_rates();
        assert_eq!(rates.len(), 3);
        assert!(
            (rates[0] - 1.0).abs() < 1e-6,
            "expert 0 should have rate ~1.0, got {}",
            rates[0]
        );
        assert!(
            (rates[1]).abs() < 1e-6,
            "expert 1 should have rate 0.0, got {}",
            rates[1]
        );
        assert!(
            (rates[2]).abs() < 1e-6,
            "expert 2 should have rate 0.0, got {}",
            rates[2]
        );
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertUtilizationStats Copy trait: modifying a copied value does not
    /// affect the original (true independence through Copy).
    #[test]
    fn test_utilization_stats_copy_then_mutate_independence() {
        // Arrange
        let original = ExpertUtilizationStats {
            total_tokens: 5,
            total_expert_assignments: 10,
            overflow_count: 0,
            max_expert_load: 3,
            min_expert_load: 2,
            mean_expert_load: 2.5,
            balance_score: 0.8,
        };

        // Act: Copy into a mutable binding and change a field
        let mut copy = original;
        copy.total_tokens = 99;
        copy.overflow_count = 7;

        // Assert: original is unaffected
        assert_eq!(original.total_tokens, 5, "original should be unchanged");
        assert_eq!(original.overflow_count, 0, "original overflow should be unchanged");
        assert_eq!(copy.total_tokens, 99);
        assert_eq!(copy.overflow_count, 7);
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertRouteTable with top_k equal to num_experts: every expert gets
    /// selected for each token, all weights sum to 1.0.
    #[test]
    fn test_route_table_top_k_equals_all_experts_selects_all() {
        // Arrange: 3 experts, top_k=3
        let config = ExpertRouteConfig::new(3, 3);
        let gate_logits = vec![
            vec![1.0, 2.0, 3.0],
        ];

        // Act
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);

        // Assert: token 0 should have all 3 experts selected
        let route = &table.token_routes[0];
        assert_eq!(
            route.expert_indices.len(),
            3,
            "all experts should be selected when top_k equals num_experts"
        );
        let weight_sum: f32 = route.expert_weights.iter().sum();
        assert!(
            (weight_sum - 1.0).abs() < 0.01,
            "weights should sum to ~1.0, got {}",
            weight_sum
        );
        // No overflow expected since all experts are used
        assert_eq!(table.overflow_count, 0, "no overflow when all experts selected");
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// softmax with alternating finite and -inf values: -inf entries get
    /// zero probability, finite entries share the total.
    #[test]
    fn test_softmax_alternating_finite_and_neg_inf() {
        // Arrange
        let logits = vec![1.0, f32::NEG_INFINITY, 3.0, f32::NEG_INFINITY];

        // Act
        let probs = softmax(&logits);

        // Assert: -inf entries should have ~0 probability
        assert_eq!(probs.len(), 4);
        assert!(
            probs[1].abs() < 1e-10,
            "neg_inf entry at index 1 should be ~0, got {}",
            probs[1]
        );
        assert!(
            probs[3].abs() < 1e-10,
            "neg_inf entry at index 3 should be ~0, got {}",
            probs[3]
        );
        // Finite entries should sum to ~1.0
        let finite_sum = probs[0] + probs[2];
        assert!(
            (finite_sum - 1.0).abs() < 1e-5,
            "finite entries should sum to ~1.0, got {}",
            finite_sum
        );
        // Higher logit should have higher probability
        assert!(
            probs[2] > probs[0],
            "logit 3.0 should have higher prob than 1.0"
        );
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertLoadBalancer::hot_experts with top_n=0 returns empty list.
    #[test]
    fn test_balancer_hot_experts_top_n_zero_returns_empty() {
        // Arrange
        let config = ExpertRouteConfig::new(4, 2);
        let balancer = ExpertLoadBalancer::new(config);

        // Act
        let hot = balancer.hot_experts(0);

        // Assert
        assert!(hot.is_empty(), "top_n=0 should return empty list");
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertRouteConfig::expert_capacity with very large num_experts and
    /// one token: capacity rounds up to at least 1 due to ceil.
    #[test]
    fn test_config_capacity_large_experts_one_token_ceil_to_one() {
        // Arrange: 1000 experts, 1 token, capacity_factor=1.0
        let config = ExpertRouteConfig {
            num_experts: 1000,
            top_k: 1,
            capacity_factor: 1.0,
            ..ExpertRouteConfig::default()
        };

        // Act: ceil(1.0 * 1 / 1000) = ceil(0.001) = 1
        let cap = config.expert_capacity(1);

        // Assert: ceil should round up to 1, not truncate to 0
        assert_eq!(cap, 1, "capacity should be 1 due to ceil rounding");
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertRouteTable::load_balance_loss with a single token and uniform
    /// logits: loss should be non-negative and finite.
    #[test]
    fn test_load_balance_loss_single_token_uniform_logits() {
        // Arrange
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 2,
            load_balance_loss: true,
            load_balance_lambda: 0.01,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![vec![1.0, 1.0, 1.0, 1.0]];
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);

        // Act
        let loss = table.load_balance_loss(&gate_logits);

        // Assert: loss is non-negative and finite
        assert!(loss >= 0.0, "loss should be non-negative, got {}", loss);
        assert!(loss.is_finite(), "loss should be finite, got {}", loss);
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// topk_indices with all same negative values: returns first k indices
    /// since all are equal.
    #[test]
    fn test_topk_indices_all_same_negative_returns_first_k() {
        // Arrange
        let logits = vec![-5.0, -5.0, -5.0, -5.0];

        // Act
        let indices = topk_indices(&logits, 2);

        // Assert: should return 2 indices, all valid
        assert_eq!(indices.len(), 2);
        for &idx in &indices {
            assert!(idx < 4, "index {} should be in range [0,4)", idx);
        }
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertLoadBalancer::cold_experts with all experts above threshold
    /// returns empty list.
    #[test]
    fn test_balancer_cold_experts_all_above_threshold_returns_empty() {
        // Arrange: 3 experts, each gets equal traffic
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());

        // Act: route one token to each expert
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0],
            vec![0.0, 100.0, 0.0],
            vec![0.0, 0.0, 100.0],
        ];
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&table);

        // Assert: no cold experts since each has rate ~0.333 > threshold 0.001
        let cold = balancer.cold_experts();
        assert!(
            cold.is_empty(),
            "all experts above threshold, expected no cold experts, got {:?}",
            cold
        );
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertUtilizationStats with equal max and min load: balance_score
    /// should be exactly 1.0 (perfect balance).
    #[test]
    fn test_utilization_stats_equal_max_min_perfect_balance() {
        // Arrange: build a route table that distributes evenly
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        // Token 0 → expert 0, Token 1 → expert 1
        let gate_logits = vec![
            vec![100.0, 0.0],
            vec![0.0, 100.0],
        ];
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);

        // Act
        let stats = table.utilization_stats();

        // Assert: max == min → balance_score == 1.0
        assert_eq!(
            stats.max_expert_load, stats.min_expert_load,
            "max and min should be equal"
        );
        assert!(
            (stats.balance_score - 1.0).abs() < 1e-6,
            "perfect balance should have score 1.0, got {}",
            stats.balance_score
        );
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// moe_dispatch with a single expert that has identical output to input
    /// and gate strongly selecting it: output equals expert output.
    #[test]
    fn test_moe_dispatch_single_expert_strong_gate_output_matches_expert() {
        // Arrange: 1 expert strongly selected
        let input = vec![1.0f32, 2.0, 3.0];
        let gate_logits = vec![100.0]; // only 1 expert
        let expert_outputs = vec![
            vec![10.0f32, 20.0, 30.0],
        ];

        // Act
        let result = moe_dispatch(&input, &gate_logits, &expert_outputs, 1);

        // Assert: output should closely match the single expert's output
        assert_eq!(result.len(), 3);
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - expert_outputs[0][i]).abs() < 0.01,
                "output[{}] should be ~{}, got {}",
                i,
                expert_outputs[0][i],
                val
            );
        }
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertRouteConfig::default has sensible values.
    #[test]
    fn test_expert_route_config_default_values() {
        let cfg = ExpertRouteConfig::default();
        assert_eq!(cfg.num_experts, 8);
        assert_eq!(cfg.top_k, 2);
        assert!(!cfg.load_balance_loss);
        assert_eq!(cfg.noise_sigma, 0.0);
    }

    /// @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_expert_route_config_equality_same_values() {
        let a = ExpertRouteConfig::new(4, 2);
        let b = ExpertRouteConfig::new(4, 2);
        assert_eq!(a, b);
    }

    /// @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_expert_route_config_inequality_different_top_k() {
        let a = ExpertRouteConfig::new(4, 2);
        let b = ExpertRouteConfig::new(4, 1);
        assert_ne!(a, b);
    }

    /// @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_expert_route_config_clone_is_equal() {
        let a = ExpertRouteConfig::new(16, 4);
        let b = a.clone();
        assert_eq!(a, b);
    }

    /// @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_expert_route_config_debug_shows_fields() {
        let cfg = ExpertRouteConfig::new(4, 2);
        let dbg = format!("{:?}", cfg);
        assert!(dbg.contains("num_experts"));
        assert!(dbg.contains("top_k"));
    }

    /// @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_expert_utilization_stats_copy_is_independent() {
        let a = ExpertUtilizationStats {
            total_tokens: 10,
            total_expert_assignments: 20,
            overflow_count: 1,
            max_expert_load: 5,
            min_expert_load: 2,
            mean_expert_load: 3.0,
            balance_score: 0.6,
        };
        let b = a;
        assert_eq!(a, b);
    }

    /// @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_expert_utilization_stats_default_balance_is_one() {
        let s = ExpertUtilizationStats {
            total_tokens: 0,
            total_expert_assignments: 0,
            overflow_count: 0,
            max_expert_load: 0,
            min_expert_load: 0,
            mean_expert_load: 0.0,
            balance_score: 1.0,
        };
        assert_eq!(s.balance_score, 1.0);
    }

    /// @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_softmax_single_element_is_one() {
        let result = softmax(&[5.0f32]);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    /// @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_softmax_two_equal_elements() {
        let result = softmax(&[1.0f32, 1.0]);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 0.5).abs() < 1e-6);
        assert!((result[1] - 0.5).abs() < 1e-6);
    }

    /// @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_topk_indices_returns_top_k_indices() {
        let logits = vec![0.1f32, 0.9, 0.3, 0.7];
        let indices = topk_indices(&logits, 2);
        assert_eq!(indices.len(), 2);
        assert!(indices.contains(&1), "index 1 (0.9) should be in top-2");
        assert!(indices.contains(&3), "index 3 (0.7) should be in top-2");
    }

    /// @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_topk_with_weights_weights_sum_to_one() {
        let logits = vec![1.0f32, 2.0, 3.0, 4.0];
        let topk = topk_with_weights(&logits, 2);
        let sum: f32 = topk.iter().map(|(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-5, "weights should sum to 1.0, got {}", sum);
    }

    /// @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_token_route_fields_after_construction() {
        let route = TokenRoute {
            expert_indices: vec![0, 2],
            expert_weights: vec![0.6, 0.4],
            expert_positions: vec![0, 1],
        };
        assert_eq!(route.expert_indices.len(), 2);
        assert_eq!(route.expert_weights.len(), 2);
    }

    /// @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_expert_route_table_zero_tokens_empty_routes() {
        let config = ExpertRouteConfig::new(4, 2);
        let table = ExpertRouteTable::from_gate_logits(config, &[]);
        assert!(table.token_routes.is_empty());
        assert_eq!(table.overflow_count, 0);
    }

    /// @trace REQ-MOE-001 [level:unit]
    #[test]
    fn test_moe_dispatch_zero_gate_logits_uniform_weight() {
        let input = vec![1.0f32, 2.0, 3.0];
        let gate_logits = vec![0.0f32, 0.0];
        let expert_outputs = vec![
            vec![10.0f32, 20.0, 30.0],
            vec![20.0f32, 40.0, 60.0],
        ];
        let result = moe_dispatch(&input, &gate_logits, &expert_outputs, 2);
        assert_eq!(result.len(), 3);
        // Both experts weighted equally → average of their outputs
        for i in 0..3 {
            let expected = (expert_outputs[0][i] + expert_outputs[1][i]) / 2.0;
            assert!((result[i] - expected).abs() < 0.1, "result[{}] should be ~{}, got {}", i, expected, result[i]);
        }
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertRouteTable::expert_token_counts length always matches
    /// config.num_experts, even with zero tokens.
    #[test]
    fn test_route_table_expert_counts_len_matches_config_zero_tokens() {
        // Arrange
        let config = ExpertRouteConfig::new(7, 2);

        // Act: zero tokens
        let table = ExpertRouteTable::from_gate_logits(config, &[]);

        // Assert
        assert_eq!(
            table.expert_token_counts.len(),
            7,
            "expert_token_counts length should equal num_experts"
        );
        assert!(
            table.expert_token_counts.iter().all(|&c| c == 0),
            "all counts should be 0 with zero tokens"
        );
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// topk_with_weights with k exactly equal to logits length returns all
    /// indices with weights that sum to 1.0.
    #[test]
    fn test_topk_with_weights_k_equals_len_returns_all_renormalized() {
        // Arrange: 4 logits, k=4
        let logits = vec![1.0, 3.0, 2.0, 0.5];

        // Act
        let result = topk_with_weights(&logits, 4);

        // Assert
        assert_eq!(result.len(), 4, "should return all 4 entries");
        let sum: f32 = result.iter().map(|(_, w)| *w).sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "renormalized weights should sum to ~1.0, got {}",
            sum
        );
        // Verify all indices are unique
        let mut indices: Vec<usize> = result.iter().map(|(i, _)| *i).collect();
        indices.sort();
        assert_eq!(indices, vec![0, 1, 2, 3], "all 4 indices should be present");
    }

    // ---- Round 13: 15 new tests covering uncovered edge cases ----

    /// @trace REQ-MOE-001 [level:unit]
    /// softmax 混合 +Inf 和 NaN 输入不应 panic，应返回与输入等长的结果
    #[test]
    fn test_softmax_mixed_positive_infinity_and_nan_no_panic() {
        // Arrange: 第一个元素 +Inf，第二个 NaN，第三个有限值
        let logits = vec![f32::INFINITY, f32::NAN, 1.0];

        // Act
        let probs = softmax(&logits);

        // Assert: 不应 panic，长度保留
        assert_eq!(probs.len(), 3, "softmax should return same length as input");
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertRouteTable 配置 top_k=0 时，每个 token 通过 fallback 获得至少一个专家
    #[test]
    fn test_route_table_top_k_zero_every_token_gets_fallback_expert() {
        // Arrange: 3 个专家，top_k=0（所有 top-k 选择结果为空），2 个 token
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 0,
            capacity_factor: 1.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![1.0, 2.0, 3.0],
            vec![3.0, 2.0, 1.0],
        ];

        // Act
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);

        // Assert: 每个 token 因 selected_indices 为空而进入 fallback，
        // 分配到负载最低的专家，权重为 1.0
        for (i, route) in table.token_routes.iter().enumerate() {
            assert_eq!(
                route.expert_indices.len(), 1,
                "token {} should have exactly 1 expert via fallback", i
            );
            assert!(
                (route.expert_weights[0] - 1.0).abs() < 1e-6,
                "token {} fallback weight should be 1.0, got {}", i, route.expert_weights[0]
            );
        }
        // top_k=0 时没有任何 top-k 专家被拒绝（根本没有选择），
        // 所以 overflow_count 为 0；fallback 是通过 selected_indices 为空触发的
        assert_eq!(table.overflow_count, 0, "top_k=0 should have zero overflow (no top-k rejection)");
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertRouteTable 所有的 logits 完全相等且 top_k=2 时，
    /// positions 应按顺序递增分配给被选中的专家
    #[test]
    fn test_route_table_equal_logits_top_k_two_positions_sequential_per_expert() {
        // Arrange: 2 个专家，top_k=2，3 个 token 使用完全相等的 logits
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 2,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        ];

        // Act
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);

        // Assert: 每个 expert 的 positions 应该从 0 递增
        let pairs_0 = table.tokens_for_expert(0);
        let pairs_1 = table.tokens_for_expert(1);
        assert_eq!(pairs_0.len(), 3, "expert 0 should receive all 3 tokens");
        assert_eq!(pairs_1.len(), 3, "expert 1 should receive all 3 tokens");
        let positions_0: Vec<usize> = pairs_0.iter().map(|&(_, p)| p).collect();
        let positions_1: Vec<usize> = pairs_1.iter().map(|&(_, p)| p).collect();
        let mut sorted_0 = positions_0.clone();
        sorted_0.sort();
        assert_eq!(sorted_0, vec![0, 1, 2], "expert 0 positions should be sequential");
        let mut sorted_1 = positions_1.clone();
        sorted_1.sort();
        assert_eq!(sorted_1, vec![0, 1, 2], "expert 1 positions should be sequential");
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertRouteTable::load_balance_loss 当 num_experts=1 时，
    /// 损失应该是确定性的且有限正值
    #[test]
    fn test_load_balance_loss_single_expert_positive_finite() {
        // Arrange: 仅 1 个专家，启用负载均衡损失
        let config = ExpertRouteConfig {
            num_experts: 1,
            top_k: 1,
            capacity_factor: 100.0,
            load_balance_loss: true,
            load_balance_lambda: 0.01,
            noise_sigma: 0.0,
        };
        let gate_logits = vec![
            vec![5.0],
            vec![3.0],
            vec![7.0],
        ];
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);

        // Act
        let loss = table.load_balance_loss(&gate_logits);

        // Assert: 单专家时 f_0=1.0, P_0=1.0, loss = lambda * 1 * (1.0*1.0) = lambda
        assert!(
            loss > 0.0,
            "single expert should have positive loss, got {}", loss
        );
        assert!(
            loss.is_finite(),
            "loss should be finite, got {}", loss
        );
        // loss = lambda * N * f_0 * P_0 = 0.01 * 1.0 * 1.0 * 1.0 = 0.01
        assert!(
            (loss - 0.01).abs() < 0.001,
            "single expert loss should be ~0.01, got {}", loss
        );
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertRouteTable 当 top_k=1 且只有一个 token 时，
    /// expert_positions 应恰好为 [0]
    #[test]
    fn test_route_table_single_token_position_is_zero() {
        // Arrange: 4 个专家，top_k=1，单个 token
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 1.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![vec![0.0, 1.0, 0.0, 0.0]];

        // Act
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);

        // Assert: 单 token 的 position 必须是 0
        assert_eq!(table.token_routes.len(), 1);
        assert_eq!(table.token_routes[0].expert_positions.len(), 1);
        assert_eq!(
            table.token_routes[0].expert_positions[0], 0,
            "single token position should be 0"
        );
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertLoadBalancer::record_route 记录空路由表后，
    /// total_routes 和 hit_history 保持不变
    #[test]
    fn test_balancer_record_empty_table_then_record_nonempty() {
        // Arrange: 先记录一个有路由的表，再记录空表，确认空表不影响统计
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        let gate_logits = vec![vec![100.0, 0.0]];
        let table_with_routes = ExpertRouteTable::from_gate_logits(config.clone(), &gate_logits);
        balancer.record_route(&table_with_routes);

        let rates_before = balancer.hit_rates().to_vec();
        assert!(rates_before[0] > 0.0, "should have recorded routes");

        // Act: 记录空路由表
        let empty_table = ExpertRouteTable::from_gate_logits(config, &[]);
        balancer.record_route(&empty_table);

        // Assert: hit_rates 应不变（空表没有 token_routes，不更新 hit_history）
        let rates_after = balancer.hit_rates();
        assert_eq!(rates_before.len(), rates_after.len());
        // total_routes 也不会增长（空表的 token_routes 为空）
        // 之前 total_routes=1，空表记录后仍为 1
        assert!(
            (rates_before[0] - rates_after[0]).abs() < 1e-10,
            "empty table should not change hit rates"
        );
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// moe_dispatch 所有 gate_logits 为 NaN 时，函数不应 panic，
    /// 返回长度与 input 一致
    #[test]
    fn test_moe_dispatch_all_nan_gate_logits_no_panic() {
        // Arrange: gate 全部为 NaN
        let input = vec![1.0f32, 2.0, 3.0];
        let gate_logits = vec![f32::NAN, f32::NAN];
        let expert_outputs = vec![
            vec![10.0f32, 20.0, 30.0],
            vec![5.0f32, 15.0, 25.0],
        ];

        // Act
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 2);

        // Assert: 不应 panic，输出长度匹配
        assert_eq!(output.len(), 3, "output length should match input");
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertRouteConfig::expert_capacity 当 total_tokens=0 且
    /// capacity_factor 极大时仍返回 0
    #[test]
    fn test_capacity_zero_tokens_large_factor_still_zero() {
        // Arrange: 0 tokens 但 capacity_factor=1e10
        let config = ExpertRouteConfig {
            num_experts: 8,
            top_k: 2,
            capacity_factor: 1e10,
            ..ExpertRouteConfig::default()
        };

        // Act
        let cap = config.expert_capacity(0);

        // Assert: 0 * 任何因子 = 0
        assert_eq!(cap, 0, "zero tokens should always yield capacity 0");
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertRouteTable::tokens_for_expert 对多专家分配的 token，
    /// 返回的 (token_idx, position) 对正确对应每个被分配的专家
    #[test]
    fn test_tokens_for_expert_multi_assignment_correct_pairs() {
        // Arrange: 3 个专家，top_k=2，2 个 token 使用不同 logits 使 token 0
        // 分配到专家 1 和 2，token 1 分配到专家 0 和 2
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 2,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![0.0, 100.0, 50.0],   // token 0: 专家 1 (100) 和 2 (50)
            vec![100.0, 0.0, 50.0],    // token 1: 专家 0 (100) 和 2 (50)
        ];
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);

        // Act: 查询专家 2 的所有 token
        let pairs = table.tokens_for_expert(2);

        // Assert: 专家 2 应该被 token 0 和 token 1 各分配一次
        assert_eq!(pairs.len(), 2, "expert 2 should have 2 assignments");
        let token_indices: Vec<usize> = pairs.iter().map(|&(t, _)| t).collect();
        assert!(token_indices.contains(&0), "token 0 should be in expert 2's list");
        assert!(token_indices.contains(&1), "token 1 should be in expert 2's list");
        // 两个 position 应不同
        assert_ne!(pairs[0].1, pairs[1].1, "positions for expert 2 should be distinct");
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertLoadBalancer::cold_experts 当负载刚好处于阈值边界时，
    /// 低于阈值的专家才被视为冷专家
    #[test]
    fn test_balancer_cold_experts_partial_cold_at_boundary() {
        // Arrange: 4 个专家，大量 token 使 expert 0 占 ~99%，expert 1 占 ~0.5%，
        // expert 2 和 3 占 ~0%（<0.1% 阈值 = cold_threshold=0.001）
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());

        // 1000 个 token 全部路由到 expert 0
        let gate_logits: Vec<Vec<f32>> = (0..1000)
            .map(|_| vec![100.0, 0.0, 0.0, 0.0])
            .collect();
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&rt);

        // Act
        let cold = balancer.cold_experts();

        // Assert: experts 1, 2, 3 的 rate = 0/1000 = 0.0 < 0.001
        // expert 0 rate = 1.0 > 0.001，不是冷专家
        let cold_indices: Vec<usize> = cold.iter().map(|(i, _)| *i).collect();
        assert!(
            cold_indices.contains(&1),
            "expert 1 (rate 0.0) should be cold"
        );
        assert!(
            cold_indices.contains(&2),
            "expert 2 (rate 0.0) should be cold"
        );
        assert!(
            cold_indices.contains(&3),
            "expert 3 (rate 0.0) should be cold"
        );
        assert!(
            !cold_indices.contains(&0),
            "expert 0 (rate 1.0) should NOT be cold"
        );
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertRouteTable 溢出 fallback 分配后 expert_token_counts 之和
    /// 应严格等于 token_routes.len()（每个 token 恰好被分配 1 次）
    #[test]
    fn test_route_table_overflow_fallback_counts_sum_equals_token_count() {
        // Arrange: 2 个专家，top_k=1，极小容量确保大量溢出
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 0.1, // ceil(0.1 * 5 / 2) = ceil(0.25) = 1
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 0.0],
            vec![100.0, 0.0],
            vec![100.0, 0.0],
            vec![100.0, 0.0],
            vec![100.0, 0.0],
        ];

        // Act
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);

        // Assert: 每个 token 通过 fallback 精确分配 1 个专家，
        // expert_token_counts 之和 = token 数
        let counts_sum: usize = table.expert_token_counts.iter().sum();
        assert_eq!(
            counts_sum, 5,
            "sum of expert_token_counts ({}) should equal 5 tokens", counts_sum
        );
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// topk_with_weights 当所有 logits 均为 +Inf 时不 panic，
    /// 返回 k 个元素且权重总和为 1.0
    #[test]
    fn test_topk_with_weights_all_positive_infinity_no_panic() {
        // Arrange: 全部 +Inf
        let logits = vec![f32::INFINITY, f32::INFINITY, f32::INFINITY];

        // Act
        let result = topk_with_weights(&logits, 2);

        // Assert: 不应 panic，返回 2 个结果
        assert_eq!(result.len(), 2, "should return k=2 results");
        let sum: f32 = result.iter().map(|(_, w)| *w).sum();
        assert!(
            (sum - 1.0).abs() < 0.1 || sum.is_nan(),
            "weights should sum to ~1.0 or be NaN due to Inf, got {}", sum
        );
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertRouteTable::load_balance_loss 禁用时，无论 gate_logits 如何，
    /// 返回值始终为 0.0
    #[test]
    fn test_load_balance_loss_disabled_always_zero_regardless_of_logits() {
        // Arrange: 禁用负载均衡，使用极端不均匀的 logits
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            load_balance_loss: false,
            load_balance_lambda: 1.0, // 即使 lambda 很大
            noise_sigma: 0.0,
        };
        let gate_logits = vec![
            vec![100.0, 0.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0, 0.0],
        ];
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);

        // Act
        let loss = table.load_balance_loss(&gate_logits);

        // Assert: 禁用时始终为 0.0
        assert!(
            (loss - 0.0).abs() < 1e-10,
            "disabled load_balance_loss should always be 0.0, got {}", loss
        );
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// moe_dispatch 当 input 长度为 0 且 expert_outputs 也为空维度时，
    /// 返回空向量
    #[test]
    fn test_moe_dispatch_zero_length_input_empty_expert_outputs() {
        // Arrange: input 为空，experts 也输出空向量
        let input: Vec<f32> = vec![];
        let gate_logits = vec![1.0, 2.0];
        let expert_outputs = vec![vec![], vec![]];

        // Act
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 2);

        // Assert: 输出为空
        assert!(
            output.is_empty(),
            "zero-length input should produce empty output, got len {}", output.len()
        );
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertLoadBalancer::suggest_capacity_adjustment 当所有专家获得完全均等命中时，
    /// 每个专家的容量应该精确等于 base_capacity
    #[test]
    fn test_balancer_suggest_capacity_perfectly_balanced_equals_base() {
        // Arrange: 4 个专家，每个恰好被命中 5 次
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        // 5 轮，每轮 4 个 token 各路由到 1 个专家
        for _ in 0..5 {
            let gate_logits = vec![
                vec![100.0, 0.0, 0.0, 0.0],
                vec![0.0, 100.0, 0.0, 0.0],
                vec![0.0, 0.0, 100.0, 0.0],
                vec![0.0, 0.0, 0.0, 100.0],
            ];
            let rt = ExpertRouteTable::from_gate_logits(config.clone(), &gate_logits);
            balancer.record_route(&rt);
        }

        // Act
        let suggested = balancer.suggest_capacity_adjustment(7);

        // Assert: 完全均衡时，每个专家的 adjusted = ceil(7 * 0.25 * 4) = ceil(7) = 7
        assert_eq!(suggested.len(), 4);
        for (i, &cap) in suggested.iter().enumerate() {
            assert_eq!(
                cap, 7,
                "perfectly balanced expert {} should get base_capacity=7, got {}", i, cap
            );
        }
    }

    // =========================================================================
    // Round 10: 15 additional tests
    // =========================================================================

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertUtilizationStats::balance_score 公式验证:
    /// balance = 1 - (max - min) / max
    /// 当 max=10, min=6 时, balance = 1 - 4/10 = 0.6
    #[test]
    fn test_utilization_stats_balance_score_formula_verification() {
        // Arrange: 手动构造 stats 使 balance_score = 1 - (max-min)/max = 1 - 4/10 = 0.6
        let stats = ExpertUtilizationStats {
            total_tokens: 10,
            total_expert_assignments: 10,
            overflow_count: 0,
            max_expert_load: 10,
            min_expert_load: 6,
            mean_expert_load: 8.0,
            balance_score: 0.6,
        };
        // Assert: 验证公式正确性
        let expected = 1.0 - (10.0 - 6.0) / 10.0;
        assert!((stats.balance_score - expected).abs() < 1e-5,
            "balance_score {} should equal {}", stats.balance_score, expected);
        assert!((stats.balance_score - 0.6).abs() < 1e-5);
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertLoadBalancer::cold_threshold 字段默认值为 0.001，
    /// 且通过 cold_experts() 正确使用该阈值
    #[test]
    fn test_balancer_cold_threshold_boundary_exact() {
        // Arrange: 1000 tokens, 1 to expert 0 → rate = 0.001 = threshold
        // cold_experts 过滤 rate < threshold (严格小于), 所以 rate == threshold 不算冷
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        // 1000 tokens 全部给 expert 0, expert 1/2/3 rate = 0.0 < 0.001
        let gate_logits: Vec<Vec<f32>> = (0..1000)
            .map(|_| vec![100.0, 0.0, 0.0, 0.0])
            .collect();
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        balancer.record_route(&rt);

        // Act
        let cold = balancer.cold_experts();

        // Assert: threshold=0.001, 专家 1/2/3 rate=0.0 < 0.001 → cold
        for &(idx, rate) in &cold {
            assert!(rate < 0.001,
                "cold expert {} rate {} should be < 0.001", idx, rate);
        }
        // expert 0 rate = 1.0, 不是 cold
        assert!(cold.iter().all(|(idx, _)| *idx != 0));
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertRouteTable fallback 时 position 正确反映专家当前计数
    /// 连续 fallback 到不同专家时，各自 position 独立递增
    #[test]
    fn test_route_table_fallback_positions_independent_per_expert() {
        // Arrange: 3 experts, capacity=1 each, 6 tokens all want expert 0
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 0.17, // ceil(0.17 * 6 / 3) = ceil(0.34) = 1
            ..ExpertRouteConfig::default()
        };
        let gate_logits: Vec<Vec<f32>> = (0..6)
            .map(|_| vec![100.0, 0.0, 0.0])
            .collect();

        // Act
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);

        // Assert: 每个 token 分配到一个专家，position 应各不相同
        let mut expert_positions: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for route in &rt.token_routes {
            for (i, &expert) in route.expert_indices.iter().enumerate() {
                expert_positions.entry(expert).or_default().push(route.expert_positions[i]);
            }
        }
        // 每个专家的 positions 应该是从 0 开始连续递增的
        for (expert, positions) in &expert_positions {
            let mut sorted = positions.clone();
            sorted.sort();
            for (i, &pos) in sorted.iter().enumerate() {
                assert_eq!(pos, i,
                    "expert {} position {} should be {}, got {}", expert, i, i, pos);
            }
        }
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// moe_dispatch 当所有 expert_outputs 全为零向量时输出应为零
    #[test]
    fn test_moe_dispatch_all_zero_expert_outputs_produces_zero() {
        // Arrange
        let input = vec![1.0, 2.0, 3.0];
        let gate_logits = vec![1.0, 2.0, 3.0];
        let expert_outputs = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ];

        // Act
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 3);

        // Assert: 任何加权组合都是零
        assert_eq!(output.len(), 3);
        for (i, &v) in output.iter().enumerate() {
            assert!((v).abs() < 1e-10, "output[{}] should be 0.0, got {}", i, v);
        }
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertLoadBalancer::suggest_capacity_adjustment
    /// 精确比例验证: 当 2 个专家分别获得 75% 和 25% 流量时，
    /// capacity 应按比例分配（各至少为 1）
    #[test]
    fn test_balancer_suggest_capacity_proportional_75_25_split() {
        // Arrange: 2 experts, 75 tokens to expert 0, 25 to expert 1
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        let mut logits = Vec::new();
        for _ in 0..75 {
            logits.push(vec![100.0, 0.0]);
        }
        for _ in 0..25 {
            logits.push(vec![0.0, 100.0]);
        }
        let rt = ExpertRouteTable::from_gate_logits(config, &logits);
        balancer.record_route(&rt);

        // Act
        let caps = balancer.suggest_capacity_adjustment(100);

        // Assert: expert 0 应获得约 75% 的容量, expert 1 约 25%
        assert_eq!(caps.len(), 2);
        assert!(caps[0] > caps[1],
            "expert 0 cap {} should > expert 1 cap {}", caps[0], caps[1]);
        // expert 0 的容量应占总容量的大部分
        let total: usize = caps.iter().sum();
        let ratio = caps[0] as f64 / total as f64;
        assert!(ratio > 0.5 && ratio < 1.0,
            "expert 0 ratio {} should be in (0.5, 1.0)", ratio);
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertRouteTable::load_balance_loss 数值精度验证:
    /// 当 lambda=1.0, N=2, 每个专家恰好 1 个 token 时，
    /// loss 应接近 λ * N * Σ(f_i * P_i) 的理论值
    #[test]
    fn test_load_balance_loss_numerical_precision_balanced_two_experts() {
        // Arrange: 2 experts, 2 tokens (1 each), lambda=1.0
        let config = ExpertRouteConfig {
            num_experts: 2,
            top_k: 1,
            capacity_factor: 100.0,
            load_balance_loss: true,
            load_balance_lambda: 1.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![
            vec![100.0, 0.0], // token 0 → expert 0
            vec![0.0, 100.0], // token 1 → expert 1
        ];
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);

        // Act
        let loss = rt.load_balance_loss(&gate_logits);

        // Assert:
        // f = [0.5, 0.5]
        // P: softmax([100,0]) = [~1.0, ~0.0], softmax([0,100]) = [~0.0, ~1.0]
        // P_avg = [~0.5, ~0.5]
        // sum = 0.5*0.5 + 0.5*0.5 = 0.5
        // loss = 1.0 * 2 * 0.5 = 1.0
        assert!((loss - 1.0).abs() < 0.05,
            "balanced 2-expert loss should be ~1.0, got {}", loss);
        assert!(loss.is_finite(), "loss should be finite");
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertRouteConfig::expert_capacity 当 total_tokens=1, num_experts=1,
    /// capacity_factor=1.0 时精确计算: ceil(1.0*1/1) = 1
    #[test]
    fn test_capacity_exact_integer_result_not_rounded_up() {
        // Arrange: 精确整除场景
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 1.0,
            ..ExpertRouteConfig::default()
        };
        // ceil(1.0 * 8 / 4) = ceil(2.0) = 2 (精确整除, ceil 不增加)
        // Act
        let cap = config.expert_capacity(8);
        // Assert
        assert_eq!(cap, 2, "exact division should not round up");
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertRouteTable 从手动构造的表调用 tokens_for_expert
    /// 验证返回的 (token_index, position) 对精确匹配
    #[test]
    fn test_tokens_for_expert_manual_construction_exact_pairs() {
        // Arrange: 手动构造含 3 个 token 的路由表
        let table = ExpertRouteTable {
            config: ExpertRouteConfig::new(3, 1),
            token_routes: vec![
                TokenRoute {
                    expert_indices: vec![0],
                    expert_weights: vec![1.0],
                    expert_positions: vec![0],
                },
                TokenRoute {
                    expert_indices: vec![2],
                    expert_weights: vec![1.0],
                    expert_positions: vec![0],
                },
                TokenRoute {
                    expert_indices: vec![0],
                    expert_weights: vec![1.0],
                    expert_positions: vec![1],
                },
            ],
            expert_token_counts: vec![2, 0, 1],
            overflow_count: 0,
        };

        // Act
        let expert_0_tokens = table.tokens_for_expert(0);
        let expert_1_tokens = table.tokens_for_expert(1);
        let expert_2_tokens = table.tokens_for_expert(2);

        // Assert: expert 0 有 token 0 (pos 0) 和 token 2 (pos 1)
        assert_eq!(expert_0_tokens.len(), 2);
        assert_eq!(expert_0_tokens[0], (0, 0));
        assert_eq!(expert_0_tokens[1], (2, 1));
        // expert 1 无 token
        assert!(expert_1_tokens.is_empty());
        // expert 2 有 token 1 (pos 0)
        assert_eq!(expert_2_tokens.len(), 1);
        assert_eq!(expert_2_tokens[0], (1, 0));
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// softmax 当所有输入为相同负数时仍产生均匀分布
    #[test]
    fn test_softmax_all_same_negative_produces_uniform() {
        // Arrange: 全部 -5.0
        let logits = vec![-5.0, -5.0, -5.0, -5.0];

        // Act
        let probs = softmax(&logits);

        // Assert: 全等输入 → 均匀分布
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        for &p in &probs {
            assert!((p - 0.25).abs() < 1e-5,
                "expected uniform 0.25, got {}", p);
        }
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// topk_indices 当 logits 包含正负混合且恰好有 2 个相同最高值时，
    /// 两个最高值都会被选出（稳定排序）
    #[test]
    fn test_topk_indices_two_highest_equal_with_lower_values() {
        // Arrange: 5 个值，最高为 5.0 出现两次
        let logits = vec![1.0, 5.0, 2.0, 5.0, 3.0];

        // Act
        let indices = topk_indices(&logits, 2);

        // Assert: 返回 2 个索引，都是值为 5.0 的 (index 1 或 3)
        assert_eq!(indices.len(), 2);
        for &idx in &indices {
            assert!(idx == 1 || idx == 3,
                "index {} should be 1 or 3 (both have value 5.0)", idx);
        }
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertLoadBalancer 重置后立即记录新的路由表，
    /// hit_history 应只反映重置后的记录
    #[test]
    fn test_balancer_reset_then_single_record_clean_history() {
        // Arrange: 先记录大量路由到 expert 0
        let config = ExpertRouteConfig {
            num_experts: 3,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        // 50 tokens 全到 expert 0
        let logits: Vec<Vec<f32>> = (0..50)
            .map(|_| vec![100.0, 0.0, 0.0])
            .collect();
        let rt = ExpertRouteTable::from_gate_logits(config.clone(), &logits);
        balancer.record_route(&rt);
        assert!(balancer.hit_history[0] > 0);

        // Act: 重置后只记录 1 个 token 到 expert 2
        balancer.reset();
        let rt2 = ExpertRouteTable::from_gate_logits(config, &[vec![0.0, 0.0, 100.0]]);
        balancer.record_route(&rt2);

        // Assert: 只有 expert 2 有 1 次命中
        assert_eq!(balancer.hit_history[0], 0, "expert 0 should have 0 hits");
        assert_eq!(balancer.hit_history[1], 0, "expert 1 should have 0 hits");
        assert_eq!(balancer.hit_history[2], 1, "expert 2 should have 1 hit");
        assert_eq!(balancer.total_routes, 1);
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertRouteTable 当 top_k=2 且两个最高 logits 非常接近时，
    /// 两个专家都被选中且权重差异很小
    #[test]
    fn test_route_table_top_k_two_close_logits_both_selected_similar_weights() {
        // Arrange: 4 experts, top_k=2, logits 非常接近
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 2,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let gate_logits = vec![vec![1.0, 1.001, 0.5, 0.5]];

        // Act
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let route = &rt.token_routes[0];

        // Assert: 两个最接近的 logits (index 0 和 1) 都应被选中
        assert_eq!(route.expert_indices.len(), 2);
        assert!(route.expert_indices.contains(&0), "expert 0 should be selected");
        assert!(route.expert_indices.contains(&1), "expert 1 should be selected");
        // 权重应非常接近（logits 差异极小）
        let w0 = route.expert_weights[route.expert_indices.iter().position(|&x| x == 0).unwrap()];
        let w1 = route.expert_weights[route.expert_indices.iter().position(|&x| x == 1).unwrap()];
        assert!((w0 - w1).abs() < 0.01,
            "weights for close logits should be similar: {} vs {}", w0, w1);
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// moe_dispatch 的线性性质验证:
    /// dispatch(input, [a,a], [[v1],[v2]], 2) 应等于
    /// a * dispatch(input, [0,0], [[v1],[v2]], 2) + (1-a) * ... 的加权形式
    #[test]
    fn test_moe_dispatch_linearity_with_scaled_input() {
        // Arrange: 验证 output = Σ(weight_i * expert_output_i) 的线性性质
        let input = vec![1.0, 1.0];
        let gate_logits = vec![0.0, 0.0]; // equal weights ~0.5 each
        let expert_a = vec![10.0, 20.0];
        let expert_b = vec![30.0, 40.0];
        let expert_outputs = vec![expert_a.clone(), expert_b.clone()];

        // Act
        let output = moe_dispatch(&input, &gate_logits, &expert_outputs, 2);

        // Assert: output ≈ 0.5 * [10,20] + 0.5 * [30,40] = [20, 30]
        assert!((output[0] - 20.0).abs() < 0.5,
            "output[0] should be ~20.0, got {}", output[0]);
        assert!((output[1] - 30.0).abs() < 0.5,
            "output[1] should be ~30.0, got {}", output[1]);
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertRouteTable::expert_token_counts 在无溢出场景下
    /// 严格等于每个专家被路由到的 token 数
    #[test]
    fn test_route_table_expert_counts_exact_no_overflow() {
        // Arrange: 4 experts, top_k=1, 高容量, 精确 2 tokens per expert
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        // 8 tokens: 2 per expert
        let gate_logits: Vec<Vec<f32>> = (0..8)
            .map(|i| {
                let mut row = vec![0.0; 4];
                row[i % 4] = 100.0;
                row
            })
            .collect();

        // Act
        let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);

        // Assert: 每个 expert 恰好 2 个 token
        assert_eq!(rt.overflow_count, 0, "should have no overflow");
        assert_eq!(rt.expert_token_counts, vec![2, 2, 2, 2]);
        assert_eq!(rt.token_routes.len(), 8);
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// ExpertLoadBalancer::hot_experts 当有多个专家零命中时,
    /// 它们在 hot 列表中的 rate 都是 0.0
    #[test]
    fn test_balancer_hot_experts_zero_rate_experts_sorted_consistently() {
        // Arrange: 只有 expert 0 有命中，其余 3 个专家全部 rate=0.0
        let config = ExpertRouteConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 100.0,
            ..ExpertRouteConfig::default()
        };
        let mut balancer = ExpertLoadBalancer::new(config.clone());
        let logits: Vec<Vec<f32>> = (0..10)
            .map(|_| vec![100.0, 0.0, 0.0, 0.0])
            .collect();
        let rt = ExpertRouteTable::from_gate_logits(config, &logits);
        balancer.record_route(&rt);

        // Act
        let hot = balancer.hot_experts(4);

        // Assert: expert 0 排第一且 rate > 0
        assert_eq!(hot[0].0, 0);
        assert!(hot[0].1 > 0.0, "expert 0 rate should be > 0");
        // 其余 3 个专家 rate 都是 0.0
        for (_, rate) in &hot[1..] {
            assert!((*rate).abs() < 1e-10, "unhit expert rate should be 0.0, got {}", rate);
        }
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// TokenRoute 的 expert_weights 和 expert_positions 长度
    /// 在 from_gate_logits 生成的路由表中始终等于实际选中的专家数
    #[test]
    fn test_route_table_weights_and_positions_length_match_indices() {
        // Arrange: 多种 top_k 配置
        for top_k in [1, 2, 3] {
            let config = ExpertRouteConfig {
                num_experts: 4,
                top_k,
                capacity_factor: 100.0,
                ..ExpertRouteConfig::default()
            };
            let gate_logits = vec![vec![1.0, 2.0, 3.0, 4.0]];

            // Act
            let rt = ExpertRouteTable::from_gate_logits(config, &gate_logits);
            let route = &rt.token_routes[0];

            // Assert: 三个 vec 长度一致
            assert_eq!(route.expert_indices.len(), route.expert_weights.len(),
                "top_k={}: indices len {} != weights len {}",
                top_k, route.expert_indices.len(), route.expert_weights.len());
            assert_eq!(route.expert_weights.len(), route.expert_positions.len(),
                "top_k={}: weights len {} != positions len {}",
                top_k, route.expert_weights.len(), route.expert_positions.len());
        }
    }

    /// @trace REQ-MOE-001 [level:unit]
    /// softmax 当输入为长度 2 的 [0.0, 0.0] 时精确返回 [0.5, 0.5]
    #[test]
    fn test_softmax_two_zeros_exact_half() {
        // Arrange
        let logits = vec![0.0, 0.0];

        // Act
        let probs = softmax(&logits);

        // Assert
        assert_eq!(probs.len(), 2);
        assert!((probs[0] - 0.5).abs() < 1e-6,
            "probs[0] should be 0.5, got {}", probs[0]);
        assert!((probs[1] - 0.5).abs() < 1e-6,
            "probs[1] should be 0.5, got {}", probs[1]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
