//! MoE 异构极致 — 集成示例
//!
//! 演示如何使用 ExpertRouteTable、ExpertLoadBalancer 和 ExpertThermalManager 实现 MoE 优化。

use gllm::moe::{
    ExpertLoadBalancer, ExpertRouteConfig, ExpertRouteTable,
    ExpertThermalManager, ExpertHeatLevel, DeoptRequest,
};

fn main() {
    println!("MoE 异构极致 — 核内分发 + 负载均衡 + 冷板凳封杀\n");

    // 1. 核内分发 (In-Kernel Expert Dispatch)
    println!("=== 1. 核内分发 (Top-K Routing) ===");
    let num_experts = 8;
    let top_k = 2;
    let config = ExpertRouteConfig::new(num_experts, top_k);

    // 模拟 3 个 token 的 gate logits
    let gate_logits = vec![
        vec![0.1, 0.5, 0.3, 0.2, 0.4, 0.6, 0.7, 0.8],
        vec![0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0],
        vec![0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    ];

    let route_table = ExpertRouteTable::from_gate_logits(config.clone(), &gate_logits);

    println!("Gate logits (3 tokens):");
    for (i, logits) in gate_logits.iter().enumerate() {
        println!("  Token {}: {:?}", i, logits);
    }
    println!("\nRouting results:");
    for (i, route) in route_table.token_routes.iter().enumerate() {
        println!(
            "  Token {} -> experts {:?} (weights: {:?})",
            i, route.expert_indices, route.expert_weights
        );
    }
    println!(
        "Expert token counts: {:?}",
        route_table.expert_token_counts
    );
    println!();

    // 2. 负载均衡 (Load Balancing)
    println!("=== 2. 负载均衡 ===");
    let mut balancer = ExpertLoadBalancer::new(config.clone());
    println!("Before balancing: all experts have equal load");

    // 模拟不均衡负载：只使用 expert 0 和 1
    let biased_logits = vec![1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    for _ in 0..100 {
        let biased_table = ExpertRouteTable::from_gate_logits(
            config.clone(),
            &[biased_logits.clone()],
        );
        balancer.record_route(&biased_table);
    }

    let cold = balancer.cold_experts();
    let hot = balancer.hot_experts(3);
    println!("After 100 steps with biased input:");
    println!("  Hot experts: {:?}", hot);
    println!("  Cold experts: {:?}", cold);
    println!();

    // 3. 冷板凳封杀 (Thermal Management / Deopt)
    println!("=== 3. 冷板凳封杀 (Thermal Management) ===");
    let mut thermal_mgr = ExpertThermalManager::new(num_experts)
        .with_eviction_threshold(10); // 降低阈值便于演示

    // 模拟不均衡负载: expert 0-1 热门，其余冷门
    for _ in 0..100 {
        // 只有 expert 0-1 有流量
        let route_counts: Vec<usize> = vec![100, 90, 0, 0, 0, 0, 0, 0];
        thermal_mgr.step(&route_counts);
    }

    let states = thermal_mgr.states();
    println!("Expert heat states after 100 steps:");
    for state in states {
        let level_str = match state.heat_level {
            ExpertHeatLevel::Hot => "HOT",
            ExpertHeatLevel::Warm => "WARM",
            ExpertHeatLevel::Cold => "COLD",
            ExpertHeatLevel::Evicted => "EVICTED",
        };
        println!(
            "  Expert {}: {} (hit_rate: {:.3}, streak: {})",
            state.expert_idx, level_str, state.hit_rate, state.consecutive_zero_streak
        );
    }

    // 检查需要封杀的专家
    let to_evict = thermal_mgr.experts_to_evict();
    if !to_evict.is_empty() {
        println!("Experts to evict: {:?}", to_evict);
        for &expert_id in &to_evict {
            thermal_mgr.evict_expert(expert_id);
            println!("  Expert {} evicted", expert_id);
        }
    }

    // 模拟 Uncommon Trap: 访问了已封杀的专家
    if !to_evict.is_empty() {
        let expert_id = to_evict[0];
        let request = DeoptRequest {
            request_id: 1,
            expert_idx: expert_id,
            layer_idx: 0,
            step: 100,
        };
        let result = thermal_mgr.handle_deopt_request(request);
        println!("\nDeopt triggered for expert {}: {:?}", expert_id, result);
        println!("Expert {} reactivated via OSR Bailout", expert_id);
    }

    println!("\nMoE 异构极致实现完成");
}
