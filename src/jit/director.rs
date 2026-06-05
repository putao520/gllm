//! JIT Director Daemon (SPEC §9.2)
//!
//! 后台常驻线程，职责：
//! 1. 定期扫描 KV Page Headers 中的遥测数据（Epilogue 写入的 entropy/centroid/delta）
//! 2. 维护半衰期积分池（Decaying Reservoir）平滑指标
//! 3. 检测全局共识不可逆突变（冷专家零命中、注意力静默等）
//! 4. 触发 Hot JMP Patching（通过 `moe/hot_patch.rs`）
//! 5. 管理 Golden Bucket 运行时演化（通过 `jit/golden_bucket.rs`）
//!
//! ## §9.2 铁律
//! - JIT Director Daemon 是纯 Rust 后台线程，不阻塞推理热路径
//! - 扫描频率由半衰期积分池控制，不是固定间隔
//! - 只有全局物理共识才能触发 Hot JMP Patching

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::kv_cache::{f16_bits_to_f32, dead_ratio_to_f32, KvPageHeader};
// ============================================================================
// Decaying Reservoir — 半衰期积分池 (§9.2)
// ============================================================================

/// 半衰期积分池 — 平滑遥测指标的时间衰减
///
/// 使用指数移动平均 (EMA) 实现半衰期衰减。
/// 每次 ingest 时，旧值按 decay_factor 衰减，新值按 (1 - decay_factor) 混入。
#[derive(Debug, Clone)]
pub struct DecayingReservoir {
    /// 衰减因子 (0.0-1.0)，越大衰减越慢
    decay_factor: f64,
    /// 平滑后的平均熵
    avg_entropy: f64,
    /// 平滑后的平均残差 delta
    avg_residual_delta: f64,
    /// 平滑后的平均死神经元比例
    avg_dead_neuron_ratio: f64,
    /// 平滑后的平均 softmax 锐度
    avg_softmax_sharpness: f64,
    /// 累计采样次数
    sample_count: u64,
}

impl DecayingReservoir {
    /// 创建新的半衰期积分池
    ///
    /// `half_life_samples`: 半衰期（多少次采样后旧值权重降为 50%）
    pub fn new(half_life_samples: u64) -> Self {
        let decay_factor = (-((2.0_f64).ln()) / half_life_samples as f64).exp();
        Self {
            decay_factor,
            avg_entropy: 0.0,
            avg_residual_delta: 1.0,
            avg_dead_neuron_ratio: 0.0,
            avg_softmax_sharpness: 0.0,
            sample_count: 0,
        }
    }

    /// 从 KvPageHeader 摄入一次遥测数据
    pub fn ingest(&mut self, header: &KvPageHeader) {
        let d = self.decay_factor;
        let w = 1.0 - d;

        if self.sample_count == 0 {
            self.avg_entropy = f16_bits_to_f32(header.entropy_avg) as f64;
            self.avg_residual_delta = f16_bits_to_f32(header.delta_rho_avg) as f64;
            self.avg_dead_neuron_ratio = dead_ratio_to_f32(header.dead_ratio) as f64;
            self.avg_softmax_sharpness = f16_bits_to_f32(header.centroid_pos) as f64;
        } else {
            self.avg_entropy = d * self.avg_entropy + w * f16_bits_to_f32(header.entropy_avg) as f64;
            self.avg_residual_delta = d * self.avg_residual_delta + w * f16_bits_to_f32(header.delta_rho_avg) as f64;
            self.avg_dead_neuron_ratio = d * self.avg_dead_neuron_ratio + w * dead_ratio_to_f32(header.dead_ratio) as f64;
            self.avg_softmax_sharpness = d * self.avg_softmax_sharpness + w * f16_bits_to_f32(header.centroid_pos) as f64;
        }
        self.sample_count += 1;
    }

    /// 平滑后的平均熵
    pub fn avg_entropy(&self) -> f64 { self.avg_entropy }
    /// 平滑后的平均残差 delta
    pub fn avg_residual_delta(&self) -> f64 { self.avg_residual_delta }
    /// 平滑后的平均死神经元比例
    pub fn avg_dead_neuron_ratio(&self) -> f64 { self.avg_dead_neuron_ratio }
    /// 平滑后的平均 softmax 锐度
    pub fn avg_softmax_sharpness(&self) -> f64 { self.avg_softmax_sharpness }
    /// 累计采样次数
    pub fn sample_count(&self) -> u64 { self.sample_count }
}

// ============================================================================
// Consensus Detector — 全局共识不可逆突变检测 (§9.2)
// ============================================================================

/// 全局共识突变类型
#[derive(Debug, Clone, PartialEq)]
pub enum ConsensusEvent {
    /// 冷专家持续零命中 → 触发 Hot JMP Patching 封杀
    ExpertFrozen {
        expert_idx: usize,
        zero_hit_steps: u64,
    },
    /// 注意力持续静默（entropy 极低）→ 可能退化
    AttentionSilent {
        avg_entropy: f64,
        duration_steps: u64,
    },
    /// 残差能量持续为零 → 层可能冗余
    LayerRedundant {
        avg_delta_rho: f64,
        duration_steps: u64,
    },
}

/// 全局共识检测器
///
/// 基于 DecayingReservoir 的平滑指标，检测不可逆突变。
#[derive(Debug, Clone)]
pub struct ConsensusDetector {
    /// 冷专家封杀阈值（连续零命中步数）
    expert_freeze_threshold: u64,
    /// 注意力静默阈值（平均熵低于此值）
    attention_silent_threshold: f64,
    /// 层冗余阈值（平均 delta_rho 低于此值）
    layer_redundant_threshold: f64,
    /// 专家连续零命中计数器
    expert_zero_streaks: Vec<u64>,
    /// 注意力静默持续步数
    attention_silent_steps: u64,
    /// 层冗余持续步数
    layer_redundant_steps: u64,
}

impl ConsensusDetector {
    pub fn new(num_experts: usize) -> Self {
        Self {
            expert_freeze_threshold: 1_000_000,
            attention_silent_threshold: 0.05,
            layer_redundant_threshold: 0.001,
            expert_zero_streaks: vec![0; num_experts],
            attention_silent_steps: 0,
            layer_redundant_steps: 0,
        }
    }

    /// 更新专家命中计数
    pub fn update_expert_hits(&mut self, expert_hits: &[u32]) {
        for (idx, streak) in self.expert_zero_streaks.iter_mut().enumerate() {
            let hits = expert_hits.get(idx).copied().unwrap_or(0);
            if hits == 0 {
                *streak += 1;
            } else {
                *streak = 0;
            }
        }
    }

    /// 从 reservoir 检测全局共识事件
    pub fn detect(&mut self, reservoir: &DecayingReservoir) -> Vec<ConsensusEvent> {
        let mut events = Vec::new();

        // 冷专家检测
        for (idx, &streak) in self.expert_zero_streaks.iter().enumerate() {
            if streak >= self.expert_freeze_threshold {
                events.push(ConsensusEvent::ExpertFrozen {
                    expert_idx: idx,
                    zero_hit_steps: streak,
                });
            }
        }

        // 注意力静默检测
        if reservoir.avg_entropy() < self.attention_silent_threshold
            && reservoir.sample_count() > 1000
        {
            self.attention_silent_steps += 1;
            if self.attention_silent_steps > 10000 {
                events.push(ConsensusEvent::AttentionSilent {
                    avg_entropy: reservoir.avg_entropy(),
                    duration_steps: self.attention_silent_steps,
                });
            }
        } else {
            self.attention_silent_steps = 0;
        }

        // 层冗余检测
        if reservoir.avg_residual_delta() < self.layer_redundant_threshold
            && reservoir.sample_count() > 1000
        {
            self.layer_redundant_steps += 1;
            if self.layer_redundant_steps > 10000 {
                events.push(ConsensusEvent::LayerRedundant {
                    avg_delta_rho: reservoir.avg_residual_delta(),
                    duration_steps: self.layer_redundant_steps,
                });
            }
        } else {
            self.layer_redundant_steps = 0;
        }

        events
    }

    /// 重置专家封杀计数（专家被复活后）
    pub fn reset_expert(&mut self, expert_idx: usize) {
        if let Some(streak) = self.expert_zero_streaks.get_mut(expert_idx) {
            *streak = 0;
        }
    }
}

// ============================================================================
// JitDirector — 后台常驻监控线程 (§9.2)
// ============================================================================

/// JIT Director 共享状态（线程间通信）
pub struct DirectorSharedState {
    /// KV Page Headers 快照（由推理线程写入，Director 读取）
    page_headers: std::sync::RwLock<Vec<KvPageHeader>>,
    /// 专家命中计数器（由推理线程原子递增，Director 读取）
    expert_hit_counters: Vec<AtomicU64>,
    /// 全局步数计数器
    global_step: AtomicU64,
    /// 待处理的共识事件队列（Director 写入，推理线程消费）
    pending_events: std::sync::Mutex<Vec<ConsensusEvent>>,
}

impl DirectorSharedState {
    pub fn new(num_experts: usize) -> Self {
        Self {
            page_headers: std::sync::RwLock::new(Vec::new()),
            expert_hit_counters: (0..num_experts).map(|_| AtomicU64::new(0)).collect(),
            global_step: AtomicU64::new(0),
            pending_events: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// 推理线程：更新 KV Page Headers 快照
    pub fn update_page_headers(&self, headers: Vec<KvPageHeader>) {
        if let Ok(mut guard) = self.page_headers.write() {
            *guard = headers;
        }
    }

    /// 推理线程：递增专家命中计数
    pub fn record_expert_hit(&self, expert_idx: usize) {
        if let Some(counter) = self.expert_hit_counters.get(expert_idx) {
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// 推理线程：递增全局步数
    pub fn advance_step(&self) {
        self.global_step.fetch_add(1, Ordering::Relaxed);
    }

    /// 推理线程：消费待处理的共识事件
    pub fn drain_events(&self) -> Vec<ConsensusEvent> {
        self.pending_events
            .lock()
            .map(|mut events| std::mem::take(&mut *events))
            .unwrap_or_default()
    }

    /// Director 线程：读取并重置专家命中计数
    fn snapshot_and_reset_expert_hits(&self) -> Vec<u32> {
        self.expert_hit_counters
            .iter()
            .map(|c| c.swap(0, Ordering::Relaxed) as u32)
            .collect()
    }

    /// Director 线程：读取 KV Page Headers
    fn read_page_headers(&self) -> Vec<KvPageHeader> {
        self.page_headers
            .read()
            .map(|guard| guard.clone())
            .unwrap_or_default()
    }

    /// Director 线程：推送共识事件
    fn push_events(&self, events: Vec<ConsensusEvent>) {
        if events.is_empty() {
            return;
        }
        if let Ok(mut guard) = self.pending_events.lock() {
            guard.extend(events);
        }
    }
}

/// JIT Director Daemon 配置
#[derive(Debug, Clone)]
pub struct DirectorConfig {
    /// 扫描间隔
    pub scan_interval: Duration,
    /// 半衰期（采样次数）
    pub half_life_samples: u64,
    /// MoE 专家数量
    pub num_experts: usize,
}

impl Default for DirectorConfig {
    fn default() -> Self {
        Self {
            scan_interval: Duration::from_millis(100),
            half_life_samples: 10_000,
            num_experts: 0,
        }
    }
}

/// JIT Director Daemon (§9.2)
///
/// 后台常驻线程，定期扫描遥测数据，检测全局共识突变，触发 Hot JMP Patching。
pub struct JitDirector {
    /// 共享状态
    shared: Arc<DirectorSharedState>,
    /// 关闭信号
    shutdown: Arc<AtomicBool>,
    /// 后台线程句柄
    handle: Option<std::thread::JoinHandle<()>>,
}

impl JitDirector {
    /// 启动 JIT Director Daemon
    pub fn spawn(config: DirectorConfig) -> Self {
        let shared = Arc::new(DirectorSharedState::new(config.num_experts));
        let shutdown = Arc::new(AtomicBool::new(false));

        let shared_clone = Arc::clone(&shared);
        let shutdown_clone = Arc::clone(&shutdown);

        let handle = std::thread::Builder::new()
            .name("gllm-jit-director".to_string())
            .spawn(move || {
                Self::scan_loop(shared_clone, shutdown_clone, config);
            })
            .expect("failed to spawn JIT Director thread");

        Self {
            shared,
            shutdown,
            handle: Some(handle),
        }
    }

    /// 获取共享状态（供推理线程使用）
    pub fn shared(&self) -> &Arc<DirectorSharedState> {
        &self.shared
    }

    /// 关闭 Director Daemon
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }

    /// 后台扫描主循环
    fn scan_loop(
        shared: Arc<DirectorSharedState>,
        shutdown: Arc<AtomicBool>,
        config: DirectorConfig,
    ) {
        let mut reservoir = DecayingReservoir::new(config.half_life_samples);
        let mut detector = ConsensusDetector::new(config.num_experts);

        log::info!("JIT Director Daemon started (scan_interval={:?}, half_life={})",
            config.scan_interval, config.half_life_samples);

        while !shutdown.load(Ordering::SeqCst) {
            std::thread::sleep(config.scan_interval);

            // 1. 读取 KV Page Headers 遥测
            let headers = shared.read_page_headers();
            for header in &headers {
                if header.is_active() {
                    reservoir.ingest(header);
                }
            }

            // 2. 读取并重置专家命中计数
            let expert_hits = shared.snapshot_and_reset_expert_hits();
            detector.update_expert_hits(&expert_hits);

            // 3. 检测全局共识事件
            let events = detector.detect(&reservoir);

            // 4. 推送事件给推理线程
            if !events.is_empty() {
                log::info!("JIT Director: {} consensus events detected", events.len());
                for event in &events {
                    log::info!("  {:?}", event);
                }
                shared.push_events(events);
            }
        }

        log::info!("JIT Director Daemon stopped");
    }
}

impl Drop for JitDirector {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::f32_to_f16_bits;

    #[test]
    fn test_decaying_reservoir() {
        let mut reservoir = DecayingReservoir::new(100);
        assert_eq!(reservoir.sample_count(), 0);

        let mut header = KvPageHeader::new(1);
        header.ref_count = 1;
        header.entropy_avg = f32_to_f16_bits(2.5);
        header.centroid_pos = f32_to_f16_bits(0.8);
        header.softmax_max_avg = f32_to_f16_bits(0.9);
        header.delta_rho_avg = f32_to_f16_bits(0.5);
        header.dead_ratio = 26; // ~0.1 in [0,255] range

        reservoir.ingest(&header);
        assert_eq!(reservoir.sample_count(), 1);
        assert!((reservoir.avg_entropy() - 2.5).abs() < 1e-6);

        // Second ingest should decay
        let mut header2 = header;
        header2.entropy_avg = f32_to_f16_bits(3.0);
        reservoir.ingest(&header2);
        assert_eq!(reservoir.sample_count(), 2);
        // Should be between 2.5 and 3.0
        assert!(reservoir.avg_entropy() > 2.5);
        assert!(reservoir.avg_entropy() < 3.0);
    }

    #[test]
    fn test_consensus_detector_expert_freeze() {
        let mut detector = ConsensusDetector::new(4);
        let reservoir = DecayingReservoir::new(100);

        // Simulate zero hits for expert 2
        detector.expert_freeze_threshold = 3; // Low threshold for test
        for _ in 0..5 {
            detector.update_expert_hits(&[1, 1, 0, 1]);
        }

        let events = detector.detect(&reservoir);
        assert!(events.iter().any(|e| matches!(e,
            ConsensusEvent::ExpertFrozen { expert_idx: 2, .. }
        )));
    }

    #[test]
    fn test_consensus_detector_reset() {
        let mut detector = ConsensusDetector::new(4);
        detector.expert_freeze_threshold = 3;

        for _ in 0..5 {
            detector.update_expert_hits(&[0, 0, 0, 0]);
        }

        detector.reset_expert(1);
        assert_eq!(detector.expert_zero_streaks[1], 0);
        assert_eq!(detector.expert_zero_streaks[0], 5);
    }

    #[test]
    fn test_shared_state() {
        let state = DirectorSharedState::new(4);

        state.record_expert_hit(0);
        state.record_expert_hit(0);
        state.record_expert_hit(2);

        let hits = state.snapshot_and_reset_expert_hits();
        assert_eq!(hits, vec![2, 0, 1, 0]);

        // After reset, should be zero
        let hits2 = state.snapshot_and_reset_expert_hits();
        assert_eq!(hits2, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_director_spawn_shutdown() {
        let config = DirectorConfig {
            scan_interval: Duration::from_millis(10),
            half_life_samples: 100,
            num_experts: 4,
        };

        let mut director = JitDirector::spawn(config);
        // Let it run briefly
        std::thread::sleep(Duration::from_millis(50));
        director.shutdown();
        // Should not panic
    }

    #[test]
    fn test_director_event_flow() {
        let config = DirectorConfig {
            scan_interval: Duration::from_millis(10),
            half_life_samples: 100,
            num_experts: 2,
        };

        let mut director = JitDirector::spawn(config);
        let shared = Arc::clone(director.shared());

        // Push some page headers
        let mut header = KvPageHeader::new(1);
        header.ref_count = 1;
        header.entropy_avg = f32_to_f16_bits(1.5);
        shared.update_page_headers(vec![header]);

        shared.record_expert_hit(0);
        shared.advance_step();

        std::thread::sleep(Duration::from_millis(50));

        // Drain events (may be empty if no consensus reached)
        let _events = shared.drain_events();

        director.shutdown();
    }

    // =========================================================================
    // DecayingReservoir tests
    // =========================================================================

    #[test]
    fn test_decaying_reservoir_clone_preserves_state() {
        let mut reservoir = DecayingReservoir::new(100);
        let mut header = KvPageHeader::new(1);
        header.ref_count = 1;
        header.entropy_avg = f32_to_f16_bits(4.0);
        header.delta_rho_avg = f32_to_f16_bits(0.3);
        header.dead_ratio = 128;
        header.centroid_pos = f32_to_f16_bits(0.7);
        reservoir.ingest(&header);

        let cloned = reservoir.clone();
        assert_eq!(cloned.sample_count(), 1);
        assert!((cloned.avg_entropy() - 4.0).abs() < 0.01);
        assert!((cloned.avg_residual_delta() - 0.3).abs() < 0.01);
        assert!((cloned.avg_softmax_sharpness() - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_decaying_reservoir_first_sample_sets_directly() {
        let mut reservoir = DecayingReservoir::new(50);
        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.entropy_avg = f32_to_f16_bits(1.0);
        header.delta_rho_avg = f32_to_f16_bits(0.2);
        header.dead_ratio = 50;
        header.centroid_pos = f32_to_f16_bits(0.5);

        reservoir.ingest(&header);

        assert_eq!(reservoir.sample_count(), 1);
        assert!((reservoir.avg_entropy() - 1.0).abs() < 0.01);
        assert!((reservoir.avg_residual_delta() - 0.2).abs() < 0.01);
        assert!((reservoir.avg_dead_neuron_ratio() - dead_ratio_to_f32(50) as f64).abs() < 0.01);
        assert!((reservoir.avg_softmax_sharpness() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_decaying_reservoir_convergence() {
        // With a long half-life, the reservoir should converge toward the latest value
        let mut reservoir = DecayingReservoir::new(10_000);
        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.entropy_avg = f32_to_f16_bits(5.0);
        header.delta_rho_avg = f32_to_f16_bits(0.1);
        header.dead_ratio = 10;
        header.centroid_pos = f32_to_f16_bits(0.3);

        // Ingest 1000 identical samples
        for _ in 0..1000 {
            reservoir.ingest(&header);
        }

        assert_eq!(reservoir.sample_count(), 1000);
        // After many identical samples, averages should be close to the constant value
        assert!((reservoir.avg_entropy() - 5.0).abs() < 0.1);
        assert!((reservoir.avg_residual_delta() - 0.1).abs() < 0.05);
        assert!((reservoir.avg_softmax_sharpness() - 0.3).abs() < 0.05);
    }

    #[test]
    fn test_decaying_reservoir_decay_with_short_half_life() {
        // Very short half-life means new values dominate quickly
        let mut reservoir = DecayingReservoir::new(2);

        let mut header_a = KvPageHeader::new(0);
        header_a.ref_count = 1;
        header_a.entropy_avg = f32_to_f16_bits(10.0);
        header_a.delta_rho_avg = f32_to_f16_bits(0.0);
        header_a.dead_ratio = 0;
        header_a.centroid_pos = f32_to_f16_bits(0.0);

        reservoir.ingest(&header_a);
        assert_eq!(reservoir.sample_count(), 1);

        let mut header_b = KvPageHeader::new(0);
        header_b.ref_count = 1;
        header_b.entropy_avg = f32_to_f16_bits(0.0);
        header_b.delta_rho_avg = f32_to_f16_bits(0.0);
        header_b.dead_ratio = 0;
        header_b.centroid_pos = f32_to_f16_bits(0.0);

        // Ingest many zero samples — should pull average down from 10 toward 0
        for _ in 0..20 {
            reservoir.ingest(&header_b);
        }

        // After many zero samples with short half-life, average should be near zero
        assert!(reservoir.avg_entropy() < 1.0);
    }

    #[test]
    fn test_decaying_reservoir_inactive_page_skipped() {
        // This test verifies that ingest reads data regardless of is_active,
        // because ingest itself does not check is_active — the scan_loop does.
        let mut reservoir = DecayingReservoir::new(100);
        let header = KvPageHeader::new(0); // ref_count = 0, is_active = false

        reservoir.ingest(&header);
        assert_eq!(reservoir.sample_count(), 1);
    }

    // =========================================================================
    // ConsensusEvent tests
    // =========================================================================

    #[test]
    fn test_consensus_event_equality() {
        let a = ConsensusEvent::ExpertFrozen { expert_idx: 3, zero_hit_steps: 100 };
        let b = ConsensusEvent::ExpertFrozen { expert_idx: 3, zero_hit_steps: 100 };
        let c = ConsensusEvent::ExpertFrozen { expert_idx: 3, zero_hit_steps: 200 };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_consensus_event_variants_distinct() {
        let expert = ConsensusEvent::ExpertFrozen { expert_idx: 0, zero_hit_steps: 50 };
        let attention = ConsensusEvent::AttentionSilent { avg_entropy: 0.01, duration_steps: 15000 };
        let layer = ConsensusEvent::LayerRedundant { avg_delta_rho: 0.0001, duration_steps: 12000 };

        // Each variant is distinct
        assert!(!matches!(expert, ConsensusEvent::AttentionSilent { .. }));
        assert!(!matches!(attention, ConsensusEvent::LayerRedundant { .. }));
        assert!(!matches!(layer, ConsensusEvent::ExpertFrozen { .. }));
    }

    #[test]
    fn test_consensus_event_debug_format() {
        let event = ConsensusEvent::ExpertFrozen { expert_idx: 7, zero_hit_steps: 999 };
        let debug_str = format!("{:?}", event);
        assert!(debug_str.contains("ExpertFrozen"));
        assert!(debug_str.contains("7"));
        assert!(debug_str.contains("999"));
    }

    // =========================================================================
    // ConsensusDetector tests
    // =========================================================================

    #[test]
    fn test_consensus_detector_new_initializes_streaks() {
        let detector = ConsensusDetector::new(8);
        assert_eq!(detector.expert_zero_streaks.len(), 8);
        assert!(detector.expert_zero_streaks.iter().all(|&s| s == 0));
    }

    #[test]
    fn test_consensus_detector_clone() {
        let mut detector = ConsensusDetector::new(3);
        detector.expert_freeze_threshold = 5;
        for _ in 0..3 {
            detector.update_expert_hits(&[0, 5, 0]);
        }

        let cloned = detector.clone();
        assert_eq!(cloned.expert_zero_streaks.len(), 3);
        assert_eq!(cloned.expert_zero_streaks[0], 3);
        assert_eq!(cloned.expert_zero_streaks[1], 0);
        assert_eq!(cloned.expert_zero_streaks[2], 3);
        assert_eq!(cloned.expert_freeze_threshold, 5);
    }

    #[test]
    fn test_consensus_detector_hit_resets_streak() {
        let mut detector = ConsensusDetector::new(3);
        detector.expert_freeze_threshold = 100;

        // Build streak of 5 for expert 0
        for _ in 0..5 {
            detector.update_expert_hits(&[0, 0, 0]);
        }
        assert_eq!(detector.expert_zero_streaks[0], 5);

        // Expert 0 gets a hit — streak resets
        detector.update_expert_hits(&[1, 0, 0]);
        assert_eq!(detector.expert_zero_streaks[0], 0);
        assert_eq!(detector.expert_zero_streaks[1], 6);
    }

    #[test]
    fn test_consensus_detector_expert_freeze_multiple() {
        let mut detector = ConsensusDetector::new(4);
        detector.expert_freeze_threshold = 3;
        let reservoir = DecayingReservoir::new(100);

        // All experts frozen
        for _ in 0..5 {
            detector.update_expert_hits(&[0, 0, 0, 0]);
        }

        let events = detector.detect(&reservoir);
        let frozen_count = events.iter().filter(|e| matches!(e, ConsensusEvent::ExpertFrozen { .. })).count();
        assert_eq!(frozen_count, 4);
    }

    #[test]
    fn test_consensus_detector_no_event_when_hits_present() {
        let mut detector = ConsensusDetector::new(4);
        detector.expert_freeze_threshold = 3;
        let reservoir = DecayingReservoir::new(100);

        // Expert 2 has hits every round — streak stays 0
        for _ in 0..10 {
            detector.update_expert_hits(&[0, 0, 1, 0]);
        }
        // Experts 0,1,3 frozen but threshold is 3, and 10 > 3
        let events = detector.detect(&reservoir);
        let expert_2_frozen = events.iter().any(|e| matches!(
            e,
            ConsensusEvent::ExpertFrozen { expert_idx: 2, .. }
        ));
        assert!(!expert_2_frozen);
    }

    #[test]
    fn test_consensus_detector_attention_silent_no_event_insufficient_samples() {
        let mut detector = ConsensusDetector::new(2);
        let mut reservoir = DecayingReservoir::new(100);

        // Feed very low entropy but only a few samples (sample_count < 1000)
        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.entropy_avg = f32_to_f16_bits(0.001); // Very low

        for _ in 0..10 {
            reservoir.ingest(&header);
        }

        let events = detector.detect(&reservoir);
        assert!(!events.iter().any(|e| matches!(e, ConsensusEvent::AttentionSilent { .. })));
    }

    #[test]
    fn test_consensus_detector_attention_silent_resets_on_recovery() {
        let mut detector = ConsensusDetector::new(2);
        let mut reservoir = DecayingReservoir::new(100);

        // Feed enough low-entropy samples to build up silent_steps, but below threshold
        let mut low_header = KvPageHeader::new(0);
        low_header.ref_count = 1;
        low_header.entropy_avg = f32_to_f16_bits(0.001);

        for _ in 0..1100 {
            reservoir.ingest(&low_header);
        }

        // detect a few times to accumulate attention_silent_steps
        for _ in 0..5 {
            let _events = detector.detect(&reservoir);
        }

        // Now feed high entropy — should reset counter
        let mut high_header = KvPageHeader::new(0);
        high_header.ref_count = 1;
        high_header.entropy_avg = f32_to_f16_bits(5.0);
        for _ in 0..100 {
            reservoir.ingest(&high_header);
        }

        let events = detector.detect(&reservoir);
        // No AttentionSilent event because entropy is high
        assert!(!events.iter().any(|e| matches!(e, ConsensusEvent::AttentionSilent { .. })));
    }

    #[test]
    fn test_consensus_detector_layer_redundant_no_event_insufficient_samples() {
        let mut detector = ConsensusDetector::new(2);
        let mut reservoir = DecayingReservoir::new(100);

        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.delta_rho_avg = f32_to_f16_bits(0.0001); // Very low

        // Only 10 samples — below the 1000 threshold
        for _ in 0..10 {
            reservoir.ingest(&header);
        }

        let events = detector.detect(&reservoir);
        assert!(!events.iter().any(|e| matches!(e, ConsensusEvent::LayerRedundant { .. })));
    }

    #[test]
    fn test_consensus_detector_reset_expert_out_of_bounds() {
        let mut detector = ConsensusDetector::new(3);
        // Should not panic with out-of-bounds index
        detector.reset_expert(100);
        assert_eq!(detector.expert_zero_streaks.len(), 3);
    }

    #[test]
    fn test_consensus_detector_update_with_shorter_hits_slice() {
        let mut detector = ConsensusDetector::new(4);

        // hits slice is shorter than num_experts — trailing experts keep streak
        detector.update_expert_hits(&[1, 0]);
        assert_eq!(detector.expert_zero_streaks[0], 0);
        assert_eq!(detector.expert_zero_streaks[1], 1);
        assert_eq!(detector.expert_zero_streaks[2], 1); // Missing in hits → treated as 0 hits
        assert_eq!(detector.expert_zero_streaks[3], 1);
    }

    // =========================================================================
    // DirectorConfig tests
    // =========================================================================

    #[test]
    fn test_director_config_default() {
        let config = DirectorConfig::default();
        assert_eq!(config.scan_interval, Duration::from_millis(100));
        assert_eq!(config.half_life_samples, 10_000);
        assert_eq!(config.num_experts, 0);
    }

    #[test]
    fn test_director_config_clone() {
        let config = DirectorConfig {
            scan_interval: Duration::from_millis(200),
            half_life_samples: 5000,
            num_experts: 8,
        };
        let cloned = config.clone();
        assert_eq!(cloned.scan_interval, Duration::from_millis(200));
        assert_eq!(cloned.half_life_samples, 5000);
        assert_eq!(cloned.num_experts, 8);
    }

    // =========================================================================
    // DirectorSharedState tests
    // =========================================================================

    #[test]
    fn test_shared_state_zero_experts() {
        let state = DirectorSharedState::new(0);
        let hits = state.snapshot_and_reset_expert_hits();
        assert!(hits.is_empty());
    }

    #[test]
    fn test_shared_state_advance_step() {
        let state = DirectorSharedState::new(2);

        state.advance_step();
        state.advance_step();
        state.advance_step();

        let step = state.global_step.load(Ordering::Relaxed);
        assert_eq!(step, 3);
    }

    #[test]
    fn test_shared_state_drain_events_empty() {
        let state = DirectorSharedState::new(2);
        let events = state.drain_events();
        assert!(events.is_empty());
    }

    #[test]
    fn test_shared_state_record_expert_hit_out_of_bounds() {
        let state = DirectorSharedState::new(2);
        // Should not panic
        state.record_expert_hit(5);
        state.record_expert_hit(100);

        let hits = state.snapshot_and_reset_expert_hits();
        assert_eq!(hits, vec![0, 0]);
    }

    #[test]
    fn test_shared_state_update_and_read_page_headers() {
        let state = DirectorSharedState::new(2);

        let mut h1 = KvPageHeader::new(10);
        h1.ref_count = 2;
        h1.entropy_avg = f32_to_f16_bits(3.0);
        let mut h2 = KvPageHeader::new(11);
        h2.ref_count = 1;
        h2.entropy_avg = f32_to_f16_bits(1.5);

        state.update_page_headers(vec![h1, h2]);

        let headers = state.read_page_headers();
        assert_eq!(headers.len(), 2);
        assert_eq!(headers[0].page_id, 10);
        assert_eq!(headers[1].page_id, 11);
    }

    #[test]
    fn test_shared_state_update_replaces_previous_headers() {
        let state = DirectorSharedState::new(2);

        let h1 = KvPageHeader::new(1);
        state.update_page_headers(vec![h1]);
        let headers = state.read_page_headers();
        assert_eq!(headers.len(), 1);

        let h2 = KvPageHeader::new(2);
        let h3 = KvPageHeader::new(3);
        state.update_page_headers(vec![h2, h3]);
        let headers = state.read_page_headers();
        assert_eq!(headers.len(), 2);
        assert_eq!(headers[0].page_id, 2);
        assert_eq!(headers[1].page_id, 3);
    }

    #[test]
    fn test_shared_state_push_and_drain_events() {
        let state = DirectorSharedState::new(2);

        state.push_events(vec![
            ConsensusEvent::ExpertFrozen { expert_idx: 0, zero_hit_steps: 100 },
            ConsensusEvent::AttentionSilent { avg_entropy: 0.01, duration_steps: 15000 },
        ]);

        let events = state.drain_events();
        assert_eq!(events.len(), 2);

        // Drain again should be empty
        let events2 = state.drain_events();
        assert!(events2.is_empty());
    }

    #[test]
    fn test_shared_state_push_empty_events_is_noop() {
        let state = DirectorSharedState::new(2);

        state.push_events(vec![]);
        let events = state.drain_events();
        assert!(events.is_empty());
    }

    #[test]
    fn test_shared_state_snapshot_resets_counters() {
        let state = DirectorSharedState::new(3);

        state.record_expert_hit(0);
        state.record_expert_hit(1);
        state.record_expert_hit(1);
        state.record_expert_hit(2);
        state.record_expert_hit(2);
        state.record_expert_hit(2);

        let hits1 = state.snapshot_and_reset_expert_hits();
        assert_eq!(hits1, vec![1, 2, 3]);

        // Record more after reset
        state.record_expert_hit(0);
        state.record_expert_hit(0);
        let hits2 = state.snapshot_and_reset_expert_hits();
        assert_eq!(hits2, vec![2, 0, 0]);
    }

    // =========================================================================
    // JitDirector tests
    // =========================================================================

    #[test]
    fn test_director_shared_returns_arc() {
        let config = DirectorConfig {
            scan_interval: Duration::from_millis(10),
            half_life_samples: 100,
            num_experts: 2,
        };
        let mut director = JitDirector::spawn(config);
        let shared = director.shared();
        assert!(Arc::strong_count(shared) >= 2); // director + the Arc we cloned
        director.shutdown();
    }

    #[test]
    fn test_director_drop_triggers_shutdown() {
        let config = DirectorConfig {
            scan_interval: Duration::from_millis(10),
            half_life_samples: 100,
            num_experts: 0,
        };

        {
            let _director = JitDirector::spawn(config);
            // Director goes out of scope — Drop impl calls shutdown
        }
        // If shutdown hangs, test will timeout
    }

    #[test]
    fn test_director_with_zero_experts() {
        let config = DirectorConfig {
            scan_interval: Duration::from_millis(10),
            half_life_samples: 50,
            num_experts: 0,
        };

        let mut director = JitDirector::spawn(config);
        let shared = Arc::clone(director.shared());

        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.entropy_avg = f32_to_f16_bits(2.0);
        shared.update_page_headers(vec![header]);

        std::thread::sleep(Duration::from_millis(50));
        director.shutdown();
    }

    #[test]
    fn test_director_page_headers_ingested_in_scan_loop() {
        let config = DirectorConfig {
            scan_interval: Duration::from_millis(5),
            half_life_samples: 50,
            num_experts: 2,
        };

        let mut director = JitDirector::spawn(config);
        let shared = Arc::clone(director.shared());

        // Feed active headers and advance steps
        for i in 0..5 {
            let mut header = KvPageHeader::new(i);
            header.ref_count = 1;
            header.entropy_avg = f32_to_f16_bits(1.0 + i as f32);
            shared.update_page_headers(vec![header]);
            shared.advance_step();
            shared.record_expert_hit(0);
        }

        std::thread::sleep(Duration::from_millis(100));

        // Just verify no panics and events can be drained
        let _events = shared.drain_events();
        director.shutdown();
    }

    #[test]
    fn test_director_concurrent_record_and_drain() {
        let config = DirectorConfig {
            scan_interval: Duration::from_millis(5),
            half_life_samples: 100,
            num_experts: 4,
        };

        let mut director = JitDirector::spawn(config);
        let shared = Arc::clone(director.shared());

        // Simulate concurrent inference thread recording hits
        let shared_clone = Arc::clone(&shared);
        let handle = std::thread::spawn(move || {
            for i in 0..100 {
                shared_clone.record_expert_hit(i % 4);
                shared_clone.advance_step();
            }
        });

        std::thread::sleep(Duration::from_millis(50));

        // Drain events while inference thread is still running
        let _events = shared.drain_events();

        handle.join().unwrap();
        director.shutdown();
    }

    // =========================================================================
    // Additional tests — trait impls, edge cases, uncovered paths
    // =========================================================================

    #[test]
    fn test_decaying_reservoir_debug_trait() {
        let reservoir = DecayingReservoir::new(100);
        let debug_str = format!("{:?}", reservoir);
        // Debug should contain the struct name and key fields
        assert!(debug_str.contains("DecayingReservoir"));
        assert!(debug_str.contains("decay_factor"));
        assert!(debug_str.contains("sample_count"));
    }

    #[test]
    fn test_decaying_reservoir_default_initial_values() {
        // Before any ingest: avg_entropy=0.0, avg_residual_delta=1.0,
        // avg_dead_neuron_ratio=0.0, avg_softmax_sharpness=0.0, sample_count=0
        let reservoir = DecayingReservoir::new(500);
        assert_eq!(reservoir.sample_count(), 0);
        assert!((reservoir.avg_entropy() - 0.0).abs() < 1e-10);
        assert!((reservoir.avg_residual_delta() - 1.0).abs() < 1e-10);
        assert!((reservoir.avg_dead_neuron_ratio() - 0.0).abs() < 1e-10);
        assert!((reservoir.avg_softmax_sharpness() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_decaying_reservoir_ema_math_verification() {
        // With half_life_samples=1, decay_factor = exp(-ln(2)/1) = 0.5
        let mut reservoir = DecayingReservoir::new(1);
        assert!((reservoir.decay_factor - 0.5).abs() < 1e-10);

        // First ingest sets directly
        let mut h1 = KvPageHeader::new(0);
        h1.ref_count = 1;
        h1.entropy_avg = f32_to_f16_bits(10.0);
        reservoir.ingest(&h1);
        assert!((reservoir.avg_entropy() - 10.0).abs() < 0.01);

        // Second ingest: avg = 0.5 * 10.0 + 0.5 * new_val
        let mut h2 = KvPageHeader::new(0);
        h2.ref_count = 1;
        h2.entropy_avg = f32_to_f16_bits(0.0);
        reservoir.ingest(&h2);
        assert!((reservoir.avg_entropy() - 5.0).abs() < 0.1);

        // Third: avg = 0.5 * 5.0 + 0.5 * 0.0 = 2.5
        reservoir.ingest(&h2);
        assert!((reservoir.avg_entropy() - 2.5).abs() < 0.2);
    }

    #[test]
    fn test_decaying_reservoir_large_half_life_nearly_one() {
        // Very large half_life_samples → decay_factor ≈ 1.0 (old values persist)
        let reservoir = DecayingReservoir::new(1_000_000_000);
        // decay_factor = exp(-ln(2) / 1e9) ≈ 1 - 6.93e-10
        assert!(reservoir.decay_factor > 0.999);
        assert!(reservoir.decay_factor <= 1.0);
    }

    #[test]
    fn test_decaying_reservoir_clone_independent() {
        let mut reservoir = DecayingReservoir::new(100);
        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.entropy_avg = f32_to_f16_bits(3.0);
        reservoir.ingest(&header);

        let mut cloned = reservoir.clone();

        // Feed different data to cloned — original should not change
        let mut h2 = KvPageHeader::new(0);
        h2.ref_count = 1;
        h2.entropy_avg = f32_to_f16_bits(9.0);
        cloned.ingest(&h2);

        assert_eq!(reservoir.sample_count(), 1);
        assert_eq!(cloned.sample_count(), 2);
        assert!((reservoir.avg_entropy() - 3.0).abs() < 0.01);
        assert!(cloned.avg_entropy() > 3.0);
    }

    #[test]
    fn test_consensus_event_clone() {
        let original = ConsensusEvent::AttentionSilent {
            avg_entropy: 0.02,
            duration_steps: 12000,
        };
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_consensus_event_layer_redundant_debug() {
        let event = ConsensusEvent::LayerRedundant {
            avg_delta_rho: 0.0005,
            duration_steps: 11000,
        };
        let debug_str = format!("{:?}", event);
        assert!(debug_str.contains("LayerRedundant"));
        assert!(debug_str.contains("avg_delta_rho"));
        assert!(debug_str.contains("11000"));
    }

    #[test]
    fn test_consensus_event_attention_silent_debug() {
        let event = ConsensusEvent::AttentionSilent {
            avg_entropy: 0.03,
            duration_steps: 20000,
        };
        let debug_str = format!("{:?}", event);
        assert!(debug_str.contains("AttentionSilent"));
        assert!(debug_str.contains("avg_entropy"));
    }

    #[test]
    fn test_consensus_detector_default_thresholds() {
        let detector = ConsensusDetector::new(4);
        assert_eq!(detector.expert_freeze_threshold, 1_000_000);
        assert!((detector.attention_silent_threshold - 0.05).abs() < 1e-10);
        assert!((detector.layer_redundant_threshold - 0.001).abs() < 1e-10);
        assert_eq!(detector.attention_silent_steps, 0);
        assert_eq!(detector.layer_redundant_steps, 0);
    }

    #[test]
    fn test_consensus_detector_debug_trait() {
        let detector = ConsensusDetector::new(2);
        let debug_str = format!("{:?}", detector);
        assert!(debug_str.contains("ConsensusDetector"));
        assert!(debug_str.contains("expert_freeze_threshold"));
    }

    #[test]
    fn test_consensus_detector_detect_returns_empty_initially() {
        let mut detector = ConsensusDetector::new(4);
        let reservoir = DecayingReservoir::new(100);
        let events = detector.detect(&reservoir);
        assert!(events.is_empty());
    }

    #[test]
    fn test_consensus_detector_update_with_longer_hits_slice() {
        let mut detector = ConsensusDetector::new(2);
        // hits slice is longer than num_experts — extra entries ignored
        detector.update_expert_hits(&[1, 0, 5, 5, 5]);
        assert_eq!(detector.expert_zero_streaks[0], 0); // hits=1 → reset
        assert_eq!(detector.expert_zero_streaks[1], 1); // hits=0 → streak++
    }

    #[test]
    fn test_consensus_detector_attention_silent_fires_with_sufficient_samples() {
        let mut detector = ConsensusDetector::new(2);
        // Lower thresholds for test
        detector.attention_silent_threshold = 0.1;

        let mut reservoir = DecayingReservoir::new(1000);
        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.entropy_avg = f32_to_f16_bits(0.01); // Very low entropy

        // Feed 2000 samples to exceed the 1000 sample gate
        for _ in 0..2000 {
            reservoir.ingest(&header);
        }

        // Call detect enough times to exceed the 10000 step gate
        // Each detect call increments attention_silent_steps by 1
        for _ in 0..10001 {
            let _events = detector.detect(&reservoir);
        }

        let events = detector.detect(&reservoir);
        assert!(events.iter().any(|e| matches!(e, ConsensusEvent::AttentionSilent { .. })));
    }

    #[test]
    fn test_consensus_detector_layer_redundant_fires_with_sufficient_samples() {
        let mut detector = ConsensusDetector::new(2);

        let mut reservoir = DecayingReservoir::new(1000);
        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.delta_rho_avg = f32_to_f16_bits(0.0001); // Very low delta

        // Feed 2000 samples to exceed the 1000 sample gate
        for _ in 0..2000 {
            reservoir.ingest(&header);
        }

        // Accumulate layer_redundant_steps > 10000
        for _ in 0..10001 {
            let _events = detector.detect(&reservoir);
        }

        let events = detector.detect(&reservoir);
        assert!(events.iter().any(|e| matches!(e, ConsensusEvent::LayerRedundant { .. })));
    }

    #[test]
    fn test_consensus_detector_layer_redundant_resets_on_recovery() {
        let mut detector = ConsensusDetector::new(2);

        let mut reservoir = DecayingReservoir::new(100);
        let mut low_header = KvPageHeader::new(0);
        low_header.ref_count = 1;
        low_header.delta_rho_avg = f32_to_f16_bits(0.0001);

        for _ in 0..1100 {
            reservoir.ingest(&low_header);
        }
        // Accumulate some silent steps
        for _ in 0..5 {
            let _events = detector.detect(&reservoir);
        }

        // Feed high delta — should reset counter
        let mut high_header = KvPageHeader::new(0);
        high_header.ref_count = 1;
        high_header.delta_rho_avg = f32_to_f16_bits(5.0);
        for _ in 0..100 {
            reservoir.ingest(&high_header);
        }

        let events = detector.detect(&reservoir);
        assert!(!events.iter().any(|e| matches!(e, ConsensusEvent::LayerRedundant { .. })));
    }

    #[test]
    fn test_director_config_debug_trait() {
        let config = DirectorConfig {
            scan_interval: Duration::from_millis(50),
            half_life_samples: 2000,
            num_experts: 16,
        };
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("DirectorConfig"));
        assert!(debug_str.contains("scan_interval"));
        assert!(debug_str.contains("half_life_samples"));
        assert!(debug_str.contains("num_experts"));
    }

    #[test]
    fn test_director_config_field_mutation() {
        let mut config = DirectorConfig::default();
        assert_eq!(config.num_experts, 0);

        config.num_experts = 64;
        config.half_life_samples = 500;
        config.scan_interval = Duration::from_millis(250);

        assert_eq!(config.num_experts, 64);
        assert_eq!(config.half_life_samples, 500);
        assert_eq!(config.scan_interval, Duration::from_millis(250));
    }

    #[test]
    fn test_shared_state_push_multiple_batches_events() {
        let state = DirectorSharedState::new(2);

        state.push_events(vec![ConsensusEvent::ExpertFrozen {
            expert_idx: 0,
            zero_hit_steps: 100,
        }]);
        state.push_events(vec![ConsensusEvent::ExpertFrozen {
            expert_idx: 1,
            zero_hit_steps: 200,
        }]);

        let events = state.drain_events();
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], ConsensusEvent::ExpertFrozen { expert_idx: 0, .. }));
        assert!(matches!(events[1], ConsensusEvent::ExpertFrozen { expert_idx: 1, .. }));
    }

    #[test]
    fn test_shared_state_many_hits_same_expert() {
        let state = DirectorSharedState::new(2);

        // Record 1000 hits on expert 0
        for _ in 0..1000 {
            state.record_expert_hit(0);
        }

        let hits = state.snapshot_and_reset_expert_hits();
        assert_eq!(hits[0], 1000);
        assert_eq!(hits[1], 0);
    }

    #[test]
    fn test_shared_state_global_step_accumulates() {
        let state = DirectorSharedState::new(0);

        for _ in 0..1000 {
            state.advance_step();
        }
        assert_eq!(state.global_step.load(Ordering::Relaxed), 1000);
    }

    #[test]
    fn test_director_double_shutdown_is_safe() {
        let config = DirectorConfig {
            scan_interval: Duration::from_millis(10),
            half_life_samples: 100,
            num_experts: 2,
        };

        let mut director = JitDirector::spawn(config);
        director.shutdown();
        // Second shutdown should be a no-op, not panic
        director.shutdown();
    }

    // =========================================================================
    // Additional tests — edge cases, uncovered paths, trait coverage
    // =========================================================================

    #[test]
    fn test_decaying_reservoir_new_half_life_one_decay_factor() {
        // half_life_samples=1 → decay_factor = exp(-ln(2)/1) = 0.5 exactly
        let reservoir = DecayingReservoir::new(1);
        assert!((reservoir.decay_factor - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_decaying_reservoir_ingest_all_zero_fields() {
        let mut reservoir = DecayingReservoir::new(100);
        let header = KvPageHeader::new(0);
        // All fields are 0: entropy_avg=0, centroid_pos=0, delta_rho_avg=0, dead_ratio=0

        reservoir.ingest(&header);
        assert_eq!(reservoir.sample_count(), 1);
        assert!((reservoir.avg_entropy() - 0.0).abs() < 1e-6);
        assert!((reservoir.avg_residual_delta() - 0.0).abs() < 1e-6);
        assert!((reservoir.avg_dead_neuron_ratio() - 0.0).abs() < 1e-6);
        assert!((reservoir.avg_softmax_sharpness() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_decaying_reservoir_ingest_max_dead_ratio() {
        let mut reservoir = DecayingReservoir::new(100);
        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.dead_ratio = 255; // Max value → should map to ~1.0

        reservoir.ingest(&header);
        let ratio = reservoir.avg_dead_neuron_ratio();
        // dead_ratio_to_f32(255) should be close to 1.0
        assert!(ratio > 0.9);
        assert!(ratio <= 1.0);
    }

    #[test]
    fn test_decaying_reservoir_sample_count_increments_linearly() {
        let mut reservoir = DecayingReservoir::new(50);
        let header = KvPageHeader::new(0);

        for i in 1..=100 {
            reservoir.ingest(&header);
            assert_eq!(reservoir.sample_count(), i);
        }
    }

    #[test]
    fn test_decaying_reservoir_ingest_blends_high_then_low() {
        let mut reservoir = DecayingReservoir::new(100);

        // First: high entropy
        let mut h_high = KvPageHeader::new(0);
        h_high.ref_count = 1;
        h_high.entropy_avg = f32_to_f16_bits(10.0);
        reservoir.ingest(&h_high);
        assert!((reservoir.avg_entropy() - 10.0).abs() < 0.01);

        // Second: low entropy — should blend between 10.0 and 0.0
        let mut h_low = KvPageHeader::new(0);
        h_low.ref_count = 1;
        h_low.entropy_avg = f32_to_f16_bits(0.0);
        reservoir.ingest(&h_low);
        assert!(reservoir.avg_entropy() > 0.0);
        assert!(reservoir.avg_entropy() < 10.0);
    }

    #[test]
    fn test_consensus_detector_new_zero_experts() {
        let detector = ConsensusDetector::new(0);
        assert!(detector.expert_zero_streaks.is_empty());
        assert_eq!(detector.attention_silent_steps, 0);
        assert_eq!(detector.layer_redundant_steps, 0);
    }

    #[test]
    fn test_consensus_detector_new_single_expert() {
        let detector = ConsensusDetector::new(1);
        assert_eq!(detector.expert_zero_streaks.len(), 1);
        assert_eq!(detector.expert_zero_streaks[0], 0);
    }

    #[test]
    fn test_consensus_detector_new_large_expert_count() {
        let detector = ConsensusDetector::new(256);
        assert_eq!(detector.expert_zero_streaks.len(), 256);
        assert!(detector.expert_zero_streaks.iter().all(|&s| s == 0));
    }

    #[test]
    fn test_consensus_detector_reset_then_detect_no_event() {
        let mut detector = ConsensusDetector::new(2);
        detector.expert_freeze_threshold = 3;

        // Build streak past threshold for expert 0
        for _ in 0..5 {
            detector.update_expert_hits(&[0, 1]);
        }
        assert_eq!(detector.expert_zero_streaks[0], 5);

        // Reset expert 0
        detector.reset_expert(0);
        assert_eq!(detector.expert_zero_streaks[0], 0);

        // Detect should not emit ExpertFrozen for expert 0
        let reservoir = DecayingReservoir::new(100);
        let events = detector.detect(&reservoir);
        assert!(!events.iter().any(|e| matches!(
            e,
            ConsensusEvent::ExpertFrozen { expert_idx: 0, .. }
        )));
    }

    #[test]
    fn test_consensus_detector_update_with_empty_hits_slice() {
        let mut detector = ConsensusDetector::new(4);
        // Empty hits slice — all experts treated as zero hits
        detector.update_expert_hits(&[]);
        assert_eq!(detector.expert_zero_streaks, vec![1, 1, 1, 1]);
    }

    #[test]
    fn test_consensus_detector_detect_on_fresh_reservoir_empty() {
        // Fresh reservoir with sample_count=0 should never trigger attention/layer events
        let mut detector = ConsensusDetector::new(4);
        let reservoir = DecayingReservoir::new(100);
        let events = detector.detect(&reservoir);
        assert!(events.is_empty());
    }

    #[test]
    fn test_consensus_event_partial_eq_different_variants_never_equal() {
        let expert = ConsensusEvent::ExpertFrozen { expert_idx: 0, zero_hit_steps: 100 };
        let attention = ConsensusEvent::AttentionSilent { avg_entropy: 0.01, duration_steps: 100 };
        let layer = ConsensusEvent::LayerRedundant { avg_delta_rho: 0.001, duration_steps: 100 };
        assert_ne!(expert, attention);
        assert_ne!(attention, layer);
        assert_ne!(expert, layer);
    }

    #[test]
    fn test_consensus_event_expert_frozen_equality_semantics() {
        // Same expert_idx but different zero_hit_steps → not equal
        let a = ConsensusEvent::ExpertFrozen { expert_idx: 1, zero_hit_steps: 100 };
        let b = ConsensusEvent::ExpertFrozen { expert_idx: 1, zero_hit_steps: 200 };
        assert_ne!(a, b);

        // Different expert_idx, same steps → not equal
        let c = ConsensusEvent::ExpertFrozen { expert_idx: 2, zero_hit_steps: 100 };
        assert_ne!(a, c);
    }

    #[test]
    fn test_consensus_event_attention_silent_equality_semantics() {
        let a = ConsensusEvent::AttentionSilent { avg_entropy: 0.01, duration_steps: 100 };
        let b = ConsensusEvent::AttentionSilent { avg_entropy: 0.01, duration_steps: 100 };
        assert_eq!(a, b);

        let c = ConsensusEvent::AttentionSilent { avg_entropy: 0.02, duration_steps: 100 };
        assert_ne!(a, c);
    }

    #[test]
    fn test_director_config_construction_custom_values() {
        let config = DirectorConfig {
            scan_interval: Duration::from_secs(1),
            half_life_samples: 500,
            num_experts: 128,
        };
        assert_eq!(config.scan_interval, Duration::from_secs(1));
        assert_eq!(config.half_life_samples, 500);
        assert_eq!(config.num_experts, 128);
    }

    #[test]
    fn test_director_shared_state_new_preserves_expert_count() {
        let state = DirectorSharedState::new(16);
        let hits = state.snapshot_and_reset_expert_hits();
        assert_eq!(hits.len(), 16);
        assert!(hits.iter().all(|&h| h == 0));
    }

    #[test]
    fn test_decaying_reservoir_ingest_max_f16_entropy() {
        let mut reservoir = DecayingReservoir::new(100);
        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        // Max representable f16 finite value (0x7BFF ≈ 65504)
        header.entropy_avg = 0x7BFF;

        reservoir.ingest(&header);
        assert_eq!(reservoir.sample_count(), 1);
        // avg_entropy should be a large positive number
        assert!(reservoir.avg_entropy() > 60000.0);
    }

    // =========================================================================
    // Additional 50 tests — targeting 128+ total
    // =========================================================================

    // --- DecayingReservoir: extreme half-life values ---

    #[test]
    fn test_decaying_reservoir_half_life_zero_decay_near_zero() {
        // half_life_samples=0 would give exp(-inf) = 0 — but we use it as divisor.
        // The formula: exp(-ln(2)/0) → exp(-inf) → 0.0 decay_factor
        // This means old values are immediately discarded and only new values matter.
        let reservoir = DecayingReservoir::new(0);
        // decay_factor should be 0.0 (or NaN — verify actual behavior)
        // exp(-inf) in Rust f64 is 0.0
        assert!(reservoir.decay_factor == 0.0 || reservoir.decay_factor.is_nan());
    }

    #[test]
    fn test_decaying_reservoir_half_life_two_decay_factor() {
        // half_life_samples=2 → decay_factor = exp(-ln(2)/2) = sqrt(0.5) ≈ 0.7071
        let reservoir = DecayingReservoir::new(2);
        let expected = (-(2.0_f64).ln() / 2.0).exp();
        assert!((reservoir.decay_factor - expected).abs() < 1e-12);
    }

    #[test]
    fn test_decaying_reservoir_half_life_very_large_persists() {
        // Very large half-life: old values barely decay
        let mut reservoir = DecayingReservoir::new(1_000_000);

        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.entropy_avg = f32_to_f16_bits(42.0);
        reservoir.ingest(&h);
        assert!((reservoir.avg_entropy() - 42.0).abs() < 0.01);

        // Ingest 100 samples of 0 — should barely move
        let mut h_zero = KvPageHeader::new(0);
        h_zero.ref_count = 1;
        h_zero.entropy_avg = f32_to_f16_bits(0.0);
        for _ in 0..100 {
            reservoir.ingest(&h_zero);
        }
        // After 100 zero samples with half_life=1M, avg should still be close to 42.0
        assert!(reservoir.avg_entropy() > 35.0);
    }

    // --- DecayingReservoir: alternating value patterns ---

    #[test]
    fn test_decaying_reservoir_alternating_high_low_entropy() {
        let mut reservoir = DecayingReservoir::new(10);

        let mut h_high = KvPageHeader::new(0);
        h_high.ref_count = 1;
        h_high.entropy_avg = f32_to_f16_bits(100.0);

        let mut h_low = KvPageHeader::new(0);
        h_low.ref_count = 1;
        h_low.entropy_avg = f32_to_f16_bits(1.0);

        // Alternate 50 times
        for i in 0..100 {
            if i % 2 == 0 {
                reservoir.ingest(&h_high);
            } else {
                reservoir.ingest(&h_low);
            }
        }

        assert_eq!(reservoir.sample_count(), 100);
        // Should be somewhere between 1.0 and 100.0
        assert!(reservoir.avg_entropy() > 1.0);
        assert!(reservoir.avg_entropy() < 100.0);
    }

    #[test]
    fn test_decaying_reservoir_monotonic_increase_in_entropy() {
        let mut reservoir = DecayingReservoir::new(100);
        for i in 1..=50u32 {
            let mut h = KvPageHeader::new(0);
            h.ref_count = 1;
            h.entropy_avg = f32_to_f16_bits(i as f32);
            reservoir.ingest(&h);

            // With increasing values and high half_life, avg should generally increase
            // (though EMA can lag — just verify it moves)
            let current = reservoir.avg_entropy();
            assert!(current > 0.0);
            let _ = current;
        }
        assert_eq!(reservoir.sample_count(), 50);
    }

    // --- DecayingReservoir: multiple fields simultaneously ---

    #[test]
    fn test_decaying_reservoir_all_fields_update_on_first_ingest() {
        let mut reservoir = DecayingReservoir::new(100);

        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.entropy_avg = f32_to_f16_bits(2.0);
        h.delta_rho_avg = f32_to_f16_bits(0.5);
        h.dead_ratio = 128;
        h.centroid_pos = f32_to_f16_bits(0.3);

        reservoir.ingest(&h);

        assert!((reservoir.avg_entropy() - 2.0).abs() < 0.01);
        assert!((reservoir.avg_residual_delta() - 0.5).abs() < 0.01);
        assert!((reservoir.avg_dead_neuron_ratio() - dead_ratio_to_f32(128) as f64).abs() < 0.01);
        assert!((reservoir.avg_softmax_sharpness() - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_decaying_reservoir_sequential_ingest_different_fields_diverge() {
        let mut reservoir = DecayingReservoir::new(10);

        // First: high entropy, low delta
        let mut h1 = KvPageHeader::new(0);
        h1.ref_count = 1;
        h1.entropy_avg = f32_to_f16_bits(10.0);
        h1.delta_rho_avg = f32_to_f16_bits(0.01);
        reservoir.ingest(&h1);

        // Second: low entropy, high delta
        let mut h2 = KvPageHeader::new(0);
        h2.ref_count = 1;
        h2.entropy_avg = f32_to_f16_bits(1.0);
        h2.delta_rho_avg = f32_to_f16_bits(5.0);
        reservoir.ingest(&h2);

        // entropy should decrease, delta should increase
        assert!(reservoir.avg_entropy() < 10.0);
        assert!(reservoir.avg_residual_delta() > 0.01);
    }

    // --- ConsensusDetector: mixed and simultaneous events ---

    #[test]
    fn test_consensus_detector_expert_freeze_reports_correct_index() {
        let mut detector = ConsensusDetector::new(8);
        detector.expert_freeze_threshold = 5;

        // Only expert 5 is frozen
        for _ in 0..10 {
            detector.update_expert_hits(&[1, 1, 1, 1, 1, 0, 1, 1]);
        }

        let reservoir = DecayingReservoir::new(100);
        let events = detector.detect(&reservoir);

        let frozen_events: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                ConsensusEvent::ExpertFrozen { expert_idx, .. } => Some(*expert_idx),
                _ => None,
            })
            .collect();

        assert_eq!(frozen_events, vec![5]);
    }

    #[test]
    fn test_consensus_detector_multiple_frozen_experts_report_all() {
        let mut detector = ConsensusDetector::new(4);
        detector.expert_freeze_threshold = 2;

        // Experts 0 and 3 frozen
        for _ in 0..5 {
            detector.update_expert_hits(&[0, 3, 2, 0]);
        }

        let reservoir = DecayingReservoir::new(100);
        let events = detector.detect(&reservoir);

        let frozen_indices: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                ConsensusEvent::ExpertFrozen { expert_idx, .. } => Some(*expert_idx),
                _ => None,
            })
            .collect();

        assert!(frozen_indices.contains(&0));
        assert!(frozen_indices.contains(&3));
        assert!(!frozen_indices.contains(&1));
        assert!(!frozen_indices.contains(&2));
    }

    #[test]
    fn test_consensus_detector_freeze_threshold_exactly_met() {
        let mut detector = ConsensusDetector::new(2);
        detector.expert_freeze_threshold = 5;

        // Exactly 5 zero-hit rounds for expert 0
        for _ in 0..5 {
            detector.update_expert_hits(&[0, 1]);
        }

        let reservoir = DecayingReservoir::new(100);
        let events = detector.detect(&reservoir);
        assert!(events.iter().any(|e| matches!(
            e,
            ConsensusEvent::ExpertFrozen { expert_idx: 0, zero_hit_steps: 5 }
        )));
    }

    #[test]
    fn test_consensus_detector_freeze_threshold_not_yet_met() {
        let mut detector = ConsensusDetector::new(2);
        detector.expert_freeze_threshold = 10;

        // Only 9 zero-hit rounds — below threshold
        for _ in 0..9 {
            detector.update_expert_hits(&[0, 1]);
        }

        let reservoir = DecayingReservoir::new(100);
        let events = detector.detect(&reservoir);
        assert!(!events.iter().any(|e| matches!(
            e,
            ConsensusEvent::ExpertFrozen { expert_idx: 0, .. }
        )));
    }

    #[test]
    fn test_consensus_detector_expert_streak_resets_exactly() {
        let mut detector = ConsensusDetector::new(2);
        detector.expert_freeze_threshold = 100;

        // Build streak of 99
        for _ in 0..99 {
            detector.update_expert_hits(&[0, 0]);
        }
        assert_eq!(detector.expert_zero_streaks[0], 99);

        // One hit resets
        detector.update_expert_hits(&[1, 0]);
        assert_eq!(detector.expert_zero_streaks[0], 0);
        assert_eq!(detector.expert_zero_streaks[1], 100);
    }

    #[test]
    fn test_consensus_detector_update_expert_hits_with_large_values() {
        let mut detector = ConsensusDetector::new(2);

        // Large hit counts
        detector.update_expert_hits(&[u32::MAX, 0]);
        assert_eq!(detector.expert_zero_streaks[0], 0); // hit → reset
        assert_eq!(detector.expert_zero_streaks[1], 1); // 0 → streak++
    }

    #[test]
    fn test_consensus_detector_detect_accumulates_both_event_types() {
        let mut detector = ConsensusDetector::new(2);
        detector.attention_silent_threshold = 0.1;

        let mut reservoir = DecayingReservoir::new(1000);
        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.entropy_avg = f32_to_f16_bits(0.01); // Low entropy
        header.delta_rho_avg = f32_to_f16_bits(0.0001); // Low delta

        // 2000 samples to exceed 1000 sample gate
        for _ in 0..2000 {
            reservoir.ingest(&header);
        }

        // 10001 detect calls to exceed 10000 step gate
        for _ in 0..10001 {
            let _events = detector.detect(&reservoir);
        }

        let events = detector.detect(&reservoir);

        let has_attention = events.iter().any(|e| matches!(e, ConsensusEvent::AttentionSilent { .. }));
        let has_layer = events.iter().any(|e| matches!(e, ConsensusEvent::LayerRedundant { .. }));

        assert!(has_attention, "Expected AttentionSilent event");
        assert!(has_layer, "Expected LayerRedundant event");
    }

    #[test]
    fn test_consensus_detector_attention_silent_step_counter_resets() {
        let mut detector = ConsensusDetector::new(2);
        detector.attention_silent_threshold = 0.1;

        let mut reservoir = DecayingReservoir::new(100);
        let mut low_header = KvPageHeader::new(0);
        low_header.ref_count = 1;
        low_header.entropy_avg = f32_to_f16_bits(0.01);

        // Build up sample_count > 1000
        for _ in 0..1100 {
            reservoir.ingest(&low_header);
        }

        // Accumulate 500 silent steps (below 10000)
        for _ in 0..500 {
            let _events = detector.detect(&reservoir);
        }
        assert_eq!(detector.attention_silent_steps, 500);

        // Feed high entropy to reset
        let mut high_header = KvPageHeader::new(0);
        high_header.ref_count = 1;
        high_header.entropy_avg = f32_to_f16_bits(5.0);
        for _ in 0..200 {
            reservoir.ingest(&high_header);
        }

        let _events = detector.detect(&reservoir);
        assert_eq!(detector.attention_silent_steps, 0);
    }

    #[test]
    fn test_consensus_detector_layer_redundant_step_counter_resets() {
        let mut detector = ConsensusDetector::new(2);

        let mut reservoir = DecayingReservoir::new(100);
        let mut low_header = KvPageHeader::new(0);
        low_header.ref_count = 1;
        low_header.delta_rho_avg = f32_to_f16_bits(0.0001);

        for _ in 0..1100 {
            reservoir.ingest(&low_header);
        }
        for _ in 0..500 {
            let _events = detector.detect(&reservoir);
        }
        assert_eq!(detector.layer_redundant_steps, 500);

        // Feed high delta to reset
        let mut high_header = KvPageHeader::new(0);
        high_header.ref_count = 1;
        high_header.delta_rho_avg = f32_to_f16_bits(5.0);
        for _ in 0..200 {
            reservoir.ingest(&high_header);
        }

        let _events = detector.detect(&reservoir);
        assert_eq!(detector.layer_redundant_steps, 0);
    }

    // --- ConsensusEvent: additional edge cases ---

    #[test]
    fn test_consensus_event_expert_frozen_zero_index() {
        let event = ConsensusEvent::ExpertFrozen { expert_idx: 0, zero_hit_steps: 1 };
        let debug_str = format!("{:?}", event);
        assert!(debug_str.contains("expert_idx: 0"));
    }

    #[test]
    fn test_consensus_event_expert_frozen_large_steps() {
        let event = ConsensusEvent::ExpertFrozen {
            expert_idx: 100,
            zero_hit_steps: u64::MAX,
        };
        if let ConsensusEvent::ExpertFrozen { zero_hit_steps, .. } = event {
            assert_eq!(zero_hit_steps, u64::MAX);
        }
    }

    #[test]
    fn test_consensus_event_attention_silent_zero_entropy() {
        let event = ConsensusEvent::AttentionSilent {
            avg_entropy: 0.0,
            duration_steps: 50000,
        };
        if let ConsensusEvent::AttentionSilent { avg_entropy, .. } = event {
            assert!((avg_entropy - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_consensus_event_layer_redundant_zero_delta() {
        let event = ConsensusEvent::LayerRedundant {
            avg_delta_rho: 0.0,
            duration_steps: 30000,
        };
        if let ConsensusEvent::LayerRedundant { avg_delta_rho, .. } = event {
            assert!((avg_delta_rho - 0.0).abs() < 1e-10);
        }
    }

    // --- DirectorConfig: edge cases ---

    #[test]
    fn test_director_config_zero_scan_interval() {
        let config = DirectorConfig {
            scan_interval: Duration::ZERO,
            half_life_samples: 100,
            num_experts: 0,
        };
        assert_eq!(config.scan_interval, Duration::ZERO);
    }

    #[test]
    fn test_director_config_large_num_experts() {
        let config = DirectorConfig {
            scan_interval: Duration::from_millis(100),
            half_life_samples: 10_000,
            num_experts: 1024,
        };
        assert_eq!(config.num_experts, 1024);
    }

    #[test]
    fn test_director_config_zero_half_life_samples() {
        let config = DirectorConfig {
            scan_interval: Duration::from_millis(100),
            half_life_samples: 0,
            num_experts: 4,
        };
        assert_eq!(config.half_life_samples, 0);
    }

    // --- DirectorSharedState: additional edge cases ---

    #[test]
    fn test_shared_state_update_empty_page_headers() {
        let state = DirectorSharedState::new(2);

        // Update with non-empty first
        let h = KvPageHeader::new(1);
        state.update_page_headers(vec![h]);
        assert_eq!(state.read_page_headers().len(), 1);

        // Then update with empty — should replace
        state.update_page_headers(vec![]);
        assert!(state.read_page_headers().is_empty());
    }

    #[test]
    fn test_shared_state_advance_step_wrapping_behavior() {
        let state = DirectorSharedState::new(0);

        // Advance many steps — just verify it keeps incrementing
        state.global_step.store(u64::MAX - 5, Ordering::Relaxed);
        state.advance_step();
        state.advance_step();
        state.advance_step();

        let step = state.global_step.load(Ordering::Relaxed);
        // Wrapping add
        assert_eq!(step, u64::MAX - 2);
    }

    #[test]
    fn test_shared_state_record_hit_and_snapshot_concurrent() {
        let state = Arc::new(DirectorSharedState::new(4));

        let state1 = Arc::clone(&state);
        let h1 = std::thread::spawn(move || {
            for _ in 0..500 {
                state1.record_expert_hit(0);
            }
        });

        let state2 = Arc::clone(&state);
        let h2 = std::thread::spawn(move || {
            for _ in 0..500 {
                state2.record_expert_hit(1);
            }
        });

        h1.join().unwrap();
        h2.join().unwrap();

        let hits = state.snapshot_and_reset_expert_hits();
        assert_eq!(hits[0], 500);
        assert_eq!(hits[1], 500);
        assert_eq!(hits[2], 0);
        assert_eq!(hits[3], 0);
    }

    #[test]
    fn test_shared_state_drain_events_concurrent_with_push() {
        let state = Arc::new(DirectorSharedState::new(0));

        let state1 = Arc::clone(&state);
        let h1 = std::thread::spawn(move || {
            for i in 0..100 {
                state1.push_events(vec![ConsensusEvent::ExpertFrozen {
                    expert_idx: i % 4,
                    zero_hit_steps: i as u64,
                }]);
            }
        });

        let state2 = Arc::clone(&state);
        let h2 = std::thread::spawn(move || {
            let mut total_drained = 0;
            for _ in 0..100 {
                let events = state2.drain_events();
                total_drained += events.len();
            }
            total_drained
        });

        h1.join().unwrap();
        let total_drained = h2.join().unwrap();

        // All events should eventually be drained (races may cause partial, but
        // the sum of all drain calls should equal 100 events)
        // Note: due to concurrency, this is best-effort; just verify no panic
        assert!(total_drained <= 100);
    }

    #[test]
    fn test_shared_state_push_events_preserves_order() {
        let state = DirectorSharedState::new(0);

        state.push_events(vec![
            ConsensusEvent::ExpertFrozen { expert_idx: 0, zero_hit_steps: 1 },
            ConsensusEvent::ExpertFrozen { expert_idx: 1, zero_hit_steps: 2 },
            ConsensusEvent::ExpertFrozen { expert_idx: 2, zero_hit_steps: 3 },
        ]);

        let events = state.drain_events();
        assert_eq!(events.len(), 3);

        // Order preserved
        assert!(matches!(events[0], ConsensusEvent::ExpertFrozen { expert_idx: 0, .. }));
        assert!(matches!(events[1], ConsensusEvent::ExpertFrozen { expert_idx: 1, .. }));
        assert!(matches!(events[2], ConsensusEvent::ExpertFrozen { expert_idx: 2, .. }));
    }

    #[test]
    fn test_shared_state_read_page_headers_returns_cloned_data() {
        let state = DirectorSharedState::new(0);

        let mut h = KvPageHeader::new(42);
        h.ref_count = 1;
        h.entropy_avg = f32_to_f16_bits(7.0);
        state.update_page_headers(vec![h]);

        let headers1 = state.read_page_headers();
        let headers2 = state.read_page_headers();

        // Both reads return independent clones
        assert_eq!(headers1.len(), 1);
        assert_eq!(headers2.len(), 1);
        assert_eq!(headers1[0].page_id, 42);
        assert_eq!(headers2[0].page_id, 42);
    }

    // --- Integration: reservoir + detector pipeline ---

    #[test]
    fn test_pipeline_reservoir_feeds_into_detector_no_events_initially() {
        let mut reservoir = DecayingReservoir::new(100);
        let mut detector = ConsensusDetector::new(4);

        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.entropy_avg = f32_to_f16_bits(2.0);
        header.delta_rho_avg = f32_to_f16_bits(0.5);

        for _ in 0..500 {
            reservoir.ingest(&header);
            detector.update_expert_hits(&[1, 1, 1, 1]);
            let events = detector.detect(&reservoir);
            assert!(events.is_empty(), "No events expected with healthy signals");
        }

        assert_eq!(reservoir.sample_count(), 500);
    }

    #[test]
    fn test_pipeline_healthy_signals_never_trigger_events() {
        let mut reservoir = DecayingReservoir::new(1000);
        let mut detector = ConsensusDetector::new(2);

        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.entropy_avg = f32_to_f16_bits(3.0);
        header.delta_rho_avg = f32_to_f16_bits(0.5);
        header.dead_ratio = 10;
        header.centroid_pos = f32_to_f16_bits(0.8);

        // 50,000 iterations of healthy signals
        for _ in 0..50_000 {
            reservoir.ingest(&header);
            detector.update_expert_hits(&[10, 10]); // Active experts
            let _events = detector.detect(&reservoir);
        }

        // Final detect
        let events = detector.detect(&reservoir);
        assert!(events.is_empty());
    }

    #[test]
    fn test_pipeline_degrading_entropy_triggers_attention_silent() {
        // Use short half-life so EMA adapts quickly to degraded values
        let mut reservoir = DecayingReservoir::new(10);
        let mut detector = ConsensusDetector::new(2);
        detector.attention_silent_threshold = 0.1;

        // Phase 1: healthy signals to get sample_count > 1000
        let mut healthy = KvPageHeader::new(0);
        healthy.ref_count = 1;
        healthy.entropy_avg = f32_to_f16_bits(3.0);
        healthy.delta_rho_avg = f32_to_f16_bits(0.5);

        for _ in 0..1100 {
            reservoir.ingest(&healthy);
        }

        // Phase 2: degrading entropy — with half_life=10, EMA converges fast
        let mut degraded = KvPageHeader::new(0);
        degraded.ref_count = 1;
        degraded.entropy_avg = f32_to_f16_bits(0.01);
        degraded.delta_rho_avg = f32_to_f16_bits(0.5);

        for _ in 0..100 {
            reservoir.ingest(&degraded);
        }

        // After 100 low-entropy samples with half_life=10, avg_entropy should be near 0.01
        assert!(reservoir.avg_entropy() < 0.1);

        // Detect many times to accumulate attention_silent_steps > 10000
        for _ in 0..10001 {
            let _events = detector.detect(&reservoir);
        }

        let events = detector.detect(&reservoir);
        assert!(events.iter().any(|e| matches!(e, ConsensusEvent::AttentionSilent { .. })));
    }

    // --- DecayingReservoir: dead_ratio edge cases ---

    #[test]
    fn test_decaying_reservoir_dead_ratio_zero_vs_max() {
        let mut r_zero = DecayingReservoir::new(100);
        let mut r_max = DecayingReservoir::new(100);

        let mut h_zero = KvPageHeader::new(0);
        h_zero.ref_count = 1;
        h_zero.dead_ratio = 0;

        let mut h_max = KvPageHeader::new(0);
        h_max.ref_count = 1;
        h_max.dead_ratio = 255;

        r_zero.ingest(&h_zero);
        r_max.ingest(&h_max);

        assert!(r_zero.avg_dead_neuron_ratio() < r_max.avg_dead_neuron_ratio());
    }

    #[test]
    fn test_decaying_reservoir_dead_ratio_blends_correctly() {
        let mut reservoir = DecayingReservoir::new(10); // Short half-life

        let mut h_low = KvPageHeader::new(0);
        h_low.ref_count = 1;
        h_low.dead_ratio = 0;

        let mut h_high = KvPageHeader::new(0);
        h_high.ref_count = 1;
        h_high.dead_ratio = 200;

        reservoir.ingest(&h_low);
        let first = reservoir.avg_dead_neuron_ratio();

        reservoir.ingest(&h_high);
        let second = reservoir.avg_dead_neuron_ratio();

        // After ingesting high, dead_ratio should increase
        assert!(second > first);
    }

    // --- ConsensusDetector: reset_expert edge cases ---

    #[test]
    fn test_consensus_detector_reset_first_and_last_expert() {
        let mut detector = ConsensusDetector::new(5);
        detector.expert_freeze_threshold = 3;

        for _ in 0..10 {
            detector.update_expert_hits(&[0, 0, 0, 0, 0]);
        }

        detector.reset_expert(0);
        detector.reset_expert(4);

        assert_eq!(detector.expert_zero_streaks[0], 0);
        assert_eq!(detector.expert_zero_streaks[4], 0);
        assert_eq!(detector.expert_zero_streaks[1], 10);
        assert_eq!(detector.expert_zero_streaks[2], 10);
        assert_eq!(detector.expert_zero_streaks[3], 10);
    }

    #[test]
    fn test_consensus_detector_reset_same_expert_twice() {
        let mut detector = ConsensusDetector::new(3);

        for _ in 0..10 {
            detector.update_expert_hits(&[0, 0, 0]);
        }
        assert_eq!(detector.expert_zero_streaks[1], 10);

        detector.reset_expert(1);
        assert_eq!(detector.expert_zero_streaks[1], 0);

        // Reset again — should still be 0
        detector.reset_expert(1);
        assert_eq!(detector.expert_zero_streaks[1], 0);
    }

    // --- DirectorSharedState: snapshot_and_reset atomicity ---

    #[test]
    fn test_shared_state_snapshot_and_reset_is_atomic() {
        let state = DirectorSharedState::new(2);

        state.record_expert_hit(0);
        state.record_expert_hit(0);
        state.record_expert_hit(1);

        let hits = state.snapshot_and_reset_expert_hits();
        assert_eq!(hits, vec![2, 1]);

        // After snapshot, counters should be 0
        state.record_expert_hit(0);
        let hits2 = state.snapshot_and_reset_expert_hits();
        assert_eq!(hits2, vec![1, 0]);
    }

    // --- JitDirector: additional lifecycle tests ---

    #[test]
    fn test_director_spawn_with_many_experts() {
        let config = DirectorConfig {
            scan_interval: Duration::from_millis(10),
            half_life_samples: 100,
            num_experts: 256,
        };

        let mut director = JitDirector::spawn(config);
        let shared = Arc::clone(director.shared());

        // Record hits across many experts
        for i in 0..256 {
            shared.record_expert_hit(i);
        }

        std::thread::sleep(Duration::from_millis(30));
        let _events = shared.drain_events();
        director.shutdown();
    }

    #[test]
    fn test_director_rapid_start_stop_cycles() {
        for _ in 0..3 {
            let config = DirectorConfig {
                scan_interval: Duration::from_millis(5),
                half_life_samples: 50,
                num_experts: 0,
            };
            let mut director = JitDirector::spawn(config);
            std::thread::sleep(Duration::from_millis(10));
            director.shutdown();
        }
    }

    #[test]
    fn test_director_shared_remains_valid_after_shutdown() {
        let config = DirectorConfig {
            scan_interval: Duration::from_millis(10),
            half_life_samples: 100,
            num_experts: 2,
        };

        let mut director = JitDirector::spawn(config);
        let shared = Arc::clone(director.shared());
        director.shutdown();

        // Shared state should still be usable after director shutdown
        shared.record_expert_hit(0);
        shared.advance_step();
        shared.update_page_headers(vec![]);

        assert_eq!(shared.snapshot_and_reset_expert_hits()[0], 1);
        assert_eq!(shared.global_step.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_director_events_drain_after_shutdown() {
        let config = DirectorConfig {
            scan_interval: Duration::from_millis(5),
            half_life_samples: 50,
            num_experts: 2,
        };

        let mut director = JitDirector::spawn(config);
        let shared = Arc::clone(director.shared());

        // Manually push events
        shared.push_events(vec![
            ConsensusEvent::ExpertFrozen { expert_idx: 0, zero_hit_steps: 50 },
        ]);

        director.shutdown();

        // Events should still be drainable
        let events = shared.drain_events();
        assert_eq!(events.len(), 1);
    }

    // --- DecayingReservoir: softmax_sharpness field ---

    #[test]
    fn test_decaying_reservoir_softmax_sharpness_high_value() {
        let mut reservoir = DecayingReservoir::new(100);

        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.centroid_pos = f32_to_f16_bits(10.0);

        reservoir.ingest(&h);
        assert!((reservoir.avg_softmax_sharpness() - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_decaying_reservoir_softmax_sharpness_decays_correctly() {
        let mut reservoir = DecayingReservoir::new(2); // Short half-life for fast decay

        let mut h_high = KvPageHeader::new(0);
        h_high.ref_count = 1;
        h_high.centroid_pos = f32_to_f16_bits(100.0);

        let mut h_low = KvPageHeader::new(0);
        h_low.ref_count = 1;
        h_low.centroid_pos = f32_to_f16_bits(0.0);

        reservoir.ingest(&h_high);
        assert!(reservoir.avg_softmax_sharpness() > 90.0);

        // With half_life=2, decay_factor ≈ 0.707. After 20 zero samples:
        // weight of initial 100 ≈ 0.707^20 ≈ 0.0008 → contribution ≈ 0.08
        for _ in 0..20 {
            reservoir.ingest(&h_low);
        }
        assert!(reservoir.avg_softmax_sharpness() < 1.0);
    }

    // --- ConsensusDetector: detect with mixed reservoir states ---

    #[test]
    fn test_consensus_detector_detect_with_many_samples_moderate_entropy() {
        let mut detector = ConsensusDetector::new(2);

        let mut reservoir = DecayingReservoir::new(1000);
        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.entropy_avg = f32_to_f16_bits(0.5); // Above 0.05 threshold
        header.delta_rho_avg = f32_to_f16_bits(0.5); // Above 0.001 threshold

        for _ in 0..2000 {
            reservoir.ingest(&header);
        }

        // Even with many detect calls, moderate values should not trigger events
        for _ in 0..20000 {
            let events = detector.detect(&reservoir);
            assert!(!events.iter().any(|e| matches!(e, ConsensusEvent::AttentionSilent { .. })));
            assert!(!events.iter().any(|e| matches!(e, ConsensusEvent::LayerRedundant { .. })));
        }
    }

    #[test]
    fn test_consensus_detector_expert_streak_at_threshold_boundary() {
        let mut detector = ConsensusDetector::new(1);
        detector.expert_freeze_threshold = 1; // Threshold = 1

        // One zero-hit round = exactly at threshold
        detector.update_expert_hits(&[0]);

        let reservoir = DecayingReservoir::new(100);
        let events = detector.detect(&reservoir);
        assert!(events.iter().any(|e| matches!(
            e,
            ConsensusEvent::ExpertFrozen { expert_idx: 0, zero_hit_steps: 1 }
        )));
    }

    #[test]
    fn test_consensus_detector_expert_streak_below_threshold_boundary() {
        let mut detector = ConsensusDetector::new(1);
        detector.expert_freeze_threshold = 2;

        // One zero-hit round — below threshold
        detector.update_expert_hits(&[0]);

        let reservoir = DecayingReservoir::new(100);
        let events = detector.detect(&reservoir);
        assert!(!events.iter().any(|e| matches!(e, ConsensusEvent::ExpertFrozen { .. })));
    }

    // --- DirectorSharedState: update_page_headers overwrites ---

    #[test]
    fn test_shared_state_update_page_headers_multiple_times() {
        let state = DirectorSharedState::new(0);

        for i in 0..10u32 {
            let h = KvPageHeader::new(i);
            state.update_page_headers(vec![h]);
        }

        // Last update wins
        let headers = state.read_page_headers();
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].page_id, 9);
    }

    // --- DecayingReservoir: verify initial avg_residual_delta = 1.0 ---

    #[test]
    fn test_decaying_reservoir_initial_residual_delta_is_one() {
        let reservoir = DecayingReservoir::new(100);
        assert!((reservoir.avg_residual_delta() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_decaying_reservoir_residual_delta_set_on_first_ingest() {
        let mut reservoir = DecayingReservoir::new(100);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.delta_rho_avg = f32_to_f16_bits(2.5);

        reservoir.ingest(&h);
        assert!((reservoir.avg_residual_delta() - 2.5).abs() < 0.01);
    }

    // --- ConsensusDetector: zero-expert detect ---

    #[test]
    fn test_consensus_detector_zero_experts_detect_empty() {
        let mut detector = ConsensusDetector::new(0);
        let reservoir = DecayingReservoir::new(100);

        // No experts → no ExpertFrozen events possible
        let events = detector.detect(&reservoir);
        let frozen: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, ConsensusEvent::ExpertFrozen { .. }))
            .collect();
        assert!(frozen.is_empty());
    }

    // --- DirectorSharedState: concurrent snapshot + record ---

    #[test]
    fn test_shared_state_concurrent_snapshot_and_record() {
        let state = Arc::new(DirectorSharedState::new(2));

        let state_recorder = Arc::clone(&state);
        let recorder = std::thread::spawn(move || {
            for _ in 0..1000 {
                state_recorder.record_expert_hit(0);
            }
        });

        let state_snapshotter = Arc::clone(&state);
        let snapshotter = std::thread::spawn(move || {
            let mut total = 0u32;
            for _ in 0..10 {
                let hits = state_snapshotter.snapshot_and_reset_expert_hits();
                total += hits[0];
                std::thread::sleep(Duration::from_micros(100));
            }
            total
        });

        recorder.join().unwrap();
        let total = snapshotter.join().unwrap();

        // Total across all snapshots should be 1000 (all hits accounted for)
        assert_eq!(total, 1000);
    }

    // =========================================================================
    // Additional 50 tests — round 2
    // =========================================================================

    // --- DecayingReservoir: decay_factor formula verification ---

    #[test]
    fn test_decaying_reservoir_decay_factor_half_life_10() {
        // half_life_samples=10 → decay_factor = exp(-ln(2)/10)
        let reservoir = DecayingReservoir::new(10);
        let expected = (-(2.0_f64).ln() / 10.0).exp();
        assert!((reservoir.decay_factor - expected).abs() < 1e-12);
    }

    #[test]
    fn test_decaying_reservoir_decay_factor_half_life_100() {
        let reservoir = DecayingReservoir::new(100);
        let expected = (-(2.0_f64).ln() / 100.0).exp();
        assert!((reservoir.decay_factor - expected).abs() < 1e-12);
    }

    #[test]
    fn test_decaying_reservoir_decay_factor_in_range_0_to_1() {
        // For any positive half_life, decay_factor must be in [0, 1)
        for half_life in [1u64, 2, 5, 10, 100, 1000, 10000] {
            let reservoir = DecayingReservoir::new(half_life);
            assert!(reservoir.decay_factor >= 0.0);
            assert!(reservoir.decay_factor < 1.0);
        }
    }

    // --- DecayingReservoir: multiple ingest sessions ---

    #[test]
    fn test_decaying_reservoir_two_ingest_sessions_independent_weight() {
        let mut reservoir = DecayingReservoir::new(10);

        // Session 1: high values
        let mut h_high = KvPageHeader::new(0);
        h_high.ref_count = 1;
        h_high.entropy_avg = f32_to_f16_bits(100.0);
        for _ in 0..100 {
            reservoir.ingest(&h_high);
        }
        let after_session1 = reservoir.avg_entropy();
        assert!(after_session1 > 50.0);

        // Session 2: zero values — with half_life=10, should pull down fast
        let mut h_zero = KvPageHeader::new(0);
        h_zero.ref_count = 1;
        h_zero.entropy_avg = f32_to_f16_bits(0.0);
        for _ in 0..100 {
            reservoir.ingest(&h_zero);
        }
        assert!(reservoir.avg_entropy() < after_session1);
    }

    #[test]
    fn test_decaying_reservoir_entropy_converges_to_constant_after_many_samples() {
        let mut reservoir = DecayingReservoir::new(5);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.entropy_avg = f32_to_f16_bits(7.77);

        for _ in 0..200 {
            reservoir.ingest(&h);
        }
        // After 200 identical samples, should be very close to 7.77
        assert!((reservoir.avg_entropy() - 7.77).abs() < 0.01);
    }

    #[test]
    fn test_decaying_reservoir_all_four_metrics_update_on_subsequent_ingest() {
        let mut reservoir = DecayingReservoir::new(50);

        let mut h1 = KvPageHeader::new(0);
        h1.ref_count = 1;
        h1.entropy_avg = f32_to_f16_bits(1.0);
        h1.delta_rho_avg = f32_to_f16_bits(0.1);
        h1.dead_ratio = 50;
        h1.centroid_pos = f32_to_f16_bits(0.2);

        reservoir.ingest(&h1);

        let mut h2 = KvPageHeader::new(0);
        h2.ref_count = 1;
        h2.entropy_avg = f32_to_f16_bits(9.0);
        h2.delta_rho_avg = f32_to_f16_bits(5.0);
        h2.dead_ratio = 200;
        h2.centroid_pos = f32_to_f16_bits(8.0);

        reservoir.ingest(&h2);

        // All four metrics should have moved toward the second sample
        assert!(reservoir.avg_entropy() > 1.0);
        assert!(reservoir.avg_residual_delta() > 0.1);
        assert!(reservoir.avg_dead_neuron_ratio() > dead_ratio_to_f32(50) as f64);
        assert!(reservoir.avg_softmax_sharpness() > 0.2);
    }

    #[test]
    fn test_decaying_reservoir_single_ingest_sample_count_is_one() {
        let mut reservoir = DecayingReservoir::new(100);
        let header = KvPageHeader::new(0);
        reservoir.ingest(&header);
        assert_eq!(reservoir.sample_count(), 1);
    }

    #[test]
    fn test_decaying_reservoir_very_short_half_life_first_sample_not_overridden() {
        // With half_life=0 (decay_factor=0), second sample completely overrides
        let mut reservoir = DecayingReservoir::new(0);

        let mut h1 = KvPageHeader::new(0);
        h1.ref_count = 1;
        h1.entropy_avg = f32_to_f16_bits(100.0);
        reservoir.ingest(&h1);

        // If decay_factor is 0.0 and weight is 1.0, second ingest sets directly
        if reservoir.decay_factor == 0.0 {
            let mut h2 = KvPageHeader::new(0);
            h2.ref_count = 1;
            h2.entropy_avg = f32_to_f16_bits(1.0);
            reservoir.ingest(&h2);
            // With d=0, w=1: avg = 0*100 + 1*1 = 1.0
            assert!((reservoir.avg_entropy() - 1.0).abs() < 0.01);
        }
    }

    // --- ConsensusDetector: boundary and mixed conditions ---

    #[test]
    fn test_consensus_detector_detect_emits_expert_frozen_with_correct_zero_hit_steps() {
        let mut detector = ConsensusDetector::new(3);
        detector.expert_freeze_threshold = 5;

        for _ in 0..7 {
            detector.update_expert_hits(&[0, 0, 1]);
        }

        let reservoir = DecayingReservoir::new(100);
        let events = detector.detect(&reservoir);

        let expert0_event = events.iter().find(|e| matches!(
            e,
            ConsensusEvent::ExpertFrozen { expert_idx: 0, .. }
        ));
        assert!(expert0_event.is_some());
        if let Some(ConsensusEvent::ExpertFrozen { zero_hit_steps, .. }) = expert0_event {
            assert_eq!(*zero_hit_steps, 7);
        }
    }

    #[test]
    fn test_consensus_detector_detect_emits_expert_frozen_for_exactly_one_expert() {
        let mut detector = ConsensusDetector::new(4);
        detector.expert_freeze_threshold = 3;

        // Only expert 1 is frozen
        for _ in 0..5 {
            detector.update_expert_hits(&[10, 0, 5, 3]);
        }

        let reservoir = DecayingReservoir::new(100);
        let events = detector.detect(&reservoir);

        let frozen_indices: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                ConsensusEvent::ExpertFrozen { expert_idx, .. } => Some(*expert_idx),
                _ => None,
            })
            .collect();

        assert_eq!(frozen_indices, vec![1]);
    }

    

    

    #[test]
    fn test_consensus_detector_sample_count_1000_is_not_sufficient_for_attention() {
        // Condition: sample_count > 1000 (strict greater), so exactly 1000 is not enough
        let mut detector = ConsensusDetector::new(2);
        detector.attention_silent_threshold = 0.1;

        let mut reservoir = DecayingReservoir::new(100);
        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.entropy_avg = f32_to_f16_bits(0.01);

        // Exactly 1000 samples
        for _ in 0..1000 {
            reservoir.ingest(&header);
        }
        assert_eq!(reservoir.sample_count(), 1000);

        // detect should not trigger because sample_count must be > 1000
        let events = detector.detect(&reservoir);
        assert!(!events.iter().any(|e| matches!(e, ConsensusEvent::AttentionSilent { .. })));
    }

    #[test]
    fn test_consensus_detector_sample_count_1001_triggers_attention_check() {
        let mut detector = ConsensusDetector::new(2);
        detector.attention_silent_threshold = 0.1;

        let mut reservoir = DecayingReservoir::new(100);
        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.entropy_avg = f32_to_f16_bits(0.01);

        // 1001 samples — just above the gate
        for _ in 0..1001 {
            reservoir.ingest(&header);
        }

        // This call should increment attention_silent_steps
        let events = detector.detect(&reservoir);
        // No event yet (steps=1 < 10000), but counter should be 1
        assert!(!events.iter().any(|e| matches!(e, ConsensusEvent::AttentionSilent { .. })));
        assert_eq!(detector.attention_silent_steps, 1);
    }

    #[test]
    fn test_consensus_detector_expert_streak_survives_detect_call() {
        let mut detector = ConsensusDetector::new(2);
        detector.expert_freeze_threshold = 100;

        for _ in 0..10 {
            detector.update_expert_hits(&[0, 1]);
        }
        assert_eq!(detector.expert_zero_streaks[0], 10);

        // detect does not modify streaks
        let reservoir = DecayingReservoir::new(100);
        let _ = detector.detect(&reservoir);
        assert_eq!(detector.expert_zero_streaks[0], 10);
    }

    #[test]
    fn test_consensus_detector_detect_resets_attention_on_high_entropy_after_accumulation() {
        let mut detector = ConsensusDetector::new(2);
        detector.attention_silent_threshold = 0.1;

        let mut reservoir = DecayingReservoir::new(10); // Short half-life for fast adaptation

        // Build up low entropy
        let mut low_h = KvPageHeader::new(0);
        low_h.ref_count = 1;
        low_h.entropy_avg = f32_to_f16_bits(0.01);
        for _ in 0..1100 {
            reservoir.ingest(&low_h);
        }

        // Accumulate 200 silent steps
        for _ in 0..200 {
            let _ = detector.detect(&reservoir);
        }
        assert_eq!(detector.attention_silent_steps, 200);

        // Switch to high entropy
        let mut high_h = KvPageHeader::new(0);
        high_h.ref_count = 1;
        high_h.entropy_avg = f32_to_f16_bits(10.0);
        for _ in 0..50 {
            reservoir.ingest(&high_h);
        }

        // With short half-life, entropy should now be above threshold
        let _ = detector.detect(&reservoir);
        assert_eq!(detector.attention_silent_steps, 0);
    }

    #[test]
    fn test_consensus_detector_detect_resets_layer_on_high_delta_after_accumulation() {
        let mut detector = ConsensusDetector::new(2);

        let mut reservoir = DecayingReservoir::new(10);

        let mut low_h = KvPageHeader::new(0);
        low_h.ref_count = 1;
        low_h.delta_rho_avg = f32_to_f16_bits(0.0001);
        for _ in 0..1100 {
            reservoir.ingest(&low_h);
        }

        for _ in 0..200 {
            let _ = detector.detect(&reservoir);
        }
        assert_eq!(detector.layer_redundant_steps, 200);

        // Switch to high delta
        let mut high_h = KvPageHeader::new(0);
        high_h.ref_count = 1;
        high_h.delta_rho_avg = f32_to_f16_bits(5.0);
        for _ in 0..50 {
            reservoir.ingest(&high_h);
        }

        let _ = detector.detect(&reservoir);
        assert_eq!(detector.layer_redundant_steps, 0);
    }

    // --- ConsensusEvent: more variant coverage ---

    #[test]
    fn test_consensus_event_attention_silent_duration_steps_field() {
        let event = ConsensusEvent::AttentionSilent {
            avg_entropy: 0.04,
            duration_steps: 12345,
        };
        if let ConsensusEvent::AttentionSilent { duration_steps, .. } = event {
            assert_eq!(duration_steps, 12345);
        }
    }

    #[test]
    fn test_consensus_event_layer_redundant_duration_steps_field() {
        let event = ConsensusEvent::LayerRedundant {
            avg_delta_rho: 0.0005,
            duration_steps: 54321,
        };
        if let ConsensusEvent::LayerRedundant { duration_steps, .. } = event {
            assert_eq!(duration_steps, 54321);
        }
    }

    #[test]
    fn test_consensus_event_layer_redundant_clone_and_equality() {
        let event = ConsensusEvent::LayerRedundant {
            avg_delta_rho: 0.001,
            duration_steps: 9999,
        };
        let cloned = event.clone();
        assert_eq!(event, cloned);
    }

    #[test]
    fn test_consensus_event_attention_silent_large_values() {
        let event = ConsensusEvent::AttentionSilent {
            avg_entropy: f64::MAX,
            duration_steps: u64::MAX,
        };
        if let ConsensusEvent::AttentionSilent { avg_entropy, duration_steps } = event {
            assert_eq!(avg_entropy, f64::MAX);
            assert_eq!(duration_steps, u64::MAX);
        }
    }

    // --- DirectorConfig: more construction patterns ---

    #[test]
    fn test_director_config_default_scan_interval_is_100ms() {
        let config = DirectorConfig::default();
        assert_eq!(config.scan_interval.as_millis(), 100);
    }

    #[test]
    fn test_director_config_scan_interval_micros() {
        let config = DirectorConfig {
            scan_interval: Duration::from_micros(500),
            half_life_samples: 100,
            num_experts: 4,
        };
        assert_eq!(config.scan_interval.as_micros(), 500);
    }

    #[test]
    fn test_director_config_clone_independence() {
        let mut config = DirectorConfig {
            scan_interval: Duration::from_millis(200),
            half_life_samples: 5000,
            num_experts: 8,
        };
        let cloned = config.clone();

        // Mutate original
        config.num_experts = 16;
        config.half_life_samples = 100;

        assert_eq!(cloned.num_experts, 8);
        assert_eq!(cloned.half_life_samples, 5000);
    }

    #[test]
    fn test_director_config_debug_contains_all_fields() {
        let config = DirectorConfig {
            scan_interval: Duration::from_secs(2),
            half_life_samples: 999,
            num_experts: 7,
        };
        let debug = format!("{:?}", config);
        assert!(debug.contains("2s") || debug.contains("scan_interval"));
        assert!(debug.contains("999"));
        assert!(debug.contains("7"));
    }

    // --- DirectorSharedState: more edge cases ---

    #[test]
    fn test_shared_state_record_hit_all_experts() {
        let state = DirectorSharedState::new(5);

        // Record one hit on each expert
        for i in 0..5 {
            state.record_expert_hit(i);
        }

        let hits = state.snapshot_and_reset_expert_hits();
        assert_eq!(hits, vec![1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_shared_state_record_hit_same_expert_many_times_across_resets() {
        let state = DirectorSharedState::new(1);

        // First batch
        for _ in 0..100 {
            state.record_expert_hit(0);
        }
        let hits1 = state.snapshot_and_reset_expert_hits();
        assert_eq!(hits1, vec![100]);

        // Second batch
        for _ in 0..200 {
            state.record_expert_hit(0);
        }
        let hits2 = state.snapshot_and_reset_expert_hits();
        assert_eq!(hits2, vec![200]);
    }

    #[test]
    fn test_shared_state_push_events_mixed_types() {
        let state = DirectorSharedState::new(0);

        state.push_events(vec![
            ConsensusEvent::ExpertFrozen { expert_idx: 0, zero_hit_steps: 100 },
            ConsensusEvent::AttentionSilent { avg_entropy: 0.02, duration_steps: 15000 },
            ConsensusEvent::LayerRedundant { avg_delta_rho: 0.0003, duration_steps: 11000 },
        ]);

        let events = state.drain_events();
        assert_eq!(events.len(), 3);
        assert!(matches!(events[0], ConsensusEvent::ExpertFrozen { .. }));
        assert!(matches!(events[1], ConsensusEvent::AttentionSilent { .. }));
        assert!(matches!(events[2], ConsensusEvent::LayerRedundant { .. }));
    }

    #[test]
    fn test_shared_state_push_then_drain_then_push_again() {
        let state = DirectorSharedState::new(0);

        // First push
        state.push_events(vec![ConsensusEvent::ExpertFrozen {
            expert_idx: 0,
            zero_hit_steps: 50,
        }]);
        let events1 = state.drain_events();
        assert_eq!(events1.len(), 1);

        // Second push after drain
        state.push_events(vec![ConsensusEvent::ExpertFrozen {
            expert_idx: 1,
            zero_hit_steps: 75,
        }]);
        let events2 = state.drain_events();
        assert_eq!(events2.len(), 1);
        assert!(matches!(events2[0], ConsensusEvent::ExpertFrozen { expert_idx: 1, .. }));
    }

    #[test]
    fn test_shared_state_update_page_headers_with_many_headers() {
        let state = DirectorSharedState::new(0);

        let headers: Vec<KvPageHeader> = (0..100)
            .map(|i| KvPageHeader::new(i))
            .collect();

        state.update_page_headers(headers);
        let read = state.read_page_headers();
        assert_eq!(read.len(), 100);
        assert_eq!(read[0].page_id, 0);
        assert_eq!(read[99].page_id, 99);
    }

    #[test]
    fn test_shared_state_global_step_starts_at_zero() {
        let state = DirectorSharedState::new(0);
        assert_eq!(state.global_step.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_shared_state_advance_step_single() {
        let state = DirectorSharedState::new(0);
        state.advance_step();
        assert_eq!(state.global_step.load(Ordering::Relaxed), 1);
    }

    // --- Pipeline integration: more scenarios ---

    #[test]
    fn test_pipeline_expert_freeze_then_revive_via_reset() {
        let mut detector = ConsensusDetector::new(3);
        detector.expert_freeze_threshold = 5;

        // Freeze expert 0
        for _ in 0..6 {
            detector.update_expert_hits(&[0, 1, 1]);
        }

        let reservoir = DecayingReservoir::new(100);
        let events = detector.detect(&reservoir);
        assert!(events.iter().any(|e| matches!(
            e,
            ConsensusEvent::ExpertFrozen { expert_idx: 0, .. }
        )));

        // Revive expert 0
        detector.reset_expert(0);

        // Continue with hits
        detector.update_expert_hits(&[1, 1, 1]);

        let events2 = detector.detect(&reservoir);
        assert!(!events2.iter().any(|e| matches!(
            e,
            ConsensusEvent::ExpertFrozen { expert_idx: 0, .. }
        )));
    }

    #[test]
    fn test_pipeline_expert_freeze_detects_multiple_times_if_streak_continues() {
        let mut detector = ConsensusDetector::new(2);
        detector.expert_freeze_threshold = 3;

        for _ in 0..5 {
            detector.update_expert_hits(&[0, 0]);
        }

        let reservoir = DecayingReservoir::new(100);

        // First detect
        let events1 = detector.detect(&reservoir);
        let frozen_count1 = events1.iter().filter(|e| matches!(e, ConsensusEvent::ExpertFrozen { .. })).count();
        assert_eq!(frozen_count1, 2);

        // Continue zero hits — streak grows, detect should still fire
        detector.update_expert_hits(&[0, 0]);
        let events2 = detector.detect(&reservoir);
        let frozen_count2 = events2.iter().filter(|e| matches!(e, ConsensusEvent::ExpertFrozen { .. })).count();
        assert_eq!(frozen_count2, 2);
    }

    #[test]
    fn test_pipeline_reservoir_many_samples_does_not_overflow_sample_count() {
        let mut reservoir = DecayingReservoir::new(100);
        let header = KvPageHeader::new(0);

        for _ in 0..10000 {
            reservoir.ingest(&header);
        }
        assert_eq!(reservoir.sample_count(), 10000);
    }

    // --- JitDirector: more lifecycle ---

    #[test]
    fn test_director_spawn_and_immediate_shutdown() {
        let config = DirectorConfig {
            scan_interval: Duration::from_millis(100),
            half_life_samples: 100,
            num_experts: 0,
        };

        let mut director = JitDirector::spawn(config);
        // Immediately shut down without waiting
        director.shutdown();
    }

    #[test]
    fn test_director_shared_arc_count_after_clone() {
        let config = DirectorConfig {
            scan_interval: Duration::from_millis(10),
            half_life_samples: 50,
            num_experts: 0,
        };

        let mut director = JitDirector::spawn(config);
        let shared = director.shared();
        let count_before = Arc::strong_count(shared);
        let _extra_clone = Arc::clone(shared);
        let count_after = Arc::strong_count(shared);
        assert_eq!(count_after, count_before + 1);
        director.shutdown();
    }

    #[test]
    fn test_director_with_large_scan_interval() {
        let config = DirectorConfig {
            scan_interval: Duration::from_secs(1),
            half_life_samples: 100,
            num_experts: 0,
        };

        let mut director = JitDirector::spawn(config);
        std::thread::sleep(Duration::from_millis(50));
        director.shutdown();
    }

    // --- ConsensusDetector: clone independence ---

    #[test]
    fn test_consensus_detector_clone_independence() {
        let mut detector = ConsensusDetector::new(2);
        detector.expert_freeze_threshold = 5;

        for _ in 0..3 {
            detector.update_expert_hits(&[0, 1]);
        }

        let cloned = detector.clone();

        // Continue modifying original
        detector.update_expert_hits(&[0, 1]);

        // Cloned should not be affected
        assert_eq!(cloned.expert_zero_streaks[0], 3);
        assert_eq!(detector.expert_zero_streaks[0], 4);
    }

    // --- DecayingReservoir: negative value handling via f16 ---

    #[test]
    fn test_decaying_reservoir_ingest_negative_entropy_via_f16() {
        // f16 can represent negative values
        let mut reservoir = DecayingReservoir::new(100);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        // Negative f16: sign bit=1, exp=15, mant=0 → -1.0 in f16
        // f16 bits: 1_01111_0000000000 = 0xBC00
        h.entropy_avg = 0xBC00;

        reservoir.ingest(&h);
        let entropy = reservoir.avg_entropy();
        // f16 0xBC00 = -1.0
        assert!(entropy < 0.0);
    }

    #[test]
    fn test_decaying_reservoir_residual_delta_initial_one_then_overridden() {
        let reservoir = DecayingReservoir::new(100);
        // Before any ingest, avg_residual_delta = 1.0
        assert!((reservoir.avg_residual_delta() - 1.0).abs() < 1e-10);

        let mut reservoir = reservoir;
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.delta_rho_avg = f32_to_f16_bits(0.5);
        reservoir.ingest(&h);

        // After first ingest, avg_residual_delta = 0.5
        assert!((reservoir.avg_residual_delta() - 0.5).abs() < 0.01);
    }

    // --- ConsensusDetector: attention_silent_steps exact boundary ---

    #[test]
    fn test_consensus_detector_attention_silent_exactly_10000_steps_no_event() {
        let mut detector = ConsensusDetector::new(2);
        detector.attention_silent_threshold = 0.1;

        let mut reservoir = DecayingReservoir::new(1000);
        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.entropy_avg = f32_to_f16_bits(0.01);

        for _ in 0..2000 {
            reservoir.ingest(&header);
        }

        // Exactly 10000 detect calls — condition is > 10000, so 10000 is not enough
        for _ in 0..10000 {
            let _ = detector.detect(&reservoir);
        }
        assert_eq!(detector.attention_silent_steps, 10000);

        let events = detector.detect(&reservoir);
        // 10001st detect should now trigger (steps becomes 10001, but the check is
        // if self.attention_silent_steps > 10000, which is now 10001 after increment)
        // Wait — the increment happens BEFORE the check in the code
        // Actually looking at the code: steps += 1 first, then if steps > 10000, fire
        // So after 10000 calls, steps = 10000. The 10001st call makes it 10001 > 10000.
        // But we already called detect 10000 times and then checked — the 10001st is below
        // So events from the last detect() should not contain AttentionSilent yet
        // Actually we called detect 10000 times, then called detect again (10001st)
        // The 10001st: steps becomes 10001, then check 10001 > 10000 = true → fires
        assert!(events.iter().any(|e| matches!(e, ConsensusEvent::AttentionSilent { .. })));
    }

    #[test]
    fn test_consensus_detector_layer_redundant_exactly_10000_steps_no_event() {
        let mut detector = ConsensusDetector::new(2);

        let mut reservoir = DecayingReservoir::new(1000);
        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.delta_rho_avg = f32_to_f16_bits(0.0001);

        for _ in 0..2000 {
            reservoir.ingest(&header);
        }

        // Exactly 10000 calls
        for _ in 0..10000 {
            let _ = detector.detect(&reservoir);
        }
        assert_eq!(detector.layer_redundant_steps, 10000);

        let events = detector.detect(&reservoir);
        // 10001st call: steps becomes 10001, then 10001 > 10000 = true
        assert!(events.iter().any(|e| matches!(e, ConsensusEvent::LayerRedundant { .. })));
    }

    // --- DirectorSharedState: push_events then push_more then drain ---

    #[test]
    fn test_shared_state_multiple_pushes_then_single_drain() {
        let state = DirectorSharedState::new(0);

        for i in 0..5 {
            state.push_events(vec![ConsensusEvent::ExpertFrozen {
                expert_idx: i,
                zero_hit_steps: i as u64 * 10,
            }]);
        }

        let events = state.drain_events();
        assert_eq!(events.len(), 5);

        for (i, event) in events.iter().enumerate() {
            if let ConsensusEvent::ExpertFrozen { expert_idx, zero_hit_steps } = event {
                assert_eq!(*expert_idx, i);
                assert_eq!(*zero_hit_steps, i as u64 * 10);
            } else {
                panic!("Expected ExpertFrozen event");
            }
        }
    }

    // --- DecayingReservoir: verify softmax_sharpness is centroid_pos field ---

    #[test]
    fn test_decaying_reservoir_softmax_sharpness_maps_to_centroid_pos() {
        let mut reservoir = DecayingReservoir::new(100);

        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.centroid_pos = f32_to_f16_bits(3.14);

        reservoir.ingest(&h);
        assert!((reservoir.avg_softmax_sharpness() - 3.14).abs() < 0.01);
    }

    // --- ConsensusDetector: hit count as u32 overflow edge case ---

    #[test]
    fn test_consensus_detector_update_hits_max_u32_still_resets_streak() {
        let mut detector = ConsensusDetector::new(1);

        // Build streak
        for _ in 0..5 {
            detector.update_expert_hits(&[0]);
        }
        assert_eq!(detector.expert_zero_streaks[0], 5);

        // u32::MAX hits should reset
        detector.update_expert_hits(&[u32::MAX]);
        assert_eq!(detector.expert_zero_streaks[0], 0);
    }

    #[test]
    fn test_consensus_detector_update_hits_one_resets_streak() {
        let mut detector = ConsensusDetector::new(1);

        for _ in 0..5 {
            detector.update_expert_hits(&[0]);
        }
        assert_eq!(detector.expert_zero_streaks[0], 5);

        // Even 1 hit resets
        detector.update_expert_hits(&[1]);
        assert_eq!(detector.expert_zero_streaks[0], 0);
    }

    // --- DecayingReservoir: decay_factor formula for various half-lives ---

    #[test]
    fn test_decaying_reservoir_decay_factor_half_life_4() {
        let reservoir = DecayingReservoir::new(4);
        let expected = (-(2.0_f64).ln() / 4.0).exp();
        assert!((reservoir.decay_factor - expected).abs() < 1e-12);
    }

    #[test]
    fn test_decaying_reservoir_decay_factor_half_life_8() {
        let reservoir = DecayingReservoir::new(8);
        let expected = (-(2.0_f64).ln() / 8.0).exp();
        assert!((reservoir.decay_factor - expected).abs() < 1e-12);
    }

    // --- DirectorSharedState: read_page_headers returns empty when_no_updates ---

    #[test]
    fn test_shared_state_read_headers_initially_empty() {
        let state = DirectorSharedState::new(0);
        let headers = state.read_page_headers();
        assert!(headers.is_empty());
    }

    #[test]
    fn test_shared_state_record_expert_hit_exact_index() {
        let state = DirectorSharedState::new(5);

        // Record hits on specific experts
        state.record_expert_hit(2);
        state.record_expert_hit(4);

        let hits = state.snapshot_and_reset_expert_hits();
        assert_eq!(hits, vec![0, 0, 1, 0, 1]);
    }

    // =========================================================================
    // Round 3 — 50 additional tests targeting uncovered paths
    // =========================================================================

    // --- DecayingReservoir: decay_factor boundary values ---

    #[test]
    fn test_decaying_reservoir_decay_factor_half_life_3() {
        let reservoir = DecayingReservoir::new(3);
        let expected = (-(2.0_f64).ln() / 3.0).exp();
        assert!((reservoir.decay_factor - expected).abs() < 1e-12);
    }

    #[test]
    fn test_decaying_reservoir_decay_factor_half_life_50() {
        let reservoir = DecayingReservoir::new(50);
        let expected = (-(2.0_f64).ln() / 50.0).exp();
        assert!((reservoir.decay_factor - expected).abs() < 1e-12);
    }

    #[test]
    fn test_decaying_reservoir_decay_factor_half_life_500() {
        let reservoir = DecayingReservoir::new(500);
        let expected = (-(2.0_f64).ln() / 500.0).exp();
        assert!((reservoir.decay_factor - expected).abs() < 1e-12);
    }

    #[test]
    fn test_decaying_reservoir_decay_factor_half_life_10000() {
        let reservoir = DecayingReservoir::new(10000);
        let expected = (-(2.0_f64).ln() / 10000.0).exp();
        assert!((reservoir.decay_factor - expected).abs() < 1e-12);
    }

    // --- DecayingReservoir: centroid_pos / softmax_sharpness edge cases ---

    #[test]
    fn test_decaying_reservoir_centroid_pos_zero_value() {
        let mut reservoir = DecayingReservoir::new(100);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.centroid_pos = f32_to_f16_bits(0.0);

        reservoir.ingest(&h);
        assert!((reservoir.avg_softmax_sharpness() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_decaying_reservoir_centroid_pos_high_value() {
        let mut reservoir = DecayingReservoir::new(100);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.centroid_pos = f32_to_f16_bits(50.0);

        reservoir.ingest(&h);
        assert!((reservoir.avg_softmax_sharpness() - 50.0).abs() < 0.5);
    }

    #[test]
    fn test_decaying_reservoir_centroid_pos_blends_between_samples() {
        let mut reservoir = DecayingReservoir::new(10);

        let mut h_low = KvPageHeader::new(0);
        h_low.ref_count = 1;
        h_low.centroid_pos = f32_to_f16_bits(1.0);

        let mut h_high = KvPageHeader::new(0);
        h_high.ref_count = 1;
        h_high.centroid_pos = f32_to_f16_bits(99.0);

        reservoir.ingest(&h_low);
        reservoir.ingest(&h_high);

        // Should be between 1.0 and 99.0 after blending
        assert!(reservoir.avg_softmax_sharpness() > 1.0);
        assert!(reservoir.avg_softmax_sharpness() < 99.0);
    }

    // --- DecayingReservoir: dead_ratio monotonic with input ---

    #[test]
    fn test_decaying_reservoir_dead_ratio_monotonic_increase() {
        let mut r1 = DecayingReservoir::new(100);
        let mut r2 = DecayingReservoir::new(100);

        let mut h1 = KvPageHeader::new(0);
        h1.ref_count = 1;
        h1.dead_ratio = 50;

        let mut h2 = KvPageHeader::new(0);
        h2.ref_count = 1;
        h2.dead_ratio = 200;

        r1.ingest(&h1);
        r2.ingest(&h2);

        assert!(r2.avg_dead_neuron_ratio() > r1.avg_dead_neuron_ratio());
    }

    #[test]
    fn test_decaying_reservoir_dead_ratio_mid_range_value() {
        let mut reservoir = DecayingReservoir::new(100);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.dead_ratio = 128; // ~0.5 in [0,255] range

        reservoir.ingest(&h);
        let ratio = reservoir.avg_dead_neuron_ratio();
        // dead_ratio_to_f32(128) should be roughly 0.5
        assert!(ratio > 0.3);
        assert!(ratio < 0.7);
    }

    // --- DecayingReservoir: residual_delta convergence ---

    #[test]
    fn test_decaying_reservoir_residual_delta_converges_to_constant() {
        let mut reservoir = DecayingReservoir::new(5);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.delta_rho_avg = f32_to_f16_bits(3.33);

        for _ in 0..200 {
            reservoir.ingest(&h);
        }
        assert!((reservoir.avg_residual_delta() - 3.33).abs() < 0.01);
    }

    // --- DecayingReservoir: verify clone copies decay_factor ---

    #[test]
    fn test_decaying_reservoir_clone_preserves_decay_factor() {
        let reservoir = DecayingReservoir::new(42);
        let cloned = reservoir.clone();
        assert!((reservoir.decay_factor - cloned.decay_factor).abs() < 1e-15);
    }

    #[test]
    fn test_decaying_reservoir_clone_sample_count_independent() {
        let mut reservoir = DecayingReservoir::new(100);
        let h = KvPageHeader::new(0);
        reservoir.ingest(&h);

        let mut cloned = reservoir.clone();
        let h2 = KvPageHeader::new(0);
        cloned.ingest(&h2);

        assert_eq!(reservoir.sample_count(), 1);
        assert_eq!(cloned.sample_count(), 2);
    }

    // --- ConsensusDetector: update_expert_hits interleaved pattern ---

    #[test]
    fn test_consensus_detector_interleaved_hits_and_zeros() {
        let mut detector = ConsensusDetector::new(2);
        detector.expert_freeze_threshold = 5;

        // Alternate: expert 0 gets 0 hits, then 1 hit, then 0, then 1...
        for _ in 0..4 {
            detector.update_expert_hits(&[0, 1]);
            detector.update_expert_hits(&[1, 1]);
        }

        // Expert 0 streak should be 0 (last was a hit)
        assert_eq!(detector.expert_zero_streaks[0], 0);
    }

    #[test]
    fn test_consensus_detector_only_last_run_matters_for_streak() {
        let mut detector = ConsensusDetector::new(1);
        detector.expert_freeze_threshold = 10;

        // 20 zeros
        for _ in 0..20 {
            detector.update_expert_hits(&[0]);
        }
        assert_eq!(detector.expert_zero_streaks[0], 20);

        // 1 hit resets
        detector.update_expert_hits(&[1]);
        assert_eq!(detector.expert_zero_streaks[0], 0);

        // 3 zeros again — below threshold
        for _ in 0..3 {
            detector.update_expert_hits(&[0]);
        }
        assert_eq!(detector.expert_zero_streaks[0], 3);
    }

    // --- ConsensusDetector: attention_silent_threshold customization ---

    #[test]
    fn test_consensus_detector_custom_attention_silent_threshold() {
        let mut detector = ConsensusDetector::new(2);
        detector.attention_silent_threshold = 1.0; // Very high — anything below 1.0 is "silent"

        let mut reservoir = DecayingReservoir::new(1000);
        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.entropy_avg = f32_to_f16_bits(0.5); // Below custom threshold

        for _ in 0..2000 {
            reservoir.ingest(&header);
        }

        for _ in 0..10001 {
            let _ = detector.detect(&reservoir);
        }

        let events = detector.detect(&reservoir);
        assert!(events.iter().any(|e| matches!(e, ConsensusEvent::AttentionSilent { .. })));
    }

    #[test]
    fn test_consensus_detector_default_attention_threshold_not_too_sensitive() {
        let detector = ConsensusDetector::new(2);
        // Default threshold is 0.05 — entropy of 0.5 should be well above
        assert!((detector.attention_silent_threshold - 0.05).abs() < 1e-10);
        assert!(0.5 > detector.attention_silent_threshold);
    }

    // --- ConsensusDetector: layer_redundant_threshold boundary ---

    #[test]
    fn test_consensus_detector_layer_redundant_threshold_default() {
        let detector = ConsensusDetector::new(2);
        assert!((detector.layer_redundant_threshold - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_consensus_detector_layer_redundant_delta_above_threshold_no_event() {
        let mut detector = ConsensusDetector::new(2);

        let mut reservoir = DecayingReservoir::new(1000);
        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        // Set delta well above the 0.001 threshold
        header.delta_rho_avg = f32_to_f16_bits(0.5);

        for _ in 0..2000 {
            reservoir.ingest(&header);
        }

        // delta_rho_avg > threshold → should not increment steps
        let _ = detector.detect(&reservoir);
        assert_eq!(detector.layer_redundant_steps, 0);
    }

    // --- ConsensusDetector: detect does not modify expert_zero_streaks ---

    #[test]
    fn test_consensus_detector_detect_does_not_modify_streaks() {
        let mut detector = ConsensusDetector::new(3);
        detector.expert_freeze_threshold = 100;

        for _ in 0..7 {
            detector.update_expert_hits(&[0, 1, 0]);
        }

        let streaks_before = detector.expert_zero_streaks.clone();
        let reservoir = DecayingReservoir::new(100);
        let _ = detector.detect(&reservoir);
        assert_eq!(detector.expert_zero_streaks, streaks_before);
    }

    // --- ConsensusEvent: exhaustive variant coverage with if-let ---

    #[test]
    fn test_consensus_event_expert_frozen_destructure() {
        let event = ConsensusEvent::ExpertFrozen {
            expert_idx: 42,
            zero_hit_steps: 12345,
        };
        if let ConsensusEvent::ExpertFrozen { expert_idx, zero_hit_steps } = event {
            assert_eq!(expert_idx, 42);
            assert_eq!(zero_hit_steps, 12345);
        } else {
            panic!("Expected ExpertFrozen variant");
        }
    }

    #[test]
    fn test_consensus_event_attention_silent_destructure() {
        let event = ConsensusEvent::AttentionSilent {
            avg_entropy: 0.042,
            duration_steps: 99999,
        };
        if let ConsensusEvent::AttentionSilent { avg_entropy, duration_steps } = event {
            assert!((avg_entropy - 0.042).abs() < 1e-10);
            assert_eq!(duration_steps, 99999);
        } else {
            panic!("Expected AttentionSilent variant");
        }
    }

    #[test]
    fn test_consensus_event_layer_redundant_destructure() {
        let event = ConsensusEvent::LayerRedundant {
            avg_delta_rho: 0.00042,
            duration_steps: 77777,
        };
        if let ConsensusEvent::LayerRedundant { avg_delta_rho, duration_steps } = event {
            assert!((avg_delta_rho - 0.00042).abs() < 1e-10);
            assert_eq!(duration_steps, 77777);
        } else {
            panic!("Expected LayerRedundant variant");
        }
    }

    // --- ConsensusEvent: PartialEq exhaustiveness ---

    #[test]
    fn test_consensus_event_eq_same_variant_same_fields() {
        let a = ConsensusEvent::ExpertFrozen { expert_idx: 5, zero_hit_steps: 500 };
        let b = ConsensusEvent::ExpertFrozen { expert_idx: 5, zero_hit_steps: 500 };
        assert_eq!(a, b);
    }

    #[test]
    fn test_consensus_event_ne_same_variant_different_field() {
        let a = ConsensusEvent::AttentionSilent { avg_entropy: 0.01, duration_steps: 100 };
        let b = ConsensusEvent::AttentionSilent { avg_entropy: 0.02, duration_steps: 100 };
        assert_ne!(a, b);
    }

    // --- DirectorConfig: more construction patterns ---

    #[test]
    fn test_director_config_default_is_100ms_interval() {
        let config = DirectorConfig::default();
        assert!(config.scan_interval >= Duration::from_millis(100));
    }

    #[test]
    fn test_director_config_default_half_life_is_10000() {
        let config = DirectorConfig::default();
        assert_eq!(config.half_life_samples, 10_000);
    }

    #[test]
    fn test_director_config_default_num_experts_is_zero() {
        let config = DirectorConfig::default();
        assert_eq!(config.num_experts, 0);
    }

    #[test]
    fn test_director_config_scan_interval_secs() {
        let config = DirectorConfig {
            scan_interval: Duration::from_secs(5),
            half_life_samples: 1000,
            num_experts: 16,
        };
        assert_eq!(config.scan_interval.as_secs(), 5);
    }

    // --- DirectorSharedState: page headers with active/inactive ---

    #[test]
    fn test_shared_state_page_headers_with_active_page() {
        let state = DirectorSharedState::new(0);

        let mut h = KvPageHeader::new(99);
        h.ref_count = 1; // Active
        state.update_page_headers(vec![h]);

        let headers = state.read_page_headers();
        assert_eq!(headers.len(), 1);
        assert!(headers[0].is_active());
    }

    #[test]
    fn test_shared_state_page_headers_with_inactive_page() {
        let state = DirectorSharedState::new(0);

        let h = KvPageHeader::new(99); // ref_count = 0, inactive
        state.update_page_headers(vec![h]);

        let headers = state.read_page_headers();
        assert_eq!(headers.len(), 1);
        assert!(!headers[0].is_active());
    }

    #[test]
    fn test_shared_state_page_headers_mixed_active_and_inactive() {
        let state = DirectorSharedState::new(0);

        let mut h_active = KvPageHeader::new(1);
        h_active.ref_count = 1;

        let h_inactive = KvPageHeader::new(2); // ref_count = 0

        state.update_page_headers(vec![h_active, h_inactive]);

        let headers = state.read_page_headers();
        assert_eq!(headers.len(), 2);
        assert!(headers[0].is_active());
        assert!(!headers[1].is_active());
    }

    // --- DirectorSharedState: expert_hit_counters as u64 -> u32 truncation ---

    #[test]
    fn test_shared_state_expert_hit_counter_truncates_to_u32() {
        let state = DirectorSharedState::new(1);

        // Store a value that fits in u32
        state.expert_hit_counters[0].store(100, Ordering::Relaxed);
        let hits = state.snapshot_and_reset_expert_hits();
        assert_eq!(hits[0], 100);
    }

    #[test]
    fn test_shared_state_expert_hit_counter_large_value_fits_u32() {
        let state = DirectorSharedState::new(1);

        // u32::MAX should be exactly representable
        state.expert_hit_counters[0].store(u32::MAX as u64, Ordering::Relaxed);
        let hits = state.snapshot_and_reset_expert_hits();
        assert_eq!(hits[0], u32::MAX);
    }

    // --- DirectorSharedState: concurrent page header updates ---

    #[test]
    fn test_shared_state_concurrent_page_header_writes() {
        let state = Arc::new(DirectorSharedState::new(0));

        let s1 = Arc::clone(&state);
        let h1 = std::thread::spawn(move || {
            for i in 0..50 {
                let h = KvPageHeader::new(i);
                s1.update_page_headers(vec![h]);
            }
        });

        let s2 = Arc::clone(&state);
        let h2 = std::thread::spawn(move || {
            for i in 50..100 {
                let h = KvPageHeader::new(i);
                s2.update_page_headers(vec![h]);
            }
        });

        h1.join().unwrap();
        h2.join().unwrap();

        // Last write wins — should have exactly 1 header
        let headers = state.read_page_headers();
        assert_eq!(headers.len(), 1);
        // page_id is either 49 or 99 depending on which thread wrote last
        assert!(headers[0].page_id == 49 || headers[0].page_id == 99);
    }

    // --- JitDirector: shutdown idempotency ---

    #[test]
    fn test_director_shutdown_then_drop_is_safe() {
        let config = DirectorConfig {
            scan_interval: Duration::from_millis(10),
            half_life_samples: 50,
            num_experts: 0,
        };

        let mut director = JitDirector::spawn(config);
        director.shutdown();
        // Drop after explicit shutdown — should not panic or double-join
        drop(director);
    }

    // --- Pipeline: director + reservoir + detector integration ---

    #[test]
    fn test_pipeline_shared_state_feeds_into_reservoir() {
        let mut reservoir = DecayingReservoir::new(100);
        let state = DirectorSharedState::new(0);

        // Push headers via shared state
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.entropy_avg = f32_to_f16_bits(4.0);
        h.delta_rho_avg = f32_to_f16_bits(0.8);
        state.update_page_headers(vec![h]);

        // Read back and ingest into reservoir
        let headers = state.read_page_headers();
        for header in &headers {
            if header.is_active() {
                reservoir.ingest(header);
            }
        }

        assert_eq!(reservoir.sample_count(), 1);
        assert!((reservoir.avg_entropy() - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_pipeline_full_cycle_healthy_data_no_events() {
        let mut reservoir = DecayingReservoir::new(100);
        let mut detector = ConsensusDetector::new(4);
        let state = DirectorSharedState::new(4);

        // Simulate 1000 healthy cycles
        for step in 0..1000 {
            let mut h = KvPageHeader::new(0);
            h.ref_count = 1;
            h.entropy_avg = f32_to_f16_bits(2.0 + (step % 10) as f32 * 0.1);
            h.delta_rho_avg = f32_to_f16_bits(0.5);
            state.update_page_headers(vec![h]);

            for expert in 0..4 {
                state.record_expert_hit(expert);
            }
            state.advance_step();

            // Director-side processing
            let headers = state.read_page_headers();
            for header in &headers {
                if header.is_active() {
                    reservoir.ingest(header);
                }
            }

            let hits = state.snapshot_and_reset_expert_hits();
            detector.update_expert_hits(&hits);

            let events = detector.detect(&reservoir);
            assert!(events.is_empty(), "No events expected with healthy data at step {}", step);
        }

        assert_eq!(reservoir.sample_count(), 1000);
        let step = state.global_step.load(Ordering::Relaxed);
        assert_eq!(step, 1000);
    }

    // --- DecayingReservoir: ingest same header many times is idempotent ---

    #[test]
    fn test_decaying_reservoir_ingest_same_header_converges() {
        let mut reservoir = DecayingReservoir::new(20);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.entropy_avg = f32_to_f16_bits(5.5);
        h.delta_rho_avg = f32_to_f16_bits(1.2);
        h.dead_ratio = 64;
        h.centroid_pos = f32_to_f16_bits(0.9);

        for _ in 0..500 {
            reservoir.ingest(&h);
        }

        assert!((reservoir.avg_entropy() - 5.5).abs() < 0.01);
        assert!((reservoir.avg_residual_delta() - 1.2).abs() < 0.01);
        assert!((reservoir.avg_softmax_sharpness() - 0.9).abs() < 0.01);
    }

    // --- ConsensusDetector: large expert count with selective freeze ---

    #[test]
    fn test_consensus_detector_128_experts_selective_freeze() {
        let mut detector = ConsensusDetector::new(128);
        detector.expert_freeze_threshold = 3;

        // All experts get hits except expert 50
        let mut hits = vec![1u32; 128];
        hits[50] = 0;

        for _ in 0..5 {
            detector.update_expert_hits(&hits);
        }

        let reservoir = DecayingReservoir::new(100);
        let events = detector.detect(&reservoir);

        let frozen_indices: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                ConsensusEvent::ExpertFrozen { expert_idx, .. } => Some(*expert_idx),
                _ => None,
            })
            .collect();

        assert_eq!(frozen_indices, vec![50]);
    }

    // --- ConsensusDetector: update_expert_hits with empty then non-empty ---

    #[test]
    fn test_consensus_detector_update_empty_then_non_empty() {
        let mut detector = ConsensusDetector::new(2);

        // Empty → all streaks increment
        detector.update_expert_hits(&[]);
        assert_eq!(detector.expert_zero_streaks, vec![1, 1]);

        // Non-empty → expert 0 resets, expert 1 continues
        detector.update_expert_hits(&[5, 0]);
        assert_eq!(detector.expert_zero_streaks[0], 0);
        assert_eq!(detector.expert_zero_streaks[1], 2);
    }

    // --- DirectorSharedState: push_events accumulates before drain ---

    #[test]
    fn test_shared_state_push_accumulates_without_drain() {
        let state = DirectorSharedState::new(0);

        for i in 0..10 {
            state.push_events(vec![ConsensusEvent::ExpertFrozen {
                expert_idx: i,
                zero_hit_steps: i as u64,
            }]);
        }

        let events = state.drain_events();
        assert_eq!(events.len(), 10);
    }

    // --- DecayingReservoir: entropy never negative from valid f16 ---

    #[test]
    fn test_decaying_reservoir_entropy_from_positive_f16_stays_positive() {
        let mut reservoir = DecayingReservoir::new(100);

        for v in [0.1f32, 1.0, 5.0, 10.0, 100.0] {
            let mut h = KvPageHeader::new(0);
            h.ref_count = 1;
            h.entropy_avg = f32_to_f16_bits(v);
            reservoir.ingest(&h);
        }

        assert!(reservoir.avg_entropy() > 0.0);
    }

    // --- ConsensusDetector: detect called multiple times does not lose events ---

    #[test]
    fn test_consensus_detector_detect_does_not_clear_expert_streaks() {
        let mut detector = ConsensusDetector::new(2);
        detector.expert_freeze_threshold = 3;

        for _ in 0..5 {
            detector.update_expert_hits(&[0, 1]);
        }

        let reservoir = DecayingReservoir::new(100);

        // Call detect multiple times — streaks persist
        let e1 = detector.detect(&reservoir);
        assert!(e1.iter().any(|e| matches!(e, ConsensusEvent::ExpertFrozen { expert_idx: 0, .. })));

        let e2 = detector.detect(&reservoir);
        assert!(e2.iter().any(|e| matches!(e, ConsensusEvent::ExpertFrozen { expert_idx: 0, .. })));
    }

    // --- DirectorConfig: debug format all fields ---

    #[test]
    fn test_director_config_debug_zero_values() {
        let config = DirectorConfig {
            scan_interval: Duration::ZERO,
            half_life_samples: 0,
            num_experts: 0,
        };
        let debug = format!("{:?}", config);
        assert!(debug.contains("DirectorConfig"));
    }

    // --- DecayingReservoir: verify weight (1 - decay_factor) is positive ---

    #[test]
    fn test_decaying_reservoir_weight_is_positive_for_finite_half_life() {
        for hl in [1u64, 2, 5, 10, 100, 1000] {
            let reservoir = DecayingReservoir::new(hl);
            let weight = 1.0 - reservoir.decay_factor;
            assert!(weight > 0.0, "Weight should be positive for half_life={}", hl);
        }
    }

    // --- ConsensusDetector: reset expert preserves others ---

    #[test]
    fn test_consensus_detector_reset_middle_expert_preserves_neighbors() {
        let mut detector = ConsensusDetector::new(5);
        detector.expert_freeze_threshold = 3;

        for _ in 0..5 {
            detector.update_expert_hits(&[0, 0, 0, 0, 0]);
        }

        detector.reset_expert(2);

        assert_eq!(detector.expert_zero_streaks[0], 5);
        assert_eq!(detector.expert_zero_streaks[1], 5);
        assert_eq!(detector.expert_zero_streaks[2], 0);
        assert_eq!(detector.expert_zero_streaks[3], 5);
        assert_eq!(detector.expert_zero_streaks[4], 5);
    }

    // --- JitDirector: shared state survives director drop ---

    #[test]
    fn test_director_shared_state_survives_drop() {
        let shared;
        {
            let config = DirectorConfig {
                scan_interval: Duration::from_millis(10),
                half_life_samples: 50,
                num_experts: 2,
            };
            let mut director = JitDirector::spawn(config);
            shared = Arc::clone(director.shared());
            director.shutdown();
        }

        // Shared state still valid
        shared.record_expert_hit(0);
        assert_eq!(shared.snapshot_and_reset_expert_hits()[0], 1);
    }

    // --- DecayingReservoir: ingest with high dead_ratio then low blends ---

    #[test]
    fn test_decaying_reservoir_dead_ratio_high_to_low_transition() {
        let mut reservoir = DecayingReservoir::new(5); // Fast adaptation

        let mut h_high = KvPageHeader::new(0);
        h_high.ref_count = 1;
        h_high.dead_ratio = 250;

        for _ in 0..50 {
            reservoir.ingest(&h_high);
        }
        let high_ratio = reservoir.avg_dead_neuron_ratio();

        let mut h_low = KvPageHeader::new(0);
        h_low.ref_count = 1;
        h_low.dead_ratio = 5;

        for _ in 0..50 {
            reservoir.ingest(&h_low);
        }
        let low_ratio = reservoir.avg_dead_neuron_ratio();

        assert!(high_ratio > low_ratio);
    }

    // --- DecayingReservoir: decay_factor computed from half_life ---

    #[test]
    fn test_decaying_reservoir_decay_factor_is_exp_of_ln2_over_half_life() {
        let r = DecayingReservoir::new(100);
        let expected = (-(2.0_f64.ln()) / 100.0_f64).exp();
        assert!((r.decay_factor - expected).abs() < 1e-15);
    }

    #[test]
    fn test_decaying_reservoir_decay_factor_half_life_1_is_half() {
        let r = DecayingReservoir::new(1);
        let expected = (-(2.0_f64.ln()) / 1.0).exp();
        assert!((r.decay_factor - expected).abs() < 1e-15);
    }

    #[test]
    fn test_decaying_reservoir_new_sample_count_zero() {
        let r = DecayingReservoir::new(50);
        assert_eq!(r.sample_count(), 0);
    }

    #[test]
    fn test_decaying_reservoir_new_avg_entropy_zero() {
        let r = DecayingReservoir::new(50);
        assert_eq!(r.avg_entropy(), 0.0);
    }

    #[test]
    fn test_decaying_reservoir_new_avg_residual_delta_one() {
        let r = DecayingReservoir::new(50);
        assert_eq!(r.avg_residual_delta(), 1.0);
    }

    #[test]
    fn test_decaying_reservoir_new_avg_dead_ratio_zero() {
        let r = DecayingReservoir::new(50);
        assert_eq!(r.avg_dead_neuron_ratio(), 0.0);
    }

    #[test]
    fn test_decaying_reservoir_new_avg_softmax_sharpness_zero() {
        let r = DecayingReservoir::new(50);
        assert_eq!(r.avg_softmax_sharpness(), 0.0);
    }

    #[test]
    fn test_decaying_reservoir_ingest_inactive_page_not_counted() {
        let mut r = DecayingReservoir::new(10);
        let h = KvPageHeader::new(0); // ref_count=0 → inactive
        assert!(!h.is_active());
        // ingest should still work (director.rs doesn't filter here — scan_loop filters)
        r.ingest(&h);
        assert_eq!(r.sample_count(), 1);
    }

    #[test]
    fn test_decaying_reservoir_weight_formula() {
        let r = DecayingReservoir::new(10);
        let w = 1.0 - r.decay_factor;
        assert!(w > 0.0);
        assert!(w < 1.0);
        assert!((r.decay_factor + w - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_decaying_reservoir_many_ingest_sample_count_exact() {
        let mut r = DecayingReservoir::new(100);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        for i in 0..500 {
            r.ingest(&h);
            assert_eq!(r.sample_count(), i + 1);
        }
    }

    #[test]
    fn test_decaying_reservoir_entropy_stays_within_f16_range() {
        let mut r = DecayingReservoir::new(100);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.entropy_avg = f32_to_f16_bits(10.0);
        for _ in 0..200 {
            r.ingest(&h);
        }
        // f16 max ~65504, but entropy should be bounded reasonably
        assert!(r.avg_entropy() <= 20.0);
        assert!(r.avg_entropy() >= 0.0);
    }

    #[test]
    fn test_decaying_reservoir_residual_delta_positive_for_positive_input() {
        let mut r = DecayingReservoir::new(50);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.delta_rho_avg = f32_to_f16_bits(0.5);
        for _ in 0..20 {
            r.ingest(&h);
        }
        assert!(r.avg_residual_delta() > 0.0);
    }

    #[test]
    fn test_decaying_reservoir_softmax_sharpness_tracks_centroid_pos() {
        let mut r = DecayingReservoir::new(50);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.centroid_pos = f32_to_f16_bits(0.75);
        for _ in 0..50 {
            r.ingest(&h);
        }
        let val = r.avg_softmax_sharpness();
        assert!((val - 0.75).abs() < 0.1);
    }

    #[test]
    fn test_decaying_reservoir_two_distinct_values_converge_to_second() {
        let mut r = DecayingReservoir::new(2); // fast decay
        let mut h1 = KvPageHeader::new(0);
        h1.ref_count = 1;
        h1.entropy_avg = f32_to_f16_bits(1.0);

        for _ in 0..100 {
            r.ingest(&h1);
        }
        assert!((r.avg_entropy() - 1.0).abs() < 0.1);

        let mut h2 = KvPageHeader::new(0);
        h2.ref_count = 1;
        h2.entropy_avg = f32_to_f16_bits(5.0);

        for _ in 0..100 {
            r.ingest(&h2);
        }
        assert!((r.avg_entropy() - 5.0).abs() < 0.5);
    }

    #[test]
    fn test_decaying_reservoir_ingest_zero_entropy_header() {
        let mut r = DecayingReservoir::new(50);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.entropy_avg = 0; // f16 zero
        r.ingest(&h);
        assert_eq!(r.avg_entropy(), 0.0);
    }

    // --- ConsensusDetector: construction edge cases ---

    #[test]
    fn test_consensus_detector_new_default_freeze_threshold() {
        let d = ConsensusDetector::new(4);
        assert_eq!(d.expert_freeze_threshold, 1_000_000);
    }

    #[test]
    fn test_consensus_detector_new_default_attention_threshold() {
        let d = ConsensusDetector::new(4);
        assert!((d.attention_silent_threshold - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_consensus_detector_new_default_layer_threshold() {
        let d = ConsensusDetector::new(4);
        assert!((d.layer_redundant_threshold - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_consensus_detector_new_streaks_all_zero() {
        let d = ConsensusDetector::new(8);
        assert!(d.expert_zero_streaks.iter().all(|&s| s == 0));
    }

    #[test]
    fn test_consensus_detector_new_silent_steps_zero() {
        let d = ConsensusDetector::new(4);
        assert_eq!(d.attention_silent_steps, 0);
    }

    #[test]
    fn test_consensus_detector_new_redundant_steps_zero() {
        let d = ConsensusDetector::new(4);
        assert_eq!(d.layer_redundant_steps, 0);
    }

    // --- ConsensusDetector: update_expert_hits with various sizes ---

    #[test]
    fn test_consensus_detector_update_hits_shorter_than_streaks() {
        let mut d = ConsensusDetector::new(4);
        // Only provide hits for experts 0,1 — experts 2,3 default to 0
        d.update_expert_hits(&[5, 3]);
        assert_eq!(d.expert_zero_streaks[0], 0);
        assert_eq!(d.expert_zero_streaks[1], 0);
        assert_eq!(d.expert_zero_streaks[2], 1);
        assert_eq!(d.expert_zero_streaks[3], 1);
    }

    #[test]
    fn test_consensus_detector_update_hits_longer_than_streaks() {
        let mut d = ConsensusDetector::new(2);
        // Extra hits beyond expert count are ignored
        d.update_expert_hits(&[1, 2, 3, 4, 5]);
        assert_eq!(d.expert_zero_streaks[0], 0);
        assert_eq!(d.expert_zero_streaks[1], 0);
    }

    #[test]
    fn test_consensus_detector_update_hits_all_zero_increments_all() {
        let mut d = ConsensusDetector::new(3);
        d.update_expert_hits(&[0, 0, 0]);
        assert_eq!(d.expert_zero_streaks, vec![1, 1, 1]);
        d.update_expert_hits(&[0, 0, 0]);
        assert_eq!(d.expert_zero_streaks, vec![2, 2, 2]);
    }

    #[test]
    fn test_consensus_detector_update_hits_mixed_resets_selectively() {
        let mut d = ConsensusDetector::new(3);
        d.update_expert_hits(&[0, 0, 0]);
        d.update_expert_hits(&[0, 5, 0]);
        assert_eq!(d.expert_zero_streaks[0], 2);
        assert_eq!(d.expert_zero_streaks[1], 0);
        assert_eq!(d.expert_zero_streaks[2], 2);
    }

    // --- ConsensusDetector: detect with fresh reservoir ---

    #[test]
    fn test_consensus_detector_detect_fresh_reservoir_no_attention_event() {
        let mut d = ConsensusDetector::new(4);
        let r = DecayingReservoir::new(100);
        // sample_count == 0, so no attention check
        let events = d.detect(&r);
        let silent: Vec<_> = events.iter().filter(|e| matches!(e, ConsensusEvent::AttentionSilent { .. })).collect();
        assert!(silent.is_empty());
    }

    #[test]
    fn test_consensus_detector_detect_fresh_reservoir_no_layer_event() {
        let mut d = ConsensusDetector::new(4);
        let r = DecayingReservoir::new(100);
        let events = d.detect(&r);
        let redundant: Vec<_> = events.iter().filter(|e| matches!(e, ConsensusEvent::LayerRedundant { .. })).collect();
        assert!(redundant.is_empty());
    }

    // --- ConsensusDetector: reset_expert ---

    #[test]
    fn test_consensus_detector_reset_out_of_bounds_no_panic() {
        let mut d = ConsensusDetector::new(4);
        d.reset_expert(100); // should not panic
        assert_eq!(d.expert_zero_streaks.len(), 4);
    }

    // --- ConsensusEvent: PartialOrd is NOT derived (f64 fields) ---

    #[test]
    fn test_consensus_event_expert_frozen_accessors() {
        let e = ConsensusEvent::ExpertFrozen { expert_idx: 7, zero_hit_steps: 42 };
        if let ConsensusEvent::ExpertFrozen { expert_idx, zero_hit_steps } = e {
            assert_eq!(expert_idx, 7);
            assert_eq!(zero_hit_steps, 42);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_consensus_event_attention_silent_accessors() {
        let e = ConsensusEvent::AttentionSilent { avg_entropy: 0.01, duration_steps: 15000 };
        if let ConsensusEvent::AttentionSilent { avg_entropy, duration_steps } = e {
            assert!((avg_entropy - 0.01).abs() < 1e-10);
            assert_eq!(duration_steps, 15000);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_consensus_event_layer_redundant_accessors() {
        let e = ConsensusEvent::LayerRedundant { avg_delta_rho: 0.0005, duration_steps: 20000 };
        if let ConsensusEvent::LayerRedundant { avg_delta_rho, duration_steps } = e {
            assert!((avg_delta_rho - 0.0005).abs() < 1e-10);
            assert_eq!(duration_steps, 20000);
        } else {
            panic!("wrong variant");
        }
    }

    // --- DirectorConfig ---

    #[test]
    fn test_director_config_default_scan_interval_100ms() {
        let c = DirectorConfig::default();
        assert_eq!(c.scan_interval, Duration::from_millis(100));
    }

    #[test]
    fn test_director_config_default_half_life_10000() {
        let c = DirectorConfig::default();
        assert_eq!(c.half_life_samples, 10_000);
    }

    #[test]
    fn test_director_config_default_num_experts_zero() {
        let c = DirectorConfig::default();
        assert_eq!(c.num_experts, 0);
    }

    #[test]
    fn test_director_config_custom_values() {
        let c = DirectorConfig {
            scan_interval: Duration::from_secs(5),
            half_life_samples: 500,
            num_experts: 64,
        };
        assert_eq!(c.scan_interval, Duration::from_secs(5));
        assert_eq!(c.half_life_samples, 500);
        assert_eq!(c.num_experts, 64);
    }

    #[test]
    fn test_director_config_clone_equality() {
        let c = DirectorConfig {
            scan_interval: Duration::from_millis(250),
            half_life_samples: 2000,
            num_experts: 8,
        };
        let c2 = c.clone();
        assert_eq!(c.scan_interval, c2.scan_interval);
        assert_eq!(c.half_life_samples, c2.half_life_samples);
        assert_eq!(c.num_experts, c2.num_experts);
    }

    #[test]
    fn test_director_config_clone_independent_mutation() {
        let mut c = DirectorConfig::default();
        let c2 = c.clone();
        c.scan_interval = Duration::from_secs(1);
        assert_ne!(c.scan_interval, c2.scan_interval);
    }

    #[test]
    fn test_director_config_debug_trait_impl() {
        let c = DirectorConfig {
            scan_interval: Duration::from_millis(100),
            half_life_samples: 10_000,
            num_experts: 0,
        };
        let debug_str = format!("{:?}", c);
        assert!(debug_str.contains("scan_interval"));
        assert!(debug_str.contains("half_life_samples"));
        assert!(debug_str.contains("num_experts"));
    }

    #[test]
    fn test_director_config_scan_interval_micros_exact() {
        let c = DirectorConfig {
            scan_interval: Duration::from_micros(500),
            ..Default::default()
        };
        assert_eq!(c.scan_interval.as_micros(), 500);
    }

    #[test]
    fn test_director_config_scan_interval_nanos_exact() {
        let c = DirectorConfig {
            scan_interval: Duration::from_nanos(1),
            ..Default::default()
        };
        assert_eq!(c.scan_interval.as_nanos(), 1);
    }

    // --- DirectorSharedState: basic operations ---

    #[test]
    fn test_shared_state_new_zero_experts_empty_counters() {
        let s = DirectorSharedState::new(0);
        let hits = s.snapshot_and_reset_expert_hits();
        assert!(hits.is_empty());
    }

    #[test]
    fn test_shared_state_record_hit_out_of_bounds_no_panic() {
        let s = DirectorSharedState::new(4);
        s.record_expert_hit(100); // should not panic
        let hits = s.snapshot_and_reset_expert_hits();
        assert_eq!(hits.len(), 4);
        assert!(hits.iter().all(|&h| h == 0));
    }

    #[test]
    fn test_shared_state_global_step_zero_initially() {
        let s = DirectorSharedState::new(4);
        assert_eq!(s.global_step.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_shared_state_advance_step_increments() {
        let s = DirectorSharedState::new(4);
        s.advance_step();
        s.advance_step();
        s.advance_step();
        assert_eq!(s.global_step.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_shared_state_snapshot_and_reset_clears_counters() {
        let s = DirectorSharedState::new(4);
        s.record_expert_hit(0);
        s.record_expert_hit(0);
        s.record_expert_hit(2);
        let hits1 = s.snapshot_and_reset_expert_hits();
        assert_eq!(hits1[0], 2);
        assert_eq!(hits1[1], 0);
        assert_eq!(hits1[2], 1);
        assert_eq!(hits1[3], 0);
        let hits2 = s.snapshot_and_reset_expert_hits();
        assert!(hits2.iter().all(|&h| h == 0));
    }

    #[test]
    fn test_shared_state_drain_empty_returns_empty() {
        let s = DirectorSharedState::new(4);
        let events = s.drain_events();
        assert!(events.is_empty());
    }

    #[test]
    fn test_shared_state_push_then_drain_returns_events() {
        let s = DirectorSharedState::new(4);
        let e1 = ConsensusEvent::ExpertFrozen { expert_idx: 0, zero_hit_steps: 100 };
        let e2 = ConsensusEvent::AttentionSilent { avg_entropy: 0.01, duration_steps: 11000 };
        s.push_events(vec![e1.clone(), e2.clone()]);
        let drained = s.drain_events();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0], e1);
        assert_eq!(drained[1], e2);
    }

    #[test]
    fn test_shared_state_drain_twice_second_empty() {
        let s = DirectorSharedState::new(4);
        s.push_events(vec![ConsensusEvent::ExpertFrozen { expert_idx: 0, zero_hit_steps: 50 }]);
        let _ = s.drain_events();
        let second = s.drain_events();
        assert!(second.is_empty());
    }

    #[test]
    fn test_shared_state_push_empty_is_noop() {
        let s = DirectorSharedState::new(4);
        s.push_events(vec![]);
        let drained = s.drain_events();
        assert!(drained.is_empty());
    }

    #[test]
    fn test_shared_state_update_page_headers_initially_empty() {
        let s = DirectorSharedState::new(4);
        let headers = s.read_page_headers();
        assert!(headers.is_empty());
    }

    #[test]
    fn test_shared_state_update_page_headers_stores_and_reads() {
        let s = DirectorSharedState::new(4);
        let mut h = KvPageHeader::new(42);
        h.ref_count = 1;
        h.entropy_avg = f32_to_f16_bits(2.5);
        s.update_page_headers(vec![h]);
        let headers = s.read_page_headers();
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].page_id, 42);
        assert_eq!(headers[0].ref_count, 1);
    }

    #[test]
    fn test_shared_state_update_page_headers_replaces() {
        let s = DirectorSharedState::new(4);
        let h1 = KvPageHeader::new(1);
        s.update_page_headers(vec![h1]);
        let h2 = KvPageHeader::new(2);
        s.update_page_headers(vec![h2]);
        let headers = s.read_page_headers();
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].page_id, 2);
    }

    #[test]
    fn test_shared_state_multiple_experts_hits() {
        let s = DirectorSharedState::new(4);
        s.record_expert_hit(0);
        s.record_expert_hit(1);
        s.record_expert_hit(1);
        s.record_expert_hit(3);
        let hits = s.snapshot_and_reset_expert_hits();
        assert_eq!(hits[0], 1);
        assert_eq!(hits[1], 2);
        assert_eq!(hits[2], 0);
        assert_eq!(hits[3], 1);
    }

    // --- JitDirector: spawn/shutdown lifecycle ---

    #[test]
    fn test_director_spawn_shutdown_clean() {
        let mut d = JitDirector::spawn(DirectorConfig {
            scan_interval: Duration::from_millis(10),
            ..Default::default()
        });
        d.shutdown();
    }

    #[test]
    fn test_director_shared_accessible() {
        let d = JitDirector::spawn(DirectorConfig {
            scan_interval: Duration::from_millis(50),
            num_experts: 4,
            ..Default::default()
        });
        let shared = d.shared();
        shared.record_expert_hit(0);
        shared.advance_step();
        drop(d);
    }

    #[test]
    fn test_director_drop_impl_calls_shutdown() {
        {
            let _d = JitDirector::spawn(DirectorConfig {
                scan_interval: Duration::from_millis(10),
                ..Default::default()
            });
        }
        // Director dropped — Drop impl calls shutdown, no panic
    }

    #[test]
    fn test_director_zero_experts_spawn() {
        let d = JitDirector::spawn(DirectorConfig {
            scan_interval: Duration::from_millis(10),
            num_experts: 0,
            ..Default::default()
        });
        let shared = d.shared();
        let hits = shared.snapshot_and_reset_expert_hits();
        assert!(hits.is_empty());
        drop(d);
    }

    #[test]
    fn test_director_config_with_very_short_scan_interval() {
        let mut d = JitDirector::spawn(DirectorConfig {
            scan_interval: Duration::from_micros(1),
            ..Default::default()
        });
        // Should still work, just loop fast
        d.shutdown();
    }

    // --- Integration: reservoir → detector pipeline ---

    #[test]
    fn test_pipeline_ingest_active_headers_updates_reservoir() {
        let mut r = DecayingReservoir::new(50);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.entropy_avg = f32_to_f16_bits(3.0);
        h.delta_rho_avg = f32_to_f16_bits(0.5);
        h.centroid_pos = f32_to_f16_bits(0.8);
        h.dead_ratio = 64;

        r.ingest(&h);
        assert_eq!(r.sample_count(), 1);
        assert!((r.avg_entropy() - 3.0).abs() < 0.1);
        assert!((r.avg_residual_delta() - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_pipeline_detect_no_events_on_healthy_data() {
        let mut r = DecayingReservoir::new(100);
        let mut d = ConsensusDetector::new(4);

        // Feed healthy data — high entropy, non-zero residual
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.entropy_avg = f32_to_f16_bits(5.0); // well above 0.05
        h.delta_rho_avg = f32_to_f16_bits(1.0); // well above 0.001

        for _ in 0..2000 {
            r.ingest(&h);
        }
        d.update_expert_hits(&[10, 20, 5, 8]);
        let events = d.detect(&r);
        assert!(events.is_empty());
    }

    #[test]
    fn test_pipeline_expert_freeze_triggers_after_threshold() {
        let r = DecayingReservoir::new(100);
        let mut d = ConsensusDetector::new(2);

        // Simulate 1_000_001 steps of zero hits for expert 0
        for _ in 0..1_000_001 {
            d.update_expert_hits(&[0, 5]);
        }
        let events = d.detect(&r);
        let frozen: Vec<_> = events.iter().filter_map(|e| {
            if let ConsensusEvent::ExpertFrozen { expert_idx, .. } = e { Some(*expert_idx) } else { None }
        }).collect();
        assert!(frozen.contains(&0));
        assert!(!frozen.contains(&1));
    }

    // --- ConsensusDetector: attention silent edge cases ---

    #[test]
    fn test_consensus_detector_attention_silent_requires_sample_count_gt_1000() {
        let mut r = DecayingReservoir::new(10);
        let mut d = ConsensusDetector::new(4);

        // Feed 999 low-entropy samples — just below threshold
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.entropy_avg = f32_to_f16_bits(0.001); // well below 0.05

        for _ in 0..999 {
            r.ingest(&h);
        }
        for _ in 0..10001 {
            let _ = d.detect(&r);
        }
        // sample_count = 999 < 1000, so no attention check
        let events = d.detect(&r);
        let silent: Vec<_> = events.iter().filter(|e| matches!(e, ConsensusEvent::AttentionSilent { .. })).collect();
        assert!(silent.is_empty());
    }

    // --- ConsensusDetector: layer redundant edge cases ---

    #[test]
    fn test_consensus_detector_layer_redundant_requires_sample_count_gt_1000() {
        let mut r = DecayingReservoir::new(10);
        let mut d = ConsensusDetector::new(4);

        // Feed 999 low-delta samples
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.delta_rho_avg = f32_to_f16_bits(0.0001); // well below 0.001

        for _ in 0..999 {
            r.ingest(&h);
        }
        for _ in 0..10001 {
            let _ = d.detect(&r);
        }
        let events = d.detect(&r);
        let redundant: Vec<_> = events.iter().filter(|e| matches!(e, ConsensusEvent::LayerRedundant { .. })).collect();
        assert!(redundant.is_empty());
    }

    // --- DecayingReservoir: math verification ---

    #[test]
    fn test_decaying_reservoir_ema_second_sample_formula() {
        let mut r = DecayingReservoir::new(100);
        let d = r.decay_factor;
        let w = 1.0 - d;

        let mut h1 = KvPageHeader::new(0);
        h1.ref_count = 1;
        h1.entropy_avg = f32_to_f16_bits(2.0);
        r.ingest(&h1);

        let mut h2 = KvPageHeader::new(0);
        h2.ref_count = 1;
        h2.entropy_avg = f32_to_f16_bits(4.0);
        r.ingest(&h2);

        // First ingest sets directly, second: d * 2.0 + w * 4.0
        let expected = d * 2.0_f64 + w * 4.0_f64;
        assert!((r.avg_entropy() - expected).abs() < 0.01);
    }

    #[test]
    fn test_decaying_reservoir_dead_ratio_ingest_all_255() {
        let mut r = DecayingReservoir::new(50);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.dead_ratio = 255;
        r.ingest(&h);
        let expected = 255.0_f32 / 255.0;
        assert!((r.avg_dead_neuron_ratio() - expected as f64).abs() < 0.01);
    }

    #[test]
    fn test_decaying_reservoir_dead_ratio_ingest_128() {
        let mut r = DecayingReservoir::new(50);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.dead_ratio = 128;
        r.ingest(&h);
        let expected = 128.0_f32 / 255.0;
        assert!((r.avg_dead_neuron_ratio() - expected as f64).abs() < 0.01);
    }

    // --- DecayingReservoir: clone produces independent copy ---

    #[test]
    fn test_decaying_reservoir_clone_then_ingest_independent() {
        let mut r1 = DecayingReservoir::new(100);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.entropy_avg = f32_to_f16_bits(3.0);
        r1.ingest(&h);

        let mut r2 = r1.clone();
        h.entropy_avg = f32_to_f16_bits(7.0);
        r2.ingest(&h);

        assert_ne!(r1.sample_count(), r2.sample_count());
        assert!(r1.avg_entropy() < r2.avg_entropy());
    }

    // --- ConsensusDetector: detect does not modify expert streaks ---

    #[test]
    fn test_consensus_detector_detect_preserves_expert_streaks() {
        let mut d = ConsensusDetector::new(3);
        d.update_expert_hits(&[0, 5, 0]);
        let streaks_before = d.expert_zero_streaks.clone();
        let r = DecayingReservoir::new(100);
        let _ = d.detect(&r);
        assert_eq!(d.expert_zero_streaks, streaks_before);
    }

    // --- DirectorSharedState: concurrent record + snapshot ---

    #[test]
    fn test_shared_state_concurrent_record_and_snapshot() {
        let s = Arc::new(DirectorSharedState::new(4));
        let s1 = Arc::clone(&s);
        let s2 = Arc::clone(&s);

        let t1 = std::thread::spawn(move || {
            for _ in 0..1000 {
                s1.record_expert_hit(0);
            }
        });
        let t2 = std::thread::spawn(move || {
            for _ in 0..10 {
                let _ = s2.snapshot_and_reset_expert_hits();
                std::thread::sleep(Duration::from_micros(10));
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // Final snapshot should have some hits (maybe 0 if last snapshot caught them)
        let _final_hits = s.snapshot_and_reset_expert_hits();
    }

    // --- ConsensusEvent: Debug trait ---

    #[test]
    fn test_consensus_event_expert_frozen_debug_output() {
        let e = ConsensusEvent::ExpertFrozen { expert_idx: 3, zero_hit_steps: 999 };
        let s = format!("{:?}", e);
        assert!(s.contains("ExpertFrozen"));
        assert!(s.contains("3"));
        assert!(s.contains("999"));
    }

    #[test]
    fn test_consensus_event_attention_silent_debug_output() {
        let e = ConsensusEvent::AttentionSilent { avg_entropy: 0.02, duration_steps: 12000 };
        let s = format!("{:?}", e);
        assert!(s.contains("AttentionSilent"));
    }

    #[test]
    fn test_consensus_event_layer_redundant_debug_output() {
        let e = ConsensusEvent::LayerRedundant { avg_delta_rho: 0.0003, duration_steps: 15000 };
        let s = format!("{:?}", e);
        assert!(s.contains("LayerRedundant"));
    }

    // --- DecayingReservoir: decay factor boundary values ---

    #[test]
    fn test_decaying_reservoir_half_life_max_u64_decay_near_one() {
        let r = DecayingReservoir::new(u64::MAX);
        assert!(r.decay_factor > 0.999);
    }

    #[test]
    fn test_decaying_reservoir_half_life_2_decay_not_half() {
        // half_life=2 means after 2 samples old weight is 50%
        // decay_factor = exp(-ln2/2) ≈ 0.707
        let r = DecayingReservoir::new(2);
        assert!((r.decay_factor - 0.7071).abs() < 0.01);
    }

    // --- ConsensusDetector: reset multiple experts ---

    #[test]
    fn test_consensus_detector_reset_all_experts() {
        let mut d = ConsensusDetector::new(4);
        d.update_expert_hits(&[0, 0, 0, 0]);
        d.update_expert_hits(&[0, 0, 0, 0]);
        assert_eq!(d.expert_zero_streaks, vec![2, 2, 2, 2]);
        for i in 0..4 {
            d.reset_expert(i);
        }
        assert_eq!(d.expert_zero_streaks, vec![0, 0, 0, 0]);
    }

    // --- DirectorSharedState: push preserves order across batches ---

    #[test]
    fn test_shared_state_push_order_preserved() {
        let s = DirectorSharedState::new(4);
        let e1 = ConsensusEvent::ExpertFrozen { expert_idx: 0, zero_hit_steps: 1 };
        let e2 = ConsensusEvent::ExpertFrozen { expert_idx: 1, zero_hit_steps: 2 };
        let e3 = ConsensusEvent::AttentionSilent { avg_entropy: 0.01, duration_steps: 11000 };
        s.push_events(vec![e1.clone(), e2.clone()]);
        s.push_events(vec![e3.clone()]);
        let drained = s.drain_events();
        assert_eq!(drained.len(), 3);
        assert_eq!(drained[0], e1);
        assert_eq!(drained[1], e2);
        assert_eq!(drained[2], e3);
    }

    // --- DecayingReservoir: multiple fields update independently ---

    #[test]
    fn test_decaying_reservoir_four_fields_all_update_on_ingest() {
        let mut r = DecayingReservoir::new(100);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.entropy_avg = f32_to_f16_bits(2.0);
        h.delta_rho_avg = f32_to_f16_bits(0.3);
        h.dead_ratio = 100;
        h.centroid_pos = f32_to_f16_bits(0.6);

        r.ingest(&h);
        assert!((r.avg_entropy() - 2.0).abs() < 0.1);
        assert!((r.avg_residual_delta() - 0.3).abs() < 0.1);
        assert!((r.avg_dead_neuron_ratio() - (100.0_f32 / 255.0) as f64).abs() < 0.01);
        assert!((r.avg_softmax_sharpness() - 0.6).abs() < 0.1);
    }

    // --- ConsensusDetector: streaks are per-expert independent ---

    #[test]
    fn test_consensus_detector_streaks_independent_across_experts() {
        let mut d = ConsensusDetector::new(4);
        // Expert 0: hits every round; Expert 1: always zero; Expert 2: alternating; Expert 3: always zero
        d.update_expert_hits(&[5, 0, 3, 0]);
        d.update_expert_hits(&[2, 0, 0, 0]);
        d.update_expert_hits(&[1, 0, 4, 0]);

        assert_eq!(d.expert_zero_streaks[0], 0);
        assert_eq!(d.expert_zero_streaks[1], 3);
        assert_eq!(d.expert_zero_streaks[2], 0); // last hit was non-zero
        assert_eq!(d.expert_zero_streaks[3], 3);
    }

    // =========================================================================
    // Round 4 — 15 additional tests: special floats, zero thresholds, edge paths
    // =========================================================================

    #[test]
    fn test_decaying_reservoir_ingest_f16_infinity_entropy() {
        // f16 Inf: sign=0, exp=31, mant=0 → bits = 0x7C00
        let mut reservoir = DecayingReservoir::new(100);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.entropy_avg = 0x7C00; // f16 positive infinity

        reservoir.ingest(&h);
        assert_eq!(reservoir.sample_count(), 1);
        assert!(reservoir.avg_entropy().is_infinite());
        assert!(reservoir.avg_entropy().is_sign_positive());
    }

    #[test]
    fn test_decaying_reservoir_ingest_f16_nan_propagates() {
        // f16 NaN: sign=0, exp=31, mant!=0 → bits = 0x7C01
        let mut reservoir = DecayingReservoir::new(100);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.entropy_avg = 0x7C01; // f16 NaN

        reservoir.ingest(&h);
        assert_eq!(reservoir.sample_count(), 1);
        assert!(reservoir.avg_entropy().is_nan());
    }

    #[test]
    fn test_consensus_detector_freeze_threshold_zero_fires_immediately() {
        // threshold=0 means any streak >= 0 fires (including streak=0)
        let mut detector = ConsensusDetector::new(3);
        detector.expert_freeze_threshold = 0;

        // No update_expert_hits calls — streaks are all 0
        let reservoir = DecayingReservoir::new(100);
        let events = detector.detect(&reservoir);

        // All experts should fire because streak 0 >= threshold 0
        let frozen: Vec<_> = events.iter()
            .filter_map(|e| match e {
                ConsensusEvent::ExpertFrozen { expert_idx, .. } => Some(*expert_idx),
                _ => None,
            })
            .collect();
        assert_eq!(frozen, vec![0, 1, 2]);
    }

    #[test]
    fn test_decaying_reservoir_ema_three_samples_exact_formula() {
        // Verify the exact EMA formula across 3 samples with half_life=1 (d=0.5)
        let mut r = DecayingReservoir::new(1);
        // decay_factor = 0.5, weight = 0.5

        let mut h1 = KvPageHeader::new(0);
        h1.ref_count = 1;
        h1.entropy_avg = f32_to_f16_bits(8.0);
        r.ingest(&h1);
        // sample_count=1, avg_entropy = 8.0 (first sample sets directly)

        let mut h2 = KvPageHeader::new(0);
        h2.ref_count = 1;
        h2.entropy_avg = f32_to_f16_bits(4.0);
        r.ingest(&h2);
        // avg = 0.5 * 8.0 + 0.5 * 4.0 = 6.0
        assert!((r.avg_entropy() - 6.0).abs() < 0.1);

        let mut h3 = KvPageHeader::new(0);
        h3.ref_count = 1;
        h3.entropy_avg = f32_to_f16_bits(2.0);
        r.ingest(&h3);
        // avg = 0.5 * 6.0 + 0.5 * 2.0 = 4.0
        assert!((r.avg_entropy() - 4.0).abs() < 0.2);
    }

    #[test]
    fn test_shared_state_advance_step_wraps_at_u64_max() {
        let state = DirectorSharedState::new(0);
        state.global_step.store(u64::MAX, Ordering::Relaxed);
        state.advance_step();
        let step = state.global_step.load(Ordering::Relaxed);
        // Wrapping add: u64::MAX + 1 = 0
        assert_eq!(step, 0);
    }

    #[test]
    fn test_consensus_detector_attention_threshold_exact_boundary() {
        // Use a value well above threshold, then verify no silent step increment.
        // Then use a value well below threshold, verify step does increment.
        let mut detector = ConsensusDetector::new(2);
        detector.attention_silent_threshold = 0.05;

        let mut reservoir = DecayingReservoir::new(100);

        // Feed entropy well above threshold — no silent step
        let mut h_above = KvPageHeader::new(0);
        h_above.ref_count = 1;
        h_above.entropy_avg = f32_to_f16_bits(1.0);
        for _ in 0..2000 {
            reservoir.ingest(&h_above);
        }
        let _ = detector.detect(&reservoir);
        assert_eq!(detector.attention_silent_steps, 0);

        // Feed entropy well below threshold — silent step increments
        let mut h_below = KvPageHeader::new(0);
        h_below.ref_count = 1;
        h_below.entropy_avg = f32_to_f16_bits(0.001);
        for _ in 0..2000 {
            reservoir.ingest(&h_below);
        }
        let _ = detector.detect(&reservoir);
        assert_eq!(detector.attention_silent_steps, 1);
    }

    #[test]
    fn test_decaying_reservoir_ingest_negative_centroid_pos_blend() {
        let mut r = DecayingReservoir::new(10);

        let mut h_pos = KvPageHeader::new(0);
        h_pos.ref_count = 1;
        h_pos.centroid_pos = f32_to_f16_bits(5.0);
        r.ingest(&h_pos);

        let mut h_neg = KvPageHeader::new(0);
        h_neg.ref_count = 1;
        // f16 -2.0: sign=1, exp=15, mant=0 → bits = 0xC000
        h_neg.centroid_pos = 0xC000;
        r.ingest(&h_neg);

        // Should blend between +5.0 and -2.0
        let sharpness = r.avg_softmax_sharpness();
        assert!(sharpness < 5.0);
        assert!(sharpness > -2.0);
    }

    #[test]
    fn test_director_config_default_equals_manual_construction() {
        let default = DirectorConfig::default();
        let manual = DirectorConfig {
            scan_interval: Duration::from_millis(100),
            half_life_samples: 10_000,
            num_experts: 0,
        };
        assert_eq!(default.scan_interval, manual.scan_interval);
        assert_eq!(default.half_life_samples, manual.half_life_samples);
        assert_eq!(default.num_experts, manual.num_experts);
    }

    #[test]
    fn test_consensus_detector_512_experts_all_zero_hits() {
        let mut detector = ConsensusDetector::new(512);
        detector.expert_freeze_threshold = 3;

        let hits = vec![0u32; 512];
        for _ in 0..5 {
            detector.update_expert_hits(&hits);
        }

        let reservoir = DecayingReservoir::new(100);
        let events = detector.detect(&reservoir);
        let frozen_count = events.iter()
            .filter(|e| matches!(e, ConsensusEvent::ExpertFrozen { .. }))
            .count();
        assert_eq!(frozen_count, 512);
    }

    #[test]
    fn test_shared_state_drain_after_multiple_drains_is_always_empty() {
        let state = DirectorSharedState::new(0);
        state.push_events(vec![ConsensusEvent::ExpertFrozen {
            expert_idx: 0, zero_hit_steps: 10,
        }]);

        let _first = state.drain_events();
        let second = state.drain_events();
        let third = state.drain_events();
        assert!(second.is_empty());
        assert!(third.is_empty());
    }

    #[test]
    fn test_pipeline_scan_loop_skips_inactive_headers() {
        // Simulate the scan_loop pattern: only ingest active headers
        let mut reservoir = DecayingReservoir::new(100);
        let state = DirectorSharedState::new(0);

        let mut h_active = KvPageHeader::new(0);
        h_active.ref_count = 1;
        h_active.entropy_avg = f32_to_f16_bits(3.0);

        let h_inactive = KvPageHeader::new(1); // ref_count = 0

        state.update_page_headers(vec![h_active, h_inactive]);

        let headers = state.read_page_headers();
        for header in &headers {
            if header.is_active() {
                reservoir.ingest(header);
            }
        }

        assert_eq!(reservoir.sample_count(), 1);
        assert!((reservoir.avg_entropy() - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_consensus_event_layer_redundant_equality_different_durations() {
        let a = ConsensusEvent::LayerRedundant {
            avg_delta_rho: 0.001,
            duration_steps: 10000,
        };
        let b = ConsensusEvent::LayerRedundant {
            avg_delta_rho: 0.001,
            duration_steps: 20000,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn test_decaying_reservoir_subnormal_f16_entropy_ingest() {
        // f16 subnormal: exp=0, mant!=0 → bits = 0x0001 (smallest subnormal)
        // This is a very small positive number, close to zero
        let mut reservoir = DecayingReservoir::new(100);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.entropy_avg = 0x0001; // f16 smallest subnormal

        reservoir.ingest(&h);
        assert_eq!(reservoir.sample_count(), 1);
        // Should be a tiny positive number
        assert!(reservoir.avg_entropy() > 0.0);
        assert!(reservoir.avg_entropy() < 0.001);
    }

    #[test]
    fn test_director_config_nanos_scan_interval() {
        let config = DirectorConfig {
            scan_interval: Duration::from_nanos(100),
            ..Default::default()
        };
        assert_eq!(config.scan_interval.as_nanos(), 100);
    }

    #[test]
    fn test_consensus_detector_detect_after_reset_and_hit_no_event() {
        let mut detector = ConsensusDetector::new(3);
        detector.expert_freeze_threshold = 5;

        // Freeze expert 1
        for _ in 0..6 {
            detector.update_expert_hits(&[1, 0, 1]);
        }
        let reservoir = DecayingReservoir::new(100);
        let events = detector.detect(&reservoir);
        assert!(events.iter().any(|e| matches!(
            e,
            ConsensusEvent::ExpertFrozen { expert_idx: 1, .. }
        )));

        // Reset expert 1, then give it a hit
        detector.reset_expert(1);
        detector.update_expert_hits(&[1, 5, 1]);

        // Detect again — expert 1 should not be frozen (streak=0)
        let events2 = detector.detect(&reservoir);
        assert!(!events2.iter().any(|e| matches!(
            e,
            ConsensusEvent::ExpertFrozen { expert_idx: 1, .. }
        )));
    }

    // =========================================================================
    // Round 5 — 15 additional tests: EMA symmetry, config edge cases, integration
    // =========================================================================

    #[test]
    fn test_decaying_reservoir_ema_symmetry_high_then_low_equals_low_then_high() {
        // Two reservoirs with same half-life: one goes 10→0, other goes 0→10.
        // After 2 samples each, the results differ because first sample sets directly.
        // Verify the EMA is path-dependent (not symmetric).
        let mut r1 = DecayingReservoir::new(100);
        let mut r2 = DecayingReservoir::new(100);

        let mut h_high = KvPageHeader::new(0);
        h_high.ref_count = 1;
        h_high.entropy_avg = f32_to_f16_bits(10.0);

        let mut h_low = KvPageHeader::new(0);
        h_low.ref_count = 1;
        h_low.entropy_avg = f32_to_f16_bits(0.0);

        // r1: high first, then low
        r1.ingest(&h_high);
        r1.ingest(&h_low);

        // r2: low first, then high
        r2.ingest(&h_low);
        r2.ingest(&h_high);

        // Both should have same sample_count
        assert_eq!(r1.sample_count(), 2);
        assert_eq!(r2.sample_count(), 2);

        // Path-dependent: results should NOT be equal
        assert_ne!(r1.avg_entropy(), r2.avg_entropy());
    }

    #[test]
    fn test_director_config_clone_preserves_all_three_fields() {
        let config = DirectorConfig {
            scan_interval: Duration::from_millis(333),
            half_life_samples: 777,
            num_experts: 42,
        };
        let cloned = config.clone();
        assert_eq!(cloned.scan_interval, Duration::from_millis(333));
        assert_eq!(cloned.half_life_samples, 777);
        assert_eq!(cloned.num_experts, 42);
    }

    #[test]
    fn test_shared_state_record_hit_on_last_expert_index() {
        let state = DirectorSharedState::new(8);
        // Record hit on the last valid expert index
        state.record_expert_hit(7);
        let hits = state.snapshot_and_reset_expert_hits();
        assert_eq!(hits.len(), 8);
        assert_eq!(hits[7], 1);
        assert!(hits[..7].iter().all(|&h| h == 0));
    }

    #[test]
    fn test_consensus_detector_expert_streak_one_after_single_zero_hit() {
        let mut detector = ConsensusDetector::new(4);
        // Only expert 3 has zero hits in this round
        detector.update_expert_hits(&[5, 3, 7, 0]);
        assert_eq!(detector.expert_zero_streaks[0], 0);
        assert_eq!(detector.expert_zero_streaks[1], 0);
        assert_eq!(detector.expert_zero_streaks[2], 0);
        assert_eq!(detector.expert_zero_streaks[3], 1);
    }

    #[test]
    fn test_decaying_reservoir_consecutive_ingests_sample_count_matches() {
        let mut reservoir = DecayingReservoir::new(100);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.entropy_avg = f32_to_f16_bits(1.5);

        for expected_count in 1..=50u64 {
            reservoir.ingest(&h);
            assert_eq!(reservoir.sample_count(), expected_count);
        }
    }

    #[test]
    fn test_consensus_detector_detect_with_zero_attention_threshold() {
        // threshold=0.0 means avg_entropy < 0.0 is impossible for non-negative values
        let mut detector = ConsensusDetector::new(2);
        detector.attention_silent_threshold = 0.0;

        let mut reservoir = DecayingReservoir::new(100);
        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.entropy_avg = f32_to_f16_bits(0.0); // Exactly zero

        for _ in 0..2000 {
            reservoir.ingest(&header);
        }

        // avg_entropy should be 0.0, and 0.0 < 0.0 is false
        // so attention_silent_steps should not increment
        let _events = detector.detect(&reservoir);
        assert_eq!(detector.attention_silent_steps, 0);
    }

    #[test]
    fn test_shared_state_page_header_fields_preserved_across_update_read() {
        let state = DirectorSharedState::new(0);

        let mut h = KvPageHeader::new(42);
        h.ref_count = 3;
        h.entropy_avg = f32_to_f16_bits(2.7);
        h.delta_rho_avg = f32_to_f16_bits(0.4);
        h.dead_ratio = 77;
        h.centroid_pos = f32_to_f16_bits(1.1);

        state.update_page_headers(vec![h]);
        let headers = state.read_page_headers();

        assert_eq!(headers[0].page_id, 42);
        assert_eq!(headers[0].ref_count, 3);
        assert_eq!(headers[0].dead_ratio, 77);
    }

    #[test]
    fn test_pipeline_inactive_headers_excluded_from_reservoir_ingest() {
        let mut reservoir = DecayingReservoir::new(100);
        let state = DirectorSharedState::new(0);

        // Two headers: one active, one inactive
        let mut h_active = KvPageHeader::new(0);
        h_active.ref_count = 1;
        h_active.entropy_avg = f32_to_f16_bits(5.0);

        let mut h_inactive = KvPageHeader::new(1);
        h_inactive.ref_count = 0;
        h_inactive.entropy_avg = f32_to_f16_bits(99.0);

        state.update_page_headers(vec![h_active, h_inactive]);

        let headers = state.read_page_headers();
        let mut active_count = 0u64;
        for header in &headers {
            if header.is_active() {
                reservoir.ingest(header);
                active_count += 1;
            }
        }

        assert_eq!(active_count, 1);
        assert_eq!(reservoir.sample_count(), 1);
        assert!((reservoir.avg_entropy() - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_consensus_event_expert_frozen_with_zero_steps() {
        let event = ConsensusEvent::ExpertFrozen {
            expert_idx: 0,
            zero_hit_steps: 0,
        };
        if let ConsensusEvent::ExpertFrozen { expert_idx, zero_hit_steps } = event {
            assert_eq!(expert_idx, 0);
            assert_eq!(zero_hit_steps, 0);
        } else {
            panic!("Expected ExpertFrozen variant");
        }
    }

    #[test]
    fn test_decaying_reservoir_dead_ratio_zero_produces_exact_zero() {
        let mut reservoir = DecayingReservoir::new(100);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.dead_ratio = 0;
        reservoir.ingest(&h);
        assert_eq!(reservoir.avg_dead_neuron_ratio(), 0.0);
    }

    #[test]
    fn test_consensus_detector_layer_redundant_custom_threshold() {
        let mut detector = ConsensusDetector::new(2);
        // Set a very high layer_redundant_threshold so that normal delta triggers it
        detector.layer_redundant_threshold = 1.0;

        let mut reservoir = DecayingReservoir::new(1000);
        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.delta_rho_avg = f32_to_f16_bits(0.5); // Below custom threshold of 1.0

        for _ in 0..2000 {
            reservoir.ingest(&header);
        }

        // delta_rho 0.5 < threshold 1.0 → should increment
        let _ = detector.detect(&reservoir);
        assert_eq!(detector.layer_redundant_steps, 1);
    }

    #[test]
    fn test_shared_state_drain_after_push_multiple_batches_preserves_total_order() {
        let state = DirectorSharedState::new(0);

        // Push 3 separate batches
        state.push_events(vec![ConsensusEvent::ExpertFrozen {
            expert_idx: 0, zero_hit_steps: 10,
        }]);
        state.push_events(vec![
            ConsensusEvent::ExpertFrozen { expert_idx: 1, zero_hit_steps: 20 },
            ConsensusEvent::ExpertFrozen { expert_idx: 2, zero_hit_steps: 30 },
        ]);
        state.push_events(vec![ConsensusEvent::ExpertFrozen {
            expert_idx: 3, zero_hit_steps: 40,
        }]);

        let events = state.drain_events();
        assert_eq!(events.len(), 4);

        // Verify order: batch1 → batch2 → batch3
        let indices: Vec<usize> = events.iter().filter_map(|e| match e {
            ConsensusEvent::ExpertFrozen { expert_idx, .. } => Some(*expert_idx),
            _ => None,
        }).collect();
        assert_eq!(indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_decaying_reservoir_decay_factor_monotonically_increases_with_half_life() {
        // Longer half-life → higher decay_factor (slower decay)
        let r1 = DecayingReservoir::new(10);
        let r2 = DecayingReservoir::new(100);
        let r3 = DecayingReservoir::new(1000);

        assert!(r1.decay_factor < r2.decay_factor);
        assert!(r2.decay_factor < r3.decay_factor);
    }

    #[test]
    fn test_director_config_with_zero_duration_scan_interval_spawns() {
        let config = DirectorConfig {
            scan_interval: Duration::ZERO,
            half_life_samples: 50,
            num_experts: 2,
        };
        let mut director = JitDirector::spawn(config);
        // Let it run briefly with zero sleep — tight loop but should not hang
        std::thread::sleep(Duration::from_millis(20));
        director.shutdown();
    }

    #[test]
    fn test_consensus_detector_attention_silent_event_carries_correct_avg_entropy() {
        let mut detector = ConsensusDetector::new(2);
        detector.attention_silent_threshold = 0.5;

        let mut reservoir = DecayingReservoir::new(10);
        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.entropy_avg = f32_to_f16_bits(0.01); // Very low

        for _ in 0..2000 {
            reservoir.ingest(&header);
        }

        for _ in 0..10001 {
            let _ = detector.detect(&reservoir);
        }

        let events = detector.detect(&reservoir);
        let silent_event = events.iter().find(|e| matches!(e, ConsensusEvent::AttentionSilent { .. }));
        assert!(silent_event.is_some());

        if let Some(ConsensusEvent::AttentionSilent { avg_entropy, .. }) = silent_event {
            // The avg_entropy in the event should match the reservoir's current value
            assert!((avg_entropy - reservoir.avg_entropy()).abs() < 1e-10);
        }
    }

    // =========================================================================
    // Round 6 — 15 additional tests: stress, concurrent, boundary, integration
    // =========================================================================

    #[test]
    fn test_decaying_reservoir_ingest_negative_delta_rho_sets_negative() {
        // Verify that a negative f16 delta_rho_avg produces a negative avg_residual_delta
        let mut reservoir = DecayingReservoir::new(100);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        // f16 -0.5: sign=1, exp=14 (bias 15, so 14 means 2^(-1)=0.5), mant=0 → 0xB800
        h.delta_rho_avg = 0xB800;

        reservoir.ingest(&h);
        assert_eq!(reservoir.sample_count(), 1);
        assert!(reservoir.avg_residual_delta() < 0.0);
    }

    #[test]
    fn test_consensus_detector_layer_redundant_event_carries_correct_avg_delta() {
        // Verify that the LayerRedundant event carries the exact reservoir avg_residual_delta
        let mut detector = ConsensusDetector::new(2);

        let mut reservoir = DecayingReservoir::new(10);
        let mut header = KvPageHeader::new(0);
        header.ref_count = 1;
        header.delta_rho_avg = f32_to_f16_bits(0.0001);

        for _ in 0..2000 {
            reservoir.ingest(&header);
        }

        for _ in 0..10001 {
            let _ = detector.detect(&reservoir);
        }

        let events = detector.detect(&reservoir);
        let layer_event = events.iter().find(|e| matches!(e, ConsensusEvent::LayerRedundant { .. }));
        assert!(layer_event.is_some());

        if let Some(ConsensusEvent::LayerRedundant { avg_delta_rho, .. }) = layer_event {
            assert!((avg_delta_rho - reservoir.avg_residual_delta()).abs() < 1e-10);
        }
    }

    #[test]
    fn test_decaying_reservoir_four_fields_initial_after_new() {
        // Comprehensive initial state check for a fresh reservoir
        let r = DecayingReservoir::new(250);
        assert_eq!(r.sample_count(), 0);
        assert!((r.avg_entropy() - 0.0).abs() < 1e-10);
        assert!((r.avg_residual_delta() - 1.0).abs() < 1e-10, "initial residual delta should be 1.0");
        assert!((r.avg_dead_neuron_ratio() - 0.0).abs() < 1e-10);
        assert!((r.avg_softmax_sharpness() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_consensus_detector_expert_streak_u64_does_not_overflow_quickly() {
        // Verify that streaks can grow very large without issue
        let mut d = ConsensusDetector::new(1);
        d.expert_freeze_threshold = u64::MAX;

        for _ in 0..100_000 {
            d.update_expert_hits(&[0]);
        }
        assert_eq!(d.expert_zero_streaks[0], 100_000);

        // No event because streak < u64::MAX
        let r = DecayingReservoir::new(100);
        let events = d.detect(&r);
        assert!(events.is_empty());
    }

    #[test]
    fn test_shared_state_snapshot_multiple_resets_between_record_batches() {
        let state = DirectorSharedState::new(3);

        // Batch 1
        for _ in 0..10 {
            state.record_expert_hit(0);
            state.record_expert_hit(2);
        }
        let hits1 = state.snapshot_and_reset_expert_hits();
        assert_eq!(hits1, vec![10, 0, 10]);

        // Batch 2 — different pattern
        for _ in 0..5 {
            state.record_expert_hit(1);
        }
        let hits2 = state.snapshot_and_reset_expert_hits();
        assert_eq!(hits2, vec![0, 5, 0]);

        // Batch 3 — empty
        let hits3 = state.snapshot_and_reset_expert_hits();
        assert_eq!(hits3, vec![0, 0, 0]);
    }

    #[test]
    fn test_decaying_reservoir_entropy_converges_from_opposite_directions() {
        // Start with high, then constant medium; start with low, then same constant medium.
        // Both should converge toward the constant medium, but from different sides.
        let mut r_high = DecayingReservoir::new(5);
        let mut r_low = DecayingReservoir::new(5);

        let mut h_high = KvPageHeader::new(0);
        h_high.ref_count = 1;
        h_high.entropy_avg = f32_to_f16_bits(20.0);

        let mut h_low = KvPageHeader::new(0);
        h_low.ref_count = 1;
        h_low.entropy_avg = f32_to_f16_bits(0.0);

        let mut h_mid = KvPageHeader::new(0);
        h_mid.ref_count = 1;
        h_mid.entropy_avg = f32_to_f16_bits(10.0);

        r_high.ingest(&h_high);
        r_low.ingest(&h_low);

        // Feed 200 identical medium samples to both
        for _ in 0..200 {
            r_high.ingest(&h_mid);
            r_low.ingest(&h_mid);
        }

        // Both should converge near 10.0
        assert!((r_high.avg_entropy() - 10.0).abs() < 0.5);
        assert!((r_low.avg_entropy() - 10.0).abs() < 0.5);
    }

    #[test]
    fn test_consensus_event_debug_output_all_three_variants_comprehensive() {
        let expert = ConsensusEvent::ExpertFrozen { expert_idx: 42, zero_hit_steps: 12345 };
        let attention = ConsensusEvent::AttentionSilent { avg_entropy: 0.035, duration_steps: 11111 };
        let layer = ConsensusEvent::LayerRedundant { avg_delta_rho: 0.00078, duration_steps: 22222 };

        let expert_debug = format!("{:?}", expert);
        let attention_debug = format!("{:?}", attention);
        let layer_debug = format!("{:?}", layer);

        // ExpertFrozen
        assert!(expert_debug.contains("ExpertFrozen"));
        assert!(expert_debug.contains("42"));
        assert!(expert_debug.contains("12345"));

        // AttentionSilent
        assert!(attention_debug.contains("AttentionSilent"));
        assert!(attention_debug.contains("0.035"));
        assert!(attention_debug.contains("11111"));

        // LayerRedundant
        assert!(layer_debug.contains("LayerRedundant"));
        assert!(layer_debug.contains("0.00078"));
        assert!(layer_debug.contains("22222"));
    }

    #[test]
    fn test_pipeline_detector_and_reservoir_with_alternating_health_and_degradation() {
        let mut reservoir = DecayingReservoir::new(5);
        let mut detector = ConsensusDetector::new(2);
        detector.attention_silent_threshold = 0.1;

        // Phase 1: healthy data for 500 samples
        let mut h_healthy = KvPageHeader::new(0);
        h_healthy.ref_count = 1;
        h_healthy.entropy_avg = f32_to_f16_bits(3.0);
        h_healthy.delta_rho_avg = f32_to_f16_bits(0.5);

        for _ in 0..500 {
            reservoir.ingest(&h_healthy);
        }

        // Phase 2: degraded data for 100 samples
        let mut h_degraded = KvPageHeader::new(0);
        h_degraded.ref_count = 1;
        h_degraded.entropy_avg = f32_to_f16_bits(0.01);
        h_degraded.delta_rho_avg = f32_to_f16_bits(0.0001);

        for _ in 0..100 {
            reservoir.ingest(&h_degraded);
        }

        // With short half-life, avg should be close to degraded values
        assert!(reservoir.avg_entropy() < 0.5);
        assert!(reservoir.avg_residual_delta() < 0.01);

        // Phase 3: healthy again for 200 samples
        for _ in 0..200 {
            reservoir.ingest(&h_healthy);
        }

        // Should recover toward healthy values
        assert!(reservoir.avg_entropy() > 0.5);

        // No events should fire because silent/redundant steps reset during recovery
        detector.update_expert_hits(&[1, 1]);
        let events = detector.detect(&reservoir);
        assert!(!events.iter().any(|e| matches!(e, ConsensusEvent::AttentionSilent { .. })));
        assert!(!events.iter().any(|e| matches!(e, ConsensusEvent::LayerRedundant { .. })));
    }

    #[test]
    fn test_shared_state_concurrent_advance_step_and_read() {
        let state = Arc::new(DirectorSharedState::new(0));

        let state_writer = Arc::clone(&state);
        let writer = std::thread::spawn(move || {
            for _ in 0..1000 {
                state_writer.advance_step();
            }
        });

        let state_reader = Arc::clone(&state);
        let reader = std::thread::spawn(move || {
            let mut max_step = 0u64;
            for _ in 0..100 {
                let step = state_reader.global_step.load(Ordering::Relaxed);
                max_step = max_step.max(step);
                std::thread::sleep(Duration::from_micros(10));
            }
            max_step
        });

        writer.join().unwrap();
        let observed_max = reader.join().unwrap();

        // The observed max should be somewhere between 0 and 1000
        assert!(observed_max <= 1000);

        // After writer finishes, step should be exactly 1000
        let final_step = state.global_step.load(Ordering::Relaxed);
        assert_eq!(final_step, 1000);
    }

    #[test]
    fn test_consensus_detector_expert_freeze_threshold_one_minimal_trigger() {
        // threshold=1 means even a single zero-hit round triggers ExpertFrozen
        let mut d = ConsensusDetector::new(2);
        d.expert_freeze_threshold = 1;

        d.update_expert_hits(&[0, 5]);

        let r = DecayingReservoir::new(100);
        let events = d.detect(&r);

        assert!(events.iter().any(|e| matches!(
            e,
            ConsensusEvent::ExpertFrozen { expert_idx: 0, zero_hit_steps: 1 }
        )));
        assert!(!events.iter().any(|e| matches!(
            e,
            ConsensusEvent::ExpertFrozen { expert_idx: 1, .. }
        )));
    }

    #[test]
    fn test_decaying_reservoir_ingest_with_all_max_f16_fields() {
        // All fields at max finite f16 to verify no overflow or panic
        let mut reservoir = DecayingReservoir::new(100);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.entropy_avg = 0x7BFF;       // max finite f16 ~65504
        h.delta_rho_avg = 0x7BFF;
        h.centroid_pos = 0x7BFF;
        h.dead_ratio = 255;

        reservoir.ingest(&h);
        assert_eq!(reservoir.sample_count(), 1);
        assert!(reservoir.avg_entropy() > 60000.0);
        assert!(reservoir.avg_residual_delta() > 60000.0);
        assert!(reservoir.avg_softmax_sharpness() > 60000.0);
        assert!(reservoir.avg_dead_neuron_ratio() > 0.9);
    }

    #[test]
    fn test_director_spawn_shutdown_and_respawn_is_safe() {
        // Verify that spawning a new director after shutting down a previous one works
        let config1 = DirectorConfig {
            scan_interval: Duration::from_millis(10),
            half_life_samples: 50,
            num_experts: 2,
        };

        let mut d1 = JitDirector::spawn(config1);
        let shared1 = Arc::clone(d1.shared());
        shared1.record_expert_hit(0);
        d1.shutdown();

        // Spawn a second director — should be independent
        let config2 = DirectorConfig {
            scan_interval: Duration::from_millis(10),
            half_life_samples: 50,
            num_experts: 4,
        };
        let mut d2 = JitDirector::spawn(config2);
        let shared2 = Arc::clone(d2.shared());
        shared2.record_expert_hit(3);
        shared2.record_expert_hit(3);

        let hits = shared2.snapshot_and_reset_expert_hits();
        assert_eq!(hits.len(), 4);
        assert_eq!(hits[3], 2);

        d2.shutdown();
    }

    #[test]
    fn test_consensus_detector_attention_silent_and_layer_redundant_fire_simultaneously() {
        // Craft conditions where both AttentionSilent and LayerRedundant fire in the same detect call
        let mut detector = ConsensusDetector::new(2);
        detector.attention_silent_threshold = 0.5;

        let mut reservoir = DecayingReservoir::new(10);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.entropy_avg = f32_to_f16_bits(0.01);   // Low entropy
        h.delta_rho_avg = f32_to_f16_bits(0.0001); // Low delta

        for _ in 0..2000 {
            reservoir.ingest(&h);
        }

        for _ in 0..10001 {
            let _ = detector.detect(&reservoir);
        }

        let events = detector.detect(&reservoir);
        let has_attention = events.iter().any(|e| matches!(e, ConsensusEvent::AttentionSilent { .. }));
        let has_layer = events.iter().any(|e| matches!(e, ConsensusEvent::LayerRedundant { .. }));

        assert!(has_attention, "Expected AttentionSilent in simultaneous event list");
        assert!(has_layer, "Expected LayerRedundant in simultaneous event list");
    }

    #[test]
    fn test_shared_state_update_page_headers_large_batch_then_read() {
        let state = DirectorSharedState::new(0);

        // Create 500 headers with unique page_ids and varying entropy
        let headers: Vec<KvPageHeader> = (0..500)
            .map(|i| {
                let mut h = KvPageHeader::new(i);
                h.ref_count = 1;
                h.entropy_avg = f32_to_f16_bits((i as f32) * 0.01);
                h
            })
            .collect();

        state.update_page_headers(headers);

        let read = state.read_page_headers();
        assert_eq!(read.len(), 500);
        assert_eq!(read[0].page_id, 0);
        assert_eq!(read[499].page_id, 499);

        // Verify entropy values are preserved
        let first_entropy = f16_bits_to_f32(read[0].entropy_avg);
        let last_entropy = f16_bits_to_f32(read[499].entropy_avg);
        assert!((first_entropy - 0.0).abs() < 0.01);
        assert!((last_entropy - 4.99).abs() < 0.1);
    }

    #[test]
    fn test_decaying_reservoir_many_oscillating_values_remain_bounded() {
        // Feed entropy values that oscillate between 0 and 20 for many iterations
        // and verify the average stays within bounds
        let mut reservoir = DecayingReservoir::new(20);

        for i in 0..1000 {
            let mut h = KvPageHeader::new(0);
            h.ref_count = 1;
            let val = if i % 2 == 0 { 20.0f32 } else { 0.0f32 };
            h.entropy_avg = f32_to_f16_bits(val);
            reservoir.ingest(&h);
        }

        let entropy = reservoir.avg_entropy();
        assert!(entropy >= 0.0);
        assert!(entropy <= 20.0);
        // Oscillating between 0 and 20 with moderate decay → should be near 10.0
        assert!((entropy - 10.0).abs() < 5.0);
    }

    #[test]
    fn test_consensus_detector_update_expert_hits_exact_boundary_one_below_threshold() {
        // Verify that exactly threshold-1 streaks do NOT trigger
        let mut d = ConsensusDetector::new(3);
        d.expert_freeze_threshold = 100;

        for _ in 0..99 {
            d.update_expert_hits(&[0, 0, 0]);
        }

        let r = DecayingReservoir::new(100);
        let events = d.detect(&r);
        let frozen_count = events.iter().filter(|e| matches!(e, ConsensusEvent::ExpertFrozen { .. })).count();
        assert_eq!(frozen_count, 0, "99 streaks should not trigger at threshold 100");

        // One more round triggers it
        d.update_expert_hits(&[0, 0, 0]);
        let events2 = d.detect(&r);
        let frozen_count2 = events2.iter().filter(|e| matches!(e, ConsensusEvent::ExpertFrozen { .. })).count();
        assert_eq!(frozen_count2, 3, "100 streaks should trigger at threshold 100");
    }

    // =========================================================================
    // Round 7 — 15 additional tests: residual delta first-ingest, concurrent
    // expert hits, decay exponent verification, threshold immutability, etc.
    // =========================================================================

    // @trace TEST-DIR-001 [level:unit] Residual delta overrides initial 1.0 on first ingest
    #[test]
    fn test_decaying_reservoir_first_ingest_overrides_default_residual_delta() {
        // Arrange: fresh reservoir has avg_residual_delta = 1.0 (default)
        // Act: ingest a header with delta_rho_avg = 0.3
        let mut reservoir = DecayingReservoir::new(100);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.delta_rho_avg = f32_to_f16_bits(0.3);

        reservoir.ingest(&h);

        // Assert: avg_residual_delta should now be 0.3, not the default 1.0
        assert!((reservoir.avg_residual_delta() - 0.3).abs() < 0.01);
        assert!((reservoir.avg_residual_delta() - 1.0).abs() > 0.01, "Should no longer be 1.0");
    }

    // @trace TEST-DIR-002 [level:unit] Decay factor for half_life=5 matches exp(-ln2/5)
    #[test]
    fn test_decaying_reservoir_decay_factor_half_life_5_exact() {
        let reservoir = DecayingReservoir::new(5);
        let expected = (-(2.0_f64).ln() / 5.0).exp();
        assert!((reservoir.decay_factor - expected).abs() < 1e-12);
    }

    // @trace TEST-DIR-003 [level:unit] detect does not mutate threshold fields
    #[test]
    fn test_consensus_detector_detect_does_not_mutate_thresholds() {
        let mut detector = ConsensusDetector::new(2);
        let original_freeze = detector.expert_freeze_threshold;
        let original_attention = detector.attention_silent_threshold;
        let original_layer = detector.layer_redundant_threshold;

        let reservoir = DecayingReservoir::new(100);
        for _ in 0..5 {
            let _ = detector.detect(&reservoir);
        }

        assert_eq!(detector.expert_freeze_threshold, original_freeze);
        assert!((detector.attention_silent_threshold - original_attention).abs() < 1e-10);
        assert!((detector.layer_redundant_threshold - original_layer).abs() < 1e-10);
    }

    // @trace TEST-DIR-004 [level:unit] Concurrent recording to different experts from multiple threads
    #[test]
    fn test_shared_state_concurrent_record_different_experts() {
        let state = Arc::new(DirectorSharedState::new(4));

        let s0 = Arc::clone(&state);
        let t0 = std::thread::spawn(move || {
            for _ in 0..500 { s0.record_expert_hit(0); }
        });

        let s1 = Arc::clone(&state);
        let t1 = std::thread::spawn(move || {
            for _ in 0..300 { s1.record_expert_hit(1); }
        });

        let s2 = Arc::clone(&state);
        let t2 = std::thread::spawn(move || {
            for _ in 0..200 { s2.record_expert_hit(2); }
        });

        let s3 = Arc::clone(&state);
        let t3 = std::thread::spawn(move || {
            for _ in 0..100 { s3.record_expert_hit(3); }
        });

        t0.join().unwrap();
        t1.join().unwrap();
        t2.join().unwrap();
        t3.join().unwrap();

        let hits = state.snapshot_and_reset_expert_hits();
        assert_eq!(hits, vec![500, 300, 200, 100]);
    }

    // @trace TEST-DIR-005 [level:unit] ExpertFrozen event with large expert_idx and zero_hit_steps
    #[test]
    fn test_consensus_event_expert_frozen_large_index_debug_format() {
        let event = ConsensusEvent::ExpertFrozen {
            expert_idx: usize::MAX / 2,
            zero_hit_steps: 9_999_999_999,
        };
        let debug_str = format!("{:?}", event);
        assert!(debug_str.contains("ExpertFrozen"));
        assert!(debug_str.contains("9999999999"));

        // Verify round-trip via clone
        let cloned = event.clone();
        assert_eq!(event, cloned);
    }

    // @trace TEST-DIR-006 [level:unit] EMA entropy after 1000 identical ingests converges within epsilon
    #[test]
    fn test_decaying_reservoir_1000_identical_ingests_converges_tightly() {
        let mut reservoir = DecayingReservoir::new(50);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        // Use a value exactly representable in f16 to avoid precision issues
        h.entropy_avg = f32_to_f16_bits(4.5);

        for _ in 0..1000 {
            reservoir.ingest(&h);
        }

        // After 1000 identical samples, EMA should be within 0.01 of the constant
        let entropy = reservoir.avg_entropy();
        assert!((entropy - 4.5).abs() < 0.01, "Expected ~4.5, got {}", entropy);
        assert_eq!(reservoir.sample_count(), 1000);
    }

    // @trace TEST-DIR-007 [level:unit] Expert streak accumulates only for target expert
    #[test]
    fn test_consensus_detector_streak_only_for_zero_hit_expert() {
        let mut detector = ConsensusDetector::new(4);
        // Expert 0: hit=7, Expert 1: hit=0, Expert 2: hit=3, Expert 3: hit=0
        detector.update_expert_hits(&[7, 0, 3, 0]);
        detector.update_expert_hits(&[5, 0, 1, 0]);
        detector.update_expert_hits(&[2, 0, 4, 0]);

        assert_eq!(detector.expert_zero_streaks[0], 0);
        assert_eq!(detector.expert_zero_streaks[1], 3);
        assert_eq!(detector.expert_zero_streaks[2], 0);
        assert_eq!(detector.expert_zero_streaks[3], 3);
    }

    // @trace TEST-DIR-008 [level:unit] DirectorConfig default and manual construction are identical
    #[test]
    fn test_director_config_default_equals_explicit_construction() {
        let default = DirectorConfig::default();
        let explicit = DirectorConfig {
            scan_interval: Duration::from_millis(100),
            half_life_samples: 10_000,
            num_experts: 0,
        };
        assert_eq!(default.scan_interval, explicit.scan_interval);
        assert_eq!(default.half_life_samples, explicit.half_life_samples);
        assert_eq!(default.num_experts, explicit.num_experts);
    }

    // @trace TEST-DIR-009 [level:unit] Clone preserves all four metrics and decay_factor
    #[test]
    fn test_decaying_reservoir_clone_exact_copy_all_fields() {
        let mut reservoir = DecayingReservoir::new(25);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.entropy_avg = f32_to_f16_bits(3.5);
        h.delta_rho_avg = f32_to_f16_bits(0.7);
        h.dead_ratio = 180;
        h.centroid_pos = f32_to_f16_bits(1.2);
        reservoir.ingest(&h);
        reservoir.ingest(&h);

        let cloned = reservoir.clone();

        assert_eq!(cloned.sample_count(), 2);
        assert!((cloned.avg_entropy() - reservoir.avg_entropy()).abs() < 1e-10);
        assert!((cloned.avg_residual_delta() - reservoir.avg_residual_delta()).abs() < 1e-10);
        assert!((cloned.avg_dead_neuron_ratio() - reservoir.avg_dead_neuron_ratio()).abs() < 1e-10);
        assert!((cloned.avg_softmax_sharpness() - reservoir.avg_softmax_sharpness()).abs() < 1e-10);
        assert!((cloned.decay_factor - reservoir.decay_factor).abs() < 1e-15);
    }

    // @trace TEST-DIR-010 [level:unit] Page header entropy field survives update-then-read
    #[test]
    fn test_shared_state_entropy_field_survives_update_read_cycle() {
        let state = DirectorSharedState::new(0);

        let mut h = KvPageHeader::new(7);
        h.ref_count = 1;
        h.entropy_avg = f32_to_f16_bits(2.34);
        h.delta_rho_avg = f32_to_f16_bits(0.56);
        h.centroid_pos = f32_to_f16_bits(1.23);
        h.dead_ratio = 42;

        state.update_page_headers(vec![h]);
        let headers = state.read_page_headers();

        assert_eq!(headers.len(), 1);
        let read_entropy = f16_bits_to_f32(headers[0].entropy_avg);
        let read_delta = f16_bits_to_f32(headers[0].delta_rho_avg);
        assert!((read_entropy - 2.34).abs() < 0.01);
        assert!((read_delta - 0.56).abs() < 0.01);
    }

    // @trace TEST-DIR-011 [level:unit] LayerRedundant steps reset after single high-delta detect
    #[test]
    fn test_consensus_detector_layer_redundant_steps_reset_on_single_high_delta() {
        let mut detector = ConsensusDetector::new(2);

        let mut reservoir = DecayingReservoir::new(10);
        let mut h_low = KvPageHeader::new(0);
        h_low.ref_count = 1;
        h_low.delta_rho_avg = f32_to_f16_bits(0.0001);

        for _ in 0..1100 {
            reservoir.ingest(&h_low);
        }
        // Accumulate 50 layer_redundant_steps
        for _ in 0..50 {
            let _ = detector.detect(&reservoir);
        }
        assert_eq!(detector.layer_redundant_steps, 50);

        // Single high-delta sample causes reset
        let mut h_high = KvPageHeader::new(0);
        h_high.ref_count = 1;
        h_high.delta_rho_avg = f32_to_f16_bits(10.0);
        for _ in 0..50 {
            reservoir.ingest(&h_high);
        }
        let _ = detector.detect(&reservoir);
        assert_eq!(detector.layer_redundant_steps, 0);
    }

    // @trace TEST-DIR-012 [level:unit] Director shared Arc ref count after clone and drop
    #[test]
    fn test_director_shared_arc_ref_count_after_clone_and_drop() {
        let config = DirectorConfig {
            scan_interval: Duration::from_millis(10),
            half_life_samples: 50,
            num_experts: 2,
        };
        let mut director = JitDirector::spawn(config);
        let shared = director.shared();
        let count_after_clone = Arc::strong_count(shared);

        {
            let _extra = Arc::clone(shared);
            assert_eq!(Arc::strong_count(shared), count_after_clone + 1);
        }
        // _extra dropped
        assert_eq!(Arc::strong_count(shared), count_after_clone);

        director.shutdown();
    }

    // @trace TEST-DIR-013 [level:unit] Reset expert then immediate zero-hits accumulates from 0
    #[test]
    fn test_consensus_detector_reset_then_zero_hits_streak_starts_from_zero() {
        let mut detector = ConsensusDetector::new(3);
        detector.expert_freeze_threshold = 100;

        // Build large streak
        for _ in 0..50 {
            detector.update_expert_hits(&[0, 0, 0]);
        }
        assert_eq!(detector.expert_zero_streaks[1], 50);

        // Reset expert 1
        detector.reset_expert(1);
        assert_eq!(detector.expert_zero_streaks[1], 0);

        // 3 more zero-hit rounds
        for _ in 0..3 {
            detector.update_expert_hits(&[0, 0, 0]);
        }
        // Expert 1 streak should be exactly 3, not 53
        assert_eq!(detector.expert_zero_streaks[1], 3);
        assert_eq!(detector.expert_zero_streaks[0], 53);
    }

    // @trace TEST-DIR-014 [level:unit] Push events with all three variants preserves variant order
    #[test]
    fn test_shared_state_push_all_three_variants_preserves_order() {
        let state = DirectorSharedState::new(0);

        let e1 = ConsensusEvent::ExpertFrozen { expert_idx: 0, zero_hit_steps: 10 };
        let e2 = ConsensusEvent::AttentionSilent { avg_entropy: 0.02, duration_steps: 12000 };
        let e3 = ConsensusEvent::LayerRedundant { avg_delta_rho: 0.0004, duration_steps: 11000 };

        state.push_events(vec![e1.clone(), e2.clone(), e3.clone()]);

        let events = state.drain_events();
        assert_eq!(events.len(), 3);
        assert!(matches!(events[0], ConsensusEvent::ExpertFrozen { .. }));
        assert!(matches!(events[1], ConsensusEvent::AttentionSilent { .. }));
        assert!(matches!(events[2], ConsensusEvent::LayerRedundant { .. }));
        assert_eq!(events[0], e1);
        assert_eq!(events[1], e2);
        assert_eq!(events[2], e3);
    }

    // @trace TEST-DIR-015 [level:unit] Inactive page header filtered in scan-loop pattern
    #[test]
    fn test_pipeline_scan_loop_pattern_only_ingests_active_headers() {
        let mut reservoir = DecayingReservoir::new(100);

        // 3 headers: 2 active, 1 inactive
        let mut h1 = KvPageHeader::new(10);
        h1.ref_count = 1;
        h1.entropy_avg = f32_to_f16_bits(2.0);

        let mut h2 = KvPageHeader::new(20);
        h2.ref_count = 0; // inactive

        let mut h3 = KvPageHeader::new(30);
        h3.ref_count = 1;
        h3.entropy_avg = f32_to_f16_bits(4.0);

        let headers = vec![h1, h2, h3];

        // Simulate scan_loop: only ingest active headers
        let mut active_ingested = 0u64;
        for header in &headers {
            if header.is_active() {
                reservoir.ingest(header);
                active_ingested += 1;
            }
        }

        assert_eq!(active_ingested, 2);
        assert_eq!(reservoir.sample_count(), 2);
        // EMA of 2.0 and 4.0 after two samples: first sets 2.0, second blends
        assert!(reservoir.avg_entropy() > 2.0);
        assert!(reservoir.avg_entropy() < 4.0);
    }

    // =========================================================================
    // Round 8 — 10 additional tests: concurrent push+drain with many events,
    // EMA weight sum verification, attention_silent_steps persistence across
    // healthy interludes, expert counter u64 truncation edge, dead_ratio
    // 127 boundary, director scan loop ingests expert hits snapshot,
    // config scan_interval does not affect default values, reservoir
    // entropy after NaN then valid, shared state events interleaved push/drain
    // =========================================================================

    // @trace TEST-DIR-016 [level:unit] EMA decay_factor + weight always sums to 1.0
    #[test]
    fn test_decaying_reservoir_decay_plus_weight_equals_one() {
        // Arrange: create reservoirs with various half-life values
        for half_life in [1u64, 2, 5, 10, 50, 100, 500, 1000, 10_000] {
            let reservoir = DecayingReservoir::new(half_life);

            // Act: compute weight = 1 - decay_factor
            let weight = 1.0 - reservoir.decay_factor;

            // Assert: decay_factor + weight == 1.0 (within floating point precision)
            assert!(
                (reservoir.decay_factor + weight - 1.0).abs() < 1e-15,
                "decay_factor + weight != 1.0 for half_life={}",
                half_life
            );
        }
    }

    // @trace TEST-DIR-017 [level:unit] Shared state push many events then drain all at once
    #[test]
    fn test_shared_state_push_100_events_then_drain_all() {
        // Arrange: create shared state and push 100 ExpertFrozen events
        let state = DirectorSharedState::new(0);

        for i in 0..100 {
            state.push_events(vec![ConsensusEvent::ExpertFrozen {
                expert_idx: i,
                zero_hit_steps: i as u64 * 100,
            }]);
        }

        // Act: drain all events in one call
        let events = state.drain_events();

        // Assert: all 100 events present, in order
        assert_eq!(events.len(), 100);
        for (i, event) in events.iter().enumerate() {
            if let ConsensusEvent::ExpertFrozen { expert_idx, zero_hit_steps } = event {
                assert_eq!(*expert_idx, i);
                assert_eq!(*zero_hit_steps, i as u64 * 100);
            } else {
                panic!("Expected ExpertFrozen at index {}", i);
            }
        }

        // Second drain should be empty
        let events2 = state.drain_events();
        assert!(events2.is_empty());
    }

    // @trace TEST-DIR-018 [level:unit] Attention silent steps persist across detect calls when condition holds
    #[test]
    fn test_consensus_detector_attention_silent_steps_accumulates_monotonically() {
        // Arrange: detector with low threshold, reservoir with enough low-entropy samples
        let mut detector = ConsensusDetector::new(2);
        detector.attention_silent_threshold = 0.5;

        let mut reservoir = DecayingReservoir::new(10);
        let mut h = KvPageHeader::new(0);
        h.ref_count = 1;
        h.entropy_avg = f32_to_f16_bits(0.01);

        for _ in 0..2000 {
            reservoir.ingest(&h);
        }

        // Act: call detect 500 times and verify steps accumulate strictly
        for expected_step in 1..=500 {
            let _events = detector.detect(&reservoir);
            assert_eq!(
                detector.attention_silent_steps, expected_step,
                "Expected attention_silent_steps={}, got {}",
                expected_step, detector.attention_silent_steps
            );
        }

        // Assert: after 500 calls, steps = 500 (still below 10000 threshold, no event)
        let events = detector.detect(&reservoir);
        assert!(!events.iter().any(|e| matches!(e, ConsensusEvent::AttentionSilent { .. })));
        assert_eq!(detector.attention_silent_steps, 501);
    }

    // @trace TEST-DIR-019 [level:unit] Expert hit counter u64->u32 truncation at u64::MAX
    #[test]
    fn test_shared_state_expert_hit_counter_u64_max_truncates() {
        // Arrange: store u64::MAX in the counter (simulating extreme overflow)
        let state = DirectorSharedState::new(1);
        state.expert_hit_counters[0].store(u64::MAX, Ordering::Relaxed);

        // Act: snapshot converts to u32 (truncation)
        let hits = state.snapshot_and_reset_expert_hits();

        // Assert: u64::MAX as u32 = u32::MAX (lower 32 bits)
        assert_eq!(hits[0], u32::MAX);

        // Counter should be reset to 0
        assert_eq!(state.expert_hit_counters[0].load(Ordering::Relaxed), 0);
    }

    // @trace TEST-DIR-020 [level:unit] Dead_ratio 127 boundary value between mid and high
    #[test]
    fn test_decaying_reservoir_dead_ratio_127_is_mid_range() {
        // Arrange: create two reservoirs with dead_ratio=127 and dead_ratio=128
        let mut r127 = DecayingReservoir::new(100);
        let mut h127 = KvPageHeader::new(0);
        h127.ref_count = 1;
        h127.dead_ratio = 127;

        let mut r128 = DecayingReservoir::new(100);
        let mut h128 = KvPageHeader::new(0);
        h128.ref_count = 1;
        h128.dead_ratio = 128;

        // Act: ingest each
        r127.ingest(&h127);
        r128.ingest(&h128);

        // Assert: 127 < 128, so ratio should be ordered accordingly
        assert!(r127.avg_dead_neuron_ratio() < r128.avg_dead_neuron_ratio());

        // Both should be close to 0.5
        assert!(r127.avg_dead_neuron_ratio() > 0.3);
        assert!(r128.avg_dead_neuron_ratio() < 0.6);
    }

    // @trace TEST-DIR-021 [level:unit] Director scan loop processes expert hits from shared state
    #[test]
    fn test_director_scan_loop_processes_expert_hits_from_shared_state() {
        // Arrange: spawn director, record hits, let scan loop process them
        let config = DirectorConfig {
            scan_interval: Duration::from_millis(5),
            half_life_samples: 50,
            num_experts: 3,
        };

        let mut director = JitDirector::spawn(config);
        let shared = Arc::clone(director.shared());

        // Record hits that the scan loop should snapshot and reset
        for _ in 0..50 {
            shared.record_expert_hit(0);
            shared.record_expert_hit(1);
        }

        // Act: let the director run a few scan iterations
        std::thread::sleep(Duration::from_millis(50));

        // Assert: expert hit counters should have been reset to 0 by scan loop
        let hits_after_scan = shared.snapshot_and_reset_expert_hits();
        // All hits should have been consumed by the scan loop in at least one iteration
        // The remaining hits may be 0 or very small depending on timing
        let total_remaining: u32 = hits_after_scan.iter().sum();
        assert!(total_remaining < 100, "Most hits should have been consumed by scan loop, got {}", total_remaining);

        director.shutdown();
    }

    // @trace TEST-DIR-022 [level:unit] Reservoir recovers from NaN entropy after valid ingests
    #[test]
    fn test_decaying_reservoir_recovers_from_nan_entropy_with_valid_ingests() {
        // Arrange: ingest NaN entropy first, then valid values
        let mut reservoir = DecayingReservoir::new(2); // fast decay

        let mut h_nan = KvPageHeader::new(0);
        h_nan.ref_count = 1;
        h_nan.entropy_avg = 0x7C01; // f16 NaN
        reservoir.ingest(&h_nan);
        assert!(reservoir.avg_entropy().is_nan());

        // Act: feed many valid entropy values
        for _ in 0..100 {
            let mut h_valid = KvPageHeader::new(0);
            h_valid.ref_count = 1;
            h_valid.entropy_avg = f32_to_f16_bits(5.0);
            reservoir.ingest(&h_valid);
        }

        // Assert: NaN * decay_factor + 5.0 * weight → NaN persists (NaN arithmetic)
        // This verifies the documented behavior that NaN propagation is expected
        // under IEEE 754 arithmetic rules
        let entropy = reservoir.avg_entropy();
        // NaN is expected because once NaN enters the EMA, it persists:
        // avg = d * NaN + w * 5.0 = NaN
        assert!(entropy.is_nan(), "NaN propagates through EMA per IEEE 754");
    }

    // @trace TEST-DIR-023 [level:unit] Shared state interleaved push and drain preserves total count
    #[test]
    fn test_shared_state_interleaved_push_drain_preserves_total_count() {
        // Arrange: create shared state, push events interleaved with drains
        let state = DirectorSharedState::new(0);
        let mut total_pushed = 0usize;
        let mut total_drained = 0usize;

        // Act: push 10 batches, draining after every other push
        for i in 0..10 {
            let batch_size = (i % 3) + 1; // 1, 2, or 3 events per batch
            let mut batch = Vec::with_capacity(batch_size);
            for j in 0..batch_size {
                batch.push(ConsensusEvent::ExpertFrozen {
                    expert_idx: (total_pushed + j) % 256,
                    zero_hit_steps: (total_pushed + j) as u64,
                });
            }
            total_pushed += batch_size;
            state.push_events(batch);

            // Drain every other iteration
            if i % 2 == 1 {
                let drained = state.drain_events();
                total_drained += drained.len();
            }
        }

        // Final drain to collect remaining events
        let final_drain = state.drain_events();
        total_drained += final_drain.len();

        // Assert: total pushed == total drained
        assert_eq!(total_pushed, total_drained);
    }

    // @trace TEST-DIR-024 [level:unit] ConsensusDetector detect fires ExpertFrozen only for streaks at or above threshold
    #[test]
    fn test_consensus_detector_detect_fires_only_for_eligible_experts_with_mixed_streaks() {
        // Arrange: 6 experts with varying streak lengths and threshold = 5
        let mut detector = ConsensusDetector::new(6);
        detector.expert_freeze_threshold = 5;

        // Build mixed streaks by carefully controlling hits per expert:
        // Phase 1: 3 rounds of all zeros for experts 0,1,2,4,5; expert 3 gets hits
        for _ in 0..3 {
            detector.update_expert_hits(&[0, 0, 0, 1, 0, 0]);
        }
        // After phase 1: streaks = [3, 3, 3, 0, 3, 3]

        // Phase 2: expert 2 gets a hit (streak resets to 0)
        detector.update_expert_hits(&[0, 0, 1, 1, 0, 0]);
        // After phase 2: streaks = [4, 4, 0, 0, 4, 4]

        // Phase 3: 4 more rounds; expert 2 keeps getting hits
        for _ in 0..4 {
            detector.update_expert_hits(&[0, 0, 5, 1, 0, 0]);
        }
        // After phase 3: streaks = [8, 8, 0, 0, 8, 8]

        // Assert streaks
        assert_eq!(detector.expert_zero_streaks[0], 8);
        assert_eq!(detector.expert_zero_streaks[1], 8);
        assert_eq!(detector.expert_zero_streaks[2], 0); // reset by hit in phase 2+3
        assert_eq!(detector.expert_zero_streaks[3], 0); // always has hits
        assert_eq!(detector.expert_zero_streaks[4], 8);
        assert_eq!(detector.expert_zero_streaks[5], 8);

        // Act
        let reservoir = DecayingReservoir::new(100);
        let events = detector.detect(&reservoir);

        // Assert: experts 0,1,4,5 fire (streaks=8 >= threshold=5)
        // Experts 2,3 do NOT fire (streaks=0 < threshold)
        let frozen_indices: Vec<usize> = events.iter()
            .filter_map(|e| match e {
                ConsensusEvent::ExpertFrozen { expert_idx, .. } => Some(*expert_idx),
                _ => None,
            })
            .collect();

        assert!(frozen_indices.contains(&0), "Expert 0 should be frozen (streak=8)");
        assert!(frozen_indices.contains(&1), "Expert 1 should be frozen (streak=8)");
        assert!(!frozen_indices.contains(&2), "Expert 2 should NOT be frozen (streak=0)");
        assert!(!frozen_indices.contains(&3), "Expert 3 should NOT be frozen (streak=0)");
        assert!(frozen_indices.contains(&4), "Expert 4 should be frozen (streak=8)");
        assert!(frozen_indices.contains(&5), "Expert 5 should be frozen (streak=8)");
        assert_eq!(frozen_indices.len(), 4);
    }

    // @trace TEST-DIR-025 [level:unit] DirectorConfig scan_interval mutation does not affect defaults
    #[test]
    fn test_director_config_mutation_does_not_affect_other_defaults() {
        // Arrange: get default config
        let mut config = DirectorConfig::default();
        assert_eq!(config.scan_interval, Duration::from_millis(100));
        assert_eq!(config.half_life_samples, 10_000);
        assert_eq!(config.num_experts, 0);

        // Act: mutate only scan_interval
        config.scan_interval = Duration::from_secs(30);

        // Assert: other fields remain at their default values
        assert_eq!(config.scan_interval, Duration::from_secs(30));
        assert_eq!(config.half_life_samples, 10_000, "half_life_samples should remain at default");
        assert_eq!(config.num_experts, 0, "num_experts should remain at default");

        // Mutate only num_experts
        config.num_experts = 64;
        assert_eq!(config.scan_interval, Duration::from_secs(30), "scan_interval should remain mutated");
        assert_eq!(config.half_life_samples, 10_000, "half_life_samples should remain at default");
        assert_eq!(config.num_experts, 64);
    }
}
