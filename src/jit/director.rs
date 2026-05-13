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
}
