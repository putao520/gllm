//! §9.3 残差数据总线物理结构 (The Residual Bus)
//!
//! SPEC §9.3 定义: 残差流 `x_out = x_in + Layer(x_in)` 必须被编译器重构为一条开放的
//! **插入端口 (Injection Port)** 与 **召回端口 (Recall Port)**。
//!
//! ## 四大物理应用 (SPEC §16)
//! - 16.1 Late-Fusion RAG: 外部检索向量通过 Injection Port 植入语义深处
//! - 16.2 PGSLE Early-Exit: 中间层挂载微型 lm_head 截断计算
//! - 16.3 Pure_Decode Intent NLU: 提取语义核心区残差特征做意图识别
//! - 16.4 In-Flight Guardrail: 极简线性探针寄生挂载做安全护栏
//!
//! ## 数据流
//! ```text
//! Layer L:
//!   x_in ──→ [ResidualBus Injection Point] ──→ Layer(x_in) ──→ [ResidualBus Recall Point]
//!             ↑                                           ↓
//!        inject_rag()                              extract_for_exit()
//!        inject_guard()                            extract_for_intent()
//!         ↓                                              ↑
//!   x_out = x_in + Layer(x_in) + injected
//! ```

use std::sync::atomic::{AtomicU32, Ordering};

/// 残差总线端口类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusPortKind {
    /// 插入端口 — 外部数据注入到指定层的残差流
    Injection,
    /// 召回端口 — 从指定层的残差流中提取数据
    Recall,
}

/// 残差总线挂载点 — 描述一个端口在模型中的物理位置
#[derive(Debug)]
pub struct BusPort {
    /// 端口类型
    pub kind: BusPortKind,
    /// 挂载层索引 (0-based, 包含)
    pub layer: usize,
    /// 端口标识
    pub tag: BusPortTag,
    /// 是否激活 — 可在运行时通过 Hot JMP Patching 动态开关
    pub active: AtomicU32,
}

/// 残差总线端口用途标识
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BusPortTag {
    /// §16.1 Late-Fusion RAG 知识注入
    RagInjection,
    /// §16.2 PGSLE Early-Exit 截断信号
    EarlyExit,
    /// §16.3 Pure_Decode 意图识别召回
    IntentRecall,
    /// §16.4 In-Flight Guardrail 安全探针
    Guardrail,
    /// §17.5 ADEPT 阴影 KV 投影
    ShadowKv,
    /// 自定义端口
    Custom(u32),
}

impl BusPort {
    /// 创建一个插入端口
    pub fn injection(layer: usize, tag: BusPortTag) -> Self {
        Self {
            kind: BusPortKind::Injection,
            layer,
            tag,
            active: AtomicU32::new(1),
        }
    }

    /// 创建一个召回端口
    pub fn recall(layer: usize, tag: BusPortTag) -> Self {
        Self {
            kind: BusPortKind::Recall,
            layer,
            tag,
            active: AtomicU32::new(1),
        }
    }

    /// 端口是否激活
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire) != 0
    }

    /// 激活端口
    pub fn activate(&self) {
        self.active.store(1, Ordering::Release);
    }

    /// 停用端口
    pub fn deactivate(&self) {
        self.active.store(0, Ordering::Release);
    }
}

/// 注入数据包 — 携带外部向量注入到残差总线
#[derive(Debug, Clone)]
pub struct InjectionPayload {
    /// 目标端口 tag
    pub target: BusPortTag,
    /// 注入向量数据 (hidden_size 维度)
    pub data: Vec<f32>,
    /// 缩放因子 (默认 1.0)
    pub scale: f32,
}

/// 召回数据包 — 从残差总线提取的数据
#[derive(Debug, Clone)]
pub struct RecallPayload {
    /// 来源端口 tag
    pub source: BusPortTag,
    /// 提取的残差向量
    pub data: Vec<f32>,
    /// 提取时的元数据
    pub meta: RecallMeta,
}

/// 召回元数据
#[derive(Debug, Clone)]
pub struct RecallMeta {
    /// 提取时的层索引
    pub layer: usize,
    /// 残差能量 (L2 范数)
    pub energy: f32,
    /// 余弦相似度 (与前一层残差)
    pub cosine_sim: f32,
    /// 熵 (从 Epilogue Telemetry 白嫖)
    pub entropy: f32,
}

/// 残差总线 — 模型级物理结构
///
/// 管理所有 Injection/Recall 端口，提供数据注入和召回的 API。
/// 端口按层索引排序，支持 O(log n) 查找。
#[derive(Debug)]
pub struct ResidualBus {
    /// 所有已注册端口，按层索引排序
    ports: Vec<BusPort>,
    /// 模型隐藏维度 (用于验证注入向量大小)
    hidden_size: usize,
    /// 模型层数
    num_layers: usize,
}

impl ResidualBus {
    /// 创建空的残差总线
    pub fn new(hidden_size: usize, num_layers: usize) -> Self {
        Self {
            ports: Vec::new(),
            hidden_size,
            num_layers,
        }
    }

    /// 注册端口 — 插入并保持按层排序
    pub fn register(&mut self, port: BusPort) {
        assert!(
            port.layer < self.num_layers,
            "BusPort layer {} exceeds num_layers {}",
            port.layer,
            self.num_layers
        );
        let pos = self.ports.partition_point(|p| p.layer <= port.layer);
        self.ports.insert(pos, port);
    }

    /// 获取指定层的所有激活端口
    pub fn active_ports_at_layer(&self, layer: usize) -> impl Iterator<Item = &BusPort> {
        self.ports.iter().filter(move |p| {
            p.layer == layer && p.is_active()
        })
    }

    /// 获取指定 tag 的端口
    pub fn find_port(&self, tag: BusPortTag) -> Option<&BusPort> {
        self.ports.iter().find(|p| p.tag == tag)
    }

    /// 获取指定 tag 的端口 (mutable)
    pub fn find_port_mut(&mut self, tag: BusPortTag) -> Option<&mut BusPort> {
        self.ports.iter_mut().find(|p| p.tag == tag)
    }

    /// 执行注入 — 将外部向量加到指定层的残差流
    ///
    /// 物理操作: `x_residual += payload.data * scale`
    pub fn inject(
        &self,
        payload: &InjectionPayload,
        residual_buffer: &mut [f32],
    ) -> Result<(), ResidualBusError> {
        let port = self.find_port(payload.target)
            .ok_or(ResidualBusError::PortNotFound(payload.target))?;

        if !port.is_active() {
            return Err(ResidualBusError::PortInactive(payload.target));
        }
        if port.kind != BusPortKind::Injection {
            return Err(ResidualBusError::WrongPortType {
                expected: BusPortKind::Injection,
                actual: port.kind,
            });
        }
        if payload.data.len() != self.hidden_size {
            return Err(ResidualBusError::DimensionMismatch {
                expected: self.hidden_size,
                actual: payload.data.len(),
            });
        }

        // 注入: residual += data * scale
        for (r, d) in residual_buffer.iter_mut().zip(payload.data.iter()) {
            *r += d * payload.scale;
        }

        Ok(())
    }

    /// 执行召回 — 从指定层提取残差数据
    ///
    /// 返回提取的残差向量和元数据
    pub fn recall(
        &self,
        tag: BusPortTag,
        residual_buffer: &[f32],
        layer: usize,
        prev_residual: Option<&[f32]>,
        entropy: f32,
    ) -> Result<RecallPayload, ResidualBusError> {
        let port = self.find_port(tag)
            .ok_or(ResidualBusError::PortNotFound(tag))?;

        if !port.is_active() {
            return Err(ResidualBusError::PortInactive(tag));
        }
        if port.kind != BusPortKind::Recall {
            return Err(ResidualBusError::WrongPortType {
                expected: BusPortKind::Recall,
                actual: port.kind,
            });
        }
        if residual_buffer.len() != self.hidden_size {
            return Err(ResidualBusError::DimensionMismatch {
                expected: self.hidden_size,
                actual: residual_buffer.len(),
            });
        }

        let energy = residual_buffer.iter().map(|x| x * x).sum::<f32>().sqrt();

        let cosine_sim = if let Some(prev) = prev_residual {
            let dot: f32 = residual_buffer.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
            let norm_prev: f32 = prev.iter().map(|x| x * x).sum::<f32>().sqrt();
            if energy > 1e-8 && norm_prev > 1e-8 {
                dot / (energy * norm_prev)
            } else {
                0.0
            }
        } else {
            1.0
        };

        Ok(RecallPayload {
            source: tag,
            data: residual_buffer.to_vec(),
            meta: RecallMeta {
                layer,
                energy,
                cosine_sim,
                entropy,
            },
        })
    }

    /// 获取所有端口
    pub fn ports(&self) -> &[BusPort] {
        &self.ports
    }

    /// 获取 hidden_size
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// 获取 num_layers
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// 统计激活端口数
    pub fn active_port_count(&self) -> usize {
        self.ports.iter().filter(|p| p.is_active()).count()
    }
}

/// 残差总线错误
#[derive(Debug)]
pub enum ResidualBusError {
    /// 端口未找到
    PortNotFound(BusPortTag),
    /// 端口未激活
    PortInactive(BusPortTag),
    /// 端口类型不匹配
    WrongPortType {
        expected: BusPortKind,
        actual: BusPortKind,
    },
    /// 维度不匹配
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
}

impl std::fmt::Display for ResidualBusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PortNotFound(tag) => write!(f, "ResidualBus port not found: {:?}", tag),
            Self::PortInactive(tag) => write!(f, "ResidualBus port inactive: {:?}", tag),
            Self::WrongPortType { expected, actual } => {
                write!(f, "ResidualBus wrong port type: expected {:?}, got {:?}", expected, actual)
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "ResidualBus dimension mismatch: expected {}, got {}", expected, actual)
            }
        }
    }
}

impl std::error::Error for ResidualBusError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residual_bus_register_and_find() {
        let mut bus = ResidualBus::new(64, 32);
        bus.register(BusPort::injection(5, BusPortTag::RagInjection));
        bus.register(BusPort::recall(20, BusPortTag::EarlyExit));
        bus.register(BusPort::recall(25, BusPortTag::IntentRecall));

        assert!(bus.find_port(BusPortTag::RagInjection).is_some());
        assert!(bus.find_port(BusPortTag::EarlyExit).is_some());
        assert!(bus.find_port(BusPortTag::IntentRecall).is_some());
        assert!(bus.find_port(BusPortTag::Guardrail).is_none());
    }

    #[test]
    fn test_residual_bus_inject() {
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::injection(2, BusPortTag::RagInjection));

        let payload = InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0, 2.0, 3.0, 4.0],
            scale: 0.5,
        };

        let mut residual = vec![10.0, 20.0, 30.0, 40.0];
        bus.inject(&payload, &mut residual).unwrap();

        assert_eq!(residual, vec![10.5, 21.0, 31.5, 42.0]);
    }

    #[test]
    fn test_residual_bus_recall() {
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::recall(3, BusPortTag::EarlyExit));

        let residual = vec![1.0, 2.0, 3.0, 4.0];
        let prev = vec![1.0, 0.0, 3.0, 0.0];
        let result = bus.recall(
            BusPortTag::EarlyExit,
            &residual,
            3,
            Some(&prev),
            2.5,
        ).unwrap();

        assert_eq!(result.source, BusPortTag::EarlyExit);
        assert_eq!(result.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result.meta.layer, 3);
        assert!(result.meta.energy > 0.0);
        assert!(result.meta.entropy > 0.0);
    }

    #[test]
    fn test_residual_bus_port_activation() {
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::injection(1, BusPortTag::Guardrail));

        let port = bus.find_port(BusPortTag::Guardrail).unwrap();
        assert!(port.is_active());

        port.deactivate();
        assert!(!port.is_active());

        // 注入应该失败
        let payload = InjectionPayload {
            target: BusPortTag::Guardrail,
            data: vec![1.0; 4],
            scale: 1.0,
        };
        let mut buf = vec![0.0; 4];
        assert!(bus.inject(&payload, &mut buf).is_err());

        port.activate();
        assert!(bus.inject(&payload, &mut buf).is_ok());
    }

    #[test]
    fn test_residual_bus_active_ports_at_layer() {
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::injection(3, BusPortTag::RagInjection));
        bus.register(BusPort::recall(3, BusPortTag::EarlyExit));
        bus.register(BusPort::recall(5, BusPortTag::IntentRecall));

        let layer3_ports: Vec<_> = bus.active_ports_at_layer(3).collect();
        assert_eq!(layer3_ports.len(), 2);

        let layer5_ports: Vec<_> = bus.active_ports_at_layer(5).collect();
        assert_eq!(layer5_ports.len(), 1);

        let layer0_ports: Vec<_> = bus.active_ports_at_layer(0).collect();
        assert_eq!(layer0_ports.len(), 0);
    }

    #[test]
    fn test_residual_bus_dimension_mismatch() {
        let mut bus = ResidualBus::new(8, 8);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));

        let payload = InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0; 4], // wrong size
            scale: 1.0,
        };
        let mut buf = vec![0.0; 8];
        assert!(matches!(
            bus.inject(&payload, &mut buf),
            Err(ResidualBusError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_residual_bus_wrong_port_type() {
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::recall(0, BusPortTag::EarlyExit)); // recall, not injection

        let payload = InjectionPayload {
            target: BusPortTag::EarlyExit,
            data: vec![1.0; 4],
            scale: 1.0,
        };
        let mut buf = vec![0.0; 4];
        assert!(matches!(
            bus.inject(&payload, &mut buf),
            Err(ResidualBusError::WrongPortType { .. })
        ));
    }

    #[test]
    fn bus_port_kind_equality() {
        assert_eq!(BusPortKind::Injection, BusPortKind::Injection);
        assert_ne!(BusPortKind::Injection, BusPortKind::Recall);
    }

    #[test]
    fn bus_port_tag_variants_distinct() {
        let tags = [
            BusPortTag::RagInjection,
            BusPortTag::EarlyExit,
            BusPortTag::IntentRecall,
            BusPortTag::Guardrail,
            BusPortTag::ShadowKv,
            BusPortTag::Custom(42),
        ];
        for i in 0..tags.len() {
            for j in (i + 1)..tags.len() {
                assert_ne!(tags[i], tags[j], "tags[{i}] == tags[{j}]");
            }
        }
    }

    #[test]
    fn bus_port_tag_copy_clone() {
        let t = BusPortTag::Custom(99);
        let t2 = t;
        assert_eq!(t, t2);
        let t3 = t.clone();
        assert_eq!(t3, BusPortTag::Custom(99));
    }

    #[test]
    fn bus_port_default_active() {
        let p = BusPort::injection(5, BusPortTag::RagInjection);
        assert!(p.is_active());
        assert_eq!(p.kind, BusPortKind::Injection);
        assert_eq!(p.layer, 5);
        assert_eq!(p.tag, BusPortTag::RagInjection);
    }

    #[test]
    fn bus_port_recall_constructor() {
        let p = BusPort::recall(10, BusPortTag::IntentRecall);
        assert_eq!(p.kind, BusPortKind::Recall);
        assert_eq!(p.layer, 10);
        assert!(p.is_active());
    }

    #[test]
    fn bus_port_activate_deactivate_cycle() {
        let p = BusPort::injection(0, BusPortTag::Guardrail);
        assert!(p.is_active());
        p.deactivate();
        assert!(!p.is_active());
        p.deactivate(); // idempotent
        assert!(!p.is_active());
        p.activate();
        assert!(p.is_active());
    }

    #[test]
    fn residual_bus_accessors() {
        let bus = ResidualBus::new(128, 24);
        assert_eq!(bus.hidden_size(), 128);
        assert_eq!(bus.num_layers(), 24);
        assert!(bus.ports().is_empty());
    }

    #[test]
    fn residual_bus_register_preserves_order() {
        let mut bus = ResidualBus::new(4, 10);
        bus.register(BusPort::injection(5, BusPortTag::RagInjection));
        bus.register(BusPort::recall(2, BusPortTag::EarlyExit));
        bus.register(BusPort::injection(8, BusPortTag::Guardrail));

        let layers: Vec<usize> = bus.ports().iter().map(|p| p.layer).collect();
        // partition_point inserts after all <=, so order by insertion but sorted by layer
        // First insert: [5], Second: [2,5], Third: [2,5,8]
        assert_eq!(layers, vec![2, 5, 8]);
    }

    #[test]
    fn residual_bus_active_port_count() {
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::injection(1, BusPortTag::RagInjection));
        bus.register(BusPort::recall(2, BusPortTag::EarlyExit));
        assert_eq!(bus.active_port_count(), 2);

        bus.find_port(BusPortTag::RagInjection).unwrap().deactivate();
        assert_eq!(bus.active_port_count(), 1);
    }

    #[test]
    fn residual_bus_find_port_mut() {
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::injection(1, BusPortTag::RagInjection));

        let port = bus.find_port_mut(BusPortTag::RagInjection).unwrap();
        port.deactivate();
        assert!(!bus.find_port(BusPortTag::RagInjection).unwrap().is_active());
    }

    #[test]
    fn residual_bus_inject_scale_zero() {
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));

        let payload = InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![10.0, 20.0, 30.0, 40.0],
            scale: 0.0,
        };
        let mut buf = vec![1.0, 2.0, 3.0, 4.0];
        bus.inject(&payload, &mut buf).unwrap();
        assert_eq!(buf, vec![1.0, 2.0, 3.0, 4.0]); // scale=0 → no change
    }

    #[test]
    fn residual_bus_recall_no_prev_cosine_is_one() {
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::recall(3, BusPortTag::EarlyExit));

        let residual = vec![1.0, 2.0, 3.0, 4.0];
        let result = bus.recall(BusPortTag::EarlyExit, &residual, 3, None, 0.0).unwrap();
        assert!((result.meta.cosine_sim - 1.0).abs() < 1e-5); // no prev → 1.0
    }

    #[test]
    fn residual_bus_recall_zero_energy_cosine_is_zero() {
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::recall(0, BusPortTag::EarlyExit));

        let residual = vec![0.0, 0.0, 0.0, 0.0];
        let prev = vec![1.0, 2.0, 3.0, 4.0];
        let result = bus.recall(BusPortTag::EarlyExit, &residual, 0, Some(&prev), 0.0).unwrap();
        assert!((result.meta.energy).abs() < 1e-5);
        assert!((result.meta.cosine_sim).abs() < 1e-5); // zero vector → cosine = 0
    }

    #[test]
    fn residual_bus_error_display() {
        let e = ResidualBusError::PortNotFound(BusPortTag::RagInjection);
        assert!(e.to_string().contains("not found"));

        let e = ResidualBusError::PortInactive(BusPortTag::Guardrail);
        assert!(e.to_string().contains("inactive"));

        let e = ResidualBusError::WrongPortType {
            expected: BusPortKind::Injection,
            actual: BusPortKind::Recall,
        };
        assert!(e.to_string().contains("wrong port type"));

        let e = ResidualBusError::DimensionMismatch { expected: 4, actual: 8 };
        assert!(e.to_string().contains("4"));
        assert!(e.to_string().contains("8"));
    }

    #[test]
    fn residual_bus_inject_port_not_found() {
        let bus = ResidualBus::new(4, 8);
        let payload = InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0; 4],
            scale: 1.0,
        };
        let mut buf = vec![0.0; 4];
        assert!(matches!(bus.inject(&payload, &mut buf), Err(ResidualBusError::PortNotFound(_))));
    }

    #[test]
    fn injection_payload_clone() {
        let p = InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0, 2.0],
            scale: 0.5,
        };
        let p2 = p.clone();
        assert_eq!(p2.target, BusPortTag::RagInjection);
        assert_eq!(p2.data, vec![1.0, 2.0]);
        assert!((p2.scale - 0.5).abs() < 1e-6);
    }

    #[test]
    fn recall_payload_clone() {
        let p = RecallPayload {
            source: BusPortTag::EarlyExit,
            data: vec![1.0, 2.0],
            meta: RecallMeta { layer: 3, energy: 5.0, cosine_sim: 0.9, entropy: 2.1 },
        };
        let p2 = p.clone();
        assert_eq!(p2.source, BusPortTag::EarlyExit);
        assert_eq!(p2.meta.layer, 3);
    }

    // --- New tests ---

    #[test]
    fn bus_port_kind_debug_format() {
        let injection = BusPortKind::Injection;
        let recall = BusPortKind::Recall;
        let s_injection = format!("{:?}", injection);
        let s_recall = format!("{:?}", recall);
        assert!(s_injection.contains("Injection"), "Debug for Injection should contain 'Injection'");
        assert!(s_recall.contains("Recall"), "Debug for Recall should contain 'Recall'");
    }

    #[test]
    fn bus_port_tag_debug_format_all_variants() {
        let variants: Vec<(BusPortTag, &str)> = vec![
            (BusPortTag::RagInjection, "RagInjection"),
            (BusPortTag::EarlyExit, "EarlyExit"),
            (BusPortTag::IntentRecall, "IntentRecall"),
            (BusPortTag::Guardrail, "Guardrail"),
            (BusPortTag::ShadowKv, "ShadowKv"),
            (BusPortTag::Custom(7), "Custom"),
        ];
        for (tag, expected_substr) in variants {
            let s = format!("{:?}", tag);
            assert!(
                s.contains(expected_substr),
                "Debug for {:?} should contain '{}'",
                tag,
                expected_substr
            );
        }
    }

    #[test]
    fn bus_port_tag_custom_same_value_equal() {
        let a = BusPortTag::Custom(42);
        let b = BusPortTag::Custom(42);
        assert_eq!(a, b);
    }

    #[test]
    fn bus_port_tag_custom_different_value_not_equal() {
        let a = BusPortTag::Custom(1);
        let b = BusPortTag::Custom(2);
        assert_ne!(a, b);
    }

    #[test]
    fn bus_port_tag_custom_zero() {
        let t = BusPortTag::Custom(0);
        assert_eq!(t, BusPortTag::Custom(0));
        assert_ne!(t, BusPortTag::Custom(1));
    }

    #[test]
    fn bus_port_debug_format() {
        let p = BusPort::injection(3, BusPortTag::RagInjection);
        let s = format!("{:?}", p);
        // BusPort has kind, layer, tag fields; just verify it doesn't panic and contains expected info
        assert!(s.contains("injection") || s.contains("Injection") || s.contains("BusPort"));
    }

    #[test]
    fn residual_bus_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(ResidualBusError::PortNotFound(BusPortTag::ShadowKv));
        assert!(e.to_string().contains("not found"));
    }

    #[test]
    fn residual_bus_error_debug_format() {
        let e = ResidualBusError::PortNotFound(BusPortTag::Guardrail);
        let s = format!("{:?}", e);
        assert!(s.contains("PortNotFound"), "Debug should contain variant name");

        let e2 = ResidualBusError::DimensionMismatch { expected: 64, actual: 128 };
        let s2 = format!("{:?}", e2);
        assert!(s2.contains("64"));
        assert!(s2.contains("128"));
    }

    #[test]
    fn recall_meta_debug_and_clone() {
        let m = RecallMeta {
            layer: 7,
            energy: 3.14,
            cosine_sim: 0.95,
            entropy: 1.5,
        };
        let m2 = m.clone();
        assert_eq!(m2.layer, 7);
        assert!((m2.energy - 3.14).abs() < 1e-6);
        assert!((m2.cosine_sim - 0.95).abs() < 1e-6);
        assert!((m2.entropy - 1.5).abs() < 1e-6);

        let debug_s = format!("{:?}", m);
        assert!(debug_s.contains("RecallMeta"));
    }

    #[test]
    fn recall_payload_debug_format() {
        let p = RecallPayload {
            source: BusPortTag::IntentRecall,
            data: vec![1.0, 2.0, 3.0],
            meta: RecallMeta { layer: 0, energy: 0.0, cosine_sim: 1.0, entropy: 0.0 },
        };
        let s = format!("{:?}", p);
        assert!(s.contains("RecallPayload"));
    }

    #[test]
    fn injection_payload_debug_format() {
        let p = InjectionPayload {
            target: BusPortTag::Guardrail,
            data: vec![0.5, 1.5],
            scale: 2.0,
        };
        let s = format!("{:?}", p);
        assert!(s.contains("InjectionPayload"));
    }

    #[test]
    fn residual_bus_debug_format() {
        let bus = ResidualBus::new(64, 12);
        let s = format!("{:?}", bus);
        assert!(s.contains("ResidualBus"));
    }

    #[test]
    fn residual_bus_register_at_layer_boundary() {
        let mut bus = ResidualBus::new(4, 3);
        // layer 0 (minimum)
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));
        // layer 2 (maximum valid = num_layers - 1)
        bus.register(BusPort::recall(2, BusPortTag::EarlyExit));
        assert_eq!(bus.ports().len(), 2);
        assert_eq!(bus.ports()[0].layer, 0);
        assert_eq!(bus.ports()[1].layer, 2);
    }

    #[test]
    #[should_panic(expected = "exceeds num_layers")]
    fn residual_bus_register_layer_out_of_bounds_panics() {
        let mut bus = ResidualBus::new(4, 4);
        bus.register(BusPort::injection(4, BusPortTag::RagInjection)); // layer == num_layers, out of bounds
    }

    #[test]
    fn residual_bus_inject_negative_scale() {
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));

        let payload = InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0, 2.0, 3.0, 4.0],
            scale: -1.0,
        };
        let mut buf = vec![10.0, 20.0, 30.0, 40.0];
        bus.inject(&payload, &mut buf).unwrap();
        assert_eq!(buf, vec![9.0, 18.0, 27.0, 36.0]); // 10 + 1*(-1), 20 + 2*(-1), ...
    }

    #[test]
    fn residual_bus_inject_negative_values() {
        let mut bus = ResidualBus::new(3, 8);
        bus.register(BusPort::injection(1, BusPortTag::Guardrail));

        let payload = InjectionPayload {
            target: BusPortTag::Guardrail,
            data: vec![-1.0, -2.0, -3.0],
            scale: 1.0,
        };
        let mut buf = vec![5.0, 5.0, 5.0];
        bus.inject(&payload, &mut buf).unwrap();
        assert_eq!(buf, vec![4.0, 3.0, 2.0]);
    }

    #[test]
    fn residual_bus_inject_accumulates() {
        let mut bus = ResidualBus::new(2, 8);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));

        let mut buf = vec![0.0, 0.0];

        let p1 = InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0, 1.0],
            scale: 1.0,
        };
        bus.inject(&p1, &mut buf).unwrap();
        assert_eq!(buf, vec![1.0, 1.0]);

        let p2 = InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![2.0, 3.0],
            scale: 0.5,
        };
        bus.inject(&p2, &mut buf).unwrap();
        assert_eq!(buf, vec![2.0, 2.5]); // 1+2*0.5, 1+3*0.5
    }

    #[test]
    fn residual_bus_recall_orthogonal_vectors_cosine_zero() {
        let mut bus = ResidualBus::new(2, 8);
        bus.register(BusPort::recall(0, BusPortTag::IntentRecall));

        // (1,0) dot (0,1) = 0; both have norm 1
        let residual = vec![1.0, 0.0];
        let prev = vec![0.0, 1.0];
        let result = bus.recall(BusPortTag::IntentRecall, &residual, 0, Some(&prev), 0.0).unwrap();
        assert!((result.meta.cosine_sim).abs() < 1e-5, "orthogonal vectors should have cosine ~0, got {}", result.meta.cosine_sim);
    }

    #[test]
    fn residual_bus_recall_parallel_vectors_cosine_one() {
        let mut bus = ResidualBus::new(3, 8);
        bus.register(BusPort::recall(0, BusPortTag::IntentRecall));

        let residual = vec![2.0, 4.0, 6.0];
        let prev = vec![1.0, 2.0, 3.0]; // same direction, different magnitude
        let result = bus.recall(BusPortTag::IntentRecall, &residual, 0, Some(&prev), 0.0).unwrap();
        assert!((result.meta.cosine_sim - 1.0).abs() < 1e-5, "parallel vectors should have cosine ~1, got {}", result.meta.cosine_sim);
    }

    #[test]
    fn residual_bus_recall_with_entropy() {
        let mut bus = ResidualBus::new(2, 8);
        bus.register(BusPort::recall(1, BusPortTag::ShadowKv));

        let residual = vec![1.0, 0.0];
        let result = bus.recall(BusPortTag::ShadowKv, &residual, 1, None, 4.2).unwrap();
        assert!((result.meta.entropy - 4.2).abs() < 1e-6);
        assert_eq!(result.meta.layer, 1);
    }

    #[test]
    fn residual_bus_recall_dimension_mismatch() {
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::recall(0, BusPortTag::EarlyExit));

        let short_residual = vec![1.0, 2.0]; // only 2, expected 4
        let result = bus.recall(BusPortTag::EarlyExit, &short_residual, 0, None, 0.0);
        assert!(matches!(result, Err(ResidualBusError::DimensionMismatch { expected: 4, actual: 2 })));
    }

    #[test]
    fn residual_bus_recall_wrong_port_type() {
        let mut bus = ResidualBus::new(4, 8);
        // Register an injection port, then try to recall from it
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));

        let residual = vec![1.0, 2.0, 3.0, 4.0];
        let result = bus.recall(BusPortTag::RagInjection, &residual, 0, None, 0.0);
        assert!(matches!(result, Err(ResidualBusError::WrongPortType { .. })));
    }

    #[test]
    fn residual_bus_register_multiple_ports_same_layer() {
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::injection(3, BusPortTag::RagInjection));
        bus.register(BusPort::recall(3, BusPortTag::EarlyExit));
        bus.register(BusPort::recall(3, BusPortTag::Guardrail));

        let at_layer: Vec<_> = bus.active_ports_at_layer(3).collect();
        assert_eq!(at_layer.len(), 3);
        // all at layer 3
        assert!(at_layer.iter().all(|p| p.layer == 3));
    }

    #[test]
    fn residual_bus_empty_no_ports_operations() {
        let bus = ResidualBus::new(8, 4);
        assert_eq!(bus.active_port_count(), 0);
        assert!(bus.ports().is_empty());
        assert_eq!(bus.active_ports_at_layer(0).count(), 0);
        assert!(bus.find_port(BusPortTag::RagInjection).is_none());
    }

    // --- Additional coverage tests ---

    #[test]
    fn bus_port_kind_copy_clone() {
        let a = BusPortKind::Injection;
        let b = a; // Copy
        assert_eq!(a, b);
        let c = a.clone();
        assert_eq!(c, BusPortKind::Injection);

        let x = BusPortKind::Recall;
        let y = x;
        assert_eq!(x, y);
    }

    #[test]
    fn bus_port_kind_equality_exhaustive() {
        let variants = [BusPortKind::Injection, BusPortKind::Recall];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn bus_port_tag_custom_max_u32() {
        let t = BusPortTag::Custom(u32::MAX);
        assert_eq!(t, BusPortTag::Custom(u32::MAX));
        assert_ne!(t, BusPortTag::Custom(0));
        assert_ne!(t, BusPortTag::RagInjection);
    }

    #[test]
    fn bus_port_injection_at_layer_zero() {
        let p = BusPort::injection(0, BusPortTag::RagInjection);
        assert_eq!(p.kind, BusPortKind::Injection);
        assert_eq!(p.layer, 0);
        assert_eq!(p.tag, BusPortTag::RagInjection);
        assert!(p.is_active());
    }

    #[test]
    fn bus_port_recall_at_layer_zero() {
        let p = BusPort::recall(0, BusPortTag::EarlyExit);
        assert_eq!(p.kind, BusPortKind::Recall);
        assert_eq!(p.layer, 0);
        assert_eq!(p.tag, BusPortTag::EarlyExit);
        assert!(p.is_active());
    }

    #[test]
    fn bus_port_all_tag_variants_as_injection() {
        let tags = [
            BusPortTag::RagInjection,
            BusPortTag::EarlyExit,
            BusPortTag::IntentRecall,
            BusPortTag::Guardrail,
            BusPortTag::ShadowKv,
            BusPortTag::Custom(100),
        ];
        for (i, tag) in tags.iter().enumerate() {
            let p = BusPort::injection(i, *tag);
            assert_eq!(p.kind, BusPortKind::Injection);
            assert_eq!(p.layer, i);
            assert_eq!(p.tag, *tag);
        }
    }

    #[test]
    fn bus_port_all_tag_variants_as_recall() {
        let tags = [
            BusPortTag::RagInjection,
            BusPortTag::EarlyExit,
            BusPortTag::IntentRecall,
            BusPortTag::Guardrail,
            BusPortTag::ShadowKv,
            BusPortTag::Custom(200),
        ];
        for (i, tag) in tags.iter().enumerate() {
            let p = BusPort::recall(i, *tag);
            assert_eq!(p.kind, BusPortKind::Recall);
            assert_eq!(p.layer, i);
            assert_eq!(p.tag, *tag);
        }
    }

    #[test]
    fn injection_payload_construction_fields() {
        let p = InjectionPayload {
            target: BusPortTag::ShadowKv,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            scale: 2.5,
        };
        assert_eq!(p.target, BusPortTag::ShadowKv);
        assert_eq!(p.data.len(), 5);
        assert_eq!(p.data[2], 3.0);
        assert!((p.scale - 2.5).abs() < 1e-6);
    }

    #[test]
    fn injection_payload_empty_data() {
        let p = InjectionPayload {
            target: BusPortTag::Custom(0),
            data: vec![],
            scale: 1.0,
        };
        assert!(p.data.is_empty());
        assert_eq!(p.target, BusPortTag::Custom(0));
    }

    #[test]
    fn recall_meta_zero_values() {
        let m = RecallMeta {
            layer: 0,
            energy: 0.0,
            cosine_sim: 0.0,
            entropy: 0.0,
        };
        assert_eq!(m.layer, 0);
        assert_eq!(m.energy, 0.0);
        assert_eq!(m.cosine_sim, 0.0);
        assert_eq!(m.entropy, 0.0);
    }

    #[test]
    fn recall_meta_negative_values() {
        let m = RecallMeta {
            layer: 5,
            energy: -1.5,
            cosine_sim: -0.99,
            entropy: -3.0,
        };
        let m2 = m.clone();
        assert!((m2.energy - (-1.5)).abs() < 1e-6);
        assert!((m2.cosine_sim - (-0.99)).abs() < 1e-6);
        assert!((m2.entropy - (-3.0)).abs() < 1e-6);
    }

    #[test]
    fn recall_payload_field_access() {
        let p = RecallPayload {
            source: BusPortTag::ShadowKv,
            data: vec![10.0, 20.0],
            meta: RecallMeta {
                layer: 99,
                energy: 42.0,
                cosine_sim: 0.75,
                entropy: 3.14,
            },
        };
        assert_eq!(p.source, BusPortTag::ShadowKv);
        assert_eq!(p.data.len(), 2);
        assert_eq!(p.meta.layer, 99);
        assert!((p.meta.energy - 42.0).abs() < 1e-6);
        assert!((p.meta.cosine_sim - 0.75).abs() < 1e-6);
        assert!((p.meta.entropy - 3.14).abs() < 1e-6);
    }

    #[test]
    fn residual_bus_zero_hidden_size() {
        let bus = ResidualBus::new(0, 8);
        assert_eq!(bus.hidden_size(), 0);
        assert_eq!(bus.num_layers(), 8);
        assert!(bus.ports().is_empty());
    }

    #[test]
    #[should_panic(expected = "exceeds num_layers")]
    fn residual_bus_register_layer_zero_with_zero_layers_panics() {
        let mut bus = ResidualBus::new(4, 0);
        // num_layers=0 means no valid layer; layer 0 >= 0 triggers the assert
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));
    }

    #[test]
    fn residual_bus_register_max_valid_layer() {
        let mut bus = ResidualBus::new(4, 100);
        let max_layer = 99; // num_layers - 1
        bus.register(BusPort::injection(max_layer, BusPortTag::RagInjection));
        assert_eq!(bus.ports().len(), 1);
        assert_eq!(bus.ports()[0].layer, 99);
    }

    #[test]
    #[should_panic(expected = "exceeds num_layers")]
    fn residual_bus_register_layer_one_past_max_panics() {
        let mut bus = ResidualBus::new(4, 100);
        bus.register(BusPort::injection(100, BusPortTag::RagInjection)); // 100 == num_layers
    }

    #[test]
    fn residual_bus_active_ports_at_layer_with_deactivated() {
        let mut bus = ResidualBus::new(4, 10);
        bus.register(BusPort::injection(3, BusPortTag::RagInjection));
        bus.register(BusPort::recall(3, BusPortTag::EarlyExit));
        bus.register(BusPort::recall(3, BusPortTag::IntentRecall));

        // All active initially
        assert_eq!(bus.active_ports_at_layer(3).count(), 3);

        // Deactivate one
        bus.find_port(BusPortTag::EarlyExit).unwrap().deactivate();
        let active: Vec<_> = bus.active_ports_at_layer(3).collect();
        assert_eq!(active.len(), 2);
        assert!(active.iter().all(|p| p.is_active()));

        // Deactivate all
        bus.find_port(BusPortTag::RagInjection).unwrap().deactivate();
        bus.find_port(BusPortTag::IntentRecall).unwrap().deactivate();
        assert_eq!(bus.active_ports_at_layer(3).count(), 0);

        // Ports still exist, just inactive
        assert_eq!(bus.ports().len(), 3);
    }

    #[test]
    fn residual_bus_register_sorting_many_ports() {
        let mut bus = ResidualBus::new(4, 100);
        // Register out-of-order
        bus.register(BusPort::injection(50, BusPortTag::RagInjection));
        bus.register(BusPort::recall(10, BusPortTag::EarlyExit));
        bus.register(BusPort::injection(30, BusPortTag::Guardrail));
        bus.register(BusPort::recall(10, BusPortTag::ShadowKv));
        bus.register(BusPort::recall(99, BusPortTag::IntentRecall));

        let layers: Vec<usize> = bus.ports().iter().map(|p| p.layer).collect();
        // Should be non-decreasing
        for window in layers.windows(2) {
            assert!(window[0] <= window[1], "ports not sorted: {:?}", layers);
        }
        assert_eq!(layers.len(), 5);
    }

    #[test]
    fn residual_bus_recall_port_not_found() {
        let bus = ResidualBus::new(4, 8);
        let residual = vec![1.0, 2.0, 3.0, 4.0];
        let result = bus.recall(BusPortTag::EarlyExit, &residual, 0, None, 0.0);
        assert!(matches!(result, Err(ResidualBusError::PortNotFound(_))));
    }

    // --- Wave 2: Additional 42 tests ---

    #[test]
    fn bus_port_tag_hash_consistency() {
        // Same tag produces same hash
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(BusPortTag::Custom(42), "first");
        map.insert(BusPortTag::Custom(42), "second");
        assert_eq!(map.get(&BusPortTag::Custom(42)), Some(&"second"));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn bus_port_tag_distinct_hash_keys() {
        use std::collections::HashSet;
        let set: HashSet<BusPortTag> = [
            BusPortTag::RagInjection,
            BusPortTag::EarlyExit,
            BusPortTag::IntentRecall,
            BusPortTag::Guardrail,
            BusPortTag::ShadowKv,
            BusPortTag::Custom(1),
            BusPortTag::Custom(2),
        ].into_iter().collect();
        assert_eq!(set.len(), 7);
    }

    #[test]
    fn residual_bus_register_same_layer_descending_order() {
        // Register ports at same layer in reverse tag order; all should appear
        let mut bus = ResidualBus::new(4, 10);
        bus.register(BusPort::recall(5, BusPortTag::ShadowKv));
        bus.register(BusPort::recall(5, BusPortTag::Guardrail));
        bus.register(BusPort::injection(5, BusPortTag::EarlyExit));
        let at_5: Vec<_> = bus.active_ports_at_layer(5).collect();
        assert_eq!(at_5.len(), 3);
        assert!(at_5.iter().all(|p| p.layer == 5));
    }

    #[test]
    fn residual_bus_register_six_ports_all_found() {
        let mut bus = ResidualBus::new(4, 20);
        let tags = [
            BusPortTag::RagInjection,
            BusPortTag::EarlyExit,
            BusPortTag::IntentRecall,
            BusPortTag::Guardrail,
            BusPortTag::ShadowKv,
            BusPortTag::Custom(7),
        ];
        for (i, &tag) in tags.iter().enumerate() {
            bus.register(BusPort::injection(i, tag));
        }
        assert_eq!(bus.ports().len(), 6);
        for &tag in &tags {
            assert!(bus.find_port(tag).is_some(), "tag {:?} not found", tag);
        }
    }

    #[test]
    fn residual_bus_inject_large_scale() {
        let mut bus = ResidualBus::new(3, 8);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));

        let payload = InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![0.01, 0.01, 0.01],
            scale: 1000.0,
        };
        let mut buf = vec![0.0, 0.0, 0.0];
        bus.inject(&payload, &mut buf).unwrap();
        assert!((buf[0] - 10.0).abs() < 1e-3);
        assert!((buf[1] - 10.0).abs() < 1e-3);
        assert!((buf[2] - 10.0).abs() < 1e-3);
    }

    #[test]
    fn residual_bus_inject_very_small_scale() {
        let mut bus = ResidualBus::new(2, 8);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));

        let payload = InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1e6, 1e6],
            scale: 1e-9,
        };
        let mut buf = vec![1.0, 1.0];
        bus.inject(&payload, &mut buf).unwrap();
        // 1.0 + 1e6 * 1e-9 = 1.0 + 1e-3 = 1.001
        assert!((buf[0] - 1.001).abs() < 1e-6);
    }

    #[test]
    fn residual_bus_inject_mixed_sign_data() {
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::injection(0, BusPortTag::Guardrail));

        let payload = InjectionPayload {
            target: BusPortTag::Guardrail,
            data: vec![1.0, -1.0, 2.0, -2.0],
            scale: 1.0,
        };
        let mut buf = vec![5.0, 5.0, 5.0, 5.0];
        bus.inject(&payload, &mut buf).unwrap();
        assert_eq!(buf, vec![6.0, 4.0, 7.0, 3.0]);
    }

    #[test]
    fn residual_bus_inject_three_rounds_accumulation() {
        let mut bus = ResidualBus::new(2, 8);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));

        let mut buf = vec![0.0, 0.0];
        for i in 1..=3 {
            let payload = InjectionPayload {
                target: BusPortTag::RagInjection,
                data: vec![1.0, 1.0],
                scale: 1.0,
            };
            bus.inject(&payload, &mut buf).unwrap();
            assert!((buf[0] - i as f32).abs() < 1e-6);
        }
        assert_eq!(buf, vec![3.0, 3.0]);
    }

    #[test]
    fn residual_bus_inject_does_not_affect_different_bus() {
        let mut bus1 = ResidualBus::new(2, 8);
        let mut bus2 = ResidualBus::new(2, 8);
        bus1.register(BusPort::injection(0, BusPortTag::RagInjection));
        bus2.register(BusPort::injection(0, BusPortTag::RagInjection));

        let mut buf1 = vec![0.0, 0.0];
        let buf2 = vec![0.0, 0.0];

        let payload = InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![5.0, 5.0],
            scale: 1.0,
        };
        bus1.inject(&payload, &mut buf1).unwrap();
        assert_eq!(buf1, vec![5.0, 5.0]);
        assert_eq!(buf2, vec![0.0, 0.0]);
    }

    #[test]
    fn residual_bus_recall_energy_calculation_known_values() {
        let mut bus = ResidualBus::new(3, 8);
        bus.register(BusPort::recall(0, BusPortTag::IntentRecall));

        // [3.0, 4.0, 0.0] → energy = sqrt(9+16+0) = 5.0
        let residual = vec![3.0, 4.0, 0.0];
        let result = bus.recall(BusPortTag::IntentRecall, &residual, 0, None, 0.0).unwrap();
        assert!((result.meta.energy - 5.0).abs() < 1e-5, "energy should be 5.0, got {}", result.meta.energy);
    }

    #[test]
    fn residual_bus_recall_energy_unit_vector() {
        let mut bus = ResidualBus::new(2, 8);
        bus.register(BusPort::recall(0, BusPortTag::IntentRecall));

        let residual = vec![1.0, 0.0];
        let result = bus.recall(BusPortTag::IntentRecall, &residual, 0, None, 0.0).unwrap();
        assert!((result.meta.energy - 1.0).abs() < 1e-5);
    }

    #[test]
    fn residual_bus_recall_anti_parallel_cosine_minus_one() {
        let mut bus = ResidualBus::new(2, 8);
        bus.register(BusPort::recall(0, BusPortTag::IntentRecall));

        let residual = vec![1.0, 0.0];
        let prev = vec![-1.0, 0.0]; // opposite direction
        let result = bus.recall(BusPortTag::IntentRecall, &residual, 0, Some(&prev), 0.0).unwrap();
        assert!((result.meta.cosine_sim - (-1.0)).abs() < 1e-5,
            "anti-parallel should have cosine ~-1, got {}", result.meta.cosine_sim);
    }

    #[test]
    fn residual_bus_recall_prev_zero_vector_cosine_zero() {
        let mut bus = ResidualBus::new(2, 8);
        bus.register(BusPort::recall(0, BusPortTag::EarlyExit));

        let residual = vec![1.0, 2.0];
        let prev = vec![0.0, 0.0]; // zero prev → cosine = 0
        let result = bus.recall(BusPortTag::EarlyExit, &residual, 0, Some(&prev), 0.0).unwrap();
        assert!((result.meta.cosine_sim).abs() < 1e-5);
    }

    #[test]
    fn residual_bus_recall_data_is_copy() {
        let mut bus = ResidualBus::new(3, 8);
        bus.register(BusPort::recall(0, BusPortTag::IntentRecall));

        let residual = vec![1.0, 2.0, 3.0];
        let result = bus.recall(BusPortTag::IntentRecall, &residual, 0, None, 0.0).unwrap();
        // Returned data is an owned copy, independent of the original residual buffer
        let mut returned_data = result.data;
        returned_data[0] = 99.0;
        assert_eq!(residual[0], 1.0); // original unchanged
    }

    #[test]
    fn residual_bus_recall_with_large_entropy() {
        let mut bus = ResidualBus::new(2, 8);
        bus.register(BusPort::recall(0, BusPortTag::ShadowKv));

        let residual = vec![1.0, 0.0];
        let result = bus.recall(BusPortTag::ShadowKv, &residual, 0, None, 100.0).unwrap();
        assert!((result.meta.entropy - 100.0).abs() < 1e-5);
    }

    #[test]
    fn residual_bus_recall_negative_entropy() {
        let mut bus = ResidualBus::new(2, 8);
        bus.register(BusPort::recall(0, BusPortTag::Guardrail));

        let residual = vec![1.0, 0.0];
        let result = bus.recall(BusPortTag::Guardrail, &residual, 0, None, -5.5).unwrap();
        assert!((result.meta.entropy - (-5.5)).abs() < 1e-5);
    }

    #[test]
    fn residual_bus_inactive_port_inject_returns_inactive_error() {
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));
        bus.find_port(BusPortTag::RagInjection).unwrap().deactivate();

        let payload = InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0; 4],
            scale: 1.0,
        };
        let mut buf = vec![0.0; 4];
        let err = bus.inject(&payload, &mut buf).unwrap_err();
        assert!(matches!(err, ResidualBusError::PortInactive(BusPortTag::RagInjection)));
    }

    #[test]
    fn residual_bus_inactive_port_recall_returns_inactive_error() {
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::recall(0, BusPortTag::EarlyExit));
        bus.find_port(BusPortTag::EarlyExit).unwrap().deactivate();

        let residual = vec![1.0; 4];
        let err = bus.recall(BusPortTag::EarlyExit, &residual, 0, None, 0.0).unwrap_err();
        assert!(matches!(err, ResidualBusError::PortInactive(BusPortTag::EarlyExit)));
    }

    #[test]
    fn residual_bus_inject_with_no_registered_ports() {
        let bus = ResidualBus::new(4, 8);
        let payload = InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0; 4],
            scale: 1.0,
        };
        let mut buf = vec![0.0; 4];
        assert!(matches!(bus.inject(&payload, &mut buf), Err(ResidualBusError::PortNotFound(_))));
        assert_eq!(buf, vec![0.0; 4]); // unchanged
    }

    #[test]
    fn residual_bus_find_port_mut_changes_reflection() {
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::injection(3, BusPortTag::RagInjection));
        bus.register(BusPort::recall(3, BusPortTag::EarlyExit));

        // Deactivate via mutable find
        bus.find_port_mut(BusPortTag::EarlyExit).unwrap().deactivate();
        assert_eq!(bus.active_port_count(), 1);

        // Reactivate via mutable find
        bus.find_port_mut(BusPortTag::EarlyExit).unwrap().activate();
        assert_eq!(bus.active_port_count(), 2);
    }

    #[test]
    fn residual_bus_find_port_mut_nonexistent_returns_none() {
        let mut bus = ResidualBus::new(4, 8);
        assert!(bus.find_port_mut(BusPortTag::RagInjection).is_none());
    }

    #[test]
    fn bus_port_activate_deactivate_repeated_cycles() {
        let p = BusPort::injection(0, BusPortTag::Guardrail);
        for _ in 0..10 {
            p.deactivate();
            assert!(!p.is_active());
            p.activate();
            assert!(p.is_active());
        }
    }

    #[test]
    fn bus_port_double_activate_idempotent() {
        let p = BusPort::recall(5, BusPortTag::ShadowKv);
        p.activate();
        p.activate();
        assert!(p.is_active());
    }

    #[test]
    fn injection_payload_scale_one_identity() {
        let mut bus = ResidualBus::new(3, 8);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));

        let data = vec![1.0, 2.0, 3.0];
        let payload = InjectionPayload {
            target: BusPortTag::RagInjection,
            data: data.clone(),
            scale: 1.0,
        };
        let mut buf = vec![0.0, 0.0, 0.0];
        bus.inject(&payload, &mut buf).unwrap();
        // buf = 0 + data * 1.0 = data
        for i in 0..3 {
            assert!((buf[i] - data[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn injection_payload_large_data_vector() {
        let mut bus = ResidualBus::new(1024, 8);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));

        let data = vec![1.0; 1024];
        let payload = InjectionPayload {
            target: BusPortTag::RagInjection,
            data: data.clone(),
            scale: 2.0,
        };
        let mut buf = vec![0.0; 1024];
        bus.inject(&payload, &mut buf).unwrap();
        assert!(buf.iter().all(|&v| (v - 2.0).abs() < 1e-6));
    }

    #[test]
    fn residual_bus_new_accessors_consistency() {
        for &(hs, nl) in &[(0, 0), (1, 1), (64, 32), (4096, 128)] {
            let bus = ResidualBus::new(hs, nl);
            assert_eq!(bus.hidden_size(), hs);
            assert_eq!(bus.num_layers(), nl);
            assert!(bus.ports().is_empty());
            assert_eq!(bus.active_port_count(), 0);
        }
    }

    #[test]
    fn residual_bus_ports_returns_all_registered() {
        let mut bus = ResidualBus::new(4, 10);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));
        bus.register(BusPort::recall(5, BusPortTag::EarlyExit));
        bus.register(BusPort::injection(9, BusPortTag::Guardrail));

        assert_eq!(bus.ports().len(), 3);
        assert_eq!(bus.ports()[0].tag, BusPortTag::RagInjection);
        assert_eq!(bus.ports()[1].tag, BusPortTag::EarlyExit);
        assert_eq!(bus.ports()[2].tag, BusPortTag::Guardrail);
    }

    #[test]
    fn residual_bus_active_ports_empty_for_unregistered_layer() {
        let mut bus = ResidualBus::new(4, 10);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));
        bus.register(BusPort::recall(9, BusPortTag::EarlyExit));

        assert_eq!(bus.active_ports_at_layer(5).count(), 0);
        assert_eq!(bus.active_ports_at_layer(1).count(), 0);
    }

    #[test]
    fn residual_bus_error_display_port_inactive() {
        let e = ResidualBusError::PortInactive(BusPortTag::ShadowKv);
        let s = e.to_string();
        assert!(s.contains("inactive"));
        assert!(s.contains("ShadowKv"));
    }

    #[test]
    fn residual_bus_error_display_wrong_port_type() {
        let e = ResidualBusError::WrongPortType {
            expected: BusPortKind::Recall,
            actual: BusPortKind::Injection,
        };
        let s = e.to_string();
        assert!(s.contains("wrong port type"));
        assert!(s.contains("Recall"));
        assert!(s.contains("Injection"));
    }

    #[test]
    fn residual_bus_error_display_dimension_mismatch() {
        let e = ResidualBusError::DimensionMismatch { expected: 512, actual: 256 };
        let s = e.to_string();
        assert!(s.contains("512"));
        assert!(s.contains("256"));
    }

    #[test]
    fn recall_meta_large_values() {
        let m = RecallMeta {
            layer: usize::MAX,
            energy: f32::MAX,
            cosine_sim: -1.0,
            entropy: f32::MIN,
        };
        let m2 = m.clone();
        assert_eq!(m2.layer, usize::MAX);
        assert_eq!(m2.energy, f32::MAX);
        assert!((m2.cosine_sim - (-1.0)).abs() < 1e-6);
        assert_eq!(m2.entropy, f32::MIN);
    }

    #[test]
    fn recall_payload_source_matches_tag() {
        let mut bus = ResidualBus::new(2, 8);
        bus.register(BusPort::recall(0, BusPortTag::ShadowKv));

        let residual = vec![1.0, 0.0];
        let result = bus.recall(BusPortTag::ShadowKv, &residual, 0, None, 0.0).unwrap();
        assert_eq!(result.source, BusPortTag::ShadowKv);
    }

    #[test]
    fn residual_bus_recall_layer_reflected_in_meta() {
        let mut bus = ResidualBus::new(2, 50);
        bus.register(BusPort::recall(42, BusPortTag::IntentRecall));

        let residual = vec![1.0, 1.0];
        let result = bus.recall(BusPortTag::IntentRecall, &residual, 42, None, 1.0).unwrap();
        assert_eq!(result.meta.layer, 42);
    }

    #[test]
    fn bus_port_kind_all_variants_debug() {
        let variants = [BusPortKind::Injection, BusPortKind::Recall];
        for v in &variants {
            let s = format!("{:?}", v);
            assert!(!s.is_empty(), "Debug should not be empty for {:?}", v);
        }
    }

    #[test]
    fn bus_port_tag_all_variants_clone_independent() {
        let variants = [
            BusPortTag::RagInjection,
            BusPortTag::EarlyExit,
            BusPortTag::IntentRecall,
            BusPortTag::Guardrail,
            BusPortTag::ShadowKv,
            BusPortTag::Custom(42),
        ];
        for v in &variants {
            let cloned = v.clone();
            assert_eq!(*v, cloned);
        }
    }

    #[test]
    fn bus_port_tag_custom_distinct_from_all_named() {
        let named = [
            BusPortTag::RagInjection,
            BusPortTag::EarlyExit,
            BusPortTag::IntentRecall,
            BusPortTag::Guardrail,
            BusPortTag::ShadowKv,
        ];
        let custom = BusPortTag::Custom(0);
        for named_tag in &named {
            assert_ne!(*named_tag, custom);
        }
    }

    #[test]
    fn residual_bus_register_at_first_layer() {
        let mut bus = ResidualBus::new(4, 100);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));
        assert_eq!(bus.ports()[0].layer, 0);
        assert_eq!(bus.ports()[0].kind, BusPortKind::Injection);
    }

    #[test]
    fn residual_bus_register_at_last_layer() {
        let mut bus = ResidualBus::new(4, 100);
        bus.register(BusPort::recall(99, BusPortTag::EarlyExit));
        assert_eq!(bus.ports()[0].layer, 99);
        assert_eq!(bus.ports()[0].kind, BusPortKind::Recall);
    }

    #[test]
    fn residual_bus_inject_on_injection_port_only() {
        // Cannot inject into a recall port
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::recall(0, BusPortTag::EarlyExit));

        let payload = InjectionPayload {
            target: BusPortTag::EarlyExit,
            data: vec![1.0; 4],
            scale: 1.0,
        };
        let mut buf = vec![0.0; 4];
        assert!(matches!(
            bus.inject(&payload, &mut buf),
            Err(ResidualBusError::WrongPortType { .. })
        ));
    }

    #[test]
    fn residual_bus_recall_on_recall_port_only() {
        // Cannot recall from an injection port
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));

        let residual = vec![1.0; 4];
        assert!(matches!(
            bus.recall(BusPortTag::RagInjection, &residual, 0, None, 0.0),
            Err(ResidualBusError::WrongPortType { .. })
        ));
    }

    #[test]
    fn residual_bus_inject_and_recall_different_ports_same_bus() {
        let mut bus = ResidualBus::new(4, 10);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));
        bus.register(BusPort::recall(5, BusPortTag::IntentRecall));

        // Inject
        let payload = InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0, 2.0, 3.0, 4.0],
            scale: 1.0,
        };
        let mut inject_buf = vec![10.0, 20.0, 30.0, 40.0];
        bus.inject(&payload, &mut inject_buf).unwrap();
        assert_eq!(inject_buf, vec![11.0, 22.0, 33.0, 44.0]);

        // Recall (using a different buffer)
        let residual = vec![1.0, 0.0, 0.0, 0.0];
        let result = bus.recall(BusPortTag::IntentRecall, &residual, 5, None, 2.0).unwrap();
        assert_eq!(result.source, BusPortTag::IntentRecall);
        assert_eq!(result.meta.layer, 5);
        assert!((result.meta.entropy - 2.0).abs() < 1e-6);
    }

    #[test]
    fn residual_bus_active_count_after_deactivate_all() {
        let mut bus = ResidualBus::new(4, 10);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));
        bus.register(BusPort::recall(1, BusPortTag::EarlyExit));
        bus.register(BusPort::recall(2, BusPortTag::IntentRecall));

        assert_eq!(bus.active_port_count(), 3);

        for tag in &[BusPortTag::RagInjection, BusPortTag::EarlyExit, BusPortTag::IntentRecall] {
            bus.find_port(*tag).unwrap().deactivate();
        }
        assert_eq!(bus.active_port_count(), 0);
        // Ports still registered
        assert_eq!(bus.ports().len(), 3);
    }

    #[test]
    fn residual_bus_reactivate_all_ports() {
        let mut bus = ResidualBus::new(4, 10);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));
        bus.register(BusPort::recall(1, BusPortTag::EarlyExit));

        bus.find_port(BusPortTag::RagInjection).unwrap().deactivate();
        bus.find_port(BusPortTag::EarlyExit).unwrap().deactivate();
        assert_eq!(bus.active_port_count(), 0);

        bus.find_port(BusPortTag::RagInjection).unwrap().activate();
        bus.find_port(BusPortTag::EarlyExit).unwrap().activate();
        assert_eq!(bus.active_port_count(), 2);
    }

    // --- Wave 3: 10 additional high-quality tests ---

    #[test]
    fn residual_bus_error_display_port_not_found_custom_tag() {
        // Arrange: create a PortNotFound error with a Custom tag
        let e = ResidualBusError::PortNotFound(BusPortTag::Custom(999));
        // Act
        let s = e.to_string();
        // Assert: Display contains "not found" and the custom discriminant info
        assert!(s.contains("not found"), "Display should contain 'not found', got: {}", s);
    }

    #[test]
    fn injection_payload_with_nan_scale_does_not_panic() {
        // Arrange: construct injection payload with NaN scale
        let mut bus = ResidualBus::new(2, 8);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));

        let payload = InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0, 1.0],
            scale: f32::NAN,
        };
        let mut buf = vec![0.0, 0.0];
        // Act: inject with NaN scale — should succeed (0.0 + 1.0 * NaN = NaN)
        bus.inject(&payload, &mut buf).unwrap();
        // Assert: result is NaN (IEEE 754 semantics)
        assert!(buf[0].is_nan(), "buf[0] should be NaN, got {}", buf[0]);
        assert!(buf[1].is_nan(), "buf[1] should be NaN, got {}", buf[1]);
    }

    #[test]
    fn injection_payload_with_inf_scale_saturation() {
        // Arrange: inject with positive infinity scale
        let mut bus = ResidualBus::new(2, 8);
        bus.register(BusPort::injection(0, BusPortTag::Guardrail));

        let payload = InjectionPayload {
            target: BusPortTag::Guardrail,
            data: vec![1.0, 1.0],
            scale: f32::INFINITY,
        };
        let mut buf = vec![0.0, 0.0];
        // Act
        bus.inject(&payload, &mut buf).unwrap();
        // Assert: 0.0 + 1.0 * inf = +inf
        assert!(buf[0].is_infinite() && buf[0].is_sign_positive());
        assert!(buf[1].is_infinite() && buf[1].is_sign_positive());
    }

    #[test]
    fn recall_payload_data_independence_after_modification() {
        // Arrange: recall into a payload, then mutate payload.data
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::recall(0, BusPortTag::IntentRecall));

        let mut residual = vec![1.0, 2.0, 3.0, 4.0];
        let result = bus.recall(BusPortTag::IntentRecall, &residual, 0, None, 0.0).unwrap();
        // Act: mutate both returned data and original residual
        let mut returned = result.data.clone();
        returned[0] = 99.0;
        residual[0] = 88.0;
        // Assert: mutations are independent
        assert_eq!(result.data[0], 1.0, "RecallPayload.data should be an owned copy");
        assert_eq!(returned[0], 99.0, "local binding should reflect our mutation");
    }

    #[test]
    fn bus_port_tag_custom_used_as_hashmap_key_with_overwrite() {
        // Arrange: use Custom tags as HashMap keys, test overwrite semantics
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(BusPortTag::Custom(10), "alpha");
        map.insert(BusPortTag::Custom(20), "beta");
        map.insert(BusPortTag::Custom(10), "gamma"); // overwrite key 10
        // Act & Assert
        assert_eq!(map.len(), 2, "only 2 distinct Custom keys");
        assert_eq!(map.get(&BusPortTag::Custom(10)), Some(&"gamma"));
        assert_eq!(map.get(&BusPortTag::Custom(20)), Some(&"beta"));
        assert_eq!(map.get(&BusPortTag::Custom(30)), None);
    }

    #[test]
    fn residual_bus_register_custom_tag_inject_and_recall() {
        // Arrange: register a Custom tag as both injection and recall on different layers
        let mut bus = ResidualBus::new(3, 10);
        bus.register(BusPort::injection(1, BusPortTag::Custom(55)));
        bus.register(BusPort::recall(5, BusPortTag::Custom(66)));

        // Act: inject into Custom(55)
        let payload = InjectionPayload {
            target: BusPortTag::Custom(55),
            data: vec![10.0, 20.0, 30.0],
            scale: 0.1,
        };
        let mut buf = vec![100.0, 100.0, 100.0];
        bus.inject(&payload, &mut buf).unwrap();
        // Assert: 100 + 10*0.1, 100 + 20*0.1, 100 + 30*0.1
        assert_eq!(buf, vec![101.0, 102.0, 103.0]);

        // Act: recall from Custom(66)
        let residual = vec![3.0, 0.0, 4.0];
        let result = bus.recall(BusPortTag::Custom(66), &residual, 5, None, 7.7).unwrap();
        assert_eq!(result.source, BusPortTag::Custom(66));
        assert!((result.meta.entropy - 7.7).abs() < 1e-5);
    }

    #[test]
    fn residual_bus_find_port_mut_deactivate_reflected_in_find_port() {
        // Arrange: register two ports, deactivate one via find_port_mut
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));
        bus.register(BusPort::recall(2, BusPortTag::ShadowKv));

        // Act: deactivate via mutable accessor
        bus.find_port_mut(BusPortTag::ShadowKv).unwrap().deactivate();
        // Assert: immutable accessor sees the change
        let port = bus.find_port(BusPortTag::ShadowKv).unwrap();
        assert!(!port.is_active(), "deactivation via find_port_mut should be visible in find_port");
        // Assert: other port unaffected
        assert!(bus.find_port(BusPortTag::RagInjection).unwrap().is_active());
    }

    #[test]
    fn residual_bus_recall_energy_all_ones_sqrt_dim() {
        // Arrange: residual vector of all 1.0s, energy = sqrt(dim)
        let dim = 9usize;
        let mut bus = ResidualBus::new(dim, 8);
        bus.register(BusPort::recall(0, BusPortTag::EarlyExit));

        let residual = vec![1.0; dim];
        // Act
        let result = bus.recall(BusPortTag::EarlyExit, &residual, 0, None, 0.0).unwrap();
        // Assert: energy = sqrt(9) = 3.0
        let expected_energy = (dim as f32).sqrt();
        assert!(
            (result.meta.energy - expected_energy).abs() < 1e-5,
            "energy should be {}, got {}",
            expected_energy,
            result.meta.energy
        );
    }

    #[test]
    fn residual_bus_inject_preserves_buffer_length() {
        // Arrange: verify inject never changes buffer length
        let mut bus = ResidualBus::new(7, 8);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));

        let payload = InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0; 7],
            scale: 1.0,
        };
        let mut buf = vec![0.0; 7];
        let original_len = buf.len();
        // Act
        bus.inject(&payload, &mut buf).unwrap();
        // Assert
        assert_eq!(buf.len(), original_len, "buffer length must be preserved after inject");
    }

    #[test]
    fn recall_meta_f32_special_values_preserved() {
        // Arrange: construct RecallMeta with f32 special values
        let meta = RecallMeta {
            layer: 0,
            energy: f32::INFINITY,
            cosine_sim: f32::NAN,
            entropy: f32::NEG_INFINITY,
        };
        // Act: clone
        let cloned = meta.clone();
        // Assert: special f32 values survive cloning
        assert!(cloned.energy.is_infinite() && cloned.energy.is_sign_positive());
        assert!(cloned.cosine_sim.is_nan());
        assert!(cloned.entropy.is_infinite() && cloned.entropy.is_sign_negative());
        assert_eq!(cloned.layer, 0);
    }
}
