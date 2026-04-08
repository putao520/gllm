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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}
