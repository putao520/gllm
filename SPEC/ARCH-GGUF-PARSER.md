# GGUF 解析器架构设计

> **定位**: gllm 内部使用的 GGUF 格式解析器，**不是通用工具库**
>
> **核心目标**: 替换有 bug 的 gguf-rs，修复 ARRAY[STRING] 解析问题
>
> **设计约束**:
> - Ω1 真实性原则：一切从元数据读取，禁止硬编码默认值
> - 零拷贝：Tensor 数据直接传递给 gllm-kernels
> - 最小化 API：只暴露推理必需的接口

---

## 1. 核心问题

### 1.1 gguf-rs 的 Bug

**ARRAY[STRING] 解析错误**：
- 预期：`tokenizer.ggml.tokens` 返回 49152 个 token
- 实际：只返回 3 个 token

**根本原因**：gguf-rs 在解析嵌套数组时没有正确处理元素类型和长度。

### 1.2 解决方案

自实现 GGUF 解析器，确保：
1. 正确解析所有 GGUF 值类型（包括 ARRAY[STRING]）
2. 支持所有 28 种量化类型的数据读取
3. 零拷贝传递 Tensor 数据给 gllm-kernels

---

## 2. GGUF 文件格式

```
+-------------------+
| Magic (4 bytes)   | 0x46554747 ("GGUF")
+-------------------+
| Version (4 bytes) | 当前版本: 3
+-------------------+
| Tensor Count (8)  |
+-------------------+
| KV Count (8)      |
+-------------------+
| KV Pairs...       | 变长
+-------------------+
| Tensor Info...    | 变长
+-------------------+
| Padding           | 对齐到 alignment
+-------------------+
| Tensor Data...    |
+-------------------+
```

---

## 3. 核心解析器

### 3.1 公共 API (极简化)

```rust
/// GGUF 文件解析器
pub struct GgufReader {
    mmap: Arc<Mmap>,
    metadata: BTreeMap<String, GgufValue>,
    tensors: Vec<TensorInfo>,
    data_offset: usize,
}

/// Tensor 信息
pub struct TensorInfo {
    name: Arc<str>,
    dtype: GgmlDType,
    shape: Vec<u64>,
    offset: usize,
    size: usize,
}

impl GgufReader {
    /// 打开 GGUF 文件
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError>;

    /// Ω1: 从元数据读取架构类型
    pub fn architecture(&self) -> Result<&str, GgufError>;

    /// Ω1: 读取元数据值（通用方法）
    pub fn get_metadata_u64(&self, key: &str) -> Option<u64>;
    pub fn get_metadata_f32(&self, key: &str) -> Option<f32>;
    pub fn get_metadata_str(&self, key: &str) -> Option<&str>;
    pub fn get_metadata_array(&self, key: &str) -> Option<&GgufArray>;

    /// Tokenizer tokens (修复 ARRAY[STRING] bug)
    pub fn tokenizer_tokens(&self) -> Result<Vec<&str>, GgufError>;

    /// 获取 Tensor (零拷贝)
    pub fn tensor(&self, name: &str) -> Result<TensorSlice, GgufError>;

    /// Tensor 数量
    pub fn tensor_count(&self) -> usize;
}
```

### 3.2 TensorSlice (零拷贝)

```rust
/// Tensor 数据的零拷贝视图
pub struct TensorSlice<'a> {
    dtype: GgmlDType,
    shape: Vec<u64>,
    data: &'a [u8],
}

impl<'a> TensorSlice<'a> {
    /// 原始字节（传递给 gllm-kernels）
    pub fn as_bytes(&self) -> &[u8];

    /// 量化类型
    pub fn dtype(&self) -> GgmlDType;

    /// 形状
    pub fn shape(&self) -> &[u64];
}
```

### 3.3 类型定义 (内部)

```rust
/// GGUF 值类型
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufValueType {
    Uint8 = 0, Int8 = 1, Uint16 = 2, Int16 = 3,
    Uint32 = 4, Int32 = 5, Float32 = 6, Bool = 7,
    String = 8, Array = 9, Uint64 = 10, Int64 = 11, Float64 = 12,
}

/// GGML 量化类型标识符
///
/// **🚨 重要约束**: 此枚举仅用于 GGUF 文件格式解析和类型标识，
/// **不是** gllm-kernels 的量化类型定义。
///
/// gllm-kernels 使用泛型架构 (ARCH-QUANT-GENERIC):
/// `QuantizedStorage<const N: usize, const BITS: u8>`
///
/// 类型映射关系在适配层实现 (见 4.3 节)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlDType {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7,
    Q8_0 = 8, Q8_1 = 9, Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13,
    Q6_K = 14, Q8_K = 15, IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18,
    IQ1_S = 19, IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23,
    I8 = 24, I16 = 25, I32 = 26, I64 = 27, F64 = 28, IQ1_M = 29,
    BF16 = 30, TQ1_0 = 34, TQ2_0 = 35, MXFP4 = 39,
}

/// 元数据值 (内部)
#[derive(Debug, Clone)]
enum GgufValue {
    Uint8(u8), Int8(i8), Uint16(u16), Int16(i16),
    Uint32(u32), Int32(i32), Uint64(u64), Int64(i64),
    Float32(f32), Float64(f64), Bool(bool),
    String(Arc<str>),
    Array(GgufArray),
}

struct GgufArray {
    item_type: GgufValueType,
    items: Vec<GgufValue>,
}
```

### 3.4 量化类型映射约束 (ARCH-GGUF-QUANT-MAPPING)

> **关联架构**: gllm-kernels ARCH-QUANT-GENERIC
> **核心约束**: GGUF 解析器不定义量化计算逻辑，只负责类型标识

| GGUF 类型 | 块大小 (N) | 位数 (BITS) | gllm-kernels 类型别名 |
|-----------|-----------|-------------|----------------------|
| F32 | 1 | 32 | `F32Block = Block<4>` |
| F16 | 1 | 16 | `F16Block = Block<2>` |
| Q4_0 | 32 | 4 | `Q4_0Block = Block<18>` |
| Q8_0 | 32 | 8 | `Q8_0Block = Block<34>` |
| Q5_K | 256 | 5 | `Q5_KBlock = Block<...>` |

**关键原则**：
- GGUF 解析器返回 `(GgmlDType, &[u8])` 元组
- 适配层 (adapter) 负责将 `(dtype, data)` 转换为 gllm-kernels 的泛型类型
- **禁止** 在解析器中直接实例化量化 Block 类型

---

## 4. 关键实现

### 4.1 修复 ARRAY[STRING] 解析

```rust
impl GgufReader {
    /// 解析 ARRAY[STRING]（修复 gguf-rs bug）
    fn parse_array_string(
        data: &[u8],
        pos: &mut usize,
        byte_order: ByteOrder,
    ) -> Result<GgufArray, GgufError> {
        let count = read_u64(data, pos, byte_order)? as usize;
        let mut items = Vec::with_capacity(count);

        for _ in 0..count {
            // 读取字符串长度
            let len = read_u64(data, pos, byte_order)? as usize;
            // 读取字符串内容
            let bytes = &data[*pos..*pos + len];
            *pos += len;
            // 转换为 UTF-8
            let s = std::str::from_utf8(bytes)?;
            items.push(GgufValue::String(Arc::from(s)));
        }

        Ok(GgufArray {
            item_type: GgufValueType::String,
            items,
        })
    }
}
```

### 4.2 Ω1 真实性原则实现

```rust
/// 配置从 GGUF 元数据读取（禁止硬编码默认值）
pub struct ModelConfig {
    pub vocab_size: u64,
    pub hidden_size: u64,
    pub num_layers: u64,
    pub num_heads: u64,
    pub num_kv_heads: u64,
    pub max_context: u64,
    pub rope_theta: f32,
}

impl ModelConfig {
    pub fn from_gguf(reader: &GgufReader) -> Result<Self, GgufError> {
        let arch = reader.architecture()?;
        let prefix = arch;

        Ok(Self {
            // Ω1: 从元数据读取，禁止默认值
            vocab_size: reader.get_metadata_u64(&format!("{}.vocab_size", prefix))
                .ok_or_else(|| GgufError::MissingMetadata("vocab_size"))?,
            hidden_size: reader.get_metadata_u64(&format!("{}.embedding_length", prefix))
                .ok_or_else(|| GgufError::MissingMetadata("embedding_length"))?,
            num_layers: reader.get_metadata_u64(&format!("{}.block_count", prefix))
                .ok_or_else(|| GgufError::MissingMetadata("block_count"))?,
            // ... 其他字段类似
        })
    }
}
```

### 4.3 与 gllm-kernels 集成

> **🚨 边界约束**: GGUF 解析器只负责文件格式解析，不负责类型适配
>
> **职责分离**:
> - GGUF 解析器: 返回原始字节 + 类型标识符
> - 适配层 (adapter): 负责类型映射和 gllm-kernels 集成

```rust
// GGUF 解析器接口 (src/loader/gguf/)
impl GgufReader {
    /// 获取 Tensor 原始数据（零拷贝）
    pub fn tensor(&self, name: &str) -> Result<TensorSlice<'_>, GgufError>;

    /// 获取 Tensor 元信息
    pub fn tensor_info(&self, name: &str) -> Result<&TensorInfo, GgufError>;
}

// TensorSlice 定义
pub struct TensorSlice<'a> {
    dtype: GgmlDType,  // 类型标识符
    shape: Vec<u64>,
    data: &'a [u8],    // 原始字节
}

// 适配层接口 (src/loader/adapter.rs)
//
// 适配层负责：
// 1. 将 GgmlDType 映射到 gllm-kernels 泛型参数
// 2. 创建正确的 QuantizedStorage 实例
// 3. 处理字节序和内存对齐
pub struct GgufAdapter {
    reader: GgufReader,
    // gllm-kernels 相关状态
}

impl GgufAdapter {
    /// 将 GGUF Tensor 转换为 gllm-kernels 格式
    pub fn tensor_for_kernel(
        &self,
        name: &str,
    ) -> Result<gllm_kernels::TensorData, GgufError> {
        let slice = self.reader.tensor(name)?;

        // 类型映射在适配层完成
        let kernel_dtype = match slice.dtype() {
            GgmlDType::F32 => gllm_kernels::DType::F32,
            GgmlDType::F16 => gllm_kernels::DType::F16,
            GgmlDType::Q4_0 => gllm_kernels::DType::Q4_0,
            // ... 其他类型映射
            _ => return Err(GgufError::UnsupportedType(slice.dtype())),
        };

        Ok(gllm_kernels::TensorData {
            dtype: kernel_dtype,
            shape: slice.shape().iter().map(|&v| v as usize).collect(),
            data: slice.as_bytes().as_ptr(),
            len: slice.as_bytes().len(),
        })
    }
}
```

### 4.4 类型映射表 (完整)

| GgmlDType | Block Size (N) | Block Bytes | gllm-kernels 映射 |
|-----------|----------------|-------------|------------------|
| F32 | 1 | 4 | `F32Block` |
| F16 | 1 | 2 | `F16Block` |
| Q4_0 | 32 | 18 | `Q4_0Block` |
| Q4_1 | 32 | 20 | `Q4_1Block` |
| Q5_0 | 32 | 22 | `Q5_0Block` |
| Q5_1 | 32 | 24 | `Q5_1Block` |
| Q8_0 | 32 | 34 | `Q8_0Block` |
| Q8_1 | 32 | 40 | `Q8_1Block` |
| Q2_K | 256 | 变长 | `Q2_KBlock` |
| Q3_K | 256 | 变长 | `Q3_KBlock` |
| Q4_K | 256 | 变长 | `Q4_KBlock` |
| Q5_K | 256 | 变长 | `Q5_KBlock` |
| Q6_K | 256 | 变长 | `Q6_KBlock` |
| Q8_K | 256 | 变长 | `Q8_KBlock` |
| IQ2_XXS | 256 | 变长 | `IQ2_XXSBlock` |
| IQ2_XS | 256 | 变长 | `IQ2_XSBlock` |
| IQ3_XXS | 256 | 变长 | `IQ3_XXSBlock` |
| IQ1_S | 256 | 变长 | `IQ1_SBlock` |
| IQ4_NL | 32 | 18 | `IQ4_NLBlock` |
| IQ3_S | 256 | 变长 | `IQ3_SBlock` |
| IQ2_S | 256 | 变长 | `IQ2_SBlock` |
| IQ4_XS | 256 | 变长 | `IQ4_XSBlock` |
| I8 | 1 | 1 | `I8Block` |
| I16 | 1 | 2 | `I16Block` |
| I32 | 1 | 4 | `I32Block` |
| I64 | 1 | 8 | `I64Block` |
| F64 | 1 | 8 | `F64Block` |
| IQ1_M | 256 | 变长 | `IQ1_MBlock` |
| BF16 | 1 | 2 | `BF16Block` |
| TQ1_0 | 256 | 变长 | `TQ1_0Block` |
| TQ2_0 | 256 | 变长 | `TQ2_0Block` |
| MXFP4 | 32 | 17 | `MXFP4Block` |

**🚨 gllm-kernels 扩展需求**:
- 当前 gllm-kernels 只实现 3 种类型 (Q4_0, Q8_0, Q5_K)
- 需要扩展到支持全部 28 种类型
- 每种类型需要:
  1. 块大小常量 (e.g., `Q4_0_BLOCK_BYTES`)
  2. 类型别名 (e.g., `pub type Q4_0Block = Block<Q4_0_BLOCK_BYTES>;`)
  3. `QuantizedStorage` 实现

---

## 5. 错误处理

```rust
#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("Invalid GGUF magic: 0x{08x}")]
    InvalidMagic(u32),

    #[error("Missing metadata: {0}")]
    MissingMetadata(&'static str),

    #[error("Invalid metadata: {0}")]
    InvalidMetadata(&'static str),

    #[error("Tensor not found: {0}")]
    TensorNotFound(String),

    #[error("Tensor out of bounds")]
    TensorOutOfBounds,

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("UTF-8 error: {0}")]
    Utf8(#[from] std::str::Utf8Error),
}
```

---

## 6. 模块结构

```
src/loader/
├── gguf/
│   ├── mod.rs          # 公共 API 导出
│   ├── reader.rs       # GgufReader 核心实现
│   ├── types.rs        # 类型定义 (GgmlDType, GgufValue)
│   └── parse.rs        # 解析逻辑 (KV pairs, Tensor info)
├── adapter.rs          # GGUF -> gllm-kernels 适配层
└── mod.rs              # Loader 模块入口
```

---

## 7. 实现计划

| 阶段 | 内容 | 关联约束 |
|------|------|----------|
| 1 | 类型定义 (GgufValueType, GgmlDType, GgufValue) | 3.3 节 |
| 2 | 头部解析 (magic, version, counts) | 2 节 |
| 3 | KV metadata 解析（修复 ARRAY[STRING]） | 4.1 节 |
| 4 | Tensor info 解析 | 3.1 节 |
| 5 | TensorSlice 零拷贝访问 | 3.2 节 |
| 6 | 适配层实现 (GGUF -> gllm-kernels) | 4.3 节 |
| 7 | gllm-kernels 扩展 (支持 28 种量化类型) | 4.4 节 |

### 7.1 gllm-kernels 扩展优先级

| 优先级 | 量化类型 | 常用度 | 状态 |
|--------|----------|--------|------|
| P0 | Q4_0, Q8_0, Q5_K | 最常用 | ✅ 已实现 |
| P1 | Q4_K, Q5_K, Q6_K, Q8_K | K-量化系列 | 🔨 待实现 |
| P2 | Q2_K, Q3_K | 极端压缩 | 📋 计划中 |
| P3 | IQ 系列, TQ 系列 | 实验性 | 📋 未来 |
| P4 | F16, BF16, F32 | 非量化 | 📋 待实现 |

### 7.2 Ω1 依赖关系

> **核心约束**: gllm (GGUF 解析器) 和 gllm-kernels (量化计算) 必须同步开发

```
GGUF 解析器 (gllm)
    ↓ 提供原始字节 + 类型标识
适配层 (gllm/src/loader/adapter.rs)
    ↓ 类型映射
gllm-kernels 泛型量化 (QuantizedStorage<N, BITS>)
    ↓ 反量化计算
GPU/CPU 后端
```

**禁止的依赖方向**:
- ❌ gllm-kernels 依赖 GGUF 格式细节
- ❌ GGUF 解析器依赖 gllm-kernels 具体类型
- ✅ 双方通过类型标识符 (u32) 和原始字节交互

---

## 8. 验证

```bash
# 对比 Python gguf-dump 输出
python3 -m gguf --dump model.gguf > expected.json
cargo run --example debug_gguf -- model.gguf > actual.json
diff expected.json actual.json
```
