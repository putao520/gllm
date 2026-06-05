//! GGUF String Table 解析器
//!
//! 目标：理解 GGUF 如何存储字符串，以及如何解析 tokenizer.ggml.tokens
//! 中的整数偏移量

use std::fs::File;
use std::io::Read;

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
enum GGUFValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GGUFValueType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

#[derive(Debug)]
#[allow(dead_code)]
enum GGUFValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(GGUFValueType, Vec<GGUFValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

struct GGUFParser<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> GGUFParser<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_u8(&mut self) -> Option<u8> {
        if self.pos + 1 > self.data.len() {
            return None;
        }
        let val = self.data[self.pos];
        self.pos += 1;
        Some(val)
    }

    fn read_u32(&mut self) -> Option<u32> {
        if self.pos + 4 > self.data.len() {
            return None;
        }
        let val = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Some(val)
    }

    fn read_u64(&mut self) -> Option<u64> {
        if self.pos + 8 > self.data.len() {
            return None;
        }
        let val = u64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Some(val)
    }

    fn read_i32(&mut self) -> Option<i32> {
        if self.pos + 4 > self.data.len() {
            return None;
        }
        let val = i32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Some(val)
    }

    fn read_string(&mut self) -> Option<String> {
        let len = self.read_u64()? as usize;
        if self.pos + len > self.data.len() {
            return None;
        }
        let s = std::str::from_utf8(&self.data[self.pos..self.pos + len])
            .ok()?
            .to_string();
        self.pos += len;
        Some(s)
    }

    fn read_value(&mut self, value_type: GGUFValueType) -> Option<GGUFValue> {
        match value_type {
            GGUFValueType::Uint8 => Some(GGUFValue::Uint8(self.read_u8()?)),
            GGUFValueType::Int8 => {
                let b = self.read_u8()?;
                Some(GGUFValue::Int8(b as i8))
            }
            GGUFValueType::Uint16 => {
                let b = self.read_u16()?;
                Some(GGUFValue::Uint16(b))
            }
            GGUFValueType::Int16 => {
                let b = self.read_u16()?;
                Some(GGUFValue::Int16(b as i16))
            }
            GGUFValueType::Uint32 => Some(GGUFValue::Uint32(self.read_u32()?)),
            GGUFValueType::Int32 => Some(GGUFValue::Int32(self.read_i32()?)),
            GGUFValueType::Float32 => {
                let b = self.read_u32()?;
                Some(GGUFValue::Float32(f32::from_bits(b)))
            }
            GGUFValueType::Bool => Some(GGUFValue::Bool(self.read_u8()? != 0)),
            GGUFValueType::String => {
                let s = self.read_string()?;
                Some(GGUFValue::String(s))
            }
            GGUFValueType::Uint64 => Some(GGUFValue::Uint64(self.read_u64()?)),
            GGUFValueType::Int64 => {
                let b = self.read_u64()?;
                Some(GGUFValue::Int64(b as i64))
            }
            GGUFValueType::Float64 => {
                let b = self.read_u64()?;
                Some(GGUFValue::Float64(f64::from_bits(b)))
            }
            GGUFValueType::Array => {
                // 🚨 重要：GGUF 数组长度是 uint32_t，不是 uint64_t！
                // 这是从实际文件分析得出的结论（0x400_0000_008 只能解释为 u32）
                let len = self.read_u32()? as u64;
                let elem_type_raw = self.read_u32()?;
                let elem_type = GGUFValueType::from_u32(elem_type_raw)?;

                // 限制读取长度，避免卡死
                let safe_len = len.min(10000) as usize;
                let mut values = Vec::with_capacity(safe_len);
                for _ in 0..safe_len {
                    let val = self.read_value(elem_type)?;
                    values.push(val);
                }

                // 跳过剩余元素
                if len > 10000 {
                    // 估算每个元素大小并跳过
                    // 这是一个粗略估计
                    let skip_size = (len - 10000) as usize * 8; // 假设每个元素最多 8 字节
                    self.pos = self.pos.saturating_add(skip_size).min(self.data.len());
                }

                Some(GGUFValue::Array(elem_type, values))
            }
        }
    }

    // 需要添加 read_u16 方法
    fn read_u16(&mut self) -> Option<u16> {
        if self.pos + 2 > self.data.len() {
            return None;
        }
        let val = u16::from_le_bytes(self.data[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        Some(val)
    }
}

fn main() {
    let path = std::path::Path::new(
        "/home/putao/.gllm/models/Mungert--SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-bf16.gguf"
    );

    let mut f = File::open(path).expect("open file");
    let mut data = Vec::new();
    f.read_to_end(&mut data).expect("read file");

    println!("=== GGUF Header ===");
    println!("File size: {} bytes", data.len());

    // Magic number (4 bytes)
    let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
    println!("Magic: {:#x} (expected: 0x46554747)", magic);

    // Version (4 bytes)
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    println!("Version: {}", version);

    // Tensor count (8 bytes)
    let tensor_count = u64::from_le_bytes(data[8..16].try_into().unwrap());
    println!("Tensor count: {}", tensor_count);

    // KV count (8 bytes)
    let kv_count = u64::from_le_bytes(data[16..24].try_into().unwrap());
    println!("KV count: {}", kv_count);

    println!("\n=== KV Metadata ===");

    let mut parser = GGUFParser::new(&data);
    parser.pos = 24; // Skip header

    let mut tokenizer_tokens: Option<Vec<u32>> = None;
    let mut tokenizer_merges: Option<Vec<u32>> = None;
    let mut tokenizer_model: Option<u32> = None;
    let mut tokenizer_vocab_size: Option<u32> = None;

    for i in 0..kv_count {
        let key = match parser.read_string() {
            Some(k) => k,
            None => {
                println!("KV {}: Failed to read key at pos {}", i, parser.pos);
                break;
            }
        };

        let value_type_raw = match parser.read_u32() {
            Some(v) => v,
            None => {
                println!("KV {}: Failed to read value type", i);
                break;
            }
        };

        let value_type = match GGUFValueType::from_u32(value_type_raw) {
            Some(t) => t,
            None => {
                println!("KV {}: Unknown value type: {}", i, value_type_raw);
                break;
            }
        };

        let value_start = parser.pos;

        // 读取值
        match value_type {
            GGUFValueType::Array => {
                // 🚨 重要：GGUF 数组长度是 uint32_t，不是 uint64_t！
                let len = parser.read_u32().unwrap_or(0) as usize;
                let elem_type_raw = parser.read_u32().unwrap_or(0);
                let elem_type = GGUFValueType::from_u32(elem_type_raw);

                println!(
                    "KV {}: key='{}', type=ARRAY<{}>, len={}",
                    i,
                    key,
                    elem_type
                        .map(|t| format!("{:?}", t))
                        .unwrap_or("unknown".to_string()),
                    len
                );

                // 对于 tokenizer 相关字段，详细解析
                if key.contains("tokenizer.ggml") {
                    if let Some(etype) = elem_type {
                        match etype {
                            GGUFValueType::Uint32 => {
                                // 读取前几个元素
                                let display_count = len.min(10);
                                let mut vals = Vec::new();
                                for _ in 0..display_count {
                                    if let Some(GGUFValue::Uint32(v)) = parser.read_value(etype) {
                                        vals.push(v);
                                    }
                                }
                                println!("  First values: {:?}", vals);

                                // 保存数据
                                if key == "tokenizer.ggml.tokens" {
                                    // 重新读取完整数据
                                    parser.pos = value_start;
                                    if let Some(GGUFValue::Array(_, elements)) =
                                        parser.read_value(value_type)
                                    {
                                        tokenizer_tokens = Some(
                                            elements
                                                .iter()
                                                .filter_map(|v| {
                                                    if let GGUFValue::Uint32(u) = v {
                                                        Some(*u)
                                                    } else {
                                                        None
                                                    }
                                                })
                                                .collect(),
                                        );
                                    }
                                } else if key == "tokenizer.ggml.merges" {
                                    parser.pos = value_start;
                                    if let Some(GGUFValue::Array(_, elements)) =
                                        parser.read_value(value_type)
                                    {
                                        tokenizer_merges = Some(
                                            elements
                                                .iter()
                                                .filter_map(|v| {
                                                    if let GGUFValue::Uint32(u) = v {
                                                        Some(*u)
                                                    } else {
                                                        None
                                                    }
                                                })
                                                .collect(),
                                        );
                                    }
                                }
                            }
                            GGUFValueType::String => {
                                // 读取前几个元素
                                let display_count = len.min(5);
                                for j in 0..display_count {
                                    if let Some(GGUFValue::String(s)) = parser.read_value(etype) {
                                        println!("  [{}]: '{}'", j, s);
                                    }
                                }
                            }
                            _ => {
                                println!("  (skipping {} elements of type {:?})", len, etype);
                                // 跳过
                                for _ in 0..len.min(1000) {
                                    let _ = parser.read_value(etype);
                                }
                            }
                        }
                    }
                } else {
                    // 跳过非 tokenizer 数组
                    let etype = elem_type.unwrap_or(GGUFValueType::Uint8);
                    for _ in 0..len.min(100) {
                        let _ = parser.read_value(etype);
                    }
                    if len > 100 {
                        println!("  (skipped {} elements)", len - 100);
                    }
                }
            }
            GGUFValueType::String => {
                if let Some(GGUFValue::String(s)) = parser.read_value(value_type) {
                    println!("KV {}: key='{}', value='{}'", i, key, s);

                    if key == "tokenizer.ggml.model" {
                        if let Ok(n) = s.parse::<u32>() {
                            tokenizer_model = Some(n);
                        }
                    } else if key == "tokenizer.ggml.vocab_size" {
                        if let Ok(n) = s.parse::<u32>() {
                            tokenizer_vocab_size = Some(n);
                        }
                    }
                }
            }
            GGUFValueType::Uint32 => {
                if let Some(GGUFValue::Uint32(v)) = parser.read_value(value_type) {
                    println!("KV {}: key='{}', value={}", i, key, v);
                }
            }
            GGUFValueType::Int32 => {
                if let Some(GGUFValue::Int32(v)) = parser.read_value(value_type) {
                    println!("KV {}: key='{}', value={}", i, key, v);
                }
            }
            GGUFValueType::Float32 => {
                if let Some(GGUFValue::Float32(v)) = parser.read_value(value_type) {
                    println!("KV {}: key='{}', value={}", i, key, v);
                }
            }
            GGUFValueType::Bool => {
                if let Some(GGUFValue::Bool(v)) = parser.read_value(value_type) {
                    println!("KV {}: key='{}', value={}", i, key, v);
                }
            }
            _ => {
                println!("KV {}: key='{}', type={:?}", i, key, value_type);
                // 尝试读取并跳过
                let _ = parser.read_value(value_type);
            }
        }
    }

    println!("\n=== Summary ===");
    println!("Position after KV parsing: {}", parser.pos);
    println!("Remaining bytes: {}", parser.remaining());

    if let Some(model) = tokenizer_model {
        let type_name = match model {
            0 => "normal (Raw)",
            1 => "bpe (Byte Pair Encoding)",
            2 => "spx (SentencePiece with unigram)",
            3 => "unigram",
            4 => "llama (SentencePiece)",
            _ => "unknown",
        };
        println!("Tokenizer model: {} ({})", model, type_name);
    }

    if let Some(vocab_size) = tokenizer_vocab_size {
        println!("Vocab size: {}", vocab_size);
    }

    if let Some(ref tokens) = tokenizer_tokens {
        println!("Token offsets: {} entries", tokens.len());
        println!("First 10 offsets: {:?}", &tokens[..10.min(tokens.len())]);
        println!(
            "Last 5 offsets: {:?}",
            &tokens[tokens.len().saturating_sub(5)..]
        );
    }

    if let Some(ref merges) = tokenizer_merges {
        println!("Merge offsets: {} entries", merges.len());
        println!("First 10 offsets: {:?}", &merges[..10.min(merges.len())]);
    }

    // 分析偏移量模式
    if let (Some(tokens), Some(merges)) = (tokenizer_tokens, tokenizer_merges) {
        println!("\n=== Offset Analysis ===");
        println!(
            "Token offset range: {} -> {}",
            tokens.first().unwrap_or(&0),
            tokens.last().unwrap_or(&0)
        );
        println!(
            "Merge offset range: {} -> {}",
            merges.first().unwrap_or(&0),
            merges.last().unwrap_or(&0)
        );

        // 检查是否有重叠或模式
        let mut all_offsets = tokens.clone();
        all_offsets.extend_from_slice(&merges);
        all_offsets.sort();
        all_offsets.dedup();

        println!("Unique offsets: {}", all_offsets.len());
        println!(
            "First 20 unique offsets: {:?}",
            &all_offsets[..20.min(all_offsets.len())]
        );

        // 计算平均间距
        let mut gaps = Vec::new();
        for i in 1..all_offsets.len() {
            gaps.push(all_offsets[i] - all_offsets[i - 1]);
        }
        if !gaps.is_empty() {
            let total: u32 = gaps.iter().sum();
            println!("Average gap: {:.1}", total as f64 / gaps.len() as f64);
            println!(
                "Min gap: {}, Max gap: {}",
                gaps.iter().min().unwrap(),
                gaps.iter().max().unwrap()
            );
        }
    }

    // 检查 pos 附近的数据
    println!("\n=== Data at KV end ===");
    let start = parser.pos;
    for i in 0..64.min(parser.remaining()) {
        print!("{:02x} ", data[start + i]);
        if (i + 1) % 16 == 0 {
            println!();
        }
    }
    println!();
}
