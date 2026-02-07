//! GGUF 字符串表偏移量解析器
//!
//! 核心发现：
//! 1. GGUF v3 在 KV metadata 之后有一个独立的字符串表段
//! 2. 某些 ARRAY 类型的值（如 general.tags）包含的是字符串表的索引/偏移量
//! 3. 需要先解析字符串表，再解析依赖它的 KV 对

use std::fs::File;
use std::io::Read;

fn main() {
    let path = std::path::Path::new(
        "/home/putao/.gllm/models/Mungert--SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-bf16.gguf"
    );

    let mut f = File::open(path).expect("open file");
    let mut data = Vec::new();
    f.read_to_end(&mut data).expect("read file");

    println!("=== GGUF Header ===");
    println!("File size: {} bytes", data.len());

    // Header
    let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    let tensor_count = u64::from_le_bytes(data[8..16].try_into().unwrap());
    let kv_count = u64::from_le_bytes(data[16..24].try_into().unwrap());

    println!("Magic: {:#x}", magic);
    println!("Version: {}", version);
    println!("Tensor count: {}", tensor_count);
    println!("KV count: {}", kv_count);

    let mut pos = 24usize;

    // 🚨 关键发现：先收集所有 KV 的 key 和位置，但不解析值
    // 因为某些值的解析依赖于字符串表
    let mut kv_keys: Vec<(String, usize)> = Vec::new();

    println!("\n=== Step 1: Collect KV Keys ===");
    for i in 0..kv_count {
        let key_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;

        let key = std::str::from_utf8(&data[pos..pos + key_len])
            .unwrap_or("invalid")
            .to_string();
        pos += key_len;

        // 保存 key 和当前值开始位置
        let value_start = pos;

        // 跳过 value_type
        pos += 4;

        // 根据类型跳过值数据
        let value_type = u32::from_le_bytes(data[value_start..value_start + 4].try_into().unwrap());

        match value_type {
            8 => {
                // STRING
                let str_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
                pos += 8 + str_len;
            }
            9 => {
                // ARRAY
                let arr_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
                pos += 4;
                let elem_type = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                pos += 4;

                // 跳过数组元素（粗略估计）
                match elem_type {
                    0 => pos += arr_len,     // UINT8
                    4 => pos += arr_len * 4, // UINT32
                    5 => pos += arr_len * 4, // INT32
                    6 => pos += arr_len * 4, // FLOAT32
                    8 => {
                        // STRING - 复杂，需要逐个跳过
                        for _ in 0..arr_len.min(100) {
                            let s_len =
                                u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
                            pos += 8 + s_len;
                        }
                        if arr_len > 100 {
                            // 超过100个元素的，粗略估计
                            pos += (arr_len - 100) * 10;
                        }
                    }
                    _ => pos += arr_len * 4,
                }
            }
            4 => pos += 4,  // UINT32
            5 => pos += 4,  // INT32
            6 => pos += 4,  // FLOAT32
            7 => pos += 1,  // BOOL
            10 => pos += 8, // UINT64
            11 => pos += 8, // INT64
            12 => pos += 8, // FLOAT64
            _ => pos += 8,
        }

        println!("KV {}: key='{}' at pos {}", i, key, value_start);
        kv_keys.push((key, value_start));
    }

    let string_table_start = pos;
    println!(
        "\n=== Step 2: Parse String Table (starting at {}) ===",
        string_table_start
    );

    // 解析字符串表
    // 字符串表格式：连续的 gguf_string_t 结构
    let mut string_table: Vec<String> = Vec::new();
    let mut st_pos = string_table_start;

    // 限制解析的字符串数量，避免无限循环
    let max_strings = 10000;

    for _ in 0..max_strings {
        if st_pos + 8 > data.len() {
            break;
        }

        let str_len = u64::from_le_bytes(data[st_pos..st_pos + 8].try_into().unwrap()) as usize;
        st_pos += 8;

        if st_pos + str_len > data.len() {
            println!(
                "ERROR: String length {} exceeds file size at pos {}",
                str_len, st_pos
            );
            break;
        }

        let s = std::str::from_utf8(&data[st_pos..st_pos + str_len]);
        match s {
            Ok(text) => {
                string_table.push(text.to_string());
                st_pos += str_len;
            }
            Err(_) => {
                // 无效 UTF-8，可能到达了字符串表末尾
                println!(
                    "Invalid UTF-8 at pos {}, likely end of string table",
                    st_pos
                );
                break;
            }
        }
    }

    println!("String table: {} entries", string_table.len());

    // 显示前50个字符串
    println!("\n=== First 50 Strings ===");
    for (i, s) in string_table.iter().take(50).enumerate() {
        println!("  [{}]: '{}'", i, s);
    }

    println!("\n=== Step 3: Parse KV Values with String Table ===");

    // 现在重新解析 KV 值，这次可以正确处理字符串引用
    for (i, (key, value_start)) in kv_keys.iter().enumerate() {
        let value_type =
            u32::from_le_bytes(data[*value_start..*value_start + 4].try_into().unwrap());
        let mut pos = *value_start + 4;

        println!("\nKV {}: key='{}', type={}", i, key, value_type);

        match value_type {
            9 => {
                // ARRAY
                let arr_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
                pos += 4;
                let elem_type = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                pos += 4;

                let elem_type_name = match elem_type {
                    0 => "UINT8",
                    1 => "INT8",
                    2 => "UINT16",
                    3 => "INT16",
                    4 => "UINT32",
                    5 => "INT32",
                    6 => "FLOAT32",
                    7 => "BOOL",
                    8 => "STRING",
                    9 => "ARRAY",
                    10 => "UINT64",
                    11 => "INT64",
                    12 => "FLOAT64",
                    _ => "UNKNOWN",
                };

                println!("  ARRAY<{}> len={}", elem_type_name, arr_len);

                // 如果是 tokenizer 相关的 UINT32 数组，尝试解释为字符串表索引
                if key.contains("tokenizer") && elem_type == 4 {
                    println!("  (Interpreting UINT32 values as string table indices)");

                    for j in 0..arr_len.min(20) {
                        let idx =
                            u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
                        pos += 4;

                        if idx < string_table.len() {
                            println!("    [{}]: idx={} -> '{}'", j, idx, string_table[idx]);
                        } else {
                            println!("    [{}]: idx={} -> (out of bounds)", j, idx);
                        }
                    }

                    if arr_len > 20 {
                        println!("    ... ({} more elements)", arr_len - 20);
                    }
                } else if arr_len <= 10 {
                    // 显示小数组的内容
                    match elem_type {
                        4 => {
                            // UINT32
                            for j in 0..arr_len {
                                let val =
                                    u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                                pos += 4;

                                // 尝试解释为字符串表索引
                                let idx = val as usize;
                                if idx < string_table.len() {
                                    println!("    [{}]: {} -> '{}'", j, val, string_table[idx]);
                                } else {
                                    println!("    [{}]: {}", j, val);
                                }
                            }
                        }
                        8 => {
                            // STRING
                            for j in 0..arr_len.min(5) {
                                let s_len =
                                    u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap())
                                        as usize;
                                pos += 8;
                                let s = std::str::from_utf8(&data[pos..pos + s_len.min(50)])
                                    .unwrap_or("invalid");
                                pos += s_len;
                                println!(
                                    "    [{}]: '{}'",
                                    j,
                                    if s.len() < 50 { s } else { &s[..47] }
                                );
                            }
                        }
                        _ => {}
                    }
                }
            }
            8 => {
                // STRING
                let str_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
                pos += 8;
                let s =
                    std::str::from_utf8(&data[pos..pos + str_len.min(100)]).unwrap_or("invalid");
                println!("  value: '{}'", if s.len() < 100 { s } else { &s[..97] });
            }
            4 => {
                // UINT32
                let val = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                println!("  value: {}", val);
            }
            5 => {
                // INT32
                let val = i32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                println!("  value: {}", val);
            }
            6 => {
                // FLOAT32
                let val = f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                println!("  value: {}", val);
            }
            7 => {
                // BOOL
                let val = data[pos] != 0;
                println!("  value: {}", val);
            }
            10 => {
                // UINT64
                let val = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
                println!("  value: {}", val);
            }
            _ => {
                println!("  (type {} not implemented)", value_type);
            }
        }
    }

    println!("\n=== Summary ===");
    println!("String table starts at: {}", string_table_start);
    println!("String table size: {} entries", string_table.len());
    println!("Data after string table: {}", st_pos);
    println!("File size: {}", data.len());
}
