//! Debug GGUF KV parsing - 专门调试 KV 9 (general.tags)

use std::fs::File;
use std::io::Read;

fn main() {
    let path = std::path::Path::new(
        "/home/putao/.gllm/models/Mungert--SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-bf16.gguf"
    );

    let mut f = File::open(path).expect("open file");
    let mut data = Vec::new();
    f.read_to_end(&mut data).expect("read file");

    println!("File size: {}", data.len());

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

    // 解析前 10 个 KV 对
    for i in 0..10.min(kv_count) {
        println!("\n=== KV {} ===", i);

        // Key length
        let key_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        println!("Key length: {} bytes", key_len);

        // Key string
        let key = std::str::from_utf8(&data[pos..pos + key_len]).unwrap_or("invalid");
        pos += key_len;
        println!("Key: '{}'", key);

        // Value type
        let value_type = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        println!("Value type: {}", value_type);

        // Value type name
        let type_name = match value_type {
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
        println!("Type: {}", type_name);

        // 解析值
        if value_type == 8 {
            // STRING
            let str_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8;
            println!("String length: {}", str_len);

            if pos + str_len <= data.len() {
                let s =
                    std::str::from_utf8(&data[pos..pos + str_len.min(100)]).unwrap_or("invalid");
                pos += str_len;
                println!("Value: '{}'", if s.len() < 100 { s } else { &s[..97] });
            } else {
                println!("ERROR: String extends beyond file!");
            }
        } else if value_type == 9 {
            // ARRAY
            // 🚨 重要：GGUF 数组长度是 uint32_t，不是 uint64_t！
            let arr_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as u64;
            pos += 4;
            println!("Array length: {}", arr_len);

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
            println!("Element type: {}", elem_type_name);

            // 显示前几个元素
            if arr_len > 0 {
                println!("First {} elements:", arr_len.min(5));
                for j in 0..arr_len.min(5) {
                    match elem_type {
                        0 => {
                            // UINT8
                            let val = data[pos];
                            pos += 1;
                            println!("  [{}]: {} (u8)", j, val);
                        }
                        1 => {
                            // INT8
                            let val = data[pos] as i8;
                            pos += 1;
                            println!("  [{}]: {} (i8)", j, val);
                        }
                        4 => {
                            // UINT32
                            let val = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                            pos += 4;
                            println!("  [{}]: {} (u32)", j, val);
                        }
                        5 => {
                            // INT32
                            let val = i32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                            pos += 4;
                            println!("  [{}]: {} (i32)", j, val);
                        }
                        8 => {
                            // STRING
                            let str_len =
                                u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
                            pos += 8;
                            let s = std::str::from_utf8(&data[pos..pos + str_len.min(50)])
                                .unwrap_or("invalid");
                            pos += str_len;
                            println!("  [{}]: '{}'", j, if s.len() < 50 { s } else { &s[..47] });
                        }
                        _ => {
                            println!("  [{}]: (type {} not implemented)", j, elem_type);
                            break;
                        }
                    }
                }

                // 跳过剩余元素
                if arr_len > 5 {
                    println!("(skipping {} more elements)", arr_len - 5);

                    // 估算每个元素大小并跳过
                    let elem_size = match elem_type {
                        0 | 1 => 1,
                        2 | 3 => 2,
                        4 | 5 | 6 => 4,
                        7 => 1,
                        10 | 11 | 12 => 8,
                        _ => 1,
                    };

                    let skip_bytes = (arr_len - 5) as usize * elem_size;
                    println!("Skipping approx {} bytes", skip_bytes);
                    pos = pos.saturating_add(skip_bytes).min(data.len());
                }
            }
        } else if value_type == 7 {
            // BOOL
            let val = data[pos] != 0;
            pos += 1;
            println!("Value: {}", val);
        } else if value_type == 4 {
            // UINT32
            let val = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;
            println!("Value: {}", val);
        } else if value_type == 5 {
            // INT32
            let val = i32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;
            println!("Value: {}", val);
        } else if value_type == 6 {
            // FLOAT32
            let val = f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;
            println!("Value: {}", val);
        } else if value_type == 10 {
            // UINT64
            let val = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
            pos += 8;
            println!("Value: {}", val);
        } else if value_type == 11 {
            // INT64
            let val = i64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
            pos += 8;
            println!("Value: {}", val);
        } else if value_type == 12 {
            // FLOAT64
            let val = f64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
            pos += 8;
            println!("Value: {}", val);
        } else {
            println!("Type {} not handled, skipping 8 bytes", value_type);
            pos += 8;
        }

        println!("Current position: {}", pos);
    }

    // 继续解析 tokenizer 相关的 KV
    println!("\n\n=== Searching for tokenizer keys ===");

    // 从当前位置继续扫描
    let mut found_tokenizer_keys = 0;
    let max_search = 20; // 最多搜索 20 个 KV
    let search_start = pos;

    for _ in 0..max_search {
        if pos + 12 > data.len() {
            break;
        }

        let key_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;

        if pos + key_len > data.len() {
            break;
        }

        let key = std::str::from_utf8(&data[pos..pos + key_len]).unwrap_or("invalid");
        pos += key_len;

        if key.contains("tokenizer") {
            found_tokenizer_keys += 1;
            println!("\nFound tokenizer key: '{}'", key);

            let value_type = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;

            let type_name = match value_type {
                8 => "STRING",
                9 => "ARRAY",
                _ => "OTHER",
            };
            println!("Type: {}", type_name);

            if value_type == 8 {
                let str_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
                pos += 8;
                let s =
                    std::str::from_utf8(&data[pos..pos + str_len.min(100)]).unwrap_or("invalid");
                pos += str_len;
                println!("Value: '{}'", if s.len() < 100 { s } else { &s[..97] });
            } else if value_type == 9 {
                let arr_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
                pos += 8;
                let elem_type = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                pos += 4;
                println!("Array: len={}, elem_type={}", arr_len, elem_type);

                // 只显示前几个，不实际跳过
                if found_tokenizer_keys >= 3 {
                    break;
                }
            } else {
                // 跳过未知类型，最多跳过 16 字节
                pos += 16.min(data.len() - pos);
            }

            if found_tokenizer_keys >= 10 {
                break;
            }
        } else {
            // 跳过非 tokenizer key 的值
            let value_type = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;

            if value_type == 8 {
                // STRING
                let str_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
                pos += 8 + str_len;
            } else if value_type == 9 {
                // ARRAY
                let arr_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
                pos += 8;
                let elem_type = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                pos += 4;

                // 估算跳过
                let elem_size = match elem_type {
                    0 | 1 => 1,
                    2 | 3 => 2,
                    4 | 5 | 6 => 4,
                    7 => 1,
                    10 | 11 | 12 => 8,
                    _ => 1,
                };
                pos = pos.saturating_add((arr_len as usize).min(10000) * elem_size);
            } else {
                pos += 8;
            }

            // 安全检查
            if pos > data.len() {
                break;
            }
        }
    }

    println!("\n\nFinal position: {}", pos);
    println!("File size: {}", data.len());
}
