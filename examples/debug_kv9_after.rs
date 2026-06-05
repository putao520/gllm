//! Debug KV 9 and beyond in GGUF file

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

    // 解析前 15 个 KV 对
    for i in 0..15.min(kv_count) {
        println!("\n=== KV {} ===", i);
        println!("Current position: {}", pos);

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
            // 数组长度是 uint32_t
            let arr_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
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

            // 跳过数组元素
            match elem_type {
                0 => {
                    pos += arr_len;
                } // UINT8
                4 => {
                    pos += arr_len * 4;
                } // UINT32
                8 => {
                    // STRING - 需要跳过每个字符串的长度前缀
                    for _ in 0..arr_len.min(5) {
                        let str_len =
                            u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
                        pos += 8;
                        println!("  String element len: {}", str_len);
                        let s = std::str::from_utf8(&data[pos..pos + str_len.min(50)])
                            .unwrap_or("invalid");
                        pos += str_len;
                        println!(
                            "  String value: '{}'",
                            if s.len() < 50 { s } else { &s[..47] }
                        );
                    }
                    // 跳过剩余元素
                    if arr_len > 5 {
                        println!("  (skipping {} more elements)", arr_len - 5);
                        // 粗略估计每个字符串平均 10 字节
                        pos += (arr_len - 5) * 10;
                    }
                }
                _ => {
                    println!("  (unknown element type, skipping {} bytes)", arr_len * 4);
                    pos += arr_len * 4;
                }
            }
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
        } else {
            println!("Type {} not handled, skipping 8 bytes", value_type);
            pos += 8;
        }

        println!("Position after KV: {}", pos);

        // 安全检查
        if pos > data.len() {
            println!("ERROR: Position exceeds file size!");
            break;
        }
    }

    println!("\n\nFinal position: {}", pos);
    println!("File size: {}", data.len());
}
