use std::fs::File;
use std::io::Read;

fn main() {
    let path = std::path::Path::new("/home/putao/.gllm/models/Mungert--SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-bf16.gguf");

    let mut f = File::open(path).expect("open file");
    let mut data = Vec::new();
    f.read_to_end(&mut data).expect("read file");

    // 解析 GGUF 头部
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    println!("Magic: {:#x}", magic);

    let _version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let tensor_count = u64::from_le_bytes([
        data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23],
    ]);
    let kv_count = u64::from_le_bytes([
        data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31],
    ]);

    println!("Tensor count: {tensor_count}");
    println!("KV count: {kv_count}");

    // 扫描 KV 对
    let mut pos = 32usize;
    for i in 0..kv_count {
        // 检查边界
        if pos + 8 > data.len() {
            break;
        }

        // 读取 key length
        let key_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;

        // 检查边界
        if pos + key_len > data.len() {
            break;
        }

        // 读取 key
        let key = std::str::from_utf8(&data[pos..pos + key_len]).unwrap_or("invalid");
        pos += key_len;

        // 检查边界
        if pos + 4 > data.len() {
            break;
        }

        // 读取 value type
        let vtype = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;

        // 如果是 tokenizer.ggml 相关字段
        if key.contains("tokenizer.ggml") {
            println!("KV {i}: key='{key}', vtype={vtype}");

            if vtype == 9 {
                // ARRAY
                if pos + 8 > data.len() {
                    break;
                }
                let arr_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
                pos += 8;

                if pos + 4 > data.len() {
                    break;
                }
                let arr_type = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                pos += 4;

                println!("  ARRAY: len={arr_len}, elem_type={arr_type}");

                // 显示前几个元素
                for j in 0..arr_len.min(5) {
                    if arr_type == 4 {
                        // UINT32
                        if pos + 4 > data.len() {
                            break;
                        }
                        let val = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                        pos += 4;
                        println!("    [{j}]: {val} (u32)");
                    } else if arr_type == 8 {
                        // STRING
                        if pos + 8 > data.len() {
                            break;
                        }
                        let str_len =
                            u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
                        pos += 8;
                        if pos + str_len > data.len() {
                            break;
                        }
                        let s = std::str::from_utf8(&data[pos..pos + str_len.min(100)])
                            .unwrap_or("invalid");
                        pos += str_len;
                        println!("    [{j}]: '{s}'");
                    } else {
                        break;
                    }
                }
            }
            continue;
        }

        // 跳过这个值（简化处理）
        if vtype == 8 {
            // STRING
            if pos + 8 > data.len() {
                break;
            }
            let str_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8 + str_len;
        } else if vtype == 4 || vtype == 5 {
            // UINT32/INT32
            pos += 4;
        } else if vtype == 6 {
            // FLOAT32
            pos += 4;
        } else if vtype == 7 {
            // BOOL
            pos += 1;
        } else if vtype == 9 {
            // ARRAY
            if pos + 12 > data.len() {
                break;
            }
            let arr_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
            pos += 8;
            let arr_type = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;

            // 简单跳过 - 每个元素最多 100 字节
            let skip = (arr_len as usize).min(1000) * 8;
            pos += skip;
        } else {
            pos += 8;
        }

        if pos > data.len().min(200000) || pos > data.len() / 2 {
            break;
        }
    }

    println!("\nFinal position: {pos}, data len: {}", data.len());
}
