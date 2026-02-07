use std::fs::File;
use std::io::Read;

fn main() {
    let path = std::path::Path::new("/home/putao/.gllm/models/Mungert--SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-bf16.gguf");

    let mut f = File::open(path).expect("open file");
    let mut data = Vec::new();
    f.read_to_end(&mut data).expect("read file");

    let kv_count = u64::from_le_bytes([
        data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31],
    ]);

    println!("KV count: {kv_count}\n");

    let mut pos = 32usize;
    for i in 0..kv_count {
        let key_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;

        let key = std::str::from_utf8(&data[pos..pos + key_len]).unwrap_or("invalid");
        pos += key_len;

        let vtype = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;

        let type_name = match vtype {
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

        // 跳过值（简化处理）
        if vtype == 8 {
            // STRING
            let str_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8 + str_len;
        } else if vtype <= 7 {
            pos += 8;
        } else if vtype == 9 {
            // ARRAY
            let arr_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
            pos += 8;
            let arr_type = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;
            // 简单跳过
            pos += (arr_len as usize).min(100) * 4;
        } else {
            pos += 16;
        }

        println!("{i:2}. {key:50} {type_name}");
    }
}
