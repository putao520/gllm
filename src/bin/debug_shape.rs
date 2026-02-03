fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let safetensors_path = if args.len() > 1 {
        args[1].clone()
    } else {
        // 默认使用示例路径（可根据需要修改）
        std::env::var("HOME")
            .map(|h| format!("{}/.gllm/models/models--microsoft--Phi-4-mini-instruct/snapshots/*/model-00001-of-00002.safetensors", h))
            .unwrap_or_else(|_| "./model.safetensors".to_string())
    };

    // 读取 safetensors 文件头部
    let bytes = std::fs::read(&safetensors_path)?;

    // safetensors 格式: 8字节长度 + JSON元数据 + 数据
    let header_len = u64::from_le_bytes(bytes[0..8].try_into()?) as usize;
    let json_bytes = &bytes[8..8 + header_len];
    let json_str = std::str::from_utf8(json_bytes)?;

    println!("=== safetensors 元数据 ===");
    println!("文件: {}", safetensors_path);
    println!("{}", json_str);

    Ok(())
}
