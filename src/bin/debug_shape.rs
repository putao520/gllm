use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let safetensors_path = Path::new(
        "/home/putao/.gllm/models/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083/model-00001-of-00002.safetensors"
    );

    // 读取 safetensors 文件头部
    let bytes = std::fs::read(safetensors_path)?;

    // safetensors 格式: 8字节长度 + JSON元数据 + 数据
    let header_len = u64::from_le_bytes(bytes[0..8].try_into()?) as usize;
    let json_bytes = &bytes[8..8 + header_len];
    let json_str = std::str::from_utf8(json_bytes)?;

    println!("=== safetensors 元数据 ===");
    println!("{}", json_str);

    Ok(())
}
