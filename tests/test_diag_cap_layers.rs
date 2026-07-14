//! 第一性原理: 开启 diagnostic-layer-capture, 逐层 dump 输出, 定位腐败起始层
//! 假设1: layer0-2 正常, layer3 开始腐败 -> layer3 计算本身有问题
//! 假设2: layer2 已腐败, layer3 继承 -> 残差累积问题
#![cfg(test)]
use gllm::{Client, ModelKind};

#[test]
#[ignore]
fn diag_cap_layers() {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "4");
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode(" ").expect("encode");
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    let stride = c.diagnostic_layer_capture_stride();
    use std::io::Write;
    let mut f = std::fs::File::create("/tmp/cap_layers.txt").expect("f");
    let _ = writeln!(f, "capture_stride={} data.len={}", stride, sp.data.len());
    if stride == 0 {
        let _ = writeln!(f, "capture feature 未开启, 需 --features diagnostic-layer-capture");
        let _ = f.flush();
        return;
    }
    // capture_base: 找 capture 相关 offset, 或从 abi 推断
    // 通常 capture buffer 在 scratchpad 末尾或专门区域
    // 试探几个候选 base
    // 先 dump named_offsets 找 capture
    let cap_names: Vec<_> = sp.named_offsets.iter().filter(|(n,_,_)| n.to_lowercase().contains("cap")).collect();
    let _ = writeln!(f, "capture named_offsets: {:?}", cap_names);
    let _ = f.flush();
}
