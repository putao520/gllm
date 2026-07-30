//! 本地 W256 Q8_0 VmInstr dump（对照 5070Ti W512 Gemma4）
#![cfg(test)]
use gllm::{Client, ModelKind};
#[test]
#[ignore]
fn q8_dump_local() {
    let outdir = "/tmp/local-q8-disasm";
    let _ = std::fs::remove_dir_all(outdir);
    std::fs::create_dir_all(outdir).unwrap();
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "1");
    std::env::set_var("GLLM_DUMP_MEGA", outdir);
    std::env::set_var("GLLM_DUMP_OFFSETMAP", "1");
    let c = Client::builder()
        .model("bartowski/Qwen_Qwen3-0.6B-GGUF")
        .kind(ModelKind::Chat)
        .gguf_file_filter("q8_0")
        .build()
        .unwrap_or_else(|e| panic!("load: {e:?}"));
    let t = c.encode("The").unwrap();
    let _ = c.diagnostic_prefill_logits(&t);
    eprintln!("dump at {outdir}/mega_kernel_vm.txt");
}
