use std::path::Path;

fn main() {
    compile_onnx_proto();
    generate_template_list();
}

fn compile_onnx_proto() {
    let proto_path = "src/loader/onnx/onnx3.proto";
    if !Path::new(proto_path).exists() {
        return;
    }

    println!("cargo:rerun-if-changed={}", proto_path);

    let mut config = prost_build::Config::new();
    config.bytes([".onnx.TensorProto.raw_data"]);
    config
        .compile_protos(&[proto_path], &["src/loader/onnx"])
        .expect("failed to compile onnx proto");
}

/// 扫描 `src/arch/templates/` 目录，生成编译期模板列表。
///
/// 输出 `$OUT_DIR/template_list.rs`，内容为：
/// ```rust
/// const SCANNED_TEMPLATES: &[&str] = &[
///     include_str!("/.../src/arch/templates/qwen3.yaml"),
///     include_str!("/.../src/arch/templates/mistral3.yaml"),
///     ...
/// ];
/// ```
///
/// registry.rs 通过 `include!(concat!(env!("OUT_DIR"), "/template_list.rs"))` 引入。
fn generate_template_list() {
    let templates_dir = Path::new("src/arch/templates");
    println!("cargo:rerun-if-changed={}", templates_dir.display());

    if !templates_dir.is_dir() {
        return;
    }

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let mut entries: Vec<String> = Vec::new();

    for entry in std::fs::read_dir(templates_dir).expect("failed to read templates dir") {
        let entry = entry.expect("failed to read dir entry");
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "yaml" || ext == "yml") {
            // 使用绝对路径确保 include_str! 能找到文件
            let abs_path = Path::new(&manifest_dir).join(&path);
            let abs_str = abs_path.to_str().expect("non-utf8 path");
            entries.push(format!("    include_str!(\"{abs_str}\")"));
            // 监听每个 YAML 文件的变更
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }

    // 按文件名排序保证确定性
    entries.sort();

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir).join("template_list.rs");
    let content = format!(
        "const SCANNED_TEMPLATES: &[&str] = &[\n{}\n];\n",
        entries.join(",\n")
    );
    std::fs::write(&out_path, content).expect("failed to write template_list.rs");
}
