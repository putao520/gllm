use std::path::Path;

fn main() {
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
