//! Test GGUF aux files

use gllm::loader::Loader;

#[test]
fn test_gguf_aux_files() {
    let loader = Loader::from("Mungert/SmolLM2-135M-Instruct-GGUF").expect("Failed to load");
    println!("aux_files = {:?}", loader.aux_files());
    println!("config_path = {:?}", loader.config_path());
}
