use gllm::loader::Loader;
use gllm::loader::LoaderConfig;
use gllm::manifest::ModelManifest;
use std::collections::HashMap;

fn main() {
    let config = LoaderConfig::from_env();
    let mut loader = Loader::from_source_with_config("intfloat/e5-small-v2".to_string(), config.clone()).unwrap();
    
    // Check what weight format is being loaded
    println!("Weight format: {:?}", loader.weight_format());
    
    // Load the model
    loader = loader.load().unwrap();
    
    // Detect architecture
    let arch = loader.detect_architecture();
    println!("Detected architecture: {}", arch);
    
    // List some tensor names
    if let Some(st) = loader.safetensors_ref() {
        println!("\nFirst 10 tensor names:");
        let mut count = 0;
        for meta in st.iter_tensors() {
            println!("  {}", meta.name);
            count += 1;
            if count >= 10 { break; }
        }
    }
}
