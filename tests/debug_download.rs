
#[cfg(test)]
mod tests {
    use gllm::loader::hf_hub::HfHubClient;
    use gllm::manifest::EMPTY_FILE_MAP;
    use gllm::loader::parallel::ParallelLoader;
    use std::path::PathBuf;

    #[test]
    fn debug_safetensors_tokenizer_download() {
        let repo = "HuggingFaceTB/SmolLM2-135M-Instruct";
        let cache_dir = PathBuf::from("/tmp/gllm_debug_cache");
        let client = HfHubClient::new(cache_dir.clone()).expect("Failed to create client");

        println!("Checking repo: {}", repo);

        // 1. 直接尝试下载 tokenizer.json
        match client.download_tokenizer_file(repo, EMPTY_FILE_MAP) {
            Ok(path) => println!("✅ Directly found tokenizer: {:?}", path),
            Err(e) => println!("❌ Failed to find tokenizer directly: {}", e),
        }

        // 2. 模拟 Loader 的完整流程
        let parallel = ParallelLoader::new(false);
        match client.download_model_files(repo, EMPTY_FILE_MAP, parallel) {
            Ok(files) => {
                println!("✅ Downloaded files:");
                for path in &files.aux_files {
                    println!("   - {:?}", path.file_name().unwrap());
                }

                let has_tokenizer = files.aux_files.iter().any(|p| p.file_name().unwrap() == "tokenizer.json");
                if has_tokenizer {
                    println!("✅ tokenizer.json is present in aux_files");
                } else {
                    println!("❌ tokenizer.json is MISSING from aux_files");
                }
            },
            Err(e) => println!("❌ Full download failed: {}", e),
        }
    }

    #[test]
    fn debug_gguf_base_model_resolution() {
        let repo = "Qwen/Qwen3-0.6B-GGUF";
        let cache_dir = PathBuf::from("/tmp/gllm_debug_cache");
        let client = HfHubClient::new(cache_dir).expect("Failed to create client");

        println!("Checking GGUF repo: {}", repo);

        // 1. 验证是否能找到 tokenizer.json (预期失败，因为 GGUF 仓库通常没有)
        match client.download_tokenizer_file(repo, EMPTY_FILE_MAP) {
            Ok(path) => println!("✅ Found tokenizer in GGUF repo: {:?}", path),
            Err(_) => println!("ℹ️ Tokenizer not in GGUF repo (Expected)"),
        }

        // 2. 验证 Base Model 解析能力
        // 我们不能直接测试私有方法 resolve_base_model_repo，但可以通过 download_tokenizer_file 的行为推断
        // 如果 resolve 生效，它应该能从 Qwen/Qwen3-0.6B 下载
        println!("Attempting to resolve from base model...");
        // 这里我们重新尝试，逻辑内部应该会自动 fallback
        match client.download_tokenizer_file(repo, EMPTY_FILE_MAP) {
            Ok(path) => println!("✅ Successfully resolved tokenizer from base model: {:?}", path),
            Err(e) => println!("❌ Failed to resolve from base model: {}", e),
        }
    }
}
