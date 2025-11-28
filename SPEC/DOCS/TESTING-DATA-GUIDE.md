# gllm 测试数据指南

## 测试数据生成策略

### 原则
- **避免真实下载**: 不在测试中下载真实模型文件
- **使用假数据**: 创建最小化的 SafeTensors 文件用于测试
- **隔离测试**: 每个测试使用独立的临时目录

### 核心测试辅助函数

#### SafeTensors 文件生成
```rust
fn write_dummy_weights(path: &std::path::Path) {
    // 创建 4x4 的 FP32 权重矩阵 (64 bytes)
    let weights: Vec<u8> = vec![0u8; 64];
    let shape = vec![4usize, 4];
    let tensor = TensorView::new(Dtype::F32, shape, &weights)
        .expect("tensor view");
    let data = serialize([("dense.weight", tensor)].into_iter(), &None)
        .expect("serialize");
    fs::write(path, data).expect("write weights");
}
```

#### 临时模型目录创建
```rust
fn create_temp_model_dir() -> (tempfile::TempDir, ModelInfo) {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let model_path = temp_dir.path().join("model.safetensors");
    write_dummy_weights(&model_path);

    let model_info = ModelInfo {
        repo_id: "test/test-model".to_string(),
        model_type: ModelType::Embedding,
        architecture: Architecture::Bert,
        hidden_size: 4,
        num_attention_heads: 2,
        num_hidden_layers: 2,
        vocab_size: 1000,
        max_position_embeddings: 512,
        safe_tensors_file: "model.safetensors".to_string(),
        tokenizer_file: None,
    };

    (temp_dir, model_info)
}
```

## 测试数据类型

### 文本测试数据
- **短文本**: "Hello world", "How are you?"
- **长文本**: 生成的长段落用于测试性能
- **多语言**: 中文、英文混合测试
- **特殊字符**: 包含 emoji、符号的文本

### 模型测试数据
- **嵌入模型**: BERT 架构的小型测试模型
- **重排序模型**: Cross-Encoder 架构的小型测试模型
- **tokenizer**: 使用简单的字符级 tokenizer

### 错误测试数据
- **无效模型**: 不存在的 repo id
- **损坏文件**: 无效的 SafeTensors 格式
- **权限错误**: 只读目录
- **网络错误**: 模拟网络超时

## 数据依赖顺序

### Embedding 流程测试数据
1. 创建临时模型目录
2. 生成假 SafeTensors 文件
3. 初始化 ModelManager
4. 创建 EmbeddingEngine
5. 执行推理测试

### Rerank 流程测试数据
1. 创建临时模型目录
2. 生成假 CrossEncoder 权重
3. 初始化 RerankEngine
4. 创建测试查询和文档
5. 执行重排序测试

## 清理策略

### 自动清理
```rust
// 使用 tempfile 确保自动清理
let temp_dir = tempfile::tempdir()?;
// temp_dir 在离开作用域时自动删除
```

### 手动清理
```rust
// 清理临时模型目录
fn cleanup_temp_model_dir(path: &Path) -> Result<()> {
    if path.exists() {
        fs::remove_dir_all(path)?;
    }
    Ok(())
}
```

### 测试后清理钩子
```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn setup_test_env() {
        env::set_var("GLLM_TEST_MODE", "1");
    }

    fn cleanup_test_env() {
        env::remove_var("GLLM_TEST_MODE");
    }

    #[test]
    fn test_something() {
        setup_test_env();
        // ... 测试逻辑
        cleanup_test_env();
    }
}
```

## 随机数据生成

### 文本生成
```rust
fn generate_test_text(length: usize) -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let chars: Vec<char> = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
        .chars().collect();

    (0..length)
        .map(|_| chars[rng.gen_range(0..chars.len())])
        .collect()
}
```

### 文档列表生成
```rust
fn generate_test_documents(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| format!("Test document number {}", i))
        .collect()
}
```

## 测试数据验证

### 维度验证
```rust
fn assert_embedding_dims(embedding: &[f32], expected_dims: usize) {
    assert_eq!(embedding.len(), expected_dims,
        "Embedding dimension mismatch: expected {}, got {}",
        expected_dims, embedding.len());
}
```

### 分数验证
```rust
fn assert_rerank_scores(scores: &[f32]) {
    for (i, score) in scores.iter().enumerate() {
        assert!(score >= 0.0 && score <= 1.0,
            "Score {} out of range [0,1]: {}", i, score);
    }
}
```

### 模型加载验证
```rust
fn assert_model_loaded(model_manager: &ModelManager, model_id: &str) {
    assert!(model_manager.is_model_loaded(model_id),
        "Model {} should be loaded", model_id);
}
```