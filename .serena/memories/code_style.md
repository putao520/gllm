# gllm 代码风格和约定

## Rust 风格

- **Edition**: Rust 2021
- **格式化**: 使用 `cargo fmt`（rustfmt 默认配置）
- **Lint**: 使用 `cargo clippy`

## 命名约定

| 类型 | 风格 | 示例 |
|------|------|------|
| 类型/结构体 | PascalCase | `Client`, `ModelConfig` |
| 函数/方法 | snake_case | `load_model`, `execute_generation` |
| 常量 | SCREAMING_SNAKE_CASE | `MAX_BATCH_SIZE` |
| 模块 | snake_case | `model_config`, `weight_loader` |
| Feature | kebab-case | `paged-attention`, `flash-attention` |

## 模块组织

- 每个模型架构一个适配器文件：`src/adapter/<model>.rs`
- Trait 定义在 `src/adapter/trait.rs`
- 模块入口在 `mod.rs` 中 re-export

## 错误处理

```rust
// 使用 thiserror 定义错误类型
#[derive(thiserror::Error, Debug)]
pub enum ClientError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    // ...
}

// 使用 Result 返回
pub fn load_model(&self) -> Result<(), ClientError> { ... }
```

## 文档注释

```rust
/// 简短描述（一行）
///
/// 详细描述（可选）
///
/// # Arguments
/// * `arg1` - 参数说明
///
/// # Returns
/// 返回值说明
///
/// # Errors
/// 可能的错误
pub fn example(arg1: &str) -> Result<(), Error> { ... }
```

## 类型安全

- 避免 `unwrap()`，使用 `?` 或 `expect("reason")`
- 使用强类型而非原始类型（如 `ModelId` 而非 `String`）
- 使用泛型和 trait 提高复用

## 性能考虑

- 零拷贝优先：使用 `memmap2` 映射文件
- 避免不必要的 clone
- 使用 `rayon` 进行并行处理

## Profile 配置（Release）

```toml
[profile.release]
lto = "thin"
codegen-units = 1
panic = "abort"
opt-level = 3
```

## 禁止事项

- ❌ 禁止擅自添加环境变量
- ❌ 禁止使用 `unsafe` 除非绝对必要
- ❌ 禁止 `println!` 调试（使用 `log` crate）
- ❌ 禁止硬编码路径
