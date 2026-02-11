# gllm 任务完成检查清单

## 编码完成后必做

### 1. 代码质量检查

```bash
# 编译检查
cargo check

# 格式化
cargo fmt

# Lint 检查
cargo clippy
```

### 2. 测试验证

```bash
# 单元测试
cargo test --lib

# 相关集成测试
cargo test --test <test_name>

# 🚨 E2E 测试必须单线程
cargo test --test test_e2e_* -- --test-threads=1
```

### 3. 文档更新

- [ ] 新增公共 API 需要文档注释
- [ ] 重大变更需更新 README.md
- [ ] 架构变更需更新 SPEC/ 文档

### 4. SPEC 一致性

- [ ] 检查实现与 SPEC 文档一致
- [ ] 新功能需要更新 SPEC/01-REQUIREMENTS.md
- [ ] API 变更需要更新 SPEC/04-API-DESIGN.md

## 提交前检查

```bash
# 完整检查流程
cargo fmt --check
cargo clippy -- -D warnings
cargo test
```

## E2E 测试约束

⚠️ **E2E 测试涉及真实模型下载和推理，必须单线程运行**

原因：
- 磁盘 I/O 竞争
- 模型缓存冲突
- CPU 内存超限
- 测试结果不可预测

## 版本更新

如果需要发布新版本：
1. 更新 `Cargo.toml` 中的 version
2. 更新 README.md 中的版本号
3. 确保所有测试通过
