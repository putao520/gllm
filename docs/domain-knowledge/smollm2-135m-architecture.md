# SmolLM2-135M-Instruct 架构资料库（C-9，堵 AI 幻觉）

> 来源：本地推理用的 config.json（ground truth，非训练数据猜）
> 路径：`~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/*/config.json`
> 建库触发（C-9）：CPU BUG 诊断 8 轮，head_dim/num_kv_heads 等基础事实多次猜错（以为 kv_heads=8/head_dim=32，实际 kv_heads=3/head_dim=64）
> 最后验证：2026-07-05

## 确定性架构事实（config.json，ground truth）

```json
{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "hidden_size": 576,
  "num_hidden_layers": 30,
  "num_attention_heads": 9,
  "num_key_value_heads": 3,
  "vocab_size": 49152,
  "intermediate_size": 1536,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "rope_theta": 100000,
  "rope_interleaved": false,
  "rope_scaling": null,
  "max_position_embeddings": 8192,
  "rms_norm_eps": 1e-05,
  "hidden_act": "silu",
  "attention_bias": false,
  "mlp_bias": false,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "use_cache": true
}
```

## 关键派生量

| 量 | 值 | 计算 |
|----|-----|------|
| head_dim | **64** | hidden_size / num_attention_heads = 576 / 9 |
| num_kv_heads | **3** | config 直读（GQA，不是 8） |
| kv_heads 与 q_heads 比 | 3:9 = 1:3 GQA | 每个 KV head 服务 3 个 Q head |
| head_dim × num_kv_heads | 3×64 = 192 | kv_proj 输出维度（K/V 各 192 维） |
| lm_head.weight | == embed_tokens.weight | tie_word_embeddings=true（tied） |
| weight dtype | **BF16** | torch_dtype=bfloat16（blob 保留原始 BF16，禁转换） |
| position embedding | **无** | Llama 用 RoPE，无 learnable position embedding |

## AI 易误判点（本轮诊断踩坑）

| ❌ 误判 | ✅ 正解（config 证明） |
|--------|---------|
| num_kv_heads=8 | num_key_value_heads=**3** |
| head_dim=32 | head_dim=**64**（576/9） |
| SmolLM2 有 position embedding | **无**（Llama 系 RoPE，无 learnable PE） |
| lm_head 独立权重 | **tied**（lm_head.weight == embed_tokens.weight） |
| 权重 F32 | **BF16**（torch_dtype=bfloat16，blob 保留原始） |

## 解决问题时参考

### KV cache 大小计算
```
kv_cache_bytes = num_layers × 2 (K+V) × max_seq_len × num_kv_heads × head_dim × elem_bytes
             = 30 × 2 × max_seq_len × 3 × 64 × elem_bytes
             = 11520 × max_seq_len × elem_bytes
```
注意 num_kv_heads=3（不是 num_attention_heads=9），head_dim=64（不是 32）。

### Embedding/Gather 维度
- embed_tokens.weight shape: [vocab_size=49152, hidden_size=576]
- embedding 输出: [seq_len, hidden=576]
- gather token i → weight row i (576 维)，row stride = 576 × elem_bytes (BF16=2, F32=4)

### RoPE 配置
- theta = 100000（不是 Llama 默认 10000）
- interleaved = false（GPT-NeoX 风格，非 GPT-J interleaved）
- scaling = null（无 long-context scaling）
- **partial = 1.0（全维度旋转，标准 RoPE）** — config.json 无 rope_partial_ratio 字段，gllm 默认 unwrap_or(1.0)（types.inc.rs:151）。SmolLM2 是 Llama 架构，用标准全维度 RoPE，非 Gemma 4 的 p-RoPE 0.25。

### 推理模式（prefill vs decode）
- prefill: prompt_len 个 token 一次过（M=seq_len），Gather 应写 [seq_len, 576]
- decode: 每迭代 1 token（M=1），Gather 写 [1, 576]
- 当前 BUG: SmolLM2 prefill 5 token，但实测 Gather 只写 row0（疑似 decode M=1 路径误用，待证）

## 验证命令（确定性）
```bash
# 读 config（ground truth，非猜）
cat ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/*/config.json | jq .

# 读 weight dtype（BF16 确认）
python3 -c "
from safetensors import safe_open
import glob
f = glob.glob('/home/putao/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/*/model.safetensors')[0]
with safe_open(f, framework='pt') as sf:
    for k in ['model.embed_tokens.weight', 'lm_head.weight']:
        t = sf.get_tensor(k)
        print(k, t.shape, t.dtype)
"
```

## 与其他资料库关系
- `cuda-driver-api.md`: GPU launch ABI（host/device ptr）
- 本文件: SmolLM2 架构事实（CPU/GPU 共用）
