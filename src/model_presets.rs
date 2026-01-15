use crate::model_config::ModelConfig;
use serde_json::{json, Map, Value};

pub(crate) fn model_defaults(repo_id: &str) -> ModelConfig {
    let key = repo_id.to_ascii_lowercase();
    match key.as_str() {
        // BGE Models
        "baai/bge-small-zh-v1.5" => preset(
            512,
            4,
            8,
            2048,
            512,
            21128,
            "bert",
            &["BertModel"],
            1e-12,
            2,
            0,
            0.1,
            0.1,
        ),
        "baai/bge-small-en-v1.5" => preset(
            384,
            12,
            12,
            1536,
            512,
            30522,
            "bert",
            &["BertModel"],
            1e-12,
            2,
            0,
            0.1,
            0.1,
        ),
        "baai/bge-base-en-v1.5" => preset(
            768,
            12,
            12,
            3072,
            512,
            30522,
            "bert",
            &["BertModel"],
            1e-12,
            2,
            0,
            0.1,
            0.1,
        ),
        "baai/bge-large-en-v1.5" => preset(
            1024,
            24,
            16,
            4096,
            512,
            30522,
            "bert",
            &["BertModel"],
            1e-12,
            2,
            0,
            0.1,
            0.1,
        ),

        // Sentence Transformers
        "sentence-transformers/all-minilm-l6-v2" => preset(
            384,
            6,
            12,
            1536,
            512,
            30522,
            "bert",
            &["BertModel"],
            1e-12,
            2,
            0,
            0.1,
            0.1,
        ),
        "sentence-transformers/all-minilm-l12-v2" => preset(
            384,
            12,
            12,
            1536,
            512,
            30522,
            "bert",
            &["BertModel"],
            1e-12,
            2,
            0,
            0.1,
            0.1,
        ),
        "sentence-transformers/all-mpnet-base-v2" => preset(
            768,
            12,
            12,
            3072,
            514,
            30527,
            "mpnet",
            &["MPNetForMaskedLM"],
            1e-5,
            1,
            1,
            0.1,
            0.1,
        ),
        "sentence-transformers/paraphrase-minilm-l6-v2" => preset(
            384,
            6,
            12,
            1536,
            512,
            30522,
            "bert",
            &["BertModel"],
            1e-12,
            2,
            0,
            0.1,
            0.1,
        ),
        "sentence-transformers/multi-qa-mpnet-base-dot-v1" => preset(
            768,
            12,
            12,
            3072,
            514,
            30527,
            "mpnet",
            &["MPNetForMaskedLM"],
            1e-5,
            1,
            1,
            0.1,
            0.1,
        ),
        "sentence-transformers/all-distilroberta-v1" => preset(
            768,
            6,
            12,
            3072,
            514,
            50265,
            "roberta",
            &["RobertaForMaskedLM"],
            1e-5,
            1,
            1,
            0.1,
            0.1,
        ),
        "sentence-transformers/paraphrase-multilingual-minilm-l12-v2" => preset(
            384,
            12,
            12,
            1536,
            512,
            250037,
            "bert",
            &["BertModel"],
            1e-12,
            2,
            0,
            0.1,
            0.1,
        ),
        "sentence-transformers/distiluse-base-multilingual-cased-v1" => preset(
            768,
            6,
            12,
            3072,
            512,
            119547,
            "distilbert",
            &["DistilBertModel"],
            1e-12,
            2,
            0,
            0.1,
            0.1,
        ),

        // Chinese Models
        "moka-ai/m3e-base" => preset(
            768,
            12,
            12,
            3072,
            512,
            21128,
            "bert",
            &["BertModel"],
            1e-12,
            2,
            0,
            0.1,
            0.1,
        ),

        // Code Models
        "microsoft/codebert-base" => preset(
            768,
            12,
            12,
            3072,
            514,
            50265,
            "roberta",
            &["RobertaModel"],
            1e-5,
            1,
            1,
            0.1,
            0.1,
        ),
        "microsoft/graphcodebert-base" => preset(
            768,
            12,
            12,
            3072,
            514,
            50265,
            "roberta",
            &["RobertaModel"],
            1e-5,
            1,
            1,
            0.1,
            0.1,
        ),
        "microsoft/unixcoder-base" => preset(
            768,
            12,
            12,
            3072,
            514,
            50265,
            "roberta",
            &["RobertaModel"],
            1e-5,
            1,
            1,
            0.1,
            0.1,
        ),
        "bigcode/starencoder" => preset(
            768,
            12,
            12,
            3072,
            512,
            49152,
            "bert",
            &["BertModel"],
            1e-12,
            0, // type_vocab_size unknown for starencoder, usually 0 or 2 for BERT
            0,
            0.1,
            0.1,
        ),
        "salesforce/sfr-embedding-code-2b_r" => codex_embed_qwen2_preset(),
        "salesforce/sfr-embedding-code-7b_r" => codex_embed_mistral_preset(),
        "qwen/qwen2-7b-instruct" => qwen2_instruct_preset(),
        "qwen/qwen2.5-0.5b-instruct" => qwen25_instruct_preset(
            896,
            24,
            14,
            2,
            4864,
            32768,
            151936,
            1_000_000.0,
            1e-6,
        ),
        "qwen/qwen2.5-1.5b-instruct" => qwen25_instruct_preset(
            1536,
            28,
            12,
            2,
            8960,
            32768,
            151936,
            1_000_000.0,
            1e-6,
        ),
        "qwen/qwen2.5-3b-instruct" => qwen25_instruct_preset(
            2048,
            36,
            16,
            2,
            11008,
            32768,
            151936,
            1_000_000.0,
            1e-6,
        ),
        "qwen/qwen2.5-7b-instruct" => qwen25_instruct_preset(
            3584,
            28,
            28,
            4,
            18944,
            32768,
            152064,
            1_000_000.0,
            1e-6,
        ),
        "qwen/qwen2.5-14b-instruct" => qwen25_instruct_preset(
            5120,
            48,
            40,
            8,
            13824,
            32768,
            152064,
            1_000_000.0,
            1e-6,
        ),
        "qwen/qwen2.5-32b-instruct" => qwen25_instruct_preset(
            5120,
            64,
            40,
            8,
            27648,
            32768,
            152064,
            1_000_000.0,
            1e-6,
        ),
        "qwen/qwen2.5-72b-instruct" => qwen25_instruct_preset(
            8192,
            80,
            64,
            8,
            29568,
            32768,
            152064,
            1_000_000.0,
            1e-6,
        ),
        // Qwen3 Generator Models
        "qwen/qwen3-0.6b" => qwen3_preset(
            1024,
            28,
            16,
            8,
            3072,
            151936,
            40960,
            1_000_000.0,
            1e-6,
        ),
        "qwen/qwen3-1.7b" => qwen3_preset(
            2048,
            28,
            16,
            8,
            6144,
            151936,
            40960,
            1_000_000.0,
            1e-6,
        ),
        "qwen/qwen3-4b" => qwen3_preset(
            2560,
            36,
            32,
            8,
            9728,
            151936,
            40960,
            1_000_000.0,
            1e-6,
        ),
        "qwen/qwen3-8b" => qwen3_preset(
            4096,
            36,
            32,
            8,
            12288,
            151936,
            40960,
            1_000_000.0,
            1e-6,
        ),
        "qwen/qwen3-14b" => qwen3_preset(
            5120,
            40,
            40,
            8,
            17408,
            151936,
            40960,
            1_000_000.0,
            1e-6,
        ),
        "qwen/qwen3-32b" => qwen3_preset(
            5120,
            64,
            64,
            8,
            25600,
            151936,
            40960,
            1_000_000.0,
            1e-6,
        ),
        "qwen/qwen3-30b-a3b" => qwen3_moe_preset(
            2048,   // hidden_size
            48,     // layers
            32,     // heads
            4,      // kv_heads
            128,    // head_dim (explicit, non-standard: 32*128=4096 != 2048)
            6144,   // intermediate
            151936, // vocab
            40960,  // max_pos
            128,    // num_experts
            8,      // num_experts_per_tok
            None,
            1_000_000.0,
            1e-6,
        ),
        "qwen/qwen3-235b-a22b" => qwen3_moe_preset(
            4096,   // hidden_size
            94,     // layers
            64,     // heads
            4,      // kv_heads
            128,    // head_dim (explicit, non-standard: 64*128=8192 != 4096)
            12288,  // intermediate
            151936, // vocab
            40960,  // max_pos
            128,    // num_experts
            8,      // num_experts_per_tok
            Some(1536),
            1_000_000.0,
            1e-6,
        ),
        // Phi-4 Models
        "microsoft/phi-4" => phi4_preset(
            5120,
            40,
            40,
            10,
            17920,
            100352,
            16384,
            250_000.0,
            1e-5,
        ),
        "microsoft/phi-4-mini-instruct" => phi4_preset(
            3072,
            32,
            24,
            8,
            8192,
            200064,
            131072,
            10000.0,
            1e-5,
        ),
        // SmolLM3 Models
        "huggingfacetb/smollm3-3b" => smollm3_preset(
            2048,
            36,
            16,
            4,
            11008,
            128256,
            65536,
            5_000_000.0,
            1e-6,
        ),
        // InternLM3 Models
        "internlm/internlm3-8b-instruct" => internlm3_preset(
            4096,
            48,
            32,
            2,
            10240,
            128512,
            32768,
            50_000_000.0,
            1e-5,
        ),
        "mistralai/mixtral-8x7b-instruct-v0.1" => mixtral_preset(
            4096,
            32,
            32,
            8,
            14336,
            32768,
            32000,
            8,
            2,
            1_000_000.0,
            1e-5,
        ),
        "mistralai/mixtral-8x22b-instruct-v0.1" => mixtral_preset(
            6144,
            56,
            48,
            8,
            16384,
            65536,
            32768,
            8,
            2,
            1_000_000.0,
            1e-5,
        ),
        "thudm/glm-4-9b-chat-hf" => glm4_chat_preset(
            4096,
            40,
            32,
            2,
            13696,
            131072,
            151552,
            10000.0,
            1.5625e-7,
        ),
        "zai-org/glm-4.7" => glm47_moe_preset(
            5120,   // hidden_size
            92,     // layers
            96,     // heads
            8,      // kv_heads
            128,    // head_dim (explicit: 96*128=12288 != hidden_size=5120)
            12288,  // intermediate
            202752, // max_pos
            151552, // vocab
            1_000_000.0,
            1e-5,
        ),
        "deepseek-ai/deepseek-v3" => deepseek_v3_preset(),

        // GPT-OSS Models (OpenAI 2025 Open Source MoE)
        "openai/gpt-oss-20b" => gpt_oss_preset(
            2880,   // hidden_size
            24,     // layers
            64,     // heads
            8,      // kv_heads
            64,     // head_dim (non-standard: 64*64=4096 != hidden_size)
            2880,   // intermediate_size (same as hidden_size for GPT-OSS)
            201088, // vocab
            131072, // max_pos (128K context via YaRN)
            32,     // num_experts
            4,      // experts_per_tok
            150000.0, // rope_theta
            1e-5,   // rms_norm_eps
        ),
        "openai/gpt-oss-120b" => gpt_oss_preset(
            2880,   // hidden_size
            36,     // layers
            64,     // heads
            8,      // kv_heads
            64,     // head_dim (non-standard)
            2880,   // intermediate_size
            201088, // vocab
            131072, // max_pos (128K context via YaRN)
            128,    // num_experts
            4,      // experts_per_tok
            150000.0, // rope_theta
            1e-5,   // rms_norm_eps
        ),

        // E5 Models
        repo if repo.starts_with("intfloat/e5-small") => preset(
            384,
            12,
            12,
            1536,
            512,
            30522,
            "bert",
            &["BertModel"],
            1e-12,
            2,
            0,
            0.1,
            0.1,
        ),
        repo if repo.starts_with("intfloat/e5-base") => preset(
            768,
            12,
            12,
            3072,
            512,
            30522,
            "bert",
            &["BertModel"],
            1e-12,
            2,
            0,
            0.1,
            0.1,
        ),
        repo if repo.starts_with("intfloat/e5-large") => preset(
            1024,
            24,
            16,
            4096,
            512,
            30522,
            "bert",
            &["BertModel"],
            1e-12,
            2,
            0,
            0.1,
            0.1,
        ),

        // JINA Models
        repo if repo.starts_with("jinaai/jina-embeddings-v2-small") => preset(
            512,
            4,
            8,
            2048,
            8192,
            30528,
            "bert",
            &["JinaBertForMaskedLM"],
            1e-12,
            2,
            0,
            0.0,
            0.1,
        ),
        repo if repo.starts_with("jinaai/jina-embeddings-v2-base") => preset(
            768,
            12,
            12,
            3072,
            8192,
            30528,
            "bert",
            &["JinaBertForMaskedLM"],
            1e-12,
            2,
            0,
            0.0,
            0.1,
        ),
        "jinaai/jina-embeddings-v4" => jina_v4_preset(),

        // Qwen3 Embedding Models
        "qwen/qwen3-embedding-0.6b" => qwen3_embedding_preset(
            896,
            24,
            14,
            2,
            4864,
            32768,
            151936,
            1_000_000.0,
            1e-6,
        ),
        "qwen/qwen3-embedding-4b" => qwen3_embedding_preset(
            2560,
            36,
            32,
            4,
            13824,
            32768,
            151936,
            1_000_000.0,
            1e-6,
        ),
        "qwen/qwen3-embedding-8b" => qwen3_embedding_preset(
            4096,
            36,
            32,
            8,
            14336,
            32768,
            151936,
            1_000_000.0,
            1e-6,
        ),

        // NVIDIA Embedding
        "nvidia/llama-embed-nemotron-8b" => nvidia_nemotron_preset(
            4096,
            32,
            32,
            8,
            14336,
            131072,
            128256,
            500_000.0,
            1e-5,
        ),

        // BGE Rerankers
        "baai/bge-reranker-v2-m3" => with_rerank(preset(
            1024,
            24,
            16,
            4096,
            8194,
            250002,
            "xlm-roberta",
            &["XLMRobertaForSequenceClassification"],
            1e-5,
            1,
            1,
            0.1,
            0.1,
        )),
        "baai/bge-reranker-large" => with_rerank(preset(
            1024,
            24,
            16,
            4096,
            514,
            250002,
            "xlm-roberta",
            &["XLMRobertaForSequenceClassification"],
            1e-5,
            1,
            1,
            0.1,
            0.1,
        )),
        "baai/bge-reranker-base" => with_rerank(preset(
            768,
            12,
            12,
            3072,
            514,
            250002,
            "xlm-roberta",
            &["XLMRobertaForSequenceClassification"],
            1e-5,
            1,
            1,
            0.1,
            0.1,
        )),

        // MS MARCO Rerankers
        repo if repo.starts_with("cross-encoder/ms-marco-minilm-l-6-v2") => with_rerank(preset(
            384,
            6,
            12,
            1536,
            512,
            30522,
            "bert",
            &["BertForSequenceClassification"],
            1e-12,
            2,
            0,
            0.1,
            0.1,
        )),
        repo if repo.starts_with("cross-encoder/ms-marco-minilm-l-12-v2") => with_rerank(preset(
            384,
            12,
            12,
            1536,
            512,
            30522,
            "bert",
            &["BertForSequenceClassification"],
            1e-12,
            2,
            0,
            0.1,
            0.1,
        )),
        repo if repo.starts_with("cross-encoder/ms-marco-tinybert-l-2-v2") => with_rerank(preset(
            128,
            2,
            2,
            512,
            512,
            30522,
            "bert",
            &["BertForSequenceClassification"],
            1e-12,
            2,
            0,
            0.1,
            0.1,
        )),
        repo if repo.starts_with("cross-encoder/ms-marco-electra-base") => with_rerank(preset(
            768,
            12,
            12,
            3072,
            512,
            30522,
            "electra",
            &["ElectraForSequenceClassification"],
            1e-12,
            2,
            0,
            0.1,
            0.1,
        )),
        "cross-encoder/quora-distilroberta-base" => with_rerank(preset(
            768,
            6,
            12,
            3072,
            514,
            50265,
            "roberta",
            &["RobertaForSequenceClassification"],
            1e-5,
            1,
            1,
            0.1,
            0.1,
        )),

        // Qwen3 Reranker Models
        "qwen/qwen3-reranker-0.6b" => qwen3_reranker_preset(
            896,
            24,
            14,
            2,
            4864,
            32768,
            151936,
            1_000_000.0,
            1e-6,
        ),
        "qwen/qwen3-reranker-4b" => qwen3_reranker_preset(
            2560,
            36,
            32,
            4,
            13824,
            32768,
            151936,
            1_000_000.0,
            1e-6,
        ),
        "qwen/qwen3-reranker-8b" => qwen3_reranker_preset(
            4096,
            36,
            32,
            8,
            14336,
            32768,
            151936,
            1_000_000.0,
            1e-6,
        ),

        // Jina Reranker V3
        "jinaai/jina-reranker-v3" => jina_reranker_v3_preset(),

        // Default
        _ => preset(
            768,
            12,
            12,
            3072,
            512,
            30522,
            "bert",
            &["BertModel"],
            1e-12,
            2,
            0,
            0.1,
            0.1,
        ),
    }
}

fn qwen3_embedding_preset(
    hidden_size: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    intermediate: usize,
    max_pos: usize,
    vocab: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
) -> ModelConfig {
    ModelConfig {
        architectures: Some(vec!["Qwen3ForCausalLM".to_string()]),
        model_type: Some("qwen3".to_string()),
        hidden_size,
        num_hidden_layers: layers,
        num_attention_heads: heads,
        num_key_value_heads: Some(kv_heads),
        head_dim: None,
        vocab_size: vocab,
        max_position_embeddings: max_pos,
        attention_probs_dropout_prob: Some(0.0),
        hidden_dropout_prob: Some(0.0),
        intermediate_size: Some(intermediate),
        num_experts: None,
        num_experts_per_tok: None,
        n_shared_experts: None,
        moe_intermediate_size: None,
        max_batch_size: None,
        memory_limit_mb: None,
        gpu_memory_fraction: None,
        hidden_act: Some("silu".to_string()),
        initializer_range: None,
        layer_norm_eps: None,
        rms_norm_eps: Some(rms_norm_eps),
        rope_theta: Some(rope_theta),
        rope_scaling: None,
        sliding_window: None,
        use_cache: Some(true),
        position_embedding_type: Some("rope".to_string()),
        pooler_hidden_act: None,
        pooler_dropout: None,
        pooling_type: None,
        num_labels: None,
        classifier_dropout: None,
        tie_word_embeddings: Some(true),
        is_decoder: Some(false),
        cross_attention_hidden_size: None,
        pad_token_id: Some(0),
        bos_token_id: None,
        eos_token_id: None,
        type_vocab_size: Some(1),
        extra: Value::Object(Map::new()),
        // Engram conditional memory (disabled by default)
        engram_enabled: None,
        engram_ngram_size: None,
        engram_num_buckets: None,
        engram_embedding_dim: None,
        engram_scale: None,
    }
}

fn codex_embed_qwen2_preset() -> ModelConfig {
    decoder_embedding_preset(
        1536,
        28,
        12,
        2,
        128,
        8960,
        131072,
        151936,
        1_000_000.0,
        1e-6,
        None,
        "qwen2",
        &["Qwen2ForCausalLM"],
    )
}

fn codex_embed_mistral_preset() -> ModelConfig {
    decoder_embedding_preset(
        4096,
        32,
        32,
        8,
        128,
        14336,
        32768,
        32000,
        10_000.0,
        1e-5,
        Some(4096),
        "mistral",
        &["MistralForCausalLM"],
    )
}

fn qwen2_instruct_preset() -> ModelConfig {
    decoder_generation_preset(
        4096,
        32,
        32,
        32,
        128,
        11008,
        32768,
        152064,
        1_000_000.0,
        1e-6,
        None,
        "qwen2",
        &["Qwen2ForCausalLM"],
    )
}

fn qwen3_preset(
    hidden_size: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    intermediate: usize,
    vocab: usize,
    max_pos: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
) -> ModelConfig {
    let head_dim = hidden_size / heads;
    decoder_generation_preset(
        hidden_size,
        layers,
        heads,
        kv_heads,
        head_dim,
        intermediate,
        max_pos,
        vocab,
        rope_theta,
        rms_norm_eps,
        None,
        "qwen3",
        &["Qwen3ForCausalLM"],
    )
}

fn qwen3_moe_preset(
    hidden_size: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize, // Explicit head_dim (Qwen3-MoE uses non-standard head_dim)
    intermediate: usize,
    vocab: usize,
    max_pos: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate_size: Option<usize>,
    rope_theta: f64,
    rms_norm_eps: f64,
) -> ModelConfig {
    let mut config = decoder_generation_preset(
        hidden_size,
        layers,
        heads,
        kv_heads,
        head_dim,
        intermediate,
        max_pos,
        vocab,
        rope_theta,
        rms_norm_eps,
        None,
        "qwen3_moe",
        &["Qwen3MoeForCausalLM"],
    );
    config.num_experts = Some(num_experts);
    config.num_experts_per_tok = Some(num_experts_per_tok);
    config.moe_intermediate_size = moe_intermediate_size;
    let mut extra = json!({
        "n_routed_experts": num_experts,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
    });
    if let Some(size) = moe_intermediate_size {
        extra["moe_intermediate_size"] = json!(size);
    }
    config.extra = extra;
    config
}

fn mixtral_preset(
    hidden_size: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    intermediate: usize,
    max_pos: usize,
    vocab: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
) -> ModelConfig {
    let head_dim = hidden_size / heads;
    let mut config = decoder_generation_preset(
        hidden_size,
        layers,
        heads,
        kv_heads,
        head_dim,
        intermediate,
        max_pos,
        vocab,
        rope_theta,
        rms_norm_eps,
        None,
        "mixtral",
        &["MixtralForCausalLM"],
    );
    config.num_experts = Some(num_experts);
    config.num_experts_per_tok = Some(num_experts_per_tok);
    config.extra = json!({
        "num_local_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
    });
    config
}

fn qwen25_instruct_preset(
    hidden_size: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    intermediate: usize,
    max_pos: usize,
    vocab: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
) -> ModelConfig {
    let head_dim = hidden_size / heads;
    decoder_generation_preset(
        hidden_size,
        layers,
        heads,
        kv_heads,
        head_dim,
        intermediate,
        max_pos,
        vocab,
        rope_theta,
        rms_norm_eps,
        None,
        "qwen2",
        &["Qwen2ForCausalLM"],
    )
}

/// GPT-OSS preset (OpenAI 2025 Open Source MoE)
///
/// Key characteristics:
/// - MoE with MXFP4 quantization on expert weights
/// - Non-standard head_dim (64*64=4096 != hidden_size=2880)
/// - YaRN rope scaling (4K→128K context)
/// - Alternating sliding/full attention pattern
fn gpt_oss_preset(
    hidden_size: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    intermediate: usize,
    vocab: usize,
    max_pos: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
) -> ModelConfig {
    let mut config = decoder_generation_preset(
        hidden_size,
        layers,
        heads,
        kv_heads,
        head_dim,
        intermediate,
        max_pos,
        vocab,
        rope_theta,
        rms_norm_eps,
        Some(128), // sliding_window for alternating attention
        "gpt_oss",
        &["GptOssForCausalLM"],
    );
    config.num_experts = Some(num_experts);
    config.num_experts_per_tok = Some(num_experts_per_tok);
    config.extra = json!({
        "num_local_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 32.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "original_max_position_embeddings": 4096
        },
        "quantization_config": {
            "quant_method": "mxfp4"
        }
    });
    config
}

fn phi4_preset(
    hidden_size: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    intermediate: usize,
    vocab: usize,
    max_pos: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
) -> ModelConfig {
    let head_dim = hidden_size / heads;
    decoder_generation_preset(
        hidden_size,
        layers,
        heads,
        kv_heads,
        head_dim,
        intermediate,
        max_pos,
        vocab,
        rope_theta,
        rms_norm_eps,
        None,
        "phi3",
        &["Phi3ForCausalLM"],
    )
}

fn smollm3_preset(
    hidden_size: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    intermediate: usize,
    vocab: usize,
    max_pos: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
) -> ModelConfig {
    let head_dim = hidden_size / heads;
    decoder_generation_preset(
        hidden_size,
        layers,
        heads,
        kv_heads,
        head_dim,
        intermediate,
        max_pos,
        vocab,
        rope_theta,
        rms_norm_eps,
        None,
        "smollm3",
        &["SmolLM3ForCausalLM"],
    )
}

fn internlm3_preset(
    hidden_size: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    intermediate: usize,
    vocab: usize,
    max_pos: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
) -> ModelConfig {
    let head_dim = hidden_size / heads;
    decoder_generation_preset(
        hidden_size,
        layers,
        heads,
        kv_heads,
        head_dim,
        intermediate,
        max_pos,
        vocab,
        rope_theta,
        rms_norm_eps,
        None,
        "internlm3",
        &["InternLM3ForCausalLM"],
    )
}

fn glm4_chat_preset(
    hidden_size: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    intermediate: usize,
    max_pos: usize,
    vocab: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
) -> ModelConfig {
    let head_dim = hidden_size / heads;
    decoder_generation_preset(
        hidden_size,
        layers,
        heads,
        kv_heads,
        head_dim,
        intermediate,
        max_pos,
        vocab,
        rope_theta,
        rms_norm_eps,
        None,
        "glm",
        &["GlmForCausalLM"],
    )
}

fn glm47_moe_preset(
    hidden_size: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize, // GLM-4.7 uses non-standard head_dim (96*128=12288 != 5120)
    intermediate: usize,
    max_pos: usize,
    vocab: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
) -> ModelConfig {
    let mut config = decoder_generation_preset(
        hidden_size,
        layers,
        heads,
        kv_heads,
        head_dim,
        intermediate,
        max_pos,
        vocab,
        rope_theta,
        rms_norm_eps,
        None,
        "glm4_moe",
        &["Glm4MoeForCausalLM"],
    );
    config.num_experts = Some(160);
    config.num_experts_per_tok = Some(8);
    config.n_shared_experts = Some(1);
    config.extra = json!({
        "n_routed_experts": 160,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1
    });
    config
}

fn deepseek_v3_preset() -> ModelConfig {
    // DeepSeek-V3 uses MLA (Multi-head Latent Attention) with:
    // - qk_rope_head_dim = 64
    // - qk_nope_head_dim = 128
    // - v_head_dim = 128 (used for output projection)
    let v_head_dim = 128;
    let mut config = decoder_generation_preset(
        7168,   // hidden_size
        61,     // layers
        128,    // heads
        128,    // kv_heads
        v_head_dim, // Use v_head_dim for attention output (MLA architecture)
        18432,  // intermediate
        163840, // max_pos
        129280, // vocab
        10000.0,
        1e-6,
        None,
        "deepseek_v3",
        &["DeepseekV3ForCausalLM"],
    );
    config.num_experts = Some(256);
    config.num_experts_per_tok = Some(8);
    config.n_shared_experts = Some(1);
    config.moe_intermediate_size = Some(2048);
    config.extra = json!({
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "moe_intermediate_size": 2048,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
    });
    config
}

fn decoder_embedding_preset(
    hidden_size: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    intermediate: usize,
    max_pos: usize,
    vocab: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
    sliding_window: Option<usize>,
    model_type: &str,
    architectures: &[&str],
) -> ModelConfig {
    ModelConfig {
        architectures: Some(architectures.iter().map(|s| s.to_string()).collect()),
        model_type: Some(model_type.to_string()),
        hidden_size,
        num_hidden_layers: layers,
        num_attention_heads: heads,
        num_key_value_heads: Some(kv_heads),
        head_dim: Some(head_dim),
        vocab_size: vocab,
        max_position_embeddings: max_pos,
        attention_probs_dropout_prob: Some(0.0),
        hidden_dropout_prob: Some(0.0),
        intermediate_size: Some(intermediate),
        num_experts: None,
        num_experts_per_tok: None,
        n_shared_experts: None,
        moe_intermediate_size: None,
        max_batch_size: None,
        memory_limit_mb: None,
        gpu_memory_fraction: None,
        hidden_act: Some("silu".to_string()),
        initializer_range: None,
        layer_norm_eps: None,
        rms_norm_eps: Some(rms_norm_eps),
        rope_theta: Some(rope_theta),
        rope_scaling: None,
        sliding_window,
        use_cache: Some(true),
        position_embedding_type: Some("rope".to_string()),
        pooler_hidden_act: None,
        pooler_dropout: None,
        pooling_type: Some("last_token".to_string()),
        num_labels: None,
        classifier_dropout: None,
        tie_word_embeddings: Some(true),
        is_decoder: Some(true),
        cross_attention_hidden_size: None,
        pad_token_id: Some(0),
        bos_token_id: None,
        eos_token_id: None,
        type_vocab_size: Some(1),
        extra: Value::Object(Map::new()),
        // Engram conditional memory (disabled by default)
        engram_enabled: None,
        engram_ngram_size: None,
        engram_num_buckets: None,
        engram_embedding_dim: None,
        engram_scale: None,
    }
}

fn decoder_generation_preset(
    hidden_size: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    intermediate: usize,
    max_pos: usize,
    vocab: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
    sliding_window: Option<usize>,
    model_type: &str,
    architectures: &[&str],
) -> ModelConfig {
    ModelConfig {
        architectures: Some(architectures.iter().map(|s| s.to_string()).collect()),
        model_type: Some(model_type.to_string()),
        hidden_size,
        num_hidden_layers: layers,
        num_attention_heads: heads,
        num_key_value_heads: Some(kv_heads),
        head_dim: Some(head_dim),
        vocab_size: vocab,
        max_position_embeddings: max_pos,
        attention_probs_dropout_prob: Some(0.0),
        hidden_dropout_prob: Some(0.0),
        intermediate_size: Some(intermediate),
        num_experts: None,
        num_experts_per_tok: None,
        n_shared_experts: None,
        moe_intermediate_size: None,
        max_batch_size: None,
        memory_limit_mb: None,
        gpu_memory_fraction: None,
        hidden_act: Some("silu".to_string()),
        initializer_range: None,
        layer_norm_eps: None,
        rms_norm_eps: Some(rms_norm_eps),
        rope_theta: Some(rope_theta),
        rope_scaling: None,
        sliding_window,
        use_cache: Some(true),
        position_embedding_type: Some("rope".to_string()),
        pooler_hidden_act: None,
        pooler_dropout: None,
        pooling_type: None,
        num_labels: None,
        classifier_dropout: None,
        tie_word_embeddings: Some(true),
        is_decoder: Some(true),
        cross_attention_hidden_size: None,
        pad_token_id: Some(0),
        bos_token_id: None,
        eos_token_id: None,
        type_vocab_size: Some(1),
        extra: Value::Object(Map::new()),
        // Engram conditional memory (disabled by default)
        engram_enabled: None,
        engram_ngram_size: None,
        engram_num_buckets: None,
        engram_embedding_dim: None,
        engram_scale: None,
    }
}

fn qwen3_reranker_preset(
    hidden_size: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    intermediate: usize,
    max_pos: usize,
    vocab: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
) -> ModelConfig {
    with_rerank(qwen3_embedding_preset(
        hidden_size,
        layers,
        heads,
        kv_heads,
        intermediate,
        max_pos,
        vocab,
        rope_theta,
        rms_norm_eps,
    ))
}

fn nvidia_nemotron_preset(
    hidden_size: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    intermediate: usize,
    max_pos: usize,
    vocab: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
) -> ModelConfig {
    ModelConfig {
        architectures: Some(vec!["LlamaBidirectionalModel".to_string()]),
        model_type: Some("llama_bidirec".to_string()),
        hidden_size,
        num_hidden_layers: layers,
        num_attention_heads: heads,
        num_key_value_heads: Some(kv_heads),
        head_dim: Some(128),
        vocab_size: vocab,
        max_position_embeddings: max_pos,
        attention_probs_dropout_prob: Some(0.0),
        hidden_dropout_prob: Some(0.0),
        intermediate_size: Some(intermediate),
        num_experts: None,
        num_experts_per_tok: None,
        n_shared_experts: None,
        moe_intermediate_size: None,
        max_batch_size: None,
        memory_limit_mb: None,
        gpu_memory_fraction: None,
        hidden_act: Some("silu".to_string()),
        initializer_range: None,
        layer_norm_eps: None,
        rms_norm_eps: Some(rms_norm_eps),
        rope_theta: Some(rope_theta),
        rope_scaling: Some(json!({
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        })),
        sliding_window: None,
        use_cache: Some(false),
        position_embedding_type: Some("rope".to_string()),
        pooler_hidden_act: None,
        pooler_dropout: None,
        pooling_type: None,
        num_labels: None,
        classifier_dropout: None,
        tie_word_embeddings: Some(false),
        is_decoder: Some(false),
        cross_attention_hidden_size: None,
        pad_token_id: Some(0),
        bos_token_id: None,
        eos_token_id: None,
        type_vocab_size: Some(1),
        extra: Value::Object(Map::new()),
        // Engram conditional memory (disabled by default)
        engram_enabled: None,
        engram_ngram_size: None,
        engram_num_buckets: None,
        engram_embedding_dim: None,
        engram_scale: None,
    }
}

fn jina_v4_preset() -> ModelConfig {
    ModelConfig {
        architectures: Some(vec!["JinaEmbeddingsV4Model".to_string()]),
        model_type: Some("qwen2_5_vl_text".to_string()),
        hidden_size: 2048,
        num_hidden_layers: 36,
        num_attention_heads: 16,
        num_key_value_heads: Some(2),
        head_dim: None,
        vocab_size: 151936,
        max_position_embeddings: 128000,
        attention_probs_dropout_prob: Some(0.0),
        hidden_dropout_prob: Some(0.0),
        intermediate_size: Some(11008),
        num_experts: None,
        num_experts_per_tok: None,
        n_shared_experts: None,
        moe_intermediate_size: None,
        max_batch_size: None,
        memory_limit_mb: None,
        gpu_memory_fraction: None,
        hidden_act: Some("silu".to_string()),
        initializer_range: None,
        layer_norm_eps: None,
        rms_norm_eps: Some(1e-6),
        rope_theta: Some(1_000_000.0),
        rope_scaling: Some(json!({
            "mrope_section": [16, 24, 24],
            "rope_type": "default",
            "type": "default",
        })),
        sliding_window: Some(32768),
        use_cache: Some(true),
        position_embedding_type: Some("rope".to_string()),
        pooler_hidden_act: None,
        pooler_dropout: None,
        pooling_type: None,
        num_labels: None,
        classifier_dropout: None,
        tie_word_embeddings: Some(true),
        is_decoder: Some(false),
        cross_attention_hidden_size: None,
        pad_token_id: Some(0),
        bos_token_id: None,
        eos_token_id: None,
        type_vocab_size: Some(1),
        extra: Value::Object(Map::new()),
        // Engram conditional memory (disabled by default)
        engram_enabled: None,
        engram_ngram_size: None,
        engram_num_buckets: None,
        engram_embedding_dim: None,
        engram_scale: None,
    }
}

fn jina_reranker_v3_preset() -> ModelConfig {
    with_rerank(ModelConfig {
        architectures: Some(vec!["JinaForRanking".to_string()]),
        model_type: Some("qwen3".to_string()),
        hidden_size: 1024,
        num_hidden_layers: 28,
        num_attention_heads: 16,
        num_key_value_heads: Some(8),
        head_dim: Some(128),
        vocab_size: 151936,
        max_position_embeddings: 131072,
        attention_probs_dropout_prob: Some(0.0),
        hidden_dropout_prob: Some(0.0),
        intermediate_size: Some(3072),
        num_experts: None,
        num_experts_per_tok: None,
        n_shared_experts: None,
        moe_intermediate_size: None,
        max_batch_size: None,
        memory_limit_mb: None,
        gpu_memory_fraction: None,
        hidden_act: Some("silu".to_string()),
        initializer_range: None,
        layer_norm_eps: None,
        rms_norm_eps: Some(1e-6),
        rope_theta: Some(1_000_000.0),
        rope_scaling: None,
        sliding_window: None,
        use_cache: Some(false),
        position_embedding_type: Some("rope".to_string()),
        pooler_hidden_act: None,
        pooler_dropout: None,
        pooling_type: None,
        num_labels: None,
        classifier_dropout: None,
        tie_word_embeddings: Some(true),
        is_decoder: Some(false),
        cross_attention_hidden_size: None,
        pad_token_id: Some(0),
        bos_token_id: None,
        eos_token_id: None,
        type_vocab_size: Some(1),
        extra: Value::Object(Map::new()),
        // Engram conditional memory (disabled by default)
        engram_enabled: None,
        engram_ngram_size: None,
        engram_num_buckets: None,
        engram_embedding_dim: None,
        engram_scale: None,
    })
}

fn preset(
    hidden_size: usize,
    layers: usize,
    heads: usize,
    intermediate: usize,
    max_pos: usize,
    vocab: usize,
    model_type: &str,
    architectures: &[&str],
    layer_norm_eps: f64,
    type_vocab_size: usize,
    pad_token_id: i64,
    attention_dropout: f32,
    hidden_dropout: f32,
) -> ModelConfig {
    ModelConfig {
        architectures: Some(architectures.iter().map(|s| s.to_string()).collect()),
        model_type: Some(model_type.to_string()),
        hidden_size,
        num_hidden_layers: layers,
        num_attention_heads: heads,
        num_key_value_heads: None,
        head_dim: None,
        vocab_size: vocab,
        max_position_embeddings: max_pos,
        attention_probs_dropout_prob: Some(attention_dropout),
        hidden_dropout_prob: Some(hidden_dropout),
        intermediate_size: Some(intermediate),
        num_experts: None,
        num_experts_per_tok: None,
        n_shared_experts: None,
        moe_intermediate_size: None,
        max_batch_size: None,
        memory_limit_mb: None,
        gpu_memory_fraction: None,
        hidden_act: Some("gelu".to_string()),
        initializer_range: Some(0.02),
        layer_norm_eps: Some(layer_norm_eps),
        rms_norm_eps: None,
        rope_theta: None,
        rope_scaling: None,
        sliding_window: None,
        use_cache: Some(true),
        position_embedding_type: Some("absolute".to_string()),
        pooler_hidden_act: None,
        pooler_dropout: None,
        pooling_type: None,
        num_labels: None,
        classifier_dropout: None,
        tie_word_embeddings: Some(true),
        is_decoder: Some(false),
        cross_attention_hidden_size: None,
        pad_token_id: Some(pad_token_id),
        bos_token_id: None,
        eos_token_id: None,
        type_vocab_size: Some(type_vocab_size),
        extra: Value::Object(Map::new()),
        // Engram conditional memory (disabled by default)
        engram_enabled: None,
        engram_ngram_size: None,
        engram_num_buckets: None,
        engram_embedding_dim: None,
        engram_scale: None,
    }
}

fn with_rerank(mut config: ModelConfig) -> ModelConfig {
    config.num_labels = Some(1);
    config.classifier_dropout = Some(config.classifier_dropout.unwrap_or(0.1));
    config
}

#[cfg(test)]
mod tests {
    use super::model_defaults;

    #[test]
    fn presets_cover_required_models() {
        let repos = [
            "baai/bge-small-zh-v1.5",
            "baai/bge-small-en-v1.5",
            "baai/bge-base-en-v1.5",
            "baai/bge-large-en-v1.5",
            "sentence-transformers/all-minilm-l6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-minilm-l6-v2",
            "sentence-transformers/multi-qa-mpnet-base-dot-v1",
            "sentence-transformers/all-minilm-l12-v2",
            "sentence-transformers/all-distilroberta-v1",
            "sentence-transformers/paraphrase-multilingual-minilm-l12-v2",
            "sentence-transformers/distiluse-base-multilingual-cased-v1",
            "intfloat/e5-large",
            "intfloat/e5-base",
            "intfloat/e5-small",
            "jinaai/jina-embeddings-v2-base-en",
            "jinaai/jina-embeddings-v2-small-en",
            "jinaai/jina-embeddings-v4",
            "moka-ai/m3e-base",
            "nvidia/llama-embed-nemotron-8b",
            "qwen/qwen3-embedding-0.6b",
            "qwen/qwen3-embedding-4b",
            "qwen/qwen3-embedding-8b",
            "salesforce/sfr-embedding-code-2b_r",
            "salesforce/sfr-embedding-code-7b_r",
            "qwen/qwen2.5-0.5b-instruct",
            "qwen/qwen2.5-1.5b-instruct",
            "qwen/qwen2.5-3b-instruct",
            "qwen/qwen2.5-7b-instruct",
            "qwen/qwen2.5-14b-instruct",
            "qwen/qwen2.5-32b-instruct",
            "qwen/qwen2.5-72b-instruct",
            "qwen/qwen3-0.6b",
            "qwen/qwen3-1.7b",
            "qwen/qwen3-4b",
            "qwen/qwen3-8b",
            "qwen/qwen3-14b",
            "qwen/qwen3-32b",
            "qwen/qwen3-30b-a3b",
            "qwen/qwen3-235b-a22b",
            "microsoft/phi-4",
            "microsoft/phi-4-mini-instruct",
            "huggingfacetb/smollm3-3b",
            "internlm/internlm3-8b-instruct",
            "mistralai/mixtral-8x7b-instruct-v0.1",
            "mistralai/mixtral-8x22b-instruct-v0.1",
            "thudm/glm-4-9b-chat-hf",
            "zai-org/glm-4.7",
            "deepseek-ai/deepseek-v3",
            "baai/bge-reranker-v2-m3",
            "baai/bge-reranker-large",
            "baai/bge-reranker-base",
            "cross-encoder/ms-marco-minilm-l-6-v2",
            "cross-encoder/ms-marco-minilm-l-12-v2",
            "cross-encoder/ms-marco-tinybert-l-2-v2",
            "cross-encoder/ms-marco-electra-base",
            "cross-encoder/quora-distilroberta-base",
            "qwen/qwen3-reranker-0.6b",
            "qwen/qwen3-reranker-4b",
            "qwen/qwen3-reranker-8b",
            "jinaai/jina-reranker-v3",
        ];

        for repo in repos {
            let cfg = model_defaults(repo);
            assert!(
                cfg.hidden_size > 0 && cfg.num_hidden_layers > 0 && cfg.num_attention_heads > 0,
                "preset should not contain zero values for {}",
                repo
            );
            assert!(
                cfg.type_vocab_size.unwrap_or(0) > 0,
                "type vocab missing for {repo}"
            );
            assert!(
                cfg.max_position_embeddings > 0,
                "max position missing for {repo}"
            );
        }
    }

    #[test]
    fn mpnet_defaults_match_roberta_variants() {
        let cfg = model_defaults("sentence-transformers/all-mpnet-base-v2");
        assert_eq!(cfg.model_type.as_deref(), Some("mpnet"));
        assert_eq!(cfg.layer_norm_eps.unwrap(), 1e-5);
        assert_eq!(cfg.pad_token_id, Some(1));
        assert_eq!(cfg.type_vocab_size, Some(1));
    }

    #[test]
    fn qwen3_embedding_presets_match_spec() {
        let cfg = model_defaults("qwen/qwen3-embedding-0.6b");
        assert_eq!(cfg.hidden_size, 896);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.num_attention_heads, 14);
        assert_eq!(cfg.num_key_value_heads, Some(2));
        assert_eq!(cfg.intermediate_size, Some(4864));
        assert_eq!(cfg.vocab_size, 151936);
        assert_eq!(cfg.rope_theta, Some(1_000_000.0));
        assert_eq!(cfg.rms_norm_eps, Some(1e-6));

        let cfg = model_defaults("qwen/qwen3-embedding-4b");
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.num_hidden_layers, 36);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, Some(4));
        assert_eq!(cfg.intermediate_size, Some(13824));

        let cfg = model_defaults("qwen/qwen3-embedding-8b");
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 36);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, Some(8));
        assert_eq!(cfg.intermediate_size, Some(14336));
    }

    #[test]
    fn moe_presets_match_spec() {
        let cfg = model_defaults("qwen/qwen3-30b-a3b");
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_hidden_layers, 48);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, Some(4));
        assert_eq!(cfg.intermediate_size, Some(6144));
        assert_eq!(cfg.vocab_size, 151936);
        assert_eq!(cfg.max_position_embeddings, 40960);
        assert_eq!(cfg.num_experts, Some(128));
        assert_eq!(cfg.num_experts_per_tok, Some(8));
        assert_eq!(cfg.rope_theta, Some(1_000_000.0));
        assert_eq!(cfg.rms_norm_eps, Some(1e-6));

        let cfg = model_defaults("qwen/qwen3-235b-a22b");
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 94);
        assert_eq!(cfg.num_attention_heads, 64);
        assert_eq!(cfg.num_key_value_heads, Some(4));
        assert_eq!(cfg.intermediate_size, Some(12288));
        assert_eq!(cfg.num_experts, Some(128));
        assert_eq!(cfg.num_experts_per_tok, Some(8));
        assert_eq!(cfg.moe_intermediate_size, Some(1536));

        let cfg = model_defaults("mistralai/mixtral-8x7b-instruct-v0.1");
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, Some(8));
        assert_eq!(cfg.intermediate_size, Some(14336));
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.max_position_embeddings, 32768);
        assert_eq!(cfg.num_experts, Some(8));
        assert_eq!(cfg.num_experts_per_tok, Some(2));
        assert_eq!(cfg.rope_theta, Some(1_000_000.0));
        assert_eq!(cfg.rms_norm_eps, Some(1e-5));

        let cfg = model_defaults("mistralai/mixtral-8x22b-instruct-v0.1");
        assert_eq!(cfg.hidden_size, 6144);
        assert_eq!(cfg.num_hidden_layers, 56);
        assert_eq!(cfg.num_attention_heads, 48);
        assert_eq!(cfg.num_key_value_heads, Some(8));
        assert_eq!(cfg.intermediate_size, Some(16384));
        assert_eq!(cfg.max_position_embeddings, 65536);
        assert_eq!(cfg.num_experts, Some(8));
        assert_eq!(cfg.num_experts_per_tok, Some(2));

        let cfg = model_defaults("deepseek-ai/deepseek-v3");
        assert_eq!(cfg.hidden_size, 7168);
        assert_eq!(cfg.num_hidden_layers, 61);
        assert_eq!(cfg.num_attention_heads, 128);
        assert_eq!(cfg.num_key_value_heads, Some(128));
        assert_eq!(cfg.head_dim, Some(128)); // v_head_dim for MLA architecture
        assert_eq!(cfg.intermediate_size, Some(18432));
        assert_eq!(cfg.vocab_size, 129280);
        assert_eq!(cfg.max_position_embeddings, 163840);
        assert_eq!(cfg.num_experts, Some(256));
        assert_eq!(cfg.num_experts_per_tok, Some(8));
        assert_eq!(cfg.n_shared_experts, Some(1));
        assert_eq!(cfg.moe_intermediate_size, Some(2048));
        assert_eq!(cfg.rope_theta, Some(10000.0));
        assert_eq!(cfg.rms_norm_eps, Some(1e-6));

        // GLM-4.7 MoE (non-standard head_dim: 96*128=12288 != hidden_size=5120)
        let cfg = model_defaults("zai-org/glm-4.7");
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.num_hidden_layers, 92);
        assert_eq!(cfg.num_attention_heads, 96);
        assert_eq!(cfg.num_key_value_heads, Some(8));
        assert_eq!(cfg.head_dim, Some(128)); // Explicit head_dim
        assert_eq!(cfg.intermediate_size, Some(12288));
        assert_eq!(cfg.vocab_size, 151552);
        assert_eq!(cfg.max_position_embeddings, 202752);
        assert_eq!(cfg.num_experts, Some(160));
        assert_eq!(cfg.num_experts_per_tok, Some(8));
        assert_eq!(cfg.n_shared_experts, Some(1));
        assert_eq!(cfg.rope_theta, Some(1_000_000.0));
        assert_eq!(cfg.rms_norm_eps, Some(1e-5));
    }

    #[test]
    fn codex_embed_presets_match_spec() {
        let cfg = model_defaults("salesforce/sfr-embedding-code-2b_r");
        assert_eq!(cfg.hidden_size, 1536);
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.num_attention_heads, 12);
        assert_eq!(cfg.num_key_value_heads, Some(2));
        assert_eq!(cfg.intermediate_size, Some(8960));
        assert_eq!(cfg.rope_theta, Some(1_000_000.0));
        assert_eq!(cfg.rms_norm_eps, Some(1e-6));
        assert_eq!(cfg.pooling_type.as_deref(), Some("last_token"));
        assert_eq!(cfg.is_decoder, Some(true));

        let cfg = model_defaults("salesforce/sfr-embedding-code-7b_r");
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, Some(8));
        assert_eq!(cfg.intermediate_size, Some(14336));
        assert_eq!(cfg.sliding_window, Some(4096));
        assert_eq!(cfg.rms_norm_eps, Some(1e-5));
        assert_eq!(cfg.pooling_type.as_deref(), Some("last_token"));
        assert_eq!(cfg.is_decoder, Some(true));
    }

    #[test]
    fn reranker_presets_set_num_labels() {
        let cfg = model_defaults("qwen/qwen3-reranker-0.6b");
        assert_eq!(cfg.num_labels, Some(1));
        let cfg = model_defaults("jinaai/jina-reranker-v3");
        assert_eq!(cfg.num_labels, Some(1));
    }

    #[test]
    fn nemotron_preset_includes_rope_scaling() {
        let cfg = model_defaults("nvidia/llama-embed-nemotron-8b");
        assert_eq!(cfg.num_key_value_heads, Some(8));
        assert_eq!(cfg.rope_theta, Some(500_000.0));
        assert_eq!(cfg.rms_norm_eps, Some(1e-5));
        assert!(cfg.rope_scaling.is_some());
    }

    #[test]
    fn jina_v4_preset_includes_sliding_window() {
        let cfg = model_defaults("jinaai/jina-embeddings-v4");
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_key_value_heads, Some(2));
        assert_eq!(cfg.sliding_window, Some(32768));
        assert!(cfg.rope_scaling.is_some());
    }
}
