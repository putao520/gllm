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
