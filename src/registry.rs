use crate::types::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model type supported by the library.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelType {
    /// Embedding encoder model.
    Embedding,
    /// Cross-encoder reranker model.
    Rerank,
    /// Decoder generator model.
    Generator,
}

/// Architecture of a model.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Architecture {
    /// BERT-like encoder.
    Bert,
    /// Cross-encoder architecture.
    CrossEncoder,
    /// Qwen2 embedding decoder.
    Qwen2Embedding,
    /// Mistral embedding decoder.
    MistralEmbedding,
    /// Qwen2 decoder for generation.
    Qwen2Generator,
    /// Mistral decoder for generation.
    MistralGenerator,
    /// Qwen3 embedding encoder.
    Qwen3Embedding,
    /// Qwen3 cross-encoder reranker.
    Qwen3Reranker,
    /// Qwen3 decoder for generation.
    Qwen3Generator,
    /// Phi-4 decoder for generation.
    Phi4Generator,
    /// SmolLM2 decoder for generation.
    SmolLM2Generator,
    /// SmolLM3 decoder for generation.
    SmolLM3Generator,
    /// InternLM3 decoder for generation.
    InternLM3Generator,
    /// Jina v4 embedding.
    JinaV4,
    /// Jina reranker v3.
    JinaRerankerV3,
    /// NVIDIA Llama-Embed-Nemotron.
    NVIDIANemotron,
    /// Google Gemma 3n.
    Gemma3n,
    /// Zhipu GLM-4.
    GLM4,
    /// Zhipu GLM-4 MoE.
    GLM4MoE,
    /// Qwen3 MoE decoder for generation.
    Qwen3MoE,
    /// Mixtral decoder for generation.
    Mixtral,
    /// DeepSeek-V3 decoder for generation.
    DeepSeekV3,
    /// GPT-OSS MoE decoder for generation (OpenAI 2025).
    GptOss,
}

/// Quantization type for models.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum Quantization {
    /// Full precision (fp16/bf16).
    #[default]
    None,
    /// 4-bit integer quantization.
    Int4,
    /// 8-bit integer quantization.
    Int8,
    /// AWQ quantization.
    AWQ,
    /// GPTQ quantization.
    GPTQ,
    /// GGUF format (for llama.cpp compatibility).
    GGUF,
    /// BNB (bitsandbytes) 4-bit.
    BNB4,
    /// BNB (bitsandbytes) 8-bit.
    BNB8,
    /// FP8 quantization.
    FP8,
}

impl Quantization {
    /// Parse quantization from string suffix.
    pub fn from_suffix(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "int4" | "4bit" => Some(Self::Int4),
            "int8" | "8bit" => Some(Self::Int8),
            "awq" => Some(Self::AWQ),
            "gptq" => Some(Self::GPTQ),
            "gguf" => Some(Self::GGUF),
            "bnb4" | "bnb-4bit" => Some(Self::BNB4),
            "bnb8" | "bnb-8bit" => Some(Self::BNB8),
            "fp8" => Some(Self::FP8),
            _ => None,
        }
    }

    /// Get the repo suffix for this quantization type.
    pub fn repo_suffix(&self) -> &'static str {
        match self {
            Self::None => "",
            Self::Int4 => "-Int4",
            Self::Int8 => "-Int8",
            Self::AWQ => "-AWQ",
            Self::GPTQ => "-GPTQ",
            Self::GGUF => "-GGUF",
            Self::BNB4 => "-bnb-4bit",
            Self::BNB8 => "-bnb-8bit",
            Self::FP8 => "-FP8",
        }
    }

    /// Check if this is a quantized format.
    pub fn is_quantized(&self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Metadata describing a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Alias for quick reference.
    pub alias: String,
    /// HuggingFace repository ID.
    pub repo_id: String,
    /// Model type.
    pub model_type: ModelType,
    /// Architecture descriptor.
    pub architecture: Architecture,
    /// Quantization type.
    #[serde(default)]
    pub quantization: Quantization,
}

/// Internal registry entry with base model information.
#[derive(Debug, Clone)]
struct RegistryEntry {
    /// Organization name (e.g., "Qwen", "BAAI").
    org: String,
    /// Base model name (e.g., "Qwen3-Embedding-0.6B").
    base_name: String,
    /// Model type.
    model_type: ModelType,
    /// Architecture.
    architecture: Architecture,
    /// Optional override for GGUF repo (for third-party GGUF providers like bartowski).
    gguf_override: Option<String>,
}

impl RegistryEntry {
    fn new(
        repo_id: &str,
        model_type: ModelType,
        architecture: Architecture,
    ) -> Self {
        let parts: Vec<&str> = repo_id.split('/').collect();
        let (org, base_name) = if parts.len() == 2 {
            (parts[0].to_string(), parts[1].to_string())
        } else {
            (String::new(), repo_id.to_string())
        };

        Self {
            org,
            base_name,
            model_type,
            architecture,
            gguf_override: None,
        }
    }

    fn with_gguf(mut self, gguf_repo: &str) -> Self {
        self.gguf_override = Some(gguf_repo.to_string());
        self
    }

    fn to_model_info(&self, alias: &str, quantization: Quantization) -> ModelInfo {
        let repo_id = match quantization {
            Quantization::GGUF => {
                // Use GGUF override if available, otherwise default pattern
                if let Some(ref gguf_repo) = self.gguf_override {
                    gguf_repo.clone()
                } else {
                    format!("{}/{}{}", self.org, self.base_name, quantization.repo_suffix())
                }
            }
            _ if quantization.is_quantized() => {
                // For other quantization types, use standard pattern
                format!("{}/{}{}", self.org, self.base_name, quantization.repo_suffix())
            }
            _ => {
                // Default (no quantization)
                format!("{}/{}", self.org, self.base_name)
            }
        };

        ModelInfo {
            alias: alias.to_string(),
            repo_id,
            model_type: self.model_type,
            architecture: self.architecture,
            quantization,
        }
    }
}

/// Registry of built-in model aliases.
pub struct ModelRegistry {
    entries: HashMap<String, RegistryEntry>,
}

impl ModelRegistry {
    /// Build a registry with built-in aliases.
    pub fn new() -> Self {
        let mut entries = HashMap::new();

        // (alias, repo_id, model_type, architecture)
        // All models support quantization suffixes - repo path: {org}/{model}{suffix}
        let model_entries = [
            // BGE Embedding Models
            ("bge-small-zh", "BAAI/bge-small-zh-v1.5", ModelType::Embedding, Architecture::Bert),
            ("bge-small-en", "BAAI/bge-small-en-v1.5", ModelType::Embedding, Architecture::Bert),
            ("bge-base-en", "BAAI/bge-base-en-v1.5", ModelType::Embedding, Architecture::Bert),
            ("bge-large-en", "BAAI/bge-large-en-v1.5", ModelType::Embedding, Architecture::Bert),

            // Sentence Transformers Models
            ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", ModelType::Embedding, Architecture::Bert),
            ("all-mpnet-base-v2", "sentence-transformers/all-mpnet-base-v2", ModelType::Embedding, Architecture::Bert),
            ("paraphrase-MiniLM-L6-v2", "sentence-transformers/paraphrase-MiniLM-L6-v2", ModelType::Embedding, Architecture::Bert),
            ("multi-qa-mpnet-base-dot-v1", "sentence-transformers/multi-qa-mpnet-base-dot-v1", ModelType::Embedding, Architecture::Bert),

            // E5 Models
            ("e5-large", "intfloat/e5-large", ModelType::Embedding, Architecture::Bert),
            ("e5-base", "intfloat/e5-base", ModelType::Embedding, Architecture::Bert),
            ("e5-small", "intfloat/e5-small", ModelType::Embedding, Architecture::Bert),

            // JINA Embeddings
            ("jina-embeddings-v2-base-en", "jinaai/jina-embeddings-v2-base-en", ModelType::Embedding, Architecture::Bert),
            ("jina-embeddings-v2-small-en", "jinaai/jina-embeddings-v2-small-en", ModelType::Embedding, Architecture::Bert),
            ("jina-embeddings-v4", "jinaai/jina-embeddings-v4", ModelType::Embedding, Architecture::JinaV4),

            // Chinese Models
            ("m3e-base", "moka-ai/m3e-base", ModelType::Embedding, Architecture::Bert),

            // Multilingual Models
            ("multilingual-MiniLM-L12-v2", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", ModelType::Embedding, Architecture::Bert),
            ("distiluse-base-multilingual-cased-v1", "sentence-transformers/distiluse-base-multilingual-cased-v1", ModelType::Embedding, Architecture::Bert),

            // Code Models (Legacy - BERT-based)
            ("codebert-base", "claudios/codebert-base", ModelType::Embedding, Architecture::Bert),
            ("starencoder", "bigcode/starencoder", ModelType::Embedding, Architecture::Bert),
            ("graphcodebert-base", "claudios/graphcodebert-base", ModelType::Embedding, Architecture::Bert),
            ("unixcoder-base", "claudios/unixcoder-base", ModelType::Embedding, Architecture::Bert),

            // Code Models (2024 SOTA - CodeXEmbed / SFR-Embedding-Code)
            // CoIR benchmark SOTA: outperforms Voyage-Code by 20%+
            ("codexembed-400m", "Salesforce/SFR-Embedding-Code-400M_R", ModelType::Embedding, Architecture::Bert),
            ("sfr-embedding-code-400m", "Salesforce/SFR-Embedding-Code-400M_R", ModelType::Embedding, Architecture::Bert),
            ("codexembed-2b", "Salesforce/SFR-Embedding-Code-2B_R", ModelType::Embedding, Architecture::Qwen2Embedding),
            ("codexembed-7b", "Salesforce/SFR-Embedding-Code-7B_R", ModelType::Embedding, Architecture::MistralEmbedding),
            ("sfr-embedding-code-2b", "Salesforce/SFR-Embedding-Code-2B_R", ModelType::Embedding, Architecture::Qwen2Embedding),
            ("sfr-embedding-code-7b", "Salesforce/SFR-Embedding-Code-7B_R", ModelType::Embedding, Architecture::MistralEmbedding),

            // Qwen3/Qwen3-next Generator Models (2025 - all support GGUF via {org}/{model}-GGUF repos)
            // Qwen2.5 removed: 2024 release, use Qwen3/Qwen3-next instead
            ("qwen3-0.6b", "Qwen/Qwen3-0.6B", ModelType::Generator, Architecture::Qwen3Generator),
            ("qwen3-1.7b", "Qwen/Qwen3-1.7B", ModelType::Generator, Architecture::Qwen3Generator),
            ("qwen3-4b", "Qwen/Qwen3-4B", ModelType::Generator, Architecture::Qwen3Generator),
            ("qwen3-8b", "Qwen/Qwen3-8B", ModelType::Generator, Architecture::Qwen3Generator),
            ("qwen3-14b", "Qwen/Qwen3-14B", ModelType::Generator, Architecture::Qwen3Generator),
            ("qwen3-32b", "Qwen/Qwen3-32B", ModelType::Generator, Architecture::Qwen3Generator),
            ("qwen3-30b-a3b", "Qwen/Qwen3-30B-A3B", ModelType::Generator, Architecture::Qwen3MoE),
            ("qwen3-235b-a22b", "Qwen/Qwen3-235B-A22B", ModelType::Generator, Architecture::Qwen3MoE),

            // Qwen3-next Models (2025 - faster inference, better quality)
            ("qwen3-next-0.6b", "Qwen/Qwen3-next-0.6B", ModelType::Generator, Architecture::Qwen3Generator),
            ("qwen3-next-2b", "Qwen/Qwen3-next-2B", ModelType::Generator, Architecture::Qwen3Generator),
            ("qwen3-next-4b", "Qwen/Qwen3-next-4B", ModelType::Generator, Architecture::Qwen3Generator),
            ("qwen3-next-8b", "Qwen/Qwen3-next-8B", ModelType::Generator, Architecture::Qwen3Generator),
            ("qwen3-next-32b", "Qwen/Qwen3-next-32B", ModelType::Generator, Architecture::Qwen3Generator),

            // Ministral Models (2024 - small efficient models, perfect for FP16 testing)
            ("ministral-3b-instruct", "mistralai/Ministral-3B-Instruct-2410", ModelType::Generator, Architecture::MistralGenerator),
            ("ministral-8b-instruct", "mistralai/Ministral-8B-Instruct-2410", ModelType::Generator, Architecture::MistralGenerator),
            // Mistral/Mixtral Models
            ("mistral-7b-instruct", "mistralai/Mistral-7B-Instruct-v0.3", ModelType::Generator, Architecture::MistralGenerator),
            ("mixtral-8x7b-instruct", "mistralai/Mixtral-8x7B-Instruct-v0.1", ModelType::Generator, Architecture::Mixtral),
            ("mixtral-8x22b-instruct", "mistralai/Mixtral-8x22B-Instruct-v0.1", ModelType::Generator, Architecture::Mixtral),

            // GLM/DeepSeek Models
            ("glm-4-9b-chat", "THUDM/glm-4-9b-chat-hf", ModelType::Generator, Architecture::GLM4),
            ("glm-4.7", "zai-org/GLM-4.7", ModelType::Generator, Architecture::GLM4MoE),
            ("deepseek-v3", "deepseek-ai/DeepSeek-V3", ModelType::Generator, Architecture::DeepSeekV3),

            // GPT-OSS Models (OpenAI 2025 Open Source MoE)
            ("gpt-oss-20b", "openai/gpt-oss-20b", ModelType::Generator, Architecture::GptOss),
            ("gpt-oss-120b", "openai/gpt-oss-120b", ModelType::Generator, Architecture::GptOss),

            // Phi-4 Models (Microsoft 2024)
            ("phi-4", "microsoft/phi-4", ModelType::Generator, Architecture::Phi4Generator),
            ("phi-4-mini-instruct", "microsoft/phi-4-mini-instruct", ModelType::Generator, Architecture::Phi4Generator),
            ("smollm3-3b", "HuggingFaceTB/SmolLM3-3B", ModelType::Generator, Architecture::SmolLM3Generator),
            ("smollm2-135m-instruct", "HuggingFaceTB/SmolLM2-135M-Instruct", ModelType::Generator, Architecture::SmolLM2Generator),
            ("internlm3-8b-instruct", "internlm/internlm3-8b-instruct", ModelType::Generator, Architecture::InternLM3Generator),

            // Light Models for Edge Devices
            ("all-MiniLM-L12-v2", "sentence-transformers/all-MiniLM-L12-v2", ModelType::Embedding, Architecture::Bert),
            ("all-distilroberta-v1", "sentence-transformers/all-distilroberta-v1", ModelType::Embedding, Architecture::Bert),

            // Qwen3 Embedding Models
            ("qwen3-embedding-0.6b", "Qwen/Qwen3-Embedding-0.6B", ModelType::Embedding, Architecture::Qwen3Embedding),
            ("qwen3-embedding-4b", "Qwen/Qwen3-Embedding-4B", ModelType::Embedding, Architecture::Qwen3Embedding),
            ("qwen3-embedding-8b", "Qwen/Qwen3-Embedding-8B", ModelType::Embedding, Architecture::Qwen3Embedding),

            // NVIDIA Embedding
            ("llama-embed-nemotron-8b", "nvidia/llama-embed-nemotron-8b", ModelType::Embedding, Architecture::NVIDIANemotron),

            // BGE Rerankers
            ("bge-reranker-v2", "BAAI/bge-reranker-v2-m3", ModelType::Rerank, Architecture::CrossEncoder),
            ("bge-reranker-large", "BAAI/bge-reranker-large", ModelType::Rerank, Architecture::CrossEncoder),
            ("bge-reranker-base", "BAAI/bge-reranker-base", ModelType::Rerank, Architecture::CrossEncoder),

            // MS MARCO Rerankers
            ("ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L-6-v2", ModelType::Rerank, Architecture::CrossEncoder),
            ("ms-marco-MiniLM-L-12-v2", "cross-encoder/ms-marco-MiniLM-L-12-v2", ModelType::Rerank, Architecture::CrossEncoder),
            ("ms-marco-TinyBERT-L-2-v2", "cross-encoder/ms-marco-TinyBERT-L-2-v2", ModelType::Rerank, Architecture::CrossEncoder),
            ("ms-marco-electra-base", "cross-encoder/ms-marco-electra-base", ModelType::Rerank, Architecture::CrossEncoder),

            // Specialized Rerankers
            ("quora-distilroberta-base", "cross-encoder/quora-distilroberta-base", ModelType::Rerank, Architecture::CrossEncoder),

            // Qwen3 Reranker Models
            ("qwen3-reranker-0.6b", "Qwen/Qwen3-Reranker-0.6B", ModelType::Rerank, Architecture::Qwen3Reranker),
            ("qwen3-reranker-4b", "Qwen/Qwen3-Reranker-4B", ModelType::Rerank, Architecture::Qwen3Reranker),
            ("qwen3-reranker-8b", "Qwen/Qwen3-Reranker-8B", ModelType::Rerank, Architecture::Qwen3Reranker),

            // Jina Reranker V3
            ("jina-reranker-v3", "jinaai/jina-reranker-v3", ModelType::Rerank, Architecture::JinaRerankerV3),
        ];

        for (alias, repo_id, model_type, architecture) in model_entries {
            let alias_key = alias.to_ascii_lowercase();
            entries.insert(
                alias_key,
                RegistryEntry::new(repo_id, model_type, architecture),
            );
        }

        // Models with third-party GGUF providers (bartowski, etc.)
        // These need explicit GGUF repo overrides
        let gguf_overrides: &[(&str, &str)] = &[
            ("ministral-3b-instruct", "bartowski/Ministral-3B-Instruct-2410-GGUF"),
            ("ministral-8b-instruct", "bartowski/Ministral-8B-Instruct-2410-GGUF"),
            ("mistral-7b-instruct", "bartowski/Mistral-7B-Instruct-v0.3-GGUF"),
            ("glm-4-9b-chat", "bartowski/glm-4-9b-chat-GGUF"),
            ("phi-4-mini-instruct", "bartowski/phi-4-mini-instruct-GGUF"),
            ("smollm3-3b", "bartowski/SmolLM3-3B-GGUF"),
            ("internlm3-8b-instruct", "bartowski/internlm3-8b-instruct-GGUF"),
            ("mixtral-8x7b-instruct", "bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF"),
            ("mixtral-8x22b-instruct", "bartowski/Mixtral-8x22B-Instruct-v0.1-GGUF"),
        ];

        for (alias, gguf_repo) in gguf_overrides {
            let alias_key = alias.to_ascii_lowercase();
            if let Some(entry) = entries.get_mut(&alias_key) {
                entry.gguf_override = Some(gguf_repo.to_string());
            }
        }

        Self { entries }
    }

    /// Resolve an alias or repo ID into a model info record.
    ///
    /// Supports quantization suffix:
    /// - `qwen3-embedding-0.6b` - default (fp16/bf16)
    /// - `qwen3-embedding-0.6b:int4` - Int4 quantization
    /// - `qwen3-embedding-0.6b:gguf` - GGUF format
    /// - `qwen3-embedding-0.6b:awq` - AWQ quantization
    /// - `Qwen/Qwen3-Embedding-0.6B-Int4` - direct repo ID
    ///
    /// All models support quantization suffixes. The repo path is constructed as:
    /// {org}/{base_name}{repo_suffix} -> e.g., Qwen/Qwen3-0.6B-GGUF
    pub fn resolve(&self, name: &str) -> Result<ModelInfo> {
        let name = name.trim();

        // Check for quantization suffix (alias:quant format)
        if let Some((base, quant_str)) = name.rsplit_once(':') {
            // Try to parse quantization
            if let Some(quantization) = Quantization::from_suffix(quant_str) {
                let base_key = base.to_ascii_lowercase();
                if let Some(entry) = self.entries.get(&base_key) {
                    // All models support quantization - repo path: {org}/{model}{suffix}
                    return Ok(entry.to_model_info(&format!("{}:{}", base, quant_str), quantization));
                }
            }
            // Not a valid quantization suffix, fall through to other parsing
        }

        // Try direct alias lookup
        let key = name.to_ascii_lowercase();
        if let Some(entry) = self.entries.get(&key) {
            return Ok(entry.to_model_info(name, Quantization::None));
        }

        // Allow direct HF repo IDs without registering
        if name.contains('/') {
            return Ok(self.infer_from_repo(name));
        }

        Err(Error::ModelNotFound(name.to_string()))
    }

    /// List all registered model aliases.
    pub fn list_aliases(&self) -> Vec<&str> {
        self.entries.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a model supports quantization variants.
    /// All registered models now support quantization suffixes.
    pub fn supports_quantization(&self, alias: &str) -> bool {
        let key = alias.to_ascii_lowercase();
        self.entries.contains_key(&key)
    }

    /// Get available quantization variants for a model.
    /// All models support the full set of quantization formats.
    pub fn available_quantizations(&self, _alias: &str) -> Vec<Quantization> {
        vec![
            Quantization::None,
            Quantization::Int4,
            Quantization::Int8,
            Quantization::AWQ,
            Quantization::GPTQ,
            Quantization::GGUF,
            Quantization::BNB4,
            Quantization::BNB8,
            Quantization::FP8,
        ]
    }

    fn infer_from_repo(&self, repo_id: &str) -> ModelInfo {
        let lower = repo_id.to_ascii_lowercase();

        // Detect quantization from repo name
        let quantization = if lower.contains("-int4") || lower.contains("-4bit") {
            Quantization::Int4
        } else if lower.contains("-int8") || lower.contains("-8bit") {
            Quantization::Int8
        } else if lower.contains("-awq") {
            Quantization::AWQ
        } else if lower.contains("-gptq") {
            Quantization::GPTQ
        } else if lower.contains("-gguf") {
            Quantization::GGUF
        } else if lower.contains("-fp8") {
            Quantization::FP8
        } else {
            Quantization::None
        };

        let is_generator = lower.contains("generator")
            || lower.contains("instruct")
            || lower.contains("chat")
            || lower.contains("glm-4.7")
            || lower.contains("glm4_moe")
            || lower.contains("glm4-moe")
            || lower.contains("mixtral")
            || lower.contains("deepseek")
            || lower.contains("phi-4")
            || lower.contains("phi4")
            || lower.contains("smollm3")
            || lower.contains("internlm3")
            || ((lower.contains("qwen3") || lower.contains("qwen-3"))
                && !lower.contains("embedding")
                && !lower.contains("reranker"));

        let model_type = if lower.contains("reranker") {
            ModelType::Rerank
        } else if is_generator {
            ModelType::Generator
        } else {
            ModelType::Embedding
        };

        let architecture = if lower.contains("sfr-embedding-code-2b")
            || lower.contains("codexembed-2b")
        {
            Architecture::Qwen2Embedding
        } else if lower.contains("sfr-embedding-code-7b") || lower.contains("codexembed-7b") {
            Architecture::MistralEmbedding
        } else if lower.contains("qwen2.5")
            || lower.contains("qwen-2.5")
            || lower.contains("qwen2_5")
        {
            match model_type {
                ModelType::Generator => Architecture::Qwen2Generator,
                _ => Architecture::Qwen2Embedding,
            }
        } else if lower.contains("qwen2") || lower.contains("qwen-2") {
            match model_type {
                ModelType::Generator => Architecture::Qwen2Generator,
                _ => Architecture::Qwen2Embedding,
            }
        } else if lower.contains("mistral") {
            match model_type {
                ModelType::Generator => Architecture::MistralGenerator,
                _ => Architecture::MistralEmbedding,
            }
        } else if lower.contains("mixtral") {
            Architecture::Mixtral
        } else if lower.contains("deepseek") && lower.contains("v3") {
            Architecture::DeepSeekV3
        } else if lower.contains("phi-4") || lower.contains("phi4") {
            Architecture::Phi4Generator
        } else if lower.contains("smollm3") {
            Architecture::SmolLM3Generator
        } else if lower.contains("smollm2") || lower.contains("smollm-2") {
            Architecture::SmolLM2Generator
        } else if lower.contains("internlm3") {
            Architecture::InternLM3Generator
        } else if lower.contains("qwen3") || lower.contains("qwen-3") {
            match model_type {
                ModelType::Embedding => Architecture::Qwen3Embedding,
                ModelType::Rerank => Architecture::Qwen3Reranker,
                ModelType::Generator => {
                    if lower.contains("moe") || (lower.contains("-a") && lower.contains("b")) {
                        Architecture::Qwen3MoE
                    } else {
                        Architecture::Qwen3Generator
                    }
                }
            }
        } else if lower.contains("jina") {
            if lower.contains("reranker") {
                Architecture::JinaRerankerV3
            } else {
                Architecture::JinaV4
            }
        } else if lower.contains("nemotron") {
            Architecture::NVIDIANemotron
        } else if lower.contains("glm-4.7")
            || lower.contains("glm4_moe")
            || lower.contains("glm4-moe")
        {
            Architecture::GLM4MoE
        } else if lower.contains("glm") {
            Architecture::GLM4
        } else if lower.contains("gemma") {
            Architecture::Gemma3n
        } else {
            match model_type {
                ModelType::Embedding => Architecture::Bert,
                ModelType::Rerank => Architecture::CrossEncoder,
                ModelType::Generator => Architecture::Qwen3Generator,
            }
        };

        let alias = repo_id
            .rsplit('/')
            .next()
            .unwrap_or(repo_id)
            .to_ascii_lowercase();

        ModelInfo {
            alias,
            repo_id: repo_id.to_string(),
            model_type,
            architecture,
            quantization,
        }
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_builtin_model() {
        let registry = ModelRegistry::new();
        let info = registry.resolve("bge-small-en").unwrap();
        assert_eq!(info.repo_id, "BAAI/bge-small-en-v1.5");
        assert!(matches!(info.architecture, Architecture::Bert));
        assert!(matches!(info.quantization, Quantization::None));
    }

    #[test]
    fn resolves_repo_id_directly() {
        let registry = ModelRegistry::new();
        let repo = "BAAI/custom-model";
        let info = registry.resolve(repo).unwrap();
        assert_eq!(info.repo_id, repo);
        assert_eq!(info.alias, "custom-model");
    }

    #[test]
    fn resolves_quantized_model() {
        let registry = ModelRegistry::new();

        // Int4 quantization
        let info = registry.resolve("qwen3-embedding-0.6b:int4").unwrap();
        assert_eq!(info.repo_id, "Qwen/Qwen3-Embedding-0.6B-Int4");
        assert!(matches!(info.quantization, Quantization::Int4));

        // AWQ quantization
        let info = registry.resolve("qwen3-embedding-8b:awq").unwrap();
        assert_eq!(info.repo_id, "Qwen/Qwen3-Embedding-8B-AWQ");
        assert!(matches!(info.quantization, Quantization::AWQ));

        // GPTQ quantization
        let info = registry.resolve("qwen3-reranker-4b:gptq").unwrap();
        assert_eq!(info.repo_id, "Qwen/Qwen3-Reranker-4B-GPTQ");
        assert!(matches!(info.quantization, Quantization::GPTQ));
    }

    #[test]
    fn all_models_support_quantization() {
        let registry = ModelRegistry::new();

        // All models now support quantization - repo path includes suffix
        let info = registry.resolve("bge-small-en:int4").unwrap();
        assert_eq!(info.repo_id, "BAAI/bge-small-en-v1.5-Int4");
        assert!(matches!(info.quantization, Quantization::Int4));

        // GGUF quantization
        let info = registry.resolve("bge-small-en:gguf").unwrap();
        assert_eq!(info.repo_id, "BAAI/bge-small-en-v1.5-GGUF");
        assert!(matches!(info.quantization, Quantization::GGUF));
    }

    #[test]
    fn infers_quantization_from_repo_id() {
        let registry = ModelRegistry::new();

        let info = registry.resolve("Qwen/Qwen3-Embedding-8B-Int4").unwrap();
        assert!(matches!(info.quantization, Quantization::Int4));

        let info = registry.resolve("some-org/model-name-AWQ").unwrap();
        assert!(matches!(info.quantization, Quantization::AWQ));
    }

    #[test]
    fn supports_quantization_check() {
        let registry = ModelRegistry::new();

        // All registered models now support quantization
        assert!(registry.supports_quantization("qwen3-embedding-0.6b"));
        assert!(registry.supports_quantization("qwen3-reranker-8b"));
        assert!(registry.supports_quantization("jina-embeddings-v4"));
        assert!(registry.supports_quantization("bge-small-en"));
        assert!(registry.supports_quantization("all-MiniLM-L6-v2"));

        // Unregistered model returns false
        assert!(!registry.supports_quantization("non-existent-model"));
    }

    #[test]
    fn resolves_expanded_models() {
        let registry = ModelRegistry::new();
        let cases = [
            ("qwen3-embedding-0.6b", "Qwen/Qwen3-Embedding-0.6B", ModelType::Embedding, Architecture::Qwen3Embedding),
            ("qwen3-embedding-4b", "Qwen/Qwen3-Embedding-4B", ModelType::Embedding, Architecture::Qwen3Embedding),
            ("qwen3-embedding-8b", "Qwen/Qwen3-Embedding-8B", ModelType::Embedding, Architecture::Qwen3Embedding),
            ("codexembed-2b", "Salesforce/SFR-Embedding-Code-2B_R", ModelType::Embedding, Architecture::Qwen2Embedding),
            ("codexembed-7b", "Salesforce/SFR-Embedding-Code-7B_R", ModelType::Embedding, Architecture::MistralEmbedding),
            ("sfr-embedding-code-2b", "Salesforce/SFR-Embedding-Code-2B_R", ModelType::Embedding, Architecture::Qwen2Embedding),
            ("sfr-embedding-code-7b", "Salesforce/SFR-Embedding-Code-7B_R", ModelType::Embedding, Architecture::MistralEmbedding),
            ("qwen3-next-0.6b", "Qwen/Qwen3-next-0.6B", ModelType::Generator, Architecture::Qwen3Generator),
            ("qwen3-next-2b", "Qwen/Qwen3-next-2B", ModelType::Generator, Architecture::Qwen3Generator),
            ("qwen3-0.6b", "Qwen/Qwen3-0.6B", ModelType::Generator, Architecture::Qwen3Generator),
            ("qwen3-1.7b", "Qwen/Qwen3-1.7B", ModelType::Generator, Architecture::Qwen3Generator),
            ("qwen3-4b", "Qwen/Qwen3-4B", ModelType::Generator, Architecture::Qwen3Generator),
            ("qwen3-8b", "Qwen/Qwen3-8B", ModelType::Generator, Architecture::Qwen3Generator),
            ("qwen3-14b", "Qwen/Qwen3-14B", ModelType::Generator, Architecture::Qwen3Generator),
            ("qwen3-32b", "Qwen/Qwen3-32B", ModelType::Generator, Architecture::Qwen3Generator),
            ("qwen3-30b-a3b", "Qwen/Qwen3-30B-A3B", ModelType::Generator, Architecture::Qwen3MoE),
            ("qwen3-235b-a22b", "Qwen/Qwen3-235B-A22B", ModelType::Generator, Architecture::Qwen3MoE),
            ("ministral-3b-instruct", "mistralai/Ministral-3B-Instruct-2410", ModelType::Generator, Architecture::MistralGenerator),
            ("ministral-8b-instruct", "mistralai/Ministral-8B-Instruct-2410", ModelType::Generator, Architecture::MistralGenerator),
            ("mistral-7b-instruct", "mistralai/Mistral-7B-Instruct-v0.3", ModelType::Generator, Architecture::MistralGenerator),
            ("mixtral-8x7b-instruct", "mistralai/Mixtral-8x7B-Instruct-v0.1", ModelType::Generator, Architecture::Mixtral),
            ("mixtral-8x22b-instruct", "mistralai/Mixtral-8x22B-Instruct-v0.1", ModelType::Generator, Architecture::Mixtral),
            ("glm-4-9b-chat", "THUDM/glm-4-9b-chat-hf", ModelType::Generator, Architecture::GLM4),
            ("glm-4.7", "zai-org/GLM-4.7", ModelType::Generator, Architecture::GLM4MoE),
            ("deepseek-v3", "deepseek-ai/DeepSeek-V3", ModelType::Generator, Architecture::DeepSeekV3),
            ("phi-4", "microsoft/phi-4", ModelType::Generator, Architecture::Phi4Generator),
            ("phi-4-mini-instruct", "microsoft/phi-4-mini-instruct", ModelType::Generator, Architecture::Phi4Generator),
            ("smollm3-3b", "HuggingFaceTB/SmolLM3-3B", ModelType::Generator, Architecture::SmolLM3Generator),
            ("internlm3-8b-instruct", "internlm/internlm3-8b-instruct", ModelType::Generator, Architecture::InternLM3Generator),
            ("qwen3-reranker-0.6b", "Qwen/Qwen3-Reranker-0.6B", ModelType::Rerank, Architecture::Qwen3Reranker),
            ("qwen3-reranker-4b", "Qwen/Qwen3-Reranker-4B", ModelType::Rerank, Architecture::Qwen3Reranker),
            ("qwen3-reranker-8b", "Qwen/Qwen3-Reranker-8B", ModelType::Rerank, Architecture::Qwen3Reranker),
            ("llama-embed-nemotron-8b", "nvidia/llama-embed-nemotron-8b", ModelType::Embedding, Architecture::NVIDIANemotron),
            ("jina-embeddings-v4", "jinaai/jina-embeddings-v4", ModelType::Embedding, Architecture::JinaV4),
            ("jina-reranker-v3", "jinaai/jina-reranker-v3", ModelType::Rerank, Architecture::JinaRerankerV3),
        ];

        for (alias, repo_id, model_type, architecture) in cases {
            let info = registry.resolve(alias).expect("resolve model");
            assert_eq!(info.repo_id, repo_id);
            assert_eq!(info.model_type, model_type);
            assert_eq!(info.architecture, architecture);
        }
    }

    #[test]
    fn list_available_quantizations() {
        let registry = ModelRegistry::new();

        // All models return full quantization list
        let quants = registry.available_quantizations("qwen3-embedding-8b");
        assert!(quants.len() > 1);
        assert!(quants.contains(&Quantization::Int4));
        assert!(quants.contains(&Quantization::GGUF));

        // Even BERT models now return full list
        let quants = registry.available_quantizations("bge-small-en");
        assert!(quants.len() > 1);
        assert!(quants.contains(&Quantization::None));
        assert!(quants.contains(&Quantization::GGUF));
    }

    #[test]
    fn gguf_override_resolves_to_third_party_repo() {
        let registry = ModelRegistry::new();

        // Models with official GGUF repos use default pattern
        let info = registry.resolve("qwen3-next-0.6b:gguf").unwrap();
        assert_eq!(info.repo_id, "Qwen/Qwen3-next-0.6B-GGUF");

        let info = registry.resolve("qwen3-0.6b:gguf").unwrap();
        assert_eq!(info.repo_id, "Qwen/Qwen3-0.6B-GGUF");

        // Models with GGUF override use third-party repos (bartowski)
        let info = registry.resolve("glm-4-9b-chat:gguf").unwrap();
        assert_eq!(info.repo_id, "bartowski/glm-4-9b-chat-GGUF");

        let info = registry.resolve("mistral-7b-instruct:gguf").unwrap();
        assert_eq!(info.repo_id, "bartowski/Mistral-7B-Instruct-v0.3-GGUF");

        let info = registry.resolve("phi-4-mini-instruct:gguf").unwrap();
        assert_eq!(info.repo_id, "bartowski/phi-4-mini-instruct-GGUF");

        let info = registry.resolve("smollm3-3b:gguf").unwrap();
        assert_eq!(info.repo_id, "bartowski/SmolLM3-3B-GGUF");

        let info = registry.resolve("internlm3-8b-instruct:gguf").unwrap();
        assert_eq!(info.repo_id, "bartowski/internlm3-8b-instruct-GGUF");

        let info = registry.resolve("mixtral-8x7b-instruct:gguf").unwrap();
        assert_eq!(info.repo_id, "bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF");
    }
}
