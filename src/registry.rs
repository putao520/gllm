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
    /// Whether this model supports quantization variants.
    supports_quantization: bool,
}

impl RegistryEntry {
    fn new(
        repo_id: &str,
        model_type: ModelType,
        architecture: Architecture,
        supports_quantization: bool,
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
            supports_quantization,
        }
    }

    fn to_model_info(&self, alias: &str, quantization: Quantization) -> ModelInfo {
        let repo_id = if quantization.is_quantized() && self.supports_quantization {
            format!("{}/{}{}", self.org, self.base_name, quantization.repo_suffix())
        } else {
            format!("{}/{}", self.org, self.base_name)
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

        // (alias, repo_id, model_type, architecture, supports_quantization)
        let model_entries = [
            // BGE Embedding Models (no quantization variants)
            ("bge-small-zh", "BAAI/bge-small-zh-v1.5", ModelType::Embedding, Architecture::Bert, false),
            ("bge-small-en", "BAAI/bge-small-en-v1.5", ModelType::Embedding, Architecture::Bert, false),
            ("bge-base-en", "BAAI/bge-base-en-v1.5", ModelType::Embedding, Architecture::Bert, false),
            ("bge-large-en", "BAAI/bge-large-en-v1.5", ModelType::Embedding, Architecture::Bert, false),

            // Sentence Transformers Models (no quantization)
            ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", ModelType::Embedding, Architecture::Bert, false),
            ("all-mpnet-base-v2", "sentence-transformers/all-mpnet-base-v2", ModelType::Embedding, Architecture::Bert, false),
            ("paraphrase-MiniLM-L6-v2", "sentence-transformers/paraphrase-MiniLM-L6-v2", ModelType::Embedding, Architecture::Bert, false),
            ("multi-qa-mpnet-base-dot-v1", "sentence-transformers/multi-qa-mpnet-base-dot-v1", ModelType::Embedding, Architecture::Bert, false),

            // E5 Models (no quantization)
            ("e5-large", "intfloat/e5-large", ModelType::Embedding, Architecture::Bert, false),
            ("e5-base", "intfloat/e5-base", ModelType::Embedding, Architecture::Bert, false),
            ("e5-small", "intfloat/e5-small", ModelType::Embedding, Architecture::Bert, false),

            // JINA Embeddings
            ("jina-embeddings-v2-base-en", "jinaai/jina-embeddings-v2-base-en", ModelType::Embedding, Architecture::Bert, false),
            ("jina-embeddings-v2-small-en", "jinaai/jina-embeddings-v2-small-en", ModelType::Embedding, Architecture::Bert, false),
            ("jina-embeddings-v4", "jinaai/jina-embeddings-v4", ModelType::Embedding, Architecture::JinaV4, true),

            // Chinese Models
            ("m3e-base", "moka-ai/m3e-base", ModelType::Embedding, Architecture::Bert, false),

            // Multilingual Models
            ("multilingual-MiniLM-L12-v2", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", ModelType::Embedding, Architecture::Bert, false),
            ("distiluse-base-multilingual-cased-v1", "sentence-transformers/distiluse-base-multilingual-cased-v1", ModelType::Embedding, Architecture::Bert, false),

            // Code Models (Legacy - BERT-based)
            ("codebert-base", "claudios/codebert-base", ModelType::Embedding, Architecture::Bert, false),
            ("starencoder", "bigcode/starencoder", ModelType::Embedding, Architecture::Bert, false),
            ("graphcodebert-base", "claudios/graphcodebert-base", ModelType::Embedding, Architecture::Bert, false),
            ("unixcoder-base", "claudios/unixcoder-base", ModelType::Embedding, Architecture::Bert, false),

            // Code Models (2024 SOTA - CodeXEmbed / SFR-Embedding-Code)
            // CoIR benchmark SOTA: outperforms Voyage-Code by 20%+
            ("codexembed-400m", "Salesforce/SFR-Embedding-Code-400M_R", ModelType::Embedding, Architecture::Bert, false),
            ("sfr-embedding-code-400m", "Salesforce/SFR-Embedding-Code-400M_R", ModelType::Embedding, Architecture::Bert, false),
            ("codexembed-2b", "Salesforce/SFR-Embedding-Code-2B_R", ModelType::Embedding, Architecture::Qwen2Embedding, false),
            ("codexembed-7b", "Salesforce/SFR-Embedding-Code-7B_R", ModelType::Embedding, Architecture::MistralEmbedding, false),
            ("sfr-embedding-code-2b", "Salesforce/SFR-Embedding-Code-2B_R", ModelType::Embedding, Architecture::Qwen2Embedding, false),
            ("sfr-embedding-code-7b", "Salesforce/SFR-Embedding-Code-7B_R", ModelType::Embedding, Architecture::MistralEmbedding, false),

            // Qwen2/Mistral Generator Models
            ("qwen2-7b-instruct", "Qwen/Qwen2-7B-Instruct", ModelType::Generator, Architecture::Qwen2Generator, false),
            ("mistral-7b-instruct", "mistralai/Mistral-7B-Instruct-v0.2", ModelType::Generator, Architecture::MistralGenerator, false),

            // Light Models for Edge Devices
            ("all-MiniLM-L12-v2", "sentence-transformers/all-MiniLM-L12-v2", ModelType::Embedding, Architecture::Bert, false),
            ("all-distilroberta-v1", "sentence-transformers/all-distilroberta-v1", ModelType::Embedding, Architecture::Bert, false),

            // Qwen3 Embedding Models (supports quantization)
            ("qwen3-embedding-0.6b", "Qwen/Qwen3-Embedding-0.6B", ModelType::Embedding, Architecture::Qwen3Embedding, true),
            ("qwen3-embedding-4b", "Qwen/Qwen3-Embedding-4B", ModelType::Embedding, Architecture::Qwen3Embedding, true),
            ("qwen3-embedding-8b", "Qwen/Qwen3-Embedding-8B", ModelType::Embedding, Architecture::Qwen3Embedding, true),

            // NVIDIA Embedding (supports quantization)
            ("llama-embed-nemotron-8b", "nvidia/llama-embed-nemotron-8b", ModelType::Embedding, Architecture::NVIDIANemotron, true),

            // BGE Rerankers (no quantization)
            ("bge-reranker-v2", "BAAI/bge-reranker-v2-m3", ModelType::Rerank, Architecture::CrossEncoder, false),
            ("bge-reranker-large", "BAAI/bge-reranker-large", ModelType::Rerank, Architecture::CrossEncoder, false),
            ("bge-reranker-base", "BAAI/bge-reranker-base", ModelType::Rerank, Architecture::CrossEncoder, false),

            // MS MARCO Rerankers (no quantization)
            ("ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L-6-v2", ModelType::Rerank, Architecture::CrossEncoder, false),
            ("ms-marco-MiniLM-L-12-v2", "cross-encoder/ms-marco-MiniLM-L-12-v2", ModelType::Rerank, Architecture::CrossEncoder, false),
            ("ms-marco-TinyBERT-L-2-v2", "cross-encoder/ms-marco-TinyBERT-L-2-v2", ModelType::Rerank, Architecture::CrossEncoder, false),
            ("ms-marco-electra-base", "cross-encoder/ms-marco-electra-base", ModelType::Rerank, Architecture::CrossEncoder, false),

            // Specialized Rerankers
            ("quora-distilroberta-base", "cross-encoder/quora-distilroberta-base", ModelType::Rerank, Architecture::CrossEncoder, false),

            // Qwen3 Reranker Models (supports quantization)
            ("qwen3-reranker-0.6b", "Qwen/Qwen3-Reranker-0.6B", ModelType::Rerank, Architecture::Qwen3Reranker, true),
            ("qwen3-reranker-4b", "Qwen/Qwen3-Reranker-4B", ModelType::Rerank, Architecture::Qwen3Reranker, true),
            ("qwen3-reranker-8b", "Qwen/Qwen3-Reranker-8B", ModelType::Rerank, Architecture::Qwen3Reranker, true),

            // Jina Reranker V3 (supports quantization)
            ("jina-reranker-v3", "jinaai/jina-reranker-v3", ModelType::Rerank, Architecture::JinaRerankerV3, true),
        ];

        for (alias, repo_id, model_type, architecture, supports_quant) in model_entries {
            let alias_key = alias.to_ascii_lowercase();
            entries.insert(
                alias_key,
                RegistryEntry::new(repo_id, model_type, architecture, supports_quant),
            );
        }

        Self { entries }
    }

    /// Resolve an alias or repo ID into a model info record.
    ///
    /// Supports quantization suffix:
    /// - `qwen3-embedding-0.6b` - default (fp16/bf16)
    /// - `qwen3-embedding-0.6b:int4` - Int4 quantization
    /// - `qwen3-embedding-0.6b:awq` - AWQ quantization
    /// - `Qwen/Qwen3-Embedding-0.6B-Int4` - direct repo ID
    pub fn resolve(&self, name: &str) -> Result<ModelInfo> {
        let name = name.trim();

        // Check for quantization suffix (alias:quant format)
        if let Some((base, quant_str)) = name.rsplit_once(':') {
            // Try to parse quantization
            if let Some(quantization) = Quantization::from_suffix(quant_str) {
                let base_key = base.to_ascii_lowercase();
                if let Some(entry) = self.entries.get(&base_key) {
                    if entry.supports_quantization {
                        return Ok(entry.to_model_info(&format!("{}:{}", base, quant_str), quantization));
                    } else {
                        // Model doesn't support quantization, ignore suffix
                        return Ok(entry.to_model_info(base, Quantization::None));
                    }
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
    pub fn supports_quantization(&self, alias: &str) -> bool {
        let key = alias.to_ascii_lowercase();
        self.entries.get(&key).map(|e| e.supports_quantization).unwrap_or(false)
    }

    /// Get available quantization variants for a model.
    pub fn available_quantizations(&self, alias: &str) -> Vec<Quantization> {
        if self.supports_quantization(alias) {
            vec![
                Quantization::None,
                Quantization::Int4,
                Quantization::Int8,
                Quantization::AWQ,
                Quantization::GPTQ,
            ]
        } else {
            vec![Quantization::None]
        }
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

        let model_type = if lower.contains("reranker") {
            ModelType::Rerank
        } else if lower.contains("generator") || lower.contains("instruct") || lower.contains("chat") {
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
        } else if lower.contains("qwen3") || lower.contains("qwen-3") {
            match model_type {
                ModelType::Embedding => Architecture::Qwen3Embedding,
                ModelType::Rerank => Architecture::Qwen3Reranker,
                ModelType::Generator => Architecture::Qwen3Generator,
            }
        } else if lower.contains("jina") {
            if lower.contains("reranker") {
                Architecture::JinaRerankerV3
            } else {
                Architecture::JinaV4
            }
        } else if lower.contains("nemotron") {
            Architecture::NVIDIANemotron
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
    fn quantization_ignored_for_unsupported_models() {
        let registry = ModelRegistry::new();

        // BERT models don't support quantization
        let info = registry.resolve("bge-small-en:int4").unwrap();
        assert_eq!(info.repo_id, "BAAI/bge-small-en-v1.5");
        assert!(matches!(info.quantization, Quantization::None));
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

        assert!(registry.supports_quantization("qwen3-embedding-0.6b"));
        assert!(registry.supports_quantization("qwen3-reranker-8b"));
        assert!(registry.supports_quantization("jina-embeddings-v4"));
        assert!(!registry.supports_quantization("bge-small-en"));
        assert!(!registry.supports_quantization("all-MiniLM-L6-v2"));
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
            ("qwen2-7b-instruct", "Qwen/Qwen2-7B-Instruct", ModelType::Generator, Architecture::Qwen2Generator),
            ("mistral-7b-instruct", "mistralai/Mistral-7B-Instruct-v0.2", ModelType::Generator, Architecture::MistralGenerator),
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

        let quants = registry.available_quantizations("qwen3-embedding-8b");
        assert!(quants.len() > 1);
        assert!(quants.contains(&Quantization::Int4));

        let quants = registry.available_quantizations("bge-small-en");
        assert_eq!(quants.len(), 1);
        assert!(quants.contains(&Quantization::None));
    }
}
