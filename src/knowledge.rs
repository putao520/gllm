//! Knowledge Injection API (API-KNOWLEDGE-INJECTION)
//!
//! This module implements the knowledge injection infrastructure defined in
//! SPEC 04-API-DESIGN.md §7 and §8. It provides semantic anchor-based layer
//! targeting and polymorphic knowledge source abstraction.
//!
//! # Core Concepts
//!
//! - **LayerTarget**: Semantic anchors (ShallowSyntax, MidSemantic, DeepLogic)
//!   instead of hardcoded layer numbers
//! - **KnowledgeSource**: Polymorphic data source abstraction (VectorDB, Text,
//!   pre-computed embeddings)
//! - **KnowledgeInjectionConfig**: Configuration for injection operations
//!
//! # Example
//!
//! ```ignore
//! use gllm::knowledge::{KnowledgeSource, LayerTarget, KnowledgeInjectionConfig};
//!
//! let config = KnowledgeInjectionConfig {
//!     source: KnowledgeSource::Text("Company policy document...".to_string()),
//!     target: LayerTarget::MidSemantic,
//! };
//!
//! client.inject_knowledge(config)?;
//! ```

use std::path::PathBuf;
use std::time::Duration;

/// Semantic anchor for layer targeting (SPEC 04-API-DESIGN.md §7.1).
///
/// Instead of using hardcoded layer numbers (e.g., `layer=15`), the engine
/// dynamically calculates the mapping to physical layer depths based on
/// the model's topology and entropy distribution curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerTarget {
    /// Shallow lexical processing zone (early layers, ~0-25% depth)
    ///
    /// Characterized by token-level pattern matching, basic syntax recognition,
    /// and surface-level feature extraction.
    ShallowSyntax,

    /// Mid-level semantic zone (middle layers, ~25-75% depth)
    ///
    /// Characterized by semantic understanding, context integration, and
    /// abstract feature formation. Ideal for knowledge injection.
    MidSemantic,

    /// Deep logic zone (late layers, ~75-100% depth)
    ///
    /// Characterized by complex reasoning, logical synthesis, and decision
    /// formation. Just before token generation ("爆词前夕").
    DeepLogic,
}

impl LayerTarget {
    /// Returns the normalized depth (0.0 to 1.0) for this layer target.
    ///
    /// This is used by the engine to map semantic anchors to physical layer
    /// numbers based on the loaded model's topology.
    pub fn normalized_depth(&self) -> f32 {
        match self {
            LayerTarget::ShallowSyntax => 0.125,  // ~12.5% depth
            LayerTarget::MidSemantic => 0.5,      // ~50% depth
            LayerTarget::DeepLogic => 0.875,      // ~87.5% depth
        }
    }

    /// Maps the normalized depth to a physical layer number.
    ///
    /// # Arguments
    ///
    /// * `total_layers` - Total number of layers in the model
    ///
    /// # Returns
    ///
    /// The physical layer index (0-based) corresponding to this semantic anchor.
    pub fn to_physical_layer(&self, total_layers: usize) -> usize {
        let depth = self.normalized_depth();
        (depth * total_layers as f32).floor() as usize
    }
}

/// Knowledge injection kind (SPEC 04-API-DESIGN.md §8.1).
///
/// Identifies the physical injection mechanism for compiler dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InjectionKind {
    /// Frozen KV chunk: Pre-computed and stored KV cache from SSD or network.
    ///
    /// Example: 4-bit quantized financial reports stored as `.kv` files.
    FrozenKvChunk,

    /// Late fusion vector: Dense feature vectors from upstream models.
    ///
    /// Example: BERT-encoded document representations injected as residual.
    LateFusionVector,

    /// Dynamic LoRA: Domain-adapted weight slices with scaling factors.
    ///
    /// Example: Legal/medical domain adapters loaded at runtime.
    DynamicLora,
}

/// Materialized knowledge payload ready for engine consumption.
///
/// This is the output of `KnowledgeSource::materialize()` and contains
/// engine-ready data structures.
#[derive(Debug, Clone)]
pub struct MaterializedPayload {
    /// The injection kind for compiler dispatch
    pub kind: InjectionKind,

    /// Raw bytes of the knowledge data
    pub data: Vec<u8>,

    /// Optional shape information for tensor-like data
    pub shape: Vec<usize>,

    /// Metadata for the payload (e.g., dtype, format version)
    pub metadata: std::collections::HashMap<String, String>,
}

/// Polymorphic knowledge source abstraction (SPEC 04-API-DESIGN.md §8.1).
///
/// Developers interact with this pure data interface without understanding
/// underlying concepts like `LDG.E` or virtual page tables.
///
/// # Example
///
/// ```ignore
/// // VectorDB source
/// let source = KnowledgeSource::VectorDb {
///     connection_string: "postgresql://localhost/vectordb".to_string(),
///     collection: "company_docs".to_string(),
///     query: "Q4 2025 financial results".to_string(),
///     top_k: 10,
/// };
///
/// // Text source
/// let source = KnowledgeSource::Text("Knowledge content here".to_string());
/// ```
#[derive(Debug, Clone)]
pub enum KnowledgeSource {
    /// Vector database source (RAG retrieval).
    ///
    /// Connects to a vector database (e.g., pgvector, Qdrant, Milvus) to
    /// retrieve relevant documents based on semantic similarity.
    VectorDb {
        /// Database connection string
        connection_string: String,

        /// Collection/table name
        collection: String,

        /// Query embedding or text
        query: String,

        /// Number of top results to retrieve
        top_k: usize,
    },

    /// Direct text content.
    ///
    /// Simple text knowledge that will be embedded and injected.
    Text(String),

    /// Pre-computed embedding vector.
    ///
    /// Already-embedded dense vector ready for injection.
    EmbeddedVector {
        /// The embedding values
        embedding: Vec<f32>,

        /// Optional metadata about the source
        metadata: Option<String>,
    },

    /// Frozen KV cache file.
    ///
    /// Pre-computed KV cache stored on disk (zero-copy page table injection).
    FrozenKv {
        /// Path to the KV cache file
        path: PathBuf,

        /// Optional offset within the file
        offset: Option<usize>,

        /// Optional length to read (None = entire file)
        length: Option<usize>,
    },

    /// LoRA adapter file.
    ///
    /// Domain-specific adapter weights for dynamic injection.
    LoRA {
        /// Path to the LoRA weights file
        path: PathBuf,

        /// Scaling factor for the adapter
        scaling: f32,
    },
}

impl KnowledgeSource {
    /// Creates a knowledge source from a frozen KV cache file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the `.kv` file containing pre-computed KV cache
    ///
    /// # Example
    ///
    /// ```ignore
    /// let source = KnowledgeSource::from_frozen_kv("company_logs_dec_2025.kv");
    /// ```
    pub fn from_frozen_kv<P: Into<PathBuf>>(path: P) -> Self {
        KnowledgeSource::FrozenKv {
            path: path.into(),
            offset: None,
            length: None,
        }
    }

    /// Creates a knowledge source from direct text content.
    ///
    /// # Arguments
    ///
    /// * `text` - The text content to be injected
    ///
    /// # Example
    ///
    /// ```ignore
    /// let source = KnowledgeSource::from_text("Company policy: ...");
    /// ```
    pub fn from_text<S: Into<String>>(text: S) -> Self {
        KnowledgeSource::Text(text.into())
    }

    /// Creates a knowledge source from a pre-computed embedding.
    ///
    /// # Arguments
    ///
    /// * `embedding` - The embedding vector
    ///
    /// # Example
    ///
    /// ```ignore
    /// let source = KnowledgeSource::from_embedding(vec![0.1, 0.2, 0.3, ...]);
    /// ```
    pub fn from_embedding(embedding: Vec<f32>) -> Self {
        KnowledgeSource::EmbeddedVector {
            embedding,
            metadata: None,
        }
    }

    /// Returns the injection kind for compiler dispatch.
    pub fn injection_kind(&self) -> InjectionKind {
        match self {
            KnowledgeSource::VectorDb { .. } | KnowledgeSource::Text(_) => {
                InjectionKind::LateFusionVector
            }
            KnowledgeSource::EmbeddedVector { .. } => InjectionKind::LateFusionVector,
            KnowledgeSource::FrozenKv { .. } => InjectionKind::FrozenKvChunk,
            KnowledgeSource::LoRA { .. } => InjectionKind::DynamicLora,
        }
    }
}

/// Knowledge injection configuration.
///
/// Combines a knowledge source with a target layer for injection.
#[derive(Debug, Clone)]
pub struct KnowledgeInjectionConfig {
    /// The source of knowledge to inject
    pub source: KnowledgeSource,

    /// The semantic anchor target for injection
    pub target: LayerTarget,

    /// Optional time-to-live for the injected knowledge
    pub ttl: Option<Duration>,

    /// Optional priority for injection scheduling
    pub priority: Option<u8>,
}

impl KnowledgeInjectionConfig {
    /// Creates a new knowledge injection configuration.
    ///
    /// # Arguments
    ///
    /// * `source` - The knowledge source to inject
    /// * `target` - The semantic anchor for layer targeting
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = KnowledgeInjectionConfig::new(
    ///     KnowledgeSource::from_text("Knowledge content"),
    ///     LayerTarget::MidSemantic,
    /// );
    /// ```
    pub fn new(source: KnowledgeSource, target: LayerTarget) -> Self {
        Self {
            source,
            target,
            ttl: None,
            priority: None,
        }
    }

    /// Sets the time-to-live for the injected knowledge.
    ///
    /// # Arguments
    ///
    /// * `ttl` - Duration after which the knowledge should be evicted
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Sets the priority for injection scheduling.
    ///
    /// # Arguments
    ///
    /// * `priority` - Priority value (higher = more important, 0-255)
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = Some(priority);
        self
    }

    /// Materializes the knowledge source into an engine-ready payload.
    ///
    /// This is a skeleton implementation that returns a placeholder payload.
    /// A full implementation would:
    /// 1. For VectorDb: Query the database and retrieve documents
    /// 2. For Text: Run embedding inference to generate vectors
    /// 3. For FrozenKv: Memory-map the file and create page table entries
    /// 4. For LoRA: Load and validate the adapter weights
    pub fn materialize(&self) -> Result<MaterializedPayload, KnowledgeError> {
        // Skeleton implementation - returns placeholder data
        let kind = self.source.injection_kind();
        let data = match &self.source {
            KnowledgeSource::Text(text) => {
                // In a real implementation, this would run embedding inference
                text.as_bytes().to_vec()
            }
            KnowledgeSource::EmbeddedVector { embedding, .. } => {
                // Serialize the embedding to bytes
                let mut bytes = Vec::with_capacity(embedding.len() * 4);
                for &val in embedding {
                    bytes.extend_from_slice(&val.to_le_bytes());
                }
                bytes
            }
            _ => Vec::new(), // Placeholder for other sources
        };

        let mut metadata = std::collections::HashMap::new();
        metadata.insert("target".to_string(), format!("{:?}", self.target));
        if let Some(ttl) = &self.ttl {
            metadata.insert("ttl_secs".to_string(), ttl.as_secs().to_string());
        }

        Ok(MaterializedPayload {
            kind,
            data,
            shape: vec![],
            metadata,
        })
    }
}

/// Knowledge injection error types.
#[derive(Debug, thiserror::Error)]
pub enum KnowledgeError {
    /// VectorDB connection or query failed
    #[error("VectorDB error: {0}")]
    VectorDb(String),

    /// File I/O error for frozen KV or LoRA files
    #[error("File error: {0}")]
    File(#[from] std::io::Error),

    /// Invalid knowledge source configuration
    #[error("Invalid knowledge source: {0}")]
    InvalidSource(String),

    /// Engine not ready for knowledge injection
    #[error("Engine not ready: {0}")]
    EngineNotReady(String),

    /// Materialization failed
    #[error("Failed to materialize knowledge: {0}")]
    MaterializationFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_target_normalized_depth() {
        assert_eq!(LayerTarget::ShallowSyntax.normalized_depth(), 0.125);
        assert_eq!(LayerTarget::MidSemantic.normalized_depth(), 0.5);
        assert_eq!(LayerTarget::DeepLogic.normalized_depth(), 0.875);
    }

    #[test]
    fn test_layer_target_to_physical_layer() {
        // 32-layer model
        let total = 32;
        assert_eq!(LayerTarget::ShallowSyntax.to_physical_layer(total), 4);
        assert_eq!(LayerTarget::MidSemantic.to_physical_layer(total), 16);
        assert_eq!(LayerTarget::DeepLogic.to_physical_layer(total), 28);

        // 64-layer model
        let total = 64;
        assert_eq!(LayerTarget::ShallowSyntax.to_physical_layer(total), 8);
        assert_eq!(LayerTarget::MidSemantic.to_physical_layer(total), 32);
        assert_eq!(LayerTarget::DeepLogic.to_physical_layer(total), 56);
    }

    #[test]
    fn test_knowledge_source_from_text() {
        let source = KnowledgeSource::from_text("Test content");
        assert!(matches!(source, KnowledgeSource::Text(_)));
        assert_eq!(source.injection_kind(), InjectionKind::LateFusionVector);
    }

    #[test]
    fn test_knowledge_source_from_frozen_kv() {
        let source = KnowledgeSource::from_frozen_kv("/path/to/cache.kv");
        assert!(matches!(source, KnowledgeSource::FrozenKv { .. }));
        assert_eq!(source.injection_kind(), InjectionKind::FrozenKvChunk);
    }

    #[test]
    fn test_knowledge_source_from_embedding() {
        let embedding = vec![0.1_f32, 0.2, 0.3, 0.4];
        let source = KnowledgeSource::from_embedding(embedding.clone());
        assert!(matches!(source, KnowledgeSource::EmbeddedVector { .. }));
        assert_eq!(source.injection_kind(), InjectionKind::LateFusionVector);
    }

    #[test]
    fn test_injection_config_builder() {
        let source = KnowledgeSource::from_text("Test");
        let config = KnowledgeInjectionConfig::new(source, LayerTarget::MidSemantic)
            .with_ttl(Duration::from_secs(300))
            .with_priority(10);

        assert_eq!(config.target, LayerTarget::MidSemantic);
        assert_eq!(config.ttl, Some(Duration::from_secs(300)));
        assert_eq!(config.priority, Some(10));
    }

    #[test]
    fn test_materialize_text_source() {
        let source = KnowledgeSource::from_text("Hello, world!");
        let config = KnowledgeInjectionConfig::new(source, LayerTarget::MidSemantic);
        let payload = config.materialize().unwrap();

        assert_eq!(payload.kind, InjectionKind::LateFusionVector);
        assert_eq!(payload.data, b"Hello, world!");
        assert_eq!(payload.metadata.get("target"), Some(&"MidSemantic".to_string()));
        assert_eq!(payload.metadata.get("ttl_secs"), Some(&"300".to_string()));
    }

    #[test]
    fn test_materialize_embedding_source() {
        let embedding = vec![1.0_f32, 2.0, 3.0];
        let source = KnowledgeSource::from_embedding(embedding);
        let config = KnowledgeInjectionConfig::new(source, LayerTarget::DeepLogic);
        let payload = config.materialize().unwrap();

        assert_eq!(payload.kind, InjectionKind::LateFusionVector);
        // Check that embedding was serialized correctly
        assert_eq!(payload.data.len(), 12); // 3 floats * 4 bytes
    }

    #[test]
    fn test_layer_target_equality() {
        assert_eq!(LayerTarget::ShallowSyntax, LayerTarget::ShallowSyntax);
        assert_ne!(LayerTarget::ShallowSyntax, LayerTarget::MidSemantic);
    }

    #[test]
    fn test_layer_target_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(LayerTarget::ShallowSyntax);
        set.insert(LayerTarget::MidSemantic);
        set.insert(LayerTarget::DeepLogic);
        assert_eq!(set.len(), 3);
    }
}
