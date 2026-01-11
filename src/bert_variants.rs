use crate::model_config::ModelConfig;
use crate::pooling::PoolingStrategy;

/// Supported BERT-family variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BertVariant {
    Bert,
    Roberta,
    DistilBert,
    Electra,
    Albert,
    /// GTE/CodeXEmbed models with RoPE and LastToken pooling.
    Gte,
    Unknown,
}

impl BertVariant {
    /// Detect variant from config hints.
    pub fn detect(config: &ModelConfig) -> Self {
        let ty = config
            .model_type
            .as_deref()
            .unwrap_or_default()
            .to_ascii_lowercase();

        // Check for GTE/CodeXEmbed models (use RoPE position embeddings)
        let is_rope = config
            .position_embedding_type
            .as_ref()
            .map_or(false, |t| t == "rope" || t == "rotary");

        // Check architectures for NewModel (GTE/CodeXEmbed signature)
        let is_new_model = config
            .architectures
            .as_ref()
            .map_or(false, |arches| {
                arches.iter().any(|arch| arch.contains("NewModel"))
            });

        if is_rope || is_new_model || ty.contains("new") || ty.contains("gte") {
            return BertVariant::Gte;
        }

        if ty.contains("roberta") {
            BertVariant::Roberta
        } else if ty.contains("distilbert") {
            BertVariant::DistilBert
        } else if ty.contains("electra") {
            BertVariant::Electra
        } else if ty.contains("albert") {
            BertVariant::Albert
        } else if ty.contains("bert") {
            BertVariant::Bert
        } else {
            // Try to infer from architecture hints if provided.
            if config
                .architectures
                .as_ref()
                .and_then(|arches| {
                    arches
                        .iter()
                        .find(|arch| arch.to_ascii_lowercase().contains("roberta"))
                })
                .is_some()
            {
                BertVariant::Roberta
            } else if config
                .architectures
                .as_ref()
                .and_then(|arches| {
                    arches
                        .iter()
                        .find(|arch| arch.to_ascii_lowercase().contains("distilbert"))
                })
                .is_some()
            {
                BertVariant::DistilBert
            } else {
                BertVariant::Unknown
            }
        }
    }

    /// Token type vocabulary size for the variant; zero disables token type embeddings.
    pub fn type_vocab_size(&self, config: &ModelConfig) -> usize {
        if let Some(size) = config.type_vocab_size {
            return size;
        }

        match self {
            BertVariant::Roberta => 1,
            BertVariant::DistilBert => 0,
            BertVariant::Gte => 2, // GTE/CodeXEmbed uses type_vocab_size=2
            _ => 2,
        }
    }

    /// Default pooling strategy per variant, overridable by config.
    pub fn pooling_strategy(&self, config: &ModelConfig) -> PoolingStrategy {
        if let Some(kind) = config.pooling_type.as_deref() {
            match kind.to_ascii_lowercase().as_str() {
                "cls" => return PoolingStrategy::Cls,
                "max" => return PoolingStrategy::Max,
                "last" | "last_token" => return PoolingStrategy::LastToken,
                "weighted_mean" => return PoolingStrategy::WeightedMean,
                "mean" => return PoolingStrategy::Mean,
                _ => {}
            }
        }

        match self {
            BertVariant::Bert | BertVariant::Roberta | BertVariant::Albert => PoolingStrategy::Cls,
            BertVariant::DistilBert | BertVariant::Electra => PoolingStrategy::Mean,
            // GTE/CodeXEmbed models use LastToken pooling for code embeddings
            BertVariant::Gte => PoolingStrategy::LastToken,
            BertVariant::Unknown => PoolingStrategy::Mean,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_variant_from_model_type() {
        let mut cfg = ModelConfig::default();
        cfg.model_type = Some("roberta".into());
        assert!(matches!(BertVariant::detect(&cfg), BertVariant::Roberta));
        cfg.model_type = Some("distilbert".into());
        assert!(matches!(BertVariant::detect(&cfg), BertVariant::DistilBert));
    }

    #[test]
    fn selects_pooling_from_config_override() {
        let mut cfg = ModelConfig::default();
        cfg.pooling_type = Some("max".into());
        let variant = BertVariant::detect(&cfg);
        assert!(matches!(
            variant.pooling_strategy(&cfg),
            PoolingStrategy::Max
        ));
    }
}
