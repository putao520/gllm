//! Parallel loading helpers (rayon).

use rayon::prelude::*;
use std::path::PathBuf;

use super::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelLoader {
    enabled: bool,
}

impl ParallelLoader {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn map_paths<T, F>(&self, paths: &[PathBuf], f: F) -> Result<Vec<T>>
    where
        T: Send,
        F: Fn(&PathBuf) -> Result<T> + Sync,
    {
        let results: Vec<Result<T>> = if self.enabled {
            paths.par_iter().map(&f).collect()
        } else {
            paths.iter().map(f).collect()
        };

        let mut out = Vec::with_capacity(results.len());
        for result in results {
            out.push(result?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ParallelLoader ────────────────────────────────────────────────

    #[test]
    fn parallel_loader_new_enabled() {
        let loader = ParallelLoader::new(true);
        assert!(loader.enabled());
    }

    #[test]
    fn parallel_loader_new_disabled() {
        let loader = ParallelLoader::new(false);
        assert!(!loader.enabled());
    }

    #[test]
    fn parallel_loader_map_paths_disabled() {
        let loader = ParallelLoader::new(false);
        let paths = vec![PathBuf::from("/a"), PathBuf::from("/b")];
        let results = loader.map_paths(&paths, |p| Ok(p.to_string_lossy().to_string())).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn parallel_loader_map_paths_enabled() {
        let loader = ParallelLoader::new(true);
        let paths = vec![PathBuf::from("/x"), PathBuf::from("/y")];
        let results = loader.map_paths(&paths, |p| Ok(p.to_string_lossy().to_string())).unwrap();
        assert_eq!(results, vec!["/x", "/y"]);
    }

    #[test]
    fn parallel_loader_map_paths_propagates_error() {
        let loader = ParallelLoader::new(false);
        let paths = vec![PathBuf::from("/bad")];
        let result: Result<Vec<String>> = loader.map_paths(&paths, |_| {
            Err(super::super::LoaderError::Onnx("test error".to_string()))
        });
        assert!(result.is_err());
    }

    #[test]
    fn parallel_loader_map_paths_empty() {
        let loader = ParallelLoader::new(true);
        let paths: Vec<PathBuf> = vec![];
        let results: Vec<String> = loader.map_paths(&paths, |_| Ok("x".to_string())).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn parallel_loader_clone() {
        let loader = ParallelLoader::new(true);
        let cloned = loader;
        assert!(cloned.enabled());
    }

    // ── Additional tests (TEST-PAR-14 .. TEST-PAR-36) ─────────────────────

    #[test]
    // @trace TEST-PAR-14 [req:REQ-LOADER] [level:unit]
    fn map_paths_enabled_propagates_error() {
        // Arrange
        let loader = ParallelLoader::new(true);
        let paths = vec![PathBuf::from("/first"), PathBuf::from("/fail")];

        // Act
        let result: Result<Vec<String>> = loader.map_paths(&paths, |p| {
            if p.to_str() == Some("/fail") {
                Err(super::super::LoaderError::Onnx("parallel failure".to_string()))
            } else {
                Ok("ok".to_string())
            }
        });

        // Assert
        assert!(result.is_err());
    }

    #[test]
    // @trace TEST-PAR-15 [req:REQ-LOADER] [level:unit]
    fn map_paths_disabled_single_path() {
        // Arrange
        let loader = ParallelLoader::new(false);
        let paths = vec![PathBuf::from("/only")];

        // Act
        let results = loader
            .map_paths(&paths, |p| Ok(p.to_string_lossy().to_string()))
            .unwrap();

        // Assert
        assert_eq!(results, vec!["/only"]);
    }

    #[test]
    // @trace TEST-PAR-16 [req:REQ-LOADER] [level:unit]
    fn map_paths_enabled_single_path() {
        // Arrange
        let loader = ParallelLoader::new(true);
        let paths = vec![PathBuf::from("/solo")];

        // Act
        let results = loader
            .map_paths(&paths, |p| Ok(p.to_string_lossy().to_string()))
            .unwrap();

        // Assert
        assert_eq!(results, vec!["/solo"]);
    }

    #[test]
    // @trace TEST-PAR-17 [req:REQ-LOADER] [level:unit]
    fn map_paths_disabled_empty() {
        // Arrange
        let loader = ParallelLoader::new(false);
        let paths: Vec<PathBuf> = vec![];

        // Act
        let results: Vec<String> = loader.map_paths(&paths, |_| Ok("x".to_string())).unwrap();

        // Assert
        assert!(results.is_empty());
    }

    #[test]
    // @trace TEST-PAR-18 [req:REQ-LOADER] [level:unit]
    fn map_paths_preserves_order_disabled() {
        // Arrange
        let loader = ParallelLoader::new(false);
        let paths: Vec<PathBuf> = (0..10).map(|i| PathBuf::from(format!("/path{}", i))).collect();

        // Act
        let results = loader
            .map_paths(&paths, |p| Ok(p.to_string_lossy().to_string()))
            .unwrap();

        // Assert
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r, &format!("/path{}", i));
        }
    }

    #[test]
    // @trace TEST-PAR-19 [req:REQ-LOADER] [level:unit]
    fn map_paths_preserves_order_enabled() {
        // Arrange
        let loader = ParallelLoader::new(true);
        let paths: Vec<PathBuf> = (0..10).map(|i| PathBuf::from(format!("/p{}", i))).collect();

        // Act
        let results = loader
            .map_paths(&paths, |p| Ok(p.to_string_lossy().to_string()))
            .unwrap();

        // Assert
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r, &format!("/p{}", i));
        }
    }

    #[test]
    // @trace TEST-PAR-20 [req:REQ-LOADER] [level:unit]
    fn map_paths_returns_integer_type() {
        // Arrange
        let loader = ParallelLoader::new(true);
        let paths = vec![PathBuf::from("/a"), PathBuf::from("/b")];

        // Act — closure returns u64, not String
        let results = loader
            .map_paths(&paths, |p| Ok(p.to_string_lossy().len() as u64))
            .unwrap();

        // Assert
        assert_eq!(results, vec![2u64, 2u64]);
    }

    #[test]
    // @trace TEST-PAR-21 [req:REQ-LOADER] [level:unit]
    fn parallel_loader_debug_format() {
        // Arrange
        let loader = ParallelLoader::new(true);

        // Act
        let debug_str = format!("{:?}", loader);

        // Assert
        assert!(debug_str.contains("ParallelLoader"));
        assert!(debug_str.contains("enabled"));
    }

    #[test]
    // @trace TEST-PAR-26 [req:REQ-LOADER] [level:unit]
    fn map_paths_disabled_all_errors() {
        // Arrange
        let loader = ParallelLoader::new(false);
        let paths = vec![PathBuf::from("/e1"), PathBuf::from("/e2")];

        // Act — all paths produce errors
        let result: Result<Vec<String>> = loader.map_paths(&paths, |_| {
            Err(super::super::LoaderError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "file not found",
            )))
        });

        // Assert — first error propagates (short-circuits in sequential mode)
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("not found"), "error message should contain 'not found'");
    }

    // ── Additional tests (TEST-PAR-27 .. TEST-PAR-36) ─────────────────────

    #[test]
    // @trace TEST-PAR-27 [req:REQ-LOADER] [level:unit]
    fn parallel_loader_copy_semantics() {
        // Arrange
        let loader = ParallelLoader::new(true);
        let copy = loader; // Copy, not move (ParallelLoader is Copy)
        let another = loader; // original still usable after copy

        // Assert — all copies share the same state
        assert!(loader.enabled());
        assert!(copy.enabled());
        assert!(another.enabled());
    }

    #[test]
    // @trace TEST-PAR-28 [req:REQ-LOADER] [level:unit]
    fn parallel_loader_equality() {
        // Arrange
        let a = ParallelLoader::new(true);
        let b = ParallelLoader::new(true);
        let c = ParallelLoader::new(false);

        // Assert — same construction yields equal values
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    // @trace TEST-PAR-29 [req:REQ-LOADER] [level:unit]
    fn map_paths_enabled_multiple_error_variants() {
        // Arrange — closure returns different LoaderError variants
        let loader = ParallelLoader::new(true);
        let paths = vec![PathBuf::from("/a"), PathBuf::from("/b")];

        // Act — use MissingWeights variant
        let result: Result<Vec<String>> = loader.map_paths(&paths, |_| {
            Err(super::super::LoaderError::MissingWeights)
        });

        // Assert
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Missing weights"),
            "expected MissingWeights message, got: {msg}"
        );
    }

    #[test]
    // @trace TEST-PAR-30 [req:REQ-LOADER] [level:unit]
    fn map_paths_sequential_order_preserved() {
        let loader = ParallelLoader::new(false);
        let paths = vec![PathBuf::from("/a"), PathBuf::from("/b"), PathBuf::from("/c")];

        let results: Vec<String> = loader
            .map_paths(&paths, |p| Ok(p.to_string_lossy().to_string()))
            .unwrap();

        assert_eq!(results, vec!["/a", "/b", "/c"]);
    }

    #[test]
    // @trace TEST-PAR-31 [req:REQ-LOADER] [level:unit]
    fn map_paths_preserves_capacity_hint() {
        // Arrange — verify output Vec has correct capacity for input size
        let loader = ParallelLoader::new(false);
        let paths: Vec<PathBuf> = (0..7).map(|i| PathBuf::from(format!("/f{i}"))).collect();

        // Act
        let results: Vec<String> = loader
            .map_paths(&paths, |p| Ok(p.to_string_lossy().to_string()))
            .unwrap();

        // Assert
        assert_eq!(results.len(), 7);
        assert!(results.capacity() >= 7);
    }

    #[test]
    // @trace TEST-PAR-32 [req:REQ-LOADER] [level:unit]
    fn map_paths_unicode_path() {
        // Arrange — paths with unicode characters
        let loader = ParallelLoader::new(true);
        let paths = vec![
            PathBuf::from("/模型/权重"),
            PathBuf::from("/données/fichier"),
        ];

        // Act
        let results = loader
            .map_paths(&paths, |p| Ok(p.to_string_lossy().to_string()))
            .unwrap();

        // Assert — unicode preserved
        assert_eq!(results.len(), 2);
        assert!(results[0].contains("模型"));
        assert!(results[1].contains("données"));
    }

    #[test]
    // @trace TEST-PAR-34 [req:REQ-LOADER] [level:unit]
    fn map_paths_returns_unit_tuple() {
        // Arrange — T = (), testing zero-sized return type
        let loader = ParallelLoader::new(false);
        let paths = vec![PathBuf::from("/a"), PathBuf::from("/b")];

        // Act
        let results: Vec<()> = loader.map_paths(&paths, |_| Ok(())).unwrap();

        // Assert
        assert_eq!(results.len(), 2);
    }

    #[test]
    // @trace TEST-PAR-36 [req:REQ-LOADER] [level:unit]
    fn map_paths_disabled_propagates_network_error() {
        // Arrange
        let loader = ParallelLoader::new(false);
        let paths = vec![PathBuf::from("/remote")];

        // Act
        let result: Result<Vec<String>> = loader.map_paths(&paths, |_| {
            Err(super::super::LoaderError::Network("connection refused".to_string()))
        });

        // Assert — Network variant propagates with correct message
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("connection refused"),
            "expected network error message, got: {msg}"
        );
    }
}
