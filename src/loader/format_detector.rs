//! Auto-detect model file formats from paths or directory contents.

use std::path::{Path, PathBuf};

use super::{LoaderError, Result, WeightFormat};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalModelFiles {
    pub format: WeightFormat,
    pub weights: Vec<PathBuf>,
    pub aux_files: Vec<PathBuf>,
}

pub fn detect_format_from_path(path: &Path) -> Result<WeightFormat> {
    if path.is_dir() {
        let formats = detect_formats_in_dir(path)?;
        return select_single_format(formats);
    }
    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("") // LEGAL: 无扩展名时返回空字符串
        .to_ascii_lowercase();
    match ext.as_str() {
        "safetensors" => Ok(WeightFormat::SafeTensors),
        "gguf" => Ok(WeightFormat::Gguf),
        "gllm" => Ok(WeightFormat::Gllm),
        "onnx" => Ok(WeightFormat::Onnx),
        _ => Err(LoaderError::UnsupportedWeightExtension(ext)),
    }
}

pub fn detect_format_from_paths(paths: &[PathBuf]) -> Result<WeightFormat> {
    if paths.is_empty() {
        return Err(LoaderError::MissingWeights);
    }
    let mut formats = Vec::new();
    for path in paths {
        let format = detect_format_from_path(path)?;
        if !formats.contains(&format) {
            formats.push(format);
        }
    }
    select_single_format(formats)
}

pub fn collect_local_files(
    path: &Path,
    format_hint: Option<WeightFormat>,
) -> Result<LocalModelFiles> {
    if path.is_file() {
        let detected = detect_format_from_path(path)?;
        let format = format_hint.unwrap_or(detected); // LEGAL: 无 hint 时使用检测到的格式
        if detected != format {
            return Err(LoaderError::FormatNotFound(format));
        }
        return Ok(LocalModelFiles {
            format,
            weights: vec![path.to_path_buf()],
            aux_files: Vec::new(),
        });
    }

    if !path.is_dir() {
        return Err(LoaderError::MissingWeights);
    }

    let formats = detect_formats_in_dir(path)?;
    if formats.is_empty() {
        return Err(LoaderError::MissingWeights);
    }

    let format = if let Some(hint) = format_hint {
        if !formats.contains(&hint) {
            return Err(LoaderError::FormatNotFound(hint));
        }
        hint
    } else {
        select_preferred_format(&formats)
    };

    let weights = collect_weights_in_dir(path, format)?;
    let aux_files = collect_aux_files(path);

    Ok(LocalModelFiles {
        format,
        weights,
        aux_files,
    })
}

fn detect_formats_in_dir(dir: &Path) -> Result<Vec<WeightFormat>> {
    let mut formats = Vec::new();

    if !collect_safetensors_candidates(dir).is_empty() || has_safetensors_index(dir) {
        formats.push(WeightFormat::SafeTensors);
    }
    if !collect_gguf_candidates(dir).is_empty() {
        formats.push(WeightFormat::Gguf);
    }
    if !collect_gllm_candidates(dir).is_empty() {
        formats.push(WeightFormat::Gllm);
    }
    if !collect_onnx_candidates(dir).is_empty() {
        formats.push(WeightFormat::Onnx);
    }
    if formats.is_empty() && !collect_pytorch_candidates(dir).is_empty() {
        formats.push(WeightFormat::PyTorch);
    }

    Ok(formats)
}

fn select_single_format(formats: Vec<WeightFormat>) -> Result<WeightFormat> {
    match formats.len() {
        0 => Err(LoaderError::MissingWeights),
        1 => Ok(formats[0]),
        _ => Err(LoaderError::MultipleWeightFormats(formats)),
    }
}

pub fn select_preferred_format(formats: &[WeightFormat]) -> WeightFormat {
    if formats.contains(&WeightFormat::Gllm) {
        return WeightFormat::Gllm;
    }
    if formats.contains(&WeightFormat::SafeTensors) {
        return WeightFormat::SafeTensors;
    }
    if formats.contains(&WeightFormat::Gguf) {
        return WeightFormat::Gguf;
    }
    if formats.contains(&WeightFormat::Onnx) {
        return WeightFormat::Onnx;
    }
    WeightFormat::PyTorch
}

fn collect_weights_in_dir(dir: &Path, format: WeightFormat) -> Result<Vec<PathBuf>> {
    match format {
        WeightFormat::SafeTensors => {
            let mut files = collect_safetensors_candidates(dir);
            files.sort();
            if files.is_empty() {
                return Err(LoaderError::MissingWeights);
            }
            Ok(files)
        }
        WeightFormat::Gguf => {
            let mut files = collect_gguf_candidates(dir);
            if files.is_empty() {
                return Err(LoaderError::MissingWeights);
            }
            // Ω1: 不基于文件名推测，选择第一个
            // 用户如需特定文件，应直接指定文件路径
            files.sort();
            Ok(vec![files.into_iter().next().unwrap()])
        }
        WeightFormat::Onnx => {
            let mut files = collect_onnx_candidates(dir);
            if files.is_empty() {
                return Err(LoaderError::MissingWeights);
            }
            // 优先选择 onnx/ 目录下的文件
            if let Some(first) = files.iter().find(|p| p.to_string_lossy().contains("onnx/")) {
                return Ok(vec![first.clone()]);
            }
            files.sort();
            Ok(vec![files.into_iter().next().unwrap()])
        }
        WeightFormat::PyTorch => {
            let mut files = Vec::new();
            for candidate_dir in candidate_dirs(dir) {
                files.extend(find_files_with_extension(&candidate_dir, "bin"));
                files.extend(find_files_with_extension(&candidate_dir, "pth"));
                files.extend(find_files_with_extension(&candidate_dir, "pt"));
            }
            if files.is_empty() {
                return Err(LoaderError::MissingWeights);
            }
            files.sort();
            Ok(files)
        }
        WeightFormat::Gllm => {
            let mut files = collect_gllm_candidates(dir);
            if files.is_empty() {
                return Err(LoaderError::MissingWeights);
            }
            files.sort();
            Ok(vec![files.into_iter().next().unwrap()])
        }
    }
}

fn collect_aux_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for name in [
        "config.json",
        "configuration.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
    ] {
        let candidate = dir.join(name);
        if candidate.exists() {
            files.push(candidate);
        }
    }
    files
}

fn collect_safetensors_candidates(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for candidate_dir in candidate_dirs(dir) {
        files.extend(find_files_with_extension(&candidate_dir, "safetensors"));
    }
    files
}

fn has_safetensors_index(dir: &Path) -> bool {
    for candidate_dir in candidate_dirs(dir) {
        if candidate_dir.join("model.safetensors.index.json").exists() {
            return true;
        }
    }
    false
}

fn collect_gguf_candidates(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for candidate_dir in candidate_dirs(dir) {
        files.extend(find_files_with_extension(&candidate_dir, "gguf"));
    }
    files
}

fn collect_gllm_candidates(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for candidate_dir in candidate_dirs(dir) {
        files.extend(find_files_with_extension(&candidate_dir, "gllm"));
    }
    files
}

fn collect_onnx_candidates(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let onnx_dir = dir.join("onnx");
    if onnx_dir.exists() {
        files.extend(find_files_with_extension(&onnx_dir, "onnx"));
        if !files.is_empty() {
            return files;
        }
    }
    files.extend(find_files_with_extension(dir, "onnx"));
    files
}

fn collect_pytorch_candidates(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for candidate_dir in candidate_dirs(dir) {
        files.extend(find_files_with_extension(&candidate_dir, "bin"));
        files.extend(find_files_with_extension(&candidate_dir, "pth"));
        files.extend(find_files_with_extension(&candidate_dir, "pt"));
    }
    files
}

fn candidate_dirs(dir: &Path) -> Vec<PathBuf> {
    let mut dirs = vec![dir.to_path_buf()];
    for sub in ["model", "weights"] {
        let candidate = dir.join(sub);
        if candidate.exists() {
            dirs.push(candidate);
        }
    }
    dirs
}

fn find_files_with_extension(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            log::warn!("cannot read directory {}: {e}", dir.display());
            return files;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            continue;
        }
        let matches = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case(ext))
            .unwrap_or(false); // LEGAL: 无扩展名时视为不匹配
        if matches {
            files.push(path);
        }
    }
    files
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_format_from_extension() {
        let path = PathBuf::from("model.safetensors");
        assert_eq!(
            detect_format_from_path(&path).unwrap(),
            WeightFormat::SafeTensors
        );
        let path = PathBuf::from("model.gguf");
        assert_eq!(detect_format_from_path(&path).unwrap(), WeightFormat::Gguf);
        let path = PathBuf::from("model.onnx");
        assert_eq!(detect_format_from_path(&path).unwrap(), WeightFormat::Onnx);
    }

    // ── detect_format_from_path edge cases ───────────────────────────────

    #[test]
    fn detect_format_gllm_extension() {
        let path = PathBuf::from("model.gllm");
        assert_eq!(detect_format_from_path(&path).unwrap(), WeightFormat::Gllm);
    }

    #[test]
    fn detect_format_case_insensitive() {
        let path = PathBuf::from("model.GGUF");
        assert_eq!(detect_format_from_path(&path).unwrap(), WeightFormat::Gguf);
        let path = PathBuf::from("model.SafeTensors");
        assert_eq!(detect_format_from_path(&path).unwrap(), WeightFormat::SafeTensors);
    }

    #[test]
    fn detect_format_unknown_extension_returns_error() {
        let path = PathBuf::from("model.bin");
        assert!(detect_format_from_path(&path).is_err());
        let path = PathBuf::from("model.txt");
        assert!(detect_format_from_path(&path).is_err());
    }

    #[test]
    fn detect_format_no_extension_returns_error() {
        let path = PathBuf::from("model_weights");
        assert!(detect_format_from_path(&path).is_err());
    }

    // ── detect_format_from_paths ─────────────────────────────────────────

    #[test]
    fn detect_format_from_paths_empty_returns_error() {
        assert!(detect_format_from_paths(&[]).is_err());
    }

    #[test]
    fn detect_format_from_paths_single() {
        let paths = vec![PathBuf::from("a.gguf")];
        assert_eq!(detect_format_from_paths(&paths).unwrap(), WeightFormat::Gguf);
    }

    // ── select_single_format ─────────────────────────────────────────────

    #[test]
    fn select_single_format_empty_returns_error() {
        assert!(select_single_format(vec![]).is_err());
    }

    #[test]
    fn select_single_format_one_returns_ok() {
        assert_eq!(select_single_format(vec![WeightFormat::Gguf]).unwrap(), WeightFormat::Gguf);
    }

    #[test]
    fn select_single_format_multiple_returns_error() {
        let formats = vec![WeightFormat::SafeTensors, WeightFormat::Gguf];
        assert!(select_single_format(formats).is_err());
    }

    // ── select_preferred_format ──────────────────────────────────────────

    #[test]
    fn preferred_format_gllm_first() {
        let formats = &[WeightFormat::Gguf, WeightFormat::Gllm, WeightFormat::SafeTensors];
        assert_eq!(select_preferred_format(formats), WeightFormat::Gllm);
    }

    #[test]
    fn preferred_format_safetensors_second() {
        let formats = &[WeightFormat::Gguf, WeightFormat::SafeTensors];
        assert_eq!(select_preferred_format(formats), WeightFormat::SafeTensors);
    }

    #[test]
    fn preferred_format_gguf_third() {
        let formats = &[WeightFormat::Onnx, WeightFormat::Gguf];
        assert_eq!(select_preferred_format(formats), WeightFormat::Gguf);
    }

    #[test]
    fn preferred_format_onnx_fourth() {
        let formats = &[WeightFormat::Onnx, WeightFormat::PyTorch];
        assert_eq!(select_preferred_format(formats), WeightFormat::Onnx);
    }

    #[test]
    fn preferred_format_pytorch_last() {
        assert_eq!(select_preferred_format(&[WeightFormat::PyTorch]), WeightFormat::PyTorch);
    }

    // ── LocalModelFiles struct ───────────────────────────────────────────

    #[test]
    fn local_model_files_debug_clone() {
        let files = LocalModelFiles {
            format: WeightFormat::Gguf,
            weights: vec![PathBuf::from("model.gguf")],
            aux_files: vec![PathBuf::from("config.json")],
        };
        assert_eq!(files.format, WeightFormat::Gguf);
        assert_eq!(files.weights.len(), 1);
        assert_eq!(files.aux_files.len(), 1);
        let cloned = files.clone();
        assert_eq!(cloned.format, files.format);
    }

    // ── WeightFormat variants and trait coverage ─────────────────────────

    #[test]
    fn weight_format_all_variants_are_distinct() {
        let all = [
            WeightFormat::SafeTensors,
            WeightFormat::Gguf,
            WeightFormat::Onnx,
            WeightFormat::PyTorch,
            WeightFormat::Gllm,
        ];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                assert_eq!(i == j, a == b, "variant equality mismatch at {i},{j}");
            }
        }
    }

    #[test]
    fn weight_format_copy_semantics() {
        let a = WeightFormat::Gllm;
        let b = a; // Copy, not move
        let c = a; // Still valid because Copy
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn weight_format_clone_produces_equal() {
        let original = WeightFormat::Onnx;
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn weight_format_debug_output() {
        let variants = [
            (WeightFormat::SafeTensors, "SafeTensors"),
            (WeightFormat::Gguf, "Gguf"),
            (WeightFormat::Onnx, "Onnx"),
            (WeightFormat::PyTorch, "PyTorch"),
            (WeightFormat::Gllm, "Gllm"),
        ];
        for (variant, name) in variants {
            let debug_str = format!("{variant:?}");
            assert!(
                debug_str.contains(name),
                "Debug output for {name} should contain variant name, got: {debug_str}"
            );
        }
    }

    // ── LocalModelFiles struct coverage ──────────────────────────────────

    #[test]
    fn local_model_files_empty_aux_files() {
        let files = LocalModelFiles {
            format: WeightFormat::SafeTensors,
            weights: vec![PathBuf::from("model.safetensors")],
            aux_files: vec![],
        };
        assert!(files.aux_files.is_empty());
        assert_eq!(files.weights.len(), 1);
    }

    #[test]
    fn local_model_files_multiple_weights() {
        let files = LocalModelFiles {
            format: WeightFormat::SafeTensors,
            weights: vec![
                PathBuf::from("model-00001.safetensors"),
                PathBuf::from("model-00002.safetensors"),
                PathBuf::from("model-00003.safetensors"),
            ],
            aux_files: vec![
                PathBuf::from("config.json"),
                PathBuf::from("tokenizer.json"),
            ],
        };
        assert_eq!(files.weights.len(), 3);
        assert_eq!(files.aux_files.len(), 2);
    }

    #[test]
    fn local_model_files_clone_independence() {
        let files = LocalModelFiles {
            format: WeightFormat::Gllm,
            weights: vec![PathBuf::from("model.gllm")],
            aux_files: vec![PathBuf::from("config.json")],
        };
        let mut cloned = files.clone();
        cloned.weights.push(PathBuf::from("extra.gllm"));
        assert_eq!(files.weights.len(), 1, "original should not be mutated");
        assert_eq!(cloned.weights.len(), 2, "clone should have the new entry");
    }

    #[test]
    fn local_model_files_debug_format() {
        let files = LocalModelFiles {
            format: WeightFormat::Gguf,
            weights: vec![PathBuf::from("model.gguf")],
            aux_files: vec![],
        };
        let debug = format!("{files:?}");
        assert!(debug.contains("LocalModelFiles"), "Debug should contain struct name");
        assert!(debug.contains("Gguf"), "Debug should contain format variant");
    }

    // ── detect_format_from_path extended coverage ────────────────────────

    #[test]
    fn detect_format_from_path_with_parent_directory() {
        let path = PathBuf::from("models/llama/model.gguf");
        assert_eq!(detect_format_from_path(&path).unwrap(), WeightFormat::Gguf);
    }

    #[test]
    fn detect_format_from_path_deeply_nested() {
        let path = PathBuf::from("/home/user/.cache/models/org/repo/model.onnx");
        assert_eq!(detect_format_from_path(&path).unwrap(), WeightFormat::Onnx);
    }

    #[test]
    fn detect_format_from_path_dots_in_filename() {
        let path = PathBuf::from("my.model.v2.safetensors");
        assert_eq!(
            detect_format_from_path(&path).unwrap(),
            WeightFormat::SafeTensors
        );
    }

    #[test]
    fn detect_format_from_path_lowercase_gllm() {
        let path = PathBuf::from("weights.gllm");
        assert_eq!(detect_format_from_path(&path).unwrap(), WeightFormat::Gllm);
    }

    #[test]
    fn detect_format_from_path_mixed_case_onnx() {
        let path = PathBuf::from("model.OnNx");
        assert_eq!(detect_format_from_path(&path).unwrap(), WeightFormat::Onnx);
    }

    #[test]
    fn detect_format_from_path_unsupported_returns_extension_in_error() {
        let path = PathBuf::from("archive.zip");
        let err = detect_format_from_path(&path).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("zip"),
            "Error message should contain the unsupported extension, got: {msg}"
        );
    }

    // ── detect_format_from_paths extended coverage ───────────────────────

    #[test]
    fn detect_format_from_paths_deduplicates_same_format() {
        let paths = vec![
            PathBuf::from("model-00001.safetensors"),
            PathBuf::from("model-00002.safetensors"),
            PathBuf::from("model-00003.safetensors"),
        ];
        assert_eq!(
            detect_format_from_paths(&paths).unwrap(),
            WeightFormat::SafeTensors
        );
    }

    #[test]
    fn detect_format_from_paths_mixed_formats_returns_error() {
        let paths = vec![
            PathBuf::from("model.safetensors"),
            PathBuf::from("model.gguf"),
        ];
        let err = detect_format_from_paths(&paths).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Multiple"),
            "Error should mention multiple formats, got: {msg}"
        );
    }

    #[test]
    fn detect_format_from_paths_all_five_formats_mixed() {
        let paths = vec![
            PathBuf::from("a.safetensors"),
            PathBuf::from("b.gguf"),
            PathBuf::from("c.onnx"),
            PathBuf::from("d.gllm"),
        ];
        assert!(detect_format_from_paths(&paths).is_err());
    }

    // ── select_preferred_format extended coverage ────────────────────────

    #[test]
    fn preferred_format_empty_slice_defaults_to_pytorch() {
        assert_eq!(select_preferred_format(&[]), WeightFormat::PyTorch);
    }

    #[test]
    fn preferred_format_all_five_picks_gllm() {
        let all = &[
            WeightFormat::SafeTensors,
            WeightFormat::Gguf,
            WeightFormat::Onnx,
            WeightFormat::PyTorch,
            WeightFormat::Gllm,
        ];
        assert_eq!(select_preferred_format(all), WeightFormat::Gllm);
    }

    #[test]
    fn preferred_format_gllm_over_safetensors() {
        let formats = &[WeightFormat::Gllm, WeightFormat::SafeTensors];
        assert_eq!(select_preferred_format(formats), WeightFormat::Gllm);
    }

    #[test]
    fn preferred_format_safetensors_over_gguf_onnx_pytorch() {
        let formats = &[
            WeightFormat::Gguf,
            WeightFormat::Onnx,
            WeightFormat::PyTorch,
        ];
        assert_eq!(select_preferred_format(formats), WeightFormat::Gguf);
    }

    #[test]
    fn preferred_format_single_gguf() {
        assert_eq!(
            select_preferred_format(&[WeightFormat::Gguf]),
            WeightFormat::Gguf
        );
    }

    #[test]
    fn preferred_format_single_onnx() {
        assert_eq!(
            select_preferred_format(&[WeightFormat::Onnx]),
            WeightFormat::Onnx
        );
    }

    // ── select_single_format extended coverage ───────────────────────────

    #[test]
    fn select_single_format_same_variant_twice() {
        // Even though same variant appears twice, len > 1 still errors
        let formats = vec![WeightFormat::Gguf, WeightFormat::Gguf];
        assert!(select_single_format(formats).is_err());
    }

    #[test]
    fn select_single_format_safetensors_onnx_pair() {
        assert!(select_single_format(vec![WeightFormat::SafeTensors, WeightFormat::Onnx]).is_err());
    }

    #[test]
    fn select_single_format_pytorch_gllm_pair() {
        assert!(select_single_format(vec![WeightFormat::PyTorch, WeightFormat::Gllm]).is_err());
    }

    #[test]
    fn select_single_format_all_five() {
        let formats = vec![
            WeightFormat::SafeTensors,
            WeightFormat::Gguf,
            WeightFormat::Onnx,
            WeightFormat::PyTorch,
            WeightFormat::Gllm,
        ];
        assert!(select_single_format(formats).is_err());
    }

    // ── LoaderError variant coverage ─────────────────────────────────────

    #[test]
    fn loader_error_unsupported_extension_display() {
        let err = LoaderError::UnsupportedWeightExtension("xyz".to_string());
        let msg = err.to_string();
        assert!(msg.contains("xyz"), "Display should contain the extension, got: {msg}");
    }

    #[test]
    fn loader_error_missing_weights_display() {
        let err = LoaderError::MissingWeights;
        let msg = err.to_string();
        assert!(!msg.is_empty(), "MissingWeights Display should not be empty");
    }

    #[test]
    fn loader_error_format_not_found_display() {
        let err = LoaderError::FormatNotFound(WeightFormat::Gllm);
        let msg = err.to_string();
        assert!(
            msg.contains("Gllm"),
            "FormatNotFound Display should contain the format, got: {msg}"
        );
    }

    #[test]
    fn loader_error_multiple_weight_formats_display() {
        let err = LoaderError::MultipleWeightFormats(vec![
            WeightFormat::SafeTensors,
            WeightFormat::Gguf,
        ]);
        let msg = err.to_string();
        assert!(!msg.is_empty(), "MultipleWeightFormats Display should not be empty");
    }

    // ── WeightFormat Hash trait ──────────────────────────────────────────

    #[test]
    fn weight_format_hash_set_dedup() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        assert!(set.insert(WeightFormat::Gguf));
        assert!(!set.insert(WeightFormat::Gguf), "duplicate insert should return false");
        assert!(set.insert(WeightFormat::SafeTensors));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn weight_format_hash_map_key() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(WeightFormat::Gguf, "gguf_value");
        map.insert(WeightFormat::SafeTensors, "st_value");
        assert_eq!(map.get(&WeightFormat::Gguf), Some(&"gguf_value"));
        assert_eq!(map.get(&WeightFormat::SafeTensors), Some(&"st_value"));
        assert_eq!(map.get(&WeightFormat::Onnx), None);
    }

    #[test]
    fn weight_format_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let variants = [
            WeightFormat::SafeTensors,
            WeightFormat::Gguf,
            WeightFormat::Onnx,
            WeightFormat::PyTorch,
            WeightFormat::Gllm,
        ];
        for variant in &variants {
            let mut h1 = DefaultHasher::new();
            let mut h2 = DefaultHasher::new();
            variant.hash(&mut h1);
            variant.hash(&mut h2);
            assert_eq!(h1.finish(), h2.finish(), "hash should be deterministic for {variant:?}");
        }
    }

    #[test]
    fn weight_format_hash_distinct_across_variants() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let variants = [
            WeightFormat::SafeTensors,
            WeightFormat::Gguf,
            WeightFormat::Onnx,
            WeightFormat::PyTorch,
            WeightFormat::Gllm,
        ];
        let hashes: Vec<u64> = variants.iter().map(|v| {
            let mut h = DefaultHasher::new();
            v.hash(&mut h);
            h.finish()
        }).collect();
        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(hashes[i], hashes[j], "variants {:?} and {:?} should produce different hashes", variants[i], variants[j]);
            }
        }
    }

    // ── LocalModelFiles PartialEq/Eq ────────────────────────────────────

    #[test]
    fn local_model_files_equal_when_fields_match() {
        let a = LocalModelFiles {
            format: WeightFormat::Gguf,
            weights: vec![PathBuf::from("model.gguf")],
            aux_files: vec![],
        };
        let b = LocalModelFiles {
            format: WeightFormat::Gguf,
            weights: vec![PathBuf::from("model.gguf")],
            aux_files: vec![],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn local_model_files_not_equal_different_format() {
        let a = LocalModelFiles {
            format: WeightFormat::Gguf,
            weights: vec![],
            aux_files: vec![],
        };
        let b = LocalModelFiles {
            format: WeightFormat::SafeTensors,
            weights: vec![],
            aux_files: vec![],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn local_model_files_not_equal_different_weights() {
        let a = LocalModelFiles {
            format: WeightFormat::Gguf,
            weights: vec![PathBuf::from("a.gguf")],
            aux_files: vec![],
        };
        let b = LocalModelFiles {
            format: WeightFormat::Gguf,
            weights: vec![PathBuf::from("b.gguf")],
            aux_files: vec![],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn local_model_files_not_equal_different_aux() {
        let a = LocalModelFiles {
            format: WeightFormat::Gguf,
            weights: vec![],
            aux_files: vec![PathBuf::from("config.json")],
        };
        let b = LocalModelFiles {
            format: WeightFormat::Gguf,
            weights: vec![],
            aux_files: vec![PathBuf::from("tokenizer.json")],
        };
        assert_ne!(a, b);
    }

    // ── collect_local_files with filesystem ──────────────────────────────

    #[test]
    fn collect_local_files_single_gguf_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("model.gguf");
        std::fs::write(&file_path, b"fake").unwrap();

        let result = collect_local_files(&file_path, None).unwrap();
        assert_eq!(result.format, WeightFormat::Gguf);
        assert_eq!(result.weights, vec![file_path.clone()]);
        assert!(result.aux_files.is_empty());
    }

    #[test]
    fn collect_local_files_single_safetensors_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("model.safetensors");
        std::fs::write(&file_path, b"fake").unwrap();

        let result = collect_local_files(&file_path, None).unwrap();
        assert_eq!(result.format, WeightFormat::SafeTensors);
        assert_eq!(result.weights, vec![file_path.clone()]);
    }

    #[test]
    fn collect_local_files_single_onnx_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("model.onnx");
        std::fs::write(&file_path, b"fake").unwrap();

        let result = collect_local_files(&file_path, None).unwrap();
        assert_eq!(result.format, WeightFormat::Onnx);
    }

    #[test]
    fn collect_local_files_single_gllm_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("model.gllm");
        std::fs::write(&file_path, b"fake").unwrap();

        let result = collect_local_files(&file_path, None).unwrap();
        assert_eq!(result.format, WeightFormat::Gllm);
    }

    #[test]
    fn collect_local_files_file_with_matching_hint() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("model.gguf");
        std::fs::write(&file_path, b"fake").unwrap();

        let result = collect_local_files(&file_path, Some(WeightFormat::Gguf)).unwrap();
        assert_eq!(result.format, WeightFormat::Gguf);
    }

    #[test]
    fn collect_local_files_file_with_conflicting_hint_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("model.gguf");
        std::fs::write(&file_path, b"fake").unwrap();

        let result = collect_local_files(&file_path, Some(WeightFormat::SafeTensors));
        assert!(result.is_err());
    }

    #[test]
    fn collect_local_files_nonexistent_path_returns_error() {
        let path = PathBuf::from("/nonexistent/path/model.gguf");
        let result = collect_local_files(&path, None);
        assert!(result.is_err());
    }

    #[test]
    fn collect_local_files_dir_with_gguf_and_aux() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.gguf"), b"fake").unwrap();
        std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.format, WeightFormat::Gguf);
        assert_eq!(result.weights.len(), 1);
        assert_eq!(result.aux_files.len(), 2);
    }

    #[test]
    fn collect_local_files_empty_dir_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let result = collect_local_files(dir.path(), None);
        assert!(result.is_err());
    }

    #[test]
    fn collect_local_files_dir_with_hint_matching() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.gguf"), b"fake").unwrap();
        std::fs::write(dir.path().join("model.safetensors"), b"fake").unwrap();

        let result = collect_local_files(dir.path(), Some(WeightFormat::Gguf)).unwrap();
        assert_eq!(result.format, WeightFormat::Gguf);
    }

    #[test]
    fn collect_local_files_dir_with_hint_not_found_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.gguf"), b"fake").unwrap();

        let result = collect_local_files(dir.path(), Some(WeightFormat::Onnx));
        assert!(result.is_err());
    }

    #[test]
    fn collect_local_files_dir_prefers_gllm_over_others() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.gguf"), b"fake").unwrap();
        std::fs::write(dir.path().join("model.gllm"), b"fake").unwrap();

        // Without hint, should pick Gllm (highest priority)
        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.format, WeightFormat::Gllm);
    }

    // ── collect_local_files directory with safetensors index ─────────────

    #[test]
    fn collect_local_files_dir_with_safetensors_index() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.safetensors.index.json"), b"{}").unwrap();
        std::fs::write(dir.path().join("model-00001.safetensors"), b"fake").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.format, WeightFormat::SafeTensors);
        assert_eq!(result.weights.len(), 1);
    }

    // ── collect_local_files multiple safetensors files ───────────────────

    #[test]
    fn collect_local_files_dir_multiple_safetensors_sorted() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model-00002.safetensors"), b"fake").unwrap();
        std::fs::write(dir.path().join("model-00001.safetensors"), b"fake").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.format, WeightFormat::SafeTensors);
        assert_eq!(result.weights.len(), 2);
        // Weights should be sorted
        assert!(result.weights[0] < result.weights[1], "weights should be sorted");
    }

    // ── collect_local_files with onnx subdirectory ───────────────────────

    #[test]
    fn collect_local_files_dir_onnx_subdir_preferred() {
        let dir = tempfile::tempdir().unwrap();
        let onnx_dir = dir.path().join("onnx");
        std::fs::create_dir(&onnx_dir).unwrap();
        std::fs::write(onnx_dir.join("model.onnx"), b"fake").unwrap();
        // Also an onnx file in root — should prefer onnx/ subdir
        std::fs::write(dir.path().join("other.onnx"), b"fake").unwrap();

        let result = collect_local_files(dir.path(), Some(WeightFormat::Onnx)).unwrap();
        assert_eq!(result.format, WeightFormat::Onnx);
        assert_eq!(result.weights.len(), 1);
        let weight_str = result.weights[0].to_string_lossy();
        assert!(weight_str.contains("onnx/"), "should prefer onnx/ subdirectory, got: {weight_str}");
    }

    // ── collect_local_files with subdirectories model/ and weights/ ──────

    #[test]
    fn collect_local_files_dir_model_subdir() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join("model");
        std::fs::create_dir(&model_dir).unwrap();
        std::fs::write(model_dir.join("weights.gguf"), b"fake").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.format, WeightFormat::Gguf);
    }

    #[test]
    fn collect_local_files_dir_weights_subdir() {
        let dir = tempfile::tempdir().unwrap();
        let weights_dir = dir.path().join("weights");
        std::fs::create_dir(&weights_dir).unwrap();
        std::fs::write(weights_dir.join("model.safetensors"), b"fake").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.format, WeightFormat::SafeTensors);
    }

    // ── collect_aux_files via collect_local_files ────────────────────────

    #[test]
    fn collect_local_files_aux_config_json() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.gguf"), b"fake").unwrap();
        std::fs::write(dir.path().join("config.json"), b"{}").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert!(result.aux_files.iter().any(|p| p.file_name() == Some(std::ffi::OsStr::new("config.json"))));
    }

    #[test]
    fn collect_local_files_aux_tokenizer_config() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.gguf"), b"fake").unwrap();
        std::fs::write(dir.path().join("tokenizer_config.json"), b"{}").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert!(result.aux_files.iter().any(|p| p.file_name() == Some(std::ffi::OsStr::new("tokenizer_config.json"))));
    }

    #[test]
    fn collect_local_files_aux_all_six_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.gguf"), b"fake").unwrap();
        for name in ["config.json", "configuration.json", "tokenizer.json",
                      "tokenizer_config.json", "special_tokens_map.json", "vocab.json"] {
            std::fs::write(dir.path().join(name), b"{}").unwrap();
        }

        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.aux_files.len(), 6, "should find all 6 aux files");
    }

    #[test]
    fn collect_local_files_aux_no_matching_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.gguf"), b"fake").unwrap();
        std::fs::write(dir.path().join("random.txt"), b"not aux").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert!(result.aux_files.is_empty());
    }

    // ── collect_local_files pytorch format ───────────────────────────────

    #[test]
    fn collect_local_files_dir_pytorch_bin_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("pytorch_model-00001.bin"), b"fake").unwrap();
        std::fs::write(dir.path().join("pytorch_model-00002.bin"), b"fake").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.format, WeightFormat::PyTorch);
        assert_eq!(result.weights.len(), 2);
    }

    #[test]
    fn collect_local_files_dir_pytorch_in_model_subdir() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join("model");
        std::fs::create_dir(&model_dir).unwrap();
        std::fs::write(model_dir.join("model.bin"), b"fake").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.format, WeightFormat::PyTorch);
    }

    // ── detect_format_from_path with directory ───────────────────────────

    #[test]
    fn detect_format_from_path_directory_with_single_format() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.gguf"), b"fake").unwrap();

        let result = detect_format_from_path(dir.path()).unwrap();
        assert_eq!(result, WeightFormat::Gguf);
    }

    #[test]
    fn detect_format_from_path_directory_with_multiple_formats_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.gguf"), b"fake").unwrap();
        std::fs::write(dir.path().join("model.safetensors"), b"fake").unwrap();

        let result = detect_format_from_path(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn detect_format_from_path_empty_directory_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let result = detect_format_from_path(dir.path());
        assert!(result.is_err());
    }

    // ── detect_format_from_paths with unsupported file ───────────────────

    #[test]
    fn detect_format_from_paths_first_unsupported_propagates_error() {
        let paths = vec![PathBuf::from("archive.zip"), PathBuf::from("model.gguf")];
        assert!(detect_format_from_paths(&paths).is_err());
    }

    #[test]
    fn detect_format_from_paths_second_unsupported_propagates_error() {
        let paths = vec![PathBuf::from("model.gguf"), PathBuf::from("archive.zip")];
        assert!(detect_format_from_paths(&paths).is_err());
    }

    // ── Edge cases: files with unusual names ─────────────────────────────

    #[test]
    fn detect_format_hidden_file_with_known_extension() {
        let path = PathBuf::from(".hidden_model.gguf");
        assert_eq!(detect_format_from_path(&path).unwrap(), WeightFormat::Gguf);
    }

    #[test]
    fn detect_format_file_name_is_only_extension() {
        let path = PathBuf::from(".gguf");
        // .gguf has no filename before the dot; extension() returns None
        assert!(detect_format_from_path(&path).is_err());
    }

    #[test]
    fn detect_format_multiple_dots_before_extension() {
        let path = PathBuf::from("my.model.v2.final.gguf");
        assert_eq!(detect_format_from_path(&path).unwrap(), WeightFormat::Gguf);
    }

    #[test]
    fn detect_format_path_with_spaces() {
        let path = PathBuf::from("my model file.gguf");
        assert_eq!(detect_format_from_path(&path).unwrap(), WeightFormat::Gguf);
    }

    #[test]
    fn detect_format_unicode_filename() {
        let path = PathBuf::from("模型权重.gguf");
        assert_eq!(detect_format_from_path(&path).unwrap(), WeightFormat::Gguf);
    }

    // ── Additional LoaderError display coverage ──────────────────────────

    #[test]
    fn loader_error_io_display() {
        let err = LoaderError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "file not found"));
        let msg = err.to_string();
        assert!(msg.contains("file not found"), "IO error display should contain message, got: {msg}");
    }

    #[test]
    fn loader_error_network_display() {
        let err = LoaderError::Network("connection timeout".to_string());
        let msg = err.to_string();
        assert!(msg.contains("connection timeout"), "Network error display should contain message, got: {msg}");
    }

    #[test]
    fn loader_error_cache_display() {
        let err = LoaderError::Cache("disk full".to_string());
        let msg = err.to_string();
        assert!(msg.contains("disk full"), "Cache error display should contain message, got: {msg}");
    }

    #[test]
    fn loader_error_duplicate_tensor_display() {
        let err = LoaderError::DuplicateTensor("layer.0.weight".to_string());
        let msg = err.to_string();
        assert!(msg.contains("layer.0.weight"), "DuplicateTensor display should contain tensor name, got: {msg}");
    }

    #[test]
    fn loader_error_missing_tensor_display() {
        let err = LoaderError::MissingTensor("output.weight".to_string());
        let msg = err.to_string();
        assert!(msg.contains("output.weight"), "MissingTensor display should contain tensor name, got: {msg}");
    }

    #[test]
    fn loader_error_onnx_display() {
        let err = LoaderError::Onnx("invalid protobuf".to_string());
        let msg = err.to_string();
        assert!(msg.contains("invalid protobuf"), "Onnx error display should contain message, got: {msg}");
    }

    #[test]
    fn loader_error_gguf_display() {
        let err = LoaderError::Gguf("bad header".to_string());
        let msg = err.to_string();
        assert!(msg.contains("bad header"), "Gguf error display should contain message, got: {msg}");
    }

    #[test]
    fn loader_error_gllm_format_display() {
        let err = LoaderError::Gllm("corrupt page".to_string());
        let msg = err.to_string();
        assert!(msg.contains("corrupt page"), "Gllm error display should contain message, got: {msg}");
    }

    #[test]
    fn loader_error_arch_detection_display() {
        let err = LoaderError::ArchDetection("unknown architecture".to_string());
        let msg = err.to_string();
        assert!(msg.contains("unknown architecture"), "ArchDetection display should contain message, got: {msg}");
    }

    #[test]
    fn loader_error_authentication_display() {
        let err = LoaderError::AuthenticationError { hint: "set HF_TOKEN".to_string() };
        let msg = err.to_string();
        assert!(msg.contains("set HF_TOKEN"), "AuthenticationError display should contain hint, got: {msg}");
    }

    #[test]
    fn loader_error_backend_display() {
        let err = LoaderError::Backend("no CUDA".to_string());
        let msg = err.to_string();
        assert!(msg.contains("no CUDA"), "Backend error display should contain message, got: {msg}");
    }

    #[test]
    fn loader_error_pytorch_display() {
        let err = LoaderError::Pytorch("bad pickle".to_string());
        let msg = err.to_string();
        assert!(msg.contains("bad pickle"), "Pytorch error display should contain message, got: {msg}");
    }

    #[test]
    fn loader_error_invalid_quantization_display() {
        let err = LoaderError::InvalidQuantization("unknown quant type".to_string());
        let msg = err.to_string();
        assert!(msg.contains("unknown quant type"), "InvalidQuantization display should contain message, got: {msg}");
    }

    #[test]
    fn loader_error_hf_hub_display() {
        let err = LoaderError::HfHub("rate limited".to_string());
        let msg = err.to_string();
        assert!(msg.contains("rate limited"), "HfHub error display should contain message, got: {msg}");
    }

    // ── WeightFormat Eq trait ────────────────────────────────────────────

    #[test]
    fn weight_format_eq_reflexive() {
        let variants = [
            WeightFormat::SafeTensors,
            WeightFormat::Gguf,
            WeightFormat::Onnx,
            WeightFormat::PyTorch,
            WeightFormat::Gllm,
        ];
        for v in &variants {
            assert_eq!(v, v, "Eq reflexive property failed for {v:?}");
        }
    }

    #[test]
    fn weight_format_eq_symmetric() {
        let a = WeightFormat::Gguf;
        let b = WeightFormat::Gguf;
        assert_eq!(a, b);
        assert_eq!(b, a, "Eq symmetric property failed");
    }

    #[test]
    fn weight_format_eq_transitive() {
        let a = WeightFormat::Onnx;
        let b = WeightFormat::Onnx;
        let c = WeightFormat::Onnx;
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c, "Eq transitive property failed");
    }

    // ── collect_local_files with gllm format ─────────────────────────────

    #[test]
    fn collect_local_files_dir_gllm_picks_first_sorted() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model-b.gllm"), b"fake").unwrap();
        std::fs::write(dir.path().join("model-a.gllm"), b"fake").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.format, WeightFormat::Gllm);
        assert_eq!(result.weights.len(), 1);
        let name = result.weights[0].file_name().unwrap().to_string_lossy();
        assert_eq!(name, "model-a.gllm", "should pick first after sort");
    }

    #[test]
    fn collect_local_files_dir_gguf_picks_first_sorted() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model-b.gguf"), b"fake").unwrap();
        std::fs::write(dir.path().join("model-a.gguf"), b"fake").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.format, WeightFormat::Gguf);
        assert_eq!(result.weights.len(), 1);
        let name = result.weights[0].file_name().unwrap().to_string_lossy();
        assert_eq!(name, "model-a.gguf", "should pick first after sort");
    }

    // ── detect_format_from_path all variants via file ────────────────────

    #[test]
    fn detect_format_all_extensions() {
        let cases = [
            ("file.safetensors", WeightFormat::SafeTensors),
            ("file.gguf", WeightFormat::Gguf),
            ("file.onnx", WeightFormat::Onnx),
            ("file.gllm", WeightFormat::Gllm),
        ];
        for (filename, expected) in cases {
            let path = PathBuf::from(filename);
            assert_eq!(detect_format_from_path(&path).unwrap(), expected, "failed for {filename}");
        }
    }

    // ── select_preferred_format with duplicate entries ───────────────────

    #[test]
    fn preferred_format_duplicate_entries_still_picks_highest() {
        let formats = &[WeightFormat::Gllm, WeightFormat::Gllm, WeightFormat::Gguf];
        assert_eq!(select_preferred_format(formats), WeightFormat::Gllm);
    }

    #[test]
    fn preferred_format_only_pytorch_returns_pytorch() {
        assert_eq!(select_preferred_format(&[WeightFormat::PyTorch]), WeightFormat::PyTorch);
    }

    // ── collect_local_files non-file non-dir path ────────────────────────

    #[test]
    fn collect_local_files_nonexistent_file_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let nonexistent = dir.path().join("phantom.gguf");
        let result = collect_local_files(&nonexistent, None);
        assert!(result.is_err());
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  45+ NEW TESTS below
    // ═══════════════════════════════════════════════════════════════════════

    // ── WeightFormat Ord-like ordering via select_preferred_format ─────────

    #[test]
    fn preferred_format_safetensors_beats_gguf_onnx_pytorch() {
        let formats = &[WeightFormat::SafeTensors, WeightFormat::Onnx, WeightFormat::PyTorch];
        assert_eq!(select_preferred_format(formats), WeightFormat::SafeTensors);
    }

    #[test]
    fn preferred_format_gllm_beats_all_individual() {
        assert_eq!(
            select_preferred_format(&[WeightFormat::Gllm, WeightFormat::Onnx]),
            WeightFormat::Gllm
        );
        assert_eq!(
            select_preferred_format(&[WeightFormat::Gllm, WeightFormat::PyTorch]),
            WeightFormat::Gllm
        );
        assert_eq!(
            select_preferred_format(&[WeightFormat::Gllm, WeightFormat::SafeTensors]),
            WeightFormat::Gllm
        );
    }

    #[test]
    fn preferred_format_onnx_beats_pytorch() {
        let formats = &[WeightFormat::Onnx, WeightFormat::PyTorch];
        assert_eq!(select_preferred_format(formats), WeightFormat::Onnx);
    }

    // ── WeightFormat exhaustive variant detection ──────────────────────────

    #[test]
    fn detect_safetensors_uppercase() {
        let path = PathBuf::from("model.SAFETENSORS");
        assert_eq!(detect_format_from_path(&path).unwrap(), WeightFormat::SafeTensors);
    }

    #[test]
    fn detect_gguf_mixed_case() {
        assert_eq!(detect_format_from_path(&PathBuf::from("a.GgUf")).unwrap(), WeightFormat::Gguf);
    }

    #[test]
    fn detect_onnx_uppercase() {
        assert_eq!(detect_format_from_path(&PathBuf::from("model.ONNX")).unwrap(), WeightFormat::Onnx);
    }

    #[test]
    fn detect_gllm_uppercase() {
        assert_eq!(detect_format_from_path(&PathBuf::from("model.GLLM")).unwrap(), WeightFormat::Gllm);
    }

    // ── Unsupported extensions ─────────────────────────────────────────────

    #[test]
    fn detect_format_tar_extension_error() {
        let err = detect_format_from_path(&PathBuf::from("archive.tar")).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("tar"), "expected 'tar' in error, got: {msg}");
    }

    #[test]
    fn detect_format_json_extension_error() {
        let err = detect_format_from_path(&PathBuf::from("config.json")).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("json"), "expected 'json' in error, got: {msg}");
    }

    #[test]
    fn detect_format_csv_extension_error() {
        let err = detect_format_from_path(&PathBuf::from("data.csv")).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("csv"), "expected 'csv' in error, got: {msg}");
    }

    #[test]
    fn detect_format_md_extension_error() {
        let err = detect_format_from_path(&PathBuf::from("readme.md")).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("md"), "expected 'md' in error, got: {msg}");
    }

    // ── select_single_format edge cases ────────────────────────────────────

    #[test]
    fn select_single_format_three_variants_error() {
        let formats = vec![WeightFormat::Gguf, WeightFormat::Onnx, WeightFormat::PyTorch];
        assert!(select_single_format(formats).is_err());
    }

    #[test]
    fn select_single_format_safetensors_only() {
        assert_eq!(
            select_single_format(vec![WeightFormat::SafeTensors]).unwrap(),
            WeightFormat::SafeTensors
        );
    }

    #[test]
    fn select_single_format_gllm_only() {
        assert_eq!(
            select_single_format(vec![WeightFormat::Gllm]).unwrap(),
            WeightFormat::Gllm
        );
    }

    // ── detect_format_from_paths edge cases ────────────────────────────────

    #[test]
    fn detect_format_from_paths_single_onnx() {
        assert_eq!(
            detect_format_from_paths(&[PathBuf::from("model.onnx")]).unwrap(),
            WeightFormat::Onnx
        );
    }

    #[test]
    fn detect_format_from_paths_single_gllm() {
        assert_eq!(
            detect_format_from_paths(&[PathBuf::from("weights.gllm")]).unwrap(),
            WeightFormat::Gllm
        );
    }

    #[test]
    fn detect_format_from_paths_single_pytorch_bin() {
        assert!(detect_format_from_paths(&[PathBuf::from("model.bin")]).is_err());
    }

    #[test]
    fn detect_format_from_paths_two_gguf_deduplicates() {
        let paths = vec![PathBuf::from("a.gguf"), PathBuf::from("b.gguf")];
        assert_eq!(detect_format_from_paths(&paths).unwrap(), WeightFormat::Gguf);
    }

    // ── LoaderError Display for each remaining variant ─────────────────────

    #[test]
    fn loader_error_safetensors_display() {
        // SafeTensors error comes from the safetensors crate; test with a known variant
        let st_err = safetensors::SafeTensorError::InvalidHeader;
        let err = LoaderError::SafeTensors(st_err);
        let msg = err.to_string();
        assert!(!msg.is_empty(), "SafeTensors error display should not be empty");
    }

    #[test]
    fn loader_error_unsupported_dtype_display() {
        let err = LoaderError::UnsupportedDtype(safetensors::Dtype::BOOL);
        let msg = err.to_string();
        assert!(msg.contains("BOOL"), "UnsupportedDtype display should contain dtype, got: {msg}");
    }

    #[test]
    fn loader_error_json_display() {
        let json_str = r#"{"key": invalid}"#;
        let json_err: serde_json::Result<serde_json::Value> = serde_json::from_str(json_str);
        let err = LoaderError::Json(json_err.unwrap_err());
        let msg = err.to_string();
        assert!(!msg.is_empty(), "Json error display should not be empty");
    }

    // ── LoaderError From conversions ───────────────────────────────────────

    #[test]
    fn loader_error_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let loader_err: LoaderError = io_err.into();
        let msg = loader_err.to_string();
        assert!(msg.contains("access denied"), "From<io::Error> should propagate, got: {msg}");
    }

    #[test]
    fn loader_error_from_json_error() {
        let json_err = serde_json::from_str::<serde_json::Value>("bad").unwrap_err();
        let loader_err: LoaderError = json_err.into();
        let msg = loader_err.to_string();
        assert!(msg.contains("JSON"), "From<serde_json::Error> should mention JSON, got: {msg}");
    }

    #[test]
    fn loader_error_from_safetensors_error() {
        let st_err = safetensors::SafeTensorError::InvalidHeader;
        let loader_err: LoaderError = st_err.into();
        let msg = loader_err.to_string();
        assert!(msg.contains("SafeTensors"), "From<SafeTensorError> should mention SafeTensors, got: {msg}");
    }

    // ── WeightFormat in collections ─────────────────────────────────────────

    #[test]
    fn weight_format_vec_dedup_preserves_order() {
        let v = vec![WeightFormat::Gguf, WeightFormat::Gguf, WeightFormat::Onnx];
        let deduped: Vec<WeightFormat> = {
            let mut seen = std::collections::HashSet::new();
            v.into_iter().filter(|f| seen.insert(*f)).collect()
        };
        assert_eq!(deduped, vec![WeightFormat::Gguf, WeightFormat::Onnx]);
    }

    #[test]
    fn weight_format_all_five_unique_in_hashset() {
        use std::collections::HashSet;
        let all: HashSet<WeightFormat> = [
            WeightFormat::SafeTensors,
            WeightFormat::Gguf,
            WeightFormat::Onnx,
            WeightFormat::PyTorch,
            WeightFormat::Gllm,
        ].into_iter().collect();
        assert_eq!(all.len(), 5, "all 5 variants should be unique in HashSet");
    }

    // ── LocalModelFiles with empty weights ─────────────────────────────────

    #[test]
    fn local_model_files_empty_weights_vec() {
        let files = LocalModelFiles {
            format: WeightFormat::Gguf,
            weights: vec![],
            aux_files: vec![],
        };
        assert!(files.weights.is_empty());
        assert!(files.aux_files.is_empty());
    }

    #[test]
    fn local_model_files_equality_both_empty() {
        let a = LocalModelFiles {
            format: WeightFormat::Onnx,
            weights: vec![],
            aux_files: vec![],
        };
        let b = LocalModelFiles {
            format: WeightFormat::Onnx,
            weights: vec![],
            aux_files: vec![],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn local_model_files_inequality_empty_vs_nonempty_weights() {
        let a = LocalModelFiles {
            format: WeightFormat::Gguf,
            weights: vec![],
            aux_files: vec![],
        };
        let b = LocalModelFiles {
            format: WeightFormat::Gguf,
            weights: vec![PathBuf::from("model.gguf")],
            aux_files: vec![],
        };
        assert_ne!(a, b);
    }

    // ── detect_format_from_path with special path characters ───────────────

    #[test]
    fn detect_format_relative_path_dots() {
        let path = PathBuf::from("../sibling/model.gguf");
        assert_eq!(detect_format_from_path(&path).unwrap(), WeightFormat::Gguf);
    }

    #[test]
    fn detect_format_absolute_path() {
        let path = PathBuf::from("/opt/models/llama/model.safetensors");
        assert_eq!(detect_format_from_path(&path).unwrap(), WeightFormat::SafeTensors);
    }

    #[test]
    fn detect_format_path_with_double_extension() {
        let path = PathBuf::from("archive.tar.gguf");
        assert_eq!(detect_format_from_path(&path).unwrap(), WeightFormat::Gguf);
    }

    #[test]
    fn detect_format_empty_filename_only_extension() {
        let path = PathBuf::from(".safetensors");
        assert!(detect_format_from_path(&path).is_err());
    }

    // ── collect_local_files with pytorch .pth and .pt files ────────────────

    #[test]
    fn collect_local_files_dir_pytorch_pth_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.pth"), b"fake").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.format, WeightFormat::PyTorch);
        assert_eq!(result.weights.len(), 1);
    }

    #[test]
    fn collect_local_files_dir_pytorch_pt_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("checkpoint.pt"), b"fake").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.format, WeightFormat::PyTorch);
    }

    #[test]
    fn collect_local_files_dir_pytorch_mixed_bin_pth_pt() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.bin"), b"fake").unwrap();
        std::fs::write(dir.path().join("b.pth"), b"fake").unwrap();
        std::fs::write(dir.path().join("c.pt"), b"fake").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.format, WeightFormat::PyTorch);
        assert_eq!(result.weights.len(), 3, "should collect all 3 pytorch files");
    }

    // ── collect_local_files: safetensors with model/ subdir ────────────────

    #[test]
    fn collect_local_files_safetensors_in_model_subdir() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join("model");
        std::fs::create_dir(&model_dir).unwrap();
        std::fs::write(model_dir.join("weights.safetensors"), b"fake").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.format, WeightFormat::SafeTensors);
        assert_eq!(result.weights.len(), 1);
    }

    // ── collect_local_files: gllm in weights/ subdir ───────────────────────

    #[test]
    fn collect_local_files_gllm_in_weights_subdir() {
        let dir = tempfile::tempdir().unwrap();
        let weights_dir = dir.path().join("weights");
        std::fs::create_dir(&weights_dir).unwrap();
        std::fs::write(weights_dir.join("model.gllm"), b"fake").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.format, WeightFormat::Gllm);
    }

    // ── detect_format_from_path with trailing slash (directory) ────────────

    #[test]
    fn detect_format_directory_trailing_slash() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.gguf"), b"fake").unwrap();

        // Path with trailing slash still works because is_dir() returns true
        let path = dir.path();
        assert!(path.is_dir());
        let result = detect_format_from_path(path).unwrap();
        assert_eq!(result, WeightFormat::Gguf);
    }

    // ── collect_aux_files: partial aux files ───────────────────────────────

    #[test]
    fn collect_local_files_aux_only_vocab() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.gguf"), b"fake").unwrap();
        std::fs::write(dir.path().join("vocab.json"), b"{}").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.aux_files.len(), 1);
        assert!(result.aux_files[0].ends_with("vocab.json"));
    }

    #[test]
    fn collect_local_files_aux_only_special_tokens_map() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.gguf"), b"fake").unwrap();
        std::fs::write(dir.path().join("special_tokens_map.json"), b"{}").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.aux_files.len(), 1);
        let name = result.aux_files[0].file_name().unwrap().to_string_lossy();
        assert_eq!(name, "special_tokens_map.json");
    }

    #[test]
    fn collect_local_files_aux_configuration_json() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.gguf"), b"fake").unwrap();
        std::fs::write(dir.path().join("configuration.json"), b"{}").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.aux_files.len(), 1);
        let name = result.aux_files[0].file_name().unwrap().to_string_lossy();
        assert_eq!(name, "configuration.json");
    }

    // ── collect_local_files: multiple gguf files sorted, only first picked ─

    #[test]
    fn collect_local_files_gguf_multiple_picks_first_sorted() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("zzz.gguf"), b"fake").unwrap();
        std::fs::write(dir.path().join("aaa.gguf"), b"fake").unwrap();
        std::fs::write(dir.path().join("mmm.gguf"), b"fake").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.weights.len(), 1, "GGUF should pick only first after sort");
        let name = result.weights[0].file_name().unwrap().to_string_lossy();
        assert_eq!(name, "aaa.gguf");
    }

    // ── collect_local_files: multiple gllm files sorted, only first picked ─

    #[test]
    fn collect_local_files_gllm_multiple_picks_first_sorted() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("zzz.gllm"), b"fake").unwrap();
        std::fs::write(dir.path().join("aaa.gllm"), b"fake").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.weights.len(), 1, "GLLM should pick only first after sort");
        let name = result.weights[0].file_name().unwrap().to_string_lossy();
        assert_eq!(name, "aaa.gllm");
    }

    // ── WeightFormat Debug format consistency ──────────────────────────────

    #[test]
    fn weight_format_debug_is_not_empty_for_all_variants() {
        let variants = [
            WeightFormat::SafeTensors,
            WeightFormat::Gguf,
            WeightFormat::Onnx,
            WeightFormat::PyTorch,
            WeightFormat::Gllm,
        ];
        for v in &variants {
            let s = format!("{v:?}");
            assert!(!s.is_empty(), "Debug should not be empty for {v:?}");
        }
    }

    // ── detect_format_from_paths: all same format still ok ─────────────────

    #[test]
    fn detect_format_from_paths_all_same_safetensors() {
        let paths = vec![
            PathBuf::from("m1.safetensors"),
            PathBuf::from("m2.safetensors"),
            PathBuf::from("m3.safetensors"),
            PathBuf::from("m4.safetensors"),
        ];
        assert_eq!(
            detect_format_from_paths(&paths).unwrap(),
            WeightFormat::SafeTensors
        );
    }

    // ── collect_local_files: mixed formats in dir without hint picks gllm ──

    #[test]
    fn collect_local_files_mixed_dir_no_hint_picks_gllm() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.gguf"), b"fake").unwrap();
        std::fs::write(dir.path().join("model.safetensors"), b"fake").unwrap();
        std::fs::write(dir.path().join("model.gllm"), b"fake").unwrap();
        std::fs::write(dir.path().join("model.onnx"), b"fake").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.format, WeightFormat::Gllm);
    }

    // ── collect_local_files: mixed formats in dir, hint picks gguf ─────────

    #[test]
    fn collect_local_files_mixed_dir_hint_gguf() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.gguf"), b"fake").unwrap();
        std::fs::write(dir.path().join("model.safetensors"), b"fake").unwrap();

        let result = collect_local_files(dir.path(), Some(WeightFormat::Gguf)).unwrap();
        assert_eq!(result.format, WeightFormat::Gguf);
    }

    // ── select_preferred_format: only safetensors in slice ─────────────────

    #[test]
    fn preferred_format_only_safetensors() {
        assert_eq!(
            select_preferred_format(&[WeightFormat::SafeTensors]),
            WeightFormat::SafeTensors
        );
    }

    // ── WeightFormat Copy used in match arms ────────────────────────────────

    #[test]
    fn weight_format_copy_in_array() {
        let arr = [WeightFormat::Gguf; 3];
        assert!(arr.iter().all(|v| *v == WeightFormat::Gguf));
    }

    // ── GgufError and GllmError From conversions to LoaderError ────────────

    #[test]
    fn loader_error_from_gguf_error() {
        use crate::loader::gguf::GgufError;
        let gguf_err = GgufError::InvalidMagic(0xDEAD_BEEF);
        let loader_err: LoaderError = gguf_err.into();
        let msg = loader_err.to_string();
        assert!(
            msg.contains("GGUF") || msg.contains("magic") || msg.contains("0x"),
            "GgufError conversion should preserve information, got: {msg}"
        );
    }

    #[test]
    fn loader_error_from_gllm_error() {
        use crate::loader::gllm::GllmError;
        let gllm_err = GllmError::InvalidMagic(0xDEAD_BEEF);
        let loader_err: LoaderError = gllm_err.into();
        let msg = loader_err.to_string();
        assert!(
            msg.contains("GLLM") || msg.contains("magic"),
            "GllmError conversion should preserve information, got: {msg}"
        );
    }

    // ── detect_format_from_path: empty string path ─────────────────────────

    #[test]
    fn detect_format_empty_path_returns_error() {
        let path = PathBuf::from("");
        assert!(detect_format_from_path(&path).is_err());
    }

    // ── collect_local_files: file path with hint overrides detection ────────

    #[test]
    fn collect_local_files_file_hint_matches_detected() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("weights.safetensors");
        std::fs::write(&file, b"fake").unwrap();

        let result = collect_local_files(&file, Some(WeightFormat::SafeTensors)).unwrap();
        assert_eq!(result.format, WeightFormat::SafeTensors);
        assert_eq!(result.weights[0], file);
    }

    // ── select_single_format with all pairs ────────────────────────────────

    #[test]
    fn select_single_format_gguf_onnx_pair() {
        assert!(select_single_format(vec![WeightFormat::Gguf, WeightFormat::Onnx]).is_err());
    }

    #[test]
    fn select_single_format_gllm_safetensors_pair() {
        assert!(select_single_format(vec![WeightFormat::Gllm, WeightFormat::SafeTensors]).is_err());
    }

    // ── Case-insensitive directory scan for weight files ────────────────────

    #[test]
    fn collect_local_files_dir_uppercase_gguf_extension() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.GGUF"), b"fake").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.format, WeightFormat::Gguf);
        assert_eq!(result.weights.len(), 1);
    }

    #[test]
    fn collect_local_files_dir_mixed_case_safetensors_extension() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.SafeTensors"), b"fake").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.format, WeightFormat::SafeTensors);
    }

    #[test]
    fn collect_local_files_dir_uppercase_onnx_extension() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.ONNX"), b"fake").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.format, WeightFormat::Onnx);
    }

    #[test]
    fn collect_local_files_dir_uppercase_gllm_extension() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.GLLM"), b"fake").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.format, WeightFormat::Gllm);
    }

    // ── Directory with both model/ and weights/ subdirs simultaneously ──────

    #[test]
    fn collect_local_files_model_and_weights_subdirs() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join("model");
        let weights_dir = dir.path().join("weights");
        std::fs::create_dir(&model_dir).unwrap();
        std::fs::create_dir(&weights_dir).unwrap();
        std::fs::write(model_dir.join("a.gguf"), b"fake").unwrap();
        std::fs::write(weights_dir.join("b.gguf"), b"fake").unwrap();

        let result = collect_local_files(dir.path(), None).unwrap();
        assert_eq!(result.format, WeightFormat::Gguf);
        // Both files from model/ and weights/ should be found
        assert!(result.weights.len() >= 1);
    }

    // ── Safetensors index file only (no actual shards) ─────────────────────

    #[test]
    fn collect_local_files_safetensors_index_only_no_shards_error() {
        let dir = tempfile::tempdir().unwrap();
        // Index file exists but no actual safetensors shard files
        std::fs::write(dir.path().join("model.safetensors.index.json"), b"{}").unwrap();

        let result = collect_local_files(dir.path(), None);
        assert!(result.is_err(), "index without shards should fail with MissingWeights");
    }

    // ── detect_format_from_paths with 5 identical gllm paths ───────────────

    #[test]
    fn detect_format_from_paths_five_identical_gllm() {
        let paths = vec![
            PathBuf::from("a.gllm"),
            PathBuf::from("b.gllm"),
            PathBuf::from("c.gllm"),
            PathBuf::from("d.gllm"),
            PathBuf::from("e.gllm"),
        ];
        assert_eq!(
            detect_format_from_paths(&paths).unwrap(),
            WeightFormat::Gllm
        );
    }

    // ── detect_format common unsupported model extensions ──────────────────

    #[test]
    fn detect_format_ckpt_extension_error() {
        let err = detect_format_from_path(&PathBuf::from("model.ckpt")).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("ckpt"), "expected 'ckpt' in error, got: {msg}");
    }

    #[test]
    fn detect_format_h5_extension_error() {
        let err = detect_format_from_path(&PathBuf::from("weights.h5")).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("h5"), "expected 'h5' in error, got: {msg}");
    }

    // ── LocalModelFiles Debug includes weights path ─────────────────────────

    #[test]
    fn local_model_files_debug_shows_weights_path() {
        let files = LocalModelFiles {
            format: WeightFormat::SafeTensors,
            weights: vec![PathBuf::from("model-00001.safetensors")],
            aux_files: vec![],
        };
        let debug = format!("{files:?}");
        assert!(
            debug.contains("model-00001.safetensors"),
            "Debug should contain weights path, got: {debug}"
        );
    }

    // ── select_preferred_format: all 5 in non-priority order ────────────────

    #[test]
    fn preferred_format_all_five_reversed_order() {
        let formats = &[
            WeightFormat::PyTorch,
            WeightFormat::Onnx,
            WeightFormat::Gguf,
            WeightFormat::SafeTensors,
            WeightFormat::Gllm,
        ];
        // Gllm should still win regardless of order in the slice
        assert_eq!(select_preferred_format(formats), WeightFormat::Gllm);
    }
}
