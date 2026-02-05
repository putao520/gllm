//! Auto-detect model file formats from paths or directory contents.

use std::path::{Path, PathBuf};

use super::naming_parser::{gguf_candidate_rank, onnx_candidate_rank};
use super::{LoaderError, Result, WeightFormat};

#[derive(Debug, Clone)]
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
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "safetensors" => Ok(WeightFormat::SafeTensors),
        "gguf" => Ok(WeightFormat::Gguf),
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
        let format = format_hint.unwrap_or(detected);
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
    if !collect_onnx_candidates(dir).is_empty() {
        formats.push(WeightFormat::Onnx);
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
    if formats.contains(&WeightFormat::SafeTensors) {
        return WeightFormat::SafeTensors;
    }
    if formats.contains(&WeightFormat::Gguf) {
        return WeightFormat::Gguf;
    }
    WeightFormat::Onnx
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
            select_best_by_rank(collect_gguf_candidates(dir), gguf_candidate_rank)
        }
        WeightFormat::Onnx => {
            select_best_by_rank(collect_onnx_candidates(dir), onnx_candidate_rank)
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
        Err(_) => return files,
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
            .unwrap_or(false);
        if matches {
            files.push(path);
        }
    }
    files
}

fn select_best_by_rank<F>(candidates: Vec<PathBuf>, ranker: F) -> Result<Vec<PathBuf>>
where
    F: Fn(&str) -> Option<(u8, u8)>,
{
    if candidates.is_empty() {
        return Err(LoaderError::MissingWeights);
    }

    let mut scored: Vec<(u8, u8, PathBuf)> = candidates
        .into_iter()
        .filter_map(|path| {
            let name = path.to_string_lossy();
            ranker(&name).map(|(primary, secondary)| (primary, secondary, path))
        })
        .collect();

    if scored.is_empty() {
        return Err(LoaderError::MissingWeights);
    }

    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    Ok(vec![scored.remove(0).2])
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
}
