use crate::engine::executor::ExecutorError;
use crate::loader::LoaderError;
use crate::engine::executor::BackendError;

use super::{BackendContext, BackendContextError, BackendExecutor};

/// Result wrapper that indicates whether a fallback path was used.
pub struct FallbackResult<T> {
    pub value: T,
    pub fallback_used: bool,
}

pub struct OomFallback<'a> {
    context: &'a BackendContext,
}

impl<'a> OomFallback<'a> {
    pub fn new(context: &'a BackendContext) -> Self {
        Self { context }
    }

    pub fn run<F, T>(&self, operation: &str, mut op: F) -> Result<FallbackResult<T>, BackendContextError>
    where
        F: FnMut(&mut BackendExecutor) -> Result<T, ExecutorError>,
    {
        let (first_error, should_retry) = {
            let mut executor = self.context.executor_mut();
            match op(&mut executor) {
                Ok(value) => return Ok(FallbackResult { value, fallback_used: false }),
                Err(err) => {
                    let retry = executor.is_gpu() && is_oom_error(&err);
                    (err, retry)
                }
            }
        };

        if !should_retry {
            return Err(first_error.into());
        }

        log::warn!("OOM fallback triggered: GPU→CPU for {operation}");
        self.context.rebuild_cpu()?;
        let mut executor = self.context.executor_mut();
        let value = op(&mut executor)?;
        Ok(FallbackResult { value, fallback_used: true })
    }
}

pub struct FallbackGenerator<'a> {
    fallback: OomFallback<'a>,
}

impl<'a> FallbackGenerator<'a> {
    pub fn new(context: &'a BackendContext) -> Self {
        Self {
            fallback: OomFallback::new(context),
        }
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Result<FallbackResult<String>, BackendContextError> {
        self.fallback
            .run("generate", |executor| executor.generate(prompt, max_tokens, temperature, top_k, top_p))
    }

    pub fn generate_with_session(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        session_id: u64,
    ) -> Result<FallbackResult<String>, BackendContextError> {
        self.fallback.run("generate_with_session", |executor| {
            executor.generate_with_session(
                prompt, max_tokens, temperature, top_k, top_p, session_id,
            )
        })
    }
}

pub struct FallbackEmbedder<'a> {
    fallback: OomFallback<'a>,
}

impl<'a> FallbackEmbedder<'a> {
    pub fn new(context: &'a BackendContext) -> Self {
        Self {
            fallback: OomFallback::new(context),
        }
    }

    pub fn embed_batch(&mut self, inputs: &[String]) -> Result<FallbackResult<Vec<Vec<f32>>>, BackendContextError> {
        self.fallback.run("embed_batch", |executor| {
            let mut embeddings = Vec::with_capacity(inputs.len());
            for input in inputs {
                embeddings.push(executor.embed(input)?);
            }
            Ok(embeddings)
        })
    }
}

pub struct FallbackReranker<'a> {
    fallback: OomFallback<'a>,
}

impl<'a> FallbackReranker<'a> {
    pub fn new(context: &'a BackendContext) -> Self {
        Self {
            fallback: OomFallback::new(context),
        }
    }

    pub fn rerank_batch(
        &mut self,
        query: &str,
        documents: &[String],
    ) -> Result<FallbackResult<Vec<f32>>, BackendContextError> {
        self.fallback.run("rerank_batch", |executor| {
            let mut scores = Vec::with_capacity(documents.len());
            for doc in documents.iter() {
                let score = executor.rerank_pair(query, doc)?;
                let val = score.first().copied().ok_or_else(|| {
                    crate::engine::executor::BackendError::Cpu(
                        "rerank_pair returned empty scores for query/doc pair".into(),
                    )
                })?;
                scores.push(val);
            }
            Ok(scores)
        })
    }
}

pub fn is_oom_context_error(err: &BackendContextError) -> bool {
    match err {
        BackendContextError::Executor(err) => is_oom_error(err),
        BackendContextError::Loader(err) => is_loader_oom(err),
        BackendContextError::UnsupportedArchitecture(_) => false,
        BackendContextError::Backend(err) => is_backend_oom(err),
    }
}

pub fn is_oom_error(err: &ExecutorError) -> bool {
    match err {
        ExecutorError::Backend(backend) => is_backend_oom(backend),
        ExecutorError::Loader(loader) => is_loader_oom(loader),
        _ => false,
    }
}

fn is_loader_oom(err: &LoaderError) -> bool {
    match err {
        LoaderError::Backend(message) => is_oom_message(message),
        _ => false,
    }
}

fn is_backend_oom(err: &BackendError) -> bool {
    match err {
        BackendError::Cuda(message) => is_oom_message(message),
        BackendError::Hip(message) => is_oom_message(message),
        BackendError::Metal(message) => is_oom_message(message),
        _ => false,
    }
}

fn is_oom_message(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("out of memory")
        || lower.contains("outofmemory")
        || lower.contains("cuda_error_out_of_memory")
        || lower.contains("cuda error out of memory")
        || lower.contains("hip_error_out_of_memory")
        || lower.contains("hipErrorOutOfMemory")
        || lower.contains("device out of memory")
        || lower.contains("insufficient memory")
        || lower.contains("not enough memory")
        || lower.contains("memory allocation")
        || lower.contains("alloc failed")
        || lower.contains("can't allocate")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_backend_oom_messages() {
        let err =
            ExecutorError::Backend(BackendError::Cuda("CUDA_ERROR_OUT_OF_MEMORY".to_string()));
        assert!(is_oom_error(&err));

        let err = ExecutorError::Backend(BackendError::Cuda("unknown".to_string()));
        assert!(!is_oom_error(&err));
    }

    #[test]
    fn detects_hip_oom_messages() {
        let err =
            ExecutorError::Backend(BackendError::Hip("HIP_ERROR_OUT_OF_MEMORY".to_string()));
        assert!(is_oom_error(&err));

        let err =
            ExecutorError::Backend(BackendError::Hip("device out of memory".to_string()));
        assert!(is_oom_error(&err));

        let err = ExecutorError::Backend(BackendError::Hip("unknown".to_string()));
        assert!(!is_oom_error(&err));
    }

    #[test]
    fn detects_metal_oom_messages() {
        let err =
            ExecutorError::Backend(BackendError::Metal("can't allocate buffer".to_string()));
        assert!(is_oom_error(&err));

        let err =
            ExecutorError::Backend(BackendError::Metal("insufficient memory".to_string()));
        assert!(is_oom_error(&err));

        let err = ExecutorError::Backend(BackendError::Metal("unknown".to_string()));
        assert!(!is_oom_error(&err));
    }

    #[test]
    fn detects_loader_backend_oom_messages() {
        let loader_err = LoaderError::Backend("out of memory".to_string());
        let err = ExecutorError::Loader(loader_err);
        assert!(is_oom_error(&err));
    }
}
