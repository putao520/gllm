use gllm_kernels::backend_trait::BackendError;

use crate::adapter::AdapterError;
use crate::engine::executor::ExecutorError;
use crate::loader::LoaderError;

use super::{BackendContext, BackendContextError, BackendExecutor};

pub struct OomFallback<'a> {
    context: &'a BackendContext,
}

impl<'a> OomFallback<'a> {
    pub fn new(context: &'a BackendContext) -> Self {
        Self { context }
    }

    pub fn run<F, T>(&self, mut op: F) -> Result<T, BackendContextError>
    where
        F: FnMut(&mut BackendExecutor) -> Result<T, ExecutorError>,
    {
        let (first_error, should_retry) = {
            let mut executor = self.context.executor_mut();
            match op(&mut executor) {
                Ok(value) => return Ok(value),
                Err(err) => {
                    let retry = executor.is_cuda() && is_oom_error(&err);
                    (err, retry)
                }
            }
        };

        if !should_retry {
            return Err(first_error.into());
        }

        self.context.rebuild_cpu()?;
        let mut executor = self.context.executor_mut();
        Ok(op(&mut executor)?)
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
    ) -> Result<String, BackendContextError> {
        self.fallback
            .run(|executor| executor.generate(prompt, max_tokens, temperature, top_k, top_p))
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

    pub fn embed_batch(&mut self, inputs: &[String]) -> Result<Vec<Vec<f32>>, BackendContextError> {
        self.fallback.run(|executor| {
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
    ) -> Result<Vec<f32>, BackendContextError> {
        self.fallback.run(|executor| {
            let mut scores = Vec::with_capacity(documents.len());
            for doc in documents {
                let mut payload = String::new();
                payload.push_str(query);
                payload.push('\n');
                payload.push_str(doc);
                let score = executor.rerank(&payload)?.first().copied().unwrap_or(0.0);
                scores.push(score);
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
        ExecutorError::Adapter(adapter) => is_adapter_oom(adapter),
        _ => false,
    }
}

fn is_adapter_oom(err: &AdapterError) -> bool {
    match err {
        AdapterError::Loader(loader) => is_loader_oom(loader),
        AdapterError::UnsupportedArchitecture => false,
        AdapterError::Backend(backend) => is_backend_oom(backend),
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
        _ => false,
    }
}

fn is_oom_message(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("out of memory")
        || lower.contains("outofmemory")
        || lower.contains("cuda_error_out_of_memory")
        || lower.contains("cuda error out of memory")
        || lower.contains("device out of memory")
        || lower.contains("insufficient memory")
        || lower.contains("not enough memory")
        || lower.contains("memory allocation")
        || lower.contains("alloc failed")
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
    fn detects_loader_backend_oom_messages() {
        let loader_err = LoaderError::Backend("out of memory".to_string());
        let adapter_err = AdapterError::Loader(loader_err);
        let err = ExecutorError::Adapter(adapter_err);
        assert!(is_oom_error(&err));
    }
}
