//! Macro to generate the `Backend<E>` trait impl for GPU backends (CUDA/HIP/Metal).
//!
//! All three GPU backends share an identical trait impl structure:
//! - Each method is cfg-gated on the backend feature
//! - Enabled path delegates to `gpu_helpers::*` or `gpu_compile::*`
//! - Disabled path suppresses unused-variable warnings and returns `Err(BE::Unimplemented(...))`
//!
//! The macro eliminates ~540 lines of triplicated boilerplate.

/// Generate a complete `Backend<E>` implementation for a GPU backend.
///
/// # Parameters
/// - `$backend_ty`: The backend struct name (e.g. `CudaBackend`)
/// - `$cfg_pred`: The cfg predicate as `meta` (e.g. `feature = "cuda"`)
/// - `$feature_label`: A string literal for error messages (e.g. `"cuda"`)
/// - `$upload_err_variant`: The `BackendError` variant for upload errors (e.g. `Other`, `Metal`)
/// - `$decoder_forward_fn`: The decoder forward function in `gpu_compile` (e.g. `cuda_decoder_forward`)
/// - `$bert_forward_fn`: The BERT encoder forward function in `gpu_compile` (e.g. `cuda_bert_encoder_forward`)
macro_rules! impl_gpu_backend {
    (
        backend = $backend_ty:ident,
        cfg_pred = [ $($cfg_pred:tt)+ ],
        feature_label = $feature_label:expr,
        upload_err = $upload_err_variant:ident,
        decoder_forward = $decoder_forward_fn:ident,
        bert_forward = $bert_forward_fn:ident $(,)?
    ) => {
        impl<E: Element> Backend<E> for $backend_ty<E> {
            type Tensor = Vec<E>;

            fn alloc_kv_cache(
                &self,
                config: &KvCacheConfig,
            ) -> Result<KvCacheHandle, BE> {
                #[cfg( $($cfg_pred)+ )]
                {
                    super::gpu_helpers::gpu_alloc_kv_cache(
                        unsafe {
                            &*(self as *const $backend_ty<E> as *const $backend_ty<f32>)
                        },
                        config,
                    )
                }
                #[cfg(not( $($cfg_pred)+ ))]
                {
                    let _ = config;
                    Err(BE::Unimplemented(concat!($feature_label, " feature not enabled")))
                }
            }

            fn batch_forward_gpu_pure(
                &self,
                input: &BatchInput,
                topology: &AttentionTopology,
                weights: &dyn backend_trait::TensorLookup<E, Self>,
                kv_caches: &mut [KvCacheHandle],
                config: &GeneratorForwardConfig,
            ) -> Result<
                (Vec<LogitsHandle>, f32, Vec<crate::scheduler::SequenceTelemetry>),
                BE,
            > {
                #[cfg( $($cfg_pred)+ )]
                {
                    super::gpu_compile::$decoder_forward_fn(
                        self, input, topology, weights, kv_caches, config,
                    )
                    .map(|(logits, telemetry)| {
                        (logits, 0.0, telemetry)
                    })
                }
                #[cfg(not( $($cfg_pred)+ ))]
                {
                    let _ = (input, topology, weights, kv_caches, config);
                    Err(BE::Unimplemented(concat!($feature_label, " feature not enabled")))
                }
            }

            fn sample_from_tensor(
                &self,
                logits: &LogitsHandle,
                topology: &AttentionTopology,
                vocab_size: usize,
                sampling: &SamplingConfig,
            ) -> Result<Vec<u32>, BE> {
                #[cfg( $($cfg_pred)+ )]
                {
                    super::gpu_helpers::gpu_sample_from_tensor(
                        logits, topology, vocab_size, sampling,
                    )
                }
                #[cfg(not( $($cfg_pred)+ ))]
                {
                    let _ = (logits, topology, vocab_size, sampling);
                    Err(BE::Unimplemented(concat!($feature_label, " feature not enabled")))
                }
            }

            fn embedding_forward_gpu_pure(
                &self,
                tokens: &[u32],
                _topology: &AttentionTopology,
                weights: &dyn backend_trait::TensorLookup<E, Self>,
                config: &GeneratorForwardConfig,
            ) -> Result<Vec<f32>, BE> {
                #[cfg( $($cfg_pred)+ )]
                {
                    super::gpu_compile::$bert_forward_fn(self, tokens, weights, config)
                }
                #[cfg(not( $($cfg_pred)+ ))]
                {
                    let _ = (tokens, weights, config);
                    Err(BE::Unimplemented(concat!($feature_label, " feature not enabled")))
                }
            }

            fn rerank_forward_gpu_pure(
                &self,
                tokens: &[u32],
                _topology: &AttentionTopology,
                weights: &dyn backend_trait::TensorLookup<E, Self>,
                config: &GeneratorForwardConfig,
            ) -> Result<Vec<f32>, BE> {
                #[cfg( $($cfg_pred)+ )]
                {
                    super::gpu_compile::$bert_forward_fn(self, tokens, weights, config)
                }
                #[cfg(not( $($cfg_pred)+ ))]
                {
                    let _ = (tokens, weights, config);
                    Err(BE::Unimplemented(concat!($feature_label, " feature not enabled")))
                }
            }

            fn classify_forward_gpu_pure(
                &self,
                tokens: &[u32],
                _topology: &AttentionTopology,
                weights: &dyn backend_trait::TensorLookup<E, Self>,
                config: &GeneratorForwardConfig,
            ) -> Result<Vec<f32>, BE> {
                #[cfg( $($cfg_pred)+ )]
                {
                    super::gpu_compile::$bert_forward_fn(self, tokens, weights, config)
                }
                #[cfg(not( $($cfg_pred)+ ))]
                {
                    let _ = (tokens, weights, config);
                    Err(BE::Unimplemented(concat!($feature_label, " feature not enabled")))
                }
            }

            fn get_memory_pressure(&self) -> Result<f32, BE> {
                #[cfg( $($cfg_pred)+ )]
                {
                    use gllm_kernels::gpu::GpuDevice;
                    let total = self.device.total_memory();
                    if total == 0 {
                        return Ok(0.0);
                    }
                    let free = self.device.free_memory();
                    Ok(1.0 - (free as f32 / total as f32))
                }
                #[cfg(not( $($cfg_pred)+ ))]
                {
                    Ok(0.0)
                }
            }

            fn swap_out_pages(
                &self,
                handle: &mut KvCacheHandle,
                mappings: &[(PageId, StorageKey)],
            ) -> Result<(), BE> {
                #[cfg( $($cfg_pred)+ )]
                {
                    super::gpu_helpers::gpu_swap_out_pages(
                        unsafe {
                            &*(self as *const $backend_ty<E> as *const $backend_ty<f32>)
                        },
                        handle,
                        mappings,
                    )
                }
                #[cfg(not( $($cfg_pred)+ ))]
                {
                    let _ = (handle, mappings);
                    Err(BE::Unimplemented(concat!($feature_label, " feature not enabled")))
                }
            }

            fn swap_in_pages(
                &self,
                handle: &mut KvCacheHandle,
                mappings: &[(PageId, StorageKey)],
            ) -> Result<(), BE> {
                #[cfg( $($cfg_pred)+ )]
                {
                    super::gpu_helpers::gpu_swap_in_pages(
                        unsafe {
                            &*(self as *const $backend_ty<E> as *const $backend_ty<f32>)
                        },
                        handle,
                        mappings,
                    )
                }
                #[cfg(not( $($cfg_pred)+ ))]
                {
                    let _ = (handle, mappings);
                    Err(BE::Unimplemented(concat!($feature_label, " feature not enabled")))
                }
            }

            fn get_page_states(
                &self,
                handle: &KvCacheHandle,
            ) -> Result<Vec<(PageId, PageState)>, BE> {
                #[cfg( $($cfg_pred)+ )]
                {
                    super::gpu_helpers::gpu_get_page_states(
                        unsafe {
                            &*(self as *const $backend_ty<E> as *const $backend_ty<f32>)
                        },
                        handle,
                    )
                }
                #[cfg(not( $($cfg_pred)+ ))]
                {
                    let _ = handle;
                    Err(BE::Unimplemented(concat!($feature_label, " feature not enabled")))
                }
            }

            fn upload_weights(&self, data: &[E]) -> Result<Self::Tensor, BE> {
                #[cfg( $($cfg_pred)+ )]
                {
                    use gllm_kernels::gpu::GpuDevice;
                    let bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            data.as_ptr() as *const u8,
                            data.len() * std::mem::size_of::<E>(),
                        )
                    };
                    let mut buf = self.device.alloc(bytes.len()).map_err(|e| {
                        BE::$upload_err_variant(format!("GPU alloc failed: {e}"))
                    })?;
                    let stream = self.device.default_stream();
                    self.device.htod(bytes, &mut buf, stream).map_err(|e| {
                        BE::$upload_err_variant(format!("GPU htod failed: {e}"))
                    })?;
                    Ok(data.to_vec())
                }
                #[cfg(not( $($cfg_pred)+ ))]
                {
                    let _ = data;
                    Err(BE::Unimplemented(concat!($feature_label, " feature not enabled")))
                }
            }
        }
    };
}

// Macro is exported via #[macro_use] on the module in mod.rs
