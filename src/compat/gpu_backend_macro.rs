//! Macro to generate the `Backend<E>` trait impl for GPU backends (CUDA/HIP/Metal).
//!
//! **ARCH-FULL-JIT + ARCH-CPU-GPU-UNIFIED migration (in progress)**:
//! - Hand-written GPU decoder/encoder forwards have been deleted from `gpu_compile`
//!   (violated ARCH-FULL-JIT).
//! - GPU backend forward paths must go through the same `FusedGraphExecutor`
//!   pipeline as CPU, with JIT codegen driven by `DeviceProfile` producing
//!   PTX/AMDGPU/AIR kernels.
//! - The required host-side glue (GPU KV-cache binding, `run_gpu_with_kv_cache`
//!   on `FusedGraphExecutor`, multi-layer GPU execution across symbolic dims)
//!   is pending a dedicated SPEC workstream. Until that lands, forward methods
//!   return an explicit `Err(BE::Unimplemented(...))` — this is not a fallback
//!   (no silent degradation) but an explicit contractual "not yet implemented"
//!   surfaced at the call site, consistent with the `Backend::quantized_matmul`
//!   / `Backend::dequantize` trait default pattern.
//!
//! Hardware resource methods (alloc_kv_cache, upload_weights, swap_*,
//! get_memory_pressure, get_page_states, sample_from_tensor) continue to use
//! `gpu_helpers::*` and the surviving JIT infrastructure in `gpu_compile`
//! (compile_graph_to_ptx, cuda_compile_graph, cuda_launch_graph and their
//! HIP / Metal counterparts).

/// Generate a complete `Backend<E>` implementation for a GPU backend.
///
/// # Parameters
/// - `$backend_ty`: The backend struct name (e.g. `CudaBackend`)
/// - `$cfg_pred`: The cfg predicate as `meta` (e.g. `feature = "cuda"`)
/// - `$feature_label`: A string literal for error messages (e.g. `"cuda"`)
/// - `$upload_err_variant`: The `BackendError` variant for upload errors
macro_rules! impl_gpu_backend {
    (
        backend = $backend_ty:ident,
        cfg_pred = [ $($cfg_pred:tt)+ ],
        feature_label = $feature_label:expr,
        upload_err = $upload_err_variant:ident $(,)?
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
                let _ = (input, topology, weights, kv_caches, config);
                Err(BE::Unimplemented(concat!(
                    $feature_label,
                    " decoder forward pending ARCH-CPU-GPU-UNIFIED migration (FusedGraphExecutor::run_gpu_with_kv_cache not implemented)"
                )))
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
                let _ = (tokens, weights, config);
                Err(BE::Unimplemented(concat!(
                    $feature_label,
                    " encoder forward pending ARCH-CPU-GPU-UNIFIED migration (FusedGraphExecutor::run_gpu host glue not implemented)"
                )))
            }

            fn rerank_forward_gpu_pure(
                &self,
                tokens: &[u32],
                _topology: &AttentionTopology,
                weights: &dyn backend_trait::TensorLookup<E, Self>,
                config: &GeneratorForwardConfig,
            ) -> Result<Vec<f32>, BE> {
                let _ = (tokens, weights, config);
                Err(BE::Unimplemented(concat!(
                    $feature_label,
                    " rerank forward pending ARCH-CPU-GPU-UNIFIED migration (FusedGraphExecutor::run_gpu host glue not implemented)"
                )))
            }

            fn classify_forward_gpu_pure(
                &self,
                tokens: &[u32],
                _topology: &AttentionTopology,
                weights: &dyn backend_trait::TensorLookup<E, Self>,
                config: &GeneratorForwardConfig,
            ) -> Result<Vec<f32>, BE> {
                let _ = (tokens, weights, config);
                Err(BE::Unimplemented(concat!(
                    $feature_label,
                    " classify forward pending ARCH-CPU-GPU-UNIFIED migration (FusedGraphExecutor::run_gpu host glue not implemented)"
                )))
            }

            fn score_tokens_forward_gpu_pure(
                &self,
                tokens: &[u32],
                target_token_ids: &[u32],
                _topology: &AttentionTopology,
                weights: &dyn backend_trait::TensorLookup<E, Self>,
                config: &GeneratorForwardConfig,
            ) -> Result<Vec<f32>, BE> {
                let _ = (tokens, target_token_ids, weights, config);
                Err(BE::Unimplemented(concat!(
                    $feature_label,
                    " score_tokens forward (Head Routing SDK) pending ARCH-CPU-GPU-UNIFIED migration (FusedGraphExecutor::run_gpu host glue not implemented)"
                )))
            }

            fn encode_at_layer_forward_gpu_pure(
                &self,
                tokens: &[u32],
                anchor_layer: usize,
                _topology: &AttentionTopology,
                weights: &dyn backend_trait::TensorLookup<E, Self>,
                config: &GeneratorForwardConfig,
            ) -> Result<Vec<f32>, BE> {
                let _ = (tokens, anchor_layer, weights, config);
                Err(BE::Unimplemented(concat!(
                    $feature_label,
                    " encode_at_layer forward (HR/Intent) pending ARCH-CPU-GPU-UNIFIED migration"
                )))
            }

            fn apply_guardrail_probe(
                &self,
                tokens: &[u32],
                _topology: &AttentionTopology,
                weights: &dyn backend_trait::TensorLookup<E, Self>,
                config: &GeneratorForwardConfig,
            ) -> Result<Vec<f32>, BE> {
                let _ = (tokens, weights, config);
                Err(BE::Unimplemented(concat!(
                    $feature_label,
                    " apply_guardrail_probe (Guardrail SDK) pending ARCH-CPU-GPU-UNIFIED migration"
                )))
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
