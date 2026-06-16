//! Macro to generate the `Backend<E>` trait impl for GPU backends (CUDA/HIP/Metal).
//!
//! **ARCH-CPU-GPU-UNIFIED**: GPU backends launch the same mega-kernel PTX/HIP/MSL
//! code that was compiled by `MegaKernelExecutor`. The compilation product is shared;
//! only the execution layer differs (cuLaunchKernel vs direct CALL).

/// Generate a complete `Backend<E>` implementation for a GPU backend.
///
/// Each GPU backend struct must have these `pub(super)` fields/methods:
/// - `compiled_ptx: Mutex<HashMap<String, Vec<u8>>>` — GPU code cache
/// - `weight_blob_gpu: Mutex<Option<(u64, usize)>>` — cached weight blob ptr
/// - `upload_weight_blob(&[u8]) -> Result<u64, String>`
/// - `alloc_scratchpad_gpu(usize) -> Result<u64, String>`
/// - `upload_to_gpu<T: Copy>(&[T]) -> Result<u64, String>`
/// - `download_from_gpu(u64, usize) -> Result<Vec<u8>, String>`
/// - `get_cached_ptx(&str) -> Option<Vec<u8>>`
/// - `get_weight_gpu_ptr() -> Option<u64>`
/// - `get_cached_scratchpad_bytes() -> usize`
/// - `gpu_launch_mega_kernel(&[u8], &str, &[usize; 23]) -> Result<(), String>`
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

            // ------------------------------------------------------------------
            // GPU Mega-Kernel Artifact Management
            // ------------------------------------------------------------------

            fn prepare_gpu_mega_kernel(
                &self,
                weight_blob: &[u8],
                gpu_code: Option<&[u8]>,
                scratchpad_bytes: usize,
            ) -> Result<(), BE> {
                #[cfg( $($cfg_pred)+ )]
                {
                    let backend_f32 = unsafe {
                        &*(self as *const $backend_ty<E> as *const $backend_ty<f32>)
                    };

                    backend_f32.upload_weight_blob(weight_blob)
                        .map_err(|e| BE::$upload_err_variant(e))?;

                    if let Some(code) = gpu_code {
                        let mut cache = backend_f32.compiled_ptx.lock()
                            .map_err(|e| BE::$upload_err_variant(format!("lock poisoned: {e}")))?;
                        cache.insert("mega_kernel".to_string(), code.to_vec());
                    }

                    {
                        let mut cache = backend_f32.compiled_ptx.lock()
                            .map_err(|e| BE::$upload_err_variant(format!("lock poisoned: {e}")))?;
                        cache.insert("__scratchpad_bytes__".to_string(), scratchpad_bytes.to_le_bytes().to_vec());
                    }

                    Ok(())
                }
                #[cfg(not( $($cfg_pred)+ ))]
                {
                    let _ = (weight_blob, gpu_code, scratchpad_bytes);
                    Ok(())
                }
            }

            // ------------------------------------------------------------------
            // REQ-GPU-001: batch_forward_gpu_pure
            // ------------------------------------------------------------------

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
                    let _ = (topology, weights);
                    let bf = unsafe { &*(self as *const $backend_ty<E> as *const $backend_ty<f32>) };

                    let ptx = bf.get_cached_ptx("mega_kernel")
                        .ok_or_else(|| BE::$upload_err_variant("mega-kernel GPU code not cached".into()))?;
                    let weight_gpu = bf.get_weight_gpu_ptr()
                        .ok_or_else(|| BE::$upload_err_variant("weight blob not uploaded".into()))?;
                    let scratch_bytes = bf.get_cached_scratchpad_bytes();

                    let mut all_logits = Vec::with_capacity(input.sequences.len());
                    let vocab_size = config.geometry.vocab_size;

                    for (seq_idx, seq) in input.sequences.iter().enumerate() {
                        let seq_len = seq.tokens.len();
                        let ids_gpu = bf.upload_to_gpu(&seq.tokens)
                            .map_err(|e| BE::$upload_err_variant(e))?;
                        let positions: Vec<u32> = (seq.position as u32..(seq.position + seq_len) as u32).collect();
                        let pos_gpu = bf.upload_to_gpu(&positions)
                            .map_err(|e| BE::$upload_err_variant(e))?;
                        let sp_gpu = bf.alloc_scratchpad_gpu(scratch_bytes.max(1024))
                            .map_err(|e| BE::$upload_err_variant(e))?;
                        let out_bytes = vocab_size * 4;
                        let out_gpu = bf.alloc_scratchpad_gpu(out_bytes)
                            .map_err(|e| BE::$upload_err_variant(e))?;
                        let kv_ptr = kv_caches.get(seq_idx).map(|h| h.0).unwrap_or(0);
                        let page_table_gpu = match &seq.page_table {
                            Some(pt) => bf.upload_to_gpu(pt.as_slice())
                                .map_err(|e| BE::$upload_err_variant(e))?,
                            None => 0,
                        };

                        let args = super::gpu_helpers::build_mega_kernel_args(
                            ids_gpu, weight_gpu, kv_ptr, pos_gpu,
                            0, 1, seq_len, sp_gpu, out_gpu,
                            0, 0, 0, 1, 0, 0,
                            0, 0, seq.position, 0, 0, 0,
                            page_table_gpu,
                            0,
                        );
                        bf.gpu_launch_mega_kernel(&ptx, "mega_kernel", &args)
                            .map_err(|e| BE::$upload_err_variant(e))?;

                        let data = bf.download_from_gpu(out_gpu, out_bytes)
                            .map_err(|e| BE::$upload_err_variant(e))?;
                        all_logits.push(LogitsHandle {
                            data: super::gpu_helpers::bytes_to_f32_vec(&data),
                        });
                    }
                    Ok((all_logits, 0.0, Vec::new()))
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

            // ------------------------------------------------------------------
            // REQ-GPU-003: rerank_forward_gpu_pure
            // REQ-GPU-004: classify_forward_gpu_pure
            // (All use forward-only kernel; pooling is in the JIT graph)
            // ------------------------------------------------------------------

            fn rerank_forward_gpu_pure(
                &self,
                tokens: &[u32],
                _topology: &AttentionTopology,
                weights: &dyn backend_trait::TensorLookup<E, Self>,
                config: &GeneratorForwardConfig,
            ) -> Result<Vec<f32>, BE> {
                #[cfg( $($cfg_pred)+ )]
                {
                    let _ = weights;
                    let bf = unsafe { &*(self as *const $backend_ty<E> as *const $backend_ty<f32>) };
                    super::gpu_backend_macro::gpu_encoder_forward_impl(bf, tokens, config, 1)
                        .map_err(|e| BE::$upload_err_variant(e))
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
                    let _ = weights;
                    let bf = unsafe { &*(self as *const $backend_ty<E> as *const $backend_ty<f32>) };
                    super::gpu_backend_macro::gpu_encoder_forward_impl(bf, tokens, config, 2)
                        .map_err(|e| BE::$upload_err_variant(e))
                }
                #[cfg(not( $($cfg_pred)+ ))]
                {
                    let _ = (tokens, weights, config);
                    Err(BE::Unimplemented(concat!($feature_label, " feature not enabled")))
                }
            }

            // ------------------------------------------------------------------
            // REQ-GPU-005: score_tokens_forward_gpu_pure
            // ------------------------------------------------------------------

            fn score_tokens_forward_gpu_pure(
                &self,
                tokens: &[u32],
                target_token_ids: &[u32],
                _topology: &AttentionTopology,
                weights: &dyn backend_trait::TensorLookup<E, Self>,
                config: &GeneratorForwardConfig,
            ) -> Result<Vec<f32>, BE> {
                #[cfg( $($cfg_pred)+ )]
                {
                    let _ = weights;
                    if target_token_ids.is_empty() { return Ok(vec![]); }
                    let bf = unsafe { &*(self as *const $backend_ty<E> as *const $backend_ty<f32>) };

                    let ptx = bf.get_cached_ptx("mega_kernel")
                        .ok_or_else(|| BE::$upload_err_variant("mega-kernel GPU code not cached".into()))?;
                    let weight_gpu = bf.get_weight_gpu_ptr()
                        .ok_or_else(|| BE::$upload_err_variant("weight blob not uploaded".into()))?;
                    let scratch_bytes = bf.get_cached_scratchpad_bytes();
                    let vocab_size = config.geometry.vocab_size;

                    let seq_len = tokens.len();
                    let ids_gpu = bf.upload_to_gpu(tokens).map_err(|e| BE::$upload_err_variant(e))?;
                    let positions: Vec<u32> = (0..seq_len as u32).collect();
                    let pos_gpu = bf.upload_to_gpu(&positions).map_err(|e| BE::$upload_err_variant(e))?;
                    let sp_gpu = bf.alloc_scratchpad_gpu(scratch_bytes.max(1024))
                        .map_err(|e| BE::$upload_err_variant(e))?;
                    let logits_bytes = vocab_size * 4;
                    let out_gpu = bf.alloc_scratchpad_gpu(logits_bytes)
                        .map_err(|e| BE::$upload_err_variant(e))?;

                    let args = super::gpu_helpers::build_mega_kernel_args(
                        ids_gpu, weight_gpu, 0, pos_gpu,
                        0, 1, seq_len, sp_gpu, out_gpu,
                        0, 0, 0, 1, 0, 0,
                        0, 0, 0, 0, 0, 0,
                        0,
                        0,
                    );
                    bf.gpu_launch_mega_kernel(&ptx, "mega_kernel", &args)
                        .map_err(|e| BE::$upload_err_variant(e))?;

                    let data = bf.download_from_gpu(out_gpu, logits_bytes)
                        .map_err(|e| BE::$upload_err_variant(e))?;
                    let all_logits = super::gpu_helpers::bytes_to_f32_vec(&data);
                    let offset = all_logits.len().saturating_sub(vocab_size);
                    Ok(target_token_ids.iter()
                        .map(|&id| *all_logits.get(offset + id as usize).unwrap_or(&0.0))
                        .collect())
                }
                #[cfg(not( $($cfg_pred)+ ))]
                {
                    let _ = (tokens, target_token_ids, weights, config);
                    Err(BE::Unimplemented(concat!($feature_label, " feature not enabled")))
                }
            }

            // ------------------------------------------------------------------
            // REQ-GPU-006: encode_at_layer_forward_gpu_pure
            // ------------------------------------------------------------------

            fn encode_at_layer_forward_gpu_pure(
                &self,
                tokens: &[u32],
                _anchor_layer: usize,
                _topology: &AttentionTopology,
                weights: &dyn backend_trait::TensorLookup<E, Self>,
                config: &GeneratorForwardConfig,
            ) -> Result<Vec<f32>, BE> {
                #[cfg( $($cfg_pred)+ )]
                {
                    let _ = weights;
                    let bf = unsafe { &*(self as *const $backend_ty<E> as *const $backend_ty<f32>) };

                    let ptx = bf.get_cached_ptx("mega_kernel")
                        .ok_or_else(|| BE::$upload_err_variant("mega-kernel GPU code not cached".into()))?;
                    let weight_gpu = bf.get_weight_gpu_ptr()
                        .ok_or_else(|| BE::$upload_err_variant("weight blob not uploaded".into()))?;
                    let scratch_bytes = bf.get_cached_scratchpad_bytes();

                    let seq_len = tokens.len();
                    let hidden_size = config.geometry.hidden_size;
                    let ids_gpu = bf.upload_to_gpu(tokens).map_err(|e| BE::$upload_err_variant(e))?;
                    let positions: Vec<u32> = (0..seq_len as u32).collect();
                    let pos_gpu = bf.upload_to_gpu(&positions).map_err(|e| BE::$upload_err_variant(e))?;
                    let sp_gpu = bf.alloc_scratchpad_gpu(scratch_bytes.max(1024))
                        .map_err(|e| BE::$upload_err_variant(e))?;

                    let args = super::gpu_helpers::build_mega_kernel_args(
                        ids_gpu, weight_gpu, 0, pos_gpu,
                        0, 1, seq_len, sp_gpu, 0,
                        0, 0, 0, 1, 0,
                        0, 0, 0, 0, 0, 0,
                        0,
                        0,
                    );
                    bf.gpu_launch_mega_kernel(&ptx, "mega_kernel", &args)
                        .map_err(|e| BE::$upload_err_variant(e))?;

                    let out_bytes = (seq_len * hidden_size * 4).min(scratch_bytes);
                    let data = bf.download_from_gpu(sp_gpu, out_bytes)
                        .map_err(|e| BE::$upload_err_variant(e))?;
                    Ok(super::gpu_helpers::bytes_to_f32_vec(&data))
                }
                #[cfg(not( $($cfg_pred)+ ))]
                {
                    let _ = (tokens, _anchor_layer, weights, config);
                    Err(BE::Unimplemented(concat!($feature_label, " feature not enabled")))
                }
            }

            fn apply_guardrail_probe(
                &self,
                tokens: &[u32],
                _topology: &AttentionTopology,
                weights: &dyn backend_trait::TensorLookup<E, Self>,
                config: &GeneratorForwardConfig,
            ) -> Result<Vec<f32>, BE> {
                #[cfg( $($cfg_pred)+ )]
                {
                    let _ = weights;
                    let bf = unsafe { &*(self as *const $backend_ty<E> as *const $backend_ty<f32>) };

                    let ptx = bf.get_cached_ptx("mega_kernel")
                        .ok_or_else(|| BE::$upload_err_variant("mega-kernel GPU code not cached".into()))?;
                    let weight_gpu = bf.get_weight_gpu_ptr()
                        .ok_or_else(|| BE::$upload_err_variant("weight blob not uploaded".into()))?;
                    let scratch_bytes = bf.get_cached_scratchpad_bytes();

                    let seq_len = tokens.len();
                    let hidden_size = config.geometry.hidden_size;
                    let ids_gpu = bf.upload_to_gpu(tokens).map_err(|e| BE::$upload_err_variant(e))?;
                    let positions: Vec<u32> = (0..seq_len as u32).collect();
                    let pos_gpu = bf.upload_to_gpu(&positions).map_err(|e| BE::$upload_err_variant(e))?;
                    let sp_gpu = bf.alloc_scratchpad_gpu(scratch_bytes.max(1024))
                        .map_err(|e| BE::$upload_err_variant(e))?;

                    let args = super::gpu_helpers::build_mega_kernel_args(
                        ids_gpu, weight_gpu, 0, pos_gpu,
                        0, 1, seq_len, sp_gpu, 0,
                        0, 0, 0, 1, 0,
                        0, 0, 0, 0, 0, 0,
                        0,
                        0,
                    );
                    bf.gpu_launch_mega_kernel(&ptx, "mega_kernel", &args)
                        .map_err(|e| BE::$upload_err_variant(e))?;

                    let out_bytes = (seq_len * hidden_size * 4).min(scratch_bytes);
                    let data = bf.download_from_gpu(sp_gpu, out_bytes)
                        .map_err(|e| BE::$upload_err_variant(e))?;
                    Ok(super::gpu_helpers::bytes_to_f32_vec(&data))
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
                    if total == 0 { return Ok(0.0); }
                    Ok(1.0 - (self.device.free_memory() as f32 / total as f32))
                }
                #[cfg(not( $($cfg_pred)+ ))]
                { Ok(0.0) }
            }

            fn swap_out_pages(
                &self,
                handle: &mut KvCacheHandle,
                mappings: &[(PageId, StorageKey)],
            ) -> Result<(), BE> {
                #[cfg( $($cfg_pred)+ )]
                {
                    super::gpu_helpers::gpu_swap_out_pages(
                        unsafe { &*(self as *const $backend_ty<E> as *const $backend_ty<f32>) },
                        handle, mappings,
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
                        unsafe { &*(self as *const $backend_ty<E> as *const $backend_ty<f32>) },
                        handle, mappings,
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
                        unsafe { &*(self as *const $backend_ty<E> as *const $backend_ty<f32>) },
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

            fn upload_weights_with_placement(
                &self,
                data: Vec<f32>,
                placement: backend_trait::WeightPlacement,
            ) -> Result<(Self::Tensor, backend_trait::WeightPlacement), BE> {
                match placement {
                    backend_trait::WeightPlacement::DeviceLocal => {
                        #[cfg( $($cfg_pred)+ )]
                        {
                            use gllm_kernels::gpu::GpuDevice;
                            let bytes: &[u8] = unsafe {
                                std::slice::from_raw_parts(
                                    data.as_ptr() as *const u8,
                                    data.len() * std::mem::size_of::<f32>(),
                                )
                            };
                            let mut buf = self.device.alloc(bytes.len()).map_err(|e| {
                                BE::$upload_err_variant(format!("GPU alloc failed: {e}"))
                            })?;
                            let stream = self.device.default_stream();
                            self.device.htod(bytes, &mut buf, stream).map_err(|e| {
                                BE::$upload_err_variant(format!("GPU htod failed: {e}"))
                            })?;
                            Ok((data.iter().map(|&f| unsafe { std::mem::transmute_copy::<f32, E>(&f) }).collect(), backend_trait::WeightPlacement::DeviceLocal))
                        }
                        #[cfg(not( $($cfg_pred)+ ))]
                        {
                            let _ = data;
                            Err(BE::Unimplemented(concat!($feature_label, " feature not enabled")))
                        }
                    }
                    backend_trait::WeightPlacement::HostLocal => {
                        let tensor = { let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect(); self.upload_weights_owned(bytes, gllm_kernels::types::DType::F32)? };
                        Ok((tensor, backend_trait::WeightPlacement::HostLocal))
                    }
                }
            }

            fn device_memory_capacity(&self) -> usize {
                #[cfg( $($cfg_pred)+ )]
                { use gllm_kernels::gpu::GpuDevice; self.device.total_memory() }
                #[cfg(not( $($cfg_pred)+ ))]
                { 0 }
            }

            fn gpu_sm_version(&self) -> u32 {
                #[cfg( $($cfg_pred)+ )]
                { self.device_info().sm_version }
                #[cfg(not( $($cfg_pred)+ ))]
                { 0 }
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Encoder forward helper — DEPRECATED (SPEC/39)
// ---------------------------------------------------------------------------
// SPEC/39: encoder models now use the unified mega-kernel path via
// execute_encode(). This dead code remains until
// the Backend trait is cleaned up to remove rerank_forward_gpu_pure and
// classify_forward_gpu_pure.

/// DEPRECATED (SPEC/39): Use execute_encode() via mega-kernel instead.
/// Previously launched a forward-only kernel for encoder tasks using the
/// separate "forward_kernel" PTX. Now superseded by the unified mega-kernel.
#[allow(dead_code)]
pub(crate) fn gpu_encoder_forward_impl<B>(
    backend: &B,
    tokens: &[u32],
    config: &crate::engine::executor::GeneratorForwardConfig,
    output_elems: usize,
) -> Result<Vec<f32>, String>
where
    B: GpuEncoderOps,
{
    backend.gpu_encoder_forward(tokens, config, output_elems)
}

/// Trait for GPU encoder forward operations.
/// Each GPU backend implements this with its specific device API calls.
pub(super) trait GpuEncoderOps {
    fn gpu_encoder_forward(
        &self,
        tokens: &[u32],
        config: &crate::engine::executor::GeneratorForwardConfig,
        output_elems: usize,
    ) -> Result<Vec<f32>, String>;
}

#[cfg(feature = "cuda")]
impl GpuEncoderOps for super::cuda_backend::CudaBackend<f32> {
    fn gpu_encoder_forward(
        &self,
        tokens: &[u32],
        config: &crate::engine::executor::GeneratorForwardConfig,
        output_elems: usize,
    ) -> Result<Vec<f32>, String> {
        let ptx = self.get_cached_ptx("forward_kernel")
            .ok_or("forward_kernel GPU code not cached — encoder model not compiled with GPU support")?;
        let weight_gpu = self.get_weight_gpu_ptr()
            .ok_or("weight blob not uploaded")?;
        let scratch_bytes = self.get_cached_scratchpad_bytes();

        let seq_len = tokens.len();
        let ids_gpu = self.upload_to_gpu(tokens)?;
        let positions: Vec<u32> = (0..seq_len as u32).collect();
        let pos_gpu = self.upload_to_gpu(&positions)?;
        let sp_gpu = self.alloc_scratchpad_gpu(scratch_bytes.max(1024))?;
        let out_bytes = output_elems * 4;
        let out_gpu = self.alloc_scratchpad_gpu(out_bytes)?;

        let args = super::gpu_helpers::build_mega_kernel_args(
            ids_gpu, weight_gpu, 0, pos_gpu,
            0, 1, seq_len, sp_gpu, out_gpu,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0,
            0,
        );
        self.gpu_launch_mega_kernel(&ptx, "forward_kernel", &args)?;

        let data = self.download_from_gpu(out_gpu, out_bytes)?;
        Ok(super::gpu_helpers::bytes_to_f32_vec(&data))
    }
}

#[cfg(feature = "hip")]
impl GpuEncoderOps for super::hip_backend::HipBackend<f32> {
    fn gpu_encoder_forward(
        &self,
        tokens: &[u32],
        config: &crate::engine::executor::GeneratorForwardConfig,
        output_elems: usize,
    ) -> Result<Vec<f32>, String> {
        let ptx = self.get_cached_ptx("forward_kernel")
            .ok_or("forward_kernel GPU code not cached")?;
        let weight_gpu = self.get_weight_gpu_ptr()
            .ok_or("weight blob not uploaded")?;
        let scratch_bytes = self.get_cached_scratchpad_bytes();

        let seq_len = tokens.len();
        let ids_gpu = self.upload_to_gpu(tokens)?;
        let positions: Vec<u32> = (0..seq_len as u32).collect();
        let pos_gpu = self.upload_to_gpu(&positions)?;
        let sp_gpu = self.alloc_scratchpad_gpu(scratch_bytes.max(1024))?;
        let out_bytes = output_elems * 4;
        let out_gpu = self.alloc_scratchpad_gpu(out_bytes)?;

        let args = super::gpu_helpers::build_mega_kernel_args(
            ids_gpu, weight_gpu, 0, pos_gpu,
            0, 1, seq_len, sp_gpu, out_gpu,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0,
            0,
        );
        self.gpu_launch_mega_kernel(&ptx, "forward_kernel", &args)?;

        let data = self.download_from_gpu(out_gpu, out_bytes)?;
        Ok(super::gpu_helpers::bytes_to_f32_vec(&data))
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl GpuEncoderOps for super::metal_backend::MetalBackend<f32> {
    fn gpu_encoder_forward(
        &self,
        tokens: &[u32],
        config: &crate::engine::executor::GeneratorForwardConfig,
        output_elems: usize,
    ) -> Result<Vec<f32>, String> {
        let ptx = self.get_cached_ptx("forward_kernel")
            .ok_or("forward_kernel GPU code not cached")?;
        let weight_gpu = self.get_weight_gpu_ptr()
            .ok_or("weight blob not uploaded")?;
        let scratch_bytes = self.get_cached_scratchpad_bytes();

        let seq_len = tokens.len();
        let ids_gpu = self.upload_to_gpu(tokens)?;
        let positions: Vec<u32> = (0..seq_len as u32).collect();
        let pos_gpu = self.upload_to_gpu(&positions)?;
        let sp_gpu = self.alloc_scratchpad_gpu(scratch_bytes.max(1024))?;
        let out_bytes = output_elems * 4;
        let out_gpu = self.alloc_scratchpad_gpu(out_bytes)?;

        let args = super::gpu_helpers::build_mega_kernel_args(
            ids_gpu, weight_gpu, 0, pos_gpu,
            0, 1, seq_len, sp_gpu, out_gpu,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0,
            0,
        );
        self.gpu_launch_mega_kernel(&ptx, "forward_kernel", &args)?;

        let data = self.download_from_gpu(out_gpu, out_bytes)?;
        Ok(super::gpu_helpers::bytes_to_f32_vec(&data))
    }
}
