# model-knowledge GAP REPORT

- gllm FIELD_DEFS entries: 51
- gllm unique json_keys: 91
- gllm unique gguf_keys: 28
- gllm ARCH_TABLE canonicals: 36

## [MISSING GGUF KEYS]

llama.cpp GGUF keys (non-arch-specific) not covered by FIELD_DEFS.gguf_keys.

**156 keys missing:**

- `adapter.alora.invocation_tokens`
- `adapter.lora.alpha`
- `adapter.lora.prompt_prefix`
- `adapter.lora.task_name`
- `adapter.type`
- `clip.audio.attention.head_count`
- `clip.audio.attention.layer_norm_epsilon`
- `clip.audio.block_count`
- `clip.audio.chunk_size`
- `clip.audio.conv_kernel_size`
- `clip.audio.embedding_length`
- `clip.audio.feed_forward_length`
- `clip.audio.max_pos_emb`
- `clip.audio.num_mel_bins`
- `clip.audio.projection_dim`
- `clip.audio.projector.downsample_rate`
- `clip.audio.projector.head_count`
- `clip.audio.projector.stack_factor`
- `clip.audio.projector.window_size`
- `clip.audio.projector_type`
- `clip.has_audio_encoder`
- `clip.has_llava_projector`
- `clip.has_vision_encoder`
- `clip.projector_type`
- `clip.use_gelu`
- `clip.use_silu`
- `clip.vision.attention.head_count`
- `clip.vision.attention.layer_norm_epsilon`
- `clip.vision.block_count`
- `clip.vision.embedding_length`
- `clip.vision.feed_forward_length`
- `clip.vision.image_max_pixels`
- `clip.vision.image_mean`
- `clip.vision.image_min_pixels`
- `clip.vision.image_size`
- `clip.vision.image_std`
- `clip.vision.is_deepstack_layers`
- `clip.vision.n_wa_pattern`
- `clip.vision.patch_size`
- `clip.vision.preproc_image_size`
- `clip.vision.preproc_max_tiles`
- `clip.vision.preproc_min_tiles`
- `clip.vision.projection_dim`
- `clip.vision.projector.scale_factor`
- `clip.vision.projector_type`
- `clip.vision.sam.block_count`
- `clip.vision.sam.embedding_length`
- `clip.vision.sam.head_count`
- `clip.vision.spatial_merge_size`
- `clip.vision.wa_layer_indexes`
- `clip.vision.window_size`
- `diffusion.shift_logits`
- `general.alignment`
- `general.architecture`
- `general.author`
- `general.base_model.count`
- `general.base_model.{id}.author`
- `general.base_model.{id}.description`
- `general.base_model.{id}.doi`
- `general.base_model.{id}.name`
- `general.base_model.{id}.organization`
- `general.base_model.{id}.repo_url`
- `general.base_model.{id}.url`
- `general.base_model.{id}.uuid`
- `general.base_model.{id}.version`
- `general.basename`
- `general.dataset.count`
- `general.dataset.{id}.author`
- `general.dataset.{id}.description`
- `general.dataset.{id}.doi`
- `general.dataset.{id}.name`
- `general.dataset.{id}.organization`
- `general.dataset.{id}.repo_url`
- `general.dataset.{id}.url`
- `general.dataset.{id}.uuid`
- `general.dataset.{id}.version`
- `general.description`
- `general.doi`
- `general.file_type`
- `general.finetune`
- `general.languages`
- `general.license`
- `general.license.link`
- `general.license.name`
- `general.name`
- `general.organization`
- `general.quantization_version`
- `general.quantized_by`
- `general.repo_url`
- `general.sampling.min_p`
- `general.sampling.mirostat`
- `general.sampling.mirostat_eta`
- `general.sampling.mirostat_tau`
- `general.sampling.penalty_last_n`
- `general.sampling.penalty_repeat`
- `general.sampling.sequence`
- `general.sampling.temp`
- `general.sampling.top_k`
- `general.sampling.top_p`
- `general.sampling.xtc_probability`
- `general.sampling.xtc_threshold`
- `general.size_label`
- `general.source.doi`
- `general.source.repo_url`
- `general.source.url`
- `general.source.uuid`
- `general.tags`
- `general.type`
- `general.url`
- `general.uuid`
- `general.version`
- `imatrix.chunk_count`
- `imatrix.chunk_size`
- `imatrix.datasets`
- `split.count`
- `split.no`
- `split.tensors.count`
- `tokenizer.chat_template`
- `tokenizer.chat_template.{name}`
- `tokenizer.chat_templates`
- `tokenizer.ggml.add_bos_token`
- `tokenizer.ggml.add_eos_token`
- `tokenizer.ggml.add_sep_token`
- `tokenizer.ggml.add_space_prefix`
- `tokenizer.ggml.bos_token_id`
- `tokenizer.ggml.eom_token_id`
- `tokenizer.ggml.eos_token_id`
- `tokenizer.ggml.eot_token_id`
- `tokenizer.ggml.fim_mid_token_id`
- `tokenizer.ggml.fim_pad_token_id`
- `tokenizer.ggml.fim_pre_token_id`
- `tokenizer.ggml.fim_rep_token_id`
- `tokenizer.ggml.fim_sep_token_id`
- `tokenizer.ggml.fim_suf_token_id`
- `tokenizer.ggml.mask_token_id`
- `tokenizer.ggml.merges`
- `tokenizer.ggml.middle_token_id`
- `tokenizer.ggml.model`
- `tokenizer.ggml.padding_token_id`
- `tokenizer.ggml.pre`
- `tokenizer.ggml.precompiled_charsmap`
- `tokenizer.ggml.prefix_token_id`
- `tokenizer.ggml.remove_extra_whitespaces`
- `tokenizer.ggml.scores`
- `tokenizer.ggml.seperator_token_id`
- `tokenizer.ggml.suffix_token_id`
- `tokenizer.ggml.token_type`
- `tokenizer.ggml.token_type_count`
- `tokenizer.ggml.tokens`
- `tokenizer.ggml.unknown_token_id`
- `tokenizer.huggingface.json`
- `tokenizer.rwkv.world`
- `xielu.alpha_n`
- `xielu.alpha_p`
- `xielu.beta`
- `xielu.eps`

## [MISSING ARCHITECTURES]

llama.cpp MODEL_ARCH names not in gllm ARCH_TABLE (canonical set).

**119 architectures missing:**

- `AFMOE`
- `APERTUS`
- `ARCEE`
- `ARCTIC`
- `ARWKV7`
- `BAICHUAN`
- `BAILINGMOE`
- `BAILINGMOE2`
- `BERT`
- `BITNET`
- `BLOOM`
- `CHAMELEON`
- `CHATGLM`
- `CODESHELL`
- `COGVLM`
- `COHERE2`
- `COMMAND_R`
- `DBRX`
- `DECI`
- `DEEPSEEK2`
- `DEEPSEEK2OCR`
- `DOTS1`
- `DREAM`
- `ERNIE4_5`
- `ERNIE4_5_MOE`
- `EUROBERT`
- `EXAONE`
- `EXAONE4`
- `EXAONE_MOE`
- `FALCON`
- `FALCON_H1`
- `GEMMA`
- `GEMMA2`
- `GEMMA3`
- `GEMMA3N`
- `GEMMA_EMBEDDING`
- `GLM4_MOE`
- `GLM_DSA`
- `GPT2`
- `GPTJ`
- `GPTNEOX`
- `GPT_OSS`
- `GRANITE`
- `GRANITE_HYBRID`
- `GRANITE_MOE`
- `GROK`
- `GROVEMOE`
- `HUNYUAN_DENSE`
- `HUNYUAN_MOE`
- `HUNYUAN_VL`
- `INTERNLM2`
- `JAIS`
- `JAIS2`
- `JAMBA`
- `JINA_BERT_V2`
- `JINA_BERT_V3`
- `KIMI_LINEAR`
- `LFM2`
- `LFM2MOE`
- `LLADA`
- `LLADA_MOE`
- `LLAMA_EMBED`
- `MAINCODER`
- `MAMBA`
- `MAMBA2`
- `MIMO2`
- `MINICPM`
- `MINICPM3`
- `MINIMAXM2`
- `MISTRAL4`
- `MMPROJ`
- `MODERN_BERT`
- `MPT`
- `NEMOTRON`
- `NEMOTRON_H`
- `NEMOTRON_H_MOE`
- `NEO_BERT`
- `NOMIC_BERT`
- `NOMIC_BERT_MOE`
- `OLMO`
- `OLMO2`
- `OLMOE`
- `OPENELM`
- `ORION`
- `PADDLEOCR`
- `PANGU_EMBED`
- `PHI2`
- `PHI3`
- `PHIMOE`
- `PLAMO`
- `PLAMO2`
- `PLAMO3`
- `PLM`
- `QWEN`
- `QWEN2`
- `QWEN2MOE`
- `QWEN2VL`
- `QWEN35`
- `QWEN35MOE`
- `QWEN3MOE`
- `QWEN3NEXT`
- `QWEN3VL`
- `QWEN3VLMOE`
- `REFACT`
- `RND1`
- `RWKV6`
- `RWKV6QWEN2`
- `RWKV7`
- `SEED_OSS`
- `SMALLTHINKER`
- `SMOLLM3`
- `STABLELM`
- `STARCODER`
- `STARCODER2`
- `STEP35`
- `T5`
- `T5ENCODER`
- `WAVTOKENIZER_DEC`
- `XVERSE`

## [HF ALIASES NOT COVERED]

transformers attribute_map aliases not in gllm FIELD_DEFS.json_keys.

**49 aliases not covered:**

| model_type | alias | canonical |
|------------|-------|-----------|
| bigbird_pegasus | `attention_probs_dropout_prob` | `attention_dropout` |
| led | `attention_probs_dropout_prob` | `attention_dropout` |
| flaubert | `bos_index` | `bos_token_id` |
| xlm | `bos_index` | `bos_token_id` |
| csm | `codebook_size` | `vocab_size` |
| csm_depth_decoder_model | `codebook_size` | `vocab_size` |
| pix2struct_text_model | `decoder_attention_heads` | `num_heads` |
| pix2struct_text_model | `decoder_layers` | `num_layers` |
| audioflamingo3_encoder | `encoder_attention_heads` | `num_attention_heads` |
| voxtral_encoder | `encoder_attention_heads` | `num_attention_heads` |
| voxtral_realtime_encoder | `encoder_attention_heads` | `num_attention_heads` |
| pix2struct_text_model | `encoder_attention_heads` | `num_heads` |
| audioflamingo3_encoder | `encoder_ffn_dim` | `intermediate_size` |
| voxtral_encoder | `encoder_ffn_dim` | `intermediate_size` |
| voxtral_realtime_encoder | `encoder_ffn_dim` | `intermediate_size` |
| omdet-turbo | `encoder_hidden_dim` | `d_model` |
| audioflamingo3_encoder | `encoder_layerdrop` | `layerdrop` |
| voxtral_encoder | `encoder_layerdrop` | `layerdrop` |
| voxtral_realtime_encoder | `encoder_layerdrop` | `layerdrop` |
| audioflamingo3_encoder | `encoder_layers` | `num_hidden_layers` |
| voxtral_encoder | `encoder_layers` | `num_hidden_layers` |
| voxtral_realtime_encoder | `encoder_layers` | `num_hidden_layers` |
| pix2struct_text_model | `encoder_layers` | `num_layers` |
| flaubert | `eos_index` | `eos_token_id` |
| xlm | `eos_index` | `eos_token_id` |
| informer | `initializer_range` | `init_std` |
| led | `initializer_range` | `init_std` |
| plbart | `initializer_range` | `init_std` |
| bamba | `layer_types` | `layers_block_type` |
| falcon_h1 | `layer_types` | `layers_block_type` |
| nemotron_h | `layer_types` | `layers_block_type` |
| zamba | `layer_types` | `layers_block_type` |
| zamba2 | `layer_types` | `layers_block_type` |
| granitemoehybrid | `layers_block_type` | `layer_types` |
| xlnet | `n_token` | `vocab_size` |
| flaubert | `n_words` | `vocab_size` |
| xlm | `n_words` | `vocab_size` |
| granite_speech_encoder | `num_mel_bins` | `input_dim` |
| granite_speech_plus_encoder | `num_mel_bins` | `input_dim` |
| flaubert | `pad_index` | `pad_token_id` |
| xlm | `pad_index` | `pad_token_id` |
| instructblipvideo | `video_token_id` | `video_token_index` |
| llava_next_video | `video_token_id` | `video_token_index` |
| llava_onevision | `video_token_id` | `video_token_index` |
| minimax_m3_vl | `video_token_id` | `video_token_index` |
| qwen2_5_omni_talker | `video_token_id` | `video_token_index` |
| qwen2_5_omni_thinker | `video_token_id` | `video_token_index` |
| video_llava | `video_token_id` | `video_token_index` |
|  | `window_size` | `block_size` |

## [SEMANTIC MAPPING COVERAGE]

llama.cpp get_key(KV, hparams.X) — X covered by a gllm canonical field?

| KV enum | hparams field | gllm canonical match |
|---------|---------------|----------------------|
| LLM_KV_ATTENTION_CAUSAL | causal_attn | no |
| LLM_KV_ATTENTION_KEY_LENGTH | n_embd_head_k_full | no |
| LLM_KV_ATTENTION_KEY_LENGTH_SWA | n_embd_head_k_swa | no |
| LLM_KV_ATTENTION_VALUE_LENGTH | n_embd_head_v_full | no |
| LLM_KV_ATTENTION_VALUE_LENGTH_SWA | n_embd_head_v_swa | no |
| LLM_KV_BLOCK_COUNT | n_layer_all | no |
| LLM_KV_CONTEXT_LENGTH | n_ctx_train | no |
| LLM_KV_CONVNEXT_BLOCK_COUNT | n_layer | no |
| LLM_KV_CONVNEXT_EMBEDDING_LENGTH | n_embd | no |
| LLM_KV_EMBEDDING_LENGTH | n_embd | no |
| LLM_KV_EMBEDDING_LENGTH | n_embd_out_impl | no |
| LLM_KV_EMBEDDING_LENGTH_OUT | n_embd_out_impl | no |
| LLM_KV_EXPERT_COUNT | n_expert | no |
| LLM_KV_EXPERT_GROUP_COUNT | n_expert_groups | no |
| LLM_KV_EXPERT_GROUP_USED_COUNT | n_group_used | no |
| LLM_KV_EXPERT_USED_COUNT | n_expert_used | no |
| LLM_KV_FEATURES_LENGTH | n_embd | no |
| LLM_KV_GENERAL_NAME | name | no |
| LLM_KV_POOLING_TYPE | pooling_type | no |
| LLM_KV_POSNET_BLOCK_COUNT | n_layer | no |
| LLM_KV_POSNET_EMBEDDING_LENGTH | n_embd | no |
| LLM_KV_ROPE_DIMENSION_COUNT | n_rot_full | no |
| LLM_KV_ROPE_DIMENSION_COUNT_SWA | n_rot_swa | no |
| LLM_KV_ROPE_FREQ_BASE | rope_freq_base_train | no |
| LLM_KV_ROPE_SCALE_LINEAR | ropescale | no |
| LLM_KV_ROPE_SCALING_ALPHA | rope_scaling_alpha | no |
| LLM_KV_ROPE_SCALING_ATTN_FACTOR | rope_attn_factor | no |
| LLM_KV_ROPE_SCALING_FACTOR | ropescale | no |
| LLM_KV_ROPE_SCALING_FINETUNED | rope_finetuned | no |
| LLM_KV_ROPE_SCALING_ORIG_CTX_LEN | n_ctx_orig_yarn | no |
| LLM_KV_ROPE_SCALING_TYPE | rope_scaling | yes |

