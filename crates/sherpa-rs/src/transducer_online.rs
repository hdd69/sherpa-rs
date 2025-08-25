use crate::utils::cstr_to_string;
use crate::{get_default_provider, utils::cstring_from_str};
use eyre::{bail, Result};
use std::mem;

pub struct OnlineTransducerRecognizer {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOnlineRecognizer,
    stream: *const sherpa_rs_sys::SherpaOnnxOnlineStream,
}

#[derive(Debug, Clone)]
pub struct OnlineTransducerConfig {
    pub decoder: String,
    pub encoder: String,
    pub joiner: String,
    pub tokens: String,
    pub num_threads: i32,
    pub sample_rate: i32,
    pub feature_dim: i32,
    pub decoding_method: String,
    pub hotwords_file: String,
    pub hotwords_score: f32,
    pub modeling_unit: String,
    pub bpe_vocab: String,
    pub blank_penalty: f32,
    pub model_type: String,
    pub debug: bool,
    pub provider: Option<String>,
    // Online-specific fields (optional, with defaults)
    pub enable_endpoint: bool,
    pub rule1_min_trailing_silence: f32,
    pub rule2_min_trailing_silence: f32,
    pub rule3_min_utterance_length: f32,
    pub max_active_paths: i32,
}

impl Default for OnlineTransducerConfig {
    fn default() -> Self {
        OnlineTransducerConfig {
            decoder: String::new(),
            encoder: String::new(),
            joiner: String::new(),
            tokens: String::new(),
            model_type: String::from("transducer"),
            num_threads: 1,
            sample_rate: 16000, // Default for most models like Parakeet
            feature_dim: 80,    // Default Mel dim
            decoding_method: String::from("greedy_search"),
            hotwords_file: String::new(),
            hotwords_score: 1.0,
            modeling_unit: String::new(),
            bpe_vocab: String::new(),
            blank_penalty: 0.0,
            debug: false,
            provider: None,
            // Online defaults
            enable_endpoint: true,
            rule1_min_trailing_silence: 2.4,
            rule2_min_trailing_silence: 1.2,
            rule3_min_utterance_length: 20.0,
            max_active_paths: 4,
        }
    }
}

impl OnlineTransducerRecognizer {
    pub fn new(config: OnlineTransducerConfig) -> Result<Self> {
        let recognizer = unsafe {
            let debug = config.debug.into();
            let provider = config.provider.unwrap_or(get_default_provider());
            let provider_ptr = cstring_from_str(&provider);

            let encoder = cstring_from_str(&config.encoder);
            let decoder = cstring_from_str(&config.decoder);
            let joiner = cstring_from_str(&config.joiner);
            let model_type = cstring_from_str(&config.model_type);
            let modeling_unit = cstring_from_str(&config.modeling_unit);
            let bpe_vocab = cstring_from_str(&config.bpe_vocab);
            let hotwords_file = cstring_from_str(&config.hotwords_file);
            let tokens = cstring_from_str(&config.tokens);
            let decoding_method = cstring_from_str(&config.decoding_method);

            let online_model_config = sherpa_rs_sys::SherpaOnnxOnlineModelConfig {
                transducer: sherpa_rs_sys::SherpaOnnxOnlineTransducerModelConfig {
                    encoder: encoder.as_ptr(),
                    decoder: decoder.as_ptr(),
                    joiner: joiner.as_ptr(),
                },
                tokens: tokens.as_ptr(),
                num_threads: config.num_threads,
                debug,
                provider: provider_ptr.as_ptr(),
                model_type: model_type.as_ptr(),
                modeling_unit: modeling_unit.as_ptr(),
                bpe_vocab: bpe_vocab.as_ptr(),
                tokens_buf: std::ptr::null(),
                tokens_buf_size: 0,
                // NULLs for other models (similar to offline)
                paraformer: mem::zeroed::<_>(),
                nemo_ctc: mem::zeroed::<_>(),
                zipformer2_ctc: mem::zeroed::<_>(),
                // Add any other unused fields as zeroed if present in bindings
            };

            let recognizer_config = sherpa_rs_sys::SherpaOnnxOnlineRecognizerConfig {
                feat_config: sherpa_rs_sys::SherpaOnnxFeatureConfig {
                    sample_rate: config.sample_rate,
                    feature_dim: config.feature_dim,
                    // Other feat fields default to zeroed or standard (e.g., low_freq=20, high_freq=-400 for Parakeet)
                    ..unsafe { mem::zeroed() } // Or set explicitly if needed
                },
                model_config: online_model_config,
                decoding_method: decoding_method.as_ptr(),
                max_active_paths: config.max_active_paths,
                enable_endpoint: config.enable_endpoint.into(),
                rule1_min_trailing_silence: config.rule1_min_trailing_silence,
                rule2_min_trailing_silence: config.rule2_min_trailing_silence,
                rule3_min_utterance_length: config.rule3_min_utterance_length,
                hotwords_file: hotwords_file.as_ptr(),
                hotwords_score: config.hotwords_score,
                // Other fields zeroed (e.g., lm_config, ctc_fst_decoder_config, etc.)
                // lm_config: mem::zeroed::<_>(),
                ctc_fst_decoder_config: mem::zeroed::<_>(),
                rule_fsts: std::ptr::null(),
                rule_fars: std::ptr::null(),
                blank_penalty: config.blank_penalty,
                hotwords_buf: std::ptr::null(),
                hotwords_buf_size: 0,
                // Add HR or other if present: mem::zeroed()
                hr: mem::zeroed::<_>(),
            };

            let recognizer = sherpa_rs_sys::SherpaOnnxCreateOnlineRecognizer(&recognizer_config);
            if recognizer.is_null() {
                bail!("SherpaOnnxCreateOnlineRecognizer failed");
            }
            recognizer
        };

        let stream = unsafe {
            let stream = sherpa_rs_sys::SherpaOnnxCreateOnlineStream(recognizer);
            if stream.is_null() {
                unsafe { sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizer(recognizer) };
                bail!("SherpaOnnxCreateOnlineStream failed");
            }
            stream
        };

        Ok(Self { recognizer, stream })
    }

    /// Feed a chunk of audio samples to the recognizer (call in a loop for streaming)
    pub fn accept_waveform(&mut self, sample_rate: u32, samples: &[f32]) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamAcceptWaveform(
                self.stream,
                sample_rate as i32,
                samples.as_ptr(),
                samples.len().try_into().unwrap(),
            );
        }
    }

    /// Decode the current stream state (call after accept_waveform)
    pub fn decode(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDecodeOnlineStream(self.recognizer, self.stream);
        }
    }

    /// Check if a partial result is ready
    pub fn is_ready(&self) -> bool {
        unsafe { sherpa_rs_sys::SherpaOnnxIsOnlineStreamReady(self.recognizer, self.stream) != 0 }
    }

    /// Get the current transcription result (partial or final; call while is_ready())
    pub fn get_result(&self) -> String {
        unsafe {
            let result_ptr =
                sherpa_rs_sys::SherpaOnnxGetOnlineStreamResult(self.recognizer, self.stream);
            if result_ptr.is_null() {
                return String::new();
            }
            let raw_result = *result_ptr;
            let text = cstr_to_string(raw_result.text as _);
            sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizerResult(result_ptr);
            text
        }
    }

    /// Check if an endpoint (end of utterance) is detected
    pub fn is_endpoint(&self) -> bool {
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamIsEndpoint(self.recognizer, self.stream) != 0
        }
    }

    /// Reset the stream for a new utterance (call after endpoint)
    pub fn reset(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamReset(self.recognizer, self.stream);
        }
    }

    /// Signal end of input and finalize any pending decoding (call at session end)
    pub fn input_finished(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamInputFinished(self.stream);
            // Decode remaining while ready
            while sherpa_rs_sys::SherpaOnnxIsOnlineStreamReady(self.recognizer, self.stream) != 0 {
                sherpa_rs_sys::SherpaOnnxDecodeOnlineStream(self.recognizer, self.stream);
            }
        }
    }
}

unsafe impl Send for OnlineTransducerRecognizer {}
unsafe impl Sync for OnlineTransducerRecognizer {}

impl Drop for OnlineTransducerRecognizer {
    fn drop(&mut self) {
        unsafe {
            // Optional: Call input_finished here if not called manually
            sherpa_rs_sys::SherpaOnnxDestroyOnlineStream(self.stream);
            sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizer(self.recognizer);
        }
    }
}
