use crate::{
    get_default_provider,
    utils::{cstr_to_string, cstring_from_str},
};
use eyre::Result;
use std::ffi::CStr;
use std::mem;

#[derive(Debug, Default)]
pub struct ZipFormerOnlineConfig {
    pub decoder: String,
    pub encoder: String,
    pub joiner: String,
    pub tokens: String,

    pub num_threads: Option<i32>,
    pub provider: Option<String>,
    pub debug: bool,

    // For online, explicitly set feature config (defaults to 16000 Hz, 80 dim if zeroed, but set for clarity)
    pub sample_rate: Option<i32>,
    pub feature_dim: Option<i32>,

    // Decoding method (e.g., "greedy_search" for streaming)
    pub decoding_method: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum StreamingError {
    #[error("Decoding failed with code: {0}")]
    DecodingFailed(i32),

    #[error("Stream not ready for processing")]
    StreamNotReady,

    #[error("Invalid stream state")]
    InvalidState,

    #[error("Model configuration error")]
    ConfigError,
}

#[derive(Debug, Clone)]
pub struct StreamingState {
    pub session_active: bool,
    pub total_samples_processed: usize,
    pub last_partial_result: String,
    pub endpoint_detected: bool,
    pub session_start_time: std::time::Instant,
}

pub struct ZipFormerOnline {
    recognizer_ptr: *mut sherpa_rs_sys::SherpaOnnxOnlineRecognizer,
    // stream_ptr: *mut sherpa_rs_sys::SherpaOnnxOnlineStream,
}

impl ZipFormerOnline {
    pub fn new(config: ZipFormerOnlineConfig) -> Result<Self, StreamingError> {
        // Online Zipformer transducer config
        let decoder_ptr = cstring_from_str(&config.decoder);
        let encoder_ptr = cstring_from_str(&config.encoder);
        let joiner_ptr = cstring_from_str(&config.joiner);
        let provider_ptr =
            cstring_from_str(&config.provider.clone().unwrap_or_else(get_default_provider));
        let tokens_ptr = cstring_from_str(&config.tokens);
        let decoding_method_ptr = cstring_from_str(
            &config
                .decoding_method
                .clone()
                .unwrap_or_else(|| "greedy_search".to_string()),
        );

        let transducer_config = {
            sherpa_rs_sys::SherpaOnnxOnlineTransducerModelConfig {
                encoder: encoder_ptr.as_ptr(),
                decoder: decoder_ptr.as_ptr(),
                joiner: joiner_ptr.as_ptr(),
            }
        };

        // Online model config
        let model_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineModelConfig {
                transducer: transducer_config,
                tokens: tokens_ptr.as_ptr(),
                num_threads: config.num_threads.unwrap_or(1),
                debug: config.debug.into(),
                provider: provider_ptr.as_ptr(),
                // Zero other fields (paraformer, etc.)
                paraformer: mem::zeroed(),
                zipformer2_ctc: mem::zeroed(),
                model_type: mem::zeroed(),
                modeling_unit: mem::zeroed(),
                bpe_vocab: mem::zeroed(),
                tokens_buf: mem::zeroed(),
                tokens_buf_size: mem::zeroed(),
                nemo_ctc: mem::zeroed(),
            }
        };

        // Feature config (set defaults if not provided)
        let feat_config = {
            sherpa_rs_sys::SherpaOnnxFeatureConfig {
                sample_rate: config.sample_rate.unwrap_or(16000),
                feature_dim: config.feature_dim.unwrap_or(80),
            }
        };

        // Online recognizer config
        let recognizer_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineRecognizerConfig {
                feat_config,
                model_config,
                decoding_method: decoding_method_ptr.as_ptr(),
                // Zero other fields (endpoint, hotwords, etc.)
                enable_endpoint: mem::zeroed(),
                hotwords_file: mem::zeroed(),
                hotwords_score: mem::zeroed(),
                ctc_fst_decoder_config: mem::zeroed(),
                rule_fsts: mem::zeroed(),
                rule_fars: mem::zeroed(),
                blank_penalty: mem::zeroed(),
                hotwords_buf: mem::zeroed(),
                hotwords_buf_size: mem::zeroed(),
                hr: mem::zeroed(),
                max_active_paths: mem::zeroed(),
                rule1_min_trailing_silence: mem::zeroed(),
                rule2_min_trailing_silence: mem::zeroed(),
                rule3_min_utterance_length: mem::zeroed(),
            }
        };

        let recognizer =
            unsafe { sherpa_rs_sys::SherpaOnnxCreateOnlineRecognizer(&recognizer_config) };
        if recognizer.is_null() {
            return Err(StreamingError::ConfigError);
        }

        let stream = unsafe { sherpa_rs_sys::SherpaOnnxCreateOnlineStream(recognizer) };
        if stream.is_null() {
            unsafe { sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizer(recognizer) };
            return Err(StreamingError::ConfigError);
        }

        Ok(Self {
            recognizer_ptr: recognizer as *mut _,
        })
    }

    pub fn accept_waveform(
        &mut self,
        stream: *const sherpa_rs_sys::SherpaOnnxOnlineStream,
        sample_rate: u32,
        samples: &[f32],
    ) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamAcceptWaveform(
                stream,
                sample_rate as i32,
                samples.as_ptr(),
                samples.len() as i32,
            );
        }
    }

    pub fn decode(&mut self, stream: *const sherpa_rs_sys::SherpaOnnxOnlineStream) -> String {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDecodeOnlineStream(self.recognizer_ptr, stream);
            let result_ptr =
                sherpa_rs_sys::SherpaOnnxGetOnlineStreamResult(self.recognizer_ptr, stream);
            if result_ptr.is_null() {
                return String::new();
            }
            let raw_result = *result_ptr; // Assuming deref to get the struct
            let text = cstr_to_string(raw_result.text as *const _);

            // Note: Do not destroy the result here if it's a pointer to internal; check API. In C API, it's a copy, so destroy.
            sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizerResult(result_ptr);
            text
        }
    }

    pub fn input_finished(&mut self, stream: *const sherpa_rs_sys::SherpaOnnxOnlineStream) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamInputFinished(stream);
        }
    }

    pub fn reset(&mut self, stream: *const sherpa_rs_sys::SherpaOnnxOnlineStream) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamReset(self.recognizer_ptr, stream);
        }
    }

    // Streaming-specific methods

    /// Check if the streaming recognizer is ready to process audio
    /// This replaces manual chunk size checking with model-driven processing
    pub fn is_ready(&self, stream: *const sherpa_rs_sys::SherpaOnnxOnlineStream) -> bool {
        unsafe { sherpa_rs_sys::SherpaOnnxIsOnlineStreamReady(self.recognizer_ptr, stream) != 0 }
    }

    /// Get current recognition result
    /// Returns empty string if no result available
    pub fn get_result(&self, stream: *const sherpa_rs_sys::SherpaOnnxOnlineStream) -> String {
        unsafe {
            let result_ptr =
                sherpa_rs_sys::SherpaOnnxGetOnlineStreamResult(self.recognizer_ptr, stream);
            if result_ptr.is_null() {
                return String::new();
            }

            // Convert C string to Rust string
            let raw_result = *result_ptr;
            let text = if !raw_result.text.is_null() {
                CStr::from_ptr(raw_result.text)
                    .to_string_lossy()
                    .into_owned()
            } else {
                String::new()
            };

            // Free the result (if required by sherpa-onnx API)
            sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizerResult(result_ptr);

            text
        }
    }

    /// Check if endpoint (end of utterance) has been detected
    pub fn is_endpoint(&self, stream: *const sherpa_rs_sys::SherpaOnnxOnlineStream) -> bool {
        unsafe { sherpa_rs_sys::SherpaOnnxOnlineStreamIsEndpoint(self.recognizer_ptr, stream) != 0 }
    }

    /// Decode the current streaming audio
    /// This is called when is_ready() returns true
    pub fn decode_stream(
        &mut self,
        stream: *const sherpa_rs_sys::SherpaOnnxOnlineStream,
    ) -> Result<(), StreamingError> {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDecodeOnlineStream(self.recognizer_ptr, stream);

            Ok(())
        }
    }

    pub fn create_stream(&mut self) -> *const sherpa_rs_sys::SherpaOnnxOnlineStream {
        unsafe { sherpa_rs_sys::SherpaOnnxCreateOnlineStream(self.recognizer_ptr) }
    }

    pub fn destroy_stream(&mut self, stream: *const sherpa_rs_sys::SherpaOnnxOnlineStream) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOnlineStream(stream);
        }
    }
}

unsafe impl Send for ZipFormerOnline {}
unsafe impl Sync for ZipFormerOnline {}

impl Drop for ZipFormerOnline {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizer(self.recognizer_ptr);
        }
    }
}
