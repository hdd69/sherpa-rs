use crate::{
    get_default_provider,
    utils::{cstr_to_string, cstring_from_str},
};
use eyre::{bail, Result};
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

pub struct ZipFormerOnline {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOnlineRecognizer,
    stream: *const sherpa_rs_sys::SherpaOnnxOnlineStream,
}

impl ZipFormerOnline {
    pub fn new(config: ZipFormerOnlineConfig) -> Result<Self> {
        // Online Zipformer transducer config
        let decoder_ptr = cstring_from_str(&config.decoder);
        let encoder_ptr = cstring_from_str(&config.encoder);
        let joiner_ptr = cstring_from_str(&config.joiner);
        let provider_ptr = cstring_from_str(&config.provider.unwrap_or_else(get_default_provider));
        let tokens_ptr = cstring_from_str(&config.tokens);
        let decoding_method_ptr = cstring_from_str(
            &config
                .decoding_method
                .unwrap_or_else(|| "greedy_search".to_string()),
        );

        let transducer_config = unsafe {
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
        let feat_config = unsafe {
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
            bail!("Failed to create online recognizer");
        }

        let stream = unsafe { sherpa_rs_sys::SherpaOnnxCreateOnlineStream(recognizer) };
        if stream.is_null() {
            unsafe { sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizer(recognizer) };
            bail!("Failed to create online stream");
        }

        Ok(Self { recognizer, stream })
    }

    pub fn accept_waveform(&mut self, sample_rate: u32, samples: &[f32]) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamAcceptWaveform(
                self.stream,
                sample_rate as i32,
                samples.as_ptr(),
                samples.len() as i32,
            );
        }
    }

    pub fn decode(&mut self) -> String {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDecodeOnlineStream(self.recognizer, self.stream);
            let result_ptr =
                sherpa_rs_sys::SherpaOnnxGetOnlineStreamResult(self.recognizer, self.stream);
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

    pub fn input_finished(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamInputFinished(self.stream);
        }
    }

    pub fn reset(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamReset(self.recognizer, self.stream);
        }
    }
}

unsafe impl Send for ZipFormerOnline {}
unsafe impl Sync for ZipFormerOnline {}

impl Drop for ZipFormerOnline {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOnlineStream(self.stream);
            sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizer(self.recognizer);
        }
    }
}
