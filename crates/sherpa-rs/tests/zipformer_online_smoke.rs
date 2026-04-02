use sherpa_rs::zipformer_online::{ZipFormerOnline, ZipFormerOnlineConfig};

fn required_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("missing required env var {name}"))
}

#[test]
#[ignore = "requires local sherpa-onnx libraries and model files"]
fn creates_online_zipformer_recognizer_with_local_models() {
    let config = ZipFormerOnlineConfig {
        encoder: required_env("SHERPA_RS_TEST_ENCODER"),
        decoder: required_env("SHERPA_RS_TEST_DECODER"),
        joiner: required_env("SHERPA_RS_TEST_JOINER"),
        tokens: required_env("SHERPA_RS_TEST_TOKENS"),
        sample_rate: Some(16000),
        decoding_method: Some("modified_beam_search".to_string()),
        max_active_paths: Some(4),
        enable_endpoint: Some(1),
        rule1_min_trailing_silence: Some(1000.0),
        rule2_min_trailing_silence: Some(0.8),
        rule3_min_utterance_length: Some(0.0),
        num_threads: Some(4),
        debug: false,
        ..Default::default()
    };

    let _recognizer = ZipFormerOnline::new(config)
        .expect("ZipFormerOnline::new should succeed with the local 1.12.34 runtime");
}
