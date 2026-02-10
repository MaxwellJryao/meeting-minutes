// audio/transcription/qwen_asr_provider.rs
//
// Qwen3-ASR transcription provider implementation.

use super::provider::{TranscriptionError, TranscriptionProvider, TranscriptResult};
use async_trait::async_trait;
use log::warn;
use std::sync::Arc;

/// Qwen3-ASR transcription provider (wraps QwenAsrEngine)
pub struct QwenAsrProvider {
    engine: Arc<crate::qwen_asr_engine::QwenAsrEngine>,
}

impl QwenAsrProvider {
    pub fn new(engine: Arc<crate::qwen_asr_engine::QwenAsrEngine>) -> Self {
        Self { engine }
    }
}

#[async_trait]
impl TranscriptionProvider for QwenAsrProvider {
    async fn transcribe(
        &self,
        audio: Vec<f32>,
        language: Option<String>,
    ) -> std::result::Result<TranscriptResult, TranscriptionError> {
        // Qwen3-ASR supports multilingual transcription natively
        if let Some(ref lang) = language {
            log::debug!("Qwen3-ASR transcribing with language hint: {}", lang);
        }

        match self.engine.transcribe_audio(audio).await {
            Ok(text) => Ok(TranscriptResult {
                text: text.trim().to_string(),
                confidence: None, // Qwen3-ASR doesn't provide confidence scores
                is_partial: false,
            }),
            Err(e) => Err(TranscriptionError::EngineFailed(e.to_string())),
        }
    }

    async fn is_model_loaded(&self) -> bool {
        self.engine.is_model_loaded().await
    }

    async fn get_current_model(&self) -> Option<String> {
        self.engine.get_current_model().await
    }

    fn provider_name(&self) -> &'static str {
        "QwenASR"
    }
}
