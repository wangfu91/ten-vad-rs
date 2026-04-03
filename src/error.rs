use ort::session::builder::SessionBuilder;
use thiserror::Error;

/// Custom error types for TenVAD operations
#[derive(Error, Debug)]
pub enum TenVadError {
    #[error("Failed to run ONNX session: {0}")]
    OnnxRuntimeError(#[from] ort::Error),

    #[error("Empty audio data provided")]
    EmptyAudioData,

    #[error("Unsupported sample rate: {0}Hz. TEN VAD only supports 16kHz")]
    UnsupportedSampleRate(u32),

    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
}

/// Implement conversion from ort::Error to TenVadError
impl From<ort::Error<SessionBuilder>> for TenVadError {
    fn from(value: ort::Error<SessionBuilder>) -> Self {
        Self::OnnxRuntimeError(value.into())
    }
}

/// Type alias for TenVAD results
pub type TenVadResult<T> = Result<T, TenVadError>;
