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
}

/// Type alias for TenVAD results
pub type TenVadResult<T> = Result<T, TenVadError>;
