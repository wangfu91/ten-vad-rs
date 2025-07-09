use thiserror::Error;

/// Custom error types for TenVAD operations
#[derive(Error, Debug)]
pub enum TenVadError {
    #[error("Failed to run ONNX session: {0}")]
    OnnxRuntimeError(#[from] ort::Error),

    #[error("Empty audio data provided")]
    EmptyAudioData,
}

/// Type alias for TenVAD results
pub type TenVadResult<T> = Result<T, TenVadError>;
