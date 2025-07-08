use thiserror::Error;

/// Custom error types for TenVAD operations
#[derive(Error, Debug)]
pub enum TenVadError {
    #[error("Failed to run ONNX session: {0}")]
    OnnxRuntimeError(#[from] ort::Error),

    #[error("Invalid audio frame size: expected {expected}, got {actual}")]
    InvalidFrameSize { expected: usize, actual: usize },

    #[error("Invalid sample rate: {rate} Hz (supported rates: 8000, 16000, 22050, 44100, 48000)")]
    InvalidSampleRate { rate: u32 },

    #[error("Empty audio data provided")]
    EmptyAudioData,

    #[error("Model initialization failed: {message}")]
    ModelInitializationError { message: String },

    #[error("Feature extraction failed: {reason}")]
    FeatureExtractionError { reason: String },

    #[error("Audio preprocessing error: {0}")]
    AudioPreprocessingError(#[from] Box<dyn std::error::Error + Send + Sync>),
}

/// Type alias for TenVAD results
pub type TenVadResult<T> = Result<T, TenVadError>;
