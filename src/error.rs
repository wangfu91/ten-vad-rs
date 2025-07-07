use thiserror::Error;

/// Custom error types for TenVAD operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum TenVadError {
    /// Audio data size mismatch
    #[error("Audio size mismatch: expected {expected}, got {actual}")]
    AudioSizeMismatch { expected: usize, actual: usize },

    /// Native library error
    #[error("Native library error: {0}")]
    NativeError(i32),

    /// Resource allocation failure
    #[error("Failed to allocate resources")]
    AllocationError,

    /// Invalid threshold value
    #[error("Invalid threshold {0}: must be between 0.0 and 1.0")]
    InvalidThreshold(f32),

    /// Invalid hop size
    #[error("Invalid hop size {0}: must be greater than 0")]
    InvalidHopSize(usize),
}

// Conversion from TenVadError to String for backwards compatibility
impl From<TenVadError> for String {
    fn from(error: TenVadError) -> Self {
        error.to_string()
    }
}

/// Type alias for TenVAD results
pub type TenVadResult<T> = Result<T, TenVadError>;
