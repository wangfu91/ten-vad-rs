/// Custom error types for TenVAD operations
#[derive(Debug, Clone, PartialEq)]
pub enum TenVadError {
    /// Invalid parameter provided
    InvalidParameter(String),
    /// Audio data size mismatch
    AudioSizeMismatch { expected: usize, actual: usize },
    /// Native library error
    NativeError(i32),
    /// Resource allocation failure
    AllocationError,
    /// Invalid threshold value
    InvalidThreshold(f32),
    /// Invalid hop size
    InvalidHopSize(usize),
}

impl std::fmt::Display for TenVadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TenVadError::InvalidParameter(msg) => write!(f, "Invalid parameter: {msg}"),
            TenVadError::AudioSizeMismatch { expected, actual } => {
                write!(f, "Audio size mismatch: expected {expected}, got {actual}")
            }
            TenVadError::NativeError(code) => write!(f, "Native library error: {code}"),
            TenVadError::AllocationError => write!(f, "Failed to allocate resources"),
            TenVadError::InvalidThreshold(threshold) => {
                write!(
                    f,
                    "Invalid threshold {threshold}: must be between 0.0 and 1.0"
                )
            }
            TenVadError::InvalidHopSize(hop_size) => {
                write!(f, "Invalid hop size {hop_size}: must be greater than 0")
            }
        }
    }
}

impl std::error::Error for TenVadError {}

// Conversion from TenVadError to String for backwards compatibility
impl From<TenVadError> for String {
    fn from(error: TenVadError) -> Self {
        error.to_string()
    }
}

/// Type alias for TenVAD results
pub type TenVadResult<T> = Result<T, TenVadError>;
