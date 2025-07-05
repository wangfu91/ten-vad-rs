use crate::bindings::{
    ten_vad_create, ten_vad_destroy, ten_vad_get_version, ten_vad_handle_t, ten_vad_process,
};

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

#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(non_upper_case_globals)]
#[allow(dead_code)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

/// Result of VAD processing for a single frame
#[derive(Debug, Clone, PartialEq)]
pub struct VadResult {
    /// Voice activity probability in the range [0.0, 1.0]
    pub probability: f32,
    /// Binary voice activity decision (true: voice detected, false: no voice)
    pub is_voice: bool,
}

/// Ten VAD (Voice Activity Detection) wrapper
#[derive(Debug)]
pub struct TenVAD {
    ten_vad_handle: ten_vad_handle_t,
    hop_size: usize,
    threshold: f32,
}

impl TenVAD {
    /// Create a new TenVAD instance
    ///
    /// # Arguments
    /// * `hop_size` - The number of samples between the start points of two consecutive analysis frames (e.g., 256)
    /// * `threshold` - VAD detection threshold ranging from [0.0, 1.0]. When probability >= threshold, voice is detected.
    ///
    /// # Returns
    /// Returns `Ok(TenVAD)` on success, or `Err(TenVadError)` on failure.
    pub fn new(hop_size: usize, threshold: f32) -> TenVadResult<Self> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(TenVadError::InvalidThreshold(threshold));
        }

        if hop_size == 0 {
            return Err(TenVadError::InvalidHopSize(hop_size));
        }

        let mut ten_vad_handle: ten_vad_handle_t = std::ptr::null_mut();
        let result = unsafe { ten_vad_create(&mut ten_vad_handle, hop_size, threshold) };
        if result != 0 {
            return Err(TenVadError::NativeError(result));
        }

        if ten_vad_handle.is_null() {
            return Err(TenVadError::AllocationError);
        }

        Ok(TenVAD {
            ten_vad_handle,
            hop_size,
            threshold,
        })
    }

    /// Create a new TenVAD instance with a preset configuration
    ///
    /// # Arguments
    /// * `preset` - A function returning (hop_size, threshold) configuration
    ///
    /// # Returns
    /// Returns `Ok(TenVAD)` on success, or `Err(TenVadError)` on failure.
    ///
    /// # Examples
    /// ```ignore
    /// # use ten_vad_rs::{TenVAD, VadPresets};
    /// let vad = TenVAD::with_preset(VadPresets::low_latency).unwrap();
    /// let vad = TenVAD::with_preset(VadPresets::high_accuracy).unwrap();
    /// ```
    pub fn with_preset<F>(preset: F) -> TenVadResult<Self>
    where
        F: FnOnce() -> (usize, f32),
    {
        let (hop_size, threshold) = preset();
        Self::new(hop_size, threshold)
    }

    /// Process a single frame of audio data
    ///
    /// # Arguments
    /// * `audio_data` - Audio samples as i16 PCM data. Length must equal hop_size.
    ///
    /// # Returns
    /// Returns `Ok(VadResult)` with probability and voice detection result, or `Err(TenVadError)` on failure.
    pub fn process_frame(&self, audio_data: &[i16]) -> TenVadResult<VadResult> {
        if audio_data.len() != self.hop_size {
            return Err(TenVadError::AudioSizeMismatch {
                expected: self.hop_size,
                actual: audio_data.len(),
            });
        }

        let mut out_probability: f32 = 0.0;
        let mut out_flag: i32 = 0;

        let result = unsafe {
            ten_vad_process(
                self.ten_vad_handle,
                audio_data.as_ptr(),
                audio_data.len(),
                &mut out_probability,
                &mut out_flag,
            )
        };

        if result != 0 {
            return Err(TenVadError::NativeError(result));
        }

        Ok(VadResult {
            probability: out_probability,
            is_voice: out_flag != 0,
        })
    }

    /// Process multiple frames of audio data
    ///
    /// # Arguments
    /// * `audio_data` - Audio samples as i16 PCM data. Length must be a multiple of hop_size.
    ///
    /// # Returns
    /// Returns `Ok(Vec<VadResult>)` with results for each frame, or `Err(TenVadError)` on failure.
    pub fn process_frames(&self, audio_data: &[i16]) -> TenVadResult<Vec<VadResult>> {
        if audio_data.len() % self.hop_size != 0 {
            return Err(TenVadError::AudioSizeMismatch {
                expected: audio_data.len() - (audio_data.len() % self.hop_size),
                actual: audio_data.len(),
            });
        }

        let frame_count = audio_data.len() / self.hop_size;
        let mut results = Vec::with_capacity(frame_count);

        for i in 0..frame_count {
            let start_idx = i * self.hop_size;
            let end_idx = start_idx + self.hop_size;
            let frame_data = &audio_data[start_idx..end_idx];

            let result = self.process_frame(frame_data)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Process multiple frames with parallel execution (requires 'parallel' feature)
    ///
    /// # Arguments
    /// * `audio_data` - Audio samples as i16 PCM data. Length must be a multiple of hop_size.
    ///
    /// # Returns
    /// Returns `Ok(Vec<VadResult>)` with results for each frame, or `Err(TenVadError)` on failure.
    #[cfg(feature = "parallel")]
    pub fn process_frames_parallel(&self, audio_data: &[i16]) -> TenVadResult<Vec<VadResult>> {
        use rayon::prelude::*;

        if audio_data.len() % self.hop_size != 0 {
            return Err(TenVadError::AudioSizeMismatch {
                expected: audio_data.len() - (audio_data.len() % self.hop_size),
                actual: audio_data.len(),
            });
        }

        let frame_count = audio_data.len() / self.hop_size;
        let hop_size = self.hop_size;

        // Create separate VAD instances for parallel processing
        let vad_instances: TenVadResult<Vec<_>> = (0..rayon::current_num_threads())
            .map(|_| TenVAD::new(hop_size, self.threshold))
            .collect();

        let vad_instances = vad_instances?;

        // Process frames in parallel
        let results: TenVadResult<Vec<_>> = (0..frame_count)
            .into_par_iter()
            .map(|i| {
                let start_idx = i * hop_size;
                let end_idx = start_idx + hop_size;
                let frame_data = &audio_data[start_idx..end_idx];

                let vad = &vad_instances
                    [rayon::current_thread_index().unwrap_or(0) % vad_instances.len()];
                vad.process_frame(frame_data)
            })
            .collect();

        results
    }

    /// Process frames with a streaming callback for memory efficiency
    ///
    /// # Arguments
    /// * `audio_data` - Audio samples as i16 PCM data. Length must be a multiple of hop_size.
    /// * `callback` - Function called for each processed frame with (frame_index, VadResult)
    ///
    /// # Returns
    /// Returns `Ok(())` on success, or `Err(TenVadError)` on failure.
    pub fn process_frames_streaming<F>(
        &self,
        audio_data: &[i16],
        mut callback: F,
    ) -> TenVadResult<()>
    where
        F: FnMut(usize, VadResult) -> bool, // Return false to stop processing
    {
        if audio_data.len() % self.hop_size != 0 {
            return Err(TenVadError::AudioSizeMismatch {
                expected: audio_data.len() - (audio_data.len() % self.hop_size),
                actual: audio_data.len(),
            });
        }

        let frame_count = audio_data.len() / self.hop_size;

        for i in 0..frame_count {
            let start_idx = i * self.hop_size;
            let end_idx = start_idx + self.hop_size;
            let frame_data = &audio_data[start_idx..end_idx];

            let result = self.process_frame(frame_data)?;

            if !callback(i, result) {
                break; // Early termination requested
            }
        }

        Ok(())
    }

    /// Process frames and return only frames where voice was detected
    ///
    /// # Arguments
    /// * `audio_data` - Audio samples as i16 PCM data. Length must be a multiple of hop_size.
    ///
    /// # Returns
    /// Returns `Ok(Vec<(usize, VadResult)>)` with frame indices and results for voice frames only.
    pub fn process_frames_voice_only(
        &self,
        audio_data: &[i16],
    ) -> TenVadResult<Vec<(usize, VadResult)>> {
        if audio_data.len() % self.hop_size != 0 {
            return Err(TenVadError::AudioSizeMismatch {
                expected: audio_data.len() - (audio_data.len() % self.hop_size),
                actual: audio_data.len(),
            });
        }

        let frame_count = audio_data.len() / self.hop_size;
        let mut voice_frames = Vec::new();

        for i in 0..frame_count {
            let start_idx = i * self.hop_size;
            let end_idx = start_idx + self.hop_size;
            let frame_data = &audio_data[start_idx..end_idx];

            let result = self.process_frame(frame_data)?;

            if result.is_voice {
                voice_frames.push((i, result));
            }
        }

        Ok(voice_frames)
    }

    /// Get the hop size used by this VAD instance
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    /// Get the threshold used by this VAD instance
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Get the ten_vad library version string
    pub fn get_version() -> String {
        unsafe {
            let version_ptr = ten_vad_get_version();
            if version_ptr.is_null() {
                return "Unknown version".to_string();
            }
            let version_cstr = std::ffi::CStr::from_ptr(version_ptr);
            version_cstr.to_string_lossy().into_owned()
        }
    }
}

impl Drop for TenVAD {
    fn drop(&mut self) {
        unsafe {
            let result = ten_vad_destroy(&mut self.ten_vad_handle);
            if result != 0 {
                eprintln!("Warning: Failed to destroy TenVAD handle: error code {result}");
            }
        }
    }
}

/// Configuration presets for different use cases
pub struct VadPresets;

impl VadPresets {
    /// Low-latency preset for real-time applications
    /// - Small hop size (128 samples = 8ms frames)
    /// - Moderate sensitivity
    pub fn low_latency() -> (usize, f32) {
        (128, 0.5)
    }

    /// Balanced preset for general use
    /// - Standard hop size (256 samples = 16ms frames)
    /// - Moderate sensitivity
    pub fn balanced() -> (usize, f32) {
        (256, 0.5)
    }

    /// High-accuracy preset for offline processing
    /// - Large hop size (512 samples = 32ms frames)
    /// - Lower sensitivity for fewer false positives
    pub fn high_accuracy() -> (usize, f32) {
        (512, 0.3)
    }

    /// Sensitive preset for quiet speech
    /// - Standard hop size
    /// - High sensitivity
    pub fn sensitive() -> (usize, f32) {
        (256, 0.2)
    }

    /// Conservative preset for noisy environments
    /// - Standard hop size
    /// - Low sensitivity to reduce false positives
    pub fn conservative() -> (usize, f32) {
        (256, 0.8)
    }

    /// Battery-efficient preset for mobile devices
    /// - Large hop size for fewer computations
    /// - Moderate sensitivity
    pub fn battery_efficient() -> (usize, f32) {
        (512, 0.5)
    }
}

unsafe impl Send for TenVAD {}
unsafe impl Sync for TenVAD {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_version() {
        let version = TenVAD::get_version();
        assert!(!version.is_empty());
        println!("TenVAD version: {version}");
    }

    #[test]
    fn test_create_vad() {
        let vad = TenVAD::new(256, 0.5);
        assert!(vad.is_ok());

        let vad = vad.unwrap();
        assert_eq!(vad.hop_size(), 256);
        assert_eq!(vad.threshold(), 0.5);
    }

    #[test]
    fn test_create_vad_invalid_threshold() {
        let vad = TenVAD::new(256, -0.1);
        assert!(vad.is_err());
        assert!(matches!(vad.unwrap_err(), TenVadError::InvalidThreshold(_)));

        let vad = TenVAD::new(256, 1.1);
        assert!(vad.is_err());
        assert!(matches!(vad.unwrap_err(), TenVadError::InvalidThreshold(_)));
    }

    #[test]
    fn test_create_vad_invalid_hop_size() {
        let vad = TenVAD::new(0, 0.5);
        assert!(vad.is_err());
        assert!(matches!(vad.unwrap_err(), TenVadError::InvalidHopSize(_)));
    }

    #[test]
    fn test_process_frame() {
        let vad = TenVAD::new(256, 0.5).unwrap();

        // Create dummy audio data (silence)
        let audio_data = vec![0i16; 256];
        let result = vad.process_frame(&audio_data);
        assert!(result.is_ok());

        let vad_result = result.unwrap();
        assert!(vad_result.probability >= 0.0 && vad_result.probability <= 1.0);
        println!(
            "Silence frame - Probability: {}, Is voice: {}",
            vad_result.probability, vad_result.is_voice
        );
    }

    #[test]
    fn test_process_frame_wrong_size() {
        let vad = TenVAD::new(256, 0.5).unwrap();

        // Create audio data with wrong size
        let audio_data = vec![0i16; 128]; // Wrong size
        let result = vad.process_frame(&audio_data);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TenVadError::AudioSizeMismatch { .. }
        ));
    }

    #[test]
    fn test_process_frames() {
        let vad = TenVAD::new(256, 0.5).unwrap();

        // Create dummy audio data for 3 frames
        let audio_data = vec![0i16; 256 * 3];
        let results = vad.process_frames(&audio_data);
        assert!(results.is_ok());

        let vad_results = results.unwrap();
        assert_eq!(vad_results.len(), 3);

        for (i, result) in vad_results.iter().enumerate() {
            assert!(result.probability >= 0.0 && result.probability <= 1.0);
            println!(
                "Frame {} - Probability: {}, Is voice: {}",
                i, result.probability, result.is_voice
            );
        }
    }

    #[test]
    fn test_process_frames_wrong_size() {
        let vad = TenVAD::new(256, 0.5).unwrap();

        // Create audio data with wrong size (not multiple of hop_size)
        let audio_data = vec![0i16; 256 * 2 + 128]; // Wrong size
        let result = vad.process_frames(&audio_data);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TenVadError::AudioSizeMismatch { .. }
        ));
    }

    #[test]
    fn test_process_synthetic_audio() {
        let vad = TenVAD::new(256, 0.5).unwrap();

        // Create synthetic audio data with some signal
        let mut audio_data = vec![0i16; 256];

        // Add some sine wave to simulate voice-like signal
        for (i, sample) in audio_data.iter_mut().enumerate() {
            let sine_value = (1000.0 * (i as f32 * 0.1).sin()) as i16;
            *sample = sine_value;
        }

        let result = vad.process_frame(&audio_data);
        assert!(result.is_ok());

        let vad_result = result.unwrap();
        assert!(vad_result.probability >= 0.0 && vad_result.probability <= 1.0);
        println!(
            "Synthetic audio - Probability: {}, Is voice: {}",
            vad_result.probability, vad_result.is_voice
        );
    }

    #[test]
    fn test_different_thresholds() {
        let thresholds = [0.1, 0.3, 0.5, 0.7, 0.9];

        for threshold in thresholds {
            let vad = TenVAD::new(256, threshold).unwrap();
            assert_eq!(vad.threshold(), threshold);

            // Test with some audio data
            let audio_data = vec![100i16; 256]; // Some non-zero audio
            let result = vad.process_frame(&audio_data);
            assert!(result.is_ok());

            let vad_result = result.unwrap();
            println!(
                "Threshold {}: Probability: {}, Is voice: {}",
                threshold, vad_result.probability, vad_result.is_voice
            );
        }
    }

    #[test]
    fn test_different_hop_sizes() {
        let hop_sizes = [128, 256, 512, 1024];

        for hop_size in hop_sizes {
            let vad = TenVAD::new(hop_size, 0.5).unwrap();
            assert_eq!(vad.hop_size(), hop_size);

            // Test with matching audio data size
            let audio_data = vec![0i16; hop_size];
            let result = vad.process_frame(&audio_data);
            assert!(result.is_ok());

            let vad_result = result.unwrap();
            println!(
                "Hop size {}: Probability: {}, Is voice: {}",
                hop_size, vad_result.probability, vad_result.is_voice
            );
        }
    }

    #[test]
    fn test_vad_result_debug() {
        let result = VadResult {
            probability: 0.75,
            is_voice: true,
        };

        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("0.75"));
        assert!(debug_str.contains("true"));
    }

    #[test]
    fn test_multiple_instances() {
        let vad1 = TenVAD::new(256, 0.3).unwrap();
        let vad2 = TenVAD::new(512, 0.7).unwrap();

        let audio_data1 = vec![0i16; 256];
        let audio_data2 = vec![0i16; 512];

        let result1 = vad1.process_frame(&audio_data1);
        let result2 = vad2.process_frame(&audio_data2);

        assert!(result1.is_ok());
        assert!(result2.is_ok());

        println!("VAD1 result: {:?}", result1.unwrap());
        println!("VAD2 result: {:?}", result2.unwrap());
    }

    #[test]
    fn test_presets() {
        // Test all presets can be created successfully
        let vad = TenVAD::with_preset(VadPresets::low_latency);
        assert!(vad.is_ok());
        let vad = vad.unwrap();
        assert_eq!(vad.hop_size(), 128);
        assert_eq!(vad.threshold(), 0.5);

        let vad = TenVAD::with_preset(VadPresets::balanced).unwrap();
        assert_eq!(vad.hop_size(), 256);
        assert_eq!(vad.threshold(), 0.5);

        let vad = TenVAD::with_preset(VadPresets::high_accuracy).unwrap();
        assert_eq!(vad.hop_size(), 512);
        assert_eq!(vad.threshold(), 0.3);

        let vad = TenVAD::with_preset(VadPresets::sensitive).unwrap();
        assert_eq!(vad.hop_size(), 256);
        assert_eq!(vad.threshold(), 0.2);

        let vad = TenVAD::with_preset(VadPresets::conservative).unwrap();
        assert_eq!(vad.hop_size(), 256);
        assert_eq!(vad.threshold(), 0.8);

        let vad = TenVAD::with_preset(VadPresets::battery_efficient).unwrap();
        assert_eq!(vad.hop_size(), 512);
        assert_eq!(vad.threshold(), 0.5);
    }

    #[test]
    fn test_voice_only_processing() {
        let vad = TenVAD::new(256, 0.3).unwrap();

        // Create audio with some voice-like patterns
        let mut audio_data = vec![0i16; 256 * 10]; // 10 frames

        // Add some "voice-like" signal to middle frames
        for i in 2..5 {
            let start = i * 256;
            let end = start + 256;
            for (j, sample) in audio_data[start..end].iter_mut().enumerate() {
                *sample = (((start + j) as f32 * 0.1).sin() * 1000.0) as i16;
            }
        }

        let voice_frames = vad.process_frames_voice_only(&audio_data).unwrap();

        // Should have detected some voice frames
        assert!(!voice_frames.is_empty());

        // All returned frames should have is_voice = true
        for (_, result) in &voice_frames {
            assert!(result.is_voice);
        }
    }

    #[test]
    fn test_streaming_processing() {
        let vad = TenVAD::new(256, 0.5).unwrap();
        let audio_data = vec![0i16; 256 * 5]; // 5 frames

        let mut frame_count = 0;
        let mut voice_count = 0;

        let result = vad.process_frames_streaming(&audio_data, |frame_idx, result| {
            frame_count += 1;
            if result.is_voice {
                voice_count += 1;
            }
            assert_eq!(frame_idx, frame_count - 1);
            true // Continue processing
        });

        assert!(result.is_ok());
        assert_eq!(frame_count, 5);
    }

    #[test]
    fn test_streaming_early_termination() {
        let vad = TenVAD::new(256, 0.5).unwrap();
        let audio_data = vec![0i16; 256 * 10]; // 10 frames

        let mut frame_count = 0;

        let result = vad.process_frames_streaming(&audio_data, |_frame_idx, _result| {
            frame_count += 1;
            frame_count < 3 // Stop after 3 frames
        });

        assert!(result.is_ok());
        assert_eq!(frame_count, 3);
    }

    #[test]
    fn test_error_types() {
        // Test invalid threshold
        let result = TenVAD::new(256, -0.1);
        assert!(matches!(result, Err(TenVadError::InvalidThreshold(_))));

        let result = TenVAD::new(256, 1.5);
        assert!(matches!(result, Err(TenVadError::InvalidThreshold(_))));

        // Test invalid hop size
        let result = TenVAD::new(0, 0.5);
        assert!(matches!(result, Err(TenVadError::InvalidHopSize(_))));

        // Test audio size mismatch
        let vad = TenVAD::new(256, 0.5).unwrap();
        let wrong_size_audio = vec![0i16; 128];
        let result = vad.process_frame(&wrong_size_audio);
        assert!(matches!(result, Err(TenVadError::AudioSizeMismatch { .. })));
    }
}
