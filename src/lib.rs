use crate::bindings::{ten_vad_create, ten_vad_handle_t};

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
    /// Returns `Ok(TenVAD)` on success, or `Err(String)` with error message on failure.
    pub fn new(hop_size: usize, threshold: f32) -> Result<Self, String> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err("Threshold must be between 0.0 and 1.0".into());
        }

        if hop_size == 0 {
            return Err("Hop size must be greater than 0".into());
        }

        let mut ten_vad_handle: ten_vad_handle_t = std::ptr::null_mut();
        let result = unsafe { ten_vad_create(&mut ten_vad_handle, hop_size, threshold) };
        if result != 0 {
            return Err(format!("Failed to create TenVAD: error code {result}"));
        }

        if ten_vad_handle.is_null() {
            return Err("Failed to create TenVAD: null handle returned".into());
        }

        Ok(TenVAD {
            ten_vad_handle,
            hop_size,
            threshold,
        })
    }

    /// Process a single frame of audio data
    ///
    /// # Arguments
    /// * `audio_data` - Audio samples as i16 PCM data. Length must equal hop_size.
    ///
    /// # Returns
    /// Returns `Ok(VadResult)` with probability and voice detection result, or `Err(String)` on failure.
    pub fn process_frame(&self, audio_data: &[i16]) -> Result<VadResult, String> {
        if audio_data.len() != self.hop_size {
            return Err(format!(
                "Audio data length {} does not match hop_size {}",
                audio_data.len(),
                self.hop_size
            ));
        }

        let mut out_probability: f32 = 0.0;
        let mut out_flag: i32 = 0;

        let result = unsafe {
            bindings::ten_vad_process(
                self.ten_vad_handle,
                audio_data.as_ptr(),
                audio_data.len(),
                &mut out_probability,
                &mut out_flag,
            )
        };

        if result != 0 {
            return Err(format!("Failed to process audio data: error code {result}"));
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
    /// Returns `Ok(Vec<VadResult>)` with results for each frame, or `Err(String)` on failure.
    pub fn process_frames(&self, audio_data: &[i16]) -> Result<Vec<VadResult>, String> {
        if audio_data.len() % self.hop_size != 0 {
            return Err(format!(
                "Audio data length {} is not a multiple of hop_size {}",
                audio_data.len(),
                self.hop_size
            ));
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
            let version_ptr = bindings::ten_vad_get_version();
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
            let result = bindings::ten_vad_destroy(&mut self.ten_vad_handle);
            if result != 0 {
                eprintln!("Warning: Failed to destroy TenVAD handle: error code {result}");
            }
        }
    }
}

// Thread safety: TenVAD is Send + Sync because the underlying C library should be thread-safe
// and we don't share mutable state between threads (each instance has its own handle)
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
        assert!(
            vad.unwrap_err()
                .contains("Threshold must be between 0.0 and 1.0")
        );

        let vad = TenVAD::new(256, 1.1);
        assert!(vad.is_err());
        assert!(
            vad.unwrap_err()
                .contains("Threshold must be between 0.0 and 1.0")
        );
    }

    #[test]
    fn test_create_vad_invalid_hop_size() {
        let vad = TenVAD::new(0, 0.5);
        assert!(vad.is_err());
        assert!(vad.unwrap_err().contains("Hop size must be greater than 0"));
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
        assert!(result.unwrap_err().contains("does not match hop_size"));
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
        assert!(
            result
                .unwrap_err()
                .contains("is not a multiple of hop_size")
        );
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
}
