use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

/// Resample mono i16 audio samples from `input_sample_rate` to `output_sample_rate`.
/// # Arguments:
/// - `mono_i16_samples`: Input audio samples in i16 format.
/// - `input_sample_rate`: Sample rate of the input audio.
/// - `output_sample_rate`: Desired sample rate for the output audio.
/// # Returns:
/// - `Ok(Vec<i16>)`: Resampled audio samples in i16 format.
/// - `Err(Box<dyn std::error::Error>)`: If resampling fails, returns an error boxed trait object.
pub fn resampling(
    mono_i16_samples: &[i16],
    input_sample_rate: u32,
    output_sample_rate: u32,
) -> Result<Vec<i16>, Box<dyn std::error::Error>> {
    if input_sample_rate == output_sample_rate {
        return Ok(mono_i16_samples.to_vec());
    }

    // Convert i16 to f32 for resampling
    let f32_samples: Vec<f32> = mono_i16_samples
        .iter()
        .map(|&s| s as f32 / i16::MAX as f32)
        .collect();

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        output_sample_rate as f64 / input_sample_rate as f64,
        2.0, // max_relative_ratio
        params,
        f32_samples.len(),
        1, // channels
    )?;

    // Perform resampling
    let output_f32_samples = resampler.process(&[f32_samples], None)?;

    // Convert back to i16
    let output_i16_samples: Vec<i16> = output_f32_samples[0]
        .iter()
        .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
        .collect();

    Ok(output_i16_samples)
}

/// Convert stereo audio samples to mono by averaging the channels.
/// # Arguments:
/// - `data`: Input audio samples in i16 format.
/// - `channels`: Number of channels in the input audio (1 for mono, 2 for stereo).
/// # Returns:
/// - `Vec<i16>`: Mono audio samples in i16 format.
pub fn convert_to_mono(data: &[i16], channels: u32) -> Vec<i16> {
    if channels == 1 {
        data.to_vec()
    } else {
        data.chunks(channels as usize)
            .map(|frame| {
                let sum: i32 = frame.iter().map(|&s| s as i32).sum();
                let divisor = if frame.len() < channels as usize {
                    frame.len() as i32
                } else {
                    channels as i32
                };
                (sum / divisor) as i16
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_to_mono_already_mono() {
        let mono_data = vec![100i16, 200, 300, 400];
        let result = convert_to_mono(&mono_data, 1);
        assert_eq!(result, mono_data);
    }

    #[test]
    fn test_convert_to_mono_stereo() {
        let stereo_data = vec![100i16, 200, 300, 400, 500, 600];
        let result = convert_to_mono(&stereo_data, 2);
        // Expected: [(100+200)/2, (300+400)/2, (500+600)/2] = [150, 350, 550]
        assert_eq!(result, vec![150, 350, 550]);
    }

    #[test]
    fn test_convert_to_mono_empty() {
        let empty_data: Vec<i16> = vec![];
        let result = convert_to_mono(&empty_data, 2);
        assert_eq!(result, vec![]);
    }

    #[test]
    fn test_convert_to_mono_single_sample_stereo() {
        let data = vec![100i16, 200];
        let result = convert_to_mono(&data, 2);
        assert_eq!(result, vec![150]);
    }

    #[test]
    fn test_convert_to_mono_incomplete_frame() {
        // 5 samples with 2 channels means last frame is incomplete
        let data = vec![100i16, 200, 300, 400, 500];
        let result = convert_to_mono(&data, 2);
        // Expected: [(100+200)/2, (300+400)/2, 500/1] = [150, 350, 500]
        assert_eq!(result, vec![150, 350, 500]);
    }

    #[test]
    fn test_convert_to_mono_quad_audio() {
        // 4-channel audio
        let quad_data = vec![100i16, 200, 300, 400, 500, 600, 700, 800];
        let result = convert_to_mono(&quad_data, 4);
        // Expected: [(100+200+300+400)/4, (500+600+700+800)/4] = [250, 650]
        assert_eq!(result, vec![250, 650]);
    }

    #[test]
    fn test_convert_to_mono_negative_values() {
        let stereo_data = vec![-100i16, 200, -300, 400];
        let result = convert_to_mono(&stereo_data, 2);
        // Expected: [(-100+200)/2, (-300+400)/2] = [50, 50]
        assert_eq!(result, vec![50, 50]);
    }

    #[test]
    fn test_convert_to_mono_overflow_protection() {
        // Test with values that might cause overflow when summing
        let stereo_data = vec![i16::MAX, i16::MAX, i16::MIN, i16::MIN];
        let result = convert_to_mono(&stereo_data, 2);
        // Should handle overflow gracefully
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_resampling_same_rate() {
        let samples = vec![100i16, 200, 300, 400];
        let result = resampling(&samples, 16000, 16000).unwrap();
        assert_eq!(result, samples);
    }

    #[test]
    fn test_resampling_upsampling() {
        // Test with longer input to get more predictable upsampling behavior
        let samples = vec![100i16; 1000];
        let result = resampling(&samples, 8000, 16000);
        
        // Should succeed
        assert!(result.is_ok(), "Upsampling should succeed");
        // This test mainly verifies that upsampling doesn't crash or error
    }

    #[test]
    fn test_resampling_downsampling() {
        let samples = vec![100i16; 1000]; // 1000 samples
        let result = resampling(&samples, 48000, 16000).unwrap();
        // Should have approximately 1/3 the samples
        assert!(result.len() < samples.len());
        assert!(result.len() > samples.len() / 4); // Allow some variance
    }

    #[test]
    fn test_resampling_empty_input() {
        let empty_samples: Vec<i16> = vec![];
        let result = resampling(&empty_samples, 16000, 8000);
        // Should handle empty input gracefully
        assert!(result.is_ok());
    }

    #[test]
    fn test_resampling_single_sample() {
        let samples = vec![100i16];
        let result = resampling(&samples, 16000, 8000);
        // Single sample may not produce output depending on resampler implementation
        assert!(result.is_ok(), "Resampling single sample should not error");
    }

    #[test]
    fn test_resampling_common_rates() {
        let samples = vec![100i16, -200, 300, -400, 500];
        
        // Test common sample rate conversions
        let rates = vec![8000, 16000, 22050, 44100, 48000];
        
        for &input_rate in &rates {
            for &output_rate in &rates {
                let result = resampling(&samples, input_rate, output_rate);
                assert!(result.is_ok(), "Failed to resample from {input_rate} to {output_rate}");
                
                if input_rate == output_rate {
                    assert_eq!(result.unwrap(), samples);
                }
            }
        }
    }

    #[test]
    fn test_resampling_preserves_signal_characteristics() {
        // Create a longer sine-like wave for more reliable resampling
        let samples: Vec<i16> = (0..1000)
            .map(|i| ((i as f32 * 0.01).sin() * 1000.0) as i16)
            .collect();
        
        let result = resampling(&samples, 16000, 32000).unwrap();
        
        // Basic sanity checks - longer input should produce output
        assert!(!result.is_empty() || samples.is_empty(), "Non-empty input should produce output");
        
        if !result.is_empty() {
            // Check that values are within reasonable bounds (not clipped to extremes)
            let has_extreme_values = result.iter().any(|&x| x == i16::MAX || x == i16::MIN);
            assert!(!has_extreme_values, "Resampling should not produce extreme clipping");
        }
    }

    #[test]
    fn test_resampling_with_silence() {
        let silence = vec![0i16; 1000];
        let result = resampling(&silence, 44100, 16000).unwrap();
        
        // Resampled silence should still be silence
        assert!(result.iter().all(|&x| x.abs() < 10)); // Allow for tiny rounding errors
    }

    #[test]
    fn test_resampling_extreme_ratios() {
        let samples = vec![1000i16, -1000, 1000, -1000];
        
        // Test extreme downsampling
        let result = resampling(&samples, 48000, 8000);
        assert!(result.is_ok());
        
        // Test extreme upsampling
        let result = resampling(&samples, 8000, 48000);
        assert!(result.is_ok());
    }
}
