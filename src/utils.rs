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
