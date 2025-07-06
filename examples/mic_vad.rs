use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use ten_vad_rs::AudioSegment;

const HOP_SIZE: usize = 256; // 16ms at 16kHz
const THRESHOLD: f32 = 0.5;

fn main() -> anyhow::Result<()> {
    println!("TenVAD Microphone Example");

    // Example usage of TenVAD with a microphone input
    use ten_vad_rs::TenVAD;

    let vad = TenVAD::new(HOP_SIZE, THRESHOLD)?;
    println!("TenVAD Version: {}", TenVAD::get_version());
    println!(
        "Using hop size: {}, threshold: {}",
        vad.hop_size(),
        vad.threshold()
    );
    println!("Listening for speech (16ms frames)...");

    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device found"))?;
    let input_stream_config = input_device
        .default_input_config()
        .map_err(|e| anyhow::anyhow!("Failed to get default input config: {}", e))?;

    let sample_rate = input_stream_config.sample_rate().0;
    let channels = input_stream_config.channels() as usize;

    println!("Input device: {}", input_device.name()?);
    println!("Sample rate: {sample_rate} Hz, Channels: {channels}");

    let mut audio_segment = AudioSegment::new(HOP_SIZE);

    let input_stream = input_device.build_input_stream(
        &input_stream_config.into(),
        move |data: &[i16], _| {
            let samples = preprocess_audio(
                data,
                channels as u32,
                sample_rate,
                16000, // Resample to 16kHz
            )
            .unwrap();

            while let Some(frame) = audio_segment.append_samples(&samples) {
                // Process each frame of audio data
                match vad.process_frame(&frame) {
                    Ok(result) => {
                        if result.is_voice {
                            println!(
                                "++++++ Detected voice in frame: probability {}",
                                result.probability,
                            );
                        }
                    }
                    Err(e) => eprintln!("Error processing frame: {e}"),
                }
            }
        },
        move |err| eprintln!("Input stream error: {err}"),
        None,
    )?;

    input_stream.play()?;

    std::thread::sleep(std::time::Duration::from_secs(300));

    Ok(())
}

fn preprocess_audio(
    samples: &[i16],
    channels: u32,
    input_sample_rate: u32,
    output_sample_rate: u32,
) -> anyhow::Result<Vec<i16>> {
    // Convert stereo to mono if needed
    let mono_samples = if channels > 1 {
        convert_to_mono(samples, channels)
    } else {
        samples.to_vec()
    };

    // Resample to 16kHz if necessary
    resample_to_16khz(&mono_samples, input_sample_rate, output_sample_rate)
}

fn resample_to_16khz(
    mono_i16_samples: &[i16],
    input_sample_rate: u32,
    output_sample_rate: u32,
) -> anyhow::Result<Vec<i16>> {
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

fn convert_to_mono(data: &[i16], channels: u32) -> Vec<i16> {
    if channels == 1 {
        data.to_vec()
    } else {
        data.chunks_exact(channels as usize)
            .map(|frame| {
                let sum: i32 = frame.iter().map(|&s| s as i32).sum();
                (sum / channels as i32) as i16
            })
            .collect()
    }
}
