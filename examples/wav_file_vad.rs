use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::env;
use std::fs::File;
use std::io::BufReader;
use ten_vad_rs::{AudioSegment, TenVad};

const HOP_SIZE: usize = 256; // 16ms at 16kHz
const THRESHOLD: f32 = 0.5; // Default threshold for VAD
const TARGET_SAMPLE_RATE: u32 = 16000; // Required sample rate for TEN VAD (16kHz)

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <wav_file_path>", args[0]);
        eprintln!("Example: {} input.wav", args[0]);
        std::process::exit(1);
    }

    let wav_file_path = &args[1];

    let mut vad = TenVad::new("onnx/ten-vad.onnx")?;
    process_wav_file(wav_file_path, &mut vad)?;

    Ok(())
}

fn process_wav_file(wav_file_path: &str, vad: &mut TenVad) -> anyhow::Result<()> {
    println!("Processing WAV file: {wav_file_path}");

    let file = File::open(wav_file_path)?;
    let mut reader = hound::WavReader::new(BufReader::new(file))?;

    let spec = reader.spec();
    println!("Input WAV spec: {spec:?}");

    let mut audio_segment = AudioSegment::new();

    let all_f32_samples = if spec.sample_format == hound::SampleFormat::Float {
        reader.samples::<f32>().collect::<Result<Vec<f32>, _>>()?
    } else {
        let i16_samples = reader.samples::<i16>().collect::<Result<Vec<i16>, _>>()?;
        i16_samples
            .iter()
            .map(|&s| s as f32 / i16::MAX as f32)
            .collect()
    };

    let resampled_i16_samples = preprocess_audio(
        &all_f32_samples,
        spec.channels as usize,
        spec.sample_rate,
        TARGET_SAMPLE_RATE,
    )?;

    audio_segment.append_samples(&resampled_i16_samples);

    let mut wav_writer: hound::WavWriter<std::io::BufWriter<File>> = hound::WavWriter::create(
        "processed_wav_output.wav",
        hound::WavSpec {
            channels: 1,
            sample_rate: TARGET_SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;

    for sample in resampled_i16_samples {
        wav_writer.write_sample(sample)?;
    }
    wav_writer.finalize()?;

    while let Some(frame) = audio_segment.get_audio_frame(HOP_SIZE) {
        match vad.process_frame(&frame) {
            Ok(vad_score) => {
                if vad_score >= THRESHOLD {
                    println!("++++++ Detected voice in frame: probability {vad_score:2}");
                } else {
                    println!("------ No voice detected in frame");
                }
            }
            Err(e) => {
                eprintln!("Error processing frame: {e}");
            }
        }
    }

    Ok(())
}

fn preprocess_audio(
    samples: &[f32],
    channels: usize,
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
    resampling(&mono_samples, input_sample_rate, output_sample_rate).map_err(|e| {
        anyhow::anyhow!(
            "Failed to resample audio from {input_sample_rate}Hz to {output_sample_rate}Hz: {e}"
        )
    })
}

pub fn resampling(
    mono_f32_samples: &[f32],
    input_sample_rate: u32,
    output_sample_rate: u32,
) -> Result<Vec<i16>, Box<dyn std::error::Error>> {
    if input_sample_rate == output_sample_rate {
        return Ok(mono_f32_samples
            .iter()
            .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
            .collect());
    }

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
        mono_f32_samples.len(),
        1, // channels
    )?;

    // Perform resampling
    let output_f32_samples = resampler.process(&[mono_f32_samples], None)?;

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
pub fn convert_to_mono(data: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        data.to_vec()
    } else {
        data.chunks(channels)
            .map(|frame| {
                let sum: f32 = frame.iter().sum();
                let divisor = if frame.len() < channels {
                    frame.len() as f32
                } else {
                    channels as f32
                };
                sum / divisor
            })
            .collect()
    }
}
