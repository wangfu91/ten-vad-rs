use anyhow::anyhow;
use std::env;
use std::fs::File;
use std::io::BufReader;
use ten_vad_rs::{AudioSegment, TenVad, utils};

const HOP_SIZE: usize = 256; // 16ms at 16kHz
const THRESHOLD: f32 = 0.5; // Default threshold for VAD
const TARGET_SAMPLE_RATE: u32 = 16000; // Required sample rate for TEN VAD (16kHz)

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() <= 2 {
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
    println!(
        "Input WAV: {}Hz, {} channels, {} bits",
        spec.sample_rate, spec.channels, spec.bits_per_sample
    );

    // Validate input format
    if spec.sample_format != hound::SampleFormat::Int {
        return Err(anyhow!("Only 16-bit PCM WAV files are supported"));
    }

    let mut audio_segment = AudioSegment::new();

    let all_samples = reader.samples::<i16>().collect::<Result<Vec<i16>, _>>()?;

    let processed_samples = preprocess_audio(
        &all_samples,
        spec.channels as u32,
        spec.sample_rate,
        TARGET_SAMPLE_RATE,
    )?;

    audio_segment.append_samples(&processed_samples);

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
    samples: &[i16],
    channels: u32,
    input_sample_rate: u32,
    output_sample_rate: u32,
) -> anyhow::Result<Vec<i16>> {
    // Convert stereo to mono if needed
    let mono_samples = if channels > 1 {
        utils::convert_to_mono(samples, channels)
    } else {
        samples.to_vec()
    };

    // Resample to 16kHz if necessary
    utils::resampling(&mono_samples, input_sample_rate, output_sample_rate).map_err(|e| {
        anyhow::anyhow!(
            "Failed to resample audio from {input_sample_rate}Hz to {output_sample_rate}Hz: {e}"
        )
    })
}
