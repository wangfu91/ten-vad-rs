use std::env;
use std::fs::File;
use std::io::BufReader;
use ten_vad_rs::{AudioFrameBuffer, TARGET_SAMPLE_RATE, TenVad};

const HOP_SIZE: usize = 256; // 16ms at 16kHz
const THRESHOLD: f32 = 0.5; // Default threshold for VAD

// Embed the ONNX model into the binary at compile time
static MODEL_BYTES: &[u8] = include_bytes!("../onnx/ten-vad.onnx");

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <wav_file_path>", args[0]);
        eprintln!("Example: {} input.wav", args[0]);
        std::process::exit(1);
    }

    let wav_file_path = &args[1];

    // Create TenVad from embedded model bytes
    let mut vad = TenVad::new_from_bytes(MODEL_BYTES, TARGET_SAMPLE_RATE)?;
    process_wav_file(wav_file_path, &mut vad)?;

    Ok(())
}

fn process_wav_file(wav_file_path: &str, vad: &mut TenVad) -> anyhow::Result<()> {
    println!("Processing WAV file: {wav_file_path}");

    let file = File::open(wav_file_path)?;
    let mut reader = hound::WavReader::new(BufReader::new(file))?;

    let spec = reader.spec();
    println!("Input WAV spec: {spec:?}");

    if spec.sample_rate != TARGET_SAMPLE_RATE {
        return Err(anyhow::anyhow!(
            "Unsupported sample rate: {} Hz. TEN VAD requires {} Hz",
            spec.sample_rate,
            TARGET_SAMPLE_RATE
        ));
    }

    if spec.channels != 1 {
        return Err(anyhow::anyhow!(
            "Unsupported number of channels: {}. TEN VAD requires: 1 (mono)",
            spec.channels
        ));
    }

    let mut audio_buffer = AudioFrameBuffer::new();

    let all_i16_samples = if spec.sample_format == hound::SampleFormat::Float {
        reader
            .samples::<f32>()
            .map(|s| {
                let s = s.unwrap_or(0.0);
                (s * i16::MAX as f32)
                    .round()
                    .clamp(i16::MIN as f32, i16::MAX as f32) as i16
            })
            .collect::<Vec<i16>>()
    } else {
        reader
            .samples::<i16>()
            .map(|s| s.unwrap_or(0))
            .collect::<Vec<i16>>()
    };

    audio_buffer.append_samples(all_i16_samples);

    while let Some(frame) = audio_buffer.pop_frame(HOP_SIZE) {
        match vad.process_frame(&frame) {
            Ok(vad_score) => {
                if vad_score >= THRESHOLD {
                    println!("++++++ Voice detected: probability {vad_score:.2}");
                } else {
                    println!("----- No voice detected");
                }
            }
            Err(e) => {
                eprintln!("Error running VAD on audio frame: {e}");
            }
        }
    }

    Ok(())
}
