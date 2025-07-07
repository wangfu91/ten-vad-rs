use anyhow::anyhow;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::env;
use std::fs::File;
use std::io::BufReader;
use ten_vad_rs::{AudioSegment, TenVAD, utils};

const HOP_SIZE: usize = 256; // 16ms at 16kHz
const THRESHOLD: f32 = 0.5; // Default threshold for VAD

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <wav_file_path>", args[0]);
        eprintln!("Example: {} input.wav", args[0]);
        std::process::exit(1);
    }

    let wav_file_path = &args[1];

    // Create TenVAD instance with optimal settings
    // Using 256 samples (16ms) hop size as recommended for 16kHz
    let vad = TenVAD::new(HOP_SIZE, THRESHOLD)?;

    println!("TenVAD Version: {}", TenVAD::get_version());
    println!("Processing WAV file: {wav_file_path}");
    println!(
        "VAD Settings: hop_size={}, threshold={}",
        vad.hop_size(),
        vad.threshold()
    );

    process_wav_file(wav_file_path, &vad)?;

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

fn process_wav_file(wav_file_path: &str, vad: &TenVAD) -> anyhow::Result<()> {
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

    let mut audio_segment = AudioSegment::new(HOP_SIZE);

    let all_samples = reader.samples::<i16>().collect::<Result<Vec<i16>, _>>()?;

    let processed_samples =
        preprocess_audio(&all_samples, spec.channels as u32, spec.sample_rate, 16000)?;

    while let Some(frame) = audio_segment.append_samples(&processed_samples) {
        match vad.process_frame(&frame) {
            Ok(result) => {
                if result.is_voice {
                    println!(
                        "++++++ Detected voice in frame: probability {}",
                        result.probability,
                    );
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
