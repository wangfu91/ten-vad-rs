use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::env;
use std::fs::File;
use std::io::BufReader;
use ten_vad_rs::TenVAD;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <wav_file_path>", args[0]);
        eprintln!("Example: {} input.wav", args[0]);
        std::process::exit(1);
    }

    let wav_file_path = &args[1];

    // Create TenVAD instance with optimal settings
    // Using 256 samples (16ms) hop size as recommended for 16kHz
    let vad = TenVAD::new(256, 0.5).unwrap_or_else(|e| {
        eprintln!("Error creating TenVAD: {e}");
        std::process::exit(1);
    });

    println!("TenVAD Version: {}", TenVAD::get_version());
    println!("Processing WAV file: {wav_file_path}");
    println!(
        "VAD Settings: hop_size={}, threshold={}",
        vad.hop_size(),
        vad.threshold()
    );
    println!("Listening for speech (16ms frames)...\n");

    if let Err(e) = process_wav_file(wav_file_path, &vad) {
        eprintln!("Error processing WAV file: {e}");
        std::process::exit(1);
    }
}

fn process_wav_file(wav_file_path: &str, vad: &TenVAD) -> Result<(), String> {
    let file = File::open(wav_file_path).map_err(|e| format!("Failed to open file: {e}"))?;
    let mut reader = hound::WavReader::new(BufReader::new(file)).map_err(|e| e.to_string())?;

    let spec = reader.spec();
    println!(
        "Input WAV: {}Hz, {} channels, {} bits",
        spec.sample_rate, spec.channels, spec.bits_per_sample
    );

    // Validate input format
    if spec.sample_format != hound::SampleFormat::Int {
        return Err("Only 16-bit PCM WAV files are supported".to_string());
    }

    if spec.bits_per_sample != 16 {
        return Err("Only 16-bit audio is supported".to_string());
    }

    // Read and process audio samples
    let all_samples: Vec<i16> = reader
        .samples::<i16>()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;

    if all_samples.is_empty() {
        return Err("WAV file contains no audio data".to_string());
    }

    println!(
        "Read {} samples ({:.2}s)\n",
        all_samples.len(),
        all_samples.len() as f32 / spec.sample_rate as f32
    );

    // Convert to mono if needed
    let mono_samples = if spec.channels == 1 {
        all_samples
    } else {
        convert_to_mono(&all_samples, spec.channels as usize)
    };

    // Resample to 16kHz if needed
    let resampled_samples = if spec.sample_rate == 16000 {
        mono_samples
    } else {
        resample_to_16khz(&mono_samples, spec.sample_rate)?
    };

    // Process audio in chunks
    process_audio_chunks(&resampled_samples, vad)?;

    Ok(())
}

fn convert_to_mono(samples: &[i16], channels: usize) -> Vec<i16> {
    println!("Converting {channels} channels to mono...");

    samples
        .chunks_exact(channels)
        .map(|frame| {
            // Average all channels to create mono
            let sum: i32 = frame.iter().map(|&s| s as i32).sum();
            (sum / channels as i32) as i16
        })
        .collect()
}

fn resample_to_16khz(samples: &[i16], input_sample_rate: u32) -> Result<Vec<i16>, String> {
    if input_sample_rate == 16000 {
        return Ok(samples.to_vec());
    }

    println!("Resampling from {input_sample_rate}Hz to 16000Hz...");

    // Convert i16 to f32 for resampling
    let input_f32: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();

    // Create resampler
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        16000.0 / input_sample_rate as f64,
        2.0, // max_relative_ratio
        params,
        input_f32.len(),
        1, // channels
    )
    .map_err(|e| format!("Failed to create resampler: {e}"))?;

    // Perform resampling
    let output_f32 = resampler
        .process(&[input_f32], None)
        .map_err(|e| format!("Resampling failed: {e}"))?;

    // Convert back to i16
    let output_i16: Vec<i16> = output_f32[0]
        .iter()
        .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
        .collect();

    println!("Resampled to {} samples", output_i16.len());
    Ok(output_i16)
}

fn process_audio_chunks(samples: &[i16], vad: &TenVAD) -> Result<(), String> {
    let hop_size = vad.hop_size();
    let mut frame_index = 0;
    let mut speech_segments = Vec::new();
    let mut current_speech_start: Option<f32> = None;

    // Process samples in chunks of hop_size
    for chunk in samples.chunks(hop_size) {
        // Skip incomplete frames at the end
        if chunk.len() < hop_size {
            break;
        }

        let timestamp_ms = (frame_index * hop_size) as f32 / 16.0; // 16kHz = 16 samples per ms

        match vad.process_frame(chunk) {
            Ok(result) => {
                if result.is_voice {
                    // Speech detected
                    if current_speech_start.is_none() {
                        current_speech_start = Some(timestamp_ms);
                        print!("🎤 Speech start: {timestamp_ms:.0}ms");
                    }
                } else {
                    // No speech
                    if let Some(start_time) = current_speech_start {
                        let duration = timestamp_ms - start_time;
                        println!(" → end: {timestamp_ms:.0}ms (duration: {duration:.0}ms)");
                        speech_segments.push((start_time, timestamp_ms, duration));
                        current_speech_start = None;
                    }
                }
            }
            Err(e) => {
                eprintln!("Warning: Failed to process frame {frame_index}: {e}");
            }
        }

        frame_index += 1;
    }

    // Handle case where speech continues until the end
    if let Some(start_time) = current_speech_start {
        let end_time = (frame_index * hop_size) as f32 / 16.0;
        let duration = end_time - start_time;
        println!(" → end: {end_time:.0}ms (duration: {duration:.0}ms)");
        speech_segments.push((start_time, end_time, duration));
    }

    // Summary statistics
    println!("\n📊 Summary:");
    println!("Total frames processed: {frame_index}");
    println!(
        "Total audio duration: {:.2}s",
        (frame_index * hop_size) as f32 / 16000.0
    );
    println!("Speech segments detected: {}", speech_segments.len());

    if !speech_segments.is_empty() {
        let total_speech_duration: f32 = speech_segments
            .iter()
            .map(|(_, _, duration)| duration)
            .sum();
        println!(
            "Total speech duration: {:.2}s",
            total_speech_duration / 1000.0
        );
        println!(
            "Speech percentage: {:.1}%",
            (total_speech_duration / 1000.0) / ((frame_index * hop_size) as f32 / 16000.0) * 100.0
        );

        println!("\nDetailed segments:");
        for (i, (start, end, duration)) in speech_segments.iter().enumerate() {
            println!(
                "  {}. {:.0}ms - {:.0}ms ({:.0}ms)",
                i + 1,
                start,
                end,
                duration
            );
        }
    } else {
        println!("No speech detected in the audio file.");
    }

    Ok(())
}
