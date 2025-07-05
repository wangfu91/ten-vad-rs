use std::env;
use std::time::Instant;
use ten_vad_rs::TenVAD;
use hound::{WavReader, WavWriter, WavSpec, SampleFormat};
use std::fs::File;
use std::io::BufReader;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args.len() > 3 {
        eprintln!("Usage: {} <input_wav> [output_wav]", args[0]);
        eprintln!("Example: {} input.wav output_vad.wav", args[0]);
        eprintln!("  This will process the input WAV and optionally create an output WAV with only voice segments");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = args.get(2);

    // Create VAD with optimal settings for real-time processing
    let vad = TenVAD::new(256, 0.5).unwrap_or_else(|e| {
        eprintln!("Error creating TenVAD: {e}");
        std::process::exit(1);
    });

    println!("🎤 Real-time VAD Processing Demo");
    println!("TenVAD Version: {}", TenVAD::get_version());
    println!("Input: {input_path}");
    if let Some(output) = output_path {
        println!("Output: {output}");
    }
    println!();

    if let Err(e) = process_realtime_vad(input_path, output_path, &vad) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn process_realtime_vad(input_path: &str, output_path: Option<&String>, vad: &TenVAD) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(input_path)?;
    let mut reader = WavReader::new(BufReader::new(file))?;
    let spec = reader.spec();
    
    println!("Input format: {}Hz, {} channels, {} bits", 
             spec.sample_rate, spec.channels, spec.bits_per_sample);
    
    if spec.sample_format != SampleFormat::Int || spec.bits_per_sample != 16 {
        return Err("Only 16-bit PCM WAV files are supported".into());
    }

    // Setup output writer if requested
    let mut writer = if let Some(output) = output_path {
        let output_spec = WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        Some(WavWriter::create(output, output_spec)?)
    } else {
        None
    };

    // Read and process audio in chunks (simulating real-time processing)
    let hop_size = vad.hop_size();
    let mut samples = Vec::new();
    let mut voice_segments = Vec::new();
    let mut current_voice_segment: Option<Vec<i16>> = None;
    let mut frame_count = 0;
    let mut voice_frame_count = 0;
    let mut total_processing_time = 0.0;

    println!("Processing audio in real-time simulation...");
    println!("Press Ctrl+C to stop (simulated)\n");

    // Read all samples first (in real-time, this would be a stream)
    let all_samples: Vec<i16> = reader
        .samples::<i16>()
        .collect::<Result<Vec<_>, _>>()?;

    // Convert to mono if needed
    let mono_samples = if spec.channels == 1 {
        all_samples
    } else {
        all_samples
            .chunks(spec.channels as usize)
            .map(|chunk| {
                let sum: i32 = chunk.iter().map(|&s| s as i32).sum();
                (sum / chunk.len() as i32) as i16
            })
            .collect()
    };

    // Resample if needed (simplified - in real implementation you'd use proper resampling)
    let processed_samples = if spec.sample_rate != 16000 {
        println!("Note: Resampling from {}Hz to 16000Hz (simplified)", spec.sample_rate);
        // Simple decimation for demo purposes
        let ratio = spec.sample_rate as f64 / 16000.0;
        mono_samples
            .iter()
            .step_by(ratio.round() as usize)
            .cloned()
            .collect()
    } else {
        mono_samples
    };

    // Process in streaming fashion
    for chunk in processed_samples.chunks(hop_size) {
        if chunk.len() < hop_size {
            break; // Skip incomplete frames
        }

        samples.extend_from_slice(chunk);
        
        // Simulate real-time processing delay
        let start_time = Instant::now();
        
        // Process the frame
        let result = vad.process_frame(chunk)?;
        
        let processing_time = start_time.elapsed().as_secs_f64() * 1000.0; // ms
        total_processing_time += processing_time;
        
        let timestamp_ms = (frame_count * hop_size) as f64 / 16.0; // 16kHz = 16 samples per ms
        
        if result.is_voice {
            voice_frame_count += 1;
            
            // Start new voice segment or continue existing one
            if current_voice_segment.is_none() {
                current_voice_segment = Some(Vec::new());
                print!("🎤 Voice detected at {:.0}ms (prob: {:.2})", timestamp_ms, result.probability);
            }
            
            // Add samples to current voice segment
            if let Some(ref mut segment) = current_voice_segment {
                segment.extend_from_slice(chunk);
            }
        } else {
            // End current voice segment if it exists
            if let Some(segment) = current_voice_segment.take() {
                let end_time = timestamp_ms;
                let duration = segment.len() as f64 / 16.0; // Duration in ms
                println!(" → ended at {end_time:.0}ms ({duration:.0}ms duration)");
                
                // Write to output file if requested
                if let Some(ref mut w) = writer {
                    for sample in &segment {
                        w.write_sample(*sample)?;
                    }
                }
                
                voice_segments.push((timestamp_ms - duration, end_time, segment));
            }
        }
        
        frame_count += 1;
        
        // Show progress every 1000 frames (~16 seconds)
        if frame_count % 1000 == 0 {
            println!("📊 Processed {:.1}s of audio ({} frames)", 
                     timestamp_ms / 1000.0, frame_count);
        }
    }

    // Handle case where voice continues until the end
    if let Some(segment) = current_voice_segment.take() {
        let end_time = (frame_count * hop_size) as f64 / 16.0;
        let duration = segment.len() as f64 / 16.0;
        println!(" → ended at {end_time:.0}ms ({duration:.0}ms duration)");
        
        if let Some(ref mut w) = writer {
            for sample in &segment {
                w.write_sample(*sample)?;
            }
        }
        
        voice_segments.push((end_time - duration, end_time, segment));
    }

    // Finalize output file
    if let Some(w) = writer {
        w.finalize()?;
        println!("✅ Voice segments saved to output file");
    }

    // Performance statistics
    let total_audio_duration = (frame_count * hop_size) as f64 / 16000.0; // seconds
    let total_voice_duration: f64 = voice_segments
        .iter()
        .map(|(_, _, segment)| segment.len() as f64 / 16000.0)
        .sum();
    let avg_processing_time = total_processing_time / frame_count as f64;
    let real_time_factor = (total_audio_duration * 1000.0) / total_processing_time;
    
    println!("\n📊 Processing Statistics:");
    println!("Total audio duration: {total_audio_duration:.2}s");
    println!("Total frames processed: {frame_count}");
    println!("Voice frames detected: {voice_frame_count} ({:.1}%)", 
             (voice_frame_count as f64 / frame_count as f64) * 100.0);
    println!("Voice segments found: {}", voice_segments.len());
    println!("Total voice duration: {total_voice_duration:.2}s ({:.1}%)", 
             (total_voice_duration / total_audio_duration) * 100.0);
    println!("Average processing time per frame: {avg_processing_time:.3}ms");
    println!("Real-time factor: {real_time_factor:.1}x (higher is better)");
    
    if real_time_factor > 1.0 {
        println!("✅ Processing is faster than real-time!");
    } else {
        println!("⚠️  Processing is slower than real-time");
    }

    if !voice_segments.is_empty() {
        println!("\n🎯 Voice Segments Details:");
        for (i, (start, end, segment)) in voice_segments.iter().enumerate() {
            let duration = segment.len() as f64 / 16000.0;
            println!("  {}. {:.2}s - {:.2}s ({:.2}s duration, {} samples)", 
                     i + 1, start / 1000.0, end / 1000.0, duration, segment.len());
        }
    }

    Ok(())
}
