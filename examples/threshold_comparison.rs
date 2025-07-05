use std::env;
use std::fs::File;
use std::io::BufReader;
use ten_vad_rs::TenVAD;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <wav_file_path>", args[0]);
        eprintln!("This example demonstrates how different threshold values affect VAD detection");
        std::process::exit(1);
    }

    let wav_file_path = &args[1];
    
    // Test different threshold values
    let thresholds = [0.1, 0.3, 0.5, 0.7, 0.9];
    
    println!("Comparing VAD performance with different thresholds");
    println!("File: {wav_file_path}\n");

    for threshold in thresholds {
        println!("🎯 Testing with threshold = {threshold:.1}");
        
        let vad = TenVAD::new(256, threshold).unwrap_or_else(|e| {
            eprintln!("Error creating TenVAD: {e}");
            std::process::exit(1);
        });

        match analyze_file(wav_file_path, &vad) {
            Ok(stats) => {
                println!("   Speech segments: {}", stats.segment_count);
                println!("   Total speech time: {:.2}s ({:.1}%)", 
                         stats.total_speech_duration, stats.speech_percentage);
                println!("   Average segment length: {:.0}ms", stats.avg_segment_length);
            }
            Err(e) => {
                eprintln!("   Error: {e}");
            }
        }
        println!();
    }
}

#[derive(Debug)]
struct VadStats {
    segment_count: usize,
    total_speech_duration: f32,
    speech_percentage: f32,
    avg_segment_length: f32,
}

fn analyze_file(wav_file_path: &str, vad: &TenVAD) -> Result<VadStats, String> {
    let file = File::open(wav_file_path).map_err(|e| format!("Failed to open file: {e}"))?;
    let mut reader = hound::WavReader::new(BufReader::new(file)).map_err(|e| e.to_string())?;
    
    let spec = reader.spec();
    
    // For simplicity, this example only handles 16kHz mono files
    if spec.sample_rate != 16000 {
        return Err("This threshold comparison example only supports 16kHz files".to_string());
    }
    if spec.channels != 1 {
        return Err("This threshold comparison example only supports mono files".to_string());
    }
    if spec.sample_format != hound::SampleFormat::Int || spec.bits_per_sample != 16 {
        return Err("Only 16-bit PCM is supported".to_string());
    }

    let samples: Vec<i16> = reader
        .samples::<i16>()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;

    let hop_size = vad.hop_size();
    let mut speech_segments = Vec::new();
    let mut current_speech_start: Option<f32> = None;
    let mut frame_index = 0;

    for chunk in samples.chunks(hop_size) {
        if chunk.len() < hop_size {
            break;
        }

        let timestamp_ms = (frame_index * hop_size) as f32 / 16.0;
        
        if let Ok(result) = vad.process_frame(chunk) {
            if result.is_voice {
                if current_speech_start.is_none() {
                    current_speech_start = Some(timestamp_ms);
                }
            } else if let Some(start_time) = current_speech_start {
                let duration = timestamp_ms - start_time;
                speech_segments.push((start_time, timestamp_ms, duration));
                current_speech_start = None;
            }
        }
        
        frame_index += 1;
    }

    // Handle case where speech continues until the end
    if let Some(start_time) = current_speech_start {
        let end_time = (frame_index * hop_size) as f32 / 16.0;
        let duration = end_time - start_time;
        speech_segments.push((start_time, end_time, duration));
    }

    let total_duration = (frame_index * hop_size) as f32 / 16000.0;
    let total_speech_duration: f32 = speech_segments.iter().map(|(_, _, duration)| duration).sum::<f32>() / 1000.0;
    let speech_percentage = if total_duration > 0.0 { (total_speech_duration / total_duration) * 100.0 } else { 0.0 };
    let avg_segment_length = if !speech_segments.is_empty() {
        speech_segments.iter().map(|(_, _, duration)| duration).sum::<f32>() / speech_segments.len() as f32
    } else {
        0.0
    };

    Ok(VadStats {
        segment_count: speech_segments.len(),
        total_speech_duration,
        speech_percentage,
        avg_segment_length,
    })
}
