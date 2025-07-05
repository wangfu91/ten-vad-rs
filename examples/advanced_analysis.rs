/// Advanced WAV File VAD Analysis
///
/// This example demonstrates advanced usage of the TenVAD library with:
/// - Real-time processing simulation
/// - Multiple threshold analysis
/// - Speech segment statistics
/// - Performance monitoring
use std::env;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;
use ten_vad_rs::{TenVAD, VadResult};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Advanced VAD Analysis Tool");
        eprintln!("Usage: {} <wav_file> [threshold]", args[0]);
        eprintln!("  wav_file:  Path to 16kHz mono WAV file");
        eprintln!("  threshold: VAD threshold (0.0-1.0, default: 0.5)");
        eprintln!("\nExample: {} speech.wav 0.3", args[0]);
        std::process::exit(1);
    }

    let wav_file = &args[1];
    let threshold = args
        .get(2)
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.5);

    if !(0.0..=1.0).contains(&threshold) {
        eprintln!("Error: Threshold must be between 0.0 and 1.0");
        std::process::exit(1);
    }

    println!("🎙️  Advanced VAD Analysis");
    println!("File: {wav_file}");
    println!("Threshold: {threshold:.2}");
    println!("{}", "─".repeat(50));

    match analyze_audio(wav_file, threshold) {
        Ok(analysis) => print_analysis(&analysis),
        Err(e) => {
            eprintln!("❌ Analysis failed: {e}");
            std::process::exit(1);
        }
    }
}

#[derive(Debug)]
struct AudioAnalysis {
    file_info: FileInfo,
    vad_settings: VadSettings,
    processing_stats: ProcessingStats,
    speech_analysis: SpeechAnalysis,
}

#[derive(Debug)]
struct FileInfo {
    duration_secs: f32,
    sample_rate: u32,
    channels: u16,
    total_samples: usize,
}

#[derive(Debug)]
struct VadSettings {
    hop_size: usize,
    threshold: f32,
    frame_duration_ms: f32,
}

#[derive(Debug)]
struct ProcessingStats {
    total_frames: usize,
    processing_time_ms: f32,
    frames_per_second: f32,
    real_time_factor: f32,
}

#[derive(Debug)]
struct SpeechAnalysis {
    segments: Vec<SpeechSegment>,
    total_speech_duration: f32,
    speech_percentage: f32,
    avg_segment_duration: f32,
    longest_segment: f32,
    shortest_segment: f32,
    speech_density: f32, // speech segments per minute
}

#[derive(Debug, Clone)]
struct SpeechSegment {
    start_ms: f32,
    end_ms: f32,
    duration_ms: f32,
    avg_probability: f32,
    max_probability: f32,
}

fn analyze_audio(wav_file: &str, threshold: f32) -> Result<AudioAnalysis, String> {
    // Load audio file
    let file = File::open(wav_file).map_err(|e| format!("Failed to open file: {e}"))?;
    let mut reader = hound::WavReader::new(BufReader::new(file)).map_err(|e| e.to_string())?;
    let spec = reader.spec();

    // Validate format (simplified for this example)
    if spec.sample_rate != 16000 || spec.channels != 1 || spec.bits_per_sample != 16 {
        return Err("This example requires 16kHz mono 16-bit WAV files".to_string());
    }

    let samples: Vec<i16> = reader
        .samples::<i16>()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;

    let file_info = FileInfo {
        duration_secs: samples.len() as f32 / spec.sample_rate as f32,
        sample_rate: spec.sample_rate,
        channels: spec.channels,
        total_samples: samples.len(),
    };

    // Create VAD
    let vad = TenVAD::new(256, threshold).map_err(|e| format!("VAD creation failed: {e}"))?;

    let vad_settings = VadSettings {
        hop_size: vad.hop_size(),
        threshold: vad.threshold(),
        frame_duration_ms: (vad.hop_size() as f32 / 16.0), // 16kHz
    };

    // Process audio
    let start_time = Instant::now();
    let mut results = Vec::new();
    let mut frame_count = 0;

    for chunk in samples.chunks(vad.hop_size()) {
        if chunk.len() < vad.hop_size() {
            break;
        }

        match vad.process_frame(chunk) {
            Ok(result) => {
                results.push((frame_count, result));
                frame_count += 1;
            }
            Err(e) => {
                eprintln!("Warning: Frame {frame_count} processing failed: {e}");
            }
        }
    }

    let processing_time = start_time.elapsed();

    let processing_stats = ProcessingStats {
        total_frames: frame_count,
        processing_time_ms: processing_time.as_secs_f32() * 1000.0,
        frames_per_second: frame_count as f32 / processing_time.as_secs_f32(),
        real_time_factor: file_info.duration_secs / processing_time.as_secs_f32(),
    };

    // Analyze speech segments
    let speech_analysis = analyze_speech_segments(&results, &vad_settings, file_info.duration_secs);

    Ok(AudioAnalysis {
        file_info,
        vad_settings,
        processing_stats,
        speech_analysis,
    })
}

fn analyze_speech_segments(
    results: &[(usize, VadResult)],
    settings: &VadSettings,
    total_duration: f32,
) -> SpeechAnalysis {
    let mut segments = Vec::new();
    let mut current_segment: Option<(f32, Vec<f32>)> = None; // (start_time, probabilities)

    for (frame_idx, result) in results {
        let timestamp_ms = *frame_idx as f32 * settings.frame_duration_ms;

        if result.is_voice {
            if let Some((_, ref mut probs)) = current_segment {
                probs.push(result.probability);
            } else {
                current_segment = Some((timestamp_ms, vec![result.probability]));
            }
        } else if let Some((start_time, probabilities)) = current_segment.take() {
            // End of speech segment
            let duration = timestamp_ms - start_time;
            let avg_prob = probabilities.iter().sum::<f32>() / probabilities.len() as f32;
            let max_prob = probabilities.iter().fold(0.0f32, |a, &b| a.max(b));

            segments.push(SpeechSegment {
                start_ms: start_time,
                end_ms: timestamp_ms,
                duration_ms: duration,
                avg_probability: avg_prob,
                max_probability: max_prob,
            });
        }
    }

    // Handle case where speech continues to the end
    if let Some((start_time, probabilities)) = current_segment {
        let end_time = results.len() as f32 * settings.frame_duration_ms;
        let duration = end_time - start_time;
        let avg_prob = probabilities.iter().sum::<f32>() / probabilities.len() as f32;
        let max_prob = probabilities.iter().fold(0.0f32, |a, &b| a.max(b));

        segments.push(SpeechSegment {
            start_ms: start_time,
            end_ms: end_time,
            duration_ms: duration,
            avg_probability: avg_prob,
            max_probability: max_prob,
        });
    }

    // Calculate statistics
    let total_speech_duration = segments.iter().map(|s| s.duration_ms).sum::<f32>() / 1000.0;
    let speech_percentage = (total_speech_duration / total_duration) * 100.0;
    let avg_segment_duration = if !segments.is_empty() {
        segments.iter().map(|s| s.duration_ms).sum::<f32>() / segments.len() as f32
    } else {
        0.0
    };

    let longest_segment = segments
        .iter()
        .map(|s| s.duration_ms)
        .fold(0.0f32, |a, b| a.max(b));
    let shortest_segment = segments
        .iter()
        .map(|s| s.duration_ms)
        .fold(f32::INFINITY, |a, b| a.min(b));
    let shortest_segment = if shortest_segment == f32::INFINITY {
        0.0
    } else {
        shortest_segment
    };

    let speech_density = (segments.len() as f32) / (total_duration / 60.0); // segments per minute

    SpeechAnalysis {
        segments,
        total_speech_duration,
        speech_percentage,
        avg_segment_duration,
        longest_segment,
        shortest_segment,
        speech_density,
    }
}

fn print_analysis(analysis: &AudioAnalysis) {
    println!("📁 File Information:");
    println!("   Duration: {:.2}s", analysis.file_info.duration_secs);
    println!("   Sample Rate: {}Hz", analysis.file_info.sample_rate);
    println!("   Channels: {}", analysis.file_info.channels);
    println!("   Total Samples: {}", analysis.file_info.total_samples);

    println!("\n⚙️  VAD Settings:");
    println!(
        "   Hop Size: {} samples ({:.1}ms)",
        analysis.vad_settings.hop_size, analysis.vad_settings.frame_duration_ms
    );
    println!("   Threshold: {:.2}", analysis.vad_settings.threshold);

    println!("\n🚀 Processing Performance:");
    println!(
        "   Total Frames: {}",
        analysis.processing_stats.total_frames
    );
    println!(
        "   Processing Time: {:.1}ms",
        analysis.processing_stats.processing_time_ms
    );
    println!(
        "   Processing Speed: {:.0} frames/sec",
        analysis.processing_stats.frames_per_second
    );
    println!(
        "   Real-time Factor: {:.1}x",
        analysis.processing_stats.real_time_factor
    );

    println!("\n🎤 Speech Analysis:");
    println!(
        "   Speech Segments: {}",
        analysis.speech_analysis.segments.len()
    );
    println!(
        "   Total Speech: {:.2}s ({:.1}%)",
        analysis.speech_analysis.total_speech_duration, analysis.speech_analysis.speech_percentage
    );
    println!(
        "   Average Segment: {:.0}ms",
        analysis.speech_analysis.avg_segment_duration
    );
    println!(
        "   Longest Segment: {:.0}ms",
        analysis.speech_analysis.longest_segment
    );
    println!(
        "   Shortest Segment: {:.0}ms",
        analysis.speech_analysis.shortest_segment
    );
    println!(
        "   Speech Density: {:.1} segments/minute",
        analysis.speech_analysis.speech_density
    );

    if !analysis.speech_analysis.segments.is_empty() {
        println!("\n📋 Detailed Segments:");
        for (i, segment) in analysis.speech_analysis.segments.iter().enumerate() {
            println!(
                "   {}. {:.0}-{:.0}ms ({:.0}ms) avg_prob={:.3} max_prob={:.3}",
                i + 1,
                segment.start_ms,
                segment.end_ms,
                segment.duration_ms,
                segment.avg_probability,
                segment.max_probability
            );
        }
    }

    println!("\n✅ Analysis complete!");
}
