use std::env;
use std::fs::File;
use std::io::BufReader;
use ten_vad_rs::{AudioFrameBuffer, TARGET_SAMPLE_RATE, TenVad};

const HOP_SIZE: usize = 256; // 16ms at 16kHz
const THRESHOLD: f32 = 0.5; // Default threshold for VAD

/// Minimum silence duration in ms to end a speech segment
const MIN_SILENCE_DURATION_MS: u32 = 300;
/// Padding to add before/after speech segments in ms
const SPEECH_PAD_MS: u32 = 30;
/// Minimum speech duration in ms to be considered valid
const MIN_SPEECH_DURATION_MS: u32 = 250;
/// Frame duration in ms (256 samples at 16kHz = 16ms)
const FRAME_SIZE_MS: u32 = 16;

/// Represents a detected speech segment with start and end sample positions
#[derive(Debug, Clone, Default)]
struct SpeechSegment {
    start: u32,
    end: u32,
    max_probability: f32,
}

/// Parameters for speech detection algorithm
struct SpeechParams {
    min_silence_samples: u32,
    speech_pad_samples: u32,
    min_speech_samples: u32,
}

impl SpeechParams {
    fn new(sample_rate: u32) -> Self {
        Self {
            min_silence_samples: sample_rate * MIN_SILENCE_DURATION_MS / 1000,
            speech_pad_samples: sample_rate * SPEECH_PAD_MS / 1000,
            min_speech_samples: sample_rate * MIN_SPEECH_DURATION_MS / 1000,
        }
    }
}

/// State machine for tracking speech segments
struct SpeechState {
    triggered: bool,
    temp_end: u32,
    current_sample: u32,
    current_speech: SpeechSegment,
}

impl SpeechState {
    fn new() -> Self {
        Self {
            triggered: false,
            temp_end: 0,
            current_sample: 0,
            current_speech: SpeechSegment::default(),
        }
    }

    fn update(&mut self, params: &SpeechParams, vad_score: f32) -> Option<SpeechSegment> {
        let speech_prob = vad_score > THRESHOLD;
        let window_size_samples = FRAME_SIZE_MS * TARGET_SAMPLE_RATE / 1000;
        self.current_sample += window_size_samples;

        // Track max probability for the current segment
        if speech_prob && vad_score > self.current_speech.max_probability {
            self.current_speech.max_probability = vad_score;
        }

        if speech_prob {
            if !self.triggered {
                self.triggered = true;
                self.current_speech.start =
                    self.current_sample.saturating_sub(params.speech_pad_samples);
                self.current_speech.max_probability = vad_score;
            }
            self.temp_end = 0;
        } else if self.triggered {
            if self.temp_end == 0 {
                self.temp_end = self.current_sample;
            }
            if self.current_sample.saturating_sub(self.temp_end) > params.min_silence_samples {
                self.current_speech.end = self.temp_end + params.speech_pad_samples;
                if self.current_speech.end.saturating_sub(self.current_speech.start)
                    > params.min_speech_samples
                {
                    let speech = std::mem::take(&mut self.current_speech);
                    self.triggered = false;
                    self.temp_end = 0;
                    return Some(speech);
                } else {
                    // Speech too short, ignore and reset
                    self.triggered = false;
                    self.temp_end = 0;
                    self.current_speech = SpeechSegment::default();
                }
            }
        }

        None
    }

    fn finalize(&mut self, params: &SpeechParams) -> Option<SpeechSegment> {
        if self.current_speech.start > 0 {
            self.current_speech.end = self.current_sample + params.speech_pad_samples;
            if self.current_speech.end.saturating_sub(self.current_speech.start)
                > params.min_speech_samples
            {
                let speech = std::mem::take(&mut self.current_speech);
                self.triggered = false;
                return Some(speech);
            }
        }
        None
    }
}

fn format_timestamp(samples: u32, sample_rate: u32) -> String {
    let total_seconds = samples as f32 / sample_rate as f32;
    let minutes = (total_seconds / 60.0) as u32;
    let seconds = total_seconds % 60.0;
    format!("{:02}:{:05.2}", minutes, seconds)
}

fn print_speech_segment(segment: &SpeechSegment, sample_rate: u32, index: usize) {
    let start_time = format_timestamp(segment.start, sample_rate);
    let end_time = format_timestamp(segment.end, sample_rate);
    let duration = (segment.end - segment.start) as f32 / sample_rate as f32;
    println!(
        "  [{index}] {start_time} - {end_time} (duration: {duration:.2}s, peak: {:.0}%)",
        segment.max_probability * 100.0
    );
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <wav_file_path>", args[0]);
        eprintln!("Example: {} input.wav", args[0]);
        std::process::exit(1);
    }

    let wav_file_path = &args[1];

    let mut vad = TenVad::new("onnx/ten-vad.onnx", TARGET_SAMPLE_RATE)?;
    process_wav_file(wav_file_path, &mut vad)?;

    Ok(())
}

fn process_wav_file(wav_file_path: &str, vad: &mut TenVad) -> anyhow::Result<()> {
    let file = File::open(wav_file_path)?;
    let mut reader = hound::WavReader::new(BufReader::new(file))?;

    let spec = reader.spec();

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

    let total_samples = all_i16_samples.len();
    let total_duration = total_samples as f32 / TARGET_SAMPLE_RATE as f32;

    println!("WAV file: {wav_file_path}");
    println!(
        "Sample rate: {} Hz, Channels: {}, Bits per sample: {}",
        spec.sample_rate, spec.channels, spec.bits_per_sample
    );
    println!("Duration: {:.2}s ({} samples)", total_duration, total_samples);
    println!();

    audio_buffer.append_samples(all_i16_samples);

    // Initialize speech detection
    let params = SpeechParams::new(TARGET_SAMPLE_RATE);
    let mut state = SpeechState::new();
    let mut segments: Vec<SpeechSegment> = Vec::new();

    while let Some(frame) = audio_buffer.pop_frame(HOP_SIZE) {
        match vad.process_frame(&frame) {
            Ok(vad_score) => {
                if let Some(segment) = state.update(&params, vad_score) {
                    segments.push(segment);
                }
            }
            Err(e) => {
                eprintln!("Error running VAD on audio frame: {e}");
            }
        }
    }

    // Finalize any ongoing speech segment
    if let Some(segment) = state.finalize(&params) {
        segments.push(segment);
    }

    // Print results
    if segments.is_empty() {
        println!("No speech detected in the audio file.");
    } else {
        println!("🎤 Detected {} speech segment(s):", segments.len());
        println!();
        for (i, segment) in segments.iter().enumerate() {
            print_speech_segment(segment, TARGET_SAMPLE_RATE, i + 1);
        }

        // Print summary
        let total_speech_duration: f32 = segments
            .iter()
            .map(|s| (s.end - s.start) as f32 / TARGET_SAMPLE_RATE as f32)
            .sum();
        let speech_percentage = (total_speech_duration / total_duration) * 100.0;

        println!();
        println!("Summary:");
        println!(
            "  Total speech: {:.2}s ({:.1}% of audio)",
            total_speech_duration, speech_percentage
        );
        println!(
            "  Average peak probability: {:.0}%",
            segments
                .iter()
                .map(|s| s.max_probability)
                .sum::<f32>()
                / segments.len() as f32
                * 100.0
        );
    }

    Ok(())
}
