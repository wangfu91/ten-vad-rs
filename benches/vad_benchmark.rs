//! Benchmarks for TEN VAD performance testing
//!
//! Run benchmarks:
//!   cargo bench
//!
//! Save baseline (on master branch):
//!   cargo bench -- --save-baseline master
//!
//! Compare against baseline (on perf branch):
//!   cargo bench -- --baseline master
//!
//! Compare specific benchmark:
//!   cargo bench process_frame -- --baseline master

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::f32::consts::PI;
use ten_vad_rs::{AudioFrameBuffer, TenVad, TARGET_SAMPLE_RATE};

const MODEL_PATH: &str = "onnx/ten-vad.onnx";
const SAMPLE_WAV_PATH: &str = "assets/sample.wav";

/// Load WAV file and return samples as i16
fn load_wav_samples() -> Vec<i16> {
    let reader = hound::WavReader::open(SAMPLE_WAV_PATH).expect("Failed to open WAV file");
    let spec = reader.spec();
    
    assert_eq!(spec.channels, 1, "Expected mono audio");
    assert_eq!(spec.sample_rate, TARGET_SAMPLE_RATE, "Expected 16kHz sample rate");
    assert_eq!(spec.bits_per_sample, 16, "Expected 16-bit audio");
    
    reader
        .into_samples::<i16>()
        .map(|s| s.expect("Failed to read sample"))
        .collect()
}

/// Generate synthetic audio for controlled benchmarks
fn generate_sine_wave(num_samples: usize, frequency: f32) -> Vec<i16> {
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / TARGET_SAMPLE_RATE as f32;
            ((2.0 * PI * frequency * t).sin() * 20000.0) as i16
        })
        .collect()
}

/// Benchmark: Single frame processing
fn bench_process_single_frame(c: &mut Criterion) {
    let mut vad = TenVad::new(MODEL_PATH, TARGET_SAMPLE_RATE).expect("Failed to create VAD");
    
    // Standard frame sizes used in real applications
    let frame_sizes = [160, 256, 320, 480, 512];
    
    let mut group = c.benchmark_group("process_single_frame");
    
    for &size in &frame_sizes {
        let audio_frame = generate_sine_wave(size, 200.0);
        
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{size}_samples")),
            &audio_frame,
            |b, frame| {
                b.iter(|| {
                    vad.reset();
                    vad.process_frame(black_box(frame)).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark: Processing real WAV file (full pipeline)
fn bench_process_wav_file(c: &mut Criterion) {
    let samples = load_wav_samples();
    let frame_size = 256;
    
    c.bench_function("process_wav_file_full", |b| {
        b.iter(|| {
            let mut vad = TenVad::new(MODEL_PATH, TARGET_SAMPLE_RATE).expect("Failed to create VAD");
            let mut buffer = AudioFrameBuffer::new();
            buffer.append_samples(samples.iter().copied());
            
            let mut scores = Vec::new();
            while let Some(frame) = buffer.pop_frame(frame_size) {
                let score = vad.process_frame(black_box(&frame)).unwrap();
                scores.push(score);
            }
            scores
        })
    });
}

/// Benchmark: Sequential frame processing (simulates real-time streaming)
fn bench_sequential_frames(c: &mut Criterion) {
    let samples = load_wav_samples();
    let frame_size = 256;
    
    // Pre-split into frames
    let frames: Vec<Vec<i16>> = samples
        .chunks(frame_size)
        .filter(|chunk| chunk.len() == frame_size)
        .map(|chunk| chunk.to_vec())
        .collect();
    
    let num_frames = frames.len();
    
    let mut group = c.benchmark_group("sequential_frames");
    group.throughput(Throughput::Elements(num_frames as u64));
    
    group.bench_function("process_sequence", |b| {
        b.iter(|| {
            let mut vad = TenVad::new(MODEL_PATH, TARGET_SAMPLE_RATE).expect("Failed to create VAD");
            let mut scores = Vec::with_capacity(num_frames);
            
            for frame in &frames {
                let score = vad.process_frame(black_box(frame)).unwrap();
                scores.push(score);
            }
            scores
        })
    });
    
    group.finish();
}

/// Benchmark: VAD initialization
fn bench_vad_initialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("initialization");
    
    // From file path
    group.bench_function("from_file", |b| {
        b.iter(|| {
            TenVad::new(black_box(MODEL_PATH), TARGET_SAMPLE_RATE).unwrap()
        })
    });
    
    // From bytes (pre-loaded model)
    let model_bytes = std::fs::read(MODEL_PATH).expect("Failed to read model file");
    group.bench_function("from_bytes", |b| {
        b.iter(|| {
            TenVad::new_from_bytes(black_box(&model_bytes), TARGET_SAMPLE_RATE).unwrap()
        })
    });
    
    group.finish();
}

/// Benchmark: Reset operation
fn bench_reset(c: &mut Criterion) {
    let mut vad = TenVad::new(MODEL_PATH, TARGET_SAMPLE_RATE).expect("Failed to create VAD");
    
    // Process some frames first to populate state
    let audio = generate_sine_wave(256, 200.0);
    for _ in 0..10 {
        let _ = vad.process_frame(&audio);
    }
    
    c.bench_function("reset", |b| {
        b.iter(|| {
            vad.reset();
        })
    });
}

/// Benchmark: AudioFrameBuffer operations
fn bench_audio_buffer(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_buffer");
    
    let samples: Vec<i16> = (0..10000).map(|i| (i % 32767) as i16).collect();
    
    // Benchmark append
    group.bench_function("append_1000_samples", |b| {
        b.iter(|| {
            let mut buffer = AudioFrameBuffer::<i16>::new();
            buffer.append_samples(black_box(samples[..1000].iter().copied()));
            buffer
        })
    });
    
    // Benchmark pop_frame
    group.bench_function("pop_frame_256", |b| {
        b.iter_batched(
            || {
                let mut buffer = AudioFrameBuffer::<i16>::new();
                buffer.append_samples(samples.iter().copied());
                buffer
            },
            |mut buffer| {
                let mut frames = Vec::new();
                while let Some(frame) = buffer.pop_frame(256) {
                    frames.push(frame);
                }
                frames
            },
            criterion::BatchSize::SmallInput,
        )
    });
    
    group.finish();
}

/// Benchmark: Different audio content types
fn bench_audio_content_types(c: &mut Criterion) {
    let mut vad = TenVad::new(MODEL_PATH, TARGET_SAMPLE_RATE).expect("Failed to create VAD");
    let frame_size = 256;
    
    let mut group = c.benchmark_group("audio_content");
    
    // Silence
    let silence: Vec<i16> = vec![0; frame_size];
    group.bench_function("silence", |b| {
        b.iter(|| {
            vad.reset();
            vad.process_frame(black_box(&silence)).unwrap()
        })
    });
    
    // Low frequency tone (100 Hz)
    let low_tone = generate_sine_wave(frame_size, 100.0);
    group.bench_function("tone_100hz", |b| {
        b.iter(|| {
            vad.reset();
            vad.process_frame(black_box(&low_tone)).unwrap()
        })
    });
    
    // Speech-like frequency (200 Hz)
    let speech_tone = generate_sine_wave(frame_size, 200.0);
    group.bench_function("tone_200hz", |b| {
        b.iter(|| {
            vad.reset();
            vad.process_frame(black_box(&speech_tone)).unwrap()
        })
    });
    
    // High frequency tone (1000 Hz)
    let high_tone = generate_sine_wave(frame_size, 1000.0);
    group.bench_function("tone_1000hz", |b| {
        b.iter(|| {
            vad.reset();
            vad.process_frame(black_box(&high_tone)).unwrap()
        })
    });
    
    // White noise approximation
    let noise: Vec<i16> = (0..frame_size)
        .map(|i| (((i * 12345 + 6789) % 65536) as i32 - 32768) as i16)
        .collect();
    group.bench_function("noise", |b| {
        b.iter(|| {
            vad.reset();
            vad.process_frame(black_box(&noise)).unwrap()
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_process_single_frame,
    bench_process_wav_file,
    bench_sequential_frames,
    bench_vad_initialization,
    bench_reset,
    bench_audio_buffer,
    bench_audio_content_types,
);

criterion_main!(benches);
