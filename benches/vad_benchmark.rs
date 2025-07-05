use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use ten_vad_rs::TenVAD;

fn generate_test_audio(sample_rate: u32, duration_ms: u32) -> Vec<i16> {
    let total_samples = (sample_rate * duration_ms / 1000) as usize;
    let mut samples = Vec::with_capacity(total_samples);

    // Generate a simple sine wave with some noise
    for i in 0..total_samples {
        let t = i as f32 / sample_rate as f32;
        let sine_wave = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        let noise = (i as f32 * 0.1).sin() * 0.1;
        let sample = ((sine_wave + noise) * 16384.0) as i16;
        samples.push(sample);
    }

    samples
}

fn bench_vad_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("vad_creation");

    for hop_size in [128, 256, 512, 1024].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(hop_size),
            hop_size,
            |b, &hop_size| {
                b.iter(|| {
                    let _vad = TenVAD::new(hop_size, 0.5).unwrap();
                });
            },
        );
    }

    group.finish();
}

fn bench_single_frame_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_frame_processing");

    for hop_size in [128, 256, 512, 1024].iter() {
        let vad = TenVAD::new(*hop_size, 0.5).unwrap();
        let test_audio = generate_test_audio(16000, 100); // 100ms of audio
        let frame = &test_audio[0..*hop_size];

        group.bench_with_input(BenchmarkId::from_parameter(hop_size), hop_size, |b, _| {
            b.iter(|| {
                let _result = vad.process_frame(black_box(frame)).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");

    let hop_size = 256;
    let vad = TenVAD::new(hop_size, 0.5).unwrap();

    for duration_ms in [100, 500, 1000, 5000].iter() {
        // Calculate exact number of samples needed for the duration
        let total_samples = (16000 * duration_ms / 1000) as usize;
        // Round down to nearest multiple of hop_size to ensure exact fit
        let aligned_samples = (total_samples / hop_size) * hop_size;
        let test_audio = generate_test_audio(16000, (aligned_samples * 1000 / 16000) as u32);

        group.bench_with_input(
            BenchmarkId::from_parameter(duration_ms),
            duration_ms,
            |b, _| {
                b.iter(|| {
                    let _results = vad.process_frames(black_box(&test_audio)).unwrap();
                });
            },
        );
    }

    group.finish();
}

fn bench_frame_by_frame_vs_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("frame_by_frame_vs_batch");

    let hop_size = 256;
    let vad = TenVAD::new(hop_size, 0.5).unwrap();

    // Generate exactly 1 second of audio, aligned to hop_size
    let total_samples = (16000 / hop_size) * hop_size; // Exactly 1 second, aligned
    let test_audio = generate_test_audio(16000, (total_samples * 1000 / 16000) as u32);

    group.bench_function("frame_by_frame", |b| {
        b.iter(|| {
            let mut results = Vec::new();
            for chunk in test_audio.chunks(hop_size) {
                if chunk.len() == hop_size {
                    let result = vad.process_frame(black_box(chunk)).unwrap();
                    results.push(result);
                }
            }
            black_box(results);
        });
    });

    group.bench_function("batch_processing", |b| {
        b.iter(|| {
            let _results = vad.process_frames(black_box(&test_audio)).unwrap();
        });
    });

    group.finish();
}

fn bench_different_thresholds(c: &mut Criterion) {
    let mut group = c.benchmark_group("different_thresholds");

    let hop_size = 256;
    // Generate 1 second of audio, aligned to hop_size
    let total_samples = (16000 / hop_size) * hop_size;
    let test_audio = generate_test_audio(16000, (total_samples * 1000 / 16000) as u32);

    for threshold in [0.1, 0.3, 0.5, 0.7, 0.9].iter() {
        let vad = TenVAD::new(hop_size, *threshold).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(threshold), threshold, |b, _| {
            b.iter(|| {
                let _results = vad.process_frames(black_box(&test_audio)).unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_vad_creation,
    bench_single_frame_processing,
    bench_batch_processing,
    bench_frame_by_frame_vs_batch,
    bench_different_thresholds
);
criterion_main!(benches);
