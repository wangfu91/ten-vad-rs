use std::env;
use std::fs;
use std::io::{BufReader, Write};
use std::path::Path;
use ten_vad_rs::{AudioFrameBuffer, TARGET_SAMPLE_RATE, TenVad};

const HOP_SIZE: usize = 256;

fn parse_scv_to_framewise(path: &Path, hop_size: usize) -> anyhow::Result<Vec<u8>> {
    let content = fs::read_to_string(path)?;
    let line = content.lines().next().unwrap_or("").trim();
    let tokens: Vec<&str> = line.split(',').collect();

    let rest = &tokens[1..];
    let frame_duration = hop_size as f64 / 16000.0;

    let mut labels: Vec<u8> = Vec::new();
    let mut first_start: Option<f64> = None;
    let mut last_end: f64 = 0.0;

    let mut i = 0;
    while i + 2 < rest.len() {
        let start: f64 = rest[i].parse()?;
        let end: f64 = rest[i + 1].parse()?;
        let label: u8 = rest[i + 2].parse()?;

        if first_start.is_none() {
            first_start = Some(start);
        }
        last_end = end;

        let num_frames = ((end - start) / frame_duration).round() as usize;
        labels.extend(std::iter::repeat_n(label, num_frames));

        i += 3;
    }

    let total_frames = ((last_end - first_start.unwrap_or(0.0)) / frame_duration) as usize;
    labels.truncate(total_frames);

    Ok(labels)
}

fn process_wav(path: &Path, vad: &mut TenVad) -> anyhow::Result<Vec<f32>> {
    let file = fs::File::open(path)?;
    let mut reader = hound::WavReader::new(BufReader::new(file))?;
    let spec = reader.spec();

    if spec.sample_rate != TARGET_SAMPLE_RATE {
        return Err(anyhow::anyhow!(
            "Unsupported sample rate: {} Hz, expected {} Hz",
            spec.sample_rate,
            TARGET_SAMPLE_RATE
        ));
    }
    if spec.channels != 1 {
        return Err(anyhow::anyhow!(
            "Unsupported channels: {}, expected 1 (mono)",
            spec.channels
        ));
    }

    let all_samples: Vec<i16> = if spec.sample_format == hound::SampleFormat::Float {
        reader
            .samples::<f32>()
            .map(|s| {
                let s = s.unwrap_or(0.0);
                (s * i16::MAX as f32)
                    .round()
                    .clamp(i16::MIN as f32, i16::MAX as f32) as i16
            })
            .collect()
    } else {
        reader.samples::<i16>().map(|s| s.unwrap_or(0)).collect()
    };

    let mut audio_buffer = AudioFrameBuffer::new();
    audio_buffer.append_samples(all_samples);

    let mut scores = Vec::new();
    while let Some(frame) = audio_buffer.pop_frame(HOP_SIZE) {
        let score = vad.process_frame(&frame)?;
        scores.push(score);
    }

    vad.reset();
    Ok(scores)
}

fn precision_recall(scores: &[f32], labels: &[u8], threshold: f64) -> (f64, f64) {
    let (mut tp, mut fp, mut fn_count) = (0u64, 0u64, 0u64);

    for (&score, &label) in scores.iter().zip(labels.iter()) {
        let predicted = if (score as f64) >= threshold {
            1u8
        } else {
            0u8
        };
        match (predicted, label) {
            (1, 1) => tp += 1,
            (1, 0) => fp += 1,
            (0, 1) => fn_count += 1,
            _ => {}
        }
    }

    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_count > 0 {
        tp as f64 / (tp + fn_count) as f64
    } else {
        0.0
    };

    (precision, recall)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <testset_dir> <onnx_model_path>", args[0]);
        eprintln!("Example: {} ten-vad/testset onnx/ten-vad.onnx", args[0]);
        std::process::exit(1);
    }

    let testset_dir = Path::new(&args[1]);
    let onnx_model_path = &args[2];

    let mut wav_files: Vec<_> = fs::read_dir(testset_dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("wav") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    wav_files.sort();

    println!(
        "Found {} WAV files in {}",
        wav_files.len(),
        testset_dir.display()
    );

    let mut vad = TenVad::new(onnx_model_path, TARGET_SAMPLE_RATE)?;
    let mut all_scores: Vec<f32> = Vec::new();
    let mut all_labels: Vec<u8> = Vec::new();

    for wav_path in &wav_files {
        let scv_path = wav_path.with_extension("scv");
        if !scv_path.exists() {
            eprintln!("Warning: no .scv file for {}, skipping", wav_path.display());
            continue;
        }

        let labels = parse_scv_to_framewise(&scv_path, HOP_SIZE)?;
        let scores = process_wav(wav_path, &mut vad)?;

        let frame_num = labels.len().min(scores.len());
        if frame_num < 2 {
            eprintln!(
                "Warning: too few frames for {}, skipping",
                wav_path.display()
            );
            continue;
        }

        all_scores.extend_from_slice(&scores[1..frame_num]);
        all_labels.extend_from_slice(&labels[..frame_num - 1]);

        println!(
            "Processed {} — {} frames",
            wav_path.file_name().unwrap_or_default().to_string_lossy(),
            frame_num
        );
    }

    println!(
        "\nTotal frames collected: {} scores, {} labels",
        all_scores.len(),
        all_labels.len()
    );

    let output_path = "PR_data_TEN_VAD_RS.txt";
    let mut out_file = fs::File::create(output_path)?;

    println!("\n{:<12} {:<12} {:<12}", "Threshold", "Precision", "Recall");
    println!("{}", "-".repeat(36));

    for i in 0..=100 {
        let threshold = i as f64 / 100.0;
        let (precision, recall) = precision_recall(&all_scores, &all_labels, threshold);
        writeln!(out_file, "{threshold:.2} {precision:.4} {recall:.4}")?;
        println!("{threshold:<12.2} {precision:<12.4} {recall:<12.4}");
    }

    println!("\nPR data written to {output_path}");
    Ok(())
}
