use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::thread;
use ten_vad_rs::{AudioSegment, TenVad};

const HOP_SIZE: usize = 256; // 16ms at 16kHz
const THRESHOLD: f32 = 0.5; // Default threshold for VAD
const TARGET_SAMPLE_RATE: u32 = 16000; // Required sample rate for TEN VAD (16kHz)

fn main() -> anyhow::Result<()> {
    println!("TenVAD Microphone Example");

    let mut vad = TenVad::new("onnx/ten-vad.onnx")?;

    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device found"))?;
    let input_stream_config = input_device
        .default_input_config()
        .map_err(|e| anyhow::anyhow!("Failed to get default input config: {}", e))?;

    let input_sample_rate = input_stream_config.sample_rate().0;
    let input_channels = input_stream_config.channels() as usize;

    println!("Input device: {}", input_device.name()?);
    println!("Sample rate: {input_sample_rate} Hz, Channels: {input_channels}");

    let (tx, rx) = std::sync::mpsc::channel();

    let input_stream = input_device.build_input_stream(
        &input_stream_config.into(),
        move |data: &[f32], _| {
            tx.send(data.to_vec()).unwrap();
        },
        move |err| eprintln!("Input stream error: {err}"),
        None,
    )?;

    input_stream.play()?;

    let join_handle = thread::spawn(move || {
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };

        let mut resampler = SincFixedIn::<f32>::new(
            TARGET_SAMPLE_RATE as f64 / input_sample_rate as f64,
            2.0, // max_relative_ratio
            params,
            480,
            1,
        )
        .unwrap();
        let mut resampler_output_buffer = resampler.output_buffer_allocate(true);

        let mut audio_segment = AudioSegment::new();

        loop {
            if let Ok(f32_samples) = rx.recv() {
                let mono_f32_samples = if input_channels == 1 {
                    f32_samples
                } else {
                    // Convert stereo to mono by averaging the channels
                    let mut mono_samples = vec![0.0; f32_samples.len() / input_channels];
                    for (i, &sample) in f32_samples.iter().enumerate() {
                        mono_samples[i / input_channels] += sample;
                    }
                    mono_samples
                        .iter_mut()
                        .for_each(|s| *s /= input_channels as f32);
                    mono_samples
                };

                let (_, out_len) = resampler
                    .process_into_buffer(&[&mono_f32_samples], &mut resampler_output_buffer, None)
                    .expect("Failed to resample audio");

                let resampled_f32_samples = &resampler_output_buffer[0][..out_len];

                let resampled_i16_samples: Vec<i16> = resampled_f32_samples
                    .iter()
                    .map(|&s| (s * i16::MAX as f32).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16)
                    .collect();

                audio_segment.append_samples(&resampled_i16_samples);

                while let Some(frame) = audio_segment.get_audio_frame(HOP_SIZE) {
                    // Process each frame of audio data
                    match vad.process_frame(&frame) {
                        Ok(vad_score) => {
                            if vad_score >= THRESHOLD {
                                println!(
                                    "++++++ Detected voice in frame: probability {vad_score:2}"
                                );
                            } else {
                                println!("------ No voice detected in frame");
                            }
                        }
                        Err(e) => eprintln!("Error processing frame: {e}"),
                    }
                }
            } else {
                eprintln!("Error receiving audio samples from the channel.");
                break;
            }
        }
    });

    join_handle.join().expect("Thread panicked");

    Ok(())
}
