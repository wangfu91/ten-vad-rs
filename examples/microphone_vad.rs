use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::thread;
use ten_vad_rs::{AudioFrameBuffer, TenVad};

const HOP_SIZE: usize = 256; // 16ms at 16kHz
const THRESHOLD: f32 = 0.5; // Default threshold for VAD
const TARGET_SAMPLE_RATE: u32 = 16000; // Required sample rate for TEN VAD (16kHz)
const RESAMPLER_CHUNK_SIZE: usize = 1024; // Fixed chunk size for the resampler

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
    println!("Input sample rate: {input_sample_rate} Hz, Channels: {input_channels}");

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
            RESAMPLER_CHUNK_SIZE,
            1,
        )
        .unwrap();
        let mut resampler_output_buffer = resampler.output_buffer_allocate(true);

        let mut audio_resample_buffer = AudioFrameBuffer::new();
        let mut audio_vad_buffer = AudioFrameBuffer::new();

        loop {
            if let Ok(input_f32_samples) = rx.recv() {
                // Check and convert stereo to mono by averaging the channels
                let mono_f32_samples = if input_channels == 1 {
                    input_f32_samples
                } else {
                    let mut mono_samples = vec![0.0; input_f32_samples.len() / input_channels];
                    for (i, &sample) in input_f32_samples.iter().enumerate() {
                        mono_samples[i / input_channels] += sample;
                    }
                    mono_samples
                        .iter_mut()
                        .for_each(|s| *s /= input_channels as f32);
                    mono_samples
                };

                // Resample the audio to the target sample rate
                audio_resample_buffer.append_samples(mono_f32_samples);
                if let Some(resample_frame) = audio_resample_buffer.pop_frame(RESAMPLER_CHUNK_SIZE)
                {
                    let (_, out_len) = resampler
                        .process_into_buffer(&[&resample_frame], &mut resampler_output_buffer, None)
                        .expect("Failed to resample audio");

                    let resampled_f32_samples = &resampler_output_buffer[0][..out_len];

                    let resampled_i16_samples: Vec<i16> = resampled_f32_samples
                        .iter()
                        .map(|&s| {
                            (s * i16::MAX as f32)
                                .round()
                                .clamp(i16::MIN as f32, i16::MAX as f32)
                                as i16
                        })
                        .collect();

                    // Run VAD on the resampled audio
                    audio_vad_buffer.append_samples(resampled_i16_samples);
                    while let Some(frame) = audio_vad_buffer.pop_frame(HOP_SIZE) {
                        // Process each frame of audio data
                        match vad.process_frame(&frame) {
                            Ok(vad_score) => {
                                if vad_score >= THRESHOLD {
                                    println!("++++++ Voice detected: probability {vad_score:2}");
                                } else {
                                    println!("----- No voice detected");
                                }
                            }
                            Err(e) => eprintln!("Error running VAD on audio frame: {e}"),
                        }
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
