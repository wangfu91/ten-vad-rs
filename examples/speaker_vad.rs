// This example is for Windows only.
#![cfg(target_os = "windows")]

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ten_vad_rs::{AudioSegment, TenVad, utils};

const HOP_SIZE: usize = 256; // 16ms at 16kHz
const THRESHOLD: f32 = 0.5; // Default threshold for VAD
const TARGET_SAMPLE_RATE: u32 = 16000; // Required sample rate for TEN VAD (16kHz)

fn main() -> anyhow::Result<()> {
    println!("TenVAD Speaker Example");

    let mut vad = TenVad::new("onnx/ten-vad.onnx")?;

    let host = cpal::default_host();
    let output_device = host
        .default_output_device()
        .ok_or_else(|| anyhow::anyhow!("No output device found"))?;
    let output_stream_config = output_device
        .default_output_config()
        .map_err(|e| anyhow::anyhow!("Failed to get default output config: {}", e))?;

    let sample_rate = output_stream_config.sample_rate().0;
    let channels = output_stream_config.channels() as usize;

    println!("Output device: {}", output_device.name()?);
    println!("Sample rate: {sample_rate} Hz, Channels: {channels}");

    let mut audio_segment = AudioSegment::new();

    let input_stream = output_device.build_input_stream(
        &output_stream_config.into(),
        move |data: &[i16], _| {
            let samples =
                preprocess_audio(data, channels as u32, sample_rate, TARGET_SAMPLE_RATE).unwrap();

            audio_segment.append_samples(&samples);

            while let Some(frame) = audio_segment.get_audio_frame(HOP_SIZE) {
                // Process each frame of audio data
                match vad.process_frame(&frame) {
                    Ok(vad_score) => {
                        if vad_score >= THRESHOLD {
                            println!("++++++ Detected voice in frame: probability {vad_score:2}");
                        } else {
                            println!("------ No voice detected in frame");
                        }
                    }
                    Err(e) => eprintln!("Error processing frame: {e}"),
                }
            }
        },
        move |err| eprintln!("Input stream error: {err}"),
        None,
    )?;

    input_stream.play()?;

    std::thread::sleep(std::time::Duration::from_secs(300));

    Ok(())
}

fn preprocess_audio(
    samples: &[i16],
    channels: u32,
    input_sample_rate: u32,
    output_sample_rate: u32,
) -> anyhow::Result<Vec<i16>> {
    // Convert stereo to mono if needed
    let mono_samples = if channels > 1 {
        utils::convert_to_mono(samples, channels)
    } else {
        samples.to_vec()
    };

    // Resample to 16kHz if necessary
    utils::resampling(&mono_samples, input_sample_rate, output_sample_rate).map_err(|e| {
        anyhow::anyhow!(
            "Failed to resample audio from {input_sample_rate}Hz to {output_sample_rate}Hz: {e}"
        )
    })
}
