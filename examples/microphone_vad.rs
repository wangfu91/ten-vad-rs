use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ten_vad_rs::{AudioSegment, utils};

const HOP_SIZE: usize = 256; // 16ms at 16kHz
const THRESHOLD: f32 = 0.5; // Default threshold for VAD
const TARGET_SAMPLE_RATE: u32 = 16000; // Target sample rate for VAD (16kHz)

fn main() -> anyhow::Result<()> {
    println!("TenVAD Microphone Example");

    // Example usage of TenVAD with a microphone input
    use ten_vad_rs::TenVAD;

    let vad = TenVAD::new(HOP_SIZE, THRESHOLD)?;
    println!("TenVAD Version: {}", TenVAD::get_version());
    println!(
        "Using hop size: {}, threshold: {}",
        vad.hop_size(),
        vad.threshold()
    );

    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device found"))?;
    let input_stream_config = input_device
        .default_input_config()
        .map_err(|e| anyhow::anyhow!("Failed to get default input config: {}", e))?;

    let sample_rate = input_stream_config.sample_rate().0;
    let channels = input_stream_config.channels() as usize;

    println!("Input device: {}", input_device.name()?);
    println!("Sample rate: {sample_rate} Hz, Channels: {channels}");

    let mut audio_segment = AudioSegment::new(HOP_SIZE);

    let input_stream = input_device.build_input_stream(
        &input_stream_config.into(),
        move |data: &[i16], _| {
            let samples =
                preprocess_audio(data, channels as u32, sample_rate, TARGET_SAMPLE_RATE).unwrap();

            audio_segment.append_samples(&samples);

            while let Some(frame) = audio_segment.get_fixed_size_samples() {
                // Process each frame of audio data
                match vad.process_frame(&frame) {
                    Ok(result) => {
                        if result.is_voice {
                            println!(
                                "++++++ Detected voice in frame: probability {}",
                                result.probability,
                            );
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
