use std::f32::consts::PI;
use hound::{WavWriter, WavSpec, SampleFormat};

fn main() {
    create_test_wav("test_speech.wav", 44100, 5.0).unwrap();
    create_test_wav("test_16khz.wav", 16000, 3.0).unwrap();
    println!("Created test WAV files:");
    println!("  test_speech.wav - 44.1kHz, 5 seconds with speech patterns");
    println!("  test_16khz.wav - 16kHz, 3 seconds with speech patterns");
}

fn create_test_wav(filename: &str, sample_rate: u32, duration_secs: f32) -> Result<(), Box<dyn std::error::Error>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(filename, spec)?;
    let total_samples = (sample_rate as f32 * duration_secs) as usize;

    for i in 0..total_samples {
        let t = i as f32 / sample_rate as f32;
        
        // Create speech-like patterns with pauses
        let sample = if (t % 2.0) < 1.5 {
            // Speech segment: mix of frequencies to simulate voice
            let freq1 = 200.0; // Fundamental frequency
            let freq2 = 400.0; // First harmonic
            let freq3 = 800.0; // Second harmonic
            
            let amplitude = if (t % 1.5) < 0.1 || (t % 1.5) > 1.4 {
                0.0 // Brief pauses within speech
            } else {
                0.3 * (t * 10.0).sin() // Amplitude modulation
            };
            
            amplitude * (
                0.5 * (2.0 * PI * freq1 * t).sin() +
                0.3 * (2.0 * PI * freq2 * t).sin() +
                0.2 * (2.0 * PI * freq3 * t).sin()
            )
        } else {
            // Silence segment
            0.0
        };

        let sample_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(sample_i16)?;
    }

    writer.finalize()?;
    Ok(())
}
