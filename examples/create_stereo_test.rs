use std::f32::consts::PI;
use hound::{WavWriter, WavSpec, SampleFormat};

fn main() {
    create_stereo_test_wav("test_stereo.wav", 22050, 2.0).unwrap();
    println!("Created test_stereo.wav - 22.05kHz, 2 channels (stereo), 2 seconds");
}

fn create_stereo_test_wav(filename: &str, sample_rate: u32, duration_secs: f32) -> Result<(), Box<dyn std::error::Error>> {
    let spec = WavSpec {
        channels: 2, // Stereo
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(filename, spec)?;
    let total_frames = (sample_rate as f32 * duration_secs) as usize;

    for i in 0..total_frames {
        let t = i as f32 / sample_rate as f32;
        
        // Create different content for left and right channels
        let left_sample = if t < 1.0 {
            // Left channel: speech-like pattern
            0.3 * (2.0 * PI * 300.0 * t).sin() * (t * 8.0).sin()
        } else {
            // Left channel: silence
            0.0
        };

        let right_sample = if t > 0.5 {
            // Right channel: different speech-like pattern  
            0.4 * (2.0 * PI * 400.0 * t).sin() * ((t - 0.5) * 6.0).sin()
        } else {
            // Right channel: silence
            0.0
        };

        // Write interleaved stereo samples
        let left_i16 = (left_sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        let right_i16 = (right_sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        
        writer.write_sample(left_i16)?;
        writer.write_sample(right_i16)?;
    }

    writer.finalize()?;
    Ok(())
}
