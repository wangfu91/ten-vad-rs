use std::f32::consts::PI;
use std::io::BufReader;
use ten_vad_rs::TenVadResult;
use ten_vad_rs::onnx::TenVadOnnx;

fn main() -> anyhow::Result<()> {
    println!("TEN VAD ONNX Inference Demo");
    println!("============================");

    let model_path = r"onnx\ten-vad.onnx";

    match TenVadOnnx::new(model_path, 0.5) {
        Ok(mut vad) => {
            println!("Successfully loaded ONNX model: {model_path}");

            let test_wav_file = r"D:\playground\others\ten-vad\examples\s0724-s0730.wav";
            println!("Processing WAV file: {test_wav_file}");

            let test_wav_file = std::fs::File::open(test_wav_file)?;

            let mut reader = hound::WavReader::new(BufReader::new(test_wav_file))?;

            let audio_data = reader.samples::<i16>().collect::<Result<Vec<i16>, _>>()?;

            // Process the audio in chunks
            let hop_size = 256; // 16ms at 16kHz
            let mut voice_detected_count = 0;
            let mut total_frames = 0;

            for chunk in audio_data.chunks(hop_size) {
                if chunk.len() == hop_size {
                    match vad.process_frame(chunk) {
                        Ok((score, is_voice)) => {
                            total_frames += 1;
                            if is_voice {
                                voice_detected_count += 1;
                            }
                            println!(
                                "Frame {total_frames}: VAD score = {score:.3}, Voice detected = {is_voice}"
                            );
                        }
                        Err(e) => {
                            println!("Error processing frame: {e:?}");
                        }
                    }
                }
            }

            println!("\nSummary:");
            println!("Total frames processed: {total_frames}");
            println!(
                "Voice detected in {} frames ({:.1}%)",
                voice_detected_count,
                100.0 * voice_detected_count as f32 / total_frames as f32
            );

            // Test threshold adjustment
            println!("\nTesting threshold adjustment:");
            println!("Current threshold: {:.3}", vad.get_threshold());

            vad.set_threshold(0.3)?;
            println!("New threshold: {:.3}", vad.get_threshold());

            vad.set_threshold(0.7)?;
            println!("Updated threshold: {:.3}", vad.get_threshold());
        }
        Err(e) => {
            eprintln!("Failed to load ONNX model: {e}");
            return Err(anyhow::anyhow!("Failed to load ONNX model: {e}"));
        }
    }

    Ok(())
}
