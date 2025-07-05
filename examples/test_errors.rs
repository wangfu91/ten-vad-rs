use ten_vad_rs::{TenVAD, VadPresets};

fn main() {
    println!("Testing thiserror error formatting...");

    // Test invalid threshold error
    match TenVAD::new(256, -0.5) {
        Ok(_) => println!("Should not reach here"),
        Err(e) => println!("Invalid threshold error: {e}"),
    }

    // Test invalid hop size error
    match TenVAD::new(0, 0.5) {
        Ok(_) => println!("Should not reach here"),
        Err(e) => println!("Invalid hop size error: {e}"),
    }

    // Test successful creation and audio size mismatch
    let vad = TenVAD::with_preset(VadPresets::balanced).unwrap();
    let wrong_size_audio = vec![0i16; 100]; // Wrong size for hop_size 256

    match vad.process_frame(&wrong_size_audio) {
        Ok(_) => println!("Should not reach here"),
        Err(e) => println!("Audio size mismatch error: {e}"),
    }

    println!("All error formatting tests completed successfully!");
}
