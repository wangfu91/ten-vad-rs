use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let onnx_dir = PathBuf::from("onnx");

    let onnx_model_path = onnx_dir.join("ten-vad.onnx");

    // Get the target directory (e.g., target/debug or target/release)
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let target_dir = env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target"));
    let exe_dir = target_dir.join(&profile);

    // Ensure the target directory exists
    fs::create_dir_all(&exe_dir).expect("Failed to create target directory");

    let target_onnx_model_path = exe_dir.join("ten-vad.onnx");
    fs::copy(&onnx_model_path, &target_onnx_model_path).expect("Failed to copy ten-vad.onnx");
}
