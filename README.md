# TenVAD-RS

A Rust wrapper for the TenVAD (Voice Activity Detection) library.

## Features

- **Safe Rust API**: Memory-safe wrapper around the C library
- **Frame-by-frame processing**: Process single audio frames
- **Batch processing**: Process multiple frames at once
- **Comprehensive error handling**: Detailed error messages for debugging
- **Thread-safe**: Can be used across threads safely
- **Extensive testing**: Comprehensive test suite included

## Usage

### Basic Example

```rust
use ten_vad_rs::TenVAD;

// Create a VAD instance
let vad = TenVAD::new(256, 0.5)?; // hop_size=256, threshold=0.5

// Process a single frame (256 samples of i16 PCM data)
let audio_data = vec![0i16; 256]; // silence
let result = vad.process_frame(&audio_data)?;

println!("Probability: {:.6}, Is voice: {}", result.probability, result.is_voice);
```

### Batch Processing

```rust
use ten_vad_rs::TenVAD;

let vad = TenVAD::new(256, 0.5)?;

// Process multiple frames at once
let audio_data = vec![0i16; 256 * 3]; // 3 frames of data
let results = vad.process_frames(&audio_data)?;

for (i, result) in results.iter().enumerate() {
    println!("Frame {}: Probability: {:.6}, Is voice: {}", 
             i, result.probability, result.is_voice);
}
```

## API Reference

### `TenVAD::new(hop_size: usize, threshold: f32) -> Result<Self, String>`

Creates a new VAD instance.

- `hop_size`: Number of samples between consecutive analysis frames (e.g., 256)
- `threshold`: VAD detection threshold [0.0, 1.0]. Voice is detected when probability >= threshold

### `process_frame(&self, audio_data: &[i16]) -> Result<VadResult, String>`

Processes a single frame of audio data.

- `audio_data`: Audio samples as i16 PCM data. Length must equal hop_size.
- Returns: `VadResult` with probability and voice detection flag

### `process_frames(&self, audio_data: &[i16]) -> Result<Vec<VadResult>, String>`

Processes multiple frames of audio data.

- `audio_data`: Audio samples as i16 PCM data. Length must be a multiple of hop_size.
- Returns: Vector of `VadResult` for each frame

### `VadResult`

```rust
pub struct VadResult {
    pub probability: f32,  // Voice activity probability [0.0, 1.0]
    pub is_voice: bool,    // true if voice detected, false otherwise
}
```

## Requirements

- Rust 2024 edition
- TenVAD C library (included in `lib/` directory)

## Building

```bash
cargo build
```

## Testing

```bash
cargo test
```

## Examples

The library includes several practical examples:

### WAV File Processing

Process any WAV file with automatic format conversion:

```bash
# Basic usage
cargo run --example wav_file_vad input.wav

# The example automatically handles:
# - Sample rate conversion (any rate → 16kHz)
# - Channel mixing (stereo → mono)
# - Provides detailed speech timing analysis
```

### Threshold Comparison

Compare different threshold values on the same file:

```bash
cargo run --example threshold_comparison test_file.wav
```

### Create Test Files

Generate test WAV files for experimentation:

```bash
# Create test files with different formats
cargo run --example create_test_wav
cargo run --example create_stereo_test
```

## Audio Format Support

TenVAD operates on **16kHz mono audio**. The examples automatically handle:

- **Sample Rate**: Any input rate is resampled to 16kHz using high-quality interpolation
- **Channels**: Stereo/multi-channel audio is mixed down to mono
- **Bit Depth**: Only 16-bit PCM is currently supported
- **Hop Sizes**: Optimized for 160/256 samples (10/16ms frames)

## Requirements

- Rust 2024 edition
- Windows x64 (current build configuration)  
- TenVAD C library (included in `lib/` directory)

## Dependencies

- `hound`: WAV file reading/writing
- `rubato`: High-quality audio resampling

## Building

```bash
cargo build
```

## Testing

```bash
cargo test
```

## Running Examples

```bash
# Basic demo
cargo run

# WAV file analysis  
cargo run --example wav_file_vad your_file.wav

# Threshold comparison
cargo run --example threshold_comparison your_file.wav
```

## C Library Compatibility

This wrapper is compatible with the TenVAD C library and follows the same processing patterns as shown in the original C examples. The key differences from the C API:

1. **Memory Safety**: Automatic resource management with RAII
2. **Error Handling**: Rust's `Result` type instead of error codes
3. **Type Safety**: Proper type checking and validation
4. **Batch Processing**: Convenient batch processing methods

## Performance

The Rust wrapper adds minimal overhead over the C library. The underlying processing is performed by the same C library, with the Rust wrapper providing safe memory management and ergonomic APIs.

## License

This project follows the same license as the original TEN Framework.
