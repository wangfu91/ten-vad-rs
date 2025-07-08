# 🎤 ten-vad-rs

[![Crates.io](https://img.shields.io/crates/v/ten-vad-rs.svg)](https://crates.io/crates/ten-vad-rs)
[![Docs.rs](https://docs.rs/ten-vad-rs/badge.svg)](https://docs.rs/ten-vad-rs)
[![License](https://img.shields.io/crates/l/ten-vad-rs.svg)](./LICENSE)

A Rust library for working with the TEN VAD (Voice Activity Detection) ONNX model. Detect speech in audio streams with high accuracy and performance! 🚀

## ✨ Features
- 🎙️ Real-time voice activity detection
- 🦀 Pure Rust API
- 🧠 Powered by ONNX Runtime
- 📦 Easy integration into your audio projects
- 🛠️ Example code for microphone, speaker, and WAV file VAD

## 📦 Installation
Add to your `Cargo.toml`:

```toml
[dependencies]
ten-vad-rs = "0.1.0" # Replace with the latest version
```

## 🚀 Quick Start
Here's a simple example using a WAV file:

```rust
use ten_vad_rs::TenVad;

let mut vad = TenVad::new("onnx/ten-vad.onnx").unwrap();
let speech_segments = vad.process_wav("path/to/audio.wav").unwrap();
for segment in speech_segments {
    println!("Speech from {} to {}", segment.start, segment.end);
}
```

See the [`examples/`](examples/) directory for more advanced usage:
- `wav_file_vad.rs` — Run VAD on a WAV file
- `microphone_vad.rs` — Real-time VAD from microphone
- `speaker_vad.rs` — Real-time VAD from speaker output

## 🛠️ Building
Requires Rust 1.76+ and a working ONNX Runtime environment. Build with:

```sh
cargo build --release
```

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to open an [issue](https://github.com/wangfu91/ten-vad-rs/issues) or submit a pull request.

## 📄 License
Licensed under the [Apache-2.0](./LICENSE) license.

