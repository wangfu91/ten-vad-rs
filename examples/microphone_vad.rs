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

// Speech detection parameters (based on Silero VAD approach here: silero-vad/examples/rust-example/src/vad_iter.rs)
const FRAME_SIZE_MS: usize = 16; // Frame size in milliseconds
const MIN_SILENCE_DURATION_MS: usize = 300; // Minimum silence duration to end speech
const SPEECH_PAD_MS: usize = 30; // Padding around speech segments
const MIN_SPEECH_DURATION_MS: usize = 250; // Minimum speech duration to be considered valid
const MAX_SPEECH_DURATION_S: f32 = 30.0; // Maximum speech duration in seconds

// Speech detection structures
#[derive(Debug, Default, Clone)]
struct SpeechSegment {
    start: i64,
    end: i64,
}

#[derive(Debug)]
struct SpeechParams {
    frame_size_samples: usize,
    threshold: f32,
    min_speech_samples: usize,
    max_speech_samples: f32,
    min_silence_samples: usize,
    min_silence_samples_at_max_speech: usize,
}

impl SpeechParams {
    fn new(sample_rate: u32, threshold: f32) -> Self {
        let sample_rate = sample_rate as usize;
        let sr_per_ms = sample_rate / 1000;
        let frame_size_samples = FRAME_SIZE_MS * sr_per_ms;
        let min_speech_samples = sr_per_ms * MIN_SPEECH_DURATION_MS;
        let speech_pad_samples = sr_per_ms * SPEECH_PAD_MS;
        let max_speech_samples = sample_rate as f32 * MAX_SPEECH_DURATION_S
            - frame_size_samples as f32
            - 2.0 * speech_pad_samples as f32;
        let min_silence_samples = sr_per_ms * MIN_SILENCE_DURATION_MS;
        let min_silence_samples_at_max_speech = sr_per_ms * 98;

        Self {
            frame_size_samples,
            threshold,
            min_speech_samples,
            max_speech_samples,
            min_silence_samples,
            min_silence_samples_at_max_speech,
        }
    }
}

#[derive(Debug, Default)]
struct SpeechState {
    current_sample: usize,
    temp_end: usize,
    next_start: usize,
    prev_end: usize,
    triggered: bool,
    current_speech: SpeechSegment,
}

impl SpeechState {
    fn new() -> Self {
        Default::default()
    }

    fn update(&mut self, params: &SpeechParams, speech_prob: f32) -> Option<SpeechSegment> {
        self.current_sample += params.frame_size_samples;

        if speech_prob > params.threshold {
            if self.temp_end != 0 {
                self.temp_end = 0;
                if self.next_start < self.prev_end {
                    self.next_start = self
                        .current_sample
                        .saturating_sub(params.frame_size_samples)
                }
            }
            if !self.triggered {
                self.triggered = true;
                self.current_speech.start =
                    self.current_sample as i64 - params.frame_size_samples as i64;
            }
            return None;
        }

        if self.triggered
            && (self.current_sample as i64 - self.current_speech.start) as f32
                > params.max_speech_samples
        {
            if self.prev_end > 0 {
                self.current_speech.end = self.prev_end as _;
                let speech = self.take_speech();
                if self.next_start < self.prev_end {
                    self.triggered = false
                } else {
                    self.current_speech.start = self.next_start as _;
                }
                self.prev_end = 0;
                self.next_start = 0;
                self.temp_end = 0;
                return Some(speech);
            } else {
                self.current_speech.end = self.current_sample as _;
                let speech = self.take_speech();
                self.prev_end = 0;
                self.next_start = 0;
                self.temp_end = 0;
                self.triggered = false;
                return Some(speech);
            }
        }

        if self.triggered && speech_prob < (params.threshold - 0.15) {
            if self.temp_end == 0 {
                self.temp_end = self.current_sample;
            }
            if self.current_sample.saturating_sub(self.temp_end)
                > params.min_silence_samples_at_max_speech
            {
                self.prev_end = self.temp_end;
            }
            if self.current_sample.saturating_sub(self.temp_end) >= params.min_silence_samples {
                self.current_speech.end = self.temp_end as _;
                if self.current_speech.end - self.current_speech.start
                    > params.min_speech_samples as _
                {
                    let speech = std::mem::take(&mut self.current_speech);
                    self.prev_end = 0;
                    self.next_start = 0;
                    self.temp_end = 0;
                    self.triggered = false;
                    return Some(speech);
                }
            }
        }

        None
    }

    fn take_speech(&mut self) -> SpeechSegment {
        // Take the current speech segment and reset it
        std::mem::take(&mut self.current_speech)
    }

    fn finalize(&mut self) -> Option<SpeechSegment> {
        if self.current_speech.start > 0 {
            self.current_speech.end = self.current_sample as _;
            let speech = self.take_speech();
            self.prev_end = 0;
            self.next_start = 0;
            self.temp_end = 0;
            self.triggered = false;
            Some(speech)
        } else {
            None
        }
    }
}

fn print_speech_segment(segment: &SpeechSegment, sample_rate: u32) {
    let start_time = segment.start as f32 / sample_rate as f32;
    let end_time = segment.end as f32 / sample_rate as f32;
    let duration = end_time - start_time;
    println!("🎤 Speech detected: {start_time:.2}s - {end_time:.2}s (duration: {duration:.2}s)");
}

fn run_audio_processing(
    rx: std::sync::mpsc::Receiver<Vec<f32>>,
    mut vad: TenVad,
    input_sample_rate: u32,
    input_channels: usize,
) -> anyhow::Result<()> {
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
    )?;
    let mut resampler_output_buffer = resampler.output_buffer_allocate(true);

    let mut audio_resample_buffer = AudioFrameBuffer::new();
    let mut audio_vad_buffer = AudioFrameBuffer::new();

    // Initialize speech detection
    let speech_params = SpeechParams::new(TARGET_SAMPLE_RATE, THRESHOLD);
    let mut speech_state = SpeechState::new();

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
            if let Some(resample_frame) = audio_resample_buffer.pop_frame(RESAMPLER_CHUNK_SIZE) {
                let (_, out_len) = resampler.process_into_buffer(
                    &[&resample_frame],
                    &mut resampler_output_buffer,
                    None,
                )?;

                let resampled_f32_samples = &resampler_output_buffer[0][..out_len];

                let resampled_i16_samples: Vec<i16> = resampled_f32_samples
                    .iter()
                    .map(|&s| {
                        (s * i16::MAX as f32)
                            .round()
                            .clamp(i16::MIN as f32, i16::MAX as f32) as i16
                    })
                    .collect();

                // Run VAD on the resampled audio with robust speech detection
                audio_vad_buffer.append_samples(resampled_i16_samples);
                while let Some(frame) = audio_vad_buffer.pop_frame(HOP_SIZE) {
                    // Process each frame of audio data
                    match vad.process_frame(&frame) {
                        Ok(vad_score) => {
                            // Update speech detection state
                            if let Some(speech_segment) =
                                speech_state.update(&speech_params, vad_score)
                            {
                                print_speech_segment(&speech_segment, TARGET_SAMPLE_RATE);
                            }
                        }
                        Err(e) => eprintln!("Error running VAD on audio frame: {e}"),
                    }
                }
            }
        } else {
            eprintln!("Error receiving audio samples from the channel.");
            // Finalize any ongoing speech before breaking
            if let Some(speech_segment) = speech_state.finalize() {
                print_speech_segment(&speech_segment, TARGET_SAMPLE_RATE);
            }
            break;
        }
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    println!("TenVAD Microphone Example with Robust Speech Detection");

    let vad = TenVad::new("onnx/ten-vad.onnx")?;

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
    println!("Listening for speech... (Ctrl+C to stop)");

    let (tx, rx) = std::sync::mpsc::channel();

    let input_stream = input_device.build_input_stream(
        &input_stream_config.into(),
        move |data: &[f32], _| {
            if let Err(e) = tx.send(data.to_vec()) {
                eprintln!("Error sending audio data: {e}");
            }
        },
        move |err| eprintln!("Input stream error: {err}"),
        None,
    )?;

    input_stream.play()?;

    let join_handle = thread::spawn(move || {
        if let Err(e) = run_audio_processing(rx, vad, input_sample_rate, input_channels) {
            eprintln!("Audio processing error: {e}");
        }
    });

    join_handle.join().expect("Thread panicked");

    Ok(())
}
