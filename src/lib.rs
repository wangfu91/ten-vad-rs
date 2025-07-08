#![allow(clippy::excessive_precision)]

mod audio_segment;
mod error;
pub mod utils;

// Re-export error types for public API
pub use crate::audio_segment::AudioSegment;
pub use crate::error::{TenVadError, TenVadResult};

use ndarray::{Array1, Array2, Axis};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{SessionInputValue, SessionInputs};
use ort::{session::Session, value::TensorRef};
use rustfft::{Fft, FftPlanner, num_complex::Complex32};
use std::{f32::consts::PI, sync::Arc};

const FFT_SIZE: usize = 1024;
const WINDOW_SIZE: usize = 768;
const MEL_FILTER_BANK_NUM: usize = 40;
const FEATURE_LEN: usize = MEL_FILTER_BANK_NUM + 1; // 40 mel features + 1 pitch feature
const CONTEXT_WINDOW_LEN: usize = 3;
const MODEL_HIDDEN_DIM: usize = 64;
const MODEL_IO_NUM: usize = 5;
const EPS: f32 = 1e-20;
const PRE_EMPHASIS_COEFF: f32 = 0.97;

/// Means of input-mel-filterbank (from coeff.h)
#[rustfmt::skip]
const FEATURE_MEANS: [f32; 41] = [
    -8.198236465454e+00, -6.265716552734e+00, -5.483818531036e+00,
    -4.758691310883e+00, -4.417088985443e+00, -4.142892837524e+00,
    -3.912850379944e+00, -3.845927953720e+00, -3.657090425491e+00,
    -3.723418712616e+00, -3.876134157181e+00, -3.843890905380e+00,
    -3.690405130386e+00, -3.756065845490e+00, -3.698696136475e+00,
    -3.650463104248e+00, -3.700468778610e+00, -3.567321300507e+00,
    -3.498900175095e+00, -3.477807044983e+00, -3.458816051483e+00,
    -3.444923877716e+00, -3.401328563690e+00, -3.306261301041e+00,
    -3.278556823730e+00, -3.233250856400e+00, -3.198616027832e+00,
    -3.204526424408e+00, -3.208798646927e+00, -3.257838010788e+00,
    -3.381376743317e+00, -3.534021377563e+00, -3.640867948532e+00,
    -3.726858854294e+00, -3.773730993271e+00, -3.804667234421e+00,
    -3.832901000977e+00, -3.871120452881e+00, -3.990592956543e+00,
    -4.480289459229e+00, 9.235690307617e+01
];

/// Stds of input-mel-filterbank (from coeff.h)
#[rustfmt::skip]
const FEATURE_STDS: [f32; 41] = [
    5.166063785553e+00, 4.977209568024e+00, 4.698895931244e+00,
    4.630621433258e+00, 4.634347915649e+00, 4.641156196594e+00,
    4.640676498413e+00, 4.666367053986e+00, 4.650534629822e+00,
    4.640020847321e+00, 4.637400150299e+00, 4.620099067688e+00,
    4.596316337585e+00, 4.562654972076e+00, 4.554360389709e+00,
    4.566910743713e+00, 4.562489986420e+00, 4.562412738800e+00,
    4.585299491882e+00, 4.600179672241e+00, 4.592845916748e+00,
    4.585922718048e+00, 4.583496570587e+00, 4.626092910767e+00,
    4.626957893372e+00, 4.626289367676e+00, 4.637005805969e+00,
    4.683015823364e+00, 4.726813793182e+00, 4.734289646149e+00,
    4.753227233887e+00, 4.849722862244e+00, 4.869434833527e+00,
    4.884482860565e+00, 4.921327114105e+00, 4.959212303162e+00,
    4.996619224548e+00, 5.044823646545e+00, 5.072216987610e+00,
    5.096439361572e+00, 1.152136917114e+02
];

/// TEN VAD ONNX model wrapper
pub struct TenVad {
    session: Session,                // ONNX session for inference
    hidden_states: Vec<Array2<f32>>, // Vector of 2D arrays: [MODEL_IO_NUM - 1] each [1, MODEL_HIDDEN_DIM]
    feature_buffer: Array2<f32>,     // 2D array: [CONTEXT_WINDOW_LEN, FEATURE_LEN]
    pre_emphasis_prev: f32,          // Previous value for pre-emphasis filtering
    mel_filters: Array2<f32>,        // 2D array: [MEL_FILTER_BANK_NUM, n_bins]
    window: Array1<f32>,             // 1D array: [WINDOW_SIZE]
    fft_instance: Arc<dyn Fft<f32>>, // Cached FFT instance
    fft_buffer: Vec<Complex32>,      // Reusable FFT buffer
}

impl TenVad {
    /// Create a new TenVadOnnx instance with the specified ONNX model path and VAD threshold.
    /// # Arguments
    /// * `onnx_model_path` - Path to the ONNX model file.
    /// # Returns
    /// * A `TenVadResult` containing the initialized `TenVadOnnx` instance or an error.
    pub fn new(onnx_model_path: &str) -> TenVadResult<Self> {
        // Create ONNX session
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .commit_from_file(onnx_model_path)?;

        // Initialize hidden states: Vector of 2D arrays [MODEL_IO_NUM - 1] each [1, MODEL_HIDDEN_DIM]
        let mut hidden_states = Vec::new();
        for _ in 0..MODEL_IO_NUM - 1 {
            hidden_states.push(Array2::zeros((1, MODEL_HIDDEN_DIM)));
        }

        // Initialize feature buffer: 2D array [CONTEXT_WINDOW_LEN, FEATURE_LEN]
        let feature_buffer = Array2::zeros((CONTEXT_WINDOW_LEN, FEATURE_LEN));

        // Initialize pre-emphasis previous value
        let pre_emphasis_prev = 0.0f32;

        // Generate mel filter bank
        let mel_filters = Self::generate_mel_filters();

        // Generate Hann window
        let window = Self::generate_hann_window();

        // Create and cache FFT planner and instance
        let mut fft_planner = FftPlanner::new();
        let fft_instance = fft_planner.plan_fft_forward(FFT_SIZE);
        let fft_buffer = vec![Complex32::new(0.0, 0.0); FFT_SIZE];

        log::debug!("Loaded ONNX model: {onnx_model_path}");

        Ok(Self {
            session,
            hidden_states,
            feature_buffer,
            pre_emphasis_prev,
            mel_filters,
            window,
            fft_instance,
            fft_buffer,
        })
    }

    /// Generate mel filter-bank coefficients(Adapted from aed.cc).
    ///
    /// A mel filter bank is a set of filters used in audio processing to mimic the human ear's perception of sound frequencies.
    /// These filters are spaced according to the mel scale, which is more sensitive to lower frequencies and less sensitive to higher frequencies.
    fn generate_mel_filters() -> Array2<f32> {
        let n_bins = FFT_SIZE / 2 + 1;

        // Generate mel frequency points
        let low_mel = 2595.0f32 * (1.0f32 + 0.0f32 / 700.0f32).log10();
        let high_mel = 2595.0f32 * (1.0f32 + 8000.0f32 / 700.0f32).log10();

        // Create mel points
        let mut mel_points = Vec::new();
        for i in 0..=MEL_FILTER_BANK_NUM + 1 {
            let mel = low_mel + (high_mel - low_mel) * i as f32 / (MEL_FILTER_BANK_NUM + 1) as f32;
            mel_points.push(mel);
        }

        // Convert to Hz
        let mut hz_points = Vec::new();
        for mel in mel_points {
            let hz = 700.0f32 * (10.0f32.powf(mel / 2595.0f32) - 1.0f32);
            hz_points.push(hz);
        }

        // Convert to FFT bin indices
        let mut bin_points = Vec::new();
        for hz in hz_points {
            let bin = ((FFT_SIZE + 1) as f32 * hz / 16000.0f32).floor() as usize;
            bin_points.push(bin);
        }

        // Build mel filter bank as 2D array
        let mut mel_filters = Array2::zeros((MEL_FILTER_BANK_NUM, n_bins));

        for i in 0..MEL_FILTER_BANK_NUM {
            // Left slope
            for j in bin_points[i]..bin_points[i + 1] {
                if j < n_bins {
                    mel_filters[[i, j]] =
                        (j - bin_points[i]) as f32 / (bin_points[i + 1] - bin_points[i]) as f32;
                }
            }

            // Right slope
            for j in bin_points[i + 1]..bin_points[i + 2] {
                if j < n_bins {
                    mel_filters[[i, j]] = (bin_points[i + 2] - j) as f32
                        / (bin_points[i + 2] - bin_points[i + 1]) as f32;
                }
            }
        }

        mel_filters
    }

    /// Generate Hann window coefficients
    fn generate_hann_window() -> Array1<f32> {
        let mut window = Array1::zeros(WINDOW_SIZE);
        for i in 0..WINDOW_SIZE {
            let value = 0.5 * (1.0 - (2.0 * PI * i as f32 / (WINDOW_SIZE - 1) as f32).cos());
            window[i] = value;
        }
        window
    }

    /// Pre-emphasis filtering
    fn pre_emphasis(&mut self, audio_frame: &[f32]) -> Array1<f32> {
        let mut emphasized = Array1::zeros(audio_frame.len());

        // First sample
        emphasized[0] = audio_frame[0] - PRE_EMPHASIS_COEFF * self.pre_emphasis_prev;

        // Remaining samples
        for i in 1..audio_frame.len() {
            emphasized[i] = audio_frame[i] - PRE_EMPHASIS_COEFF * audio_frame[i - 1];
        }

        // Update previous value for next call
        self.pre_emphasis_prev = audio_frame[audio_frame.len() - 1];

        emphasized
    }

    /// Extract features from audio frame
    fn extract_features(&mut self, audio_frame: &[f32]) -> Array1<f32> {
        // Pre-emphasis
        let emphasized = self.pre_emphasis(audio_frame);

        // Zero-padding to window size
        let mut padded = Array1::zeros(WINDOW_SIZE);
        let copy_len = emphasized.len().min(WINDOW_SIZE);
        padded
            .slice_mut(ndarray::s![..copy_len])
            .assign(&emphasized.slice(ndarray::s![..copy_len]));

        // Windowing
        let windowed = &padded * &self.window;

        // FFT - use cached FFT instance and reusable buffer
        self.fft_buffer.clear();
        self.fft_buffer.resize(FFT_SIZE, Complex32::new(0.0, 0.0));

        // Prepare input for FFT (real to complex)
        for i in 0..WINDOW_SIZE.min(FFT_SIZE) {
            self.fft_buffer[i] = Complex32::new(windowed[i], 0.0);
        }

        // Perform FFT using cached instance
        self.fft_instance.process(&mut self.fft_buffer);

        // Compute power spectrum (only positive frequencies)
        let n_bins = FFT_SIZE / 2 + 1;
        let mut power_spectrum = Array1::zeros(n_bins);
        for i in 0..n_bins {
            power_spectrum[i] = self.fft_buffer[i].norm_sqr();
        }

        // Normalization (corresponding to powerNormal = 32768^2 in C++)
        let power_normal = 32768.0f32.powi(2);
        power_spectrum /= power_normal;

        // Mel filter bank features
        let mel_features = self.mel_filters.dot(&power_spectrum);
        let mel_features = mel_features.mapv(|x| (x + EPS).ln());

        // Simple pitch estimation (using 0 here, actual C++ code has complex pitch estimation)
        let pitch_freq = 0.0f32;

        // Combine features
        let mut features = Array1::zeros(FEATURE_LEN);
        features
            .slice_mut(ndarray::s![..MEL_FILTER_BANK_NUM])
            .assign(&mel_features);
        features[MEL_FILTER_BANK_NUM] = pitch_freq;

        // Feature normalization
        for i in 0..FEATURE_LEN {
            features[i] = (features[i] - FEATURE_MEANS[i]) / (FEATURE_STDS[i] + EPS);
        }

        features
    }

    /// Process a single audio frame and return VAD score and decision
    /// # Arguments
    /// * `audio_frame` - A slice of i16 audio samples (e.g., from a microphone)
    /// # Returns
    /// * The VAD score (f32)
    pub fn process_frame(&mut self, audio_frame: &[i16]) -> TenVadResult<f32> {
        // Convert i16 to f32
        let audio_f32: Vec<f32> = audio_frame.iter().map(|&x| x as f32).collect();

        // Extract features
        let features = self.extract_features(&audio_f32);

        // Update feature buffer (sliding window)
        // Shift existing features up and add new features at the end
        if CONTEXT_WINDOW_LEN > 1 {
            // Use a simple loop to shift rows up
            for i in 0..CONTEXT_WINDOW_LEN - 1 {
                // Copy row i+1 to row i
                let src_row = self.feature_buffer.row(i + 1).to_owned();
                self.feature_buffer.row_mut(i).assign(&src_row);
            }
        }
        // Set the last row to new features
        self.feature_buffer
            .row_mut(CONTEXT_WINDOW_LEN - 1)
            .assign(&features);

        // Prepare ONNX inference input
        // Reshape feature buffer, [CONTEXT_WINDOW_LEN, FEATURE_LEN] to [1, CONTEXT_WINDOW_LEN, FEATURE_LEN]
        let input_features = self.feature_buffer.view().insert_axis(Axis(0)); // shape: (1, CONTEXT_WINDOW_LEN, FEATURE_LEN)

        // Build input array directly
        let input_tensors: [SessionInputValue; MODEL_IO_NUM] = [
            // Input features as first input
            SessionInputValue::from(TensorRef::from_array_view(input_features)?),
            // Add hidden states as inputs
            SessionInputValue::from(TensorRef::from_array_view(self.hidden_states[0].view())?),
            SessionInputValue::from(TensorRef::from_array_view(self.hidden_states[1].view())?),
            SessionInputValue::from(TensorRef::from_array_view(self.hidden_states[2].view())?),
            SessionInputValue::from(TensorRef::from_array_view(self.hidden_states[3].view())?),
        ];

        let session_inputs = SessionInputs::ValueArray(input_tensors);

        // Run inference with all inputs
        let outputs = self.session.run(session_inputs)?;

        // Get VAD score from first output (outputs[0])
        let vad_score = outputs[0].try_extract_array::<f32>()?[[0, 0, 0]];

        // Update hidden states with outputs[1], outputs[2], outputs[3], outputs[4]
        for i in 0..MODEL_IO_NUM - 1 {
            let output_tensor = outputs[i + 1].try_extract_array::<f32>()?;
            self.hidden_states[i].assign(&output_tensor);
        }

        Ok(vad_score)
    }

    /// Reset the VAD state
    pub fn reset(&mut self) {
        // Reset hidden states
        for hidden_state in &mut self.hidden_states {
            hidden_state.fill(0.0f32);
        }

        // Reset feature buffer
        self.feature_buffer.fill(0.0f32);

        // Reset pre-emphasis previous value
        self.pre_emphasis_prev = 0.0f32;
    }
}

impl std::fmt::Debug for TenVad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TenVadOnnx")
            .field("session", &"Session")
            .field("hidden_states", &self.hidden_states.len())
            .field("feature_buffer", &self.feature_buffer.shape())
            .field("pre_emphasis_prev", &self.pre_emphasis_prev)
            .field("mel_filters", &self.mel_filters.shape())
            .field("window", &self.window.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_mel_filters() {
        let mel_filters = TenVad::generate_mel_filters();
        assert_eq!(
            mel_filters.shape(),
            &[MEL_FILTER_BANK_NUM, FFT_SIZE / 2 + 1]
        );
    }

    #[test]
    fn test_generate_hann_window() {
        let window = TenVad::generate_hann_window();
        assert_eq!(window.len(), WINDOW_SIZE);
        assert!(window.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }

    #[test]
    fn test_pre_emphasis() {
        let mut vad = TenVad::new("onnx/ten-vad.onnx").unwrap();
        let audio_frame = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let emphasized = vad.pre_emphasis(&audio_frame);
        assert_eq!(emphasized.len(), audio_frame.len());
    }

    #[test]
    fn test_extract_features() {
        let mut vad = TenVad::new("onnx/ten-vad.onnx").unwrap();
        let audio_frame = vec![0.0; WINDOW_SIZE]; // Zero input for simplicity
        let features = vad.extract_features(&audio_frame);
        assert_eq!(features.len(), FEATURE_LEN);
    }
}
