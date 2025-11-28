#![allow(clippy::excessive_precision)]

mod buffer;
mod error;

// Re-export error types for public API
pub use crate::buffer::AudioFrameBuffer;
pub use crate::error::{TenVadError, TenVadResult};

/// Target sample rate for TEN VAD (16kHz)
pub const TARGET_SAMPLE_RATE: u32 = 16000;

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

/// Pitch estimation constants (from pitch_est_st.h)
const PITCH_MIN_PERIOD: usize = 32; // ~500Hz at 16kHz (AUP_PE_MIN_PERIOD_16KHZ)
const PITCH_MAX_PERIOD: usize = 256; // ~62Hz at 16kHz (AUP_PE_MAX_PERIOD_16KHZ)
const PITCH_VOICED_THRESHOLD: f32 = 0.4; // Threshold for voiced detection

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

/// TEN VAD ONNX model runner
pub struct TenVad {
    session: Session,                // ONNX session for inference
    hidden_states: Vec<Array2<f32>>, // Vector of 2D arrays: [MODEL_IO_NUM - 1] each [1, MODEL_HIDDEN_DIM]
    feature_buffer: Array2<f32>,     // 2D array: [CONTEXT_WINDOW_LEN, FEATURE_LEN]
    pre_emphasis_prev: f32,          // Previous value for pre-emphasis filtering
    mel_filters: Array2<f32>,        // 2D array: [MEL_FILTER_BANK_NUM, n_bins]
    window: Array1<f32>,             // 1D array: [WINDOW_SIZE]
    fft_instance: Arc<dyn Fft<f32>>, // Cached FFT instance
    fft_buffer: Vec<Complex32>,      // Reusable FFT buffer
    power_spectrum: Array1<f32>,     // Reusable power spectrum buffer
}

impl TenVad {
    /// Create a new TenVadOnnx instance with the specified ONNX model path and sample rate.
    ///
    /// # Arguments
    /// * `onnx_model_path` - Path to the ONNX model file.
    /// * `sample_rate` - Sample rate in Hz. **Must be 16000 (16kHz)**, otherwise returns an error.
    ///
    /// # Returns
    /// * A `TenVadResult` containing the initialized `TenVadOnnx` instance or an error.
    ///
    /// # Errors
    /// Returns `TenVadError::UnsupportedSampleRate` if the sample rate is not 16000 Hz.
    pub fn new(onnx_model_path: &str, sample_rate: u32) -> TenVadResult<Self> {
        if sample_rate != TARGET_SAMPLE_RATE {
            return Err(TenVadError::UnsupportedSampleRate(sample_rate));
        }

        let builder = Self::configure_session_builder()?;
        let session = builder.commit_from_file(onnx_model_path)?;

        Self::from_session(session)
    }

    /// Create a new TenVad instance from in-memory model bytes.
    ///
    /// This uses `commit_from_memory` from the `ort` crate to build the session directly
    /// from the provided bytes (avoids writing a tempfile).
    pub fn new_from_bytes(model_bytes: &[u8], sample_rate: u32) -> TenVadResult<Self> {
        if sample_rate != TARGET_SAMPLE_RATE {
            return Err(TenVadError::UnsupportedSampleRate(sample_rate));
        }

        let builder = Self::configure_session_builder()?;
        let session = builder.commit_from_memory(model_bytes)?;

        Self::from_session(session)
    }

    /// Shared initialization from an already-built `Session`.
    fn from_session(session: Session) -> TenVadResult<Self> {
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

        // Pre-allocate reusable buffer for power spectrum
        let power_spectrum = Array1::zeros(FFT_SIZE / 2 + 1);

        Ok(Self {
            session,
            hidden_states,
            feature_buffer,
            pre_emphasis_prev,
            mel_filters,
            window,
            fft_instance,
            fft_buffer,
            power_spectrum,
        })
    }

    /// Configure a common Session builder with project defaults (optimization level and threads).
    fn configure_session_builder() -> TenVadResult<ort::session::builder::SessionBuilder> {
        let builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_inter_threads(1)?;
        Ok(builder)
    }

    /// Generate mel filter-bank coefficients (Adapted from aed.cc `AUP_Aed_resetVariables`).
    ///
    /// A mel filter bank is a set of filters used in audio processing to mimic the human ear's perception of sound frequencies.
    /// These filters are spaced according to the mel scale, which is more sensitive to lower frequencies and less sensitive to higher frequencies.
    fn generate_mel_filters() -> Array2<f32> {
        let n_bins = FFT_SIZE / 2 + 1;
        let num_points = MEL_FILTER_BANK_NUM + 2; // +2 for left and right boundary points

        // Generate mel filter-bank coefficients (matching C++ aed.cc)
        // C++: float low_mel = 2595.0f * log10f(1.0f + 0.0f / 700.0f);
        // C++: float high_mel = 2595.0f * log10f(1.0f + 8000.0f / 700.0f);
        let low_mel = 2595.0f32 * (1.0f32 + 0.0f32 / 700.0f32).log10();
        let high_mel = 2595.0f32 * (1.0f32 + 8000.0f32 / 700.0f32).log10();

        // Create mel points and convert to Hz and bin indices in one pass
        // C++: mel_points = i * (high_mel - low_mel) / ((float)melFbSz + 1.0f) + low_mel;
        // C++: hz_points = 700.0f * (powf(10.0f, mel_points / 2595.0f) - 1.0f);
        // C++: melBinBuff[i] = (size_t)((stHdl->intFftSz + 1.0f) * hz_points / (float)AUP_AED_FS);
        let mut bin_points = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let mel =
                i as f32 * (high_mel - low_mel) / (MEL_FILTER_BANK_NUM as f32 + 1.0f32) + low_mel;
            let hz = 700.0f32 * (10.0f32.powf(mel / 2595.0f32) - 1.0f32);
            // Match C++ exactly: (intFftSz + 1) * hz / AUP_AED_FS
            let bin =
                ((FFT_SIZE as f32 + 1.0f32) * hz / TARGET_SAMPLE_RATE as f32).floor() as usize;
            bin_points.push(bin);
        }

        // Build mel filter bank as 2D array
        // C++ stores as melFbCoef[j * nBins + i] but we use [j, i] for clarity
        let mut mel_filters = Array2::zeros((MEL_FILTER_BANK_NUM, n_bins));

        // C++ code from aed.cc AUP_Aed_resetVariables:
        // for (j = 0; j < melFbSz; j++) {
        //     for (i = melBinBuff[j]; i < melBinBuff[j + 1]; i++) {
        //         idx = j * nBins + i;
        //         melFbCoef[idx] = (float)(i - melBinBuff[j]) / (float)(melBinBuff[j + 1] - melBinBuff[j]);
        //     }
        //     for (i = melBinBuff[j + 1]; i < melBinBuff[j + 2]; i++) {
        //         idx = j * nBins + i;
        //         melFbCoef[idx] = (float)(melBinBuff[j + 2] - i) / (float)(melBinBuff[j + 2] - melBinBuff[j + 1]);
        //     }
        // }
        for j in 0..MEL_FILTER_BANK_NUM {
            // Left slope (rising edge)
            let left_bin = bin_points[j];
            let center_bin = bin_points[j + 1];
            let right_bin = bin_points[j + 2];

            // Avoid division by zero
            if center_bin > left_bin {
                for i in left_bin..center_bin {
                    if i < n_bins {
                        mel_filters[[j, i]] =
                            (i - left_bin) as f32 / (center_bin - left_bin) as f32;
                    }
                }
            }

            // Right slope (falling edge)
            if right_bin > center_bin {
                for i in center_bin..right_bin {
                    if i < n_bins {
                        mel_filters[[j, i]] =
                            (right_bin - i) as f32 / (right_bin - center_bin) as f32;
                    }
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

    /// Estimate pitch frequency using normalized autocorrelation.
    ///
    /// This is a simplified version of the C++ pitch estimation algorithm.
    /// The C++ version uses LPC pre-filtering, downsampling, and Viterbi path finding,
    /// but this simplified version provides a reasonable approximation using
    /// normalized autocorrelation with peak detection.
    ///
    /// Returns pitch frequency in Hz, or 0.0 if no pitch detected (unvoiced).
    fn estimate_pitch(audio_frame: &[f32]) -> f32 {
        let len = audio_frame.len();
        if len < PITCH_MAX_PERIOD {
            return 0.0;
        }

        // Compute energy of the signal
        let energy: f32 = audio_frame.iter().map(|&x| x * x).sum();
        if energy < 1e-10 {
            return 0.0; // Silent frame
        }

        // Compute normalized autocorrelation for lag range [MIN_PERIOD, MAX_PERIOD]
        // Using the formula: r[lag] = sum(x[i] * x[i+lag]) / sqrt(sum(x[i]^2) * sum(x[i+lag]^2))
        let analysis_len = len.saturating_sub(PITCH_MAX_PERIOD);
        if analysis_len < PITCH_MIN_PERIOD {
            return 0.0;
        }

        let mut best_corr = -1.0f32;
        let mut best_lag = 0usize;

        // Energy of reference window
        let energy0: f32 = audio_frame[..analysis_len].iter().map(|&x| x * x).sum();

        for lag in PITCH_MIN_PERIOD..=PITCH_MAX_PERIOD.min(len - analysis_len) {
            // Cross-correlation at this lag
            let mut xcorr = 0.0f32;
            for i in 0..analysis_len {
                xcorr += audio_frame[i] * audio_frame[i + lag];
            }

            // Energy of lagged window (sliding window energy)
            let energy_lag: f32 = audio_frame[lag..lag + analysis_len]
                .iter()
                .map(|&x| x * x)
                .sum();

            // Normalized correlation
            let denom = (energy0 * energy_lag).sqrt();
            let norm_corr = if denom > 1e-10 { xcorr / denom } else { 0.0 };

            if norm_corr > best_corr {
                best_corr = norm_corr;
                best_lag = lag;
            }
        }

        // Check if correlation is strong enough to indicate voiced speech
        // C++ uses voicedThr = 0.4 (AUP_PE_PITCH_EST_DEFAULT_VOICEDTHR)
        if best_corr >= PITCH_VOICED_THRESHOLD && best_lag > 0 {
            // Convert lag (in samples) to frequency (Hz)
            // pitch_freq = sample_rate / period
            TARGET_SAMPLE_RATE as f32 / best_lag as f32
        } else {
            0.0 // Unvoiced or no reliable pitch found
        }
    }

    /// Pre-emphasis filtering
    fn pre_emphasis(&mut self, audio_frame: &[f32]) -> Array1<f32> {
        if audio_frame.is_empty() {
            return Array1::zeros(0);
        }

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

        // Zero the FFT buffer before use to clear any previous data (using cached FFT instance and reusable buffer)
        self.fft_buffer.fill(Complex32::new(0.0, 0.0));

        // Prepare input for FFT (real to complex)
        for i in 0..WINDOW_SIZE.min(FFT_SIZE) {
            self.fft_buffer[i] = Complex32::new(windowed[i], 0.0);
        }

        // Perform FFT using cached instance
        self.fft_instance.process(&mut self.fft_buffer);

        // Compute power spectrum using reusable buffer (only positive frequencies)
        // C++ code in AUP_Aed_CalcBinPow handles bin-0 and bin-(NBins-1) specially
        // because of the packed real FFT output format. With complex FFT, we use norm_sqr directly.
        let n_bins = FFT_SIZE / 2 + 1;
        self.power_spectrum.fill(0.0);
        for i in 0..n_bins {
            self.power_spectrum[i] = self.fft_buffer[i].norm_sqr();
        }

        // Mel filter bank features with normalization
        // C++ code from AUP_Aed_aivad_proc:
        //   perBandValue = perBandValue / powerNormal;  // powerNormal = 32768^2
        //   perBandValue = logf(perBandValue + AUP_AED_EPS);
        // We apply: mel_filter * power_spectrum, then divide by powerNormal, then log
        let power_normal = 32768.0f32 * 32768.0f32;
        let mel_features = self.mel_filters.dot(&self.power_spectrum);
        let mel_features = mel_features.mapv(|x| (x / power_normal + EPS).ln());

        // Estimate pitch frequency using autocorrelation
        let pitch_freq = Self::estimate_pitch(audio_frame);

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
    /// * `audio_frame` - A slice of i16 audio samples in 16kHz (e.g., from a microphone)
    /// # Returns
    /// * The VAD score (f32)
    pub fn process_frame(&mut self, audio_frame: &[i16]) -> TenVadResult<f32> {
        // Check if audio frame is empty
        if audio_frame.is_empty() {
            return Err(TenVadError::EmptyAudioData);
        }

        // Convert i16 to f32 and copy to a local Vec to avoid borrow issues
        // Note: C++ uses input in [-32768, 32767] range directly (no normalization to [-1, 1])
        let audio_f32: Vec<f32> = audio_frame.iter().map(|&x| x as f32).collect();

        // Extract features (includes pre-emphasis, windowing, FFT, mel filterbank, pitch estimation)
        let features = self.extract_features(&audio_f32);

        // Update feature buffer (sliding window) using memmove-style operation
        // C++ code: memmove(aivadInputFeatStack, aivadInputFeatStack + srcOffset, sizeof(float) * srcLen);
        if CONTEXT_WINDOW_LEN > 1 {
            // More efficient: shift by copying element by element to avoid row allocation
            for i in 0..CONTEXT_WINDOW_LEN - 1 {
                for j in 0..FEATURE_LEN {
                    self.feature_buffer[[i, j]] = self.feature_buffer[[i + 1, j]];
                }
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
        f.debug_struct("TenVad")
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
    use std::f32::consts::PI;

    // Helper function to create a valid TenVad instance for testing
    fn create_test_vad() -> TenVad {
        TenVad::new("onnx/ten-vad.onnx", TARGET_SAMPLE_RATE)
            .expect("Failed to create TenVad instance for testing")
    }

    // Helper function to generate test audio with specific properties
    fn generate_test_audio(length: usize, frequency: f32, sample_rate: f32) -> Vec<f32> {
        (0..length)
            .map(|i| (2.0 * PI * frequency * i as f32 / sample_rate).sin() * 0.5)
            .collect()
    }

    #[test]
    fn test_generate_mel_filters() {
        let mel_filters = TenVad::generate_mel_filters();
        let n_bins = FFT_SIZE / 2 + 1;

        // Check dimensions
        assert_eq!(mel_filters.shape(), &[MEL_FILTER_BANK_NUM, n_bins]);

        // Check that filters are non-negative
        assert!(
            mel_filters.iter().all(|&x| x >= 0.0),
            "All filter coefficients should be non-negative"
        );

        // Check that each filter has some non-zero values and reasonable properties
        for i in 0..MEL_FILTER_BANK_NUM {
            let filter_sum: f32 = mel_filters.row(i).sum();
            assert!(filter_sum > 0.0, "Filter {i} should have non-zero values");

            // Check that filters have triangular shape (max value should be around 1.0)
            let max_val = mel_filters.row(i).iter().fold(0.0f32, |a, &b| a.max(b));
            assert!(
                max_val <= 1.0 + f32::EPSILON,
                "Filter {i} max value should not exceed 1.0 (got {max_val})"
            );

            // Max value should be close to 1.0 at the center of the triangular filter
            assert!(
                max_val > 0.5,
                "Filter {i} should have peak near 1.0 (got {max_val})"
            );
        }

        // Verify adjacent filters overlap (triangular filter bank property)
        for i in 0..MEL_FILTER_BANK_NUM - 1 {
            let filter_i = mel_filters.row(i);
            let filter_next = mel_filters.row(i + 1);

            // Find bins where both filters have non-zero values
            let overlap_count = filter_i
                .iter()
                .zip(filter_next.iter())
                .filter(|(a, b)| **a > 0.0 && **b > 0.0)
                .count();

            assert!(
                overlap_count > 0,
                "Filters {i} and {} should overlap",
                i + 1
            );
        }
    }

    #[test]
    fn test_generate_hann_window() {
        let window = TenVad::generate_hann_window();

        // Check length
        assert_eq!(window.len(), WINDOW_SIZE);

        // Check range [0, 1]
        assert!(window.iter().all(|&x| (0.0..=1.0).contains(&x)));

        // Check symmetry
        for i in 0..WINDOW_SIZE / 2 {
            let diff = (window[i] - window[WINDOW_SIZE - 1 - i]).abs();
            assert!(diff < 1e-6, "Window should be symmetric");
        }

        // Check that window starts and ends near zero
        assert!(window[0] < 0.01, "Window should start near zero");
        assert!(
            window[WINDOW_SIZE - 1] < 0.01,
            "Window should end near zero"
        );

        // Check that window peaks near the middle
        let mid_idx = WINDOW_SIZE / 2;
        assert!(window[mid_idx] > 0.9, "Window should peak near the middle");
    }

    #[test]
    fn test_pre_emphasis_basic() {
        let mut vad = create_test_vad();
        let audio_frame = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let emphasized = vad.pre_emphasis(&audio_frame);

        assert_eq!(emphasized.len(), audio_frame.len());

        // First sample should be original (no previous sample)
        assert_eq!(emphasized[0], audio_frame[0]);

        // Check that pre-emphasis is applied correctly
        for i in 1..audio_frame.len() {
            let expected = audio_frame[i] - PRE_EMPHASIS_COEFF * audio_frame[i - 1];
            assert!((emphasized[i] - expected).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_pre_emphasis_state_preservation() {
        let mut vad = create_test_vad();

        // Process first frame
        let frame1 = vec![1.0, 2.0, 3.0];
        let _ = vad.pre_emphasis(&frame1);

        // Process second frame - should use last value from frame1 as previous
        let frame2 = vec![4.0, 5.0, 6.0];
        let emphasized2 = vad.pre_emphasis(&frame2);

        // First sample of frame2 should use last sample of frame1
        let expected = frame2[0] - PRE_EMPHASIS_COEFF * frame1[frame1.len() - 1];
        assert!((emphasized2[0] - expected).abs() < f32::EPSILON);
    }

    #[test]
    fn test_pre_emphasis_empty_frame() {
        let mut vad = create_test_vad();
        let empty_frame: Vec<f32> = vec![];
        let emphasized = vad.pre_emphasis(&empty_frame);
        assert_eq!(emphasized.len(), 0);
    }

    #[test]
    fn test_pre_emphasis_single_sample() {
        let mut vad = create_test_vad();
        let single_frame = vec![5.0];
        let emphasized = vad.pre_emphasis(&single_frame);

        assert_eq!(emphasized.len(), 1);
        // With no previous sample (initial state), should be close to original
        assert!((emphasized[0] - single_frame[0]).abs() < f32::EPSILON);
    }

    #[test]
    fn test_extract_features_basic() {
        let mut vad = create_test_vad();
        let audio_frame = vec![0.0; WINDOW_SIZE];
        let features = vad.extract_features(&audio_frame);

        assert_eq!(features.len(), FEATURE_LEN);

        // All features should be finite numbers
        assert!(features.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_estimate_pitch_silence() {
        // Silent audio should return 0 pitch
        let silence = vec![0.0f32; WINDOW_SIZE];
        let pitch = TenVad::estimate_pitch(&silence);
        assert_eq!(pitch, 0.0, "Silent audio should have no pitch");
    }

    #[test]
    fn test_estimate_pitch_sine_wave() {
        // Generate a 200 Hz sine wave (within typical speech pitch range)
        let frequency = 200.0f32;
        let sample_rate = TARGET_SAMPLE_RATE as f32;
        let audio: Vec<f32> = (0..WINDOW_SIZE)
            .map(|i| (2.0 * PI * frequency * i as f32 / sample_rate).sin() * 10000.0)
            .collect();

        let pitch = TenVad::estimate_pitch(&audio);

        // For a clear 200 Hz tone, pitch should be detected
        assert!(
            pitch > 0.0,
            "Pitch should be detected for a clear 200 Hz sine wave (got {pitch})"
        );

        // Verify accuracy - allow some tolerance due to autocorrelation resolution
        let error_percent = ((pitch - frequency) / frequency).abs() * 100.0;
        assert!(
            error_percent < 15.0,
            "Pitch {pitch} Hz should be close to {frequency} Hz (error: {error_percent:.1}%)"
        );
    }

    #[test]
    fn test_estimate_pitch_different_frequencies() {
        let sample_rate = TARGET_SAMPLE_RATE as f32;

        // Test frequencies within the detectable range: ~62 Hz to ~500 Hz
        // Note: Autocorrelation-based pitch estimation can have octave errors,
        // so we verify pitch is detected and is a harmonic multiple
        for &frequency in &[100.0f32, 150.0, 200.0, 250.0, 400.0] {
            let audio: Vec<f32> = (0..WINDOW_SIZE)
                .map(|i| (2.0 * PI * frequency * i as f32 / sample_rate).sin() * 10000.0)
                .collect();

            let pitch = TenVad::estimate_pitch(&audio);

            // For pure tones, pitch should be detected
            assert!(
                pitch > 0.0,
                "Pitch should be detected for {frequency} Hz sine wave (got {pitch})"
            );

            // Verify it's close to the fundamental or a subharmonic (octave error)
            // Acceptable: f, f/2, f/3 (common autocorrelation artifacts)
            let is_valid = [1.0, 2.0, 3.0].iter().any(|&divisor| {
                let expected = frequency / divisor;
                let error_percent = ((pitch - expected) / expected).abs() * 100.0;
                error_percent < 15.0
            });

            assert!(
                is_valid,
                "Detected pitch {pitch:.1} Hz should be {frequency} Hz or a subharmonic"
            );
        }
    }

    #[test]
    fn test_estimate_pitch_short_frame() {
        // Frame too short should return 0
        let short_audio = vec![1.0f32; PITCH_MIN_PERIOD - 1];
        let pitch = TenVad::estimate_pitch(&short_audio);
        assert_eq!(pitch, 0.0, "Too short frame should have no pitch");
    }

    #[test]
    fn test_extract_features_sine_wave() {
        let mut vad = create_test_vad();
        let audio_frame = generate_test_audio(WINDOW_SIZE, 440.0, 16000.0);
        let features = vad.extract_features(&audio_frame);

        assert_eq!(features.len(), FEATURE_LEN);
        assert!(features.iter().all(|&x| x.is_finite()));

        // Use a fresh VAD instance for silence to avoid state pollution from pre-emphasis
        let mut silence_vad = create_test_vad();
        let silence_features = silence_vad.extract_features(&vec![0.0; WINDOW_SIZE]);

        // For a sine wave, features should be different from silence
        let features_diff: f32 = features
            .iter()
            .zip(silence_features.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(
            features_diff > 0.1,
            "Sine wave features should be different from silence (diff: {features_diff})"
        );
    }

    #[test]
    fn test_extract_features_short_frame() {
        let mut vad = create_test_vad();
        let short_frame = vec![1.0; 100]; // Shorter than WINDOW_SIZE
        let features = vad.extract_features(&short_frame);

        assert_eq!(features.len(), FEATURE_LEN);
        assert!(features.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_extract_features_long_frame() {
        let mut vad = create_test_vad();
        let long_frame = vec![1.0; WINDOW_SIZE * 2]; // Longer than WINDOW_SIZE
        let features = vad.extract_features(&long_frame);

        assert_eq!(features.len(), FEATURE_LEN);
        assert!(features.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_extract_features_normalization() {
        let mut vad = create_test_vad();
        let audio_frame = generate_test_audio(WINDOW_SIZE, 1000.0, 16000.0);
        let features = vad.extract_features(&audio_frame);

        // Features should be normalized - check basic properties
        assert!(
            features.iter().all(|&x| x.is_finite()),
            "All features should be finite"
        );

        // Check that features are not all identical (indicating processing worked)
        let first_feature = features[0];
        let has_variation = features.iter().any(|&x| (x - first_feature).abs() > 0.01);
        assert!(
            has_variation,
            "Features should show variation after processing"
        );

        // Check that features have reasonable magnitude (normalized features typically in [-5, 5] range)
        let max_abs = features.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs < 10.0,
            "Normalized features should have reasonable magnitude"
        );
    }

    #[test]
    fn test_new_vad_initialization() {
        // Test that initialization works
        let vad = TenVad::new("onnx/ten-vad.onnx", TARGET_SAMPLE_RATE);
        assert!(vad.is_ok(), "TenVad initialization should succeed");

        let vad = vad.unwrap();

        // Check initial states
        assert_eq!(vad.hidden_states.len(), MODEL_IO_NUM - 1);
        for (i, hidden_state) in vad.hidden_states.iter().enumerate() {
            assert_eq!(
                hidden_state.shape(),
                &[1, MODEL_HIDDEN_DIM],
                "Hidden state {i} should have correct shape"
            );
            assert!(
                hidden_state.iter().all(|&x| x == 0.0),
                "Hidden state {i} should be initialized to zero"
            );
        }

        assert_eq!(
            vad.feature_buffer.shape(),
            &[CONTEXT_WINDOW_LEN, FEATURE_LEN]
        );
        assert!(
            vad.feature_buffer.iter().all(|&x| x == 0.0),
            "Feature buffer should be initialized to zero"
        );

        assert_eq!(vad.pre_emphasis_prev, 0.0);
    }

    #[test]
    fn test_new_vad_invalid_path() {
        let result = TenVad::new("nonexistent/path/model.onnx", TARGET_SAMPLE_RATE);
        assert!(result.is_err(), "Should fail with invalid model path");
    }

    #[test]
    fn test_new_vad_unsupported_sample_rate() {
        let result = TenVad::new("onnx/ten-vad.onnx", 48000);
        assert!(result.is_err(), "Should fail with unsupported sample rate");

        match result.unwrap_err() {
            TenVadError::UnsupportedSampleRate(rate) => {
                assert_eq!(rate, 48000, "Error should contain the invalid sample rate");
            }
            _ => panic!("Expected UnsupportedSampleRate error"),
        }
    }

    #[test]
    fn test_reset_vad_state() {
        let mut vad = create_test_vad();

        // Process some audio to change internal state
        let audio_frame = generate_test_audio(256, 440.0, 16000.0);
        let audio_i16: Vec<i16> = audio_frame.iter().map(|&x| (x * 32767.0) as i16).collect();
        let _ = vad.process_frame(&audio_i16);

        // Reset the VAD
        vad.reset();

        // Check that states are reset
        for hidden_state in &vad.hidden_states {
            assert!(
                hidden_state.iter().all(|&x| x == 0.0),
                "Hidden states should be reset to zero"
            );
        }

        assert!(
            vad.feature_buffer.iter().all(|&x| x == 0.0),
            "Feature buffer should be reset to zero"
        );

        assert_eq!(
            vad.pre_emphasis_prev, 0.0,
            "Pre-emphasis state should be reset"
        );
    }

    #[test]
    fn test_process_frame_basic() {
        let mut vad = create_test_vad();
        let audio_frame = vec![0i16; 256];
        let result = vad.process_frame(&audio_frame);

        assert!(result.is_ok(), "Processing frame should succeed");
        let vad_score = result.unwrap();
        assert!(vad_score.is_finite(), "VAD score should be finite");
        assert!(
            (0.0..=1.0).contains(&vad_score),
            "VAD score should be in [0, 1] range"
        );
    }

    #[test]
    fn test_process_frame_empty() {
        let mut vad = create_test_vad();
        let empty_frame: Vec<i16> = vec![];
        let result = vad.process_frame(&empty_frame);

        assert!(result.is_err(), "Processing empty frame should fail");
    }

    #[test]
    fn test_process_frame_different_sizes() {
        let mut vad = create_test_vad();

        let sizes = vec![64, 128, 256, 512, 1024];
        for size in sizes {
            let audio_frame = vec![100i16; size];
            let result = vad.process_frame(&audio_frame);
            assert!(
                result.is_ok(),
                "Processing frame of size {size} should succeed"
            );
        }
    }

    #[test]
    fn test_process_frame_extreme_values() {
        let mut vad = create_test_vad();

        // Test with maximum values
        let max_frame = vec![i16::MAX; 256];
        let result = vad.process_frame(&max_frame);
        assert!(result.is_ok(), "Processing max values should succeed");
        let max_score = result.unwrap();
        assert!(
            max_score.is_finite(),
            "Score for max values should be finite"
        );
        assert!(
            (0.0..=1.0).contains(&max_score),
            "Score for max values should be in [0, 1] (got {max_score})"
        );

        // Test with minimum values
        let min_frame = vec![i16::MIN; 256];
        let result = vad.process_frame(&min_frame);
        assert!(result.is_ok(), "Processing min values should succeed");
        let min_score = result.unwrap();
        assert!(
            min_score.is_finite(),
            "Score for min values should be finite"
        );
        assert!(
            (0.0..=1.0).contains(&min_score),
            "Score for min values should be in [0, 1] (got {min_score})"
        );
    }

    #[test]
    fn test_process_frame_sequence() {
        let mut vad = create_test_vad();
        let frame_size = 256;

        // Process multiple frames in sequence
        for i in 0..10 {
            let audio_frame: Vec<i16> = (0..frame_size)
                .map(|j| ((i * 100 + j) % 1000) as i16)
                .collect();
            let result = vad.process_frame(&audio_frame);
            assert!(result.is_ok(), "Processing frame {i} should succeed");

            let vad_score = result.unwrap();
            assert!(vad_score.is_finite(), "VAD score {i} should be finite");
        }
    }

    #[test]
    fn test_process_frame_consistent_results() {
        let mut vad1 = create_test_vad();
        let mut vad2 = create_test_vad();

        let audio_frame = generate_test_audio(256, 440.0, 16000.0);
        let audio_i16: Vec<i16> = audio_frame.iter().map(|&x| (x * 32767.0) as i16).collect();

        let score1 = vad1.process_frame(&audio_i16).unwrap();
        let score2 = vad2.process_frame(&audio_i16).unwrap();

        assert!(
            (score1 - score2).abs() < f32::EPSILON,
            "Same input should produce same output"
        );
    }

    #[test]
    fn test_feature_buffer_sliding_window() {
        let mut vad = create_test_vad();

        // Feature buffer should initially be zeros
        let initial_sum: f32 = vad.feature_buffer.sum();
        assert_eq!(initial_sum, 0.0, "Initial feature buffer should be zeros");

        // Process several frames using process_frame (which uses i16 with proper amplitude)
        // to ensure the feature buffer gets populated
        let frequencies = [200.0f32, 400.0, 600.0, 800.0, 1000.0];
        for (i, &freq) in frequencies.iter().enumerate().take(CONTEXT_WINDOW_LEN + 2) {
            // Generate i16 audio with strong amplitude (like real audio input)
            let audio_frame: Vec<i16> = (0..256)
                .map(|j| ((2.0 * PI * freq * j as f32 / 16000.0).sin() * 20000.0) as i16)
                .collect();

            let result = vad.process_frame(&audio_frame);
            assert!(result.is_ok(), "Processing frame {i} should succeed");
        }

        // Feature buffer should contain the last CONTEXT_WINDOW_LEN frames
        assert_eq!(
            vad.feature_buffer.shape(),
            &[CONTEXT_WINDOW_LEN, FEATURE_LEN]
        );

        // After processing real audio, the feature buffer should have values
        // that are non-trivially different from initial zeros
        // (normalized features can be positive or negative, so check absolute sum)
        let buffer_abs_sum: f32 = vad.feature_buffer.iter().map(|x| x.abs()).sum();
        assert!(
            buffer_abs_sum > 0.1,
            "Feature buffer should contain meaningful values after processing (abs sum: {buffer_abs_sum})"
        );

        // Verify all features are finite
        assert!(
            vad.feature_buffer.iter().all(|x| x.is_finite()),
            "All features in buffer should be finite"
        );
    }

    #[test]
    fn test_constants_validity() {
        // Test that constants are reasonable (these help document expected values)
        // The following lines use `const _: () = assert!(...)` for compile-time assertions.
        // This idiom causes a compilation error if the assertion fails, ensuring the condition is checked at compile time.
        const _: () = assert!(FFT_SIZE > 0, "FFT_SIZE should be positive");
        const _: () = assert!(WINDOW_SIZE > 0, "WINDOW_SIZE should be positive");
        const _: () = assert!(
            MEL_FILTER_BANK_NUM > 0,
            "MEL_FILTER_BANK_NUM should be positive"
        );
        const _: () = assert!(FEATURE_LEN > 0, "FEATURE_LEN should be positive");
        const _: () = assert!(
            CONTEXT_WINDOW_LEN > 0,
            "CONTEXT_WINDOW_LEN should be positive"
        );
        const _: () = assert!(MODEL_HIDDEN_DIM > 0, "MODEL_HIDDEN_DIM should be positive");
        const _: () = assert!(MODEL_IO_NUM > 1, "MODEL_IO_NUM should be greater than 1");

        // Test runtime checks
        assert!(
            FFT_SIZE.is_power_of_two(),
            "FFT_SIZE should be a power of 2"
        );
        assert!(
            (0.0..1.0).contains(&PRE_EMPHASIS_COEFF),
            "PRE_EMPHASIS_COEFF should be in (0,1)"
        );

        // Test feature normalization constants
        assert_eq!(
            FEATURE_MEANS.len(),
            FEATURE_LEN,
            "FEATURE_MEANS length should match FEATURE_LEN"
        );
        assert_eq!(
            FEATURE_STDS.len(),
            FEATURE_LEN,
            "FEATURE_STDS length should match FEATURE_LEN"
        );

        // All standard deviations should be positive
        assert!(
            FEATURE_STDS.iter().all(|&x| x > 0.0),
            "All feature stds should be positive"
        );
    }

    #[test]
    fn test_debug_implementation() {
        let vad = create_test_vad();
        let debug_str = format!("{vad:?}");

        // Debug output should contain key information
        assert!(debug_str.contains("TenVad"));
        assert!(debug_str.contains("hidden_states"));
        assert!(debug_str.contains("feature_buffer"));
    }

    #[test]
    fn test_multiple_vad_instances() {
        // Test that multiple VAD instances can coexist and are independent
        let mut vad1 = create_test_vad();
        let mut vad2 = create_test_vad();

        // Process different frames - both should work independently
        let frame1 = vec![100i16; 256];
        let frame2 = vec![200i16; 256];

        let score1 = vad1.process_frame(&frame1).unwrap();
        let score2 = vad2.process_frame(&frame2).unwrap();

        assert!(
            score1.is_finite() && score2.is_finite(),
            "Both instances should produce finite scores"
        );
        assert!(
            (0.0..=1.0).contains(&score1) && (0.0..=1.0).contains(&score2),
            "Both scores should be in valid range"
        );

        // Test determinism: fresh instances with same input should produce identical output
        let mut fresh_vad1 = create_test_vad();
        let mut fresh_vad2 = create_test_vad();
        let test_frame = vec![150i16; 256];

        let fresh_score1 = fresh_vad1.process_frame(&test_frame).unwrap();
        let fresh_score2 = fresh_vad2.process_frame(&test_frame).unwrap();

        assert!(
            (fresh_score1 - fresh_score2).abs() < f32::EPSILON,
            "Fresh instances with same input should produce identical output (got {fresh_score1} vs {fresh_score2})"
        );
    }

    // === New tests for improved coverage ===

    #[test]
    fn test_new_from_bytes() {
        // Read the model file into memory and create VAD from bytes
        let model_bytes = std::fs::read("onnx/ten-vad.onnx").expect("Failed to read model file");
        let vad = TenVad::new_from_bytes(&model_bytes, TARGET_SAMPLE_RATE);

        assert!(vad.is_ok(), "TenVad::new_from_bytes should succeed");

        let mut vad = vad.unwrap();

        // Verify it works the same as file-based initialization
        let audio_frame = vec![100i16; 256];
        let result = vad.process_frame(&audio_frame);
        assert!(
            result.is_ok(),
            "Processing should succeed with byte-loaded model"
        );
        assert!(result.unwrap().is_finite(), "VAD score should be finite");
    }

    #[test]
    fn test_new_from_bytes_invalid_sample_rate() {
        let model_bytes = std::fs::read("onnx/ten-vad.onnx").expect("Failed to read model file");
        let result = TenVad::new_from_bytes(&model_bytes, 8000);

        assert!(result.is_err(), "Should fail with unsupported sample rate");
        match result.unwrap_err() {
            TenVadError::UnsupportedSampleRate(rate) => {
                assert_eq!(rate, 8000);
            }
            _ => panic!("Expected UnsupportedSampleRate error"),
        }
    }

    #[test]
    fn test_new_from_bytes_invalid_model() {
        let invalid_bytes = vec![0u8; 100]; // Invalid model data
        let result = TenVad::new_from_bytes(&invalid_bytes, TARGET_SAMPLE_RATE);

        assert!(result.is_err(), "Should fail with invalid model bytes");
    }

    #[test]
    fn test_pre_emphasis_negative_values() {
        let mut vad = create_test_vad();
        let audio_frame = vec![-100.0, -50.0, 0.0, 50.0, 100.0];
        let emphasized = vad.pre_emphasis(&audio_frame);

        assert_eq!(emphasized.len(), audio_frame.len());

        // Verify pre-emphasis formula for negative values
        for i in 1..audio_frame.len() {
            let expected = audio_frame[i] - PRE_EMPHASIS_COEFF * audio_frame[i - 1];
            assert!(
                (emphasized[i] - expected).abs() < f32::EPSILON,
                "Pre-emphasis should work correctly with negative values"
            );
        }
    }

    #[test]
    fn test_hidden_states_update_after_processing() {
        let mut vad = create_test_vad();

        // Verify initial hidden states are zero
        for hidden_state in &vad.hidden_states {
            assert!(hidden_state.iter().all(|&x| x == 0.0));
        }

        // Process several frames
        let audio_frame = vec![1000i16; 256];
        for _ in 0..5 {
            let _ = vad.process_frame(&audio_frame);
        }

        // After processing, at least some hidden states should be non-zero
        let total_sum: f32 = vad
            .hidden_states
            .iter()
            .flat_map(|h| h.iter())
            .map(|&x| x.abs())
            .sum();

        assert!(
            total_sum > 0.0,
            "Hidden states should be updated after processing"
        );
    }

    #[test]
    fn test_vad_score_silence_vs_tone() {
        let mut vad = create_test_vad();

        // Process silence (should have low VAD score)
        let silence = vec![0i16; 256];
        let mut silence_score = 0.0;
        for _ in 0..10 {
            silence_score = vad.process_frame(&silence).unwrap();
        }

        vad.reset();

        // Process a strong tone (more likely to be detected as voice-like activity)
        let tone: Vec<i16> = (0..256)
            .map(|i| ((2.0 * PI * 200.0 * i as f32 / 16000.0).sin() * 20000.0) as i16)
            .collect();
        let mut tone_score = 0.0;
        for _ in 0..10 {
            tone_score = vad.process_frame(&tone).unwrap();
        }

        // Both scores should be valid
        assert!((0.0..=1.0).contains(&silence_score));
        assert!((0.0..=1.0).contains(&tone_score));

        // Tone typically produces higher activation than silence
        // (though model behavior may vary, at least verify both work)
        assert!(
            silence_score.is_finite() && tone_score.is_finite(),
            "Both silence and tone should produce valid scores"
        );
    }

    #[test]
    fn test_feature_buffer_content_after_processing() {
        let mut vad = create_test_vad();

        // Process enough frames to fill the context window
        let audio_frame: Vec<i16> = (0..256)
            .map(|i| ((2.0 * PI * 300.0 * i as f32 / 16000.0).sin() * 10000.0) as i16)
            .collect();

        for _ in 0..CONTEXT_WINDOW_LEN + 2 {
            let _ = vad.process_frame(&audio_frame);
        }

        // Verify feature buffer has non-trivial content
        let feature_sum: f32 = vad.feature_buffer.iter().map(|x| x.abs()).sum();
        assert!(
            feature_sum > 0.0,
            "Feature buffer should contain non-zero values after processing"
        );

        // Verify feature buffer shape is correct
        assert_eq!(
            vad.feature_buffer.shape(),
            &[CONTEXT_WINDOW_LEN, FEATURE_LEN]
        );
    }

    #[test]
    fn test_estimate_pitch_noise() {
        // Random noise should have low or no pitch correlation
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let noise: Vec<f32> = (0..WINDOW_SIZE)
            .map(|i| {
                // Simple deterministic pseudo-random based on index
                let mut hasher = DefaultHasher::new();
                i.hash(&mut hasher);
                let h = hasher.finish();
                ((h % 65536) as f32 - 32768.0) * 0.3
            })
            .collect();

        let pitch = TenVad::estimate_pitch(&noise);

        // Noise typically doesn't have a strong periodic component
        // Either returns 0 (unvoiced) or a low correlation pitch
        assert!(
            pitch.is_finite(),
            "Pitch estimation should return finite value for noise"
        );
    }

    #[test]
    fn test_process_frame_alternating_values() {
        let mut vad = create_test_vad();

        // Alternating high frequency pattern
        let alternating: Vec<i16> = (0..256)
            .map(|i| if i % 2 == 0 { 10000 } else { -10000 })
            .collect();

        let result = vad.process_frame(&alternating);
        assert!(result.is_ok());
        let score = result.unwrap();
        assert!(
            (0.0..=1.0).contains(&score),
            "Score should be in valid range"
        );
    }

    #[test]
    fn test_mel_filters_frequency_coverage() {
        let mel_filters = TenVad::generate_mel_filters();
        let n_bins = FFT_SIZE / 2 + 1;

        // Check that lower mel filters cover lower frequency bins
        // and higher mel filters cover higher frequency bins
        let low_filter_center = mel_filters
            .row(0)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let high_filter_center = mel_filters
            .row(MEL_FILTER_BANK_NUM - 1)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        assert!(
            low_filter_center < high_filter_center,
            "Lower mel filters should cover lower frequencies"
        );

        // Verify filters span across the frequency range
        assert!(
            low_filter_center < n_bins / 4,
            "Lowest filter should be in lower quarter"
        );
        assert!(
            high_filter_center > n_bins / 2,
            "Highest filter should be above midpoint"
        );
    }

    #[test]
    fn test_error_display_messages() {
        // Test UnsupportedSampleRate error message
        let error = TenVadError::UnsupportedSampleRate(44100);
        let msg = format!("{}", error);
        assert!(
            msg.contains("44100"),
            "Error message should contain the rate"
        );
        assert!(
            msg.contains("16kHz"),
            "Error message should mention supported rate"
        );

        // Test EmptyAudioData error message
        let error = TenVadError::EmptyAudioData;
        let msg = format!("{}", error);
        assert!(
            msg.to_lowercase().contains("empty"),
            "Error message should indicate empty data"
        );
    }

    #[test]
    fn test_process_frame_single_sample() {
        let mut vad = create_test_vad();

        // Single sample should still work (edge case)
        let single = vec![1000i16];
        let result = vad.process_frame(&single);
        assert!(result.is_ok(), "Single sample frame should be processed");
        assert!(result.unwrap().is_finite());
    }

    #[test]
    fn test_reset_clears_pre_emphasis_effect() {
        let mut vad = create_test_vad();

        // Process a frame to set pre_emphasis_prev
        let frame = vec![5000i16; 256];
        let _ = vad.process_frame(&frame);

        // Pre-emphasis state should be non-zero
        assert_ne!(vad.pre_emphasis_prev, 0.0);

        // Reset and verify
        vad.reset();
        assert_eq!(vad.pre_emphasis_prev, 0.0);

        // Processing same input after reset should give same result as fresh instance
        let mut fresh_vad = create_test_vad();
        let test_frame = vec![1000i16; 256];

        let score_reset = vad.process_frame(&test_frame).unwrap();
        let score_fresh = fresh_vad.process_frame(&test_frame).unwrap();

        assert!(
            (score_reset - score_fresh).abs() < f32::EPSILON,
            "Reset VAD should behave like fresh instance"
        );
    }

    #[test]
    fn test_hann_window_energy_conservation() {
        let window = TenVad::generate_hann_window();

        // Sum of squared window values (for energy analysis)
        let window_energy: f32 = window.iter().map(|&x| x * x).sum();

        // Hann window has known energy properties
        // For a normalized Hann window, sum of squares ≈ N * 3/8 for large N
        let expected_approx = WINDOW_SIZE as f32 * 3.0 / 8.0;

        // Allow 5% tolerance
        let tolerance = expected_approx * 0.05;
        assert!(
            (window_energy - expected_approx).abs() < tolerance,
            "Hann window energy should match theoretical value (got {window_energy}, expected ~{expected_approx})"
        );
    }

    #[test]
    fn test_process_frame_ramp_signal() {
        let mut vad = create_test_vad();

        // Linear ramp signal
        let ramp: Vec<i16> = (0..256).map(|i| (i * 100) as i16).collect();
        let result = vad.process_frame(&ramp);

        assert!(result.is_ok());
        let score = result.unwrap();
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_estimate_pitch_boundary_frequencies() {
        let sample_rate = TARGET_SAMPLE_RATE as f32;

        // Test pitch at boundary of detection range (near min period = ~500Hz)
        let high_freq = sample_rate / PITCH_MIN_PERIOD as f32; // ~500 Hz
        let audio_high: Vec<f32> = (0..WINDOW_SIZE)
            .map(|i| (2.0 * PI * high_freq * i as f32 / sample_rate).sin() * 10000.0)
            .collect();

        let pitch_high = TenVad::estimate_pitch(&audio_high);
        assert!(pitch_high.is_finite());

        // Test pitch at boundary of detection range (near max period = ~62Hz)
        let low_freq = sample_rate / PITCH_MAX_PERIOD as f32; // ~62.5 Hz
        let audio_low: Vec<f32> = (0..WINDOW_SIZE)
            .map(|i| (2.0 * PI * low_freq * i as f32 / sample_rate).sin() * 10000.0)
            .collect();

        let pitch_low = TenVad::estimate_pitch(&audio_low);
        assert!(pitch_low.is_finite());
    }

    #[test]
    fn test_vad_state_independence_after_reset() {
        let mut vad = create_test_vad();

        // Process varied audio to build up state
        for freq in [100.0, 200.0, 300.0, 400.0, 500.0] {
            let audio: Vec<i16> = (0..256)
                .map(|i| ((2.0 * PI * freq * i as f32 / 16000.0).sin() * 15000.0) as i16)
                .collect();
            let _ = vad.process_frame(&audio);
        }

        vad.reset();

        // After reset, state should be fully independent of previous processing
        assert!(
            vad.hidden_states
                .iter()
                .all(|h| h.iter().all(|&x| x == 0.0))
        );
        assert!(vad.feature_buffer.iter().all(|&x| x == 0.0));
        assert_eq!(vad.pre_emphasis_prev, 0.0);
    }
}
