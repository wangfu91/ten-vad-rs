use crate::biquad::{AUP_PE_A_4KHZ, AUP_PE_B_4KHZ, AUP_PE_G_4KHZ, BiquadFilter};
use rustfft::{Fft, FftPlanner, num_complex::Complex32};
use std::f32::consts::PI;
use std::sync::Arc;

pub const NB_BANDS: usize = 18;
pub const LPC_ORDER: usize = 16;
pub const MIN_PERIOD_16KHZ: usize = 32;
pub const MAX_PERIOD_16KHZ: usize = 256;
pub const XCORR_TRAINING_OFFSET: usize = 80;
pub const FEAT_TIME_WINDOW: usize = 40;
pub const FEAT_MAX_NFRM: usize = 12;
pub const TOTAL_NFEAT: usize = 55;
pub const PITCHMAXPATH_W: f32 = 0.02;
pub const ASSUMED_FFT_4_BAND_ENG: f32 = 80.0;
pub const BAND_START_INDEX: [usize; NB_BANDS] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40,
];
pub const BAND_LPC_COMP: [f32; NB_BANDS] = [
    0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.666667, 0.5, 0.5, 0.5, 0.333333, 0.25, 0.25, 0.2,
    0.166667, 0.173913,
];

#[derive(Clone, Debug)]
pub struct PitchEstConfig {
    pub fft_size: usize,
    pub ana_window_size: usize,
    pub hop_size: usize,
    pub use_lpc_pre_filtering: bool,
    pub proc_fs: usize,
    pub voiced_thr: f32,
}

impl Default for PitchEstConfig {
    fn default() -> Self {
        Self {
            fft_size: 1024,
            ana_window_size: 768,
            hop_size: 256,
            use_lpc_pre_filtering: true,
            proc_fs: 4000,
            voiced_thr: 0.4,
        }
    }
}

#[derive(Clone)]
pub struct PitchEstimator {
    config: PitchEstConfig,
    proc_resample_rate: usize,
    min_period: usize,
    max_period: usize,
    dif_period: usize,
    n_feat: usize,
    n_bins: usize,
    dct_table: [f32; NB_BANDS * NB_BANDS],
    input_resample_buf: Vec<f32>,
    input_resample_buf_idx: usize,
    input_q: Vec<f32>,
    aligned_in: Vec<f32>,
    lpc_filter_out_buf: Vec<f32>,
    exc_buf: Vec<f32>,
    exc_buf_sq: Vec<f32>,
    lpc: [f32; LPC_ORDER],
    lpc_scratch: [f32; LPC_ORDER],
    per_bin_scratch: Vec<f32>,
    ac_scratch: [f32; LPC_ORDER + 1],
    pitch_mem: [f32; LPC_ORDER],
    pitch_filt: f32,
    filt_scratch: Vec<f32>,
    decimated_scratch: Vec<f32>,
    tmp_feat: [f32; TOTAL_NFEAT],
    xcorr_offset_idx: usize,
    xcorr_inst: Vec<f32>,
    xcorr: Vec<Vec<f32>>,
    xcorr_tmp: Vec<Vec<f32>>,
    xcorr_mov_scratch: Vec<f32>,
    path_scratch: Vec<usize>,
    best_period_est_local: Vec<usize>,
    frm_weight: Vec<f32>,
    frm_weight_norm: Vec<f32>,
    pitch_max_path_reg: [Vec<f32>; 2],
    pitch_prev: Vec<Vec<i32>>,
    pitch_max_path_all: f32,
    best_period_est: usize,
    voiced: bool,
    pitch_est_result: f32,
    biquad_filter: BiquadFilter,
    ifft_instance: Arc<dyn Fft<f32>>,
    ifft_buffer: Vec<Complex32>,
}

impl std::fmt::Debug for PitchEstimator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PitchEstimator")
            .field("proc_resample_rate", &self.proc_resample_rate)
            .field("min_period", &self.min_period)
            .field("max_period", &self.max_period)
            .field("n_feat", &self.n_feat)
            .field("n_bins", &self.n_bins)
            .field("best_period_est", &self.best_period_est)
            .field("voiced", &self.voiced)
            .field("pitch_est_result", &self.pitch_est_result)
            .finish()
    }
}

impl PitchEstimator {
    pub fn new() -> Self {
        let config = PitchEstConfig::default();
        let proc_resample_rate = 16000 / config.proc_fs;
        let min_period = MIN_PERIOD_16KHZ / proc_resample_rate;
        let max_period = MAX_PERIOD_16KHZ / proc_resample_rate;
        let dif_period = max_period - min_period;
        let _ana_window_size = config.ana_window_size;
        let n_feat = FEAT_MAX_NFRM.min(
            ((FEAT_TIME_WINDOW as f32 * 16000.0) / (config.hop_size as f32 * 1000.0)).ceil()
                as usize,
        );
        let n_bins = config.fft_size / 2 + 1;
        let mut dct_table = [0.0f32; NB_BANDS * NB_BANDS];
        for idx in 0..NB_BANDS {
            for jdx in 0..NB_BANDS {
                let mut v = ((idx as f32 + 0.5) * jdx as f32 * PI / NB_BANDS as f32).cos();
                if jdx == 0 {
                    v *= (0.5f32).sqrt();
                }
                dct_table[idx * NB_BANDS + jdx] = v;
            }
        }

        let mut planner = FftPlanner::<f32>::new();
        let ifft_instance = planner.plan_fft_inverse(config.fft_size);
        let ifft_buffer = vec![Complex32::new(0.0, 0.0); config.fft_size];

        let frame_w = (n_feat * 2).max(1);
        let hop_size = config.hop_size;
        let decimation_step = proc_resample_rate.max(1);
        let decimated_scratch_len = (hop_size + decimation_step - 1) / decimation_step;
        let input_q_len = XCORR_TRAINING_OFFSET.max(config.hop_size) + config.hop_size;
        let exc_buf_shift_len = (config.hop_size + proc_resample_rate - 1) / proc_resample_rate;
        let exc_buf_len = max_period + exc_buf_shift_len + 1;
        Self {
            config,
            proc_resample_rate,
            min_period,
            max_period,
            dif_period,
            n_feat,
            n_bins,
            dct_table,
            input_resample_buf: vec![0.0; 1024],
            input_resample_buf_idx: 0,
            input_q: vec![0.0; input_q_len],
            aligned_in: vec![0.0; 256],
            lpc_filter_out_buf: vec![0.0; 256],
            exc_buf: vec![0.0; exc_buf_len],
            exc_buf_sq: vec![0.0; exc_buf_len],
            lpc: [0.0; LPC_ORDER],
            lpc_scratch: [0.0f32; LPC_ORDER],
            per_bin_scratch: vec![0.0f32; n_bins],
            ac_scratch: [0.0f32; LPC_ORDER + 1],
            pitch_mem: [0.0; LPC_ORDER],
            pitch_filt: 0.0,
            filt_scratch: vec![0.0f32; hop_size],
            decimated_scratch: vec![0.0f32; decimated_scratch_len.max(1)],
            tmp_feat: [0.0; TOTAL_NFEAT],
            xcorr_offset_idx: 0,
            xcorr_inst: vec![0.0; max_period],
            xcorr: vec![vec![0.0; max_period + 1]; frame_w],
            xcorr_tmp: vec![vec![0.0; max_period + 1]; frame_w],
            xcorr_mov_scratch: vec![0.0f32; max_period],
            path_scratch: vec![0usize; n_feat * 2],
            best_period_est_local: vec![0usize; n_feat * 2],
            frm_weight: vec![0.0; frame_w],
            frm_weight_norm: vec![0.0; frame_w],
            pitch_max_path_reg: [vec![0.0; max_period], vec![0.0; max_period]],
            pitch_prev: vec![vec![0; max_period]; frame_w],
            pitch_max_path_all: 0.0,
            best_period_est: min_period,
            voiced: false,
            pitch_est_result: 0.0,
            biquad_filter: BiquadFilter::new(&AUP_PE_B_4KHZ, &AUP_PE_A_4KHZ, &AUP_PE_G_4KHZ),
            ifft_instance,
            ifft_buffer,
        }
    }

    pub fn reset(&mut self) {
        self.input_resample_buf.fill(0.0);
        self.input_resample_buf_idx = 0;
        self.input_q.fill(0.0);
        self.aligned_in.fill(0.0);
        self.lpc_filter_out_buf.fill(0.0);
        self.exc_buf.fill(0.0);
        self.exc_buf_sq.fill(0.0);
        self.lpc = [0.0; LPC_ORDER];
        self.lpc_scratch.fill(0.0);
        self.per_bin_scratch.fill(0.0);
        self.ac_scratch.fill(0.0);
        self.pitch_mem = [0.0; LPC_ORDER];
        self.pitch_filt = 0.0;
        self.filt_scratch.fill(0.0);
        self.decimated_scratch.fill(0.0);
        self.tmp_feat = [0.0; TOTAL_NFEAT];
        self.xcorr_offset_idx = 0;
        self.xcorr_inst.fill(0.0);
        for row in &mut self.xcorr {
            row.fill(0.0);
        }
        for row in &mut self.xcorr_tmp {
            row.fill(0.0);
        }
        self.xcorr_mov_scratch.fill(0.0);
        self.path_scratch.fill(0);
        self.frm_weight.fill(0.0);
        self.frm_weight_norm.fill(0.0);
        for reg in &mut self.pitch_max_path_reg {
            reg.fill(0.0);
        }
        for row in &mut self.pitch_prev {
            row.fill(0);
        }
        self.pitch_max_path_all = 0.0;
        self.best_period_est = self.min_period;
        self.voiced = false;
        self.pitch_est_result = 0.0;
        self.biquad_filter.reset();
    }

    fn compute_band_energy(&self, in_bin_power: &[f32], band_e: &mut [f32; NB_BANDS]) {
        let n_bins = in_bin_power.len();
        if n_bins == 0 {
            band_e.fill(0.0);
            return;
        }
        let index_conv_rate = self.config.fft_size as f32 / ASSUMED_FFT_4_BAND_ENG;
        band_e.fill(0.0);
        for i in 0..(NB_BANDS - 1) {
            let band_sz = (((BAND_START_INDEX[i + 1] - BAND_START_INDEX[i]) as f32)
                * index_conv_rate)
                .round() as usize;
            if band_sz == 0 {
                continue;
            }
            let index_offset = ((BAND_START_INDEX[i] as f32) * index_conv_rate).round() as usize;
            for j in 0..band_sz {
                let frac = j as f32 / band_sz as f32;
                let acc_idx = (index_offset + j).min(n_bins - 1);
                band_e[i] += (1.0 - frac) * in_bin_power[acc_idx];
                band_e[i + 1] += frac * in_bin_power[acc_idx];
            }
        }
        band_e[0] *= 2.0;
        band_e[NB_BANDS - 1] *= 2.0;
    }

    fn dct(&self, input: &[f32; NB_BANDS], out: &mut [f32; NB_BANDS]) {
        let ratio = (2.0 / NB_BANDS as f32).sqrt();
        for idx in 0..NB_BANDS {
            let mut sum = 0.0;
            for j in 0..NB_BANDS {
                sum += input[j] * self.dct_table[j * NB_BANDS + idx];
            }
            out[idx] = sum * ratio;
        }
    }

    fn idct(&self, input: &[f32; NB_BANDS], out: &mut [f32; NB_BANDS]) {
        let ratio = (2.0 / NB_BANDS as f32).sqrt();
        for idx in 0..NB_BANDS {
            let mut sum = 0.0;
            for j in 0..NB_BANDS {
                sum += input[j] * self.dct_table[idx * NB_BANDS + j];
            }
            out[idx] = sum * ratio;
        }
    }

    fn interp_band_gain(band_gain: &[f32; NB_BANDS], gain_per_bin: &mut [f32]) {
        let n_bins = gain_per_bin.len();
        let fft_sz = (n_bins - 1) * 2;
        let index_conv_rate = fft_sz as f32 / ASSUMED_FFT_4_BAND_ENG;
        gain_per_bin.fill(0.0);
        for idx in 0..(NB_BANDS - 1) {
            let band_sz = (((BAND_START_INDEX[idx + 1] - BAND_START_INDEX[idx]) as f32)
                * index_conv_rate)
                .round() as usize;
            if band_sz == 0 {
                continue;
            }
            let index_offset = (BAND_START_INDEX[idx] as f32 * index_conv_rate).round() as usize;
            for j in 0..band_sz {
                let frac = j as f32 / band_sz as f32;
                let acc_idx = (index_offset + j).min(n_bins - 1);
                gain_per_bin[acc_idx] = (1.0 - frac) * band_gain[idx] + frac * band_gain[idx + 1];
            }
        }
    }

    fn celt_lpc(ac: &[f32], lpc: &mut [f32], lpc_scratch: &mut [f32; LPC_ORDER]) {
        let p = lpc.len();
        lpc_scratch[..p].fill(0.0);
        if ac[0] != 0.0 {
            let mut error = ac[0];
            for i in 0..p {
                let mut rr = ac[i + 1];
                for j in 0..i {
                    rr += lpc_scratch[j] * ac[i - j];
                }
                let r = -rr / error;
                lpc_scratch[i] = r;
                for j in 0..(i / 2) {
                    let aj = lpc_scratch[j];
                    let ai = lpc_scratch[i - 1 - j];
                    lpc_scratch[j] = aj + r * ai;
                    lpc_scratch[i - 1 - j] = ai + r * aj;
                }
                if i % 2 == 1 {
                    let j = i / 2;
                    lpc_scratch[j] += lpc_scratch[j] * r;
                }
                error *= (1.0 - r * r).max(1e-6);
                if error < 0.001 * ac[0] {
                    break;
                }
            }
        }
        lpc.copy_from_slice(&lpc_scratch[..p]);
    }

    fn lpc_from_bands(&mut self, band_gain: &[f32; NB_BANDS], lpc: &mut [f32; LPC_ORDER]) {
        let n = self.config.fft_size;
        let half = n / 2 + 1;
        Self::interp_band_gain(band_gain, &mut self.per_bin_scratch[..half]);
        self.per_bin_scratch[half - 1] = 0.0;

        self.ifft_buffer.fill(Complex32::new(0.0, 0.0));
        for (i, &bin) in self.per_bin_scratch.iter().enumerate().take(half) {
            self.ifft_buffer[i] = Complex32::new(bin, 0.0);
        }
        for i in 1..(n / 2) {
            self.ifft_buffer[n - i] = self.ifft_buffer[i].conj();
        }
        self.ifft_instance.process(&mut self.ifft_buffer);

        self.ac_scratch.fill(0.0);
        for (i, a) in self.ac_scratch.iter_mut().enumerate().take(LPC_ORDER + 1) {
            *a = self.ifft_buffer[i].re / n as f32;
        }
        let dc0_bias = self.config.ana_window_size as f32 / 12.0 / 38.0;
        self.ac_scratch[0] += self.ac_scratch[0] * 1e-4 + dc0_bias;
        for i in 1..=LPC_ORDER {
            self.ac_scratch[i] *= 1.0 - 6e-5 * i as f32 * i as f32;
        }
        Self::celt_lpc(&self.ac_scratch, lpc, &mut self.lpc_scratch);
    }

    fn lpc_compute(&mut self, cepstrum: &[f32; NB_BANDS], lpc: &mut [f32; LPC_ORDER]) {
        let mut log_band = [0.0f32; NB_BANDS];
        self.idct(cepstrum, &mut log_band);
        for i in 0..NB_BANDS {
            log_band[i] = 10.0f32.powf(log_band[i]) * BAND_LPC_COMP[i];
        }
        self.lpc_from_bands(&log_band, lpc);
    }

    fn celt_inner_prod(x: &[f32], y: &[f32]) -> f32 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }

    fn moving_xcorr(
        corr_window_len: usize,
        corr_shift_times: usize,
        ref_in: &[f32],
        y_in_to_shift: &[f32],
        xcorr: &mut [f32],
    ) {
        for i in 0..corr_shift_times {
            if i + corr_window_len > y_in_to_shift.len() || corr_window_len > ref_in.len() {
                xcorr[i] = 0.0;
            } else {
                xcorr[i] = Self::celt_inner_prod(
                    &ref_in[..corr_window_len],
                    &y_in_to_shift[i..i + corr_window_len],
                );
            }
        }
    }

    pub fn process(&mut self, raw_signal: &[f32], bin_power: &[f32]) -> f32 {
        if raw_signal.is_empty() || bin_power.is_empty() {
            self.pitch_est_result = 0.0;
            return 0.0;
        }

        // Phase A: LPC-based pre-filtering and 4kHz resampling.
        let mut band_energy = [0.0f32; NB_BANDS];
        self.compute_band_energy(bin_power, &mut band_energy);
        let mut log_max = -2.0f32;
        let mut follow = -2.0f32;
        for b in &mut band_energy {
            let mut ly = (1e-2 + *b).log10();
            ly = (log_max - 8.0).max((follow - 2.5).max(ly));
            log_max = log_max.max(ly);
            follow = (follow - 2.5).max(ly);
            *b = ly;
        }
        let mut cep = [0.0f32; NB_BANDS];
        self.dct(&band_energy, &mut cep);
        if self.config.use_lpc_pre_filtering {
            let mut lpc_new = [0.0f32; LPC_ORDER];
            self.lpc_compute(&cep, &mut lpc_new);
            self.lpc = lpc_new;
        }

        let hop = self.aligned_in.len();
        let shift = raw_signal.len().min(hop);
        if shift < self.input_q.len() {
            self.input_q.copy_within(shift.., 0);
            let tail_start = self.input_q.len() - shift;
            self.input_q[tail_start..].copy_from_slice(&raw_signal[..shift]);
        } else {
            let start = raw_signal.len() - self.input_q.len();
            self.input_q.copy_from_slice(&raw_signal[start..]);
        }

        let offset = self
            .input_q
            .len()
            .saturating_sub(hop)
            .saturating_sub(XCORR_TRAINING_OFFSET);
        self.aligned_in[..shift].copy_from_slice(&self.input_q[offset..offset + shift]);
        for i in 0..shift {
            let mut slid_win_sum = self.aligned_in[i];
            for j in 0..LPC_ORDER {
                slid_win_sum += self.lpc[j] * self.pitch_mem[j];
            }
            self.pitch_mem.copy_within(0..LPC_ORDER - 1, 1);
            self.pitch_mem[0] = self.aligned_in[i];
            self.lpc_filter_out_buf[i] = slid_win_sum + 0.7 * self.pitch_filt;
            self.pitch_filt = slid_win_sum;
        }

        self.biquad_filter.process(
            &self.lpc_filter_out_buf[..shift],
            &mut self.filt_scratch[..shift],
        );
        let mut dshift_count = 0usize;
        for idx in (0..shift).step_by(self.proc_resample_rate.max(1)) {
            debug_assert!(dshift_count < self.decimated_scratch.len());
            self.decimated_scratch[dshift_count] = self.filt_scratch[idx];
            dshift_count += 1;
        }
        let decimated = &self.decimated_scratch[..dshift_count];

        let dshift = decimated.len().min(self.exc_buf.len());
        if dshift < self.exc_buf.len() {
            self.exc_buf.copy_within(dshift.., 0);
            let start = self.exc_buf.len() - dshift;
            self.exc_buf[start..].copy_from_slice(&decimated[..dshift]);
        }

        // Phase B: normalized moving xcorr.
        for (dst, &x) in self.exc_buf_sq.iter_mut().zip(self.exc_buf.iter()) {
            *dst = x * x;
        }
        for idx in 0..(self.n_feat - 1) {
            self.frm_weight[2 * idx] = self.frm_weight[2 * (idx + 1)];
            self.frm_weight[2 * idx + 1] = self.frm_weight[2 * (idx + 1) + 1];
        }

        let corr_half_hop_sz = hop / (self.proc_resample_rate * 2);
        for sub in 0..2 {
            let xcorr_acc_idx = 2 * self.xcorr_offset_idx + sub;
            let corr_offset = sub * corr_half_hop_sz;
            let ref_seq = &self.exc_buf[self.max_period + corr_offset..];
            let mv_seq = &self.exc_buf[corr_offset..];
            Self::moving_xcorr(
                corr_half_hop_sz,
                self.max_period,
                ref_seq,
                mv_seq,
                &mut self.xcorr_inst,
            );

            let mut energy0 = 0.0f32;
            for idx in 0..corr_half_hop_sz {
                energy0 += self.exc_buf_sq[self.max_period + corr_offset + idx];
            }
            self.frm_weight[2 * (self.n_feat - 1) + sub] = energy0;

            let mut slid_win_sum = 0.0f32;
            for idx in 0..corr_half_hop_sz {
                slid_win_sum += self.exc_buf_sq[corr_offset + idx];
            }

            let mut tmp_denom = (slid_win_sum + (1.0 + energy0)).max(1e-12);
            self.xcorr[xcorr_acc_idx][0] = 2.0 * self.xcorr_inst[0] / tmp_denom;
            for idx in 1..self.max_period {
                slid_win_sum = (slid_win_sum - self.exc_buf_sq[corr_offset + idx - 1]).max(0.0);
                slid_win_sum += self.exc_buf_sq[corr_offset + idx + corr_half_hop_sz - 1];
                tmp_denom = (slid_win_sum + (1.0 + energy0)).max(1e-12);
                self.xcorr[xcorr_acc_idx][idx] = 2.0 * self.xcorr_inst[idx] / tmp_denom;
            }

            for idx in 0..(self.max_period - 2 * self.min_period) {
                let mut td = self.xcorr[xcorr_acc_idx][(self.max_period + idx) / 2];
                td = td.max(self.xcorr[xcorr_acc_idx][(self.max_period + idx + 2) / 2]);
                td = td.max(self.xcorr[xcorr_acc_idx][(self.max_period + idx - 1) / 2]);
                if self.xcorr[xcorr_acc_idx][idx] < td * 1.1 {
                    self.xcorr[xcorr_acc_idx][idx] *= 0.8;
                }
            }
        }
        self.xcorr_offset_idx = (self.xcorr_offset_idx + 1) % self.n_feat;

        // Phase C: DP tracking + weighted regression.
        let mut slid_win_sum = 1e-15f32;
        for sub in 0..(self.n_feat * 2) {
            slid_win_sum += self.frm_weight[sub];
        }
        for sub in 0..(self.n_feat * 2) {
            self.frm_weight_norm[sub] =
                self.frm_weight[sub] * ((self.n_feat * 2) as f32 / slid_win_sum);
        }
        for (dst_row, src_row) in self.xcorr_tmp.iter_mut().zip(self.xcorr.iter()) {
            dst_row.copy_from_slice(src_row);
        }
        for sub in (0..(self.n_feat * 2 - 2)).step_by(2) {
            let (head, tail) = self.pitch_prev.split_at_mut(sub + 2);
            head[sub].copy_from_slice(&tail[0]);
            head[sub + 1].copy_from_slice(&tail[1]);
        }

        self.pitch_max_path_reg[0].fill(0.0);
        self.pitch_max_path_reg[1].fill(0.0);
        for sub in (self.n_feat * 2 - 2)..(self.n_feat * 2) {
            let mut xc_idx = sub + self.xcorr_offset_idx * 2;
            if xc_idx >= 2 * self.n_feat {
                xc_idx -= 2 * self.n_feat;
            }
            for idx in 0..self.dif_period {
                let mut max_track_reg = self.pitch_max_path_all - 1e10;
                self.pitch_prev[sub][idx] = self.best_period_est as i32;
                let sidxt = std::cmp::min(0, 4 - idx as i32);
                for jdx in sidxt..=4 {
                    let cand = idx as i32 + jdx;
                    if cand < 0 || cand as usize >= self.dif_period {
                        continue;
                    }
                    let tmp_denom = self.pitch_max_path_reg[0][cand as usize]
                        - (PITCHMAXPATH_W * (jdx.abs() as f32) * (jdx.abs() as f32));
                    if tmp_denom > max_track_reg {
                        max_track_reg = tmp_denom;
                        self.pitch_prev[sub][idx] = cand;
                    }
                }
                self.pitch_max_path_reg[1][idx] =
                    max_track_reg + self.frm_weight_norm[sub] * self.xcorr_tmp[xc_idx][idx];
            }

            let mut max_path_reg = -1e15f32;
            let mut tmp_int = 0usize;
            for idx in 0..self.dif_period {
                if self.pitch_max_path_reg[1][idx] > max_path_reg {
                    max_path_reg = self.pitch_max_path_reg[1][idx];
                    tmp_int = idx;
                }
            }
            self.pitch_max_path_all = max_path_reg;
            self.best_period_est = tmp_int;
            let (curr, next) = self.pitch_max_path_reg.split_at_mut(1);
            curr[0].copy_from_slice(&next[0]);
            for idx in 0..self.dif_period {
                self.pitch_max_path_reg[0][idx] -= max_path_reg;
            }
        }

        self.best_period_est_local.fill(0);
        let mut tmp_int = self.best_period_est;
        let mut frm_corr = 0.0f32;
        for sub in (0..(self.n_feat * 2)).rev() {
            self.best_period_est_local[sub] = self.max_period - tmp_int;
            let mut xc_idx = sub + self.xcorr_offset_idx * 2;
            if xc_idx >= 2 * self.n_feat {
                xc_idx -= 2 * self.n_feat;
            }
            frm_corr += self.frm_weight_norm[sub] * self.xcorr_tmp[xc_idx][tmp_int];
            tmp_int = self.pitch_prev[sub][tmp_int] as usize;
        }
        frm_corr = (frm_corr / (self.n_feat * 2) as f32).max(0.0);
        self.voiced = frm_corr >= self.config.voiced_thr;

        let mut sx = 0.0f32;
        let mut sxx = 0.0f32;
        let mut sxy = 0.0f32;
        let mut sy = 0.0f32;
        let mut sw = 0.0f32;
        for sub in 0..(self.n_feat * 2) {
            let w = self.frm_weight_norm[sub];
            let sf = sub as f32;
            let pf = self.best_period_est_local[sub] as f32;
            sw += w;
            sx += w * sf;
            sxx += w * sf * sf;
            sxy += w * sf * pf;
            sy += w * pf;
        }
        let denom = sw * sxx - sx * sx;
        let mut best_a = if denom == 0.0 {
            (sw * sxy - sx * sy) / 1e-15
        } else {
            (sw * sxy - sx * sy) / denom
        };
        if self.voiced {
            let slope_lim = (sy / sw.max(1e-15)) / (4.0 * 2.0 * self.n_feat as f32);
            best_a = best_a.clamp(-slope_lim, slope_lim);
        } else {
            best_a = 0.0;
        }
        let best_b = (sy - best_a * sx) / sw.max(1e-15);
        let estimated_period = best_b + 5.5 * best_a;

        self.pitch_est_result = if self.voiced {
            self.config.proc_fs as f32 / estimated_period.max(1.0)
        } else {
            0.0
        };
        self.pitch_est_result
    }
}
