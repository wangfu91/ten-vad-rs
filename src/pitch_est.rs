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
pub const BAND_START_INDEX: [usize; 41] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
];
pub const BAND_LPC_COMP: [f32; NB_BANDS] = [
    0.80, 0.86, 0.92, 0.97, 1.00, 1.02, 1.03, 1.02, 1.01, 1.00, 0.99, 0.98, 0.97, 0.96, 0.95,
    0.94, 0.93, 0.92,
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
    pitch_mem: [f32; LPC_ORDER],
    pitch_filt: f32,
    tmp_feat: [f32; TOTAL_NFEAT],
    xcorr_offset_idx: usize,
    xcorr_inst: Vec<f32>,
    xcorr: Vec<Vec<f32>>,
    xcorr_tmp: Vec<Vec<f32>>,
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
        let n_feat = FEAT_MAX_NFRM.min(FEAT_TIME_WINDOW * 1000 / (config.hop_size * 1000 / 16));
        let n_bins = config.fft_size / 2 + 1;
        let mut dct_table = [0.0f32; NB_BANDS * NB_BANDS];
        let scale = (2.0 / NB_BANDS as f32).sqrt();
        for k in 0..NB_BANDS {
            for n in 0..NB_BANDS {
                let alpha = if k == 0 { (0.5f32).sqrt() } else { 1.0 };
                dct_table[k * NB_BANDS + n] =
                    alpha * scale * ((PI / NB_BANDS as f32) * (n as f32 + 0.5) * k as f32).cos();
            }
        }

        let mut planner = FftPlanner::<f32>::new();
        let ifft_instance = planner.plan_fft_inverse(config.fft_size);
        let ifft_buffer = vec![Complex32::new(0.0, 0.0); config.fft_size];

        let frame_w = n_feat.max(1);
        let periods = dif_period + 1;
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
            input_q: vec![0.0; MAX_PERIOD_16KHZ + 2048],
            aligned_in: vec![0.0; 256],
            lpc_filter_out_buf: vec![0.0; 256],
            exc_buf: vec![0.0; 4096],
            exc_buf_sq: vec![0.0; 4096],
            lpc: [0.0; LPC_ORDER],
            pitch_mem: [0.0; LPC_ORDER],
            pitch_filt: 0.0,
            tmp_feat: [0.0; TOTAL_NFEAT],
            xcorr_offset_idx: 0,
            xcorr_inst: vec![0.0; periods],
            xcorr: vec![vec![0.0; periods]; frame_w],
            xcorr_tmp: vec![vec![0.0; periods]; frame_w],
            frm_weight: vec![0.0; frame_w],
            frm_weight_norm: vec![0.0; frame_w],
            pitch_max_path_reg: [vec![0.0; periods], vec![0.0; periods]],
            pitch_prev: vec![vec![0; periods]; frame_w],
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
        self.pitch_mem = [0.0; LPC_ORDER];
        self.pitch_filt = 0.0;
        self.tmp_feat = [0.0; TOTAL_NFEAT];
        self.xcorr_offset_idx = 0;
        self.xcorr_inst.fill(0.0);
        for row in &mut self.xcorr {
            row.fill(0.0);
        }
        for row in &mut self.xcorr_tmp {
            row.fill(0.0);
        }
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

    fn compute_band_energy(&self, bin_power: &[f32], band_energy: &mut [f32; NB_BANDS]) {
        let n_bins = bin_power.len().max(1);
        for (b, be) in band_energy.iter_mut().enumerate() {
            let bs0 = BAND_START_INDEX[b * (BAND_START_INDEX.len() - 1) / NB_BANDS];
            let bs1 = BAND_START_INDEX[(b + 1) * (BAND_START_INDEX.len() - 1) / NB_BANDS];
            let start = bs0 * (n_bins - 1) / BAND_START_INDEX[BAND_START_INDEX.len() - 1].max(1);
            let end = (bs1 * (n_bins - 1)
                / BAND_START_INDEX[BAND_START_INDEX.len() - 1].max(1))
            .max(start + 1);
            let mut sum = 0.0;
            for &p in &bin_power[start..=end] {
                sum += p.max(0.0);
            }
            *be = sum / (end - start + 1) as f32 + 1e-12;
        }
    }

    fn dct(&self, input: &[f32; NB_BANDS], out: &mut [f32; NB_BANDS]) {
        for (k, out_k) in out.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (n, &x) in input.iter().enumerate() {
                acc += self.dct_table[k * NB_BANDS + n] * x;
            }
            *out_k = acc;
        }
    }

    fn idct(&self, input: &[f32; NB_BANDS], out: &mut [f32; NB_BANDS]) {
        for (n, out_n) in out.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (k, &x) in input.iter().enumerate() {
                acc += self.dct_table[k * NB_BANDS + n] * x;
            }
            *out_n = acc;
        }
    }

    fn interp_band_gain(&self, band_gain: &[f32; NB_BANDS], gain_per_bin: &mut [f32]) {
        let n = gain_per_bin.len();
        for (i, g) in gain_per_bin.iter_mut().enumerate() {
            let x = i as f32 * (NB_BANDS - 1) as f32 / (n - 1) as f32;
            let i0 = x.floor() as usize;
            let i1 = (i0 + 1).min(NB_BANDS - 1);
            let t = x - i0 as f32;
            *g = band_gain[i0] * (1.0 - t) + band_gain[i1] * t;
        }
    }

    fn celt_lpc(&self, ac: &[f32], lpc: &mut [f32]) {
        let p = lpc.len();
        let mut error = ac[0].max(1e-9);
        let mut a = vec![0.0f32; p];
        for i in 0..p {
            let mut rr = ac[i + 1];
            for j in 0..i {
                rr += a[j] * ac[i - j];
            }
            let r = -rr / error;
            a[i] = r;
            for j in 0..(i / 2) {
                let aj = a[j];
                let ai = a[i - 1 - j];
                a[j] = aj + r * ai;
                a[i - 1 - j] = ai + r * aj;
            }
            if i % 2 == 1 {
                let j = i / 2;
                a[j] += a[j] * r;
            }
            error *= (1.0 - r * r).max(1e-6);
        }
        lpc.copy_from_slice(&a);
    }

    fn lpc_from_bands(&mut self, band_gain: &[f32; NB_BANDS], lpc: &mut [f32; LPC_ORDER]) {
        let n = self.config.fft_size;
        let half = n / 2 + 1;
        let mut per_bin = vec![0.0f32; half];
        self.interp_band_gain(band_gain, &mut per_bin);

        self.ifft_buffer.fill(Complex32::new(0.0, 0.0));
        for (i, &bin) in per_bin.iter().enumerate().take(half) {
            self.ifft_buffer[i] = Complex32::new(bin, 0.0);
        }
        for i in 1..(n / 2) {
            self.ifft_buffer[n - i] = self.ifft_buffer[i].conj();
        }
        self.ifft_instance.process(&mut self.ifft_buffer);

        let mut ac = vec![0.0f32; LPC_ORDER + 1];
        for (i, a) in ac.iter_mut().enumerate().take(LPC_ORDER + 1) {
            *a = self.ifft_buffer[i].re / n as f32;
        }
        self.celt_lpc(&ac, lpc);
    }

    fn lpc_compute(&mut self, cepstrum: &[f32; NB_BANDS], lpc: &mut [f32; LPC_ORDER]) {
        let mut log_band = [0.0f32; NB_BANDS];
        self.idct(cepstrum, &mut log_band);
        for i in 0..NB_BANDS {
            log_band[i] = (log_band[i] * BAND_LPC_COMP[i]).exp().max(1e-7);
        }
        self.lpc_from_bands(&log_band, lpc);
    }

    fn xcorr_kernel(x: &[f32], y: &[f32]) -> f32 {
        let n = x.len().min(y.len());
        let mut s0 = 0.0;
        let mut s1 = 0.0;
        let mut s2 = 0.0;
        let mut s3 = 0.0;
        let mut i = 0;
        while i + 3 < n {
            s0 += x[i] * y[i];
            s1 += x[i + 1] * y[i + 1];
            s2 += x[i + 2] * y[i + 2];
            s3 += x[i + 3] * y[i + 3];
            i += 4;
        }
        let mut sum = s0 + s1 + s2 + s3;
        while i < n {
            sum += x[i] * y[i];
            i += 1;
        }
        sum
    }

    fn celt_inner_prod(x: &[f32], y: &[f32]) -> f32 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }

    fn moving_xcorr(&self, x: &[f32], y: &[f32], out: &mut [f32]) {
        for (i, o) in out.iter_mut().enumerate() {
            if i >= y.len() {
                *o = 0.0;
                continue;
            }
            let y_off = &y[i..];
            *o = Self::xcorr_kernel(x, y_off);
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
        for b in &mut band_energy {
            *b = b.max(1e-9).ln().clamp(-20.0, 20.0);
        }
        let mut cep = [0.0f32; NB_BANDS];
        self.dct(&band_energy, &mut cep);
        if self.config.use_lpc_pre_filtering {
            let mut lpc_new = [0.0f32; LPC_ORDER];
            self.lpc_compute(&cep, &mut lpc_new);
            self.lpc = lpc_new;
        }

        let shift = raw_signal.len().min(self.input_q.len());
        if shift < self.input_q.len() {
            self.input_q.copy_within(shift.., 0);
            let tail_start = self.input_q.len() - shift;
            self.input_q[tail_start..].copy_from_slice(&raw_signal[..shift]);
        }

        self.aligned_in[..shift].copy_from_slice(&raw_signal[..shift]);
        for i in 0..shift {
            let mut pred = 0.0;
            for j in 0..LPC_ORDER {
                let past = if i > j {
                    self.aligned_in[i - j - 1]
                } else {
                    self.pitch_mem[LPC_ORDER - (j + 1 - i)]
                };
                pred += self.lpc[j] * past;
            }
            self.lpc_filter_out_buf[i] = self.aligned_in[i] - pred;
        }
        if shift >= LPC_ORDER {
            self.pitch_mem
                .copy_from_slice(&self.aligned_in[shift - LPC_ORDER..shift]);
        } else {
            let keep = LPC_ORDER - shift;
            self.pitch_mem.copy_within(shift.., 0);
            self.pitch_mem[keep..].copy_from_slice(&self.aligned_in[..shift]);
        }

        let mut filt = vec![0.0f32; shift];
        self.biquad_filter
            .process(&self.lpc_filter_out_buf[..shift], &mut filt);
        let mut decimated = Vec::with_capacity((shift / self.proc_resample_rate).max(1));
        for idx in (0..filt.len()).step_by(self.proc_resample_rate.max(1)) {
            decimated.push(filt[idx]);
        }

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
        self.frm_weight.rotate_left(1);
        let frm_last = self.frm_weight.len() - 1;
        self.frm_weight[frm_last] =
            decimated.iter().map(|x| x * x).sum::<f32>().sqrt() + 1e-9;

        let n = self.exc_buf.len();
        let periods = self.dif_period + 1;
        let recent_len = (self.max_period + XCORR_TRAINING_OFFSET)
            .min(n.saturating_sub(1))
            .max(self.max_period + 1);
        let base_start = n - recent_len;
        let x = &self.exc_buf[base_start + self.max_period..];
        let mut xcorr_mov = vec![0.0f32; periods];
        self.moving_xcorr(x, &self.exc_buf[base_start..base_start + self.max_period], &mut xcorr_mov);
        for p in self.min_period..=self.max_period {
            let y_start = base_start + self.max_period - p;
            let y = &self.exc_buf[y_start..y_start + x.len()];
            let num = Self::celt_inner_prod(x, y);
            let den =
                (Self::celt_inner_prod(x, x) * Self::celt_inner_prod(y, y)).sqrt().max(1e-8);
            let mut val = (0.5 * (num / den + xcorr_mov[p - self.min_period] / den)).clamp(-1.0, 1.0);
            if p * 2 <= self.max_period {
                val *= 0.98;
            }
            if p <= self.min_period + 2 {
                val *= 0.95;
            }
            self.xcorr_inst[p - self.min_period] = val.max(0.0);
        }
        self.xcorr_offset_idx = (self.xcorr_offset_idx + 1) % self.n_feat;
        self.xcorr[self.xcorr_offset_idx].copy_from_slice(&self.xcorr_inst);

        // Phase C: DP tracking + weighted regression.
        let wsum: f32 = self.frm_weight.iter().sum::<f32>().max(1e-9);
        for (w, wn) in self.frm_weight.iter().zip(self.frm_weight_norm.iter_mut()) {
            *wn = *w / wsum;
        }
        self.xcorr_tmp.clone_from(&self.xcorr);
        self.pitch_prev.rotate_left(1);
        let pitch_prev_last = self.pitch_prev.len() - 1;
        self.pitch_prev[pitch_prev_last].fill(0);

        self.pitch_max_path_reg[0].fill(0.0);
        self.pitch_max_path_reg[1].fill(0.0);
        for t in 0..self.n_feat {
            let curr = t % 2;
            let prev = 1 - curr;
            let row_idx = (self.xcorr_offset_idx + self.n_feat - (self.n_feat - 1 - t)) % self.n_feat;
            for p in 0..periods {
                let obs = self.xcorr_tmp[row_idx][p];
                let mut best = -1e30;
                let mut best_prev = 0;
                for q in 0..periods {
                    let d = p as f32 - q as f32;
                    let score = self.pitch_max_path_reg[prev][q] - PITCHMAXPATH_W * d * d;
                    if score > best {
                        best = score;
                        best_prev = q as i32;
                    }
                }
                self.pitch_max_path_reg[curr][p] = best + obs * self.frm_weight_norm[t];
                self.pitch_prev[t][p] = best_prev;
            }
        }
        let last = (self.n_feat - 1) % 2;
        let mut best_p = 0usize;
        let mut best_s = -1e30;
        for p in 0..periods {
            let s = self.pitch_max_path_reg[last][p];
            if s > best_s {
                best_s = s;
                best_p = p;
            }
        }
        self.pitch_max_path_all = best_s;
        let mut path = vec![0usize; self.n_feat];
        path[self.n_feat - 1] = best_p;
        for t in (1..self.n_feat).rev() {
            path[t - 1] = self.pitch_prev[t][path[t]] as usize;
        }
        self.best_period_est = path[self.n_feat - 1] + self.min_period;

        let mut corr = 0.0;
        for (t, p) in path.iter().enumerate() {
            corr += self.frm_weight_norm[t] * self.xcorr_tmp[t][*p];
        }
        self.voiced = corr >= self.config.voiced_thr;

        let mut sw = 0.0;
        let mut st = 0.0;
        let mut spp = 0.0;
        let mut stp = 0.0;
        for (t, &p) in path.iter().enumerate() {
            let w = self.frm_weight_norm[t];
            let tt = t as f32;
            let pp = (p + self.min_period) as f32;
            sw += w;
            st += w * tt;
            spp += w * pp;
            stp += w * tt * pp;
        }
        let st2: f32 = self
            .frm_weight_norm
            .iter()
            .enumerate()
            .map(|(t, w)| *w * (t as f32) * (t as f32))
            .sum();
        let denom = (sw * st2 - st * st).abs().max(1e-6);
        let slope = (sw * stp - st * spp) / denom;
        let intercept = (spp - slope * st) / sw.max(1e-6);
        let period_est = (intercept + slope * (self.n_feat.saturating_sub(1)) as f32)
            .clamp(self.min_period as f32, self.max_period as f32);

        self.pitch_est_result = if self.voiced {
            self.config.proc_fs as f32 / period_est.max(1.0)
        } else {
            0.0
        };
        self.pitch_est_result
    }
}
