const AGORA_UAP_BIQUAD_MAX_SECTION: usize = 20;

pub const ANTI_ALIAS_SECTIONS: usize = 5;
pub const AUP_PE_B_4KHZ: [[f32; 3]; ANTI_ALIAS_SECTIONS] = [
    [1.0, 1.198825, 1.0],
    [1.0, -0.567461, 1.0],
    [1.0, -1.099061, 1.0],
    [1.0, -1.265846, 1.0],
    [1.0, -1.318849, 1.0],
];
pub const AUP_PE_A_4KHZ: [[f32; 3]; ANTI_ALIAS_SECTIONS] = [
    [1.0, -1.445267, 0.5463974],
    [1.0, -1.426720, 0.6820138],
    [1.0, -1.408255, 0.8286664],
    [1.0, -1.400909, 0.9240320],
    [1.0, -1.408242, 0.9789776],
];
pub const AUP_PE_G_4KHZ: [f32; ANTI_ALIAS_SECTIONS] =
    [0.2692541, 0.2692541, 0.2692541, 0.2692541, 0.2692541];

#[derive(Clone, Debug)]
pub struct BiquadFilter {
    nsect: usize,
    b_coeff: [[f32; 3]; AGORA_UAP_BIQUAD_MAX_SECTION],
    a_coeff: [[f32; 3]; AGORA_UAP_BIQUAD_MAX_SECTION],
    g_coeff: [f32; AGORA_UAP_BIQUAD_MAX_SECTION],
    sect_w: [[f32; 2]; AGORA_UAP_BIQUAD_MAX_SECTION],
}

impl BiquadFilter {
    pub fn new(b: &[[f32; 3]], a: &[[f32; 3]], g: &[f32]) -> Self {
        assert_eq!(b.len(), a.len());
        assert_eq!(b.len(), g.len());
        assert!(b.len() <= AGORA_UAP_BIQUAD_MAX_SECTION);

        let nsect = b.len();
        let mut filter = Self {
            nsect,
            b_coeff: [[0.0; 3]; AGORA_UAP_BIQUAD_MAX_SECTION],
            a_coeff: [[0.0; 3]; AGORA_UAP_BIQUAD_MAX_SECTION],
            g_coeff: [0.0; AGORA_UAP_BIQUAD_MAX_SECTION],
            sect_w: [[0.0; 2]; AGORA_UAP_BIQUAD_MAX_SECTION],
        };

        filter.b_coeff[..nsect].copy_from_slice(&b[..nsect]);
        filter.a_coeff[..nsect].copy_from_slice(&a[..nsect]);
        filter.g_coeff[..nsect].copy_from_slice(&g[..nsect]);
        filter
    }

    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        output.copy_from_slice(input);

        for sect in 0..self.nsect {
            let b = self.b_coeff[sect];
            let a = self.a_coeff[sect];
            let g = self.g_coeff[sect];
            let mut w1 = self.sect_w[sect][0];
            let mut w2 = self.sect_w[sect][1];

            for y in output.iter_mut() {
                let x = *y;
                let w0 = x - a[1] * w1 - a[2] * w2;
                let out = g * (b[0] * w0 + b[1] * w1 + b[2] * w2);
                w2 = w1;
                w1 = w0;
                *y = out;
            }

            self.sect_w[sect][0] = w1;
            self.sect_w[sect][1] = w2;
        }
    }

    pub fn reset(&mut self) {
        for state in &mut self.sect_w {
            *state = [0.0; 2];
        }
    }
}
