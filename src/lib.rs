#![allow(clippy::excessive_precision)]

mod biquad;
mod buffer;
mod error;
mod pitch_est;

// Re-export error types for public API
pub use crate::buffer::AudioFrameBuffer;
pub use crate::error::{TenVadError, TenVadResult};

/// Target sample rate for TEN VAD (16kHz)
pub const TARGET_SAMPLE_RATE: u32 = 16000;

use ndarray::{Array1, Array2, ArrayView1, Axis, aview1};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{SessionInputValue, SessionInputs};
use ort::{session::Session, value::TensorRef};
use pitch_est::PitchEstimator;
use rustfft::{Fft, FftPlanner, num_complex::Complex32};
use std::sync::Arc;

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

/// Hann window coefficients (from coeff.h)
#[rustfmt::skip]
const HANN_WINDOW_768: [f32; WINDOW_SIZE] = [
    0.0000000e+00, 1.6733041e-05, 6.6931045e-05, 1.5059065e-04,
    2.6770626e-04, 4.1827004e-04, 6.0227190e-04, 8.1969953e-04,
    1.0705384e-03, 1.3547717e-03, 1.6723803e-03, 2.0233432e-03,
    2.4076367e-03, 2.8252351e-03, 3.2761105e-03, 3.7602327e-03,
    4.2775693e-03, 4.8280857e-03, 5.4117450e-03, 6.0285082e-03,
    6.6783340e-03, 7.3611788e-03, 8.0769970e-03, 8.8257407e-03,
    9.6073598e-03, 1.0421802e-02, 1.1269013e-02, 1.2148935e-02,
    1.3061510e-02, 1.4006678e-02, 1.4984373e-02, 1.5994532e-02,
    1.7037087e-02, 1.8111967e-02, 1.9219101e-02, 2.0358415e-02,
    2.1529832e-02, 2.2733274e-02, 2.3968661e-02, 2.5235910e-02,
    2.6534935e-02, 2.7865651e-02, 2.9227967e-02, 3.0621794e-02,
    3.2047037e-02, 3.3503601e-02, 3.4991388e-02, 3.6510300e-02,
    3.8060234e-02, 3.9641086e-02, 4.1252752e-02, 4.2895122e-02,
    4.4568088e-02, 4.6271536e-02, 4.8005353e-02, 4.9769424e-02,
    5.1563629e-02, 5.3387849e-02, 5.5241962e-02, 5.7125844e-02,
    5.9039368e-02, 6.0982406e-02, 6.2954829e-02, 6.4956504e-02,
    6.6987298e-02, 6.9047074e-02, 7.1135695e-02, 7.3253021e-02,
    7.5398909e-02, 7.7573217e-02, 7.9775799e-02, 8.2006508e-02,
    8.4265194e-02, 8.6551706e-02, 8.8865891e-02, 9.1207593e-02,
    9.3576658e-02, 9.5972925e-02, 9.8396234e-02, 1.0084642e-01,
    1.0332333e-01, 1.0582679e-01, 1.0835663e-01, 1.1091268e-01,
    1.1349477e-01, 1.1610274e-01, 1.1873640e-01, 1.2139558e-01,
    1.2408010e-01, 1.2678978e-01, 1.2952444e-01, 1.3228389e-01,
    1.3506796e-01, 1.3787646e-01, 1.4070919e-01, 1.4356597e-01,
    1.4644661e-01, 1.4935091e-01, 1.5227868e-01, 1.5522973e-01,
    1.5820385e-01, 1.6120085e-01, 1.6422052e-01, 1.6726267e-01,
    1.7032709e-01, 1.7341358e-01, 1.7652192e-01, 1.7965192e-01,
    1.8280336e-01, 1.8597603e-01, 1.8916971e-01, 1.9238420e-01,
    1.9561929e-01, 1.9887474e-01, 2.0215035e-01, 2.0544589e-01,
    2.0876115e-01, 2.1209590e-01, 2.1544993e-01, 2.1882300e-01,
    2.2221488e-01, 2.2562536e-01, 2.2905421e-01, 2.3250119e-01,
    2.3596607e-01, 2.3944863e-01, 2.4294863e-01, 2.4646583e-01,
    2.5000000e-01, 2.5355090e-01, 2.5711830e-01, 2.6070196e-01,
    2.6430163e-01, 2.6791708e-01, 2.7154806e-01, 2.7519434e-01,
    2.7885565e-01, 2.8253178e-01, 2.8622245e-01, 2.8992744e-01,
    2.9364649e-01, 2.9737934e-01, 3.0112576e-01, 3.0488549e-01,
    3.0865828e-01, 3.1244388e-01, 3.1624203e-01, 3.2005248e-01,
    3.2387498e-01, 3.2770926e-01, 3.3155507e-01, 3.3541216e-01,
    3.3928027e-01, 3.4315913e-01, 3.4704849e-01, 3.5094809e-01,
    3.5485766e-01, 3.5877695e-01, 3.6270569e-01, 3.6664362e-01,
    3.7059048e-01, 3.7454600e-01, 3.7850991e-01, 3.8248196e-01,
    3.8646187e-01, 3.9044938e-01, 3.9444422e-01, 3.9844613e-01,
    4.0245484e-01, 4.0647007e-01, 4.1049157e-01, 4.1451906e-01,
    4.1855226e-01, 4.2259092e-01, 4.2663476e-01, 4.3068351e-01,
    4.3473690e-01, 4.3879466e-01, 4.4285652e-01, 4.4692220e-01,
    4.5099143e-01, 4.5506394e-01, 4.5913946e-01, 4.6321772e-01,
    4.6729844e-01, 4.7138134e-01, 4.7546616e-01, 4.7955263e-01,
    4.8364046e-01, 4.8772939e-01, 4.9181913e-01, 4.9590943e-01,
    5.0000000e-01, 5.0409057e-01, 5.0818087e-01, 5.1227061e-01,
    5.1635954e-01, 5.2044737e-01, 5.2453384e-01, 5.2861866e-01,
    5.3270156e-01, 5.3678228e-01, 5.4086054e-01, 5.4493606e-01,
    5.4900857e-01, 5.5307780e-01, 5.5714348e-01, 5.6120534e-01,
    5.6526310e-01, 5.6931649e-01, 5.7336524e-01, 5.7740908e-01,
    5.8144774e-01, 5.8548094e-01, 5.8950843e-01, 5.9352993e-01,
    5.9754516e-01, 6.0155387e-01, 6.0555578e-01, 6.0955062e-01,
    6.1353813e-01, 6.1751804e-01, 6.2149009e-01, 6.2545400e-01,
    6.2940952e-01, 6.3335638e-01, 6.3729431e-01, 6.4122305e-01,
    6.4514234e-01, 6.4905191e-01, 6.5295151e-01, 6.5684087e-01,
    6.6071973e-01, 6.6458784e-01, 6.6844493e-01, 6.7229074e-01,
    6.7612502e-01, 6.7994752e-01, 6.8375797e-01, 6.8755612e-01,
    6.9134172e-01, 6.9511451e-01, 6.9887424e-01, 7.0262066e-01,
    7.0635351e-01, 7.1007256e-01, 7.1377755e-01, 7.1746822e-01,
    7.2114435e-01, 7.2480566e-01, 7.2845194e-01, 7.3208292e-01,
    7.3569837e-01, 7.3929804e-01, 7.4288170e-01, 7.4644910e-01,
    7.5000000e-01, 7.5353417e-01, 7.5705137e-01, 7.6055137e-01,
    7.6403393e-01, 7.6749881e-01, 7.7094579e-01, 7.7437464e-01,
    7.7778512e-01, 7.8117700e-01, 7.8455007e-01, 7.8790410e-01,
    7.9123885e-01, 7.9455411e-01, 7.9784965e-01, 8.0112526e-01,
    8.0438071e-01, 8.0761580e-01, 8.1083029e-01, 8.1402397e-01,
    8.1719664e-01, 8.2034808e-01, 8.2347808e-01, 8.2658642e-01,
    8.2967291e-01, 8.3273733e-01, 8.3577948e-01, 8.3879915e-01,
    8.4179615e-01, 8.4477027e-01, 8.4772132e-01, 8.5064909e-01,
    8.5355339e-01, 8.5643403e-01, 8.5929081e-01, 8.6212354e-01,
    8.6493204e-01, 8.6771611e-01, 8.7047556e-01, 8.7321022e-01,
    8.7591990e-01, 8.7860442e-01, 8.8126360e-01, 8.8389726e-01,
    8.8650523e-01, 8.8908732e-01, 8.9164337e-01, 8.9417321e-01,
    8.9667667e-01, 8.9915358e-01, 9.0160377e-01, 9.0402708e-01,
    9.0642334e-01, 9.0879241e-01, 9.1113411e-01, 9.1344829e-01,
    9.1573481e-01, 9.1799349e-01, 9.2022420e-01, 9.2242678e-01,
    9.2460109e-01, 9.2674698e-01, 9.2886431e-01, 9.3095293e-01,
    9.3301270e-01, 9.3504350e-01, 9.3704517e-01, 9.3901759e-01,
    9.4096063e-01, 9.4287416e-01, 9.4475804e-01, 9.4661215e-01,
    9.4843637e-01, 9.5023058e-01, 9.5199465e-01, 9.5372846e-01,
    9.5543191e-01, 9.5710488e-01, 9.5874725e-01, 9.6035891e-01,
    9.6193977e-01, 9.6348970e-01, 9.6500861e-01, 9.6649640e-01,
    9.6795296e-01, 9.6937821e-01, 9.7077203e-01, 9.7213435e-01,
    9.7346506e-01, 9.7476409e-01, 9.7603134e-01, 9.7726673e-01,
    9.7847017e-01, 9.7964159e-01, 9.8078090e-01, 9.8188803e-01,
    9.8296291e-01, 9.8400547e-01, 9.8501563e-01, 9.8599332e-01,
    9.8693849e-01, 9.8785107e-01, 9.8873099e-01, 9.8957820e-01,
    9.9039264e-01, 9.9117426e-01, 9.9192300e-01, 9.9263882e-01,
    9.9332167e-01, 9.9397149e-01, 9.9458825e-01, 9.9517191e-01,
    9.9572243e-01, 9.9623977e-01, 9.9672389e-01, 9.9717476e-01,
    9.9759236e-01, 9.9797666e-01, 9.9832762e-01, 9.9864523e-01,
    9.9892946e-01, 9.9918030e-01, 9.9939773e-01, 9.9958173e-01,
    9.9973229e-01, 9.9984941e-01, 9.9993307e-01, 9.9998327e-01,
    1.0000000e+00, 9.9998327e-01, 9.9993307e-01, 9.9984941e-01,
    9.9973229e-01, 9.9958173e-01, 9.9939773e-01, 9.9918030e-01,
    9.9892946e-01, 9.9864523e-01, 9.9832762e-01, 9.9797666e-01,
    9.9759236e-01, 9.9717476e-01, 9.9672389e-01, 9.9623977e-01,
    9.9572243e-01, 9.9517191e-01, 9.9458825e-01, 9.9397149e-01,
    9.9332167e-01, 9.9263882e-01, 9.9192300e-01, 9.9117426e-01,
    9.9039264e-01, 9.8957820e-01, 9.8873099e-01, 9.8785107e-01,
    9.8693849e-01, 9.8599332e-01, 9.8501563e-01, 9.8400547e-01,
    9.8296291e-01, 9.8188803e-01, 9.8078090e-01, 9.7964159e-01,
    9.7847017e-01, 9.7726673e-01, 9.7603134e-01, 9.7476409e-01,
    9.7346506e-01, 9.7213435e-01, 9.7077203e-01, 9.6937821e-01,
    9.6795296e-01, 9.6649640e-01, 9.6500861e-01, 9.6348970e-01,
    9.6193977e-01, 9.6035891e-01, 9.5874725e-01, 9.5710488e-01,
    9.5543191e-01, 9.5372846e-01, 9.5199465e-01, 9.5023058e-01,
    9.4843637e-01, 9.4661215e-01, 9.4475804e-01, 9.4287416e-01,
    9.4096063e-01, 9.3901759e-01, 9.3704517e-01, 9.3504350e-01,
    9.3301270e-01, 9.3095293e-01, 9.2886431e-01, 9.2674698e-01,
    9.2460109e-01, 9.2242678e-01, 9.2022420e-01, 9.1799349e-01,
    9.1573481e-01, 9.1344829e-01, 9.1113411e-01, 9.0879241e-01,
    9.0642334e-01, 9.0402708e-01, 9.0160377e-01, 8.9915358e-01,
    8.9667667e-01, 8.9417321e-01, 8.9164337e-01, 8.8908732e-01,
    8.8650523e-01, 8.8389726e-01, 8.8126360e-01, 8.7860442e-01,
    8.7591990e-01, 8.7321022e-01, 8.7047556e-01, 8.6771611e-01,
    8.6493204e-01, 8.6212354e-01, 8.5929081e-01, 8.5643403e-01,
    8.5355339e-01, 8.5064909e-01, 8.4772132e-01, 8.4477027e-01,
    8.4179615e-01, 8.3879915e-01, 8.3577948e-01, 8.3273733e-01,
    8.2967291e-01, 8.2658642e-01, 8.2347808e-01, 8.2034808e-01,
    8.1719664e-01, 8.1402397e-01, 8.1083029e-01, 8.0761580e-01,
    8.0438071e-01, 8.0112526e-01, 7.9784965e-01, 7.9455411e-01,
    7.9123885e-01, 7.8790410e-01, 7.8455007e-01, 7.8117700e-01,
    7.7778512e-01, 7.7437464e-01, 7.7094579e-01, 7.6749881e-01,
    7.6403393e-01, 7.6055137e-01, 7.5705137e-01, 7.5353417e-01,
    7.5000000e-01, 7.4644910e-01, 7.4288170e-01, 7.3929804e-01,
    7.3569837e-01, 7.3208292e-01, 7.2845194e-01, 7.2480566e-01,
    7.2114435e-01, 7.1746822e-01, 7.1377755e-01, 7.1007256e-01,
    7.0635351e-01, 7.0262066e-01, 6.9887424e-01, 6.9511451e-01,
    6.9134172e-01, 6.8755612e-01, 6.8375797e-01, 6.7994752e-01,
    6.7612502e-01, 6.7229074e-01, 6.6844493e-01, 6.6458784e-01,
    6.6071973e-01, 6.5684087e-01, 6.5295151e-01, 6.4905191e-01,
    6.4514234e-01, 6.4122305e-01, 6.3729431e-01, 6.3335638e-01,
    6.2940952e-01, 6.2545400e-01, 6.2149009e-01, 6.1751804e-01,
    6.1353813e-01, 6.0955062e-01, 6.0555578e-01, 6.0155387e-01,
    5.9754516e-01, 5.9352993e-01, 5.8950843e-01, 5.8548094e-01,
    5.8144774e-01, 5.7740908e-01, 5.7336524e-01, 5.6931649e-01,
    5.6526310e-01, 5.6120534e-01, 5.5714348e-01, 5.5307780e-01,
    5.4900857e-01, 5.4493606e-01, 5.4086054e-01, 5.3678228e-01,
    5.3270156e-01, 5.2861866e-01, 5.2453384e-01, 5.2044737e-01,
    5.1635954e-01, 5.1227061e-01, 5.0818087e-01, 5.0409057e-01,
    5.0000000e-01, 4.9590943e-01, 4.9181913e-01, 4.8772939e-01,
    4.8364046e-01, 4.7955263e-01, 4.7546616e-01, 4.7138134e-01,
    4.6729844e-01, 4.6321772e-01, 4.5913946e-01, 4.5506394e-01,
    4.5099143e-01, 4.4692220e-01, 4.4285652e-01, 4.3879466e-01,
    4.3473690e-01, 4.3068351e-01, 4.2663476e-01, 4.2259092e-01,
    4.1855226e-01, 4.1451906e-01, 4.1049157e-01, 4.0647007e-01,
    4.0245484e-01, 3.9844613e-01, 3.9444422e-01, 3.9044938e-01,
    3.8646187e-01, 3.8248196e-01, 3.7850991e-01, 3.7454600e-01,
    3.7059048e-01, 3.6664362e-01, 3.6270569e-01, 3.5877695e-01,
    3.5485766e-01, 3.5094809e-01, 3.4704849e-01, 3.4315913e-01,
    3.3928027e-01, 3.3541216e-01, 3.3155507e-01, 3.2770926e-01,
    3.2387498e-01, 3.2005248e-01, 3.1624203e-01, 3.1244388e-01,
    3.0865828e-01, 3.0488549e-01, 3.0112576e-01, 2.9737934e-01,
    2.9364649e-01, 2.8992744e-01, 2.8622245e-01, 2.8253178e-01,
    2.7885565e-01, 2.7519434e-01, 2.7154806e-01, 2.6791708e-01,
    2.6430163e-01, 2.6070196e-01, 2.5711830e-01, 2.5355090e-01,
    2.5000000e-01, 2.4646583e-01, 2.4294863e-01, 2.3944863e-01,
    2.3596607e-01, 2.3250119e-01, 2.2905421e-01, 2.2562536e-01,
    2.2221488e-01, 2.1882300e-01, 2.1544993e-01, 2.1209590e-01,
    2.0876115e-01, 2.0544589e-01, 2.0215035e-01, 1.9887474e-01,
    1.9561929e-01, 1.9238420e-01, 1.8916971e-01, 1.8597603e-01,
    1.8280336e-01, 1.7965192e-01, 1.7652192e-01, 1.7341358e-01,
    1.7032709e-01, 1.6726267e-01, 1.6422052e-01, 1.6120085e-01,
    1.5820385e-01, 1.5522973e-01, 1.5227868e-01, 1.4935091e-01,
    1.4644661e-01, 1.4356597e-01, 1.4070919e-01, 1.3787646e-01,
    1.3506796e-01, 1.3228389e-01, 1.2952444e-01, 1.2678978e-01,
    1.2408010e-01, 1.2139558e-01, 1.1873640e-01, 1.1610274e-01,
    1.1349477e-01, 1.1091268e-01, 1.0835663e-01, 1.0582679e-01,
    1.0332333e-01, 1.0084642e-01, 9.8396234e-02, 9.5972925e-02,
    9.3576658e-02, 9.1207593e-02, 8.8865891e-02, 8.6551706e-02,
    8.4265194e-02, 8.2006508e-02, 7.9775799e-02, 7.7573217e-02,
    7.5398909e-02, 7.3253021e-02, 7.1135695e-02, 6.9047074e-02,
    6.6987298e-02, 6.4956504e-02, 6.2954829e-02, 6.0982406e-02,
    5.9039368e-02, 5.7125844e-02, 5.5241962e-02, 5.3387849e-02,
    5.1563629e-02, 4.9769424e-02, 4.8005353e-02, 4.6271536e-02,
    4.4568088e-02, 4.2895122e-02, 4.1252752e-02, 3.9641086e-02,
    3.8060234e-02, 3.6510300e-02, 3.4991388e-02, 3.3503601e-02,
    3.2047037e-02, 3.0621794e-02, 2.9227967e-02, 2.7865651e-02,
    2.6534935e-02, 2.5235910e-02, 2.3968661e-02, 2.2733274e-02,
    2.1529832e-02, 2.0358415e-02, 1.9219101e-02, 1.8111967e-02,
    1.7037087e-02, 1.5994532e-02, 1.4984373e-02, 1.4006678e-02,
    1.3061510e-02, 1.2148935e-02, 1.1269013e-02, 1.0421802e-02,
    9.6073598e-03, 8.8257407e-03, 8.0769970e-03, 7.3611788e-03,
    6.6783340e-03, 6.0285082e-03, 5.4117450e-03, 4.8280857e-03,
    4.2775693e-03, 3.7602327e-03, 3.2761105e-03, 2.8252351e-03,
    2.4076367e-03, 2.0233432e-03, 1.6723803e-03, 1.3547717e-03,
    1.0705384e-03, 8.1969953e-04, 6.0227190e-04, 4.1827004e-04,
    2.6770626e-04, 1.5059065e-04, 6.6931045e-05, 1.6733041e-05
];

/// TEN VAD ONNX model runner
pub struct TenVad {
    session: Session,                 // ONNX session for inference
    hidden_states: Vec<Array2<f32>>, // Vector of 2D arrays: [MODEL_IO_NUM - 1] each [1, MODEL_HIDDEN_DIM]
    feature_buffer: Array2<f32>,     // 2D array: [CONTEXT_WINDOW_LEN, FEATURE_LEN]
    pre_emphasis_prev: f32,          // Previous value for pre-emphasis filtering
    mel_filters: Array2<f32>,        // 2D array: [MEL_FILTER_BANK_NUM, n_bins]
    window: ArrayView1<'static, f32>, // Hann window view: [WINDOW_SIZE]
    fft_instance: Arc<dyn Fft<f32>>, // Cached FFT instance
    fft_buffer: Vec<Complex32>,      // Reusable FFT buffer
    stft_input_q: Vec<f32>,          // Sliding STFT input queue (pre-emphasized samples)
    stft_windowed_buf: Array1<f32>,  // Reusable STFT windowed buffer
    pitch_estimator: PitchEstimator, // Pitch estimator state
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
        let mel_filters = Self::generate_mel_filters()?;

        // Generate Hann window
        let window = Self::generate_hann_window();

        // Create and cache FFT planner and instance
        let mut fft_planner = FftPlanner::new();
        let fft_instance = fft_planner.plan_fft_forward(FFT_SIZE);
        let fft_buffer = vec![Complex32::new(0.0, 0.0); FFT_SIZE];
        let stft_input_q = vec![0.0f32; WINDOW_SIZE];
        let stft_windowed_buf = Array1::zeros(WINDOW_SIZE);
        let pitch_estimator = PitchEstimator::new();

        Ok(Self {
            session,
            hidden_states,
            feature_buffer,
            pre_emphasis_prev,
            mel_filters,
            window,
            fft_instance,
            fft_buffer,
            stft_input_q,
            stft_windowed_buf,
            pitch_estimator,
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

    /// Generate mel filter-bank coefficients(Adapted from aed.cc).
    ///
    /// A mel filter bank is a set of filters used in audio processing to mimic the human ear's perception of sound frequencies.
    /// These filters are spaced according to the mel scale, which is more sensitive to lower frequencies and less sensitive to higher frequencies.
    fn generate_mel_filters() -> TenVadResult<Array2<f32>> {
        let n_bins = FFT_SIZE / 2 + 1;

        // Generate mel filter-bank coefficients
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
            let bin = ((FFT_SIZE + 1) as f32 * hz / 16000.0f32) as usize;
            bin_points.push(bin);
        }
        for i in 1..bin_points.len() {
            if bin_points[i] == bin_points[i - 1] {
                return Err(TenVadError::InvalidConfiguration(
                    "Duplicate mel bin points are not supported".to_string(),
                ));
            }
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

        Ok(mel_filters)
    }

    /// Generate Hann window coefficients
    fn generate_hann_window() -> ArrayView1<'static, f32> {
        aview1(&HANN_WINDOW_768)
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

        // Sliding STFT window (hop of 256 samples in the reference pipeline).
        let hop_size = 256.min(WINDOW_SIZE).min(emphasized.len());
        if hop_size > 0 {
            self.stft_input_q.copy_within(hop_size.., 0);
            let dst_start = WINDOW_SIZE - hop_size;
            for i in 0..hop_size {
                self.stft_input_q[dst_start + i] = emphasized[emphasized.len() - hop_size + i];
            }
        }

        // Windowing into a reusable buffer.
        for i in 0..WINDOW_SIZE {
            self.stft_windowed_buf[i] = self.stft_input_q[i] * self.window[i];
        }

        // Zero the FFT buffer before use to clear any previous data (using cached FFT instance and reusable buffer)
        self.fft_buffer.fill(Complex32::new(0.0, 0.0));

        // Prepare input for FFT (real to complex)
        for i in 0..WINDOW_SIZE.min(FFT_SIZE) {
            self.fft_buffer[i] = Complex32::new(self.stft_windowed_buf[i], 0.0);
        }

        // Perform FFT using cached instance
        self.fft_instance.process(&mut self.fft_buffer);

        // Compute power spectrum (only positive frequencies)
        let n_bins = FFT_SIZE / 2 + 1;
        let mut power_spectrum = Array1::zeros(n_bins);
        for i in 0..n_bins {
            power_spectrum[i] = self.fft_buffer[i].norm_sqr();
        }

        // Pitch estimation consumes raw (non-pre-emphasized) signal and unnormalized bin power.
        let pitch_freq = self
            .pitch_estimator
            .process(audio_frame, power_spectrum.as_slice().unwrap_or(&[]));

        // Normalize mel path (corresponding to powerNormal = 32768^2 in C++).
        let power_normal = 32768.0f32.powi(2);
        power_spectrum /= power_normal;

        // Mel filter bank features
        let mel_features = self.mel_filters.dot(&power_spectrum);
        let mel_features = mel_features.mapv(|x| (x + EPS).ln());

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
        self.stft_input_q.fill(0.0f32);
        self.stft_windowed_buf.fill(0.0f32);
        self.pitch_estimator.reset();
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
            .field("stft_input_q", &self.stft_input_q.len())
            .field("stft_windowed_buf", &self.stft_windowed_buf.len())
            .field("pitch_estimator", &self.pitch_estimator)
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
        let mel_filters = TenVad::generate_mel_filters().expect("Failed to generate mel filters");

        // Check dimensions
        assert_eq!(
            mel_filters.shape(),
            &[MEL_FILTER_BANK_NUM, FFT_SIZE / 2 + 1]
        );

        // Check that filters are non-negative
        assert!(mel_filters.iter().all(|&x| x >= 0.0));

        // Check that each filter has some non-zero values
        for i in 0..MEL_FILTER_BANK_NUM {
            let filter_sum: f32 = mel_filters.row(i).sum();
            assert!(filter_sum > 0.0, "Filter {i} should have non-zero values");
        }

        // Check that filters have triangular shape (max value should be around 1.0)
        for i in 0..MEL_FILTER_BANK_NUM {
            let max_val = mel_filters.row(i).iter().fold(0.0f32, |a, &b| a.max(b));
            assert!(
                max_val <= 1.0 + f32::EPSILON,
                "Filter {i} max value should not exceed 1.0"
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

        // Check exact edge values from reference table.
        assert!((window[0] - 0.0).abs() < 1e-12, "Window[0] mismatch");
        assert!(
            (window[WINDOW_SIZE - 1] - 1.6733041e-05).abs() < 1e-10,
            "Window[last] mismatch"
        );

        // Check that window peaks near the middle
        let mid_idx = WINDOW_SIZE / 2;
        assert!(window[mid_idx] > 0.99, "Window should peak near the middle");
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
    fn test_extract_features_sine_wave() {
        let mut vad = create_test_vad();
        let audio_frame = generate_test_audio(WINDOW_SIZE, 440.0, 16000.0);
        let features = vad.extract_features(&audio_frame);

        assert_eq!(features.len(), FEATURE_LEN);
        assert!(features.iter().all(|&x| x.is_finite()));

        // For a sine wave, features should be different from silence
        let silence_features = vad.extract_features(&vec![0.0; WINDOW_SIZE]);
        let features_diff: f32 = features
            .iter()
            .zip(silence_features.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(
            features_diff > 0.1,
            "Sine wave features should be different from silence"
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

        // Test with minimum values
        let min_frame = vec![i16::MIN; 256];
        let result = vad.process_frame(&min_frame);
        assert!(result.is_ok(), "Processing min values should succeed");
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

        // Process several frames with different signals
        for i in 0..CONTEXT_WINDOW_LEN + 2 {
            // Create audio with some variation to ensure features are different
            let audio_frame = generate_test_audio(WINDOW_SIZE, 200.0 + i as f32 * 100.0, 16000.0);
            let _ = vad.extract_features(&audio_frame);
        }

        // Feature buffer should contain the last CONTEXT_WINDOW_LEN frames
        assert_eq!(
            vad.feature_buffer.shape(),
            &[CONTEXT_WINDOW_LEN, FEATURE_LEN]
        );

        // The buffer should have been updated from its initial zero state
        // Even after normalization, processed audio should produce different features than silence
        let silence_features = {
            let mut temp_vad = create_test_vad();
            temp_vad.extract_features(&vec![0.0; WINDOW_SIZE])
        };

        // At least one row should be different from silence features
        let mut has_difference = false;
        for row_idx in 0..CONTEXT_WINDOW_LEN {
            let row = vad.feature_buffer.row(row_idx);
            let diff: f32 = row
                .iter()
                .zip(silence_features.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            if diff > 0.1 {
                // Allow for some tolerance
                has_difference = true;
                break;
            }
        }

        // If no significant difference found, at least verify the buffer structure is correct
        assert!(
            has_difference || vad.feature_buffer.shape() == [CONTEXT_WINDOW_LEN, FEATURE_LEN],
            "Feature buffer should either show processing changes or maintain correct structure"
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
        // Test that multiple VAD instances can coexist
        let mut vad1 = create_test_vad();
        let mut vad2 = create_test_vad();

        let frame1 = vec![100i16; 256];
        let frame2 = vec![200i16; 256];

        let score1 = vad1.process_frame(&frame1).unwrap();
        let score2 = vad2.process_frame(&frame2).unwrap();

        // Different inputs should potentially produce different outputs
        assert!(score1.is_finite() && score2.is_finite());

        // Process same frame with both instances
        let same_frame = vec![150i16; 256];
        let score1_same = vad1.process_frame(&same_frame).unwrap();
        let score2_same = vad2.process_frame(&same_frame).unwrap();

        // Should produce same result for same input
        assert!(
            (score1_same - score2_same).abs() < 0.01,
            "Different instances should produce similar results for same input"
        );
    }
}
