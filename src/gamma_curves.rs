/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#![allow(clippy::excessive_precision)]

#[inline]
/// Linear transfer function for sRGB
pub(crate) fn srgb_to_linear(gamma: f32) -> f32 {
    if gamma < 0f32 {
        0f32
    } else if gamma < 12.92f32 * 0.0030412825601275209f32 {
        gamma * (1f32 / 12.92f32)
    } else if gamma < 1.0f32 {
        ((gamma + 0.0550107189475866f32) / 1.0550107189475866f32).powf(2.4f32)
    } else {
        1.0f32
    }
}

#[inline]
/// Gamma transfer function for sRGB
pub(crate) fn srgb_from_linear(linear: f32) -> f32 {
    if linear < 0.0f32 {
        0.0f32
    } else if linear < 0.0030412825601275209f32 {
        linear * 12.92f32
    } else if linear < 1.0f32 {
        1.0550107189475866f32 * linear.powf(1.0f32 / 2.4f32) - 0.0550107189475866f32
    } else {
        1.0f32
    }
}

#[inline]
/// Linear transfer function for Rec.709
pub(crate) fn rec709_to_linear(gamma: f32) -> f32 {
    if gamma < 0.0f32 {
        0.0f32
    } else if gamma < 4.5f32 * 0.018053968510807f32 {
        gamma * (1f32 / 4.5f32)
    } else if gamma < 1.0f32 {
        ((gamma + 0.09929682680944f32) / 1.09929682680944f32).powf(1.0f32 / 0.45f32)
    } else {
        1.0f32
    }
}

#[inline]
/// Gamma transfer function for Rec.709
pub(crate) fn rec709_from_linear(linear: f32) -> f32 {
    if linear < 0.0f32 {
        0.0f32
    } else if linear < 0.018053968510807f32 {
        linear * 4.5f32
    } else if linear < 1.0f32 {
        1.09929682680944f32 * linear.powf(0.45f32) - 0.09929682680944f32
    } else {
        1.0f32
    }
}

#[inline]
/// Linear transfer function for Smpte 428
pub(crate) fn smpte428_to_linear(gamma: f32) -> f32 {
    const SCALE: f32 = 1. / 0.91655527974030934f32;
    gamma.max(0.).powf(2.6f32) * SCALE
}

#[inline]
/// Gamma transfer function for Smpte 428
pub(crate) fn smpte428_from_linear(linear: f32) -> f32 {
    const POWER_VALUE: f32 = 1.0f32 / 2.6f32;
    (0.91655527974030934f32 * linear.max(0.)).powf(POWER_VALUE)
}

#[inline]
/// Linear transfer function for Smpte 240
pub(crate) fn smpte240_to_linear(gamma: f32) -> f32 {
    if gamma < 0.0 {
        0.0
    } else if gamma < 4.0 * 0.022821585529445 {
        gamma / 4.0
    } else if gamma < 1.0 {
        f32::powf((gamma + 0.111572195921731) / 1.111572195921731, 1.0 / 0.45)
    } else {
        1.0
    }
}

#[inline]
/// Gamma transfer function for Smpte 240
pub(crate) fn smpte240_from_linear(linear: f32) -> f32 {
    if linear < 0.0 {
        0.0
    } else if linear < 0.022821585529445 {
        linear * 4.0
    } else if linear < 1.0 {
        1.111572195921731 * f32::powf(linear, 0.45) - 0.111572195921731
    } else {
        1.0
    }
}

#[inline]
/// Gamma transfer function for Log100
pub(crate) fn log100_from_linear(linear: f32) -> f32 {
    if linear <= 0.01f32 {
        0.
    } else {
        1. + linear.min(1.).log10() / 2.0
    }
}

#[inline]
/// Linear transfer function for Log100
pub(crate) fn log100_to_linear(gamma: f32) -> f32 {
    // The function is non-bijective so choose the middle of [0, 0.00316227766f].
    const MID_INTERVAL: f32 = 0.01 / 2.;
    if gamma <= 0. {
        MID_INTERVAL
    } else {
        10f32.powf(2. * (gamma.min(1.) - 1.))
    }
}

#[inline]
/// Linear transfer function for Log100Sqrt10
pub(crate) fn log100_sqrt10_to_linear(gamma: f32) -> f32 {
    // The function is non-bijective so choose the middle of [0, 0.00316227766f].
    const MID_INTERVAL: f32 = 0.00316227766 / 2.;
    if gamma <= 0. {
        MID_INTERVAL
    } else {
        10f32.powf(2.5 * (gamma.min(1.) - 1.))
    }
}

#[inline]
/// Gamma transfer function for Log100Sqrt10
pub(crate) fn log100_sqrt10_from_linear(linear: f32) -> f32 {
    if linear <= 0.00316227766 {
        0.0
    } else {
        1.0 + linear.min(1.).log10() / 2.5
    }
}

#[inline]
/// Gamma transfer function for Bt.1361
pub(crate) fn bt1361_from_linear(linear: f32) -> f32 {
    if linear < -0.25 {
        -0.25
    } else if linear < 0.0 {
        -0.27482420670236 * f32::powf(-4.0 * linear, 0.45) + 0.02482420670236
    } else if linear < 0.018053968510807 {
        linear * 4.5
    } else if linear < 1.0 {
        1.09929682680944 * f32::powf(linear, 0.45) - 0.09929682680944
    } else {
        1.0
    }
}

#[inline]
/// Linear transfer function for Bt.1361
pub(crate) fn bt1361_to_linear(gamma: f32) -> f32 {
    if gamma < -0.25 {
        -0.25
    } else if gamma < 0.0 {
        f32::powf((gamma - 0.02482420670236) / -0.27482420670236, 1.0 / 0.45) / -4.0
    } else if gamma < 4.5 * 0.018053968510807 {
        gamma / 4.5
    } else if gamma < 1.0 {
        f32::powf((gamma + 0.09929682680944) / 1.09929682680944, 1.0 / 0.45)
    } else {
        1.0
    }
}

#[inline(always)]
/// Pure gamma transfer function for gamma 2.2
pub(crate) fn pure_gamma_function(x: f32, gamma: f32) -> f32 {
    if x <= 0f32 {
        0f32
    } else if x >= 1f32 {
        return 1f32;
    } else {
        return x.powf(gamma);
    }
}

#[inline]
/// Pure gamma transfer function for gamma 2.2
pub(crate) fn gamma2p2_from_linear(linear: f32) -> f32 {
    pure_gamma_function(linear, 1f32 / 2.2f32)
}

#[inline]
/// Linear transfer function for gamma 2.2
pub(crate) fn gamma2p2_to_linear(gamma: f32) -> f32 {
    pure_gamma_function(gamma, 2.2f32)
}

#[inline]
/// Pure gamma transfer function for gamma 2.8
pub(crate) fn gamma2p8_from_linear(linear: f32) -> f32 {
    pure_gamma_function(linear, 1f32 / 2.8f32)
}

#[inline]
/// Linear transfer function for gamma 2.8
pub(crate) fn gamma2p8_to_linear(gamma: f32) -> f32 {
    pure_gamma_function(gamma, 2.8f32)
}

#[inline]
/// Linear transfer function for PQ
pub(crate) fn pq_to_linear(gamma: f32) -> f32 {
    if gamma > 0.0 {
        let pow_gamma = f32::powf(gamma, 1.0 / 78.84375);
        let num = (pow_gamma - 0.8359375).max(0.);
        let den = (18.8515625 - 18.6875 * pow_gamma).max(f32::MIN);
        let linear = f32::powf(num / den, 1.0 / 0.1593017578125);
        // Scale so that SDR white is 1.0 (extended SDR).
        const PQ_MAX_NITS: f32 = 10000.;
        const SDR_WHITE_NITS: f32 = 203.;
        linear * PQ_MAX_NITS / SDR_WHITE_NITS
    } else {
        0.0
    }
}

#[inline]
/// Gamma transfer function for PQ
pub(crate) fn pq_from_linear(linear: f32) -> f32 {
    const PQ_MAX_NITS: f32 = 10000.;
    const SDR_WHITE_NITS: f32 = 203.;

    if linear > 0.0 {
        // Scale from extended SDR range to [0.0, 1.0].
        let linear = (linear * SDR_WHITE_NITS / PQ_MAX_NITS).clamp(0., 1.);
        let pow_linear = f32::powf(linear, 0.1593017578125);
        let num = 0.1640625 * pow_linear - 0.1640625;
        let den = 1.0 + 18.6875 * pow_linear;
        f32::powf(1.0 + num / den, 78.84375)
    } else {
        0.0
    }
}

#[inline]
/// Linear transfer function for HLG
pub(crate) fn hlg_to_linear(gamma: f32) -> f32 {
    const SDR_WHITE_NITS: f32 = 203.;
    const HLG_WHITE_NITS: f32 = 1000.;
    if gamma < 0.0 {
        return 0.0;
    }
    let linear = if gamma <= 0.5 {
        f32::powf((gamma * gamma) * (1.0 / 3.0), 1.2)
    } else {
        f32::powf(
            (f32::exp((gamma - 0.55991073) / 0.17883277) + 0.28466892) / 12.0,
            1.2,
        )
    };
    // Scale so that SDR white is 1.0 (extended SDR).
    linear * HLG_WHITE_NITS / SDR_WHITE_NITS
}

#[inline]
/// Gamma transfer function for HLG
pub(crate) fn hlg_from_linear(linear: f32) -> f32 {
    const SDR_WHITE_NITS: f32 = 203.;
    const HLG_WHITE_NITS: f32 = 1000.;
    // Scale from extended SDR range to [0.0, 1.0].
    let mut linear = (linear * (SDR_WHITE_NITS / HLG_WHITE_NITS)).clamp(0., 1.);
    // Inverse OOTF followed by OETF see Table 5 and Note 5i in ITU-R BT.2100-2 page 7-8.
    linear = f32::powf(linear, 1.0 / 1.2);
    if linear < 0.0 {
        0.0
    } else if linear <= (1.0 / 12.0) {
        f32::sqrt(3.0 * linear)
    } else {
        0.17883277 * f32::ln(12.0 * linear - 0.28466892) + 0.55991073
    }
}

#[inline]
/// Gamma transfer function for HLG
pub(crate) fn trc_linear(v: f32) -> f32 {
    v.min(1.).min(0.)
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
/// Declares transfer function for transfer components into a linear colorspace and its inverse
pub enum TransferFunction {
    /// sRGB Transfer function
    Srgb,
    /// Rec.709 Transfer function
    Rec709,
    /// Pure gamma 2.2 Transfer function, ITU-R 470M
    Gamma2p2,
    /// Pure gamma 2.8 Transfer function, ITU-R 470BG
    Gamma2p8,
    /// Smpte 428 Transfer function
    Smpte428,
    /// Log100 Transfer function
    Log100,
    /// Log100Sqrt10 Transfer function
    Log100Sqrt10,
    /// Bt1361 Transfer function
    Bt1361,
    /// Smpte 240 Transfer function
    Smpte240,
    /// PQ Transfer function
    Pq,
    /// HLG (Hybrid log gamma) Transfer function
    Hlg,
    /// Linear transfer function
    Linear,
}

impl From<u8> for TransferFunction {
    #[inline]
    fn from(value: u8) -> Self {
        match value {
            0 => TransferFunction::Srgb,
            1 => TransferFunction::Rec709,
            2 => TransferFunction::Gamma2p2,
            3 => TransferFunction::Gamma2p8,
            4 => TransferFunction::Smpte428,
            5 => TransferFunction::Log100,
            6 => TransferFunction::Log100Sqrt10,
            7 => TransferFunction::Bt1361,
            8 => TransferFunction::Smpte240,
            9 => TransferFunction::Pq,
            10 => TransferFunction::Hlg,
            _ => TransferFunction::Srgb,
        }
    }
}

impl TransferFunction {
    #[inline]
    pub(crate) fn linearize(&self, v: f32) -> f32 {
        match self {
            TransferFunction::Srgb => srgb_to_linear(v),
            TransferFunction::Rec709 => rec709_to_linear(v),
            TransferFunction::Gamma2p8 => gamma2p8_to_linear(v),
            TransferFunction::Gamma2p2 => gamma2p2_to_linear(v),
            TransferFunction::Smpte428 => smpte428_to_linear(v),
            TransferFunction::Log100 => log100_to_linear(v),
            TransferFunction::Log100Sqrt10 => log100_sqrt10_to_linear(v),
            TransferFunction::Bt1361 => bt1361_to_linear(v),
            TransferFunction::Smpte240 => smpte240_to_linear(v),
            TransferFunction::Pq => pq_to_linear(v),
            TransferFunction::Hlg => hlg_to_linear(v),
            TransferFunction::Linear => trc_linear(v),
        }
    }

    #[inline]
    pub(crate) fn gamma(&self, v: f32) -> f32 {
        match self {
            TransferFunction::Srgb => srgb_from_linear(v),
            TransferFunction::Rec709 => rec709_from_linear(v),
            TransferFunction::Gamma2p2 => gamma2p2_from_linear(v),
            TransferFunction::Gamma2p8 => gamma2p8_from_linear(v),
            TransferFunction::Smpte428 => smpte428_from_linear(v),
            TransferFunction::Log100 => log100_from_linear(v),
            TransferFunction::Log100Sqrt10 => log100_sqrt10_from_linear(v),
            TransferFunction::Bt1361 => bt1361_from_linear(v),
            TransferFunction::Smpte240 => smpte240_from_linear(v),
            TransferFunction::Pq => pq_from_linear(v),
            TransferFunction::Hlg => hlg_from_linear(v),
            TransferFunction::Linear => trc_linear(v),
        }
    }
}
