//! # Luv
/// Struct representing a color in CIALuv, a.k.a. L\*u\*v\*, color space
#[derive(Debug, Copy, Clone, Default)]
pub struct Luv {
    /// The L\* value (achromatic luminance) of the colour in 0–100 range.
    pub l: f32,
    /// The u\* value of the colour.
    ///
    /// Together with v\* value, it defines chromaticity of the colour.  The u\*
    /// coordinate represents colour’s position on red-green axis with negative
    /// values indicating more red and positive more green colour.  Typical
    /// values are in -100–100 range (but exact range for ‘valid’ colours
    /// depends on luminance and v\* value).
    pub u: f32,
    /// The u\* value of the colour.
    ///
    /// Together with u\* value, it defines chromaticity of the colour.  The v\*
    /// coordinate represents colour’s position on blue-yellow axis with
    /// negative values indicating more blue and positive more yellow colour.
    /// Typical values are in -100–100 range (but exact range for ‘valid’
    /// colours depends on luminance and u\* value).
    pub v: f32,
}

/// Struct representing a color in cylindrical CIELCh(uv) color space
#[derive(Debug, Copy, Clone, Default)]
pub struct LCh {
    /// The L\* value (achromatic luminance) of the colour in 0–100 range.
    ///
    /// This is the same value as in the [`Luv`] object.
    pub l: f32,
    /// The C\*_uv value (chroma) of the colour.
    ///
    /// Together with h_uv, it defines chromaticity of the colour.  The typical
    /// values of the coordinate go from zero up to around 150 (but exact range
    /// for ‘valid’ colours depends on luminance and hue).  Zero represents
    /// shade of grey.
    pub c: f32,
    /// The h_uv value (hue) of the colour measured in radians.
    ///
    /// Together with C\*_uv, it defines chromaticity of the colour.  The value
    /// represents an angle thus it wraps around τ.  Typically, the value will
    /// be in the -π–π range.  The value is undefined if C\*_uv is zero.
    pub h: f32,
}

const D65_XYZ: [f32; 3] = [95.047f32, 100.0f32, 108.883f32];

use crate::android_rgba::Rgba;
use crate::hsl::Rgb;
use crate::xyz::Xyz;

const WHITE_U_PRIME: f32 =
    4.0f32 * D65_XYZ[1] / (D65_XYZ[0] + 15.0 * D65_XYZ[1] + 3.0 * D65_XYZ[2]);
const WHITE_V_PRIME: f32 =
    9.0f32 * D65_XYZ[1] / (D65_XYZ[0] + 15.0 * D65_XYZ[1] + 3.0 * D65_XYZ[2]);

const CUTOFF_FORWARD_Y: f32 = (6f32 / 29f32) * (6f32 / 29f32) * (6f32 / 29f32);
const MULTIPLIER_FORWARD_Y: f32 = (29f32 / 3f32) * (29f32 / 3f32) * (29f32 / 3f32);
const SCALE_XYZ_FORWARD: f32 = 1f32 / 100f32;
const MULTIPLIER_INVERSE_Y: f32 = (3f32 / 29f32) * (3f32 / 29f32) * (3f32 / 29f32);

impl Luv {
    /// Constructs a new `Luv` from a three-element array of `u8`s
    ///
    /// # Examples
    ///
    /// ```
    /// let luv = luv::Luv::from_rgb(&[240, 33, 95]);
    /// assert_eq!(luv::Luv { l: 52.334686, u: 138.98639, v: 7.8476787 }, luv);
    /// ```
    pub fn from_rgb(rgb: &Rgb<u8>) -> Self {
        let xyz = Xyz::from_rgb(rgb);
        let [x, y, z] = [
            xyz.x * SCALE_XYZ_FORWARD,
            xyz.y * SCALE_XYZ_FORWARD,
            xyz.z * SCALE_XYZ_FORWARD,
        ];

        let l = (if y < CUTOFF_FORWARD_Y {
            MULTIPLIER_FORWARD_Y * y
        } else {
            116f32 * y.cbrt() - 16f32
        })
            .min(100f32)
            .max(0f32);
        let m_scale = 1f32 / (x + 15.0 * y + 3.0 * z);
        let u_prime = 4.0 * x * m_scale;
        let v_prime = 9.0 * y * m_scale;
        let u = 13f32 * l * (u_prime - WHITE_U_PRIME);
        let v = 13f32 * l * (v_prime - WHITE_V_PRIME);

        Luv { l, u, v }
    }

    /// Constructs a new `Luv` from a four-element array of `u8`s
    ///
    /// The `Luv` struct does not store alpha channel information, so the last
    /// `u8` representing alpha is discarded. This convenience method exists
    /// in order to easily measure colors already stored in an RGBA array.
    ///
    /// # Examples
    ///
    /// ```
    /// let luv = luv::Luv::from_rgba(&[240, 33, 95, 255]);
    /// assert_eq!(luv::Luv { l: 52.334686, u: 138.98639, v: 7.8476787 }, luv);
    /// ```
    pub fn from_rgba(rgba: &Rgba<u8>) -> Self {
        Luv::from_rgb(&rgba.to_rgb())
    }

    /// Returns the `Luv`'s color in RGB, in a 3-element array.
    ///
    /// # Examples
    ///
    /// ```
    /// let luv = luv::Luv { l: 52.334686, u: 138.98636, v: 7.8476787 };
    /// assert_eq!([240, 33, 95], luv.to_rgb());
    /// ```
    #[allow(dead_code)]
    pub fn to_rgb(&self) -> Rgb<u8> {
        if self.l <= 0f32 {
            return Xyz::new(0f32, 0f32, 0f32).to_rgb();
        }
        let l13 = 1f32 / (13f32 * self.l);
        let u = self.u * l13 + WHITE_U_PRIME;
        let v = self.v * l13 + WHITE_V_PRIME;
        let y = if self.l > 8f32 {
            ((self.l + 16f32) / 116f32).powf(3f32)
        } else {
            self.l * MULTIPLIER_INVERSE_Y
        };
        let den = 1f32 / (4f32 * v);
        let x = y * 9f32 * u * den;
        let z = y * (12.0 - 3.0 * u - 20.0 * v) * den;

        Xyz::new(
            Xyz::saturate_x(x * 100f32),
            Xyz::saturate_y(y * 100f32),
            Xyz::saturate_z(z * 100f32),
        ).to_rgb()
    }

    pub fn new(l: f32, u: f32, v: f32) -> Luv {
        Luv { l, u, v }
    }
}

impl LCh {
    /// Constructs a new `LCh` from a three-element array of `u8`s
    ///
    /// # Examples
    ///
    /// ```
    /// let rgb = [240, 33, 95];
    /// let lch = luv::LCh::from_rgb(&rgb);
    /// assert_eq!(luv::LCh {l: 52.334686, c: 139.20776, h: 0.056403756}, lch);
    /// assert_eq!(lch, luv::LCh::from_luv(luv::Luv::from_rgb(&rgb)));
    /// ```
    #[allow(dead_code)]
    pub fn from_rgb(rgb: &Rgb<u8>) -> Self {
        LCh::from_luv(Luv::from_rgb(rgb))
    }

    /// Constructs a new `LCh` from a four-element array of `u8`s
    ///
    /// The `LCh` struct does not store alpha channel information, so the last
    /// `u8` representing alpha is discarded. This convenience method exists
    /// in order to easily measure colors already stored in an RGBA array.
    ///
    /// # Examples
    ///
    /// ```
    /// let rgba = [240, 33, 95, 255];
    /// let lch = luv::LCh::from_rgba(&rgba);
    /// assert_eq!(luv::LCh {l: 52.334686, c: 139.20776, h: 0.056403756}, lch);
    /// assert_eq!(lch, luv::LCh::from_luv(luv::Luv::from_rgba(&rgba)));
    /// ```
    #[allow(dead_code)]
    pub fn from_rgba(rgba: &Rgba<u8>) -> Self {
        LCh::from_luv(Luv::from_rgba(rgba))
    }

    /// Constructs a new `LCh` from a `Luv`
    ///
    /// # Examples
    ///
    /// ```
    /// let luv = luv::Luv { l: 52.33686, u: 75.5516, v: 19.998878 };
    /// let lch = luv::LCh::from_luv(luv);
    /// assert_eq!(luv::LCh { l: 52.33686, c: 78.15369, h: 0.25877 }, lch);
    ///
    /// let luv = luv::Luv { l: 52.33686, u: 0.0, v: 0.0 };
    /// let lch = luv::LCh::from_luv(luv);
    /// assert_eq!(luv::LCh { l: 52.33686, c: 0.0, h: 0.0 }, lch);
    /// ```
    pub fn from_luv(luv: Luv) -> Self {
        LCh {
            l: luv.l,
            c: luv.u.hypot(luv.v),
            h: luv.v.atan2(luv.u),
        }
    }

    /// Returns the `LCh`'s color in RGB, in a 3-element array
    ///
    /// # Examples
    ///
    /// ```
    /// let mut lch = luv::LCh { l: 52.334686, c: 139.20773, h: 0.05640377 };
    /// assert_eq!([240, 33, 95], lch.to_rgb());
    ///
    /// lch.h += std::f32::consts::TAU;
    /// assert_eq!([240, 33, 95], lch.to_rgb());
    /// ```
    #[allow(dead_code)]
    pub fn to_rgb(&self) -> Rgb<u8> {
        self.to_luv().to_rgb()
    }

    /// Returns the `LCh`'s color in `Luv`
    ///
    /// Note that due to imprecision of floating point arithmetic, conversions
    /// between Luv and LCh are not stable.  A chain of Luv→LCh→Luv or
    /// LCh→Luv→LCh operations isn’t guaranteed to give back the source colour.
    ///
    /// # Examples
    ///
    /// ```
    /// let lch = luv::LCh { l: 52.33686, c: 78.15369, h: 0.25877 };
    /// let luv = lch.to_luv();
    /// assert_eq!(luv::Luv { l: 52.33686, u: 75.5516, v: 19.998878 }, luv);
    ///
    /// let lch = luv::LCh { l: 52.33686, c: 0.0, h: 0.25877 };
    /// let luv = lch.to_luv();
    /// assert_eq!(luv::Luv { l: 52.33686, u: 0.0, v: 0.0 }, luv);
    ///
    /// let inp = luv::Luv { l: 29.52658, u: 58.595745, v: -36.281406 };
    /// let lch = luv::LCh { l: 29.52658, c: 68.91881,  h: -0.5544043 };
    /// let out = luv::Luv { l: 29.52658, u: 58.59575,  v: -36.281406 };
    /// assert_eq!(lch, luv::LCh::from_luv(inp));
    /// assert_eq!(out, lch.to_luv());
    /// ```
    pub fn to_luv(&self) -> Luv {
        Luv {
            l: self.l,
            u: self.c * self.h.cos(),
            v: self.c * self.h.sin(),
        }
    }
}

impl std::cmp::PartialEq<Luv> for Luv {
    /// Compares two colours ignoring chromaticity if L\* is zero.
    fn eq(&self, other: &Self) -> bool {
        if self.l != other.l {
            false
        } else if self.l == 0.0 {
            true
        } else {
            self.u == other.u && self.v == other.v
        }
    }
}

impl PartialEq<LCh> for LCh {
    /// Compares two colours ignoring chromaticity if L\* is zero and hue if C\*
    /// is zero.  Hues which are τ apart are compared equal.
    fn eq(&self, other: &Self) -> bool {
        if self.l != other.l {
            false
        } else if self.l == 0.0 {
            true
        } else if self.c != other.c {
            false
        } else if self.c == 0.0 {
            true
        } else {
            use std::f32::consts::TAU;
            self.h.rem_euclid(TAU) == other.h.rem_euclid(TAU)
        }
    }
}
