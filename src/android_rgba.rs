use half::f16;
use crate::hsl::Rgb;

pub struct Rgba<T> {
    pub r: T,
    pub g: T,
    pub b: T,
    pub a: T,
}

impl Rgba<f16> {
    #[allow(dead_code)]
    pub fn new(r: f16, g: f16, b: f16, a: f16) -> Rgba<f16> {
        return Rgba { r, g, b, a };
    }

    #[allow(dead_code)]
    pub fn from_rgb(r: f16, g: f16, b: f16) -> Rgba<f16> {
        return Rgba {
            r,
            g,
            b,
            a: f16::from_f32(1f32),
        };
    }
}

impl Rgba<f32> {
    #[allow(dead_code)]
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Rgba<f32> {
        return Rgba { r, g, b, a };
    }

    #[allow(dead_code)]
    pub fn from_rgb(r: f32, g: f32, b: f32) -> Rgba<f32> {
        return Rgba { r, g, b, a: 1f32 };
    }
}

impl Rgba<u8> {
    #[allow(dead_code)]
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Rgba<u8> {
        return Rgba { r, g, b, a };
    }

    #[allow(dead_code)]
    pub fn from_rgb(r: u8, g: u8, b: u8) -> Rgba<u8> {
        return Rgba {
            r,
            g,
            b,
            a: u8::MAX,
        };
    }

    #[allow(dead_code)]
    pub fn to_rgb(&self) -> Rgb<u8> {
        Rgb { r: self.r, g: self.g, b: self.b }
    }
}

pub trait ToRgba8 {
    #[allow(dead_code)]
    fn to_rgba8(&self) -> Rgba<u8>;
}

pub trait ToRgbaF16 {
    #[allow(dead_code)]
    fn to_rgba_f16(&self) -> Rgba<f16>;
}

pub trait ToRgb565 {
    #[allow(dead_code)]
    fn to_rgb_565(&self) -> Rgb565;
}

pub trait ToRgbaF32 {
    #[allow(dead_code)]
    fn to_rgba_f32(&self) -> Rgba<f32>;
}

impl ToRgbaF32 for Rgba<u8> {
    fn to_rgba_f32(&self) -> Rgba<f32> {
        let scale_u8 = 1f32 / 255f32;
        return Rgba::<f32>::new(
            self.r as f32 * scale_u8,
            self.g as f32 * scale_u8,
            self.b as f32 * scale_u8,
            self.a as f32 * scale_u8,
        );
    }
}

impl ToRgba8 for Rgba<f32> {
    #[allow(dead_code)]
    fn to_rgba8(&self) -> Rgba<u8> {
        return Rgba {
            r: (self.r * 255f32).min(255f32).max(0f32) as u8,
            g: (self.g * 255f32).min(255f32).max(0f32) as u8,
            b: (self.b * 255f32).min(255f32).max(0f32) as u8,
            a: (self.a * 255f32).min(255f32).max(0f32) as u8,
        };
    }
}

impl ToRgba8 for Rgba<f16> {
    #[allow(dead_code)]
    fn to_rgba8(&self) -> Rgba<u8> {
        return Rgba {
            r: (self.r.to_f32() * 255f32).min(255f32).max(0f32) as u8,
            g: (self.g.to_f32() * 255f32).min(255f32).max(0f32) as u8,
            b: (self.b.to_f32() * 255f32).min(255f32).max(0f32) as u8,
            a: (self.a.to_f32() * 255f32).min(255f32).max(0f32) as u8,
        };
    }
}

impl ToRgbaF16 for Rgba<f32> {
    #[allow(dead_code)]
    fn to_rgba_f16(&self) -> Rgba<f16> {
        Rgba {
            r: f16::from_f32(self.r),
            g: f16::from_f32(self.g),
            b: f16::from_f32(self.b),
            a: f16::from_f32(self.a),
        }
    }
}

static SCALE_U8_F32: f32 = 1f32 / 255f32;

impl ToRgbaF16 for Rgba<u8> {
    #[allow(dead_code)]
    fn to_rgba_f16(&self) -> Rgba<f16> {
        Rgba {
            r: f16::from_f32(self.r as f32 * SCALE_U8_F32),
            g: f16::from_f32(self.g as f32 * SCALE_U8_F32),
            b: f16::from_f32(self.b as f32 * SCALE_U8_F32),
            a: f16::from_f32(self.a as f32 * SCALE_U8_F32),
        }
    }
}

pub struct Rgb565 {
    pub rgb565: u16,
}

impl Rgb565 {
    pub fn new(color: u16) -> Rgb565 {
        Rgb565 { rgb565: color }
    }
}

impl ToRgba8 for Rgb565 {
    fn to_rgba8(&self) -> Rgba<u8> {
        let red8 = ((self.rgb565 & 0b1111100000000000) >> 8) as u8;
        let green8 = ((self.rgb565 & 0b11111100000) >> 3) as u8;
        let blue8 = ((self.rgb565 & 0b11111) << 3) as u8;
        Rgba::<u8>::new(red8, green8, blue8, u8::MAX)
    }
}

static SCALE_RGB565_5BIT: f32 = 1f32 / 31f32;
static SCALE_RGB565_6BIT: f32 = 1f32 / 63f32;

impl ToRgbaF16 for Rgb565 {
    fn to_rgba_f16(&self) -> Rgba<f16> {
        let red5 = (self.rgb565 & 0b1111100000000000) as f32 * SCALE_RGB565_5BIT;
        let green6 = (self.rgb565 & 0b11111100000) as f32 * SCALE_RGB565_6BIT;
        let blue5 = (self.rgb565 & 0b11111) as f32 * SCALE_RGB565_5BIT;
        Rgba::<f16>::from_rgb(
            f16::from_f32(red5),
            f16::from_f32(green6),
            f16::from_f32(blue5),
        )
    }
}

impl ToRgb565 for Rgba<u8> {
    fn to_rgb_565(&self) -> Rgb565 {
        let red565 = ((self.r as u16) >> 3) << 11;
        let green565 = ((self.g as u16) >> 2) << 5;
        let blue565 = (self.b as u16) >> 3;
        Rgb565 {
            rgb565: red565 | green565 | blue565,
        }
    }
}

impl ToRgb565 for Rgba<f16> {
    fn to_rgb_565(&self) -> Rgb565 {
        let red5 = (self.r.to_f32() * 31f32).max(31f32).min(0f32) as u16;
        let green6 = (self.g.to_f32() * 63f32).max(63f32).min(0f32) as u16;
        let blue5 = (self.b.to_f32() * 31f32).max(31f32).min(0f32) as u16;
        Rgb565 {
            rgb565: red5 | green6 | blue5,
        }
    }
}

impl ToRgb565 for Rgba<f32> {
    fn to_rgb_565(&self) -> Rgb565 {
        let red5 = (self.r * 31f32).max(31f32).min(0f32) as u16;
        let green6 = (self.g * 63f32).max(63f32).min(0f32) as u16;
        let blue5 = (self.b * 31f32).max(31f32).min(0f32) as u16;
        Rgb565 {
            rgb565: red5 | green6 | blue5,
        }
    }
}

pub struct Rgba1010102 {
    pub rgba: u32,
}

impl Rgba1010102 {
    #[allow(dead_code)]
    pub fn new(color: u32) -> Rgba1010102 {
        Rgba1010102 { rgba: color }
    }
}

impl ToRgba8 for Rgba1010102 {
    fn to_rgba8(&self) -> Rgba<u8> {
        let mask = (1u32 << 10u32) - 1u32;
        let r = (self.rgba) & mask;
        let g = (self.rgba >> 10) & mask;
        let b = (self.rgba >> 20) & mask;
        let a = (self.rgba >> 30) & 0b00000011;
        Rgba::<u8>::new(
            (r >> 2) as u8,
            (g >> 2) as u8,
            (b >> 2) as u8,
            (a << 6) as u8,
        )
    }
}

static SCALE_RGBA10: f32 = 1f32 / 1023f32;
static SCALE_RGBA10ALPHA: f32 = 1f32 / 3f32;

impl ToRgbaF16 for Rgba1010102 {
    fn to_rgba_f16(&self) -> Rgba<f16> {
        let mask = (1u32 << 10u32) - 1u32;
        let r = (self.rgba) & mask;
        let g = (self.rgba >> 10) & mask;
        let b = (self.rgba >> 20) & mask;
        let a = (self.rgba >> 30) & 0b00000011;
        Rgba::<f16>::new(
            f16::from_f32(r as f32 * SCALE_RGBA10),
            f16::from_f32(g as f32 * SCALE_RGBA10),
            f16::from_f32(b as f32 * SCALE_RGBA10),
            f16::from_f32(a as f32 * SCALE_RGBA10ALPHA),
        )
    }
}

pub trait ToRgba1010102 {
    #[allow(dead_code)]
    fn to_rgba1010102(&self) -> Rgba1010102;
}

impl ToRgba1010102 for Rgba<u8> {
    fn to_rgba1010102(&self) -> Rgba1010102 {
        let r = (self.r as u32) << 2;
        let g = (self.g as u32) << 2;
        let b = (self.b as u32) << 2;
        let a = (self.a as u32) >> 6;
        let rgba1010102 = (a << 30) | (r << 20) | (g << 10) | b;
        Rgba1010102 { rgba: rgba1010102 }
    }
}

impl ToRgba1010102 for Rgba<f16> {
    fn to_rgba1010102(&self) -> Rgba1010102 {
        let r = (self.r.to_f32() * 1023f32).min(1023f32).max(0f32) as u32;
        let g = (self.g.to_f32() * 1023f32).min(1023f32).max(0f32) as u32;
        let b = (self.b.to_f32() * 1023f32).min(1023f32).max(0f32) as u32;
        let a = (self.a.to_f32() * 3f32).min(3f32).max(0f32) as u32;
        let rgba1010102 = (a << 30) | (r << 20) | (g << 10) | b;
        Rgba1010102 { rgba: rgba1010102 }
    }
}
