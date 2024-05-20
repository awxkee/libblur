use crate::hsv::Hsv;
use crate::lab::Lab;
use crate::luv::Luv;

pub struct Hsl {
    pub h: f32,
    pub s: f32,
    pub l: f32,
}

pub struct Rgb<T> {
    pub r: T,
    pub g: T,
    pub b: T,
}

impl Rgb<u8> {
    pub fn to_hsl(&self) -> Hsl {
        rgb2hsl(self.r, self.g, self.b)
    }

    pub fn to_hsv(&self) -> Hsv {
        Hsv::from(self)
    }

    pub fn to_lab(&self) -> Lab {
        Lab::from_rgb(self)
    }

    pub fn to_luv(&self) -> Luv {
        Luv::from_rgb(self)
    }
}

impl<T> Rgb<T> {
    pub(crate) fn new(r: T, g: T, b: T) -> Rgb<T> {
        Rgb { r, g, b }
    }
}

impl Hsl {
    pub fn new(h: u16, s: u16, l: u16) -> Hsl {
        Hsl { h: h as f32, s: s as f32 / 100f32, l: l as f32 / 100f32 }
    }

    pub fn to_rgb8(&self) -> Rgb<u8> {
        let c = (1.0 - (2.0 * self.l - 1.0).abs()) * self.s;
        let x = c * (1.0 - ((self.h / 60.0) % 2.0 - 1.0).abs());
        let m = self.l - c / 2.0;

        let (r, g, b) = if self.h >= 0.0 && self.h < 60.0 {
            (c, x, 0.0)
        } else if self.h >= 60.0 && self.h < 120.0 {
            (x, c, 0.0)
        } else if self.h >= 120.0 && self.h < 180.0 {
            (0.0, c, x)
        } else if self.h >= 180.0 && self.h < 240.0 {
            (0.0, x, c)
        } else if self.h >= 240.0 && self.h < 300.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };

        Rgb::<u8> {
            r: ((r + m) * 255.0).round() as u8,
            g: ((g + m) * 255.0).round() as u8,
            b: ((b + m) * 255.0).round() as u8,
        }
    }

    pub fn get_saturation(&self) -> u16 {
        ((self.s * 100f32) as u16).min(100u16)
    }

    pub fn get_lightness(&self) -> u16 {
        ((self.l * 100f32) as u16).min(100u16)
    }

    pub fn get_hue(&self) -> u16 {
        (self.h as u16).min(360)
    }
}

fn rgb2hsl(o_r: u8, o_g: u8, o_b: u8) -> Hsl {
    let r = o_r as f32 / 255.0;
    let g = o_g as f32 / 255.0;
    let b = o_b as f32 / 255.0;

    let c_max = r.max(g).max(b);
    let c_min = r.min(g).min(b);
    let delta = c_max - c_min;

    let mut h = if delta == 0.0 {
        0.0
    } else if c_max == r {
        60.0 * (((g - b) / delta) % 6.0)
    } else if c_max == g {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };

    if h < 0.0 {
        h += 360.0;
    }

    let l = 0.5 * (c_max + c_min);
    let s = if delta == 0.0 {
        0.0
    } else {
        delta / (1.0 - (2.0 * l - 1.0).abs())
    };

    Hsl { h, s, l }
}