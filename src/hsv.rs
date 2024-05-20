use crate::hsl::Rgb;

pub struct Hsv {
    pub h: f32,
    pub s: f32,
    pub v: f32,
}

static HSV_U8_SCALE: f32 = 1f32 / 255f32;
static HSV_PERCENTAGE_SCALE: f32 = 1f32 / 100f32;

impl Hsv {
    #[allow(dead_code)]
    pub fn new(h: u16, s: u16, l: u16) -> Hsv {
        Hsv {
            h: h as f32,
            s: s as f32 * HSV_PERCENTAGE_SCALE,
            v: l as f32 * HSV_PERCENTAGE_SCALE,
        }
    }

    #[allow(dead_code)]
    pub fn from_components(h: f32, s: f32, v: f32) -> Hsv {
        Hsv { h, s, v }
    }

    #[allow(dead_code)]
    pub fn from(rgb: &Rgb<u8>) -> Hsv {
        let (h, s, v) = rgb_to_hsv(
            rgb.r as f32 * HSV_U8_SCALE,
            rgb.g as f32 * HSV_U8_SCALE,
            rgb.b as f32 * HSV_U8_SCALE,
        );
        return Hsv { h, s, v };
    }

    #[allow(dead_code)]
    pub fn to_rgb8(&self) -> Rgb<u8> {
        let (rf, gf, bf) = hsv_to_rgb(self.h, self.s, self.v);
        return Rgb {
            r: (rf * 255f32) as u8,
            g: (gf * 255f32) as u8,
            b: (bf * 255f32) as u8,
        };
    }

    #[allow(dead_code)]
    pub fn get_hue(&self) -> f32 {
        self.h
    }

    #[allow(dead_code)]
    pub fn get_saturation(&self) -> f32 {
        self.s
    }

    #[allow(dead_code)]
    pub fn get_value(&self) -> f32 {
        self.v
    }

    #[allow(dead_code)]
    pub fn get_hue_p(&self) -> u16 {
        self.h.max(0f32).min(360f32) as u16
    }

    #[allow(dead_code)]
    pub fn get_saturation_p(&self) -> u16 {
        (self.s * 100f32).max(0f32).min(100f32) as u16
    }

    #[allow(dead_code)]
    pub fn get_value_p(&self) -> u16 {
        (self.v * 100f32).max(0f32).min(100f32) as u16
    }
}

fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let c_max = r.max(g).max(b);
    let c_min = r.min(g).min(b);
    let delta = c_max - c_min;

    let mut h = 0f32;
    let mut s = 0f32;
    let v = c_max;

    if delta > 0f32 {
        if c_max == r {
            h = 60f32 * (((g - b) / delta) % 6f32);
        } else if c_max == g {
            h = 60f32 * (((b - r) / delta) + 2f32);
        } else if c_max == b {
            h = 60f32 * (((r - g) / delta) + 4f32);
        }

        if c_max > 0f32 {
            s = delta / c_max;
        }
    }

    if h < 0f32 {
        h += 360f32;
    }

    (h, s, v)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let c = v * s;
    let h_prime = (h / 60f32) % 6f32;
    let x = c * (1f32 - ((h_prime % 2f32) - 1f32).abs());
    let m = v - c;

    let (r, g, b) = if h_prime >= 0f32 && h_prime < 1f32 {
        (c, x, 0f32)
    } else if h_prime >= 1f32 && h_prime < 2f32 {
        (x, c, 0f32)
    } else if h_prime >= 2f32 && h_prime < 3f32 {
        (0f32, c, x)
    } else if h_prime >= 3f32 && h_prime < 4f32 {
        (0f32, x, c)
    } else if h_prime >= 4f32 && h_prime < 5f32 {
        (x, 0f32, c)
    } else if h_prime >= 5f32 && h_prime < 6f32 {
        (c, 0f32, x)
    } else {
        (0f32, 0f32, 0f32)
    };

    (r + m, g + m, b + m)
}
