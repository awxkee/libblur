use crate::hsl::Rgb;
use crate::xyz::Xyz;

/// A CIELAB color.
#[derive(Copy, Clone, Debug, Default)]
pub struct Lab {
    pub l: f32,
    pub a: f32,
    pub b: f32,
}

impl Lab {
    /// Create a new CIELAB color.
    ///
    /// `l`: lightness component (0 to 100)
    ///
    /// `a`: green (negative) and red (positive) component.
    ///
    /// `b`: blue (negative) and yellow (positive) component.
    #[inline]
    pub fn new(l: f32, a: f32, b: f32) -> Self {
        Self { l, a, b }
    }
}

impl Lab {
    pub fn from_rgb(rgb: &Rgb<u8>) -> Self {
        let xyz = Xyz::from_rgb(rgb);
        let x = xyz.x / 95.047;
        let y = xyz.y / 100.0;
        let z = xyz.z / 108.883;
        let x = if x > 0.008856 { x.cbrt() } else { 7.787 * x + 16.0 / 116.0 };
        let y = if y > 0.008856 { y.cbrt() } else { 7.787 * y + 16.0 / 116.0 };
        let z = if z > 0.008856 { z.cbrt() } else { 7.787 * z + 16.0 / 116.0 };
        Self::new(
            (116.0 * y) - 16.0,
            500.0 * (x - y),
            200.0 * (y - z)
        )
    }
}

impl Lab {
    pub fn to_rgb8(&self) -> Rgb<u8> {
        let y = (self.l + 16.0) / 116.0;
        let x = self.a / 500.0 + y;
        let z = y - self.b / 200.0;
        let x3 = x.powf(3.0);
        let y3 = y.powf(3.0);
        let z3 = z.powf(3.0);
        let x = 95.047 * if x3 > 0.008856 { x3 } else { (x - 16.0 / 116.0) / 7.787 };
        let y = 100.0 * if y3 > 0.008856 { y3 } else { (y - 16.0 / 116.0) / 7.787 };
        let z = 108.883 * if z3 > 0.008856 { z3 } else { (z - 16.0 / 116.0) / 7.787 };
        Xyz::new(x, y, z).to_rgb()
    }

    pub fn to_rgb(&self) -> Rgb<u8> {
        self.to_rgb8()
    }
}