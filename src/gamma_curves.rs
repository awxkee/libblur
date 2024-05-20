pub mod gamma_curve {
    pub fn srgb_to_linear(gamma: f32) -> f32 {
        return if gamma < 0f32 {
            0f32
        } else if gamma < 12.92f32 * 0.0030412825601275209f32 {
            gamma / 12.92f32
        } else if gamma < 1.0f32 {
            ((gamma + 0.0550107189475866f32) / 1.0550107189475866f32).powf(2.4f32)
        } else {
            1.0f32
        }
    }

    pub fn srgb_from_linear(linear: f32) -> f32 {
        return if linear < 0.0f32 {
            0.0f32
        } else if linear < 0.0030412825601275209f32 {
            linear * 12.92f32
        } else if linear < 1.0f32 {
            1.0550107189475866f32 * linear.powf(1.0f32 / 2.4f32) - 0.0550107189475866f32
        } else {
            1.0f32
        }
    }
}