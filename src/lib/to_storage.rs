use half::f16;
use num_traits::Float;

/// Helper trait to convert and round if we are storing in integral type
pub(crate) trait ToStorage<T>: 'static + Copy + Float
where
    T: 'static + Copy,
{
    /// Convert a value to another, using the `to` operator.
    fn to_(self) -> T;
}

macro_rules! impl_to_integral_storage {
    ($from:ty, $to:ty) => {
        impl ToStorage<$to> for $from {
            fn to_(self) -> $to {
                self.round() as $to
            }
        }
    };
}

impl_to_integral_storage!(f32, u8);
impl_to_integral_storage!(f64, u8);
impl_to_integral_storage!(f32, u16);
impl_to_integral_storage!(f64, u16);
impl_to_integral_storage!(f32, u32);
impl_to_integral_storage!(f64, u32);
impl_to_integral_storage!(f32, usize);
impl_to_integral_storage!(f64, usize);

macro_rules! impl_to_float_storage {
    ($from:ty, $to:ty) => {
        impl ToStorage<$to> for $from {
            fn to_(self) -> $to {
                self as $to
            }
        }
    };
}

impl_to_float_storage!(f32, f32);
impl_to_float_storage!(f64, f64);
impl_to_float_storage!(f64, f32);
impl_to_float_storage!(f32, f64);

impl ToStorage<f16> for f32 {
    fn to_(self) -> f16 {
        f16::from_f32(self)
    }
}

impl ToStorage<f16> for f64 {
    fn to_(self) -> f16 {
        f16::from_f64(self)
    }
}
