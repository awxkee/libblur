use num_traits::{AsPrimitive, Euclid, FromPrimitive, Signed};

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Default)]
/// Declares an edge handling mode
pub enum EdgeMode {
    /// If kernel goes out of bounds it will be clipped to an edge and edge pixel replicated across filter
    #[default]
    Clamp = 0,
    /// If kernel goes out of bounds it will be clipped, this is a slightly faster than clamp, however have different visual effects at the edge.
    /// *Kernel clip is supported only for clear gaussian blur and not supported in any approximations!*
    KernelClip = 1,
    /// If filter goes out of bounds image will be replicated with rule `cdefgh|abcdefgh|abcdefg`
    Wrap = 2,
    /// If filter goes out of bounds image will be replicated with rule `fedcba|abcdefgh|hgfedcb`
    Reflect = 3,
}

impl From<usize> for EdgeMode {
    fn from(value: usize) -> Self {
        return match value {
            0 => EdgeMode::Clamp,
            1 => EdgeMode::KernelClip,
            2 => EdgeMode::Wrap,
            3 => EdgeMode::Reflect,
            _ => {
                panic!("Unknown edge mode for value: {}", value);
            }
        };
    }
}

#[inline(always)]
pub(crate) fn reflect_index<
    T: Copy
        + 'static
        + PartialOrd
        + PartialEq
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + Euclid
        + FromPrimitive
        + Signed
        + AsPrimitive<usize>,
>(
    i: T,
    n: T,
) -> usize
where
    i64: AsPrimitive<T>,
{
    let i = (i - n).rem_euclid(&(2i64.as_() * n));
    let i = (i - n).abs();
    return i.as_();
}

/**
    RRRRRR  OOOOO  U     U TTTTTTT IIIII NN   N EEEEEEE SSSSS
    R     R O     O U     U   T     I   I N N  N E       S
   RRRRRR  O     O U     U   T     I   I N  N N EEEEE    SSS
   R   R   O     O U     U   T     I   I N   NN E            S
   R    R   OOOOO   UUUUU    T    IIIII N    N EEEEEEE  SSSSS
**/

#[macro_export]
macro_rules! clamp_edge {
    ($edge_mode:expr, $value:expr, $min:expr, $max:expr) => {{
        match $edge_mode {
            EdgeMode::Clamp | EdgeMode::KernelClip => {
                (std::cmp::min(std::cmp::max($value, $min), $max) as u32) as usize
            }
            EdgeMode::Wrap => {
                let cx = $value.rem_euclid($max);
                cx as usize
            }
            EdgeMode::Reflect => {
                let cx = reflect_index($value, $max);
                cx as usize
            }
        }
    }};
}
