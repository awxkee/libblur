#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Default)]
/// Declares an edge handling mode
pub enum EdgeMode {
    /// If kernel goes out of bounds it will be clipped to an edge and edge pixel replicated across filter
    Clamp = 0,
    #[default]
    /// If kernel goes out of bounds it will be clipped, this is a slightly faster than clamp, however have different visual effects at the edge
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
pub fn reflect_index(i: i64, n: i64) -> usize {
    let i = (i - n).rem_euclid(2i64 * n);
    let i = (i - n).abs();
    return i as usize;
}