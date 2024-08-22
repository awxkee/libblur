// Copyright (c) Radzivon Bartoshyk. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1.  Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2.  Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3.  Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
    /// Note that for gaussian blur *Wrap* is significantly slower when NEON or SSE is available than *KernelClip* and *Clamp*
    Wrap = 2,
    /// If filter goes out of bounds image will be replicated with rule `fedcba|abcdefgh|hgfedcb`
    /// Note that for gaussian blur *Reflect* is significantly slower when NEON or SSE is available than *KernelClip* and *Clamp*
    Reflect = 3,
    /// If filter goes out of bounds image will be replicated with rule `gfedcb|abcdefgh|gfedcba`
    /// Note that for gaussian blur *Reflect101* is significantly slower when NEON or SSE is available than *KernelClip* and *Clamp*
    Reflect101 = 4,
}

impl From<usize> for EdgeMode {
    fn from(value: usize) -> Self {
        match value {
            0 => EdgeMode::Clamp,
            1 => EdgeMode::KernelClip,
            2 => EdgeMode::Wrap,
            3 => EdgeMode::Reflect,
            4 => EdgeMode::Reflect101,
            _ => {
                panic!("Unknown edge mode for value: {}", value);
            }
        }
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
    i.as_()
}

#[inline(always)]
#[allow(dead_code)]
pub(crate) fn reflect_index_101<
    T: Copy
        + 'static
        + PartialOrd
        + PartialEq
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + Euclid
        + FromPrimitive
        + Signed
        + AsPrimitive<usize>
        + Ord,
>(
    i: T,
    n: T,
) -> usize
where
    i64: AsPrimitive<T>,
{
    if i < T::from_i32(0i32).unwrap() {
        let i = (i - n).rem_euclid(&(2i64.as_() * n));
        let i = (i - n).abs();
        return (i + T::from_i32(1).unwrap()).min(n).as_();
    }
    if i > n {
        let i = (i - n).rem_euclid(&(2i64.as_() * n));
        let i = (i - n).abs();
        return (i - T::from_i32(1i32).unwrap())
            .max(T::from_i32(0i32).unwrap())
            .as_();
    }
    i.as_()
}

#[macro_export]
macro_rules! reflect_101 {
    ($i:expr, $n:expr) => {{
        if $i < 0 {
            let i = ($i - $n).rem_euclid(2i64 * $n as i64);
            let i = (i - $n).abs();
            (i + 1).min($n) as usize
        } else if $i > $n {
            let i = ($i - $n).rem_euclid(2i64 * $n as i64);
            let i = (i - $n).abs();
            (i - 1).max(0) as usize
        } else {
            $i as usize
        }
    }};
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
            EdgeMode::Reflect101 => {
                let cx = reflect_101!($value, $max);
                cx as usize
            }
        }
    }};
}
