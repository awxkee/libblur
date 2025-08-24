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

use std::ops::Index;

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Default)]
/// Declares a 2D edge handling mode.
///
/// `EdgeMode2D` allows specifying how out-of-bounds pixel access
/// is handled independently in the horizontal and vertical directions.
pub struct EdgeMode2D {
    /// Edge handling mode in the horizontal direction.
    pub horizontal: EdgeMode,
    /// Edge handling mode in the vertical direction.
    pub vertical: EdgeMode,
}

impl EdgeMode2D {
    pub const fn new(mode: EdgeMode) -> Self {
        Self {
            horizontal: mode,
            vertical: mode,
        }
    }

    pub const fn anisotropy(horizontal: EdgeMode, vertical: EdgeMode) -> Self {
        Self {
            horizontal,
            vertical,
        }
    }
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Default)]
/// Declares an edge handling mode
pub enum EdgeMode {
    /// If kernel goes out of bounds it will be clipped to an edge and edge pixel replicated across filter
    #[default]
    Clamp = 0,
    /// If kernel goes out of bounds it will be clipped, this is a slightly faster than clamp, however have different visual effects at the edge.
    Wrap = 1,
    /// If filter goes out of bounds image will be replicated with rule `fedcba|abcdefgh|hgfedcb`.
    Reflect = 2,
    /// If filter goes out of bounds image will be replicated with rule `gfedcb|abcdefgh|gfedcba`.
    Reflect101 = 3,
    /// If filter goes out of bounds image will be replicated with provided constant.
    /// Works only for filter APIs.
    Constant = 4,
}

impl From<usize> for EdgeMode {
    fn from(value: usize) -> Self {
        match value {
            0 => EdgeMode::Clamp,
            1 => EdgeMode::Wrap,
            2 => EdgeMode::Reflect,
            3 => EdgeMode::Reflect101,
            4 => EdgeMode::Constant,
            _ => {
                unreachable!("Unknown edge mode for value: {value}");
            }
        }
    }
}

impl EdgeMode {
    pub const fn as_2d(self) -> EdgeMode2D {
        EdgeMode2D::new(self)
    }
}

#[inline(always)]
pub(crate) fn reflect_index(i: isize, n: isize) -> usize {
    (n - i.rem_euclid(n) - 1) as usize
}

#[inline(always)]
#[allow(dead_code)]
pub(crate) fn reflect_index_101(i: isize, n: isize) -> usize {
    let n_r = n - 1;
    if n_r == 0 {
        return 0;
    }
    (n_r - i.rem_euclid(n_r)) as usize
}

#[allow(clippy::int_plus_one)]
macro_rules! clamp_edge {
    ($edge_mode:expr, $value:expr, $min:expr, $max:expr) => {{
        use crate::edge_mode::EdgeMode;
        match $edge_mode {
            EdgeMode::Clamp | EdgeMode::Constant => $value.max($min).min($max - 1) as usize,
            EdgeMode::Wrap => {
                if $value < $min || $value >= $max {
                    $value.rem_euclid($max) as usize
                } else {
                    $value as usize
                }
            }
            EdgeMode::Reflect => {
                if $value < $min || $value >= $max {
                    use crate::edge_mode::reflect_index;
                    let cx = reflect_index($value as isize, $max as isize);
                    cx as usize
                } else {
                    $value as usize
                }
            }
            EdgeMode::Reflect101 => {
                if $value < $min || $value >= $max {
                    use crate::edge_mode::reflect_index_101;
                    reflect_index_101($value as isize, $max as isize)
                } else {
                    $value as usize
                }
            }
        }
    }};
}

#[derive(Clone, Copy)]
pub struct BorderHandle {
    pub edge_mode: EdgeMode,
    pub scalar: Scalar,
}

macro_rules! border_interpolate {
    ($slice: expr, $edge_mode:expr, $value:expr, $min:expr, $max:expr, $scale: expr, $cn: expr) => {{
        use crate::edge_mode::EdgeMode;
        use num_traits::AsPrimitive;
        match $edge_mode.edge_mode {
            EdgeMode::Constant => {
                if $value < $min || $value >= $max {
                    $edge_mode.scalar[$cn].as_()
                } else {
                    *$slice.get_unchecked($value as usize * $scale + $cn)
                }
            }
            EdgeMode::Clamp => {
                *$slice.get_unchecked($value.max($min).min($max - 1) as usize * $scale + $cn)
            }
            EdgeMode::Wrap => {
                if $value < $min || $value >= $max {
                    *$slice.get_unchecked($value.rem_euclid($max) as usize * $scale + $cn)
                } else {
                    *$slice.get_unchecked($value as usize * $scale + $cn)
                }
            }
            EdgeMode::Reflect => {
                if $value < $min || $value >= $max {
                    use crate::edge_mode::reflect_index;
                    let cx = reflect_index($value as isize, $max as isize);
                    *$slice.get_unchecked(cx as usize * $scale + $cn)
                } else {
                    *$slice.get_unchecked($value as usize * $scale + $cn)
                }
            }
            EdgeMode::Reflect101 => {
                if $value < $min || $value >= $max {
                    use crate::edge_mode::reflect_index_101;
                    let cx = reflect_index_101($value as isize, $max as isize);
                    *$slice.get_unchecked(cx as usize * $scale + $cn)
                } else {
                    *$slice.get_unchecked($value as usize * $scale + $cn)
                }
            }
        }
    }};
}

pub(crate) use border_interpolate;
pub(crate) use clamp_edge;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialOrd, PartialEq)]
pub struct Scalar {
    pub v0: f64,
    pub v1: f64,
    pub v2: f64,
    pub v3: f64,
}

impl Scalar {
    pub fn new(v0: f64, v1: f64, v2: f64, v3: f64) -> Self {
        Self { v0, v1, v2, v3 }
    }

    pub fn dup(v: f64) -> Self {
        Scalar::new(v, v, v, v)
    }
}

impl Default for Scalar {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }
}

impl Index<usize> for Scalar {
    type Output = f64;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.v0,
            1 => &self.v1,
            2 => &self.v2,
            3 => &self.v3,
            _ => {
                unimplemented!("Index out of bounds: {}", index);
            }
        }
    }
}
