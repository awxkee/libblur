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
/// Declares an edge handling mode
pub enum EdgeMode {
    /// If kernel goes out of bounds it will be clipped to an edge and edge pixel replicated across filter
    #[default]
    Clamp = 0,
    /// If kernel goes out of bounds it will be clipped, this is a slightly faster than clamp, however have different visual effects at the edge.
    /// *Kernel clip is supported only for clear gaussian blur and not supported in any approximations!*
    Wrap = 1,
    /// If filter goes out of bounds image will be replicated with rule `fedcba|abcdefgh|hgfedcb`
    Reflect = 2,
    /// If filter goes out of bounds image will be replicated with rule `gfedcb|abcdefgh|gfedcba`
    Reflect101 = 3,
    /// If filter goes out of bounds image will be replicated with provided constant
    /// Works only for clear filter, otherwise ignored
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
                panic!("Unknown edge mode for value: {}", value);
            }
        }
    }
}

#[inline]
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

pub(crate) use reflect_101;

macro_rules! clamp_edge {
    ($edge_mode:expr, $value:expr, $min:expr, $max:expr) => {{
        match $edge_mode {
            EdgeMode::Clamp | EdgeMode::Constant => {
                (std::cmp::min(std::cmp::max($value, $min), $max - 1) as u32) as usize
            }
            EdgeMode::Wrap => {
                if $value < $min || $value >= $max {
                    $value.rem_euclid($max) as usize
                } else {
                    $value as usize
                }
            }
            EdgeMode::Reflect => {
                let cx = reflect_index($value as isize, $max as isize);
                cx as usize
            }
            EdgeMode::Reflect101 => {
                let cx = reflect_101!($value, $max);
                cx as usize
            }
        }
    }};
}

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
