/*
 * // Copyright (c) Radzivon Bartoshyk 11/2025. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#![allow(unused)]
use num_traits::Euclid;
use std::ops::Div;

#[inline(always)]
fn mulhi_i64(x: i64, y: i64) -> i64 {
    let xl = x as i128;
    let yl = y as i128;
    ((xl * yl) >> 64) as i64
}

#[inline(always)]
fn mulhi_i32(x: i32, y: i32) -> i32 {
    let xl = x as i64;
    let yl = y as i64;
    ((xl * yl) >> 32) as i32
}

#[inline(always)]
fn mulhi_isize(x: isize, y: isize) -> isize {
    #[cfg(target_pointer_width = "64")]
    {
        let xl = x as i128;
        let yl = y as i128;
        ((xl * yl) >> 64) as isize
    }
    #[cfg(target_pointer_width = "32")]
    {
        let xl = x as i64;
        let yl = y as i64;
        ((xl * yl) >> 32) as isize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DividerI64 {
    magic: i64,
    more: u8,
    divisor: i64,
    abs_divisor: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DividerI32 {
    magic: i32,
    more: u8,
    divisor: i32,
    abs_divisor: i32,
}

#[derive(Copy, Clone)]
pub(crate) enum DividerIsize {
    #[cfg(target_pointer_width = "32")]
    I32(DividerI32),
    #[cfg(target_pointer_width = "64")]
    I64(DividerI64),
}

impl DividerI32 {
    pub(crate) fn new(divisor: i32) -> DividerI32 {
        let ud: u32 = divisor as u32;
        let ud: u32 = if divisor < 0 {
            divisor.wrapping_neg() as u32
        } else {
            ud
        };
        let floor_log_2_d = 31 - ud.leading_zeros();
        if (ud & (ud - 1)) == 0 {
            // Branchfree and non-branchfree cases are the same
            return DividerI32 {
                magic: 0,
                more: floor_log_2_d as u8,
                divisor,
                abs_divisor: divisor.abs(),
            };
        }
        // the dividend here is 2**(floor_log_2_d + 63), so the low 64 bit word
        // is 0 and the high word is floor_log_2_d - 1
        let num = 1u64 << (floor_log_2_d - 1);
        let u_div = ud as u64;
        let v = ((num) << 32).div_rem_euclid(&u_div);
        let (mut proposed_m, rem) = (v.0 as u32, v.1 as u32);
        let e = ud - rem;

        // We are going to start with a power of floor_log_2_d - 1.
        // This works if e < 2**floor_log_2_d.
        let mut more: u8 = if e < (1u32 << floor_log_2_d) {
            // This power works
            (floor_log_2_d - 1) as u8
        } else {
            // We need to go one higher. This should not make proposed_m
            // overflow, but it will make it negative when interpreted as an
            // int32_t.
            proposed_m += proposed_m;
            let twice_rem = rem + rem;
            if twice_rem >= ud || twice_rem < rem {
                proposed_m += 1;
            }
            (floor_log_2_d | 0x40) as u8
        };
        proposed_m += 1;
        let mut magic: i32 = proposed_m as i32;

        // Mark if we are negative
        if divisor < 0 {
            more |= 0x80;
            magic = -magic;
        }
        DividerI32 {
            magic,
            more,
            divisor,
            abs_divisor: divisor.abs(),
        }
    }
}

impl DividerI64 {
    pub(crate) fn new(divisor: i64) -> DividerI64 {
        let ud: u64 = divisor as u64;
        let ud: u64 = if divisor < 0 {
            divisor.wrapping_neg() as u64
        } else {
            ud
        };
        let floor_log_2_d = 63 - ud.leading_zeros();
        if (ud & (ud - 1)) == 0 {
            // Branchfree and non-branchfree cases are the same
            return DividerI64 {
                magic: 0,
                more: floor_log_2_d as u8,
                divisor,
                abs_divisor: divisor.abs(),
            };
        }
        // the dividend here is 2**(floor_log_2_d + 63), so the low 64 bit word
        // is 0 and the high word is floor_log_2_d - 1
        let num = 1u64 << (floor_log_2_d - 1);
        let u_div = ud as u128;
        let v = ((num as u128) << 64).div_rem_euclid(&u_div);
        let (mut proposed_m, rem) = (v.0 as u64, v.1 as u64);
        let e = ud - rem;

        // We are going to start with a power of floor_log_2_d - 1.
        // This works if e < 2**floor_log_2_d.
        let mut more: u8 = if e < (1u64 << floor_log_2_d) {
            // This power works
            (floor_log_2_d - 1) as u8
        } else {
            // We need to go one higher. This should not make proposed_m
            // overflow, but it will make it negative when interpreted as an
            // int32_t.
            proposed_m += proposed_m;
            let twice_rem = rem + rem;
            if twice_rem >= ud || twice_rem < rem {
                proposed_m += 1;
            }
            (floor_log_2_d | 0x40) as u8
        };
        proposed_m += 1;
        let mut magic: i64 = proposed_m as i64;

        // Mark if we are negative
        if divisor < 0 {
            more |= 0x80;
            magic = -magic;
        }
        DividerI64 {
            magic,
            more,
            divisor,
            abs_divisor: divisor.abs(),
        }
    }
}

impl Div<DividerI64> for i64 {
    type Output = i64;

    #[inline(always)]
    fn div(self, denom: DividerI64) -> Self::Output {
        let shift = denom.more & 0x3F;

        if denom.magic == 0 {
            // shift path
            let mask = (1u64 << shift) - 1;
            let uq: u64 = (self as u64).wrapping_add((self >> 63) as u64 & mask);
            let mut q = uq as i64;
            q >>= shift;
            // must be arithmetic shift and then sign-extend
            let sign: i64 = (denom.more >> 7) as i64;
            q = (q ^ sign) - sign;
            q
        } else {
            let mut uq = mulhi_i64(self, denom.magic);
            if (denom.more & 0x40) != 0 {
                // must be arithmetic shift and then sign extend
                let sign: i64 = (denom.more >> 7) as i64;
                // q += (more < 0 ? -numer : numer)
                uq = uq.wrapping_add(((self as u64) ^ sign as u64) as i64 - sign);
            }
            let mut q = uq;
            q >>= shift;
            q += (q < 0) as i64;
            q
        }
    }
}

impl Div<DividerI32> for i32 {
    type Output = i32;

    #[inline(always)]
    fn div(self, denom: DividerI32) -> Self::Output {
        let shift = denom.more & 0x3F;

        if denom.magic == 0 {
            // shift path
            let mask = (1u32 << shift) - 1;
            let uq: u32 = (self as u32).wrapping_add((self >> 31) as u32 & mask);
            let mut q = uq as i32;
            q >>= shift;
            // must be arithmetic shift and then sign-extend
            let sign: i32 = (denom.more >> 7) as i32;
            q = (q ^ sign) - sign;
            q
        } else {
            let mut uq = mulhi_i32(self, denom.magic);
            if (denom.more & 0x40) != 0 {
                // must be arithmetic shift and then sign extend
                let sign: i32 = (denom.more >> 7) as i32;
                // q += (more < 0 ? -numer : numer)
                uq = uq.wrapping_add(((self as u32) ^ sign as u32) as i32 - sign);
            }
            let mut q = uq;
            q >>= shift;
            q += (q < 0) as i32;
            q
        }
    }
}

pub(crate) trait RemEuclidFast<Divider> {
    fn rem_euclid_fast(self, divider: &Divider) -> Self;
}

macro_rules! impl_rem_euclid_fast {
    ($div_typ: ident, $int_typ: ident) => {
        impl RemEuclidFast<$div_typ> for $int_typ {
            #[inline]
            fn rem_euclid_fast(self, divider: &$div_typ) -> $int_typ {
                let q = self / *divider;
                let mut r = self - q * divider.divisor;
                if r < 0 {
                    r = r.wrapping_add(divider.abs_divisor);
                }
                r
            }
        }
    };
}

impl DividerIsize {
    #[inline(always)]
    pub(crate) fn new(divisor: isize) -> Self {
        #[cfg(target_pointer_width = "32")]
        {
            Self::I32(DividerI32::new(divisor as i32))
        }

        #[cfg(target_pointer_width = "64")]
        {
            Self::I64(DividerI64::new(divisor as i64))
        }
    }

    #[inline(always)]
    pub(crate) fn divisor(&self) -> isize {
        match self {
            #[cfg(target_pointer_width = "32")]
            Self::I32(d) => d.divisor as isize,
            #[cfg(target_pointer_width = "64")]
            Self::I64(d) => d.divisor as isize,
        }
    }

    #[inline(always)]
    pub(crate) fn abs_divisor(&self) -> isize {
        match self {
            #[cfg(target_pointer_width = "32")]
            Self::I32(d) => d.abs_divisor as isize,
            #[cfg(target_pointer_width = "64")]
            Self::I64(d) => d.abs_divisor as isize,
        }
    }
}

impl Div<DividerIsize> for isize {
    type Output = isize;

    #[inline(always)]
    fn div(self, denom: DividerIsize) -> Self::Output {
        match denom {
            #[cfg(target_pointer_width = "32")]
            DividerIsize::I32(d) => (self as i32 / d) as isize,
            #[cfg(target_pointer_width = "64")]
            DividerIsize::I64(d) => (self as i64 / d) as isize,
        }
    }
}

impl_rem_euclid_fast!(DividerI64, i64);
impl_rem_euclid_fast!(DividerI32, i32);

impl RemEuclidFast<DividerIsize> for isize {
    #[inline]
    fn rem_euclid_fast(self, divider: &DividerIsize) -> isize {
        let q = self / *divider;
        let mut r = self - q * divider.divisor();
        if r < 0 {
            r = r.wrapping_add(divider.abs_divisor());
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rem_euclid_fast_i32_basic() {
        let divisors = [1, 2, 3, 7, 15, 150, i32::MAX, i32::MIN + 1];
        let test_values = [
            -500,
            -150,
            -15,
            -1,
            0,
            1,
            15,
            150,
            500,
            i32::MAX,
            i32::MIN + 1,
        ];

        for &d in &divisors {
            let divider = DividerI32::new(d);
            for &x in &test_values {
                assert_eq!(
                    x.rem_euclid_fast(&divider),
                    x.rem_euclid(d),
                    "Failed i32: x = {}, divisor = {}",
                    x,
                    d
                );
            }
        }
    }

    #[test]
    fn test_rem_euclid_fast_i64_basic() {
        let divisors = [1, 2, 3, 7, 15, 150, i64::MAX, i64::MIN + 1];
        let test_values = [
            -500,
            -150,
            -15,
            -1,
            0,
            1,
            15,
            150,
            500,
            i64::MAX,
            i64::MIN + 1,
        ];

        for &d in &divisors {
            let divider = DividerI64::new(d);
            for &x in &test_values {
                assert_eq!(
                    x.rem_euclid_fast(&divider),
                    x.rem_euclid(d),
                    "Failed i64: x = {}, divisor = {}",
                    x,
                    d
                );
            }
        }
    }

    #[test]
    fn test_rem_euclid_fast_isize_basic() {
        let divisors = [
            1isize,
            2,
            3,
            7,
            15,
            150,
            isize::MAX,
            isize::MIN + 1,
            -150,
            -150,
        ];
        let test_values = [
            -500isize,
            -150,
            -15,
            -1,
            0,
            1,
            15,
            150,
            500,
            isize::MAX,
            isize::MIN + 1,
            75,
            -250,
        ];

        for &d in &divisors {
            let divider = DividerIsize::new(d);
            for &x in &test_values {
                assert_eq!(
                    x.rem_euclid_fast(&divider),
                    x.rem_euclid(d),
                    "Failed isize: x = {}, divisor = {}",
                    x,
                    d
                );
            }
        }
    }
}
