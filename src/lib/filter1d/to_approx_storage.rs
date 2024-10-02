/*
 * // Copyright (c) Radzivon Bartoshyk. All rights reserved.
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
use std::ops::Shr;

pub trait ToApproxStorage<T>: 'static + Copy + Shr<T>
where
    T: 'static + Copy,
{
    /// Convert a value to another, using the `to` operator.
    fn to_approx_(self) -> T;
}

macro_rules! impl_to_approx_storage {
    ($from:ty, $to:ty) => {
        impl ToApproxStorage<$to> for $from {
            #[inline]
            fn to_approx_(self) -> $to {
                ((self + (1 << (<$from>::approx_level() - 1))) >> <$from>::approx_level())
                    .max(<$from>::MIN.into())
                    .max(<$to>::MIN.into()) as $to
            }
        }
    };
}

impl_to_approx_storage!(i32, i8);
impl_to_approx_storage!(i64, i8);
impl_to_approx_storage!(i16, i8);
impl_to_approx_storage!(i32, u8);
impl_to_approx_storage!(i64, u8);
impl_to_approx_storage!(u32, u8);
impl_to_approx_storage!(u64, u8);
impl_to_approx_storage!(i16, u8);
impl_to_approx_storage!(u16, u8);
impl_to_approx_storage!(i32, u16);
impl_to_approx_storage!(i64, u16);
impl_to_approx_storage!(u32, u16);
impl_to_approx_storage!(u64, u16);
impl_to_approx_storage!(i32, i16);
impl_to_approx_storage!(i64, i16);

pub trait ApproxLevel {
    fn approx_level() -> i32;
}

impl ApproxLevel for i16 {
    #[inline]
    fn approx_level() -> i32 {
        7
    }
}

impl ApproxLevel for u16 {
    #[inline]
    fn approx_level() -> i32 {
        7
    }
}

impl ApproxLevel for i32 {
    #[inline]
    fn approx_level() -> i32 {
        15
    }
}

impl ApproxLevel for u32 {
    #[inline]
    fn approx_level() -> i32 {
        15
    }
}

impl ApproxLevel for i64 {
    #[inline]
    fn approx_level() -> i32 {
        31
    }
}

impl ApproxLevel for u64 {
    #[inline]
    fn approx_level() -> i32 {
        31
    }
}
