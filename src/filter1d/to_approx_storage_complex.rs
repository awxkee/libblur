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

pub trait ToApproxStorageComplex<T>: 'static + Copy + Shr<T>
where
    T: 'static + Copy,
{
    /// Convert a value to another, using the `to` operator.
    fn to_c_approx_(self) -> T;
}

macro_rules! impl_to_approx_storage {
    ($from:ty, $to:ty) => {
        impl ToApproxStorageComplex<$to> for $from {
            #[inline(always)]
            fn to_c_approx_(self) -> $to {
                ((self + (1 << (<$to>::Q - 1))) >> <$to>::Q)
                    .max(<$from>::MIN.into())
                    .max(<$to>::MIN.into()) as $to
            }
        }
    };
}

impl_to_approx_storage!(i32, i8);
impl_to_approx_storage!(i32, u8);
impl_to_approx_storage!(i32, u16);
impl_to_approx_storage!(i32, i16);

pub trait ApproxComplexLevel {
    const Q: i32;
}

impl ApproxComplexLevel for i8 {
    const Q: i32 = 15;
}

impl ApproxComplexLevel for u8 {
    const Q: i32 = 15;
}

impl ApproxComplexLevel for u16 {
    const Q: i32 = 14;
}

impl ApproxComplexLevel for i16 {
    const Q: i32 = 14;
}
