/*
 * // Copyright (c) Radzivon Bartoshyk 8/2025. All rights reserved.
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

#[cfg(feature = "nightly_f16")]
use core::f16;

pub trait PrimitiveCast<T>: 'static + Copy
where
    T: 'static + Copy,
{
    /// Convert a value to another, using the `as` operator.
    fn cast_(self) -> T;
}

macro_rules! impl_as_primitive {
    (@ $T: ty => $(#[$cfg:meta])* impl $U: ty ) => {
        $(#[$cfg])*
        impl PrimitiveCast<$U> for $T {
            #[inline] fn cast_(self) -> $U { self as $U }
        }
    };
    (@ $T: ty => { $( $U: ty ),* } ) => {$(
        impl_as_primitive!(@ $T => impl $U);
    )*};
    ($T: ty => { $( $U: ty ),* } ) => {
        impl_as_primitive!(@ $T => { $( $U ),* });
        impl_as_primitive!(@ $T => { u8, u16, u32, u64, u128, usize });
        impl_as_primitive!(@ $T => { i8, i16, i32, i64, i128, isize });
    };
}

impl_as_primitive!(u8 => { char, f32, f64 });
impl_as_primitive!(i8 => { f32, f64 });
impl_as_primitive!(u16 => { f32, f64 });
impl_as_primitive!(i16 => { f32, f64 });
impl_as_primitive!(u32 => { f32, f64 });
impl_as_primitive!(i32 => { f32, f64 });
impl_as_primitive!(u64 => { f32, f64 });
#[cfg(not(feature = "nightly_f16"))]
impl_as_primitive!(i64 => { f32, f64 });
#[cfg(feature = "nightly_f16")]
impl_as_primitive!(i64 => { f32, f64, f16 });
impl_as_primitive!(u128 => { f32, f64 });
impl_as_primitive!(i128 => { f32, f64 });
impl_as_primitive!(usize => { f32, f64 });
impl_as_primitive!(isize => { f32, f64 });
#[cfg(feature = "nightly_f16")]
impl_as_primitive!(f32 => { f32, f64, f16 });
#[cfg(not(feature = "nightly_f16"))]
impl_as_primitive!(f32 => { f32, f64 });
#[cfg(feature = "nightly_f16")]
impl_as_primitive!(f16 => { f32, f16, f64 });
#[cfg(feature = "nightly_f16")]
impl_as_primitive!(f64 => { f16, f32, f64 });
#[cfg(not(feature = "nightly_f16"))]
impl_as_primitive!(f64 => { f32, f64 });
impl_as_primitive!(char => { char });
impl_as_primitive!(bool => {});
