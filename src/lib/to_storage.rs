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

use half::f16;

/// Helper trait to convert and round if we are storing in integral type
pub trait ToStorage<T>: 'static + Copy
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
                self.round().max(0 as $from).min(<$to>::MAX as $from) as $to
            }
        }
    };
}

impl_to_integral_storage!(f32, i8);
impl_to_integral_storage!(f64, i8);
impl_to_integral_storage!(f32, u8);
impl_to_integral_storage!(f64, u8);
impl_to_integral_storage!(f32, u16);
impl_to_integral_storage!(f64, u16);
impl_to_integral_storage!(f32, i16);
impl_to_integral_storage!(f64, i16);
impl_to_integral_storage!(f32, u32);
impl_to_integral_storage!(f64, u32);
impl_to_integral_storage!(f32, i32);
impl_to_integral_storage!(f64, i32);
impl_to_integral_storage!(f32, usize);
impl_to_integral_storage!(f64, usize);

macro_rules! impl_to_direct_storage {
    ($from:ty, $to:ty) => {
        impl ToStorage<$to> for $from {
            fn to_(self) -> $to {
                self as $to
            }
        }
    };
}

impl_to_direct_storage!(f32, f32);
impl_to_direct_storage!(f64, f64);
impl_to_direct_storage!(f64, f32);
impl_to_direct_storage!(f32, f64);

macro_rules! impl_signed_to_unsigned_storage {
    ($from:ty, $to:ty) => {
        impl ToStorage<$to> for $from {
            fn to_(self) -> $to {
                self.max(0).min(<$to>::MAX as $from) as $to
            }
        }
    };
}

impl_signed_to_unsigned_storage!(i16, u8);
impl_signed_to_unsigned_storage!(i32, u8);
impl_signed_to_unsigned_storage!(i64, u8);
impl_to_direct_storage!(u16, u8);
impl_to_direct_storage!(u32, u8);
impl_to_direct_storage!(u64, u8);

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
