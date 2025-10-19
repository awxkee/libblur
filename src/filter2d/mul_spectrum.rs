/*
 * // Copyright (c) Radzivon Bartoshyk 2/2025. All rights reserved.
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
use num_traits::AsPrimitive;
use rustfft::num_complex::Complex;
use rustfft::FftNum;
use std::ops::Mul;

pub(crate) fn mul_spectrum_in_place<V: FftNum + Mul<V> + SpectrumMultiplier<V>>(
    value1: &mut [Complex<V>],
    other: &[Complex<V>],
    width: usize,
    height: usize,
    scale: V,
) where
    f64: AsPrimitive<V>,
{
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "avx"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                return mul_spectrum_in_place_avx2(value1, other, width, height, scale);
            }
        }
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            unsafe {
                return mul_spectrum_in_place_sse_4_1(value1, other, width, height, scale);
            }
        }
    }
    V::mul_spectrum(value1, other, width, height, scale);
}

pub trait SpectrumMultiplier<V> {
    fn mul_spectrum(
        value1: &mut [Complex<V>],
        other: &[Complex<V>],
        width: usize,
        height: usize,
        scale: V,
    );
}

impl SpectrumMultiplier<f32> for f32 {
    fn mul_spectrum(
        value1: &mut [Complex<f32>],
        other: &[Complex<f32>],
        width: usize,
        height: usize,
        scale: f32,
    ) {
        #[cfg(all(target_arch = "aarch64", feature = "nightly_fcma"))]
        {
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::filter2d::neon::fcma_mul_spectrum_in_place_f32;
                return fcma_mul_spectrum_in_place_f32(value1, other, width, height);
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                if std::arch::is_x86_feature_detected!("fma") {
                    use crate::filter2d::avx::avx_fma_mul_spectrum_in_place_f32;
                    return avx_fma_mul_spectrum_in_place_f32(value1, other, width, height, scale);
                }
                unsafe {
                    return mul_spectrum_in_place_avx2(value1, other, width, height, scale);
                }
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                unsafe {
                    return mul_spectrum_in_place_sse_4_1(value1, other, width, height, scale);
                }
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::filter2d::neon::neon_mul_spectrum_in_place_f32;
            neon_mul_spectrum_in_place_f32(value1, other, width, height, scale);
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            mul_spectrum_in_place_impl(value1, other, width, height, scale);
        }
    }
}

impl SpectrumMultiplier<f64> for f64 {
    fn mul_spectrum(
        value1: &mut [Complex<f64>],
        other: &[Complex<f64>],
        width: usize,
        height: usize,
        scale: f64,
    ) {
        mul_spectrum_in_place(value1, other, width, height, scale);
    }
}

#[allow(dead_code)]
#[inline(always)]
fn mul_spectrum_in_place_impl<V: FftNum + Mul<V>>(
    value1: &mut [Complex<V>],
    other: &[Complex<V>],
    width: usize,
    height: usize,
    scale: V,
) where
    f64: AsPrimitive<V>,
{
    let complex_size = height * width;
    for (dst, kernel) in value1
        .iter_mut()
        .take(complex_size)
        .zip(other.iter().take(complex_size))
    {
        *dst = (*dst) * (*kernel) * scale;
    }
}

#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
#[target_feature(enable = "sse4.1")]
unsafe fn mul_spectrum_in_place_sse_4_1<V: FftNum + Mul<V>>(
    value1: &mut [Complex<V>],
    other: &[Complex<V>],
    width: usize,
    height: usize,
    scale: V,
) where
    f64: AsPrimitive<V>,
{
    mul_spectrum_in_place_impl(value1, other, width, height, scale)
}

#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "avx"))]
#[target_feature(enable = "avx2")]
unsafe fn mul_spectrum_in_place_avx2<V: FftNum + Mul<V>>(
    value1: &mut [Complex<V>],
    other: &[Complex<V>],
    width: usize,
    height: usize,
    scale: V,
) where
    f64: AsPrimitive<V>,
{
    mul_spectrum_in_place_impl(value1, other, width, height, scale)
}
