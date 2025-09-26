/*
 * // Copyright (c) Radzivon Bartoshyk 5/2025. All rights reserved.
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
use rustfft::num_complex::Complex;
use std::arch::x86_64::*;

pub(crate) fn avx_fma_mul_spectrum_in_place_f32(
    value1: &mut [Complex<f32>],
    other: &[Complex<f32>],
    width: usize,
    height: usize,
) {
    unsafe {
        mul_spectrum_in_place_f32_impl(value1, other, width, height);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn avx_deinterleave(a: __m256, b: __m256) -> (__m256, __m256) {
    const SH: i32 = (2 * 4) + 16 + 3 * 64;
    let p0 = _mm256_shuffle_epi32::<SH>(_mm256_castps_si256(a));
    let p1 = _mm256_shuffle_epi32::<SH>(_mm256_castps_si256(b));
    let pl = _mm256_permute2x128_si256::<32>(p0, p1);
    let ph = _mm256_permute2x128_si256::<49>(p0, p1);
    let a0 = _mm256_unpacklo_epi64(pl, ph);
    let b0 = _mm256_unpackhi_epi64(pl, ph);
    (_mm256_castsi256_ps(a0), _mm256_castsi256_ps(b0))
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn avx_interleave(a: __m256, b: __m256) -> (__m256, __m256) {
    let xy_l = _mm256_unpacklo_ps(a, b);
    let xy_h = _mm256_unpackhi_ps(a, b);

    let xy0 = _mm256_permute2f128_ps::<32>(xy_l, xy_h);
    let xy1 = _mm256_permute2f128_ps::<49>(xy_l, xy_h);
    (xy0, xy1)
}

#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn complex_mul_fma(a: __m128, b: __m128) -> __m128 {
    let temp1 = _mm_shuffle_ps::<0xA0>(b, b);
    let temp2 = _mm_shuffle_ps::<0xF5>(b, b);
    let mul2 = _mm_mul_ps(a, temp2);
    let mul2 = _mm_shuffle_ps::<0xB1>(mul2, mul2);
    _mm_fmaddsub_ps(a, temp1, mul2)
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn mul_spectrum_in_place_f32_impl(
    value1: &mut [Complex<f32>],
    other: &[Complex<f32>],
    width: usize,
    height: usize,
) {
    unsafe {
        let normalization_factor = (1f64 / (width * height) as f64) as f32;
        let v_norm_factor = _mm256_set1_ps(normalization_factor);
        let complex_size = height * width;
        let value1 = &mut value1[..complex_size];
        let other = &other[..complex_size];

        for (dst, kernel) in value1.chunks_exact_mut(16).zip(other.chunks_exact(16)) {
            let vd0 = _mm256_loadu_ps(dst.as_ptr().cast());
            let vd1 = _mm256_loadu_ps(dst.get_unchecked(4..).as_ptr().cast());
            let vd2 = _mm256_loadu_ps(dst.get_unchecked(8..).as_ptr().cast());
            let vd3 = _mm256_loadu_ps(dst.get_unchecked(12..).as_ptr().cast());

            let vk0 = _mm256_loadu_ps(kernel.as_ptr().cast());
            let vk1 = _mm256_loadu_ps(kernel.get_unchecked(4..).as_ptr().cast());
            let vk2 = _mm256_loadu_ps(kernel.get_unchecked(8..).as_ptr().cast());
            let vk3 = _mm256_loadu_ps(kernel.get_unchecked(12..).as_ptr().cast());

            let (ar0, ai0) = avx_deinterleave(vd0, vd1);
            let (ar1, ai1) = avx_deinterleave(vd2, vd3);

            let (br0, bi0) = avx_deinterleave(vk0, vk1);
            let (br1, bi1) = avx_deinterleave(vk2, vk3);

            let mut prod_r0 = _mm256_mul_ps(ar0, br0);
            let mut prod_i0 = _mm256_mul_ps(ar0, bi0);
            prod_r0 = _mm256_fnmadd_ps(ai0, bi0, prod_r0);
            prod_i0 = _mm256_fmadd_ps(ai0, br0, prod_i0);

            let mut prod_r1 = _mm256_mul_ps(ar1, br1);
            let mut prod_i1 = _mm256_mul_ps(ar1, bi1);
            prod_r1 = _mm256_fnmadd_ps(ai1, bi1, prod_r1);
            prod_i1 = _mm256_fmadd_ps(ai1, br1, prod_i1);

            prod_r0 = _mm256_mul_ps(prod_r0, v_norm_factor);
            prod_i0 = _mm256_mul_ps(prod_i0, v_norm_factor);
            prod_r1 = _mm256_mul_ps(prod_r1, v_norm_factor);
            prod_i1 = _mm256_mul_ps(prod_i1, v_norm_factor);

            let (d0, d1) = avx_interleave(prod_r0, prod_i0);
            let (d2, d3) = avx_interleave(prod_r1, prod_i1);

            _mm256_storeu_ps(dst.as_mut_ptr().cast(), d0);
            _mm256_storeu_ps(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), d1);
            _mm256_storeu_ps(dst.get_unchecked_mut(8..).as_mut_ptr().cast(), d2);
            _mm256_storeu_ps(dst.get_unchecked_mut(12..).as_mut_ptr().cast(), d3);
        }

        let dst_rem = value1.chunks_exact_mut(16).into_remainder();
        let src_rem = other.chunks_exact(16).remainder();

        for (dst, kernel) in dst_rem.chunks_exact_mut(4).zip(src_rem.chunks_exact(4)) {
            let a0 = _mm256_loadu_ps(dst.as_ptr().cast());
            let b0 = _mm256_loadu_ps(kernel.as_ptr().cast());

            let (ar0, ai0) = avx_deinterleave(a0, _mm256_setzero_ps());
            let (br0, bi0) = avx_deinterleave(b0, _mm256_setzero_ps());

            let mut prod_r0 = _mm256_mul_ps(ar0, br0);
            let mut prod_i0 = _mm256_mul_ps(ar0, bi0);
            prod_r0 = _mm256_fnmadd_ps(ai0, bi0, prod_r0);
            prod_i0 = _mm256_fmadd_ps(ai0, br0, prod_i0);

            prod_r0 = _mm256_mul_ps(prod_r0, v_norm_factor);
            prod_i0 = _mm256_mul_ps(prod_i0, v_norm_factor);

            let (d0, _) = avx_interleave(prod_r0, prod_i0);

            _mm256_storeu_ps(dst.as_mut_ptr().cast(), d0);
        }

        let dst_rem = dst_rem.chunks_exact_mut(4).into_remainder();
        let src_rem = src_rem.chunks_exact(4).remainder();

        for (dst, kernel) in dst_rem.iter_mut().zip(src_rem.iter()) {
            let v0 = _mm_loadu_si64(dst as *const Complex<f32> as *const _);
            let v1 = _mm_loadu_si64(kernel as *const Complex<f32> as *const _);

            let lo = complex_mul_fma(_mm_castsi128_ps(v0), _mm_castsi128_ps(v1));

            _mm_storeu_si64(dst as *mut Complex<f32> as *mut _, _mm_castps_si128(lo));
        }
    }
}
