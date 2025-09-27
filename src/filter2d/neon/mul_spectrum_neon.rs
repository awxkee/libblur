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
use std::arch::aarch64::*;

pub(crate) fn neon_mul_spectrum_in_place_f32(
    value1: &mut [Complex<f32>],
    other: &[Complex<f32>],
    width: usize,
    height: usize,
) {
    unsafe {
        let normalization_factor = (1f64 / (width * height) as f64) as f32;
        let v_norm_factor = vdupq_n_f32(normalization_factor);
        let complex_size = height * width;
        let value1 = &mut value1[..complex_size];
        let other = &other[..complex_size];

        for (dst, kernel) in value1.chunks_exact_mut(8).zip(other.chunks_exact(8)) {
            let vd0 = vld1q_f32(dst.as_ptr().cast());
            let vd1 = vld1q_f32(dst.get_unchecked(2..).as_ptr().cast());
            let vd2 = vld1q_f32(dst.get_unchecked(4..).as_ptr().cast());
            let vd3 = vld1q_f32(dst.get_unchecked(6..).as_ptr().cast());

            let vk0 = vld1q_f32(kernel.as_ptr().cast());
            let vk1 = vld1q_f32(kernel.get_unchecked(2..).as_ptr().cast());
            let vk2 = vld1q_f32(kernel.get_unchecked(4..).as_ptr().cast());
            let vk3 = vld1q_f32(kernel.get_unchecked(6..).as_ptr().cast());

            let p0 = vmulq_f32(mul_complex_f32(vd0, vk0), v_norm_factor);
            let p1 = vmulq_f32(mul_complex_f32(vd1, vk1), v_norm_factor);
            let p2 = vmulq_f32(mul_complex_f32(vd2, vk2), v_norm_factor);
            let p3 = vmulq_f32(mul_complex_f32(vd3, vk3), v_norm_factor);

            vst1q_f32(dst.as_mut_ptr().cast(), p0);
            vst1q_f32(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), p1);
            vst1q_f32(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), p2);
            vst1q_f32(dst.get_unchecked_mut(6..).as_mut_ptr().cast(), p3);
        }

        let dst_rem = value1.chunks_exact_mut(8).into_remainder();
        let src_rem = other.chunks_exact(8).remainder();

        for (dst, kernel) in dst_rem.chunks_exact_mut(2).zip(src_rem.chunks_exact(2)) {
            let v0 = vld1q_f32(dst.as_ptr().cast());
            let v1 = vld1q_f32(kernel.as_ptr().cast());
            let p1 = vmulq_f32(mul_complex_f32(v0, v1), v_norm_factor);
            vst1q_f32(dst.as_mut_ptr().cast(), p1);
        }

        let dst_rem = dst_rem.chunks_exact_mut(2).into_remainder();
        let src_rem = src_rem.chunks_exact(2).remainder();

        for (dst, kernel) in dst_rem.iter_mut().zip(src_rem.iter()) {
            let v0 = vld1_f32(dst as *const Complex<f32> as *const f32);
            let v1 = vld1_f32(kernel as *const Complex<f32> as *const f32);
            let p1 = vmul_f32(mulh_complex_f32(v0, v1), vget_low_f32(v_norm_factor));
            vst1_f32(dst as *mut Complex<f32> as *mut f32, p1);
        }
    }
}

#[inline(always)]
unsafe fn mul_complex_f32(lhs: float32x4_t, rhs: float32x4_t) -> float32x4_t {
    unsafe {
        let temp1 = vtrn1q_f32(rhs, rhs);
        let temp2 = vtrn2q_f32(rhs, vnegq_f32(rhs));
        let temp3 = vmulq_f32(temp2, lhs);
        let temp4 = vrev64q_f32(temp3);
        vfmaq_f32(temp4, temp1, lhs)
    }
}

#[inline(always)]
unsafe fn mulh_complex_f32(lhs: float32x2_t, rhs: float32x2_t) -> float32x2_t {
    unsafe {
        let temp1 = vtrn1_f32(rhs, rhs);
        let temp2 = vtrn2_f32(rhs, vneg_f32(rhs));
        let temp3 = vmul_f32(temp2, lhs);
        let temp4 = vrev64_f32(temp3);
        vfma_f32(temp4, temp1, lhs)
    }
}
