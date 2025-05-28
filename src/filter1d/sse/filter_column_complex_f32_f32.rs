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
use crate::filter1d::arena::Arena;
use crate::filter1d::sse::filter_column_complex_u8_f32::{mq_complex_mla, mq_complex_mul};
use crate::img_size::ImageSize;
use crate::sse::_shuffle;
use crate::to_storage::ToStorage;
use num_complex::Complex;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn filter_sse_column_complex_f32_f32(
    arena: Arena,
    arena_src: &[&[Complex<f32>]],
    dst: &mut [f32],
    image_size: ImageSize,
    kernel: &[Complex<f32>],
) {
    unsafe {
        filter_sse_column_complex_f32_f32_impl(arena, arena_src, dst, image_size, kernel);
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn filter_sse_column_complex_f32_f32_impl(
    arena: Arena,
    arena_src: &[&[Complex<f32>]],
    dst: &mut [f32],
    image_size: ImageSize,
    kernel: &[Complex<f32>],
) {
    unsafe {
        let full_width = image_size.width * arena.components;

        let length = kernel.len();

        let mut cx = 0usize;

        let cwg = _mm_castsi128_ps(_mm_set1_epi64x(
            (kernel.get_unchecked(0..).as_ptr() as *const i64).read_unaligned(),
        ));
        let c_real = _mm_shuffle_ps::<0b10100000>(cwg, cwg);
        let c_img = _mm_shuffle_ps::<0b11110101>(cwg, cwg);

        while cx + 8 < full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(cx..);

            let values0 = _mm_loadu_ps(v_src.as_ptr().cast());
            let values1 = _mm_loadu_ps(v_src.get_unchecked(2..).as_ptr().cast());
            let values2 = _mm_loadu_ps(v_src.get_unchecked(4..).as_ptr().cast());
            let values3 = _mm_loadu_ps(v_src.get_unchecked(6..).as_ptr().cast());

            let mut k0 = mq_complex_mul(values0, c_real, c_img);
            let mut k1 = mq_complex_mul(values1, c_real, c_img);
            let mut k2 = mq_complex_mul(values2, c_real, c_img);
            let mut k3 = mq_complex_mul(values3, c_real, c_img);

            for i in 1..length {
                let cwg = _mm_castsi128_ps(_mm_set1_epi64x(
                    (kernel.get_unchecked(i..).as_ptr() as *const i64).read_unaligned(),
                ));
                let c_real = _mm_shuffle_ps::<0b10100000>(cwg, cwg);
                let c_img = _mm_shuffle_ps::<0b11110101>(cwg, cwg);

                let v0 = arena_src.get_unchecked(i);

                let values0 = _mm_loadu_ps(v0.get_unchecked(cx..).as_ptr().cast());
                let values1 = _mm_loadu_ps(v0.get_unchecked(cx + 2..).as_ptr().cast());
                let values2 = _mm_loadu_ps(v0.get_unchecked(cx + 4..).as_ptr().cast());
                let values3 = _mm_loadu_ps(v0.get_unchecked(cx + 6..).as_ptr().cast());

                k0 = mq_complex_mla(k0, values0, c_real, c_img);
                k1 = mq_complex_mla(k1, values1, c_real, c_img);
                k2 = mq_complex_mla(k2, values2, c_real, c_img);
                k3 = mq_complex_mla(k3, values3, c_real, c_img);
            }

            let v0 = _mm_shuffle_ps::<{ _shuffle(2, 0, 2, 0) }>(k0, k1);
            let v1 = _mm_shuffle_ps::<{ _shuffle(2, 0, 2, 0) }>(k2, k3);

            _mm_storeu_ps(dst.get_unchecked_mut(cx..).as_mut_ptr().cast(), v0);
            _mm_storeu_ps(dst.get_unchecked_mut(cx + 4..).as_mut_ptr().cast(), v1);
            cx += 8;
        }

        while cx + 4 < full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(cx..);

            let values0 = _mm_loadu_ps(v_src.as_ptr().cast());
            let values1 = _mm_loadu_ps(v_src.get_unchecked(2..).as_ptr().cast());

            let mut k0 = mq_complex_mul(values0, c_real, c_img);
            let mut k1 = mq_complex_mul(values1, c_real, c_img);

            for i in 1..length {
                let cwg = _mm_castsi128_ps(_mm_set1_epi64x(
                    (kernel.get_unchecked(i..).as_ptr() as *const i64).read_unaligned(),
                ));
                let c_real = _mm_shuffle_ps::<0b10100000>(cwg, cwg);
                let c_img = _mm_shuffle_ps::<0b11110101>(cwg, cwg);

                let ss0 = arena_src.get_unchecked(i);

                let values0 = _mm_loadu_ps(ss0.get_unchecked(cx..).as_ptr().cast());
                let values1 = _mm_loadu_ps(ss0.get_unchecked(cx + 2..).as_ptr().cast());

                k0 = mq_complex_mla(k0, values0, c_real, c_img);
                k1 = mq_complex_mla(k1, values1, c_real, c_img);
            }

            let v0 = _mm_shuffle_ps::<{ _shuffle(2, 0, 2, 0) }>(k0, k1);

            _mm_storeu_ps(dst.get_unchecked_mut(cx..).as_mut_ptr().cast(), v0);
            cx += 4;
        }

        for x in cx..full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(x..);

            let cwg = _mm_castsi128_ps(_mm_loadu_si64(v_src.as_ptr().cast()));

            let mut k0 = mq_complex_mul(cwg, c_real, c_img);

            for i in 1..length {
                let cwg = _mm_castsi128_ps(_mm_set1_epi64x(
                    (kernel.get_unchecked(i..).as_ptr() as *const i64).read_unaligned(),
                ));
                let c_real = _mm_shuffle_ps::<0b10100000>(cwg, cwg);
                let c_img = _mm_shuffle_ps::<0b11110101>(cwg, cwg);

                let ss = arena_src.get_unchecked(i).get_unchecked(x..);

                let vals = _mm_castsi128_ps(_mm_loadu_si64(ss.as_ptr().cast()));

                k0 = mq_complex_mla(k0, vals, c_real, c_img);
            }

            *dst.get_unchecked_mut(x) = f32::from_bits(_mm_extract_ps::<0>(k0) as u32).to_();
        }
    }
}
