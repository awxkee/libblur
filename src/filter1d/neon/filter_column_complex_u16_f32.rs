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
use crate::filter1d::neon::filter_column_complex_u8_f32::{
    v_complex_mla, v_complex_mul, vq_complex_mla, vq_complex_mul,
};
use crate::img_size::ImageSize;
use crate::to_storage::ToStorage;
use num_complex::Complex;
use std::arch::aarch64::*;

pub(crate) fn filter_column_complex_u16_f32(
    arena: Arena,
    arena_src: &[&[Complex<f32>]],
    dst: &mut [u16],
    image_size: ImageSize,
    kernel: &[Complex<f32>],
) {
    unsafe {
        let full_width = image_size.width * arena.components;

        let length = kernel.len();

        let mut cx = 0usize;

        let coeff = *kernel.get_unchecked(0);
        let c_re = vdupq_n_f32(coeff.re);
        let c_im = vdupq_n_f32(coeff.im);

        while cx + 8 < full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(cx..);

            let values0 = vld2q_f32(v_src.as_ptr().cast());
            let values1 = vld2q_f32(v_src.get_unchecked(4..).as_ptr().cast());

            let mut k0 = vq_complex_mul(values0.0, values0.1, c_re, c_im);
            let mut k1 = vq_complex_mul(values1.0, values1.1, c_re, c_im);

            for i in 1..length {
                let coeff = *kernel.get_unchecked(i);

                let c_re = vdupq_n_f32(coeff.re);
                let c_im = vdupq_n_f32(coeff.im);

                let values0 = vld2q_f32(
                    arena_src
                        .get_unchecked(i)
                        .get_unchecked(cx..)
                        .as_ptr()
                        .cast(),
                );

                let values1 = vld2q_f32(
                    arena_src
                        .get_unchecked(i)
                        .get_unchecked(cx + 4..)
                        .as_ptr()
                        .cast(),
                );

                k0 = vq_complex_mla(k0, values0.0, values0.1, c_re, c_im);
                k1 = vq_complex_mla(k1, values1.0, values1.1, c_re, c_im);
            }

            let real = vcombine_u16(
                vqmovn_u32(vcvtaq_u32_f32(k0.0)),
                vqmovn_u32(vcvtaq_u32_f32(k1.0)),
            );
            vst1q_u16(dst.get_unchecked_mut(cx..).as_mut_ptr().cast(), real);
            cx += 8;
        }

        while cx + 4 < full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(cx..);

            let values = vld2q_f32(v_src.as_ptr().cast());

            let mut k0 = vq_complex_mul(values.0, values.1, c_re, c_im);

            for i in 1..length {
                let coeff = *kernel.get_unchecked(i);

                let c_re = vdupq_n_f32(coeff.re);
                let c_im = vdupq_n_f32(coeff.im);

                let values = vld2q_f32(
                    arena_src
                        .get_unchecked(i)
                        .get_unchecked(cx..)
                        .as_ptr()
                        .cast(),
                );

                k0 = vq_complex_mla(k0, values.0, values.1, c_re, c_im);
            }

            let real = vqmovn_u32(vcvtaq_u32_f32(k0.0));
            vst1_u16(dst.get_unchecked_mut(cx..).as_mut_ptr().cast(), real);
            cx += 4;
        }

        for x in cx..full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(x..);

            let r0 = vdup_n_f32(v_src[0].re);
            let i0 = vdup_n_f32(v_src[0].im);

            let mut k0 = v_complex_mul(r0, i0, vget_low_f32(c_re), vget_low_f32(c_im));

            for i in 1..length {
                let coeff = *kernel.get_unchecked(i);
                let c_re = vdup_n_f32(coeff.re);
                let c_im = vdup_n_f32(coeff.im);

                let ss = arena_src.get_unchecked(i).get_unchecked(x);

                let r0 = vdup_n_f32(ss.re);
                let i0 = vdup_n_f32(ss.im);

                k0 = v_complex_mla(k0, r0, i0, c_re, c_im);
            }

            *dst.get_unchecked_mut(x) = vget_lane_f32::<0>(k0.0).to_();
        }
    }
}
