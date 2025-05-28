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
use crate::filter1d::neon::filter_column_complex_u8_f32_fcma::vqpermf;
use crate::img_size::ImageSize;
use crate::to_storage::ToStorage;
use num_complex::Complex;
use std::arch::aarch64::*;

pub(crate) fn filter_column_complex_u16_f32_fcma(
    arena: Arena,
    arena_src: &[&[Complex<f32>]],
    dst: &mut [u16],
    image_size: ImageSize,
    kernel: &[Complex<f32>],
) {
    unsafe {
        filter_column_complex_u8_f32_impl_fc(arena, arena_src, dst, image_size, kernel);
    }
}

#[target_feature(enable = "fcma")]
unsafe fn filter_column_complex_u8_f32_impl_fc(
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

        let c_c = vld1_f32(kernel.get_unchecked(0..).as_ptr().cast());
        let coeff = vcombine_f32(c_c, c_c);

        let v_index = vld1q_u8(
            [
                0, 1, 2, 3, // a0
                8, 9, 10, 11, // a2
                4, 5, 6, 7, // a1
                12, 13, 14, 15, // a3
            ]
            .as_ptr(),
        );

        while cx + 8 < full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(cx..);

            let values0 = vld1q_f32(v_src.as_ptr().cast());
            let values1 = vld1q_f32(v_src.get_unchecked(2..).as_ptr().cast());
            let values2 = vld1q_f32(v_src.get_unchecked(4..).as_ptr().cast());
            let values3 = vld1q_f32(v_src.get_unchecked(6..).as_ptr().cast());

            let mut k0 =
                vcmlaq_rot90_f32(vcmlaq_f32(vdupq_n_f32(0.), values0, coeff), values0, coeff);
            let mut k1 =
                vcmlaq_rot90_f32(vcmlaq_f32(vdupq_n_f32(0.), values1, coeff), values1, coeff);
            let mut k2 =
                vcmlaq_rot90_f32(vcmlaq_f32(vdupq_n_f32(0.), values2, coeff), values2, coeff);
            let mut k3 =
                vcmlaq_rot90_f32(vcmlaq_f32(vdupq_n_f32(0.), values3, coeff), values3, coeff);

            for i in 1..length {
                let c_c = vld1_f32(kernel.get_unchecked(i..).as_ptr().cast());
                let coeff = vcombine_f32(c_c, c_c);

                let values0 = vld1q_f32(
                    arena_src
                        .get_unchecked(i)
                        .get_unchecked(cx..)
                        .as_ptr()
                        .cast(),
                );

                let values1 = vld1q_f32(
                    arena_src
                        .get_unchecked(i)
                        .get_unchecked(cx + 2..)
                        .as_ptr()
                        .cast(),
                );

                let values2 = vld1q_f32(
                    arena_src
                        .get_unchecked(i)
                        .get_unchecked(cx + 4..)
                        .as_ptr()
                        .cast(),
                );

                let values3 = vld1q_f32(
                    arena_src
                        .get_unchecked(i)
                        .get_unchecked(cx + 6..)
                        .as_ptr()
                        .cast(),
                );

                k0 = vcmlaq_rot90_f32(vcmlaq_f32(k0, values0, coeff), values0, coeff);
                k1 = vcmlaq_rot90_f32(vcmlaq_f32(k1, values1, coeff), values1, coeff);
                k2 = vcmlaq_rot90_f32(vcmlaq_f32(k2, values2, coeff), values2, coeff);
                k3 = vcmlaq_rot90_f32(vcmlaq_f32(k3, values3, coeff), values3, coeff);
            }

            let u0 = vqpermf(k0, k1, v_index);

            let real = vcombine_u16(
                vqmovn_u32(vcvtaq_u32_f32(u0)),
                vqmovn_u32(vcvtaq_u32_f32(vqpermf(k2, k3, v_index))),
            );
            vst1q_u16(dst.get_unchecked_mut(cx..).as_mut_ptr().cast(), real);
            cx += 8;
        }

        while cx + 4 < full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(cx..);

            let values0 = vld1q_f32(v_src.as_ptr().cast());
            let values1 = vld1q_f32(v_src.get_unchecked(2..).as_ptr().cast());

            let mut k0 =
                vcmlaq_rot90_f32(vcmlaq_f32(vdupq_n_f32(0.), values0, coeff), values0, coeff);
            let mut k1 =
                vcmlaq_rot90_f32(vcmlaq_f32(vdupq_n_f32(0.), values1, coeff), values1, coeff);

            for i in 1..length {
                let c_c = vld1_f32(kernel.get_unchecked(i..).as_ptr().cast());
                let coeff = vcombine_f32(c_c, c_c);

                let values0 = vld1q_f32(
                    arena_src
                        .get_unchecked(i)
                        .get_unchecked(cx..)
                        .as_ptr()
                        .cast(),
                );

                let values1 = vld1q_f32(
                    arena_src
                        .get_unchecked(i)
                        .get_unchecked(cx + 2..)
                        .as_ptr()
                        .cast(),
                );

                k0 = vcmlaq_rot90_f32(vcmlaq_f32(k0, values0, coeff), values0, coeff);
                k1 = vcmlaq_rot90_f32(vcmlaq_f32(k1, values1, coeff), values1, coeff);
            }

            let real = vqmovn_u32(vcvtaq_u32_f32(vqpermf(k0, k1, v_index)));
            vst1_u16(dst.get_unchecked_mut(cx..).as_mut_ptr().cast(), real);
            cx += 4;
        }

        for x in cx..full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(x..);

            let r0 = vld1_f32(v_src.as_ptr().cast());

            let mut k0 = vcmla_rot90_f32(
                vcmla_f32(vdup_n_f32(0.), r0, vget_low_f32(coeff)),
                r0,
                vget_low_f32(coeff),
            );

            for i in 1..length {
                let c_c = vld1_f32(kernel.get_unchecked(i..).as_ptr().cast());

                let values0 = vld1_f32(
                    arena_src
                        .get_unchecked(i)
                        .get_unchecked(x..)
                        .as_ptr()
                        .cast(),
                );

                k0 = vcmla_rot90_f32(vcmla_f32(k0, values0, c_c), values0, c_c);
            }

            *dst.get_unchecked_mut(x) = vget_lane_f32::<0>(k0).to_();
        }
    }
}
