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
use crate::img_size::ImageSize;
use num_complex::Complex;
use std::arch::aarch64::*;

pub(crate) fn filter_row_complex_u8_i32_q(
    arena: Arena,
    arena_src: &[u8],
    dst: &mut [Complex<i16>],
    image_size: ImageSize,
    kernel: &[Complex<i16>],
) {
    unsafe {
        let width = image_size.width;

        let src = arena_src;
        let local_src = src;

        let length = kernel.len();

        let mut cx = 0usize;

        let max_width = width * arena.components;

        let coeff = vreinterpret_s16_s32(vld1_lane_s32::<0>(
            kernel.get_unchecked(0..).as_ptr().cast(),
            vdup_n_s32(0),
        ));

        while cx + 16 < max_width {
            let shifted_src = local_src.get_unchecked(cx..);

            let values = vld1q_u8(shifted_src.as_ptr().cast());

            let lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(values)));
            let hi = vreinterpretq_s16_u16(vmovl_high_u8(values));

            let mut r0 = vmull_lane_s16::<0>(vget_low_s16(lo), coeff);
            let mut r1 = vmull_high_lane_s16::<0>(lo, coeff);
            let mut r2 = vmull_lane_s16::<0>(vget_low_s16(hi), coeff);
            let mut r3 = vmull_high_lane_s16::<0>(hi, coeff);

            let mut i0 = vmull_lane_s16::<1>(vget_low_s16(lo), coeff);
            let mut i1 = vmull_high_lane_s16::<1>(lo, coeff);
            let mut i2 = vmull_lane_s16::<1>(vget_low_s16(hi), coeff);
            let mut i3 = vmull_high_lane_s16::<1>(hi, coeff);

            for i in 1..length {
                let coeff = vreinterpret_s16_s32(vld1_lane_s32::<0>(
                    kernel.get_unchecked(i..).as_ptr().cast(),
                    vdup_n_s32(0),
                ));

                let values = vld1q_u8(
                    shifted_src
                        .get_unchecked(i * arena.components..)
                        .as_ptr()
                        .cast(),
                );

                let lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(values)));
                let hi = vreinterpretq_s16_u16(vmovl_high_u8(values));

                r0 = vmlal_lane_s16::<0>(r0, vget_low_s16(lo), coeff);
                r1 = vmlal_high_lane_s16::<0>(r1, lo, coeff);
                r2 = vmlal_lane_s16::<0>(r2, vget_low_s16(hi), coeff);
                r3 = vmlal_high_lane_s16::<0>(r3, hi, coeff);

                i0 = vmlal_lane_s16::<1>(i0, vget_low_s16(lo), coeff);
                i1 = vmlal_high_lane_s16::<1>(i1, lo, coeff);
                i2 = vmlal_lane_s16::<1>(i2, vget_low_s16(hi), coeff);
                i3 = vmlal_high_lane_s16::<1>(i3, hi, coeff);
            }

            let r0 = vqrshrn_n_s32::<15>(r0);
            let r1 = vqrshrn_n_s32::<15>(r1);
            let r2 = vqrshrn_n_s32::<15>(r2);
            let r3 = vqrshrn_n_s32::<15>(r3);

            let i0 = vqrshrn_n_s32::<15>(i0);
            let i1 = vqrshrn_n_s32::<15>(i1);
            let i2 = vqrshrn_n_s32::<15>(i2);
            let i3 = vqrshrn_n_s32::<15>(i3);

            let dst0 = dst.get_unchecked_mut(cx..);
            vst2q_s16(
                dst0.as_mut_ptr().cast(),
                int16x8x2_t(vcombine_s16(r0, r1), vcombine_s16(i0, i1)),
            );
            vst2q_s16(
                dst0.get_unchecked_mut(8..).as_mut_ptr().cast(),
                int16x8x2_t(vcombine_s16(r2, r3), vcombine_s16(i2, i3)),
            );
            cx += 16;
        }

        while cx + 8 < max_width {
            let shifted_src = local_src.get_unchecked(cx..);

            let values = vld1_u8(shifted_src.as_ptr().cast());

            let lo = vreinterpretq_s16_u16(vmovl_u8(values));

            let mut r0 = vmull_lane_s16::<0>(vget_low_s16(lo), coeff);
            let mut r1 = vmull_high_lane_s16::<0>(lo, coeff);

            let mut i0 = vmull_lane_s16::<1>(vget_low_s16(lo), coeff);
            let mut i1 = vmull_high_lane_s16::<1>(lo, coeff);

            for i in 1..length {
                let coeff = vreinterpret_s16_s32(vld1_lane_s32::<0>(
                    kernel.get_unchecked(i..).as_ptr().cast(),
                    vdup_n_s32(0),
                ));

                let values = vld1_u8(
                    shifted_src
                        .get_unchecked(i * arena.components..)
                        .as_ptr()
                        .cast(),
                );

                let lo = vreinterpretq_s16_u16(vmovl_u8(values));

                r0 = vmlal_lane_s16::<0>(r0, vget_low_s16(lo), coeff);
                r1 = vmlal_high_lane_s16::<0>(r1, lo, coeff);

                i0 = vmlal_lane_s16::<1>(i0, vget_low_s16(lo), coeff);
                i1 = vmlal_high_lane_s16::<1>(i1, lo, coeff);
            }

            let r0 = vqrshrn_n_s32::<15>(r0);
            let r1 = vqrshrn_n_s32::<15>(r1);
            let i0 = vqrshrn_n_s32::<15>(i0);
            let i1 = vqrshrn_n_s32::<15>(i1);

            let dst0 = dst.get_unchecked_mut(cx..);
            vst2q_s16(
                dst0.as_mut_ptr().cast(),
                int16x8x2_t(vcombine_s16(r0, r1), vcombine_s16(i0, i1)),
            );
            cx += 8;
        }

        for x in cx..max_width {
            let shifted_src = local_src.get_unchecked(x..);
            let mut r0 =
                vmull_lane_s16::<0>(vdup_n_s16(*shifted_src.get_unchecked(0) as i16), coeff);
            let mut i0 =
                vmull_lane_s16::<1>(vdup_n_s16(*shifted_src.get_unchecked(0) as i16), coeff);

            for i in 1..length {
                let coeff = vreinterpret_s16_s32(vld1_lane_s32::<0>(
                    kernel.get_unchecked(i..).as_ptr().cast(),
                    vdup_n_s32(0),
                ));

                let a0 = vdup_n_s16(*shifted_src.get_unchecked(i * arena.components) as i16);

                r0 = vmlal_lane_s16::<0>(r0, a0, coeff);
                i0 = vmlal_lane_s16::<1>(i0, a0, coeff);
            }

            let r0 = vqrshrn_n_s32::<15>(r0);
            let i0 = vqrshrn_n_s32::<15>(i0);

            *dst.get_unchecked_mut(x) = Complex {
                re: vget_lane_s16::<0>(r0),
                im: vget_lane_s16::<1>(i0),
            };
        }
    }
}
