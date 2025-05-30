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

pub(crate) fn filter_row_complex_u16_i32_q(
    arena: Arena,
    arena_src: &[u16],
    dst: &mut [Complex<i32>],
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

        let coeff = vmovl_s16(vreinterpret_s16_s32(vld1_lane_s32::<0>(
            kernel.get_unchecked(0..).as_ptr().cast(),
            vdup_n_s32(0),
        )));

        while cx + 16 < max_width {
            let shifted_src = local_src.get_unchecked(cx..);

            let values0 = vld1q_u16(shifted_src.as_ptr().cast());
            let values1 = vld1q_u16(shifted_src.get_unchecked(8..).as_ptr().cast());

            let lo0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(values0)));
            let hi0 = vreinterpretq_s32_u32(vmovl_high_u16(values0));

            let lo1 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(values1)));
            let hi1 = vreinterpretq_s32_u32(vmovl_high_u16(values1));

            let mut r0 = vmulq_laneq_s32::<0>(lo0, coeff);
            let mut r1 = vmulq_laneq_s32::<0>(hi0, coeff);
            let mut r2 = vmulq_laneq_s32::<0>(lo1, coeff);
            let mut r3 = vmulq_laneq_s32::<0>(hi1, coeff);

            let mut i0 = vmulq_laneq_s32::<1>(lo0, coeff);
            let mut i1 = vmulq_laneq_s32::<1>(hi0, coeff);
            let mut i2 = vmulq_laneq_s32::<1>(lo1, coeff);
            let mut i3 = vmulq_laneq_s32::<1>(hi1, coeff);

            for i in 1..length {
                let coeff = vmovl_s16(vreinterpret_s16_s32(vld1_lane_s32::<0>(
                    kernel.get_unchecked(i..).as_ptr().cast(),
                    vdup_n_s32(0),
                )));

                let values0 = vld1q_u16(
                    shifted_src
                        .get_unchecked(i * arena.components..)
                        .as_ptr()
                        .cast(),
                );
                let values1 = vld1q_u16(
                    shifted_src
                        .get_unchecked(i * arena.components + 8..)
                        .as_ptr()
                        .cast(),
                );

                let lo0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(values0)));
                let hi0 = vreinterpretq_s32_u32(vmovl_high_u16(values0));

                let lo1 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(values1)));
                let hi1 = vreinterpretq_s32_u32(vmovl_high_u16(values1));

                r0 = vmlaq_laneq_s32::<0>(r0, lo0, coeff);
                r1 = vmlaq_laneq_s32::<0>(r1, hi0, coeff);
                r2 = vmlaq_laneq_s32::<0>(r2, lo1, coeff);
                r3 = vmlaq_laneq_s32::<0>(r3, hi1, coeff);

                i0 = vmlaq_laneq_s32::<1>(i0, lo0, coeff);
                i1 = vmlaq_laneq_s32::<1>(i1, hi0, coeff);
                i2 = vmlaq_laneq_s32::<1>(i2, lo1, coeff);
                i3 = vmlaq_laneq_s32::<1>(i3, hi1, coeff);
            }

            let dst0 = dst.get_unchecked_mut(cx..);
            vst2q_s32(
                dst0.as_mut_ptr().cast(),
                int32x4x2_t(vrshrq_n_s32::<14>(r0), vrshrq_n_s32::<14>(i0)),
            );
            vst2q_s32(
                dst0.get_unchecked_mut(4..).as_mut_ptr().cast(),
                int32x4x2_t(vrshrq_n_s32::<14>(r1), vrshrq_n_s32::<14>(i1)),
            );
            vst2q_s32(
                dst0.get_unchecked_mut(8..).as_mut_ptr().cast(),
                int32x4x2_t(vrshrq_n_s32::<14>(r2), vrshrq_n_s32::<14>(i2)),
            );
            vst2q_s32(
                dst0.get_unchecked_mut(12..).as_mut_ptr().cast(),
                int32x4x2_t(vrshrq_n_s32::<14>(r3), vrshrq_n_s32::<14>(i3)),
            );
            cx += 16;
        }

        while cx + 8 < max_width {
            let shifted_src = local_src.get_unchecked(cx..);

            let values = vld1q_u16(shifted_src.as_ptr().cast());

            let lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(values)));
            let hi = vreinterpretq_s32_u32(vmovl_high_u16(values));

            let mut r0 = vmulq_laneq_s32::<0>(lo, coeff);
            let mut r1 = vmulq_laneq_s32::<0>(hi, coeff);

            let mut i0 = vmulq_laneq_s32::<1>(lo, coeff);
            let mut i1 = vmulq_laneq_s32::<1>(hi, coeff);

            for i in 1..length {
                let coeff = vmovl_s16(vreinterpret_s16_s32(vld1_lane_s32::<0>(
                    kernel.get_unchecked(i..).as_ptr().cast(),
                    vdup_n_s32(0),
                )));

                let values = vld1q_u16(
                    shifted_src
                        .get_unchecked(i * arena.components..)
                        .as_ptr()
                        .cast(),
                );

                let lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(values)));
                let hi = vreinterpretq_s32_u32(vmovl_high_u16(values));

                r0 = vmlaq_laneq_s32::<0>(r0, lo, coeff);
                r1 = vmlaq_laneq_s32::<0>(r1, hi, coeff);

                i0 = vmlaq_laneq_s32::<1>(i0, lo, coeff);
                i1 = vmlaq_laneq_s32::<1>(i1, hi, coeff);
            }

            let dst0 = dst.get_unchecked_mut(cx..);
            vst2q_s32(
                dst0.as_mut_ptr().cast(),
                int32x4x2_t(vrshrq_n_s32::<14>(r0), vrshrq_n_s32::<14>(i0)),
            );
            vst2q_s32(
                dst0.get_unchecked_mut(4..).as_mut_ptr().cast(),
                int32x4x2_t(vrshrq_n_s32::<14>(r1), vrshrq_n_s32::<14>(i1)),
            );
            cx += 8;
        }

        for x in cx..max_width {
            let shifted_src = local_src.get_unchecked(x..);
            let mut r0 =
                vmulq_laneq_s32::<0>(vdupq_n_s32(*shifted_src.get_unchecked(0) as i32), coeff);
            let mut i0 =
                vmulq_laneq_s32::<1>(vdupq_n_s32(*shifted_src.get_unchecked(0) as i32), coeff);

            for i in 1..length {
                let coeff = vmovl_s16(vreinterpret_s16_s32(vld1_lane_s32::<0>(
                    kernel.get_unchecked(i..).as_ptr().cast(),
                    vdup_n_s32(0),
                )));

                let a0 = vdupq_n_s32(*shifted_src.get_unchecked(i * arena.components) as i32);

                r0 = vmlaq_laneq_s32::<0>(r0, a0, coeff);
                i0 = vmlaq_laneq_s32::<1>(i0, a0, coeff);
            }

            r0 = vrshrq_n_s32::<14>(r0);
            i0 = vrshrq_n_s32::<14>(i0);

            *dst.get_unchecked_mut(x) = Complex {
                re: vgetq_lane_s32::<0>(r0),
                im: vgetq_lane_s32::<1>(i0),
            };
        }
    }
}
