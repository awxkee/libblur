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

pub(crate) fn filter_row_complex_u8_f32(
    arena: Arena,
    arena_src: &[u8],
    dst: &mut [Complex<f32>],
    image_size: ImageSize,
    kernel: &[Complex<f32>],
) {
    unsafe {
        let width = image_size.width;

        let src = arena_src;

        let local_src = src;

        let length = kernel.len();

        let mut cx = 0usize;

        let max_width = width * arena.components;

        let coeff = vld1_f32(kernel.get_unchecked(0..).as_ptr().cast());

        while cx + 16 < max_width {
            let shifted_src = local_src.get_unchecked(cx..);

            let values = vld1q_u8(shifted_src.as_ptr().cast());

            let lo = vmovl_u8(vget_low_u8(values));
            let hi = vmovl_high_u8(values);

            let a0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo)));
            let a1 = vcvtq_f32_u32(vmovl_high_u16(lo));
            let a2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi)));
            let a3 = vcvtq_f32_u32(vmovl_high_u16(hi));

            let mut r0 = vmulq_lane_f32::<0>(a0, coeff);
            let mut r1 = vmulq_lane_f32::<0>(a1, coeff);
            let mut r2 = vmulq_lane_f32::<0>(a2, coeff);
            let mut r3 = vmulq_lane_f32::<0>(a3, coeff);

            let mut i0 = vmulq_lane_f32::<1>(a0, coeff);
            let mut i1 = vmulq_lane_f32::<1>(a1, coeff);
            let mut i2 = vmulq_lane_f32::<1>(a2, coeff);
            let mut i3 = vmulq_lane_f32::<1>(a3, coeff);

            for i in 1..length {
                let coeff = vld1_f32(kernel.get_unchecked(i..).as_ptr().cast());

                let values = vld1q_u8(
                    shifted_src
                        .get_unchecked(i * arena.components..)
                        .as_ptr()
                        .cast(),
                );

                let lo = vmovl_u8(vget_low_u8(values));
                let hi = vmovl_high_u8(values);

                let a0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo)));
                let a1 = vcvtq_f32_u32(vmovl_high_u16(lo));
                let a2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi)));
                let a3 = vcvtq_f32_u32(vmovl_high_u16(hi));

                r0 = vfmaq_lane_f32::<0>(r0, a0, coeff);
                r1 = vfmaq_lane_f32::<0>(r1, a1, coeff);
                r2 = vfmaq_lane_f32::<0>(r2, a2, coeff);
                r3 = vfmaq_lane_f32::<0>(r3, a3, coeff);

                i0 = vfmaq_lane_f32::<1>(i0, a0, coeff);
                i1 = vfmaq_lane_f32::<1>(i1, a1, coeff);
                i2 = vfmaq_lane_f32::<1>(i2, a2, coeff);
                i3 = vfmaq_lane_f32::<1>(i3, a3, coeff);
            }

            let dst0 = dst.get_unchecked_mut(cx..);
            vst2q_f32(dst0.as_mut_ptr().cast(), float32x4x2_t(r0, i0));
            vst2q_f32(
                dst0.get_unchecked_mut(4..).as_mut_ptr().cast(),
                float32x4x2_t(r1, i1),
            );
            vst2q_f32(
                dst0.get_unchecked_mut(8..).as_mut_ptr().cast(),
                float32x4x2_t(r2, i2),
            );
            vst2q_f32(
                dst0.get_unchecked_mut(12..).as_mut_ptr().cast(),
                float32x4x2_t(r3, i3),
            );
            cx += 16;
        }

        while cx + 8 < max_width {
            let shifted_src = local_src.get_unchecked(cx..);

            let values = vld1_u8(shifted_src.as_ptr().cast());

            let lo = vmovl_u8(values);

            let a0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo)));
            let a1 = vcvtq_f32_u32(vmovl_high_u16(lo));

            let mut r0 = vmulq_lane_f32::<0>(a0, coeff);
            let mut r1 = vmulq_lane_f32::<0>(a1, coeff);

            let mut i0 = vmulq_lane_f32::<1>(a0, coeff);
            let mut i1 = vmulq_lane_f32::<1>(a1, coeff);

            for i in 1..length {
                let coeff = vld1_f32(kernel.get_unchecked(i..).as_ptr().cast());

                let values = vld1_u8(
                    shifted_src
                        .get_unchecked(i * arena.components..)
                        .as_ptr()
                        .cast(),
                );

                let lo = vmovl_u8(values);

                let a0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo)));
                let a1 = vcvtq_f32_u32(vmovl_high_u16(lo));

                r0 = vfmaq_lane_f32::<0>(r0, a0, coeff);
                r1 = vfmaq_lane_f32::<0>(r1, a1, coeff);

                i0 = vfmaq_lane_f32::<1>(i0, a0, coeff);
                i1 = vfmaq_lane_f32::<1>(i1, a1, coeff);
            }

            let dst0 = dst.get_unchecked_mut(cx..);
            vst2q_f32(dst0.as_mut_ptr().cast(), float32x4x2_t(r0, i0));
            vst2q_f32(
                dst0.get_unchecked_mut(4..).as_mut_ptr().cast(),
                float32x4x2_t(r1, i1),
            );
            cx += 8;
        }

        for x in cx..max_width {
            let shifted_src = local_src.get_unchecked(x..);
            let mut r0 =
                vmul_lane_f32::<0>(vdup_n_f32(*shifted_src.get_unchecked(0) as f32), coeff);
            let mut i0 =
                vmul_lane_f32::<1>(vdup_n_f32(*shifted_src.get_unchecked(0) as f32), coeff);

            for i in 1..length {
                let coeff = vld1_f32(kernel.get_unchecked(i..).as_ptr().cast());

                let a0 = vdup_n_f32(*shifted_src.get_unchecked(i * arena.components) as f32);

                r0 = vfma_lane_f32::<0>(r0, a0, coeff);
                i0 = vfma_lane_f32::<1>(i0, a0, coeff);
            }
            *dst.get_unchecked_mut(x) = Complex {
                re: vget_lane_f32::<0>(r0),
                im: vget_lane_f32::<1>(i0),
            };
        }
    }
}
