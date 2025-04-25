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
use crate::filter1d::filter_1d_column_handler::FilterBrows;
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::neon::utils::{
    vmla_symm_hi_u8_s16, vmlaq_symm_hi_u8_s16, vmull_expand_i16, vmullq_expand_i16,
    vqmovnq_s16x2_u8, xvld1q_u8_x2, xvld4u8, xvst1q_u8_x2, xvst_u8x4_q15,
};
use crate::filter1d::to_approx_storage::ToApproxStorage;
use crate::img_size::ImageSize;
use std::arch::aarch64::*;
use std::ops::{Add, Mul};

pub(crate) fn filter_column_symm_neon_u8_i32_rdm_x3(
    arena: Arena,
    brows: FilterBrows<u8>,
    dst: &mut [u8],
    image_size: ImageSize,
    dst_stride: usize,
    scanned_kernel: &[ScanPoint1d<i32>],
) {
    unsafe {
        executor_unit(arena, brows, dst, image_size, dst_stride, scanned_kernel);
    }
}

#[target_feature(enable = "rdm")]
unsafe fn executor_unit(
    arena: Arena,
    brows: FilterBrows<u8>,
    dst: &mut [u8],
    image_size: ImageSize,
    dst_stride: usize,
    scanned_kernel: &[ScanPoint1d<i32>],
) {
    unsafe {
        let image_width = image_size.width * arena.components;

        let length = scanned_kernel.len();
        let half_len = length / 2;

        let brows0 = brows.brows[0];
        let brows1 = brows.brows[1];
        let brows2 = brows.brows[2];

        let (dst0, dst_rem) = dst.split_at_mut(dst_stride);
        let (dst1, dst2) = dst_rem.split_at_mut(dst_stride);

        let mut cx = 0usize;

        while cx + 32 < image_width {
            let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(half_len).weight as i16);

            let v_src0 = brows0.get_unchecked(half_len).get_unchecked(cx..);
            let v_src1 = brows1.get_unchecked(half_len).get_unchecked(cx..);
            let v_src2 = brows2.get_unchecked(half_len).get_unchecked(cx..);

            let source0 = xvld1q_u8_x2(v_src0.as_ptr());
            let source1 = xvld1q_u8_x2(v_src1.as_ptr());
            let source2 = xvld1q_u8_x2(v_src2.as_ptr());

            let mut k0_0 = vmullq_expand_i16(source0.0, coeff);
            let mut k1_0 = vmullq_expand_i16(source0.1, coeff);

            let mut k0_1 = vmullq_expand_i16(source1.0, coeff);
            let mut k1_1 = vmullq_expand_i16(source1.1, coeff);

            let mut k0_2 = vmullq_expand_i16(source2.0, coeff);
            let mut k1_2 = vmullq_expand_i16(source2.1, coeff);

            for i in 0..half_len {
                let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight as i16);
                let rollback = length - i - 1;
                let v_source0_0 =
                    xvld1q_u8_x2(brows0.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1_0 =
                    xvld1q_u8_x2(brows0.get_unchecked(rollback).get_unchecked(cx..).as_ptr());
                let v_source0_1 =
                    xvld1q_u8_x2(brows1.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1_1 =
                    xvld1q_u8_x2(brows1.get_unchecked(rollback).get_unchecked(cx..).as_ptr());
                let v_source0_2 =
                    xvld1q_u8_x2(brows2.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1_2 =
                    xvld1q_u8_x2(brows2.get_unchecked(rollback).get_unchecked(cx..).as_ptr());
                k0_0 = vmlaq_symm_hi_u8_s16(k0_0, v_source0_0.0, v_source1_0.0, coeff);
                k1_0 = vmlaq_symm_hi_u8_s16(k1_0, v_source0_0.1, v_source1_0.1, coeff);

                k0_1 = vmlaq_symm_hi_u8_s16(k0_1, v_source0_1.0, v_source1_1.0, coeff);
                k1_1 = vmlaq_symm_hi_u8_s16(k1_1, v_source0_1.1, v_source1_1.1, coeff);

                k0_2 = vmlaq_symm_hi_u8_s16(k0_2, v_source0_2.0, v_source1_2.0, coeff);
                k1_2 = vmlaq_symm_hi_u8_s16(k1_2, v_source0_2.1, v_source1_2.1, coeff);
            }

            let dst_ptr0 = dst0.get_unchecked_mut(cx..).as_mut_ptr();
            let dst_ptr1 = dst1.get_unchecked_mut(cx..).as_mut_ptr();
            let dst_ptr2 = dst2.get_unchecked_mut(cx..).as_mut_ptr();

            xvst1q_u8_x2(
                dst_ptr0,
                uint8x16x2_t(vqmovnq_s16x2_u8(k0_0), vqmovnq_s16x2_u8(k1_0)),
            );
            xvst1q_u8_x2(
                dst_ptr1,
                uint8x16x2_t(vqmovnq_s16x2_u8(k0_1), vqmovnq_s16x2_u8(k1_1)),
            );
            xvst1q_u8_x2(
                dst_ptr2,
                uint8x16x2_t(vqmovnq_s16x2_u8(k0_2), vqmovnq_s16x2_u8(k1_2)),
            );
            cx += 32;
        }

        while cx + 16 < image_width {
            let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(half_len).weight as i16);

            let v_src0 = brows0.get_unchecked(half_len).get_unchecked(cx..);
            let v_src1 = brows1.get_unchecked(half_len).get_unchecked(cx..);
            let v_src2 = brows2.get_unchecked(half_len).get_unchecked(cx..);

            let source0 = vld1q_u8(v_src0.as_ptr());
            let source1 = vld1q_u8(v_src1.as_ptr());
            let source2 = vld1q_u8(v_src2.as_ptr());

            let mut k0 = vmullq_expand_i16(source0, coeff);
            let mut k1 = vmullq_expand_i16(source1, coeff);
            let mut k2 = vmullq_expand_i16(source2, coeff);

            for i in 0..half_len {
                let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight as i16);
                let rollback = length - i - 1;
                let v_source0_0 = vld1q_u8(brows0.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1_0 =
                    vld1q_u8(brows0.get_unchecked(rollback).get_unchecked(cx..).as_ptr());
                let v_source0_1 = vld1q_u8(brows1.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1_1 =
                    vld1q_u8(brows1.get_unchecked(rollback).get_unchecked(cx..).as_ptr());
                let v_source0_2 = vld1q_u8(brows2.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1_2 =
                    vld1q_u8(brows2.get_unchecked(rollback).get_unchecked(cx..).as_ptr());
                k0 = vmlaq_symm_hi_u8_s16(k0, v_source0_0, v_source1_0, coeff);
                k1 = vmlaq_symm_hi_u8_s16(k1, v_source0_1, v_source1_1, coeff);
                k2 = vmlaq_symm_hi_u8_s16(k2, v_source0_2, v_source1_2, coeff);
            }

            let dst_ptr0 = dst0.get_unchecked_mut(cx..).as_mut_ptr();
            let dst_ptr1 = dst1.get_unchecked_mut(cx..).as_mut_ptr();
            let dst_ptr2 = dst2.get_unchecked_mut(cx..).as_mut_ptr();

            vst1q_u8(dst_ptr0, vqmovnq_s16x2_u8(k0));
            vst1q_u8(dst_ptr1, vqmovnq_s16x2_u8(k1));
            vst1q_u8(dst_ptr2, vqmovnq_s16x2_u8(k2));
            cx += 16;
        }

        while cx + 4 < image_width {
            let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(half_len).weight as i16);

            let v_src0 = brows0.get_unchecked(half_len).get_unchecked(cx..);
            let v_src1 = brows1.get_unchecked(half_len).get_unchecked(cx..);
            let v_src2 = brows2.get_unchecked(half_len).get_unchecked(cx..);

            let source0 = xvld4u8(v_src0.as_ptr());
            let source1 = xvld4u8(v_src1.as_ptr());
            let source2 = xvld4u8(v_src2.as_ptr());

            let mut k0 = vmull_expand_i16(source0, coeff);
            let mut k1 = vmull_expand_i16(source1, coeff);
            let mut k2 = vmull_expand_i16(source2, coeff);

            for i in 0..half_len {
                let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight as i16);
                let rollback = length - i - 1;
                let v_source0_0 = xvld4u8(brows0.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1_0 =
                    xvld4u8(brows0.get_unchecked(rollback).get_unchecked(cx..).as_ptr());
                let v_source0_1 = xvld4u8(brows1.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1_1 =
                    xvld4u8(brows1.get_unchecked(rollback).get_unchecked(cx..).as_ptr());
                let v_source0_2 = xvld4u8(brows2.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1_2 =
                    xvld4u8(brows2.get_unchecked(rollback).get_unchecked(cx..).as_ptr());
                k0 = vmla_symm_hi_u8_s16(k0, v_source0_0, v_source1_0, coeff);
                k1 = vmla_symm_hi_u8_s16(k1, v_source0_1, v_source1_1, coeff);
                k2 = vmla_symm_hi_u8_s16(k2, v_source0_2, v_source1_2, coeff);
            }

            let dst_ptr0 = dst0.get_unchecked_mut(cx..).as_mut_ptr();
            let dst_ptr1 = dst1.get_unchecked_mut(cx..).as_mut_ptr();
            let dst_ptr2 = dst2.get_unchecked_mut(cx..).as_mut_ptr();

            xvst_u8x4_q15(dst_ptr0, k0);
            xvst_u8x4_q15(dst_ptr1, k1);
            xvst_u8x4_q15(dst_ptr2, k2);
            cx += 4;
        }

        for x in cx..image_width {
            let coeff = *scanned_kernel.get_unchecked(0);

            let v_src0 = brows0.get_unchecked(half_len).get_unchecked(x..);
            let v_src1 = brows1.get_unchecked(half_len).get_unchecked(x..);
            let v_src2 = brows2.get_unchecked(half_len).get_unchecked(x..);

            let mut k0 = ((*v_src0.get_unchecked(0)) as i32).mul(coeff.weight);
            let mut k1 = ((*v_src1.get_unchecked(0)) as i32).mul(coeff.weight);
            let mut k2 = ((*v_src2.get_unchecked(0)) as i32).mul(coeff.weight);

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;
                k0 = ((*brows0.get_unchecked(i).get_unchecked(x)) as i32)
                    .add((*brows0.get_unchecked(rollback).get_unchecked(x)) as i32)
                    .mul(coeff.weight)
                    .add(k0);

                k1 = ((*brows1.get_unchecked(i).get_unchecked(x)) as i32)
                    .add((*brows1.get_unchecked(rollback).get_unchecked(x)) as i32)
                    .mul(coeff.weight)
                    .add(k1);

                k2 = ((*brows2.get_unchecked(i).get_unchecked(x)) as i32)
                    .add((*brows2.get_unchecked(rollback).get_unchecked(x)) as i32)
                    .mul(coeff.weight)
                    .add(k2);
            }

            *dst0.get_unchecked_mut(x) = k0.to_approx_();
            *dst1.get_unchecked_mut(x) = k1.to_approx_();
            *dst2.get_unchecked_mut(x) = k2.to_approx_();
        }
    }
}
