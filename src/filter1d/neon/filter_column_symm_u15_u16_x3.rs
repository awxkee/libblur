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
use crate::filter1d::arena::Arena;
use crate::filter1d::filter_1d_column_handler::FilterBrows;
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::neon::utils::{
    vfmlaq_symm_u16_u16, vmullq_u16_by_u16, vqmovnq_u32_u16, xvld1q_u16_x2, xvst1q_u16_x2,
};
use crate::filter1d::to_approx_storage::ToApproxStorage;
use crate::img_size::ImageSize;
use crate::mlaf::mlaf;
use std::arch::aarch64::*;
use std::ops::{Add, Mul};

pub(crate) fn filter_symm_column_neon_uq15_u16_x3(
    arena: Arena,
    brows: FilterBrows<u16>,
    dst: &mut [u16],
    image_size: ImageSize,
    dst_stride: usize,
    scanned_kernel: &[ScanPoint1d<u32>],
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

        while cx + 16 < image_width {
            let coeff = vdupq_n_u16(scanned_kernel.get_unchecked(half_len).weight as u16);

            let v_src0 = brows0.get_unchecked(half_len).get_unchecked(cx..);
            let v_src1 = brows1.get_unchecked(half_len).get_unchecked(cx..);
            let v_src2 = brows2.get_unchecked(half_len).get_unchecked(cx..);

            let source0 = xvld1q_u16_x2(v_src0.as_ptr());
            let source1 = xvld1q_u16_x2(v_src1.as_ptr());
            let source2 = xvld1q_u16_x2(v_src2.as_ptr());

            let mut k0_0 = vmullq_u16_by_u16(source0.0, coeff);
            let mut k1_0 = vmullq_u16_by_u16(source0.1, coeff);

            let mut k0_1 = vmullq_u16_by_u16(source1.0, coeff);
            let mut k1_1 = vmullq_u16_by_u16(source1.1, coeff);

            let mut k0_2 = vmullq_u16_by_u16(source2.0, coeff);
            let mut k1_2 = vmullq_u16_by_u16(source2.1, coeff);

            for i in 0..half_len {
                let coeff = vdupq_n_u32(scanned_kernel.get_unchecked(i).weight);
                let rollback = length - i - 1;

                let v_source0_0 =
                    xvld1q_u16_x2(brows0.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1_0 =
                    xvld1q_u16_x2(brows0.get_unchecked(rollback).get_unchecked(cx..).as_ptr());

                let v_source0_1 =
                    xvld1q_u16_x2(brows1.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1_1 =
                    xvld1q_u16_x2(brows1.get_unchecked(rollback).get_unchecked(cx..).as_ptr());

                let v_source0_2 =
                    xvld1q_u16_x2(brows2.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1_2 =
                    xvld1q_u16_x2(brows2.get_unchecked(rollback).get_unchecked(cx..).as_ptr());

                k0_0 = vfmlaq_symm_u16_u16(k0_0, v_source0_0.0, v_source1_0.0, coeff);
                k1_0 = vfmlaq_symm_u16_u16(k1_0, v_source0_0.1, v_source1_0.1, coeff);

                k0_1 = vfmlaq_symm_u16_u16(k0_1, v_source0_1.0, v_source1_1.0, coeff);
                k1_1 = vfmlaq_symm_u16_u16(k1_1, v_source0_1.1, v_source1_1.1, coeff);

                k0_2 = vfmlaq_symm_u16_u16(k0_2, v_source0_2.0, v_source1_2.0, coeff);
                k1_2 = vfmlaq_symm_u16_u16(k1_2, v_source0_2.1, v_source1_2.1, coeff);
            }

            let dst_ptr0 = dst0.get_unchecked_mut(cx..).as_mut_ptr();
            let dst_ptr1 = dst1.get_unchecked_mut(cx..).as_mut_ptr();
            let dst_ptr2 = dst2.get_unchecked_mut(cx..).as_mut_ptr();

            xvst1q_u16_x2(
                dst_ptr0,
                uint16x8x2_t(vqmovnq_u32_u16(k0_0), vqmovnq_u32_u16(k1_0)),
            );
            xvst1q_u16_x2(
                dst_ptr1,
                uint16x8x2_t(vqmovnq_u32_u16(k0_1), vqmovnq_u32_u16(k1_1)),
            );
            xvst1q_u16_x2(
                dst_ptr2,
                uint16x8x2_t(vqmovnq_u32_u16(k0_2), vqmovnq_u32_u16(k1_2)),
            );
            cx += 16;
        }

        while cx + 8 < image_width {
            let coeff = vdupq_n_u16(scanned_kernel.get_unchecked(half_len).weight as u16);

            let v_src0 = brows0.get_unchecked(half_len).get_unchecked(cx..);
            let v_src1 = brows1.get_unchecked(half_len).get_unchecked(cx..);
            let v_src2 = brows2.get_unchecked(half_len).get_unchecked(cx..);

            let source0 = vld1q_u16(v_src0.as_ptr());
            let source1 = vld1q_u16(v_src1.as_ptr());
            let source2 = vld1q_u16(v_src2.as_ptr());

            let mut k0 = vmullq_u16_by_u16(source0, coeff);
            let mut k1 = vmullq_u16_by_u16(source1, coeff);
            let mut k2 = vmullq_u16_by_u16(source2, coeff);

            for i in 0..half_len {
                let coeff = vdupq_n_u32(scanned_kernel.get_unchecked(i).weight as u32);
                let rollback = length - i - 1;
                let v_source0_0 = vld1q_u16(brows0.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1_0 =
                    vld1q_u16(brows0.get_unchecked(rollback).get_unchecked(cx..).as_ptr());

                let v_source0_1 = vld1q_u16(brows1.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1_1 =
                    vld1q_u16(brows1.get_unchecked(rollback).get_unchecked(cx..).as_ptr());

                let v_source0_2 = vld1q_u16(brows2.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1_2 =
                    vld1q_u16(brows2.get_unchecked(rollback).get_unchecked(cx..).as_ptr());

                k0 = vfmlaq_symm_u16_u16(k0, v_source0_0, v_source1_0, coeff);
                k1 = vfmlaq_symm_u16_u16(k1, v_source0_1, v_source1_1, coeff);
                k2 = vfmlaq_symm_u16_u16(k2, v_source0_2, v_source1_2, coeff);
            }

            let dst_ptr0 = dst0.get_unchecked_mut(cx..).as_mut_ptr();
            let dst_ptr1 = dst1.get_unchecked_mut(cx..).as_mut_ptr();
            let dst_ptr2 = dst2.get_unchecked_mut(cx..).as_mut_ptr();

            vst1q_u16(dst_ptr0, vqmovnq_u32_u16(k0));
            vst1q_u16(dst_ptr1, vqmovnq_u32_u16(k1));
            vst1q_u16(dst_ptr2, vqmovnq_u32_u16(k2));

            cx += 8;
        }

        while cx + 4 < image_width {
            let coeff = vdup_n_u16(scanned_kernel.get_unchecked(half_len).weight as u16);

            let v_src0 = brows0.get_unchecked(half_len).get_unchecked(cx..);
            let v_src1 = brows1.get_unchecked(half_len).get_unchecked(cx..);
            let v_src2 = brows2.get_unchecked(half_len).get_unchecked(cx..);

            let source0 = vld1_u16(v_src0.as_ptr());
            let source1 = vld1_u16(v_src1.as_ptr());
            let source2 = vld1_u16(v_src2.as_ptr());

            let mut k0 = vmull_u16(source0, coeff);
            let mut k1 = vmull_u16(source1, coeff);
            let mut k2 = vmull_u16(source2, coeff);

            for i in 0..half_len {
                let coeff = vdupq_n_u32(scanned_kernel.get_unchecked(i).weight as u32);
                let rollback = length - i - 1;
                let v_source0_0 = vld1_u16(brows0.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1_0 =
                    vld1_u16(brows0.get_unchecked(rollback).get_unchecked(cx..).as_ptr());

                let v_source0_1 = vld1_u16(brows1.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1_1 =
                    vld1_u16(brows1.get_unchecked(rollback).get_unchecked(cx..).as_ptr());

                let v_source0_2 = vld1_u16(brows2.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1_2 =
                    vld1_u16(brows2.get_unchecked(rollback).get_unchecked(cx..).as_ptr());

                k0 = vmlaq_u32(k0, vaddl_u16(v_source0_0, v_source1_0), coeff);
                k1 = vmlaq_u32(k1, vaddl_u16(v_source0_1, v_source1_1), coeff);
                k2 = vmlaq_u32(k2, vaddl_u16(v_source0_2, v_source1_2), coeff);
            }

            let dst_ptr0 = dst0.get_unchecked_mut(cx..).as_mut_ptr();
            let dst_ptr1 = dst1.get_unchecked_mut(cx..).as_mut_ptr();
            let dst_ptr2 = dst2.get_unchecked_mut(cx..).as_mut_ptr();

            vst1_u16(dst_ptr0, vqshrn_n_u32::<15>(k0));
            vst1_u16(dst_ptr1, vqshrn_n_u32::<15>(k1));
            vst1_u16(dst_ptr2, vqshrn_n_u32::<15>(k2));

            cx += 4;
        }

        for x in cx..dst_stride {
            let coeff = scanned_kernel.get_unchecked(half_len).weight;

            let mut k0 = (*brows0.get_unchecked(half_len).get_unchecked(x) as u32).mul(coeff);
            let mut k1 = (*brows1.get_unchecked(half_len).get_unchecked(x) as u32).mul(coeff);
            let mut k2 = (*brows2.get_unchecked(half_len).get_unchecked(x) as u32).mul(coeff);

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;
                k0 = mlaf(
                    k0,
                    ((*brows0.get_unchecked(i).get_unchecked(x)) as u32)
                        .add(*brows0.get_unchecked(rollback).get_unchecked(x) as u32),
                    coeff.weight,
                );

                k1 = mlaf(
                    k1,
                    ((*brows1.get_unchecked(i).get_unchecked(x)) as u32)
                        .add(*brows1.get_unchecked(rollback).get_unchecked(x) as u32),
                    coeff.weight,
                );

                k2 = mlaf(
                    k2,
                    ((*brows2.get_unchecked(i).get_unchecked(x)) as u32)
                        .add(*brows2.get_unchecked(rollback).get_unchecked(x) as u32),
                    coeff.weight,
                );
            }

            *dst0.get_unchecked_mut(x) = k0.to_approx_();
            *dst1.get_unchecked_mut(x) = k1.to_approx_();
            *dst2.get_unchecked_mut(x) = k2.to_approx_();
        }
    }
}
