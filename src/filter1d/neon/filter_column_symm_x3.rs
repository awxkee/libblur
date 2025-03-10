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
    vfmlaq_symm_u8_f32, vmulq_u8_by_f32, vqmovnq_f32_u8, xvld1q_u8_x2,
    xvst1q_u8_x2,
};
use crate::img_size::ImageSize;
use crate::mlaf::mlaf;
use crate::to_storage::ToStorage;
use std::arch::aarch64::*;
use std::ops::{Add, Mul};

pub(crate) fn filter_symm_column_neon_u8_f32_x3(
    arena: Arena,
    brows: FilterBrows<u8>,
    dst: &mut [u8],
    image_size: ImageSize,
    dst_stride: usize,
    scanned_kernel: &[ScanPoint1d<f32>],
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
            let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(half_len).weight);

            let shifted_src0 = brows0.get_unchecked(half_len).get_unchecked(cx..);
            let shifted_src1 = brows1.get_unchecked(half_len).get_unchecked(cx..);
            let shifted_src2 = brows2.get_unchecked(half_len).get_unchecked(cx..);

            let source0 = xvld1q_u8_x2(shifted_src0.as_ptr());
            let source1 = xvld1q_u8_x2(shifted_src1.as_ptr());
            let source2 = xvld1q_u8_x2(shifted_src2.as_ptr());

            let mut k0_0 = vmulq_u8_by_f32(source0.0, coeff);
            let mut k1_0 = vmulq_u8_by_f32(source0.1, coeff);

            let mut k0_1 = vmulq_u8_by_f32(source1.0, coeff);
            let mut k1_1 = vmulq_u8_by_f32(source1.1, coeff);

            let mut k0_2 = vmulq_u8_by_f32(source2.0, coeff);
            let mut k1_2 = vmulq_u8_by_f32(source2.1, coeff);

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(i).weight);
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

                k0_0 = vfmlaq_symm_u8_f32(k0_0, v_source0_0.0, v_source1_0.0, coeff);
                k1_0 = vfmlaq_symm_u8_f32(k1_0, v_source0_0.1, v_source1_0.1, coeff);

                k0_1 = vfmlaq_symm_u8_f32(k0_1, v_source0_1.0, v_source1_1.0, coeff);
                k1_1 = vfmlaq_symm_u8_f32(k1_1, v_source0_1.1, v_source1_1.1, coeff);

                k0_2 = vfmlaq_symm_u8_f32(k0_2, v_source0_2.0, v_source1_2.0, coeff);
                k1_2 = vfmlaq_symm_u8_f32(k1_2, v_source0_2.1, v_source1_2.1, coeff);
            }

            let dst_ptr0 = dst0.get_unchecked_mut(cx..).as_mut_ptr();
            let dst_ptr1 = dst1.get_unchecked_mut(cx..).as_mut_ptr();
            let dst_ptr2 = dst2.get_unchecked_mut(cx..).as_mut_ptr();

            xvst1q_u8_x2(
                dst_ptr0,
                uint8x16x2_t(vqmovnq_f32_u8(k0_0), vqmovnq_f32_u8(k1_0)),
            );
            xvst1q_u8_x2(
                dst_ptr1,
                uint8x16x2_t(vqmovnq_f32_u8(k0_1), vqmovnq_f32_u8(k1_1)),
            );
            xvst1q_u8_x2(
                dst_ptr2,
                uint8x16x2_t(vqmovnq_f32_u8(k0_2), vqmovnq_f32_u8(k1_2)),
            );

            cx += 32;
        }

        while cx + 16 < image_width {
            let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(half_len).weight);

            let shifted_src0 = brows0.get_unchecked(half_len).get_unchecked(cx..);
            let shifted_src1 = brows1.get_unchecked(half_len).get_unchecked(cx..);
            let shifted_src2 = brows2.get_unchecked(half_len).get_unchecked(cx..);

            let source0 = vld1q_u8(shifted_src0.as_ptr());
            let source1 = vld1q_u8(shifted_src1.as_ptr());
            let source2 = vld1q_u8(shifted_src2.as_ptr());

            let mut k0_0 = vmulq_u8_by_f32(source0, coeff);

            let mut k0_1 = vmulq_u8_by_f32(source1, coeff);

            let mut k0_2 = vmulq_u8_by_f32(source2, coeff);

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(i).weight);
                let v_source0_0 = vld1q_u8(brows0.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1_0 =
                    vld1q_u8(brows0.get_unchecked(rollback).get_unchecked(cx..).as_ptr());

                let v_source0_1 = vld1q_u8(brows1.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1_1 =
                    vld1q_u8(brows1.get_unchecked(rollback).get_unchecked(cx..).as_ptr());

                let v_source0_2 = vld1q_u8(brows2.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1_2 =
                    vld1q_u8(brows2.get_unchecked(rollback).get_unchecked(cx..).as_ptr());

                k0_0 = vfmlaq_symm_u8_f32(k0_0, v_source0_0, v_source1_0, coeff);
                k0_1 = vfmlaq_symm_u8_f32(k0_1, v_source0_1, v_source1_1, coeff);
                k0_2 = vfmlaq_symm_u8_f32(k0_2, v_source0_2, v_source1_2, coeff);
            }

            let dst_ptr0 = dst0.get_unchecked_mut(cx..).as_mut_ptr();
            let dst_ptr1 = dst1.get_unchecked_mut(cx..).as_mut_ptr();
            let dst_ptr2 = dst2.get_unchecked_mut(cx..).as_mut_ptr();

            vst1q_u8(dst_ptr0, vqmovnq_f32_u8(k0_0));
            vst1q_u8(dst_ptr1, vqmovnq_f32_u8(k0_1));
            vst1q_u8(dst_ptr2, vqmovnq_f32_u8(k0_2));
            cx += 16;
        }

        while cx + 4 < image_width {
            let coeff = scanned_kernel.get_unchecked(half_len).weight;

            let mut k0_0 = (*brows0.get_unchecked(half_len).get_unchecked(cx) as f32).mul(coeff);
            let mut k1_0 =
                (*brows0.get_unchecked(half_len).get_unchecked(cx + 1) as f32).mul(coeff);
            let mut k2_0 =
                (*brows0.get_unchecked(half_len).get_unchecked(cx + 2) as f32).mul(coeff);
            let mut k3_0 =
                (*brows0.get_unchecked(half_len).get_unchecked(cx + 3) as f32).mul(coeff);

            let mut k0_1 = (*brows1.get_unchecked(half_len).get_unchecked(cx) as f32).mul(coeff);
            let mut k1_1 =
                (*brows1.get_unchecked(half_len).get_unchecked(cx + 1) as f32).mul(coeff);
            let mut k2_1 =
                (*brows1.get_unchecked(half_len).get_unchecked(cx + 2) as f32).mul(coeff);
            let mut k3_1 =
                (*brows1.get_unchecked(half_len).get_unchecked(cx + 3) as f32).mul(coeff);

            let mut k0_2 = (*brows2.get_unchecked(half_len).get_unchecked(cx) as f32).mul(coeff);
            let mut k1_2 =
                (*brows2.get_unchecked(half_len).get_unchecked(cx + 1) as f32).mul(coeff);
            let mut k2_2 =
                (*brows2.get_unchecked(half_len).get_unchecked(cx + 2) as f32).mul(coeff);
            let mut k3_2 =
                (*brows2.get_unchecked(half_len).get_unchecked(cx + 3) as f32).mul(coeff);

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;
                k0_0 = mlaf(
                    k0_0,
                    ((*brows0.get_unchecked(i).get_unchecked(cx)) as f32)
                        .add(*brows0.get_unchecked(rollback).get_unchecked(cx) as f32),
                    coeff.weight,
                );
                k1_0 = mlaf(
                    k1_0,
                    ((*brows0.get_unchecked(i).get_unchecked(cx + 1)) as f32)
                        .add((*brows0.get_unchecked(rollback).get_unchecked(cx + 1)) as f32),
                    coeff.weight,
                );
                k2_0 = mlaf(
                    k2_0,
                    ((*brows0.get_unchecked(i).get_unchecked(cx + 2)) as f32)
                        .add((*brows0.get_unchecked(rollback).get_unchecked(cx + 2)) as f32),
                    coeff.weight,
                );
                k3_0 = mlaf(
                    k3_0,
                    ((*brows0.get_unchecked(i).get_unchecked(cx + 3)) as f32)
                        .add((*brows0.get_unchecked(rollback).get_unchecked(cx + 3)) as f32),
                    coeff.weight,
                );

                k0_1 = mlaf(
                    k0_1,
                    ((*brows1.get_unchecked(i).get_unchecked(cx)) as f32)
                        .add(*brows1.get_unchecked(rollback).get_unchecked(cx) as f32),
                    coeff.weight,
                );
                k1_1 = mlaf(
                    k1_1,
                    ((*brows1.get_unchecked(i).get_unchecked(cx + 1)) as f32)
                        .add((*brows1.get_unchecked(rollback).get_unchecked(cx + 1)) as f32),
                    coeff.weight,
                );
                k2_1 = mlaf(
                    k2_1,
                    ((*brows1.get_unchecked(i).get_unchecked(cx + 2)) as f32)
                        .add((*brows1.get_unchecked(rollback).get_unchecked(cx + 2)) as f32),
                    coeff.weight,
                );
                k3_1 = mlaf(
                    k3_1,
                    ((*brows1.get_unchecked(i).get_unchecked(cx + 3)) as f32)
                        .add((*brows1.get_unchecked(rollback).get_unchecked(cx + 3)) as f32),
                    coeff.weight,
                );

                k0_2 = mlaf(
                    k0_2,
                    ((*brows2.get_unchecked(i).get_unchecked(cx)) as f32)
                        .add(*brows2.get_unchecked(rollback).get_unchecked(cx) as f32),
                    coeff.weight,
                );
                k1_2 = mlaf(
                    k1_2,
                    ((*brows2.get_unchecked(i).get_unchecked(cx + 1)) as f32)
                        .add((*brows2.get_unchecked(rollback).get_unchecked(cx + 1)) as f32),
                    coeff.weight,
                );
                k2_2 = mlaf(
                    k2_2,
                    ((*brows2.get_unchecked(i).get_unchecked(cx + 2)) as f32)
                        .add((*brows2.get_unchecked(rollback).get_unchecked(cx + 2)) as f32),
                    coeff.weight,
                );
                k3_2 = mlaf(
                    k3_2,
                    ((*brows2.get_unchecked(i).get_unchecked(cx + 3)) as f32)
                        .add((*brows2.get_unchecked(rollback).get_unchecked(cx + 3)) as f32),
                    coeff.weight,
                );
            }

            *dst0.get_unchecked_mut(cx) = k0_0.to_();
            *dst0.get_unchecked_mut(cx + 1) = k1_0.to_();
            *dst0.get_unchecked_mut(cx + 2) = k2_0.to_();
            *dst0.get_unchecked_mut(cx + 3) = k3_0.to_();

            *dst1.get_unchecked_mut(cx) = k0_1.to_();
            *dst1.get_unchecked_mut(cx + 1) = k1_1.to_();
            *dst1.get_unchecked_mut(cx + 2) = k2_1.to_();
            *dst1.get_unchecked_mut(cx + 3) = k3_1.to_();

            *dst2.get_unchecked_mut(cx) = k0_2.to_();
            *dst2.get_unchecked_mut(cx + 1) = k1_2.to_();
            *dst2.get_unchecked_mut(cx + 2) = k2_2.to_();
            *dst2.get_unchecked_mut(cx + 3) = k3_2.to_();

            cx += 4;
        }

        for x in cx..image_width {
            let coeff = scanned_kernel.get_unchecked(half_len).weight;

            let mut k0 = (*brows0.get_unchecked(half_len).get_unchecked(x) as f32).mul(coeff);
            let mut k1 = (*brows1.get_unchecked(half_len).get_unchecked(x) as f32).mul(coeff);
            let mut k2 = (*brows2.get_unchecked(half_len).get_unchecked(x) as f32).mul(coeff);

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;
                k0 = mlaf(
                    k0,
                    ((*brows0.get_unchecked(i).get_unchecked(x)) as f32)
                        .add(*brows0.get_unchecked(rollback).get_unchecked(x) as f32),
                    coeff.weight,
                );

                k1 = mlaf(
                    k1,
                    ((*brows1.get_unchecked(i).get_unchecked(x)) as f32)
                        .add(*brows1.get_unchecked(rollback).get_unchecked(x) as f32),
                    coeff.weight,
                );

                k2 = mlaf(
                    k2,
                    ((*brows2.get_unchecked(i).get_unchecked(x)) as f32)
                        .add(*brows2.get_unchecked(rollback).get_unchecked(x) as f32),
                    coeff.weight,
                );
            }

            *dst0.get_unchecked_mut(x) = k0.to_();
            *dst1.get_unchecked_mut(x) = k1.to_();
            *dst2.get_unchecked_mut(x) = k2.to_();
        }
    }
}
