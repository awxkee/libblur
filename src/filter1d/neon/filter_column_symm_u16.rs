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
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::neon::utils::{
    vfmlaq_symm_u16_f32, vmulq_u16_by_f32, vqmovnq_f32_u16, xvld1q_u16_x2, xvld1q_u16_x4,
    xvst1q_u16_x2, xvst1q_u16_x4,
};
use crate::filter1d::region::FilterRegion;
use crate::img_size::ImageSize;
use crate::mlaf::mlaf;
use crate::to_storage::ToStorage;
use std::arch::aarch64::*;
use std::ops::{Add, Mul};

pub(crate) fn filter_symm_column_neon_u16_f32(
    arena: Arena,
    brows: &[&[u16]],
    dst: &mut [u16],
    image_size: ImageSize,
    _: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    unsafe {
        let image_width = image_size.width * arena.components;

        let dst_stride = image_size.width * arena.components;

        let length = scanned_kernel.len();
        let half_len = length / 2;

        let ref0 = brows.get_unchecked(half_len);

        let mut cx = 0usize;

        let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(half_len).weight);

        while cx + 32 < image_width {
            let shifted_src = ref0.get_unchecked(cx..);

            let source = xvld1q_u16_x4(shifted_src.as_ptr());
            let mut k0 = vmulq_u16_by_f32(source.0, coeff);
            let mut k1 = vmulq_u16_by_f32(source.1, coeff);
            let mut k2 = vmulq_u16_by_f32(source.2, coeff);
            let mut k3 = vmulq_u16_by_f32(source.3, coeff);

            for i in 0..half_len {
                let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(i).weight);
                let rollback = length - i - 1;
                let v_source0 = xvld1q_u16_x4(brows.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1 =
                    xvld1q_u16_x4(brows.get_unchecked(rollback).get_unchecked(cx..).as_ptr());
                k0 = vfmlaq_symm_u16_f32(k0, v_source0.0, v_source1.0, coeff);
                k1 = vfmlaq_symm_u16_f32(k1, v_source0.1, v_source1.1, coeff);
                k2 = vfmlaq_symm_u16_f32(k2, v_source0.2, v_source1.2, coeff);
                k3 = vfmlaq_symm_u16_f32(k3, v_source0.3, v_source1.3, coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            xvst1q_u16_x4(
                dst_ptr0,
                uint16x8x4_t(
                    vqmovnq_f32_u16(k0),
                    vqmovnq_f32_u16(k1),
                    vqmovnq_f32_u16(k2),
                    vqmovnq_f32_u16(k3),
                ),
            );
            cx += 32;
        }

        while cx + 16 < image_width {
            let shifted_src = ref0.get_unchecked(cx..);

            let source = xvld1q_u16_x2(shifted_src.as_ptr());
            let mut k0 = vmulq_u16_by_f32(source.0, coeff);
            let mut k1 = vmulq_u16_by_f32(source.1, coeff);

            for i in 0..half_len {
                let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(i).weight);
                let rollback = length - i - 1;
                let v_source0 = xvld1q_u16_x2(brows.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1 =
                    xvld1q_u16_x2(brows.get_unchecked(rollback).get_unchecked(cx..).as_ptr());
                k0 = vfmlaq_symm_u16_f32(k0, v_source0.0, v_source1.0, coeff);
                k1 = vfmlaq_symm_u16_f32(k1, v_source0.1, v_source1.1, coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            xvst1q_u16_x2(
                dst_ptr0,
                uint16x8x2_t(vqmovnq_f32_u16(k0), vqmovnq_f32_u16(k1)),
            );
            cx += 16;
        }

        while cx + 8 < image_width {
            let shifted_src = ref0.get_unchecked(cx..);

            let source = vld1q_u16(shifted_src.as_ptr());
            let mut k0 = vmulq_u16_by_f32(source, coeff);

            for i in 0..half_len {
                let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(i).weight);
                let rollback = length - i - 1;
                let v_source0 = vld1q_u16(brows.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1 =
                    vld1q_u16(brows.get_unchecked(rollback).get_unchecked(cx..).as_ptr());
                k0 = vfmlaq_symm_u16_f32(k0, v_source0, v_source1, coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            vst1q_u16(dst_ptr0, vqmovnq_f32_u16(k0));
            cx += 8;
        }

        let coeff = scanned_kernel.get_unchecked(half_len).weight;

        while cx + 4 < image_width {
            let mut k0 = (*ref0.get_unchecked(cx) as f32).mul(coeff);
            let mut k1 = (*ref0.get_unchecked(cx + 1) as f32).mul(coeff);
            let mut k2 = (*ref0.get_unchecked(cx + 2) as f32).mul(coeff);
            let mut k3 = (*ref0.get_unchecked(cx + 3) as f32).mul(coeff);

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;
                k0 = mlaf(
                    k0,
                    ((*brows.get_unchecked(i).get_unchecked(cx)) as f32)
                        .add(*brows.get_unchecked(rollback).get_unchecked(cx) as f32),
                    coeff.weight,
                );
                k1 = mlaf(
                    k1,
                    ((*brows.get_unchecked(i).get_unchecked(cx + 1)) as f32)
                        .add((*brows.get_unchecked(rollback).get_unchecked(cx + 1)) as f32),
                    coeff.weight,
                );
                k2 = mlaf(
                    k2,
                    ((*brows.get_unchecked(i).get_unchecked(cx + 2)) as f32)
                        .add((*brows.get_unchecked(rollback).get_unchecked(cx + 2)) as f32),
                    coeff.weight,
                );
                k3 = mlaf(
                    k3,
                    ((*brows.get_unchecked(i).get_unchecked(cx + 3)) as f32)
                        .add((*brows.get_unchecked(rollback).get_unchecked(cx + 3)) as f32),
                    coeff.weight,
                );
            }

            *dst.get_unchecked_mut(cx) = k0.to_();
            *dst.get_unchecked_mut(cx + 1) = k1.to_();
            *dst.get_unchecked_mut(cx + 2) = k2.to_();
            *dst.get_unchecked_mut(cx + 3) = k3.to_();
            cx += 4;
        }

        for x in cx..dst_stride {
            let mut k0 = (*ref0.get_unchecked(x) as f32).mul(coeff);

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;
                k0 = mlaf(
                    k0,
                    ((*brows.get_unchecked(i).get_unchecked(x)) as f32)
                        .add(*brows.get_unchecked(rollback).get_unchecked(x) as f32),
                    coeff.weight,
                );
            }

            *dst.get_unchecked_mut(x) = k0.to_();
        }
    }
}
