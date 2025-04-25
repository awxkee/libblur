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
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::neon::utils::{
    vmla_symm_hi_u8_s16, vmlaq_symm_hi_u8_s16, vmull_expand_i16, vmullq_expand_i16,
    vqmovnq_s16x2_u8, xvld1q_u8_x2, xvld1q_u8_x3, xvld1q_u8_x4, xvld4u8, xvst1q_u8_x2,
    xvst1q_u8_x3, xvst1q_u8_x4, xvst_u8x4_q15,
};
use crate::filter1d::region::FilterRegion;
use crate::filter1d::to_approx_storage::ToApproxStorage;
use crate::img_size::ImageSize;
use std::arch::aarch64::*;
use std::ops::{Add, Mul};

pub(crate) fn filter_column_symm_neon_u8_i32_rdm(
    arena: Arena,
    arena_src: &[&[u8]],
    dst: &mut [u8],
    image_size: ImageSize,
    _: FilterRegion,
    scanned_kernel: &[ScanPoint1d<i32>],
) {
    unsafe {
        let image_width = image_size.width * arena.components;

        let length = scanned_kernel.len();
        let half_len = length / 2;

        let mut cx = 0usize;

        while cx + 64 < image_width {
            let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(half_len).weight as i16);

            let v_src = arena_src.get_unchecked(half_len).get_unchecked(cx..);

            let source = xvld1q_u8_x4(v_src.as_ptr());
            let mut k0 = vmullq_expand_i16(source.0, coeff);
            let mut k1 = vmullq_expand_i16(source.1, coeff);
            let mut k2 = vmullq_expand_i16(source.2, coeff);
            let mut k3 = vmullq_expand_i16(source.3, coeff);

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight as i16);
                let v_source0 =
                    xvld1q_u8_x4(arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1 = xvld1q_u8_x4(
                    arena_src
                        .get_unchecked(rollback)
                        .get_unchecked(cx..)
                        .as_ptr(),
                );
                k0 = vmlaq_symm_hi_u8_s16(k0, v_source0.0, v_source1.0, coeff);
                k1 = vmlaq_symm_hi_u8_s16(k1, v_source0.1, v_source1.1, coeff);
                k2 = vmlaq_symm_hi_u8_s16(k2, v_source0.2, v_source1.2, coeff);
                k3 = vmlaq_symm_hi_u8_s16(k3, v_source0.3, v_source1.3, coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            xvst1q_u8_x4(
                dst_ptr0,
                uint8x16x4_t(
                    vqmovnq_s16x2_u8(k0),
                    vqmovnq_s16x2_u8(k1),
                    vqmovnq_s16x2_u8(k2),
                    vqmovnq_s16x2_u8(k3),
                ),
            );
            cx += 64;
        }

        while cx + 48 < image_width {
            let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(half_len).weight as i16);

            let v_src = arena_src.get_unchecked(half_len).get_unchecked(cx..);

            let source = xvld1q_u8_x3(v_src.as_ptr());
            let mut k0 = vmullq_expand_i16(source.0, coeff);
            let mut k1 = vmullq_expand_i16(source.1, coeff);
            let mut k2 = vmullq_expand_i16(source.2, coeff);

            for i in 0..half_len {
                let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight as i16);
                let rollback = length - i - 1;
                let v_source0 =
                    xvld1q_u8_x3(arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1 = xvld1q_u8_x3(
                    arena_src
                        .get_unchecked(rollback)
                        .get_unchecked(cx..)
                        .as_ptr(),
                );
                k0 = vmlaq_symm_hi_u8_s16(k0, v_source0.0, v_source1.0, coeff);
                k1 = vmlaq_symm_hi_u8_s16(k1, v_source0.1, v_source1.1, coeff);
                k2 = vmlaq_symm_hi_u8_s16(k2, v_source0.2, v_source1.2, coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            xvst1q_u8_x3(
                dst_ptr0,
                uint8x16x3_t(
                    vqmovnq_s16x2_u8(k0),
                    vqmovnq_s16x2_u8(k1),
                    vqmovnq_s16x2_u8(k2),
                ),
            );
            cx += 48;
        }

        while cx + 32 < image_width {
            let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(half_len).weight as i16);

            let v_src = arena_src.get_unchecked(half_len).get_unchecked(cx..);

            let source = xvld1q_u8_x2(v_src.as_ptr());
            let mut k0 = vmullq_expand_i16(source.0, coeff);
            let mut k1 = vmullq_expand_i16(source.1, coeff);

            for i in 0..half_len {
                let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight as i16);
                let rollback = length - i - 1;
                let v_source0 =
                    xvld1q_u8_x2(arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1 = xvld1q_u8_x2(
                    arena_src
                        .get_unchecked(rollback)
                        .get_unchecked(cx..)
                        .as_ptr(),
                );
                k0 = vmlaq_symm_hi_u8_s16(k0, v_source0.0, v_source1.0, coeff);
                k1 = vmlaq_symm_hi_u8_s16(k1, v_source0.1, v_source1.1, coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            xvst1q_u8_x2(
                dst_ptr0,
                uint8x16x2_t(vqmovnq_s16x2_u8(k0), vqmovnq_s16x2_u8(k1)),
            );
            cx += 32;
        }

        while cx + 16 < image_width {
            let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(half_len).weight as i16);

            let v_src = arena_src.get_unchecked(half_len).get_unchecked(cx..);

            let source = vld1q_u8(v_src.as_ptr());
            let mut k0 = vmullq_expand_i16(source, coeff);

            for i in 0..half_len {
                let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight as i16);
                let rollback = length - i - 1;
                let v_source0 = vld1q_u8(arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1 = vld1q_u8(
                    arena_src
                        .get_unchecked(rollback)
                        .get_unchecked(cx..)
                        .as_ptr(),
                );
                k0 = vmlaq_symm_hi_u8_s16(k0, v_source0, v_source1, coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            vst1q_u8(dst_ptr0, vqmovnq_s16x2_u8(k0));
            cx += 16;
        }

        while cx + 4 < image_width {
            let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(half_len).weight as i16);

            let v_src = arena_src.get_unchecked(half_len).get_unchecked(cx..);

            let source = xvld4u8(v_src.as_ptr());
            let mut k0 = vmull_expand_i16(source, coeff);

            for i in 0..half_len {
                let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight as i16);
                let rollback = length - i - 1;
                let v_source0 = xvld4u8(arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1 = xvld4u8(
                    arena_src
                        .get_unchecked(rollback)
                        .get_unchecked(cx..)
                        .as_ptr(),
                );
                k0 = vmla_symm_hi_u8_s16(k0, v_source0, v_source1, coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            xvst_u8x4_q15(dst_ptr0, k0);
            cx += 4;
        }

        for x in cx..image_width {
            let coeff = *scanned_kernel.get_unchecked(0);

            let v_src = arena_src.get_unchecked(half_len).get_unchecked(x..);

            let mut k0 = ((*v_src.get_unchecked(0)) as i32).mul(coeff.weight);

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;
                k0 = ((*arena_src.get_unchecked(i).get_unchecked(x)) as i32)
                    .add((*arena_src.get_unchecked(rollback).get_unchecked(x)) as i32)
                    .mul(coeff.weight)
                    .add(k0);
            }

            *dst.get_unchecked_mut(x) = k0.to_approx_();
        }
    }
}
