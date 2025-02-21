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
    vmla_symm_hi_u8_s16, vmlaq_symm_hi_u8_s16, vmull_expand_i16, vmullq_expand_i16, vqmovn_s16_u8,
    vqmovnq_s16x2_u8, xvld1q_u8_x3, xvld1q_u8_x4, xvst1q_u8_x3, xvst1q_u8_x4,
};
use crate::filter1d::region::FilterRegion;
use crate::filter1d::to_approx_storage::ToApproxStorage;
use crate::img_size::ImageSize;
use std::arch::aarch64::*;

pub(crate) fn filter_row_symm_neon_u8_i32_rdm<const N: usize>(
    _: Arena,
    arena_src: &[u8],
    dst: &mut [u8],
    image_size: ImageSize,
    _: FilterRegion,
    scanned_kernel: &[ScanPoint1d<i32>],
) {
    unsafe {
        let width = image_size.width;

        let src = arena_src;

        let length = scanned_kernel.len();
        let half_len = length / 2;

        let local_src = src;

        let mut cx = 0usize;

        const EXPAND: i32 = 6;
        const PRECISION: i32 = 6;

        let max_width = width * N;

        while cx + 64 < max_width {
            let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(0).weight as i16);

            let shifted_src = local_src.get_unchecked(cx..);

            let source = xvld1q_u8_x4(shifted_src.get_unchecked(half_len..).as_ptr());
            let mut k0 = vmullq_expand_i16::<EXPAND>(source.0, coeff);
            let mut k1 = vmullq_expand_i16::<EXPAND>(source.1, coeff);
            let mut k2 = vmullq_expand_i16::<EXPAND>(source.2, coeff);
            let mut k3 = vmullq_expand_i16::<EXPAND>(source.3, coeff);

            for i in 0..half_len {
                let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight as i16);
                let rollback = length - i - 1;
                let v_source0 = xvld1q_u8_x4(shifted_src.get_unchecked((i * N)..).as_ptr());
                let v_source1 = xvld1q_u8_x4(shifted_src.get_unchecked((rollback * N)..).as_ptr());
                k0 = vmlaq_symm_hi_u8_s16::<EXPAND>(k0, v_source0.0, v_source1.0, coeff);
                k1 = vmlaq_symm_hi_u8_s16::<EXPAND>(k1, v_source0.1, v_source1.1, coeff);
                k2 = vmlaq_symm_hi_u8_s16::<EXPAND>(k2, v_source0.2, v_source1.2, coeff);
                k3 = vmlaq_symm_hi_u8_s16::<EXPAND>(k3, v_source0.3, v_source1.3, coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            xvst1q_u8_x4(
                dst_ptr0,
                uint8x16x4_t(
                    vqmovnq_s16x2_u8::<PRECISION>(k0),
                    vqmovnq_s16x2_u8::<PRECISION>(k1),
                    vqmovnq_s16x2_u8::<PRECISION>(k2),
                    vqmovnq_s16x2_u8::<PRECISION>(k3),
                ),
            );
            cx += 64;
        }

        while cx + 48 < max_width {
            let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(0).weight as i16);

            let shifted_src = local_src.get_unchecked(cx..);

            let source = xvld1q_u8_x3(shifted_src.get_unchecked(half_len..).as_ptr());
            let mut k0 = vmullq_expand_i16::<EXPAND>(source.0, coeff);
            let mut k1 = vmullq_expand_i16::<EXPAND>(source.1, coeff);
            let mut k2 = vmullq_expand_i16::<EXPAND>(source.2, coeff);

            for i in 0..half_len {
                let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight as i16);
                let rollback = length - i - 1;
                let v_source0 = xvld1q_u8_x3(shifted_src.get_unchecked((i * N)..).as_ptr());
                let v_source1 = xvld1q_u8_x3(shifted_src.get_unchecked((rollback * N)..).as_ptr());
                k0 = vmlaq_symm_hi_u8_s16::<EXPAND>(k0, v_source0.0, v_source1.0, coeff);
                k1 = vmlaq_symm_hi_u8_s16::<EXPAND>(k1, v_source0.1, v_source1.1, coeff);
                k2 = vmlaq_symm_hi_u8_s16::<EXPAND>(k2, v_source0.2, v_source1.2, coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            xvst1q_u8_x3(
                dst_ptr0,
                uint8x16x3_t(
                    vqmovnq_s16x2_u8::<PRECISION>(k0),
                    vqmovnq_s16x2_u8::<PRECISION>(k1),
                    vqmovnq_s16x2_u8::<PRECISION>(k2),
                ),
            );
            cx += 48;
        }

        while cx + 16 < max_width {
            let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(0).weight as i16);

            let shifted_src = local_src.get_unchecked(cx..);

            let source = vld1q_u8(shifted_src.get_unchecked(half_len..).as_ptr());
            let mut k0 = vmullq_expand_i16::<EXPAND>(source, coeff);

            for i in 0..half_len {
                let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight as i16);
                let rollback = length - i - 1;
                let v_source0 = vld1q_u8(shifted_src.get_unchecked((i * N)..).as_ptr());
                let v_source1 = vld1q_u8(shifted_src.get_unchecked((rollback * N)..).as_ptr());
                k0 = vmlaq_symm_hi_u8_s16::<EXPAND>(k0, v_source0, v_source1, coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            vst1q_u8(dst_ptr0, vqmovnq_s16x2_u8::<PRECISION>(k0));
            cx += 16;
        }

        while cx + 8 < max_width {
            let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(0).weight as i16);

            let shifted_src = local_src.get_unchecked(cx..);

            let source = vld1_u8(shifted_src.get_unchecked(half_len..).as_ptr());
            let mut k0 = vmull_expand_i16::<EXPAND>(source, coeff);

            for i in 0..half_len {
                let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight as i16);
                let rollback = length - i - 1;
                let v_source0 = vld1_u8(shifted_src.get_unchecked((i * N)..).as_ptr());
                let v_source1 = vld1_u8(shifted_src.get_unchecked((rollback * N)..).as_ptr());
                k0 = vmla_symm_hi_u8_s16::<EXPAND>(k0, v_source0, v_source1, coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            vst1_u8(dst_ptr0, vqmovn_s16_u8::<PRECISION>(k0));
            cx += 8;
        }

        const K_PRECISION: i32 = 15;
        const RND: i32 = 1 << (K_PRECISION - 1);

        while cx + 4 < max_width {
            let coeff = *scanned_kernel.get_unchecked(half_len);
            let shifted_src = local_src.get_unchecked(cx..);
            let mut k0 = RND + *shifted_src.get_unchecked(0) as i32 * coeff.weight;
            let mut k1 = RND + *shifted_src.get_unchecked(1) as i32 * coeff.weight;
            let mut k2 = RND + *shifted_src.get_unchecked(2) as i32 * coeff.weight;
            let mut k3 = RND + *shifted_src.get_unchecked(3) as i32 * coeff.weight;

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;

                k0 += (*shifted_src.get_unchecked(i * N) as i16
                    + *shifted_src.get_unchecked(rollback * N) as i16) as i32
                    * coeff.weight;

                k1 += (*shifted_src.get_unchecked(i * N + 1) as i16
                    + *shifted_src.get_unchecked(rollback * N + 1) as i16)
                    as i32
                    * coeff.weight;

                k2 += (*shifted_src.get_unchecked(i * N + 2) as i16
                    + *shifted_src.get_unchecked(rollback * N + 2) as i16)
                    as i32
                    * coeff.weight;

                k3 += (*shifted_src.get_unchecked(i * N + 3) as i16
                    + *shifted_src.get_unchecked(rollback * N + 3) as i16)
                    as i32
                    * coeff.weight;
            }

            *dst.get_unchecked_mut(cx) = k0.to_approx_();
            *dst.get_unchecked_mut(cx + 1) = k1.to_approx_();
            *dst.get_unchecked_mut(cx + 2) = k2.to_approx_();
            *dst.get_unchecked_mut(cx + 3) = k3.to_approx_();
            cx += 4;
        }

        for x in cx..max_width {
            let coeff = *scanned_kernel.get_unchecked(half_len);
            let shifted_src = local_src.get_unchecked(x..);
            let mut k0 = RND + *shifted_src.get_unchecked(0) as i32 * coeff.weight;

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;

                k0 += (*shifted_src.get_unchecked(i * N) as i16
                    + *shifted_src.get_unchecked(rollback * N) as i16) as i32
                    * coeff.weight;
            }

            *dst.get_unchecked_mut(x) = k0.to_approx_();
        }
    }
}
