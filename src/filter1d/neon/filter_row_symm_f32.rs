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
use crate::filter1d::neon::utils::{xvld1q_f32_x2, xvld1q_f32_x4, xvst1q_f32_x2, xvst1q_f32_x4};
use crate::filter1d::region::FilterRegion;
use crate::img_size::ImageSize;
use crate::mlaf::mlaf;
use crate::neon::p_vfmaq_f32;
use crate::to_storage::ToStorage;
use std::arch::aarch64::*;

pub(crate) fn filter_row_neon_symm_f32_f32<const N: usize>(
    _: Arena,
    arena_src: &[f32],
    dst: &mut [f32],
    image_size: ImageSize,
    _: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    unsafe {
        let src = arena_src;
        let local_src = src;

        let length = scanned_kernel.len();
        let half_len = length / 2;

        let mut cx = 0usize;

        let max_width = image_size.width * N;

        while cx + 16 < max_width {
            let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(half_len).weight);

            let shifted_src = local_src.get_unchecked(cx..);

            let source = xvld1q_f32_x4(shifted_src.get_unchecked(half_len * N..).as_ptr());
            let mut k0 = vmulq_f32(source.0, coeff);
            let mut k1 = vmulq_f32(source.1, coeff);
            let mut k2 = vmulq_f32(source.2, coeff);
            let mut k3 = vmulq_f32(source.3, coeff);

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(i).weight);
                let v_source0 = xvld1q_f32_x4(shifted_src.get_unchecked(i * N..).as_ptr());
                let v_source1 = xvld1q_f32_x4(shifted_src.get_unchecked(rollback * N..).as_ptr());
                k0 = p_vfmaq_f32(k0, vaddq_f32(v_source0.0, v_source1.0), coeff);
                k1 = p_vfmaq_f32(k1, vaddq_f32(v_source0.1, v_source1.1), coeff);
                k2 = p_vfmaq_f32(k2, vaddq_f32(v_source0.2, v_source1.2), coeff);
                k3 = p_vfmaq_f32(k3, vaddq_f32(v_source0.3, v_source1.3), coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            xvst1q_f32_x4(dst_ptr0, float32x4x4_t(k0, k1, k2, k3));
            cx += 16;
        }

        while cx + 8 < max_width {
            let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(half_len).weight);

            let shifted_src = local_src.get_unchecked(cx..);

            let source = xvld1q_f32_x2(shifted_src.get_unchecked(half_len * N..).as_ptr());
            let mut k0 = vmulq_f32(source.0, coeff);
            let mut k1 = vmulq_f32(source.1, coeff);

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(i).weight);
                let v_source0 = xvld1q_f32_x2(shifted_src.get_unchecked((i * N)..).as_ptr());
                let v_source1 = xvld1q_f32_x2(shifted_src.get_unchecked((rollback * N)..).as_ptr());
                k0 = p_vfmaq_f32(k0, vaddq_f32(v_source0.0, v_source1.0), coeff);
                k1 = p_vfmaq_f32(k1, vaddq_f32(v_source0.1, v_source1.1), coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            xvst1q_f32_x2(dst_ptr0, float32x4x2_t(k0, k1));
            cx += 8;
        }

        while cx + 4 < max_width {
            let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(half_len).weight);

            let shifted_src = local_src.get_unchecked(cx..);

            let source = vld1q_f32(shifted_src.get_unchecked(half_len * N..).as_ptr());
            let mut k0 = vmulq_f32(source, coeff);

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(i).weight);
                let v_source0 = vld1q_f32(shifted_src.get_unchecked((i * N)..).as_ptr());
                let v_source1 = vld1q_f32(shifted_src.get_unchecked((rollback * N)..).as_ptr());
                k0 = p_vfmaq_f32(k0, vaddq_f32(v_source0, v_source1), coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            vst1q_f32(dst_ptr0, k0);
            cx += 4;
        }

        for x in cx..max_width {
            let coeff = *scanned_kernel.get_unchecked(half_len);
            let shifted_src = local_src.get_unchecked(x..);
            let mut k0 = shifted_src.get_unchecked(half_len * N) * coeff.weight;

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = *scanned_kernel.get_unchecked(i);

                k0 = mlaf(
                    k0,
                    (*shifted_src.get_unchecked(i * N))
                        + (*shifted_src.get_unchecked(rollback * N)),
                    coeff.weight,
                );
            }

            *dst.get_unchecked_mut(x) = k0.to_();
        }
    }
}
