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
    vfmlaq_symm_u16_f32, vmulq_u16_by_f32, vqmovnq_f32_u16, xvld1q_u16_x2, xvld1q_u16_x4,
    xvst1q_u16_x2, xvst1q_u16_x4,
};
use crate::filter1d::region::FilterRegion;
use crate::img_size::ImageSize;
use crate::mlaf::mlaf;
use crate::to_storage::ToStorage;
use std::arch::aarch64::*;

pub(crate) fn filter_row_symm_neon_u16_f32<const N: usize>(
    _: Arena,
    arena_src: &[u16],
    dst: &mut [u16],
    image_size: ImageSize,
    _: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    unsafe {
        let width = image_size.width;

        let src = arena_src;

        let length = scanned_kernel.len();
        let half_len = length / 2;

        let local_src = src;

        let mut cx = 0usize;

        let max_width = width * N;

        let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(half_len).weight);

        while cx + 32 < max_width {
            let shifted_src = local_src.get_unchecked(cx..);

            let source = xvld1q_u16_x4(shifted_src.get_unchecked(half_len * N..).as_ptr());
            let mut k0 = vmulq_u16_by_f32(source.0, coeff);
            let mut k1 = vmulq_u16_by_f32(source.1, coeff);
            let mut k2 = vmulq_u16_by_f32(source.2, coeff);
            let mut k3 = vmulq_u16_by_f32(source.3, coeff);

            for i in 0..half_len {
                let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(i).weight);
                let rollback = length - i - 1;
                let v_source0 = xvld1q_u16_x4(shifted_src.get_unchecked(i * N..).as_ptr());
                let v_source1 = xvld1q_u16_x4(shifted_src.get_unchecked(rollback * N..).as_ptr());
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

        while cx + 16 < max_width {
            let shifted_src = local_src.get_unchecked(cx..);

            let source = xvld1q_u16_x2(shifted_src.get_unchecked(half_len * N..).as_ptr());
            let mut k0 = vmulq_u16_by_f32(source.0, coeff);
            let mut k1 = vmulq_u16_by_f32(source.1, coeff);

            for i in 0..half_len {
                let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(i).weight);
                let rollback = length - i - 1;
                let v_source0 = xvld1q_u16_x2(shifted_src.get_unchecked(i * N..).as_ptr());
                let v_source1 = xvld1q_u16_x2(shifted_src.get_unchecked(rollback * N..).as_ptr());
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

        while cx + 8 < max_width {
            let shifted_src = local_src.get_unchecked(cx..);

            let source = vld1q_u16(shifted_src.get_unchecked(half_len * N..).as_ptr());
            let mut k0 = vmulq_u16_by_f32(source, coeff);

            for i in 0..half_len {
                let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(i).weight);
                let rollback = length - i - 1;
                let v_source0 = vld1q_u16(shifted_src.get_unchecked(i * N..).as_ptr());
                let v_source1 = vld1q_u16(shifted_src.get_unchecked(rollback * N..).as_ptr());
                k0 = vfmlaq_symm_u16_f32(k0, v_source0, v_source1, coeff);
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            vst1q_u16(dst_ptr0, vqmovnq_f32_u16(k0));
            cx += 8;
        }

        let coeff = *scanned_kernel.get_unchecked(half_len);

        while cx + 4 < max_width {
            let shifted_src = local_src.get_unchecked(cx..);
            let mut k0 = *shifted_src.get_unchecked(half_len * N) as f32 * coeff.weight;
            let mut k1 = *shifted_src.get_unchecked(half_len * N + 1) as f32 * coeff.weight;
            let mut k2 = *shifted_src.get_unchecked(half_len * N + 2) as f32 * coeff.weight;
            let mut k3 = *shifted_src.get_unchecked(half_len * N + 3) as f32 * coeff.weight;

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;

                k0 = mlaf(
                    k0,
                    *shifted_src.get_unchecked(i * N) as f32
                        + *shifted_src.get_unchecked(rollback * N) as f32,
                    coeff.weight,
                );

                k1 = mlaf(
                    k1,
                    *shifted_src.get_unchecked(i * N + 1) as f32
                        + *shifted_src.get_unchecked(rollback * N + 1) as f32,
                    coeff.weight,
                );

                k2 = mlaf(
                    k2,
                    *shifted_src.get_unchecked(i * N + 2) as f32
                        + *shifted_src.get_unchecked(rollback * N + 2) as f32,
                    coeff.weight,
                );
                k3 = mlaf(
                    k3,
                    *shifted_src.get_unchecked(i * N + 3) as f32
                        + *shifted_src.get_unchecked(rollback * N + 3) as f32,
                    coeff.weight,
                );
            }

            *dst.get_unchecked_mut(cx) = k0.to_();
            *dst.get_unchecked_mut(cx + 1) = k1.to_();
            *dst.get_unchecked_mut(cx + 2) = k2.to_();
            *dst.get_unchecked_mut(cx + 3) = k3.to_();
            cx += 4;
        }

        for x in cx..max_width {
            let shifted_src = local_src.get_unchecked(x..);
            let mut k0 = *shifted_src.get_unchecked(half_len * N) as f32 * coeff.weight;

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;

                k0 = mlaf(
                    k0,
                    *shifted_src.get_unchecked(i * N) as f32
                        + *shifted_src.get_unchecked(rollback * N) as f32,
                    coeff.weight,
                );
            }

            *dst.get_unchecked_mut(x) = k0.to_();
        }
    }
}
