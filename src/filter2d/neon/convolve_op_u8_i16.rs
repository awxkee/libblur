/*
 * Copyright (c) Radzivon Bartoshyk. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2.  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3.  Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use crate::filter1d::neon::utils::{
    vdotq_exact_s16, vmulq_u8_by_i16, xvld1q_u8_x2, xvld1q_u8_x4, xvst1q_u8_x2, xvst1q_u8_x4,
};
use crate::filter1d::Arena;
use crate::filter2d::scan_point_2d::ScanPoint2d;
use crate::to_storage::ToStorage;
use crate::ImageSize;
use num_traits::MulAdd;
use std::arch::aarch64::*;
use std::ops::Mul;

pub(crate) fn convolve_segment_neon_2d_u8_i16(
    arena: Arena,
    arena_source: &[u8],
    dst: &mut [u8],
    image_size: ImageSize,
    prepared_kernel: &[ScanPoint2d<i16>],
    y: usize,
) {
    unsafe {
        let dx = arena.pad_w as i64;
        let dy = arena.pad_h as i64;

        let arena_stride = arena.width * arena.components;

        let offsets = prepared_kernel
            .iter()
            .map(|&x| {
                arena_source.get_unchecked(
                    ((x.y + dy + y as i64) as usize * arena_stride
                        + (x.x + dx) as usize * arena.components)..,
                )
            })
            .collect::<Vec<_>>();

        let length = prepared_kernel.len();
        let total_width = image_size.width * arena.components;

        let mut cx = 0usize;

        let k_weight = vdupq_n_s16(prepared_kernel.get_unchecked(0).weight);

        let off0 = offsets.get_unchecked(0);

        while cx + 64 < total_width {
            let items0 = xvld1q_u8_x4(off0.get_unchecked(cx..).as_ptr());
            let mut k0 = vmulq_u8_by_i16(items0.0, k_weight);
            let mut k1 = vmulq_u8_by_i16(items0.1, k_weight);
            let mut k2 = vmulq_u8_by_i16(items0.2, k_weight);
            let mut k3 = vmulq_u8_by_i16(items0.3, k_weight);
            for i in 1..length {
                let weight = vdupq_n_s16(prepared_kernel.get_unchecked(i).weight);
                let s_ptr = offsets.get_unchecked(i);
                let items0 = xvld1q_u8_x4(s_ptr.get_unchecked(cx..).as_ptr());
                k0 = vdotq_exact_s16(k0, items0.0, weight);
                k1 = vdotq_exact_s16(k1, items0.1, weight);
                k2 = vdotq_exact_s16(k2, items0.2, weight);
                k3 = vdotq_exact_s16(k3, items0.3, weight);
            }
            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            xvst1q_u8_x4(
                dst_ptr0,
                uint8x16x4_t(
                    vcombine_u8(vqmovun_s16(k0.0), vqmovun_s16(k0.1)),
                    vcombine_u8(vqmovun_s16(k1.0), vqmovun_s16(k1.1)),
                    vcombine_u8(vqmovun_s16(k2.0), vqmovun_s16(k2.1)),
                    vcombine_u8(vqmovun_s16(k3.0), vqmovun_s16(k3.1)),
                ),
            );
            cx += 64;
        }

        while cx + 32 < total_width {
            let items0 = xvld1q_u8_x2(off0.get_unchecked(cx..).as_ptr());
            let mut k0 = vmulq_u8_by_i16(items0.0, k_weight);
            let mut k1 = vmulq_u8_by_i16(items0.1, k_weight);
            for i in 1..length {
                let weight = vdupq_n_s16(prepared_kernel.get_unchecked(i).weight);
                let s_ptr = offsets.get_unchecked(i);
                let items0 = xvld1q_u8_x2(s_ptr.get_unchecked(cx..).as_ptr());
                k0 = vdotq_exact_s16(k0, items0.0, weight);
                k1 = vdotq_exact_s16(k1, items0.1, weight);
            }
            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            xvst1q_u8_x2(
                dst_ptr0,
                uint8x16x2_t(
                    vcombine_u8(vqmovun_s16(k0.0), vqmovun_s16(k0.1)),
                    vcombine_u8(vqmovun_s16(k1.0), vqmovun_s16(k1.1)),
                ),
            );
            cx += 32;
        }

        while cx + 16 < total_width {
            let items0 = vld1q_u8(off0.get_unchecked(cx..).as_ptr());
            let mut k0 = vmulq_u8_by_i16(items0, k_weight);
            for i in 1..length {
                let weight = vdupq_n_s16(prepared_kernel.get_unchecked(i).weight);
                let items0 = vld1q_u8(offsets.get_unchecked(i).get_unchecked(cx..).as_ptr());
                k0 = vdotq_exact_s16(k0, items0, weight);
            }
            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            vst1q_u8(dst_ptr0, vcombine_u8(vqmovun_s16(k0.0), vqmovun_s16(k0.1)));
            cx += 16;
        }

        let k_weight = prepared_kernel.get_unchecked(0).weight;

        while cx + 4 < total_width {
            let mut k0 = ((*off0.get_unchecked(cx)) as i16).mul(k_weight);
            let mut k1 = ((*off0.get_unchecked(cx + 1)) as i16).mul(k_weight);
            let mut k2 = ((*off0.get_unchecked(cx + 2)) as i16).mul(k_weight);
            let mut k3 = ((*off0.get_unchecked(cx + 3)) as i16).mul(k_weight);

            for i in 1..length {
                let weight = prepared_kernel.get_unchecked(i).weight;
                k0 = ((*offsets.get_unchecked(i).get_unchecked(cx)) as i16).mul_add(weight, k0);
                k1 = ((*offsets.get_unchecked(i).get_unchecked(cx + 1)) as i16).mul_add(weight, k1);
                k2 = ((*offsets.get_unchecked(i).get_unchecked(cx + 2)) as i16).mul_add(weight, k2);
                k3 = ((*offsets.get_unchecked(i).get_unchecked(cx + 3)) as i16).mul_add(weight, k3);
            }

            *dst.get_unchecked_mut(cx) = k0.to_();
            *dst.get_unchecked_mut(cx + 1) = k1.to_();
            *dst.get_unchecked_mut(cx + 2) = k2.to_();
            *dst.get_unchecked_mut(cx + 3) = k3.to_();
            cx += 4;
        }

        for x in cx..total_width {
            let mut k0 = ((*(*off0).get_unchecked(x)) as i16).mul(k_weight);

            for i in 1..length {
                let k_weight = prepared_kernel.get_unchecked(i).weight;
                k0 = ((*offsets.get_unchecked(i).get_unchecked(x)) as i16).mul_add(k_weight, k0);
            }
            *dst.get_unchecked_mut(x) = k0.to_();
        }
    }
}
