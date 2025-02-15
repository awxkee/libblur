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
    vdotq_exact_s16, vmulq_u8_by_i16, xvld1q_u8_x3, xvld1q_u8_x4, xvst1q_u8_x3, xvst1q_u8_x4,
};
use crate::filter1d::region::FilterRegion;
use crate::img_size::ImageSize;
use crate::unsafe_slice::UnsafeSlice;
use std::arch::aarch64::*;

pub fn filter_rgb_row_neon_u8_i16<const N: usize>(
    arena: Arena,
    arena_src: &[u8],
    dst: &UnsafeSlice<u8>,
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<i16>],
) {
    unsafe {
        let width = image_size.width;

        let src = arena_src;

        let dst_stride = image_size.width * arena.components;

        let length = scanned_kernel.len();

        let y = filter_region.start;
        let local_src = src;

        let mut cx = 0usize;

        let max_width = width * N;

        while cx + 64 < max_width {
            let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(0).weight);

            let shifted_src = local_src.get_unchecked(cx..);

            let source = xvld1q_u8_x4(shifted_src.as_ptr());
            let mut k0 = vmulq_u8_by_i16(source.0, coeff);
            let mut k1 = vmulq_u8_by_i16(source.1, coeff);
            let mut k2 = vmulq_u8_by_i16(source.2, coeff);
            let mut k3 = vmulq_u8_by_i16(source.3, coeff);

            for i in 1..length {
                let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight);
                let v_source = xvld1q_u8_x4(shifted_src.get_unchecked((i * N)..).as_ptr());
                k0 = vdotq_exact_s16(k0, v_source.0, coeff);
                k1 = vdotq_exact_s16(k1, v_source.1, coeff);
                k2 = vdotq_exact_s16(k2, v_source.2, coeff);
                k3 = vdotq_exact_s16(k3, v_source.3, coeff);
            }

            let dst_offset = y * dst_stride + cx;
            let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
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

        while cx + 48 < max_width {
            let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(0).weight);

            let shifted_src = local_src.get_unchecked(cx..);

            let source = xvld1q_u8_x3(shifted_src.as_ptr());
            let mut k0 = vmulq_u8_by_i16(source.0, coeff);
            let mut k1 = vmulq_u8_by_i16(source.1, coeff);
            let mut k2 = vmulq_u8_by_i16(source.2, coeff);

            for i in 1..length {
                let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight);
                let v_source = xvld1q_u8_x3(shifted_src.get_unchecked((i * N)..).as_ptr());
                k0 = vdotq_exact_s16(k0, v_source.0, coeff);
                k1 = vdotq_exact_s16(k1, v_source.1, coeff);
                k2 = vdotq_exact_s16(k2, v_source.2, coeff);
            }

            let dst_offset = y * dst_stride + cx;
            let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
            xvst1q_u8_x3(
                dst_ptr0,
                uint8x16x3_t(
                    vcombine_u8(vqmovun_s16(k0.0), vqmovun_s16(k0.1)),
                    vcombine_u8(vqmovun_s16(k1.0), vqmovun_s16(k1.1)),
                    vcombine_u8(vqmovun_s16(k2.0), vqmovun_s16(k2.1)),
                ),
            );
            cx += 48;
        }

        while cx + 16 < max_width {
            let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(0).weight);

            let shifted_src = local_src.get_unchecked(cx..);

            let source = vld1q_u8(shifted_src.as_ptr());
            let mut k0 = vmulq_u8_by_i16(source, coeff);

            for i in 1..length {
                let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight);
                let v_source = vld1q_u8(shifted_src.get_unchecked((i * N)..).as_ptr());
                k0 = vdotq_exact_s16(k0, v_source, coeff);
            }

            let dst_offset = y * dst_stride + cx;
            let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
            vst1q_u8(dst_ptr0, vcombine_u8(vqmovun_s16(k0.0), vqmovun_s16(k0.1)));
            cx += 16;
        }

        while cx + 4 < width {
            let coeff = *scanned_kernel.get_unchecked(0);
            let shifted_src = local_src.get_unchecked(cx..);
            let mut k0 = *shifted_src.get_unchecked(0) as i16 * coeff.weight;
            let mut k1 = *shifted_src.get_unchecked(1) as i16 * coeff.weight;
            let mut k2 = *shifted_src.get_unchecked(2) as i16 * coeff.weight;
            let mut k3 = *shifted_src.get_unchecked(3) as i16 * coeff.weight;

            for i in 1..length {
                let coeff = *scanned_kernel.get_unchecked(i);
                k0 += *shifted_src.get_unchecked(i * N) as i16 * coeff.weight;
                k1 += *shifted_src.get_unchecked(i * N + 1) as i16 * coeff.weight;
                k2 += *shifted_src.get_unchecked(i * N + 2) as i16 * coeff.weight;
                k3 += *shifted_src.get_unchecked(i * N + 3) as i16 * coeff.weight;
            }

            dst.write(y * dst_stride + cx, k0.max(0).min(255) as u8);
            dst.write(y * dst_stride + cx + 1, k1.max(0).min(255) as u8);
            dst.write(y * dst_stride + cx + 2, k2.max(0).min(255) as u8);
            dst.write(y * dst_stride + cx + 3, k3.max(0).min(255) as u8);
            cx += 4;
        }

        for x in cx..max_width {
            let coeff = *scanned_kernel.get_unchecked(0);
            let shifted_src = local_src.get_unchecked(x..);
            let mut k0 = *shifted_src.get_unchecked(0) as i16 * coeff.weight;

            for i in 1..length {
                let coeff = *scanned_kernel.get_unchecked(i);
                k0 += *shifted_src.get_unchecked(i * N) as i16 * coeff.weight;
            }

            dst.write(y * dst_stride + x, k0.max(0).min(255) as u8);
        }
    }
}
