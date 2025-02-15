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
    vfmla_u8_s16, vfmlaq_u8_s16, vmull_u8_by_i16, vmullq_u8_by_i16, vqmovn_s32_u8, vqmovnq_s32_u8,
};
use crate::filter1d::region::FilterRegion;
use crate::img_size::ImageSize;
use crate::unsafe_slice::UnsafeSlice;
use std::arch::aarch64::*;

pub fn filter_row_neon_u8_i32_app<const N: usize>(
    arena: Arena,
    arena_src: &[u8],
    dst: &UnsafeSlice<u8>,
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<i32>],
) {
    unsafe {
        let src = arena_src;

        let dst_stride = image_size.width * arena.components;

        let y = filter_region.start;
        let local_src = src;

        let max_width = image_size.width * N;

        let length = scanned_kernel.iter().len();

        let mut cx = 0usize;

        while cx + 64 < dst_stride {
            let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(0).weight as i16);

            let shifted_src = local_src.get_unchecked(cx..);

            let source = vld1q_u8_x4(shifted_src.as_ptr());
            let mut k0 = vmullq_u8_by_i16(source.0, coeff);
            let mut k1 = vmullq_u8_by_i16(source.1, coeff);
            let mut k2 = vmullq_u8_by_i16(source.2, coeff);
            let mut k3 = vmullq_u8_by_i16(source.3, coeff);

            for i in 1..length {
                let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight as i16);
                let v_source = vld1q_u8_x4(shifted_src.get_unchecked(i * N..).as_ptr());
                k0 = vfmlaq_u8_s16(k0, v_source.0, coeff);
                k1 = vfmlaq_u8_s16(k1, v_source.1, coeff);
                k2 = vfmlaq_u8_s16(k2, v_source.2, coeff);
                k3 = vfmlaq_u8_s16(k3, v_source.3, coeff);
            }

            let dst_offset = y * dst_stride + cx;
            let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
            vst1q_u8_x4(
                dst_ptr0,
                uint8x16x4_t(
                    vqmovnq_s32_u8(k0),
                    vqmovnq_s32_u8(k1),
                    vqmovnq_s32_u8(k2),
                    vqmovnq_s32_u8(k3),
                ),
            );
            cx += 64;
        }

        while cx + 32 < dst_stride {
            let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(0).weight as i16);

            let shifted_src = local_src.get_unchecked(cx..);

            let source = vld1q_u8_x2(shifted_src.as_ptr());
            let mut k0 = vmullq_u8_by_i16(source.0, coeff);
            let mut k1 = vmullq_u8_by_i16(source.1, coeff);

            for i in 1..length {
                let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight as i16);
                let v_source = vld1q_u8_x2(shifted_src.get_unchecked(i * N..).as_ptr());
                k0 = vfmlaq_u8_s16(k0, v_source.0, coeff);
                k1 = vfmlaq_u8_s16(k1, v_source.1, coeff);
            }

            let dst_offset = y * dst_stride + cx;
            let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
            vst1q_u8_x2(
                dst_ptr0,
                uint8x16x2_t(vqmovnq_s32_u8(k0), vqmovnq_s32_u8(k1)),
            );
            cx += 32;
        }

        while cx + 16 < dst_stride {
            let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(0).weight as i16);

            let shifted_src = local_src.get_unchecked(cx..);

            let source = vld1q_u8(shifted_src.as_ptr());
            let mut k0 = vmullq_u8_by_i16(source, coeff);

            for i in 1..length {
                let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight as i16);
                let v_source = vld1q_u8(shifted_src.get_unchecked(i * N..).as_ptr());
                k0 = vfmlaq_u8_s16(k0, v_source, coeff);
            }

            let dst_offset = y * dst_stride + cx;
            let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
            vst1q_u8(dst_ptr0, vqmovnq_s32_u8(k0));
            cx += 16;
        }

        while cx + 8 < dst_stride {
            let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(0).weight as i16);

            let shifted_src = local_src.get_unchecked(cx..);

            let source = vld1_u8(shifted_src.as_ptr());
            let mut k0 = vmull_u8_by_i16(source, vget_low_s16(coeff));

            for i in 1..length {
                let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight as i16);
                let v_source = vld1_u8(shifted_src.get_unchecked(i * N..).as_ptr());
                k0 = vfmla_u8_s16(k0, v_source, vget_low_s16(coeff));
            }

            let dst_offset = y * dst_stride + cx;
            let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
            vst1_u8(dst_ptr0, vqmovn_s32_u8(k0));
            cx += 8;
        }

        const K_PRECISION: i32 = 15;
        const RND: i32 = 1 << (K_PRECISION - 1);

        while cx + 4 < max_width {
            let coeff = *scanned_kernel.get_unchecked(0);
            let shifted_src = local_src.get_unchecked(cx..);
            let mut k0 = RND + *shifted_src.get_unchecked(0) as i32 * coeff.weight;
            let mut k1 = RND + *shifted_src.get_unchecked(1) as i32 * coeff.weight;
            let mut k2 = RND + *shifted_src.get_unchecked(2) as i32 * coeff.weight;
            let mut k3 = RND + *shifted_src.get_unchecked(3) as i32 * coeff.weight;

            for i in 1..length {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;

                k0 += *shifted_src.get_unchecked(i * N) as i32 * coeff.weight;

                k1 += *shifted_src.get_unchecked(i * N + 1) as i32 * coeff.weight;

                k2 += *shifted_src.get_unchecked(i * N + 2) as i32 * coeff.weight;

                k3 += *shifted_src.get_unchecked(rollback * N + 3) as i32 * coeff.weight;
            }

            dst.write(
                y * dst_stride + cx,
                (k0 >> K_PRECISION).max(0).min(255) as u8,
            );
            dst.write(
                y * dst_stride + cx + 1,
                (k1 >> K_PRECISION).max(0).min(255) as u8,
            );
            dst.write(
                y * dst_stride + cx + 2,
                (k2 >> K_PRECISION).max(0).min(255) as u8,
            );
            dst.write(
                y * dst_stride + cx + 3,
                (k3 >> K_PRECISION).max(0).min(255) as u8,
            );

            cx += 4;
        }

        for x in cx..max_width {
            let coeff = *scanned_kernel.get_unchecked(0);
            let shifted_src = local_src.get_unchecked(x..);
            let mut k0 = RND + *shifted_src.get_unchecked(0) as i32 * coeff.weight;

            for i in 1..length {
                let coeff = *scanned_kernel.get_unchecked(i);

                k0 += (*shifted_src.get_unchecked(i * N) as i32) * coeff.weight;
            }

            dst.write(
                y * dst_stride + x,
                (k0 >> K_PRECISION).max(0).min(255) as u8,
            );
        }
    }
}
