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
use crate::filter1d::color_group::ColorGroup;
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::neon::utils::{
    vfmla_u8_s16, vfmlaq_u8_s16, vmul_u8_by_i16, vmulq_u8_by_i16, vqmovn_s32_u8, vqmovnq_s32_u8,
};
use crate::filter1d::region::FilterRegion;
use crate::img_size::ImageSize;
use crate::unsafe_slice::UnsafeSlice;
use std::arch::aarch64::*;
use std::ops::{Add, Mul};

pub fn filter_rgb_row_neon_u8_i32(
    arena: Arena,
    arena_src: &[u8],
    dst: &UnsafeSlice<u8>,
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<i32>],
) {
    unsafe {
        let width = image_size.width;

        const N: usize = 3;

        let src = &arena_src;

        let dst_stride = image_size.width * arena.components;

        let arena_width = arena.width * N;

        for y in filter_region.start..filter_region.end {
            let local_src = src.get_unchecked((y * arena_width)..);

            let length = scanned_kernel.iter().len();

            let mut _cx = 0usize;

            while _cx + 16 < width {
                let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(0).weight as i16);

                let shifted_src = local_src.get_unchecked((_cx * N)..);

                let source = vld3q_u8(shifted_src.as_ptr());
                let mut k0 = vmulq_u8_by_i16(source.0, coeff);
                let mut k1 = vmulq_u8_by_i16(source.1, coeff);
                let mut k2 = vmulq_u8_by_i16(source.2, coeff);

                for i in 1..length {
                    let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight as i16);
                    let v_source = vld3q_u8(shifted_src.get_unchecked((i * N)..).as_ptr());
                    k0 = vfmlaq_u8_s16(k0, v_source.0, coeff);
                    k1 = vfmlaq_u8_s16(k1, v_source.1, coeff);
                    k2 = vfmlaq_u8_s16(k2, v_source.2, coeff);
                }

                let dst_offset = y * dst_stride + _cx * N;
                let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
                vst3q_u8(
                    dst_ptr0,
                    uint8x16x3_t(vqmovnq_s32_u8(k0), vqmovnq_s32_u8(k1), vqmovnq_s32_u8(k2)),
                );
                _cx += 16;
            }

            while _cx + 8 < width {
                let coeff = vdup_n_s16(scanned_kernel.get_unchecked(0).weight as i16);

                let shifted_src = local_src.get_unchecked((_cx * N)..);

                let source = vld3_u8(shifted_src.as_ptr());
                let mut k0 = vmul_u8_by_i16(source.0, coeff);
                let mut k1 = vmul_u8_by_i16(source.1, coeff);
                let mut k2 = vmul_u8_by_i16(source.2, coeff);

                for i in 1..length {
                    let coeff = vdup_n_s16(scanned_kernel.get_unchecked(i).weight as i16);
                    let v_source = vld3_u8(shifted_src.get_unchecked((i * N)..).as_ptr());
                    k0 = vfmla_u8_s16(k0, v_source.0, coeff);
                    k1 = vfmla_u8_s16(k1, v_source.1, coeff);
                    k2 = vfmla_u8_s16(k2, v_source.2, coeff);
                }

                let dst_offset = y * dst_stride + _cx * N;
                let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
                vst3_u8(
                    dst_ptr0,
                    uint8x8x3_t(vqmovn_s32_u8(k0), vqmovn_s32_u8(k1), vqmovn_s32_u8(k2)),
                );
                _cx += 8;
            }

            while _cx + 4 < width {
                let coeff = *scanned_kernel.get_unchecked(0);

                let shifted_src = local_src.get_unchecked((_cx * N)..);

                let mut k0 = ColorGroup::<N, i32>::from_slice(shifted_src, 0).mul(coeff.weight);
                let mut k1 = ColorGroup::<N, i32>::from_slice(shifted_src, N).mul(coeff.weight);
                let mut k2 = ColorGroup::<N, i32>::from_slice(shifted_src, N * 2).mul(coeff.weight);
                let mut k3 = ColorGroup::<N, i32>::from_slice(shifted_src, N * 3).mul(coeff.weight);

                for i in 1..length {
                    let coeff = *scanned_kernel.get_unchecked(i);
                    k0 = ColorGroup::<N, i32>::from_slice(shifted_src, i * N)
                        .mul(coeff.weight)
                        .add(k0);
                    k1 = ColorGroup::<N, i32>::from_slice(shifted_src, (i + 1) * N)
                        .mul(coeff.weight)
                        .add(k1);
                    k2 = ColorGroup::<N, i32>::from_slice(shifted_src, (i + 2) * N)
                        .mul(coeff.weight)
                        .add(k2);
                    k3 = ColorGroup::<N, i32>::from_slice(shifted_src, (i + 3) * N)
                        .mul(coeff.weight)
                        .add(k3);
                }

                let dst_offset = y * dst_stride + _cx * N;

                k0.to_approx_store(dst, dst_offset);
                k1.to_approx_store(dst, dst_offset + N);
                k2.to_approx_store(dst, dst_offset + N * 2);
                k3.to_approx_store(dst, dst_offset + N * 3);
                _cx += 4;
            }

            for x in _cx..width {
                let coeff = *scanned_kernel.get_unchecked(0);
                let shifted_src = local_src.get_unchecked((x * N)..);
                let mut k0 = ColorGroup::<N, i32>::from_slice(shifted_src, 0).mul(coeff.weight);

                for i in 1..length {
                    let coeff = *scanned_kernel.get_unchecked(i);
                    k0 = ColorGroup::<N, i32>::from_slice(shifted_src, i * N)
                        .mul(coeff.weight)
                        .add(k0);
                }

                k0.to_approx_store(dst, y * dst_stride + x * N);
            }
        }
    }
}
