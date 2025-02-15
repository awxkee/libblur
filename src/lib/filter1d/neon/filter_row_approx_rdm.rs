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
use crate::filter1d::neon::utils::{vmlaq_hi_u8_s16, vmullq_expand_i16, vqmovnq_s16x2_u8};
use crate::filter1d::region::FilterRegion;
use crate::img_size::ImageSize;
use crate::unsafe_slice::UnsafeSlice;
use std::arch::aarch64::*;
use std::ops::{Add, Mul};

pub(crate) fn filter_row_neon_u8_i32_rdm(
    arena: Arena,
    arena_src: &[u8],
    dst: &UnsafeSlice<u8>,
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<i32>],
) {
    unsafe {
        let src = arena_src;

        const N: usize = 1;

        let y = filter_region.start;
        let local_src = src;

        let length = scanned_kernel.iter().len();

        let max_width = N * image_size.width;
        let dst_stride = arena.width * arena.components;

        const EXPAND: i32 = 6;
        const PRECISION: i32 = 6;

        let mut cx = 0usize;

        while cx + 64 < max_width {
            let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(0).weight as i16);

            let shifted_src = local_src.get_unchecked(cx..);

            let source = vld1q_u8_x4(shifted_src.as_ptr());
            let mut k0 = vmullq_expand_i16::<EXPAND>(source.0, coeff);
            let mut k1 = vmullq_expand_i16::<EXPAND>(source.1, coeff);
            let mut k2 = vmullq_expand_i16::<EXPAND>(source.2, coeff);
            let mut k3 = vmullq_expand_i16::<EXPAND>(source.3, coeff);

            for i in 1..length {
                let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight as i16);
                let v_source = vld1q_u8_x4(shifted_src.get_unchecked(i..).as_ptr());
                k0 = vmlaq_hi_u8_s16::<EXPAND>(k0, v_source.0, coeff);
                k1 = vmlaq_hi_u8_s16::<EXPAND>(k1, v_source.1, coeff);
                k2 = vmlaq_hi_u8_s16::<EXPAND>(k2, v_source.2, coeff);
                k3 = vmlaq_hi_u8_s16::<EXPAND>(k3, v_source.3, coeff);
            }

            let dst_offset = y * dst_stride + cx;
            let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
            vst1q_u8_x4(
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

        while cx + 32 < max_width {
            let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(0).weight as i16);

            let shifted_src = local_src.get_unchecked(cx..);

            let source = vld1q_u8_x2(shifted_src.as_ptr());
            let mut k0 = vmullq_expand_i16::<EXPAND>(source.0, coeff);
            let mut k1 = vmullq_expand_i16::<EXPAND>(source.1, coeff);

            for i in 1..length {
                let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight as i16);
                let v_source = vld1q_u8_x2(shifted_src.get_unchecked(i..).as_ptr());
                k0 = vmlaq_hi_u8_s16::<EXPAND>(k0, v_source.0, coeff);
                k1 = vmlaq_hi_u8_s16::<EXPAND>(k1, v_source.1, coeff);
            }

            let dst_offset = y * dst_stride + cx;
            let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
            vst1q_u8_x2(
                dst_ptr0,
                uint8x16x2_t(
                    vqmovnq_s16x2_u8::<PRECISION>(k0),
                    vqmovnq_s16x2_u8::<PRECISION>(k1),
                ),
            );
            cx += 32;
        }

        while cx + 16 < max_width {
            let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(0).weight as i16);

            let shifted_src = local_src.get_unchecked(cx..);

            let source = vld1q_u8(shifted_src.as_ptr());
            let mut k0 = vmullq_expand_i16::<EXPAND>(source, coeff);

            for i in 1..length {
                let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight as i16);
                let v_source = vld1q_u8(shifted_src.get_unchecked(i..).as_ptr());
                k0 = vmlaq_hi_u8_s16::<EXPAND>(k0, v_source, coeff);
            }

            let dst_offset = y * dst_stride + cx;
            let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
            vst1q_u8(dst_ptr0, vqmovnq_s16x2_u8::<PRECISION>(k0));
            cx += 16;
        }

        while cx + 4 < max_width {
            let coeff = *scanned_kernel.get_unchecked(0);

            let shifted_src = local_src.get_unchecked((cx * N)..);

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

            let dst_offset = y * dst_stride + cx * N;

            k0.to_approx_store(dst, dst_offset);
            k1.to_approx_store(dst, dst_offset + N);
            k2.to_approx_store(dst, dst_offset + N * 2);
            k3.to_approx_store(dst, dst_offset + N * 3);
            cx += 4;
        }

        for x in cx..max_width {
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
