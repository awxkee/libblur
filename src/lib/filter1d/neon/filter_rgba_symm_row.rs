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
    vfmla_symm_u8_f32, vfmlaq_symm_u8_f32, vmul_u8_by_f32, vmulq_u8_by_f32, vqmovn_f32_u8,
    vqmovnq_f32_u8,
};
use crate::filter1d::region::FilterRegion;
use crate::img_size::ImageSize;
use crate::unsafe_slice::UnsafeSlice;
use num_traits::MulAdd;
use std::arch::aarch64::*;
use std::ops::{Add, Mul};

pub fn filter_rgba_row_symm_neon_u8_f32(
    arena: Arena,
    arena_src: &[u8],
    dst: &UnsafeSlice<u8>,
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    unsafe {
        let width = image_size.width;

        const N: usize = 4;

        let src = arena_src;

        let dst_stride = image_size.width * arena.components;

        let length = scanned_kernel.len();
        let half_len = length / 2;

        let y = filter_region.start;
        let local_src = src;

        let mut _cx = 0usize;

        while _cx + 16 < width {
            let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(half_len).weight);

            let shifted_src = local_src.get_unchecked((_cx * N)..);

            let source = vld4q_u8(shifted_src.get_unchecked((half_len * N)..).as_ptr());
            let mut k0 = vmulq_u8_by_f32(source.0, coeff);
            let mut k1 = vmulq_u8_by_f32(source.1, coeff);
            let mut k2 = vmulq_u8_by_f32(source.2, coeff);
            let mut k3 = vmulq_u8_by_f32(source.3, coeff);

            for i in 0..half_len {
                let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(i).weight);
                let rollback = length - i - 1;
                let v_source0 = vld4q_u8(shifted_src.get_unchecked((i * N)..).as_ptr());
                let v_source1 = vld4q_u8(shifted_src.get_unchecked((rollback * N)..).as_ptr());
                k0 = vfmlaq_symm_u8_f32(k0, v_source0.0, v_source1.0, coeff);
                k1 = vfmlaq_symm_u8_f32(k1, v_source0.1, v_source1.1, coeff);
                k2 = vfmlaq_symm_u8_f32(k2, v_source0.2, v_source1.2, coeff);
                k3 = vfmlaq_symm_u8_f32(k3, v_source0.3, v_source1.3, coeff);
            }

            let dst_offset = y * dst_stride + _cx * N;
            let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
            vst4q_u8(
                dst_ptr0,
                uint8x16x4_t(
                    vqmovnq_f32_u8(k0),
                    vqmovnq_f32_u8(k1),
                    vqmovnq_f32_u8(k2),
                    vqmovnq_f32_u8(k3),
                ),
            );
            _cx += 16;
        }

        while _cx + 8 < width {
            let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(half_len).weight);

            let shifted_src = local_src.get_unchecked((_cx * N)..);

            let source = vld4_u8(shifted_src.get_unchecked((half_len * N)..).as_ptr());
            let mut k0 = vmul_u8_by_f32(source.0, coeff);
            let mut k1 = vmul_u8_by_f32(source.1, coeff);
            let mut k2 = vmul_u8_by_f32(source.2, coeff);
            let mut k3 = vmul_u8_by_f32(source.3, coeff);

            for i in 0..half_len {
                let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(i).weight);
                let rollback = length - i - 1;
                let v_source0 = vld4_u8(shifted_src.get_unchecked((i * N)..).as_ptr());
                let v_source1 = vld4_u8(shifted_src.get_unchecked((rollback * N)..).as_ptr());
                k0 = vfmla_symm_u8_f32(k0, v_source0.0, v_source1.0, coeff);
                k1 = vfmla_symm_u8_f32(k1, v_source0.1, v_source1.1, coeff);
                k2 = vfmla_symm_u8_f32(k2, v_source0.2, v_source1.2, coeff);
                k3 = vfmla_symm_u8_f32(k3, v_source0.3, v_source1.3, coeff);
            }

            let dst_offset = y * dst_stride + _cx * N;
            let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
            vst4_u8(
                dst_ptr0,
                uint8x8x4_t(
                    vqmovn_f32_u8(k0),
                    vqmovn_f32_u8(k1),
                    vqmovn_f32_u8(k2),
                    vqmovn_f32_u8(k3),
                ),
            );
            _cx += 8;
        }

        while _cx + 4 < width {
            let coeff = *scanned_kernel.get_unchecked(half_len);

            let shifted_src = local_src.get_unchecked((_cx * N)..);

            let mut k0 =
                ColorGroup::<N, f32>::from_slice(shifted_src, half_len * N).mul(coeff.weight);
            let mut k1 =
                ColorGroup::<N, f32>::from_slice(shifted_src, (half_len * N) + N).mul(coeff.weight);
            let mut k2 = ColorGroup::<N, f32>::from_slice(shifted_src, (half_len * N) + N * 2)
                .mul(coeff.weight);
            let mut k3 = ColorGroup::<N, f32>::from_slice(shifted_src, (half_len * N) + N * 3)
                .mul(coeff.weight);

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;
                k0 = ColorGroup::<N, f32>::from_slice(shifted_src, i * N)
                    .add(ColorGroup::<N, f32>::from_slice(shifted_src, rollback * N))
                    .mul_add(k0, coeff.weight);
                k1 = ColorGroup::<N, f32>::from_slice(shifted_src, (i + 1) * N)
                    .add(ColorGroup::<N, f32>::from_slice(
                        shifted_src,
                        (rollback + 1) * N,
                    ))
                    .mul_add(k1, coeff.weight);
                k2 = ColorGroup::<N, f32>::from_slice(shifted_src, (i + 2) * N)
                    .add(ColorGroup::<N, f32>::from_slice(
                        shifted_src,
                        (rollback + 2) * N,
                    ))
                    .mul_add(k2, coeff.weight);
                k3 = ColorGroup::<N, f32>::from_slice(shifted_src, (i + 3) * N)
                    .add(ColorGroup::<N, f32>::from_slice(
                        shifted_src,
                        (rollback + 3) * N,
                    ))
                    .mul_add(k3, coeff.weight);
            }

            let dst_offset = y * dst_stride + _cx * N;

            k0.to_store(dst, dst_offset);
            k1.to_store(dst, dst_offset + N);
            k2.to_store(dst, dst_offset + N * 2);
            k3.to_store(dst, dst_offset + N * 3);
            _cx += 4;
        }

        for x in _cx..width {
            let coeff = *scanned_kernel.get_unchecked(half_len);
            let shifted_src = local_src.get_unchecked((x * N)..);
            let mut k0 =
                ColorGroup::<N, f32>::from_slice(shifted_src, half_len * N).mul(coeff.weight);

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;
                k0 = ColorGroup::<N, f32>::from_slice(shifted_src, i * N)
                    .add(ColorGroup::<N, f32>::from_slice(shifted_src, rollback * N))
                    .mul_add(k0, coeff.weight);
            }

            k0.to_store(dst, y * dst_stride + x * N);
        }
    }
}
