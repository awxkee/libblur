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
use crate::filter1d::neon::utils::{vfmlaq_u8_f32, vmulq_u8_by_f32, vqmovnq_f32_u8};
use crate::filter1d::region::FilterRegion;
use crate::img_size::ImageSize;
use crate::mlaf::mlaf;
use crate::to_storage::ToStorage;
use crate::unsafe_slice::UnsafeSlice;
use std::arch::aarch64::*;
use std::ops::Mul;

pub fn filter_column_neon_u8_f32(
    arena: Arena,
    arena_src: &[&[u8]],
    dst: &UnsafeSlice<u8>,
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f32>],
) {
    unsafe {
        let image_width = image_size.width * arena.components;

        let dst_stride = image_size.width * arena.components;

        let length = scanned_kernel.len();

        let y = filter_region.start;

        let mut _cx = 0usize;

        while _cx + 64 < image_width {
            let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(0).weight);

            let v_src = arena_src.get_unchecked(0).get_unchecked(_cx..);

            let source = vld1q_u8_x4(v_src.as_ptr());
            let mut k0 = vmulq_u8_by_f32(source.0, coeff);
            let mut k1 = vmulq_u8_by_f32(source.1, coeff);
            let mut k2 = vmulq_u8_by_f32(source.2, coeff);
            let mut k3 = vmulq_u8_by_f32(source.3, coeff);

            for i in 1..length {
                let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(i).weight);
                let v_source =
                    vld1q_u8_x4(arena_src.get_unchecked(i).get_unchecked(_cx..).as_ptr());
                k0 = vfmlaq_u8_f32(k0, v_source.0, coeff);
                k1 = vfmlaq_u8_f32(k1, v_source.1, coeff);
                k2 = vfmlaq_u8_f32(k2, v_source.2, coeff);
                k3 = vfmlaq_u8_f32(k3, v_source.3, coeff);
            }

            let dst_offset = y * dst_stride + _cx;
            let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
            vst1q_u8_x4(
                dst_ptr0,
                uint8x16x4_t(
                    vqmovnq_f32_u8(k0),
                    vqmovnq_f32_u8(k1),
                    vqmovnq_f32_u8(k2),
                    vqmovnq_f32_u8(k3),
                ),
            );
            _cx += 64;
        }

        while _cx + 48 < image_width {
            let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(0).weight);

            let v_src = arena_src.get_unchecked(0).get_unchecked(_cx..);

            let source = vld1q_u8_x3(v_src.as_ptr());
            let mut k0 = vmulq_u8_by_f32(source.0, coeff);
            let mut k1 = vmulq_u8_by_f32(source.1, coeff);
            let mut k2 = vmulq_u8_by_f32(source.2, coeff);

            for i in 1..length {
                let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(i).weight);
                let v_source =
                    vld1q_u8_x3(arena_src.get_unchecked(i).get_unchecked(_cx..).as_ptr());
                k0 = vfmlaq_u8_f32(k0, v_source.0, coeff);
                k1 = vfmlaq_u8_f32(k1, v_source.1, coeff);
                k2 = vfmlaq_u8_f32(k2, v_source.2, coeff);
            }

            let dst_offset = y * dst_stride + _cx;
            let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
            vst1q_u8_x3(
                dst_ptr0,
                uint8x16x3_t(vqmovnq_f32_u8(k0), vqmovnq_f32_u8(k1), vqmovnq_f32_u8(k2)),
            );
            _cx += 48;
        }

        while _cx + 32 < image_width {
            let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(0).weight);

            let v_src = arena_src.get_unchecked(0).get_unchecked(_cx..);

            let source = vld1q_u8_x2(v_src.as_ptr());
            let mut k0 = vmulq_u8_by_f32(source.0, coeff);
            let mut k1 = vmulq_u8_by_f32(source.1, coeff);

            for i in 1..length {
                let coeff = vdupq_n_f32(scanned_kernel.get_unchecked(i).weight);
                let v_source =
                    vld1q_u8_x2(arena_src.get_unchecked(i).get_unchecked(_cx..).as_ptr());
                k0 = vfmlaq_u8_f32(k0, v_source.0, coeff);
                k1 = vfmlaq_u8_f32(k1, v_source.1, coeff);
            }

            let dst_offset = y * dst_stride + _cx;
            let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
            vst1q_u8_x2(
                dst_ptr0,
                uint8x16x2_t(vqmovnq_f32_u8(k0), vqmovnq_f32_u8(k1)),
            );

            _cx += 32;
        }

        while _cx + 16 < image_width {
            let coeff = *scanned_kernel.get_unchecked(0);

            let v_src = arena_src.get_unchecked(0).get_unchecked(_cx..);

            let source_0 = vld1q_u8(v_src.as_ptr());
            let mut k0 = vmulq_u8_by_f32(source_0, vdupq_n_f32(coeff.weight));

            for i in 1..length {
                let coeff = *scanned_kernel.get_unchecked(i);
                let v_source_0 = vld1q_u8(arena_src.get_unchecked(i).get_unchecked(_cx..).as_ptr());
                k0 = vfmlaq_u8_f32(k0, v_source_0, vdupq_n_f32(coeff.weight));
            }

            let dst_offset = y * dst_stride + _cx;
            let dst_ptr = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
            vst1q_u8(dst_ptr, vqmovnq_f32_u8(k0));
            _cx += 16;
        }

        while _cx + 4 < image_width {
            let coeff = *scanned_kernel.get_unchecked(0);

            let v_src = arena_src.get_unchecked(0).get_unchecked(_cx..);

            let mut k0 = (*v_src.get_unchecked(0) as f32).mul(coeff.weight);
            let mut k1 = (*v_src.get_unchecked(1) as f32).mul(coeff.weight);
            let mut k2 = (*v_src.get_unchecked(2) as f32).mul(coeff.weight);
            let mut k3 = (*v_src.get_unchecked(3) as f32).mul(coeff.weight);

            for i in 1..length {
                let coeff = *scanned_kernel.get_unchecked(i);
                k0 = mlaf(
                    k0,
                    (*arena_src.get_unchecked(i).get_unchecked(_cx)) as f32,
                    coeff.weight,
                );
                k1 = mlaf(
                    k1,
                    (*arena_src.get_unchecked(i).get_unchecked(_cx + 1)) as f32,
                    coeff.weight,
                );
                k2 = mlaf(
                    k2,
                    (*arena_src.get_unchecked(i).get_unchecked(_cx + 2)) as f32,
                    coeff.weight,
                );
                k3 = mlaf(
                    k3,
                    (*arena_src.get_unchecked(i).get_unchecked(_cx + 3)) as f32,
                    coeff.weight,
                );
            }

            let dst_offset = y * dst_stride + _cx;

            dst.write(dst_offset, k0.to_());
            dst.write(dst_offset + 1, k1.to_());
            dst.write(dst_offset + 2, k2.to_());
            dst.write(dst_offset + 3, k3.to_());
            _cx += 4;
        }

        for x in _cx..image_width {
            let coeff = *scanned_kernel.get_unchecked(0);

            let v_src = arena_src.get_unchecked(0).get_unchecked(x..);

            let mut k0 = ((*v_src.get_unchecked(0)) as f32).mul(coeff.weight);

            for i in 1..length {
                let coeff = *scanned_kernel.get_unchecked(i);
                k0 = mlaf(
                    k0,
                    (*arena_src.get_unchecked(i).get_unchecked(x)) as f32,
                    coeff.weight,
                );
            }

            dst.write(y * dst_stride + x, k0.to_());
        }
    }
}
