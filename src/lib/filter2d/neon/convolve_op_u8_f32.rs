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
use crate::filter1d::neon::utils::{vfmlaq_u8_f32, vmulq_u8_by_f32, vqmovnq_f32_u8};
use crate::filter1d::Arena;
use crate::filter2d::scan_point_2d::ScanPoint2d;
use crate::to_storage::ToStorage;
use crate::unsafe_slice::UnsafeSlice;
use crate::ImageSize;
use std::arch::aarch64::*;
use std::ops::Mul;

pub fn convolve_segment_neon_2d_u8_f32(
    arena: Arena,
    arena_source: &[u8],
    dst: &UnsafeSlice<u8>,
    image_size: ImageSize,
    prepared_kernel: &[ScanPoint2d<f32>],
    y: usize,
) {
    unsafe {
        let width = image_size.width;
        let stride = image_size.width;

        let dx = arena.pad_w as i64;
        let dy = arena.pad_h as i64;

        let arena_width = arena.width;

        let offsets = prepared_kernel
            .iter()
            .map(|&x| {
                arena_source.get_unchecked(
                    ((x.y + dy + y as i64) as usize * arena_width + (x.x + dx) as usize)..,
                )
            })
            .collect::<Vec<_>>();

        let length = prepared_kernel.len();

        let mut _cx = 0usize;

        while _cx + 64 < length {
            let k_weight = vdupq_n_f32(prepared_kernel.get_unchecked(0).weight);
            let items0 = vld1q_u8_x4(offsets.get_unchecked(0).get_unchecked(_cx..).as_ptr());
            let mut k0 = vmulq_u8_by_f32(items0.0, k_weight);
            let mut k1 = vmulq_u8_by_f32(items0.1, k_weight);
            let mut k2 = vmulq_u8_by_f32(items0.2, k_weight);
            let mut k3 = vmulq_u8_by_f32(items0.3, k_weight);
            for i in 1..length {
                let weight = vdupq_n_f32(prepared_kernel.get_unchecked(i).weight);
                let s_ptr = offsets.get_unchecked(i);
                let items0 = vld1q_u8_x4(s_ptr.get_unchecked(_cx..).as_ptr());
                k0 = vfmlaq_u8_f32(k0, items0.0, weight);
                k1 = vfmlaq_u8_f32(k1, items0.1, weight);
                k2 = vfmlaq_u8_f32(k2, items0.2, weight);
                k3 = vfmlaq_u8_f32(k3, items0.3, weight);
            }
            let dst_offset = y * stride + _cx;
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

        while _cx + 32 < length {
            let k_weight = vdupq_n_f32(prepared_kernel.get_unchecked(0).weight);
            let items0 = vld1q_u8_x2(offsets.get_unchecked(0).get_unchecked(_cx..).as_ptr());
            let mut k0 = vmulq_u8_by_f32(items0.0, k_weight);
            let mut k1 = vmulq_u8_by_f32(items0.1, k_weight);
            for i in 1..length {
                let weight = vdupq_n_f32(prepared_kernel.get_unchecked(i).weight);
                let s_ptr = offsets.get_unchecked(i);
                let items0 = vld1q_u8_x2(s_ptr.get_unchecked(_cx..).as_ptr());
                k0 = vfmlaq_u8_f32(k0, items0.0, weight);
                k1 = vfmlaq_u8_f32(k1, items0.1, weight);
            }
            let dst_offset = y * stride + _cx;
            let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
            vst1q_u8_x2(
                dst_ptr0,
                uint8x16x2_t(vqmovnq_f32_u8(k0), vqmovnq_f32_u8(k1)),
            );
            _cx += 32;
        }

        while _cx + 16 < length {
            let k_weight = vdupq_n_f32(prepared_kernel.get_unchecked(0).weight);
            let items0 = vld1q_u8(offsets.get_unchecked(0).get_unchecked(_cx..).as_ptr());
            let mut k0 = vmulq_u8_by_f32(items0, k_weight);
            for i in 1..length {
                let weight = vdupq_n_f32(prepared_kernel.get_unchecked(i).weight);
                let items0 = vld1q_u8(offsets.get_unchecked(i).get_unchecked(_cx..).as_ptr());
                k0 = vfmlaq_u8_f32(k0, items0, weight);
            }
            let dst_offset = y * stride + _cx;
            let dst_ptr0 = (dst.slice.as_ptr() as *mut u8).add(dst_offset);
            vst1q_u8(dst_ptr0, vqmovnq_f32_u8(k0));
            _cx += 16;
        }

        while _cx + 4 < width {
            let k_weight = prepared_kernel.get_unchecked(0).weight;

            let mut k0 = ((*offsets.get_unchecked(0).get_unchecked(_cx)) as f32).mul(k_weight);
            let mut k1 = ((*offsets.get_unchecked(0).get_unchecked(_cx + 1)) as f32).mul(k_weight);
            let mut k2 = ((*offsets.get_unchecked(0).get_unchecked(_cx + 2)) as f32).mul(k_weight);
            let mut k3 = ((*offsets.get_unchecked(0).get_unchecked(_cx + 3)) as f32).mul(k_weight);

            for i in 1..length {
                let weight = prepared_kernel.get_unchecked(i).weight;
                k0 = ((*offsets.get_unchecked(i).get_unchecked(_cx)) as f32).mul_add(weight, k0);
                k1 =
                    ((*offsets.get_unchecked(i).get_unchecked(_cx + 1)) as f32).mul_add(weight, k1);
                k2 =
                    ((*offsets.get_unchecked(i).get_unchecked(_cx + 2)) as f32).mul_add(weight, k2);
                k3 =
                    ((*offsets.get_unchecked(i).get_unchecked(_cx + 3)) as f32).mul_add(weight, k3);
            }

            let dst_offset = y * stride + _cx;

            dst.write(dst_offset, k0.to_());
            dst.write(dst_offset + 1, k1.to_());
            dst.write(dst_offset + 2, k2.to_());
            dst.write(dst_offset + 3, k3.to_());
            _cx += 4;
        }

        for x in _cx..width {
            let k_weight = prepared_kernel.get_unchecked(0).weight;

            let mut k0 = ((*(*offsets.get_unchecked(0)).get_unchecked(x)) as f32).mul(k_weight);

            for i in 1..length {
                k0 = ((*offsets.get_unchecked(i).get_unchecked(x)) as f32).mul_add(k_weight, k0);
            }
            dst.write(y * stride + x, k0.to_());
        }
    }
}
