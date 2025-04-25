/*
 * // Copyright (c) Radzivon Bartoshyk 4/2025. All rights reserved.
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
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::region::FilterRegion;
use crate::{Arena, ImageSize};
use std::ops::{Add, Mul};

trait FastPackQ15 {
    fn pack_q8(&self) -> u16;
}

impl FastPackQ15 for u32 {
    #[inline(always)]
    fn pack_q8(&self) -> u16 {
        const RND: u32 = (1 << 14) - 1;
        ((self + RND) >> 15).min(65535) as u16
    }
}

pub(crate) fn filter_row_symmetric_approx_uq8p8_u8<const N: usize>(
    arena_src: &[u16],
    dst: &mut [u16],
    image_size: ImageSize,
    _: FilterRegion,
    scanned_kernel: &[ScanPoint1d<u16>],
) {
    unsafe {
        let width = image_size.width;

        let src = arena_src;

        let length = scanned_kernel.len();
        let half_len = length / 2;

        let max_width = width * N;

        let mut cx = 0usize;

        while cx + 4 < max_width {
            let coeff = *scanned_kernel.get_unchecked(half_len);
            let shifted_src = src.get_unchecked(cx..);
            let mut k0 = *shifted_src.get_unchecked(half_len * N) as u32 * coeff.weight as u32;
            let mut k1 = *shifted_src.get_unchecked(half_len * N + 1) as u32 * coeff.weight as u32;
            let mut k2 = *shifted_src.get_unchecked(half_len * N + 2) as u32 * coeff.weight as u32;
            let mut k3 = *shifted_src.get_unchecked(half_len * N + 3) as u32 * coeff.weight as u32;

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;

                k0 = k0
                    + (*shifted_src.get_unchecked(i * N) as u32
                        + *shifted_src.get_unchecked(rollback * N) as u32)
                        * coeff.weight as u32;

                k1 = k1
                    + (*shifted_src.get_unchecked(i * N + 1) as u32
                        + *shifted_src.get_unchecked(rollback * N + 1) as u32)
                        * coeff.weight as u32;

                k2 = k2
                    + (*shifted_src.get_unchecked(i * N + 2) as u32
                        + *shifted_src.get_unchecked(rollback * N + 2) as u32)
                        * coeff.weight as u32;

                k3 = k3
                    + (*shifted_src.get_unchecked(i * N + 3) as u32
                        + *shifted_src.get_unchecked(rollback * N + 3) as u32)
                        * coeff.weight as u32;
            }

            *dst.get_unchecked_mut(cx) = k0.pack_q8();
            *dst.get_unchecked_mut(cx + 1) = k1.pack_q8();
            *dst.get_unchecked_mut(cx + 2) = k2.pack_q8();
            *dst.get_unchecked_mut(cx + 3) = k3.pack_q8();
            cx += 4;
        }

        for x in cx..max_width {
            let coeff = *scanned_kernel.get_unchecked(half_len);
            let shifted_src = src.get_unchecked(x..);
            let mut k0 = (*shifted_src.get_unchecked(half_len * N)) as u32 * coeff.weight as u32;

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;

                k0 = k0
                    + (*shifted_src.get_unchecked(i * N) as u32
                        + *shifted_src.get_unchecked(rollback * N) as u32)
                        * coeff.weight as u32;
            }

            *dst.get_unchecked_mut(x) = k0.pack_q8();
        }
    }
}

pub(crate) fn filter_column_symmetric_approx_uq8p8_u8(
    arena: Arena,
    arena_src: &[&[u16]],
    dst: &mut [u16],
    image_size: ImageSize,
    _: FilterRegion,
    scanned_kernel: &[ScanPoint1d<u16>],
) {
    unsafe {
        let dst_stride = image_size.width * arena.components;

        let length = scanned_kernel.len();
        let half_len = length / 2;

        let mut cx = 0usize;

        while cx + 32 < dst_stride {
            let coeff = scanned_kernel[half_len].weight as u32;

            let mut store: [u32; 32] = [u32::default(); 32];

            let v_src = &arena_src[half_len][cx..(cx + 32)];

            for (dst, src) in store.iter_mut().zip(v_src) {
                *dst = (*src as u32).mul(coeff);
            }

            for (i, coeff) in scanned_kernel.iter().take(half_len).enumerate() {
                let rollback = length - i - 1;
                let fw = &arena_src[i][cx..(cx + 32)];
                let bw = &arena_src[rollback][cx..(cx + 32)];

                for ((dst, &fw), &bw) in store.iter_mut().zip(fw).zip(bw) {
                    *dst = *dst + (fw as u32).add(bw as u32).mul(coeff.weight as u32);
                }
            }

            for (y, src) in store.iter().enumerate() {
                *dst.get_unchecked_mut(cx + y) = src.pack_q8();
            }

            cx += 32;
        }

        while cx + 16 < dst_stride {
            let coeff = scanned_kernel[half_len].weight as u32;

            let mut store: [u32; 16] = [u32::default(); 16];

            let v_src = &arena_src[half_len][cx..(cx + 16)];

            for (dst, src) in store.iter_mut().zip(v_src) {
                *dst = (*src as u32).mul(coeff);
            }

            for (i, coeff) in scanned_kernel.iter().take(half_len).enumerate() {
                let rollback = length - i - 1;
                let fw = &arena_src[i][cx..(cx + 16)];
                let bw = &arena_src[rollback][cx..(cx + 16)];

                for ((dst, &fw), &bw) in store.iter_mut().zip(fw).zip(bw) {
                    *dst = *dst + (fw as u32).add(bw as u32).mul(coeff.weight as u32);
                }
            }

            for (y, src) in store.iter().enumerate() {
                *dst.get_unchecked_mut(cx + y) = src.pack_q8();
            }

            cx += 16;
        }

        while cx + 4 < dst_stride {
            let coeff = scanned_kernel[half_len].weight as u32;

            let v_src = &arena_src[half_len][cx..(cx + 4)];

            let mut k0 = (v_src[0] as u32).mul(coeff);
            let mut k1 = (v_src[1] as u32).mul(coeff);
            let mut k2 = (v_src[2] as u32).mul(coeff);
            let mut k3 = (v_src[3] as u32).mul(coeff);

            for (i, coeff) in scanned_kernel.iter().take(half_len).enumerate() {
                let rollback = length - i - 1;
                let fw = &arena_src[i][cx..(cx + 4)];
                let bw = &arena_src[rollback][cx..(cx + 4)];
                k0 = (fw[0] as u32)
                    .add(bw[0] as u32)
                    .mul(coeff.weight as u32)
                    .add(k0);
                k1 = (fw[1] as u32)
                    .add(bw[1] as u32)
                    .mul(coeff.weight as u32)
                    .add(k1);
                k2 = (fw[2] as u32)
                    .add(bw[2] as u32)
                    .mul(coeff.weight as u32)
                    .add(k2);
                k3 = (fw[3] as u32)
                    .add(bw[3] as u32)
                    .mul(coeff.weight as u32)
                    .add(k3);
            }

            *dst.get_unchecked_mut(cx) = k0.pack_q8();
            *dst.get_unchecked_mut(cx + 1) = k1.pack_q8();
            *dst.get_unchecked_mut(cx + 2) = k2.pack_q8();
            *dst.get_unchecked_mut(cx + 3) = k3.pack_q8();
            cx += 4;
        }

        for x in cx..dst_stride {
            let coeff = scanned_kernel[half_len].weight as u32;

            let v_src = &arena_src[half_len][x..(x + 1)];

            let mut k0 = (v_src[0] as u32).mul(coeff);

            for (i, coeff) in scanned_kernel.iter().take(half_len).enumerate() {
                let rollback = length - i - 1;
                let fw = &arena_src[i][x..(x + 1)];
                let bw = &arena_src[rollback][x..(x + 1)];
                k0 = (fw[0] as u32)
                    .add(bw[0] as u32)
                    .mul(coeff.weight as u32)
                    .add(k0);
            }

            *dst.get_unchecked_mut(x) = k0.pack_q8();
        }
    }
}
