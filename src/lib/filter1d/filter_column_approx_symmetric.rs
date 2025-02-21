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
use crate::filter1d::region::FilterRegion;
use crate::filter1d::to_approx_storage::ToApproxStorage;
use crate::img_size::ImageSize;
use num_traits::AsPrimitive;
use std::ops::{Add, Mul, Shr};

pub fn filter_column_symmetric_approx<T, I>(
    arena: Arena,
    arena_src: &[&[T]],
    dst: &mut [T],
    image_size: ImageSize,
    _: FilterRegion,
    scanned_kernel: &[ScanPoint1d<I>],
) where
    T: Copy + AsPrimitive<I> + Default,
    I: Copy
        + Mul<Output = I>
        + Add<Output = I>
        + Shr<I, Output = I>
        + Default
        + 'static
        + ToApproxStorage<T>,
{
    unsafe {
        let dst_stride = image_size.width * arena.components;

        let length = scanned_kernel.len();
        let half_len = length / 2;

        let mut cx = 0usize;

        while cx + 32 < dst_stride {
            let coeff = scanned_kernel[half_len].weight;

            let mut store: [I; 32] = [I::default(); 32];

            let v_src = &arena_src[half_len][cx..(cx + 32)];

            for (dst, src) in store.iter_mut().zip(v_src) {
                *dst = src.as_().mul(coeff);
            }

            for (i, coeff) in scanned_kernel.iter().take(half_len).enumerate() {
                let rollback = length - i - 1;
                let fw = &arena_src[i][cx..(cx + 32)];
                let bw = &arena_src[rollback][cx..(cx + 32)];

                for ((dst, fw), bw) in store.iter_mut().zip(fw).zip(bw) {
                    *dst = *dst + fw.as_().add(bw.as_()).mul(coeff.weight);
                }
            }

            for (y, src) in store.iter().enumerate() {
                *dst.get_unchecked_mut(cx + y) = src.to_approx_();
            }

            cx += 32;
        }

        while cx + 16 < dst_stride {
            let coeff = scanned_kernel[half_len].weight;

            let mut store: [I; 16] = [I::default(); 16];

            let v_src = &arena_src[half_len][cx..(cx + 16)];

            for (dst, src) in store.iter_mut().zip(v_src) {
                *dst = src.as_().mul(coeff);
            }

            for (i, coeff) in scanned_kernel.iter().take(half_len).enumerate() {
                let rollback = length - i - 1;
                let fw = &arena_src[i][cx..(cx + 16)];
                let bw = &arena_src[rollback][cx..(cx + 16)];

                for ((dst, fw), bw) in store.iter_mut().zip(fw).zip(bw) {
                    *dst = *dst + fw.as_().add(bw.as_()).mul(coeff.weight);
                }
            }

            for (y, src) in store.iter().enumerate() {
                *dst.get_unchecked_mut(cx + y) = src.to_approx_();
            }

            cx += 16;
        }

        while cx + 4 < dst_stride {
            let coeff = scanned_kernel[half_len].weight;

            let v_src = &arena_src[half_len][cx..(cx + 4)];

            let mut k0 = v_src[0].as_().mul(coeff);
            let mut k1 = v_src[1].as_().mul(coeff);
            let mut k2 = v_src[2].as_().mul(coeff);
            let mut k3 = v_src[3].as_().mul(coeff);

            for (i, coeff) in scanned_kernel.iter().take(half_len).enumerate() {
                let rollback = length - i - 1;
                let fw = &arena_src[i][cx..(cx + 4)];
                let bw = &arena_src[rollback][cx..(cx + 4)];
                k0 = fw[0].as_().add(bw[0].as_()).mul(coeff.weight).add(k0);
                k1 = fw[1].as_().add(bw[1].as_()).mul(coeff.weight).add(k1);
                k2 = fw[2].as_().add(bw[2].as_()).mul(coeff.weight).add(k2);
                k3 = fw[3].as_().add(bw[3].as_()).mul(coeff.weight).add(k3);
            }

            *dst.get_unchecked_mut(cx) = k0.to_approx_();
            *dst.get_unchecked_mut(cx + 1) = k1.to_approx_();
            *dst.get_unchecked_mut(cx + 2) = k2.to_approx_();
            *dst.get_unchecked_mut(cx + 3) = k3.to_approx_();
            cx += 4;
        }

        for x in cx..dst_stride {
            let coeff = scanned_kernel[half_len].weight;

            let v_src = &arena_src[half_len][x..(x + 1)];

            let mut k0 = v_src[0].as_().mul(coeff);

            for (i, coeff) in scanned_kernel.iter().take(half_len).enumerate() {
                let rollback = length - i - 1;
                let fw = &arena_src[i][x..(x + 1)];
                let bw = &arena_src[rollback][x..(x + 1)];
                k0 = fw[0].as_().add(bw[0].as_()).mul(coeff.weight).add(k0);
            }

            *dst.get_unchecked_mut(x) = k0.to_approx_();
        }
    }
}
