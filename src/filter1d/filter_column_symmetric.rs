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
use crate::img_size::ImageSize;
use crate::mlaf::mlaf;
use crate::to_storage::ToStorage;
use num_traits::{AsPrimitive, MulAdd};
use std::ops::{Add, Mul};

pub(crate) fn filter_symmetric_column<T, F>(
    arena: Arena,
    arena_src: &[&[T]],
    dst: &mut [T],
    image_size: ImageSize,
    region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<F>],
) where
    T: Copy + AsPrimitive<F>,
    F: ToStorage<T> + Mul<F, Output = F> + MulAdd<F, Output = F> + Add<F, Output = F> + Default,
{
    unsafe {
        let _yy = region.start;

        let dst_stride = image_size.width * arena.components;

        let length = scanned_kernel.len();
        let half_len = length / 2;

        let mut cx = 0usize;

        let coeff = scanned_kernel[half_len].weight;

        while cx + 32 < dst_stride {
            let mut store: [F; 32] = [F::default(); 32];

            let v_src = &arena_src[half_len][cx..(cx + 32)];

            for (dst, src) in store.iter_mut().zip(v_src) {
                *dst = src.as_().mul(coeff);
            }

            for (i, coeff) in scanned_kernel.iter().take(half_len).enumerate() {
                let rollback = length - i - 1;
                let fw = &arena_src[i][cx..(cx + 32)];
                let bw = &arena_src[rollback][cx..(cx + 32)];

                for ((dst, fw), bw) in store.iter_mut().zip(fw).zip(bw) {
                    *dst = mlaf(*dst, fw.as_().add(bw.as_()), coeff.weight);
                }
            }

            for (y, src) in store.iter().enumerate() {
                *dst.get_unchecked_mut(cx + y) = src.to_();
            }

            cx += 32;
        }

        while cx + 16 < dst_stride {
            let mut store: [F; 16] = [F::default(); 16];

            let v_src = &arena_src[half_len][cx..(cx + 16)];

            for (dst, src) in store.iter_mut().zip(v_src) {
                *dst = src.as_().mul(coeff);
            }

            for (i, coeff) in scanned_kernel.iter().take(half_len).enumerate() {
                let rollback = length - i - 1;
                let fw = &arena_src[i][cx..(cx + 16)];
                let bw = &arena_src[rollback][cx..(cx + 16)];

                for ((dst, fw), bw) in store.iter_mut().zip(fw).zip(bw) {
                    *dst = mlaf(*dst, fw.as_().add(bw.as_()), coeff.weight);
                }
            }

            for (y, src) in store.iter().enumerate() {
                *dst.get_unchecked_mut(cx + y) = src.to_();
            }

            cx += 16;
        }

        while cx + 4 < dst_stride {
            let v_src = &arena_src[half_len][cx..(cx + 4)];

            let mut k0 = v_src[0].as_().mul(coeff);
            let mut k1 = v_src[1].as_().mul(coeff);
            let mut k2 = v_src[2].as_().mul(coeff);
            let mut k3 = v_src[3].as_().mul(coeff);

            for (i, coeff) in scanned_kernel.iter().take(half_len).enumerate() {
                let rollback = length - i - 1;
                let fw = &arena_src[i][cx..(cx + 4)];
                let bw = &arena_src[rollback][cx..(cx + 4)];
                k0 = mlaf(k0, fw[0].as_().add(bw[0].as_()), coeff.weight);
                k1 = mlaf(k1, fw[1].as_().add(bw[1].as_()), coeff.weight);
                k2 = mlaf(k2, fw[2].as_().add(bw[2].as_()), coeff.weight);
                k3 = mlaf(k3, fw[3].as_().add(bw[3].as_()), coeff.weight);
            }

            *dst.get_unchecked_mut(cx) = k0.to_();
            *dst.get_unchecked_mut(cx + 1) = k1.to_();
            *dst.get_unchecked_mut(cx + 2) = k2.to_();
            *dst.get_unchecked_mut(cx + 3) = k3.to_();
            cx += 4;
        }

        for x in cx..dst_stride {
            let v_src = &arena_src[half_len][x..(x + 1)];

            let mut k0 = v_src[0].as_().mul(coeff);

            for (i, coeff) in scanned_kernel.iter().take(half_len).enumerate() {
                let rollback = length - i - 1;
                let fw = &arena_src[i][x..(x + 1)];
                let bw = &arena_src[rollback][x..(x + 1)];
                k0 = mlaf(k0, fw[0].as_().add(bw[0].as_()), coeff.weight);
            }

            *dst.get_unchecked_mut(x) = k0.to_();
        }
    }
}
