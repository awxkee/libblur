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
use crate::filter1d::Arena;
use crate::filter2d::scan_point_2d::ScanPoint2d;
use crate::mlaf::mlaf;
use crate::to_storage::ToStorage;
use crate::unsafe_slice::UnsafeSlice;
use crate::ImageSize;
use num_traits::{AsPrimitive, MulAdd};
use std::ops::{Add, Mul};

pub(crate) fn convolve_segment_2d<T, F>(
    arena: Arena,
    arena_source: &[T],
    dst: &UnsafeSlice<T>,
    image_size: ImageSize,
    prepared_kernel: &[ScanPoint2d<F>],
    y: usize,
) where
    T: Copy + AsPrimitive<F>,
    F: ToStorage<T> + Mul<Output = F> + MulAdd<F, Output = F> + Add<Output = F>,
{
    unsafe {
        let width = image_size.width;
        let stride = image_size.width * arena.components;

        let dx = arena.pad_w as i64;
        let dy = arena.pad_h as i64;

        let arena_stride = arena.width * arena.components;

        let offsets = prepared_kernel
            .iter()
            .map(|&x| {
                arena_source.get_unchecked(
                    ((x.y + dy + y as i64) as usize * arena_stride
                        + (x.x + dx) as usize * arena.components)..,
                )
            })
            .collect::<Vec<_>>();

        let length = prepared_kernel.len();
        let total_width = width * arena.components;

        let mut cx = 0usize;

        while cx + 4 < total_width {
            let k_weight = prepared_kernel.get_unchecked(0).weight;

            let mut k0 = (*offsets.get_unchecked(0))
                .get_unchecked(cx)
                .as_()
                .mul(k_weight);
            let mut k1 = (*offsets.get_unchecked(0))
                .get_unchecked(cx + 1)
                .as_()
                .mul(k_weight);
            let mut k2 = (*offsets.get_unchecked(0))
                .get_unchecked(cx + 2)
                .as_()
                .mul(k_weight);
            let mut k3 = (*offsets.get_unchecked(0).get_unchecked(cx + 3))
                .as_()
                .mul(k_weight);

            for i in 1..length {
                let weight = prepared_kernel.get_unchecked(i).weight;
                k0 = mlaf(k0, offsets.get_unchecked(i).get_unchecked(cx).as_(), weight);
                k1 = mlaf(
                    k1,
                    offsets.get_unchecked(i).get_unchecked(cx + 1).as_(),
                    weight,
                );
                k2 = mlaf(
                    k2,
                    offsets.get_unchecked(i).get_unchecked(cx + 2).as_(),
                    weight,
                );
                k3 = mlaf(
                    k3,
                    offsets.get_unchecked(i).get_unchecked(cx + 3).as_(),
                    weight,
                );
            }

            let dst_offset = y * stride + cx;

            dst.write(dst_offset, k0.to_());
            dst.write(dst_offset + 1, k1.to_());
            dst.write(dst_offset + 2, k2.to_());
            dst.write(dst_offset + 3, k3.to_());
            cx += 4;
        }

        for x in cx..total_width {
            let k_weight = prepared_kernel.get_unchecked(0).weight;

            let mut k0 = (*offsets.get_unchecked(0))
                .get_unchecked(x)
                .as_()
                .mul(k_weight);

            for i in 1..length {
                let k_weight = prepared_kernel.get_unchecked(i).weight;
                k0 = mlaf(
                    k0,
                    offsets.get_unchecked(i).get_unchecked(x).as_(),
                    k_weight,
                );
            }
            dst.write(y * stride + x, k0.to_());
        }
    }
}
