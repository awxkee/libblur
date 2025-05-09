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
#![allow(dead_code)]
use crate::filter1d::{Arena, ToApproxStorage};
use crate::filter2d::scan_point_2d::ScanPoint2d;
use crate::mlaf::mlaf;
use crate::ImageSize;
use num_traits::AsPrimitive;
use std::ops::Mul;

pub(crate) fn convolve_segment_2d_fp<T>(
    arena: Arena,
    arena_source: &[T],
    dst: &mut [T],
    image_size: ImageSize,
    prepared_kernel: &[ScanPoint2d<i16>],
    y: usize,
) where
    T: Copy + AsPrimitive<i32>,
    i32: ToApproxStorage<T>,
{
    unsafe {
        let width = image_size.width;

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

        let k_weight = prepared_kernel.get_unchecked(0).weight;

        while cx + 4 < total_width {
            let mut k0 = (*offsets.get_unchecked(0))
                .get_unchecked(cx)
                .as_()
                .mul(k_weight as i32);
            let mut k1 = (*offsets.get_unchecked(0))
                .get_unchecked(cx + 1)
                .as_()
                .mul(k_weight as i32);
            let mut k2 = (*offsets.get_unchecked(0))
                .get_unchecked(cx + 2)
                .as_()
                .mul(k_weight as i32);
            let mut k3 = (*offsets.get_unchecked(0).get_unchecked(cx + 3))
                .as_()
                .mul(k_weight as i32);

            for i in 1..length {
                let weight = prepared_kernel.get_unchecked(i).weight;
                k0 = mlaf(
                    k0,
                    offsets.get_unchecked(i).get_unchecked(cx).as_(),
                    weight as i32,
                );
                k1 = mlaf(
                    k1,
                    offsets.get_unchecked(i).get_unchecked(cx + 1).as_(),
                    weight as i32,
                );
                k2 = mlaf(
                    k2,
                    offsets.get_unchecked(i).get_unchecked(cx + 2).as_(),
                    weight as i32,
                );
                k3 = mlaf(
                    k3,
                    offsets.get_unchecked(i).get_unchecked(cx + 3).as_(),
                    weight as i32,
                );
            }

            *dst.get_unchecked_mut(cx) = k0.to_approx_();
            *dst.get_unchecked_mut(cx + 1) = k1.to_approx_();
            *dst.get_unchecked_mut(cx + 2) = k2.to_approx_();
            *dst.get_unchecked_mut(cx + 3) = k3.to_approx_();
            cx += 4;
        }

        for x in cx..total_width {
            let mut k0 = (*offsets.get_unchecked(0))
                .get_unchecked(x)
                .as_()
                .mul(k_weight as i32);

            for i in 1..length {
                let k_weight = prepared_kernel.get_unchecked(i).weight;
                k0 = mlaf(
                    k0,
                    offsets.get_unchecked(i).get_unchecked(x).as_(),
                    k_weight as i32,
                );
            }
            *dst.get_unchecked_mut(x) = k0.to_approx_();
        }
    }
}
