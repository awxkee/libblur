/*
 * // Copyright (c) Radzivon Bartoshyk 5/2025. All rights reserved.
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
#![allow(dead_code)]
use crate::edge_mode::{border_interpolate, BorderHandle};
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::to_approx_storage::ToApproxStorage;
use crate::img_size::ImageSize;
use num_traits::AsPrimitive;
use std::ops::{Add, Mul, Shr};

pub(crate) fn filter_row_symmetric_approx_binter<T, I, const N: usize>(
    edge_mode: BorderHandle,
    src: &[T],
    dst: &mut [T],
    image_size: ImageSize,
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
    i32: AsPrimitive<I>,
    f64: AsPrimitive<T>,
{
    unsafe {
        let width = image_size.width;

        let length = scanned_kernel.len();
        let half_len = length / 2;

        let s_kernel = half_len as i64;

        let mut cx = 0usize;

        let min_left = half_len.min(width);

        let width = width as i64;

        let coeff = *scanned_kernel.get_unchecked(half_len);

        while cx < min_left {
            for c in 0..N {
                let mx = cx as i64 - s_kernel;
                let mut k0 = src.get_unchecked(cx * N + c).as_() * coeff.weight;

                for i in 0..half_len {
                    let coeff = *scanned_kernel.get_unchecked(i);
                    let rollback = length - i - 1;

                    let src0 = border_interpolate!(src, edge_mode, i as i64 + mx, 0, width, N, c);
                    let src1 =
                        border_interpolate!(src, edge_mode, rollback as i64 + mx, 0, width, N, c);

                    k0 = k0 + (src0.as_() + src1.as_()) * coeff.weight;
                }

                *dst.get_unchecked_mut(cx * N + c) = k0.to_approx_();
            }
            cx += 1;
        }

        // Flat region

        let m_right = (width as usize).saturating_sub(half_len);
        let flat_max = m_right * N;

        let mut f_cx = cx * N;

        while f_cx + 4 < flat_max {
            let mx = f_cx - half_len * N;
            let shifted_src = src.get_unchecked(mx..);
            let mut k0 = shifted_src.get_unchecked(half_len * N).as_() * coeff.weight;
            let mut k1 = shifted_src.get_unchecked(half_len * N + 1).as_() * coeff.weight;
            let mut k2 = shifted_src.get_unchecked(half_len * N + 2).as_() * coeff.weight;
            let mut k3 = shifted_src.get_unchecked(half_len * N + 3).as_() * coeff.weight;

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;

                k0 = k0
                    + (shifted_src.get_unchecked(i * N).as_()
                        + shifted_src.get_unchecked(rollback * N).as_())
                        * coeff.weight;

                k1 = k1
                    + (shifted_src.get_unchecked(i * N + 1).as_()
                        + shifted_src.get_unchecked(rollback * N + 1).as_())
                        * coeff.weight;

                k2 = k2
                    + (shifted_src.get_unchecked(i * N + 2).as_()
                        + shifted_src.get_unchecked(rollback * N + 2).as_())
                        * coeff.weight;

                k3 = k3
                    + (shifted_src.get_unchecked(i * N + 3).as_()
                        + shifted_src.get_unchecked(rollback * N + 3).as_())
                        * coeff.weight;
            }

            *dst.get_unchecked_mut(f_cx) = k0.to_approx_();
            *dst.get_unchecked_mut(f_cx + 1) = k1.to_approx_();
            *dst.get_unchecked_mut(f_cx + 2) = k2.to_approx_();
            *dst.get_unchecked_mut(f_cx + 3) = k3.to_approx_();
            f_cx += 4;
        }

        while f_cx < flat_max {
            let mx = f_cx - half_len * N;
            let shifted_src = src.get_unchecked(mx..);
            let mut k0 = shifted_src.get_unchecked(half_len * N).as_() * coeff.weight;

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;

                k0 = k0
                    + (shifted_src.get_unchecked(i * N).as_()
                        + shifted_src.get_unchecked(rollback * N).as_())
                        * coeff.weight;
            }

            *dst.get_unchecked_mut(f_cx) = k0.to_approx_();
            f_cx += 1;
        }

        f_cx = m_right;

        while f_cx < width as usize {
            for c in 0..N {
                let coeff = *scanned_kernel.get_unchecked(half_len);
                let mx = f_cx as i64 - s_kernel;
                let mut k0 = src.get_unchecked(f_cx * N + c).as_() * coeff.weight;

                for i in 0..half_len {
                    let coeff = *scanned_kernel.get_unchecked(i);
                    let rollback = length - i - 1;

                    let src0 = border_interpolate!(src, edge_mode, i as i64 + mx, 0, width, N, c);
                    let src1 =
                        border_interpolate!(src, edge_mode, rollback as i64 + mx, 0, width, N, c);

                    k0 = k0 + (src0.as_() + src1.as_()) * coeff.weight;
                }

                *dst.get_unchecked_mut(f_cx * N + c) = k0.to_approx_();
            }
            f_cx += 1;
        }
    }
}
