/*
 * // Copyright (c) Radzivon Bartoshyk 6/2025. All rights reserved.
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
#![allow(clippy::manual_clamp)]
use crate::bilateral::bp8::{BilateralStore, BilateralUnit};
use crate::{Arena, BilateralBlurParams};
use std::arch::aarch64::*;

pub(crate) struct BilateralExecutionUnitNeon<'a, const N: usize> {
    pub(crate) arena: Arena,
    pub(crate) params: BilateralBlurParams,
    pub(crate) store: &'a BilateralStore,
    pub(crate) src_width: usize,
}

#[inline(always)]
unsafe fn v_lut(v: uint16x8_t, lut: &[f32; 65536]) -> (float32x4_t, float32x4_t) {
    let v0 = vld1q_lane_f32::<0>(
        lut.get_unchecked(vgetq_lane_u16::<0>(v) as usize),
        vdupq_n_f32(0.),
    );
    let v1 = vld1q_lane_f32::<1>(lut.get_unchecked(vgetq_lane_u16::<1>(v) as usize), v0);
    let v2 = vld1q_lane_f32::<2>(lut.get_unchecked(vgetq_lane_u16::<2>(v) as usize), v1);
    let v3 = vld1q_lane_f32::<3>(lut.get_unchecked(vgetq_lane_u16::<3>(v) as usize), v2);

    let v4 = vld1q_lane_f32::<0>(
        lut.get_unchecked(vgetq_lane_u16::<4>(v) as usize),
        vdupq_n_f32(0.),
    );
    let v5 = vld1q_lane_f32::<1>(lut.get_unchecked(vgetq_lane_u16::<5>(v) as usize), v4);
    let v6 = vld1q_lane_f32::<2>(lut.get_unchecked(vgetq_lane_u16::<6>(v) as usize), v5);
    let v7 = vld1q_lane_f32::<3>(lut.get_unchecked(vgetq_lane_u16::<7>(v) as usize), v6);
    (v3, v7)
}

#[inline(always)]
unsafe fn replace_zeros_with_ones(input: float32x4_t) -> float32x4_t {
    let is_zero = vceqq_f32(input, vdupq_n_f32(0.0));
    vbslq_f32(is_zero, vdupq_n_f32(1.0), input)
}

impl<const N: usize> BilateralUnit<u8> for BilateralExecutionUnitNeon<'_, N> {
    fn execute(&self, a_src: &[u8], y: usize, dst_row: &mut [u8], src_row: &[u8]) {
        let sliced_range = &self.store.range[..self.params.kernel_size * self.params.kernel_size];
        let ss = &self.store.spatial;
        let useful_width = self.src_width * N;
        let a_stride = self.arena.width * self.arena.components;
        let mut dst_row = &mut dst_row[..useful_width];
        let mut src_row = &src_row[..useful_width];

        unsafe {
            for (wx, (dst, center)) in dst_row
                .chunks_exact_mut(16)
                .zip(src_row.chunks_exact(16))
                .enumerate()
            {
                let mut sum0 = vdupq_n_f32(0.);
                let mut sum1 = vdupq_n_f32(0.);
                let mut sum2 = vdupq_n_f32(0.);
                let mut sum3 = vdupq_n_f32(0.);

                let mut iw0 = vdupq_n_f32(0.);
                let mut iw1 = vdupq_n_f32(0.);
                let mut iw2 = vdupq_n_f32(0.);
                let mut iw3 = vdupq_n_f32(0.);

                let cxz = vld1q_u8(center.as_ptr().cast());
                let cx0 = vshll_n_u8::<8>(vget_low_u8(cxz));
                let cx1 = vshll_high_n_u8::<8>(cxz);

                let x = wx * 16;

                for (ky, ky_row) in sliced_range
                    .chunks_exact(self.params.kernel_size)
                    .enumerate()
                {
                    let c_slice = (y + ky) * a_stride + x;

                    for (w, &rwz) in ky_row.iter().enumerate() {
                        let v_rwz = vdupq_n_f32(rwz);
                        let pxz = vld1q_u8(
                            a_src
                                .get_unchecked(c_slice + w * N..c_slice + w * N + 16)
                                .as_ptr()
                                .cast(),
                        );

                        let px16l = vmovl_u8(vget_low_u8(pxz));
                        let px16h = vmovl_high_u8(pxz);

                        let px0_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(px16l)));
                        let px0_hi = vcvtq_f32_u32(vmovl_high_u16(px16l));
                        let px1_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(px16h)));
                        let px1_hi = vcvtq_f32_u32(vmovl_high_u16(px16h));

                        let w0 = vaddq_u16(cx0, px16l);
                        let w1 = vaddq_u16(cx1, px16h);

                        let (mut z0, mut z1) = v_lut(w0, ss);
                        let (mut z2, mut z3) = v_lut(w1, ss);

                        z0 = vmulq_f32(z0, v_rwz);
                        z1 = vmulq_f32(z1, v_rwz);
                        z2 = vmulq_f32(z2, v_rwz);
                        z3 = vmulq_f32(z3, v_rwz);

                        sum0 = vfmaq_f32(sum0, z0, px0_lo);
                        sum1 = vfmaq_f32(sum1, z1, px0_hi);
                        sum2 = vfmaq_f32(sum2, z2, px1_lo);
                        sum3 = vfmaq_f32(sum3, z3, px1_hi);

                        iw0 = vaddq_f32(iw0, z0);
                        iw1 = vaddq_f32(iw1, z1);
                        iw2 = vaddq_f32(iw2, z2);
                        iw3 = vaddq_f32(iw3, z3);
                    }
                }

                iw0 = replace_zeros_with_ones(iw0);
                iw1 = replace_zeros_with_ones(iw1);
                iw2 = replace_zeros_with_ones(iw2);
                iw3 = replace_zeros_with_ones(iw3);

                let s0 = vcvtaq_u32_f32(vdivq_f32(sum0, iw0));
                let s1 = vcvtaq_u32_f32(vdivq_f32(sum1, iw1));
                let s2 = vcvtaq_u32_f32(vdivq_f32(sum2, iw2));
                let s3 = vcvtaq_u32_f32(vdivq_f32(sum3, iw3));

                let ss0 = vmovn_u32(s0);
                let ss1 = vmovn_u32(s1);
                let ss2 = vmovn_u32(s2);
                let ss3 = vmovn_u32(s3);

                let vss0 = vqmovn_u16(vcombine_u16(ss0, ss1));
                let vss1 = vqmovn_u16(vcombine_u16(ss2, ss3));
                vst1q_u8(dst.as_mut_ptr().cast(), vcombine_u8(vss0, vss1));
            }

            dst_row = dst_row.chunks_exact_mut(16).into_remainder();
            src_row = src_row.chunks_exact(16).remainder();
        }

        unsafe {
            for (wx, (dst, center)) in dst_row
                .chunks_exact_mut(8)
                .zip(src_row.chunks_exact(8))
                .enumerate()
            {
                let mut sum0 = vdupq_n_f32(0.);
                let mut sum1 = vdupq_n_f32(0.);

                let mut iw0 = vdupq_n_f32(0.);
                let mut iw1 = vdupq_n_f32(0.);

                let cx = vshll_n_u8::<8>(vld1_u8(center.as_ptr().cast()));

                let x = wx * 8;

                for (ky, ky_row) in sliced_range
                    .chunks_exact(self.params.kernel_size)
                    .enumerate()
                {
                    let c_slice = (y + ky) * a_stride + x;

                    for (w, &rwz) in ky_row.iter().enumerate() {
                        let v_rwz = vdupq_n_f32(rwz);
                        let px = vld1_u8(
                            a_src
                                .get_unchecked(c_slice + w * N..c_slice + w * N + 8)
                                .as_ptr()
                                .cast(),
                        );

                        let px16 = vmovl_u8(px);
                        let px_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(px16)));
                        let px_hi = vcvtq_f32_u32(vmovl_high_u16(px16));

                        let w0 = vaddq_u16(cx, px16);
                        let (mut z0, mut z1) = v_lut(w0, ss);

                        z0 = vmulq_f32(z0, v_rwz);
                        z1 = vmulq_f32(z1, v_rwz);

                        sum0 = vfmaq_f32(sum0, z0, px_lo);
                        sum1 = vfmaq_f32(sum1, z1, px_hi);

                        iw0 = vaddq_f32(iw0, z0);
                        iw1 = vaddq_f32(iw1, z1);
                    }
                }

                iw0 = replace_zeros_with_ones(iw0);
                iw1 = replace_zeros_with_ones(iw1);

                let s0 = vcvtaq_u32_f32(vdivq_f32(sum0, iw0));
                let s1 = vcvtaq_u32_f32(vdivq_f32(sum1, iw1));

                let ss0 = vmovn_u32(s0);
                let ss1 = vmovn_u32(s1);

                let vss0 = vqmovn_u16(vcombine_u16(ss0, ss1));
                vst1_u8(dst.as_mut_ptr().cast(), vss0);
            }
        }

        let dst_row = dst_row.chunks_exact_mut(8).into_remainder();
        let src_row = src_row.chunks_exact(8).remainder();

        for (x, (dst, &center)) in dst_row.iter_mut().zip(src_row.iter()).enumerate() {
            let mut sum0 = 0f32;
            let mut iw0 = 0f32;

            let sx = x % N;
            for (ky, ky_row) in sliced_range
                .chunks_exact(self.params.kernel_size)
                .enumerate()
            {
                let c_slice = (y + ky) * a_stride + x - sx;
                let c_px_slice =
                    &a_src[(c_slice + sx)..(c_slice + N * self.params.kernel_size - sx)];
                for (&c_px, &rwz) in c_px_slice.iter().step_by(N).zip(ky_row.iter()) {
                    let z0 = rwz * ss[(center as u16 * 256 + c_px as u16) as usize];
                    sum0 += z0 * c_px as f32;
                    iw0 += z0;
                }
            }

            iw0 = if iw0 == 0. { 1. } else { iw0 };

            *dst = (sum0 / iw0).round().min(255.).max(0.) as u8;
        }
    }
}
