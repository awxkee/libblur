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
use crate::img_size::ImageSize;
use num_complex::Complex;
use std::arch::aarch64::*;

#[inline(always)]
pub(crate) unsafe fn vq_complex_mla_q(
    acc_r: ((int32x4_t, int32x4_t), (int32x4_t, int32x4_t)),
    r0: int16x8_t,
    i0: int16x8_t,
    r1: int16x8_t,
    i1: int16x8_t,
) -> ((int32x4_t, int32x4_t), (int32x4_t, int32x4_t)) {
    let acc0 = acc_r.0;
    let acc1 = acc_r.1;
    let re0 = vmlsl_s16(
        vmlal_s16(acc0.0, vget_low_s16(r0), vget_low_s16(r1)),
        vget_low_s16(i0),
        vget_low_s16(i1),
    );
    let im0 = vmlal_s16(
        vmlal_s16(acc0.1, vget_low_s16(r0), vget_low_s16(i1)),
        vget_low_s16(i0),
        vget_low_s16(r1),
    );
    let re1 = vmlsl_high_s16(vmlal_high_s16(acc1.0, r0, r1), i0, i1);
    let im1 = vmlal_high_s16(vmlal_high_s16(acc1.1, r0, i1), i0, r1);
    ((re0, im0), (re1, im1))
}

#[inline(always)]
pub(crate) unsafe fn v_complex_mla_q(
    acc: (int32x4_t, int32x4_t),
    r0: int16x4_t,
    i0: int16x4_t,
    r1: int16x4_t,
    i1: int16x4_t,
) -> (int32x4_t, int32x4_t) {
    let re0 = vmlsl_s16(vmlal_s16(acc.0, r0, r1), i0, i1);
    let im0 = vmlal_s16(vmlal_s16(acc.1, r0, i1), i0, r1);
    (re0, im0)
}

#[inline(always)]
pub(crate) unsafe fn vq_complex_mul_q(
    r0: int16x8_t,
    i0: int16x8_t,
    r1: int16x8_t,
    i1: int16x8_t,
) -> ((int32x4_t, int32x4_t), (int32x4_t, int32x4_t)) {
    let re0 = vmlsl_s16(
        vmull_s16(vget_low_s16(r0), vget_low_s16(r1)),
        vget_low_s16(i0),
        vget_low_s16(i1),
    );
    let im0 = vmlal_s16(
        vmull_s16(vget_low_s16(r0), vget_low_s16(i1)),
        vget_low_s16(i0),
        vget_low_s16(r1),
    );
    let re1 = vmlsl_high_s16(vmull_high_s16(r0, r1), i0, i1);
    let im1 = vmlal_high_s16(vmull_high_s16(r0, i1), i0, r1);
    ((re0, im0), (re1, im1))
}

#[inline(always)]
pub(crate) unsafe fn v_complex_mul_q(
    r0: int16x4_t,
    i0: int16x4_t,
    r1: int16x4_t,
    i1: int16x4_t,
) -> (int32x4_t, int32x4_t) {
    let re0 = vmlsl_s16(vmull_s16(r0, r1), i0, i1);
    let im0 = vmlal_s16(vmull_s16(r0, i1), i0, r1);
    (re0, im0)
}

pub(crate) fn filter_column_complex_u8_i32(
    arena: Arena,
    arena_src: &[&[Complex<i16>]],
    dst: &mut [u8],
    image_size: ImageSize,
    kernel: &[Complex<i16>],
) {
    unsafe {
        let full_width = image_size.width * arena.components;

        let length = kernel.len();

        let mut cx = 0usize;

        let c_re = vdupq_n_s16(kernel.get_unchecked(0).re);
        let c_im = vdupq_n_s16(kernel.get_unchecked(0).im);

        while cx + 16 < full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(cx..);

            let values0 = vld2q_s16(v_src.as_ptr().cast());
            let values1 = vld2q_s16(v_src.get_unchecked(8..).as_ptr().cast());

            let (mut k0, mut k1) = vq_complex_mul_q(values0.0, values0.1, c_re, c_im);
            let (mut k2, mut k3) = vq_complex_mul_q(values1.0, values1.1, c_re, c_im);

            for i in 1..length {
                let coeff = *kernel.get_unchecked(i);

                let c_re = vdupq_n_s16(coeff.re);
                let c_im = vdupq_n_s16(coeff.im);

                let values0 = vld2q_s16(
                    arena_src
                        .get_unchecked(i)
                        .get_unchecked(cx..)
                        .as_ptr()
                        .cast(),
                );

                let values1 = vld2q_s16(
                    arena_src
                        .get_unchecked(i)
                        .get_unchecked(cx + 8..)
                        .as_ptr()
                        .cast(),
                );

                (k0, k1) = vq_complex_mla_q((k0, k1), values0.0, values0.1, c_re, c_im);
                (k2, k3) = vq_complex_mla_q((k2, k3), values1.0, values1.1, c_re, c_im);
            }

            let real0 = vcombine_u16(vqrshrun_n_s32::<15>(k0.0), vqrshrun_n_s32::<15>(k1.0));
            let real1 = vcombine_u16(vqrshrun_n_s32::<15>(k2.0), vqrshrun_n_s32::<15>(k3.0));

            let real = vcombine_u8(vqmovn_u16(real0), vqmovn_u16(real1));

            vst1q_u8(dst.get_unchecked_mut(cx..).as_mut_ptr().cast(), real);
            cx += 16;
        }

        while cx + 8 < full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(cx..);

            let values0 = vld2q_s16(v_src.as_ptr().cast());

            let (mut k0, mut k1) = vq_complex_mul_q(values0.0, values0.1, c_re, c_im);

            for i in 1..length {
                let coeff = *kernel.get_unchecked(i);

                let c_re = vdupq_n_s16(coeff.re);
                let c_im = vdupq_n_s16(coeff.im);

                let values0 = vld2q_s16(
                    arena_src
                        .get_unchecked(i)
                        .get_unchecked(cx..)
                        .as_ptr()
                        .cast(),
                );

                (k0, k1) = vq_complex_mla_q((k0, k1), values0.0, values0.1, c_re, c_im);
            }

            let p0 = vqrshrun_n_s32::<15>(k0.0);
            let p1 = vqrshrun_n_s32::<15>(k1.0);

            vst1_u8(
                dst.get_unchecked_mut(cx..).as_mut_ptr().cast(),
                vqmovn_u16(vcombine_u16(p0, p1)),
            );
            cx += 8;
        }

        while cx + 4 < full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(cx..);

            let values = vld2_s16(v_src.as_ptr().cast());

            let mut k0 =
                v_complex_mul_q(values.0, values.1, vget_low_s16(c_re), vget_low_s16(c_im));

            for i in 1..length {
                let coeff = *kernel.get_unchecked(i);

                let c_re = vdup_n_s16(coeff.re);
                let c_im = vdup_n_s16(coeff.im);
                let ss0 = arena_src.get_unchecked(i);

                let r0 = vld2_s16(ss0.get_unchecked(cx..).as_ptr().cast());

                k0 = v_complex_mla_q(k0, r0.0, r0.1, c_re, c_im);
            }

            let p0 = vqrshrun_n_s32::<15>(k0.0);

            vst1_lane_u32::<0>(
                dst.get_unchecked_mut(cx..).as_mut_ptr().cast(),
                vreinterpret_u32_u8(vqmovn_u16(vcombine_u16(p0, p0))),
            );
            cx += 4;
        }

        for x in cx..full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(x..);

            let r0 = vdup_n_s16(v_src[0].re);
            let i0 = vdup_n_s16(v_src[0].im);

            let mut k0 = v_complex_mul_q(r0, i0, vget_low_s16(c_re), vget_low_s16(c_im));

            for i in 1..length {
                let coeff = *kernel.get_unchecked(i);

                let c_re = vdup_n_s16(coeff.re);
                let c_im = vdup_n_s16(coeff.im);

                let ss = arena_src.get_unchecked(i).get_unchecked(x);

                let r0 = vdup_n_s16(ss.re);
                let i0 = vdup_n_s16(ss.im);

                k0 = v_complex_mla_q(k0, r0, i0, c_re, c_im);
            }

            let k0 = vqrshrun_n_s32::<15>(k0.0);
            vst1_lane_u8::<0>(dst.get_unchecked_mut(x), vqmovn_u16(vcombine_u16(k0, k0)));
        }
    }
}
