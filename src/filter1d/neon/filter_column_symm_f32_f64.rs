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
use crate::filter1d::neon::utils::{xvld1q_f32_x2, xvld1q_f32_x4, xvst1q_f32_x2, xvst1q_f32_x4};
use crate::filter1d::region::FilterRegion;
use crate::img_size::ImageSize;
use std::arch::aarch64::*;

pub(crate) fn filter_column_neon_symm_f32_f64(
    arena: Arena,
    arena_src: &[&[f32]],
    dst: &mut [f32],
    image_size: ImageSize,
    _: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f64>],
) {
    unsafe {
        let dst_stride = image_size.width * arena.components;

        let length = scanned_kernel.len();
        let half_len = length / 2;

        let mut cx = 0usize;

        let coeff = vdupq_n_f64(scanned_kernel.get_unchecked(half_len).weight);

        while cx + 16 < dst_stride {
            let v_src = arena_src.get_unchecked(half_len).get_unchecked(cx..);

            let source = xvld1q_f32_x4(v_src.as_ptr());

            let mut k0 = vmulq_f64(vcvt_f64_f32(vget_low_f32(source.0)), coeff);
            let mut k1 = vmulq_f64(vcvt_high_f64_f32(source.0), coeff);
            let mut k2 = vmulq_f64(vcvt_f64_f32(vget_low_f32(source.1)), coeff);
            let mut k3 = vmulq_f64(vcvt_high_f64_f32(source.1), coeff);
            let mut k4 = vmulq_f64(vcvt_f64_f32(vget_low_f32(source.2)), coeff);
            let mut k5 = vmulq_f64(vcvt_high_f64_f32(source.2), coeff);
            let mut k6 = vmulq_f64(vcvt_f64_f32(vget_low_f32(source.3)), coeff);
            let mut k7 = vmulq_f64(vcvt_high_f64_f32(source.3), coeff);

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = vdupq_n_f64(scanned_kernel.get_unchecked(i).weight);
                let v_source0 =
                    xvld1q_f32_x4(arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1 = xvld1q_f32_x4(
                    arena_src
                        .get_unchecked(rollback)
                        .get_unchecked(cx..)
                        .as_ptr(),
                );
                k0 = vfmaq_f64(
                    k0,
                    vaddq_f64(
                        vcvt_f64_f32(vget_low_f32(v_source0.0)),
                        vcvt_f64_f32(vget_low_f32(v_source1.0)),
                    ),
                    coeff,
                );
                k1 = vfmaq_f64(
                    k1,
                    vaddq_f64(
                        vcvt_high_f64_f32(v_source0.0),
                        vcvt_high_f64_f32(v_source1.0),
                    ),
                    coeff,
                );
                k2 = vfmaq_f64(
                    k2,
                    vaddq_f64(
                        vcvt_f64_f32(vget_low_f32(v_source0.1)),
                        vcvt_f64_f32(vget_low_f32(v_source1.1)),
                    ),
                    coeff,
                );
                k3 = vfmaq_f64(
                    k3,
                    vaddq_f64(
                        vcvt_high_f64_f32(v_source0.1),
                        vcvt_high_f64_f32(v_source1.1),
                    ),
                    coeff,
                );
                k4 = vfmaq_f64(
                    k4,
                    vaddq_f64(
                        vcvt_f64_f32(vget_low_f32(v_source0.2)),
                        vcvt_f64_f32(vget_low_f32(v_source1.2)),
                    ),
                    coeff,
                );
                k5 = vfmaq_f64(
                    k5,
                    vaddq_f64(
                        vcvt_high_f64_f32(v_source0.2),
                        vcvt_high_f64_f32(v_source1.2),
                    ),
                    coeff,
                );
                k6 = vfmaq_f64(
                    k6,
                    vaddq_f64(
                        vcvt_f64_f32(vget_low_f32(v_source0.3)),
                        vcvt_f64_f32(vget_low_f32(v_source1.3)),
                    ),
                    coeff,
                );
                k7 = vfmaq_f64(
                    k7,
                    vaddq_f64(
                        vcvt_high_f64_f32(v_source0.3),
                        vcvt_high_f64_f32(v_source1.3),
                    ),
                    coeff,
                );
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            xvst1q_f32_x4(
                dst_ptr0,
                float32x4x4_t(
                    vcombine_f32(vcvt_f32_f64(k0), vcvt_f32_f64(k1)),
                    vcombine_f32(vcvt_f32_f64(k2), vcvt_f32_f64(k3)),
                    vcombine_f32(vcvt_f32_f64(k4), vcvt_f32_f64(k5)),
                    vcombine_f32(vcvt_f32_f64(k6), vcvt_f32_f64(k7)),
                ),
            );
            cx += 16;
        }

        while cx + 8 < dst_stride {
            let v_src = arena_src.get_unchecked(half_len).get_unchecked(cx..);

            let source = xvld1q_f32_x2(v_src.as_ptr());
            let mut k0 = vmulq_f64(vcvt_f64_f32(vget_low_f32(source.0)), coeff);
            let mut k1 = vmulq_f64(vcvt_high_f64_f32(source.0), coeff);
            let mut k2 = vmulq_f64(vcvt_f64_f32(vget_low_f32(source.1)), coeff);
            let mut k3 = vmulq_f64(vcvt_high_f64_f32(source.1), coeff);

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = vdupq_n_f64(scanned_kernel.get_unchecked(i).weight);
                let v_source0 =
                    xvld1q_f32_x2(arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1 = xvld1q_f32_x2(
                    arena_src
                        .get_unchecked(rollback)
                        .get_unchecked(cx..)
                        .as_ptr(),
                );
                k0 = vfmaq_f64(
                    k0,
                    vaddq_f64(
                        vcvt_f64_f32(vget_low_f32(v_source0.0)),
                        vcvt_f64_f32(vget_low_f32(v_source1.0)),
                    ),
                    coeff,
                );
                k1 = vfmaq_f64(
                    k1,
                    vaddq_f64(
                        vcvt_high_f64_f32(v_source0.0),
                        vcvt_high_f64_f32(v_source1.0),
                    ),
                    coeff,
                );
                k2 = vfmaq_f64(
                    k2,
                    vaddq_f64(
                        vcvt_f64_f32(vget_low_f32(v_source0.1)),
                        vcvt_f64_f32(vget_low_f32(v_source1.1)),
                    ),
                    coeff,
                );
                k3 = vfmaq_f64(
                    k3,
                    vaddq_f64(
                        vcvt_high_f64_f32(v_source0.1),
                        vcvt_high_f64_f32(v_source1.1),
                    ),
                    coeff,
                );
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            xvst1q_f32_x2(
                dst_ptr0,
                float32x4x2_t(
                    vcombine_f32(vcvt_f32_f64(k0), vcvt_f32_f64(k1)),
                    vcombine_f32(vcvt_f32_f64(k2), vcvt_f32_f64(k3)),
                ),
            );

            cx += 8;
        }

        while cx + 4 < dst_stride {
            let v_src = arena_src.get_unchecked(half_len).get_unchecked(cx..);

            let source_0 = vld1q_f32(v_src.as_ptr());
            let mut k0 = vmulq_f64(vcvt_f64_f32(vget_low_f32(source_0)), coeff);
            let mut k1 = vmulq_f64(vcvt_high_f64_f32(source_0), coeff);

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = vdupq_n_f64(scanned_kernel.get_unchecked(i).weight);
                let v_source_0 = vld1q_f32(arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source_1 = vld1q_f32(
                    arena_src
                        .get_unchecked(rollback)
                        .get_unchecked(cx..)
                        .as_ptr(),
                );
                k0 = vfmaq_f64(
                    k0,
                    vaddq_f64(
                        vcvt_f64_f32(vget_low_f32(v_source_0)),
                        vcvt_f64_f32(vget_low_f32(v_source_1)),
                    ),
                    coeff,
                );
                k1 = vfmaq_f64(
                    k1,
                    vaddq_f64(vcvt_high_f64_f32(v_source_0), vcvt_high_f64_f32(v_source_1)),
                    coeff,
                );
            }

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            vst1q_f32(dst_ptr0, vcombine_f32(vcvt_f32_f64(k0), vcvt_f32_f64(k1)));
            cx += 4;
        }

        while cx + 2 < dst_stride {
            let v_src = arena_src.get_unchecked(half_len).get_unchecked(cx..);

            let source_0 = vld1_f32(v_src.as_ptr());
            let mut k0 = vmulq_f64(vcvt_f64_f32(source_0), coeff);

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = vdupq_n_f64(scanned_kernel.get_unchecked(i).weight);
                let v_source_0 = vld1_f32(arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source_1 = vld1_f32(
                    arena_src
                        .get_unchecked(rollback)
                        .get_unchecked(cx..)
                        .as_ptr(),
                );
                k0 = vfmaq_f64(
                    k0,
                    vaddq_f64(vcvt_f64_f32(v_source_0), vcvt_f64_f32(v_source_1)),
                    coeff,
                );
            }

            let dst_ptr = dst.get_unchecked_mut(cx..).as_mut_ptr();
            vst1_f32(dst_ptr, vcvt_f32_f64(k0));
            cx += 2;
        }

        for x in cx..dst_stride {
            let v_src = arena_src.get_unchecked(half_len).get_unchecked(x..);

            let mut k0 = vmulq_f64(
                vcvt_f64_f32(vld1_lane_f32::<0>(v_src.as_ptr(), vdup_n_f32(0.))),
                coeff,
            );

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = vdupq_n_f64(scanned_kernel.get_unchecked(i).weight);

                let z0 = vld1_lane_f32::<0>(
                    arena_src.get_unchecked(i).get_unchecked(x..).as_ptr(),
                    vdup_n_f32(0.),
                );
                let z1 = vld1_lane_f32::<0>(
                    arena_src
                        .get_unchecked(rollback)
                        .get_unchecked(x..)
                        .as_ptr(),
                    vdup_n_f32(0.),
                );

                k0 = vfmaq_f64(k0, vaddq_f64(vcvt_f64_f32(z0), vcvt_f64_f32(z1)), coeff);
            }

            vst1_lane_f32::<0>(dst.get_unchecked_mut(x..).as_mut_ptr(), vcvt_f32_f64(k0));
        }
    }
}
