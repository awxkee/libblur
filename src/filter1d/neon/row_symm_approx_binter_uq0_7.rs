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
use crate::edge_mode::border_interpolate;
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::neon::utils::{
    xvld1q_u8_x2, xvld1q_u8_x3, xvld4u8, xvst1q_u8_x2, xvst1q_u8_x3,
};
use crate::filter1d::row_handler_small_approx::{RowsHolder, RowsHolderMut};
use crate::img_size::ImageSize;
use crate::BorderHandle;
use std::arch::aarch64::*;

pub(crate) fn filter_row_symm_neon_binter_u8_u0_7<const N: usize>(
    edge_mode: BorderHandle,
    m_src: &RowsHolder<u8>,
    m_dst: &mut RowsHolderMut<u8>,
    image_size: ImageSize,
    scanned_kernel: &[ScanPoint1d<i32>],
) {
    let mut shifted = scanned_kernel
        .iter()
        .map(|&x| ((x.weight) >> 8) as u8)
        .collect::<Vec<_>>();
    let mut sum: u32 = shifted.iter().map(|&x| x as u32).sum();
    if sum > 128 {
        let half = shifted.len() / 2;
        while sum > 128 {
            shifted[half] = shifted[half].saturating_sub(1);
            sum -= 1;
        }
    } else if sum < 128 {
        let half = shifted.len() / 2;
        while sum < 128 {
            shifted[half] = shifted[half].saturating_add(1);
            sum += 1;
        }
    }
    executor_unit::<N>(edge_mode, m_src, m_dst, image_size, &shifted);
}

#[inline(always)]
pub(crate) unsafe fn xvst_u8x4_q7(a: *mut u8, v: uint16x8_t) {
    let shifted = vqrshrn_n_u16::<7>(v);
    vst1_lane_u32::<0>(a as *mut _, vreinterpret_u32_u8(shifted));
}

fn executor_unit<const N: usize>(
    edge_mode: BorderHandle,
    m_src: &RowsHolder<u8>,
    m_dst: &mut RowsHolderMut<u8>,
    image_size: ImageSize,
    scanned_kernel: &[u8],
) {
    unsafe {
        let width = image_size.width;
        let length = scanned_kernel.len();
        let half_len = length / 2;

        let min_left = half_len.min(width);
        let s_kernel = half_len as i64;

        for (src, dst) in m_src.holder.iter().zip(m_dst.holder.iter_mut()) {
            let mut f_cx = 0usize;

            let coeff = *scanned_kernel.get_unchecked(half_len);

            while f_cx < min_left {
                for c in 0..N {
                    let mx = f_cx as i64 - s_kernel;
                    let mut k0: u16 = *src.get_unchecked(f_cx * N + c) as u16 * coeff as u16;

                    for i in 0..half_len {
                        let coeff = *scanned_kernel.get_unchecked(i);
                        let rollback = length - i - 1;

                        let src0 = border_interpolate!(
                            src,
                            edge_mode,
                            i as i64 + mx,
                            0,
                            width as i64,
                            N,
                            c
                        );
                        let src1 = border_interpolate!(
                            src,
                            edge_mode,
                            rollback as i64 + mx,
                            0,
                            width as i64,
                            N,
                            c
                        );

                        k0 += (src0 as u16 + src1 as u16) * coeff as u16;
                    }

                    *dst.get_unchecked_mut(f_cx * N + c) = ((k0 + (1 << 6)) >> 7).min(255) as u8;
                }
                f_cx += 1;
            }

            let mut m_cx = f_cx * N;

            let s_half = half_len * N;
            let m_right = width.saturating_sub(half_len);
            let max_width = m_right * N;

            let coeff = vdupq_n_u8(*scanned_kernel.get_unchecked(half_len));

            while m_cx + 48 < max_width {
                let cx = m_cx - s_half;
                let shifted_src = src.get_unchecked(cx..);

                let source = xvld1q_u8_x3(shifted_src.get_unchecked(half_len * N..).as_ptr());
                let mut k0 = vmull_u8(vget_low_u8(source.0), vget_low_u8(coeff));
                let mut k1 = vmull_high_u8(source.0, coeff);
                let mut k2 = vmull_u8(vget_low_u8(source.1), vget_low_u8(coeff));
                let mut k3 = vmull_high_u8(source.1, coeff);
                let mut k4 = vmull_u8(vget_low_u8(source.2), vget_low_u8(coeff));
                let mut k5 = vmull_high_u8(source.2, coeff);

                for i in 0..half_len {
                    let coeff = vdupq_n_u16(*scanned_kernel.get_unchecked(i) as u16);
                    let rollback = length - i - 1;
                    let v_source0 = xvld1q_u8_x3(shifted_src.get_unchecked((i * N)..).as_ptr());
                    let v_source1 =
                        xvld1q_u8_x3(shifted_src.get_unchecked((rollback * N)..).as_ptr());
                    k0 = vmlaq_u16(
                        k0,
                        vaddl_u8(vget_low_u8(v_source0.0), vget_low_u8(v_source1.0)),
                        coeff,
                    );
                    k1 = vmlaq_u16(k1, vaddl_high_u8(v_source0.0, v_source1.0), coeff);

                    k2 = vmlaq_u16(
                        k2,
                        vaddl_u8(vget_low_u8(v_source0.1), vget_low_u8(v_source1.1)),
                        coeff,
                    );
                    k3 = vmlaq_u16(k3, vaddl_high_u8(v_source0.1, v_source1.1), coeff);

                    k4 = vmlaq_u16(
                        k4,
                        vaddl_u8(vget_low_u8(v_source0.2), vget_low_u8(v_source1.2)),
                        coeff,
                    );
                    k5 = vmlaq_u16(k5, vaddl_high_u8(v_source0.2, v_source1.2), coeff);
                }

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                xvst1q_u8_x3(
                    dst_ptr0,
                    uint8x16x3_t(
                        vcombine_u8(vqshrn_n_u16::<7>(k0), vqshrn_n_u16::<7>(k1)),
                        vcombine_u8(vqshrn_n_u16::<7>(k2), vqshrn_n_u16::<7>(k3)),
                        vcombine_u8(vqshrn_n_u16::<7>(k4), vqshrn_n_u16::<7>(k5)),
                    ),
                );
                m_cx += 48;
            }

            while m_cx + 32 < max_width {
                let cx = m_cx - s_half;
                let shifted_src = src.get_unchecked(cx..);

                let source = xvld1q_u8_x2(shifted_src.get_unchecked(half_len * N..).as_ptr());
                let mut k0 = vmull_u8(vget_low_u8(source.0), vget_low_u8(coeff));
                let mut k1 = vmull_high_u8(source.0, coeff);
                let mut k2 = vmull_u8(vget_low_u8(source.1), vget_low_u8(coeff));
                let mut k3 = vmull_high_u8(source.1, coeff);

                for i in 0..half_len {
                    let coeff = vdupq_n_u16(*scanned_kernel.get_unchecked(i) as u16);
                    let rollback = length - i - 1;
                    let v_source0 = xvld1q_u8_x2(shifted_src.get_unchecked((i * N)..).as_ptr());
                    let v_source1 =
                        xvld1q_u8_x2(shifted_src.get_unchecked((rollback * N)..).as_ptr());
                    k0 = vmlaq_u16(
                        k0,
                        vaddl_u8(vget_low_u8(v_source0.0), vget_low_u8(v_source1.0)),
                        coeff,
                    );
                    k1 = vmlaq_u16(k1, vaddl_high_u8(v_source0.0, v_source1.0), coeff);

                    k2 = vmlaq_u16(
                        k2,
                        vaddl_u8(vget_low_u8(v_source0.1), vget_low_u8(v_source1.1)),
                        coeff,
                    );
                    k3 = vmlaq_u16(k3, vaddl_high_u8(v_source0.1, v_source1.1), coeff);
                }

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                xvst1q_u8_x2(
                    dst_ptr0,
                    uint8x16x2_t(
                        vcombine_u8(vqshrn_n_u16::<7>(k0), vqshrn_n_u16::<7>(k1)),
                        vcombine_u8(vqshrn_n_u16::<7>(k2), vqshrn_n_u16::<7>(k3)),
                    ),
                );
                m_cx += 32;
            }

            while m_cx + 16 < max_width {
                let cx = m_cx - s_half;
                let shifted_src = src.get_unchecked(cx..);

                let source = vld1q_u8(shifted_src.get_unchecked(half_len * N..).as_ptr());
                let mut k0 = vmull_u8(vget_low_u8(source), vget_low_u8(coeff));
                let mut k1 = vmull_high_u8(source, coeff);

                for i in 0..half_len {
                    let coeff = vdupq_n_u16(*scanned_kernel.get_unchecked(i) as u16);
                    let rollback = length - i - 1;
                    let v_source0 = vld1q_u8(shifted_src.get_unchecked((i * N)..).as_ptr());
                    let v_source1 = vld1q_u8(shifted_src.get_unchecked((rollback * N)..).as_ptr());
                    k0 = vmlaq_u16(
                        k0,
                        vaddl_u8(vget_low_u8(v_source0), vget_low_u8(v_source1)),
                        coeff,
                    );
                    k1 = vmlaq_u16(k1, vaddl_high_u8(v_source0, v_source1), coeff);
                }

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                vst1q_u8(
                    dst_ptr0,
                    vcombine_u8(vqshrn_n_u16::<7>(k0), vqshrn_n_u16::<7>(k1)),
                );
                m_cx += 16;
            }

            while m_cx + 8 < max_width {
                let cx = m_cx - s_half;
                let shifted_src = src.get_unchecked(cx..);

                let source = vld1_u8(shifted_src.get_unchecked(half_len * N..).as_ptr());
                let mut k0 = vmull_u8(source, vget_low_u8(coeff));

                for i in 0..half_len {
                    let coeff = vdupq_n_u16(*scanned_kernel.get_unchecked(i) as u16);
                    let rollback = length - i - 1;
                    let v_source0 = vld1_u8(shifted_src.get_unchecked((i * N)..).as_ptr());
                    let v_source1 = vld1_u8(shifted_src.get_unchecked((rollback * N)..).as_ptr());
                    k0 = vmlaq_u16(k0, vaddl_u8(v_source0, v_source1), coeff);
                }

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                vst1_u8(dst_ptr0, vqshrn_n_u16::<7>(k0));
                m_cx += 8;
            }

            while m_cx + 4 < max_width {
                let cx = m_cx - s_half;
                let shifted_src = src.get_unchecked(cx..);

                let source = xvld4u8(shifted_src.get_unchecked(half_len * N..).as_ptr());
                let mut k0 = vmull_u8(source, vget_low_u8(coeff));

                for i in 0..half_len {
                    let coeff = vdupq_n_u16(*scanned_kernel.get_unchecked(i) as u16);
                    let rollback = length - i - 1;
                    let v_source0 = xvld4u8(shifted_src.get_unchecked((i * N)..).as_ptr());
                    let v_source1 = xvld4u8(shifted_src.get_unchecked((rollback * N)..).as_ptr());
                    k0 = vmlaq_u16(k0, vaddl_u8(v_source0, v_source1), coeff);
                }

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                xvst_u8x4_q7(dst_ptr0, k0);
                m_cx += 4;
            }

            let coeff = *scanned_kernel.get_unchecked(half_len);

            for zx in m_cx..max_width {
                let x = zx - s_half;
                let shifted_src = src.get_unchecked(x..);
                let mut k0 = *shifted_src.get_unchecked(half_len * N) as u16 * coeff as u16;

                for i in 0..half_len {
                    let coeff = *scanned_kernel.get_unchecked(i);
                    let rollback = length - i - 1;

                    k0 += (*shifted_src.get_unchecked(i * N) as u16
                        + *shifted_src.get_unchecked(rollback * N) as u16)
                        * coeff as u16;
                }

                *dst.get_unchecked_mut(zx) = ((k0 + (1 << 6)) >> 7).min(255) as u8;
            }

            f_cx = m_right;

            while f_cx < width {
                for c in 0..N {
                    let mx = f_cx as i64 - s_kernel;
                    let mut k0 = *src.get_unchecked(f_cx * N + c) as u16 * coeff as u16;

                    for i in 0..half_len {
                        let coeff = *scanned_kernel.get_unchecked(i);
                        let rollback = length - i - 1;

                        let src0 = border_interpolate!(
                            src,
                            edge_mode,
                            i as i64 + mx,
                            0,
                            width as i64,
                            N,
                            c
                        );
                        let src1 = border_interpolate!(
                            src,
                            edge_mode,
                            rollback as i64 + mx,
                            0,
                            width as i64,
                            N,
                            c
                        );

                        k0 += (src0 as u16 + src1 as u16) * coeff as u16;
                    }

                    *dst.get_unchecked_mut(f_cx * N + c) = ((k0 + (1 << 6)) >> 7).min(255) as u8;
                }
                f_cx += 1;
            }
        }
    }
}
