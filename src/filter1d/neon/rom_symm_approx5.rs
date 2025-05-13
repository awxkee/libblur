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
use crate::edge_mode::clamp_edge;
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::neon::utils::xvld4u8;
use crate::filter1d::row_handler_small_approx::{RowsHolder, RowsHolderMut};
use crate::{BorderHandle, ImageSize};
use std::arch::aarch64::*;

pub(crate) fn filter_row_symm_neon_binter_u8_uq0_7_x5<const N: usize>(
    edge_mode: BorderHandle,
    m_src: &RowsHolder<u8>,
    m_dst: &mut RowsHolderMut<u8>,
    image_size: ImageSize,
    scanned_kernel: &[ScanPoint1d<i32>],
) {
    assert!(N <= 4);
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
    executor_unit_5_q0_u8::<N>(edge_mode, m_src, m_dst, image_size, &shifted);
}

#[inline(always)]
pub(crate) unsafe fn xvst_u8x4_q0_7(a: *mut u8, v: uint16x8_t) {
    let shifted = vqrshrn_n_u16::<7>(v);
    vst1_lane_u32::<0>(a as *mut _, vreinterpret_u32_u8(shifted));
}

fn executor_unit_5_q0_u8<const N: usize>(
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
            let dst = &mut **dst;
            let mut f_cx = 0usize;

            let c0 = scanned_kernel[0];
            let c1 = scanned_kernel[1];
            let c2 = scanned_kernel[2];
            while f_cx < min_left {
                let mx = f_cx as i64 - s_kernel;
                let e0 = clamp_edge!(edge_mode.edge_mode, mx, 0, width as i64);
                let e1 = clamp_edge!(edge_mode.edge_mode, 4i64 + mx, 0, width as i64);
                let e2 = clamp_edge!(edge_mode.edge_mode, mx + 1, 0, width as i64);
                let e3 = clamp_edge!(edge_mode.edge_mode, 3i64 + mx, 0, width as i64);

                for c in 0..N {
                    let mut k0: u16 = *src.get_unchecked(f_cx * N + c) as u16 * c2 as u16;

                    let src0 = *src.get_unchecked(e0 * N + c);
                    let src1 = *src.get_unchecked(e1 * N + c);

                    let src2 = *src.get_unchecked(e2 * N + c);
                    let src3 = *src.get_unchecked(e3 * N + c);

                    k0 += (src0 as u16 + src1 as u16) * c0 as u16;
                    k0 += (src2 as u16 + src3 as u16) * c1 as u16;

                    *dst.get_unchecked_mut(f_cx * N + c) = ((k0 + (1 << 6)) >> 7).min(255) as u8;
                }
                f_cx += 1;
            }

            let mut m_cx = f_cx * N;

            let s_half = half_len * N;
            let m_right = width.saturating_sub(half_len);
            let max_width = m_right * N;

            let c0 = vdupq_n_u16(scanned_kernel[0] as u16);
            let c1 = vdupq_n_u16(scanned_kernel[1] as u16);
            let c2 = vdupq_n_u8(scanned_kernel[2]);

            while m_cx + 16 < max_width {
                let cx = m_cx - s_half;

                let shifted_src = src.get_unchecked(cx..);

                let source = vld1q_u8(shifted_src.get_unchecked(half_len * N..).as_ptr());
                let v_source0 = vld1q_u8(shifted_src.get_unchecked(0..).as_ptr());
                let v_source4 = vld1q_u8(shifted_src.get_unchecked((4 * N)..).as_ptr());
                let v_source2 = vld1q_u8(shifted_src.get_unchecked(N..).as_ptr());
                let v_source3 = vld1q_u8(shifted_src.get_unchecked((3 * N)..).as_ptr());

                let mut k0 = vmull_u8(vget_low_u8(source), vget_low_u8(c2));
                let mut k1 = vmull_high_u8(source, c2);

                k0 = vmlaq_u16(
                    k0,
                    vaddl_u8(vget_low_u8(v_source0), vget_low_u8(v_source4)),
                    c0,
                );
                k1 = vmlaq_u16(k1, vaddl_high_u8(v_source0, v_source4), c0);

                k0 = vmlaq_u16(
                    k0,
                    vaddl_u8(vget_low_u8(v_source2), vget_low_u8(v_source3)),
                    c1,
                );
                k1 = vmlaq_u16(k1, vaddl_high_u8(v_source2, v_source3), c1);

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                vst1q_u8(
                    dst_ptr0,
                    vcombine_u8(vqrshrn_n_u16::<7>(k0), vqrshrn_n_u16::<7>(k1)),
                );
                m_cx += 16;
            }

            while m_cx + 8 < max_width {
                let cx = m_cx - s_half;

                let shifted_src = src.get_unchecked(cx..);

                let v_source0 = vld1_u8(shifted_src.get_unchecked(0..).as_ptr());
                let source = vld1_u8(shifted_src.get_unchecked(half_len * N..).as_ptr());
                let v_source1 = vld1_u8(shifted_src.get_unchecked((4 * N)..).as_ptr());
                let v_source2 = vld1_u8(shifted_src.get_unchecked(N..).as_ptr());
                let v_source3 = vld1_u8(shifted_src.get_unchecked((3 * N)..).as_ptr());

                let mut k0 = vmull_u8(source, vget_low_u8(c2));

                k0 = vmlaq_u16(k0, vaddl_u8(v_source0, v_source1), c0);

                k0 = vmlaq_u16(k0, vaddl_u8(v_source2, v_source3), c1);

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                vst1_u8(dst_ptr0, vqrshrn_n_u16::<7>(k0));
                m_cx += 8;
            }

            while m_cx + 4 < max_width {
                let cx = m_cx - s_half;

                let shifted_src = src.get_unchecked(cx..);

                let source = xvld4u8(shifted_src.get_unchecked(half_len * N..).as_ptr());
                let mut k0 = vmull_u8(source, vget_low_u8(c2));

                let v_source0 = xvld4u8(shifted_src.get_unchecked(0..).as_ptr());
                let v_source1 = xvld4u8(shifted_src.get_unchecked((4 * N)..).as_ptr());
                k0 = vmlaq_u16(k0, vaddl_u8(v_source0, v_source1), c0);

                let v_source2 = xvld4u8(shifted_src.get_unchecked(N..).as_ptr());
                let v_source3 = xvld4u8(shifted_src.get_unchecked((3 * N)..).as_ptr());
                k0 = vmlaq_u16(k0, vaddl_u8(v_source2, v_source3), c1);

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                xvst_u8x4_q0_7(dst_ptr0, k0);
                m_cx += 4;
            }

            let c0 = scanned_kernel[0];
            let c1 = scanned_kernel[1];
            let c2 = scanned_kernel[2];

            for zx in m_cx..max_width {
                let x = zx - s_half;

                let shifted_src = src.get_unchecked(x..);
                let mut k0: u16 = *shifted_src.get_unchecked(half_len * N) as u16 * c2 as u16;

                k0 += (*shifted_src.get_unchecked(0) as u16
                    + *shifted_src.get_unchecked(4 * N) as u16)
                    * c0 as u16;

                k0 += (*shifted_src.get_unchecked(N) as u16
                    + *shifted_src.get_unchecked(3 * N) as u16)
                    * c1 as u16;

                *dst.get_unchecked_mut(zx) = ((k0 + (1 << 6)) >> 7).min(255) as u8;
            }

            f_cx = m_right;

            while f_cx < width {
                let mx = f_cx as i64 - s_kernel;
                let e0 = clamp_edge!(edge_mode.edge_mode, mx, 0, width as i64);
                let e1 = clamp_edge!(edge_mode.edge_mode, 4i64 + mx, 0, width as i64);
                let e2 = clamp_edge!(edge_mode.edge_mode, mx + 1, 0, width as i64);
                let e3 = clamp_edge!(edge_mode.edge_mode, 3i64 + mx, 0, width as i64);

                for c in 0..N {
                    let mut k0: u16 = *src.get_unchecked(f_cx * N + c) as u16 * c2 as u16;

                    let src0 = *src.get_unchecked(e0 * N + c);
                    let src1 = *src.get_unchecked(e1 * N + c);

                    let src2 = *src.get_unchecked(e2 * N + c);
                    let src3 = *src.get_unchecked(e3 * N + c);

                    k0 += (src0 as u16 + src1 as u16) * c0 as u16;
                    k0 += (src2 as u16 + src3 as u16) * c1 as u16;

                    *dst.get_unchecked_mut(f_cx * N + c) = ((k0 + (1 << 6)) >> 7).min(255) as u8;
                }
                f_cx += 1;
            }
        }
    }
}
