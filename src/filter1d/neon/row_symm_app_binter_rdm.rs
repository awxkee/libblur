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
use crate::edge_mode::{border_interpolate, clamp_edge};
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::neon::utils::{
    vmla_symm_hi_u8_s16, vmlaq_symm_hi_u8_s16, vmull_expand_i16, vmullq_expand_i16, vqmovn_s16_u8,
    vqmovnq_s16x2_u8, xvld1q_u8_x3, xvld1q_u8_x4, xvld4u8, xvst1q_u8_x3, xvst1q_u8_x4,
    xvst_u8x4_q15,
};
use crate::filter1d::row_handler_small_approx::{RowsHolder, RowsHolderMut};
use crate::filter1d::to_approx_storage::ToApproxStorage;
use crate::img_size::ImageSize;
use crate::BorderHandle;
use std::arch::aarch64::*;

pub(crate) fn filter_row_symm_neon_binter_u8_i32_rdm<const N: usize>(
    edge_mode: BorderHandle,
    m_src: &RowsHolder<u8>,
    m_dst: &mut RowsHolderMut<u8>,
    image_size: ImageSize,
    scanned_kernel: &[ScanPoint1d<i32>],
) {
    unsafe {
        executor_unit::<N>(edge_mode, m_src, m_dst, image_size, scanned_kernel);
    }
}

#[target_feature(enable = "rdm")]
unsafe fn executor_unit<const N: usize>(
    edge_mode: BorderHandle,
    m_src: &RowsHolder<u8>,
    m_dst: &mut RowsHolderMut<u8>,
    image_size: ImageSize,
    scanned_kernel: &[ScanPoint1d<i32>],
) {
    unsafe {
        let length = scanned_kernel.len();

        if length == 5 {
            executor_unit_5::<N>(edge_mode, m_src, m_dst, image_size, scanned_kernel);
        } else {
            executor_unit_any::<N>(edge_mode, m_src, m_dst, image_size, scanned_kernel);
        }
    }
}

#[target_feature(enable = "rdm")]
unsafe fn executor_unit_5<const N: usize>(
    edge_mode: BorderHandle,
    m_src: &RowsHolder<u8>,
    m_dst: &mut RowsHolderMut<u8>,
    image_size: ImageSize,
    scanned_kernel: &[ScanPoint1d<i32>],
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

            let c0 = scanned_kernel[0].weight;
            let c1 = scanned_kernel[1].weight;
            let c2 = scanned_kernel[2].weight;
            while f_cx < min_left {
                let mx = f_cx as i64 - s_kernel;
                let e0 = clamp_edge!(edge_mode.edge_mode, mx, 0, width as i64);
                let e1 = clamp_edge!(edge_mode.edge_mode, 4i64 + mx, 0, width as i64);
                let e2 = clamp_edge!(edge_mode.edge_mode, mx + 1, 0, width as i64);
                let e3 = clamp_edge!(edge_mode.edge_mode, 3i64 + mx, 0, width as i64);

                for c in 0..N {
                    let mut k0: i32 = *src.get_unchecked(f_cx * N + c) as i32 * c2;

                    let src0 = *src.get_unchecked(e0 * N + c);
                    let src1 = *src.get_unchecked(e1 * N + c);

                    let src2 = *src.get_unchecked(e2 * N + c);
                    let src3 = *src.get_unchecked(e3 * N + c);

                    k0 += (src0 as i32 + src1 as i32) * c0;
                    k0 += (src2 as i32 + src3 as i32) * c1;

                    *dst.get_unchecked_mut(f_cx * N + c) = k0.to_approx_();
                }
                f_cx += 1;
            }

            let mut m_cx = f_cx * N;

            let s_half = half_len * N;
            let m_right = width.saturating_sub(half_len);
            let max_width = m_right * N;

            let c0 = vdupq_n_s16(scanned_kernel[0].weight as i16);
            let c1 = vdupq_n_s16(scanned_kernel[1].weight as i16);
            let c2 = vdupq_n_s16(scanned_kernel[2].weight as i16);

            while m_cx + 64 < max_width {
                let cx = m_cx - s_half;

                let shifted_src = src.get_unchecked(cx..);

                let source = xvld1q_u8_x4(shifted_src.get_unchecked(half_len * N..).as_ptr());
                let mut k0 = vmullq_expand_i16(source.0, c2);
                let mut k1 = vmullq_expand_i16(source.1, c2);
                let mut k2 = vmullq_expand_i16(source.2, c2);
                let mut k3 = vmullq_expand_i16(source.3, c2);

                let v_source0 = xvld1q_u8_x4(shifted_src.get_unchecked(0..).as_ptr());
                let v_source1 = xvld1q_u8_x4(shifted_src.get_unchecked((4 * N)..).as_ptr());
                k0 = vmlaq_symm_hi_u8_s16(k0, v_source0.0, v_source1.0, c0);
                k1 = vmlaq_symm_hi_u8_s16(k1, v_source0.1, v_source1.1, c0);
                k2 = vmlaq_symm_hi_u8_s16(k2, v_source0.2, v_source1.2, c0);
                k3 = vmlaq_symm_hi_u8_s16(k3, v_source0.3, v_source1.3, c0);

                let v_source2 = xvld1q_u8_x4(shifted_src.get_unchecked(N..).as_ptr());
                let v_source3 = xvld1q_u8_x4(shifted_src.get_unchecked((3 * N)..).as_ptr());

                k0 = vmlaq_symm_hi_u8_s16(k0, v_source2.0, v_source3.0, c1);
                k1 = vmlaq_symm_hi_u8_s16(k1, v_source2.1, v_source3.1, c1);
                k2 = vmlaq_symm_hi_u8_s16(k2, v_source2.2, v_source3.2, c1);
                k3 = vmlaq_symm_hi_u8_s16(k3, v_source2.3, v_source3.3, c1);

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                xvst1q_u8_x4(
                    dst_ptr0,
                    uint8x16x4_t(
                        vqmovnq_s16x2_u8(k0),
                        vqmovnq_s16x2_u8(k1),
                        vqmovnq_s16x2_u8(k2),
                        vqmovnq_s16x2_u8(k3),
                    ),
                );
                m_cx += 64;
            }

            while m_cx + 48 < max_width {
                let cx = m_cx - s_half;

                let shifted_src = src.get_unchecked(cx..);

                let source = xvld1q_u8_x3(shifted_src.get_unchecked(half_len * N..).as_ptr());
                let mut k0 = vmullq_expand_i16(source.0, c2);
                let mut k1 = vmullq_expand_i16(source.1, c2);
                let mut k2 = vmullq_expand_i16(source.2, c2);

                let v_source0 = xvld1q_u8_x3(shifted_src.get_unchecked(0..).as_ptr());
                let v_source1 = xvld1q_u8_x3(shifted_src.get_unchecked((4 * N)..).as_ptr());
                k0 = vmlaq_symm_hi_u8_s16(k0, v_source0.0, v_source1.0, c0);
                k1 = vmlaq_symm_hi_u8_s16(k1, v_source0.1, v_source1.1, c0);
                k2 = vmlaq_symm_hi_u8_s16(k2, v_source0.2, v_source1.2, c0);

                let v_source2 = xvld1q_u8_x3(shifted_src.get_unchecked(N..).as_ptr());
                let v_source3 = xvld1q_u8_x3(shifted_src.get_unchecked((3 * N)..).as_ptr());
                k0 = vmlaq_symm_hi_u8_s16(k0, v_source2.0, v_source3.0, c1);
                k1 = vmlaq_symm_hi_u8_s16(k1, v_source2.1, v_source3.1, c1);
                k2 = vmlaq_symm_hi_u8_s16(k2, v_source2.2, v_source3.2, c1);

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                xvst1q_u8_x3(
                    dst_ptr0,
                    uint8x16x3_t(
                        vqmovnq_s16x2_u8(k0),
                        vqmovnq_s16x2_u8(k1),
                        vqmovnq_s16x2_u8(k2),
                    ),
                );
                m_cx += 48;
            }

            while m_cx + 16 < max_width {
                let cx = m_cx - s_half;

                let shifted_src = src.get_unchecked(cx..);

                let source = vld1q_u8(shifted_src.get_unchecked(half_len * N..).as_ptr());
                let mut k0 = vmullq_expand_i16(source, c2);

                let v_source0 = vld1q_u8(shifted_src.get_unchecked(0..).as_ptr());
                let v_source1 = vld1q_u8(shifted_src.get_unchecked((4 * N)..).as_ptr());
                k0 = vmlaq_symm_hi_u8_s16(k0, v_source0, v_source1, c0);

                let v_source2 = vld1q_u8(shifted_src.get_unchecked(N..).as_ptr());
                let v_source3 = vld1q_u8(shifted_src.get_unchecked((3 * N)..).as_ptr());
                k0 = vmlaq_symm_hi_u8_s16(k0, v_source2, v_source3, c1);

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                vst1q_u8(dst_ptr0, vqmovnq_s16x2_u8(k0));
                m_cx += 16;
            }

            while m_cx + 8 < max_width {
                let cx = m_cx - s_half;

                let shifted_src = src.get_unchecked(cx..);

                let source = vld1_u8(shifted_src.get_unchecked(half_len * N..).as_ptr());
                let mut k0 = vmull_expand_i16(source, c2);

                let v_source0 = vld1_u8(shifted_src.get_unchecked(0..).as_ptr());
                let v_source1 = vld1_u8(shifted_src.get_unchecked((4 * N)..).as_ptr());
                k0 = vmla_symm_hi_u8_s16(k0, v_source0, v_source1, c0);

                let v_source2 = vld1_u8(shifted_src.get_unchecked(N..).as_ptr());
                let v_source3 = vld1_u8(shifted_src.get_unchecked((3 * N)..).as_ptr());
                k0 = vmla_symm_hi_u8_s16(k0, v_source2, v_source3, c1);

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                vst1_u8(dst_ptr0, vqmovn_s16_u8(k0));
                m_cx += 8;
            }

            while m_cx + 4 < max_width {
                let cx = m_cx - s_half;

                let shifted_src = src.get_unchecked(cx..);

                let source = xvld4u8(shifted_src.get_unchecked(half_len * N..).as_ptr());
                let mut k0 = vmull_expand_i16(source, c2);

                let v_source0 = xvld4u8(shifted_src.get_unchecked(0..).as_ptr());
                let v_source1 = xvld4u8(shifted_src.get_unchecked((4 * N)..).as_ptr());
                k0 = vmla_symm_hi_u8_s16(k0, v_source0, v_source1, c0);

                let v_source2 = xvld4u8(shifted_src.get_unchecked(N..).as_ptr());
                let v_source3 = xvld4u8(shifted_src.get_unchecked((3 * N)..).as_ptr());
                k0 = vmla_symm_hi_u8_s16(k0, v_source2, v_source3, c1);

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                xvst_u8x4_q15(dst_ptr0, k0);
                m_cx += 4;
            }

            let c0 = scanned_kernel[0].weight;
            let c1 = scanned_kernel[1].weight;
            let c2 = scanned_kernel[2].weight;

            for zx in m_cx..max_width {
                let x = zx - s_half;

                let shifted_src = src.get_unchecked(x..);
                let mut k0 = *shifted_src.get_unchecked(half_len * N) as i32 * c2;

                k0 += (*shifted_src.get_unchecked(0) as i16
                    + *shifted_src.get_unchecked(4 * N) as i16) as i32
                    * c0;

                k0 += (*shifted_src.get_unchecked(N) as i16
                    + *shifted_src.get_unchecked(3 * N) as i16) as i32
                    * c1;

                *dst.get_unchecked_mut(zx) = k0.to_approx_();
            }

            f_cx = m_right;

            while f_cx < width {
                let mx = f_cx as i64 - s_kernel;
                let e0 = clamp_edge!(edge_mode.edge_mode, mx, 0, width as i64);
                let e1 = clamp_edge!(edge_mode.edge_mode, 4i64 + mx, 0, width as i64);
                let e2 = clamp_edge!(edge_mode.edge_mode, mx + 1, 0, width as i64);
                let e3 = clamp_edge!(edge_mode.edge_mode, 3i64 + mx, 0, width as i64);

                for c in 0..N {
                    let mut k0: i32 = *src.get_unchecked(f_cx * N + c) as i32 * c2;

                    let src0 = *src.get_unchecked(e0 * N + c);
                    let src1 = *src.get_unchecked(e1 * N + c);

                    let src2 = *src.get_unchecked(e2 * N + c);
                    let src3 = *src.get_unchecked(e3 * N + c);

                    k0 += (src0 as i32 + src1 as i32) * c0;
                    k0 += (src2 as i32 + src3 as i32) * c1;

                    *dst.get_unchecked_mut(f_cx * N + c) = k0.to_approx_();
                }
                f_cx += 1;
            }
        }
    }
}

#[target_feature(enable = "rdm")]
unsafe fn executor_unit_any<const N: usize>(
    edge_mode: BorderHandle,
    m_src: &RowsHolder<u8>,
    m_dst: &mut RowsHolderMut<u8>,
    image_size: ImageSize,
    scanned_kernel: &[ScanPoint1d<i32>],
) {
    unsafe {
        let width = image_size.width;

        let length = scanned_kernel.len();
        let half_len = length / 2;

        let min_left = half_len.min(width);
        let s_kernel = half_len as i64;

        let left_src = m_src.holder.chunks_exact(4).remainder();
        let left_dst = m_dst.holder.chunks_exact_mut(4).into_remainder();

        for (src, dst) in left_src.iter().zip(left_dst.iter_mut()) {
            let dst = &mut **dst;
            let mut f_cx = 0usize;

            let coeff = *scanned_kernel.get_unchecked(half_len);
            while f_cx < min_left {
                for c in 0..N {
                    let mx = f_cx as i64 - s_kernel;
                    let mut k0: i32 = *src.get_unchecked(f_cx * N + c) as i32 * coeff.weight;

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

                        k0 += (src0 as i32 + src1 as i32) * coeff.weight;
                    }

                    *dst.get_unchecked_mut(f_cx * N + c) = k0.to_approx_();
                }
                f_cx += 1;
            }

            let mut m_cx = f_cx * N;

            let s_half = half_len * N;
            let m_right = width.saturating_sub(half_len);
            let max_width = m_right * N;

            let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(half_len).weight as i16);

            while m_cx + 48 < max_width {
                let cx = m_cx - s_half;

                let shifted_src = src.get_unchecked(cx..);

                let source = xvld1q_u8_x3(shifted_src.get_unchecked(half_len * N..).as_ptr());
                let mut k0 = vmullq_expand_i16(source.0, coeff);
                let mut k1 = vmullq_expand_i16(source.1, coeff);
                let mut k2 = vmullq_expand_i16(source.2, coeff);

                for i in 0..half_len {
                    let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight as i16);
                    let rollback = length - i - 1;
                    let v_source0 = xvld1q_u8_x3(shifted_src.get_unchecked((i * N)..).as_ptr());
                    let v_source1 =
                        xvld1q_u8_x3(shifted_src.get_unchecked((rollback * N)..).as_ptr());
                    k0 = vmlaq_symm_hi_u8_s16(k0, v_source0.0, v_source1.0, coeff);
                    k1 = vmlaq_symm_hi_u8_s16(k1, v_source0.1, v_source1.1, coeff);
                    k2 = vmlaq_symm_hi_u8_s16(k2, v_source0.2, v_source1.2, coeff);
                }

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                xvst1q_u8_x3(
                    dst_ptr0,
                    uint8x16x3_t(
                        vqmovnq_s16x2_u8(k0),
                        vqmovnq_s16x2_u8(k1),
                        vqmovnq_s16x2_u8(k2),
                    ),
                );
                m_cx += 48;
            }

            while m_cx + 16 < max_width {
                let cx = m_cx - s_half;

                let shifted_src = src.get_unchecked(cx..);

                let source = vld1q_u8(shifted_src.get_unchecked(half_len * N..).as_ptr());
                let mut k0 = vmullq_expand_i16(source, coeff);

                for i in 0..half_len {
                    let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight as i16);
                    let rollback = length - i - 1;
                    let v_source0 = vld1q_u8(shifted_src.get_unchecked((i * N)..).as_ptr());
                    let v_source1 = vld1q_u8(shifted_src.get_unchecked((rollback * N)..).as_ptr());
                    k0 = vmlaq_symm_hi_u8_s16(k0, v_source0, v_source1, coeff);
                }

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                vst1q_u8(dst_ptr0, vqmovnq_s16x2_u8(k0));
                m_cx += 16;
            }

            while m_cx + 8 < max_width {
                let cx = m_cx - s_half;

                let shifted_src = src.get_unchecked(cx..);

                let source = vld1_u8(shifted_src.get_unchecked(half_len * N..).as_ptr());
                let mut k0 = vmull_expand_i16(source, coeff);

                for i in 0..half_len {
                    let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight as i16);
                    let rollback = length - i - 1;
                    let v_source0 = vld1_u8(shifted_src.get_unchecked((i * N)..).as_ptr());
                    let v_source1 = vld1_u8(shifted_src.get_unchecked((rollback * N)..).as_ptr());
                    k0 = vmla_symm_hi_u8_s16(k0, v_source0, v_source1, coeff);
                }

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                vst1_u8(dst_ptr0, vqmovn_s16_u8(k0));
                m_cx += 8;
            }

            while m_cx + 4 < max_width {
                let cx = m_cx - s_half;

                let shifted_src = src.get_unchecked(cx..);

                let source = xvld4u8(shifted_src.get_unchecked(half_len * N..).as_ptr());
                let mut k0 = vmull_expand_i16(source, coeff);

                for i in 0..half_len {
                    let coeff = vdupq_n_s16(scanned_kernel.get_unchecked(i).weight as i16);
                    let rollback = length - i - 1;
                    let v_source0 = xvld4u8(shifted_src.get_unchecked((i * N)..).as_ptr());
                    let v_source1 = xvld4u8(shifted_src.get_unchecked((rollback * N)..).as_ptr());
                    k0 = vmla_symm_hi_u8_s16(k0, v_source0, v_source1, coeff);
                }

                let dst_ptr0 = dst.get_unchecked_mut(m_cx..).as_mut_ptr();
                xvst_u8x4_q15(dst_ptr0, k0);
                m_cx += 4;
            }

            let coeff = *scanned_kernel.get_unchecked(half_len);

            for zx in m_cx..max_width {
                let x = zx - s_half;

                let shifted_src = src.get_unchecked(x..);
                let mut k0 = *shifted_src.get_unchecked(half_len * N) as i32 * coeff.weight;

                for i in 0..half_len {
                    let coeff = *scanned_kernel.get_unchecked(i);
                    let rollback = length - i - 1;

                    k0 += (*shifted_src.get_unchecked(i * N) as i16
                        + *shifted_src.get_unchecked(rollback * N) as i16)
                        as i32
                        * coeff.weight;
                }

                *dst.get_unchecked_mut(zx) = k0.to_approx_();
            }

            f_cx = m_right;

            while f_cx < width {
                let mx = f_cx as i64 - s_kernel;
                let coeff = *scanned_kernel.get_unchecked(half_len);

                for c in 0..N {
                    let mut k0 = *src.get_unchecked(f_cx * N + c) as i32 * coeff.weight;

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

                        k0 += (src0 as i32 + src1 as i32) * coeff.weight;
                    }

                    *dst.get_unchecked_mut(f_cx * N + c) = k0.to_approx_();
                }
                f_cx += 1;
            }
        }
    }
}
