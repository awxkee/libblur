// Copyright (c) Radzivon Bartoshyk. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1.  Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2.  Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3.  Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#![allow(unused_unsafe)]
use std::arch::aarch64::*;

pub(crate) fn neon_ring_vertical_row_summ(
    src: &[&[u8]; 2],
    dst: &mut [u8],
    working_row: &mut [u32],
    radius: u32,
) {
    let next_row = src[1];
    let previous_row = src[0];
    let weight = 1. / (radius as f32 * 2. + 1.);
    let v_weight = unsafe { vdupq_n_f32(weight) };

    let chunks = previous_row.chunks_exact(16).len();

    for (((src_next, src_previous), buffer), dst) in next_row
        .chunks_exact(16)
        .zip(previous_row.chunks_exact(16))
        .zip(working_row.chunks_exact_mut(16))
        .zip(dst.chunks_exact_mut(16))
    {
        unsafe {
            let mut weight0 = vld1q_u32(buffer.as_ptr());
            let mut weight1 = vld1q_u32(buffer.get_unchecked(4..).as_ptr());
            let mut weight2 = vld1q_u32(buffer.get_unchecked(8..).as_ptr());
            let mut weight3 = vld1q_u32(buffer.get_unchecked(12..).as_ptr());

            let z0 = vmulq_f32(vcvtq_f32_u32(weight0), v_weight);
            let z1 = vmulq_f32(vcvtq_f32_u32(weight1), v_weight);
            let z2 = vmulq_f32(vcvtq_f32_u32(weight2), v_weight);
            let z3 = vmulq_f32(vcvtq_f32_u32(weight3), v_weight);

            let k0 = vcvtaq_u32_f32(z0);
            let k1 = vcvtaq_u32_f32(z1);
            let k2 = vcvtaq_u32_f32(z2);
            let k3 = vcvtaq_u32_f32(z3);

            let r0 = vcombine_u16(vmovn_u32(k0), vmovn_u32(k1));
            let r1 = vcombine_u16(vmovn_u32(k2), vmovn_u32(k3));

            vst1q_u8(
                dst.as_mut_ptr(),
                vcombine_u8(vqmovn_u16(r0), vqmovn_u16(r1)),
            );

            let next = vld1q_u8(src_next.as_ptr());
            let previous = vld1q_u8(src_previous.as_ptr());

            let next0 = vmovl_u8(vget_low_u8(next));
            let next1 = vmovl_high_u8(next);

            let previous0 = vmovl_u8(vget_low_u8(previous));
            let previous1 = vmovl_high_u8(previous);

            weight0 = vaddw_u16(weight0, vget_low_u16(next0));
            weight1 = vaddw_high_u16(weight1, next0);
            weight2 = vaddw_u16(weight2, vget_low_u16(next1));
            weight3 = vaddw_high_u16(weight3, next1);

            weight0 = vsubw_u16(weight0, vget_low_u16(previous0));
            weight1 = vsubw_high_u16(weight1, previous0);
            weight2 = vsubw_u16(weight2, vget_low_u16(previous1));
            weight3 = vsubw_high_u16(weight3, previous1);

            vst1q_u32(buffer.as_mut_ptr(), weight0);
            vst1q_u32(buffer.get_unchecked_mut(4..).as_mut_ptr(), weight1);
            vst1q_u32(buffer.get_unchecked_mut(8..).as_mut_ptr(), weight2);
            vst1q_u32(buffer.get_unchecked_mut(12..).as_mut_ptr(), weight3);
        }
    }

    for (((src_next, src_previous), buffer), dst) in next_row
        .iter()
        .zip(previous_row.iter())
        .zip(working_row.iter_mut())
        .zip(dst.iter_mut())
        .skip(chunks * 16)
    {
        let mut weight0 = *buffer;

        *dst = ((weight0 as f32 * weight).round() as u32).min(255) as u8;

        weight0 += *src_next as u32;
        weight0 -= *src_previous as u32;

        *buffer = weight0;
    }
}

#[cfg(feature = "rdm")]
pub(crate) fn neon_ring_vertical_row_summ_rdm(
    src: &[&[u8]; 2],
    dst: &mut [u8],
    working_row: &mut [u32],
    radius: u32,
) {
    unsafe {
        neon_ring_vertical_row_summ_rdm_impl(src, dst, working_row, radius);
    }
}

#[cfg(feature = "rdm")]
#[target_feature(enable = "rdm")]
pub(crate) fn neon_ring_vertical_row_summ_rdm_impl(
    src: &[&[u8]; 2],
    dst: &mut [u8],
    working_row: &mut [u32],
    radius: u32,
) {
    let next_row = src[1];
    let previous_row = src[0];
    let weight = (((1i64 << 31) - 1) as f64 / (radius as f64 * 2. + 1.)) as i32;
    let v_weight = unsafe { vdupq_n_s32(weight) };

    let chunks = previous_row.chunks_exact(16).len();

    for (((src_next, src_previous), buffer), dst) in next_row
        .chunks_exact(16)
        .zip(previous_row.chunks_exact(16))
        .zip(working_row.chunks_exact_mut(16))
        .zip(dst.chunks_exact_mut(16))
    {
        unsafe {
            let mut weight0 = vld1q_u32(buffer.as_ptr());
            let mut weight1 = vld1q_u32(buffer.get_unchecked(4..).as_ptr());
            let mut weight2 = vld1q_u32(buffer.get_unchecked(8..).as_ptr());
            let mut weight3 = vld1q_u32(buffer.get_unchecked(12..).as_ptr());

            let z0 = vqrdmulhq_s32(vreinterpretq_s32_u32(weight0), v_weight);
            let z1 = vqrdmulhq_s32(vreinterpretq_s32_u32(weight1), v_weight);
            let z2 = vqrdmulhq_s32(vreinterpretq_s32_u32(weight2), v_weight);
            let z3 = vqrdmulhq_s32(vreinterpretq_s32_u32(weight3), v_weight);

            let k0 = vreinterpretq_u32_s32(z0);
            let k1 = vreinterpretq_u32_s32(z1);
            let k2 = vreinterpretq_u32_s32(z2);
            let k3 = vreinterpretq_u32_s32(z3);

            let r0 = vcombine_u16(vmovn_u32(k0), vmovn_u32(k1));
            let r1 = vcombine_u16(vmovn_u32(k2), vmovn_u32(k3));

            vst1q_u8(
                dst.as_mut_ptr(),
                vcombine_u8(vqmovn_u16(r0), vqmovn_u16(r1)),
            );

            let next = vld1q_u8(src_next.as_ptr());
            let previous = vld1q_u8(src_previous.as_ptr());

            let next0 = vmovl_u8(vget_low_u8(next));
            let next1 = vmovl_high_u8(next);

            let previous0 = vmovl_u8(vget_low_u8(previous));
            let previous1 = vmovl_high_u8(previous);

            weight0 = vaddw_u16(weight0, vget_low_u16(next0));
            weight1 = vaddw_high_u16(weight1, next0);
            weight2 = vaddw_u16(weight2, vget_low_u16(next1));
            weight3 = vaddw_high_u16(weight3, next1);

            weight0 = vsubw_u16(weight0, vget_low_u16(previous0));
            weight1 = vsubw_high_u16(weight1, previous0);
            weight2 = vsubw_u16(weight2, vget_low_u16(previous1));
            weight3 = vsubw_high_u16(weight3, previous1);

            vst1q_u32(buffer.as_mut_ptr(), weight0);
            vst1q_u32(buffer.get_unchecked_mut(4..).as_mut_ptr(), weight1);
            vst1q_u32(buffer.get_unchecked_mut(8..).as_mut_ptr(), weight2);
            vst1q_u32(buffer.get_unchecked_mut(12..).as_mut_ptr(), weight3);
        }
    }

    for (((src_next, src_previous), buffer), dst) in next_row
        .iter()
        .zip(previous_row.iter())
        .zip(working_row.iter_mut())
        .zip(dst.iter_mut())
        .skip(chunks * 16)
    {
        let mut weight0 = *buffer;

        *dst = ((weight0 as i64 * weight as i64 + (1 << 30)) >> 31).min(255) as u8;

        weight0 += *src_next as u32;
        weight0 -= *src_previous as u32;

        *buffer = weight0;
    }
}
