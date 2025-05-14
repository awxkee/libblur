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

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn sse_ring_vertical_row_summ16(
    src: &[&[u16]; 2],
    dst: &mut [u16],
    working_row: &mut [u32],
    radius: u32,
) {
    unsafe {
        sse_ring_vertical_row_summ_impl16(src, dst, working_row, radius);
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_ring_vertical_row_summ_impl16(
    src: &[&[u16]; 2],
    dst: &mut [u16],
    working_row: &mut [u32],
    radius: u32,
) {
    let next_row = src[1];
    let previous_row = src[0];
    let weight = 1. / (radius as f32 * 2.);
    let v_weight = _mm_set1_ps(weight);

    let chunks = previous_row.chunks_exact(16).len();

    for (((src_next, src_previous), buffer), dst) in next_row
        .chunks_exact(16)
        .zip(previous_row.chunks_exact(16))
        .zip(working_row.chunks_exact_mut(16))
        .zip(dst.chunks_exact_mut(16))
    {
        unsafe {
            let mut weight0 = _mm_loadu_si128(buffer.as_ptr() as *const _);
            let mut weight1 = _mm_loadu_si128(buffer.get_unchecked(4..).as_ptr() as *const _);
            let mut weight2 = _mm_loadu_si128(buffer.get_unchecked(8..).as_ptr() as *const _);
            let mut weight3 = _mm_loadu_si128(buffer.get_unchecked(12..).as_ptr() as *const _);

            let next0 = _mm_loadu_si128(src_next.as_ptr() as *const _);
            let next1 = _mm_loadu_si128(src_next.get_unchecked(8..).as_ptr() as *const _);
            let previous0 = _mm_loadu_si128(src_previous.as_ptr() as *const _);
            let previous1 = _mm_loadu_si128(src_previous.get_unchecked(8..).as_ptr() as *const _);

            weight0 = _mm_add_epi32(weight0, _mm_unpacklo_epi16(next0, _mm_setzero_si128()));
            weight1 = _mm_add_epi32(weight1, _mm_unpackhi_epi16(next0, _mm_setzero_si128()));
            weight2 = _mm_add_epi32(weight2, _mm_unpacklo_epi16(next1, _mm_setzero_si128()));
            weight3 = _mm_add_epi32(weight3, _mm_unpackhi_epi16(next1, _mm_setzero_si128()));

            weight0 = _mm_sub_epi32(weight0, _mm_unpacklo_epi16(previous0, _mm_setzero_si128()));
            weight1 = _mm_sub_epi32(weight1, _mm_unpackhi_epi16(previous0, _mm_setzero_si128()));
            weight2 = _mm_sub_epi32(weight2, _mm_unpacklo_epi16(previous1, _mm_setzero_si128()));
            weight3 = _mm_sub_epi32(weight3, _mm_unpackhi_epi16(previous1, _mm_setzero_si128()));

            let w0 = _mm_cvtepi32_ps(weight0);
            let w1 = _mm_cvtepi32_ps(weight1);
            let w2 = _mm_cvtepi32_ps(weight2);
            let w3 = _mm_cvtepi32_ps(weight3);

            _mm_storeu_si128(buffer.as_mut_ptr() as *mut _, weight0);
            _mm_storeu_si128(
                buffer.get_unchecked_mut(4..).as_mut_ptr() as *mut _,
                weight1,
            );
            _mm_storeu_si128(
                buffer.get_unchecked_mut(8..).as_mut_ptr() as *mut _,
                weight2,
            );
            _mm_storeu_si128(
                buffer.get_unchecked_mut(12..).as_mut_ptr() as *mut _,
                weight3,
            );

            let z0 = _mm_mul_ps(w0, v_weight);
            let z1 = _mm_mul_ps(w1, v_weight);
            let z2 = _mm_mul_ps(w2, v_weight);
            let z3 = _mm_mul_ps(w3, v_weight);

            let k0 = _mm_cvtps_epi32(z0);
            let k1 = _mm_cvtps_epi32(z1);
            let k2 = _mm_cvtps_epi32(z2);
            let k3 = _mm_cvtps_epi32(z3);

            let r0 = _mm_packus_epi32(k0, k1);
            let r1 = _mm_packus_epi32(k2, k3);

            _mm_storeu_si128(dst.as_mut_ptr() as *mut _, r0);
            _mm_storeu_si128(dst.get_unchecked_mut(8..).as_mut_ptr() as *mut _, r1);
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

        weight0 += *src_next as u32;
        weight0 -= *src_previous as u32;

        *buffer = weight0;

        *dst = ((weight0 as f32 * weight).round() as u32).min(u16::MAX as u32) as u16;
    }
}
