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

pub(crate) fn sse_ring_vertical_row_summ_f32(
    src: &[&[f32]; 2],
    dst: &mut [f32],
    working_row: &mut [f32],
    radius: u32,
) {
    unsafe {
        sse_ring_vertical_row_summ_impl_f32(src, dst, working_row, radius);
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_ring_vertical_row_summ_impl_f32(
    src: &[&[f32]; 2],
    dst: &mut [f32],
    working_row: &mut [f32],
    radius: u32,
) {
    let next_row = src[1];
    let previous_row = src[0];
    let weight = 1. / (radius as f32 * 2. + 1.);
    let v_weight = _mm_set1_ps(weight);

    let chunks = previous_row.chunks_exact(16).len();

    for (((src_next, src_previous), buffer), dst) in next_row
        .chunks_exact(16)
        .zip(previous_row.chunks_exact(16))
        .zip(working_row.chunks_exact_mut(16))
        .zip(dst.chunks_exact_mut(16))
    {
        unsafe {
            let mut weight0 = _mm_loadu_ps(buffer.as_ptr() as *const _);
            let mut weight1 = _mm_loadu_ps(buffer.get_unchecked(4..).as_ptr() as *const _);
            let mut weight2 = _mm_loadu_ps(buffer.get_unchecked(8..).as_ptr() as *const _);
            let mut weight3 = _mm_loadu_ps(buffer.get_unchecked(12..).as_ptr() as *const _);

            let z0 = _mm_mul_ps(weight0, v_weight);
            let z1 = _mm_mul_ps(weight1, v_weight);
            let z2 = _mm_mul_ps(weight2, v_weight);
            let z3 = _mm_mul_ps(weight3, v_weight);

            _mm_storeu_ps(dst.as_mut_ptr() as *mut _, z0);
            _mm_storeu_ps(dst.get_unchecked_mut(4..).as_mut_ptr() as *mut _, z1);
            _mm_storeu_ps(dst.get_unchecked_mut(8..).as_mut_ptr() as *mut _, z2);
            _mm_storeu_ps(dst.get_unchecked_mut(12..).as_mut_ptr() as *mut _, z3);

            let next0 = _mm_loadu_ps(src_next.as_ptr() as *const _);
            let next1 = _mm_loadu_ps(src_next.get_unchecked(4..).as_ptr() as *const _);
            let next2 = _mm_loadu_ps(src_next.get_unchecked(8..).as_ptr() as *const _);
            let next3 = _mm_loadu_ps(src_next.get_unchecked(12..).as_ptr() as *const _);

            let previous0 = _mm_loadu_ps(src_previous.as_ptr() as *const _);
            let previous1 = _mm_loadu_ps(src_previous.get_unchecked(4..).as_ptr() as *const _);
            let previous2 = _mm_loadu_ps(src_previous.get_unchecked(8..).as_ptr() as *const _);
            let previous3 = _mm_loadu_ps(src_previous.get_unchecked(12..).as_ptr() as *const _);

            weight0 = _mm_add_ps(weight0, next0);
            weight1 = _mm_add_ps(weight1, next1);
            weight2 = _mm_add_ps(weight2, next2);
            weight3 = _mm_add_ps(weight3, next3);

            weight0 = _mm_sub_ps(weight0, previous0);
            weight1 = _mm_sub_ps(weight1, previous1);
            weight2 = _mm_sub_ps(weight2, previous2);
            weight3 = _mm_sub_ps(weight3, previous3);

            _mm_storeu_ps(buffer.as_mut_ptr() as *mut _, weight0);
            _mm_storeu_ps(
                buffer.get_unchecked_mut(4..).as_mut_ptr() as *mut _,
                weight1,
            );
            _mm_storeu_ps(
                buffer.get_unchecked_mut(8..).as_mut_ptr() as *mut _,
                weight2,
            );
            _mm_storeu_ps(
                buffer.get_unchecked_mut(12..).as_mut_ptr() as *mut _,
                weight3,
            );
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

        *dst = weight0 * weight;

        weight0 += *src_next;
        weight0 -= *src_previous;

        *buffer = weight0;
    }
}
