/*
 * Copyright (c) Radzivon Bartoshyk. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2.  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3.  Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use crate::filter1d::sse::utils::{
    _mm_mul_add_epi8_by_ps_x4, _mm_mul_epi8_by_ps_x4, _mm_pack_ps_x4_epi8,
};
use crate::filter1d::Arena;
use crate::filter2d::scan_point_2d::ScanPoint2d;
use crate::mlaf::mlaf;
use crate::sse::{_mm_load_pack_x2, _mm_load_pack_x4, _mm_store_pack_x2, _mm_store_pack_x4};
use crate::to_storage::ToStorage;
use crate::unsafe_slice::UnsafeSlice;
use crate::ImageSize;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::Mul;

pub fn convolve_segment_sse_2d_u8_f32(
    arena: Arena,
    arena_source: &[u8],
    dst: &mut [u8],
    image_size: ImageSize,
    prepared_kernel: &[ScanPoint2d<f32>],
    y: usize,
) {
    unsafe {
        let has_fma = std::arch::is_x86_feature_detected!("fma");
        if has_fma {
            convolve_segment_2d_u8_fma(arena, arena_source, dst, image_size, prepared_kernel, y);
        } else {
            convolve_segment_2d_u8_def(arena, arena_source, dst, image_size, prepared_kernel, y);
        }
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn convolve_segment_2d_u8_def(
    arena: Arena,
    arena_source: &[u8],
    dst: &mut [u8],
    image_size: ImageSize,
    prepared_kernel: &[ScanPoint2d<f32>],
    y: usize,
) {
    convolve_segment_2d_u8_f32_impl::<false>(
        arena,
        arena_source,
        dst,
        image_size,
        prepared_kernel,
        y,
    );
}

#[target_feature(enable = "sse4.1", enable = "fma")]
unsafe fn convolve_segment_2d_u8_fma(
    arena: Arena,
    arena_source: &[u8],
    dst: &mut [u8],
    image_size: ImageSize,
    prepared_kernel: &[ScanPoint2d<f32>],
    y: usize,
) {
    convolve_segment_2d_u8_f32_impl::<true>(
        arena,
        arena_source,
        dst,
        image_size,
        prepared_kernel,
        y,
    );
}

#[inline(always)]
unsafe fn convolve_segment_2d_u8_f32_impl<const FMA: bool>(
    arena: Arena,
    arena_source: &[u8],
    dst: &mut [u8],
    image_size: ImageSize,
    prepared_kernel: &[ScanPoint2d<f32>],
    y: usize,
) {
    let width = image_size.width;

    let dx = arena.pad_w as i64;
    let dy = arena.pad_h as i64;

    let arena_stride = arena.width * arena.components;

    let offsets = prepared_kernel
        .iter()
        .map(|&x| {
            arena_source.get_unchecked(
                ((x.y + dy + y as i64) as usize * arena_stride
                    + (x.x + dx) as usize * arena.components)..,
            )
        })
        .collect::<Vec<_>>();

    let length = prepared_kernel.len();

    let total_width = width * arena.components;

    let mut cx = 0usize;

    while cx + 64 < total_width {
        let k_weight = _mm_set1_ps(prepared_kernel.get_unchecked(0).weight);
        let items0 = _mm_load_pack_x4(offsets.get_unchecked(0).get_unchecked(cx..).as_ptr());
        let mut k0 = _mm_mul_epi8_by_ps_x4::<FMA>(items0.0, k_weight);
        let mut k1 = _mm_mul_epi8_by_ps_x4::<FMA>(items0.1, k_weight);
        let mut k2 = _mm_mul_epi8_by_ps_x4::<FMA>(items0.2, k_weight);
        let mut k3 = _mm_mul_epi8_by_ps_x4::<FMA>(items0.3, k_weight);
        for i in 1..length {
            let weight = _mm_set1_ps(prepared_kernel.get_unchecked(i).weight);
            let s_ptr = offsets.get_unchecked(i);
            let items0 = _mm_load_pack_x4(s_ptr.get_unchecked(cx..).as_ptr());
            k0 = _mm_mul_add_epi8_by_ps_x4::<FMA>(k0, items0.0, weight);
            k1 = _mm_mul_add_epi8_by_ps_x4::<FMA>(k1, items0.1, weight);
            k2 = _mm_mul_add_epi8_by_ps_x4::<FMA>(k2, items0.2, weight);
            k3 = _mm_mul_add_epi8_by_ps_x4::<FMA>(k3, items0.3, weight);
        }
        let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
        _mm_store_pack_x4(
            dst_ptr0,
            (
                _mm_pack_ps_x4_epi8(k0),
                _mm_pack_ps_x4_epi8(k1),
                _mm_pack_ps_x4_epi8(k2),
                _mm_pack_ps_x4_epi8(k3),
            ),
        );
        cx += 64;
    }

    while cx + 32 < total_width {
        let k_weight = _mm_set1_ps(prepared_kernel.get_unchecked(0).weight);
        let items0 = _mm_load_pack_x2(offsets.get_unchecked(0).get_unchecked(cx..).as_ptr());
        let mut k0 = _mm_mul_epi8_by_ps_x4::<FMA>(items0.0, k_weight);
        let mut k1 = _mm_mul_epi8_by_ps_x4::<FMA>(items0.1, k_weight);
        for i in 1..length {
            let weight = _mm_set1_ps(prepared_kernel.get_unchecked(i).weight);
            let s_ptr = offsets.get_unchecked(i);
            let items0 = _mm_load_pack_x2(s_ptr.get_unchecked(cx..).as_ptr());
            k0 = _mm_mul_add_epi8_by_ps_x4::<FMA>(k0, items0.0, weight);
            k1 = _mm_mul_add_epi8_by_ps_x4::<FMA>(k1, items0.1, weight);
        }
        let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
        _mm_store_pack_x2(dst_ptr0, (_mm_pack_ps_x4_epi8(k0), _mm_pack_ps_x4_epi8(k1)));
        cx += 32;
    }

    while cx + 16 < total_width {
        let k_weight = _mm_set1_ps(prepared_kernel.get_unchecked(0).weight);
        let items0 = _mm_loadu_si128(
            offsets.get_unchecked(0).get_unchecked(cx..).as_ptr() as *const __m128i
        );
        let mut k0 = _mm_mul_epi8_by_ps_x4::<FMA>(items0, k_weight);
        for i in 1..length {
            let weight = _mm_set1_ps(prepared_kernel.get_unchecked(i).weight);
            let items0 = _mm_loadu_si128(
                offsets.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const __m128i
            );
            k0 = _mm_mul_add_epi8_by_ps_x4::<FMA>(k0, items0, weight);
        }
        let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
        _mm_storeu_si128(dst_ptr0 as *mut __m128i, _mm_pack_ps_x4_epi8(k0));
        cx += 16;
    }

    while cx + 4 < total_width {
        let k_weight = prepared_kernel.get_unchecked(0).weight;

        let mut k0 = ((*offsets.get_unchecked(0).get_unchecked(cx)) as f32).mul(k_weight);
        let mut k1 = ((*offsets.get_unchecked(0).get_unchecked(cx + 1)) as f32).mul(k_weight);
        let mut k2 = ((*offsets.get_unchecked(0).get_unchecked(cx + 2)) as f32).mul(k_weight);
        let mut k3 = ((*offsets.get_unchecked(0).get_unchecked(cx + 3)) as f32).mul(k_weight);

        for i in 1..length {
            let weight = prepared_kernel.get_unchecked(i).weight;
            k0 = mlaf(
                k0,
                (*offsets.get_unchecked(i).get_unchecked(cx)) as f32,
                weight,
            );
            k1 = mlaf(
                k1,
                (*offsets.get_unchecked(i).get_unchecked(cx + 1)) as f32,
                weight,
            );
            k2 = mlaf(
                k2,
                (*offsets.get_unchecked(i).get_unchecked(cx + 2)) as f32,
                weight,
            );
            k3 = mlaf(
                k3,
                (*offsets.get_unchecked(i).get_unchecked(cx + 3)) as f32,
                weight,
            );
        }

        *dst.get_unchecked_mut(cx) = k0.to_();
        *dst.get_unchecked_mut(cx + 1) = k1.to_();
        *dst.get_unchecked_mut(cx + 2) = k2.to_();
        *dst.get_unchecked_mut(cx + 3) = k3.to_();
        cx += 4;
    }

    for x in cx..total_width {
        let k_weight = prepared_kernel.get_unchecked(0).weight;

        let mut k0 = ((*(*offsets.get_unchecked(0)).get_unchecked(x)) as f32).mul(k_weight);

        for i in 1..length {
            let k_weight = prepared_kernel.get_unchecked(i).weight;
            k0 = mlaf(
                k0,
                (*offsets.get_unchecked(i).get_unchecked(x)) as f32,
                k_weight,
            );
        }
        *dst.get_unchecked_mut(cx) = k0.to_();
    }
}
