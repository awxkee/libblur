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
use crate::avx::{_mm256_load_pack_x2, _mm256_store_pack_x2, shuffle};
use crate::filter1d::{Arena, ToApproxStorage};
use crate::filter2d::scan_point_2d::ScanPoint2d;
use crate::mlaf::mlaf;
use crate::ImageSize;
use std::arch::x86_64::*;
use std::ops::Mul;

pub(crate) fn convolve_segment_sse_2d_u8_i16_fp(
    arena: Arena,
    arena_source: &[u8],
    dst: &mut [u8],
    image_size: ImageSize,
    prepared_kernel: &[ScanPoint2d<i16>],
    y: usize,
) {
    unsafe {
        convolve_segment_2d_u8_i16_impl(arena, arena_source, dst, image_size, prepared_kernel, y);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn convolve_segment_2d_u8_i16_impl(
    arena: Arena,
    arena_source: &[u8],
    dst: &mut [u8],
    image_size: ImageSize,
    prepared_kernel: &[ScanPoint2d<i16>],
    y: usize,
) {
    let width = image_size.width;

    let dx = arena.pad_w as i64;
    let dy = arena.pad_h as i64;

    let arena_stride = arena.width * arena.components;

    let v_prepared = prepared_kernel
        .iter()
        .map(|&x| {
            let z = x.weight.to_ne_bytes();
            i32::from_ne_bytes([z[0], z[1], z[0], z[1]])
        })
        .collect::<Vec<_>>();

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

    let k_weight = _mm256_set1_epi32(*v_prepared.get_unchecked(0));

    let rnd = _mm256_set1_epi32(1i32 << 14);

    let off0 = offsets.get_unchecked(0);

    while cx + 64 < total_width {
        let items0 = _mm256_load_pack_x2(off0.get_unchecked(cx..).as_ptr());

        let lo0 = _mm256_unpacklo_epi8(items0.0, _mm256_setzero_si256());
        let hi0 = _mm256_unpackhi_epi8(items0.0, _mm256_setzero_si256());
        let hi1 = _mm256_unpackhi_epi8(items0.1, _mm256_setzero_si256());
        let lo1 = _mm256_unpacklo_epi8(items0.1, _mm256_setzero_si256());

        let mut k0 = _mm256_add_epi32(
            rnd,
            _mm256_madd_epi16(_mm256_unpacklo_epi16(lo0, _mm256_setzero_si256()), k_weight),
        );
        let mut k1 = _mm256_add_epi32(
            rnd,
            _mm256_madd_epi16(_mm256_unpackhi_epi16(lo0, _mm256_setzero_si256()), k_weight),
        );
        let mut k2 = _mm256_add_epi32(
            rnd,
            _mm256_madd_epi16(_mm256_unpacklo_epi16(hi0, _mm256_setzero_si256()), k_weight),
        );
        let mut k3 = _mm256_add_epi32(
            rnd,
            _mm256_madd_epi16(_mm256_unpackhi_epi16(hi0, _mm256_setzero_si256()), k_weight),
        );
        let mut k4 = _mm256_add_epi32(
            rnd,
            _mm256_madd_epi16(_mm256_unpacklo_epi16(lo1, _mm256_setzero_si256()), k_weight),
        );
        let mut k5 = _mm256_add_epi32(
            rnd,
            _mm256_madd_epi16(_mm256_unpackhi_epi16(lo1, _mm256_setzero_si256()), k_weight),
        );
        let mut k6 = _mm256_add_epi32(
            rnd,
            _mm256_madd_epi16(_mm256_unpacklo_epi16(hi1, _mm256_setzero_si256()), k_weight),
        );
        let mut k7 = _mm256_add_epi32(
            rnd,
            _mm256_madd_epi16(_mm256_unpackhi_epi16(hi1, _mm256_setzero_si256()), k_weight),
        );

        for i in 1..length {
            let weight = _mm256_set1_epi32(*v_prepared.get_unchecked(i));
            let s_ptr = offsets.get_unchecked(i);
            let items0 = _mm256_load_pack_x2(s_ptr.get_unchecked(cx..).as_ptr());

            let lo0 = _mm256_unpacklo_epi8(items0.0, _mm256_setzero_si256());
            let hi0 = _mm256_unpackhi_epi8(items0.0, _mm256_setzero_si256());
            let hi1 = _mm256_unpackhi_epi8(items0.1, _mm256_setzero_si256());
            let lo1 = _mm256_unpacklo_epi8(items0.1, _mm256_setzero_si256());

            k0 = _mm256_add_epi32(
                k0,
                _mm256_madd_epi16(_mm256_unpacklo_epi16(lo0, _mm256_setzero_si256()), weight),
            );
            k1 = _mm256_add_epi32(
                k1,
                _mm256_madd_epi16(_mm256_unpackhi_epi16(lo0, _mm256_setzero_si256()), weight),
            );
            k2 = _mm256_add_epi32(
                k2,
                _mm256_madd_epi16(_mm256_unpacklo_epi16(hi0, _mm256_setzero_si256()), weight),
            );
            k3 = _mm256_add_epi32(
                k3,
                _mm256_madd_epi16(_mm256_unpackhi_epi16(hi0, _mm256_setzero_si256()), weight),
            );
            k4 = _mm256_add_epi32(
                k4,
                _mm256_madd_epi16(_mm256_unpacklo_epi16(lo1, _mm256_setzero_si256()), weight),
            );
            k5 = _mm256_add_epi32(
                k5,
                _mm256_madd_epi16(_mm256_unpackhi_epi16(lo1, _mm256_setzero_si256()), weight),
            );
            k6 = _mm256_add_epi32(
                k6,
                _mm256_madd_epi16(_mm256_unpacklo_epi16(hi1, _mm256_setzero_si256()), weight),
            );
            k7 = _mm256_add_epi32(
                k7,
                _mm256_madd_epi16(_mm256_unpackhi_epi16(hi1, _mm256_setzero_si256()), weight),
            );
        }

        k0 = _mm256_srai_epi32::<15>(k0);
        k1 = _mm256_srai_epi32::<15>(k1);
        k2 = _mm256_srai_epi32::<15>(k2);
        k3 = _mm256_srai_epi32::<15>(k3);
        k4 = _mm256_srai_epi32::<15>(k4);
        k5 = _mm256_srai_epi32::<15>(k5);
        k6 = _mm256_srai_epi32::<15>(k6);
        k7 = _mm256_srai_epi32::<15>(k7);

        let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();

        let z0 = _mm256_packus_epi32(k0, k1);
        let z1 = _mm256_packus_epi32(k2, k3);
        let z2 = _mm256_packus_epi32(k4, k5);
        let z3 = _mm256_packus_epi32(k6, k7);

        let r0 = _mm256_packus_epi16(z0, z1);
        let r1 = _mm256_packus_epi16(z2, z3);

        _mm256_store_pack_x2(dst_ptr0, (r0, r1));
        cx += 64;
    }

    while cx + 32 < total_width {
        let items0 = _mm256_loadu_si256(off0.get_unchecked(cx..).as_ptr() as *const _);

        let lo0 = _mm256_unpacklo_epi8(items0, _mm256_setzero_si256());
        let hi0 = _mm256_unpackhi_epi8(items0, _mm256_setzero_si256());

        let mut k0 = _mm256_add_epi32(
            rnd,
            _mm256_madd_epi16(_mm256_unpacklo_epi16(lo0, _mm256_setzero_si256()), k_weight),
        );
        let mut k1 = _mm256_add_epi32(
            rnd,
            _mm256_madd_epi16(_mm256_unpackhi_epi16(lo0, _mm256_setzero_si256()), k_weight),
        );
        let mut k2 = _mm256_add_epi32(
            rnd,
            _mm256_madd_epi16(_mm256_unpacklo_epi16(hi0, _mm256_setzero_si256()), k_weight),
        );
        let mut k3 = _mm256_add_epi32(
            rnd,
            _mm256_madd_epi16(_mm256_unpackhi_epi16(hi0, _mm256_setzero_si256()), k_weight),
        );

        for i in 1..length {
            let weight = _mm256_set1_epi32(*v_prepared.get_unchecked(i));
            let s_ptr = offsets.get_unchecked(i);
            let items0 = _mm256_loadu_si256(s_ptr.get_unchecked(cx..).as_ptr() as *const _);

            let lo0 = _mm256_unpacklo_epi8(items0, _mm256_setzero_si256());
            let hi0 = _mm256_unpackhi_epi8(items0, _mm256_setzero_si256());

            k0 = _mm256_add_epi32(
                k0,
                _mm256_madd_epi16(_mm256_unpacklo_epi16(lo0, _mm256_setzero_si256()), weight),
            );
            k1 = _mm256_add_epi32(
                k1,
                _mm256_madd_epi16(_mm256_unpackhi_epi16(lo0, _mm256_setzero_si256()), weight),
            );
            k2 = _mm256_add_epi32(
                k2,
                _mm256_madd_epi16(_mm256_unpacklo_epi16(hi0, _mm256_setzero_si256()), weight),
            );
            k3 = _mm256_add_epi32(
                k3,
                _mm256_madd_epi16(_mm256_unpackhi_epi16(hi0, _mm256_setzero_si256()), weight),
            );
        }

        k0 = _mm256_srai_epi32::<15>(k0);
        k1 = _mm256_srai_epi32::<15>(k1);
        k2 = _mm256_srai_epi32::<15>(k2);
        k3 = _mm256_srai_epi32::<15>(k3);

        let z0 = _mm256_packus_epi32(k0, k1);
        let z1 = _mm256_packus_epi32(k2, k3);

        let r0 = _mm256_packus_epi16(z0, z1);

        let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
        _mm256_storeu_si256(dst_ptr0 as *mut _, r0);
        cx += 32;
    }

    while cx + 16 < total_width {
        let items0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
            off0.get_unchecked(cx..).as_ptr() as *const __m128i
        ));

        let mut k0 = _mm256_add_epi32(
            rnd,
            _mm256_madd_epi16(
                _mm256_unpacklo_epi16(items0, _mm256_setzero_si256()),
                k_weight,
            ),
        );
        let mut k1 = _mm256_add_epi32(
            rnd,
            _mm256_madd_epi16(
                _mm256_unpackhi_epi16(items0, _mm256_setzero_si256()),
                k_weight,
            ),
        );

        for i in 1..length {
            let weight = _mm256_set1_epi32(*v_prepared.get_unchecked(i));
            let items0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                offsets.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const __m128i,
            ));
            k0 = _mm256_add_epi32(
                k0,
                _mm256_madd_epi16(
                    _mm256_unpacklo_epi16(items0, _mm256_setzero_si256()),
                    weight,
                ),
            );
            k1 = _mm256_add_epi32(
                k1,
                _mm256_madd_epi16(
                    _mm256_unpackhi_epi16(items0, _mm256_setzero_si256()),
                    weight,
                ),
            );
        }

        k0 = _mm256_srai_epi32::<15>(k0);
        k1 = _mm256_srai_epi32::<15>(k1);

        let z0 = _mm256_packus_epi32(k0, k1);

        let p0 = _mm256_permute4x64_epi64::<{ shuffle(3, 1, 2, 0) }>(_mm256_packus_epi16(
            z0,
            _mm256_setzero_si256(),
        ));

        let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
        _mm_storeu_si128(dst_ptr0 as *mut __m128i, _mm256_castsi256_si128(p0));
        cx += 16;
    }

    let k_weight = prepared_kernel.get_unchecked(0).weight;

    while cx + 4 < total_width {
        let mut k0 = ((*off0.get_unchecked(cx)) as i32).mul(k_weight as i32);
        let mut k1 = ((*off0.get_unchecked(cx + 1)) as i32).mul(k_weight as i32);
        let mut k2 = ((*off0.get_unchecked(cx + 2)) as i32).mul(k_weight as i32);
        let mut k3 = ((*off0.get_unchecked(cx + 3)) as i32).mul(k_weight as i32);

        for i in 1..length {
            let weight = prepared_kernel.get_unchecked(i).weight;
            k0 = mlaf(
                k0,
                (*offsets.get_unchecked(i).get_unchecked(cx)) as i32,
                weight as i32,
            );
            k1 = mlaf(
                k1,
                (*offsets.get_unchecked(i).get_unchecked(cx + 1)) as i32,
                weight as i32,
            );
            k2 = mlaf(
                k2,
                (*offsets.get_unchecked(i).get_unchecked(cx + 2)) as i32,
                weight as i32,
            );
            k3 = mlaf(
                k3,
                (*offsets.get_unchecked(i).get_unchecked(cx + 3)) as i32,
                weight as i32,
            );
        }

        *dst.get_unchecked_mut(cx) = k0.to_approx_();
        *dst.get_unchecked_mut(cx + 1) = k1.to_approx_();
        *dst.get_unchecked_mut(cx + 2) = k2.to_approx_();
        *dst.get_unchecked_mut(cx + 3) = k3.to_approx_();
        cx += 4;
    }

    for x in cx..total_width {
        let mut k0 = ((*(*off0).get_unchecked(x)) as i32).mul(k_weight as i32);

        for i in 1..length {
            let k_weight = prepared_kernel.get_unchecked(i).weight;
            k0 = mlaf(
                k0,
                (*offsets.get_unchecked(i).get_unchecked(x)) as i32,
                k_weight as i32,
            );
        }
        *dst.get_unchecked_mut(x) = k0.to_approx_();
    }
}
