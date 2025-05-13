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
use crate::avx::{_mm256_load_pack_x2, _mm256_store_pack_x2};
use crate::filter1d::arena::Arena;
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::region::FilterRegion;
use crate::img_size::ImageSize;
use std::arch::x86_64::*;
use std::ops::{Add, Mul};

pub(crate) fn filter_column_avx_symm_u8_uq0_7(
    arena: Arena,
    arena_src: &[&[u8]],
    dst: &mut [u8],
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<i32>],
) {
    unsafe {
        let mut shifted = scanned_kernel
            .iter()
            .map(|&x| ((x.weight) >> 8).min(i8::MAX as i32) as i8)
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
        filter_column_avx_symm_u8_i32_impl(
            arena,
            arena_src,
            dst,
            image_size,
            filter_region,
            &shifted,
        );
    }
}

#[target_feature(enable = "avx2")]
unsafe fn filter_column_avx_symm_u8_i32_impl(
    arena: Arena,
    arena_src: &[&[u8]],
    dst: &mut [u8],
    image_size: ImageSize,
    _: FilterRegion,
    scanned_kernel: &[i8],
) {
    unsafe {
        let image_width = image_size.width * arena.components;

        let length = scanned_kernel.len();
        let half_len = length / 2;

        let ref0 = arena_src.get_unchecked(half_len);

        let v_prepared = scanned_kernel
            .iter()
            .map(|&x| {
                let z = x.to_ne_bytes();
                i32::from_ne_bytes([z[0], z[0], z[0], z[0]])
            })
            .collect::<Vec<_>>();

        let mut cx = 0usize;

        let coeff = _mm256_set1_epi32(*v_prepared.get_unchecked(half_len));

        let rnd = _mm256_set1_epi16(1 << 6);

        while cx + 64 < image_width {
            let v_src = ref0.get_unchecked(cx..);

            let source = _mm256_load_pack_x2(v_src.as_ptr());
            let mut k0 = _mm256_add_epi16(
                rnd,
                _mm256_maddubs_epi16(
                    _mm256_unpacklo_epi8(source.0, _mm256_setzero_si256()),
                    coeff,
                ),
            );
            let mut k1 = _mm256_add_epi16(
                rnd,
                _mm256_maddubs_epi16(
                    _mm256_unpackhi_epi8(source.0, _mm256_setzero_si256()),
                    coeff,
                ),
            );
            let mut k2 = _mm256_add_epi16(
                rnd,
                _mm256_maddubs_epi16(
                    _mm256_unpacklo_epi8(source.1, _mm256_setzero_si256()),
                    coeff,
                ),
            );
            let mut k3 = _mm256_add_epi16(
                rnd,
                _mm256_maddubs_epi16(
                    _mm256_unpackhi_epi8(source.1, _mm256_setzero_si256()),
                    coeff,
                ),
            );

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = _mm256_set1_epi32(*v_prepared.get_unchecked(i));
                let v_source0 =
                    _mm256_load_pack_x2(arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr());
                let v_source1 = _mm256_load_pack_x2(
                    arena_src
                        .get_unchecked(rollback)
                        .get_unchecked(cx..)
                        .as_ptr(),
                );
                k0 = _mm256_add_epi16(
                    k0,
                    _mm256_maddubs_epi16(_mm256_unpacklo_epi8(v_source0.0, v_source1.0), coeff),
                );
                k1 = _mm256_add_epi16(
                    k1,
                    _mm256_maddubs_epi16(_mm256_unpackhi_epi8(v_source0.0, v_source1.0), coeff),
                );
                k2 = _mm256_add_epi16(
                    k2,
                    _mm256_maddubs_epi16(_mm256_unpacklo_epi8(v_source0.1, v_source1.1), coeff),
                );
                k3 = _mm256_add_epi16(
                    k3,
                    _mm256_maddubs_epi16(_mm256_unpackhi_epi8(v_source0.1, v_source1.1), coeff),
                );
            }

            k0 = _mm256_srai_epi16::<7>(k0);
            k1 = _mm256_srai_epi16::<7>(k1);
            k2 = _mm256_srai_epi16::<7>(k2);
            k3 = _mm256_srai_epi16::<7>(k3);

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            _mm256_store_pack_x2(
                dst_ptr0,
                (_mm256_packus_epi16(k0, k1), _mm256_packus_epi16(k2, k3)),
            );
            cx += 64;
        }

        while cx + 32 < image_width {
            let v_src = ref0.get_unchecked(cx..);

            let source = _mm256_loadu_si256(v_src.as_ptr() as *const __m256i);
            let mut k0 = _mm256_add_epi16(
                rnd,
                _mm256_maddubs_epi16(_mm256_unpacklo_epi8(source, _mm256_setzero_si256()), coeff),
            );
            let mut k1 = _mm256_add_epi16(
                rnd,
                _mm256_maddubs_epi16(_mm256_unpackhi_epi8(source, _mm256_setzero_si256()), coeff),
            );

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = _mm256_set1_epi32(*v_prepared.get_unchecked(i));
                let v_source0 = _mm256_loadu_si256(
                    arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const __m256i,
                );
                let v_source1 = _mm256_loadu_si256(
                    arena_src
                        .get_unchecked(rollback)
                        .get_unchecked(cx..)
                        .as_ptr() as *const __m256i,
                );
                k0 = _mm256_add_epi16(
                    k0,
                    _mm256_maddubs_epi16(_mm256_unpacklo_epi8(v_source0, v_source1), coeff),
                );
                k1 = _mm256_add_epi16(
                    k1,
                    _mm256_maddubs_epi16(_mm256_unpackhi_epi8(v_source0, v_source1), coeff),
                );
            }

            k0 = _mm256_srai_epi16::<7>(k0);
            k1 = _mm256_srai_epi16::<7>(k1);

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            _mm256_storeu_si256(dst_ptr0 as *mut __m256i, _mm256_packus_epi16(k0, k1));
            cx += 32;
        }

        while cx + 16 < image_width {
            let v_src = ref0.get_unchecked(cx..);

            let source = _mm_loadu_si128(v_src.as_ptr() as *const __m128i);
            let mut k0 = _mm_add_epi16(
                _mm256_castsi256_si128(rnd),
                _mm_maddubs_epi16(
                    _mm_unpacklo_epi8(source, _mm_setzero_si128()),
                    _mm256_castsi256_si128(coeff),
                ),
            );
            let mut k1 = _mm_add_epi16(
                _mm256_castsi256_si128(rnd),
                _mm_maddubs_epi16(
                    _mm_unpackhi_epi8(source, _mm_setzero_si128()),
                    _mm256_castsi256_si128(coeff),
                ),
            );

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = _mm_set1_epi32(*v_prepared.get_unchecked(i));
                let v_source0 = _mm_loadu_si128(
                    arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const __m128i,
                );
                let v_source1 = _mm_loadu_si128(
                    arena_src
                        .get_unchecked(rollback)
                        .get_unchecked(cx..)
                        .as_ptr() as *const __m128i,
                );
                k0 = _mm_add_epi16(
                    k0,
                    _mm_maddubs_epi16(_mm_unpacklo_epi8(v_source0, v_source1), coeff),
                );
                k1 = _mm_add_epi16(
                    k1,
                    _mm_maddubs_epi16(_mm_unpackhi_epi8(v_source0, v_source1), coeff),
                );
            }

            k0 = _mm_srai_epi16::<7>(k0);
            k1 = _mm_srai_epi16::<7>(k1);

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            _mm_storeu_si128(dst_ptr0 as *mut __m128i, _mm_packus_epi16(k0, k1));
            cx += 16;
        }

        while cx + 8 < image_width {
            let v_src = ref0.get_unchecked(cx..);

            let source = _mm_loadu_si64(v_src.as_ptr() as *const _);
            let mut k0 = _mm_add_epi16(
                _mm256_castsi256_si128(rnd),
                _mm_maddubs_epi16(
                    _mm_unpacklo_epi8(source, _mm_setzero_si128()),
                    _mm256_castsi256_si128(coeff),
                ),
            );

            for i in 0..half_len {
                let rollback = length - i - 1;
                let coeff = _mm_set1_epi32(*v_prepared.get_unchecked(i));
                let v_source0 = _mm_loadu_si64(
                    arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr() as *const _,
                );
                let v_source1 = _mm_loadu_si64(
                    arena_src
                        .get_unchecked(rollback)
                        .get_unchecked(cx..)
                        .as_ptr() as *const _,
                );
                k0 = _mm_add_epi16(
                    k0,
                    _mm_maddubs_epi16(_mm_unpacklo_epi8(v_source0, v_source1), coeff),
                );
            }

            k0 = _mm_srai_epi16::<7>(k0);

            let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
            _mm_storeu_si64(
                dst_ptr0 as *mut _,
                _mm_packus_epi16(k0, _mm_setzero_si128()),
            );
            cx += 8;
        }

        let coeff = *scanned_kernel.get_unchecked(half_len);

        for x in cx..image_width {
            let v_src = ref0.get_unchecked(x..);

            let mut k0 = ((*v_src.get_unchecked(0)) as u16).mul(coeff as u16);

            for i in 0..half_len {
                let coeff = *scanned_kernel.get_unchecked(i);
                let rollback = length - i - 1;
                k0 = ((*arena_src.get_unchecked(i).get_unchecked(x)) as u16)
                    .add((*arena_src.get_unchecked(rollback).get_unchecked(x)) as u16)
                    .mul(coeff as u16)
                    .add(k0);
            }

            *dst.get_unchecked_mut(x) = ((k0 + (1 << 6)) >> 7).min(255) as u8;
        }
    }
}
