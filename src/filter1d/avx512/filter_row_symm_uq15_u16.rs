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
use crate::filter1d::avx512::utils::{
    _mm512_mul_add_symm_epu16_by_epu16_x4, _mm512_mul_epu16_widen, _mm512_pack_epi32_x4_epu16,
    v512_shuffle,
};
use crate::filter1d::avx512::v_load::_mm512_load_pack_x2;
use crate::filter1d::avx512::v_store::_mm512_store_pack_x2;
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::region::FilterRegion;
use crate::filter1d::to_approx_storage::ToApproxStorage;
use crate::img_size::ImageSize;
use std::arch::x86_64::*;

pub(crate) fn filter_row_avx512_symm_uq15_u16<const N: usize>(
    arena: Arena,
    arena_src: &[u16],
    dst: &mut [u16],
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<u32>],
) {
    unsafe {
        filter_row_avx512_symm_uq15_u16_impl::<N>(
            arena,
            arena_src,
            dst,
            image_size,
            filter_region,
            scanned_kernel,
        );
    }
}

#[target_feature(enable = "avx512bw")]
unsafe fn filter_row_avx512_symm_uq15_u16_impl<const N: usize>(
    arena: Arena,
    arena_src: &[u16],
    dst: &mut [u16],
    image_size: ImageSize,
    _: FilterRegion,
    scanned_kernel: &[ScanPoint1d<u32>],
) {
    let width = image_size.width;

    let src = arena_src;

    let length = scanned_kernel.len();
    let half_len = length / 2;

    let local_src = src;

    let mut cx = 0usize;

    let max_width = width * arena.components;

    let coeff = _mm512_set1_epi32(scanned_kernel.get_unchecked(half_len).weight as i32);
    let rnd = _mm512_set1_epi32((1 << 14) - 1);

    while cx + 64 < max_width {
        let shifted_src = local_src.get_unchecked(cx..);

        let source =
            _mm512_load_pack_x2(shifted_src.get_unchecked(half_len * N..).as_ptr() as *const _);
        let mut k0 = _mm512_mul_epu16_widen(source.0, coeff);
        let mut k1 = _mm512_mul_epu16_widen(source.1, coeff);

        for i in 0..half_len {
            let rollback = length - i - 1;
            let coeff = _mm512_set1_epi32(scanned_kernel.get_unchecked(i).weight as i32);
            let v_source0 =
                _mm512_load_pack_x2(shifted_src.get_unchecked((i * N)..).as_ptr() as *const _);
            let v_source1 = _mm512_load_pack_x2(
                shifted_src.get_unchecked((rollback * N)..).as_ptr() as *const _,
            );
            k0 = _mm512_mul_add_symm_epu16_by_epu16_x4(k0, v_source0.0, v_source1.0, coeff);
            k1 = _mm512_mul_add_symm_epu16_by_epu16_x4(k1, v_source0.1, v_source1.1, coeff);
        }

        let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
        _mm512_store_pack_x2(
            dst_ptr0 as *mut _,
            (
                _mm512_pack_epi32_x4_epu16(k0),
                _mm512_pack_epi32_x4_epu16(k1),
            ),
        );
        cx += 64;
    }

    while cx + 32 < max_width {
        let shifted_src = local_src.get_unchecked(cx..);

        let source =
            _mm512_loadu_si512(shifted_src.get_unchecked(half_len * N..).as_ptr() as *const _);
        let mut k0 = _mm512_mul_epu16_widen(source, coeff);

        for i in 0..half_len {
            let rollback = length - i - 1;
            let coeff = _mm512_set1_epi32(scanned_kernel.get_unchecked(i).weight as i32);
            let v_source0 =
                _mm512_loadu_si512(shifted_src.get_unchecked((i * N)..).as_ptr() as *const _);
            let v_source1 = _mm512_loadu_si512(
                shifted_src.get_unchecked((rollback * N)..).as_ptr() as *const _,
            );
            k0 = _mm512_mul_add_symm_epu16_by_epu16_x4(k0, v_source0, v_source1, coeff);
        }

        let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
        _mm512_storeu_si512(dst_ptr0 as *mut _, _mm512_pack_epi32_x4_epu16(k0));
        cx += 32;
    }

    while cx + 16 < max_width {
        let shifted_src = local_src.get_unchecked(cx..);

        let sh_mask = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

        let source =
            _mm256_loadu_si256(shifted_src.get_unchecked(half_len * N..).as_ptr() as *const _);
        let mut k0 = _mm512_mullo_epi32(
            _mm512_permutexvar_epi64(sh_mask, _mm512_castsi256_si512(source)),
            coeff,
        );

        for i in 0..half_len {
            let rollback = length - i - 1;
            let coeff = _mm512_set1_epi32(scanned_kernel.get_unchecked(i).weight as i32);
            let v_source0 = _mm512_permutexvar_epi64(
                sh_mask,
                _mm512_castsi256_si512(_mm256_loadu_si256(
                    shifted_src.get_unchecked((i * N)..).as_ptr() as *const _,
                )),
            );
            let v_source1 = _mm512_permutexvar_epi64(
                sh_mask,
                _mm512_castsi256_si512(_mm256_loadu_si256(
                    shifted_src.get_unchecked((rollback * N)..).as_ptr() as *const _,
                )),
            );

            let uexp0 = _mm512_unpacklo_epi8(v_source0, _mm512_setzero_si512());
            let uexp1 = _mm512_unpacklo_epi8(v_source1, _mm512_setzero_si512());

            k0 = _mm512_add_epi32(
                k0,
                _mm512_mullo_epi32(_mm512_add_epi32(uexp0, uexp1), coeff),
            );
        }

        let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
        _mm256_storeu_si256(
            dst_ptr0 as *mut _,
            _mm512_castsi512_si256(_mm512_packus_epi32(
                _mm512_srli_epi32::<15>(_mm512_add_epi32(k0, rnd)),
                _mm512_setzero_si512(),
            )),
        );
        cx += 16;
    }

    while cx + 8 < max_width {
        let shifted_src = local_src.get_unchecked(cx..);

        const M: i32 = v512_shuffle(3, 1, 2, 0);

        let source = _mm256_castsi128_si256(_mm_loadu_si128(
            shifted_src.get_unchecked(half_len * N..).as_ptr() as *const _,
        ));
        let mut k0 = _mm256_mullo_epi32(
            _mm256_unpacklo_epi8(
                _mm256_permute4x64_epi64::<M>(source),
                _mm256_setzero_si256(),
            ),
            _mm512_castsi512_si256(coeff),
        );

        for i in 0..half_len {
            let rollback = length - i - 1;
            let coeff = _mm256_set1_epi32(scanned_kernel.get_unchecked(i).weight as i32);
            let v_source0 = _mm256_permute4x64_epi64::<M>(_mm256_castsi128_si256(_mm_loadu_si128(
                shifted_src.get_unchecked((i * N)..).as_ptr() as *const _,
            )));
            let v_source1 = _mm256_permute4x64_epi64::<M>(_mm256_castsi128_si256(_mm_loadu_si128(
                shifted_src.get_unchecked((rollback * N)..).as_ptr() as *const _,
            )));
            k0 = _mm256_add_epi32(
                k0,
                _mm256_mullo_epi32(_mm256_add_epi32(v_source0, v_source1), coeff),
            );
        }

        let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
        _mm_storeu_si128(
            dst_ptr0 as *mut _,
            _mm256_castsi256_si128(_mm256_packus_epi32(
                _mm256_srli_epi32::<15>(_mm256_add_epi32(k0, _mm512_castsi512_si256(rnd))),
                _mm256_setzero_si256(),
            )),
        );
        cx += 8;
    }

    while cx + 4 < max_width {
        let shifted_src = local_src.get_unchecked(cx..);

        let source = _mm_loadu_si64(shifted_src.get_unchecked(half_len * N..).as_ptr() as *const _);
        let mut k0 = _mm_mullo_epi32(
            _mm_unpacklo_epi16(source, _mm_setzero_si128()),
            _mm512_castsi512_si128(coeff),
        );

        for i in 0..half_len {
            let rollback = length - i - 1;
            let coeff = _mm_set1_epi32(scanned_kernel.get_unchecked(i).weight as i32);
            let v_source0 =
                _mm_loadu_si64(shifted_src.get_unchecked((i * N)..).as_ptr() as *const _);
            let v_source1 =
                _mm_loadu_si64(shifted_src.get_unchecked((rollback * N)..).as_ptr() as *const _);
            k0 = _mm_add_epi32(
                k0,
                _mm_mullo_epi32(
                    _mm_add_epi32(
                        _mm_unpacklo_epi16(v_source0, _mm_setzero_si128()),
                        _mm_unpacklo_epi16(v_source1, _mm_setzero_si128()),
                    ),
                    coeff,
                ),
            );
        }

        let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();

        _mm_storeu_si64(
            dst_ptr0 as *mut _,
            _mm_packus_epi32(
                _mm_srli_epi32::<15>(_mm_add_epi32(k0, _mm512_castsi512_si128(rnd))),
                _mm_setzero_si128(),
            ),
        );
        cx += 4;
    }

    let coeff = *scanned_kernel.get_unchecked(half_len);

    for x in cx..max_width {
        let shifted_src = local_src.get_unchecked(x..);
        let mut k0 = *shifted_src.get_unchecked(half_len * N) as u32 * coeff.weight;

        for i in 0..half_len {
            let coeff = *scanned_kernel.get_unchecked(i);
            let rollback = length - i - 1;

            k0 += (*shifted_src.get_unchecked(i * N) as u32
                + *shifted_src.get_unchecked(rollback * N) as u32)
                * coeff.weight;
        }

        *dst.get_unchecked_mut(x) = k0.to_approx_();
    }
}
