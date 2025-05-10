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
use crate::filter1d::filter_scan::ScanPoint1d;
use crate::filter1d::region::FilterRegion;
use crate::filter1d::sse::utils::_mm_fmla_pd;
use crate::img_size::ImageSize;
use crate::sse::{_mm_load_pack_ps_x2, _mm_store_pack_ps_x2};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn filter_column_sse_f32_f64(
    arena: Arena,
    arena_src: &[&[f32]],
    dst: &mut [f32],
    image_size: ImageSize,
    filter_region: FilterRegion,
    scanned_kernel: &[ScanPoint1d<f64>],
) {
    unsafe {
        let unit = ExecutionUnit::default();
        unit.pass(
            arena,
            arena_src,
            dst,
            image_size,
            filter_region,
            scanned_kernel,
        );
    }
}

#[derive(Copy, Clone, Default)]
struct ExecutionUnit {}

impl ExecutionUnit {
    #[target_feature(enable = "sse4.1")]
    unsafe fn pass(
        &self,
        arena: Arena,
        arena_src: &[&[f32]],
        dst: &mut [f32],
        image_size: ImageSize,
        _: FilterRegion,
        scanned_kernel: &[ScanPoint1d<f64>],
    ) {
        unsafe {
            let dst_stride = image_size.width * arena.components;

            let length = scanned_kernel.len();

            let mut cx = 0usize;

            let coeff = _mm_set1_pd(scanned_kernel.get_unchecked(0).weight);

            let off0 = arena_src.get_unchecked(0);

            while cx + 8 < dst_stride {
                let v_src = arena_src.get_unchecked(0).get_unchecked(cx..);

                let source = _mm_load_pack_ps_x2(v_src.as_ptr());
                let mut k0 = _mm_mul_pd(_mm_cvtps_pd(source.0), coeff);
                let mut k1 = _mm_mul_pd(_mm_cvtps_pd(_mm_movehl_ps(source.0, source.0)), coeff);
                let mut k2 = _mm_mul_pd(_mm_cvtps_pd(source.1), coeff);
                let mut k3 = _mm_mul_pd(_mm_cvtps_pd(_mm_movehl_ps(source.1, source.1)), coeff);

                for i in 1..length {
                    let coeff = _mm_set1_pd(scanned_kernel.get_unchecked(i).weight);
                    let v_source = _mm_load_pack_ps_x2(
                        arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr(),
                    );
                    k0 = _mm_fmla_pd(k0, _mm_cvtps_pd(v_source.0), coeff);
                    k1 = _mm_fmla_pd(
                        k1,
                        _mm_cvtps_pd(_mm_movehl_ps(v_source.0, v_source.0)),
                        coeff,
                    );
                    k2 = _mm_fmla_pd(k2, _mm_cvtps_pd(v_source.1), coeff);
                    k3 = _mm_fmla_pd(
                        k3,
                        _mm_cvtps_pd(_mm_movehl_ps(v_source.1, v_source.1)),
                        coeff,
                    );
                }

                let z0 = _mm_cvtpd_ps(k0);
                let z1 = _mm_cvtpd_ps(k1);
                let z2 = _mm_cvtpd_ps(k2);
                let z3 = _mm_cvtpd_ps(k3);

                let dst_ptr0 = dst.get_unchecked_mut(cx..).as_mut_ptr();
                _mm_store_pack_ps_x2(dst_ptr0, (_mm_movelh_ps(z0, z1), _mm_movelh_ps(z2, z3)));

                cx += 8;
            }

            while cx + 4 < dst_stride {
                let v_src = off0.get_unchecked(cx..);

                let source_0 = _mm_loadu_ps(v_src.as_ptr());
                let mut k0 = _mm_mul_pd(_mm_cvtps_pd(source_0), coeff);
                let mut k1 = _mm_mul_pd(_mm_cvtps_pd(_mm_movehl_ps(source_0, source_0)), coeff);

                for i in 1..length {
                    let coeff = _mm_set1_pd(scanned_kernel.get_unchecked(i).weight);
                    let v_source_0 =
                        _mm_loadu_ps(arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr());
                    k0 = _mm_fmla_pd(k0, _mm_cvtps_pd(v_source_0), coeff);
                    k1 = _mm_fmla_pd(
                        k1,
                        _mm_cvtps_pd(_mm_movehl_ps(v_source_0, v_source_0)),
                        coeff,
                    );
                }

                let z0 = _mm_cvtpd_ps(k0);
                let z1 = _mm_cvtpd_ps(k1);

                let dst_ptr = dst.get_unchecked_mut(cx..).as_mut_ptr();
                _mm_storeu_ps(dst_ptr, _mm_movelh_ps(z0, z1));
                cx += 4;
            }

            while cx < dst_stride {
                let v_src = off0.get_unchecked(cx..);

                let source_0 = _mm_load_ss(v_src.as_ptr());
                let mut k0 = _mm_mul_pd(_mm_cvtps_pd(source_0), coeff);

                for i in 1..length {
                    let coeff = *scanned_kernel.get_unchecked(i);
                    let v_source_0 =
                        _mm_load_ss(arena_src.get_unchecked(i).get_unchecked(cx..).as_ptr());
                    k0 = _mm_fmla_pd(k0, _mm_cvtps_pd(v_source_0), _mm_set1_pd(coeff.weight));
                }

                let z0 = _mm_cvtpd_ps(k0);

                let dst_ptr = dst.get_unchecked_mut(cx..).as_mut_ptr();
                _mm_store_ss(dst_ptr, z0);
                cx += 1;
            }
        }
    }
}
