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
use crate::neon::prefer_vfmaq_f32;
use std::arch::aarch64::*;

#[inline(always)]
pub(crate) unsafe fn vmullq_expand_i16<const EXPAND: i32>(
    input: uint8x16_t,
    weight: int16x8_t,
) -> (int16x8_t, int16x8_t) {
    let lo_16 = vreinterpretq_s16_u16(vshll_n_u8::<EXPAND>(vget_low_u8(input)));
    let hi_16 = vreinterpretq_s16_u16(vshll_high_n_u8::<EXPAND>(input));

    (vqrdmulhq_s16(lo_16, weight), vqrdmulhq_s16(hi_16, weight))
}

#[inline(always)]
pub(crate) unsafe fn vmull_expand_i16<const EXPAND: i32>(
    input: uint8x8_t,
    weight: int16x8_t,
) -> int16x8_t {
    let v = vreinterpretq_s16_u16(vshll_n_u8::<EXPAND>(input));
    vqrdmulhq_s16(v, weight)
}

#[inline(always)]
pub(crate) unsafe fn vmlaq_hi_u8_s16<const EXPAND: i32>(
    store: (int16x8_t, int16x8_t),
    input: uint8x16_t,
    weight: int16x8_t,
) -> (int16x8_t, int16x8_t) {
    let lo_16 = vreinterpretq_s16_u16(vshll_n_u8::<EXPAND>(vget_low_u8(input)));
    let hi_16 = vreinterpretq_s16_u16(vshll_high_n_u8::<EXPAND>(input));

    (
        vqrdmlahq_s16(store.0, lo_16, weight),
        vqrdmlahq_s16(store.1, hi_16, weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn vmla_symm_hi_u8_s16<const EXPAND: i32>(
    store: int16x8_t,
    input0: uint8x8_t,
    input1: uint8x8_t,
    weight: int16x8_t,
) -> int16x8_t {
    let lo_16 = vreinterpretq_s16_u16(vshlq_n_u16::<EXPAND>(vaddl_u8(input0, input1)));

    vqrdmlahq_s16(store, lo_16, weight)
}

#[inline(always)]
pub(crate) unsafe fn vmlaq_symm_hi_u8_s16<const EXPAND: i32>(
    store: (int16x8_t, int16x8_t),
    input0: uint8x16_t,
    input1: uint8x16_t,
    weight: int16x8_t,
) -> (int16x8_t, int16x8_t) {
    let lw = vaddl_u8(vget_low_u8(input0), vget_low_u8(input1));
    let hw = vaddl_high_u8(input0, input1);
    let lo_16 = vreinterpretq_s16_u16(vshlq_n_u16::<EXPAND>(lw));
    let hi_16 = vreinterpretq_s16_u16(vshlq_n_u16::<EXPAND>(hw));

    (
        vqrdmlahq_s16(store.0, lo_16, weight),
        vqrdmlahq_s16(store.1, hi_16, weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn vmullq_u8_by_i16(
    input: uint8x16_t,
    weight: int16x8_t,
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    let lo_16 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(input)));
    let hi_16 = vreinterpretq_s16_u16(vmovl_high_u8(input));

    (
        vmull_s16(vget_low_s16(lo_16), vget_low_s16(weight)),
        vmull_high_s16(lo_16, weight),
        vmull_s16(vget_low_s16(hi_16), vget_low_s16(weight)),
        vmull_high_s16(hi_16, weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn vmull_u8_by_i16(
    input: uint8x8_t,
    weight: int16x4_t,
) -> (int32x4_t, int32x4_t) {
    let lo_16 = vreinterpretq_s16_u16(vmovl_u8(input));

    (
        vmull_s16(vget_low_s16(lo_16), weight),
        vmull_s16(vget_high_s16(lo_16), weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn vmulq_u8_by_i16(
    input: uint8x16_t,
    weight: int16x8_t,
) -> (int16x8_t, int16x8_t) {
    let lw = vmovl_u8(vget_low_u8(input));
    let hw = vmovl_u8(vget_high_u8(input));
    (
        vmulq_s16(vreinterpretq_s16_u16(lw), weight),
        vmulq_s16(vreinterpretq_s16_u16(hw), weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn vmul_u8_by_i16(input: uint8x8_t, weight: int16x8_t) -> int16x8_t {
    vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(input)), weight)
}

#[inline(always)]
pub(crate) unsafe fn vmulq_u8_by_f32(
    input: uint8x16_t,
    weight: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
    let lo_16 = vmovl_u8(vget_low_u8(input));
    let o0 = vmovl_u16(vget_low_u16(lo_16));
    let o1 = vmovl_high_u16(lo_16);
    let lo_lo = vcvtq_f32_u32(o0);
    let lo_hi = vcvtq_f32_u32(o1);
    let hi_16 = vmovl_high_u8(input);

    let o2 = vmovl_u16(vget_low_u16(hi_16));
    let o3 = vmovl_high_u16(hi_16);

    let hi_lo = vcvtq_f32_u32(o2);
    let hi_hi = vcvtq_f32_u32(o3);

    (
        vmulq_f32(lo_lo, weight),
        vmulq_f32(lo_hi, weight),
        vmulq_f32(hi_lo, weight),
        vmulq_f32(hi_hi, weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn vmul_u8_by_f32(
    input: uint8x8_t,
    weight: float32x4_t,
) -> (float32x4_t, float32x4_t) {
    let lo_16 = vmovl_u8(input);

    let o0 = vmovl_u16(vget_low_u16(lo_16));
    let o1 = vmovl_high_u16(lo_16);

    let lo_lo = vcvtq_f32_u32(o0);
    let lo_hi = vcvtq_f32_u32(o1);

    (vmulq_f32(lo_lo, weight), vmulq_f32(lo_hi, weight))
}

#[inline(always)]
pub(crate) unsafe fn vfmlaq_u8_f32(
    store: (float32x4_t, float32x4_t, float32x4_t, float32x4_t),
    input: uint8x16_t,
    weight: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
    let lo_16 = vmovl_u8(vget_low_u8(input));
    let hi_16 = vmovl_high_u8(input);

    let o0 = vmovl_u16(vget_low_u16(lo_16));
    let o1 = vmovl_high_u16(lo_16);
    let o2 = vmovl_u16(vget_low_u16(hi_16));
    let o3 = vmovl_high_u16(hi_16);

    let lo_lo = vcvtq_f32_u32(o0);
    let lo_hi = vcvtq_f32_u32(o1);
    let hi_lo = vcvtq_f32_u32(o2);
    let hi_hi = vcvtq_f32_u32(o3);

    (
        prefer_vfmaq_f32(store.0, lo_lo, weight),
        prefer_vfmaq_f32(store.1, lo_hi, weight),
        prefer_vfmaq_f32(store.2, hi_lo, weight),
        prefer_vfmaq_f32(store.3, hi_hi, weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn vfmlaq_symm_u8_f32(
    store: (float32x4_t, float32x4_t, float32x4_t, float32x4_t),
    input0: uint8x16_t,
    input1: uint8x16_t,
    weight: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
    let lo_16 = vaddq_u16(vmovl_u8(vget_low_u8(input0)), vmovl_u8(vget_low_u8(input1)));
    let hi_16 = vaddq_u16(vmovl_high_u8(input0), vmovl_high_u8(input1));

    let o0 = vmovl_u16(vget_low_u16(lo_16));
    let o1 = vmovl_high_u16(lo_16);
    let o2 = vmovl_u16(vget_low_u16(hi_16));
    let o3 = vmovl_high_u16(hi_16);

    let lo_lo = vcvtq_f32_u32(o0);
    let lo_hi = vcvtq_f32_u32(o1);

    let hi_lo = vcvtq_f32_u32(o2);
    let hi_hi = vcvtq_f32_u32(o3);

    (
        prefer_vfmaq_f32(store.0, lo_lo, weight),
        prefer_vfmaq_f32(store.1, lo_hi, weight),
        prefer_vfmaq_f32(store.2, hi_lo, weight),
        prefer_vfmaq_f32(store.3, hi_hi, weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn vdotq_exact_s16(
    store: (int16x8_t, int16x8_t),
    input: uint8x16_t,
    weight: int16x8_t,
) -> (int16x8_t, int16x8_t) {
    let o0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(input)));
    let o1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(input)));
    (
        vmlaq_s16(store.0, o0, weight),
        vmlaq_s16(store.1, o1, weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn vdot_exact_s16(
    store: int16x8_t,
    input: uint8x8_t,
    weight: int16x8_t,
) -> int16x8_t {
    vmlaq_s16(store, vreinterpretq_s16_u16(vmovl_u8(input)), weight)
}

#[inline(always)]
pub(crate) unsafe fn vdot_exact_symm_s16(
    store: int16x8_t,
    input0: uint8x8_t,
    input1: uint8x8_t,
    weight: int16x8_t,
) -> int16x8_t {
    vmlaq_s16(
        store,
        vreinterpretq_s16_u16(vaddq_u16(vmovl_u8(input0), vmovl_u8(input1))),
        weight,
    )
}

#[inline(always)]
pub(crate) unsafe fn vdotq_exact_symm_s16(
    store: (int16x8_t, int16x8_t),
    input0: uint8x16_t,
    input1: uint8x16_t,
    weight: int16x8_t,
) -> (int16x8_t, int16x8_t) {
    let wide_lo = vaddl_u8(vget_low_u8(input0), vget_low_u8(input1));
    let wide_hi = vaddl_u8(vget_high_u8(input0), vget_high_u8(input1));
    (
        vaddq_s16(store.0, vmulq_s16(vreinterpretq_s16_u16(wide_lo), weight)),
        vaddq_s16(store.1, vmulq_s16(vreinterpretq_s16_u16(wide_hi), weight)),
    )
}

#[inline(always)]
pub(crate) unsafe fn vfmlaq_u8_s16(
    store: (int32x4_t, int32x4_t, int32x4_t, int32x4_t),
    input: uint8x16_t,
    weight: int16x8_t,
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    let lo_16 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(input)));
    let hi_16 = vreinterpretq_s16_u16(vmovl_high_u8(input));

    (
        vmlal_s16(store.0, vget_low_s16(lo_16), vget_low_s16(weight)),
        vmlal_high_s16(store.1, lo_16, weight),
        vmlal_s16(store.2, vget_low_s16(hi_16), vget_low_s16(weight)),
        vmlal_high_s16(store.3, hi_16, weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn vfmlaq_symm_u8_s16(
    store: (int32x4_t, int32x4_t, int32x4_t, int32x4_t),
    input0: uint8x16_t,
    input1: uint8x16_t,
    weight: int16x8_t,
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    let lo_16 = vreinterpretq_s16_u16(vaddq_u16(
        vmovl_u8(vget_low_u8(input0)),
        vmovl_u8(vget_low_u8(input1)),
    ));
    let hi_16 = vreinterpretq_s16_u16(vaddq_u16(vmovl_high_u8(input0), vmovl_high_u8(input1)));

    (
        vmlal_s16(store.0, vget_low_s16(lo_16), vget_low_s16(weight)),
        vmlal_high_s16(store.1, lo_16, weight),
        vmlal_s16(store.2, vget_low_s16(hi_16), vget_low_s16(weight)),
        vmlal_high_s16(store.3, hi_16, weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn vfmla_u8_s16(
    store: (int32x4_t, int32x4_t),
    input: uint8x8_t,
    weight: int16x4_t,
) -> (int32x4_t, int32x4_t) {
    let lo_16 = vreinterpretq_s16_u16(vmovl_u8(input));

    (
        vmlal_s16(store.0, vget_low_s16(lo_16), weight),
        vmlal_s16(store.1, vget_high_s16(lo_16), weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn vfmla_symm_u8_s16(
    store: (int32x4_t, int32x4_t),
    input0: uint8x8_t,
    input1: uint8x8_t,
    weight: int16x4_t,
) -> (int32x4_t, int32x4_t) {
    let lo_16 = vreinterpretq_s16_u16(vaddq_u16(vmovl_u8(input0), vmovl_u8(input1)));

    (
        vmlal_s16(store.0, vget_low_s16(lo_16), weight),
        vmlal_s16(store.1, vget_high_s16(lo_16), weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn vfmla_u8_f32(
    store: (float32x4_t, float32x4_t),
    input: uint8x8_t,
    weight: float32x4_t,
) -> (float32x4_t, float32x4_t) {
    let lo_16 = vmovl_u8(input);
    let lo_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_16)));
    let lo_hi = vcvtq_f32_u32(vmovl_high_u16(lo_16));

    (
        prefer_vfmaq_f32(store.0, lo_lo, weight),
        prefer_vfmaq_f32(store.1, lo_hi, weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn vfmla_symm_u8_f32(
    store: (float32x4_t, float32x4_t),
    input0: uint8x8_t,
    input1: uint8x8_t,
    weight: float32x4_t,
) -> (float32x4_t, float32x4_t) {
    let lo_16 = vaddq_u16(vmovl_u8(input0), vmovl_u8(input1));

    let a0 = vmovl_u16(vget_low_u16(lo_16));
    let a1 = vmovl_high_u16(lo_16);

    let lo_lo = vcvtq_f32_u32(a0);
    let lo_hi = vcvtq_f32_u32(a1);

    (
        prefer_vfmaq_f32(store.0, lo_lo, weight),
        prefer_vfmaq_f32(store.1, lo_hi, weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn vqmovnq_f32_u8(
    store: (float32x4_t, float32x4_t, float32x4_t, float32x4_t),
) -> uint8x16_t {
    let a0 = vcvtaq_u32_f32(store.0);
    let a1 = vcvtaq_u32_f32(store.1);
    let a2 = vcvtaq_u32_f32(store.2);
    let a3 = vcvtaq_u32_f32(store.3);
    let lo_s = vcombine_u16(vmovn_u32(a0), vmovn_u32(a1));
    let hi_s = vcombine_u16(vmovn_u32(a2), vmovn_u32(a3));
    vcombine_u8(vqmovn_u16(lo_s), vqmovn_u16(hi_s))
}

#[inline(always)]
pub(crate) unsafe fn vqmovnq_s32_u8(
    store: (int32x4_t, int32x4_t, int32x4_t, int32x4_t),
) -> uint8x16_t {
    let hi_s = vcombine_u16(vqrshrun_n_s32::<15>(store.2), vqrshrun_n_s32::<15>(store.3));
    let lo_s = vcombine_u16(vqrshrun_n_s32::<15>(store.0), vqrshrun_n_s32::<15>(store.1));
    vcombine_u8(vqmovn_u16(lo_s), vqmovn_u16(hi_s))
}

#[inline(always)]
pub(crate) unsafe fn vqmovnq_s16x2_u8<const PRECISION: i32>(
    store: (int16x8_t, int16x8_t),
) -> uint8x16_t {
    vcombine_u8(
        vqrshrun_n_s16::<PRECISION>(store.0),
        vqrshrun_n_s16::<PRECISION>(store.1),
    )
}

#[inline(always)]
pub(crate) unsafe fn vqmovn_s32_u8(store: (int32x4_t, int32x4_t)) -> uint8x8_t {
    let lo_s = vcombine_u16(vqrshrun_n_s32::<15>(store.0), vqrshrun_n_s32::<15>(store.1));
    vqmovn_u16(lo_s)
}

#[inline(always)]
pub(crate) unsafe fn vqmovn_s16_u8<const PRECISION: i32>(store: int16x8_t) -> uint8x8_t {
    vqrshrun_n_s16::<PRECISION>(store)
}

#[inline(always)]
pub(crate) unsafe fn vqmovn_f32_u8(store: (float32x4_t, float32x4_t)) -> uint8x8_t {
    let o0 = vcvtaq_u32_f32(store.0);
    let o1 = vcvtaq_u32_f32(store.1);
    let lo_s = vcombine_u16(vmovn_u32(o0), vmovn_u32(o1));
    vqmovn_u16(lo_s)
}
