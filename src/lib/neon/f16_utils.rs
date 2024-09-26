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
use std::arch::aarch64::*;
use std::arch::asm;

/// Provides basic support for f16

#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
#[allow(dead_code)]
pub struct x_float16x4_t(pub(crate) uint16x4_t);

#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
#[allow(dead_code)]
pub struct x_float16x8_t(pub(crate) uint16x8_t);

#[inline]
pub unsafe fn xvld_f16(ptr: *const half::f16) -> x_float16x4_t {
    let store: uint16x4_t = vld1_u16(std::mem::transmute::<*const half::f16, *const u16>(ptr));
    std::mem::transmute(store)
}

#[inline]
pub unsafe fn xreinterpret_u16_f16(x: x_float16x4_t) -> uint16x4_t {
    std::mem::transmute(x)
}

#[inline]
pub unsafe fn xreinterpret_f16_u16(x: uint16x4_t) -> x_float16x4_t {
    std::mem::transmute(x)
}

// #[inline]
// pub unsafe fn xreinterpretq_f16_u16(x: uint16x8_t) -> x_float16x8_t {
//     std::mem::transmute(x)
// }

#[inline]
pub unsafe fn xvcvt_f32_f16(x: x_float16x4_t) -> float32x4_t {
    let src: uint16x4_t = xreinterpret_u16_f16(x);
    let dst: float32x4_t;
    asm!(
    "fcvtl {0:v}.4s, {1:v}.4h",
    out(vreg) dst,
    in(vreg) src,
    options(pure, nomem, nostack));
    dst
}

#[inline]
pub(super) unsafe fn xvcvt_f16_f32(v: float32x4_t) -> x_float16x4_t {
    let result: uint16x4_t;
    asm!(
    "fcvtn {0:v}.4h, {1:v}.4s",
    out(vreg) result,
    in(vreg) v,
    options(pure, nomem, nostack));
    xreinterpret_f16_u16(result)
}

#[inline]
pub unsafe fn xvst_f16(ptr: *mut half::f16, x: x_float16x4_t) {
    vst1_u16(
        std::mem::transmute::<*mut half::f16, *mut u16>(ptr),
        xreinterpret_u16_f16(x),
    )
}
