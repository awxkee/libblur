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

use crate::neon::f16_utils::{
    xreinterpret_f16_u16, xreinterpret_u16_f16, xvcvt_f16_f32, xvcvt_f32_f16, xvld_f16, xvst_f16,
};
use half::f16;
use std::arch::aarch64::*;

#[inline(always)]
pub unsafe fn load_u8_s32_fast<const CHANNELS_COUNT: usize>(ptr: *const u8) -> int32x4_t {
    vreinterpretq_s32_u32(load_u8_u32_fast::<CHANNELS_COUNT>(ptr))
}

#[inline(always)]
pub unsafe fn store_f32<const CHANNELS_COUNT: usize>(dst_ptr: *mut f32, regi: float32x4_t) {
    if CHANNELS_COUNT == 4 {
        vst1q_f32(dst_ptr, regi);
    } else if CHANNELS_COUNT == 3 {
        vst1q_lane_f64::<0>(dst_ptr as *mut f64, vreinterpretq_f64_f32(regi));
        vst1q_lane_f32::<2>(dst_ptr.add(2), regi);
    } else if CHANNELS_COUNT == 2 {
        vst1q_lane_f64::<0>(dst_ptr as *mut f64, vreinterpretq_f64_f32(regi));
    } else {
        vst1q_lane_f32::<0>(dst_ptr, regi);
    }
}

#[inline(always)]
pub unsafe fn store_u8_s32<const CHANNELS_COUNT: usize>(dst_ptr: *mut u8, regi: int32x4_t) {
    let s16 = vqmovun_s32(regi);
    let u16_f = vcombine_u16(s16, s16);
    let v8 = vqmovn_u16(u16_f);
    if CHANNELS_COUNT == 4 {
        vst1_lane_u32::<0>(dst_ptr as *mut u32, vreinterpret_u32_u8(v8));
    } else if CHANNELS_COUNT == 3 {
        vst1_lane_u16::<0>(dst_ptr as *mut u16, vreinterpret_u16_u8(v8));
        vst1_lane_u8::<2>(dst_ptr.add(2), v8);
    } else if CHANNELS_COUNT == 2 {
        vst1_lane_u16::<0>(dst_ptr as *mut u16, vreinterpret_u16_u8(v8));
    } else {
        vst1_lane_u8::<0>(dst_ptr, v8);
    }
}

#[inline(always)]
pub unsafe fn store_u8_s32_x4<const CHANNELS_COUNT: usize>(
    dst_ptr: (*mut u8, *mut u8, *mut u8, *mut u8),
    regi: int32x4x4_t,
) {
    let s16_0 = vreinterpret_u16_s16(vqmovn_s32(regi.0));
    let s16_1 = vreinterpret_u16_s16(vqmovn_s32(regi.1));
    let s16_2 = vreinterpret_u16_s16(vqmovn_s32(regi.2));
    let s16_3 = vreinterpret_u16_s16(vqmovn_s32(regi.3));
    let u16_f_0 = vcombine_u16(s16_0, s16_0);
    let u16_f_1 = vcombine_u16(s16_1, s16_1);
    let u16_f_2 = vcombine_u16(s16_2, s16_2);
    let u16_f_3 = vcombine_u16(s16_3, s16_3);
    let v8_0 = vqmovn_u16(u16_f_0);
    let v8_1 = vqmovn_u16(u16_f_1);
    let v8_2 = vqmovn_u16(u16_f_2);
    let v8_3 = vqmovn_u16(u16_f_3);
    if CHANNELS_COUNT == 4 {
        vst1_lane_u32::<0>(dst_ptr.0 as *mut u32, vreinterpret_u32_u8(v8_0));
        vst1_lane_u32::<0>(dst_ptr.1 as *mut u32, vreinterpret_u32_u8(v8_1));
        vst1_lane_u32::<0>(dst_ptr.2 as *mut u32, vreinterpret_u32_u8(v8_2));
        vst1_lane_u32::<0>(dst_ptr.3 as *mut u32, vreinterpret_u32_u8(v8_3));
    } else if CHANNELS_COUNT == 3 {
        vst1_lane_u16::<0>(dst_ptr.0 as *mut u16, vreinterpret_u16_u8(v8_0));
        vst1_lane_u8::<2>(dst_ptr.0.add(2), v8_0);
        vst1_lane_u16::<0>(dst_ptr.1 as *mut u16, vreinterpret_u16_u8(v8_1));
        vst1_lane_u8::<2>(dst_ptr.1.add(2), v8_1);
        vst1_lane_u16::<0>(dst_ptr.2 as *mut u16, vreinterpret_u16_u8(v8_2));
        vst1_lane_u8::<2>(dst_ptr.2.add(2), v8_2);
        vst1_lane_u16::<0>(dst_ptr.3 as *mut u16, vreinterpret_u16_u8(v8_3));
        vst1_lane_u8::<2>(dst_ptr.3.add(2), v8_3);
    } else if CHANNELS_COUNT == 2 {
        vst1_lane_u16::<0>(dst_ptr.0 as *mut u16, vreinterpret_u16_u8(v8_0));
        vst1_lane_u16::<0>(dst_ptr.1 as *mut u16, vreinterpret_u16_u8(v8_1));
        vst1_lane_u16::<0>(dst_ptr.2 as *mut u16, vreinterpret_u16_u8(v8_2));
        vst1_lane_u16::<0>(dst_ptr.3 as *mut u16, vreinterpret_u16_u8(v8_3));
    } else {
        vst1_lane_u8::<0>(dst_ptr.0, v8_0);
        vst1_lane_u8::<0>(dst_ptr.1, v8_1);
        vst1_lane_u8::<0>(dst_ptr.2, v8_2);
        vst1_lane_u8::<0>(dst_ptr.3, v8_3);
    }
}

#[allow(dead_code)]
#[inline(always)]
pub unsafe fn store_u8_s32_x5<const CHANNELS_COUNT: usize>(
    dst_ptr: (*mut u8, *mut u8, *mut u8, *mut u8, *mut u8),
    regi: int32x4x4_t,
    add: int32x4_t,
) {
    let s16_0 = vreinterpret_u16_s16(vqmovn_s32(regi.0));
    let s16_1 = vreinterpret_u16_s16(vqmovn_s32(regi.1));
    let s16_2 = vreinterpret_u16_s16(vqmovn_s32(regi.2));
    let s16_3 = vreinterpret_u16_s16(vqmovn_s32(regi.3));
    let s16_4 = vreinterpret_u16_s16(vqmovn_s32(add));
    let u16_f_0 = vcombine_u16(s16_0, s16_0);
    let u16_f_1 = vcombine_u16(s16_1, s16_1);
    let u16_f_2 = vcombine_u16(s16_2, s16_2);
    let u16_f_3 = vcombine_u16(s16_3, s16_3);
    let u16_f_4 = vcombine_u16(s16_4, s16_4);
    let v8_0 = vqmovn_u16(u16_f_0);
    let v8_1 = vqmovn_u16(u16_f_1);
    let v8_2 = vqmovn_u16(u16_f_2);
    let v8_3 = vqmovn_u16(u16_f_3);
    let v8_4 = vqmovn_u16(u16_f_4);
    if CHANNELS_COUNT == 4 {
        vst1_lane_u32::<0>(dst_ptr.0 as *mut u32, vreinterpret_u32_u8(v8_0));
        vst1_lane_u32::<0>(dst_ptr.1 as *mut u32, vreinterpret_u32_u8(v8_1));
        vst1_lane_u32::<0>(dst_ptr.2 as *mut u32, vreinterpret_u32_u8(v8_2));
        vst1_lane_u32::<0>(dst_ptr.3 as *mut u32, vreinterpret_u32_u8(v8_3));
        vst1_lane_u32::<0>(dst_ptr.4 as *mut u32, vreinterpret_u32_u8(v8_4));
    } else if CHANNELS_COUNT == 3 {
        vst1_lane_u16::<0>(dst_ptr.0 as *mut u16, vreinterpret_u16_u8(v8_0));
        vst1_lane_u8::<2>(dst_ptr.0.add(2), v8_0);
        vst1_lane_u16::<0>(dst_ptr.1 as *mut u16, vreinterpret_u16_u8(v8_1));
        vst1_lane_u8::<2>(dst_ptr.1.add(2), v8_1);
        vst1_lane_u16::<0>(dst_ptr.2 as *mut u16, vreinterpret_u16_u8(v8_2));
        vst1_lane_u8::<2>(dst_ptr.2.add(2), v8_2);
        vst1_lane_u16::<0>(dst_ptr.3 as *mut u16, vreinterpret_u16_u8(v8_3));
        vst1_lane_u8::<2>(dst_ptr.3.add(2), v8_3);
        vst1_lane_u16::<0>(dst_ptr.4 as *mut u16, vreinterpret_u16_u8(v8_4));
        vst1_lane_u8::<2>(dst_ptr.4.add(2), v8_4);
    } else if CHANNELS_COUNT == 2 {
        vst1_lane_u16::<0>(dst_ptr.0 as *mut u16, vreinterpret_u16_u8(v8_0));
        vst1_lane_u16::<0>(dst_ptr.1 as *mut u16, vreinterpret_u16_u8(v8_1));
        vst1_lane_u16::<0>(dst_ptr.2 as *mut u16, vreinterpret_u16_u8(v8_2));
        vst1_lane_u16::<0>(dst_ptr.3 as *mut u16, vreinterpret_u16_u8(v8_3));
        vst1_lane_u16::<0>(dst_ptr.4 as *mut u16, vreinterpret_u16_u8(v8_4));
    } else {
        vst1_lane_u8::<0>(dst_ptr.0, v8_0);
        vst1_lane_u8::<0>(dst_ptr.1, v8_1);
        vst1_lane_u8::<0>(dst_ptr.2, v8_2);
        vst1_lane_u8::<0>(dst_ptr.3, v8_3);
        vst1_lane_u8::<0>(dst_ptr.4, v8_4);
    }
}

#[inline(always)]
pub unsafe fn store_u8_u32<const CHANNELS_COUNT: usize>(dst_ptr: *mut u8, regi: uint32x4_t) {
    let s16 = vqmovn_u32(regi);
    let u16_f = vcombine_u16(s16, s16);
    let v8 = vqmovn_u16(u16_f);
    if CHANNELS_COUNT == 4 {
        vst1_lane_u32::<0>(dst_ptr as *mut u32, vreinterpret_u32_u8(v8));
    } else if CHANNELS_COUNT == 3 {
        vst1_lane_u16::<0>(dst_ptr as *mut u16, vreinterpret_u16_u8(v8));
        vst1_lane_u8::<2>(dst_ptr.add(2), v8);
    } else if CHANNELS_COUNT == 2 {
        vst1_lane_u16::<0>(dst_ptr as *mut u16, vreinterpret_u16_u8(v8));
    } else {
        vst1_lane_u8::<0>(dst_ptr, v8);
    }
}

#[inline(always)]
pub unsafe fn load_f32_fast<const CHANNELS_COUNT: usize>(ptr: *const f32) -> float32x4_t {
    if CHANNELS_COUNT == 4 {
        return vld1q_f32(ptr);
    } else if CHANNELS_COUNT == 3 {
        return vld1q_f32(
            [
                ptr.read_unaligned(),
                ptr.add(1).read_unaligned(),
                ptr.add(2).read_unaligned(),
                0f32,
            ]
            .as_ptr(),
        );
    } else if CHANNELS_COUNT == 2 {
        return vld1q_f32(
            [
                ptr.read_unaligned(),
                ptr.add(1).read_unaligned(),
                0f32,
                0f32,
            ]
            .as_ptr(),
        );
    }
    vld1q_f32([ptr.read_unaligned(), 0f32, 0f32, 0f32].as_ptr())
}

#[inline(always)]
pub unsafe fn load_u8_u32_fast<const CHANNELS_COUNT: usize>(ptr: *const u8) -> uint32x4_t {
    if CHANNELS_COUNT == 4 {
        let v0 = vreinterpret_u8_u32(vld1_lane_u32::<0>(ptr as *const u32, vdup_n_u32(0)));
        vmovl_u16(vget_low_u16(vmovl_u8(v0)))
    } else if CHANNELS_COUNT == 3 {
        let mut v0 = vreinterpret_u8_u16(vld1_lane_u16::<0>(ptr as *const u16, vdup_n_u16(0)));
        v0 = vld1_lane_u8::<2>(ptr.add(2), v0);
        vmovl_u16(vget_low_u16(vmovl_u8(v0)))
    } else if CHANNELS_COUNT == 2 {
        let v0 = vreinterpret_u8_u16(vld1_lane_u16::<0>(ptr as *const u16, vdup_n_u16(0)));
        vmovl_u16(vget_low_u16(vmovl_u8(v0)))
    } else {
        vmovl_u16(vget_low_u16(vmovl_u8(vld1_lane_u8::<0>(ptr, vdup_n_u8(0)))))
    }
}

#[inline(always)]
pub unsafe fn load_u16_s32_fast<const CN: usize>(ptr: *const u16) -> int32x4_t {
    if CN == 4 {
        vreinterpretq_s32_u32(vmovl_u16(vld1_u16(ptr)))
    } else if CN == 3 {
        let mut v0 = vreinterpret_u16_u32(vld1_lane_u32::<0>(ptr as *const u32, vdup_n_u32(0)));
        v0 = vld1_lane_u16::<2>(ptr.add(2), v0);
        vreinterpretq_s32_u32(vmovl_u16(v0))
    } else if CN == 2 {
        let v0 = vreinterpret_u16_u32(vld1_lane_u32::<0>(ptr as *const u32, vdup_n_u32(0)));
        vreinterpretq_s32_u32(vmovl_u16(v0))
    } else {
        vreinterpretq_s32_u32(vmovl_u16(vld1_lane_u16::<0>(ptr, vdup_n_u16(0))))
    }
}

#[inline(always)]
pub(crate) unsafe fn vmulq_u32_f32(a: uint32x4_t, b: float32x4_t) -> uint32x4_t {
    let cvt = vcvtq_f32_u32(a);
    vcvtaq_u32_f32(vmulq_f32(cvt, b))
}

#[inline(always)]
pub(crate) unsafe fn vmulq_u16_low_f32(a: uint16x8_t, b: float32x4_t) -> uint32x4_t {
    let cvt = vcvtq_f32_u32(vmovl_u16(vget_low_u16(a)));
    vcvtaq_u32_f32(vmulq_f32(cvt, b))
}

#[inline(always)]
#[allow(dead_code)]
pub(crate) unsafe fn vmulq_u16_low_s32(a: uint16x8_t, b: int32x4_t) -> int32x4_t {
    let cvt = vmovl_u16(vget_low_u16(a));
    vqrdmulhq_s32(vreinterpretq_s32_u32(cvt), b)
}

#[inline(always)]
pub(crate) unsafe fn vmulq_s32_f32(a: int32x4_t, b: float32x4_t) -> int32x4_t {
    let cvt = vcvtq_f32_s32(a);
    vcvtaq_s32_f32(vmulq_f32(cvt, b))
}

#[inline(always)]
pub unsafe fn load_u8_u16<const CHANNELS_COUNT: usize>(ptr: *const u8) -> uint16x4_t {
    if CHANNELS_COUNT == 4 {
        vget_low_u16(vmovl_u8(vreinterpret_u8_u32(vld1_lane_u32::<0>(
            ptr as *mut u32,
            vdup_n_u32(0),
        ))))
    } else if CHANNELS_COUNT == 3 {
        let a0 = vreinterpret_u8_u16(vld1_lane_u16::<0>(ptr as *const _, vdup_n_u16(0)));
        let a1 = vld1_lane_u8::<2>(ptr.add(2) as *const _, a0);
        vget_low_u16(vmovl_u8(a1))
    } else if CHANNELS_COUNT == 2 {
        let a0 = vreinterpret_u8_u16(vld1_lane_u16::<0>(ptr as *const _, vdup_n_u16(0)));
        vget_low_u16(vmovl_u8(a0))
    } else {
        let a0 = vld1_lane_u8::<0>(ptr as *const _, vdup_n_u8(0));
        vget_low_u16(vmovl_u8(a0))
    }
}

#[inline(always)]
pub(crate) unsafe fn vmulq_by_3_s32(k: int32x4_t) -> int32x4_t {
    vaddq_s32(vshlq_n_s32::<1>(k), k)
}

#[inline(always)]
pub unsafe fn load_u8<const CHANNELS_COUNT: usize>(ptr: *const u8) -> uint8x8_t {
    if CHANNELS_COUNT == 4 {
        vreinterpret_u8_u32(vld1_lane_u32::<0>(ptr as *mut u32, vdup_n_u32(0)))
    } else if CHANNELS_COUNT == 3 {
        let a0 = vreinterpret_u8_u16(vld1_lane_u16::<0>(ptr as *const _, vdup_n_u16(0)));
        vld1_lane_u8::<2>(ptr.add(2) as *const _, a0)
    } else if CHANNELS_COUNT == 2 {
        vreinterpret_u8_u16(vld1_lane_u16::<0>(ptr as *const _, vdup_n_u16(0)))
    } else {
        vld1_lane_u8::<0>(ptr as *const _, vdup_n_u8(0))
    }
}

#[inline(always)]
pub unsafe fn p_vfmaq_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    #[cfg(target_arch = "aarch64")]
    {
        vfmaq_f32(a, b, c)
    }
    #[cfg(target_arch = "arm")]
    {
        vmlaq_f32(a, b, c)
    }
}

#[inline(always)]
pub unsafe fn prefer_vfma_f32(a: float32x2_t, b: float32x2_t, c: float32x2_t) -> float32x2_t {
    #[cfg(target_arch = "aarch64")]
    {
        vfma_f32(a, b, c)
    }
    #[cfg(target_arch = "arm")]
    {
        vmla_f32(a, b, c)
    }
}

#[inline(always)]
pub unsafe fn load_f32_f16<const CHANNELS_COUNT: usize>(ptr: *const f16) -> float32x4_t {
    if CHANNELS_COUNT == 4 {
        let cvt = xvld_f16(ptr);
        return xvcvt_f32_f16(cvt);
    } else if CHANNELS_COUNT == 3 {
        let mut jvet = vreinterpret_u16_u32(vld1_lane_u32::<0>(ptr as *const u32, vdup_n_u32(0)));
        jvet = vld1_lane_u16::<2>(ptr.add(2) as *const u16, jvet);
        let cvt = xreinterpret_f16_u16(jvet);
        return xvcvt_f32_f16(cvt);
    } else if CHANNELS_COUNT == 2 {
        let jvet = vld1_lane_u32::<0>(ptr as *const u32, vdup_n_u32(0));
        let cvt = xreinterpret_f16_u16(vreinterpret_u16_u32(jvet));
        return xvcvt_f32_f16(cvt);
    }
    let recast = ptr as *const u16;
    let cvt = xreinterpret_f16_u16(vld1_u16([recast.read_unaligned(), 0, 0, 0].as_ptr()));
    xvcvt_f32_f16(cvt)
}

#[inline(always)]
pub unsafe fn store_f32_f16<const CHANNELS_COUNT: usize>(dst_ptr: *mut f16, in_regi: float32x4_t) {
    let out_regi = xvcvt_f16_f32(in_regi);
    if CHANNELS_COUNT == 4 {
        xvst_f16(dst_ptr, out_regi);
    } else if CHANNELS_COUNT == 3 {
        let casted_out = xreinterpret_u16_f16(out_regi);
        let casted_ptr = dst_ptr as *mut u16;
        let lo_part = vreinterpret_u32_u16(casted_out);
        vst1_lane_u32::<0>(casted_ptr as *mut _, lo_part);
        vst1_lane_u16::<2>(casted_ptr.add(2) as *mut _, vreinterpret_u16_u32(lo_part));
    } else if CHANNELS_COUNT == 2 {
        let casted_out = xreinterpret_u16_f16(out_regi);
        let casted_ptr = dst_ptr as *mut u32;
        let lo_part = vreinterpret_u32_u16(casted_out);
        vst1_lane_u32::<0>(casted_ptr, lo_part);
    } else {
        let casted_out = xreinterpret_u16_f16(out_regi);
        vst1_lane_u16::<0>(dst_ptr as *mut u16, casted_out);
    }
}

/// Stores up to 4 values from uint8x8_t
#[inline(always)]
pub(crate) unsafe fn store_u8x8_m4<const CHANNELS_COUNT: usize>(
    dst_ptr: *mut u8,
    in_regi: uint8x8_t,
) {
    let casted_u32 = vreinterpret_u32_u8(in_regi);

    if CHANNELS_COUNT == 4 {
        vst1_lane_u32::<0>(dst_ptr as *mut _, casted_u32);
    } else if CHANNELS_COUNT == 3 {
        vst1_lane_u16::<0>(dst_ptr as *mut _, vreinterpret_u16_u32(casted_u32));
        vst1_lane_u8::<2>(dst_ptr.add(2) as *mut _, vreinterpret_u8_u32(casted_u32));
    } else if CHANNELS_COUNT == 2 {
        vst1_lane_u16::<0>(dst_ptr as *mut _, vreinterpret_u16_u32(casted_u32));
    } else {
        vst1_lane_u8::<0>(dst_ptr, vreinterpret_u8_u32(casted_u32));
    }
}

/// Stores up to 4 values from uint16x4_t
#[inline(always)]
pub(crate) unsafe fn store_u16x4<const CN: usize>(dst_ptr: *mut u16, a0: uint16x4_t) {
    if CN == 4 {
        vst1_u16(dst_ptr, a0);
    } else if CN == 3 {
        vst1_lane_u32::<0>(dst_ptr as *mut _, vreinterpret_u32_u16(a0));
        vst1_lane_u16::<2>(dst_ptr.add(2) as *mut _, a0);
    } else if CN == 2 {
        vst1_lane_u32::<0>(dst_ptr as *mut _, vreinterpret_u32_u16(a0));
    } else {
        vst1_lane_u16::<0>(dst_ptr as *mut _, a0);
    }
}

#[inline(always)]
pub unsafe fn store_u16_s32_x4<const CN: usize>(
    dst_ptr: (*mut u16, *mut u16, *mut u16, *mut u16),
    regi: int32x4x4_t,
) {
    let a0 = vqmovun_s32(regi.0);
    let a1 = vqmovun_s32(regi.1);
    let a2 = vqmovun_s32(regi.2);
    let a3 = vqmovun_s32(regi.3);
    if CN == 4 {
        vst1_u16(dst_ptr.0, a0);
        vst1_u16(dst_ptr.1, a1);
        vst1_u16(dst_ptr.2, a2);
        vst1_u16(dst_ptr.3, a3);
    } else if CN == 3 {
        vst1_lane_u32::<0>(dst_ptr.0 as *mut _, vreinterpret_u32_u16(a0));
        vst1_lane_u16::<2>(dst_ptr.0.add(2) as *mut _, a0);
        vst1_lane_u32::<0>(dst_ptr.1 as *mut _, vreinterpret_u32_u16(a1));
        vst1_lane_u16::<2>(dst_ptr.1.add(2) as *mut _, a1);
        vst1_lane_u32::<0>(dst_ptr.2 as *mut _, vreinterpret_u32_u16(a2));
        vst1_lane_u16::<2>(dst_ptr.2.add(2) as *mut _, a2);
        vst1_lane_u32::<0>(dst_ptr.3 as *mut _, vreinterpret_u32_u16(a3));
        vst1_lane_u16::<2>(dst_ptr.3.add(2) as *mut _, a3);
    } else if CN == 2 {
        vst1_lane_u32::<0>(dst_ptr.0 as *mut _, vreinterpret_u32_u16(a0));
        vst1_lane_u32::<0>(dst_ptr.1 as *mut _, vreinterpret_u32_u16(a1));
        vst1_lane_u32::<0>(dst_ptr.2 as *mut _, vreinterpret_u32_u16(a2));
        vst1_lane_u32::<0>(dst_ptr.3 as *mut _, vreinterpret_u32_u16(a3));
    } else {
        vst1_lane_u16::<0>(dst_ptr.0 as *mut _, a0);
        vst1_lane_u16::<0>(dst_ptr.1 as *mut _, a1);
        vst1_lane_u16::<0>(dst_ptr.2 as *mut _, a2);
        vst1_lane_u16::<0>(dst_ptr.3 as *mut _, a3);
    }
}

#[inline(always)]
pub unsafe fn store_u16_s32_x5<const CN: usize>(
    dst_ptr: (*mut u16, *mut u16, *mut u16, *mut u16, *mut u16),
    regi: int32x4x4_t,
    add: int32x4_t,
) {
    let a0 = vqmovun_s32(regi.0);
    let a1 = vqmovun_s32(regi.1);
    let a2 = vqmovun_s32(regi.2);
    let a3 = vqmovun_s32(regi.3);
    let a4 = vqmovun_s32(add);
    if CN == 4 {
        vst1_u16(dst_ptr.0, a0);
        vst1_u16(dst_ptr.1, a1);
        vst1_u16(dst_ptr.2, a2);
        vst1_u16(dst_ptr.3, a3);
        vst1_u16(dst_ptr.4, a4);
    } else if CN == 3 {
        vst1_lane_u32::<0>(dst_ptr.0 as *mut _, vreinterpret_u32_u16(a0));
        vst1_lane_u16::<2>(dst_ptr.0.add(2) as *mut _, a0);
        vst1_lane_u32::<0>(dst_ptr.1 as *mut _, vreinterpret_u32_u16(a1));
        vst1_lane_u16::<2>(dst_ptr.1.add(2) as *mut _, a1);
        vst1_lane_u32::<0>(dst_ptr.2 as *mut _, vreinterpret_u32_u16(a2));
        vst1_lane_u16::<2>(dst_ptr.2.add(2) as *mut _, a2);
        vst1_lane_u32::<0>(dst_ptr.3 as *mut _, vreinterpret_u32_u16(a3));
        vst1_lane_u16::<2>(dst_ptr.3.add(2) as *mut _, a3);
        vst1_lane_u32::<0>(dst_ptr.4 as *mut _, vreinterpret_u32_u16(a4));
        vst1_lane_u16::<2>(dst_ptr.4.add(2) as *mut _, a4);
    } else if CN == 2 {
        vst1_lane_u32::<0>(dst_ptr.0 as *mut _, vreinterpret_u32_u16(a0));
        vst1_lane_u32::<0>(dst_ptr.1 as *mut _, vreinterpret_u32_u16(a1));
        vst1_lane_u32::<0>(dst_ptr.2 as *mut _, vreinterpret_u32_u16(a2));
        vst1_lane_u32::<0>(dst_ptr.3 as *mut _, vreinterpret_u32_u16(a3));
        vst1_lane_u32::<0>(dst_ptr.4 as *mut _, vreinterpret_u32_u16(a4));
    } else {
        vst1_lane_u16::<0>(dst_ptr.0 as *mut _, a0);
        vst1_lane_u16::<0>(dst_ptr.1 as *mut _, a1);
        vst1_lane_u16::<0>(dst_ptr.2 as *mut _, a2);
        vst1_lane_u16::<0>(dst_ptr.3 as *mut _, a3);
        vst1_lane_u16::<0>(dst_ptr.4 as *mut _, a4);
    }
}
