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

use std::arch::aarch64::*;

#[inline(always)]
pub(crate) unsafe fn load_u8_s32_fast<const CHANNELS_COUNT: usize>(ptr: *const u8) -> int32x4_t {
    return vreinterpretq_s32_u32(load_u8_u32_fast::<CHANNELS_COUNT>(ptr));
}

#[inline(always)]
pub(crate) unsafe fn load_u8_u32_one(ptr: *const u8) -> uint32x2_t {
    let u_first = u32::from_le_bytes([ptr.read_unaligned(), 0, 0, 0]);
    return vdup_n_u32(u_first);
}

#[inline(always)]
pub(crate) unsafe fn store_f32<const CHANNELS_COUNT: usize>(dst_ptr: *mut f32, regi: float32x4_t) {
    if CHANNELS_COUNT == 4 {
        vst1q_f32(dst_ptr, regi);
    } else if CHANNELS_COUNT == 3 {
        dst_ptr.write_unaligned(vgetq_lane_f32::<0>(regi));
        dst_ptr.add(1).write_unaligned(vgetq_lane_f32::<1>(regi));
        dst_ptr.add(2).write_unaligned(vgetq_lane_f32::<2>(regi));
    } else if CHANNELS_COUNT == 2 {
        dst_ptr.write_unaligned(vgetq_lane_f32::<0>(regi));
        dst_ptr.add(1).write_unaligned(vgetq_lane_f32::<1>(regi));
    } else {
        dst_ptr.write_unaligned(vgetq_lane_f32::<0>(regi));
    }
}

#[inline(always)]
pub(crate) unsafe fn store_u8_s32<const CHANNELS_COUNT: usize>(dst_ptr: *mut u8, regi: int32x4_t) {
    let s16 = vreinterpret_u16_s16(vqmovn_s32(regi));
    let u16_f = vcombine_u16(s16, s16);
    let v8 = vqmovn_u16(u16_f);
    let pixel_u32 = vget_lane_u32::<0>(vreinterpret_u32_u8(v8));
    if CHANNELS_COUNT == 4 {
        let casted_dst = dst_ptr as *mut u32;
        casted_dst.write_unaligned(pixel_u32);
    } else {
        let pixel_bytes = pixel_u32.to_le_bytes();
        dst_ptr.write_unaligned(pixel_bytes[0]);
        dst_ptr.add(1).write_unaligned(pixel_bytes[1]);
        dst_ptr.add(2).write_unaligned(pixel_bytes[2]);
    }
}

#[inline(always)]
pub(crate) unsafe fn load_f32_fast<const CHANNELS_COUNT: usize>(ptr: *const f32) -> float32x4_t {
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
    return vld1q_f32([ptr.read_unaligned(), 0f32, 0f32, 0f32].as_ptr());
}

#[inline(always)]
pub(crate) unsafe fn load_u8_u32_fast<const CHANNELS_COUNT: usize>(ptr: *const u8) -> uint32x4_t {
    let u_first = u32::from_le_bytes([ptr.read_unaligned(), 0, 0, 0]);
    let u_second = u32::from_le_bytes([ptr.add(1).read_unaligned(), 0, 0, 0]);
    let u_third = u32::from_le_bytes([ptr.add(2).read_unaligned(), 0, 0, 0]);
    let u_fourth = match CHANNELS_COUNT {
        4 => u32::from_le_bytes([ptr.add(3).read_unaligned(), 0, 0, 0]),
        _ => 0,
    };
    let store: [u32; 4] = [u_first, u_second, u_third, u_fourth];
    return vld1q_u32(store.as_ptr());
}

#[inline(always)]
pub(crate) unsafe fn load_u8_u64_fast<const CHANNELS_COUNT: usize>(ptr: *const u8) -> uint64x2x2_t {
    let u_first = u64::from_le_bytes([ptr.read_unaligned(), 0, 0, 0, 0, 0, 0, 0]);
    let u_second = u64::from_le_bytes([ptr.add(1).read_unaligned(), 0, 0, 0, 0, 0, 0, 0]);
    let u_third = u64::from_le_bytes([ptr.add(2).read_unaligned(), 0, 0, 0, 0, 0, 0, 0]);
    let u_fourth = match CHANNELS_COUNT {
        4 => u64::from_le_bytes([ptr.add(3).read_unaligned(), 0, 0, 0, 0, 0, 0, 0]),
        _ => 0,
    };
    let store: [u64; 4] = [u_first, u_second, u_third, u_fourth];
    return vld1q_u64_x2(store.as_ptr());
}

#[inline(always)]
pub unsafe fn vaddq_s64x2(ab: int64x2x2_t, cd: int64x2x2_t) -> int64x2x2_t {
    let ux_0 = vaddq_s64(ab.0, cd.0);
    let ux_1 = vaddq_s64(ab.1, cd.1);
    int64x2x2_t(ux_0, ux_1)
}

#[inline(always)]
pub unsafe fn vsubq_s64x2(ab: int64x2x2_t, cd: int64x2x2_t) -> int64x2x2_t {
    let ux_0 = vsubq_s64(ab.0, cd.0);
    let ux_1 = vsubq_s64(ab.1, cd.1);
    int64x2x2_t(ux_0, ux_1)
}

#[inline(always)]
pub unsafe fn vmulq_s64(ab: int64x2_t, cd: int64x2_t) -> int64x2_t {
    /* ac = (ab & 0xFFFFFFFF) * (cd & 0xFFFFFFFF); */
    let ac = vmulq_u32(
        vreinterpretq_u32_u64(vreinterpretq_u64_s64(ab)),
        vreinterpretq_u32_u64(vreinterpretq_u64_s64(cd)),
    );

    /* b = ab >> 32; */
    let b = vshrq_n_u64::<32>(vreinterpretq_u64_s64(ab));

    /* bc = b * (cd & 0xFFFFFFFF); */
    let bc = vmulq_u32(
        vreinterpretq_u32_u64(b),
        vreinterpretq_u32_u64(vreinterpretq_u64_s64(cd)),
    );

    /* d = cd >> 32; */
    let d = vshrq_n_u64::<32>(vreinterpretq_u64_s64(cd));

    /* ad = (ab & 0xFFFFFFFF) * d; */
    let ad = vmulq_u32(
        vreinterpretq_u32_u64(vreinterpretq_u64_s64(ab)),
        vreinterpretq_u32_u64(d),
    );

    /* high = bc + ad; */
    let mut high = vaddq_s64(vreinterpretq_s64_u32(bc), vreinterpretq_s64_u32(ad));

    /* high <<= 32; */
    high = vshlq_n_s64::<32>(high);

    /* return ac + high; */
    return vaddq_s64(high, vreinterpretq_s64_u32(ac));
}

#[inline(always)]
pub(crate) unsafe fn vdupq_n_s64x2(v: i64) -> int64x2x2_t {
    let vl = vdupq_n_s64(v);
    int64x2x2_t(vl, vl)
}

#[inline(always)]
pub(crate) unsafe fn vmulq_u32_f32(a: uint32x4_t, b: float32x4_t) -> uint32x4_t {
    let cvt = vcvtq_f32_u32(a);
    vcvtaq_u32_f32(vmulq_f32(cvt, b))
}

#[inline(always)]
pub(crate) unsafe fn vmulq_s32_f32(a: int32x4_t, b: float32x4_t) -> int32x4_t {
    let cvt = vcvtq_f32_s32(a);
    vcvtaq_s32_f32(vmulq_f32(cvt, b))
}

#[inline(always)]
pub(crate) unsafe fn vmulq_n_s64x2(x: int64x2x2_t, v: i64) -> int64x2x2_t {
    let vl = vdupq_n_s64(v);
    let lo = vmulq_s64(x.0, vl);
    let hi = vmulq_s64(x.1, vl);
    int64x2x2_t(lo, hi)
}

#[inline(always)]
pub(crate) unsafe fn load_u8_s64x2_fast<const CHANNELS_COUNT: usize>(
    ptr: *const u8,
) -> int64x2x2_t {
    let ux = load_u8_u64_fast::<CHANNELS_COUNT>(ptr);
    let sx_0 = vreinterpretq_s64_u64(ux.0);
    let sx_1 = vreinterpretq_s64_u64(ux.1);
    int64x2x2_t(sx_0, sx_1)
}

#[inline(always)]
pub(crate) unsafe fn load_u8_u16_x2_fast<const CHANNELS_COUNT: usize>(
    ptr: *const u8,
) -> uint16x8_t {
    return if CHANNELS_COUNT == 3 {
        let first_integer_part = (ptr as *const u32).read_unaligned().to_le_bytes();
        let u_first = u16::from_le_bytes([first_integer_part[0], 0]);
        let u_second = u16::from_le_bytes([first_integer_part[1], 0]);
        let u_third = u16::from_le_bytes([first_integer_part[2], 0]);
        let u_fourth = u16::from_le_bytes([first_integer_part[3], 0]);
        let u_fifth = u16::from_le_bytes([ptr.add(4).read_unaligned(), 0]);
        let u_sixth = u16::from_le_bytes([ptr.add(5).read_unaligned(), 0]);
        let store: [u16; 8] = [u_first, u_second, u_third, u_fourth, u_fifth, u_sixth, 0, 0];
        vld1q_u16(store.as_ptr())
    } else {
        vmovl_u8(vld1_u8(ptr))
    };
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn load_u8_s16<const CHANNELS_COUNT: usize>(ptr: *const u8) -> int16x4_t {
    let pixel_color = unsafe { vreinterpret_s16_u16(load_u8_u16::<CHANNELS_COUNT>(ptr)) };
    return pixel_color;
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) unsafe fn load_u8_u16<const CHANNELS_COUNT: usize>(ptr: *const u8) -> uint16x4_t {
    let u_first = u16::from_le_bytes([ptr.read(), 0]);
    let u_second = u16::from_le_bytes([ptr.add(1).read_unaligned(), 0]);
    let u_third = u16::from_le_bytes([ptr.add(2).read_unaligned(), 0]);
    let u_fourth = match CHANNELS_COUNT {
        4 => u16::from_le_bytes([ptr.add(3).read_unaligned(), 0]),
        _ => 0,
    };
    let store: [u16; 4] = [u_first, u_second, u_third, u_fourth];
    let pixel_color = unsafe { vld1_u16(store.as_ptr()) };
    return pixel_color;
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
pub(crate) unsafe fn prefer_vfmaq_f32(
    a: float32x4_t,
    b: float32x4_t,
    c: float32x4_t,
) -> float32x4_t {
    #[cfg(target_arch = "aarch64")]
    {
        return vfmaq_f32(a, b, c);
    }
    #[cfg(target_arch = "arm")]
    {
        return vmlaq_f32(a, b, c);
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
pub(crate) unsafe fn prefer_vfma_f32(
    a: float32x2_t,
    b: float32x2_t,
    c: float32x2_t,
) -> float32x2_t {
    #[cfg(target_arch = "aarch64")]
    {
        return vfma_f32(a, b, c);
    }
    #[cfg(target_arch = "arm")]
    {
        return vmla_f32(a, b, c);
    }
}
