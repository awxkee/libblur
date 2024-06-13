#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(crate) mod neon_utils {
    use std::arch::aarch64::*;
    use std::ptr;

    #[inline(always)]
    pub(crate) unsafe fn load_u8_s32_fast<const CHANNELS_COUNT: usize>(
        ptr: *const u8,
    ) -> int32x4_t {
        return vreinterpretq_s32_u32(load_u8_u32_fast::<CHANNELS_COUNT>(ptr));
    }

    #[inline(always)]
    pub(crate) unsafe fn load_u8_u32_one(ptr: *const u8) -> uint32x2_t {
        let u_first = u32::from_le_bytes([ptr.read_unaligned(), 0, 0, 0]);
        return vdup_n_u32(u_first);
    }

    #[inline(always)]
    pub(crate) unsafe fn store_u8_s32<const CHANNELS_COUNT: usize>(
        dst_ptr: *mut u8,
        regi: int32x4_t,
    ) {
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
    pub(crate) unsafe fn load_u8_u32_fast<const CHANNELS_COUNT: usize>(
        ptr: *const u8,
    ) -> uint32x4_t {
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

    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn load_f32(ptr: *const f32, use_vld: bool, channels_count: usize) -> float32x4_t {
        let mut safe_transient_store: [f32; 4] = [0f32; 4];
        let edge_ptr: *const f32;
        if use_vld {
            edge_ptr = ptr;
        } else {
            unsafe {
                ptr::copy_nonoverlapping(ptr, safe_transient_store.as_mut_ptr(), channels_count);
            }
            edge_ptr = safe_transient_store.as_ptr();
        }
        let pixel_color = unsafe { vld1q_f32(edge_ptr) };
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
}
