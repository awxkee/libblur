#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
#[cfg(target_feature = "neon")]
pub(crate) mod neon_utils {
    use std::arch::aarch64::*;
    use std::ptr;

    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) unsafe fn load_u8_s32_fast<const CHANNELS_COUNT: usize>(
        ptr: *const u8,
    ) -> int32x4_t {
        let u_first = u32::from_le_bytes([*ptr, 0, 0, 0]);
        let u_second = u32::from_le_bytes([*ptr.add(1), 0, 0, 0]);
        let u_third = u32::from_le_bytes([*ptr.add(2), 0, 0, 0]);
        let u_fourth = match CHANNELS_COUNT {
            4 => u32::from_le_bytes([*ptr.add(3), 0, 0, 0]),
            _ => 0,
        };
        let store: [u32; 4] = [u_first, u_second, u_third, u_fourth];
        return vreinterpretq_s32_u32(vld1q_u32(store.as_ptr()));
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn load_u8_s16(ptr: *const u8, use_vld: bool, channels_count: usize) -> int16x4_t {
        let mut safe_transient_store: [u8; 8] = [0; 8];
        let edge_ptr: *const u8;
        if use_vld {
            edge_ptr = ptr;
        } else {
            unsafe {
                ptr::copy_nonoverlapping(ptr, safe_transient_store.as_mut_ptr(), channels_count);
            }
            edge_ptr = safe_transient_store.as_ptr();
        }
        let pixel_color =
            unsafe { vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vld1_u8(edge_ptr)))) };
        return pixel_color;
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn load_u8_u16(ptr: *const u8, use_vld: bool, channels_count: usize) -> uint16x4_t {
        let mut safe_transient_store: [u8; 8] = [0; 8];
        let edge_ptr: *const u8;
        if use_vld {
            edge_ptr = ptr;
        } else {
            unsafe {
                ptr::copy_nonoverlapping(ptr, safe_transient_store.as_mut_ptr(), channels_count);
            }
            edge_ptr = safe_transient_store.as_ptr();
        }
        let pixel_color = unsafe { vget_low_u16(vmovl_u8(vld1_u8(edge_ptr))) };
        return pixel_color;
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn load_u8_u32(ptr: *const u8, use_vld: bool, channels_count: usize) -> uint32x4_t {
        let mut safe_transient_store: [u8; 8] = [0; 8];
        let edge_ptr: *const u8;
        if use_vld {
            edge_ptr = ptr;
        } else {
            unsafe {
                ptr::copy_nonoverlapping(ptr, safe_transient_store.as_mut_ptr(), channels_count);
            }
            edge_ptr = safe_transient_store.as_ptr();
        }
        let pixel_color = unsafe { vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(edge_ptr)))) };
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
}
