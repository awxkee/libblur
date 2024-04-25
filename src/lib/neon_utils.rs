#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
#[cfg(target_feature = "neon")]
pub(crate) mod neon_utils {
    use std::arch::aarch64::{int16x4_t, int32x4_t, uint16x4_t, uint32x4_t, vget_low_u16, vld1_u8, vmovl_u16, vmovl_u8, vreinterpret_s16_u16, vreinterpretq_s32_u32};
    use std::ptr;

    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn load_u8_s32(
        ptr: *const u8,
        use_vld: bool,
        channels_count: usize,
    ) -> int32x4_t {
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
            unsafe { vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(edge_ptr))))) };
        return pixel_color;
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn load_u8_s16(
        ptr: *const u8,
        use_vld: bool,
        channels_count: usize,
    ) -> int16x4_t {
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
    pub(crate) fn load_u8_u16(
        ptr: *const u8,
        use_vld: bool,
        channels_count: usize,
    ) -> uint16x4_t {
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
            unsafe { vget_low_u16(vmovl_u8(vld1_u8(edge_ptr))) };
        return pixel_color;
    }


    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn load_u8_u32(
        ptr: *const u8,
        use_vld: bool,
        channels_count: usize,
    ) -> uint32x4_t {
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
            unsafe { vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(edge_ptr)))) };
        return pixel_color;
    }
}
