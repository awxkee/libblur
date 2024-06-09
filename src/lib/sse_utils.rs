#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[cfg(target_feature = "sse4.1")]
pub(crate) mod neon_utils {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;

    #[inline]
    pub(crate) unsafe fn load_u8_s32_fast<const CHANNELS_COUNT: usize>(
        ptr: *const u8,
    ) -> __m128i {
        let u_first = u32::from_le_bytes([*ptr, 0, 0, 0]);
        let u_second = u32::from_le_bytes([*ptr.add(1), 0, 0, 0]);
        let u_third = u32::from_le_bytes([*ptr.add(2), 0, 0, 0]);
        let u_fourth = match CHANNELS_COUNT {
            4 => u32::from_le_bytes([*ptr.add(3), 0, 0, 0]),
            _ => 0,
        };
        let store: [u32; 4] = [u_first, u_second, u_third, u_fourth];
        return _mm_loadu_si128(store.as_ptr() as *const __m128i);
    }
}
