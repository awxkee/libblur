#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[cfg(target_feature = "sse4.1")]
pub(crate) mod sse_utils {
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

    #[inline]
    pub(crate) unsafe fn load_u8_f32_fast<const CHANNELS_COUNT: usize>(
        ptr: *const u8,
    ) -> __m128 {
        let vl = load_u8_s32_fast::<CHANNELS_COUNT>(ptr);
        return _mm_cvtepi32_ps(vl);
    }

    #[inline(always)]
    pub(crate) unsafe fn load_u8_u32_one(
        ptr: *const u8,
    ) -> __m128i {
        let u_first = u32::from_le_bytes([*ptr, 0, 0, 0]);
        return _mm_set1_epi32(u_first as i32);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[cfg(not(target_feature = "fma"))]
    #[inline]
    pub unsafe fn _mm_prefer_fma_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
        return _mm_add_ps(_mm_mul_ps(b, c), a);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[cfg(target_feature = "fma")]
    #[inline]
    pub unsafe fn _mm_prefer_fma_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
        return _mm_fmadd_ps(b, c, a);
    }
}
