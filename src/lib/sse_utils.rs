#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[cfg(target_feature = "sse4.1")]
pub(crate) mod sse_utils {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[inline]
    pub(crate) unsafe fn load_u8_s32_fast<const CHANNELS_COUNT: usize>(ptr: *const u8) -> __m128i {
        let u_first = u32::from_le_bytes([ptr.read_unaligned(), 0, 0, 0]);
        let u_second = u32::from_le_bytes([ptr.add(1).read_unaligned(), 0, 0, 0]);
        let u_third = u32::from_le_bytes([ptr.add(2).read_unaligned(), 0, 0, 0]);
        let u_fourth = match CHANNELS_COUNT {
            4 => u32::from_le_bytes([ptr.add(3).read_unaligned(), 0, 0, 0]),
            _ => 0,
        };
        let store: [u32; 4] = [u_first, u_second, u_third, u_fourth];
        return _mm_loadu_si128(store.as_ptr() as *const __m128i);
    }

    #[inline(always)]
    pub(crate) unsafe fn _mm_mul_epi64(ab: __m128i, cd: __m128i) -> __m128i {
        /* ac = (ab & 0xFFFFFFFF) * (cd & 0xFFFFFFFF); */
        let ac = _mm_mul_epu32(ab, cd);

        /* b = ab >> 32; */
        let b = _mm_srli_epi64::<32>(ab);

        /* bc = b * (cd & 0xFFFFFFFF); */
        let bc = _mm_mul_epu32(b, cd);

        /* d = cd >> 32; */
        let d = _mm_srli_epi64::<32>(cd);

        /* ad = (ab & 0xFFFFFFFF) * d; */
        let ad = _mm_mul_epu32(ab, d);

        /* high = bc + ad; */
        let mut high = _mm_add_epi64(bc, ad);

        /* high <<= 32; */
        high = _mm_slli_epi64::<32>(high);

        /* return ac + high; */
        return _mm_add_epi64(high, ac);
    }

    #[inline]
    pub(crate) unsafe fn load_u8_f32_fast<const CHANNELS_COUNT: usize>(ptr: *const u8) -> __m128 {
        let vl = load_u8_s32_fast::<CHANNELS_COUNT>(ptr);
        return _mm_cvtepi32_ps(vl);
    }

    #[inline(always)]
    pub(crate) unsafe fn load_u8_u32_one(ptr: *const u8) -> __m128i {
        let u_first = u32::from_le_bytes([ptr.read_unaligned(), 0, 0, 0]);
        return _mm_set1_epi32(u_first as i32);
    }

    #[inline(always)]
    pub(crate) unsafe fn store_u8_s32<const CHANNELS_COUNT: usize>(
        dst_ptr: *mut u8,
        regi: __m128i,
    ) {
        let s16 = _mm_packs_epi32(regi, regi);
        let v8 = _mm_packus_epi16(s16, s16);
        let pixel_s32 = _mm_extract_epi32::<0>(v8);
        if CHANNELS_COUNT == 4 {
            let casted_dst = dst_ptr as *mut i32;
            casted_dst.write_unaligned(pixel_s32);
        } else {
            let pixel_bytes = pixel_s32.to_le_bytes();
            dst_ptr.write_unaligned(pixel_bytes[0]);
            dst_ptr.add(1).write_unaligned(pixel_bytes[1]);
            dst_ptr.add(2).write_unaligned(pixel_bytes[2]);
        }
    }

    #[cfg(not(target_feature = "fma"))]
    #[inline]
    pub unsafe fn _mm_prefer_fma_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
        return _mm_add_ps(_mm_mul_ps(b, c), a);
    }

    #[cfg(target_feature = "fma")]
    #[inline]
    pub unsafe fn _mm_prefer_fma_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
        return _mm_fmadd_ps(b, c, a);
    }
}
