use std::arch::aarch64::*;
use crate::neon::{load_u8_u32_fast, load_u8_u32_one, prefer_vfma_f32, prefer_vfmaq_f32};
use crate::unsafe_slice::UnsafeSlice;

pub fn gaussian_blur_vertical_pass_neon<T, const CHANNEL_CONFIGURATION: usize>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    height: u32,
    kernel_size: usize,
    kernel: &[f32],
    start_y: u32,
    end_y: u32,
) {
    let src: &[u8] = unsafe { std::mem::transmute(undef_src) };
    let unsafe_dst: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(undef_unsafe_dst) };
    let half_kernel = (kernel_size / 2) as i32;

    let zeros = unsafe { vdupq_n_f32(0f32) };

    let total_size = CHANNEL_CONFIGURATION * width as usize;

    for y in start_y..end_y {
        let y_dst_shift = y as usize * dst_stride as usize;

        let mut cx = 0usize;

        unsafe {
            while cx + 32 < total_size {
                let mut store0 = zeros;
                let mut store1 = zeros;
                let mut store2 = zeros;
                let mut store3 = zeros;
                let mut store4 = zeros;
                let mut store5 = zeros;
                let mut store6 = zeros;
                let mut store7 = zeros;

                let mut r = -half_kernel;
                while r <= half_kernel {
                    let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let f_weight: float32x4_t = vdupq_n_f32(weight);

                    let py =
                        std::cmp::min(std::cmp::max(y as i64 + r as i64, 0), (height - 1) as i64);
                    let y_src_shift = py as usize * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let pixels_u8x2 = vld1q_u8_x2(s_ptr);
                    let hi_16 = vmovl_high_u8(pixels_u8x2.0);
                    let lo_16 = vmovl_u8(vget_low_u8(pixels_u8x2.0));
                    let lo_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_16)));
                    store0 = prefer_vfmaq_f32(store0, lo_lo, f_weight);
                    let lo_hi = vcvtq_f32_u32(vmovl_high_u16(lo_16));
                    store1 = prefer_vfmaq_f32(store1, lo_hi, f_weight);
                    let hi_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_16)));
                    store2 = prefer_vfmaq_f32(store2, hi_lo, f_weight);
                    let hi_hi = vcvtq_f32_u32(vmovl_high_u16(hi_16));
                    store3 = prefer_vfmaq_f32(store3, hi_hi, f_weight);

                    let hi_16 = vmovl_high_u8(pixels_u8x2.1);
                    let lo_16 = vmovl_u8(vget_low_u8(pixels_u8x2.1));
                    let lo_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_16)));
                    store4 = prefer_vfmaq_f32(store4, lo_lo, f_weight);
                    let lo_hi = vcvtq_f32_u32(vmovl_high_u16(lo_16));
                    store5 = prefer_vfmaq_f32(store5, lo_hi, f_weight);
                    let hi_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_16)));
                    store6 = prefer_vfmaq_f32(store6, hi_lo, f_weight);
                    let hi_hi = vcvtq_f32_u32(vmovl_high_u16(hi_16));
                    store7 = prefer_vfmaq_f32(store7, hi_hi, f_weight);

                    r += 1;
                }

                let store_0 = vcvtaq_u32_f32(store0);
                let store_1 = vcvtaq_u32_f32(store1);
                let store_2 = vcvtaq_u32_f32(store2);
                let store_3 = vcvtaq_u32_f32(store3);
                let store_4 = vcvtaq_u32_f32(store4);
                let store_5 = vcvtaq_u32_f32(store5);
                let store_6 = vcvtaq_u32_f32(store6);
                let store_7 = vcvtaq_u32_f32(store7);

                let store_lo = vcombine_u16(vmovn_u32(store_0), vmovn_u32(store_1));
                let store_hi = vcombine_u16(vmovn_u32(store_2), vmovn_u32(store_3));
                let store_x = vcombine_u8(vqmovn_u16(store_lo), vqmovn_u16(store_hi));

                let store_lo = vcombine_u16(vmovn_u32(store_4), vmovn_u32(store_5));
                let store_hi = vcombine_u16(vmovn_u32(store_6), vmovn_u32(store_7));
                let store_k = vcombine_u8(vqmovn_u16(store_lo), vqmovn_u16(store_hi));

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u8;

                let store = uint8x16x2_t(store_x, store_k);
                vst1q_u8_x2(dst_ptr, store);

                cx += 32;
            }

            while cx + 16 < total_size {
                let mut store0: float32x4_t = zeros;
                let mut store1: float32x4_t = zeros;
                let mut store2: float32x4_t = zeros;
                let mut store3: float32x4_t = zeros;

                let mut r = -half_kernel;
                while r <= half_kernel {
                    let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let f_weight: float32x4_t = vdupq_n_f32(weight);

                    let py =
                        std::cmp::min(std::cmp::max(y as i64 + r as i64, 0), (height - 1) as i64);
                    let y_src_shift = py as usize * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let pixels_u8 = vld1q_u8(s_ptr);
                    let hi_16 = vmovl_high_u8(pixels_u8);
                    let lo_16 = vmovl_u8(vget_low_u8(pixels_u8));
                    let lo_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_16)));
                    store0 = prefer_vfmaq_f32(store0, lo_lo, f_weight);
                    let lo_hi = vcvtq_f32_u32(vmovl_high_u16(lo_16));
                    store1 = prefer_vfmaq_f32(store1, lo_hi, f_weight);
                    let hi_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_16)));
                    store2 = prefer_vfmaq_f32(store2, hi_lo, f_weight);
                    let hi_hi = vcvtq_f32_u32(vmovl_high_u16(hi_16));
                    store3 = prefer_vfmaq_f32(store3, hi_hi, f_weight);

                    r += 1;
                }

                let store_0 = vcvtaq_u32_f32(store0);
                let store_1 = vcvtaq_u32_f32(store1);
                let store_2 = vcvtaq_u32_f32(store2);
                let store_3 = vcvtaq_u32_f32(store3);

                let store_lo = vcombine_u16(vmovn_u32(store_0), vmovn_u32(store_1));
                let store_hi = vcombine_u16(vmovn_u32(store_2), vmovn_u32(store_3));
                let store = vcombine_u8(vqmovn_u16(store_lo), vqmovn_u16(store_hi));

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u8;

                vst1q_u8(dst_ptr, store);

                cx += 16;
            }

            while cx + 8 < total_size {
                let mut store0: float32x4_t = zeros;
                let mut store1: float32x4_t = zeros;

                let mut r = -half_kernel;
                while r <= half_kernel {
                    let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let f_weight: float32x4_t = vdupq_n_f32(weight);

                    let py =
                        std::cmp::min(std::cmp::max(y as i64 + r as i64, 0), (height - 1) as i64);
                    let y_src_shift = py as usize * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let pixels_u8 = vld1_u8(s_ptr);
                    let pixels_u16 = vmovl_u8(pixels_u8);
                    let lo_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(pixels_u16)));
                    store0 = prefer_vfmaq_f32(store0, lo_lo, f_weight);
                    let lo_hi = vcvtq_f32_u32(vmovl_high_u16(pixels_u16));
                    store1 = prefer_vfmaq_f32(store1, lo_hi, f_weight);

                    r += 1;
                }

                let store_0 = vcvtaq_u32_f32(store0);
                let store_1 = vcvtaq_u32_f32(store1);

                let store_lo = vcombine_u16(vmovn_u32(store_0), vmovn_u32(store_1));
                let store = vqmovn_u16(store_lo);

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u8;

                vst1_u8(dst_ptr, store);

                cx += 8;
            }

            while cx + 4 < total_size {
                let mut store0: float32x4_t = zeros;

                let mut r = -half_kernel;
                while r <= half_kernel {
                    let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let f_weight: float32x4_t = vdupq_n_f32(weight);

                    let py =
                        std::cmp::min(std::cmp::max(y as i64 + r as i64, 0), (height - 1) as i64);
                    let y_src_shift = py as usize * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let pixels_u32 = load_u8_u32_fast::<4>(s_ptr);
                    let lo_lo = vcvtq_f32_u32(pixels_u32);
                    store0 = prefer_vfmaq_f32(store0, lo_lo, f_weight);

                    r += 1;
                }

                let store_0 = vcvtaq_u32_f32(store0);

                let store_c = vmovn_u32(store_0);
                let store_lo = vcombine_u16(store_c, store_c);
                let store = vqmovn_u16(store_lo);

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u32;

                let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store));
                dst_ptr.write_unaligned(pixel);

                cx += 4;
            }

            while cx < total_size {
                let mut store0 = vdup_n_f32(0f32);

                let mut r = -half_kernel;
                while r <= half_kernel {
                    let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let f_weight = vdup_n_f32(weight);

                    let py =
                        std::cmp::min(std::cmp::max(y as i64 + r as i64, 0), (height - 1) as i64);
                    let y_src_shift = py as usize * src_stride as usize;
                    let s_ptr = src.as_ptr().add(y_src_shift + cx);
                    let pixels_u32 = load_u8_u32_one(s_ptr);
                    let lo_lo = vcvt_f32_u32(pixels_u32);
                    store0 = prefer_vfma_f32(store0, lo_lo, f_weight);

                    r += 1;
                }

                let store_0 = vcvta_u32_f32(store0);

                let store_c = vmovn_u32(vcombine_u32(store_0, store_0));
                let store_lo = vcombine_u16(store_c, store_c);
                let store = vqmovn_u16(store_lo);

                let dst_ptr = unsafe_dst.slice.as_ptr().add(y_dst_shift + cx) as *mut u8;

                let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store));
                let bytes = pixel.to_le_bytes();
                dst_ptr.write_unaligned(bytes[0]);

                cx += 1;
            }
        }
    }
}
