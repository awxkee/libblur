use std::arch::aarch64::*;
use crate::neon::{prefer_vfmaq_f32, vhsumq_f32};
use crate::unsafe_slice::UnsafeSlice;

pub fn gaussian_horiz_one_chan_f32<T>(
    undef_src: &[T],
    src_stride: u32,
    undef_unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    kernel_size: usize,
    kernel: &[f32],
    start_y: u32,
    end_y: u32,
) {
    let src: &[f32] = unsafe { std::mem::transmute(undef_src) };
    let unsafe_dst: &UnsafeSlice<'_, f32> = unsafe { std::mem::transmute(undef_unsafe_dst) };
    let half_kernel = (kernel_size / 2) as i32;

    let mut _cy = start_y;

    for y in (_cy..end_y.saturating_sub(2)).step_by(2) {
        unsafe {
            for x in 0..width {
                let y_src_shift = y as usize * src_stride as usize;
                let y_dst_shift = y as usize * dst_stride as usize;

                let y_src_shift_next = y_src_shift + src_stride as usize;
                let y_dst_shift_next = y_dst_shift + dst_stride as usize;

                let mut store0: float32x4_t = vdupq_n_f32(0f32) ;
                let mut store1: float32x4_t = vdupq_n_f32(0f32) ;

                let mut r = -half_kernel;

                let zeros = vdupq_n_f32(0f32);

                while r + 32 <= half_kernel && ((x as i64 + r as i64 + 32i64) < width as i64) {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_set0_0 = vld1q_f32_x4(s_ptr);
                    let pixel_colors_f32_set0_1 = vld1q_f32_x4(s_ptr.add(16));

                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);

                    let pixel_colors_f32_set_next_0 = vld1q_f32_x4(s_ptr_next);
                    let pixel_colors_f32_set_next_1 = vld1q_f32_x4(s_ptr_next.add(16));

                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights_set_0 = vld1q_f32_x4(weight);
                    let weights_set_1 = vld1q_f32_x4(weight.add(16));

                    store0 = prefer_vfmaq_f32(store0, pixel_colors_f32_set0_0.0, weights_set_0.0);
                    store0 = prefer_vfmaq_f32(store0, pixel_colors_f32_set0_0.1, weights_set_0.1);
                    store0 = prefer_vfmaq_f32(store0, pixel_colors_f32_set0_0.2, weights_set_0.2);
                    store0 = prefer_vfmaq_f32(store0, pixel_colors_f32_set0_0.3, weights_set_0.3);

                    store0 = prefer_vfmaq_f32(store0, pixel_colors_f32_set0_1.0, weights_set_1.0);
                    store0 = prefer_vfmaq_f32(store0, pixel_colors_f32_set0_1.1, weights_set_1.1);
                    store0 = prefer_vfmaq_f32(store0, pixel_colors_f32_set0_1.2, weights_set_1.2);
                    store0 = prefer_vfmaq_f32(store0, pixel_colors_f32_set0_1.3, weights_set_1.3);

                    store1 = prefer_vfmaq_f32(store1, pixel_colors_f32_set_next_0.0, weights_set_0.0);
                    store1 = prefer_vfmaq_f32(store1, pixel_colors_f32_set_next_0.1, weights_set_0.1);
                    store1 = prefer_vfmaq_f32(store1, pixel_colors_f32_set_next_0.2, weights_set_0.2);
                    store1 = prefer_vfmaq_f32(store1, pixel_colors_f32_set_next_0.3, weights_set_0.3);

                    store1 = prefer_vfmaq_f32(store1, pixel_colors_f32_set_next_1.0, weights_set_1.0);
                    store1 = prefer_vfmaq_f32(store1, pixel_colors_f32_set_next_1.1, weights_set_1.1);
                    store1 = prefer_vfmaq_f32(store1, pixel_colors_f32_set_next_1.2, weights_set_1.2);
                    store1 = prefer_vfmaq_f32(store1, pixel_colors_f32_set_next_1.3, weights_set_1.3);

                    r += 32;
                }

                while r + 16 <= half_kernel && ((x as i64 + r as i64 + 16i64) < width as i64) {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_set_0 = vld1q_f32_x4(s_ptr);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_f32_set_1 = vld1q_f32_x4(s_ptr_next);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights_set = vld1q_f32_x4(weight);
                    store0 = prefer_vfmaq_f32(store0, pixel_colors_f32_set_0.0, weights_set.0);
                    store0 = prefer_vfmaq_f32(store0, pixel_colors_f32_set_0.1, weights_set.1);
                    store0 = prefer_vfmaq_f32(store0, pixel_colors_f32_set_0.2, weights_set.2);
                    store0 = prefer_vfmaq_f32(store0, pixel_colors_f32_set_0.3, weights_set.3);

                    store1 = prefer_vfmaq_f32(store1, pixel_colors_f32_set_1.0, weights_set.0);
                    store1 = prefer_vfmaq_f32(store1, pixel_colors_f32_set_1.1, weights_set.1);
                    store1 = prefer_vfmaq_f32(store1, pixel_colors_f32_set_1.2, weights_set.2);
                    store1 = prefer_vfmaq_f32(store1, pixel_colors_f32_set_1.3, weights_set.3);

                    r += 16;
                }

                while r + 4 <= half_kernel && ((x as i64 + r as i64 + 4i64) < width as i64) {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_0 = vld1q_f32(s_ptr);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_f32_1 = vld1q_f32(s_ptr_next);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let f_weight: float32x4_t = vld1q_f32(weight);
                    store0 = prefer_vfmaq_f32(store0, pixel_colors_f32_0, f_weight);
                    store1 = prefer_vfmaq_f32(store1, pixel_colors_f32_1, f_weight);

                    r += 4;
                }

                while r <= half_kernel {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_0 = vld1q_lane_f32::<0>(s_ptr, zeros);
                    let s_ptr_next = src.as_ptr().add(y_src_shift_next + px);
                    let pixel_colors_f32_1 = vld1q_lane_f32::<0>(s_ptr_next, zeros);
                    let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let f_weight: float32x4_t = vdupq_n_f32(weight);
                    store0 = prefer_vfmaq_f32(store0, pixel_colors_f32_0, f_weight);
                    store1 = prefer_vfmaq_f32(store1, pixel_colors_f32_1, f_weight);

                    r += 1;
                }

                let agg0 = vhsumq_f32(store0);
                let offset = y_dst_shift + x as usize;
                let dst_ptr0 = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                dst_ptr0.write_unaligned(agg0);

                let agg1 = vhsumq_f32(store1);
                let offset = y_dst_shift_next + x as usize;
                let dst_ptr1 = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                dst_ptr1.write_unaligned(agg1);
            }
        }
        _cy = y;
    }

    for y in _cy..end_y {
        unsafe {
            for x in 0..width {
                let y_src_shift = y as usize * src_stride as usize;
                let y_dst_shift = y as usize * dst_stride as usize;

                let mut store: float32x4_t = vdupq_n_f32(0f32) ;

                let mut r = -half_kernel;

                let zeros = vdupq_n_f32(0f32);

                while r + 32 <= half_kernel && ((x as i64 + r as i64 + 32i64) < width as i64) {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_set_0 = vld1q_f32_x4(s_ptr);
                    let pixel_colors_f32_set_1 = vld1q_f32_x4(s_ptr.add(16));
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights_set_0 = vld1q_f32_x4(weight);
                    let weights_set_1 = vld1q_f32_x4(weight.add(16));

                    store = prefer_vfmaq_f32(store, pixel_colors_f32_set_0.0, weights_set_0.0);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32_set_0.1, weights_set_0.1);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32_set_0.2, weights_set_0.2);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32_set_0.3, weights_set_0.3);

                    store = prefer_vfmaq_f32(store, pixel_colors_f32_set_1.0, weights_set_1.0);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32_set_1.1, weights_set_1.1);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32_set_1.2, weights_set_1.2);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32_set_1.3, weights_set_1.3);

                    r += 32;
                }

                while r + 16 <= half_kernel && ((x as i64 + r as i64 + 16i64) < width as i64) {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32_set = vld1q_f32_x4(s_ptr);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let weights_set = vld1q_f32_x4(weight);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32_set.0, weights_set.0);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32_set.1, weights_set.1);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32_set.2, weights_set.2);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32_set.3, weights_set.3);

                    r += 16;
                }

                while r + 4 <= half_kernel && ((x as i64 + r as i64 + 4i64) < width as i64) {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32 = vld1q_f32(s_ptr);
                    let weight = kernel.as_ptr().add((r + half_kernel) as usize);
                    let f_weight: float32x4_t = vld1q_f32(weight);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32, f_weight);

                    r += 4;
                }

                while r <= half_kernel {
                    let current_x =
                        std::cmp::min(std::cmp::max(x as i64 + r as i64, 0), (width - 1) as i64)
                            as usize;
                    let px = current_x;
                    let s_ptr = src.as_ptr().add(y_src_shift + px);
                    let pixel_colors_f32 = vld1q_lane_f32::<0>(s_ptr, zeros);
                    let weight = *kernel.get_unchecked((r + half_kernel) as usize);
                    let f_weight: float32x4_t = vdupq_n_f32(weight);
                    store = prefer_vfmaq_f32(store, pixel_colors_f32, f_weight);

                    r += 1;
                }

                let agg = vhsumq_f32(store);
                let offset = y_dst_shift + x as usize;
                let dst_ptr = unsafe_dst.slice.as_ptr().add(offset) as *mut f32;
                dst_ptr.write_unaligned(agg);
            }
        }
    }
}
