use crate::gaussian::gaussian_filter::GaussianFilter;
use crate::unsafe_slice::UnsafeSlice;
use num_traits::FromPrimitive;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::gaussian::gaussian_neon_filter::neon_gaussian_filter::{gaussian_blur_vertical_pass_filter_neon};

pub fn gaussian_blur_vertical_pass_c_impl<
    T: FromPrimitive + Default + Into<f32> + Send + Sync + Copy,
    const CHANNEL_CONFIGURATION: usize,
>(
    src: &[T],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    height: u32,
    kernel_size: usize,
    kernel: &Vec<f32>,
    start_y: u32,
    end_y: u32,
) {
    let total_length = width as usize * std::mem::size_of::<T>() * CHANNEL_CONFIGURATION;
    for y in start_y..end_y {
        let mut _cx = 0usize;

        while _cx + 32 < total_length {
            gaussian_vertical_row::<T, 32>(
                src,
                src_stride,
                unsafe_dst,
                dst_stride,
                width,
                height,
                kernel_size,
                kernel,
                _cx as u32,
                y,
            );
            _cx += 32;
        }

        while _cx + 16 < total_length {
            gaussian_vertical_row::<T, 16>(
                src,
                src_stride,
                unsafe_dst,
                dst_stride,
                width,
                height,
                kernel_size,
                kernel,
                _cx as u32,
                y,
            );
            _cx += 16;
        }

        while _cx + 8 < total_length {
            gaussian_vertical_row::<T, 8>(
                src,
                src_stride,
                unsafe_dst,
                dst_stride,
                width,
                height,
                kernel_size,
                kernel,
                _cx as u32,
                y,
            );
            _cx += 8;
        }

        while _cx + 4 < total_length {
            gaussian_vertical_row::<T, 4>(
                src,
                src_stride,
                unsafe_dst,
                dst_stride,
                width,
                height,
                kernel_size,
                kernel,
                _cx as u32,
                y,
            );
            _cx += 4;
        }

        while _cx < total_length {
            gaussian_vertical_row::<T, 1>(
                src,
                src_stride,
                unsafe_dst,
                dst_stride,
                width,
                height,
                kernel_size,
                kernel,
                _cx as u32,
                y,
            );
            _cx += 1;
        }
    }
}

#[inline]
pub fn gaussian_vertical_row<
    T: FromPrimitive + Default + Into<f32> + Send + Sync + Copy,
    const ROW_SIZE: usize,
>(
    src: &[T],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    _: u32,
    height: u32,
    kernel_size: usize,
    kernel: &[f32],
    x: u32,
    y: u32,
) {
    let half_kernel = (kernel_size / 2) as i32;
    let mut weights: [f32; ROW_SIZE] = [0f32; ROW_SIZE];
    for r in -half_kernel..=half_kernel {
        let py = std::cmp::min(std::cmp::max(y as i64 + r as i64, 0), (height - 1) as i64);
        let y_src_shift = py as usize * src_stride as usize;
        let weight = unsafe { *kernel.get_unchecked((r + half_kernel) as usize) };
        for i in 0..ROW_SIZE {
            let px = x as usize + i;
            unsafe {
                let v = *src.get_unchecked(y_src_shift + px);
                let w0 = weights.get_unchecked_mut(i);
                *w0 = *w0 + v.into() * weight;
            }
        }
    }
    let y_dst_shift = y as usize * dst_stride as usize;
    unsafe {
        for i in 0..ROW_SIZE {
            let px = x as usize + i;
            unsafe_dst.write(
                y_dst_shift + px,
                T::from_f32((*weights.get_unchecked(i)).round()).unwrap_or_default(),
            );
        }
    }
}

pub fn gaussian_blur_vertical_pass_clip_edge_impl<
    T: FromPrimitive + Default + Into<f32> + Send + Sync + Copy,
    const CHANNEL_CONFIGURATION: usize,
>(
    src: &[T],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    width: u32,
    height: u32,
    filter: &Vec<GaussianFilter>,
    start_y: u32,
    end_y: u32,
) {
    if std::any::type_name::<T>() == "u8" {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            let u8_slice: &[u8] = unsafe { std::mem::transmute(src) };
            let slice: &UnsafeSlice<'_, u8> = unsafe { std::mem::transmute(unsafe_dst) };
            gaussian_blur_vertical_pass_filter_neon::<CHANNEL_CONFIGURATION>(
                u8_slice, src_stride, slice, dst_stride, width, height, filter, start_y, end_y,
            );
            return;
        }
    }
    let total_length = width as usize * std::mem::size_of::<T>() * CHANNEL_CONFIGURATION;
    for y in start_y..end_y {
        let mut _cx = 0usize;

        while _cx + 32 < total_length {
            gaussian_vertical_row_clip_edge::<T, 32>(
                src, src_stride, unsafe_dst, dst_stride, width, height, filter, _cx as u32, y,
            );
            _cx += 32;
        }

        while _cx + 16 < total_length {
            gaussian_vertical_row_clip_edge::<T, 16>(
                src, src_stride, unsafe_dst, dst_stride, width, height, filter, _cx as u32, y,
            );
            _cx += 16;
        }

        while _cx + 8 < total_length {
            gaussian_vertical_row_clip_edge::<T, 8>(
                src, src_stride, unsafe_dst, dst_stride, width, height, filter, _cx as u32, y,
            );
            _cx += 8;
        }

        while _cx + 4 < total_length {
            gaussian_vertical_row_clip_edge::<T, 4>(
                src, src_stride, unsafe_dst, dst_stride, width, height, filter, _cx as u32, y,
            );
            _cx += 4;
        }

        while _cx < total_length {
            gaussian_vertical_row_clip_edge::<T, 1>(
                src, src_stride, unsafe_dst, dst_stride, width, height, filter, _cx as u32, y,
            );
            _cx += 1;
        }
    }
}

#[inline]
pub fn gaussian_vertical_row_clip_edge<
    T: FromPrimitive + Default + Into<f32> + Send + Sync + Copy,
    const ROW_SIZE: usize,
>(
    src: &[T],
    src_stride: u32,
    unsafe_dst: &UnsafeSlice<T>,
    dst_stride: u32,
    _: u32,
    _: u32,
    filter: &Vec<GaussianFilter>,
    x: u32,
    y: u32,
) {
    let mut weights: [f32; ROW_SIZE] = [0f32; ROW_SIZE];
    let current_filter = unsafe { filter.get_unchecked(y as usize) };
    let filter_start = current_filter.start;
    let filter_weights = &current_filter.filter;
    for j in 0..current_filter.size {
        let py = filter_start + j;
        let y_src_shift = py * src_stride as usize;
        let weight = unsafe { *filter_weights.get_unchecked(j) };
        for i in 0..ROW_SIZE {
            let px = x as usize + i;
            unsafe {
                let v = *src.get_unchecked(y_src_shift + px);
                let w0 = weights.get_unchecked_mut(i);
                *w0 = *w0 + v.into() * weight;
            }
        }
    }
    let y_dst_shift = y as usize * dst_stride as usize;
    unsafe {
        for i in 0..ROW_SIZE {
            let px = x as usize + i;
            unsafe_dst.write(
                y_dst_shift + px,
                T::from_f32((*weights.get_unchecked(i)).round()).unwrap_or_default(),
            );
        }
    }
}
