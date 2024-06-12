use crate::gaussian::gaussian_filter::GaussianFilter;
use crate::gaussian::gaussian_horizontal::gaussian_blur_horizontal_pass_impl_clip_edge;
use crate::gaussian::gaussian_vertical::gaussian_blur_vertical_pass_clip_edge_impl;
use crate::unsafe_slice::UnsafeSlice;
use num_traits::FromPrimitive;
use rayon::ThreadPool;

pub(crate) fn gaussian_blur_vertical_pass_edge_clip_dispatch<
    T: FromPrimitive + Default + Into<f32> + Send + Sync,
    const CHANNEL_CONFIGURATION: usize,
>(
    src: &[T],
    src_stride: u32,
    dst: &mut [T],
    dst_stride: u32,
    width: u32,
    height: u32,
    filter: &Vec<GaussianFilter>,
    thread_pool: &ThreadPool,
    thread_count: u32,
) where
    T: std::ops::AddAssign + std::ops::SubAssign + Copy,
{
    let unsafe_dst = UnsafeSlice::new(dst);
    thread_pool.scope(|scope| {
        let segment_size = height / thread_count;

        for i in 0..thread_count {
            let start_y = i * segment_size;
            let mut end_y = (i + 1) * segment_size;
            if i == thread_count - 1 {
                end_y = height;
            }

            scope.spawn(move |_| {
                gaussian_blur_vertical_pass_clip_edge_impl::<T, CHANNEL_CONFIGURATION>(
                    src,
                    src_stride,
                    &unsafe_dst,
                    dst_stride,
                    width,
                    height,
                    filter,
                    start_y,
                    end_y,
                );
            });
        }
    });
}

pub(crate) fn gaussian_blur_horizontal_pass_edge_clip_dispatch<
    T: FromPrimitive + Default + Into<f32> + Send + Sync,
    const CHANNEL_CONFIGURATION: usize,
>(
    src: &[T],
    src_stride: u32,
    dst: &mut [T],
    dst_stride: u32,
    width: u32,
    height: u32,
    filter: &Vec<GaussianFilter>,
    thread_pool: &ThreadPool,
    thread_count: u32,
) where
    T: std::ops::AddAssign + std::ops::SubAssign + Copy,
{
    let unsafe_dst = UnsafeSlice::new(dst);
    thread_pool.scope(|scope| {
        let segment_size = height / thread_count;
        for i in 0..thread_count {
            let start_y = i * segment_size;
            let mut end_y = (i + 1) * segment_size;
            if i == thread_count - 1 {
                end_y = height;
            }

            scope.spawn(move |_| {
                gaussian_blur_horizontal_pass_impl_clip_edge::<T, CHANNEL_CONFIGURATION>(
                    src,
                    src_stride,
                    &unsafe_dst,
                    dst_stride,
                    width,
                    filter,
                    start_y,
                    end_y,
                );
            });
        }
    });
}
