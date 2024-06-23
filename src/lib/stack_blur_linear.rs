use crate::{FastBlurChannels, ThreadingPolicy};
use colorutils_rs::{
    linear_to_rgb, linear_to_rgba, rgb_to_linear, rgba_to_linear, TransferFunction,
};

/// Stack blur that will be performed in linear color space
///
/// Blurs in linear color space produces most pleasant results and mathematically correct to do so however significantly slower
/// This method is significantly slower than regular `u8` stack blur in perceptual colorspace
/// This is a very fast approximation using f32 accumulator size with radius less that *BASE_RADIUS_F64_CUTOFF*,
/// after it to avoid overflowing fallback to f64 accumulator will be used with some computational slowdown with factor ~1.5-2
///
/// # Arguments
/// * `in_place` - mutable buffer contains image data that will be used as a source and destination
/// * `stride` - Bytes per lane, default is width * channels_count if not aligned
/// * `width` - image width
/// * `height` - image height
/// * `radius` - since f32 accumulator is used under the hood radius almost is not limited
/// * `channels` - Count of channels of the image, only 3 and 4 is supported, alpha position, and channels order does not matter
/// * `threading_policy` - Threads usage policy
/// * `transfer_function` - Transfer function in linear colorspace
///
/// # Complexity
/// O(1) complexity.
pub fn stack_blur_in_linear(
    in_place: &mut [u8],
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
    threading_policy: ThreadingPolicy,
    transfer_function: TransferFunction,
) {
    let mut linear_data: Vec<f32> =
        vec![0f32; width as usize * height as usize * channels.get_channels()];

    let forward_transformer = match channels {
        FastBlurChannels::Channels3 => rgb_to_linear,
        FastBlurChannels::Channels4 => rgba_to_linear,
    };

    let inverse_transformer = match channels {
        FastBlurChannels::Channels3 => linear_to_rgb,
        FastBlurChannels::Channels4 => linear_to_rgba,
    };

    forward_transformer(
        &in_place,
        stride,
        &mut linear_data,
        width * std::mem::size_of::<f32>() as u32 * channels.get_channels() as u32,
        width,
        height,
        transfer_function,
    );

    crate::stack_blur_f32(
        &mut linear_data,
        width,
        height,
        radius,
        channels,
        threading_policy,
    );

    inverse_transformer(
        &linear_data,
        width * std::mem::size_of::<f32>() as u32 * channels.get_channels() as u32,
        in_place,
        stride,
        width,
        height,
        transfer_function,
    );
}
