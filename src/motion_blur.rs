/*
 * // Copyright (c) Radzivon Bartoshyk. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use crate::{
    filter_2d, BlurError, BlurImage, BlurImageMut, EdgeMode, KernelShape, Scalar, ThreadingPolicy,
};

#[derive(Copy, Clone)]
pub struct BresenhamPoint {
    pub x: i64,
    pub y: i64,
}

impl BresenhamPoint {
    pub fn new(x: i64, y: i64) -> BresenhamPoint {
        BresenhamPoint { x, y }
    }
}

fn draw_line_bresenham(
    width: usize,
    height: usize,
    x0: isize,
    y0: isize,
    x1: isize,
    y1: isize,
) -> Vec<BresenhamPoint> {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;

    let mut result = vec![];

    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };

    while x != x1 || y != y1 {
        if x >= 0 && x < width as isize && y >= 0 && y < height as isize {
            result.push(BresenhamPoint::new(x as i64, y as i64));
        }

        let e2 = 2 * err;

        if e2 >= dy {
            err += dy;
            x += sx;
        }

        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
    if x >= 0 && x < width as isize && y >= 0 && y < height as isize {
        result.push(BresenhamPoint::new(x as i64, y as i64));
    }

    result
}

pub fn generate_motion_kernel(size: usize, angle_deg: f32) -> Vec<f32> {
    let mut kernel = vec![0.0; size * size];

    // Convert the angle to radians
    let angle_rad = angle_deg * std::f32::consts::PI / 180.0;

    let mut sum = 0f32;

    let start_pos = size / 2;
    let end_pos = size / 2;

    let pos_x = (start_pos as f32 + size as f32 * angle_rad.cos()).ceil() as usize;
    let pos_y = (end_pos as f32 + size as f32 * angle_rad.sin()).ceil() as usize;

    let end_pos_x = (size as isize - pos_x as isize).saturating_sub(1);
    let end_pos_y = (size as isize - pos_y as isize).saturating_sub(1);

    let points = draw_line_bresenham(
        size,
        size,
        pos_x as isize,
        pos_y as isize,
        end_pos_x,
        end_pos_y,
    );

    for point in points {
        kernel[point.y as usize * size + point.x as usize] = 1.;
        sum += 1.;
    }

    for item in kernel.iter_mut() {
        *item /= sum;
    }

    kernel
}

/// Performs motion blur on the image
///
/// # Arguments
///
/// * `image`: Source image.
/// * `destination`: Destination image.
/// * `angle`: Degree of acceleration, in degrees.
/// * `kernel_size`: Convolve kernel size, must be odd!
/// * `border_mode`: See [EdgeMode] for more info.
/// * `border_constant`: If [EdgeMode::Constant] border will be replaced with this provided [Scalar] value.
/// * `threading_policy`: see [ThreadingPolicy] for more info.
///
/// returns: ()
///
pub fn motion_blur(
    image: &BlurImage<u8>,
    destination: &mut BlurImageMut<u8>,
    angle: f32,
    kernel_size: usize,
    border_mode: EdgeMode,
    border_constant: Scalar,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    image.check_layout()?;
    destination.check_layout(Some(image))?;
    image.size_matches_mut(destination)?;
    if kernel_size & 1 == 0 {
        return Err(BlurError::OddKernel(kernel_size));
    }
    let kernel = generate_motion_kernel(kernel_size, angle);
    filter_2d::<u8, f32>(
        image,
        destination,
        &kernel,
        KernelShape::new(kernel_size, kernel_size),
        border_mode,
        border_constant,
        threading_policy,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FastBlurChannels;

    #[test]
    fn test_motion_k25_a25_rgb() {
        let width: usize = 188;
        let height: usize = 188;
        let src = vec![126; width * height * 3];
        let src_image = BlurImage::borrow(
            &src,
            width as u32,
            height as u32,
            FastBlurChannels::Channels3,
        );
        let mut dst = BlurImageMut::default();
        motion_blur(
            &src_image,
            &mut dst,
            90.,
            25,
            EdgeMode::Clamp,
            Scalar::default(),
            ThreadingPolicy::Single,
        )
        .unwrap();
        for (i, &cn) in dst.data.borrow_mut().iter().enumerate() {
            let diff = (cn as i32 - 126).abs();
            assert!(
                diff <= 3,
                "Diff expected to be less than 3 but it was {diff} at {i}"
            );
        }
    }

    #[test]
    fn test_motion_k25_a25_rgba() {
        let width: usize = 188;
        let height: usize = 188;
        let src = vec![126; width * height * 4];
        let src_image = BlurImage::borrow(
            &src,
            width as u32,
            height as u32,
            FastBlurChannels::Channels4,
        );
        let mut dst = BlurImageMut::default();
        motion_blur(
            &src_image,
            &mut dst,
            90.,
            25,
            EdgeMode::Clamp,
            Scalar::default(),
            ThreadingPolicy::Single,
        )
        .unwrap();
        for (i, &cn) in dst.data.borrow_mut().iter().enumerate() {
            let diff = (cn as i32 - 126).abs();
            assert!(
                diff <= 3,
                "Diff expected to be less than 3 but it was {diff} at {i}"
            );
        }
    }

    #[test]
    fn test_motion_k25_a25_plane() {
        let width: usize = 188;
        let height: usize = 188;
        let src = vec![126; width * height];
        let src_image =
            BlurImage::borrow(&src, width as u32, height as u32, FastBlurChannels::Plane);
        let mut dst = BlurImageMut::default();
        motion_blur(
            &src_image,
            &mut dst,
            90.,
            25,
            EdgeMode::Clamp,
            Scalar::default(),
            ThreadingPolicy::Single,
        )
        .unwrap();
        for (i, &cn) in dst.data.borrow_mut().iter().enumerate() {
            let diff = (cn as i32 - 126).abs();
            assert!(
                diff <= 3,
                "Diff expected to be less than 3 but it was {diff} at {i}"
            );
        }
    }
}
