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
    filter_2d, filter_2d_rgb, filter_2d_rgba, EdgeMode, FastBlurChannels, ImageSize, KernelShape,
    Scalar, ThreadingPolicy,
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
        *item = *item / sum;
    }

    kernel
}

pub fn motion_blur(
    image: &[u8],
    destination: &mut [u8],
    image_size: ImageSize,
    angle: f32,
    kernel_size: usize,
    channels: FastBlurChannels,
    threading_policy: ThreadingPolicy,
) {
    if kernel_size & 1 == 0 {
        panic!("Kernel size must be odd");
    }
    let kernel = generate_motion_kernel(kernel_size, angle);
    let executor = match channels {
        FastBlurChannels::Plane => filter_2d,
        FastBlurChannels::Channels3 => filter_2d_rgb,
        FastBlurChannels::Channels4 => filter_2d_rgba,
    };
    executor(
        image,
        destination,
        image_size,
        &kernel,
        KernelShape::new(kernel_size, kernel_size),
        EdgeMode::Reflect101,
        Scalar::default(),
        threading_policy,
    )
    .unwrap();
}
