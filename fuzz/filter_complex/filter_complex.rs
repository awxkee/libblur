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

#![no_main]

use arbitrary::Arbitrary;
use libblur::{
    filter_1d_complex, BlurImage, BlurImageMut, EdgeMode, EdgeMode2D, FastBlurChannels, Scalar,
    ThreadingPolicy,
};
use libfuzzer_sys::fuzz_target;
use num_complex::Complex;

#[derive(Clone, Debug, Arbitrary)]
pub struct SrcImage {
    pub src_width: u16,
    pub src_height: u16,
    pub value: u8,
    pub edge_mode_horizontal: u8,
    pub edge_mode_vertical: u8,
    pub x_kernel_size: u8,
    pub y_kernel_size: u8,
    pub channel: u8,
    pub base: u8,
    pub threading: bool,
}

fn complex_gaussian_kernel(radius: f64, scale: f64, a: f64, b: f64) -> Vec<Complex<f32>> {
    let kernel_radius = radius.ceil() as usize;
    let mut kernel: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); 1 + 2 * (kernel_radius)];

    for i in -(kernel_radius as isize)..=(kernel_radius as isize) {
        let ax = i as f64 * scale / radius;
        let ax2 = ax * ax;
        let exp_a = (-a * ax2).exp();
        let val = Complex::new(exp_a * (b * ax2).cos(), exp_a * (b * ax2).sin());
        kernel[(i + kernel_radius as isize) as usize] = val;
    }

    let sum: f64 = kernel.iter().map(|z| z.norm_sqr()).sum::<f64>();
    if sum != 0.0 {
        kernel
            .iter()
            .map(|z| Complex {
                re: (z.re / sum) as f32,
                im: (z.im / sum) as f32,
            })
            .collect::<Vec<_>>()
    } else {
        kernel
            .iter()
            .map(|x| Complex {
                re: x.re as f32,
                im: x.im as f32,
            })
            .collect()
    }
}

fuzz_target!(|data: SrcImage| {
    if data.src_width > 250 || data.src_height > 250 {
        return;
    }
    if data.x_kernel_size > 45 || data.x_kernel_size == 0 {
        return;
    }
    if data.y_kernel_size > 45 || data.y_kernel_size == 0 {
        return;
    }
    let edge_mode_horizontal = match data.edge_mode_horizontal % 4 {
        0 => EdgeMode::Clamp,
        1 => EdgeMode::Wrap,
        2 => EdgeMode::Reflect,
        _ => EdgeMode::Reflect101,
    };
    let edge_mode_vertical = match data.edge_mode_vertical % 4 {
        0 => EdgeMode::Clamp,
        1 => EdgeMode::Wrap,
        2 => EdgeMode::Reflect,
        _ => EdgeMode::Reflect101,
    };
    let channel = match data.channel % 3 {
        0 => FastBlurChannels::Plane,
        1 => FastBlurChannels::Channels3,
        _ => FastBlurChannels::Channels4,
    };
    let mp = if data.threading {
        ThreadingPolicy::Adaptive
    } else {
        ThreadingPolicy::Single
    };
    match data.base % 3 {
        0 => {
            fuzz_8bit(
                data.src_width as usize,
                data.src_height as usize,
                data.x_kernel_size as usize,
                data.y_kernel_size as usize,
                channel,
                EdgeMode2D::anisotropy(edge_mode_horizontal, edge_mode_vertical),
                mp,
            );
        }
        1 => {
            fuzz_16bit(
                data.src_width as usize,
                data.src_height as usize,
                data.x_kernel_size as usize,
                data.y_kernel_size as usize,
                channel,
                EdgeMode2D::anisotropy(edge_mode_horizontal, edge_mode_vertical),
                mp,
            );
        }
        _ => {
            fuzz_f32(
                data.src_width as usize,
                data.src_height as usize,
                data.x_kernel_size as usize,
                data.y_kernel_size as usize,
                channel,
                EdgeMode2D::anisotropy(edge_mode_horizontal, edge_mode_vertical),
                mp,
            );
        }
    }
});

fn fuzz_8bit(
    width: usize,
    height: usize,
    x_kernel_size: usize,
    y_kernel_size: usize,
    channels: FastBlurChannels,
    edge_modes: EdgeMode2D,
    threading_policy: ThreadingPolicy,
) {
    if width == 0 || height == 0 || x_kernel_size == 0 || y_kernel_size == 0 {
        return;
    }
    let src_image = BlurImage::alloc(width as u32, height as u32, channels);
    let mut dst_image = BlurImageMut::alloc(width as u32, height as u32, channels);

    let x_kernel = complex_gaussian_kernel(x_kernel_size as f64, y_kernel_size as f64, 1.0, 1.0);
    let y_kernel = complex_gaussian_kernel(x_kernel_size as f64, y_kernel_size as f64, 1.0, 1.0);

    match channels {
        FastBlurChannels::Plane => {
            filter_1d_complex::<u8, f32, 1>(
                &src_image,
                &mut dst_image,
                &x_kernel,
                &y_kernel,
                edge_modes,
                Scalar::default(),
                threading_policy,
            )
            .unwrap();
        }
        FastBlurChannels::Channels3 => {
            filter_1d_complex::<u8, f32, 3>(
                &src_image,
                &mut dst_image,
                &x_kernel,
                &y_kernel,
                edge_modes,
                Scalar::default(),
                threading_policy,
            )
            .unwrap();
        }
        FastBlurChannels::Channels4 => {
            filter_1d_complex::<u8, f32, 4>(
                &src_image,
                &mut dst_image,
                &x_kernel,
                &y_kernel,
                edge_modes,
                Scalar::default(),
                threading_policy,
            )
            .unwrap();
        }
    }
}

fn fuzz_16bit(
    width: usize,
    height: usize,
    x_kernel_size: usize,
    y_kernel_size: usize,
    channels: FastBlurChannels,
    edge_mode: EdgeMode2D,
    threading_policy: ThreadingPolicy,
) {
    if width == 0 || height == 0 || x_kernel_size == 0 || y_kernel_size == 0 {
        return;
    }
    let src_image = BlurImage::alloc(width as u32, height as u32, channels);
    let mut dst_image = BlurImageMut::alloc(width as u32, height as u32, channels);

    let x_kernel = complex_gaussian_kernel(x_kernel_size as f64, y_kernel_size as f64, 1.0, 1.0);
    let y_kernel = complex_gaussian_kernel(x_kernel_size as f64, y_kernel_size as f64, 1.0, 1.0);

    match channels {
        FastBlurChannels::Plane => {
            filter_1d_complex::<u16, f32, 1>(
                &src_image,
                &mut dst_image,
                &x_kernel,
                &y_kernel,
                edge_mode,
                Scalar::default(),
                threading_policy,
            )
            .unwrap();
        }
        FastBlurChannels::Channels3 => {
            filter_1d_complex::<u16, f32, 3>(
                &src_image,
                &mut dst_image,
                &x_kernel,
                &y_kernel,
                edge_mode,
                Scalar::default(),
                threading_policy,
            )
            .unwrap();
        }
        FastBlurChannels::Channels4 => {
            filter_1d_complex::<u16, f32, 4>(
                &src_image,
                &mut dst_image,
                &x_kernel,
                &y_kernel,
                edge_mode,
                Scalar::default(),
                threading_policy,
            )
            .unwrap();
        }
    }
}

fn fuzz_f32(
    width: usize,
    height: usize,
    x_kernel_size: usize,
    y_kernel_size: usize,
    channels: FastBlurChannels,
    edge_mode: EdgeMode2D,
    threading_policy: ThreadingPolicy,
) {
    if width == 0 || height == 0 || x_kernel_size == 0 || y_kernel_size == 0 {
        return;
    }
    let src_image = BlurImage::alloc(width as u32, height as u32, channels);
    let mut dst_image = BlurImageMut::alloc(width as u32, height as u32, channels);

    let x_kernel = complex_gaussian_kernel(x_kernel_size as f64, y_kernel_size as f64, 1.0, 1.0);
    let y_kernel = complex_gaussian_kernel(x_kernel_size as f64, y_kernel_size as f64, 1.0, 1.0);

    match channels {
        FastBlurChannels::Plane => {
            filter_1d_complex::<f32, f32, 1>(
                &src_image,
                &mut dst_image,
                &x_kernel,
                &y_kernel,
                edge_mode,
                Scalar::default(),
                threading_policy,
            )
            .unwrap();
        }
        FastBlurChannels::Channels3 => {
            filter_1d_complex::<f32, f32, 3>(
                &src_image,
                &mut dst_image,
                &x_kernel,
                &y_kernel,
                edge_mode,
                Scalar::default(),
                threading_policy,
            )
            .unwrap();
        }
        FastBlurChannels::Channels4 => {
            filter_1d_complex::<f32, f32, 4>(
                &src_image,
                &mut dst_image,
                &x_kernel,
                &y_kernel,
                edge_mode,
                Scalar::default(),
                threading_policy,
            )
            .unwrap();
        }
    }
}
