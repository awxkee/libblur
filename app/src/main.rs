/*
 * // Copyright (c) Radzivon Bartoshyk 3/2025. All rights reserved.
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

mod merge;
mod split;

use image::{EncodableLayout, GenericImageView, ImageReader};
use libblur::{
    filter_1d_exact, gaussian_kernel_1d, generate_motion_kernel, motion_blur, sigma_size,
    BlurImage, BlurImageMut, BufferStore, ConvolutionMode, EdgeMode, FastBlurChannels, ImageSize,
    Scalar, ThreadingPolicy,
};
use std::arch::x86_64::*;
use std::time::Instant;

#[allow(dead_code)]
fn f32_to_f16(bytes: Vec<f32>) -> Vec<u16> {
    bytes
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect()
}

#[allow(dead_code)]
fn f16_to_f32(bytes: Vec<u16>) -> Vec<f32> {
    bytes
        .iter()
        .map(|&x| half::f16::from_bits(x).to_f32())
        .collect()
}

#[inline(always)]
pub(crate) unsafe fn _mm_mulhi_epi32(a0: __m128i, b0: __m256i) -> __m128i {
    let rnd = _mm256_set1_epi64x((1 << 30) - 1);

    let perm = _mm256_setr_epi32(0, -1, 1, -1, 2, -1, 3, -1);

    let v0p = _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(a0), perm);

    let mut values: [i64; 8] = [0; 8];
    _mm256_storeu_si256(values.as_mut_ptr() as *mut _, v0p);
    println!("{:?}", values);

    let lo = _mm256_mul_epi32(v0p, b0);

    let mut values: [i64; 8] = [0; 8];
    _mm256_storeu_si256(values.as_mut_ptr() as *mut _, lo);
    println!("{:?}", values);

    let a0 = _mm256_add_epi64(lo, rnd);

    let mut values: [i64; 8] = [0; 8];
    _mm256_storeu_si256(values.as_mut_ptr() as *mut _, lo);
    println!("{:?}", values);

    let b0 = _mm256_srli_epi64::<31>(a0);

    let mut values: [i64; 8] = [0; 8];
    _mm256_storeu_si256(values.as_mut_ptr() as *mut _, lo);
    println!("{:?}", values);

    let perm = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
    let shuffled0 = _mm256_permutevar8x32_epi32(b0, perm);
    _mm256_castsi256_si128(shuffled0)
}

fn main() {
    unsafe {
        let v0 = _mm256_set1_epi64x(10);

        let v1 = _mm_set1_epi32(((1i64 << 31) - 1) as i32);

        let mut values: [i32; 8] = [0; 8];
        _mm_storeu_si128(values.as_mut_ptr() as *mut __m128i, v1);
        println!("{:?}", values);

        let v = _mm_mulhi_epi32(v1, v0);
        let mut values: [i32; 8] = [0; 8];
        _mm_storeu_si128(values.as_mut_ptr() as *mut __m128i, v);
        println!("{:?}", values);
    }
    let dyn_image = ImageReader::open("./assets/test_image_2.png")
        .unwrap()
        .decode()
        .unwrap();

    let dimensions = dyn_image.dimensions();
    println!("dimensions {:?}", dyn_image.dimensions());
    println!("type {:?}", dyn_image.color());

    // let vldg = dyn_image.to_rgb8();
    // let new_rgb = image::imageops::blur(&vldg, 66.);
    // let dyn_image = DynamicImage::ImageRgb8(dyn_image.to_rgb8());
    // new_dyn.save("output.jpg").unwrap();

    println!("{:?}", dyn_image.color());

    let img = dyn_image.to_rgba8();
    let mut src_bytes = img.as_bytes();
    let components = 4;
    let stride = dimensions.0 as usize * components;
    let mut bytes: Vec<u8> = src_bytes.to_vec();
    let mut dst_bytes: Vec<u8> = src_bytes.to_vec();

    let start_time = Instant::now();

    println!("libblur::stack_blur: {:?}", start_time.elapsed());

    let start = Instant::now();

    let mut v_vec = src_bytes
        .to_vec()
        .iter()
        .map(|&x| u16::from_ne_bytes([x, x]))
        .collect::<Vec<u16>>();

    let mut dst_image = BlurImageMut::borrow(
        &mut v_vec,
        dyn_image.width(),
        dyn_image.height(),
        FastBlurChannels::Channels4,
    );

    // let image = BlurImage::borrow(
    //     &src_bytes,
    //     dyn_image.width(),
    //     dyn_image.height(),
    //     FastBlurChannels::Channels4,
    // );
    // let mut dst_image = BlurImageMut::default();
    //
    // libblur::gaussian_box_blur(&image, &mut dst_image, 10., ThreadingPolicy::Single).unwrap();

    libblur::fast_gaussian_next_u16(&mut dst_image, 6, ThreadingPolicy::Single, EdgeMode::Clamp)
        .unwrap();

    // libblur::motion_blur(
    //     &image,
    //     &mut dst_image,
    //     35.,
    //     43,
    //     EdgeMode::Clamp,
    //     Scalar::new(0.0, 0.0, 0.0, 0.0),
    //     ThreadingPolicy::Single,
    // )
    // .unwrap();

    dst_bytes = dst_image
        .data
        .borrow()
        .iter()
        .map(|&x| (x >> 8) as u8)
        // .map(|&x| (x * 255f32).round() as u8)
        .collect::<Vec<u8>>();

    // dst_bytes = dst_image.data.borrow().to_vec();
    //
    // filter_2d_rgba_approx::<u8, f32, i32>(
    //     &bytes,
    //     &mut dst_bytes,
    //     ImageSize::new(dimensions.0 as usize, dimensions.1 as usize),
    //     &kernel,
    //     &kernel,
    //     EdgeMode::Clamp,
    //     ThreadingPolicy::Adaptive,
    // )
    // .unwrap();

    let elapsed_time = start_time.elapsed();
    // Print the elapsed time in milliseconds
    println!("Elapsed time: {:.2?}", elapsed_time);

    // let start_time = Instant::now();
    // let blurred = dyn_image.blur(125f32);
    // println!("Gauss image: {:.2?}", start_time.elapsed());
    // blurred.save("dyn.jpg").unwrap();

    bytes = dst_bytes;

    if components == 3 {
        image::save_buffer(
            "blurred_stack_next.jpg",
            bytes.as_bytes(),
            dimensions.0,
            dimensions.1,
            image::ExtendedColorType::Rgb8,
        )
        .unwrap();
    } else {
        image::save_buffer(
            "blurred_stack_next_f.png",
            bytes.as_bytes(),
            dimensions.0,
            dimensions.1,
            if components == 3 {
                image::ExtendedColorType::Rgb8
            } else {
                image::ExtendedColorType::Rgba8
            },
        )
        .unwrap();
    }
}
