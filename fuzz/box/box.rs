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
use libblur::{BlurImage, BlurImageMut, BoxBlurParameters, FastBlurChannels, ThreadingPolicy};
use libfuzzer_sys::fuzz_target;

#[derive(Clone, Debug, Arbitrary)]
pub struct SrcImage {
    pub src_width: u16,
    pub src_height: u16,
    pub value: u16,
    pub edge_mode: u8,
    pub x_kernel_size: u8,
    pub y_kernel_size: u8,
    pub plane: u8,
    pub threading: bool,
}

fuzz_target!(|data: SrcImage| {
    if data.src_width > 250 || data.src_height > 250 {
        return;
    }
    if data.x_kernel_size == 0 || data.x_kernel_size % 2 == 0 {
        return;
    }
    if data.y_kernel_size == 0 || data.y_kernel_size % 2 == 0 {
        return;
    }
    let plane_match = match data.plane % 3 {
        0 => FastBlurChannels::Channels4,
        1 => FastBlurChannels::Channels3,
        _ => FastBlurChannels::Plane,
    };
    fuzz_8bit(
        data.src_width as usize,
        data.src_height as usize,
        data.x_kernel_size as usize,
        data.y_kernel_size as usize,
        plane_match,
        if data.threading {
            ThreadingPolicy::Adaptive
        } else {
            ThreadingPolicy::Single
        },
    );
});

fn fuzz_8bit(
    width: usize,
    height: usize,
    x_kernel_size: usize,
    y_kernel_size: usize,
    channels: FastBlurChannels,
    threading: ThreadingPolicy,
) {
    if width == 0 || height == 0 || x_kernel_size == 0 || y_kernel_size == 0 {
        return;
    }
    let mut dst_image = BlurImageMut::alloc(width as u32, height as u32, channels);
    let src_image = BlurImage::alloc(width as u32, height as u32, channels);

    libblur::box_blur(
        &src_image,
        &mut dst_image,
        BoxBlurParameters {
            x_axis_kernel: x_kernel_size as u32,
            y_axis_kernel: y_kernel_size as u32,
        },
        threading,
    )
    .unwrap();
}
