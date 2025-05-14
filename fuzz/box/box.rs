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
use libblur::{BlurImage, BlurImageMut, FastBlurChannels, ThreadingPolicy};
use libfuzzer_sys::fuzz_target;

#[derive(Clone, Debug, Arbitrary)]
pub struct SrcImage {
    pub src_width: u16,
    pub src_height: u16,
    pub value: u16,
    pub edge_mode: u8,
    pub kernel_size: u8,
    pub plane: u8,
}

fuzz_target!(|data: SrcImage| {
    if data.src_width > 250 || data.src_height > 250 {
        return;
    }
    if data.kernel_size == 0 || data.kernel_size % 2 == 0 {
        return;
    }
    fuzz_8bit(
        data.src_width as usize,
        data.src_height as usize,
        data.kernel_size as usize,
        FastBlurChannels::Channels4,
    );
    fuzz_8bit(
        data.src_width as usize,
        data.src_height as usize,
        data.kernel_size as usize,
        FastBlurChannels::Channels3,
    );
    fuzz_8bit(
        data.src_width as usize,
        data.src_height as usize,
        data.kernel_size as usize,
        FastBlurChannels::Plane,
    );
});

fn fuzz_8bit(width: usize, height: usize, kernel_size: usize, channels: FastBlurChannels) {
    if width == 0 || height == 0 || kernel_size == 0 {
        return;
    }
    let mut dst_image = BlurImageMut::alloc(width as u32, height as u32, channels);
    let src_image = BlurImage::alloc(width as u32, height as u32, channels);

    libblur::box_blur(
        &src_image,
        &mut dst_image,
        kernel_size as u32,
        ThreadingPolicy::Single,
    )
    .unwrap();
}
