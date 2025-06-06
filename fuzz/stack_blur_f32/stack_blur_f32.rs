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
use libblur::{stack_blur_f32, AnisotropicRadius, BlurImageMut, FastBlurChannels, ThreadingPolicy};
use libfuzzer_sys::fuzz_target;

#[derive(Clone, Debug, Arbitrary)]
pub struct SrcImage {
    pub src_width: u16,
    pub src_height: u16,
    pub value: u8,
    pub edge_mode: u8,
    pub x_radius: u8,
    pub y_radius: u8,
    pub threading: bool,
}

fuzz_target!(|data: SrcImage| {
    if data.src_width > 250 || data.src_height > 250 {
        return;
    }
    let mp = if data.threading {
        ThreadingPolicy::Adaptive
    } else {
        ThreadingPolicy::Single
    };
    fuzz_image(
        data.src_width as usize,
        data.src_height as usize,
        data.x_radius as usize,
        data.y_radius as usize,
        FastBlurChannels::Channels4,
        mp,
    );
    fuzz_image(
        data.src_width as usize,
        data.src_height as usize,
        data.x_radius as usize,
        data.y_radius as usize,
        FastBlurChannels::Channels3,
        mp,
    );
    fuzz_image(
        data.src_width as usize,
        data.src_height as usize,
        data.x_radius as usize,
        data.y_radius as usize,
        FastBlurChannels::Plane,
        mp,
    );
});

fn fuzz_image(
    width: usize,
    height: usize,
    x_radius: usize,
    y_radius: usize,
    channels: FastBlurChannels,
    threading_policy: ThreadingPolicy,
) {
    if width == 0 || height == 0 {
        return;
    }

    let mut dst_image = BlurImageMut::alloc(width as u32, height as u32, channels);

    stack_blur_f32(
        &mut dst_image,
        AnisotropicRadius::create(x_radius as u32, y_radius as u32),
        threading_policy,
    )
    .unwrap();
}
