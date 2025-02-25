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

use libblur::{EdgeMode, FastBlurChannels, ThreadingPolicy};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: (u8, u8, u8)| {
    fuzz_f32(
        data.0 as usize,
        data.1 as usize,
        data.2 as usize,
        FastBlurChannels::Channels4,
    );
    fuzz_f32(
        data.0 as usize,
        data.1 as usize,
        data.2 as usize,
        FastBlurChannels::Channels3,
    );
    fuzz_f32(
        data.0 as usize,
        data.1 as usize,
        data.2 as usize,
        FastBlurChannels::Plane,
    );
});

fn fuzz_f32(width: usize, height: usize, radius: usize, channels: FastBlurChannels) {
    if width == 0 || height == 0 || radius == 0 {
        return;
    }
    let src_image = vec![0.5f32; width * height * channels.channels()];
    let mut dst_image = vec![0f32; width * height * channels.channels()];

    libblur::gaussian_blur_f32(
        &src_image,
        &mut dst_image,
        width as u32,
        height as u32,
        radius as u32 * 2 + 1,
        0.,
        channels,
        EdgeMode::Clamp,
        ThreadingPolicy::Single,
    )
    .unwrap();
}
