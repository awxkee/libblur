// Copyright (c) Radzivon Bartoshyk. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1.  Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2.  Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3.  Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

mod box_blur;
mod fast_gaussian;
mod gaussian;
mod median_blur;
mod unsafe_slice;
mod channels_configuration;
mod fast_gaussian_next;
mod gaussian_neon;
mod box_blur_neon;
mod fast_gaussian_neon;
mod fast_gaussian_next_neon;

pub use box_blur::tent_blur;
pub use box_blur::tent_blur_u16;
pub use box_blur::box_blur;
pub use box_blur::box_blur_u16;
pub use box_blur::gaussian_box_blur;
pub use box_blur::gaussian_box_blur_u16;
pub use fast_gaussian_next::fast_gaussian_next;
pub use fast_gaussian_next::fast_gaussian_next_u16;
pub use median_blur::median_blur;
pub use gaussian::gaussian_blur;
pub use gaussian::gaussian_blur_u16;
pub use fast_gaussian::fast_gaussian;
pub use fast_gaussian::fast_gaussian_u16;
pub use channels_configuration::FastBlurChannels;