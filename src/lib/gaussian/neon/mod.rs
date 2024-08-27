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

mod filter_u8;
mod filter_vertical_f32;
mod horiz_four_channel;
mod horiz_four_channel_f32;
mod horiz_one_approx;
mod horiz_one_channel_f32;
mod horiz_one_channel_u8;
mod horiz_rgba_approx;
mod horiz_rgba_filter_approx;
mod vert_four_channel;
mod vertical_approx_u8;
mod vertical_f32;
mod vertical_filter_approx_u8;

pub use filter_u8::gaussian_blur_horizontal_pass_filter_neon;
pub use filter_u8::gaussian_blur_vertical_pass_filter_neon;
pub use filter_vertical_f32::gaussian_blur_vertical_pass_filter_f32_neon;
pub use horiz_four_channel::*;
pub use horiz_four_channel_f32::gaussian_horiz_t_f_chan_f32;
pub use horiz_four_channel_f32::gaussian_horiz_t_f_chan_filter_f32;
pub use horiz_one_approx::{gaussian_horiz_one_approx_u8, gaussian_horiz_one_chan_filter_approx};
pub use horiz_one_channel_f32::*;
pub use horiz_one_channel_u8::*;
pub use horiz_rgba_approx::gaussian_blur_horizontal_pass_approx_neon;
pub use horiz_rgba_filter_approx::gaussian_blur_horizontal_pass_filter_approx_neon;
pub use vert_four_channel::*;
pub use vertical_approx_u8::gaussian_blur_vertical_approx_neon;
pub use vertical_f32::gaussian_blur_vertical_pass_f32_neon;
pub use vertical_filter_approx_u8::gaussian_blur_vertical_pass_filter_approx_neon;
