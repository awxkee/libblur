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

mod fast_gaussian;
#[cfg(target_feature = "f16c")]
mod fast_gaussian_f16;
mod fast_gaussian_f32;
mod fast_gaussian_next;
#[cfg(target_feature = "f16c")]
mod fast_gaussian_next_f16;
mod fast_gaussian_next_f32;
#[cfg(target_feature = "f16c")]
mod stack_blur_f16;
mod stack_blur_f32;
mod stack_blur_i32;
mod stack_blur_i64;
mod utils;

pub use fast_gaussian::*;
#[cfg(target_feature = "f16c")]
pub use fast_gaussian_f16::{
    fast_gaussian_horizontal_pass_sse_f16, fast_gaussian_vertical_pass_sse_f16,
};
pub use fast_gaussian_f32::{
    fast_gaussian_horizontal_pass_sse_f32, fast_gaussian_vertical_pass_sse_f32,
};
pub use fast_gaussian_next::*;
#[cfg(target_feature = "f16c")]
pub use fast_gaussian_next_f16::{
    fast_gaussian_next_horizontal_pass_sse_f16, fast_gaussian_next_vertical_pass_sse_f16,
};
pub use fast_gaussian_next_f32::{
    fast_gaussian_next_horizontal_pass_sse_f32, fast_gaussian_next_vertical_pass_sse_f32,
};
#[cfg(target_feature = "f16c")]
pub use stack_blur_f16::stack_blur_pass_sse_f16;
pub use stack_blur_f32::stack_blur_pass_sse_f;
pub use stack_blur_i32::stack_blur_pass_sse;
pub use stack_blur_i64::stack_blur_pass_sse_i64;
pub use utils::*;
