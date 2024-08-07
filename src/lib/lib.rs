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

mod r#box;
mod channels_configuration;
mod edge_mode;
mod fast_gaussian;
mod fast_gaussian_next;
mod fast_gaussian_superior;
mod gaussian;
mod median_blur;
mod mul_table;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
mod sse;
mod stack_blur;
mod stack_blur_f16;
mod stack_blur_f32;
mod stack_blur_linear;
mod threading_policy;
mod to_storage;
mod unsafe_slice;

pub use channels_configuration::FastBlurChannels;
pub use colorutils_rs::TransferFunction;
pub use edge_mode::*;
pub use fast_gaussian::fast_gaussian;
pub use fast_gaussian::fast_gaussian_f16;
pub use fast_gaussian::fast_gaussian_f32;
pub use fast_gaussian::fast_gaussian_in_linear;
pub use fast_gaussian::fast_gaussian_plane;
pub use fast_gaussian::fast_gaussian_plane_f32;
pub use fast_gaussian::fast_gaussian_u16;
pub use fast_gaussian_next::fast_gaussian_next;
pub use fast_gaussian_next::fast_gaussian_next_f16;
pub use fast_gaussian_next::fast_gaussian_next_f32;
pub use fast_gaussian_next::fast_gaussian_next_in_linear;
pub use fast_gaussian_next::fast_gaussian_next_u16;
pub use fast_gaussian_superior::fast_gaussian_superior;
pub use gaussian::gaussian_blur;
pub use gaussian::gaussian_blur_f16;
pub use gaussian::gaussian_blur_f32;
pub use gaussian::gaussian_blur_in_linear;
pub use gaussian::gaussian_blur_u16;
pub use median_blur::median_blur;
pub use r#box::box_blur;
pub use r#box::box_blur_f32;
pub use r#box::box_blur_in_linear;
pub use r#box::box_blur_u16;
pub use r#box::gaussian_box_blur;
pub use r#box::gaussian_box_blur_f32;
pub use r#box::gaussian_box_blur_in_linear;
pub use r#box::gaussian_box_blur_u16;
pub use r#box::tent_blur;
pub use r#box::tent_blur_f32;
pub use r#box::tent_blur_in_linear;
pub use r#box::tent_blur_u16;
pub use stack_blur::stack_blur;
pub use stack_blur_f16::stack_blur_f16;
pub use stack_blur_f32::stack_blur_f32;
pub use stack_blur_linear::stack_blur_in_linear;
pub use threading_policy::ThreadingPolicy;
