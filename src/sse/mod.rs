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
#![deny(unreachable_pub)]
#[cfg(feature = "sse")]
mod fast_gaussian;
#[cfg(all(feature = "sse", feature = "nightly_f16"))]
mod fast_gaussian_f16;
#[cfg(feature = "sse")]
mod fast_gaussian_f32;
#[cfg(feature = "sse")]
mod fast_gaussian_next;
#[cfg(all(feature = "sse", feature = "nightly_f16"))]
mod fast_gaussian_next_f16;
#[cfg(feature = "sse")]
mod fast_gaussian_next_f32;
#[cfg(feature = "sse")]
mod fast_gaussian_next_u16;
#[cfg(feature = "sse")]
mod fast_gaussian_u16;
mod packing;
pub(crate) mod utils;
mod v_load_store;

#[cfg(feature = "sse")]
pub(crate) use fast_gaussian::{fg_horizontal_pass_sse_u8, fg_vertical_pass_sse_u8};
#[cfg(all(feature = "sse", feature = "nightly_f16"))]
pub(crate) use fast_gaussian_f16::{fg_horizontal_pass_sse_f16, fg_vertical_pass_sse_f16};
#[cfg(feature = "sse")]
pub(crate) use fast_gaussian_f32::{fg_horizontal_pass_sse_f32, fg_vertical_pass_sse_f32};
#[cfg(feature = "sse")]
pub(crate) use fast_gaussian_next::{fgn_horizontal_pass_sse_u8, fgn_vertical_pass_sse_u8};
#[cfg(all(feature = "sse", feature = "nightly_f16"))]
pub(crate) use fast_gaussian_next_f16::{
    fast_gaussian_next_horizontal_pass_sse_f16, fast_gaussian_next_vertical_pass_sse_f16,
};
#[cfg(feature = "sse")]
pub(crate) use fast_gaussian_next_f32::{fgn_horizontal_pass_sse_f32, fgn_vertical_pass_sse_f32};
#[cfg(feature = "sse")]
pub(crate) use fast_gaussian_next_u16::{fgn_horizontal_pass_sse_u16, fgn_vertical_pass_sse_u16};
#[cfg(feature = "sse")]
pub(crate) use fast_gaussian_u16::{fg_horizontal_pass_sse_u16, fg_vertical_pass_sse_u16};
pub(crate) use packing::*;
pub(crate) use utils::*;
#[allow(unused_imports)]
pub(crate) use v_load_store::*;
