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
#![allow(clippy::type_complexity)]
mod arena;
mod arena_roi;
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "avx"))]
mod avx;
mod color_group;
mod filter;
mod filter_1d_approx;
mod filter_1d_column_handler;
mod filter_1d_column_handler_approx;
mod filter_1d_row_handler;
mod filter_1d_row_handler_approx;
mod filter_column;
mod filter_column_approx;
mod filter_column_approx_symmetric;
mod filter_column_symmetric;
mod filter_element;
mod filter_row;
mod filter_row_approx;
mod filter_row_symmetric;
mod filter_row_symmetric_approx;
mod filter_scan;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(crate) mod neon;
mod region;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub(crate) mod sse;
mod to_approx_storage;

pub use arena::{make_arena, Arena, ArenaPads};
pub use filter::filter_1d_exact;
pub use filter_1d_approx::filter_1d_approx;
pub use filter_element::KernelShape;
