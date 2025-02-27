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
mod convolve_op;
#[cfg(feature = "fft")]
mod fft_utils;
mod filter_2d;
#[cfg(feature = "fft")]
mod filter_2d_fft;
mod filter_2d_handler;
mod filter_2d_rgb;
#[cfg(feature = "fft")]
mod filter_2d_rgb_fft;
mod filter_2d_rgba;
#[cfg(feature = "fft")]
mod filter_2d_rgba_fft;
#[cfg(feature = "fft")]
mod gather_channel;
#[cfg(feature = "fft")]
mod mul_spectrum;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;
mod scan_point_2d;
mod scan_se_2d;
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
mod sse;

#[cfg(feature = "fft")]
pub use fft_utils::fft_next_good_size;
pub use filter_2d::{filter_2d, filter_2d_arbitrary};
#[cfg(feature = "fft")]
pub use filter_2d_fft::filter_2d_fft;
pub use filter_2d_rgb::filter_2d_rgb;
#[cfg(feature = "fft")]
pub use filter_2d_rgb_fft::filter_2d_rgb_fft;
pub use filter_2d_rgba::filter_2d_rgba;
#[cfg(feature = "fft")]
pub use filter_2d_rgba_fft::filter_2d_rgba_fft;
