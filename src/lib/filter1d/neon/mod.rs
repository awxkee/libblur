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
mod filter_column;
mod filter_column_approx;
mod filter_column_approx_rdm;
mod filter_column_f32;
mod filter_column_symm;
mod filter_column_symm_approx;
mod filter_column_symm_approx_rdm;
mod filter_column_symm_f32;
mod filter_column_symm_u8_i16;
mod filter_column_u8_i16;
mod filter_rgb_row_symm_approx;
mod filter_rgb_row_symm_u8_i16;
mod filter_rgb_row_u8_i16;
mod filter_rgba_row_symm_approx;
mod filter_row_symm_approx_rdm;
mod filter_row_symm_f32;
mod filter_symm_row;
mod filter_row;
mod filter_row_approx;
mod filter_row_approx_rdm;
mod filter_row_f32;
pub mod utils;

pub use filter_column::filter_column_neon_u8_f32;
pub use filter_column_approx::filter_column_neon_u8_i32_app;
pub(crate) use filter_column_approx_rdm::filter_column_neon_u8_i32_i16_qrdm_app;
pub use filter_column_f32::filter_column_neon_f32_f32;
pub use filter_column_symm::filter_symm_column_neon_u8_f32;
pub use filter_column_symm_approx::filter_column_symm_neon_u8_i32_app;
pub(crate) use filter_column_symm_approx_rdm::filter_column_symm_neon_u8_i32_rdm;
pub use filter_column_symm_f32::filter_column_neon_symm_f32_f32;
pub use filter_column_symm_u8_i16::filter_column_symm_neon_u8_i16;
pub use filter_column_u8_i16::filter_column_neon_u8_i16;
pub use filter_rgb_row_symm_approx::filter_rgb_row_symm_neon_u8_i32;
pub use filter_rgb_row_symm_u8_i16::filter_rgb_row_symm_neon_u8_i16;
pub use filter_rgb_row_u8_i16::filter_rgb_row_neon_u8_i16;
pub use filter_rgba_row_symm_approx::filter_rgba_row_symm_neon_u8_i32;
pub(crate) use filter_row_symm_approx_rdm::filter_row_symm_neon_u8_i32_rdm;
pub use filter_row_symm_f32::filter_row_neon_symm_f32_f32;
pub use filter_symm_row::filter_row_symm_neon_u8_f32;
pub use filter_row::filter_row_neon_u8_f32;
pub use filter_row_approx::filter_row_neon_u8_i32_app;
pub(crate) use filter_row_approx_rdm::filter_row_neon_u8_i32_rdm;
pub use filter_row_f32::filter_row_neon_f32_f32;
