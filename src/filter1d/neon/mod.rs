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
#![allow(clippy::manual_clamp)]
mod filter_column;
mod filter_column_approx;
#[cfg(feature = "rdm")]
mod filter_column_approx_rdm;
mod filter_column_f32;
mod filter_column_symm;
mod filter_column_symm_approx;
#[cfg(feature = "rdm")]
mod filter_column_symm_approx_rdm;
#[cfg(feature = "rdm")]
mod filter_column_symm_approx_rdm_x3;
mod filter_column_symm_approx_uq0_7;
mod filter_column_symm_f32;
mod filter_column_symm_f32_f64;
mod filter_column_symm_u15_u16_x3;
mod filter_column_symm_u16;
mod filter_column_symm_u16_x3;
mod filter_column_symm_u8_i16;
mod filter_column_symm_uq15_u16;
mod filter_column_symm_x3;
mod filter_column_u8_i16;
mod filter_row;
mod filter_row_approx;
#[cfg(feature = "rdm")]
mod filter_row_approx_rdm;
mod filter_row_f32;
mod filter_row_symm;
mod filter_row_symm_approx;
#[cfg(feature = "rdm")]
mod filter_row_symm_approx_rdm;
mod filter_row_symm_f32;
mod filter_row_symm_f32_f64;
mod filter_row_symm_u16;
mod filter_row_symm_u8_i16;
mod filter_row_symm_uq15_u16;
mod filter_row_u8_i16;
mod rom_symm_approx5;
#[cfg(feature = "rdm")]
mod row_symm_app_binter_rdm;
mod row_symm_approx_binter;
mod row_symm_approx_binter_uq0_7;
pub mod utils;

pub(crate) use filter_column::filter_column_neon_u8_f32;
pub(crate) use filter_column_approx::filter_column_neon_u8_i32_app;
#[cfg(feature = "rdm")]
pub(crate) use filter_column_approx_rdm::filter_column_neon_u8_i32_i16_qrdm_app;
pub(crate) use filter_column_f32::filter_column_neon_f32_f32;
pub(crate) use filter_column_symm::filter_symm_column_neon_u8_f32;
pub(crate) use filter_column_symm_approx::filter_column_symm_neon_u8_i32_app;
#[cfg(feature = "rdm")]
pub(crate) use filter_column_symm_approx_rdm::filter_column_symm_neon_u8_i32_rdm;
#[cfg(feature = "rdm")]
pub(crate) use filter_column_symm_approx_rdm_x3::filter_column_symm_neon_u8_i32_rdm_x3;
pub(crate) use filter_column_symm_f32::filter_column_neon_symm_f32_f32;
pub(crate) use filter_column_symm_f32_f64::filter_column_neon_symm_f32_f64;
pub(crate) use filter_column_symm_u15_u16_x3::filter_symm_column_neon_uq15_u16_x3;
pub(crate) use filter_column_symm_u16::filter_symm_column_neon_u16_f32;
pub(crate) use filter_column_symm_u16_x3::filter_symm_column_neon_u16_f32_x3;
pub(crate) use filter_column_symm_u8_i16::filter_column_symm_neon_u8_i16;
pub(crate) use filter_column_symm_uq15_u16::filter_column_symm_neon_uq15_u16;
pub(crate) use filter_column_symm_x3::filter_symm_column_neon_u8_f32_x3;
pub(crate) use filter_column_u8_i16::filter_column_neon_u8_i16;
pub(crate) use filter_row::filter_row_neon_u8_f32;
pub(crate) use filter_row_approx::filter_row_neon_u8_i32_app;
#[cfg(feature = "rdm")]
pub(crate) use filter_row_approx_rdm::filter_row_neon_u8_i32_rdm;
pub(crate) use filter_row_f32::filter_row_neon_f32_f32;
pub(crate) use filter_row_symm::filter_row_symm_neon_u8_f32;
pub(crate) use filter_row_symm_approx::filter_row_symm_neon_u8_i32;
#[cfg(feature = "rdm")]
pub(crate) use filter_row_symm_approx_rdm::filter_row_symm_neon_u8_i32_rdm;
pub(crate) use filter_row_symm_f32::filter_row_neon_symm_f32_f32;
pub(crate) use filter_row_symm_f32_f64::filter_row_neon_symm_f32_f64;
pub(crate) use filter_row_symm_u16::filter_row_symm_neon_u16_f32;
pub(crate) use filter_row_symm_u8_i16::filter_rgb_row_symm_neon_u8_i16;
pub(crate) use filter_row_symm_uq15_u16::filter_row_symm_neon_uq15_u16;
pub(crate) use filter_row_u8_i16::filter_rgb_row_neon_u8_i16;
pub(crate) use rom_symm_approx5::filter_row_symm_neon_binter_u8_uq0_7_x5;
#[cfg(feature = "rdm")]
pub(crate) use row_symm_app_binter_rdm::filter_row_symm_neon_binter_u8_i32_rdm;
pub(crate) use row_symm_approx_binter::filter_row_symm_neon_binter_u8_i32;
pub(crate) use row_symm_approx_binter_uq0_7::filter_row_symm_neon_binter_u8_u0_7;
