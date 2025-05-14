/*
 * // Copyright (c) Radzivon Bartoshyk 5/2025. All rights reserved.
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
mod rgba16;
mod rgba8;
#[cfg(feature = "rdm")]
mod rgba8_q0_31;
mod rgba_f32;
mod vsum16;
mod vsum8;
mod vsum_f32;

pub(crate) use rgba16::{box_blur_horizontal_pass_neon_rgba16, box_blur_vertical_pass_neon_rgba16};
pub(crate) use rgba8::{box_blur_horizontal_pass_neon, box_blur_vertical_pass_neon};
#[cfg(feature = "rdm")]
pub(crate) use rgba8_q0_31::{box_blur_horizontal_pass_neon_rdm, box_blur_vertical_pass_neon_rdm};
pub(crate) use rgba_f32::{
    box_blur_horizontal_pass_neon_rgba_f32, box_blur_vertical_pass_neon_rgba_f32,
};
pub(crate) use vsum16::neon_ring_vertical_row_summ16;
pub(crate) use vsum8::neon_ring_vertical_row_summ;
pub(crate) use vsum_f32::neon_ring_vertical_row_summ_f32;
