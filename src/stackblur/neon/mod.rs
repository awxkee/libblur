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
mod horizontal;
#[cfg(feature = "nightly_f16")]
mod horizontal_f16;
mod horizontal_f32;
#[cfg(feature = "rdm")]
mod horizontal_q0_31;
mod vertical;
#[cfg(feature = "nightly_f16")]
mod vertical_f16;
mod vertical_f32;
#[cfg(feature = "rdm")]
mod vertical_q0_31;

pub(crate) use horizontal::HorizontalNeonStackBlurPass;
#[cfg(feature = "nightly_f16")]
pub(crate) use horizontal_f16::HorizontalNeonStackBlurPassFloat16;
pub(crate) use horizontal_f32::HorizontalNeonStackBlurPassFloat32;
#[cfg(feature = "rdm")]
pub(crate) use horizontal_q0_31::HorizontalNeonStackBlurPassQ0_31;
pub(crate) use vertical::VerticalNeonStackBlurPass;
#[cfg(feature = "nightly_f16")]
pub(crate) use vertical_f16::VerticalNeonStackBlurPassFloat16;
pub(crate) use vertical_f32::VerticalNeonStackBlurPassFloat32;
#[cfg(feature = "rdm")]
pub(crate) use vertical_q0_31::VerticalNeonStackBlurPassQ0_31;
