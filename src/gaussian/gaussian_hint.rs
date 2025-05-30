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

/// Declares preferred convolution precision mode for integer storage types.
#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Default)]
pub enum ConvolutionMode {
    /// Exact precision, f32 accumulator and weights will be used.
    Exact = 0,
    /// Convolution in numerical approximation,
    /// this is faster than exact convolution but may change result.
    ///
    /// Estimated error not less than 1-2%.
    #[default]
    FixedPoint = 1,
}

/// Specifies the preferred convolution precision mode for IEEE 754 binary32 (`f32`) data.
#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Default)]
pub enum IeeeBinaryConvolutionMode {
    /// Exact precision, f32 accumulator and weights will be used.
    Normal = 0,
    /// High precision using `f64` for intermediate accumulation.
    /// This significantly reduces numerical error in convolution results.
    #[default]
    Zealous = 1,
}
