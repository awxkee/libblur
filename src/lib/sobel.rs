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
use crate::{
    filter_1d_exact, filter_1d_rgb_exact, filter_1d_rgba_exact, BlurError, EdgeMode,
    FastBlurChannels, ImageSize, Scalar, ThreadingPolicy,
};

/// Performs sobel operator on the image
///
/// # Arguments
///
/// * `image`: Source image
/// * `destination`: Destination image
/// * `image_size`: Image size, see [ImageSize]
/// * `border_mode`: See [EdgeMode] for more info
/// * `border_constant`: If [EdgeMode::Constant] border will be replaced with this provided [Scalar] value
/// * `channels`: see [FastBlurChannels] for more info
/// * `threading_policy`: see [ThreadingPolicy] for more info
///
/// returns: ()
///
pub fn sobel(
    image: &[u8],
    destination: &mut [u8],
    image_size: ImageSize,
    border_mode: EdgeMode,
    border_constant: Scalar,
    channels: FastBlurChannels,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    let sobel_horizontal: [i16; 3] = [-1, 0, 1];
    let sobel_vertical: [i16; 3] = [1, 2, 1];
    let _dispatcher = match channels {
        FastBlurChannels::Plane => filter_1d_exact::<u8, i16>,
        FastBlurChannels::Channels3 => filter_1d_rgb_exact::<u8, i16>,
        FastBlurChannels::Channels4 => filter_1d_rgba_exact::<u8, i16>,
    };
    _dispatcher(
        image,
        destination,
        image_size,
        &sobel_horizontal,
        &sobel_vertical,
        border_mode,
        border_constant,
        threading_policy,
    )
}
