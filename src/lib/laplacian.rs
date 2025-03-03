/*
 * // Copyright (c) Radzivon Bartoshyk. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without
 * modification, // are permitted provided that the following conditions are
 * met: //
 * // 1.  Redistributions of source code must retain the above copyright
 * notice, this // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright
 * notice, // this list of conditions and the following disclaimer in the
 * documentation // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE // FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE // OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
use crate::{
    filter_2d, sigma_size, BlurError, BlurImage, BlurImageMut, EdgeMode, KernelShape, Scalar,
    ThreadingPolicy,
};

pub fn laplacian_kernel(size: usize) -> Vec<f32> {
    if size & 1 == 0 {
        panic!("Kernel size must be odd in 'get_laplacian_kernel'!");
    }
    let center_x = (size / 2) as f32;
    let center_y = (size / 2) as f32;
    let sigma = sigma_size(size as f32);

    let mut kernel = vec![f32::default(); size * size];

    let sigma_p_2 = sigma * sigma;

    let scale = -1. / (std::f32::consts::PI * sigma_p_2 * sigma_p_2);

    let mut sum = 0f32;

    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - center_x;
            let dy = y as f32 - center_y;
            let der = -((dx * dx + dy * dy) / (2. * sigma_p_2));
            let v = scale * (1. + der) * f32::exp(der);
            kernel[y * size + x] = v;
            sum += v;
        }
    }

    if sum != 0. {
        let scale = 1. / sum;

        for item in kernel.iter_mut() {
            *item *= scale;
        }
    }
    kernel
}

/// Performs laplacian of gaussian on the image
///
/// # Arguments
///
/// * `image`: Source image.
/// * `destination`: Destination image.
/// * `border_mode`: See [EdgeMode] for more info
/// * `border_constant`: If [EdgeMode::Constant] border will be replaced with
///   this provided [Scalar] value
/// * `threading_policy`: see [ThreadingPolicy] for more info
///
/// returns: ()
pub fn laplacian(
    image: &BlurImage<u8>,
    destination: &mut BlurImageMut<u8>,
    border_mode: EdgeMode,
    border_constant: Scalar,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    image.check_layout()?;
    destination.check_layout(Some(image))?;
    image.size_matches_mut(destination)?;
    let kernel = [-1, -1, -1, -1, 8, -1, -1, -1, -1];
    filter_2d::<u8, i16>(
        image,
        destination,
        &kernel,
        KernelShape::new(3, 3),
        border_mode,
        border_constant,
        threading_policy,
    )
}
