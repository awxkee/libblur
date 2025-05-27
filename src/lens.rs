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
use crate::{BlurError, KernelShape};

/// Creates lens kernel
///
/// # Arguments
///
/// * `shape`: Kernel shape must be always odd.
/// * `n`: n is number of diaphragm blades
/// * `m`: is the concavity, aka the number of vertices on straight lines
/// * `k`: is the roundness vs. linearity factor; should be in [-1; 1]
pub fn lens_kernel(
    shape: KernelShape,
    n: f32,
    m: f32,
    k: f32,
    rotation: f32,
) -> Result<Vec<f32>, BlurError> {
    if shape.width == 0 || shape.height == 0 {
        return Err(BlurError::ZeroBaseSize);
    }
    if shape.width % 2 == 0 {
        return Err(BlurError::OddKernel(shape.width));
    }
    if shape.height % 2 == 0 {
        return Err(BlurError::OddKernel(shape.height));
    }
    assert!(k.abs() <= 1., "Roundness contract is not satisfied");
    let eps = 1f32 / shape.width as f32;
    let radius = (shape.width as f32 - 1.) / 2. - 1.;
    let mut new_buffer = vec![0f32; shape.width * shape.height];

    for (j, row) in new_buffer.chunks_exact_mut(shape.width).enumerate() {
        for (i, dst) in row.iter_mut().enumerate() {
            let x = (i as f32 - 1.) / radius - 1.;
            let y = (j as f32 - 1.) / radius - 1.;
            let r = x.hypot(y);
            let m = f32::cos((2f32 * f32::asin(k) + std::f32::consts::PI * m) / (2. * n))
                / f32::cos(
                    (2. * f32::asin(k * f32::cos(n * (f32::atan2(y, x) + rotation)))
                        + std::f32::consts::PI * m)
                        / (2. * n),
                );
            *dst = if m >= r + eps { 1. } else { 0. };
        }
    }
    let sum = new_buffer.iter().sum::<f32>();
    if sum != 0. {
        let recip = 1. / sum;
        new_buffer.iter_mut().for_each(|x| *x *= recip);
    }
    Ok(new_buffer)
}
