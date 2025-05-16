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

/// Computes sigma from kernel size
pub fn sigma_size(kernel_size: f32) -> f32 {
    let safe_kernel_size = if kernel_size <= 1. { 2. } else { kernel_size };
    0.3f32 * ((safe_kernel_size - 1.) * 0.5f32 - 1f32) + 0.8f32
}

/// Computes sigma from kernel size for `f64`
pub fn sigma_size_d(kernel_size: f64) -> f64 {
    let safe_kernel_size = if kernel_size <= 1. { 2. } else { kernel_size };
    0.3 * ((safe_kernel_size - 1.) * 0.5 - 1.) + 0.8
}

/// Computes kernel size from sigma
pub fn kernel_size(sigma: f32) -> u32 {
    assert_ne!(sigma, 0.8f32);
    let possible_size = (((((sigma - 0.8f32) / 0.3f32) + 1f32) * 2f32) + 1f32).max(3f32) as u32;
    if possible_size % 2 == 0 {
        return possible_size + 1;
    }
    possible_size
}

/// Computes kernel size from sigma
pub fn kernel_size_d(sigma: f64) -> u32 {
    assert_ne!(sigma, 0.8);
    let possible_size = (((((sigma - 0.8) / 0.3) + 1.) * 2.) + 1.).max(3.) as u32;
    if possible_size % 2 == 0 {
        return possible_size + 1;
    }
    possible_size
}
