// Copyright (c) Radzivon Bartoshyk. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1.  Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2.  Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3.  Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use crate::gaussian::gaussian_approx::PRECISION;

pub(crate) fn get_gaussian_kernel_1d(width: u32, sigma: f32) -> Vec<f32> {
    let mut sum_norm: f32 = 0f32;
    let mut kernel: Vec<f32> = Vec::with_capacity(width as usize);
    let scale = 1f32 / (f32::sqrt(2f32 * std::f32::consts::PI) * sigma);
    let mean = (width / 2) as f32;

    for x in 0..width {
        let new_weight = f32::exp(-0.5f32 * f32::powf((x as f32 - mean) / sigma, 2.0f32)) * scale;
        kernel.push(new_weight);
        sum_norm += new_weight;
    }

    if sum_norm != 0f32 {
        let sum_scale = 1f32 / sum_norm;
        for x in 0..width as usize {
            unsafe {
                *kernel.get_unchecked_mut(x) = (*kernel.get_unchecked(x)) * sum_scale;
            }
        }
    }

    kernel
}

pub(crate) fn get_gaussian_kernel_1d_integral(width: u32, sigma: f32) -> Vec<i16> {
    let mut sum_norm: f32 = 0f32;
    let mut kernel: Vec<f32> = Vec::with_capacity(width as usize);
    let scale = 1f32 / (f32::sqrt(2f32 * std::f32::consts::PI) * sigma);
    let mean = (width / 2) as f32;

    for x in 0..width {
        let new_weight = f32::exp(-0.5f32 * f32::powf((x as f32 - mean) / sigma, 2.0f32)) * scale;
        kernel.push(new_weight);
        sum_norm += new_weight;
    }

    if sum_norm != 0f32 {
        let sum_scale = 1f32 / sum_norm;
        for x in 0..width as usize {
            unsafe {
                *kernel.get_unchecked_mut(x) = (*kernel.get_unchecked(x)) * sum_scale;
            }
        }
    }

    let precision_scale = (1 << PRECISION) as f32;

    kernel
        .iter()
        .map(|&x| (x * precision_scale) as i16)
        .collect()
}
