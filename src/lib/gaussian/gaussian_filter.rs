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

#[derive(Clone)]
pub(crate) struct GaussianFilter {
    pub start: usize,
    pub size: usize,
    pub filter: Vec<f32>,
}

impl GaussianFilter {
    pub(crate) fn new(start: usize, size: usize, filter: Vec<f32>) -> GaussianFilter {
        GaussianFilter {
            start,
            size,
            filter,
        }
    }
}

pub(crate) fn create_filter(length: usize, kernel_size: u32, sigma: f32) -> Vec<GaussianFilter> {
    let mut filter: Vec<GaussianFilter> = vec![GaussianFilter::new(0, 0, vec![]); length];
    let filter_radius = (kernel_size / 2) as usize;

    let filter_scale = 1f32 / (f32::sqrt(2f32 * std::f32::consts::PI) * sigma);
    for x in 0..length {
        let start = (x as i64 - filter_radius as i64).max(0) as usize;
        let end = (x + filter_radius).min(length - 1);
        let size = end - start;

        let mut real_filter = vec![];
        let mut filter_sum = 0f32;
        for j in start..end {
            let new_weight =
                f32::exp(-0.5f32 * f32::powf((j as f32 - x as f32) / sigma, 2.0f32)) * filter_scale;
            filter_sum += new_weight;
            real_filter.push(new_weight);
        }

        if filter_sum != 0f32 {
            let scale = 1f32 / filter_sum;
            let new_filter = real_filter.iter().map(|&x| x * scale).collect();
            real_filter = new_filter;
        }
        unsafe {
            *filter.get_unchecked_mut(x) = GaussianFilter::new(start, size, real_filter);
        }
    }
    filter
}
