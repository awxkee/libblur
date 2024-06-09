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

pub(crate) mod fast_gaussian_next_f16_impl {
    use crate::unsafe_slice::UnsafeSlice;
    use crate::FastBlurChannels;

    pub(crate) fn fast_gaussian_next_vertical_pass_f16(
        bytes: &UnsafeSlice<u16>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        start: u32,
        end: u32,
        channels: FastBlurChannels,
    ) {
        let mut buffer_r: [f32; 1024] = [0f32; 1024];
        let mut buffer_g: [f32; 1024] = [0f32; 1024];
        let mut buffer_b: [f32; 1024] = [0f32; 1024];
        let radius_64 = radius as i64;
        let height_wide = height as i64;
        let weight = 1.0f32 / ((radius as f32) * (radius as f32) * (radius as f32));
        let channels_count = match channels {
            FastBlurChannels::Channels3 => 3,
            FastBlurChannels::Channels4 => 4,
        };
        for x in start..std::cmp::min(width, end) {
            let mut dif_r: f32 = 0f32;
            let mut der_r: f32 = 0f32;
            let mut sum_r: f32 = 0f32;
            let mut dif_g: f32 = 0f32;
            let mut der_g: f32 = 0f32;
            let mut sum_g: f32 = 0f32;
            let mut dif_b: f32 = 0f32;
            let mut der_b: f32 = 0f32;
            let mut sum_b: f32 = 0f32;

            let current_px = (x * channels_count) as usize;

            let start_y = 0 - 3 * radius as i64;
            for y in start_y..height_wide {
                let current_y = (y * (stride as i64)) as usize;
                if y >= 0 {
                    let new_r = half::f16::from_f32(sum_r * weight).to_bits();
                    let new_g = half::f16::from_f32(sum_g * weight).to_bits();
                    let new_b = half::f16::from_f32(sum_b * weight).to_bits();

                    unsafe {
                        bytes.write(current_y + current_px, new_r);
                        bytes.write(current_y + current_px + 1, new_g);
                        bytes.write(current_y + current_px + 2, new_b);
                    }

                    let d_arr_index_1 = ((y + radius_64) & 1023) as usize;
                    let d_arr_index_2 = ((y - radius_64) & 1023) as usize;
                    let d_arr_index = (y & 1023) as usize;
                    dif_r += 3f32 * (buffer_r[d_arr_index] - buffer_r[d_arr_index_1])
                        - buffer_r[d_arr_index_2];
                    dif_g += 3f32 * (buffer_g[d_arr_index] - buffer_g[d_arr_index_1])
                        - buffer_g[d_arr_index_2];
                    dif_b += 3f32 * (buffer_b[d_arr_index] - buffer_b[d_arr_index_1])
                        - buffer_b[d_arr_index_2];
                } else if y + radius_64 >= 0 {
                    let arr_index = (y & 1023) as usize;
                    let arr_index_1 = ((y + radius_64) & 1023) as usize;
                    dif_r += 3f32 * (buffer_r[arr_index] - buffer_r[arr_index_1]);
                    dif_g += 3f32 * (buffer_g[arr_index] - buffer_g[arr_index_1]);
                    dif_b += 3f32 * (buffer_b[arr_index] - buffer_b[arr_index_1]);
                } else if y + 2 * radius_64 >= 0 {
                    let arr_index = ((y + radius_64) & 1023) as usize;
                    dif_r -= 3f32 * buffer_r[arr_index];
                    dif_g -= 3f32 * buffer_g[arr_index];
                    dif_b -= 3f32 * buffer_b[arr_index];
                }

                let next_row_y = (std::cmp::min(
                    std::cmp::max(y + ((3 * radius_64) >> 1), 0),
                    height_wide - 1,
                ) as usize)
                    * (stride as usize);
                let next_row_x = (x * channels_count) as usize;

                let px_idx = next_row_y + next_row_x;

                let rf32 = half::f16::from_bits(bytes[px_idx]).to_f32();
                let gf32 = half::f16::from_bits(bytes[px_idx + 1]).to_f32();
                let bf32 = half::f16::from_bits(bytes[px_idx + 2]).to_f32();

                let arr_index = ((y + 2 * radius_64) & 1023) as usize;

                dif_r += rf32;
                der_r += dif_r;
                sum_r += der_r;
                buffer_r[arr_index] = rf32;

                dif_g += gf32;
                der_g += dif_g;
                sum_g += der_g;
                buffer_g[arr_index] = gf32;

                dif_b += bf32;
                der_b += dif_b;
                sum_b += der_b;
                buffer_b[arr_index] = bf32;
            }
        }
    }

    pub(crate) fn fast_gaussian_next_horizontal_pass_f16(
        bytes: &UnsafeSlice<u16>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        start: u32,
        end: u32,
        channels: FastBlurChannels,
    ) {
        let mut buffer_r: [f32; 1024] = [0f32; 1024];
        let mut buffer_g: [f32; 1024] = [0f32; 1024];
        let mut buffer_b: [f32; 1024] = [0f32; 1024];
        let radius_64 = radius as i64;
        let width_wide = width as i64;
        let weight = 1.0f32 / ((radius as f32) * (radius as f32) * (radius as f32));
        let channels_count = match channels {
            FastBlurChannels::Channels3 => 3,
            FastBlurChannels::Channels4 => 4,
        };
        for y in start..std::cmp::min(height, end) {
            let mut dif_r: f32 = 0f32;
            let mut der_r: f32 = 0f32;
            let mut sum_r: f32 = 0f32;
            let mut dif_g: f32 = 0f32;
            let mut der_g: f32 = 0f32;
            let mut sum_g: f32 = 0f32;
            let mut dif_b: f32 = 0f32;
            let mut der_b: f32 = 0f32;
            let mut sum_b: f32 = 0f32;

            let current_y = ((y as i64) * (stride as i64)) as usize;

            for x in (0 - 3 * radius_64)..(width as i64) {
                if x >= 0 {
                    let current_px = ((std::cmp::max(x, 0) as u32) * channels_count) as usize;
                    let new_r = half::f16::from_f32(sum_r * weight).to_bits();
                    let new_g = half::f16::from_f32(sum_g * weight).to_bits();
                    let new_b = half::f16::from_f32(sum_b * weight).to_bits();

                    unsafe {
                        bytes.write(current_y + current_px, new_r);
                        bytes.write(current_y + current_px + 1, new_g);
                        bytes.write(current_y + current_px + 2, new_b);
                    }

                    let d_arr_index_1 = ((x + radius_64) & 1023) as usize;
                    let d_arr_index_2 = ((x - radius_64) & 1023) as usize;
                    let d_arr_index = (x & 1023) as usize;
                    dif_r += 3f32 * (buffer_r[d_arr_index] - buffer_r[d_arr_index_1])
                        - buffer_r[d_arr_index_2];
                    dif_g += 3f32 * (buffer_g[d_arr_index] - buffer_g[d_arr_index_1])
                        - buffer_g[d_arr_index_2];
                    dif_b += 3f32 * (buffer_b[d_arr_index] - buffer_b[d_arr_index_1])
                        - buffer_b[d_arr_index_2];
                } else if x + radius_64 >= 0 {
                    let arr_index = (x & 1023) as usize;
                    let arr_index_1 = ((x + radius_64) & 1023) as usize;
                    dif_r += 3f32 * (buffer_r[arr_index] - buffer_r[arr_index_1]);
                    dif_g += 3f32 * (buffer_g[arr_index] - buffer_g[arr_index_1]);
                    dif_b += 3f32 * (buffer_b[arr_index] - buffer_b[arr_index_1]);
                } else if x + 2 * radius_64 >= 0 {
                    let arr_index = ((x + radius_64) & 1023) as usize;
                    dif_r -= 3f32 * buffer_r[arr_index];
                    dif_g -= 3f32 * buffer_g[arr_index];
                    dif_b -= 3f32 * buffer_b[arr_index];
                }

                let next_row_y = (y as usize) * (stride as usize);
                let next_row_x =
                    ((std::cmp::min(std::cmp::max(x + 3 * radius_64 / 2, 0), width_wide - 1)
                        as u32)
                        * channels_count) as usize;

                let rf32 = half::f16::from_bits(bytes[next_row_y + next_row_x]).to_f32();
                let gf32 = half::f16::from_bits(bytes[next_row_y + next_row_x + 1]).to_f32();
                let bf32 = half::f16::from_bits(bytes[next_row_y + next_row_x + 2]).to_f32();

                let arr_index = ((x + 2 * radius_64) & 1023) as usize;

                dif_r += rf32;
                der_r += dif_r;
                sum_r += der_r;
                buffer_r[arr_index] = rf32;

                dif_g += gf32;
                der_g += dif_g;
                sum_g += der_g;
                buffer_g[arr_index] = gf32;

                dif_b += bf32;
                der_b += dif_b;
                sum_b += der_b;
                buffer_b[arr_index] = bf32;
            }
        }
    }
}

pub(crate) mod fast_gaussian_next_f16 {
    use crate::fast_gaussian_next_f16::fast_gaussian_next_f16_impl;
    use crate::unsafe_slice::UnsafeSlice;
    use crate::FastBlurChannels;

    pub(crate) fn fast_gaussian_next_impl_f16(
        bytes: &mut [u16],
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        channels: FastBlurChannels,
    ) {
        let unsafe_image = UnsafeSlice::new(bytes);
        let thread_count = std::cmp::max(std::cmp::min(width * height / (256 * 256), 12), 1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count as usize)
            .build()
            .unwrap();
        pool.scope(|scope| {
            let segment_size = width / thread_count;

            for i in 0..thread_count {
                let start_x = i * segment_size;
                let mut end_x = (i + 1) * segment_size;
                if i == thread_count - 1 {
                    end_x = width;
                }
                scope.spawn(move |_| {
                    fast_gaussian_next_f16_impl::fast_gaussian_next_vertical_pass_f16(
                        &unsafe_image,
                        stride,
                        width,
                        height,
                        radius,
                        start_x,
                        end_x,
                        channels,
                    );
                });
            }
        });
        pool.scope(|scope| {
            let segment_size = height / thread_count;

            for i in 0..thread_count {
                let start_y = i * segment_size;
                let mut end_y = (i + 1) * segment_size;
                if i == thread_count - 1 {
                    end_y = height;
                }

                scope.spawn(move |_| {
                    fast_gaussian_next_f16_impl::fast_gaussian_next_horizontal_pass_f16(
                        &unsafe_image,
                        stride,
                        width,
                        height,
                        radius,
                        start_y,
                        end_y,
                        channels,
                    );
                });
            }
        });
    }
}
