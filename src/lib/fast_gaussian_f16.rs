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

pub(crate) mod fast_gaussian_f16_impl {
    use crate::unsafe_slice::UnsafeSlice;
    use crate::FastBlurChannels;

    pub(crate) fn fast_gaussian_vertical_pass_f16<const CHANNELS: usize>(
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
        let mut buffer_a: [f32; 1024] = [0f32; 1024];
        let radius_64 = radius as i64;
        let height_wide = height as i64;
        let weight = 1.0f32 / ((radius as f32) * (radius as f32));
        let channels_count = match channels {
            FastBlurChannels::Channels3 => 3,
            FastBlurChannels::Channels4 => 4,
        };
        for x in start..std::cmp::min(width, end) {
            let mut dif_r: f32 = 0f32;
            let mut sum_r: f32 = 0f32;
            let mut dif_g: f32 = 0f32;
            let mut sum_g: f32 = 0f32;
            let mut dif_b: f32 = 0f32;
            let mut sum_b: f32 = 0f32;
            let mut dif_a: f32 = 0f32;
            let mut sum_a: f32 = 0f32;

            let current_px = (x * channels_count) as usize;

            let start_y = 0 - 2 * radius as i64;
            for y in start_y..height_wide {
                let current_y = (y * (stride as i64)) as usize;
                if y >= 0 {
                    let new_r = half::f16::from_f32(sum_r * weight).to_bits();
                    let new_g = half::f16::from_f32(sum_g * weight).to_bits();
                    let new_b = half::f16::from_f32(sum_b * weight).to_bits();
                    let new_a = if CHANNELS == 4 {
                        half::f16::from_f32(sum_a * weight).to_bits()
                    } else {
                        0u16
                    };

                    unsafe {
                        let pixel_offset = current_y + current_px;
                        bytes.write(pixel_offset, new_r);
                        bytes.write(pixel_offset + 1, new_g);
                        bytes.write(pixel_offset + 2, new_b);
                        if CHANNELS == 4 {
                            bytes.write(pixel_offset + 3, new_a);
                        }
                    }

                    let arr_index = ((y - radius_64) & 1023) as usize;
                    let d_arr_index = (y & 1023) as usize;
                    unsafe {
                        dif_r += *buffer_r.get_unchecked(arr_index)
                            - 2f32 * (*buffer_r.get_unchecked(d_arr_index));
                        dif_g += *buffer_g.get_unchecked(arr_index)
                            - 2f32 * (*buffer_g.get_unchecked(d_arr_index));
                        dif_b += *buffer_b.get_unchecked(arr_index)
                            - 2f32 * (*buffer_b.get_unchecked(d_arr_index));
                        dif_a += *buffer_a.get_unchecked(arr_index)
                            - 2f32 * (*buffer_a.get_unchecked(d_arr_index));
                    }
                } else if y + radius_64 >= 0 {
                    let arr_index = (y & 1023) as usize;
                    unsafe {
                        dif_r -= 2f32 * (*buffer_r.get_unchecked(arr_index));
                        dif_g -= 2f32 * (*buffer_g.get_unchecked(arr_index));
                        dif_b -= 2f32 * (*buffer_b.get_unchecked(arr_index));
                        dif_a -= 2f32 * (*buffer_a.get_unchecked(arr_index));
                    }
                }

                let next_row_y = (std::cmp::min(std::cmp::max(y + radius_64, 0), height_wide - 1)
                    as usize)
                    * (stride as usize);
                let next_row_x = (x * channels_count) as usize;

                let px_idx = next_row_y + next_row_x;

                unsafe {
                    let rf32 = half::f16::from_bits(bytes[px_idx]).to_f32();
                    let gf32 = half::f16::from_bits(bytes[px_idx + 1]).to_f32();
                    let bf32 = half::f16::from_bits(bytes[px_idx + 2]).to_f32();

                    let arr_index = ((y + radius_64) & 1023) as usize;

                    dif_r += rf32;
                    sum_r += dif_r;
                    *buffer_r.get_unchecked_mut(arr_index) = rf32;

                    dif_g += gf32;
                    sum_g += dif_g;
                    *buffer_g.get_unchecked_mut(arr_index) = gf32;

                    dif_b += bf32;
                    sum_b += dif_b;
                    *buffer_b.get_unchecked_mut(arr_index) = bf32;

                    if CHANNELS == 4 {
                        let af32 = half::f16::from_bits(bytes[px_idx + 3]).to_f32();
                        dif_a += af32;
                        sum_a += dif_a;
                        *buffer_a.get_unchecked_mut(arr_index) = af32;
                    }
                }
            }
        }
    }

    pub(crate) fn fast_gaussian_horizontal_pass_f16<const CHANNELS: usize>(
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
        let mut buffer_a: [f32; 1024] = [0f32; 1024];
        let radius_64 = radius as i64;
        let width_wide = width as i64;
        let weight = 1.0f32 / ((radius as f32) * (radius as f32));
        let channels_count = match channels {
            FastBlurChannels::Channels3 => 3,
            FastBlurChannels::Channels4 => 4,
        };
        for y in start..std::cmp::min(height, end) {
            let mut dif_r: f32 = 0f32;
            let mut sum_r: f32 = 0f32;
            let mut dif_g: f32 = 0f32;
            let mut sum_g: f32 = 0f32;
            let mut dif_b: f32 = 0f32;
            let mut sum_b: f32 = 0f32;
            let mut dif_a: f32 = 0f32;
            let mut sum_a: f32 = 0f32;

            let current_y = ((y as i64) * (stride as i64)) as usize;

            let start_x = 0 - 2 * radius_64;
            for x in start_x..(width as i64) {
                if x >= 0 {
                    let current_px = ((std::cmp::max(x, 0) as u32) * channels_count) as usize;
                    let new_r = half::f16::from_f32(sum_r * weight).to_bits();
                    let new_g = half::f16::from_f32(sum_g * weight).to_bits();
                    let new_b = half::f16::from_f32(sum_b * weight).to_bits();
                    let new_a = if CHANNELS == 4 {
                        half::f16::from_f32(sum_a * weight).to_bits()
                    } else {
                        0u16
                    };

                    unsafe {
                        let pixel_offset = current_y + current_px;
                        bytes.write(pixel_offset, new_r);
                        bytes.write(pixel_offset + 1, new_g);
                        bytes.write(pixel_offset + 2, new_b);
                        if CHANNELS == 4 {
                            bytes.write(pixel_offset + 3, new_a);
                        }
                    }

                    let arr_index = ((x - radius_64) & 1023) as usize;
                    let d_arr_index = (x & 1023) as usize;
                    unsafe {
                        dif_r += (*buffer_r.get_unchecked(arr_index))
                            - 2f32 * (*buffer_r.get_unchecked(d_arr_index));
                        dif_g += (*buffer_g.get_unchecked(arr_index))
                            - 2f32 * (*buffer_g.get_unchecked(d_arr_index));
                        dif_b += (*buffer_b.get_unchecked(arr_index))
                            - 2f32 * (*buffer_b.get_unchecked(d_arr_index));
                        if CHANNELS == 4 {
                            dif_a += (*buffer_a.get_unchecked(arr_index))
                                - 2f32 * (*buffer_a.get_unchecked(d_arr_index));
                        }
                    }
                } else if x + radius_64 >= 0 {
                    let arr_index = (x & 1023) as usize;
                    unsafe {
                        dif_r -= 2f32 * (*buffer_r.get_unchecked(arr_index));
                        dif_g -= 2f32 * (*buffer_g.get_unchecked(arr_index));
                        dif_b -= 2f32 * (*buffer_b.get_unchecked(arr_index));
                        if CHANNELS == 4 {
                            dif_a -= 2f32 * (*buffer_a.get_unchecked(arr_index));
                        }
                    }
                }

                let next_row_y = (y as usize) * (stride as usize);
                let next_row_x = ((std::cmp::min(std::cmp::max(x + radius_64, 0), width_wide - 1)
                    as u32)
                    * channels_count) as usize;

                let rf32 = half::f16::from_bits(bytes[next_row_y + next_row_x]).to_f32();
                let gf32 = half::f16::from_bits(bytes[next_row_y + next_row_x + 1]).to_f32();
                let bf32 = half::f16::from_bits(bytes[next_row_y + next_row_x + 2]).to_f32();

                let arr_index = ((x + radius_64) & 1023) as usize;

                dif_r += rf32;
                sum_r += dif_r;
                unsafe {
                    *buffer_r.get_unchecked_mut(arr_index) = rf32;
                }

                dif_g += gf32;
                sum_g += dif_g;
                unsafe {
                    *buffer_g.get_unchecked_mut(arr_index) = gf32;
                }

                dif_b += bf32;
                sum_b += dif_b;
                unsafe {
                    *buffer_b.get_unchecked_mut(arr_index) = bf32;
                }

                if CHANNELS == 4 {
                    let af32 = half::f16::from_bits(bytes[next_row_y + next_row_x + 3]).to_f32();
                    dif_a += af32;
                    sum_a += dif_a;
                    unsafe {
                        *buffer_a.get_unchecked_mut(arr_index) = af32;
                    }
                }
            }
        }
    }
}

pub(crate) mod fast_gaussian_f16 {
    use crate::fast_gaussian_f16::fast_gaussian_f16_impl;
    use crate::unsafe_slice::UnsafeSlice;
    use crate::FastBlurChannels;

    pub(crate) fn fast_gaussian_impl_f16(
        bytes: &mut [u16],
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        channels: FastBlurChannels,
    ) {
        let _dispatcher_vertical: fn(
            &UnsafeSlice<u16>,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            FastBlurChannels,
        ) = match channels {
            FastBlurChannels::Channels3 => {
                fast_gaussian_f16_impl::fast_gaussian_vertical_pass_f16::<3>
            }
            FastBlurChannels::Channels4 => {
                fast_gaussian_f16_impl::fast_gaussian_vertical_pass_f16::<4>
            }
        };
        let _dispatcher_horizontal: fn(
            &UnsafeSlice<u16>,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            FastBlurChannels,
        ) = match channels {
            FastBlurChannels::Channels3 => {
                fast_gaussian_f16_impl::fast_gaussian_horizontal_pass_f16::<3>
            }
            FastBlurChannels::Channels4 => {
                fast_gaussian_f16_impl::fast_gaussian_horizontal_pass_f16::<4>
            }
        };
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
                    _dispatcher_vertical(
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
                    _dispatcher_horizontal(
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
