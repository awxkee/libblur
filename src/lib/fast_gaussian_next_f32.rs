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

pub(crate) mod fast_gaussian_next_f32_cpu {
    use crate::unsafe_slice::UnsafeSlice;
    use crate::FastBlurChannels;
    use num_traits::{AsPrimitive, FromPrimitive};

    pub(crate) fn fast_gaussian_next_vertical_pass_f32<T, const CHANNELS_COUNT: usize>(
        bytes: &UnsafeSlice<T>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        start: u32,
        end: u32,
    ) where
        T: Copy + AsPrimitive<f32> + FromPrimitive,
    {
        let channels: FastBlurChannels = CHANNELS_COUNT.into();
        let mut buffer_r: [f32; 1024] = [0f32; 1024];
        let mut buffer_g: [f32; 1024] = [0f32; 1024];
        let mut buffer_b: [f32; 1024] = [0f32; 1024];
        let mut buffer_a: [f32; 1024] = [0f32; 1024];
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
            let mut dif_a: f32 = 0f32;
            let mut der_a: f32 = 0f32;
            let mut sum_a: f32 = 0f32;

            let current_px = (x * channels_count) as usize;

            let start_y = 0 - 3 * radius as i64;
            for y in start_y..height_wide {
                let current_y = (y * (stride as i64)) as usize;
                if y >= 0 {
                    let new_r = sum_r * weight;
                    let new_g = sum_g * weight;
                    let new_b = sum_b * weight;

                    unsafe {
                        let offset = current_y + current_px;
                        bytes.write(offset, T::from_f32(new_r).unwrap());
                        bytes.write(offset + 1, T::from_f32(new_g).unwrap());
                        bytes.write(offset + 2, T::from_f32(new_b).unwrap());
                        if CHANNELS_COUNT == 4 {
                            let new_a = sum_a * weight;
                            bytes.write(offset + 3, T::from_f32(new_a).unwrap());
                        }
                    }

                    let d_arr_index_1 = ((y + radius_64) & 1023) as usize;
                    let d_arr_index_2 = ((y - radius_64) & 1023) as usize;
                    let d_arr_index = (y & 1023) as usize;
                    unsafe {
                        dif_r += 3f32
                            * ((*buffer_r.get_unchecked(d_arr_index))
                                - (*buffer_r.get_unchecked(d_arr_index_1)))
                            - (*buffer_r.get_unchecked(d_arr_index_2));
                        dif_g += 3f32
                            * ((*buffer_g.get_unchecked(d_arr_index))
                                - (*buffer_g.get_unchecked(d_arr_index_1)))
                            - (*buffer_g.get_unchecked(d_arr_index_2));
                        dif_b += 3f32
                            * ((*buffer_b.get_unchecked(d_arr_index))
                                - (*buffer_b.get_unchecked(d_arr_index_1)))
                            - (*buffer_b.get_unchecked(d_arr_index_2));
                        if CHANNELS_COUNT == 4 {
                            dif_a += 3f32
                                * ((*buffer_a.get_unchecked(d_arr_index))
                                    - (*buffer_a.get_unchecked(d_arr_index_1)))
                                - (*buffer_a.get_unchecked(d_arr_index_2));
                        }
                    }
                } else if y + radius_64 >= 0 {
                    let arr_index = (y & 1023) as usize;
                    let arr_index_1 = ((y + radius_64) & 1023) as usize;
                    unsafe {
                        dif_r += 3f32
                            * ((*buffer_r.get_unchecked(arr_index))
                                - (*buffer_r.get_unchecked(arr_index_1)));
                        dif_g += 3f32
                            * ((*buffer_g.get_unchecked(arr_index))
                                - (*buffer_g.get_unchecked(arr_index_1)));
                        dif_b += 3f32
                            * ((*buffer_b.get_unchecked(arr_index))
                                - (*buffer_b.get_unchecked(arr_index_1)));
                        if CHANNELS_COUNT == 4 {
                            dif_a += 3f32
                                * ((*buffer_a.get_unchecked(arr_index))
                                    - (*buffer_a.get_unchecked(arr_index_1)));
                        }
                    }
                } else if y + 2 * radius_64 >= 0 {
                    let arr_index = ((y + radius_64) & 1023) as usize;
                    unsafe {
                        dif_r -= 3f32 * (*buffer_r.get_unchecked(arr_index));
                        dif_g -= 3f32 * (*buffer_g.get_unchecked(arr_index));
                        dif_b -= 3f32 * (*buffer_b.get_unchecked(arr_index));
                        if CHANNELS_COUNT == 4 {
                            dif_a -= 3f32 * (*buffer_a.get_unchecked(arr_index));
                        }
                    }
                }

                let next_row_y = (std::cmp::min(
                    std::cmp::max(y + ((3 * radius_64) >> 1), 0),
                    height_wide - 1,
                ) as usize)
                    * (stride as usize);
                let next_row_x = (x * channels_count) as usize;

                let px_idx = next_row_y + next_row_x;

                let rf32 = bytes[px_idx].as_();
                let gf32 = bytes[px_idx + 1].as_();
                let bf32 = bytes[px_idx + 2].as_();

                let arr_index = ((y + 2 * radius_64) & 1023) as usize;

                dif_r += rf32;
                der_r += dif_r;
                sum_r += der_r;
                unsafe {
                    *buffer_r.get_unchecked_mut(arr_index) = rf32;
                }

                dif_g += gf32;
                der_g += dif_g;
                sum_g += der_g;
                unsafe {
                    *buffer_g.get_unchecked_mut(arr_index) = gf32;
                }

                dif_b += bf32;
                der_b += dif_b;
                sum_b += der_b;
                unsafe {
                    *buffer_b.get_unchecked_mut(arr_index) = bf32;
                }

                if CHANNELS_COUNT == 4 {
                    let af32 = bytes[px_idx + 3].as_();
                    dif_a += af32;
                    der_a += dif_a;
                    sum_a += der_a;
                    unsafe {
                        *buffer_a.get_unchecked_mut(arr_index) = af32;
                    }
                }
            }
        }
    }

    pub(crate) fn fast_gaussian_next_horizontal_pass_f32<T, const CHANNELS_COUNT: usize>(
        bytes: &UnsafeSlice<T>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        start: u32,
        end: u32,
    ) where
        T: Copy + AsPrimitive<f32> + FromPrimitive,
    {
        let channels: FastBlurChannels = CHANNELS_COUNT.into();
        let mut buffer_r: [f32; 1024] = [0f32; 1024];
        let mut buffer_g: [f32; 1024] = [0f32; 1024];
        let mut buffer_b: [f32; 1024] = [0f32; 1024];
        let mut buffer_a: [f32; 1024] = [0f32; 1024];
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
            let mut dif_a: f32 = 0f32;
            let mut der_a: f32 = 0f32;
            let mut sum_a: f32 = 0f32;

            let current_y = ((y as i64) * (stride as i64)) as usize;

            for x in (0 - 3 * radius_64)..(width as i64) {
                if x >= 0 {
                    let current_px = ((std::cmp::max(x, 0) as u32) * channels_count) as usize;
                    let new_r = sum_r * weight;
                    let new_g = sum_g * weight;
                    let new_b = sum_b * weight;

                    unsafe {
                        let offset = current_y + current_px;
                        bytes.write(offset, T::from_f32(new_r).unwrap());
                        bytes.write(offset + 1, T::from_f32(new_g).unwrap());
                        bytes.write(offset + 2, T::from_f32(new_b).unwrap());
                        if CHANNELS_COUNT == 4 {
                            let new_a = sum_a * weight;
                            bytes.write(offset + 3, T::from_f32(new_a).unwrap());
                        }
                    }

                    let d_arr_index_1 = ((x + radius_64) & 1023) as usize;
                    let d_arr_index_2 = ((x - radius_64) & 1023) as usize;
                    let d_arr_index = (x & 1023) as usize;
                    unsafe {
                        dif_r += 3f32
                            * ((*buffer_r.get_unchecked(d_arr_index))
                                - (*buffer_r.get_unchecked(d_arr_index_1)))
                            - (*buffer_r.get_unchecked(d_arr_index_2));
                        dif_g += 3f32
                            * ((*buffer_g.get_unchecked(d_arr_index))
                                - (*buffer_g.get_unchecked(d_arr_index_1)))
                            - (*buffer_g.get_unchecked(d_arr_index_2));
                        dif_b += 3f32
                            * ((*buffer_b.get_unchecked(d_arr_index))
                                - (*buffer_b.get_unchecked(d_arr_index_1)))
                            - (*buffer_b.get_unchecked(d_arr_index_2));
                        if CHANNELS_COUNT == 4 {
                            dif_a += 3f32
                                * ((*buffer_a.get_unchecked(d_arr_index))
                                    - (*buffer_a.get_unchecked(d_arr_index_1)))
                                - (*buffer_a.get_unchecked(d_arr_index_2));
                        }
                    }
                } else if x + radius_64 >= 0 {
                    let arr_index = (x & 1023) as usize;
                    let arr_index_1 = ((x + radius_64) & 1023) as usize;
                    unsafe {
                        dif_r += 3f32
                            * ((*buffer_r.get_unchecked(arr_index))
                                - (*buffer_r.get_unchecked(arr_index_1)));
                        dif_g += 3f32
                            * ((*buffer_g.get_unchecked(arr_index))
                                - (*buffer_g.get_unchecked(arr_index_1)));
                        dif_b += 3f32
                            * ((*buffer_b.get_unchecked(arr_index))
                                - (*buffer_b.get_unchecked(arr_index_1)));
                        if CHANNELS_COUNT == 4 {
                            dif_a += 3f32
                                * ((*buffer_a.get_unchecked(arr_index))
                                    - (*buffer_a.get_unchecked(arr_index_1)));
                        }
                    }
                } else if x + 2 * radius_64 >= 0 {
                    let arr_index = ((x + radius_64) & 1023) as usize;
                    unsafe {
                        dif_r -= 3f32 * (*buffer_r.get_unchecked(arr_index));
                        dif_g -= 3f32 * (*buffer_g.get_unchecked(arr_index));
                        dif_b -= 3f32 * (*buffer_b.get_unchecked(arr_index));
                        if CHANNELS_COUNT == 4 {
                            dif_a -= 3f32 * (*buffer_a.get_unchecked(arr_index));
                        }
                    }
                }

                let next_row_y = (y as usize) * (stride as usize);
                let next_row_x =
                    ((std::cmp::min(std::cmp::max(x + 3 * radius_64 / 2, 0), width_wide - 1)
                        as u32)
                        * channels_count) as usize;

                let src_offset = next_row_y + next_row_x;
                let rf32 = bytes[src_offset].as_();
                let gf32 = bytes[src_offset + 1].as_();
                let bf32 = bytes[src_offset + 2].as_();

                let arr_index = ((x + 2 * radius_64) & 1023) as usize;

                dif_r += rf32;
                der_r += dif_r;
                sum_r += der_r;
                unsafe {
                    *buffer_r.get_unchecked_mut(arr_index) = rf32;
                }

                dif_g += gf32;
                der_g += dif_g;
                sum_g += der_g;
                unsafe {
                    *buffer_g.get_unchecked_mut(arr_index) = gf32;
                }

                dif_b += bf32;
                der_b += dif_b;
                sum_b += der_b;
                unsafe {
                    *buffer_b.get_unchecked_mut(arr_index) = bf32;
                }

                if CHANNELS_COUNT == 4 {
                    let af32 = bytes[src_offset + 3].as_();
                    dif_a += af32;
                    der_a += dif_a;
                    sum_a += der_a;
                    unsafe {
                        *buffer_a.get_unchecked_mut(arr_index) = af32;
                    }
                }
            }
        }
    }
}

pub(crate) mod fast_gaussian_next_f32 {
    use crate::fast_gaussian_next_f32::fast_gaussian_next_f32_cpu;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    use crate::neon::{
        fast_gaussian_next_horizontal_pass_neon_f32, fast_gaussian_next_vertical_pass_neon_f32,
    };
    use crate::unsafe_slice::UnsafeSlice;
    use crate::{FastBlurChannels, ThreadingPolicy};
    use num_traits::{AsPrimitive, FromPrimitive};

    pub(crate) fn fast_gaussian_next_impl_f32<T>(
        bytes: &mut [T],
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        channels: FastBlurChannels,
        threading_policy: ThreadingPolicy,
    ) where
        T: Copy + AsPrimitive<f32> + FromPrimitive + Send + Sync,
    {
        let threads_count = threading_policy.get_threads_count(width, height) as u32;
        let mut _dispatcher_horizontal: fn(
            &UnsafeSlice<T>,
            u32,
            u32,
            height: u32,
            radius: u32,
            start: u32,
            end: u32,
        ) = match channels {
            FastBlurChannels::Channels3 => {
                fast_gaussian_next_f32_cpu::fast_gaussian_next_horizontal_pass_f32::<T, 3>
            }
            FastBlurChannels::Channels4 => {
                fast_gaussian_next_f32_cpu::fast_gaussian_next_horizontal_pass_f32::<T, 4>
            }
        };
        let mut _dispatcher_vertical: fn(
            bytes: &UnsafeSlice<T>,
            stride: u32,
            width: u32,
            height: u32,
            radius: u32,
            start: u32,
            end: u32,
        ) = match channels {
            FastBlurChannels::Channels3 => {
                fast_gaussian_next_f32_cpu::fast_gaussian_next_vertical_pass_f32::<T, 3>
            }
            FastBlurChannels::Channels4 => {
                fast_gaussian_next_f32_cpu::fast_gaussian_next_vertical_pass_f32::<T, 4>
            }
        };
        if std::any::type_name::<T>() == "f32" {
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                _dispatcher_horizontal = match channels {
                    FastBlurChannels::Channels3 => {
                        fast_gaussian_next_horizontal_pass_neon_f32::<T, 3>
                    }
                    FastBlurChannels::Channels4 => {
                        fast_gaussian_next_horizontal_pass_neon_f32::<T, 4>
                    }
                };
                _dispatcher_vertical = match channels {
                    FastBlurChannels::Channels3 => {
                        fast_gaussian_next_vertical_pass_neon_f32::<T, 3>
                    }
                    FastBlurChannels::Channels4 => {
                        fast_gaussian_next_vertical_pass_neon_f32::<T, 4>
                    }
                };
            }
        }
        let unsafe_image = UnsafeSlice::new(bytes);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads_count as usize)
            .build()
            .unwrap();
        pool.scope(|scope| {
            let segment_size = width / threads_count;

            for i in 0..threads_count {
                let start_x = i * segment_size;
                let mut end_x = (i + 1) * segment_size;
                if i == threads_count - 1 {
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
                    );
                });
            }
        });
        pool.scope(|scope| {
            let segment_size = height / threads_count;

            for i in 0..threads_count {
                let start_y = i * segment_size;
                let mut end_y = (i + 1) * segment_size;
                if i == threads_count - 1 {
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
                    );
                });
            }
        });
    }
}
