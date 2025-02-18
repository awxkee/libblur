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

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::cpu_features::is_aarch_f16c_supported;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::{
    fgn_horizontal_pass_neon_f16, fgn_horizontal_pass_neon_f32, fgn_horizontal_pass_neon_u8,
    fgn_vertical_pass_neon_f16, fgn_vertical_pass_neon_f32, fgn_vertical_pass_neon_u8,
};
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::{
    fast_gaussian_next_horizontal_pass_sse_f16, fast_gaussian_next_horizontal_pass_sse_u8,
    fast_gaussian_next_vertical_pass_sse_f16, fast_gaussian_next_vertical_pass_sse_u8,
    fgn_horizontal_pass_sse_f32, fgn_vertical_pass_sse_f32,
};
use crate::to_storage::ToStorage;
use crate::unsafe_slice::UnsafeSlice;
use crate::util::check_slice_size;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::wasm32::{
    fast_gaussian_next_horizontal_pass_wasm_u8, fast_gaussian_next_vertical_pass_wasm_u8,
};
use crate::{clamp_edge, reflect_101, EdgeMode, FastBlurChannels, ThreadingPolicy};
use crate::{reflect_index, BlurError};
use colorutils_rs::linear_to_planar::linear_to_plane;
use colorutils_rs::planar_to_linear::plane_to_linear;
use colorutils_rs::{
    linear_to_rgb, linear_to_rgba, rgb_to_linear, rgba_to_linear, TransferFunction,
};
use half::f16;
use num_traits::{AsPrimitive, Float, FromPrimitive};
use std::mem::size_of;

const BASE_RADIUS_I64_CUTOFF: u32 = 125;

macro_rules! impl_generic_call {
    ($store_type:ty, $channels_type:expr, $edge_mode:expr, $bytes:expr, $stride:expr, $width:expr, $height:expr, $radius:expr, $threading_policy:expr) => {
        let _dispatcher = match $channels_type {
            FastBlurChannels::Plane => fast_gaussian_next_impl::<$store_type, 1>,
            FastBlurChannels::Channels3 => fast_gaussian_next_impl::<$store_type, 3>,
            FastBlurChannels::Channels4 => fast_gaussian_next_impl::<$store_type, 4>,
        };
        _dispatcher(
            $bytes,
            $stride,
            $width,
            $height,
            $radius,
            $threading_policy,
            $edge_mode,
        );
    };
}

macro_rules! impl_margin_call {
    ($store_type:ty, $channels_type:expr, $edge_mode:expr,
    $bytes:expr, $stride:expr, $width:expr, $height:expr,
    $radius:expr, $threading_policy:expr) => {
        impl_generic_call!(
            $store_type,
            $channels_type,
            $edge_mode,
            $bytes,
            $stride,
            $width,
            $height,
            $radius,
            $threading_policy
        );
    };
}

macro_rules! write_out_blurred {
    ($sum:expr, $weight:expr, $bytes:expr, $bytes_offset:expr) => {{
        let sum_f: M = $sum.as_();
        let new_v: T = (sum_f * $weight).to_();
        unsafe {
            $bytes.write($bytes_offset, new_v);
        }
    }};
}

macro_rules! update_differences_inside {
    ($dif:expr, $buffer:expr, $d_idx:expr, $d_idx_1:expr, $d_idx_2:expr) => {{
        let threes = J::from_i32(3i32).unwrap();
        $dif += threes
            * (unsafe { *$buffer.get_unchecked($d_idx) }
                - unsafe { *$buffer.get_unchecked($d_idx_1) })
            - unsafe { *$buffer.get_unchecked($d_idx_2) };
    }};
}

macro_rules! update_differences_one_rad {
    ($dif:expr, $buffer:expr, $d_idx:expr, $d_idx_1:expr) => {{
        let threes = J::from_i32(3i32).unwrap();
        $dif += threes
            * (unsafe { *$buffer.get_unchecked($d_idx) }
                - unsafe { *$buffer.get_unchecked($d_idx_1) });
    }};
}

macro_rules! update_differences_two_rad {
    ($dif:expr, $buffer:expr, $d_idx:expr) => {{
        let threes = J::from_i32(3i32).unwrap();
        $dif -= threes * unsafe { *$buffer.get_unchecked($d_idx) };
    }};
}

macro_rules! update_sum_in {
    ($bytes:expr, $bytes_offset:expr, $dif:expr, $der:expr, $sum:expr, $buffer:expr, $arr_index:expr) => {{
        let v: J = $bytes[$bytes_offset].as_();
        $dif += v;
        $der += $dif;
        $sum += $der;
        unsafe {
            *$buffer.get_unchecked_mut($arr_index) = v;
        }
    }};
}

/// # Params
/// `T` - type of buffer
/// `J` - accumulator type
/// `M` - multiplication type, when weight will be applied this type will be used also
fn fgn_vertical_pass<
    T: FromPrimitive
        + Default
        + std::ops::AddAssign
        + 'static
        + std::ops::SubAssign
        + Copy
        + FromPrimitive
        + Default
        + AsPrimitive<J>,
    J,
    M,
    const CN: usize,
>(
    bytes: &UnsafeSlice<T>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) where
    J: Copy
        + FromPrimitive
        + Default
        + std::ops::Mul<Output = J>
        + std::ops::Sub<Output = J>
        + std::ops::Add<Output = J>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + AsPrimitive<M>,
    M: Copy + FromPrimitive + std::ops::Mul<Output = M> + AsPrimitive<T> + Float + ToStorage<T>,
    i32: AsPrimitive<J>,
{
    let mut buffer_r = Box::new([0i32.as_(); 1024]);
    let mut buffer_g = Box::new([0i32.as_(); 1024]);
    let mut buffer_b = Box::new([0i32.as_(); 1024]);
    let mut buffer_a = Box::new([0i32.as_(); 1024]);
    let radius_64 = radius as i64;
    let height_wide = height as i64;
    let weight =
        M::from_f64(1.0f64 / ((radius as f64) * (radius as f64) * (radius as f64))).unwrap();
    for x in start..std::cmp::min(width, end) {
        let mut dif_r: J = 0i32.as_();
        let mut der_r: J = 0i32.as_();
        let mut sum_r: J = 0i32.as_();
        let mut dif_g: J = 0i32.as_();
        let mut der_g: J = 0i32.as_();
        let mut sum_g: J = 0i32.as_();
        let mut dif_b: J = 0i32.as_();
        let mut der_b: J = 0i32.as_();
        let mut sum_b: J = 0i32.as_();
        let mut dif_a: J = 0i32.as_();
        let mut der_a: J = 0i32.as_();
        let mut sum_a: J = 0i32.as_();

        let current_px = (x * CN as u32) as usize;

        let start_y = 0 - 3 * radius as i64;
        for y in start_y..height_wide {
            let current_y = (y * (stride as i64)) as usize;
            if y >= 0 {
                let bytes_offset = current_y + current_px;

                write_out_blurred!(sum_r, weight, bytes, bytes_offset);
                if CN > 1 {
                    write_out_blurred!(sum_g, weight, bytes, bytes_offset + 1);
                }
                if CN > 2 {
                    write_out_blurred!(sum_b, weight, bytes, bytes_offset + 2);
                }
                if CN == 4 {
                    write_out_blurred!(sum_a, weight, bytes, bytes_offset + 3);
                }

                let d_idx_1 = ((y + radius_64) & 1023) as usize;
                let d_idx_2 = ((y - radius_64) & 1023) as usize;
                let d_idx = (y & 1023) as usize;
                update_differences_inside!(dif_r, buffer_r, d_idx, d_idx_1, d_idx_2);
                if CN > 1 {
                    update_differences_inside!(dif_g, buffer_g, d_idx, d_idx_1, d_idx_2);
                }
                if CN > 2 {
                    update_differences_inside!(dif_b, buffer_b, d_idx, d_idx_1, d_idx_2);
                }
                if CN == 4 {
                    update_differences_inside!(dif_a, buffer_a, d_idx, d_idx_1, d_idx_2);
                }
            } else if y + radius_64 >= 0 {
                let arr_index = (y & 1023) as usize;
                let arr_index_1 = ((y + radius_64) & 1023) as usize;
                update_differences_one_rad!(dif_r, buffer_r, arr_index, arr_index_1);
                if CN > 1 {
                    update_differences_one_rad!(dif_g, buffer_g, arr_index, arr_index_1);
                }
                if CN > 2 {
                    update_differences_one_rad!(dif_b, buffer_b, arr_index, arr_index_1);
                }
                if CN == 4 {
                    update_differences_one_rad!(dif_a, buffer_a, arr_index, arr_index_1);
                }
            } else if y + 2 * radius_64 >= 0 {
                let arr_index = ((y + radius_64) & 1023) as usize;
                update_differences_two_rad!(dif_r, buffer_r, arr_index);
                if CN > 1 {
                    update_differences_two_rad!(dif_g, buffer_g, arr_index);
                }
                if CN > 2 {
                    update_differences_two_rad!(dif_b, buffer_b, arr_index);
                }
                if CN == 4 {
                    update_differences_two_rad!(dif_a, buffer_a, arr_index);
                }
            }

            let next_row_y = clamp_edge!(edge_mode, y + ((3 * radius_64) >> 1), 0, height_wide - 1)
                * (stride as usize);
            let next_row_x = (x * CN as u32) as usize;

            let px_idx = next_row_y + next_row_x;

            let arr_index = ((y + 2 * radius_64) & 1023) as usize;
            update_sum_in!(bytes, px_idx, dif_r, der_r, sum_r, buffer_r, arr_index);
            if CN > 1 {
                update_sum_in!(bytes, px_idx + 1, dif_g, der_g, sum_g, buffer_g, arr_index);
            }
            if CN > 2 {
                update_sum_in!(bytes, px_idx + 2, dif_b, der_b, sum_b, buffer_b, arr_index);
            }
            if CN == 4 {
                update_sum_in!(bytes, px_idx + 3, dif_a, der_a, sum_a, buffer_a, arr_index);
            }
        }
    }
}

/// # Params
/// `T` - type of buffer
/// `J` - accumulator type
/// `M` - multiplication type, when weight will be applied this type will be used also
fn fgn_horizontal_pass<
    T: FromPrimitive
        + Default
        + Send
        + Sync
        + std::ops::AddAssign
        + 'static
        + std::ops::SubAssign
        + Copy
        + FromPrimitive
        + Default
        + AsPrimitive<J>,
    J,
    M,
    const CN: usize,
>(
    bytes: &UnsafeSlice<T>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) where
    J: Copy
        + FromPrimitive
        + Default
        + std::ops::Mul<Output = J>
        + std::ops::Sub<Output = J>
        + std::ops::Add<Output = J>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + AsPrimitive<M>,
    M: Copy + FromPrimitive + std::ops::Mul<Output = M> + AsPrimitive<T> + Float + ToStorage<T>,
    f32: AsPrimitive<T>,
    i32: AsPrimitive<J>,
{
    let mut buffer_r = Box::new([0i32.as_(); 1024]);
    let mut buffer_g = Box::new([0i32.as_(); 1024]);
    let mut buffer_b = Box::new([0i32.as_(); 1024]);
    let mut buffer_a = Box::new([0i32.as_(); 1024]);
    let radius_64 = radius as i64;
    let width_wide = width as i64;
    let weight =
        M::from_f64(1.0f64 / ((radius as f64) * (radius as f64) * (radius as f64))).unwrap();
    for y in start..std::cmp::min(height, end) {
        let mut dif_r: J = 0i32.as_();
        let mut der_r: J = 0i32.as_();
        let mut sum_r: J = 0i32.as_();
        let mut dif_g: J = 0i32.as_();
        let mut der_g: J = 0i32.as_();
        let mut sum_g: J = 0i32.as_();
        let mut dif_b: J = 0i32.as_();
        let mut der_b: J = 0i32.as_();
        let mut sum_b: J = 0i32.as_();
        let mut dif_a: J = 0i32.as_();
        let mut der_a: J = 0i32.as_();
        let mut sum_a: J = 0i32.as_();

        let current_y = ((y as i64) * (stride as i64)) as usize;

        for x in (0 - 3 * radius_64)..(width as i64) {
            if x >= 0 {
                let current_px = x as usize * CN;

                let bytes_offset = current_y + current_px;

                write_out_blurred!(sum_r, weight, bytes, bytes_offset);
                if CN > 1 {
                    write_out_blurred!(sum_g, weight, bytes, bytes_offset + 1);
                }
                if CN > 2 {
                    write_out_blurred!(sum_b, weight, bytes, bytes_offset + 2);
                }
                if CN == 4 {
                    write_out_blurred!(sum_a, weight, bytes, bytes_offset + 3);
                }

                let d_idx_1 = ((x + radius_64) & 1023) as usize;
                let d_idx_2 = ((x - radius_64) & 1023) as usize;
                let d_idx = (x & 1023) as usize;
                update_differences_inside!(dif_r, buffer_r, d_idx, d_idx_1, d_idx_2);
                if CN > 1 {
                    update_differences_inside!(dif_g, buffer_g, d_idx, d_idx_1, d_idx_2);
                }
                if CN > 2 {
                    update_differences_inside!(dif_b, buffer_b, d_idx, d_idx_1, d_idx_2);
                }
                if CN == 4 {
                    update_differences_inside!(dif_a, buffer_a, d_idx, d_idx_1, d_idx_2);
                }
            } else if x + radius_64 >= 0 {
                let arr_index = (x & 1023) as usize;
                let arr_index_1 = ((x + radius_64) & 1023) as usize;
                update_differences_one_rad!(dif_r, buffer_r, arr_index, arr_index_1);
                if CN > 1 {
                    update_differences_one_rad!(dif_g, buffer_g, arr_index, arr_index_1);
                }
                if CN > 2 {
                    update_differences_one_rad!(dif_b, buffer_b, arr_index, arr_index_1);
                }
                if CN == 4 {
                    update_differences_one_rad!(dif_a, buffer_a, arr_index, arr_index_1);
                }
            } else if x + 2 * radius_64 >= 0 {
                let arr_index = ((x + radius_64) & 1023) as usize;
                update_differences_two_rad!(dif_r, buffer_r, arr_index);
                if CN > 1 {
                    update_differences_two_rad!(dif_g, buffer_g, arr_index);
                }
                if CN > 2 {
                    update_differences_two_rad!(dif_b, buffer_b, arr_index);
                }
                if CN == 4 {
                    update_differences_two_rad!(dif_a, buffer_a, arr_index);
                }
            }

            let next_row_y = (y as usize) * (stride as usize);
            let next_row_x = clamp_edge!(edge_mode, x + 3 * radius_64 / 2, 0, width_wide - 1) * CN;

            let px_off = next_row_y + next_row_x;

            let arr_index = ((x + 2 * radius_64) & 1023) as usize;

            update_sum_in!(bytes, px_off, dif_r, der_r, sum_r, buffer_r, arr_index);
            if CN > 1 {
                update_sum_in!(bytes, px_off + 1, dif_g, der_g, sum_g, buffer_g, arr_index);
            }
            if CN > 2 {
                update_sum_in!(bytes, px_off + 2, dif_b, der_b, sum_b, buffer_b, arr_index);
            }
            if CN == 4 {
                update_sum_in!(bytes, px_off + 3, dif_a, der_a, sum_a, buffer_a, arr_index);
            }
        }
    }
}

trait FastGaussianNextPassProvider<T> {
    fn get_horizontal<const CHANNEL_CONFIGURATION: usize>(
        radius: u32,
    ) -> fn(
        bytes: &UnsafeSlice<T>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        start: u32,
        end: u32,
        EdgeMode,
    );

    fn get_vertical<const CHANNEL_CONFIGURATION: usize>(
        radius: u32,
    ) -> fn(
        bytes: &UnsafeSlice<T>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        start: u32,
        end: u32,
        EdgeMode,
    );
}

impl FastGaussianNextPassProvider<u16> for u16 {
    fn get_horizontal<const CN: usize>(
        radius: u32,
    ) -> fn(&UnsafeSlice<u16>, u32, u32, u32, u32, u32, u32, EdgeMode) {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if BASE_RADIUS_I64_CUTOFF > radius {
                use crate::neon::fgn_horizontal_pass_neon_u16;
                return fgn_horizontal_pass_neon_u16::<CN>;
            }
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            let is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");

            if BASE_RADIUS_I64_CUTOFF > radius && is_sse_available {
                use crate::sse::fgn_horizontal_pass_sse_u16;
                return fgn_horizontal_pass_sse_u16::<CN>;
            }
        }
        if BASE_RADIUS_I64_CUTOFF > radius {
            fgn_horizontal_pass::<u16, i32, f32, CN>
        } else {
            fgn_horizontal_pass::<u16, i64, f64, CN>
        }
    }

    fn get_vertical<const CN: usize>(
        radius: u32,
    ) -> fn(&UnsafeSlice<u16>, u32, u32, u32, u32, u32, u32, EdgeMode) {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if BASE_RADIUS_I64_CUTOFF > radius {
                use crate::neon::fgn_vertical_pass_neon_u16;
                return fgn_vertical_pass_neon_u16::<CN>;
            }
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            let is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");

            if BASE_RADIUS_I64_CUTOFF > radius && is_sse_available {
                use crate::sse::fgn_vertical_pass_sse_u16;
                return fgn_vertical_pass_sse_u16::<CN>;
            }
        }
        if BASE_RADIUS_I64_CUTOFF > radius {
            fgn_vertical_pass::<u16, i32, f32, CN>
        } else {
            fgn_vertical_pass::<u16, i64, f64, CN>
        }
    }
}

impl FastGaussianNextPassProvider<u8> for u8 {
    fn get_horizontal<const CN: usize>(
        radius: u32,
    ) -> fn(&UnsafeSlice<u8>, u32, u32, u32, u32, u32, u32, EdgeMode) {
        let mut _dispatcher_horizontal: fn(
            bytes: &UnsafeSlice<u8>,
            stride: u32,
            width: u32,
            height: u32,
            radius: u32,
            start: u32,
            end: u32,
            EdgeMode,
        ) = if BASE_RADIUS_I64_CUTOFF > radius {
            fgn_horizontal_pass::<u8, i32, f32, CN>
        } else {
            fgn_horizontal_pass::<u8, i64, f64, CN>
        };

        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            let is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");

            if BASE_RADIUS_I64_CUTOFF > radius && is_sse_available {
                _dispatcher_horizontal = fast_gaussian_next_horizontal_pass_sse_u8::<u8, CN>;
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if BASE_RADIUS_I64_CUTOFF > radius {
                _dispatcher_horizontal = fgn_horizontal_pass_neon_u8::<u8, CN>;
            }
        }

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if BASE_RADIUS_I64_CUTOFF > radius {
                _dispatcher_horizontal = fast_gaussian_next_horizontal_pass_wasm_u8::<u8, CN>;
            }
        }

        _dispatcher_horizontal
    }

    fn get_vertical<const CN: usize>(
        radius: u32,
    ) -> fn(&UnsafeSlice<u8>, u32, u32, u32, u32, u32, u32, EdgeMode) {
        let mut _dispatcher_vertical: fn(
            bytes: &UnsafeSlice<u8>,
            stride: u32,
            width: u32,
            height: u32,
            radius: u32,
            start: u32,
            end: u32,
            EdgeMode,
        ) = if BASE_RADIUS_I64_CUTOFF > radius {
            fgn_vertical_pass::<u8, i32, f32, CN>
        } else {
            fgn_vertical_pass::<u8, i64, f64, CN>
        };

        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            let _is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");

            if BASE_RADIUS_I64_CUTOFF > radius && _is_sse_available {
                _dispatcher_vertical = fast_gaussian_next_vertical_pass_sse_u8::<u8, CN>;
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if BASE_RADIUS_I64_CUTOFF > radius {
                _dispatcher_vertical = fgn_vertical_pass_neon_u8::<u8, CN>;
            }
        }

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if BASE_RADIUS_I64_CUTOFF > radius {
                _dispatcher_vertical = fast_gaussian_next_vertical_pass_wasm_u8::<u8, CN>;
            }
        }
        _dispatcher_vertical
    }
}

impl FastGaussianNextPassProvider<f32> for f32 {
    fn get_horizontal<const CN: usize>(
        radius: u32,
    ) -> fn(&UnsafeSlice<f32>, u32, u32, u32, u32, u32, u32, EdgeMode) {
        let mut _dispatcher_horizontal: fn(
            &UnsafeSlice<f32>,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            EdgeMode,
        ) = if BASE_RADIUS_I64_CUTOFF > radius {
            fgn_horizontal_pass::<f32, f32, f32, CN>
        } else {
            fgn_horizontal_pass::<f32, f64, f64, CN>
        };
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            let _is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
            if _is_sse_available {
                _dispatcher_horizontal = fgn_horizontal_pass_sse_f32::<f32, CN>;
            }
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if BASE_RADIUS_I64_CUTOFF > radius {
                _dispatcher_horizontal = fgn_horizontal_pass_neon_f32::<f32, CN>;
            }
        }
        _dispatcher_horizontal
    }

    fn get_vertical<const CN: usize>(
        radius: u32,
    ) -> fn(&UnsafeSlice<f32>, u32, u32, u32, u32, u32, u32, EdgeMode) {
        let mut _dispatcher_vertical: fn(
            &UnsafeSlice<f32>,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            EdgeMode,
        ) = if BASE_RADIUS_I64_CUTOFF > radius {
            fgn_vertical_pass::<f32, f32, f32, CN>
        } else {
            fgn_vertical_pass::<f32, f64, f64, CN>
        };
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            let is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
            if is_sse_available {
                _dispatcher_vertical = fgn_vertical_pass_sse_f32::<f32, CN>;
            }
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if BASE_RADIUS_I64_CUTOFF > radius {
                _dispatcher_vertical = fgn_vertical_pass_neon_f32::<f32, CN>;
            }
        }
        _dispatcher_vertical
    }
}

impl FastGaussianNextPassProvider<f16> for f16 {
    fn get_horizontal<const CN: usize>(
        radius: u32,
    ) -> fn(&UnsafeSlice<f16>, u32, u32, u32, u32, u32, u32, EdgeMode) {
        let mut _dispatcher_horizontal: fn(
            &UnsafeSlice<f16>,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            EdgeMode,
        ) = if BASE_RADIUS_I64_CUTOFF > radius {
            fgn_horizontal_pass::<f16, f32, f32, CN>
        } else {
            fgn_horizontal_pass::<f16, f64, f64, CN>
        };
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            let _is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
            let _is_f16c_available = std::arch::is_x86_feature_detected!("f16c");
            if _is_f16c_available && _is_sse_available {
                _dispatcher_horizontal = fast_gaussian_next_horizontal_pass_sse_f16::<f16, CN>;
            }
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if is_aarch_f16c_supported() {
                _dispatcher_horizontal = fgn_horizontal_pass_neon_f16::<f16, CN>;
            }
        }
        _dispatcher_horizontal
    }

    fn get_vertical<const CN: usize>(
        radius: u32,
    ) -> fn(&UnsafeSlice<f16>, u32, u32, u32, u32, u32, u32, EdgeMode) {
        let mut _dispatcher_vertical: fn(
            &UnsafeSlice<f16>,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            EdgeMode,
        ) = if BASE_RADIUS_I64_CUTOFF > radius {
            fgn_vertical_pass::<f16, f32, f32, CN>
        } else {
            fgn_vertical_pass::<f16, f64, f64, CN>
        };
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            let _is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
            let _is_f16c_available = std::arch::is_x86_feature_detected!("f16c");
            if _is_f16c_available && _is_sse_available {
                _dispatcher_vertical = fast_gaussian_next_vertical_pass_sse_f16::<f16, CN>;
            }
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if is_aarch_f16c_supported() {
                _dispatcher_vertical = fgn_vertical_pass_neon_f16::<f16, CN>;
            }
        }
        _dispatcher_vertical
    }
}

fn fast_gaussian_next_impl<
    T: FromPrimitive
        + Default
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Copy
        + AsPrimitive<f32>
        + AsPrimitive<f64>
        + AsPrimitive<i64>
        + AsPrimitive<i32>
        + FastGaussianNextPassProvider<T>,
    const CN: usize,
>(
    bytes: &mut [T],
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    threading_policy: ThreadingPolicy,
    edge_mode: EdgeMode,
) where
    i64: AsPrimitive<T>,
    f32: AsPrimitive<T> + ToStorage<T>,
    f64: AsPrimitive<T> + ToStorage<T>,
{
    let mut _dispatcher_vertical: fn(
        bytes: &UnsafeSlice<T>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        start: u32,
        end: u32,
        EdgeMode,
    ) = T::get_vertical::<CN>(radius);
    let mut _dispatcher_horizontal: fn(
        bytes: &UnsafeSlice<T>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        start: u32,
        end: u32,
        EdgeMode,
    ) = T::get_horizontal::<CN>(radius);
    let thread_count = threading_policy.thread_count(width, height) as u32;
    if thread_count == 1 {
        let unsafe_image = UnsafeSlice::new(bytes);
        _dispatcher_vertical(
            &unsafe_image,
            stride,
            width,
            height,
            radius,
            0,
            width,
            edge_mode,
        );
        _dispatcher_horizontal(
            &unsafe_image,
            stride,
            width,
            height,
            radius,
            0,
            height,
            edge_mode,
        );
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count as usize)
            .build()
            .unwrap();

        let unsafe_image = UnsafeSlice::new(bytes);
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
                        edge_mode,
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
                        edge_mode,
                    );
                });
            }
        });
    }
}

/// Performs gaussian approximation on the image.
///
/// Fast gaussian approximation for u8 image.
/// This is also a VERY fast approximation, however producing more pleasant results than stack blu.
/// Radius is limited to 280.
/// Approximation based on binomial filter.
/// O(1) complexity.
///
/// * `stride` - Lane length, default is width * channels_count if not aligned
/// * `width` - Width of the image
/// * `height` - Height of the image
/// * `radius` - Radius is limited to 280
/// * `channels` - Count of channels of the image, see [FastBlurChannels] for more info
/// * `threading_policy` - Threads usage policy
/// * `edge_mode` - Edge handling mode, *Kernel clip* is not supported!
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn fast_gaussian_next(
    bytes: &mut [u8],
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
    threading_policy: ThreadingPolicy,
    edge_mode: EdgeMode,
) -> Result<(), BlurError> {
    check_slice_size(
        bytes,
        stride as usize,
        width as usize,
        height as usize,
        channels.get_channels(),
    )?;
    let radius = std::cmp::min(radius, 280);
    impl_margin_call!(
        u8,
        channels,
        edge_mode,
        bytes,
        stride,
        width,
        height,
        radius,
        threading_policy
    );
    Ok(())
}

/// Performs gaussian approximation on the image.
///
/// Fast gaussian approximation for u16 image.
/// This is also a VERY fast approximation, however producing more pleasant results than stack blur.
/// Radius is limited to 152.
/// Approximation based on binomial filter.
/// O(1) complexity.
///
/// * `stride` - Lane length, default is width * channels_count if not aligned
/// * `width` - Width of the image
/// * `height` - Height of the image
/// * `radius` - Radius more than ~152 is not supported. To use larger radius convert image to f32 and use function for f32
/// * `channels` - Count of channels of the image, see [FastBlurChannels] for more info
/// * `threading_policy` - Threads usage policy
/// * `edge_mode` - Edge handling mode, *Kernel clip* is not supported!
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn fast_gaussian_next_u16(
    bytes: &mut [u16],
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
    threading_policy: ThreadingPolicy,
    edge_mode: EdgeMode,
) -> Result<(), BlurError> {
    check_slice_size(
        bytes,
        stride as usize,
        width as usize,
        height as usize,
        channels.get_channels(),
    )?;
    let acq_radius = std::cmp::min(radius, 152);
    impl_margin_call!(
        u16,
        channels,
        edge_mode,
        bytes,
        stride,
        width,
        height,
        acq_radius,
        threading_policy
    );
    Ok(())
}

/// Performs gaussian approximation on the image.
///
/// Fast gaussian approximation for u16 image.
/// This is also a VERY fast approximation, however producing more pleasant results than stack blur.
/// Approximation based on binomial filter.
/// O(1) complexity.
///
/// * `stride` - Lane length, default is width * channels_count if not aligned
/// * `width` - Width of the image
/// * `height` - Height of the image
/// * `radius` - Almost any radius is supported, in real world radius > 300 is too big for this implementation
/// * `channels` - Count of channels of the image, only 3 and 4 is supported, alpha position, and channels order does not matter
/// * `threading_policy` - Threads usage policy
/// * `edge_mode` - Edge handling mode, *Kernel clip* is not supported!
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn fast_gaussian_next_f32(
    bytes: &mut [f32],
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
    threading_policy: ThreadingPolicy,
    edge_mode: EdgeMode,
) -> Result<(), BlurError> {
    check_slice_size(
        bytes,
        stride as usize,
        width as usize,
        height as usize,
        channels.get_channels(),
    )?;
    impl_margin_call!(
        f32,
        channels,
        edge_mode,
        bytes,
        stride,
        width,
        height,
        radius,
        threading_policy
    );
    Ok(())
}

/// Performs gaussian approximation on the image.
///
/// Fast gaussian approximation for f16 image.
/// This is also a VERY fast approximation, however producing more pleasant results than stack blur.
/// Approximation based on binomial filter.
/// O(1) complexity.
///
/// * `stride` - Lane length, default is width * channels_count if not aligned
/// * `width` - Width of the image
/// * `height` - Height of the image
/// * `radius` - Almost any radius is supported, in real world radius > 300 is too big for this implementation
/// * `channels` - Count of channels of the image, see [FastBlurChannels] for more info
/// * `threading_policy` - Threads usage policy
/// * `edge_mode` - Edge handling mode, *Kernel clip* is not supported!
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn fast_gaussian_next_f16(
    bytes: &mut [f16],
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
    threading_policy: ThreadingPolicy,
    edge_mode: EdgeMode,
) -> Result<(), BlurError> {
    check_slice_size(
        bytes,
        stride as usize,
        width as usize,
        height as usize,
        channels.get_channels(),
    )?;
    impl_margin_call!(
        half::f16,
        channels,
        edge_mode,
        unsafe { std::mem::transmute::<&mut [half::f16], &mut [half::f16]>(bytes) },
        stride,
        width,
        height,
        radius,
        threading_policy
    );
    Ok(())
}

/// Performs gaussian approximation on the image in linear color space
///
/// This is fast approximation that first converts in linear colorspace, performs blur and converts back,
/// operation will be performed in f32 so its cost is significant
/// Approximation based on binomial filter.
/// O(1) complexity.
///
/// * `stride` - Lane length, default is width * channels_count if not aligned
/// * `width` - Width of the image
/// * `height` - Height of the image
/// * `radius` - Almost any reasonable radius is supported
/// * `channels` - Count of channels of the image, see [FastBlurChannels] for more info
/// * `threading_policy` - Threads usage policy
/// * `transfer_function` - Transfer function in linear colorspace
/// * `edge_mode` - Edge handling mode, *Kernel clip* is not supported!
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn fast_gaussian_next_in_linear(
    in_place: &mut [u8],
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
    threading_policy: ThreadingPolicy,
    transfer_function: TransferFunction,
    edge_mode: EdgeMode,
) -> Result<(), BlurError> {
    check_slice_size(
        in_place,
        stride as usize,
        width as usize,
        height as usize,
        channels.get_channels(),
    )?;
    let mut linear_data: Vec<f32> =
        vec![0f32; width as usize * height as usize * channels.get_channels()];

    let forward_transformer = match channels {
        FastBlurChannels::Plane => plane_to_linear,
        FastBlurChannels::Channels3 => rgb_to_linear,
        FastBlurChannels::Channels4 => rgba_to_linear,
    };

    let inverse_transformer = match channels {
        FastBlurChannels::Plane => linear_to_plane,
        FastBlurChannels::Channels3 => linear_to_rgb,
        FastBlurChannels::Channels4 => linear_to_rgba,
    };

    forward_transformer(
        in_place,
        stride,
        &mut linear_data,
        width * size_of::<f32>() as u32 * channels.get_channels() as u32,
        width,
        height,
        transfer_function,
    );

    fast_gaussian_next_f32(
        &mut linear_data,
        width * channels.get_channels() as u32,
        width,
        height,
        radius,
        channels,
        threading_policy,
        edge_mode,
    )?;

    inverse_transformer(
        &linear_data,
        width * size_of::<f32>() as u32 * channels.get_channels() as u32,
        in_place,
        stride,
        width,
        height,
        transfer_function,
    );
    Ok(())
}
