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

use colorutils_rs::linear_to_planar::linear_to_plane;
use colorutils_rs::planar_to_linear::plane_to_linear;
use colorutils_rs::{
    linear_to_rgb, linear_to_rgba, rgb_to_linear, rgba_to_linear, TransferFunction,
};
use half::f16;
use num_traits::cast::FromPrimitive;
use num_traits::{AsPrimitive, Float};

use crate::channels_configuration::FastBlurChannels;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::cpu_features::is_aarch_f16c_supported;
use crate::edge_mode::reflect_index;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::{
    fg_horizontal_pass_neon_f16, fg_horizontal_pass_neon_f32, fg_horizontal_pass_neon_u8,
    fg_vertical_pass_neon_f16, fg_vertical_pass_neon_f32, fg_vertical_pass_neon_u8,
};
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
use crate::sse::{
    fg_horizontal_pass_sse_f16, fg_horizontal_pass_sse_f32, fg_horizontal_pass_sse_u8,
    fg_vertical_pass_sse_f16, fg_vertical_pass_sse_f32, fg_vertical_pass_sse_u8,
};
use crate::threading_policy::ThreadingPolicy;
use crate::to_storage::ToStorage;
use crate::unsafe_slice::UnsafeSlice;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::wasm32::{fg_horizontal_pass_wasm_u8, fg_vertical_pass_wasm_u8};
use crate::{clamp_edge, BlurError, BlurImageMut, EdgeMode};

const BASE_RADIUS_I64_CUTOFF: u32 = 180;

macro_rules! update_differences_inside {
    ($dif_r:expr, $buffer_r:expr, $arr_index:expr, $d_arr_index:expr) => {{
        let twos = J::from_i32(2i32).unwrap();
        $dif_r += unsafe { *$buffer_r.get_unchecked($arr_index) }
            - twos * unsafe { *$buffer_r.get_unchecked($d_arr_index) };
    }};
}

macro_rules! update_differences_out {
    ($dif:expr, $buffer:expr, $arr_index:expr) => {{
        let twos = J::from_i32(2i32).unwrap();
        $dif -= twos * unsafe { *$buffer.get_unchecked($arr_index) };
    }};
}

macro_rules! update_sum_in {
    ($bytes:expr, $bytes_offset:expr, $dif:expr, $sum:expr, $buffer:expr, $arr_index:expr) => {{
        let v: J = $bytes[$bytes_offset].as_();
        $dif += v;
        $sum += $dif;
        unsafe {
            *$buffer.get_unchecked_mut($arr_index) = v;
        }
    }};
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

macro_rules! impl_generic_call {
    ($store_type:ty, $channels_type:expr, $edge_mode:expr,
        $bytes:expr, $stride:expr, $width:expr, $height:expr,
        $radius:expr, $threading_policy:expr) => {
        let _dispatch = match $channels_type {
            FastBlurChannels::Plane => fast_gaussian_impl::<$store_type, 1>,
            FastBlurChannels::Channels3 => fast_gaussian_impl::<$store_type, 3>,
            FastBlurChannels::Channels4 => fast_gaussian_impl::<$store_type, 4>,
        };
        _dispatch(
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

trait InitialValue {
    fn get_initial(radius: usize) -> i64;
}

impl InitialValue for f32 {
    fn get_initial(_: usize) -> i64 {
        0i64
    }
}

impl InitialValue for f64 {
    fn get_initial(_: usize) -> i64 {
        0i64
    }
}

impl InitialValue for u8 {
    fn get_initial(radius: usize) -> i64 {
        ((radius * radius) >> 1) as i64
    }
}

impl InitialValue for u16 {
    fn get_initial(radius: usize) -> i64 {
        ((radius * radius) >> 1) as i64
    }
}

impl InitialValue for half::f16 {
    fn get_initial(_: usize) -> i64 {
        0i64
    }
}

/// # Params
/// `T` - type of buffer
/// `J` - accumulator type
/// `M` - multiplication type, when weight will be applied this type will be used also
fn fg_vertical_pass<T, J, M, const CN: usize>(
    bytes: &UnsafeSlice<T>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) where
    T: std::ops::AddAssign
        + 'static
        + std::ops::SubAssign
        + Copy
        + FromPrimitive
        + Default
        + AsPrimitive<J>
        + InitialValue,
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
    let initial = J::from_i64(T::get_initial(radius as usize)).unwrap();
    let weight = M::from_f64(1f64 / (radius as f64 * radius as f64)).unwrap();
    for x in start..width.min(end) {
        let mut dif_r: J = 0i32.as_();
        let mut sum_r: J = initial;
        let mut dif_g: J = 0i32.as_();
        let mut sum_g: J = initial;
        let mut dif_b: J = 0i32.as_();
        let mut sum_b: J = initial;
        let mut dif_a: J = 0i32.as_();
        let mut sum_a: J = initial;

        let current_px = (x * CN as u32) as usize;

        let start_y = 0 - 2 * radius as i64;
        for y in start_y..height_wide {
            if y >= 0 {
                let current_y = (y * (stride as i64)) as usize;
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

                let arr_index = ((y - radius_64) & 1023) as usize;
                let d_arr_index = (y & 1023) as usize;
                update_differences_inside!(dif_r, buffer_r, arr_index, d_arr_index);
                if CN > 1 {
                    update_differences_inside!(dif_g, buffer_g, arr_index, d_arr_index);
                }
                if CN > 2 {
                    update_differences_inside!(dif_b, buffer_b, arr_index, d_arr_index);
                }
                if CN == 4 {
                    update_differences_inside!(dif_a, buffer_a, arr_index, d_arr_index);
                }
            } else if y + radius_64 >= 0 {
                let arr_index = (y & 1023) as usize;
                update_differences_out!(dif_r, buffer_r, arr_index);
                if CN > 1 {
                    update_differences_out!(dif_g, buffer_g, arr_index);
                }
                if CN > 2 {
                    update_differences_out!(dif_b, buffer_b, arr_index);
                }
                if CN == 4 {
                    update_differences_out!(dif_a, buffer_a, arr_index);
                }
            }

            let next_row_y =
                clamp_edge!(edge_mode, y + radius_64, 0i64, height_wide) * (stride as usize);
            let next_row_x = (x * CN as u32) as usize;

            let px_idx = next_row_y + next_row_x;

            let arr_index = ((y + radius_64) & 1023) as usize;

            update_sum_in!(bytes, px_idx, dif_r, sum_r, buffer_r, arr_index);
            if CN > 1 {
                update_sum_in!(bytes, px_idx + 1, dif_g, sum_g, buffer_g, arr_index);
            }
            if CN > 2 {
                update_sum_in!(bytes, px_idx + 2, dif_b, sum_b, buffer_b, arr_index);
            }

            if CN == 4 {
                update_sum_in!(bytes, px_idx + 3, dif_a, sum_a, buffer_a, arr_index);
            }
        }
    }
}

/// # Params
/// `T` - type of buffer
/// `J` - accumulator type
/// `M` - multiplication type, when weight will be applied this type will be used also
fn fg_horizontal_pass<T, J, M, const CN: usize>(
    bytes: &UnsafeSlice<T>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    start: u32,
    end: u32,
    edge_mode: EdgeMode,
) where
    T: std::ops::AddAssign
        + 'static
        + std::ops::SubAssign
        + Copy
        + FromPrimitive
        + Default
        + AsPrimitive<J>
        + InitialValue,
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
    let width_wide = width as i64;
    let weight = M::from_f64(1f64 / (radius as f64 * radius as f64)).unwrap();
    let initial = J::from_i64(T::get_initial(radius as usize)).unwrap();
    for y in start..height.min(end) {
        let mut dif_r: J = 0i32.as_();
        let mut sum_r: J = initial;
        let mut dif_g: J = 0i32.as_();
        let mut sum_g: J = initial;
        let mut dif_b: J = 0i32.as_();
        let mut sum_b: J = initial;
        let mut dif_a: J = 0i32.as_();
        let mut sum_a: J = initial;

        let current_y = ((y as i64) * (stride as i64)) as usize;

        let start_x = 0 - 2 * radius_64;
        for x in start_x..(width as i64) {
            if x >= 0 {
                let current_px = (x * CN as i64) as usize;

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

                let arr_index = ((x - radius_64) & 1023) as usize;
                let d_arr_index = (x & 1023) as usize;
                update_differences_inside!(dif_r, buffer_r, arr_index, d_arr_index);
                if CN > 1 {
                    update_differences_inside!(dif_g, buffer_g, arr_index, d_arr_index);
                }
                if CN > 2 {
                    update_differences_inside!(dif_b, buffer_b, arr_index, d_arr_index);
                }
                if CN == 4 {
                    update_differences_inside!(dif_a, buffer_a, arr_index, d_arr_index);
                }
            } else if x + radius_64 >= 0 {
                let arr_index = (x & 1023) as usize;
                update_differences_out!(dif_r, buffer_r, arr_index);
                if CN > 1 {
                    update_differences_out!(dif_g, buffer_g, arr_index);
                }
                if CN > 2 {
                    update_differences_out!(dif_b, buffer_b, arr_index);
                }
                if CN == 4 {
                    update_differences_out!(dif_a, buffer_a, arr_index);
                }
            }

            let next_row_y = (y as usize) * (stride as usize);
            let next_row_x = clamp_edge!(edge_mode, x + radius_64, 0, width_wide) * CN;

            let bytes_offset = next_row_y + next_row_x;

            let arr_index = ((x + radius_64) & 1023) as usize;

            update_sum_in!(bytes, bytes_offset, dif_r, sum_r, buffer_r, arr_index);
            if CN > 1 {
                update_sum_in!(bytes, bytes_offset + 1, dif_g, sum_g, buffer_g, arr_index);
            }
            if CN > 2 {
                update_sum_in!(bytes, bytes_offset + 2, dif_b, sum_b, buffer_b, arr_index);
            }

            if CN == 4 {
                update_sum_in!(bytes, bytes_offset + 3, dif_a, sum_a, buffer_a, arr_index);
            }
        }
    }
}

trait FastGaussianDispatchProvider<T> {
    fn get_vertical<const CN: usize>(
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
    fn get_horizontal<const CN: usize>(
        radius: u32,
    ) -> fn(&UnsafeSlice<T>, u32, u32, u32, u32, u32, u32, EdgeMode);
}

impl FastGaussianDispatchProvider<u16> for u16 {
    fn get_vertical<const CN: usize>(
        radius: u32,
    ) -> fn(&UnsafeSlice<u16>, u32, u32, u32, u32, u32, u32, EdgeMode) {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            use crate::neon::fg_vertical_pass_neon_u16;
            if BASE_RADIUS_I64_CUTOFF > radius {
                return fg_vertical_pass_neon_u16::<CN>;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let has_avx = std::arch::is_x86_feature_detected!("avx2");
            if has_avx && BASE_RADIUS_I64_CUTOFF > radius {
                use crate::avx::fg_vertical_pass_avx_u16;
                return fg_vertical_pass_avx_u16::<CN>;
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            let is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
            if is_sse_available && BASE_RADIUS_I64_CUTOFF > radius {
                use crate::sse::fg_vertical_pass_sse_u16;
                return fg_vertical_pass_sse_u16::<CN>;
            }
        }
        if BASE_RADIUS_I64_CUTOFF > radius {
            fg_vertical_pass::<u16, i32, f32, CN>
        } else {
            fg_vertical_pass::<u16, i64, f64, CN>
        }
    }

    fn get_horizontal<const CN: usize>(
        radius: u32,
    ) -> fn(&UnsafeSlice<u16>, u32, u32, u32, u32, u32, u32, EdgeMode) {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if BASE_RADIUS_I64_CUTOFF > radius {
                use crate::neon::fg_horizontal_pass_neon_u16;
                return fg_horizontal_pass_neon_u16::<CN>;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let has_avx = std::arch::is_x86_feature_detected!("avx2");
            if has_avx && BASE_RADIUS_I64_CUTOFF > radius {
                use crate::avx::fg_horizontal_pass_avx_u16;
                return fg_horizontal_pass_avx_u16::<CN>;
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            let is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
            if is_sse_available && BASE_RADIUS_I64_CUTOFF > radius {
                use crate::sse::fg_horizontal_pass_sse_u16;
                return fg_horizontal_pass_sse_u16::<CN>;
            }
        }
        if BASE_RADIUS_I64_CUTOFF > radius {
            fg_horizontal_pass::<u16, i32, f32, CN>
        } else {
            fg_horizontal_pass::<u16, i64, f64, CN>
        }
    }
}

impl FastGaussianDispatchProvider<u8> for u8 {
    fn get_horizontal<const CN: usize>(
        radius: u32,
    ) -> fn(&UnsafeSlice<u8>, u32, u32, u32, u32, u32, u32, EdgeMode) {
        let mut _dispatcher_horizontal: fn(
            &UnsafeSlice<u8>,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            EdgeMode,
        ) = if BASE_RADIUS_I64_CUTOFF > radius {
            fg_horizontal_pass::<u8, i32, f32, CN>
        } else {
            fg_horizontal_pass::<u8, i64, f64, CN>
        };
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if BASE_RADIUS_I64_CUTOFF > radius {
                _dispatcher_horizontal = fg_horizontal_pass_neon_u8::<u8, CN>;
                #[cfg(feature = "rdm")]
                {
                    if std::arch::is_aarch64_feature_detected!("rdm") {
                        use crate::neon::fg_horizontal_pass_neon_u8_rdm;
                        _dispatcher_horizontal = fg_horizontal_pass_neon_u8_rdm::<u8, CN>;
                    }
                }
            }
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if BASE_RADIUS_I64_CUTOFF > radius {
                _dispatcher_horizontal = fg_horizontal_pass_wasm_u8::<u8, CN>;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let has_avx = std::arch::is_x86_feature_detected!("avx2");
            if has_avx && BASE_RADIUS_I64_CUTOFF > radius {
                use crate::avx::fg_horizontal_pass_sse_u8;
                return fg_horizontal_pass_sse_u8::<u8, CN>;
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            let is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
            if is_sse_available && BASE_RADIUS_I64_CUTOFF > radius {
                _dispatcher_horizontal = fg_horizontal_pass_sse_u8::<u8, CN>;
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
            fg_vertical_pass::<u8, i32, f32, CN>
        } else {
            fg_vertical_pass::<u8, i64, f64, CN>
        };
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if BASE_RADIUS_I64_CUTOFF > radius {
                _dispatcher_vertical = fg_vertical_pass_neon_u8::<u8, CN>;
                #[cfg(feature = "rdm")]
                {
                    if std::arch::is_aarch64_feature_detected!("rdm") {
                        use crate::neon::fg_vertical_pass_neon_u8_rdm;
                        _dispatcher_vertical = fg_vertical_pass_neon_u8_rdm::<u8, CN>;
                    }
                }
            }
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if BASE_RADIUS_I64_CUTOFF > radius {
                _dispatcher_vertical = fg_vertical_pass_wasm_u8::<u8, CN>;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let has_avx = std::arch::is_x86_feature_detected!("avx2");
            if has_avx && BASE_RADIUS_I64_CUTOFF > radius {
                use crate::avx::fg_vertical_pass_avx_u8;
                return fg_vertical_pass_avx_u8::<u8, CN>;
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            let is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
            if is_sse_available && BASE_RADIUS_I64_CUTOFF > radius {
                _dispatcher_vertical = fg_vertical_pass_sse_u8::<u8, CN>;
            }
        }
        _dispatcher_vertical
    }
}

impl FastGaussianDispatchProvider<f32> for f32 {
    fn get_vertical<const CN: usize>(
        radius: u32,
    ) -> fn(&UnsafeSlice<f32>, u32, u32, u32, u32, u32, u32, EdgeMode) {
        let mut _dispatcher_vertical: fn(
            bytes: &UnsafeSlice<f32>,
            stride: u32,
            width: u32,
            height: u32,
            radius: u32,
            start: u32,
            end: u32,
            EdgeMode,
        ) = if BASE_RADIUS_I64_CUTOFF > radius {
            fg_vertical_pass::<f32, f32, f32, CN>
        } else {
            fg_vertical_pass::<f32, f64, f64, CN>
        };
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            let is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
            if is_sse_available {
                _dispatcher_vertical = fg_vertical_pass_sse_f32::<f32, CN>;
            }
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _dispatcher_vertical = fg_vertical_pass_neon_f32::<f32, CN>;
        }
        _dispatcher_vertical
    }

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
            fg_horizontal_pass::<f32, f32, f32, CN>
        } else {
            fg_horizontal_pass::<f32, f64, f64, CN>
        };
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            let is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
            if is_sse_available {
                _dispatcher_horizontal = fg_horizontal_pass_sse_f32::<f32, CN>;
            }
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _dispatcher_horizontal = fg_horizontal_pass_neon_f32::<f32, CN>;
        }
        _dispatcher_horizontal
    }
}

impl FastGaussianDispatchProvider<f16> for f16 {
    fn get_vertical<const CN: usize>(
        radius: u32,
    ) -> fn(&UnsafeSlice<f16>, u32, u32, u32, u32, u32, u32, EdgeMode) {
        let mut _dispatcher_vertical: fn(
            bytes: &UnsafeSlice<f16>,
            stride: u32,
            width: u32,
            height: u32,
            radius: u32,
            start: u32,
            end: u32,
            EdgeMode,
        ) = if BASE_RADIUS_I64_CUTOFF > radius {
            fg_vertical_pass::<f16, f32, f32, CN>
        } else {
            fg_vertical_pass::<f16, f64, f64, CN>
        };
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if is_aarch_f16c_supported() {
                _dispatcher_vertical = fg_vertical_pass_neon_f16::<f16, CN>;
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1")
                && std::arch::is_x86_feature_detected!("f16c")
            {
                _dispatcher_vertical = fg_vertical_pass_sse_f16::<f16, CN>;
            }
        }
        _dispatcher_vertical
    }

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
            fg_horizontal_pass::<f16, f32, f32, CN>
        } else {
            fg_horizontal_pass::<f16, f64, f64, CN>
        };
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if is_aarch_f16c_supported() {
                _dispatcher_horizontal = fg_horizontal_pass_neon_f16::<f16, CN>;
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1")
                && std::arch::is_x86_feature_detected!("f16c")
            {
                _dispatcher_horizontal = fg_horizontal_pass_sse_f16::<f16, CN>;
            }
        }
        _dispatcher_horizontal
    }
}

fn fast_gaussian_impl<
    T: FromPrimitive
        + Default
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Copy
        + AsPrimitive<i32>
        + AsPrimitive<i64>
        + AsPrimitive<f32>
        + AsPrimitive<f64>
        + InitialValue
        + FastGaussianDispatchProvider<T>,
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
    f32: AsPrimitive<T> + ToStorage<T>,
    f64: AsPrimitive<T> + ToStorage<T>,
{
    let unsafe_image = UnsafeSlice::new(bytes);
    let thread_count = threading_policy.thread_count(width, height) as u32;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count as usize)
        .build()
        .unwrap();
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
    let mut _dispatcher_horizontal: fn(&UnsafeSlice<T>, u32, u32, u32, u32, u32, u32, EdgeMode) =
        T::get_horizontal::<CN>(radius);
    if thread_count == 1 {
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
/// Fast gaussian approximation for u8 image, limited to 319 radius, sometimes on the very bright images may start ringing on a very large radius.
/// Approximation based on binomial filter. Algorithm is close to stack blur with better results and a little slower speed
/// Results better than in stack blur however this a little slower.
/// This is a very fast approximation using i32 accumulator size with radius less that *BASE_RADIUS_I64_CUTOFF*,
/// after it to avoid overflowing fallback to i64 accumulator will be used with some computational slowdown with factor ~2.
/// O(1) complexity.
///
/// # Arguments
///
/// * `image` - Image to work in place, see [BlurImageMut] for more info.
/// * `radius` - Radius more than 319 is not supported. To use larger radius convert image to f32 and use function for f32.
/// * `threading_policy` - Threads usage policy.
/// * `edge_mode` - Edge handling mode, *Kernel clip* is not supported!
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn fast_gaussian(
    image: &mut BlurImageMut<u8>,
    radius: u32,
    threading_policy: ThreadingPolicy,
    edge_mode: EdgeMode,
) -> Result<(), BlurError> {
    image.check_layout(None)?;
    let radius = std::cmp::min(radius, 319);
    let stride = image.row_stride();
    let width = image.width;
    let height = image.height;
    let data = image.data.borrow_mut();
    impl_margin_call!(
        u8,
        image.channels,
        edge_mode,
        data,
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
/// Fast gaussian approximation for u16 image, limited to 319 radius, sometimes on the very bright images may start ringing on a very large radius.
/// Approximation based on binomial filter. Algorithm is close to stack blur with better results and a little slower speed.
/// O(1) complexity.
///
/// # Arguments
///
/// * `image` - Image to work in place, see [BlurImageMut] for more info.
/// * `radius` - Radius more than 255 is not supported. To use larger radius convert image to f32 and use function for f32.
/// * `edge_mode` - Edge handling mode, *Kernel clip* is not supported!
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn fast_gaussian_u16(
    image: &mut BlurImageMut<u16>,
    radius: u32,
    threading_policy: ThreadingPolicy,
    edge_mode: EdgeMode,
) -> Result<(), BlurError> {
    image.check_layout(None)?;
    let stride = image.row_stride();
    let width = image.width;
    let height = image.height;
    let channels = image.channels;
    let data = image.data.borrow_mut();
    let radius = std::cmp::min(radius, 255);
    impl_margin_call!(
        u16,
        channels,
        edge_mode,
        data,
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
/// Fast gaussian approximation for f32 image. No limitations are expected.
/// Approximation based on binomial filter. Algorithm is close to stack blur with better results and a little slower speed
/// O(1) complexity.
///
/// # Arguments
///
/// * `image` - Image to work in place, see [BlurImageMut] for more info.
/// * `radius` - almost any radius is supported.
/// * `transfer_function` - Transfer function in linear colorspace.
/// * `edge_mode` - Edge handling mode, *Kernel clip* is not supported!
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn fast_gaussian_f32(
    image: &mut BlurImageMut<f32>,
    radius: u32,
    threading_policy: ThreadingPolicy,
    edge_mode: EdgeMode,
) -> Result<(), BlurError> {
    image.check_layout(None)?;
    let stride = image.row_stride();
    let width = image.width;
    let height = image.height;
    let channels = image.channels;
    let data = image.data.borrow_mut();
    impl_margin_call!(
        f32,
        channels,
        edge_mode,
        data,
        stride,
        width,
        height,
        radius,
        threading_policy
    );
    Ok(())
}

/// Performs gaussian approximation on the image in linear colorspace
///
/// This is fast approximation that first converts in linear colorspace, performs blur and converts back,
/// operation will be performed in f32 so its cost is significant.
/// O(1) complexity.
///
/// # Arguments
///
/// * `image` - Image to work in place, see [BlurImageMut] for more info.
/// * `radius` - Almost any reasonable radius is supported.
/// * `threading_policy` - Threads usage policy.
/// * `transfer_function` - Transfer function in linear colorspace.
/// * `edge_mode` - Edge handling mode, *Kernel clip* is not supported!
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn fast_gaussian_in_linear(
    image: &mut BlurImageMut<u8>,
    radius: u32,
    threading_policy: ThreadingPolicy,
    transfer_function: TransferFunction,
    edge_mode: EdgeMode,
) -> Result<(), BlurError> {
    image.check_layout(None)?;
    let mut linear_data = BlurImageMut::alloc(image.width, image.height, image.channels);

    let forward_transformer = match image.channels {
        FastBlurChannels::Plane => plane_to_linear,
        FastBlurChannels::Channels3 => rgb_to_linear,
        FastBlurChannels::Channels4 => rgba_to_linear,
    };

    let inverse_transformer = match image.channels {
        FastBlurChannels::Plane => linear_to_plane,
        FastBlurChannels::Channels3 => linear_to_rgb,
        FastBlurChannels::Channels4 => linear_to_rgba,
    };

    let width = image.width;
    let height = image.height;
    let stride = image.row_stride();

    forward_transformer(
        image.data.borrow(),
        stride,
        linear_data.data.borrow_mut(),
        width * std::mem::size_of::<f32>() as u32 * image.channels.channels() as u32,
        width,
        height,
        transfer_function,
    );

    fast_gaussian_f32(&mut linear_data, radius, threading_policy, edge_mode)?;

    inverse_transformer(
        linear_data.data.borrow(),
        width * std::mem::size_of::<f32>() as u32 * image.channels.channels() as u32,
        image.data.borrow_mut(),
        stride,
        width,
        height,
        transfer_function,
    );
    Ok(())
}

/// Performs gaussian approximation on the image.
///
/// Fast gaussian approximation for f32 image. No limitations are expected.
/// Approximation based on binomial filter. Algorithm is close to stack blur with better results and a little slower speed.
/// O(1) complexity.
///
/// # Arguments
///
/// * `image` - Image to work in place, see [BlurImageMut] for more info.
/// * `radius` - almost any radius is supported.
/// * `threading_policy` - Threads usage policy.
/// * `edge_mode` - Edge handling mode, *Kernel clip* is not supported!
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn fast_gaussian_f16(
    image: &mut BlurImageMut<f16>,
    radius: u32,
    threading_policy: ThreadingPolicy,
    edge_mode: EdgeMode,
) -> Result<(), BlurError> {
    image.check_layout(None)?;
    let stride = image.row_stride();
    let width = image.width;
    let height = image.height;
    let channels = image.channels;
    let data = image.data.borrow_mut();
    impl_margin_call!(
        half::f16,
        channels,
        edge_mode,
        data,
        stride,
        width,
        height,
        radius,
        threading_policy
    );
    Ok(())
}
