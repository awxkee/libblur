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

#[cfg(all(target_arch = "aarch64", feature = "neon", feature = "nightly_f16"))]
use crate::neon::{fgn_horizontal_pass_neon_f16, fgn_vertical_pass_neon_f16};
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
use crate::neon::{
    fgn_horizontal_pass_neon_f32, fgn_horizontal_pass_neon_u8, fgn_vertical_pass_neon_f32,
    fgn_vertical_pass_neon_u8,
};
use crate::primitives::PrimitiveCast;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    feature = "sse",
    feature = "nightly_f16"
))]
use crate::sse::{
    fast_gaussian_next_horizontal_pass_sse_f16, fast_gaussian_next_vertical_pass_sse_f16,
};
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
use crate::sse::{
    fgn_horizontal_pass_sse_f32, fgn_horizontal_pass_sse_u8, fgn_vertical_pass_sse_f32,
    fgn_vertical_pass_sse_u8,
};
use crate::to_storage::ToStorage;
use crate::unsafe_slice::UnsafeSlice;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::wasm32::{
    fast_gaussian_next_horizontal_pass_wasm_u8, fast_gaussian_next_vertical_pass_wasm_u8,
};
use crate::{clamp_edge, BlurImageMut, EdgeMode, FastBlurChannels, ThreadingPolicy};
use crate::{AnisotropicRadius, BlurError};
#[cfg(feature = "nightly_f16")]
use core::f16;
use num_traits::Float;

const BASE_RADIUS_I64_CUTOFF: u32 = 150;
const BASE_RADIUS_I64_CUTOFF_U16: u32 = 32;

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
        let sum_f: M = $sum.cast_();
        let new_v: T = (sum_f * $weight).to_();
        unsafe {
            $bytes.write($bytes_offset, new_v);
        }
    }};
}

macro_rules! update_differences_inside {
    ($dif:expr, $buffer:expr, $d_idx:expr, $d_idx_1:expr, $d_idx_2:expr) => {{
        let threes: J = 3i32.cast_();
        $dif += threes
            * (unsafe { *$buffer.get_unchecked($d_idx) }
                - unsafe { *$buffer.get_unchecked($d_idx_1) })
            - unsafe { *$buffer.get_unchecked($d_idx_2) };
    }};
}

macro_rules! update_differences_one_rad {
    ($dif:expr, $buffer:expr, $d_idx:expr, $d_idx_1:expr) => {{
        let threes: J = 3i32.cast_();
        $dif += threes
            * (unsafe { *$buffer.get_unchecked($d_idx) }
                - unsafe { *$buffer.get_unchecked($d_idx_1) });
    }};
}

macro_rules! update_differences_two_rad {
    ($dif:expr, $buffer:expr, $d_idx:expr) => {{
        let threes: J = 3i32.cast_();
        $dif -= threes * unsafe { *$buffer.get_unchecked($d_idx) };
    }};
}

macro_rules! update_sum_in {
    ($bytes:expr, $bytes_offset:expr, $dif:expr, $der:expr, $sum:expr, $buffer:expr, $arr_index:expr) => {{
        let v: J = $bytes[$bytes_offset].cast_();
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
    T: Default
        + std::ops::AddAssign
        + 'static
        + std::ops::SubAssign
        + Copy
        + Default
        + PrimitiveCast<J>,
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
        + Default
        + std::ops::Mul<Output = J>
        + std::ops::Sub<Output = J>
        + std::ops::Add<Output = J>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + PrimitiveCast<M>,
    M: Copy + std::ops::Mul<Output = M> + PrimitiveCast<T> + Float + ToStorage<T>,
    i32: PrimitiveCast<J>,
    f64: PrimitiveCast<M>,
{
    let mut buffer_r = Box::new([0i32.cast_(); 1024]);
    let mut buffer_g = Box::new([0i32.cast_(); 1024]);
    let mut buffer_b = Box::new([0i32.cast_(); 1024]);
    let mut buffer_a = Box::new([0i32.cast_(); 1024]);
    let radius_64 = radius as i64;
    let height_wide = height as i64;
    let weight: M = (1.0f64 / ((radius as f64) * (radius as f64) * (radius as f64))).cast_();
    for x in start..std::cmp::min(width, end) {
        let mut dif_r: J = 0i32.cast_();
        let mut der_r: J = 0i32.cast_();
        let mut sum_r: J = 0i32.cast_();
        let mut dif_g: J = 0i32.cast_();
        let mut der_g: J = 0i32.cast_();
        let mut sum_g: J = 0i32.cast_();
        let mut dif_b: J = 0i32.cast_();
        let mut der_b: J = 0i32.cast_();
        let mut sum_b: J = 0i32.cast_();
        let mut dif_a: J = 0i32.cast_();
        let mut der_a: J = 0i32.cast_();
        let mut sum_a: J = 0i32.cast_();

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

            let next_row_y = clamp_edge!(edge_mode, y + ((3 * radius_64) >> 1), 0, height_wide)
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
    T: Default
        + Send
        + Sync
        + std::ops::AddAssign
        + 'static
        + std::ops::SubAssign
        + Copy
        + Default
        + PrimitiveCast<J>,
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
        + Default
        + std::ops::Mul<Output = J>
        + std::ops::Sub<Output = J>
        + std::ops::Add<Output = J>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + PrimitiveCast<M>,
    M: Copy + std::ops::Mul<Output = M> + PrimitiveCast<T> + Float + ToStorage<T>,
    f32: PrimitiveCast<T>,
    i32: PrimitiveCast<J>,
    f64: PrimitiveCast<M>,
{
    let mut buffer_r = Box::new([0i32.cast_(); 1024]);
    let mut buffer_g = Box::new([0i32.cast_(); 1024]);
    let mut buffer_b = Box::new([0i32.cast_(); 1024]);
    let mut buffer_a = Box::new([0i32.cast_(); 1024]);
    let radius_64 = radius as i64;
    let width_wide = width as i64;
    let weight: M = (1.0f64 / ((radius as f64) * (radius as f64) * (radius as f64))).cast_();
    for y in start..std::cmp::min(height, end) {
        let mut dif_r: J = 0i32.cast_();
        let mut der_r: J = 0i32.cast_();
        let mut sum_r: J = 0i32.cast_();
        let mut dif_g: J = 0i32.cast_();
        let mut der_g: J = 0i32.cast_();
        let mut sum_g: J = 0i32.cast_();
        let mut dif_b: J = 0i32.cast_();
        let mut der_b: J = 0i32.cast_();
        let mut sum_b: J = 0i32.cast_();
        let mut dif_a: J = 0i32.cast_();
        let mut der_a: J = 0i32.cast_();
        let mut sum_a: J = 0i32.cast_();

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
            let next_row_x = clamp_edge!(edge_mode, x + 3 * radius_64 / 2, 0, width_wide) * CN;

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
    fn get_horizontal<const CN: usize>(
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
}

impl FastGaussianNextPassProvider<u16> for u16 {
    fn get_horizontal<const CN: usize>(
        radius: u32,
    ) -> fn(&UnsafeSlice<u16>, u32, u32, u32, u32, u32, u32, EdgeMode) {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            if BASE_RADIUS_I64_CUTOFF_U16 > radius {
                #[cfg(feature = "rdm")]
                {
                    if std::arch::is_aarch64_feature_detected!("rdm") {
                        use crate::neon::fgn_horizontal_pass_neon_u16_q0_31;
                        return fgn_horizontal_pass_neon_u16_q0_31::<CN>;
                    }
                }
                use crate::neon::fgn_horizontal_pass_neon_u16;
                return fgn_horizontal_pass_neon_u16::<CN>;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let has_avx = std::arch::is_x86_feature_detected!("avx2");

            if BASE_RADIUS_I64_CUTOFF_U16 > radius && has_avx {
                use crate::avx::fgn_horizontal_pass_avx_u16;
                return fgn_horizontal_pass_avx_u16::<CN>;
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            let is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");

            if BASE_RADIUS_I64_CUTOFF_U16 > radius && is_sse_available {
                use crate::sse::fgn_horizontal_pass_sse_u16;
                return fgn_horizontal_pass_sse_u16::<CN>;
            }
        }
        if BASE_RADIUS_I64_CUTOFF_U16 > radius {
            fgn_horizontal_pass::<u16, i32, f32, CN>
        } else {
            fgn_horizontal_pass::<u16, i64, f64, CN>
        }
    }

    fn get_vertical<const CN: usize>(
        radius: u32,
    ) -> fn(&UnsafeSlice<u16>, u32, u32, u32, u32, u32, u32, EdgeMode) {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            if BASE_RADIUS_I64_CUTOFF_U16 > radius {
                #[cfg(feature = "rdm")]
                {
                    if std::arch::is_aarch64_feature_detected!("rdm") {
                        use crate::neon::fgn_vertical_pass_neon_u16_q0_31;
                        return fgn_vertical_pass_neon_u16_q0_31::<CN>;
                    }
                }
                use crate::neon::fgn_vertical_pass_neon_u16;
                return fgn_vertical_pass_neon_u16::<CN>;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let has_avx = std::arch::is_x86_feature_detected!("avx2");

            if BASE_RADIUS_I64_CUTOFF_U16 > radius && has_avx {
                use crate::avx::fgn_vertical_pass_avx_u16;
                return fgn_vertical_pass_avx_u16::<CN>;
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            let is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");

            if BASE_RADIUS_I64_CUTOFF_U16 > radius && is_sse_available {
                use crate::sse::fgn_vertical_pass_sse_u16;
                return fgn_vertical_pass_sse_u16::<CN>;
            }
        }
        if BASE_RADIUS_I64_CUTOFF_U16 > radius {
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

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let is_avx_available = std::arch::is_x86_feature_detected!("avx2");

            if BASE_RADIUS_I64_CUTOFF > radius && is_avx_available {
                use crate::avx::fgn_horizontal_pass_avx2_u8;
                return fgn_horizontal_pass_avx2_u8::<u8, CN>;
            }
        }

        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            let is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");

            if BASE_RADIUS_I64_CUTOFF > radius && is_sse_available {
                _dispatcher_horizontal = fgn_horizontal_pass_sse_u8::<u8, CN>;
            }
        }

        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            if BASE_RADIUS_I64_CUTOFF > radius {
                _dispatcher_horizontal = fgn_horizontal_pass_neon_u8::<u8, CN>;
                #[cfg(feature = "rdm")]
                {
                    if std::arch::is_aarch64_feature_detected!("rdm") {
                        use crate::neon::fgn_horizontal_pass_neon_u8_rdm;
                        _dispatcher_horizontal = fgn_horizontal_pass_neon_u8_rdm::<u8, CN>;
                    }
                }
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

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let is_avx_available = std::arch::is_x86_feature_detected!("avx2");

            if BASE_RADIUS_I64_CUTOFF > radius && is_avx_available {
                use crate::avx::fgn_vertical_pass_avx_u8;
                return fgn_vertical_pass_avx_u8::<u8, CN>;
            }
        }

        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            let is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");

            if BASE_RADIUS_I64_CUTOFF > radius && is_sse_available {
                _dispatcher_vertical = fgn_vertical_pass_sse_u8::<u8, CN>;
            }
        }

        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            if BASE_RADIUS_I64_CUTOFF > radius {
                _dispatcher_vertical = fgn_vertical_pass_neon_u8::<u8, CN>;
                #[cfg(feature = "rdm")]
                {
                    if std::arch::is_aarch64_feature_detected!("rdm") {
                        use crate::neon::fgn_vertical_pass_neon_u8_rdm;
                        _dispatcher_vertical = fgn_vertical_pass_neon_u8_rdm::<u8, CN>;
                    }
                }
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
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let has_avx = std::arch::is_x86_feature_detected!("avx2");

            if has_avx {
                use crate::avx::fgn_horizontal_pass_avx_f32;
                return fgn_horizontal_pass_avx_f32::<f32, CN>;
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            let is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
            if is_sse_available {
                _dispatcher_horizontal = fgn_horizontal_pass_sse_f32::<f32, CN>;
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
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
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let has_avx = std::arch::is_x86_feature_detected!("avx2");

            if has_avx {
                use crate::avx::fgn_vertical_pass_avx_f32;
                return fgn_vertical_pass_avx_f32::<f32, CN>;
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            let is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
            if is_sse_available {
                _dispatcher_vertical = fgn_vertical_pass_sse_f32::<f32, CN>;
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            if BASE_RADIUS_I64_CUTOFF > radius {
                _dispatcher_vertical = fgn_vertical_pass_neon_f32::<f32, CN>;
            }
        }
        _dispatcher_vertical
    }
}

#[cfg(feature = "nightly_f16")]
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
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            let is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
            let is_f16c_available = std::arch::is_x86_feature_detected!("f16c");
            if is_f16c_available && is_sse_available {
                _dispatcher_horizontal = fast_gaussian_next_horizontal_pass_sse_f16::<f16, CN>;
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _dispatcher_horizontal = fgn_horizontal_pass_neon_f16::<f16, CN>;
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
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            let _is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
            let _is_f16c_available = std::arch::is_x86_feature_detected!("f16c");
            if _is_f16c_available && _is_sse_available {
                _dispatcher_vertical = fast_gaussian_next_vertical_pass_sse_f16::<f16, CN>;
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _dispatcher_vertical = fgn_vertical_pass_neon_f16::<f16, CN>;
        }
        _dispatcher_vertical
    }
}

fn fast_gaussian_next_impl<
    T: Default
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Copy
        + PrimitiveCast<f32>
        + PrimitiveCast<f64>
        + PrimitiveCast<i64>
        + PrimitiveCast<i32>
        + FastGaussianNextPassProvider<T>,
    const CN: usize,
>(
    bytes: &mut [T],
    stride: u32,
    width: u32,
    height: u32,
    radius: AnisotropicRadius,
    threading_policy: ThreadingPolicy,
    edge_mode: EdgeMode,
) where
    i64: PrimitiveCast<T>,
    f32: PrimitiveCast<T> + ToStorage<T>,
    f64: PrimitiveCast<T> + ToStorage<T>,
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
    ) = T::get_vertical::<CN>(radius.y_axis);
    let mut _dispatcher_horizontal: fn(
        bytes: &UnsafeSlice<T>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        start: u32,
        end: u32,
        EdgeMode,
    ) = T::get_horizontal::<CN>(radius.x_axis);
    let thread_count = threading_policy.thread_count(width, height) as u32;

    let unsafe_image = UnsafeSlice::new(bytes);
    let pool = novtb::ThreadPool::new(thread_count as usize);
    pool.parallel_for(|thread_index| {
        let segment_size = width / thread_count;
        let start_x = thread_index as u32 * segment_size;
        let mut end_x = (thread_index as u32 + 1) * segment_size;
        if thread_index as u32 == thread_count - 1 {
            end_x = width;
        }
        _dispatcher_vertical(
            &unsafe_image,
            stride,
            width,
            height,
            radius.y_axis,
            start_x,
            end_x,
            edge_mode,
        );
    });
    pool.parallel_for(|thread_index| {
        let segment_size = height / thread_count;
        let start_y = thread_index as u32 * segment_size;
        let mut end_y = (thread_index as u32 + 1) * segment_size;
        if thread_index as u32 == thread_count - 1 {
            end_y = height;
        }
        _dispatcher_horizontal(
            &unsafe_image,
            stride,
            width,
            height,
            radius.x_axis,
            start_y,
            end_y,
            edge_mode,
        );
    });
}

/// Performs gaussian approximation on the image.
///
/// Fast gaussian approximation for u8 image.
/// This is also a VERY fast approximation, however producing more pleasant results than stack blu.
/// Radius is limited to 280.
/// Approximation based on binomial filter.
/// O(1) complexity.
///
/// * `image` - Image to blur in-place, see [BlurImageMut] for more info
/// * `radius` - Radius is limited to 280
/// * `threading_policy` - Threads usage policy
/// * `edge_mode` - Edge handling mode, *Kernel clip* is not supported!
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn fast_gaussian_next(
    image: &mut BlurImageMut<u8>,
    radius: AnisotropicRadius,
    threading_policy: ThreadingPolicy,
    edge_mode: EdgeMode,
) -> Result<(), BlurError> {
    image.check_layout(None)?;
    let radius = radius.clamp(1, 280);
    let stride = image.row_stride();
    let width = image.width;
    let height = image.height;
    let channels = image.channels;
    impl_margin_call!(
        u8,
        channels,
        edge_mode,
        image.data.borrow_mut(),
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
/// * `image` - Image to blur in-place, see [BlurImageMut] for more info.
/// * `radius` - Radius is limited to 152.
/// * `threading_policy` - Threads usage policy.
/// * `edge_mode` - Edge handling mode, *Kernel clip* is not supported!
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn fast_gaussian_next_u16(
    image: &mut BlurImageMut<u16>,
    radius: AnisotropicRadius,
    threading_policy: ThreadingPolicy,
    edge_mode: EdgeMode,
) -> Result<(), BlurError> {
    image.check_layout(None)?;
    let stride = image.row_stride();
    let width = image.width;
    let height = image.height;
    let channels = image.channels;
    let acq_radius = radius.clamp(1, 152);
    impl_margin_call!(
        u16,
        channels,
        edge_mode,
        image.data.borrow_mut(),
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
/// * `image` - Image to blur in-place, see [BlurImageMut] for more info.
/// * `radius` - Almost any radius is supported, in real world radius > 300 is too big for this implementation.
/// * `threading_policy` - Threads usage policy.
/// * `edge_mode` - Edge handling mode, *Kernel clip* is not supported!
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
pub fn fast_gaussian_next_f32(
    image: &mut BlurImageMut<f32>,
    radius: AnisotropicRadius,
    threading_policy: ThreadingPolicy,
    edge_mode: EdgeMode,
) -> Result<(), BlurError> {
    image.check_layout(None)?;
    let stride = image.row_stride();
    let width = image.width;
    let height = image.height;
    let channels = image.channels;
    let radius = AnisotropicRadius::create(radius.x_axis.max(1), radius.y_axis.max(1));
    impl_margin_call!(
        f32,
        channels,
        edge_mode,
        image.data.borrow_mut(),
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
/// * `image` - Image to blur in-place, see [BlurImageMut] for more info.
/// * `radius` - Almost any radius is supported, in real world radius > 300 is too big for this implementation.
/// * `threading_policy` - Threads usage policy.
/// * `edge_mode` - Edge handling mode, *Kernel clip* is not supported!.
///
/// # Panics
/// Panic is stride/width/height/channel configuration do not match provided
#[cfg(feature = "nightly_f16")]
#[cfg_attr(docsrs, doc(cfg(feature = "nightly_f16")))]
pub fn fast_gaussian_next_f16(
    in_place: &mut BlurImageMut<f16>,
    radius: AnisotropicRadius,
    threading_policy: ThreadingPolicy,
    edge_mode: EdgeMode,
) -> Result<(), BlurError> {
    in_place.check_layout(None)?;
    let channels = in_place.channels;
    let stride = in_place.row_stride();
    let width = in_place.width;
    let height = in_place.height;
    let radius = AnisotropicRadius::create(radius.x_axis.max(1), radius.y_axis.max(1));
    impl_margin_call!(
        f16,
        channels,
        edge_mode,
        unsafe { std::mem::transmute::<&mut [f16], &mut [f16]>(in_place.data.borrow_mut()) },
        stride,
        width,
        height,
        radius,
        threading_policy
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_gaussian_next_u8_q_k5() {
        let width: usize = 148;
        let height: usize = 148;
        let mut dst = vec![126; width * height * 3];
        let mut dst_image = BlurImageMut::borrow(
            &mut dst,
            width as u32,
            height as u32,
            FastBlurChannels::Channels3,
        );
        fast_gaussian_next(
            &mut dst_image,
            AnisotropicRadius::new(5),
            ThreadingPolicy::Single,
            EdgeMode::Clamp,
        )
        .unwrap();
        for (i, &cn) in dst.iter().enumerate() {
            let diff = (cn as i32 - 126).abs();
            assert!(
                diff <= 3,
                "Diff expected to be less than 3, but it was {diff} at {i}"
            );
        }
    }

    #[test]
    fn test_fast_gaussian_next_u16_fp_k25() {
        let width: usize = 148;
        let height: usize = 148;
        let mut dst = vec![17234u16; width * height * 3];
        let mut dst_image = BlurImageMut::borrow(
            &mut dst,
            width as u32,
            height as u32,
            FastBlurChannels::Channels3,
        );
        fast_gaussian_next_u16(
            &mut dst_image,
            AnisotropicRadius::new(5),
            ThreadingPolicy::Single,
            EdgeMode::Clamp,
        )
        .unwrap();
        for &cn in dst.iter() {
            let diff = (cn as i32 - 17234i32).abs();
            assert!(
                diff <= 14,
                "Diff expected to be less than 14, but it was {diff}"
            );
        }
    }

    #[test]
    fn test_fast_gaussian_next_f32_k25() {
        let width: usize = 148;
        let height: usize = 148;
        let mut dst = vec![0.432; width * height * 3];
        let mut dst_image = BlurImageMut::borrow(
            &mut dst,
            width as u32,
            height as u32,
            FastBlurChannels::Channels3,
        );
        fast_gaussian_next_f32(
            &mut dst_image,
            AnisotropicRadius::new(5),
            ThreadingPolicy::Single,
            EdgeMode::Clamp,
        )
        .unwrap();
        for &cn in dst.iter() {
            let diff = (cn - 0.432).abs();
            assert!(
                diff <= 1e-4,
                "Diff expected to be less than 1e-4, but it was {diff}"
            );
        }
    }
}
