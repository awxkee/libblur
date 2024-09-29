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

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::cpu_features::{is_x86_avx512dq_supported, is_x86_avx512vl_supported};
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::{stack_blur_pass_sse, stack_blur_pass_sse_i64};
use crate::stackblur::{HorizontalStackBlurPass, StackBlurWorkingPass, VerticalStackBlurPass};
use crate::unsafe_slice::UnsafeSlice;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::wasm32::stack_blur_pass_wasm_i32;
use crate::{FastBlurChannels, ThreadingPolicy};
use num_traits::{AsPrimitive, FromPrimitive};
use std::ops::{AddAssign, Mul, Shr, Sub, SubAssign};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::stackblur::neon::{HorizontalNeonStackBlurPass, VerticalNeonStackBlurPass};

const BASE_RADIUS_I64_CUTOFF: u32 = 150;

#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub(crate) struct BlurStack<J: Copy + FromPrimitive> {
    pub r: J,
    pub g: J,
    pub b: J,
    pub a: J,
}

impl<J> BlurStack<J>
where
    J: Copy + FromPrimitive + Default,
{
    #[inline]
    pub fn new() -> BlurStack<J> {
        BlurStack {
            r: J::default(),
            g: J::default(),
            b: J::default(),
            a: J::default(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Copy)]
pub(crate) struct SlidingWindow<const COMPS: usize, J: Copy> {
    pub r: J,
    pub g: J,
    pub b: J,
    pub a: J,
}

impl<const COMPS: usize, J> SlidingWindow<COMPS, J>
where
    J: Copy + Default,
{
    #[inline]
    pub fn new() -> SlidingWindow<COMPS, J> {
        SlidingWindow {
            r: J::default(),
            g: J::default(),
            b: J::default(),
            a: J::default(),
        }
    }

    #[inline]
    pub fn from_components(r: J, g: J, b: J, a: J) -> SlidingWindow<COMPS, J> {
        SlidingWindow { r, g, b, a }
    }

    #[inline]
    pub fn cast<T>(&self) -> SlidingWindow<COMPS, T>
    where
        J: AsPrimitive<T>,
        T: Default + Copy + 'static,
    {
        if COMPS == 1 {
            SlidingWindow::from_components(self.r.as_(), T::default(), T::default(), T::default())
        } else if COMPS == 2 {
            SlidingWindow::from_components(self.r.as_(), self.g.as_(), T::default(), T::default())
        } else if COMPS == 3 {
            SlidingWindow::from_components(self.r.as_(), self.g.as_(), self.b.as_(), T::default())
        } else if COMPS == 4 {
            SlidingWindow::from_components(self.r.as_(), self.g.as_(), self.b.as_(), self.a.as_())
        } else {
            panic!("Not implemented.");
        }
    }
}

impl<const COMPS: usize, J> SlidingWindow<COMPS, J>
where
    J: Copy + FromPrimitive + Default + 'static,
{
    #[inline]
    pub fn from_store<T>(store: &UnsafeSlice<T>, offset: usize) -> SlidingWindow<COMPS, J>
    where
        T: AsPrimitive<J>,
    {
        if COMPS == 1 {
            SlidingWindow {
                r: (*store.get(offset)).as_(),
                g: J::default(),
                b: J::default(),
                a: J::default(),
            }
        } else if COMPS == 2 {
            SlidingWindow {
                r: (*store.get(offset)).as_(),
                g: (*store.get(offset + 1)).as_(),
                b: J::default(),
                a: J::default(),
            }
        } else if COMPS == 3 {
            SlidingWindow {
                r: (*store.get(offset)).as_(),
                g: (*store.get(offset + 1)).as_(),
                b: (*store.get(offset + 2)).as_(),
                a: J::default(),
            }
        } else if COMPS == 4 {
            SlidingWindow {
                r: (*store.get(offset)).as_(),
                g: (*store.get(offset + 1)).as_(),
                b: (*store.get(offset + 2)).as_(),
                a: (*store.get(offset + 3)).as_(),
            }
        } else {
            panic!("Not implemented.")
        }
    }
}

impl<const COMPS: usize, J> Mul<J> for SlidingWindow<COMPS, J>
where
    J: Copy + Mul<Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: J) -> Self::Output {
        if COMPS == 1 {
            SlidingWindow::from_components(self.r * rhs, self.g, self.b, self.a)
        } else if COMPS == 2 {
            SlidingWindow::from_components(self.r * rhs, self.g * rhs, self.b, self.a)
        } else if COMPS == 3 {
            SlidingWindow::from_components(self.r * rhs, self.g * rhs, self.b * rhs, self.a)
        } else if COMPS == 4 {
            SlidingWindow::from_components(self.r * rhs, self.g * rhs, self.b * rhs, self.a * rhs)
        } else {
            panic!("Not implemented.");
        }
    }
}

impl<const COMPS: usize, J> Sub<J> for SlidingWindow<COMPS, J>
where
    J: Copy + Sub<Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: J) -> Self::Output {
        if COMPS == 1 {
            SlidingWindow::from_components(self.r - rhs, self.g, self.b, self.a)
        } else if COMPS == 2 {
            SlidingWindow::from_components(self.r - rhs, self.g - rhs, self.b, self.a)
        } else if COMPS == 3 {
            SlidingWindow::from_components(self.r - rhs, self.g - rhs, self.b - rhs, self.a)
        } else if COMPS == 4 {
            SlidingWindow::from_components(self.r - rhs, self.g - rhs, self.b - rhs, self.a - rhs)
        } else {
            panic!("Not implemented.");
        }
    }
}

impl<const COMPS: usize, J> Sub<SlidingWindow<COMPS, J>> for SlidingWindow<COMPS, J>
where
    J: Copy + Sub<Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: SlidingWindow<COMPS, J>) -> Self::Output {
        if COMPS == 1 {
            SlidingWindow::from_components(self.r - rhs.r, self.g, self.b, self.a)
        } else if COMPS == 2 {
            SlidingWindow::from_components(self.r - rhs.r, self.g - rhs.g, self.b, self.a)
        } else if COMPS == 3 {
            SlidingWindow::from_components(self.r - rhs.r, self.g - rhs.g, self.b - rhs.b, self.a)
        } else if COMPS == 4 {
            SlidingWindow::from_components(
                self.r - rhs.r,
                self.g - rhs.g,
                self.b - rhs.b,
                self.a - rhs.a,
            )
        } else {
            panic!("Not implemented.");
        }
    }
}

impl<const COMPS: usize, J> Shr<J> for SlidingWindow<COMPS, J>
where
    J: Copy + Shr<J, Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline]
    fn shr(self, rhs: J) -> Self::Output {
        if COMPS == 1 {
            SlidingWindow::from_components(self.r >> rhs, self.g, self.b, self.a)
        } else if COMPS == 2 {
            SlidingWindow::from_components(self.r >> rhs, self.g >> rhs, self.b, self.a)
        } else if COMPS == 3 {
            SlidingWindow::from_components(self.r >> rhs, self.g >> rhs, self.b >> rhs, self.a)
        } else if COMPS == 4 {
            SlidingWindow::from_components(
                self.r >> rhs,
                self.g >> rhs,
                self.b >> rhs,
                self.a >> rhs,
            )
        } else {
            panic!("Not implemented.");
        }
    }
}

impl<const COMPS: usize, J> AddAssign<SlidingWindow<COMPS, J>> for SlidingWindow<COMPS, J>
where
    J: Copy + AddAssign,
{
    #[inline]
    fn add_assign(&mut self, rhs: SlidingWindow<COMPS, J>) {
        if COMPS == 1 {
            self.r += rhs.r;
        } else if COMPS == 2 {
            self.r += rhs.r;
            self.g += rhs.g;
        } else if COMPS == 3 {
            self.r += rhs.r;
            self.g += rhs.g;
            self.b += rhs.b;
        } else if COMPS == 4 {
            self.r += rhs.r;
            self.g += rhs.g;
            self.b += rhs.b;
            self.a += rhs.a;
        }
    }
}

impl<const COMPS: usize, J> SubAssign<SlidingWindow<COMPS, J>> for SlidingWindow<COMPS, J>
where
    J: Copy + SubAssign,
{
    #[inline]
    fn sub_assign(&mut self, rhs: SlidingWindow<COMPS, J>) {
        if COMPS == 1 {
            self.r -= rhs.r;
        } else if COMPS == 2 {
            self.r -= rhs.r;
            self.g -= rhs.g;
        } else if COMPS == 3 {
            self.r -= rhs.r;
            self.g -= rhs.g;
            self.b -= rhs.b;
        } else if COMPS == 4 {
            self.r -= rhs.r;
            self.g -= rhs.g;
            self.b -= rhs.b;
            self.a -= rhs.a;
        }
    }
}

impl<const COMPS: usize, J> Default for SlidingWindow<COMPS, J>
where
    J: Copy + FromPrimitive + Default,
{
    #[inline]
    fn default() -> Self {
        SlidingWindow::new()
    }
}

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub(crate) enum StackBlurPass {
    Horizontal,
    Vertical,
}

///
///
/// # Generics
/// `T` - buffer type u8, u16 etc, this method expected only integral types
/// `J` - accumulator type, i32, i64
/// `I` - intermediate multiplication type, when sum will be adopting into higher it may overflow, use this parameter to control overflowing
#[allow(clippy::too_many_arguments)]
fn stack_blur_pass<T, J, I, const COMPONENTS: usize>(
    pixels: &UnsafeSlice<T>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    pass: StackBlurPass,
    thread: usize,
    total_threads: usize,
) where
    J: Copy
        + 'static
        + FromPrimitive
        + AddAssign<J>
        + Mul<Output = J>
        + Shr<Output = J>
        + Sub<Output = J>
        + AsPrimitive<f32>
        + SubAssign
        + AsPrimitive<T>
        + AsPrimitive<I>
        + Default,
    T: Copy + AsPrimitive<J> + FromPrimitive,
    I: Copy + AsPrimitive<T> + FromPrimitive + Mul<Output = I> + Shr<Output = I> + Default,
    i32: AsPrimitive<J>,
    u32: AsPrimitive<J>,
    f32: AsPrimitive<T>,
    usize: AsPrimitive<J>,
{
    let div = ((radius * 2) + 1) as usize;
    let kernel_size = 2 * radius as usize + 1;
    let mut stacks = vec![];
    for _ in 0..div {
        stacks.push(SlidingWindow::<COMPONENTS, J>::new());
    }

    let mut sum: SlidingWindow<COMPONENTS, J>;

    let wm = width - 1;
    let hm = height - 1;

    let mul_value = 1. / ((radius as f32 + 1.) * (radius as f32 + 1.));

    if pass == StackBlurPass::Horizontal {
        let min_y = thread * height as usize / total_threads;
        let max_y = (thread + 1) * height as usize / total_threads;

        let mut diff =
            vec![SlidingWindow::<COMPONENTS, J>::default(); width as usize + kernel_size];

        for y in min_y..max_y {
            let y_offset = y * stride as usize;

            let mut diff_val = SlidingWindow::<COMPONENTS, J>::default();
            let radius_mul = (radius + 2) * (radius + 1) / 2;
            let zero_pos_value = SlidingWindow::<COMPONENTS, J>::from_store(pixels, y_offset);
            sum = zero_pos_value * radius_mul.as_();

            let mut loading_pos = y_offset;

            let mut differences_iter = 0usize;

            for i in 0..radius as usize {
                if i < wm as usize {
                    loading_pos += COMPONENTS;
                }
                let c_val = SlidingWindow::<COMPONENTS, J>::from_store(pixels, loading_pos);
                let new_diff = c_val - zero_pos_value;
                unsafe {
                    *diff.get_unchecked_mut(differences_iter) = new_diff;
                }
                diff_val += new_diff;
                sum += c_val * (radius as usize - i).as_();
                differences_iter += 1;
            }

            let full_offset = (radius + 1) as usize;

            let mut max_to_the_right = (width as usize).saturating_sub(full_offset);

            for i in 0..max_to_the_right {
                let v_offset_0 = y_offset + i * COMPONENTS;
                let v_offset_1 = y_offset + (i + full_offset) * COMPONENTS;
                let new_diff = SlidingWindow::<COMPONENTS, J>::from_store(pixels, v_offset_1)
                    - SlidingWindow::<COMPONENTS, J>::from_store(pixels, v_offset_0);
                unsafe {
                    *diff.get_unchecked_mut(differences_iter) = new_diff;
                }
                differences_iter += 1;
            }

            let v_last_offset = y_offset + (width as usize - 1) * COMPONENTS;
            let last = SlidingWindow::<COMPONENTS, J>::from_store(pixels, v_last_offset);

            for i in max_to_the_right..width as usize {
                let v_offset = y_offset + i * COMPONENTS;
                unsafe {
                    *diff.get_unchecked_mut(differences_iter) =
                        last - SlidingWindow::<COMPONENTS, J>::from_store(pixels, v_offset);
                }
                differences_iter += 1;
            }

            diff_val += unsafe { *diff.get_unchecked(radius as usize) };

            let mut diff_start_end = radius as usize + 1;
            let mut diff_start = 0usize;

            for i in 0..width as usize {
                unsafe {
                    let sum_intermediate: SlidingWindow<COMPONENTS, f32> = sum.cast();
                    let dst_ptr = y_offset + i * COMPONENTS;
                    pixels.write(dst_ptr, (sum_intermediate.r * mul_value).as_());
                    if COMPONENTS > 1 {
                        pixels.write(dst_ptr + 1, (sum_intermediate.g * mul_value).as_());
                    }
                    if COMPONENTS > 2 {
                        pixels.write(dst_ptr + 2, (sum_intermediate.b * mul_value).as_());
                    }
                    if COMPONENTS == 4 {
                        pixels.write(dst_ptr + 3, (sum_intermediate.a * mul_value).as_());
                    }
                }

                sum += diff_val;

                unsafe {
                    diff_val +=
                        *diff.get_unchecked(diff_start_end) - *diff.get_unchecked(diff_start);
                }

                diff_start_end += 1;
                diff_start += 1;
            }
        }
    } else if pass == StackBlurPass::Vertical {
        let min_x = thread * width as usize / total_threads;
        let max_x = (thread + 1) * width as usize / total_threads;

        let mut diff =
            vec![SlidingWindow::<COMPONENTS, J>::default(); height as usize + kernel_size];

        for x in min_x..max_x {
            let mut diff_val = SlidingWindow::<COMPONENTS, J>::default();
            let radius_mul = (radius + 2) * (radius + 1) / 2;
            let zero_pos_value = SlidingWindow::<COMPONENTS, J>::from_store(pixels, x * COMPONENTS);
            sum = zero_pos_value * radius_mul.as_();

            let mut loading_pos = 0usize + x * COMPONENTS;

            let mut differences_iter = 0usize;

            for i in 0..radius as usize {
                if i < hm as usize {
                    loading_pos += stride as usize;
                }
                let c_val = SlidingWindow::<COMPONENTS, J>::from_store(pixels, loading_pos);
                let new_diff = c_val - zero_pos_value;
                unsafe {
                    *diff.get_unchecked_mut(differences_iter) = new_diff;
                }
                diff_val += new_diff;
                sum += c_val * (radius as usize - i).as_();
                differences_iter += 1;
            }

            let full_offset = (radius + 1) as usize;

            let mut max_to_the_bottom = (height as usize).saturating_sub(full_offset);

            for i in 0..max_to_the_bottom {
                let v_offset_0 = i * stride as usize + x * COMPONENTS;
                let v_offset_1 = (i + full_offset) * stride as usize + x * COMPONENTS;
                let new_diff = SlidingWindow::<COMPONENTS, J>::from_store(pixels, v_offset_1)
                    - SlidingWindow::<COMPONENTS, J>::from_store(pixels, v_offset_0);
                unsafe {
                    *diff.get_unchecked_mut(differences_iter) = new_diff;
                }
                differences_iter += 1;
            }

            let v_last_offset = (height as usize - 1) * stride as usize + x * COMPONENTS;
            let last = SlidingWindow::<COMPONENTS, J>::from_store(pixels, v_last_offset);

            for i in max_to_the_bottom..height as usize {
                let v_offset = i * stride as usize + x * COMPONENTS;
                unsafe {
                    *diff.get_unchecked_mut(differences_iter) =
                        last - SlidingWindow::<COMPONENTS, J>::from_store(pixels, v_offset);
                }
                differences_iter += 1;
            }

            diff_val += unsafe { *diff.get_unchecked(radius as usize) };

            let mut diff_start_end = radius as usize + 1;
            let mut diff_start = 0usize;

            for i in 0..height as usize {
                unsafe {
                    let sum_intermediate: SlidingWindow<COMPONENTS, f32> = sum.cast();
                    let dst_ptr = i * stride as usize + x * COMPONENTS;
                    pixels.write(dst_ptr, (sum_intermediate.r * mul_value).as_());
                    if COMPONENTS > 1 {
                        pixels.write(dst_ptr + 1, (sum_intermediate.g * mul_value).as_());
                    }
                    if COMPONENTS > 2 {
                        pixels.write(dst_ptr + 2, (sum_intermediate.b * mul_value).as_());
                    }
                    if COMPONENTS == 4 {
                        pixels.write(dst_ptr + 3, (sum_intermediate.a * mul_value).as_());
                    }
                }

                sum += diff_val;

                unsafe {
                    diff_val +=
                        *diff.get_unchecked(diff_start_end) - *diff.get_unchecked(diff_start);
                }

                diff_start_end += 1;
                diff_start += 1;
            }
        }
    }
}

fn stack_blur_worker_horizontal(
    slice: &UnsafeSlice<u8>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
    thread: usize,
    thread_count: usize,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let _is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let _is_avx512dq_available = is_x86_avx512dq_supported();
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let _is_avx512vl_available = is_x86_avx512vl_supported();
    match channels {
        FastBlurChannels::Plane => {
            let executor = Box::new(HorizontalStackBlurPass::<u8, i32, 1>::default());
            executor.pass(slice, stride, width, height, radius, thread, thread_count);
            // let mut _dispatcher: fn(
            //     &UnsafeSlice<u8>,
            //     u32,
            //     u32,
            //     u32,
            //     u32,
            //     StackBlurPass,
            //     usize,
            //     usize,
            // ) = if radius < BASE_RADIUS_I64_CUTOFF {
            //     stack_blur_pass::<u8, i64, i64, 1>
            // } else {
            //     stack_blur_pass::<u8, i32, i64, 1>
            // };
            // _dispatcher(
            //     slice,
            //     stride,
            //     width,
            //     height,
            //     radius,
            //     StackBlurPass::Horizontal,
            //     thread,
            //     thread_count,
            // );
        }
        FastBlurChannels::Channels3 => {
            // let mut _dispatcher: fn(
            //     &UnsafeSlice<u8>,
            //     u32,
            //     u32,
            //     u32,
            //     u32,
            //     StackBlurPass,
            //     usize,
            //     usize,
            // ) = if radius < BASE_RADIUS_I64_CUTOFF {
            //     stack_blur_pass::<u8, i32, i64, 3>
            // } else {
            //     stack_blur_pass::<u8, i32, i64, 3>
            // };

            // if radius < BASE_RADIUS_I64_CUTOFF {
            //     #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            //     {
            //         _dispatcher = stack_blur_pass_neon_i32::<3>;
            //     }
            //     #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            //     {
            //         if _is_sse_available {
            //             if _is_avx512dq_available && _is_avx512vl_available {
            //                 _dispatcher = stack_blur_pass_sse::<3, true>;
            //             } else {
            //                 _dispatcher = stack_blur_pass_sse::<3, false>;
            //             }
            //         }
            //     }
            //     #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            //     {
            //         _dispatcher = stack_blur_pass_wasm_i32::<3>;
            //     }
            // } else {
            //     #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            //     {
            //         _dispatcher = stack_blur_pass_neon_i64::<3>;
            //     }
            //     #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            //     {
            //         if _is_sse_available {
            //             if _is_avx512dq_available && _is_avx512vl_available {
            //                 _dispatcher = stack_blur_pass_sse_i64::<3, true>;
            //             } else {
            //                 _dispatcher = stack_blur_pass_sse_i64::<3, false>;
            //             }
            //         }
            //     }
            // }
            // _dispatcher(
            //     slice,
            //     stride,
            //     width,
            //     height,
            //     radius,
            //     StackBlurPass::Horizontal,
            //     thread,
            //     thread_count,
            // );
            let mut _executor: Box<dyn StackBlurWorkingPass<u8, i32, 3>> = Box::new(HorizontalStackBlurPass::<u8, i32, 3>::default());
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                _executor = Box::new(HorizontalNeonStackBlurPass::<u8, i32, 3>::default());
            }
            _executor.pass(slice, stride, width, height, radius, thread, thread_count);
        }
        FastBlurChannels::Channels4 => {
            // let mut _dispatcher: fn(
            //     &UnsafeSlice<u8>,
            //     u32,
            //     u32,
            //     u32,
            //     u32,
            //     StackBlurPass,
            //     usize,
            //     usize,
            // ) = if radius < BASE_RADIUS_I64_CUTOFF {
            //     stack_blur_pass::<u8, i64, i64, 4>
            // } else {
            //     stack_blur_pass::<u8, i32, i64, 4>
            // };
            // if radius < BASE_RADIUS_I64_CUTOFF {
            //     #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            //     {
            //         _dispatcher = stack_blur_pass_neon_i32::<4>;
            //     }
            //     #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            //     {
            //         if _is_sse_available {
            //             if _is_avx512dq_available && _is_avx512vl_available {
            //                 _dispatcher = stack_blur_pass_sse::<4, true>;
            //             } else {
            //                 _dispatcher = stack_blur_pass_sse::<4, false>;
            //             }
            //         }
            //     }
            //     #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            //     {
            //         _dispatcher = stack_blur_pass_wasm_i32::<4>;
            //     }
            // } else {
            //     #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            //     {
            //         _dispatcher = stack_blur_pass_neon_i64::<4>;
            //     }
            //     #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            //     {
            //         if _is_sse_available {
            //             if _is_avx512dq_available && _is_avx512vl_available {
            //                 _dispatcher = stack_blur_pass_sse_i64::<4, true>;
            //             } else {
            //                 _dispatcher = stack_blur_pass_sse_i64::<4, false>;
            //             }
            //         }
            //     }
            // }
            // _dispatcher(
            //     slice,
            //     stride,
            //     width,
            //     height,
            //     radius,
            //     StackBlurPass::Horizontal,
            //     thread,
            //     thread_count,
            // );
            let mut _executor: Box<dyn StackBlurWorkingPass<u8, i32, 4>> = Box::new(HorizontalStackBlurPass::<u8, i32, 4>::default());
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                _executor = Box::new(HorizontalNeonStackBlurPass::<u8, i32, 4>::default());
            }
            _executor.pass(slice, stride, width, height, radius, thread, thread_count);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn stack_blur_worker_vertical(
    slice: &UnsafeSlice<u8>,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
    thread: usize,
    thread_count: usize,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let _is_sse_available = std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let _is_avx512dq_available = is_x86_avx512dq_supported();
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let _is_avx512vl_available = is_x86_avx512vl_supported();
    match channels {
        FastBlurChannels::Plane => {
            let mut _dispatcher: fn(
                &UnsafeSlice<u8>,
                u32,
                u32,
                u32,
                u32,
                StackBlurPass,
                usize,
                usize,
            ) = if radius < BASE_RADIUS_I64_CUTOFF {
                stack_blur_pass::<u8, i64, i64, 1>
            } else {
                stack_blur_pass::<u8, i32, i64, 1>
            };
            _dispatcher(
                slice,
                stride,
                width,
                height,
                radius,
                StackBlurPass::Vertical,
                thread,
                thread_count,
            );
        }
        FastBlurChannels::Channels3 => {
            // let mut _dispatcher: fn(
            //     &UnsafeSlice<u8>,
            //     u32,
            //     u32,
            //     u32,
            //     u32,
            //     StackBlurPass,
            //     usize,
            //     usize,
            // ) = if radius < BASE_RADIUS_I64_CUTOFF {
            //     stack_blur_pass::<u8, i64, i64, 3>
            // } else {
            //     stack_blur_pass::<u8, i32, i64, 3>
            // };
            // if radius < BASE_RADIUS_I64_CUTOFF {
            //     #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            //     {
            //         _dispatcher = stack_blur_pass_neon_i32::<3>;
            //     }
            //     #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            //     {
            //         if _is_sse_available {
            //             if _is_avx512dq_available && _is_avx512vl_available {
            //                 _dispatcher = stack_blur_pass_sse::<3, true>;
            //             } else {
            //                 _dispatcher = stack_blur_pass_sse::<3, false>;
            //             }
            //         }
            //     }
            //     #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            //     {
            //         _dispatcher = stack_blur_pass_wasm_i32::<3>;
            //     }
            // } else {
            //     #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            //     {
            //         _dispatcher = stack_blur_pass_neon_i64::<3>;
            //     }
            //     #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            //     {
            //         if _is_sse_available {
            //             if _is_avx512dq_available && _is_avx512vl_available {
            //                 _dispatcher = stack_blur_pass_sse_i64::<3, true>;
            //             } else {
            //                 _dispatcher = stack_blur_pass_sse_i64::<3, false>;
            //             }
            //         }
            //     }
            // }
            // _dispatcher(
            //     slice,
            //     stride,
            //     width,
            //     height,
            //     radius,
            //     StackBlurPass::Vertical,
            //     thread,
            //     thread_count,
            // );
            let mut _executor: Box<dyn StackBlurWorkingPass<u8, i32, 3>> = Box::new(VerticalStackBlurPass::<u8, i32, 3>::default());
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                _executor = Box::new(VerticalNeonStackBlurPass::<u8, i32, 3>::default());
            }
            _executor.pass(slice, stride, width, height, radius, thread, thread_count);
        }
        FastBlurChannels::Channels4 => {
            // let mut _dispatcher: fn(
            //     &UnsafeSlice<u8>,
            //     u32,
            //     u32,
            //     u32,
            //     u32,
            //     StackBlurPass,
            //     usize,
            //     usize,
            // ) = if radius < BASE_RADIUS_I64_CUTOFF {
            //     stack_blur_pass::<u8, i64, i64, 4>
            // } else {
            //     stack_blur_pass::<u8, i32, i64, 4>
            // };
            // if radius < BASE_RADIUS_I64_CUTOFF {
            //     #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            //     {
            //         _dispatcher = stack_blur_pass_neon_i32::<4>;
            //     }
            //     #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            //     {
            //         if _is_sse_available {
            //             if _is_avx512dq_available && _is_avx512vl_available {
            //                 _dispatcher = stack_blur_pass_sse::<4, true>;
            //             } else {
            //                 _dispatcher = stack_blur_pass_sse::<4, false>;
            //             }
            //         }
            //     }
            //     #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            //     {
            //         _dispatcher = stack_blur_pass_wasm_i32::<3>;
            //     }
            // } else {
            //     #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            //     {
            //         _dispatcher = stack_blur_pass_neon_i64::<4>;
            //     }
            //     #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            //     {
            //         if _is_sse_available {
            //             if _is_avx512dq_available && _is_avx512vl_available {
            //                 _dispatcher = stack_blur_pass_sse_i64::<4, true>;
            //             } else {
            //                 _dispatcher = stack_blur_pass_sse_i64::<4, false>;
            //             }
            //         }
            //     }
            // }
            // _dispatcher(
            //     slice,
            //     stride,
            //     width,
            //     height,
            //     radius,
            //     StackBlurPass::Vertical,
            //     thread,
            //     thread_count,
            // );
            let mut _executor: Box<dyn StackBlurWorkingPass<u8, i32, 4>> = Box::new(VerticalStackBlurPass::<u8, i32, 4>::default());
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                _executor = Box::new(VerticalNeonStackBlurPass::<u8, i32, 4>::default());
            }
            _executor.pass(slice, stride, width, height, radius, thread, thread_count);
        }
    }
}

/// Fastest available blur option
///
/// Fast gaussian approximation using stack blur
/// This is a very fast approximation using i32 accumulator size with radius less that *BASE_RADIUS_I64_CUTOFF*,
/// after it to avoid overflowing fallback to i64 accumulator will be used with some computational slowdown with factor ~1.5-2
///
/// # Arguments
/// * `in_place` - mutable buffer contains image data that will be used as a source and destination
/// * `stride` - Bytes per lane, default is width * channels_count if not aligned
/// * `width` - image width
/// * `height` - image height
/// * `radius` - radius is limited into 2..254
/// * `channels` - Count of channels of the image, only 3 and 4 is supported, alpha position, and channels order does not matter
/// * `threading_policy` - Threads usage policy
///
/// # Complexity
/// O(1) complexity.
pub fn stack_blur(
    in_place: &mut [u8],
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
    threading_policy: ThreadingPolicy,
) {
    // let radius = radius.clamp(2, 254);
    let thread_count = threading_policy.get_threads_count(width, height) as u32;
    if thread_count == 1 {
        let slice = UnsafeSlice::new(in_place);
        stack_blur_worker_horizontal(&slice, stride, width, height, radius, channels, 0, 1);
        stack_blur_worker_vertical(&slice, stride, width, height, radius, channels, 0, 1);
        return;
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count as usize)
        .build()
        .unwrap();
    pool.scope(|scope| {
        let slice = UnsafeSlice::new(in_place);
        for i in 0..thread_count {
            scope.spawn(move |_| {
                stack_blur_worker_horizontal(
                    &slice,
                    stride,
                    width,
                    height,
                    radius,
                    channels,
                    i as usize,
                    thread_count as usize,
                );
            });
        }
    });
    pool.scope(|scope| {
        let slice = UnsafeSlice::new(in_place);
        for i in 0..thread_count {
            scope.spawn(move |_| {
                stack_blur_worker_vertical(
                    &slice,
                    stride,
                    width,
                    height,
                    radius,
                    channels,
                    i as usize,
                    thread_count as usize,
                );
            });
        }
    })
}
