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
use crate::mul_table::{MUL_TABLE_STACK_BLUR, SHR_TABLE_STACK_BLUR};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::{stack_blur_pass_neon_i32, stack_blur_pass_neon_i64};
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::{stack_blur_pass_sse, stack_blur_pass_sse_i64};
use crate::unsafe_slice::UnsafeSlice;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::wasm32::stack_blur_pass_wasm_i32;
use crate::{FastBlurChannels, ThreadingPolicy};
use num_traits::{AsPrimitive, FromPrimitive};
use std::ops::{AddAssign, Mul, Shr, SubAssign};

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
    pub fn cast<T>(&self) -> SlidingWindow<COMPS, T> where J: AsPrimitive<T>, T: Default + Copy + 'static {
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
    pub fn from_store<T>(store: &UnsafeSlice<T>, offset: usize) -> SlidingWindow<COMPS, J> where T: AsPrimitive<J> {
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
            SlidingWindow::from_components(self.r >> rhs, self.g >> rhs, self.b >> rhs, self.a >> rhs)
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
        + std::ops::Mul<Output = J>
        + std::ops::Shr<Output = J>
        + std::ops::SubAssign
        + AsPrimitive<T>
        + AsPrimitive<I>
        + Default,
    T: Copy + AsPrimitive<J> + FromPrimitive,
    I: Copy
        + AsPrimitive<T>
        + FromPrimitive
        + std::ops::Mul<Output = I>
        + std::ops::Shr<Output = I> + Default,
    i32: AsPrimitive<J>,
    u32: AsPrimitive<J>,
{
    let div = ((radius * 2) + 1) as usize;
    let (mut xp, mut yp);
    let mut sp;
    let mut stack_start;
    let mut stacks = vec![];
    for _ in 0..div {
        stacks.push(SlidingWindow::<COMPONENTS, J>::new());
    }

    let mut sum: SlidingWindow<COMPONENTS, J>;
    let mut sum_in: SlidingWindow<COMPONENTS, J>;
    let mut sum_out: SlidingWindow<COMPONENTS, J>;

    let wm = width - 1;
    let hm = height - 1;
    let div = (radius * 2) + 1;
    let mul_sum = I::from_i32(MUL_TABLE_STACK_BLUR[radius as usize]).unwrap();
    let shr_sum = I::from_i32(SHR_TABLE_STACK_BLUR[radius as usize]).unwrap();

    let mut src_ptr;
    let mut dst_ptr;

    if pass == StackBlurPass::Horizontal {
        let min_y = thread * height as usize / total_threads;
        let max_y = (thread + 1) * height as usize / total_threads;

        for y in min_y..max_y {
            sum = SlidingWindow::default();
            sum_in = SlidingWindow::default();
            sum_out = SlidingWindow::default();

            src_ptr = stride as usize * y; // start of line (0,y)

            let src = SlidingWindow::from_store(pixels, src_ptr);

            for i in 0..=radius {
                unsafe { *stacks.get_unchecked_mut(i as usize) = src.clone() };
                let fi = (i + 1).as_();
                sum += src * fi;
                sum_out += src;
            }

            for i in 1..=radius {
                if i <= wm {
                    src_ptr += COMPONENTS;
                }

                let src = SlidingWindow::from_store(pixels, src_ptr);

                unsafe { *stacks.get_unchecked_mut((i + radius) as usize) = src };

                let re = (radius + 1 - i).as_();
                sum += src * re;
                sum_in += src;
            }

            sp = radius;
            xp = radius;
            if xp > wm {
                xp = wm;
            }

            src_ptr = COMPONENTS * xp as usize + y * stride as usize;
            dst_ptr = y * stride as usize;
            for _ in 0..width {
                unsafe {
                    let sum_intermediate: SlidingWindow<COMPONENTS, I> = sum.cast();
                    let finalized = (sum_intermediate * mul_sum) >> shr_sum;
                    pixels.write(dst_ptr, finalized.r.as_());
                    if COMPONENTS > 1 {
                        pixels.write(dst_ptr + 1, finalized.g.as_());
                    }
                    if COMPONENTS > 2 {
                        pixels.write(dst_ptr + 2, finalized.b.as_());
                    }
                    if COMPONENTS == 4 {
                        pixels.write(dst_ptr + 3, finalized.a.as_());
                    }
                }
                dst_ptr += COMPONENTS;

                sum -= sum_out;

                stack_start = sp + div - radius;
                if stack_start >= div {
                    stack_start -= div;
                }
                let stack = unsafe { &mut *stacks.get_unchecked_mut(stack_start as usize) };

                sum_out -= *stack;

                if xp < wm {
                    src_ptr += COMPONENTS;
                    xp += 1;
                }

                let src = SlidingWindow::from_store(pixels, src_ptr);
                *stack = src;
                sum_in += src;
                sum += sum_in;

                sp += 1;
                if sp >= div {
                    sp = 0;
                }
                let stack = unsafe { &mut *stacks.get_unchecked_mut(sp as usize) };

                sum_out += *stack;
                sum_in -= *stack;
            }
        }
    } else if pass == StackBlurPass::Vertical {
        let min_x = thread * width as usize / total_threads;
        let max_x = (thread + 1) * width as usize / total_threads;

        for x in min_x..max_x {
            sum = SlidingWindow::default();
            sum_in = SlidingWindow::default();
            sum_out = SlidingWindow::default();

            src_ptr = COMPONENTS * x; // x,0

            let src = SlidingWindow::from_store(pixels, src_ptr);

            for i in 0..=radius {
                unsafe { *stacks.get_unchecked_mut(i as usize) = src }

                let ji = (i + 1).as_();
                sum += src * ji;
                sum_out += src;
            }

            for i in 1..=radius {
                if i <= hm {
                    src_ptr += stride as usize;
                }

                let src = SlidingWindow::from_store(pixels, src_ptr);

                unsafe { *stacks.get_unchecked_mut((i + radius) as usize) = src };

                let rji = (radius + 1 - i).as_();
                sum += src * rji;
                sum_in += src;
            }

            sp = radius;
            yp = radius;
            if yp > hm {
                yp = hm;
            }
            src_ptr = COMPONENTS * x + yp as usize * stride as usize;
            dst_ptr = COMPONENTS * x;
            for _ in 0..height {
                unsafe {
                    let sum_intermediate: SlidingWindow<COMPONENTS, I> = sum.cast();
                    let finalized = (sum_intermediate * mul_sum) >> shr_sum;
                    pixels.write(dst_ptr, finalized.r.as_());
                    if COMPONENTS > 1 {
                        pixels.write(dst_ptr + 1, finalized.g.as_());
                    }
                    if COMPONENTS > 2 {
                        pixels.write(dst_ptr + 2, finalized.b.as_());
                    }
                    if COMPONENTS == 4 {
                        pixels.write(dst_ptr + 3, finalized.a.as_());
                    }
                }
                dst_ptr += stride as usize;

                sum -= sum_out;

                stack_start = sp + div - radius;
                if stack_start >= div {
                    stack_start -= div;
                }
                let stack_ptr = unsafe { &mut *stacks.get_unchecked_mut(stack_start as usize) };

                sum_out -= *stack_ptr;

                if yp < hm {
                    src_ptr += stride as usize; // stride
                    yp += 1;
                }

                let src = SlidingWindow::from_store(pixels, src_ptr);

                *stack_ptr = src;

                sum_in += src;
                sum += sum_in;

                sp += 1;

                if sp >= div {
                    sp = 0;
                }
                let stack_ptr = unsafe { &mut *stacks.get_unchecked_mut(sp as usize) };

                sum_out += *stack_ptr;
                sum_in -= *stack_ptr;
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
                StackBlurPass::Horizontal,
                thread,
                thread_count,
            );
        }
        FastBlurChannels::Channels3 => {
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
                stack_blur_pass::<u8, i64, i64, 3>
            } else {
                stack_blur_pass::<u8, i32, i64, 3>
            };
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
            _dispatcher(
                slice,
                stride,
                width,
                height,
                radius,
                StackBlurPass::Horizontal,
                thread,
                thread_count,
            );
        }
        FastBlurChannels::Channels4 => {
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
                stack_blur_pass::<u8, i64, i64, 4>
            } else {
                stack_blur_pass::<u8, i32, i64, 4>
            };
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
            _dispatcher(
                slice,
                stride,
                width,
                height,
                radius,
                StackBlurPass::Horizontal,
                thread,
                thread_count,
            );
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
                stack_blur_pass::<u8, i64, i64, 3>
            } else {
                stack_blur_pass::<u8, i32, i64, 3>
            };
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
        FastBlurChannels::Channels4 => {
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
                stack_blur_pass::<u8, i64, i64, 4>
            } else {
                stack_blur_pass::<u8, i32, i64, 4>
            };
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
    let radius = radius.clamp(2, 254);
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
