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
use std::ops::AddAssign;

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
    J: Copy + FromPrimitive,
{
    pub fn new() -> BlurStack<J> {
        BlurStack {
            r: J::from_i32(0i32).unwrap(),
            g: J::from_i32(0i32).unwrap(),
            b: J::from_i32(0i32).unwrap(),
            a: J::from_i32(0i32).unwrap(),
        }
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
        + AsPrimitive<I>,
    T: Copy + AsPrimitive<J> + FromPrimitive,
    I: Copy
        + AsPrimitive<T>
        + FromPrimitive
        + std::ops::Mul<Output = I>
        + std::ops::Shr<Output = I>,
    i32: AsPrimitive<J>,
    u32: AsPrimitive<J>,
{
    let div = ((radius * 2) + 1) as usize;
    let (mut xp, mut yp);
    let mut sp;
    let mut stack_start;
    let mut stacks = vec![];
    for _ in 0..div {
        stacks.push(Box::new(BlurStack::new()));
    }

    let mut sum_r: J;
    let mut sum_g: J;
    let mut sum_b: J;
    let mut sum_a: J;
    let mut sum_in_r: J;
    let mut sum_in_g: J;
    let mut sum_in_b: J;
    let mut sum_in_a: J;
    let mut sum_out_r: J;
    let mut sum_out_g: J;
    let mut sum_out_b: J;
    let mut sum_out_a: J;

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
            sum_r = 0i32.as_();
            sum_g = 0i32.as_();
            sum_b = 0i32.as_();
            sum_a = 0i32.as_();
            sum_in_r = 0i32.as_();
            sum_in_g = 0i32.as_();
            sum_in_b = 0i32.as_();
            sum_in_a = 0i32.as_();
            sum_out_r = 0i32.as_();
            sum_out_g = 0i32.as_();
            sum_out_b = 0i32.as_();
            sum_out_a = 0i32.as_();

            src_ptr = stride as usize * y; // start of line (0,y)

            let src_r = pixels[src_ptr].as_();
            let src_g = if COMPONENTS > 1 {
                pixels[src_ptr + 1].as_()
            } else {
                0i32.as_()
            };
            let src_b = if COMPONENTS > 2 {
                pixels[src_ptr + 2].as_()
            } else {
                0i32.as_()
            };
            let src_a = if COMPONENTS == 4 {
                pixels[src_ptr + 3].as_()
            } else {
                0i32.as_()
            };

            for i in 0..=radius {
                let stack_value = unsafe { &mut *stacks.get_unchecked_mut(i as usize) };
                stack_value.r = src_r;
                if COMPONENTS > 1 {
                    stack_value.g = src_g;
                }
                if COMPONENTS > 2 {
                    stack_value.b = src_b;
                }
                if COMPONENTS == 4 {
                    stack_value.a = src_a;
                }

                let fi = (i + 1).as_();
                sum_r += src_r * fi;
                if COMPONENTS > 1 {
                    sum_g += src_g * fi;
                }
                if COMPONENTS > 2 {
                    sum_b += src_b * fi;
                }
                if COMPONENTS == 4 {
                    sum_a += src_a * fi;
                }

                sum_out_r += src_r;
                if COMPONENTS > 1 {
                    sum_out_g += src_g;
                }
                if COMPONENTS > 2 {
                    sum_out_b += src_b;
                }
                if COMPONENTS == 4 {
                    sum_out_a += src_a;
                }
            }

            for i in 1..=radius {
                if i <= wm {
                    src_ptr += COMPONENTS;
                }
                let stack_ptr = unsafe { &mut *stacks.get_unchecked_mut((i + radius) as usize) };

                let src_r = pixels[src_ptr].as_();
                let src_g = if COMPONENTS > 1 {
                    pixels[src_ptr + 1].as_()
                } else {
                    0i32.as_()
                };
                let src_b = if COMPONENTS > 2 {
                    pixels[src_ptr + 2].as_()
                } else {
                    0i32.as_()
                };
                let src_a = if COMPONENTS == 4 {
                    pixels[src_ptr + 3].as_()
                } else {
                    0i32.as_()
                };

                stack_ptr.r = src_r;
                if COMPONENTS > 1 {
                    stack_ptr.g = src_g;
                }
                if COMPONENTS > 2 {
                    stack_ptr.b = src_b;
                }
                if COMPONENTS == 4 {
                    stack_ptr.a = src_a;
                }

                let re = (radius + 1 - i).as_();
                sum_r += src_r * re;
                if COMPONENTS > 1 {
                    sum_g += src_g * re;
                }
                if COMPONENTS > 2 {
                    sum_b += src_b * re;
                }
                if COMPONENTS == 4 {
                    sum_a += src_a * re;
                }

                sum_in_r += src_r;
                if COMPONENTS > 1 {
                    sum_in_g += src_g;
                }
                if COMPONENTS > 2 {
                    sum_in_b += src_b;
                }
                if COMPONENTS == 4 {
                    sum_in_a += src_a;
                }
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
                    let sum_r_i64: I = sum_r.as_();
                    pixels.write(dst_ptr, ((sum_r_i64 * mul_sum) >> shr_sum).as_());
                    if COMPONENTS > 1 {
                        let sum_g_i64: I = sum_g.as_();
                        pixels.write(dst_ptr + 1, ((sum_g_i64 * mul_sum) >> shr_sum).as_());
                    }
                    if COMPONENTS > 2 {
                        let sum_b_i64: I = sum_b.as_();
                        pixels.write(dst_ptr + 2, ((sum_b_i64 * mul_sum) >> shr_sum).as_());
                    }
                    if COMPONENTS == 4 {
                        let sum_a_i64: I = sum_a.as_();
                        pixels.write(dst_ptr + 3, ((sum_a_i64 * mul_sum) >> shr_sum).as_());
                    }
                }
                dst_ptr += COMPONENTS;

                sum_r -= sum_out_r;
                if COMPONENTS > 1 {
                    sum_g -= sum_out_g;
                }
                if COMPONENTS > 2 {
                    sum_b -= sum_out_b;
                }
                if COMPONENTS == 4 {
                    sum_a -= sum_out_a;
                }

                stack_start = sp + div - radius;
                if stack_start >= div {
                    stack_start -= div;
                }
                let stack = unsafe { &mut *stacks.get_unchecked_mut(stack_start as usize) };

                sum_out_r -= stack.r;
                if COMPONENTS > 1 {
                    sum_out_g -= stack.g;
                }
                if COMPONENTS > 2 {
                    sum_out_b -= stack.b;
                }
                if COMPONENTS == 4 {
                    sum_out_a -= stack.a;
                }

                if xp < wm {
                    src_ptr += COMPONENTS;
                    xp += 1;
                }

                let src_r = pixels[src_ptr].as_();
                let src_g = if COMPONENTS > 1 {
                    pixels[src_ptr + 1].as_()
                } else {
                    0i32.as_()
                };
                let src_b = if COMPONENTS > 2 {
                    pixels[src_ptr + 2].as_()
                } else {
                    0i32.as_()
                };
                let src_a = if COMPONENTS == 4 {
                    pixels[src_ptr + 3].as_()
                } else {
                    0i32.as_()
                };

                stack.r = src_r;
                if COMPONENTS > 1 {
                    stack.g = src_g;
                }
                if COMPONENTS > 2 {
                    stack.b = src_b;
                }
                if COMPONENTS == 4 {
                    stack.a = src_a;
                }

                sum_in_r += src_r;
                if COMPONENTS > 1 {
                    sum_in_g += src_g;
                }
                if COMPONENTS > 2 {
                    sum_in_b += src_b;
                }
                if COMPONENTS == 4 {
                    sum_in_a += src_a;
                }

                sum_r += sum_in_r;
                if COMPONENTS > 1 {
                    sum_g += sum_in_g;
                }
                if COMPONENTS > 2 {
                    sum_b += sum_in_b;
                }
                if COMPONENTS == 4 {
                    sum_a += sum_in_a;
                }

                sp += 1;
                if sp >= div {
                    sp = 0;
                }
                let stack = unsafe { &mut *stacks.get_unchecked_mut(sp as usize) };

                sum_out_r += stack.r;
                if COMPONENTS > 1 {
                    sum_out_g += stack.g;
                }
                if COMPONENTS > 2 {
                    sum_out_b += stack.b;
                }
                if COMPONENTS == 4 {
                    sum_out_a += stack.a;
                }
                sum_in_r -= stack.r;
                if COMPONENTS > 1 {
                    sum_in_g -= stack.g;
                }
                if COMPONENTS > 2 {
                    sum_in_b -= stack.b;
                }
                if COMPONENTS == 4 {
                    sum_in_a -= stack.a;
                }
            }
        }
    } else if pass == StackBlurPass::Vertical {
        let min_x = thread * width as usize / total_threads;
        let max_x = (thread + 1) * width as usize / total_threads;

        for x in min_x..max_x {
            sum_r = 0i32.as_();
            sum_g = 0i32.as_();
            sum_b = 0i32.as_();
            sum_a = 0i32.as_();
            sum_in_r = 0i32.as_();
            sum_in_g = 0i32.as_();
            sum_in_b = 0i32.as_();
            sum_in_a = 0i32.as_();
            sum_out_r = 0i32.as_();
            sum_out_g = 0i32.as_();
            sum_out_b = 0i32.as_();
            sum_out_a = 0i32.as_();

            src_ptr = COMPONENTS * x; // x,0

            let src_r = pixels[src_ptr].as_();
            let src_g = if COMPONENTS > 1 {
                pixels[src_ptr + 1].as_()
            } else {
                0i32.as_()
            };
            let src_b = if COMPONENTS > 2 {
                pixels[src_ptr + 2].as_()
            } else {
                0i32.as_()
            };
            let src_a = if COMPONENTS == 4 {
                pixels[src_ptr + 3].as_()
            } else {
                0i32.as_()
            };

            for i in 0..=radius {
                let stack_value = unsafe { &mut *stacks.get_unchecked_mut(i as usize) };
                stack_value.r = src_r;
                if COMPONENTS > 1 {
                    stack_value.g = src_g;
                }
                if COMPONENTS > 2 {
                    stack_value.b = src_b;
                }
                if COMPONENTS == 4 {
                    stack_value.a = src_a;
                }

                let ji = (i + 1).as_();
                sum_r += src_r * ji;
                if COMPONENTS > 1 {
                    sum_g += src_g * ji;
                }
                if COMPONENTS > 2 {
                    sum_b += src_b * ji;
                }
                if COMPONENTS == 4 {
                    sum_a += src_a * ji;
                }

                sum_out_r += src_r;
                if COMPONENTS > 1 {
                    sum_out_g += src_g;
                }
                if COMPONENTS > 2 {
                    sum_out_b += src_b;
                }
                if COMPONENTS == 4 {
                    sum_out_a += src_a;
                }
            }

            for i in 1..=radius {
                if i <= hm {
                    src_ptr += stride as usize;
                }

                let stack_ptr = unsafe { &mut *stacks.get_unchecked_mut((i + radius) as usize) };

                let src_r = pixels[src_ptr].as_();
                let src_g = if COMPONENTS > 1 {
                    pixels[src_ptr + 1].as_()
                } else {
                    0i32.as_()
                };
                let src_b = if COMPONENTS > 2 {
                    pixels[src_ptr + 2].as_()
                } else {
                    0i32.as_()
                };
                let src_a = if COMPONENTS == 4 {
                    pixels[src_ptr + 3].as_()
                } else {
                    0i32.as_()
                };

                stack_ptr.r = src_r;
                if COMPONENTS > 1 {
                    stack_ptr.g = src_g;
                }
                if COMPONENTS > 2 {
                    stack_ptr.b = src_b;
                }
                if COMPONENTS == 4 {
                    stack_ptr.a = src_a;
                }

                let rji = (radius + 1 - i).as_();
                sum_r += src_r * rji;
                if COMPONENTS > 1 {
                    sum_g += src_g * rji;
                }
                if COMPONENTS > 2 {
                    sum_b += src_b * rji;
                }
                if COMPONENTS == 4 {
                    sum_a += src_a * rji;
                }
                sum_in_r += src_r;
                if COMPONENTS > 1 {
                    sum_in_g += src_g;
                }
                if COMPONENTS > 2 {
                    sum_in_b += src_b;
                }
                if COMPONENTS == 4 {
                    sum_in_a += src_a;
                }
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
                    let sum_r_i64: I = sum_r.as_();
                    pixels.write(dst_ptr, ((sum_r_i64 * mul_sum) >> shr_sum).as_());
                    if COMPONENTS > 1 {
                        let sum_g_i64: I = sum_g.as_();
                        pixels.write(dst_ptr + 1, ((sum_g_i64 * mul_sum) >> shr_sum).as_());
                    }
                    if COMPONENTS > 2 {
                        let sum_b_i64: I = sum_b.as_();
                        pixels.write(dst_ptr + 2, ((sum_b_i64 * mul_sum) >> shr_sum).as_());
                    }
                    if COMPONENTS == 4 {
                        let sum_a_i64: I = sum_a.as_();
                        pixels.write(dst_ptr + 3, ((sum_a_i64 * mul_sum) >> shr_sum).as_());
                    }
                }
                dst_ptr += stride as usize;

                sum_r -= sum_out_r;
                if COMPONENTS > 1 {
                    sum_g -= sum_out_g;
                }
                if COMPONENTS > 2 {
                    sum_b -= sum_out_b;
                }
                if COMPONENTS == 4 {
                    sum_a -= sum_out_a;
                }

                stack_start = sp + div - radius;
                if stack_start >= div {
                    stack_start -= div;
                }
                let stack_ptr = unsafe { &mut *stacks.get_unchecked_mut(stack_start as usize) };

                sum_out_r -= stack_ptr.r;
                if COMPONENTS > 1 {
                    sum_out_g -= stack_ptr.g;
                }
                if COMPONENTS > 2 {
                    sum_out_b -= stack_ptr.b;
                }
                if COMPONENTS == 4 {
                    sum_out_a -= stack_ptr.a;
                }

                if yp < hm {
                    src_ptr += stride as usize; // stride
                    yp += 1;
                }

                let src_r = pixels[src_ptr].as_();
                let src_g = if COMPONENTS > 1 {
                    pixels[src_ptr + 1].as_()
                } else {
                    0i32.as_()
                };
                let src_b = if COMPONENTS > 2 {
                    pixels[src_ptr + 2].as_()
                } else {
                    0i32.as_()
                };
                let src_a = if COMPONENTS == 4 {
                    pixels[src_ptr + 3].as_()
                } else {
                    0i32.as_()
                };

                stack_ptr.r = src_r;
                if COMPONENTS > 1 {
                    stack_ptr.g = src_g;
                }
                if COMPONENTS > 2 {
                    stack_ptr.b = src_b;
                }
                if COMPONENTS == 4 {
                    stack_ptr.a = src_a;
                }

                sum_in_r += src_r;
                if COMPONENTS > 1 {
                    sum_in_g += src_g;
                }
                if COMPONENTS > 2 {
                    sum_in_b += src_b;
                }
                if COMPONENTS == 4 {
                    sum_in_a += src_a;
                }

                sum_r += sum_in_r;
                if COMPONENTS > 1 {
                    sum_g += sum_in_g;
                }
                if COMPONENTS > 2 {
                    sum_b += sum_in_b;
                }
                if COMPONENTS == 4 {
                    sum_a += sum_in_a;
                }
                sp += 1;

                if sp >= div {
                    sp = 0;
                }
                let stack_ptr = unsafe { &mut *stacks.get_unchecked_mut(sp as usize) };

                sum_out_r += stack_ptr.r;
                if COMPONENTS > 1 {
                    sum_out_g += stack_ptr.g;
                }
                if COMPONENTS > 2 {
                    sum_out_b += stack_ptr.b;
                }
                if COMPONENTS == 4 {
                    sum_out_a += stack_ptr.a;
                }
                sum_in_r -= stack_ptr.r;
                if COMPONENTS > 1 {
                    sum_in_g -= stack_ptr.g;
                }
                if COMPONENTS > 2 {
                    sum_in_b -= stack_ptr.b;
                }
                if COMPONENTS == 4 {
                    sum_in_a -= stack_ptr.a;
                }
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
            if radius < BASE_RADIUS_I64_CUTOFF {
                #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                {
                    _dispatcher = stack_blur_pass_neon_i32::<3>;
                }
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                {
                    if _is_sse_available {
                        if _is_avx512dq_available && _is_avx512vl_available {
                            _dispatcher = stack_blur_pass_sse::<3, true>;
                        } else {
                            _dispatcher = stack_blur_pass_sse::<3, false>;
                        }
                    }
                }
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                {
                    _dispatcher = stack_blur_pass_wasm_i32::<3>;
                }
            } else {
                #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                {
                    _dispatcher = stack_blur_pass_neon_i64::<3>;
                }
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                {
                    if _is_sse_available {
                        if _is_avx512dq_available && _is_avx512vl_available {
                            _dispatcher = stack_blur_pass_sse_i64::<3, true>;
                        } else {
                            _dispatcher = stack_blur_pass_sse_i64::<3, false>;
                        }
                    }
                }
            }
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
            if radius < BASE_RADIUS_I64_CUTOFF {
                #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                {
                    _dispatcher = stack_blur_pass_neon_i32::<4>;
                }
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                {
                    if _is_sse_available {
                        if _is_avx512dq_available && _is_avx512vl_available {
                            _dispatcher = stack_blur_pass_sse::<4, true>;
                        } else {
                            _dispatcher = stack_blur_pass_sse::<4, false>;
                        }
                    }
                }
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                {
                    _dispatcher = stack_blur_pass_wasm_i32::<4>;
                }
            } else {
                #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                {
                    _dispatcher = stack_blur_pass_neon_i64::<4>;
                }
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                {
                    if _is_sse_available {
                        if _is_avx512dq_available && _is_avx512vl_available {
                            _dispatcher = stack_blur_pass_sse_i64::<4, true>;
                        } else {
                            _dispatcher = stack_blur_pass_sse_i64::<4, false>;
                        }
                    }
                }
            }
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
            if radius < BASE_RADIUS_I64_CUTOFF {
                #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                {
                    _dispatcher = stack_blur_pass_neon_i32::<3>;
                }
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                {
                    if _is_sse_available {
                        if _is_avx512dq_available && _is_avx512vl_available {
                            _dispatcher = stack_blur_pass_sse::<3, true>;
                        } else {
                            _dispatcher = stack_blur_pass_sse::<3, false>;
                        }
                    }
                }
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                {
                    _dispatcher = stack_blur_pass_wasm_i32::<3>;
                }
            } else {
                #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                {
                    _dispatcher = stack_blur_pass_neon_i64::<3>;
                }
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                {
                    if _is_sse_available {
                        if _is_avx512dq_available && _is_avx512vl_available {
                            _dispatcher = stack_blur_pass_sse_i64::<3, true>;
                        } else {
                            _dispatcher = stack_blur_pass_sse_i64::<3, false>;
                        }
                    }
                }
            }
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
            if radius < BASE_RADIUS_I64_CUTOFF {
                #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                {
                    _dispatcher = stack_blur_pass_neon_i32::<4>;
                }
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                {
                    if _is_sse_available {
                        if _is_avx512dq_available && _is_avx512vl_available {
                            _dispatcher = stack_blur_pass_sse::<4, true>;
                        } else {
                            _dispatcher = stack_blur_pass_sse::<4, false>;
                        }
                    }
                }
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                {
                    _dispatcher = stack_blur_pass_wasm_i32::<3>;
                }
            } else {
                #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                {
                    _dispatcher = stack_blur_pass_neon_i64::<4>;
                }
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                {
                    if _is_sse_available {
                        if _is_avx512dq_available && _is_avx512vl_available {
                            _dispatcher = stack_blur_pass_sse_i64::<4, true>;
                        } else {
                            _dispatcher = stack_blur_pass_sse_i64::<4, false>;
                        }
                    }
                }
            }
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
