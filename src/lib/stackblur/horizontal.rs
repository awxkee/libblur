/*
 * // Copyright (c) Radzivon Bartoshyk. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use crate::stackblur::sliding_window::SlidingWindow;
use crate::stackblur::stack_blur_pass::StackBlurWorkingPass;
use crate::unsafe_slice::UnsafeSlice;
use num_traits::{AsPrimitive, FromPrimitive};
use std::marker::PhantomData;
use std::ops::{AddAssign, Mul, Sub, SubAssign};

pub(crate) struct HorizontalStackBlurPass<T, J, F, const COMPONENTS: usize> {
    _phantom_t: PhantomData<T>,
    _phantom_j: PhantomData<J>,
    _phantom_f: PhantomData<F>,
}

impl<T, J, F, const COMPONENTS: usize> Default for HorizontalStackBlurPass<T, J, F, COMPONENTS> {
    fn default() -> Self {
        HorizontalStackBlurPass::<T, J, F, COMPONENTS> {
            _phantom_t: Default::default(),
            _phantom_j: Default::default(),
            _phantom_f: Default::default(),
        }
    }
}

/// # Generics
/// `T` - data type
/// `J` - accumulator type
impl<T, J, F, const COMPONENTS: usize> HorizontalStackBlurPass<T, J, F, COMPONENTS>
where
    J: Copy
        + 'static
        + FromPrimitive
        + AddAssign<J>
        + Mul<Output = J>
        + Sub<Output = J>
        + AsPrimitive<F>
        + SubAssign
        + AsPrimitive<T>
        + Default,
    T: Copy + AsPrimitive<J> + FromPrimitive,
    i32: AsPrimitive<J>,
    u32: AsPrimitive<J>,
    F: AsPrimitive<T> + AsPrimitive<J> + 'static + Copy + Mul<Output = F> + Default,
    usize: AsPrimitive<J>,
    f32: AsPrimitive<F>,
{
    #[inline]
    unsafe fn pass_impl(
        &self,
        pixels: &UnsafeSlice<T>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        thread: usize,
        total_threads: usize,
    ) {
        let div = ((radius * 2) + 1) as usize;
        let mut xp;
        let mut sp;
        let mut stack_start;
        let mut stacks0 = vec![SlidingWindow::<COMPONENTS, J>::new(); div];

        let rad_p_1 = radius as f32 + 1.;
        let scale_filter_value = (1. / (rad_p_1 * rad_p_1)).as_();

        let wm = width - 1;
        let div = (radius * 2) + 1;

        let min_y = thread * height as usize / total_threads;
        let max_y = (thread + 1) * height as usize / total_threads;

        let start_y = min_y;

        for y in start_y..max_y {
            let mut sum = SlidingWindow::default();
            let mut sum_in = SlidingWindow::default();
            let mut sum_out = SlidingWindow::default();

            let mut src_ptr = stride as usize * y;

            let src = SlidingWindow::from_store(pixels, src_ptr);

            for i in 0..=radius {
                unsafe { *stacks0.get_unchecked_mut(i as usize) = src };
                let fi = (i + 1).as_();
                sum += src * fi;
                sum_out += src;
            }

            for i in 1..=radius {
                if i <= wm {
                    src_ptr += COMPONENTS;
                }

                let src = SlidingWindow::from_store(pixels, src_ptr);

                unsafe { *stacks0.get_unchecked_mut((i + radius) as usize) = src };

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
            let mut dst_ptr = y * stride as usize;
            for _ in 0..width {
                let sum_intermediate: SlidingWindow<COMPONENTS, F> = sum.cast();
                let finalized: SlidingWindow<COMPONENTS, J> =
                    (sum_intermediate * scale_filter_value).cast();
                finalized.to_store(pixels, dst_ptr);

                dst_ptr += COMPONENTS;

                sum -= sum_out;

                stack_start = sp + div - radius;
                if stack_start >= div {
                    stack_start -= div;
                }
                let stack = unsafe { &mut *stacks0.get_unchecked_mut(stack_start as usize) };

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
                let stack = unsafe { &mut *stacks0.get_unchecked_mut(sp as usize) };

                sum_out += *stack;
                sum_in -= *stack;
            }
        }
    }
}

impl<T, J, F, const COMPONENTS: usize> StackBlurWorkingPass<T, COMPONENTS>
    for HorizontalStackBlurPass<T, J, F, COMPONENTS>
where
    J: Copy
        + 'static
        + FromPrimitive
        + AddAssign<J>
        + Mul<Output = J>
        + Sub<Output = J>
        + AsPrimitive<F>
        + SubAssign
        + AsPrimitive<T>
        + AsPrimitive<f32>
        + Default,
    T: Copy + AsPrimitive<J> + FromPrimitive,
    i32: AsPrimitive<J>,
    u32: AsPrimitive<J>,
    F: AsPrimitive<T> + AsPrimitive<J> + 'static + Copy + Mul<Output = F> + Default,
    usize: AsPrimitive<J>,
    f32: AsPrimitive<F> + AsPrimitive<T>,
{
    fn pass(
        &self,
        pixels: &UnsafeSlice<T>,
        stride: u32,
        width: u32,
        height: u32,
        radius: u32,
        thread: usize,
        total_threads: usize,
    ) {
        unsafe {
            self.pass_impl(pixels, stride, width, height, radius, thread, total_threads);
        }
    }
}
