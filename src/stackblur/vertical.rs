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
use crate::primitives::PrimitiveCast;
use crate::stackblur::sliding_window::SlidingWindow;
use crate::stackblur::stack_blur_pass::StackBlurWorkingPass;
use crate::unsafe_slice::UnsafeSlice;
use std::marker::PhantomData;
use std::ops::{AddAssign, Mul, Sub, SubAssign};

pub struct VerticalStackBlurPass<T, J, F, const CN: usize> {
    _phantom_t: PhantomData<T>,
    _phantom_j: PhantomData<J>,
    _phantom_f: PhantomData<F>,
}

impl<T, J, F, const CN: usize> Default for VerticalStackBlurPass<T, J, F, CN> {
    fn default() -> Self {
        VerticalStackBlurPass {
            _phantom_t: Default::default(),
            _phantom_j: Default::default(),
            _phantom_f: Default::default(),
        }
    }
}

impl<T, J, F, const CN: usize> VerticalStackBlurPass<T, J, F, CN>
where
    J: Copy
        + 'static
        + AddAssign<J>
        + Mul<Output = J>
        + Sub<Output = J>
        + PrimitiveCast<f32>
        + PrimitiveCast<F>
        + SubAssign
        + PrimitiveCast<T>
        + Default,
    T: Copy + PrimitiveCast<J> + Default,
    i32: PrimitiveCast<J>,
    u32: PrimitiveCast<J>,
    f32: PrimitiveCast<T> + PrimitiveCast<J> + PrimitiveCast<F>,
    F: PrimitiveCast<T> + 'static + Mul<Output = F>,
    usize: PrimitiveCast<J>,
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
        let mut _yp;
        let mut sp;
        let mut stack_start;
        let mut stacks = vec![SlidingWindow::<CN, J>::new(); div];

        let mut sum: SlidingWindow<CN, J>;
        let mut sum_in: SlidingWindow<CN, J>;
        let mut sum_out: SlidingWindow<CN, J>;

        let hm = height - 1;
        let div = (radius * 2) + 1;

        let mut src_ptr;
        let mut dst_ptr;

        let min_x = thread * width as usize / total_threads;
        let max_x = (thread + 1) * width as usize / total_threads;

        let mul_value = (1. / ((radius as f32 + 1.) * (radius as f32 + 1.))).cast_();

        for x in min_x..max_x {
            sum = SlidingWindow::default();
            sum_in = SlidingWindow::default();
            sum_out = SlidingWindow::default();

            src_ptr = CN * x;

            let src = SlidingWindow::from_store(pixels, src_ptr);

            for i in 0..=radius {
                unsafe { *stacks.get_unchecked_mut(i as usize) = src }

                let ji = (i + 1).cast_();
                sum += src * ji;
                sum_out += src;
            }

            for i in 1..=radius {
                if i <= hm {
                    src_ptr += stride as usize;
                }

                let src = SlidingWindow::from_store(pixels, src_ptr);

                unsafe { *stacks.get_unchecked_mut((i + radius) as usize) = src };

                let rji = (radius + 1 - i).cast_();
                sum += src * rji;
                sum_in += src;
            }

            sp = radius;
            _yp = radius;
            if _yp > hm {
                _yp = hm;
            }
            src_ptr = CN * x + _yp as usize * stride as usize;
            dst_ptr = CN * x;
            for _ in 0..height {
                let sum_intermediate: SlidingWindow<CN, f32> = sum.cast();
                let finalized: SlidingWindow<CN, J> = (sum_intermediate * mul_value).cast();
                finalized.to_store(pixels, dst_ptr);

                dst_ptr += stride as usize;

                sum -= sum_out;

                stack_start = sp + div - radius;
                if stack_start >= div {
                    stack_start -= div;
                }
                let stack_ptr = unsafe { &mut *stacks.get_unchecked_mut(stack_start as usize) };

                sum_out -= *stack_ptr;

                if _yp < hm {
                    src_ptr += stride as usize; // stride
                    _yp += 1;
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

impl<T, J, F, const CN: usize> StackBlurWorkingPass<T, CN> for VerticalStackBlurPass<T, J, F, CN>
where
    J: Copy
        + 'static
        + AddAssign<J>
        + Mul<Output = J>
        + Sub<Output = J>
        + PrimitiveCast<f32>
        + PrimitiveCast<F>
        + SubAssign
        + PrimitiveCast<T>
        + Default,
    T: Copy + PrimitiveCast<J> + Default,
    i32: PrimitiveCast<J>,
    u32: PrimitiveCast<J>,
    f32: PrimitiveCast<T> + PrimitiveCast<J> + PrimitiveCast<F>,
    F: PrimitiveCast<T> + 'static + Mul<Output = F>,
    usize: PrimitiveCast<J>,
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
