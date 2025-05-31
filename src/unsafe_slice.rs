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

use std::cell::UnsafeCell;
use std::ops::Index;

#[derive(Copy, Clone, Debug)]
pub struct UnsafeSlice<'a, T> {
    pub slice: &'a [UnsafeCell<T>],
}

unsafe impl<T: Send + Sync> Send for UnsafeSlice<'_, T> {}

unsafe impl<T: Send + Sync> Sync for UnsafeSlice<'_, T> {}

impl<'a, T> UnsafeSlice<'a, T> {
    pub fn new(slice: &'a mut [T]) -> Self {
        let ptr = slice as *mut [T] as *const [UnsafeCell<T>];
        Self {
            slice: unsafe { &*ptr },
        }
    }

    /// SAFETY: It is UB if two threads write to the same index without
    /// synchronization.
    #[inline(always)]
    pub unsafe fn write(&self, i: usize, value: T) {
        let ptr = unsafe { self.slice.get_unchecked(i) }.get();
        *ptr = value;
    }
    #[allow(dead_code)]
    #[inline(always)]
    pub fn get(&self, i: usize) -> &T {
        let ptr = unsafe { self.slice.get_unchecked(i) }.get();
        unsafe { &*ptr }
    }
    #[allow(dead_code)]
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.slice.len()
    }
}

impl<T> Index<usize> for UnsafeSlice<'_, T> {
    type Output = T;
    #[allow(dead_code)]
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        let ptr = unsafe { self.slice.get_unchecked(index) }.get();
        unsafe { &*ptr }
    }
}
