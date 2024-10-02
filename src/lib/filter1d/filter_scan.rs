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
use num_traits::AsPrimitive;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialOrd, PartialEq)]
pub struct ScanPoint1d<F> {
    pub weight: F,
}

impl<F> ScanPoint1d<F> {
    pub fn new(weight: F) -> ScanPoint1d<F> {
        ScanPoint1d { weight }
    }
}

pub(crate) unsafe fn scan_se_1d<F>(kernel: &[F]) -> Vec<ScanPoint1d<F>>
where
    F: Copy + PartialEq + 'static,
    i32: AsPrimitive<F>,
{
    let mut left_front = vec![];

    let kernel_width = kernel.len();

    for x in 0..kernel_width {
        let item = *kernel.get_unchecked(x);
        left_front.push(ScanPoint1d::new(item));
    }

    left_front.to_vec()
}

pub(crate) unsafe fn is_symmetric_1d<F>(kernel: &[F]) -> bool
where
    F: Copy + PartialEq + 'static,
    i32: AsPrimitive<F>,
{
    {
        let len = kernel.len();
        for i in 0..len / 2 {
            if kernel
                .get_unchecked(i)
                .ne(kernel.get_unchecked(len - 1 - i))
            {
                return false;
            }
        }
        true
    }
}
