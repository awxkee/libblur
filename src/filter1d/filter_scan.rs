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

pub(crate) fn scan_se_1d<F>(kernel: &[F]) -> Vec<ScanPoint1d<F>>
where
    F: Copy + PartialEq + 'static + Default,
    i32: AsPrimitive<F>,
{
    let mut left_front = vec![ScanPoint1d::new(F::default()); kernel.len()];

    for (dst, src) in left_front.iter_mut().zip(kernel.iter()) {
        *dst = ScanPoint1d::new(*src);
    }
    left_front
}

pub(crate) fn is_symmetric_1d<F>(kernel: &[F]) -> bool
where
    F: Copy + PartialEq + 'static,
{
    let len = kernel.len();
    let fw = kernel.iter().take(len / 2);
    let bw = kernel.iter().rev().take(len / 2);
    for (&f, w) in fw.rev().zip(bw.rev()) {
        if f.ne(w) {
            return false;
        }
    }
    true
}
