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
use crate::filter1d::KernelShape;
use crate::filter2d::scan_point_2d::ScanPoint2d;
use num_traits::AsPrimitive;

pub(crate) fn scan_se_2d<F>(
    structuring_element: &[F],
    structuring_element_size: KernelShape,
) -> Vec<ScanPoint2d<F>>
where
    F: Copy + PartialEq + 'static,
    i32: AsPrimitive<F>,
{
    let mut left_front = vec![];

    let kernel_width = structuring_element_size.width;
    let kernel_height = structuring_element_size.height;

    let horizontal_anchor = kernel_width as i64 / 2;
    let half_kernel_height = kernel_height as i64 / 2;

    structuring_element
        .chunks_exact(kernel_width)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, &element) in row.iter().enumerate() {
                let zero_f = 0i32.as_();
                if element.ne(&zero_f) {
                    left_front.push(ScanPoint2d::new(
                        x as i64 - horizontal_anchor,
                        y as i64 - half_kernel_height,
                        element,
                    ));
                }
            }
        });

    left_front
}
