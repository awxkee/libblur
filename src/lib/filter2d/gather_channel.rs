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
#![forbid(unsafe_code)]

use crate::{BlurImage, BlurImageMut, FastBlurChannels};
use std::fmt::Debug;

pub fn gather_channel<'a, T: Copy + Default + Debug, const CN: usize>(
    image: &BlurImage<'a, T>,
    order: usize,
) -> BlurImageMut<'a, T> {
    assert!(order < CN);
    let mut channel =
        BlurImageMut::<'a, T>::alloc(image.width, image.height, FastBlurChannels::Plane);
    let dst_stride = channel.row_stride() as usize;
    for (dst, src) in channel.data.borrow_mut().chunks_exact_mut(dst_stride).zip(
        image
            .data
            .as_ref()
            .chunks_exact(image.row_stride() as usize),
    ) {
        for (dst, src) in dst.iter_mut().zip(src.chunks_exact(CN)) {
            *dst = src[order];
        }
    }
    channel
}

pub fn squash_channel<T: Copy + Default + Debug, const CN: usize>(
    image: &mut BlurImageMut<'_, T>,
    source: &BlurImage<T>,
    order: usize,
) {
    assert!(order < CN);
    let dst_stride = image.row_stride() as usize;
    for (dst, src) in image.data.borrow_mut().chunks_exact_mut(dst_stride).zip(
        source
            .data
            .as_ref()
            .chunks_exact(source.row_stride() as usize),
    ) {
        for (dst, src) in dst.chunks_exact_mut(CN).zip(src.iter()) {
            dst[order] = *src;
        }
    }
}
