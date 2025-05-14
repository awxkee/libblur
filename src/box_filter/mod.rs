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

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
mod avx;
mod box_blur;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon;
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
mod sse;

pub use box_blur::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BlurImage, BlurImageMut, FastBlurChannels, ThreadingPolicy};

    macro_rules! compare_u8_stat {
        ($dst: expr, $radius: expr, $threading: expr) => {
            for (i, cn) in $dst.data.borrow_mut().chunks_exact(3).enumerate() {
                let diff0 = (cn[0] as i32 - 126).abs();
                let rad = $radius as i32;
                let threading = $threading;
                assert!(
                    diff0 <= 3,
                    "Diff expected to be less than 3, but it was {diff0} at {i} in channel 0, on radius {rad}, with threading {threading}"
                );
                let diff1 = (cn[1] as i32 - 66).abs();
                assert!(
                    diff1 <= 3,
                    "Diff expected to be less than 3, but it was {diff1} at {i} in channel 1, on radius {rad}, with threading {threading}"
                );
                let diff2 = (cn[2] as i32 - 77).abs();
                assert!(
                    diff2 <= 3,
                    "Diff expected to be less than 3, but it was {diff2} at {i} in channel 2, on radius {rad}, with threading {threading}"
                );
            }
        };
    }

    macro_rules! compare_f32_stat {
        ($dst: expr, $radius: expr, $threading: expr) => {
            for (i, cn) in $dst.data.borrow_mut().chunks_exact(3).enumerate() {
                let diff0 = (cn[0] as f32 - 0.532).abs();
                let rad = $radius as i32;
                let threading = $threading;
                assert!(
                    diff0 <= 1e-4,
                    "Diff expected to be less than 1e-4, but it was {diff0} at {i} in channel 0, on radius {rad}, with threading {threading}"
                );
                let diff1 = (cn[1] as f32 - 0.123).abs();
                assert!(
                    diff1 <= 1e-4,
                    "Diff expected to be less than 1e-4, but it was {diff1} at {i} in channel 1, on radius {rad}, with threading {threading}"
                );
                let diff2 = (cn[2] as f32 - 0.654).abs();
                assert!(
                    diff2 <= 1e-4,
                    "Diff expected to be less than 1e-4, but it was {diff2} at {i} in channel 2, on radius {rad}, with threading {threading}"
                );
            }
        };
    }

    fn test_box_rgb8(radius: u32, threading_policy: ThreadingPolicy) {
        let width: usize = 148;
        let height: usize = 148;
        let mut src = vec![126; width * height * 3];
        for dst in src.chunks_exact_mut(3) {
            dst[0] = 126;
            dst[1] = 66;
            dst[2] = 77;
        }
        let src_image = BlurImage::borrow(
            &src,
            width as u32,
            height as u32,
            FastBlurChannels::Channels3,
        );
        let mut dst = BlurImageMut::default();
        box_blur(&src_image, &mut dst, radius, threading_policy).unwrap();
        compare_u8_stat!(dst, radius, threading_policy);
    }

    fn test_box_rgb16(radius: u32, threading_policy: ThreadingPolicy) {
        let width: usize = 148;
        let height: usize = 148;
        let mut src = vec![126u16; width * height * 3];
        for dst in src.chunks_exact_mut(3) {
            dst[0] = 126;
            dst[1] = 66;
            dst[2] = 77;
        }
        let src_image = BlurImage::borrow(
            &src,
            width as u32,
            height as u32,
            FastBlurChannels::Channels3,
        );
        let mut dst = BlurImageMut::default();
        box_blur_u16(&src_image, &mut dst, radius, threading_policy).unwrap();
        compare_u8_stat!(dst, radius, threading_policy);
    }

    fn test_box_rgb_f32(radius: u32, threading_policy: ThreadingPolicy) {
        let width: usize = 148;
        let height: usize = 148;
        let mut src = vec![126.; width * height * 3];
        for dst in src.chunks_exact_mut(3) {
            dst[0] = 0.532;
            dst[1] = 0.123;
            dst[2] = 0.654;
        }
        let src_image = BlurImage::borrow(
            &src,
            width as u32,
            height as u32,
            FastBlurChannels::Channels3,
        );
        let mut dst = BlurImageMut::default();
        box_blur_f32(&src_image, &mut dst, radius, threading_policy).unwrap();
        compare_f32_stat!(dst, radius, threading_policy);
    }

    #[test]
    fn test_box_u8() {
        test_box_rgb8(5, ThreadingPolicy::Single);
        test_box_rgb8(70, ThreadingPolicy::Single);
        test_box_rgb8(5, ThreadingPolicy::Adaptive);
        test_box_rgb8(70, ThreadingPolicy::Adaptive);
    }

    #[test]
    fn test_box_u16() {
        test_box_rgb16(5, ThreadingPolicy::Single);
        test_box_rgb16(70, ThreadingPolicy::Single);
        test_box_rgb16(5, ThreadingPolicy::Adaptive);
        test_box_rgb16(70, ThreadingPolicy::Adaptive);
    }

    #[test]
    fn test_box_f32() {
        test_box_rgb_f32(5, ThreadingPolicy::Single);
        test_box_rgb_f32(70, ThreadingPolicy::Single);
        test_box_rgb_f32(5, ThreadingPolicy::Adaptive);
        test_box_rgb_f32(70, ThreadingPolicy::Adaptive);
    }
}
