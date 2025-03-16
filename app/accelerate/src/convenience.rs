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

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod acc_convenience {
    use crate::accelerate::*;
    #[allow(clippy::too_many_arguments)]
    pub fn box_convolve(
        src: &[u8],
        src_stride: usize,
        dst: &mut [u8],
        dst_stride: usize,
        kernel_size: usize,
        width: usize,
        height: usize,
        threading: bool,
    ) {
        let src_image = vImage_Buffer {
            data: src.as_ptr() as *mut libc::c_void,
            height,
            width,
            row_bytes: src_stride,
        };
        let mut dst_image = vImage_Buffer {
            data: dst.as_mut_ptr() as *mut libc::c_void,
            height,
            width,
            row_bytes: dst_stride,
        };
        unsafe {
            let flags = if threading {
                kvImageEdgeExtend
            } else {
                kvImageEdgeExtend | kvImageDoNotTile
            };
            let status = vImageBoxConvolve_ARGB8888(
                &src_image,
                &mut dst_image,
                std::ptr::null_mut(),
                0,
                0,
                kernel_size as libc::c_uint,
                kernel_size as libc::c_uint,
                std::ptr::null_mut(),
                flags,
            );
            if status != 0 {
                panic!("vImageBoxConvolve returned error: {}", status);
            }
        }
    }
}
