/*
 * Copyright (c) Radzivon Bartoshyk. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2.  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3.  Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod convenience;

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod accelerate {
    extern crate libc;

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {

        #[allow(non_camel_case_types)]
        #[allow(non_snake_case)]
        pub fn vImageBoxConvolve_ARGB8888(
            src: *const vImage_Buffer,
            dest: *mut vImage_Buffer,
            temp_buffer: *mut libc::c_void,
            srcOffsetToROI_X: libc::c_uint,
            srcOffsetToROI_Y: libc::c_uint,
            kernel_height: libc::c_uint,
            kernel_width: libc::c_uint,
            background_color: *mut libc::c_void,
            flags: libc::c_uint,
        ) -> libc::c_int;

        #[allow(non_camel_case_types)]
        #[allow(non_snake_case)]
        pub fn vImageScale_ARGB8888(
            src: *const vImage_Buffer,
            dest: *mut vImage_Buffer,
            temp_buffer: *mut libc::c_void,
            flags: libc::c_uint,
        ) -> libc::c_int;

        #[allow(non_camel_case_types)]
        #[allow(non_snake_case)]
        pub fn vImageScale_XRGB2101010W(
            src: *const vImage_Buffer,
            dest: *mut vImage_Buffer,
            temp_buffer: *mut libc::c_void,
            flags: libc::c_uint,
        ) -> libc::c_int;

        #[allow(non_camel_case_types)]
        #[allow(non_snake_case)]
        pub fn vImageScale_ARGBFFFF(
            src: *const vImage_Buffer,
            dest: *mut vImage_Buffer,
            temp_buffer: *mut libc::c_void,
            flags: libc::c_uint,
        ) -> libc::c_int;

        // Scales a Planar8 image
        #[allow(non_camel_case_types)]
        #[allow(non_snake_case)]
        pub fn vImageScale_Planar8(
            src: *const vImage_Buffer,
            dest: *mut vImage_Buffer,
            temp_buffer: *mut libc::c_void,
            flags: libc::c_uint,
        ) -> libc::c_int;

        // Scales a PlanarF (floating point) image
        #[allow(non_camel_case_types)]
        #[allow(non_snake_case)]
        pub fn vImageScale_PlanarF(
            src: *const vImage_Buffer,
            dest: *mut vImage_Buffer,
            temp_buffer: *mut libc::c_void,
            flags: libc::c_uint,
        ) -> libc::c_int;

        // Scales a Planar16U (16-bit unsigned) image
        #[allow(non_camel_case_types)]
        #[allow(non_snake_case)]
        pub fn vImageScale_Planar16U(
            src: *const vImage_Buffer,
            dest: *mut vImage_Buffer,
            temp_buffer: *mut libc::c_void,
            flags: libc::c_uint,
        ) -> libc::c_int;

        // Scales an ARGB16U (16-bit unsigned) image
        #[allow(non_camel_case_types)]
        #[allow(non_snake_case)]
        pub fn vImageScale_ARGB16U(
            src: *const vImage_Buffer,
            dest: *mut vImage_Buffer,
            temp_buffer: *mut libc::c_void,
            flags: libc::c_uint,
        ) -> libc::c_int;

        // Scales a Planar16F (16-bit floating point) image
        #[allow(non_camel_case_types)]
        #[allow(non_snake_case)]
        pub fn vImageScale_Planar16F(
            src: *const vImage_Buffer,
            dest: *mut vImage_Buffer,
            temp_buffer: *mut libc::c_void,
            flags: libc::c_uint,
        ) -> libc::c_int;

        // Scales an ARGB16F (16-bit floating point) image
        #[allow(non_camel_case_types)]
        #[allow(non_snake_case)]
        pub fn vImageScale_ARGB16F(
            src: *const vImage_Buffer,
            dest: *mut vImage_Buffer,
            temp_buffer: *mut libc::c_void,
            flags: libc::c_uint,
        ) -> libc::c_int;
    }

    #[allow(non_camel_case_types)]
    #[allow(non_snake_case)]
    #[repr(C)]
    pub struct vImage_Buffer {
        pub data: *mut libc::c_void,
        pub height: libc::size_t,
        pub width: libc::size_t,
        pub row_bytes: libc::size_t,
    }

    #[allow(non_camel_case_types)]
    #[allow(non_snake_case)]
    #[allow(non_upper_case_globals)]
    pub const kvImageNoFlags: libc::c_uint = 0;

    #[allow(non_camel_case_types)]
    #[allow(non_snake_case)]
    #[allow(non_upper_case_globals)]
    pub const kvImageUseFP16Accumulator: libc::c_uint = 4096;

    #[allow(non_camel_case_types)]
    #[allow(non_snake_case)]
    #[allow(non_upper_case_globals)]
    pub const kvImageDoNotTile: libc::c_uint = 16;

    #[allow(non_camel_case_types)]
    #[allow(non_snake_case)]
    #[allow(non_upper_case_globals)]
    pub const kvImageEdgeExtend: libc::c_uint = 8;

    #[allow(non_camel_case_types)]
    #[allow(non_snake_case)]
    #[allow(non_upper_case_globals)]
    pub const kvImageTruncateKernel: libc::c_uint = 64;
    /*
    kvImageInvalidEdgeStyle -21768
    kvImageRoiLargerThanInputBuffer -21766
    kvImageInvalidOffset_X -21769
    kvImageInvalidOffset_Y -21770
    kvImageMemoryAllocationError -21771
    kvImageInvalidKernelSize -21767
     */
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use accelerate::*;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use convenience::*;
