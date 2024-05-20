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

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
#[cfg(target_feature = "neon")]
pub(crate) mod neon_utils {
    use std::arch::aarch64::{
        float32x4_t, int16x4_t, int32x4_t, uint16x4_t, uint32x4_t, vget_low_u16, vld1_u8,
        vld1q_f32, vmovl_u16, vmovl_u8, vreinterpret_s16_u16, vreinterpretq_s32_u32,
    };
    use std::ptr;

    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn load_u8_s32(ptr: *const u8, use_vld: bool, channels_count: usize) -> int32x4_t {
        let mut safe_transient_store: [u8; 8] = [0; 8];
        let edge_ptr: *const u8;
        if use_vld {
            edge_ptr = ptr;
        } else {
            unsafe {
                ptr::copy_nonoverlapping(ptr, safe_transient_store.as_mut_ptr(), channels_count);
            }
            edge_ptr = safe_transient_store.as_ptr();
        }
        let pixel_color =
            unsafe { vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(edge_ptr))))) };
        return pixel_color;
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn load_u8_s16(ptr: *const u8, use_vld: bool, channels_count: usize) -> int16x4_t {
        let mut safe_transient_store: [u8; 8] = [0; 8];
        let edge_ptr: *const u8;
        if use_vld {
            edge_ptr = ptr;
        } else {
            unsafe {
                ptr::copy_nonoverlapping(ptr, safe_transient_store.as_mut_ptr(), channels_count);
            }
            edge_ptr = safe_transient_store.as_ptr();
        }
        let pixel_color =
            unsafe { vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vld1_u8(edge_ptr)))) };
        return pixel_color;
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn load_u8_u16(ptr: *const u8, use_vld: bool, channels_count: usize) -> uint16x4_t {
        let mut safe_transient_store: [u8; 8] = [0; 8];
        let edge_ptr: *const u8;
        if use_vld {
            edge_ptr = ptr;
        } else {
            unsafe {
                ptr::copy_nonoverlapping(ptr, safe_transient_store.as_mut_ptr(), channels_count);
            }
            edge_ptr = safe_transient_store.as_ptr();
        }
        let pixel_color = unsafe { vget_low_u16(vmovl_u8(vld1_u8(edge_ptr))) };
        return pixel_color;
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn load_u8_u32(ptr: *const u8, use_vld: bool, channels_count: usize) -> uint32x4_t {
        let mut safe_transient_store: [u8; 8] = [0; 8];
        let edge_ptr: *const u8;
        if use_vld {
            edge_ptr = ptr;
        } else {
            unsafe {
                ptr::copy_nonoverlapping(ptr, safe_transient_store.as_mut_ptr(), channels_count);
            }
            edge_ptr = safe_transient_store.as_ptr();
        }
        let pixel_color = unsafe { vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(edge_ptr)))) };
        return pixel_color;
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn load_f32(ptr: *const f32, use_vld: bool, channels_count: usize) -> float32x4_t {
        let mut safe_transient_store: [f32; 4] = [0f32; 4];
        let edge_ptr: *const f32;
        if use_vld {
            edge_ptr = ptr;
        } else {
            unsafe {
                ptr::copy_nonoverlapping(ptr, safe_transient_store.as_mut_ptr(), channels_count);
            }
            edge_ptr = safe_transient_store.as_ptr();
        }
        let pixel_color = unsafe { vld1q_f32(edge_ptr) };
        return pixel_color;
    }
}
