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

#[cfg(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[allow(dead_code)]
fn apple_has_cpu_feature(feature_name: &str) -> bool {
    use libc::{c_int, sysctlbyname};
    use std::ffi::CString;
    use std::ptr;

    let c_feature_name = CString::new(feature_name).expect("CString::new failed");
    let mut result: c_int = 0;
    let mut size = std::mem::size_of::<c_int>();

    unsafe {
        if sysctlbyname(
            c_feature_name.as_ptr(),
            &mut result as *mut _ as *mut libc::c_void,
            &mut size,
            ptr::null_mut(),
            0,
        ) == 0
        {
            result != 0
        } else {
            false
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(dead_code)]
pub fn is_x86_avx512dq_supported() -> bool {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        apple_has_cpu_feature("hw.optional.avx512dq")
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        std::arch::is_x86_feature_detected!("avx512dq")
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(dead_code)]
pub fn is_x86_avx512vl_supported() -> bool {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        apple_has_cpu_feature("hw.optional.avx512vl")
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        std::arch::is_x86_feature_detected!("avx512vl")
    }
}

#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    any(target_arch = "x86", target_arch = "x86_64")
)))]
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
#[allow(dead_code)]
fn apple_has_cpu_feature(_feature_name: &str) -> bool {
    false
}

/// Test aarch64 cpu with *fp16* check,
/// on *Apple* platform [libc](https://developer.apple.com/documentation/kernel/1387446-sysctlbyname/determining_instruction_set_characteristics) be used
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[allow(dead_code)]
pub fn is_aarch_f16_supported() -> bool {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        apple_has_cpu_feature("hw.optional.arm.FEAT_FP16")
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        std::arch::is_aarch64_feature_detected!("fp16")
    }
}

/// Test aarch64 cpu with *f16 conversion* instructions.
/// on *Apple* platform [libc](https://developer.apple.com/documentation/kernel/1387446-sysctlbyname/determining_instruction_set_characteristics) be used
/// otherwise consider it is always available
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[allow(dead_code)]
pub fn is_aarch_f16c_supported() -> bool {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        apple_has_cpu_feature("hw.optional.AdvSIMD_HPFPCvt")
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        true
    }
}

/// Test aarch64 cpu with *fhm conversion* instructions.
/// on *Apple* platform [libc](https://developer.apple.com/documentation/kernel/1387446-sysctlbyname/determining_instruction_set_characteristics) be used
/// otherwise consider it is always available
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[allow(dead_code)]
pub fn is_aarch_fhm_supported() -> bool {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        apple_has_cpu_feature("hw.optional.arm.FEAT_FHM")
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        std::arch::is_aarch64_feature_detected!("fhm")
    }
}
