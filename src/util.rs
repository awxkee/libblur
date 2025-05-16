/*
 * // Copyright (c) Radzivon Bartoshyk 2/2025. All rights reserved.
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
use std::error::Error;

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
/// Shows size mismatching
pub struct MismatchedSize {
    pub expected: usize,
    pub received: usize,
}

#[derive(Copy, Clone, Debug)]
pub enum BlurError {
    ZeroBaseSize,
    MinimumSliceSizeMismatch(MismatchedSize),
    MinimumStrideSizeMismatch(MismatchedSize),
    OddKernel(usize),
    KernelSizeMismatch(MismatchedSize),
    ImagesMustMatch,
    StrideIsNotSupported,
    FftChannelsNotSupported,
    ExceedingPointerSize,
    NegativeOrZeroSigma,
    InvalidArguments,
}

impl Error for BlurError {}

impl std::fmt::Display for BlurError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            BlurError::MinimumSliceSizeMismatch(size) => f.write_fmt(format_args!(
                "Minimum image slice size mismatch: expected={}, received={}",
                size.expected, size.received
            )),
            BlurError::MinimumStrideSizeMismatch(size) => f.write_fmt(format_args!(
                "Minimum stride must have size at least {} but it is {}",
                size.expected, size.received
            )),
            BlurError::ZeroBaseSize => f.write_str("Image size must not be zero"),
            BlurError::OddKernel(size) => {
                f.write_fmt(format_args!("Kernel size must be odd, but received {size}",))
            }
            BlurError::KernelSizeMismatch(size) => f.write_fmt(format_args!(
                "Kernel size mismatch: expected={}, received={}",
                size.expected, size.received
            )),
            BlurError::ImagesMustMatch => {
                f.write_str("Source and destination images must match in their dimensions")
            }
            BlurError::StrideIsNotSupported => f.write_str("Stride is not supported"),
            BlurError::FftChannelsNotSupported => f.write_str("Fft supports only planar images"),
            BlurError::ExceedingPointerSize => {
                f.write_str("Image bounds and blurring kernel/radius exceeds pointer capacity")
            }
            BlurError::NegativeOrZeroSigma => {
                f.write_str("Negative or zero sigma is not supported")
            }
            BlurError::InvalidArguments => f.write_str("Invalid arguments"),
        }
    }
}

pub(crate) fn check_slice_size<T>(
    arr: &[T],
    stride: usize,
    width: usize,
    height: usize,
    cn: usize,
) -> Result<(), BlurError> {
    if width == 0 || height == 0 {
        return Err(BlurError::ZeroBaseSize);
    }
    if arr.len() < stride * (height - 1) + width * cn {
        return Err(BlurError::MinimumSliceSizeMismatch(MismatchedSize {
            expected: stride * height,
            received: arr.len(),
        }));
    }
    if (stride) < (width * cn) {
        return Err(BlurError::MinimumStrideSizeMismatch(MismatchedSize {
            expected: width * cn,
            received: stride,
        }));
    }
    Ok(())
}
