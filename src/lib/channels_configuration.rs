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

#[repr(C)]
#[derive(Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
/// Declares channels count, generally channels order do not matter for blurring,
/// except cases when transformation into linear colorspace is performed
/// in case of linear transformation alpha plane expected to be last so if colorspace has 4 channels then it should be
/// RGBA, BGRA etc
pub enum FastBlurChannels {
    /// Single plane image
    Plane = 1,
    /// RGB, BGR etc
    Channels3 = 3,
    /// RGBA, BGRA etc
    Channels4 = 4,
}

impl FastBlurChannels {
    pub fn get_channels(&self) -> usize {
        match self {
            FastBlurChannels::Plane => 1,
            FastBlurChannels::Channels3 => 3,
            FastBlurChannels::Channels4 => 4,
        }
    }
}

impl From<usize> for FastBlurChannels {
    fn from(value: usize) -> Self {
        return match value {
            1 => FastBlurChannels::Plane,
            3 => FastBlurChannels::Channels3,
            4 => FastBlurChannels::Channels4,
            _ => {
                panic!("Unknown value");
            }
        };
    }
}
