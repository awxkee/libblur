/*
 * // Copyright (c) Radzivon Bartoshyk 5/2025. All rights reserved.
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
use crate::gamma_curves::TransferFunction;
use crate::{BlurError, BlurImage, FastBlurChannels};

struct Linearization16 {
    linearization: Box<[u16; 65536]>,
}

struct Gamma8 {
    gamma: Box<[u8; 65536]>,
}

struct Gamma16 {
    gamma: Box<[u16; 65536]>,
}

fn make_linearization16(transfer_function: TransferFunction) -> Linearization16 {
    let mut linearizing = Box::new([0u16; 65536]);
    let max_lin_depth = (1u32 << 16) - 1;

    for (i, dst) in linearizing.iter_mut().enumerate() {
        *dst = (transfer_function.linearize(i as f32 / 65535.) * max_lin_depth as f32)
            .round()
            .min(max_lin_depth as f32) as u16;
    }

    Linearization16 {
        linearization: linearizing,
    }
}

struct Linearization {
    linearization: Box<[u16; 256]>,
}

fn make_linearization(transfer_function: TransferFunction) -> Linearization {
    let mut linearizing = Box::new([0u16; 256]);
    let max_lin_depth = (1u32 << 16) - 1;

    for (i, dst) in linearizing.iter_mut().enumerate() {
        *dst = (transfer_function.linearize(i as f32 / 255.) * max_lin_depth as f32)
            .round()
            .min(max_lin_depth as f32) as u16;
    }

    Linearization {
        linearization: linearizing,
    }
}

fn make_gamma(transfer_function: TransferFunction) -> Gamma8 {
    let mut gamma = Box::new([0u8; 65536]);
    let max_lin_depth = (1u32 << 16) - 1;

    for (i, dst) in gamma.iter_mut().enumerate() {
        *dst = (transfer_function.gamma(i as f32 / max_lin_depth as f32) * 255.)
            .round()
            .min(255.) as u8;
    }

    Gamma8 { gamma }
}

fn make_gamma16(transfer_function: TransferFunction) -> Gamma16 {
    let mut gamma = Box::new([0u16; 65536]);
    let max_lin_depth = (1u32 << 16) - 1;

    for (i, dst) in gamma.iter_mut().enumerate() {
        *dst = (transfer_function.gamma(i as f32 / max_lin_depth as f32) * 65535.)
            .round()
            .min(65535.) as u16;
    }

    Gamma16 { gamma }
}

impl BlurImage<'_, u8> {
    /// Linearize image
    ///
    /// # Arguments
    ///
    /// * `transfer_function`: See [TransferFunction] for more info.
    /// * `may_have_alpha`: If image could have alpha, image with channels 2 and 4 will consider channels 1 and 3 as alpha.
    pub fn linearize(
        &self,
        transfer_function: TransferFunction,
        may_have_alpha: bool,
    ) -> Result<BlurImage<'_, u16>, BlurError> {
        self.check_layout()?;
        let mut new_image =
            vec![0u16; self.width as usize * self.height as usize * self.channels.channels()];
        let row_stride = self.width as usize * self.channels.channels();
        let linearization = make_linearization(transfer_function);
        if !may_have_alpha || (self.channels != FastBlurChannels::Channels4) {
            for (dst, src) in new_image
                .chunks_exact_mut(row_stride)
                .zip(self.data.chunks_exact(self.row_stride() as usize))
            {
                let src = &src[..self.width as usize * self.channels.channels()];
                let dst = &mut dst[..self.width as usize * self.channels.channels()];

                for (dst, src) in dst.iter_mut().zip(src.iter()) {
                    *dst = linearization.linearization[*src as usize];
                }
            }
        } else {
            // ATM only 4 channels here possible, so hardcoded 4
            for (dst, src) in new_image
                .chunks_exact_mut(row_stride)
                .zip(self.data.chunks_exact(self.row_stride() as usize))
            {
                let src = &src[..self.width as usize * self.channels.channels()];
                let dst = &mut dst[..self.width as usize * self.channels.channels()];
                for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
                    dst[0] = linearization.linearization[src[0] as usize];
                    dst[1] = linearization.linearization[src[1] as usize];
                    dst[2] = linearization.linearization[src[2] as usize];
                    dst[3] = u16::from_ne_bytes([src[3], src[3]]);
                }
            }
        }

        Ok(BlurImage {
            data: std::borrow::Cow::Owned(new_image),
            width: self.width,
            stride: row_stride as u32,
            height: self.height,
            channels: self.channels,
        })
    }
}

impl BlurImage<'_, u16> {
    /// Linearize image
    ///
    /// # Arguments
    ///
    /// * `transfer_function`: See [TransferFunction] for more info.
    /// * `may_have_alpha`: If image could have alpha, image with channels 2 and 4 will consider channels 1 and 3 as alpha.
    pub fn linearize(
        &self,
        transfer_function: TransferFunction,
        may_have_alpha: bool,
    ) -> Result<BlurImage<'_, u16>, BlurError> {
        self.check_layout()?;
        let mut new_image =
            vec![0u16; self.width as usize * self.height as usize * self.channels.channels()];
        let row_stride = self.width as usize * self.channels.channels();
        let linearization = make_linearization16(transfer_function);
        if !may_have_alpha || (self.channels != FastBlurChannels::Channels4) {
            for (dst, src) in new_image
                .chunks_exact_mut(row_stride)
                .zip(self.data.chunks_exact(self.row_stride() as usize))
            {
                let src = &src[..self.width as usize * self.channels.channels()];
                let dst = &mut dst[..self.width as usize * self.channels.channels()];
                for (dst, src) in dst.iter_mut().zip(src.iter()) {
                    *dst = linearization.linearization[*src as usize];
                }
            }
        } else {
            // ATM only 4 channels here possible, so hardcoded 4
            for (dst, src) in new_image
                .chunks_exact_mut(row_stride)
                .zip(self.data.chunks_exact(self.row_stride() as usize))
            {
                let src = &src[..self.width as usize * self.channels.channels()];
                let dst = &mut dst[..self.width as usize * self.channels.channels()];
                for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
                    dst[0] = linearization.linearization[src[0] as usize];
                    dst[1] = linearization.linearization[src[1] as usize];
                    dst[2] = linearization.linearization[src[2] as usize];
                    dst[3] = src[3];
                }
            }
        }

        Ok(BlurImage {
            data: std::borrow::Cow::Owned(new_image),
            width: self.width,
            stride: row_stride as u32,
            height: self.height,
            channels: self.channels,
        })
    }

    /// Converts an image to gamma 8-bit
    ///
    /// # Arguments
    ///
    /// * `transfer_function`: See [TransferFunction] for more info.
    /// * `may_have_alpha`: If image could have alpha, image with channels 2 and 4 will consider channels 1 and 3 as alpha.
    pub fn gamma8(
        &self,
        transfer_function: TransferFunction,
        may_have_alpha: bool,
    ) -> Result<BlurImage<'_, u8>, BlurError> {
        self.check_layout()?;
        let mut new_image =
            vec![0u8; self.width as usize * self.height as usize * self.channels.channels()];
        let row_stride = self.width as usize * self.channels.channels();
        let gamma = make_gamma(transfer_function);
        if !may_have_alpha || (self.channels != FastBlurChannels::Channels4) {
            for (dst, src) in new_image
                .chunks_exact_mut(row_stride)
                .zip(self.data.chunks_exact(self.row_stride() as usize))
            {
                let src = &src[..self.width as usize * self.channels.channels()];
                let dst = &mut dst[..self.width as usize * self.channels.channels()];
                for (dst, src) in dst.iter_mut().zip(src.iter()) {
                    *dst = gamma.gamma[*src as usize];
                }
            }
        } else {
            // ATM only 4 channels here possible, so hardcoded 4
            for (dst, src) in new_image
                .chunks_exact_mut(row_stride)
                .zip(self.data.chunks_exact(self.row_stride() as usize))
            {
                let src = &src[..self.width as usize * self.channels.channels()];
                let dst = &mut dst[..self.width as usize * self.channels.channels()];
                for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
                    dst[0] = gamma.gamma[src[0] as usize];
                    dst[1] = gamma.gamma[src[1] as usize];
                    dst[2] = gamma.gamma[src[2] as usize];
                    dst[3] = (src[3] >> 8) as u8;
                }
            }
        }

        Ok(BlurImage {
            data: std::borrow::Cow::Owned(new_image),
            width: self.width,
            stride: row_stride as u32,
            height: self.height,
            channels: self.channels,
        })
    }

    /// Converts an image to gamma 16-bit
    ///
    /// # Arguments
    ///
    /// * `transfer_function`: See [TransferFunction] for more info.
    /// * `may_have_alpha`: If image could have alpha, image with channels 2 and 4 will consider channels 1 and 3 as alpha.
    pub fn gamma16(
        &self,
        transfer_function: TransferFunction,
        may_have_alpha: bool,
    ) -> Result<BlurImage<'_, u16>, BlurError> {
        self.check_layout()?;
        let mut new_image =
            vec![0u16; self.width as usize * self.height as usize * self.channels.channels()];
        let row_stride = self.width as usize * self.channels.channels();
        let gamma = make_gamma16(transfer_function);
        if !may_have_alpha || (self.channels != FastBlurChannels::Channels4) {
            for (dst, src) in new_image
                .chunks_exact_mut(row_stride)
                .zip(self.data.chunks_exact(self.row_stride() as usize))
            {
                let src = &src[..self.width as usize * self.channels.channels()];
                let dst = &mut dst[..self.width as usize * self.channels.channels()];
                for (dst, src) in dst.iter_mut().zip(src.iter()) {
                    *dst = gamma.gamma[*src as usize];
                }
            }
        } else {
            // ATM only 4 channels here possible, so hardcoded 4
            for (dst, src) in new_image
                .chunks_exact_mut(row_stride)
                .zip(self.data.chunks_exact(self.row_stride() as usize))
            {
                let src = &src[..self.width as usize * self.channels.channels()];
                let dst = &mut dst[..self.width as usize * self.channels.channels()];
                for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
                    dst[0] = gamma.gamma[src[0] as usize];
                    dst[1] = gamma.gamma[src[1] as usize];
                    dst[2] = gamma.gamma[src[2] as usize];
                    dst[3] = src[3];
                }
            }
        }

        Ok(BlurImage {
            data: std::borrow::Cow::Owned(new_image),
            width: self.width,
            stride: row_stride as u32,
            height: self.height,
            channels: self.channels,
        })
    }
}

impl BlurImage<'_, f32> {
    /// Linearize image
    ///
    /// # Arguments
    ///
    /// * `transfer_function`: See [TransferFunction] for more info.
    /// * `may_have_alpha`: If image could have alpha, image with channels 2 and 4 will consider channels 1 and 3 as alpha.
    pub fn linearize(
        &self,
        transfer_function: TransferFunction,
        may_have_alpha: bool,
    ) -> Result<BlurImage<'_, f32>, BlurError> {
        self.check_layout()?;
        let mut new_image =
            vec![0.; self.width as usize * self.height as usize * self.channels.channels()];
        let row_stride = self.width as usize * self.channels.channels();
        if !may_have_alpha || (self.channels != FastBlurChannels::Channels4) {
            for (dst, src) in new_image
                .chunks_exact_mut(row_stride)
                .zip(self.data.chunks_exact(self.row_stride() as usize))
            {
                let src = &src[..self.width as usize * self.channels.channels()];
                let dst = &mut dst[..self.width as usize * self.channels.channels()];
                for (dst, src) in dst.iter_mut().zip(src.iter()) {
                    *dst = transfer_function.linearize(*src);
                }
            }
        } else {
            // ATM only 4 channels here possible, so hardcoded 4
            for (dst, src) in new_image
                .chunks_exact_mut(row_stride)
                .zip(self.data.chunks_exact(self.row_stride() as usize))
            {
                let src = &src[..self.width as usize * self.channels.channels()];
                let dst = &mut dst[..self.width as usize * self.channels.channels()];
                for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
                    dst[0] = transfer_function.linearize(src[0]);
                    dst[1] = transfer_function.linearize(src[1]);
                    dst[2] = transfer_function.linearize(src[2]);
                    dst[3] = src[3];
                }
            }
        }

        Ok(BlurImage {
            data: std::borrow::Cow::Owned(new_image),
            width: self.width,
            stride: row_stride as u32,
            height: self.height,
            channels: self.channels,
        })
    }

    /// Converts an image to gamma
    ///
    /// # Arguments
    ///
    /// * `transfer_function`: See [TransferFunction] for more info.
    /// * `may_have_alpha`: If image could have alpha, image with channels 2 and 4 will consider channels 1 and 3 as alpha.
    pub fn gamma(
        &self,
        transfer_function: TransferFunction,
        may_have_alpha: bool,
    ) -> Result<BlurImage<'_, f32>, BlurError> {
        self.check_layout()?;
        let mut new_image =
            vec![0.; self.width as usize * self.height as usize * self.channels.channels()];
        let row_stride = self.width as usize * self.channels.channels();
        if !may_have_alpha || (self.channels != FastBlurChannels::Channels4) {
            for (dst, src) in new_image
                .chunks_exact_mut(row_stride)
                .zip(self.data.chunks_exact(self.row_stride() as usize))
            {
                let src = &src[..self.width as usize * self.channels.channels()];
                let dst = &mut dst[..self.width as usize * self.channels.channels()];
                for (dst, src) in dst.iter_mut().zip(src.iter()) {
                    *dst = transfer_function.gamma(*src);
                }
            }
        } else {
            // ATM only 4 channels here possible, so hardcoded 4
            for (dst, src) in new_image
                .chunks_exact_mut(row_stride)
                .zip(self.data.chunks_exact(self.row_stride() as usize))
            {
                let src = &src[..self.width as usize * self.channels.channels()];
                let dst = &mut dst[..self.width as usize * self.channels.channels()];
                for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
                    dst[0] = transfer_function.gamma(src[0]);
                    dst[1] = transfer_function.gamma(src[1]);
                    dst[2] = transfer_function.gamma(src[2]);
                    dst[3] = src[3];
                }
            }
        }

        Ok(BlurImage {
            data: std::borrow::Cow::Owned(new_image),
            width: self.width,
            stride: row_stride as u32,
            height: self.height,
            channels: self.channels,
        })
    }
}
