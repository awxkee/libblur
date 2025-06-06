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
use crate::{BlurError, BlurImage, BlurImageMut, BufferStore, FastBlurChannels};

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

trait FinalImageFactory<T, W> {
    fn make_image(
        &self,
        vec: Vec<W>,
        width: usize,
        height: usize,
        stride: usize,
        channels: FastBlurChannels,
    ) -> T;
}

#[derive(Default)]
struct ReturnImmutableImage8 {}

impl<'a> FinalImageFactory<BlurImage<'a, u8>, u8> for ReturnImmutableImage8 {
    fn make_image(
        &self,
        vec: Vec<u8>,
        width: usize,
        height: usize,
        stride: usize,
        channels: FastBlurChannels,
    ) -> BlurImage<'a, u8> {
        BlurImage {
            data: std::borrow::Cow::Owned(vec),
            width: width as u32,
            stride: stride as u32,
            height: height as u32,
            channels,
        }
    }
}

#[derive(Default)]
struct ReturnMutableImage8 {}

impl<'a> FinalImageFactory<BlurImageMut<'a, u8>, u8> for ReturnMutableImage8 {
    fn make_image(
        &self,
        vec: Vec<u8>,
        width: usize,
        height: usize,
        stride: usize,
        channels: FastBlurChannels,
    ) -> BlurImageMut<'a, u8> {
        BlurImageMut {
            data: BufferStore::Owned(vec),
            width: width as u32,
            stride: stride as u32,
            height: height as u32,
            channels,
        }
    }
}

#[derive(Default)]
struct ReturnImmutableImage16 {}

impl<'a> FinalImageFactory<BlurImage<'a, u16>, u16> for ReturnImmutableImage16 {
    fn make_image(
        &self,
        vec: Vec<u16>,
        width: usize,
        height: usize,
        stride: usize,
        channels: FastBlurChannels,
    ) -> BlurImage<'a, u16> {
        BlurImage {
            data: std::borrow::Cow::Owned(vec),
            width: width as u32,
            stride: stride as u32,
            height: height as u32,
            channels,
        }
    }
}

#[derive(Default)]
struct ReturnMutableImage16 {}

impl<'a> FinalImageFactory<BlurImageMut<'a, u16>, u16> for ReturnMutableImage16 {
    fn make_image(
        &self,
        vec: Vec<u16>,
        width: usize,
        height: usize,
        stride: usize,
        channels: FastBlurChannels,
    ) -> BlurImageMut<'a, u16> {
        BlurImageMut {
            data: BufferStore::Owned(vec),
            width: width as u32,
            stride: stride as u32,
            height: height as u32,
            channels,
        }
    }
}

#[derive(Default)]
struct ReturnImmutableImageF32 {}

impl<'a> FinalImageFactory<BlurImage<'a, f32>, f32> for ReturnImmutableImageF32 {
    fn make_image(
        &self,
        vec: Vec<f32>,
        width: usize,
        height: usize,
        stride: usize,
        channels: FastBlurChannels,
    ) -> BlurImage<'a, f32> {
        BlurImage {
            data: std::borrow::Cow::Owned(vec),
            width: width as u32,
            stride: stride as u32,
            height: height as u32,
            channels,
        }
    }
}

#[derive(Default)]
struct ReturnMutableImageF32 {}

impl<'a> FinalImageFactory<BlurImageMut<'a, f32>, f32> for ReturnMutableImageF32 {
    fn make_image(
        &self,
        vec: Vec<f32>,
        width: usize,
        height: usize,
        stride: usize,
        channels: FastBlurChannels,
    ) -> BlurImageMut<'a, f32> {
        BlurImageMut {
            data: BufferStore::Owned(vec),
            width: width as u32,
            stride: stride as u32,
            height: height as u32,
            channels,
        }
    }
}

/// Linearize image
///
/// # Arguments
///
/// * `transfer_function`: See [TransferFunction] for more info.
/// * `may_have_alpha`: If image could have alpha, image with channels 2 and 4 will consider channels 1 and 3 as alpha.
fn linearize8<Z, F: FinalImageFactory<Z, u16>>(
    src_ref: &BlurImage<'_, u8>,
    transfer_function: TransferFunction,
    may_have_alpha: bool,
    factory: F,
) -> Result<Z, BlurError> {
    src_ref.check_layout()?;
    let mut new_image =
        vec![0u16; src_ref.width as usize * src_ref.height as usize * src_ref.channels.channels()];
    let row_stride = src_ref.width as usize * src_ref.channels.channels();
    let linearization = make_linearization(transfer_function);
    if !may_have_alpha || (src_ref.channels != FastBlurChannels::Channels4) {
        for (dst, src) in new_image
            .chunks_exact_mut(row_stride)
            .zip(src_ref.data.chunks_exact(src_ref.row_stride() as usize))
        {
            let src = &src[..src_ref.width as usize * src_ref.channels.channels()];
            let dst = &mut dst[..src_ref.width as usize * src_ref.channels.channels()];

            for (dst, src) in dst.iter_mut().zip(src.iter()) {
                *dst = linearization.linearization[*src as usize];
            }
        }
    } else {
        // ATM only 4 channels here possible, so hardcoded 4
        for (dst, src) in new_image
            .chunks_exact_mut(row_stride)
            .zip(src_ref.data.chunks_exact(src_ref.row_stride() as usize))
        {
            let src = &src[..src_ref.width as usize * src_ref.channels.channels()];
            let dst = &mut dst[..src_ref.width as usize * src_ref.channels.channels()];
            for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
                dst[0] = linearization.linearization[src[0] as usize];
                dst[1] = linearization.linearization[src[1] as usize];
                dst[2] = linearization.linearization[src[2] as usize];
                dst[3] = u16::from_ne_bytes([src[3], src[3]]);
            }
        }
    }

    Ok(factory.make_image(
        new_image,
        src_ref.width as usize,
        src_ref.height as usize,
        src_ref.row_stride() as usize,
        src_ref.channels,
    ))
}

/// Converts an image to gamma 8-bit
///
/// # Arguments
///
/// * `transfer_function`: See [TransferFunction] for more info.
/// * `may_have_alpha`: If image could have alpha, image with channels 2 and 4 will consider channels 1 and 3 as alpha.
fn gen_gamma8<Z, F: FinalImageFactory<Z, u8>>(
    src_ref: &BlurImage<'_, u16>,
    transfer_function: TransferFunction,
    may_have_alpha: bool,
    factory: F,
) -> Result<Z, BlurError> {
    src_ref.check_layout()?;
    let mut new_image =
        vec![0u8; src_ref.width as usize * src_ref.height as usize * src_ref.channels.channels()];
    let row_stride = src_ref.width as usize * src_ref.channels.channels();
    let gamma = make_gamma(transfer_function);
    if !may_have_alpha || (src_ref.channels != FastBlurChannels::Channels4) {
        for (dst, src) in new_image
            .chunks_exact_mut(row_stride)
            .zip(src_ref.data.chunks_exact(src_ref.row_stride() as usize))
        {
            let src = &src[..src_ref.width as usize * src_ref.channels.channels()];
            let dst = &mut dst[..src_ref.width as usize * src_ref.channels.channels()];
            for (dst, src) in dst.iter_mut().zip(src.iter()) {
                *dst = gamma.gamma[*src as usize];
            }
        }
    } else {
        // ATM only 4 channels here possible, so hardcoded 4
        for (dst, src) in new_image
            .chunks_exact_mut(row_stride)
            .zip(src_ref.data.chunks_exact(src_ref.row_stride() as usize))
        {
            let src = &src[..src_ref.width as usize * src_ref.channels.channels()];
            let dst = &mut dst[..src_ref.width as usize * src_ref.channels.channels()];
            for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
                dst[0] = gamma.gamma[src[0] as usize];
                dst[1] = gamma.gamma[src[1] as usize];
                dst[2] = gamma.gamma[src[2] as usize];
                dst[3] = (src[3] >> 8) as u8;
            }
        }
    }

    Ok(factory.make_image(
        new_image,
        src_ref.width as usize,
        src_ref.height as usize,
        src_ref.row_stride() as usize,
        src_ref.channels,
    ))
}

/// Linearize image
///
/// # Arguments
///
/// * `transfer_function`: See [TransferFunction] for more info.
/// * `may_have_alpha`: If image could have alpha, image with channels 2 and 4 will consider channels 1 and 3 as alpha.
fn linearize16<Z, F: FinalImageFactory<Z, u16>>(
    src_ref: &BlurImage<'_, u16>,
    transfer_function: TransferFunction,
    may_have_alpha: bool,
    factory: F,
) -> Result<Z, BlurError> {
    src_ref.check_layout()?;
    let mut new_image =
        vec![0u16; src_ref.width as usize * src_ref.height as usize * src_ref.channels.channels()];
    let row_stride = src_ref.width as usize * src_ref.channels.channels();
    let linearization = make_linearization16(transfer_function);
    if !may_have_alpha || (src_ref.channels != FastBlurChannels::Channels4) {
        for (dst, src) in new_image
            .chunks_exact_mut(row_stride)
            .zip(src_ref.data.chunks_exact(src_ref.row_stride() as usize))
        {
            let src = &src[..src_ref.width as usize * src_ref.channels.channels()];
            let dst = &mut dst[..src_ref.width as usize * src_ref.channels.channels()];
            for (dst, src) in dst.iter_mut().zip(src.iter()) {
                *dst = linearization.linearization[*src as usize];
            }
        }
    } else {
        // ATM only 4 channels here possible, so hardcoded 4
        for (dst, src) in new_image
            .chunks_exact_mut(row_stride)
            .zip(src_ref.data.chunks_exact(src_ref.row_stride() as usize))
        {
            let src = &src[..src_ref.width as usize * src_ref.channels.channels()];
            let dst = &mut dst[..src_ref.width as usize * src_ref.channels.channels()];
            for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
                dst[0] = linearization.linearization[src[0] as usize];
                dst[1] = linearization.linearization[src[1] as usize];
                dst[2] = linearization.linearization[src[2] as usize];
                dst[3] = src[3];
            }
        }
    }

    Ok(factory.make_image(
        new_image,
        src_ref.width as usize,
        src_ref.height as usize,
        src_ref.row_stride() as usize,
        src_ref.channels,
    ))
}

/// Converts an image to gamma 16-bit
///
/// # Arguments
///
/// * `transfer_function`: See [TransferFunction] for more info.
/// * `may_have_alpha`: If image could have alpha, image with channels 2 and 4 will consider channels 1 and 3 as alpha.
fn gen_gamma16<Z, F: FinalImageFactory<Z, u16>>(
    src_ref: &BlurImage<'_, u16>,
    transfer_function: TransferFunction,
    may_have_alpha: bool,
    factory: F,
) -> Result<Z, BlurError> {
    src_ref.check_layout()?;
    let mut new_image =
        vec![0u16; src_ref.width as usize * src_ref.height as usize * src_ref.channels.channels()];
    let row_stride = src_ref.width as usize * src_ref.channels.channels();
    let gamma = make_gamma16(transfer_function);
    if !may_have_alpha || (src_ref.channels != FastBlurChannels::Channels4) {
        for (dst, src) in new_image
            .chunks_exact_mut(row_stride)
            .zip(src_ref.data.chunks_exact(src_ref.row_stride() as usize))
        {
            let src = &src[..src_ref.width as usize * src_ref.channels.channels()];
            let dst = &mut dst[..src_ref.width as usize * src_ref.channels.channels()];
            for (dst, src) in dst.iter_mut().zip(src.iter()) {
                *dst = gamma.gamma[*src as usize];
            }
        }
    } else {
        // ATM only 4 channels here possible, so hardcoded 4
        for (dst, src) in new_image
            .chunks_exact_mut(row_stride)
            .zip(src_ref.data.chunks_exact(src_ref.row_stride() as usize))
        {
            let src = &src[..src_ref.width as usize * src_ref.channels.channels()];
            let dst = &mut dst[..src_ref.width as usize * src_ref.channels.channels()];
            for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
                dst[0] = gamma.gamma[src[0] as usize];
                dst[1] = gamma.gamma[src[1] as usize];
                dst[2] = gamma.gamma[src[2] as usize];
                dst[3] = src[3];
            }
        }
    }

    Ok(factory.make_image(
        new_image,
        src_ref.width as usize,
        src_ref.height as usize,
        src_ref.row_stride() as usize,
        src_ref.channels,
    ))
}

/// Linearize image
///
/// # Arguments
///
/// * `transfer_function`: See [TransferFunction] for more info.
/// * `may_have_alpha`: If image could have alpha, image with channels 2 and 4 will consider channels 1 and 3 as alpha.
fn linearize_f32<Z, F: FinalImageFactory<Z, f32>>(
    src_ref: &BlurImage<'_, f32>,
    transfer_function: TransferFunction,
    may_have_alpha: bool,
    factory: F,
) -> Result<Z, BlurError> {
    src_ref.check_layout()?;
    let mut new_image =
        vec![0.; src_ref.width as usize * src_ref.height as usize * src_ref.channels.channels()];
    let row_stride = src_ref.width as usize * src_ref.channels.channels();
    if !may_have_alpha || (src_ref.channels != FastBlurChannels::Channels4) {
        for (dst, src) in new_image
            .chunks_exact_mut(row_stride)
            .zip(src_ref.data.chunks_exact(src_ref.row_stride() as usize))
        {
            let src = &src[..src_ref.width as usize * src_ref.channels.channels()];
            let dst = &mut dst[..src_ref.width as usize * src_ref.channels.channels()];
            for (dst, src) in dst.iter_mut().zip(src.iter()) {
                *dst = transfer_function.linearize(*src);
            }
        }
    } else {
        // ATM only 4 channels here possible, so hardcoded 4
        for (dst, src) in new_image
            .chunks_exact_mut(row_stride)
            .zip(src_ref.data.chunks_exact(src_ref.row_stride() as usize))
        {
            let src = &src[..src_ref.width as usize * src_ref.channels.channels()];
            let dst = &mut dst[..src_ref.width as usize * src_ref.channels.channels()];
            for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
                dst[0] = transfer_function.linearize(src[0]);
                dst[1] = transfer_function.linearize(src[1]);
                dst[2] = transfer_function.linearize(src[2]);
                dst[3] = src[3];
            }
        }
    }

    Ok(factory.make_image(
        new_image,
        src_ref.width as usize,
        src_ref.height as usize,
        src_ref.row_stride() as usize,
        src_ref.channels,
    ))
}

/// Converts an image to gamma
///
/// # Arguments
///
/// * `transfer_function`: See [TransferFunction] for more info.
/// * `may_have_alpha`: If image could have alpha, image with channels 2 and 4 will consider channels 1 and 3 as alpha.
fn gamma_f32<Z, F: FinalImageFactory<Z, f32>>(
    src_ref: &BlurImage<'_, f32>,
    transfer_function: TransferFunction,
    may_have_alpha: bool,
    factory: F,
) -> Result<Z, BlurError> {
    src_ref.check_layout()?;
    let mut new_image =
        vec![0.; src_ref.width as usize * src_ref.height as usize * src_ref.channels.channels()];
    let row_stride = src_ref.width as usize * src_ref.channels.channels();
    if !may_have_alpha || (src_ref.channels != FastBlurChannels::Channels4) {
        for (dst, src) in new_image
            .chunks_exact_mut(row_stride)
            .zip(src_ref.data.chunks_exact(src_ref.row_stride() as usize))
        {
            let src = &src[..src_ref.width as usize * src_ref.channels.channels()];
            let dst = &mut dst[..src_ref.width as usize * src_ref.channels.channels()];
            for (dst, src) in dst.iter_mut().zip(src.iter()) {
                *dst = transfer_function.gamma(*src);
            }
        }
    } else {
        // ATM only 4 channels here possible, so hardcoded 4
        for (dst, src) in new_image
            .chunks_exact_mut(row_stride)
            .zip(src_ref.data.chunks_exact(src_ref.row_stride() as usize))
        {
            let src = &src[..src_ref.width as usize * src_ref.channels.channels()];
            let dst = &mut dst[..src_ref.width as usize * src_ref.channels.channels()];
            for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
                dst[0] = transfer_function.gamma(src[0]);
                dst[1] = transfer_function.gamma(src[1]);
                dst[2] = transfer_function.gamma(src[2]);
                dst[3] = src[3];
            }
        }
    }

    Ok(factory.make_image(
        new_image,
        src_ref.width as usize,
        src_ref.height as usize,
        src_ref.row_stride() as usize,
        src_ref.channels,
    ))
}

impl BlurImage<'_, u8> {
    /// Linearize image
    ///
    /// # Arguments
    ///
    /// * `transfer_function`: See [TransferFunction] for more info.
    /// * `may_have_alpha`: If image could have alpha, image with channels 2 and 4 will consider channels 1 and 3 as alpha.
    pub fn linearize<'f>(
        &self,
        transfer_function: TransferFunction,
        may_have_alpha: bool,
    ) -> Result<BlurImage<'f, u16>, BlurError> {
        linearize8::<BlurImage<'f, u16>, ReturnImmutableImage16>(
            self,
            transfer_function,
            may_have_alpha,
            ReturnImmutableImage16::default(),
        )
    }
}

impl BlurImageMut<'_, u8> {
    /// Linearize image
    ///
    /// # Arguments
    ///
    /// * `transfer_function`: See [TransferFunction] for more info.
    /// * `may_have_alpha`: If image could have alpha, image with channels 2 and 4 will consider channels 1 and 3 as alpha.
    pub fn linearize<'f>(
        &self,
        transfer_function: TransferFunction,
        may_have_alpha: bool,
    ) -> Result<BlurImageMut<'f, u16>, BlurError> {
        let ref0 = self.to_immutable_ref();
        linearize8::<BlurImageMut<'f, u16>, ReturnMutableImage16>(
            &ref0,
            transfer_function,
            may_have_alpha,
            ReturnMutableImage16::default(),
        )
    }
}

impl BlurImage<'_, u16> {
    /// Linearize image
    ///
    /// # Arguments
    ///
    /// * `transfer_function`: See [TransferFunction] for more info.
    /// * `may_have_alpha`: If image could have alpha, image with channels 2 and 4 will consider channels 1 and 3 as alpha.
    pub fn linearize<'f>(
        &self,
        transfer_function: TransferFunction,
        may_have_alpha: bool,
    ) -> Result<BlurImage<'f, u16>, BlurError> {
        linearize16::<BlurImage<'f, u16>, ReturnImmutableImage16>(
            self,
            transfer_function,
            may_have_alpha,
            ReturnImmutableImage16::default(),
        )
    }

    /// Converts an image to gamma 8-bit
    ///
    /// # Arguments
    ///
    /// * `transfer_function`: See [TransferFunction] for more info.
    /// * `may_have_alpha`: If image could have alpha, image with channels 2 and 4 will consider channels 1 and 3 as alpha.
    pub fn gamma8<'f>(
        &self,
        transfer_function: TransferFunction,
        may_have_alpha: bool,
    ) -> Result<BlurImage<'f, u8>, BlurError> {
        gen_gamma8::<BlurImage<'f, u8>, ReturnImmutableImage8>(
            self,
            transfer_function,
            may_have_alpha,
            ReturnImmutableImage8::default(),
        )
    }

    /// Converts an image to gamma 16-bit
    ///
    /// # Arguments
    ///
    /// * `transfer_function`: See [TransferFunction] for more info.
    /// * `may_have_alpha`: If image could have alpha, image with channels 2 and 4 will consider channels 1 and 3 as alpha.
    pub fn gamma16<'f>(
        &self,
        transfer_function: TransferFunction,
        may_have_alpha: bool,
    ) -> Result<BlurImage<'f, u16>, BlurError> {
        gen_gamma16::<BlurImage<'f, u16>, ReturnImmutableImage16>(
            self,
            transfer_function,
            may_have_alpha,
            ReturnImmutableImage16::default(),
        )
    }
}

impl BlurImageMut<'_, u16> {
    /// Linearize image
    ///
    /// # Arguments
    ///
    /// * `transfer_function`: See [TransferFunction] for more info.
    /// * `may_have_alpha`: If image could have alpha, image with channels 2 and 4 will consider channels 1 and 3 as alpha.
    pub fn linearize<'f>(
        &self,
        transfer_function: TransferFunction,
        may_have_alpha: bool,
    ) -> Result<BlurImageMut<'f, u16>, BlurError> {
        let src_ref = self.to_immutable_ref();
        linearize16::<BlurImageMut<'f, u16>, ReturnMutableImage16>(
            &src_ref,
            transfer_function,
            may_have_alpha,
            ReturnMutableImage16::default(),
        )
    }

    /// Converts an image to gamma 8-bit
    ///
    /// # Arguments
    ///
    /// * `transfer_function`: See [TransferFunction] for more info.
    /// * `may_have_alpha`: If image could have alpha, image with channels 2 and 4 will consider channels 1 and 3 as alpha.
    pub fn gamma8<'f>(
        &self,
        transfer_function: TransferFunction,
        may_have_alpha: bool,
    ) -> Result<BlurImageMut<'f, u8>, BlurError> {
        let src_ref = self.to_immutable_ref();
        gen_gamma8::<BlurImageMut<'f, u8>, ReturnMutableImage8>(
            &src_ref,
            transfer_function,
            may_have_alpha,
            ReturnMutableImage8::default(),
        )
    }

    /// Converts an image to gamma 16-bit
    ///
    /// # Arguments
    ///
    /// * `transfer_function`: See [TransferFunction] for more info.
    /// * `may_have_alpha`: If image could have alpha, image with channels 2 and 4 will consider channels 1 and 3 as alpha.
    pub fn gamma16<'f>(
        &self,
        transfer_function: TransferFunction,
        may_have_alpha: bool,
    ) -> Result<BlurImageMut<'f, u16>, BlurError> {
        let src_ref = self.to_immutable_ref();
        gen_gamma16::<BlurImageMut<'f, u16>, ReturnMutableImage16>(
            &src_ref,
            transfer_function,
            may_have_alpha,
            ReturnMutableImage16::default(),
        )
    }
}

impl BlurImage<'_, f32> {
    /// Linearize image
    ///
    /// # Arguments
    ///
    /// * `transfer_function`: See [TransferFunction] for more info.
    /// * `may_have_alpha`: If image could have alpha, image with channels 2 and 4 will consider channels 1 and 3 as alpha.
    pub fn linearize<'f>(
        &self,
        transfer_function: TransferFunction,
        may_have_alpha: bool,
    ) -> Result<BlurImage<'f, f32>, BlurError> {
        linearize_f32::<BlurImage<'f, f32>, ReturnImmutableImageF32>(
            self,
            transfer_function,
            may_have_alpha,
            ReturnImmutableImageF32::default(),
        )
    }

    /// Converts an image to gamma
    ///
    /// # Arguments
    ///
    /// * `transfer_function`: See [TransferFunction] for more info.
    /// * `may_have_alpha`: If image could have alpha, image with channels 2 and 4 will consider channels 1 and 3 as alpha.
    pub fn gamma<'f>(
        &self,
        transfer_function: TransferFunction,
        may_have_alpha: bool,
    ) -> Result<BlurImage<'f, f32>, BlurError> {
        gamma_f32::<BlurImage<'f, f32>, ReturnImmutableImageF32>(
            self,
            transfer_function,
            may_have_alpha,
            ReturnImmutableImageF32::default(),
        )
    }
}

impl BlurImageMut<'_, f32> {
    /// Linearize image
    ///
    /// # Arguments
    ///
    /// * `transfer_function`: See [TransferFunction] for more info.
    /// * `may_have_alpha`: If image could have alpha, image with channels 2 and 4 will consider channels 1 and 3 as alpha.
    pub fn linearize<'f>(
        &self,
        transfer_function: TransferFunction,
        may_have_alpha: bool,
    ) -> Result<BlurImageMut<'f, f32>, BlurError> {
        let src_ref = self.to_immutable_ref();
        linearize_f32::<BlurImageMut<'f, f32>, ReturnMutableImageF32>(
            &src_ref,
            transfer_function,
            may_have_alpha,
            ReturnMutableImageF32::default(),
        )
    }

    /// Converts an image to gamma
    ///
    /// # Arguments
    ///
    /// * `transfer_function`: See [TransferFunction] for more info.
    /// * `may_have_alpha`: If image could have alpha, image with channels 2 and 4 will consider channels 1 and 3 as alpha.
    pub fn gamma<'f>(
        &self,
        transfer_function: TransferFunction,
        may_have_alpha: bool,
    ) -> Result<BlurImageMut<'f, f32>, BlurError> {
        let src_ref = self.to_immutable_ref();
        gamma_f32::<BlurImageMut<'f, f32>, ReturnMutableImageF32>(
            &src_ref,
            transfer_function,
            may_have_alpha,
            ReturnMutableImageF32::default(),
        )
    }
}
