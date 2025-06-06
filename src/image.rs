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
use crate::{BlurError, FastBlurChannels, ImageSize, MismatchedSize};
use std::fmt::Debug;

#[derive(Debug)]
pub enum BufferStore<'a, T: Copy + Debug> {
    Borrowed(&'a mut [T]),
    Owned(Vec<T>),
}

impl<T: Copy + Debug> BufferStore<'_, T> {
    #[allow(clippy::should_implement_trait)]
    pub fn borrow(&self) -> &[T] {
        match self {
            Self::Borrowed(p_ref) => p_ref,
            Self::Owned(vec) => vec,
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn borrow_mut(&mut self) -> &mut [T] {
        match self {
            Self::Borrowed(p_ref) => p_ref,
            Self::Owned(vec) => vec,
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub(crate) fn resize(&mut self, new_size: usize, value: T) {
        match self {
            Self::Borrowed(_) => {}
            Self::Owned(vec) => vec.resize(new_size, value),
        }
    }
}

/// Immutable image store
pub struct BlurImage<'a, T: Clone + Copy + Default + Debug> {
    pub data: std::borrow::Cow<'a, [T]>,
    pub width: u32,
    pub height: u32,
    /// Image stride, items per row, might be 0
    pub stride: u32,
    pub channels: FastBlurChannels,
}

/// Mutable image store
/// If it owns vector it does auto resizing on methods that working out-of-place.
pub struct BlurImageMut<'a, T: Clone + Copy + Default + Debug> {
    pub data: BufferStore<'a, T>,
    pub width: u32,
    pub height: u32,
    /// Image stride, items per row, might be 0
    pub stride: u32,
    pub channels: FastBlurChannels,
}

impl<T: Clone + Copy + Default + Debug> Default for BlurImageMut<'_, T> {
    fn default() -> Self {
        BlurImageMut {
            data: BufferStore::Owned(Vec::new()),
            width: 0,
            height: 0,
            stride: 0,
            channels: FastBlurChannels::Plane,
        }
    }
}

impl<'a, T: Clone + Copy + Default + Debug> BlurImage<'a, T> {
    /// Allocates default image layout for given [FastBlurChannels]
    pub fn alloc(width: u32, height: u32, channels: FastBlurChannels) -> Self {
        Self {
            data: std::borrow::Cow::Owned(vec![
                T::default();
                width as usize
                    * height as usize
                    * channels.channels()
            ]),
            width,
            height,
            stride: width * channels.channels() as u32,
            channels,
        }
    }

    /// Borrows existing data
    /// Stride will be default `width * channels.channels()`
    pub fn borrow(arr: &'a [T], width: u32, height: u32, channels: FastBlurChannels) -> Self {
        Self {
            data: std::borrow::Cow::Borrowed(arr),
            width,
            height,
            stride: width * channels.channels() as u32,
            channels,
        }
    }

    /// Deep copy immutable image to mutable
    pub fn copy_to_mut(&self, dst: &mut BlurImageMut<T>) -> Result<(), BlurError> {
        self.check_layout()?;
        dst.check_layout(Some(self))?;
        self.size_matches_mut(dst)?;
        for (src, dst) in self
            .data
            .as_ref()
            .chunks_exact(self.row_stride() as usize)
            .zip(
                dst.data
                    .borrow_mut()
                    .chunks_exact_mut(self.row_stride() as usize),
            )
        {
            let src = &src[..self.width as usize * self.channels.channels()];
            let dst = &mut dst[..self.width as usize * self.channels.channels()];
            for (src, dst) in src.iter().zip(dst.iter_mut()) {
                *dst = *src;
            }
        }
        Ok(())
    }

    /// Checks if it is matches the size of the other image
    #[inline]
    pub fn size_matches(&self, other: &BlurImage<'_, T>) -> Result<(), BlurError> {
        if self.width == other.width
            && self.height == other.height
            && self.channels == other.channels
        {
            return Ok(());
        }
        Err(BlurError::ImagesMustMatch)
    }

    #[inline]
    pub fn size(&self) -> ImageSize {
        ImageSize::new(self.width as usize, self.height as usize)
    }

    /// Checks if it is matches the size of the other image
    #[inline]
    pub fn size_matches_mut(&self, other: &BlurImageMut<'_, T>) -> Result<(), BlurError> {
        if self.width == other.width
            && self.height == other.height
            && self.channels == other.channels
        {
            return Ok(());
        }
        Err(BlurError::ImagesMustMatch)
    }

    /// Checks if it is matches the size of the other image
    #[inline]
    pub fn only_size_matches_mut(&self, other: &BlurImageMut<'_, T>) -> Result<(), BlurError> {
        if self.width == other.width && self.height == other.height {
            return Ok(());
        }
        Err(BlurError::ImagesMustMatch)
    }

    /// Checks if layout matches necessary requirements by using external channels count
    #[inline]
    pub fn check_layout_channels(&self, cn: usize) -> Result<(), BlurError> {
        if self.width == 0 || self.height == 0 {
            return Err(BlurError::ZeroBaseSize);
        }
        let data_len = self.data.as_ref().len();
        if data_len < self.stride as usize * (self.height as usize - 1) + self.width as usize * cn {
            return Err(BlurError::MinimumSliceSizeMismatch(MismatchedSize {
                expected: self.stride as usize * self.height as usize,
                received: data_len,
            }));
        }
        if (self.stride as usize) < (self.width as usize * cn) {
            return Err(BlurError::MinimumStrideSizeMismatch(MismatchedSize {
                expected: self.width as usize * cn,
                received: self.stride as usize,
            }));
        }
        Ok(())
    }

    /// Returns row stride
    #[inline]
    pub fn row_stride(&self) -> u32 {
        if self.stride == 0 {
            self.width * self.channels.channels() as u32
        } else {
            self.stride
        }
    }

    #[inline]
    pub fn check_layout(&self) -> Result<(), BlurError> {
        if self.width == 0 || self.height == 0 {
            return Err(BlurError::ZeroBaseSize);
        }
        let cn = self.channels.channels();
        if self.data.len()
            < self.stride as usize * (self.height as usize - 1) + self.width as usize * cn
        {
            return Err(BlurError::MinimumSliceSizeMismatch(MismatchedSize {
                expected: self.stride as usize * self.height as usize,
                received: self.data.len(),
            }));
        }
        if (self.stride as usize) < (self.width as usize * cn) {
            return Err(BlurError::MinimumStrideSizeMismatch(MismatchedSize {
                expected: self.width as usize * cn,
                received: self.stride as usize,
            }));
        }
        Ok(())
    }

    /// Deep clone as mutable image
    pub fn clone_as_mut<'f>(&self) -> BlurImageMut<'f, T> {
        BlurImageMut {
            data: BufferStore::Owned(self.data.to_vec()),
            width: self.width,
            height: self.height,
            stride: self.stride,
            channels: self.channels,
        }
    }
}

impl<'a, T: Clone + Copy + Default + Debug> BlurImageMut<'a, T> {
    /// Allocates default image layout for given [FastBlurChannels]
    pub fn alloc(width: u32, height: u32, channels: FastBlurChannels) -> Self {
        Self {
            data: BufferStore::Owned(vec![
                T::default();
                width as usize * height as usize * channels.channels()
            ]),
            width,
            height,
            stride: width * channels.channels() as u32,
            channels,
        }
    }

    /// Mutable borrows existing data
    /// Stride will be default `width * channels.channels()`
    pub fn borrow(arr: &'a mut [T], width: u32, height: u32, channels: FastBlurChannels) -> Self {
        Self {
            data: BufferStore::Borrowed(arr),
            width,
            height,
            stride: width * channels.channels() as u32,
            channels,
        }
    }

    /// Returns row stride
    #[inline]
    pub fn row_stride(&self) -> u32 {
        if self.stride == 0 {
            self.width * self.channels.channels() as u32
        } else {
            self.stride
        }
    }

    #[inline]
    pub fn layout_test(&self) -> Result<(), BlurError> {
        if self.width == 0 || self.height == 0 {
            return Err(BlurError::ZeroBaseSize);
        }
        let cn = self.channels.channels();
        let data_len = self.data.borrow().len();
        if data_len < self.stride as usize * (self.height as usize - 1) + self.width as usize * cn {
            return Err(BlurError::MinimumSliceSizeMismatch(MismatchedSize {
                expected: self.stride as usize * self.height as usize,
                received: data_len,
            }));
        }
        if (self.stride as usize) < (self.width as usize * cn) {
            return Err(BlurError::MinimumStrideSizeMismatch(MismatchedSize {
                expected: self.width as usize * cn,
                received: self.stride as usize,
            }));
        }
        Ok(())
    }

    /// Checks if layout matches necessary requirements
    #[inline]
    pub fn check_layout(&mut self, other: Option<&BlurImage<'_, T>>) -> Result<(), BlurError> {
        if let Some(other) = other {
            if matches!(self.data, BufferStore::Owned(_)) {
                self.resize(other.width, other.height, other.channels);
                return Ok(());
            }
        }
        if self.width == 0 || self.height == 0 {
            return Err(BlurError::ZeroBaseSize);
        }
        let cn = self.channels.channels();
        let data_len = self.data.borrow().len();
        if data_len < self.stride as usize * (self.height as usize - 1) + self.width as usize * cn {
            return Err(BlurError::MinimumSliceSizeMismatch(MismatchedSize {
                expected: self.stride as usize * self.height as usize,
                received: data_len,
            }));
        }
        if (self.stride as usize) < (self.width as usize * cn) {
            return Err(BlurError::MinimumStrideSizeMismatch(MismatchedSize {
                expected: self.width as usize * cn,
                received: self.stride as usize,
            }));
        }
        Ok(())
    }

    /// Checks if layout matches necessary requirements by using external channels count
    #[inline]
    pub fn check_layout_channels(
        &mut self,
        cn: usize,
        other: Option<&BlurImage<'_, T>>,
    ) -> Result<(), BlurError> {
        if let Some(other) = other {
            if matches!(self.data, BufferStore::Owned(_)) {
                self.resize_arbitrary(other.width, other.height, cn);
                return Ok(());
            }
        }
        if self.width == 0 || self.height == 0 {
            return Err(BlurError::ZeroBaseSize);
        }
        let data_len = self.data.borrow().len();
        if data_len < self.stride as usize * (self.height as usize - 1) + self.width as usize * cn {
            return Err(BlurError::MinimumSliceSizeMismatch(MismatchedSize {
                expected: self.stride as usize * self.height as usize,
                received: data_len,
            }));
        }
        if (self.stride as usize) < (self.width as usize * cn) {
            return Err(BlurError::MinimumStrideSizeMismatch(MismatchedSize {
                expected: self.width as usize * cn,
                received: self.stride as usize,
            }));
        }
        Ok(())
    }

    /// Checks if it is matches the size of the other image
    #[inline]
    pub fn size_matches(&self, other: &BlurImage<'_, T>) -> Result<(), BlurError> {
        if self.width == other.width
            && self.height == other.height
            && self.channels == other.channels
        {
            return Ok(());
        }
        Err(BlurError::ImagesMustMatch)
    }

    /// Checks if it is matches the size of the other image
    #[inline]
    pub fn size_matches_mut(&self, other: &BlurImageMut<'_, T>) -> Result<(), BlurError> {
        if self.width == other.width
            && self.height == other.height
            && self.channels == other.channels
        {
            return Ok(());
        }
        Err(BlurError::ImagesMustMatch)
    }

    #[inline]
    pub fn to_immutable_ref(&self) -> BlurImage<'_, T> {
        BlurImage {
            data: std::borrow::Cow::Borrowed(self.data.borrow()),
            stride: self.row_stride(),
            width: self.width,
            height: self.height,
            channels: self.channels,
        }
    }

    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn resize(&mut self, width: u32, height: u32, channels: FastBlurChannels) {
        self.height = height;
        self.width = width;
        self.channels = channels;
        self.stride = self.width * self.channels.channels() as u32;
        self.data.resize(
            self.row_stride() as usize * self.height as usize,
            T::default(),
        );
    }

    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn resize_arbitrary(&mut self, width: u32, height: u32, cn: usize) {
        self.height = height;
        self.width = width;
        self.channels = if cn == 4 {
            FastBlurChannels::Channels4
        } else if cn == 3 {
            FastBlurChannels::Channels3
        } else {
            FastBlurChannels::Plane
        };
        self.stride = self.width * cn as u32;
        self.data.resize(
            self.row_stride() as usize * self.height as usize,
            T::default(),
        );
    }
}
