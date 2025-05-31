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
use crate::gaussian::gaussian_kernel_1d;
use crate::unsafe_slice::UnsafeSlice;
use crate::{BlurError, BlurImage, BlurImageMut, FastBlurChannels, ThreadingPolicy};
use num_traits::real::Real;
use num_traits::AsPrimitive;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::{IntoParallelIterator, ParallelSlice, ParallelSliceMut};
use rayon::ThreadPool;
use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Div, Index, Mul, MulAssign, Sub};

#[derive(Debug, Clone, Copy)]
struct Vector<T> {
    pub x: T,
    pub y: T,
}

impl<T> Mul<T> for Vector<T>
where
    T: Mul<Output = T> + Copy,
{
    type Output = Vector<T>;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Vector::new(self.x * rhs, self.y * rhs)
    }
}

impl<T> Vector<T>
where
    T: AsReal + Copy,
{
    #[inline]
    fn to_real(self) -> Vector<f32> {
        Vector::new(self.x.as_real(), self.y.as_real())
    }

    #[inline]
    fn from_real(real: &Vector<f32>) -> Vector<T> {
        Vector::new(T::from_real(real.x), T::from_real(real.y))
    }
}

impl<T> Mul<Vector<T>> for Vector<T>
where
    T: Mul<T, Output = T> + Copy,
{
    type Output = Vector<T>;

    #[inline]
    fn mul(self, rhs: Vector<T>) -> Self::Output {
        Vector::new(self.x * rhs.x, self.y * rhs.y)
    }
}

impl<T> Add<Vector<T>> for Vector<T>
where
    T: Add<T, Output = T> + Copy,
{
    type Output = Vector<T>;

    #[inline]
    fn add(self, rhs: Vector<T>) -> Self::Output {
        Vector::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl<T> Div<Vector<T>> for Vector<T>
where
    T: Div<T, Output = T> + Copy,
{
    type Output = Vector<T>;

    #[inline]
    fn div(self, rhs: Vector<T>) -> Self::Output {
        Vector::new(self.x / rhs.x, self.y / rhs.y)
    }
}

impl Mul<Vector<f32>> for f32 {
    type Output = Vector<f32>;
    #[inline]
    fn mul(self, rhs: Vector<f32>) -> Self::Output {
        Vector::new(self * rhs.x, self * rhs.y)
    }
}

impl<T> AddAssign<Vector<T>> for Vector<T>
where
    T: Add<T, Output = T> + Copy,
{
    #[inline]
    fn add_assign(&mut self, rhs: Vector<T>) {
        self.x = self.x + rhs.x;
        self.y = self.y + rhs.y;
    }
}

impl<T> Vector<T>
where
    T: Copy,
{
    #[inline]
    pub fn new(x: T, y: T) -> Vector<T> {
        Vector { x, y }
    }
}

impl<T> Default for Vector<T>
where
    T: Default + Clone,
{
    #[inline]
    fn default() -> Self {
        Vector {
            x: T::default(),
            y: T::default(),
        }
    }
}

trait AsReal {
    fn as_real(&self) -> f32;
    fn from_real(real: f32) -> Self;
}

impl AsReal for f32 {
    #[inline]
    fn as_real(&self) -> f32 {
        *self
    }
    #[inline]
    fn from_real(real: f32) -> Self {
        real
    }
}

impl AsReal for u8 {
    #[inline]
    fn as_real(&self) -> f32 {
        (*self as f32) * (1. / 255.)
    }
    #[inline]
    #[allow(clippy::manual_clamp)]
    fn from_real(real: f32) -> Self {
        (real * 255.).min(255f32).max(0.) as u8
    }
}

#[derive(Debug, Clone)]
struct Array3D<'a, T> {
    x_dim: usize,
    y_dim: usize,
    z_dim: usize,
    store: UnsafeSlice<'a, Vector<T>>,
}

impl<'a, T> Array3D<'a, T>
where
    T: Default + Clone,
{
    pub fn new(
        slice: &'a mut [Vector<T>],
        width: usize,
        height: usize,
        z: usize,
    ) -> Array3D<'a, T> {
        Array3D {
            x_dim: width,
            y_dim: height,
            z_dim: z,
            store: UnsafeSlice::new(slice),
        }
    }
}

impl<T> Index<usize> for Array3D<'_, T> {
    type Output = Vector<T>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.store.get(index)
    }
}

impl<T> Array3D<'_, T>
where
    T: Clone,
{
    #[inline]
    pub fn find(&self, x: usize, y: usize, z: usize) -> &Vector<T> {
        self.store.get((x * self.y_dim + y) * self.z_dim + z)
    }
}

impl<T> Array3D<'_, T>
where
    T: Clone + Copy,
{
    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> Vector<T> {
        *self.store.get((x * self.y_dim + y) * self.z_dim + z)
    }

    #[inline]
    pub fn set(&self, x: usize, y: usize, z: usize, val: Vector<T>) {
        unsafe {
            self.store.write((x * self.y_dim + y) * self.z_dim + z, val);
        }
    }
}

impl<T> Array3D<'_, T>
where
    T: AsReal + Clone + Copy,
{
    fn trilinear_interpolation(&self, x: f32, y: f32, z: f32) -> Vector<T> {
        let x_index = f32::clamp(x, 0., (self.x_dim - 1) as f32);
        let xx_index = f32::clamp(x_index + 1., 0., (self.x_dim - 1) as f32) as usize;

        let y_index = f32::clamp(y, 0., (self.y_dim - 1) as f32);
        let yy_index = f32::clamp(y_index + 1., 0., (self.y_dim - 1) as f32) as usize;

        let z_index = f32::clamp(z, 0., (self.z_dim - 1) as f32);
        let zz_index = f32::clamp(z_index + 1., 0., (self.z_dim - 1) as f32) as usize;

        let x_alpha = x - x_index;
        let y_alpha = y - y_index;
        let z_alpha = z - z_index;

        let x_index = x_index as usize;
        let y_index = y_index as usize;
        let z_index = z_index as usize;

        let result = self.get(x_index, y_index, z_index).to_real()
            * (1.0f32 - x_alpha)
            * (1.0f32 - y_alpha)
            * (1.0f32 - z_alpha)
            + x_alpha
                * (1.0f32 - y_alpha)
                * (1.0f32 - z_alpha)
                * self.get(xx_index, y_index, z_index).to_real()
            + (1.0f32 - x_alpha)
                * y_alpha
                * (1.0f32 - z_alpha)
                * self.get(x_index, yy_index, z_index).to_real()
            + x_alpha
                * y_alpha
                * (1.0f32 - z_alpha)
                * self.get(xx_index, yy_index, z_index).to_real()
            + (1.0f32 - x_alpha)
                * (1.0f32 - y_alpha)
                * z_alpha
                * self.get(x_index, y_index, zz_index).to_real()
            + x_alpha
                * (1.0f32 - y_alpha)
                * z_alpha
                * self.get(xx_index, y_index, zz_index).to_real()
            + (1.0f32 - x_alpha)
                * y_alpha
                * z_alpha
                * self.get(x_index, yy_index, zz_index).to_real()
            + x_alpha * y_alpha * z_alpha * self.get(xx_index, yy_index, zz_index).to_real();
        Vector::<T>::from_real(&result)
    }
}

#[allow(clippy::manual_clamp)]
fn fast_bilateral_filter_impl<
    T: Copy
        + Default
        + Clone
        + AsReal
        + Real
        + Add<T>
        + Sub<T>
        + Mul<T>
        + Div<T>
        + AsPrimitive<f32>
        + MulAssign<T>
        + AddAssign<T>
        + Display
        + Debug
        + Send
        + Sync,
>(
    img: &BlurImage<T>,
    dst: &mut [T],
    kernel_size: u32,
    spatial_sigma: f32,
    range_sigma: f32,
) where
    f32: AsPrimitive<T>,
{
    let width = img.width;
    let height = img.height;
    let mut base_max = T::min_value();
    let mut base_min = T::max_value();
    for item in img.data.as_ref().iter() {
        base_min = item.min(base_min);
        base_max = item.max(base_max);
    }

    let base_delta = base_max - base_min;

    let padding_xy = 2.;
    let padding_z = 2.;

    let spatial_sigma_scale = if spatial_sigma > 1. {
        (1. / spatial_sigma * (if spatial_sigma > 1.3 { 1.3 } else { 1. }))
            .min(1. / 1.2f32)
            .max(1. / 5f32)
    } else {
        1.
    };
    let range_sigma_scale = if range_sigma > 1. {
        (1. / range_sigma).min(1. / 1.1f32).max(1. / 5f32)
    } else {
        1.
    };

    let small_width = (((width - 1) as f32 * spatial_sigma_scale) + 1. + 2. * padding_xy) as usize;
    let small_height =
        (((height - 1) as f32 * spatial_sigma_scale) + 1. + 2. * padding_xy) as usize;
    let small_depth = ((base_delta.as_() * range_sigma_scale) + 1. + 2. * padding_z) as usize;

    let mut target = vec![Vector::<T>::default(); small_height * small_depth * small_width];

    let mut data = Array3D::<T>::new(&mut target, small_width, small_height, small_depth);

    let stride = img.row_stride() as usize;
    let img = img.data.as_ref();

    for x in 0..width as usize {
        let small_x = ((x as f32) * spatial_sigma_scale + 0.5f32) + padding_xy;
        for y in 0..height as usize {
            let pixel = unsafe { *img.get_unchecked(y * stride + x) };
            let z = pixel - base_min;

            let small_y = ((y as f32) * spatial_sigma_scale + 0.5f32) + padding_xy;
            let small_z = ((z.as_()) * range_sigma_scale + 0.5f32) + padding_z;

            let mut d = *data.find(small_x as usize, small_y as usize, small_z as usize);
            d.x += pixel;
            d.y += 1.0f32.as_();
            data.set(small_x as usize, small_y as usize, small_z as usize, d)
        }
    }

    let mut target2 = vec![Vector::<T>::default(); small_height * small_depth * small_width];

    let mut buffer = Array3D::<T>::new(&mut target2, small_width, small_height, small_depth);

    let preferred_sigma = (spatial_sigma * spatial_sigma + range_sigma * range_sigma).sqrt();

    let gaussian_kernel = gaussian_kernel_1d(kernel_size, preferred_sigma);
    let half_kernel = gaussian_kernel.len() / 2;

    // Unrolled 3D convolution

    (0..small_depth).into_par_iter().for_each(|z| {
        for y in 0..small_height {
            for x in 0..half_kernel.min(small_width) {
                let mut sum = data
                    .get(x, y, z)
                    .mul(unsafe { *gaussian_kernel.get_unchecked(half_kernel) }.as_());
                for i in 0..half_kernel {
                    let weight = unsafe { *gaussian_kernel.get_unchecked(i) }.as_();
                    let start_p = data.get(
                        (x + i).saturating_sub(half_kernel).min(small_width - 1),
                        y,
                        z,
                    );
                    let rolled = half_kernel - i - 1;
                    let end_p = data.get((x + rolled).min(small_width - 1), y, z);
                    sum += (start_p + end_p) * weight;
                }
                buffer.set(x, y, z, sum);
            }

            for x in half_kernel..small_width.saturating_sub(half_kernel) {
                let mut sum = data
                    .get(x, y, z)
                    .mul(unsafe { *gaussian_kernel.get_unchecked(half_kernel) }.as_());
                for i in 0..half_kernel {
                    let weight = unsafe { *gaussian_kernel.get_unchecked(i) }.as_();
                    let start_px = data.get((x + i).sub(half_kernel), y, z);
                    let rolled = half_kernel - i - 1;
                    let end_px = data.get(x + rolled, y, z);
                    sum += (start_px + end_px) * weight;
                }
                buffer.set(x, y, z, sum);
            }

            for x in small_width.saturating_sub(half_kernel)..small_width {
                let mut sum = data
                    .get(x, y, z)
                    .mul(unsafe { *gaussian_kernel.get_unchecked(half_kernel) }.as_());
                for i in 0..half_kernel {
                    let weight = unsafe { *gaussian_kernel.get_unchecked(i) }.as_();
                    let start_p = data.get(
                        (x + i).saturating_sub(half_kernel).min(small_width - 1),
                        y,
                        z,
                    );
                    let rolled = half_kernel - i - 1;
                    let end_p = data.get((x + rolled).min(small_width - 1), y, z);
                    sum += (start_p + end_p) * weight;
                }
                buffer.set(x, y, z, sum);
            }
        }
    });

    std::mem::swap(&mut buffer, &mut data);

    (0..small_height).into_par_iter().for_each(|y| {
        for x in 0..small_width {
            for z in 0..half_kernel.min(small_depth) {
                let mut sum = data
                    .get(x, y, z)
                    .mul(unsafe { *gaussian_kernel.get_unchecked(half_kernel) }.as_());
                for i in 0..half_kernel {
                    let weight = unsafe { *gaussian_kernel.get_unchecked(i) };
                    let rolled = half_kernel - i - 1;
                    let start_px = data.get(
                        x,
                        y,
                        (z + i).saturating_sub(half_kernel).min(small_depth - 1),
                    );
                    let end_px = data.get(x, y, (z + rolled).min(small_depth - 1));
                    sum += (start_px + end_px) * weight.as_();
                }
                buffer.set(x, y, z, sum);
            }

            for z in half_kernel..small_depth.saturating_sub(half_kernel) {
                let mut sum = data
                    .get(x, y, z)
                    .mul(unsafe { *gaussian_kernel.get_unchecked(half_kernel) }.as_());
                for i in 0..half_kernel {
                    let weight = unsafe { *gaussian_kernel.get_unchecked(i) };
                    let start_px = data.get(x, y, (z + i).sub(half_kernel));
                    let rolled = half_kernel - i - 1;
                    let end_px = data.get(x, y, z + rolled);
                    sum += (start_px + end_px) * weight.as_();
                }
                buffer.set(x, y, z, sum);
            }

            for z in small_depth.saturating_sub(small_depth)..small_depth {
                let mut sum = data
                    .get(x, y, z)
                    .mul(unsafe { *gaussian_kernel.get_unchecked(half_kernel) }.as_());
                for i in 0..half_kernel {
                    let weight = unsafe { *gaussian_kernel.get_unchecked(i) };
                    let rolled = half_kernel - i - 1;
                    let start_px = data.get(
                        x,
                        y,
                        (z + i).saturating_sub(half_kernel).min(small_depth - 1),
                    );
                    let end_px = data.get(x, y, (z + rolled).min(small_depth - 1));
                    sum += (start_px + end_px) * weight.as_();
                }
                buffer.set(x, y, z, sum);
            }
        }
    });

    std::mem::swap(&mut buffer, &mut data);

    (0..small_depth).into_par_iter().for_each(|z| {
        for x in 0..small_width {
            for y in 0..half_kernel.min(small_height) {
                let mut sum = data
                    .get(x, y, z)
                    .mul(unsafe { *gaussian_kernel.get_unchecked(half_kernel) }.as_());
                for i in 0..half_kernel {
                    let weight = unsafe { *gaussian_kernel.get_unchecked(i) };
                    let start_px = data.get(
                        x,
                        (y + i).saturating_sub(half_kernel).min(small_height - 1),
                        z,
                    );
                    let rolled = half_kernel - i - 1;
                    let end_px = data.get(x, (y + rolled).min(small_height - 1), z);
                    sum += (start_px + end_px) * weight.as_();
                }
                buffer.set(x, y, z, sum);
            }

            for y in half_kernel..small_height.saturating_sub(half_kernel) {
                let mut sum = data
                    .get(x, y, z)
                    .mul(unsafe { *gaussian_kernel.get_unchecked(half_kernel) }.as_());
                for i in 0..half_kernel {
                    let weight = unsafe { *gaussian_kernel.get_unchecked(i) };
                    let start_px = data.get(x, (y + i).sub(half_kernel), z);
                    let rolled = half_kernel - i - 1;
                    let end_px = data.get(x, y + rolled, z);
                    sum += (start_px + end_px) * weight.as_();
                }
                buffer.set(x, y, z, sum);
            }

            for y in small_height.saturating_sub(half_kernel)..small_height {
                let mut sum = data
                    .get(x, y, z)
                    .mul(unsafe { *gaussian_kernel.get_unchecked(half_kernel) }.as_());
                for i in 0..half_kernel {
                    let weight = unsafe { *gaussian_kernel.get_unchecked(i) };
                    let start_px = data.get(
                        x,
                        (y + i).saturating_sub(half_kernel).min(small_height - 1),
                        z,
                    );
                    let rolled = half_kernel - i - 1;
                    let end_px = data.get(x, (y + rolled).min(small_height - 1), z);
                    sum += (start_px + end_px) * weight.as_();
                }
                buffer.set(x, y, z, sum);
            }
        }
    });

    std::mem::swap(&mut buffer, &mut data);

    for (src, dst) in img.iter().zip(dst.iter_mut()) {
        *dst = *src;
    }

    dst.par_chunks_exact_mut(width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, t) in row.iter_mut().enumerate() {
                let z = (*t - base_min).as_();
                let d = data.trilinear_interpolation(
                    (x as f32) * spatial_sigma_scale + padding_xy,
                    (y as f32) * spatial_sigma_scale + padding_xy,
                    z * range_sigma_scale + padding_z,
                );
                *t = if d.y != 0f32.as_() {
                    d.x / d.y
                } else {
                    0f32.as_()
                };
            }
        })
}

pub trait BilinearWorkingItem<T> {
    fn to_bi_linear_f32(&self) -> f32;
    fn from_bi_linear_f32(value: f32) -> T;
}

impl BilinearWorkingItem<u8> for u8 {
    #[inline]
    #[allow(clippy::manual_clamp)]
    fn from_bi_linear_f32(value: f32) -> u8 {
        (value * 255.).min(255.).max(0.).round() as u8
    }
    #[inline]
    fn to_bi_linear_f32(&self) -> f32 {
        (*self as f32) * (1. / 255.)
    }
}

impl BilinearWorkingItem<u16> for u16 {
    #[inline]
    fn from_bi_linear_f32(value: f32) -> u16 {
        (value * u16::MAX as f32)
            .round()
            .min(u16::MAX as f32)
            .max(0.) as u16
    }
    #[inline]
    fn to_bi_linear_f32(&self) -> f32 {
        (*self as f32) * (1. / u16::MAX as f32)
    }
}

impl BilinearWorkingItem<f32> for f32 {
    #[inline]
    fn from_bi_linear_f32(value: f32) -> f32 {
        value
    }
    #[inline]
    fn to_bi_linear_f32(&self) -> f32 {
        *self
    }
}

fn fast_bilateral_filter_plane_impl<
    V: Copy + Default + 'static + BilinearWorkingItem<V> + Debug + Send + Sync,
>(
    img: &BlurImage<V>,
    dst: &mut BlurImageMut<V>,
    kernel_size: u32,
    spatial_sigma: f32,
    range_sigma: f32,
    pool: &ThreadPool,
) -> Result<(), BlurError> {
    pool.install(|| {
        img.check_layout()?;
        dst.check_layout(Some(img))?;
        img.size_matches_mut(dst)?;
        let width = img.width;
        let height = img.height;
        assert_ne!(kernel_size & 1, 0, "kernel size must be odd");
        assert!(
            !(spatial_sigma <= 0. || range_sigma <= 0.0),
            "Spatial sigma and range sigma must be more than 0"
        );
        let mut chan0 = vec![0f32; width as usize * height as usize];
        for (dst, src) in chan0
            .chunks_exact_mut(width as usize)
            .zip(img.data.chunks_exact(img.row_stride() as usize))
        {
            for (r, &src) in dst.iter_mut().zip(src.iter()) {
                *r = src.to_bi_linear_f32();
            }
        }
        let in_image = BlurImage::borrow(&chan0, width, height, FastBlurChannels::Plane);
        let mut working_dst = BlurImageMut::alloc(width, height, FastBlurChannels::Plane);

        fast_bilateral_filter_impl(
            &in_image,
            working_dst.data.borrow_mut(),
            kernel_size,
            spatial_sigma,
            range_sigma,
        );

        let dst_stride = dst.row_stride() as usize;

        for (dst, src) in dst.data.borrow_mut().chunks_exact_mut(dst_stride).zip(
            working_dst
                .data
                .borrow()
                .chunks_exact(working_dst.row_stride() as usize),
        ) {
            for (r, &src) in dst.iter_mut().zip(src.iter()) {
                *r = V::from_bi_linear_f32(src);
            }
        }
        Ok(())
    })
}

#[cfg(feature = "image")]
pub(crate) fn fast_bilateral_filter_gray_alpha_impl<
    V: Copy + Default + 'static + BilinearWorkingItem<V> + Send + Sync + Debug,
>(
    img: &BlurImage<V>,
    dst: &mut BlurImageMut<V>,
    kernel_size: u32,
    spatial_sigma: f32,
    range_sigma: f32,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    img.check_layout_channels(2)?;
    dst.check_layout_channels(2, Some(img))?;
    assert_ne!(kernel_size & 1, 0, "kernel size must be odd");
    assert!(
        !(spatial_sigma <= 0. || range_sigma <= 0.0),
        "Spatial sigma and range sigma must be more than 0"
    );

    let thread_count = threading_policy.thread_count(img.width, img.height) as u32;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count as usize)
        .build()
        .unwrap();

    pool.install(|| {
        let width = img.width;
        let height = img.height;
        let mut in_image0 = BlurImageMut::alloc(width, height, FastBlurChannels::Plane);
        let mut in_image1 = BlurImageMut::alloc(width, height, FastBlurChannels::Plane);
        for ((dst0, dst1), src) in in_image0
            .data
            .borrow_mut()
            .chunks_exact_mut(width as usize)
            .zip(in_image1.data.borrow_mut().chunks_exact_mut(width as usize))
            .zip(img.data.chunks_exact(img.row_stride() as usize))
        {
            for ((r, g), src) in dst0
                .iter_mut()
                .zip(dst1.iter_mut())
                .zip(src.chunks_exact(2))
            {
                *r = src[0].to_bi_linear_f32();
                *g = src[1].to_bi_linear_f32();
            }
        }

        let mut working_dst0 = BlurImageMut::alloc(width, height, FastBlurChannels::Plane);
        let mut working_dst1 = BlurImageMut::alloc(width, height, FastBlurChannels::Plane);

        let ref0 = in_image0.to_immutable_ref();
        let ref1 = in_image1.to_immutable_ref();

        pool.scope(|s| {
            s.spawn(|_| {
                fast_bilateral_filter_impl(
                    &ref0,
                    working_dst0.data.borrow_mut(),
                    kernel_size,
                    spatial_sigma,
                    range_sigma,
                );
            });
            s.spawn(|_| {
                fast_bilateral_filter_impl(
                    &ref1,
                    working_dst1.data.borrow_mut(),
                    kernel_size,
                    spatial_sigma,
                    range_sigma,
                );
            });
        });

        let dst_stride = dst.row_stride() as usize;

        dst.data
            .borrow_mut()
            .par_chunks_exact_mut(dst_stride)
            .zip(working_dst0.data.borrow().par_chunks_exact(width as usize))
            .zip(working_dst1.data.borrow().par_chunks_exact(width as usize))
            .for_each(|((dst, src0), src1)| {
                for ((dst, src0), src1) in dst.chunks_exact_mut(2).zip(src0.iter()).zip(src1.iter())
                {
                    dst[0] = V::from_bi_linear_f32(*src0);
                    dst[1] = V::from_bi_linear_f32(*src1);
                }
            });
        Ok(())
    })
}

fn fast_bilateral_filter_rgb_impl<
    V: Copy + Default + 'static + BilinearWorkingItem<V> + Send + Sync + Debug,
>(
    img: &BlurImage<V>,
    dst: &mut BlurImageMut<V>,
    kernel_size: u32,
    spatial_sigma: f32,
    range_sigma: f32,
    pool: &ThreadPool,
) -> Result<(), BlurError> {
    pool.install(|| {
        img.check_layout()?;
        dst.check_layout(None)?;
        img.size_matches_mut(dst)?;
        let width = img.width;
        let height = img.height;
        assert_ne!(kernel_size & 1, 0, "kernel size must be odd");
        assert!(
            !(spatial_sigma <= 0. || range_sigma <= 0.0),
            "Spatial sigma and range sigma must be more than 0"
        );
        let mut in_image0 = BlurImageMut::alloc(width, height, FastBlurChannels::Plane);
        let mut in_image1 = BlurImageMut::alloc(width, height, FastBlurChannels::Plane);
        let mut in_image2 = BlurImageMut::alloc(width, height, FastBlurChannels::Plane);
        for (((dst0, dst1), dst2), src) in in_image0
            .data
            .borrow_mut()
            .chunks_exact_mut(width as usize)
            .zip(in_image1.data.borrow_mut().chunks_exact_mut(width as usize))
            .zip(in_image2.data.borrow_mut().chunks_exact_mut(width as usize))
            .zip(img.data.chunks_exact(img.row_stride() as usize))
        {
            for (((r, g), b), src) in dst0
                .iter_mut()
                .zip(dst1.iter_mut())
                .zip(dst2.iter_mut())
                .zip(src.chunks_exact(3))
            {
                *r = src[0].to_bi_linear_f32();
                *g = src[1].to_bi_linear_f32();
                *b = src[2].to_bi_linear_f32();
            }
        }

        let mut working_dst0 = BlurImageMut::alloc(width, height, FastBlurChannels::Plane);
        let mut working_dst1 = BlurImageMut::alloc(width, height, FastBlurChannels::Plane);
        let mut working_dst2 = BlurImageMut::alloc(width, height, FastBlurChannels::Plane);

        let ref0 = in_image0.to_immutable_ref();
        let ref1 = in_image1.to_immutable_ref();
        let ref2 = in_image2.to_immutable_ref();

        pool.scope(|s| {
            s.spawn(|_| {
                fast_bilateral_filter_impl(
                    &ref0,
                    working_dst0.data.borrow_mut(),
                    kernel_size,
                    spatial_sigma,
                    range_sigma,
                );
            });
            s.spawn(|_| {
                fast_bilateral_filter_impl(
                    &ref1,
                    working_dst1.data.borrow_mut(),
                    kernel_size,
                    spatial_sigma,
                    range_sigma,
                );
            });
            s.spawn(|_| {
                fast_bilateral_filter_impl(
                    &ref2,
                    working_dst2.data.borrow_mut(),
                    kernel_size,
                    spatial_sigma,
                    range_sigma,
                );
            });
        });

        let dst_stride = dst.row_stride() as usize;

        dst.data
            .borrow_mut()
            .par_chunks_exact_mut(dst_stride)
            .zip(working_dst0.data.borrow().par_chunks_exact(width as usize))
            .zip(working_dst1.data.borrow().par_chunks_exact(width as usize))
            .zip(working_dst2.data.borrow().par_chunks_exact(width as usize))
            .for_each(|(((dst, src0), src1), src2)| {
                for (((dst, src0), src1), src2) in dst
                    .chunks_exact_mut(3)
                    .zip(src0.iter())
                    .zip(src1.iter())
                    .zip(src2)
                {
                    dst[0] = V::from_bi_linear_f32(*src0);
                    dst[1] = V::from_bi_linear_f32(*src1);
                    dst[2] = V::from_bi_linear_f32(*src2);
                }
            });

        Ok(())
    })
}

fn fast_bilateral_filter_rgba_impl<
    V: Copy + Default + 'static + BilinearWorkingItem<V> + Sync + Send + Debug,
>(
    img: &BlurImage<V>,
    dst: &mut BlurImageMut<V>,
    kernel_size: u32,
    spatial_sigma: f32,
    range_sigma: f32,
    pool: &ThreadPool,
) -> Result<(), BlurError> {
    pool.install(|| {
        img.check_layout()?;
        dst.check_layout(Some(img))?;
        img.size_matches_mut(dst)?;
        let width = img.width;
        let height = img.height;
        assert_ne!(kernel_size & 1, 0, "kernel size must be odd");
        assert!(
            !(spatial_sigma <= 0. || range_sigma <= 0.0),
            "Spatial sigma and range sigma must be more than 0"
        );
        let mut in_image0 = BlurImageMut::alloc(width, height, FastBlurChannels::Plane);
        let mut in_image1 = BlurImageMut::alloc(width, height, FastBlurChannels::Plane);
        let mut in_image2 = BlurImageMut::alloc(width, height, FastBlurChannels::Plane);
        let mut in_image3 = BlurImageMut::alloc(width, height, FastBlurChannels::Plane);
        for ((((dst0, dst1), dst2), dst3), src) in in_image0
            .data
            .borrow_mut()
            .chunks_exact_mut(width as usize)
            .zip(in_image1.data.borrow_mut().chunks_exact_mut(width as usize))
            .zip(in_image2.data.borrow_mut().chunks_exact_mut(width as usize))
            .zip(in_image3.data.borrow_mut().chunks_exact_mut(width as usize))
            .zip(img.data.chunks_exact(img.row_stride() as usize))
        {
            for ((((r, g), b), a), src) in dst0
                .iter_mut()
                .zip(dst1.iter_mut())
                .zip(dst2.iter_mut())
                .zip(dst3.iter_mut())
                .zip(src.chunks_exact(4))
            {
                *r = src[0].to_bi_linear_f32();
                *g = src[1].to_bi_linear_f32();
                *b = src[2].to_bi_linear_f32();
                *a = src[3].to_bi_linear_f32();
            }
        }

        let mut working_dst0 = BlurImageMut::alloc(width, height, FastBlurChannels::Plane);
        let mut working_dst1 = BlurImageMut::alloc(width, height, FastBlurChannels::Plane);
        let mut working_dst2 = BlurImageMut::alloc(width, height, FastBlurChannels::Plane);
        let mut working_dst3 = BlurImageMut::alloc(width, height, FastBlurChannels::Plane);

        let ref0 = in_image0.to_immutable_ref();
        let ref1 = in_image1.to_immutable_ref();
        let ref2 = in_image2.to_immutable_ref();
        let ref3 = in_image3.to_immutable_ref();

        pool.scope(|s| {
            s.spawn(|_| {
                fast_bilateral_filter_impl(
                    &ref0,
                    working_dst0.data.borrow_mut(),
                    kernel_size,
                    spatial_sigma,
                    range_sigma,
                );
            });

            s.spawn(|_| {
                fast_bilateral_filter_impl(
                    &ref1,
                    working_dst1.data.borrow_mut(),
                    kernel_size,
                    spatial_sigma,
                    range_sigma,
                );
            });

            s.spawn(|_| {
                fast_bilateral_filter_impl(
                    &ref2,
                    working_dst2.data.borrow_mut(),
                    kernel_size,
                    spatial_sigma,
                    range_sigma,
                );
            });

            s.spawn(|_| {
                fast_bilateral_filter_impl(
                    &ref3,
                    working_dst3.data.borrow_mut(),
                    kernel_size,
                    spatial_sigma,
                    range_sigma,
                );
            })
        });

        let dst_stride = dst.row_stride() as usize;

        dst.data
            .borrow_mut()
            .par_chunks_exact_mut(dst_stride)
            .zip(working_dst0.data.borrow().par_chunks_exact(width as usize))
            .zip(working_dst1.data.borrow().par_chunks_exact(width as usize))
            .zip(working_dst2.data.borrow().par_chunks_exact(width as usize))
            .zip(working_dst3.data.borrow().par_chunks_exact(width as usize))
            .for_each(|((((dst, src0), src1), src2), src3)| {
                for ((((dst, src0), src1), src2), src3) in dst
                    .chunks_exact_mut(4)
                    .zip(src0.iter())
                    .zip(src1.iter())
                    .zip(src2.iter())
                    .zip(src3.iter())
                {
                    dst[0] = V::from_bi_linear_f32(*src0);
                    dst[1] = V::from_bi_linear_f32(*src1);
                    dst[2] = V::from_bi_linear_f32(*src2);
                    dst[3] = V::from_bi_linear_f32(*src3);
                }
            });

        Ok(())
    })
}

/// Performs fast bilateral filter on the 8-bit image
///
/// This is fast bilateral approximation, note this behaviour significantly differs from OpenCV.
/// This method has high convergence and will completely blur an image very fast with increasing spatial sigma
/// By the nature of this filter the more spatial sigma are the faster method is.
///
/// # Arguments
///
/// * `src`: Source image, see [BlurImage] for more info
/// * `dst`: Destination image, see [BlurImageMut] for more info
/// * `kernel_size`: Convolution kernel size, must be odd
/// * `spatial_sigma`: Spatial sigma
/// * `range_sigma`: Range sigma
///
pub fn fast_bilateral_filter(
    src: &BlurImage<u8>,
    dst: &mut BlurImageMut<u8>,
    kernel_size: u32,
    spatial_sigma: f32,
    range_sigma: f32,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    src.check_layout()?;
    dst.check_layout(Some(src))?;
    src.size_matches_mut(dst)?;
    let channels = src.channels;
    assert_ne!(kernel_size & 1, 0, "kernel_size must be odd");

    let thread_count = threading_policy.thread_count(src.width, src.height) as u32;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count as usize)
        .build()
        .unwrap();

    match channels {
        FastBlurChannels::Plane => {
            fast_bilateral_filter_plane_impl(
                src,
                dst,
                kernel_size,
                spatial_sigma,
                range_sigma,
                &pool,
            )?;
        }
        FastBlurChannels::Channels3 => {
            fast_bilateral_filter_rgb_impl(
                src,
                dst,
                kernel_size,
                spatial_sigma,
                range_sigma,
                &pool,
            )?;
        }
        FastBlurChannels::Channels4 => {
            fast_bilateral_filter_rgba_impl(
                src,
                dst,
                kernel_size,
                spatial_sigma,
                range_sigma,
                &pool,
            )?;
        }
    }
    Ok(())
}

/// Performs fast bilateral filter on the up to 16-bit image
///
/// This is fast bilateral approximation, note this behaviour significantly differs from OpenCV.
/// This method has high convergence and will completely blur an image very fast with increasing spatial sigma
/// By the nature of this filter the more spatial sigma are the faster method is.
///
/// # Arguments
///
/// * `img`: Source image
/// * `dst`: Destination image
/// * `width`: Width of the image
/// * `height`: Height of the image
/// * `kernel_size`: Convolution kernel size, must be odd
/// * `spatial_sigma`: Spatial sigma
/// * `range_sigma`: Range sigma
/// * `channels`: See [FastBlurChannels] for more info
///
pub fn fast_bilateral_filter_u16(
    src: &BlurImage<u16>,
    dst: &mut BlurImageMut<u16>,
    kernel_size: u32,
    spatial_sigma: f32,
    range_sigma: f32,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    src.check_layout()?;
    dst.check_layout(Some(src))?;
    src.size_matches_mut(dst)?;
    let channels = src.channels;
    assert_ne!(kernel_size & 1, 0, "kernel_size must be odd");

    let thread_count = threading_policy.thread_count(src.width, src.height) as u32;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count as usize)
        .build()
        .unwrap();

    match channels {
        FastBlurChannels::Plane => {
            fast_bilateral_filter_plane_impl(
                src,
                dst,
                kernel_size,
                spatial_sigma,
                range_sigma,
                &pool,
            )?;
        }
        FastBlurChannels::Channels3 => {
            fast_bilateral_filter_rgb_impl(
                src,
                dst,
                kernel_size,
                spatial_sigma,
                range_sigma,
                &pool,
            )?;
        }
        FastBlurChannels::Channels4 => {
            fast_bilateral_filter_rgba_impl(
                src,
                dst,
                kernel_size,
                spatial_sigma,
                range_sigma,
                &pool,
            )?;
        }
    }
    Ok(())
}

/// Performs fast bilateral filter on the f32 image
///
/// This is fast bilateral approximation, note this behaviour significantly differs from OpenCV.
/// This method has high convergence and will completely blur an image very fast with increasing spatial sigma
/// By the nature of this filter the more spatial sigma are the faster method is.
///
/// # Arguments
///
/// * `img`: Source image
/// * `dst`: Destination image
/// * `width`: Width of the image
/// * `height`: Height of the image
/// * `kernel_size`: Convolution kernel size, must be odd
/// * `spatial_sigma`: Spatial sigma
/// * `range_sigma`: Range sigma
/// * `channels`: See [FastBlurChannels] for more info
///
pub fn fast_bilateral_filter_f32(
    src: &BlurImage<f32>,
    dst: &mut BlurImageMut<f32>,
    kernel_size: u32,
    spatial_sigma: f32,
    range_sigma: f32,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    src.check_layout()?;
    dst.check_layout(Some(src))?;
    src.size_matches_mut(dst)?;
    let channels = src.channels;

    let thread_count = threading_policy.thread_count(src.width, src.height) as u32;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count as usize)
        .build()
        .unwrap();

    assert_ne!(kernel_size & 1, 0, "kernel_size must be odd");
    match channels {
        FastBlurChannels::Plane => {
            fast_bilateral_filter_plane_impl(
                src,
                dst,
                kernel_size,
                spatial_sigma,
                range_sigma,
                &pool,
            )?;
        }
        FastBlurChannels::Channels3 => {
            fast_bilateral_filter_rgb_impl(
                src,
                dst,
                kernel_size,
                spatial_sigma,
                range_sigma,
                &pool,
            )?;
        }
        FastBlurChannels::Channels4 => {
            fast_bilateral_filter_rgba_impl(
                src,
                dst,
                kernel_size,
                spatial_sigma,
                range_sigma,
                &pool,
            )?;
        }
    }
    Ok(())
}
