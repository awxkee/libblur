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
use crate::gaussian::get_gaussian_kernel_1d;
use crate::FastBlurChannels;
use num_traits::real::Real;
use num_traits::AsPrimitive;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::ParallelSliceMut;
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
struct Array3D<T> {
    x_dim: usize,
    y_dim: usize,
    z_dim: usize,
    store: Vec<Vector<T>>,
}

impl<T> Array3D<T>
where
    T: Default + Clone,
{
    pub fn new(width: usize, height: usize, z: usize) -> Array3D<T> {
        Array3D {
            x_dim: width,
            y_dim: height,
            z_dim: z,
            store: vec![Vector::<T>::default(); width * height * z],
        }
    }
}

impl<T> Index<usize> for Array3D<T> {
    type Output = Vector<T>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        unsafe { self.store.get_unchecked(index) }
    }
}

impl<T> Array3D<T>
where
    T: Clone,
{
    #[inline]
    pub fn find(&self, x: usize, y: usize, z: usize) -> &Vector<T> {
        unsafe {
            self.store
                .get_unchecked((x * self.y_dim + y) * self.z_dim + z)
        }
    }

    #[inline]
    pub fn find_mut(&mut self, x: usize, y: usize, z: usize) -> &mut Vector<T> {
        unsafe {
            self.store
                .get_unchecked_mut((x * self.y_dim + y) * self.z_dim + z)
        }
    }
}

impl<T> Array3D<T>
where
    T: Clone + Copy,
{
    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> Vector<T> {
        unsafe {
            *self
                .store
                .get_unchecked((x * self.y_dim + y) * self.z_dim + z)
        }
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, val: Vector<T>) {
        unsafe {
            *self
                .store
                .get_unchecked_mut((x * self.y_dim + y) * self.z_dim + z) = val;
        }
    }
}

impl<T> Array3D<T>
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

fn get_gaussian_kernel_size(sigma: f32) -> usize {
    2 * (3.0 * sigma).ceil() as usize + 1
}

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
        + Debug,
>(
    img: &[T],
    dst: &mut [T],
    width: u32,
    height: u32,
    spatial_sigma: f32,
    range_sigma: f32,
) where
    f32: AsPrimitive<T>,
{
    let mut base_max = T::min_value();
    let mut base_min = T::max_value();
    for item in img.iter() {
        base_min = item.min(base_min);
        base_max = item.max(base_max);
    }

    let base_delta = base_max - base_min;

    let padding_xy = 2.;
    let padding_z = 2.;

    let spatial_sigma_scale = if spatial_sigma > 1. {
        (1. / spatial_sigma * (if spatial_sigma > 1.3 { 1.3 } else { 1. }))
            .min(1. / 1.2f32)
            .max(1. / 15f32)
    } else {
        1.
    };
    let range_sigma_scale = if range_sigma > 1. {
        (1. / range_sigma).min(1. / 1.1f32).max(1. / 15f32)
    } else {
        1.
    };

    let small_width = (((width - 1) as f32 * spatial_sigma_scale) + 1. + 2. * padding_xy) as usize;
    let small_height =
        (((height - 1) as f32 * spatial_sigma_scale) + 1. + 2. * padding_xy) as usize;
    let small_depth = ((base_delta.as_() * range_sigma_scale) + 1. + 2. * padding_z) as usize;

    let mut data = Array3D::<T>::new(small_width, small_height, small_depth);

    for x in 0..width as usize {
        let small_x = ((x as f32) * spatial_sigma_scale + 0.5f32) + padding_xy;
        for y in 0..height as usize {
            let pixel = unsafe { *img.get_unchecked(y * width as usize + x) };
            let z = pixel - base_min;

            let small_y = ((y as f32) * spatial_sigma_scale + 0.5f32) + padding_xy;
            let small_z = ((z.as_()) * range_sigma_scale + 0.5f32) + padding_z;

            let mut d = *data.find(small_x as usize, small_y as usize, small_z as usize);
            d.x += pixel;
            d.y += 1.0f32.as_();
            *data.find_mut(small_x as usize, small_y as usize, small_z as usize) = d;
        }
    }

    let mut buffer = Array3D::<T>::new(small_width, small_height, small_depth);

    let preferred_sigma = (spatial_sigma * spatial_sigma + range_sigma * range_sigma).sqrt();

    let vl = get_gaussian_kernel_size(preferred_sigma).max(3);

    let gaussian_kernel = get_gaussian_kernel_1d(vl as u32, preferred_sigma);
    let half_kernel = gaussian_kernel.len() / 2;
    let kernel_size = gaussian_kernel.len();

    // Unrolled 3D convolution

    for z in 0..small_depth {
        for y in 0..small_height {
            for x in 0..half_kernel.min(small_width) {
                let mut sum = Vector::default();
                for i in 0..kernel_size {
                    let weight = unsafe { *gaussian_kernel.get_unchecked(i) }.as_();
                    sum += data.get(
                        (x + i).saturating_sub(half_kernel).min(small_width - 1),
                        y,
                        z,
                    ) * weight;
                }
                buffer.set(x, y, z, sum);
            }

            for x in half_kernel..small_width.saturating_sub(half_kernel) {
                let mut sum = Vector::default();
                for i in 0..kernel_size {
                    let weight = unsafe { *gaussian_kernel.get_unchecked(i) }.as_();
                    sum += data.get((x + i).sub(half_kernel), y, z) * weight;
                }
                buffer.set(x, y, z, sum);
            }

            for x in small_width.saturating_sub(half_kernel)..small_width {
                let mut sum = Vector::default();
                for i in 0..kernel_size {
                    let weight = unsafe { *gaussian_kernel.get_unchecked(i) }.as_();
                    sum += data.get(
                        (x + i).saturating_sub(half_kernel).min(small_width - 1),
                        y,
                        z,
                    ) * weight;
                }
                buffer.set(x, y, z, sum);
            }
        }
    }

    std::mem::swap(&mut buffer, &mut data);

    for y in 0..small_height {
        for x in 0..small_width {
            for z in 0..half_kernel.min(small_depth) {
                let mut sum = Vector::default();
                for i in 0..kernel_size {
                    let weight = unsafe { *gaussian_kernel.get_unchecked(i) };
                    sum += data.get(
                        x,
                        y,
                        (z + i).saturating_sub(half_kernel).min(small_depth - 1),
                    ) * weight.as_();
                }
                buffer.set(x, y, z, sum);
            }

            for z in half_kernel..small_depth.saturating_sub(half_kernel) {
                let mut sum = Vector::default();
                for i in 0..kernel_size {
                    let weight = unsafe { *gaussian_kernel.get_unchecked(i) };
                    sum += data.get(x, y, (z + i).sub(half_kernel)) * weight.as_();
                }
                buffer.set(x, y, z, sum);
            }

            for z in small_depth.saturating_sub(small_depth)..small_depth {
                let mut sum = Vector::default();
                for i in 0..kernel_size {
                    let weight = unsafe { *gaussian_kernel.get_unchecked(i) };
                    sum += data.get(
                        x,
                        y,
                        (z + i).saturating_sub(half_kernel).min(small_depth - 1),
                    ) * weight.as_();
                }
                buffer.set(x, y, z, sum);
            }
        }
    }

    std::mem::swap(&mut buffer, &mut data);

    for z in 0..small_depth {
        for x in 0..small_width {
            for y in 0..half_kernel.min(small_height) {
                let mut sum = Vector::default();
                for i in 0..kernel_size {
                    let weight = unsafe { *gaussian_kernel.get_unchecked(i) };
                    sum += data.get(
                        x,
                        (y + i).saturating_sub(half_kernel).min(small_height - 1),
                        z,
                    ) * weight.as_();
                }
                buffer.set(x, y, z, sum);
            }

            for y in half_kernel..small_height.saturating_sub(half_kernel) {
                let mut sum = Vector::default();
                for i in 0..kernel_size {
                    let weight = unsafe { *gaussian_kernel.get_unchecked(i) };
                    sum += data.get(x, (y + i).sub(half_kernel), z) * weight.as_();
                }
                buffer.set(x, y, z, sum);
            }

            for y in small_height.saturating_sub(half_kernel)..small_height {
                let mut sum = Vector::default();
                for i in 0..kernel_size {
                    let weight = unsafe { *gaussian_kernel.get_unchecked(i) };
                    sum += data.get(
                        x,
                        (y + i).saturating_sub(half_kernel).min(small_height - 1),
                        z,
                    ) * weight.as_();
                }
                buffer.set(x, y, z, sum);
            }
        }
    }

    std::mem::swap(&mut buffer, &mut data);

    for (src, dst) in img.iter().zip(dst.iter_mut()) {
        *dst = *src;
    }

    for x in 0..width as usize {
        for y in 0..height as usize {
            let z = (*unsafe { dst.get_unchecked(y * width as usize + x) } - base_min).as_();
            let d = data.trilinear_interpolation(
                (x as f32) * spatial_sigma_scale + padding_xy,
                (y as f32) * spatial_sigma_scale + padding_xy,
                z * range_sigma_scale + padding_z,
            );
            unsafe {
                *dst.get_unchecked_mut(y * width as usize + x) = if d.y != 0f32.as_() {
                    d.x / d.y
                } else {
                    0f32.as_()
                };
            }
        }
    }
}

pub trait BilinearWorkingItem<T> {
    fn to_bi_linear_f32(&self) -> f32;
    fn from_bi_linear_f32(value: f32) -> T;
}

impl BilinearWorkingItem<u8> for u8 {
    #[inline]
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
            .min(u16::MAX as f32)
            .max(0.)
            .round() as u16
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

fn fast_bilateral_filter_plane_impl<V: Copy + Default + 'static + BilinearWorkingItem<V>>(
    img: &[V],
    dst: &mut [V],
    width: u32,
    height: u32,
    spatial_sigma: f32,
    range_sigma: f32,
) {
    if spatial_sigma <= 0. || range_sigma <= 0.0 {
        panic!("Spatial sigma and range sigma must be more than 0");
    }
    let mut chan0 = vec![0f32; width as usize * height as usize];
    for (r, &src) in chan0.iter_mut().zip(img.iter()) {
        *r = src.to_bi_linear_f32();
    }
    let mut dst_chan0 = vec![0f32; width as usize * height as usize];

    fast_bilateral_filter_impl(
        &chan0,
        &mut dst_chan0,
        width,
        height,
        spatial_sigma,
        range_sigma,
    );

    for (&r, v_dst) in dst_chan0.iter().zip(dst.iter_mut()) {
        *v_dst = V::from_bi_linear_f32(r);
    }
}

#[cfg(feature = "image")]
pub(crate) fn fast_bilateral_filter_gray_alpha_impl<
    V: Copy + Default + 'static + BilinearWorkingItem<V> + Send + Sync,
>(
    img: &[V],
    dst: &mut [V],
    width: u32,
    height: u32,
    spatial_sigma: f32,
    range_sigma: f32,
) {
    if spatial_sigma <= 0. || range_sigma <= 0.0 {
        panic!("Spatial sigma and range sigma must be more than 0");
    }
    let mut chan0 = vec![0f32; width as usize * height as usize];
    let mut chan1 = vec![0f32; width as usize * height as usize];
    for ((r, g), src) in chan0
        .iter_mut()
        .zip(chan1.iter_mut())
        .zip(img.chunks_exact(2))
    {
        *r = src[0].to_bi_linear_f32();
        *g = src[1].to_bi_linear_f32();
    }
    let mut dst_chan0 = vec![0f32; width as usize * height as usize];
    let mut dst_chan1 = vec![0f32; width as usize * height as usize];

    rayon::scope(|s| {
        s.spawn(|_| {
            fast_bilateral_filter_impl(
                &chan0,
                &mut dst_chan0,
                width,
                height,
                spatial_sigma,
                range_sigma,
            );
        });
        s.spawn(|_| {
            fast_bilateral_filter_impl(
                &chan1,
                &mut dst_chan1,
                width,
                height,
                spatial_sigma,
                range_sigma,
            );
        });
    });

    dst.par_chunks_exact_mut(2)
        .enumerate()
        .for_each(|(i, v_dst)| unsafe {
            v_dst[0] = V::from_bi_linear_f32(*dst_chan0.get_unchecked(i));
            v_dst[1] = V::from_bi_linear_f32(*dst_chan1.get_unchecked(i));
        });
}

fn fast_bilateral_filter_rgb_impl<
    V: Copy + Default + 'static + BilinearWorkingItem<V> + Send + Sync,
>(
    img: &[V],
    dst: &mut [V],
    width: u32,
    height: u32,
    spatial_sigma: f32,
    range_sigma: f32,
) {
    if spatial_sigma <= 0. || range_sigma <= 0.0 {
        panic!("Spatial sigma and range sigma must be more than 0");
    }
    let mut chan0 = vec![0f32; width as usize * height as usize];
    let mut chan1 = vec![0f32; width as usize * height as usize];
    let mut chan2 = vec![0f32; width as usize * height as usize];
    for (((r, g), b), src) in chan0
        .iter_mut()
        .zip(chan1.iter_mut())
        .zip(chan2.iter_mut())
        .zip(img.chunks_exact(3))
    {
        *r = src[0].to_bi_linear_f32();
        *g = src[1].to_bi_linear_f32();
        *b = src[2].to_bi_linear_f32();
    }
    let mut dst_chan0 = vec![0f32; width as usize * height as usize];
    let mut dst_chan1 = vec![0f32; width as usize * height as usize];
    let mut dst_chan2 = vec![0f32; width as usize * height as usize];

    rayon::scope(|s| {
        s.spawn(|_| {
            fast_bilateral_filter_impl(
                &chan0,
                &mut dst_chan0,
                width,
                height,
                spatial_sigma,
                range_sigma,
            );
        });
        s.spawn(|_| {
            fast_bilateral_filter_impl(
                &chan1,
                &mut dst_chan1,
                width,
                height,
                spatial_sigma,
                range_sigma,
            );
        });
        s.spawn(|_| {
            fast_bilateral_filter_impl(
                &chan2,
                &mut dst_chan2,
                width,
                height,
                spatial_sigma,
                range_sigma,
            );
        });
    });

    dst.par_chunks_exact_mut(3)
        .enumerate()
        .for_each(|(i, v_dst)| unsafe {
            v_dst[0] = V::from_bi_linear_f32(*dst_chan0.get_unchecked(i));
            v_dst[1] = V::from_bi_linear_f32(*dst_chan1.get_unchecked(i));
            v_dst[2] = V::from_bi_linear_f32(*dst_chan2.get_unchecked(i));
        });
}

fn fast_bilateral_filter_rgba_impl<
    V: Copy + Default + 'static + BilinearWorkingItem<V> + Sync + Send,
>(
    img: &[V],
    dst: &mut [V],
    width: u32,
    height: u32,
    spatial_sigma: f32,
    range_sigma: f32,
) {
    if spatial_sigma <= 0. || range_sigma <= 0.0 {
        panic!("Spatial sigma and range sigma must be more than 0");
    }
    let mut chan0 = vec![0f32; width as usize * height as usize];
    let mut chan1 = vec![0f32; width as usize * height as usize];
    let mut chan2 = vec![0f32; width as usize * height as usize];
    let mut chan3 = vec![0f32; width as usize * height as usize];
    for ((((r, g), b), a), src) in chan0
        .iter_mut()
        .zip(chan1.iter_mut())
        .zip(chan2.iter_mut())
        .zip(chan3.iter_mut())
        .zip(img.chunks_exact(4))
    {
        *r = src[0].to_bi_linear_f32();
        *g = src[1].to_bi_linear_f32();
        *b = src[2].to_bi_linear_f32();
        *a = src[3].to_bi_linear_f32();
    }
    let mut dst_chan0 = vec![0f32; width as usize * height as usize];
    let mut dst_chan1 = vec![0f32; width as usize * height as usize];
    let mut dst_chan2 = vec![0f32; width as usize * height as usize];
    let mut dst_chan3 = vec![0f32; width as usize * height as usize];

    rayon::scope(|s| {
        s.spawn(|_| {
            fast_bilateral_filter_impl(
                &chan0,
                &mut dst_chan0,
                width,
                height,
                spatial_sigma,
                range_sigma,
            );
        });
        s.spawn(|_| {
            fast_bilateral_filter_impl(
                &chan1,
                &mut dst_chan1,
                width,
                height,
                spatial_sigma,
                range_sigma,
            );
        });
        s.spawn(|_| {
            fast_bilateral_filter_impl(
                &chan2,
                &mut dst_chan2,
                width,
                height,
                spatial_sigma,
                range_sigma,
            );
        });
        s.spawn(|_| {
            fast_bilateral_filter_impl(
                &chan3,
                &mut dst_chan3,
                width,
                height,
                spatial_sigma,
                range_sigma,
            );
        });
    });

    dst.par_chunks_exact_mut(4)
        .enumerate()
        .for_each(|(i, v_dst)| unsafe {
            v_dst[0] = V::from_bi_linear_f32(*dst_chan0.get_unchecked(i));
            v_dst[1] = V::from_bi_linear_f32(*dst_chan1.get_unchecked(i));
            v_dst[2] = V::from_bi_linear_f32(*dst_chan2.get_unchecked(i));
            v_dst[3] = V::from_bi_linear_f32(*dst_chan3.get_unchecked(i));
        });
}

/// Performs fast bilateral filter on the 8-bit image
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
/// * `spatial_sigma`: Spatial sigma
/// * `range_sigma`: Range sigma
/// * `channels`: See [FastBlurChannels] for more info
///
pub fn fast_bilateral_filter(
    img: &[u8],
    dst: &mut [u8],
    width: u32,
    height: u32,
    spatial_sigma: f32,
    range_sigma: f32,
    channels: FastBlurChannels,
) {
    match channels {
        FastBlurChannels::Plane => {
            fast_bilateral_filter_plane_impl(img, dst, width, height, spatial_sigma, range_sigma);
        }
        FastBlurChannels::Channels3 => {
            fast_bilateral_filter_rgb_impl(img, dst, width, height, spatial_sigma, range_sigma);
        }
        FastBlurChannels::Channels4 => {
            fast_bilateral_filter_rgba_impl(img, dst, width, height, spatial_sigma, range_sigma);
        }
    }
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
/// * `spatial_sigma`: Spatial sigma
/// * `range_sigma`: Range sigma
/// * `channels`: See [FastBlurChannels] for more info
///
pub fn fast_bilateral_filter_u16(
    img: &[u16],
    dst: &mut [u16],
    width: u32,
    height: u32,
    spatial_sigma: f32,
    range_sigma: f32,
    channels: FastBlurChannels,
) {
    match channels {
        FastBlurChannels::Plane => {
            fast_bilateral_filter_plane_impl(img, dst, width, height, spatial_sigma, range_sigma);
        }
        FastBlurChannels::Channels3 => {
            fast_bilateral_filter_rgb_impl(img, dst, width, height, spatial_sigma, range_sigma);
        }
        FastBlurChannels::Channels4 => {
            fast_bilateral_filter_rgba_impl(img, dst, width, height, spatial_sigma, range_sigma);
        }
    }
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
/// * `spatial_sigma`: Spatial sigma
/// * `range_sigma`: Range sigma
/// * `channels`: See [FastBlurChannels] for more info
///
pub fn fast_bilateral_filter_f32(
    img: &[f32],
    dst: &mut [f32],
    width: u32,
    height: u32,
    spatial_sigma: f32,
    range_sigma: f32,
    channels: FastBlurChannels,
) {
    match channels {
        FastBlurChannels::Plane => {
            fast_bilateral_filter_plane_impl(img, dst, width, height, spatial_sigma, range_sigma);
        }
        FastBlurChannels::Channels3 => {
            fast_bilateral_filter_rgb_impl(img, dst, width, height, spatial_sigma, range_sigma);
        }
        FastBlurChannels::Channels4 => {
            fast_bilateral_filter_rgba_impl(img, dst, width, height, spatial_sigma, range_sigma);
        }
    }
}
