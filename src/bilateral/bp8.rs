/*
 * // Copyright (c) Radzivon Bartoshyk 6/2025. All rights reserved.
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
#![allow(clippy::manual_clamp)]
use crate::{
    make_arena, Arena, ArenaPads, BlurError, BlurImage, BlurImageMut, EdgeMode, FastBlurChannels,
    Scalar, ThreadingPolicy,
};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

/// Pre-compute exp LUTs  ────────────────────────────────────────────────────
///   range LUT  : 256 entries  — exp(-ΔI² / 2σ_r²)
///   spatial LUT: (2r+1)²      — exp(-(dx²+dy²) / 2σ_s²)
pub(crate) struct BilateralStore {
    pub(crate) range: Box<[f32; 65536]>,
    pub(crate) spatial: Box<[f32; 65536]>,
}

#[inline]
fn f32_to_q08(x: f32) -> f32 {
    x
}

impl BilateralStore {
    pub fn new(kernel: usize, sigma_spatial: f32, sigma_range: f32) -> Self {
        let mut range = Box::new([0f32; 65536]);

        let recip_d_spatial = 1.0 / (2.0 * sigma_spatial * sigma_spatial);
        let recip_d_range = 1.0 / (2.0 * sigma_range * sigma_range);

        let pad_r = kernel as i32 / 2;

        for dy in 0..kernel {
            for dx in 0..kernel {
                let zx = (pad_r - (dx as i32)) as f32;
                let zy = (pad_r - (dy as i32)) as f32;
                let dzx = zx * zx;
                let dzy = zy * zy;

                let x = -((dzx + dzy) * recip_d_spatial);
                let k = x.exp();
                range[dy * kernel + dx] = f32_to_q08(k);
            }
        }

        let mut spatial = Box::new([0f32; 65536]);

        for px in 0..256 {
            for intensity in 0..256 {
                let diff = px as f32 / 255.0 - intensity as f32 / 255.0;
                let rz = -((diff * diff) * recip_d_range);
                let d_diff = rz.exp();
                spatial[px as usize * 256 + intensity as usize] = f32_to_q08(d_diff);
            }
        }

        Self { range, spatial }
    }
}

pub(crate) trait BilateralUnit<T> {
    fn execute(&self, a_src: &[T], y: usize, dst_row: &mut [T], src_row: &[T]);
}

struct ExecutionUnit4<'a> {
    arena: Arena,
    params: BilateralBlurParams,
    store: &'a BilateralStore,
    src_width: usize,
}

struct ExecutionUnit3<'a> {
    arena: Arena,
    params: BilateralBlurParams,
    store: &'a BilateralStore,
    src_width: usize,
}

struct ExecutionUnitPlane<'a> {
    arena: Arena,
    params: BilateralBlurParams,
    store: &'a BilateralStore,
    src_width: usize,
}

impl BilateralUnit<u8> for ExecutionUnit4<'_> {
    fn execute(&self, a_src: &[u8], y: usize, dst_row: &mut [u8], src_row: &[u8]) {
        const N: usize = 4;
        let sliced_range = &self.store.range[..self.params.kernel_size * self.params.kernel_size];
        let ss = &self.store.spatial;
        let useful_width = self.src_width * N;
        let a_stride = self.arena.width * self.arena.components;
        let dst_row = &mut dst_row[..useful_width];
        let src_row = &src_row[..useful_width];
        for (x, (dst, center)) in dst_row
            .chunks_exact_mut(N)
            .zip(src_row.chunks_exact(N))
            .enumerate()
        {
            let mut sum0 = 0f32;
            let mut sum1 = 0f32;
            let mut sum2 = 0f32;
            let mut sum3 = 0f32;

            let mut iw0 = 0f32;
            let mut iw1 = 0f32;
            let mut iw2 = 0f32;
            let mut iw3 = 0f32;

            for (ky, ky_row) in sliced_range
                .chunks_exact(self.params.kernel_size)
                .enumerate()
            {
                let c_slice = (y + ky) * a_stride + x * N;
                let c_px_slice = &a_src[c_slice..(c_slice + N * self.params.kernel_size)];
                for (c_px, &rwz) in c_px_slice.chunks_exact(N).zip(ky_row.iter()) {
                    let z0 = rwz * ss[(center[0] as u16 * 256 + c_px[0] as u16) as usize];
                    let z1 = rwz * ss[(center[1] as u16 * 256 + c_px[1] as u16) as usize];
                    let z2 = rwz * ss[(center[2] as u16 * 256 + c_px[2] as u16) as usize];
                    let z3 = rwz * ss[(center[3] as u16 * 256 + c_px[3] as u16) as usize];
                    sum0 += z0 * c_px[0] as f32;
                    sum1 += z1 * c_px[1] as f32;
                    sum2 += z2 * c_px[2] as f32;
                    sum3 += z3 * c_px[3] as f32;
                    iw0 += z0;
                    iw1 += z1;
                    iw2 += z2;
                    iw3 += z3;
                }
            }

            iw0 = if iw0 == 0. { 1. } else { iw0 };
            iw1 = if iw1 == 0. { 1. } else { iw1 };
            iw2 = if iw2 == 0. { 1. } else { iw2 };
            iw3 = if iw3 == 0. { 1. } else { iw3 };

            dst[0] = (sum0 / iw0).round().min(255.).max(0.) as u8;
            dst[1] = (sum1 / iw1).round().min(255.).max(0.) as u8;
            dst[2] = (sum2 / iw2).round().min(255.).max(0.) as u8;
            dst[3] = (sum3 / iw3).round().min(255.).max(0.) as u8;
        }
    }
}

impl BilateralUnit<u8> for ExecutionUnit3<'_> {
    fn execute(&self, a_src: &[u8], y: usize, dst_row: &mut [u8], src_row: &[u8]) {
        const N: usize = 3;
        let sliced_range = &self.store.range[..self.params.kernel_size * self.params.kernel_size];
        let ss = &self.store.spatial;
        let useful_width = self.src_width * N;
        let a_stride = self.arena.width * self.arena.components;
        let dst_row = &mut dst_row[..useful_width];
        let src_row = &src_row[..useful_width];
        for (x, (dst, center)) in dst_row
            .chunks_exact_mut(N)
            .zip(src_row.chunks_exact(N))
            .enumerate()
        {
            let mut sum0 = 0f32;
            let mut sum1 = 0f32;
            let mut sum2 = 0f32;

            let mut iw0 = 0f32;
            let mut iw1 = 0f32;
            let mut iw2 = 0f32;

            for (ky, ky_row) in sliced_range
                .chunks_exact(self.params.kernel_size)
                .enumerate()
            {
                let c_slice = (y + ky) * a_stride + x * N;
                let c_px_slice = &a_src[c_slice..(c_slice + N * self.params.kernel_size)];
                for (c_px, &rwz) in c_px_slice.chunks_exact(N).zip(ky_row.iter()) {
                    let z0 = rwz * ss[(center[0] as u16 * 256 + c_px[0] as u16) as usize];
                    let z1 = rwz * ss[(center[1] as u16 * 256 + c_px[1] as u16) as usize];
                    let z2 = rwz * ss[(center[2] as u16 * 256 + c_px[2] as u16) as usize];
                    sum0 += z0 * c_px[0] as f32;
                    sum1 += z1 * c_px[1] as f32;
                    sum2 += z2 * c_px[2] as f32;
                    iw0 += z0;
                    iw1 += z1;
                    iw2 += z2;
                }
            }

            iw0 = if iw0 == 0. { 1. } else { iw0 };
            iw1 = if iw1 == 0. { 1. } else { iw1 };
            iw2 = if iw2 == 0. { 1. } else { iw2 };

            dst[0] = (sum0 / iw0).round().min(255.).max(0.) as u8;
            dst[1] = (sum1 / iw1).round().min(255.).max(0.) as u8;
            dst[2] = (sum2 / iw2).round().min(255.).max(0.) as u8;
        }
    }
}

impl BilateralUnit<u8> for ExecutionUnitPlane<'_> {
    fn execute(&self, a_src: &[u8], y: usize, dst_row: &mut [u8], src_row: &[u8]) {
        const N: usize = 1;
        let sliced_range = &self.store.range[..self.params.kernel_size * self.params.kernel_size];
        let ss = &self.store.spatial;
        let useful_width = self.src_width * N;
        let a_stride = self.arena.width * self.arena.components;
        let dst_row = &mut dst_row[..useful_width];
        let src_row = &src_row[..useful_width];
        for (x, (dst, center)) in dst_row
            .chunks_exact_mut(N)
            .zip(src_row.chunks_exact(N))
            .enumerate()
        {
            let mut sum0 = 0f32;
            let mut iw0 = 0f32;

            for (ky, ky_row) in sliced_range
                .chunks_exact(self.params.kernel_size)
                .enumerate()
            {
                let c_slice = (y + ky) * a_stride + x * N;
                let c_px_slice = &a_src[c_slice..(c_slice + N * self.params.kernel_size)];
                for (c_px, &rwz) in c_px_slice.chunks_exact(N).zip(ky_row.iter()) {
                    let z0 = rwz * ss[(center[0] as u16 * 256 + c_px[0] as u16) as usize];
                    sum0 += z0 * c_px[0] as f32;
                    iw0 += z0;
                }
            }

            iw0 = if iw0 == 0. { 1. } else { iw0 };

            dst[0] = (sum0 / iw0).round().min(255.).max(0.) as u8;
        }
    }
}

fn bilateral_filter_impl<const N: usize>(
    src: &BlurImage<u8>,
    dst: &mut BlurImageMut<u8>,
    params: BilateralBlurParams,
    edge_mode: EdgeMode,
    constant_border: Scalar,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    params.validate()?;
    src.check_layout()?;
    dst.check_layout(Some(src))?;
    src.size_matches_mut(dst)?;

    let arena = make_arena::<u8, N>(
        src.data.as_ref(),
        src.row_stride() as usize,
        src.size(),
        ArenaPads::constant(params.kernel_size / 2),
        edge_mode,
        constant_border,
    )?;

    let dst_row_stride = dst.row_stride() as usize;

    let store = BilateralStore::new(params.kernel_size, params.spatial_sigma, params.range_sigma);

    let arena_src = arena.0.as_slice();
    let arena_cfg = arena.1;

    let mut _unit: Box<dyn BilateralUnit<u8> + Send + Sync> = match N {
        1 => Box::new(ExecutionUnitPlane {
            arena: arena_cfg,
            params,
            store: &store,
            src_width: src.width as usize,
        }),
        3 => Box::new(ExecutionUnit3 {
            arena: arena_cfg,
            params,
            store: &store,
            src_width: src.width as usize,
        }),
        4 => Box::new(ExecutionUnit4 {
            arena: arena_cfg,
            params,
            store: &store,
            src_width: src.width as usize,
        }),
        _ => unimplemented!("Bilateral blur for {} channels is not implemented", N),
    };
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        use crate::bilateral::neon::BilateralExecutionUnitNeon;
        _unit = Box::new(BilateralExecutionUnitNeon::<N> {
            arena: arena_cfg,
            params,
            store: &store,
            src_width: src.width as usize,
        });
    }

    let thread_count = threading_policy.thread_count(src.width, src.height) as u32;
    if thread_count == 1 {
        dst.data
            .borrow_mut()
            .chunks_exact_mut(dst_row_stride)
            .zip(src.data.as_ref().chunks_exact(src.row_stride() as usize))
            .enumerate()
            .for_each(|(y, (dst, src_row))| _unit.execute(arena_src, y, dst, src_row));
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count as usize)
            .build()
            .unwrap();

        pool.install(|| {
            dst.data
                .borrow_mut()
                .par_chunks_exact_mut(dst_row_stride)
                .zip(
                    src.data
                        .as_ref()
                        .par_chunks_exact(src.row_stride() as usize),
                )
                .enumerate()
                .for_each(|(y, (dst, src_row))| _unit.execute(arena_src, y, dst, src_row));
        });
    }

    Ok(())
}

#[derive(Copy, Clone, Debug)]
pub struct BilateralBlurParams {
    pub kernel_size: usize,
    pub spatial_sigma: f32,
    pub range_sigma: f32,
}

impl BilateralBlurParams {
    fn validate(&self) -> Result<(), BlurError> {
        if self.kernel_size % 2 == 0 {
            return Err(BlurError::OddKernel(self.kernel_size));
        }
        if self.spatial_sigma <= 0.0 {
            return Err(BlurError::NegativeOrZeroSigma);
        }
        if self.range_sigma <= 0.0 {
            return Err(BlurError::NegativeOrZeroSigma);
        }
        Ok(())
    }
}

/// Bilateral filter.
///
/// This is very slow filter.
///
/// # Arguments
///
/// * `src`: Src image.
/// * `dst`: Dst image.
/// * `params`: See [BilateralBlurParams] for more info.
/// * `edge_mode`: Border mode, see [EdgeMode] for more info.
/// * `constant_border`: Scalar value for constant border mode.
/// * `threading_policy`: see [ThreadingPolicy] for more info.
///
/// returns: Result<(), BlurError>
///
/// # Examples
pub fn bilateral_filter(
    src: &BlurImage<u8>,
    dst: &mut BlurImageMut<u8>,
    params: BilateralBlurParams,
    edge_mode: EdgeMode,
    constant_border: Scalar,
    threading_policy: ThreadingPolicy,
) -> Result<(), BlurError> {
    params.validate()?;
    src.check_layout()?;
    dst.check_layout(Some(src))?;
    src.size_matches_mut(dst)?;
    let mut copied_params = params;
    copied_params.kernel_size = copied_params.kernel_size.min(254);
    if copied_params.kernel_size == 1 {
        return src.copy_to_mut(dst);
    }
    match src.channels {
        FastBlurChannels::Plane => bilateral_filter_impl::<1>(
            src,
            dst,
            copied_params,
            edge_mode,
            constant_border,
            threading_policy,
        ),
        FastBlurChannels::Channels3 => bilateral_filter_impl::<3>(
            src,
            dst,
            copied_params,
            edge_mode,
            constant_border,
            threading_policy,
        ),
        FastBlurChannels::Channels4 => bilateral_filter_impl::<4>(
            src,
            dst,
            copied_params,
            edge_mode,
            constant_border,
            threading_policy,
        ),
    }
}
