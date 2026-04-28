/*
 * // Copyright (c) Radzivon Bartoshyk 4/2026. All rights reserved.
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
use crate::edge_mode::{reflect_index, reflect_index_101};
use crate::filter2d::fft_utils::fft_next_good_size_real;
use crate::to_storage::ToStorage;
use crate::unsafe_slice::UnsafeSlice;
use crate::{BlurImageMut, EdgeMode, EdgeMode2D, FftNumber};
use novtb::{ThreadPool, parallel_range_with_context};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};
use std::fmt::Debug;

pub(crate) struct FftConvolve<T> {
    pub(crate) border_constant: T,
    pub(crate) edge_mode: EdgeMode2D,
    pub(crate) thread_count: usize,
    pub(crate) block_rows: usize,
    pub(crate) block_cols: usize,
}

struct TileContext<T> {
    tile: Vec<T>,
    complex_tile: Vec<Complex<T>>,
    scratch: Vec<Complex<T>>,
}

impl<T: Copy + 'static + Zero + FftNumber> FftConvolve<T>
where
    f64: AsPrimitive<T>,
{
    fn fill_tile(
        &self,
        tile: &mut [T],
        tile_cols: usize,
        signal: &[T],
        sig_rows: usize,
        sig_cols: usize,
        src_row_start: isize,
        src_col_start: isize,
    ) {
        tile.chunks_exact_mut(tile_cols)
            .enumerate()
            .for_each(|(tr, row)| {
                let sr = src_row_start + tr as isize;

                // Row completely out of signal bounds — fill entire row via sample
                if sr < 0 || sr >= sig_rows as isize {
                    row.iter_mut().enumerate().for_each(|(tc, pixel)| {
                        let sc = src_col_start + tc as isize;
                        *pixel = self.sample(signal, sig_rows, sig_cols, sr, sc);
                    });
                    return;
                }

                // Row is valid — find the column overlap range
                // [col_start_in_tile, col_end_in_tile) maps to valid signal columns
                let col_start_in_tile = ((-src_col_start).max(0) as usize).min(tile_cols);
                let col_end_in_tile =
                    ((sig_cols as isize - src_col_start).max(0) as usize).min(tile_cols);

                // Left border columns — out of bounds horizontally
                row[..col_start_in_tile]
                    .iter_mut()
                    .enumerate()
                    .for_each(|(tc, pixel)| {
                        let sc = src_col_start + tc as isize;
                        *pixel = self.sample(signal, sig_rows, sig_cols, sr, sc);
                    });

                // Center — bulk memcpy directly from signal row, no bounds check needed
                if col_start_in_tile < col_end_in_tile {
                    let sig_col_start = (src_col_start + col_start_in_tile as isize) as usize;
                    let sig_row_offset = sr as usize * sig_cols + sig_col_start;
                    let copy_len = col_end_in_tile - col_start_in_tile;
                    row[col_start_in_tile..col_end_in_tile]
                        .copy_from_slice(&signal[sig_row_offset..sig_row_offset + copy_len]);
                }

                // Right border columns — out of bounds horizontally
                row[col_end_in_tile..]
                    .iter_mut()
                    .enumerate()
                    .for_each(|(tc, pixel)| {
                        let sc = src_col_start + (col_end_in_tile + tc) as isize;
                        *pixel = self.sample(signal, sig_rows, sig_cols, sr, sc);
                    });
            });
    }

    #[inline]
    fn sample(&self, signal: &[T], sig_rows: usize, sig_cols: usize, r: isize, c: isize) -> T {
        let h = sig_rows as isize;
        let w = sig_cols as isize;

        if r >= 0 && r < h && c >= 0 && c < w {
            return *unsafe { signal.get_unchecked(r as usize * sig_cols + c as usize) };
        }

        let er = match self.edge_mode.vertical {
            EdgeMode::Constant => return self.border_constant,
            EdgeMode::Clamp => r.clamp(0, h - 1),
            EdgeMode::Reflect => reflect_index(r, h) as isize,
            EdgeMode::Reflect101 => reflect_index_101(r, h) as isize,
            EdgeMode::Wrap => r.rem_euclid(h),
        };

        let ec = match self.edge_mode.horizontal {
            EdgeMode::Constant => return self.border_constant,
            EdgeMode::Clamp => c.clamp(0, w - 1),
            EdgeMode::Reflect => reflect_index(c, w) as isize,
            EdgeMode::Reflect101 => reflect_index_101(c, w) as isize,
            EdgeMode::Wrap => c.rem_euclid(w),
        };

        signal[er as usize * sig_cols + ec as usize]
    }

    // /// full-linear convolution
    // pub fn overlap_save_2d(
    //     &self,
    //     signal: &[T],
    //     sig_rows: usize,
    //     sig_cols: usize,
    //     kernel: &[T],
    //     ker_rows: usize,
    //     ker_cols: usize,
    // ) -> Vec<T> {
    //     assert_eq!(signal.len(), sig_rows * sig_cols);
    //     assert_eq!(kernel.len(), ker_rows * ker_cols);
    //
    //     let out_rows = sig_rows + ker_rows - 1;
    //     let out_cols = sig_cols + ker_cols - 1;
    //
    //     let fft_rows = (self.block_rows + ker_rows - 1).next_power_of_two();
    //     let fft_cols = (self.block_cols + ker_cols - 1).next_power_of_two();
    //     let step_rows = fft_rows - (ker_rows - 1);
    //     let step_cols = fft_cols - (ker_cols - 1);
    //
    //     let tile_len = fft_rows * fft_cols;
    //     let complex_tile_len = fft_rows * ((fft_cols / 2) + 1);
    //
    //     let mut ker_spec = vec![Complex::new(T::zero(), T::zero()); complex_tile_len];
    //     let mut ker_arena = vec![T::zero(); fft_rows * fft_cols];
    //     ker_arena
    //         .chunks_exact_mut(fft_cols)
    //         .zip(kernel.chunks_exact(ker_cols))
    //         .for_each(|(arena_row, ker_row)| {
    //             arena_row[..ker_cols].copy_from_slice(ker_row);
    //         });
    //
    //     let r2c = T::make_r2c_executor(fft_cols, fft_rows, 1).unwrap();
    //     let c2r = T::make_c2r_executor(fft_cols, fft_rows, 1).unwrap();
    //
    //     let total_scratch = r2c.scratch_length().max(c2r.scratch_length());
    //
    //     r2c.execute(&ker_arena, &mut ker_spec).unwrap();
    //
    //     let num_row_frames = out_rows.div_ceil(step_rows);
    //     let num_col_frames = out_cols.div_ceil(step_cols);
    //     let total_frames = num_row_frames * num_col_frames;
    //
    //     let mut out = vec![T::zero(); out_rows * out_cols];
    //     let unsafe_out = UnsafeSlice::new(&mut out);
    //
    //     let novtb = ThreadPool::new(self.thread_count);
    //     parallel_range_with_context(
    //         &novtb,
    //         total_frames,
    //         || TileContext {
    //             tile: vec![T::zero(); tile_len],
    //             complex_tile: vec![Complex::new(T::zero(), T::zero()); complex_tile_len],
    //             scratch: vec![Complex::new(T::zero(), T::zero()); total_scratch],
    //         },
    //         |frame_idx, ctx| {
    //             let fr = frame_idx / num_col_frames;
    //             let fc = frame_idx % num_col_frames;
    //
    //             let out_row = fr * step_rows;
    //             let out_col = fc * step_cols;
    //             let src_row = out_row as isize - (ker_rows as isize - 1);
    //             let src_col = out_col as isize - (ker_cols as isize - 1);
    //
    //             self.fill_tile(
    //                 &mut ctx.tile,
    //                 fft_cols,
    //                 signal,
    //                 sig_rows,
    //                 sig_cols,
    //                 src_row,
    //                 src_col,
    //             );
    //
    //             _ = r2c.execute_with_scratch(&ctx.tile, &mut ctx.complex_tile, &mut ctx.scratch);
    //             T::mul_spectrum(
    //                 &mut ctx.complex_tile,
    //                 &ker_spec,
    //                 (fft_cols / 2) + 1,
    //                 fft_rows,
    //                 (1. / (fft_rows * fft_cols) as f64).as_(),
    //             );
    //             _ = c2r.execute_with_scratch(
    //                 &mut ctx.complex_tile,
    //                 &mut ctx.tile,
    //                 &mut ctx.scratch,
    //             );
    //
    //             let valid_rows = step_rows.min(out_rows.saturating_sub(out_row));
    //             let valid_cols = step_cols.min(out_cols.saturating_sub(out_col));
    //
    //             // Tiles write to non-overlapping output regions so no data races.
    //             // Each (fr, fc) pair owns exactly out[out_row..out_row+valid_rows,
    //             // out_col..out_col+valid_cols] — no two frames share a row+col pair.
    //             (0..valid_rows).for_each(|vr| {
    //                 let out_row_offset = (out_row + vr) * out_cols + out_col;
    //                 let tile_row_offset = (ker_rows - 1 + vr) * fft_cols + (ker_cols - 1);
    //                 let out_ptr = unsafe_out.get_ptr(out_row_offset);
    //                 let dst_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, valid_cols) };
    //                 dst_slice
    //                     .copy_from_slice(&ctx.tile[tile_row_offset..tile_row_offset + valid_cols]);
    //             });
    //         },
    //     );
    //     out
    // }

    pub(crate) fn overlap_save_2d<K: Clone + Debug + Default + Copy + 'static + Send + Sync>(
        &self,
        dst: &mut BlurImageMut<K>,
        signal: &[T],
        sig_rows: usize,
        sig_cols: usize,
        kernel: &[T],
        ker_rows: usize,
        ker_cols: usize,
    ) where
        T: ToStorage<K>,
    {
        assert_eq!(signal.len(), sig_rows * sig_cols);
        assert_eq!(kernel.len(), ker_rows * ker_cols);

        let out_rows = sig_rows + ker_rows - 1;
        let out_cols = sig_cols + ker_cols - 1;

        let fft_rows = fft_next_good_size_real(self.block_rows + ker_rows - 1);
        let fft_cols = fft_next_good_size_real(self.block_cols + ker_cols - 1);
        let step_rows = fft_rows - (ker_rows - 1);
        let step_cols = fft_cols - (ker_cols - 1);

        let tile_len = fft_rows * fft_cols;
        let complex_tile_len = fft_rows * ((fft_cols / 2) + 1);

        let mut ker_spec = vec![Complex::new(T::zero(), T::zero()); complex_tile_len];
        let mut ker_arena = vec![T::zero(); fft_rows * fft_cols];
        ker_arena
            .chunks_exact_mut(fft_cols)
            .zip(kernel.chunks_exact(ker_cols))
            .for_each(|(arena_row, ker_row)| {
                arena_row[..ker_cols].copy_from_slice(ker_row);
            });

        let r2c = T::make_r2c_executor(fft_cols, fft_rows, 1).unwrap();
        let c2r = T::make_c2r_executor(fft_cols, fft_rows, 1).unwrap();

        let half_r = ker_rows / 2; // row offset into full conv output
        let half_c = ker_cols / 2; // col offset into full conv output

        let total_scratch = r2c.scratch_length().max(c2r.scratch_length());

        r2c.execute(&ker_arena, &mut ker_spec).unwrap();

        let num_row_frames = out_rows.div_ceil(step_rows);
        let num_col_frames = out_cols.div_ceil(step_cols);
        let total_frames = num_row_frames * num_col_frames;

        let dst_stride = dst.row_stride() as usize;
        let unsafe_dst = UnsafeSlice::new(dst.data.borrow_mut());

        let novtb = ThreadPool::new(self.thread_count);
        parallel_range_with_context(
            &novtb,
            total_frames,
            || TileContext {
                tile: vec![T::zero(); tile_len],
                complex_tile: vec![Complex::new(T::zero(), T::zero()); complex_tile_len],
                scratch: vec![Complex::new(T::zero(), T::zero()); total_scratch],
            },
            |frame_idx, ctx| {
                let fr = frame_idx / num_col_frames;
                let fc = frame_idx % num_col_frames;

                let out_row = fr * step_rows;
                let out_col = fc * step_cols;
                let src_row = out_row as isize - (ker_rows as isize - 1);
                let src_col = out_col as isize - (ker_cols as isize - 1);

                self.fill_tile(
                    &mut ctx.tile,
                    fft_cols,
                    signal,
                    sig_rows,
                    sig_cols,
                    src_row,
                    src_col,
                );

                _ = r2c.execute_with_scratch(&ctx.tile, &mut ctx.complex_tile, &mut ctx.scratch);
                T::mul_spectrum(
                    &mut ctx.complex_tile,
                    &ker_spec,
                    (fft_cols / 2) + 1,
                    fft_rows,
                    (1. / (fft_rows * fft_cols) as f64).as_(),
                );
                _ = c2r.execute_with_scratch(
                    &mut ctx.complex_tile,
                    &mut ctx.tile,
                    &mut ctx.scratch,
                );

                let valid_rows = step_rows.min(out_rows.saturating_sub(out_row));
                let valid_cols = step_cols.min(out_cols.saturating_sub(out_col));

                // Tiles write to non-overlapping output regions so no data races.
                // Each (fr, fc) pair owns exactly out[out_row..out_row+valid_rows,
                // out_col..out_col+valid_cols] — no two frames share a row+col pair.
                (0..valid_rows).for_each(|vr| {
                    let dst_row = (out_row + vr) as isize - half_r as isize;
                    if dst_row < 0 || dst_row >= sig_rows as isize {
                        return; // this tile row falls in the ramp-up/ramp-down border
                    }

                    let tile_row_offset = (ker_rows - 1 + vr) * fft_cols + (ker_cols - 1);

                    // find the valid column range that lands inside dst
                    let dst_col_start = out_col as isize - half_c as isize;

                    // clamp to [0, sig_cols)
                    let valid_start = (-dst_col_start).max(0) as usize; // skip left ramp
                    let valid_end = (sig_cols as isize - dst_col_start)
                        .min(valid_cols as isize)
                        .max(0) as usize;

                    if valid_start >= valid_end {
                        return;
                    }

                    let dst_offset = dst_row as usize * dst_stride
                        + (dst_col_start + valid_start as isize) as usize;
                    let tile_offset = tile_row_offset + valid_start;
                    let copy_len = valid_end - valid_start;

                    let dst_slice = unsafe {
                        std::slice::from_raw_parts_mut(unsafe_dst.get_ptr(dst_offset), copy_len)
                    };

                    dst_slice
                        .iter_mut()
                        .zip(ctx.tile[tile_offset..tile_offset + copy_len].iter())
                        .for_each(|(d, s)| *d = s.to_());
                });
            },
        );
    }
}
