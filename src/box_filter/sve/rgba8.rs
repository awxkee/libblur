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
use crate::unsafe_slice::UnsafeSlice;
use std::arch::aarch64::*;

pub(crate) fn box_blur_horizontal_pass_sve<const CN: usize>(
    src: &[u8],
    src_stride: u32,
    dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    width: u32,
    radius: u32,
    start_y: u32,
    end_y: u32,
) {
    unsafe {
        box_blur_horizontal_pass_sve_impl::<CN>(
            src, src_stride, dst, dst_stride, width, radius, start_y, end_y,
        );
    }
}

#[target_feature(enable = "sve2", enable = "sve")]
fn box_blur_horizontal_pass_sve_impl<const CN: usize>(
    src: &[u8],
    src_stride: u32,
    dst: &UnsafeSlice<u8>,
    dst_stride: u32,
    width: u32,
    radius: u32,
    start_y: u32,
    end_y: u32,
) {
    unsafe {
        let kernel_size = radius * 2 + 1;
        let edge_count = ((kernel_size / 2) + 1) as i32;

        const Q: f64 = ((1i64 << 31i64) - 1) as f64;
        let weight = Q / (radius as f64 * 2. + 1.);
        let v_weight = svdup_n_s32(weight as i32);

        let pv_cn = svwhilelt_b32_u32(0u32, CN as u32);

        let half_kernel = kernel_size / 2;

        let mut yy = start_y;

        let src_stride = src_stride as usize;

        while yy + 4 <= end_y {
            let y = yy;
            let y_src_shift = y as usize * src_stride;
            let y_dst_shift = y as usize * dst_stride as usize;

            let s0 = src;
            let s1 = src.get_unchecked(y_src_shift + src_stride..);
            let s2 = src.get_unchecked(y_src_shift + src_stride * 2..);
            let s3 = src.get_unchecked(y_src_shift + src_stride * 3..);

            let mut store_0: svint32_t;
            let mut store_1: svint32_t;
            let mut store_2: svint32_t;
            let mut store_3: svint32_t;

            {
                let edge_colors_0 = svld1ub_s32(pv_cn, s0.as_ptr().cast());
                let edge_colors_1 = svld1ub_s32(pv_cn, s1.as_ptr().cast());
                let edge_colors_2 = svld1ub_s32(pv_cn, s2.as_ptr().cast());
                let edge_colors_3 = svld1ub_s32(pv_cn, s3.as_ptr().cast());

                store_0 = svmul_n_s32_x(pv_cn, edge_colors_0, edge_count);
                store_1 = svmul_n_s32_x(pv_cn, edge_colors_1, edge_count);
                store_2 = svmul_n_s32_x(pv_cn, edge_colors_2, edge_count);
                store_3 = svmul_n_s32_x(pv_cn, edge_colors_3, edge_count);
            }

            {
                for x in 1..=half_kernel as usize {
                    let px = x.min(width as usize - 1) * CN;

                    let s0 = s0.get_unchecked(px..);
                    let s1 = s1.get_unchecked(px..);
                    let s2 = s2.get_unchecked(px..);
                    let s3 = s3.get_unchecked(px..);

                    let edge_colors_0 = svld1ub_s32(pv_cn, s0.as_ptr().cast());
                    let edge_colors_1 = svld1ub_s32(pv_cn, s1.as_ptr().cast());
                    let edge_colors_2 = svld1ub_s32(pv_cn, s2.as_ptr().cast());
                    let edge_colors_3 = svld1ub_s32(pv_cn, s3.as_ptr().cast());

                    store_0 = svadd_s32_x(pv_cn, store_0, edge_colors_0);
                    store_1 = svadd_s32_x(pv_cn, store_1, edge_colors_1);
                    store_2 = svadd_s32_x(pv_cn, store_2, edge_colors_2);
                    store_3 = svadd_s32_x(pv_cn, store_3, edge_colors_3);
                }
            }

            for x in 0..width {
                let px = x as usize * CN;

                {
                    let scale_store0 = svqrdmulh_s32(store_0, v_weight);
                    let scale_store1 = svqrdmulh_s32(store_1, v_weight);
                    let scale_store2 = svqrdmulh_s32(store_2, v_weight);
                    let scale_store3 = svqrdmulh_s32(store_3, v_weight);

                    let bytes_offset_0 = y_dst_shift + px;
                    let bytes_offset_1 = y_dst_shift + dst_stride as usize + px;
                    let bytes_offset_2 = y_dst_shift + dst_stride as usize * 2 + px;
                    let bytes_offset_3 = y_dst_shift + dst_stride as usize * 3 + px;
                    svst1b_u32(
                        pv_cn,
                        dst.get(bytes_offset_0) as *mut u8 as *mut _,
                        svreinterpret_u32_s32(scale_store0),
                    );
                    svst1b_u32(
                        pv_cn,
                        dst.get(bytes_offset_1) as *mut u8 as *mut _,
                        svreinterpret_u32_s32(scale_store1),
                    );
                    svst1b_u32(
                        pv_cn,
                        dst.get(bytes_offset_2) as *mut u8 as *mut _,
                        svreinterpret_u32_s32(scale_store2),
                    );
                    svst1b_u32(
                        pv_cn,
                        dst.get(bytes_offset_3) as *mut u8 as *mut _,
                        svreinterpret_u32_s32(scale_store3),
                    );
                }

                // subtract previous
                {
                    let previous_x = (x as i64 - half_kernel as i64).max(0) as usize;
                    let previous = previous_x * CN;

                    let s_ptr_0 = s0.get_unchecked(previous..);
                    let s_ptr_1 = s1.get_unchecked(previous..);
                    let s_ptr_2 = s2.get_unchecked(previous..);
                    let s_ptr_3 = s3.get_unchecked(previous..);

                    let edge_colors_0 = svld1ub_s32(pv_cn, s_ptr_0.as_ptr().cast());
                    let edge_colors_1 = svld1ub_s32(pv_cn, s_ptr_1.as_ptr().cast());
                    let edge_colors_2 = svld1ub_s32(pv_cn, s_ptr_2.as_ptr().cast());
                    let edge_colors_3 = svld1ub_s32(pv_cn, s_ptr_3.as_ptr().cast());

                    store_0 = svsub_s32_x(pv_cn, store_0, edge_colors_0);
                    store_1 = svsub_s32_x(pv_cn, store_1, edge_colors_1);
                    store_2 = svsub_s32_x(pv_cn, store_2, edge_colors_2);
                    store_3 = svsub_s32_x(pv_cn, store_3, edge_colors_3);
                }

                // add next
                {
                    let next_x = (x + half_kernel + 1).min(width - 1) as usize;

                    let next = next_x * CN;

                    let s_ptr_0 = s0.get_unchecked(next..);
                    let s_ptr_1 = s1.get_unchecked(next..);
                    let s_ptr_2 = s2.get_unchecked(next..);
                    let s_ptr_3 = s3.get_unchecked(next..);

                    let edge_colors_0 = svld1ub_s32(pv_cn, s_ptr_0.as_ptr().cast());
                    let edge_colors_1 = svld1ub_s32(pv_cn, s_ptr_1.as_ptr().cast());
                    let edge_colors_2 = svld1ub_s32(pv_cn, s_ptr_2.as_ptr().cast());
                    let edge_colors_3 = svld1ub_s32(pv_cn, s_ptr_3.as_ptr().cast());

                    store_0 = svadd_s32_x(pv_cn, store_0, edge_colors_0);
                    store_1 = svadd_s32_x(pv_cn, store_1, edge_colors_1);
                    store_2 = svadd_s32_x(pv_cn, store_2, edge_colors_2);
                    store_3 = svadd_s32_x(pv_cn, store_3, edge_colors_3);
                }
            }

            yy += 4;
        }

        for y in yy..end_y {
            let y_src_shift = y as usize * src_stride;
            let y_dst_shift = y as usize * dst_stride as usize;

            let mut store: svint32_t = {
                let s_ptr = src.get_unchecked(y_src_shift..).as_ptr();
                let edge_colors = svld1ub_s32(pv_cn, s_ptr.cast());
                svmul_n_s32_x(pv_cn, edge_colors, edge_count)
            };

            {
                for x in 1..=half_kernel as usize {
                    let px = x.min(width as usize - 1) * CN;
                    let s_val =
                        svld1ub_s32(pv_cn, src.get_unchecked(y_src_shift + px..).as_ptr().cast());
                    store = svadd_s32_x(pv_cn, store, s_val);
                }
            }

            for x in 0..width {
                let px = x as usize * CN;

                let scale_store = svqrdmulh_s32(store, v_weight);

                let bytes_offset = y_dst_shift + px;
                {
                    svst1b_u32(
                        pv_cn,
                        dst.get_ptr(bytes_offset).cast(),
                        svreinterpret_u32_s32(scale_store),
                    );
                }

                // subtract previous
                {
                    let previous_x = (x as isize - half_kernel as isize).max(0) as usize;
                    let previous = previous_x * CN;
                    let s_ptr = src.get_unchecked(y_src_shift + previous..);
                    let edge_colors = svld1ub_s32(pv_cn, s_ptr.as_ptr().cast());
                    store = svsub_s32_x(pv_cn, store, edge_colors);
                }

                // add next
                {
                    let next_x = (x + half_kernel + 1).min(width - 1) as usize;

                    let next = next_x * CN;

                    let s_ptr = src.get_unchecked(y_src_shift + next..);
                    let edge_colors = svld1ub_s32(pv_cn, s_ptr.as_ptr().cast());
                    store = svadd_s32_x(pv_cn, store, edge_colors);
                }
            }
        }
    }
}
