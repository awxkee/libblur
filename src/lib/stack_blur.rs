use crate::mul_table::{MUL_TABLE_STACK_BLUR, SHR_TABLE_STACK_BLUR};
use crate::FastBlurChannels;

#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub(crate) struct BlurStack {
    pub r: i32,
    pub g: i32,
    pub b: i32,
    pub a: i32,
}

impl BlurStack {
    pub fn new() -> BlurStack {
        BlurStack {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        }
    }
}

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
enum StackBlurPass {
    HORIZONTAL,
    VERTICAL,
}

fn stack_blur_pass<const COMPONENTS: usize>(
    pixels: &mut [u8],
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    pass: StackBlurPass,
    thread: usize,
    total_threads: usize,
) {
    let div = ((radius * 2) + 1) as usize;
    let (mut xp, mut yp);
    let mut sp;
    let mut stack_start;
    let mut stacks = vec![];
    for _ in 0..div * total_threads {
        stacks.push(Box::new(BlurStack::new()));
    }

    let mut sum_r: i32;
    let mut sum_g: i32;
    let mut sum_b: i32;
    let mut sum_a: i32;
    let mut sum_in_r: i32;
    let mut sum_in_g: i32;
    let mut sum_in_b: i32;
    let mut sum_in_a: i32;
    let mut sum_out_r: i32;
    let mut sum_out_g: i32;
    let mut sum_out_b: i32;
    let mut sum_out_a: i32;

    let wm = width - 1;
    let hm = height - 1;
    let div = (radius * 2) + 1;
    let mul_sum = MUL_TABLE_STACK_BLUR[radius as usize];
    let shr_sum = SHR_TABLE_STACK_BLUR[radius as usize];

    let mut src_ptr;
    let mut dst_ptr;

    if pass == StackBlurPass::HORIZONTAL {
        let min_y = thread * height as usize / total_threads;
        let max_y = (thread + 1) * height as usize / total_threads;

        for y in min_y..max_y {
            sum_r = 0;
            sum_g = 0;
            sum_b = 0;
            sum_a = 0;
            sum_in_r = 0;
            sum_in_g = 0;
            sum_in_b = 0;
            sum_in_a = 0;
            sum_out_r = 0;
            sum_out_g = 0;
            sum_out_b = 0;
            sum_out_a = 0;

            src_ptr = stride as usize * y; // start of line (0,y)

            let src_r = unsafe { *pixels.get_unchecked(src_ptr + 0) as i32 };
            let src_g = unsafe { *pixels.get_unchecked(src_ptr + 1) as i32 };
            let src_b = unsafe { *pixels.get_unchecked(src_ptr + 2) as i32 };
            let src_a = if COMPONENTS == 4 {
                unsafe { *pixels.get_unchecked(src_ptr + 3) as i32 }
            } else {
                0i32
            };

            for i in 0..=radius {
                let stack_value = unsafe { &mut *stacks.get_unchecked_mut(i as usize) };
                stack_value.r = src_r;
                stack_value.g = src_g;
                stack_value.b = src_b;
                if COMPONENTS == 4 {
                    stack_value.a = src_a;
                }

                sum_r += src_r * (i + 1) as i32;
                sum_g += src_g * (i + 1) as i32;
                sum_b += src_b * (i + 1) as i32;
                if COMPONENTS == 4 {
                    sum_a += src_a * (i + 1) as i32;
                }

                sum_out_r += src_r;
                sum_out_g += src_g;
                sum_out_b += src_b;
                if COMPONENTS == 4 {
                    sum_out_a += src_a;
                }
            }

            for i in 1..=radius {
                if i <= wm {
                    src_ptr += COMPONENTS;
                }
                let stack_ptr = unsafe { &mut *stacks.get_unchecked_mut((i + radius) as usize) };

                let src_r = unsafe { *pixels.get_unchecked(src_ptr + 0) as i32 };
                let src_g = unsafe { *pixels.get_unchecked(src_ptr + 1) as i32 };
                let src_b = unsafe { *pixels.get_unchecked(src_ptr + 2) as i32 };
                let src_a = if COMPONENTS == 4 {
                    unsafe { *pixels.get_unchecked(src_ptr + 3) as i32 }
                } else {
                    0i32
                };

                stack_ptr.r = src_r;
                stack_ptr.g = src_g;
                stack_ptr.b = src_b;
                if COMPONENTS == 4 {
                    stack_ptr.a = src_a;
                }

                sum_r += src_r * (radius + 1 - i) as i32;
                sum_g += src_g * (radius + 1 - i) as i32;
                sum_b += src_b * (radius + 1 - i) as i32;
                if COMPONENTS == 4 {
                    sum_a += src_a * (radius + 1 - i) as i32;
                }

                sum_in_r += src_r;
                sum_in_g += src_g;
                sum_in_b += src_b;
                if COMPONENTS == 4 {
                    sum_in_a += src_a;
                }
            }

            sp = radius;
            xp = radius;
            if xp > wm {
                xp = wm;
            }

            src_ptr = COMPONENTS * xp as usize + y * stride as usize;
            dst_ptr = y * stride as usize;
            for _ in 0..width {
                unsafe {
                    *pixels.get_unchecked_mut(dst_ptr + 0) = ((sum_r * mul_sum) >> shr_sum) as u8;
                    *pixels.get_unchecked_mut(dst_ptr + 1) = ((sum_g * mul_sum) >> shr_sum) as u8;
                    *pixels.get_unchecked_mut(dst_ptr + 2) = ((sum_b * mul_sum) >> shr_sum) as u8;
                    if COMPONENTS == 4 {
                        *pixels.get_unchecked_mut(dst_ptr + 3) =
                            ((sum_a * mul_sum) >> shr_sum) as u8;
                    }
                }
                dst_ptr += COMPONENTS;

                sum_r -= sum_out_r;
                sum_g -= sum_out_g;
                sum_b -= sum_out_b;
                if COMPONENTS == 4 {
                    sum_a -= sum_out_a;
                }

                stack_start = sp + div - radius;
                if stack_start >= div {
                    stack_start -= div;
                }
                let stack = unsafe { &mut *stacks.get_unchecked_mut(stack_start as usize) };

                sum_out_r -= stack.r;
                sum_out_g -= stack.g;
                sum_out_b -= stack.b;
                sum_out_a -= stack.a;

                if xp < wm {
                    src_ptr += COMPONENTS;
                    xp += 1;
                }

                let src_r = unsafe { *pixels.get_unchecked(src_ptr + 0) as i32 };
                let src_g = unsafe { *pixels.get_unchecked(src_ptr + 1) as i32 };
                let src_b = unsafe { *pixels.get_unchecked(src_ptr + 2) as i32 };
                let src_a = if COMPONENTS == 4 {
                    unsafe { *pixels.get_unchecked(src_ptr + 3) as i32 }
                } else {
                    0i32
                };
                stack.r = src_r;
                stack.g = src_g;
                stack.b = src_b;
                if COMPONENTS == 4 {
                    stack.a = src_a;
                }

                sum_in_r += src_r;
                sum_in_g += src_g;
                sum_in_b += src_b;
                if COMPONENTS == 4 {
                    sum_in_a += src_a;
                }

                sum_r += sum_in_r;
                sum_g += sum_in_g;
                sum_b += sum_in_b;
                sum_a += sum_in_a;

                sp += 1;
                if sp >= div {
                    sp = 0;
                }
                let stack = unsafe { &mut *stacks.get_unchecked_mut(sp as usize) };

                sum_out_r += stack.r;
                sum_out_g += stack.g;
                sum_out_b += stack.b;
                sum_out_a += stack.a;
                sum_in_r -= stack.r;
                sum_in_g -= stack.g;
                sum_in_b -= stack.b;
                sum_in_a -= stack.a;
            }
        }
    } else if pass == StackBlurPass::VERTICAL {
        let min_x = thread * width as usize / total_threads;
        let max_x = (thread + 1) * width as usize / total_threads;

        for x in min_x..max_x {
            sum_r = 0;
            sum_g = 0;
            sum_b = 0;
            sum_a = 0;
            sum_in_r = 0;
            sum_in_g = 0;
            sum_in_b = 0;
            sum_in_a = 0;
            sum_out_r = 0;
            sum_out_g = 0;
            sum_out_b = 0;
            sum_out_a = 0;

            src_ptr = COMPONENTS * x; // x,0

            let src_r = unsafe { *pixels.get_unchecked(src_ptr + 0) as i32 };
            let src_g = unsafe { *pixels.get_unchecked(src_ptr + 1) as i32 };
            let src_b = unsafe { *pixels.get_unchecked(src_ptr + 2) as i32 };
            let src_a = if COMPONENTS == 4 {
                unsafe { *pixels.get_unchecked(src_ptr + 3) as i32 }
            } else {
                0i32
            };

            for i in 0..=radius {
                let stack_value = unsafe { &mut *stacks.get_unchecked_mut(i as usize) };
                stack_value.r = src_r;
                stack_value.g = src_g;
                stack_value.b = src_b;
                if COMPONENTS == 4 {
                    stack_value.a = src_a;
                }

                sum_r += src_r * (i + 1) as i32;
                sum_g += src_g * (i + 1) as i32;
                sum_b += src_b * (i + 1) as i32;
                if COMPONENTS == 4 {
                    sum_a += src_a * (i + 1) as i32;
                }

                sum_out_r += src_r;
                sum_out_g += src_g;
                sum_out_b += src_b;
                if COMPONENTS == 4 {
                    sum_out_a += src_a;
                }
            }

            for i in 1..=radius {
                if i <= hm {
                    src_ptr += stride as usize;
                }

                let stack_ptr = unsafe { &mut *stacks.get_unchecked_mut((i + radius) as usize) };

                let src_r = unsafe { *pixels.get_unchecked(src_ptr + 0) as i32 };
                let src_g = unsafe { *pixels.get_unchecked(src_ptr + 1) as i32 };
                let src_b = unsafe { *pixels.get_unchecked(src_ptr + 2) as i32 };
                let src_a = if COMPONENTS == 4 {
                    unsafe { *pixels.get_unchecked(src_ptr + 3) as i32 }
                } else {
                    0i32
                };

                stack_ptr.r = src_r;
                stack_ptr.g = src_g;
                stack_ptr.b = src_b;
                if COMPONENTS == 4 {
                    stack_ptr.a = src_a;
                }

                sum_r += src_r * (radius + 1 - i) as i32;
                sum_g += src_g * (radius + 1 - i) as i32;
                sum_b += src_b * (radius + 1 - i) as i32;
                if COMPONENTS == 4 {
                    sum_a += src_a * (radius + 1 - i) as i32;
                }
                sum_in_r += src_r;
                sum_in_g += src_g;
                sum_in_b += src_b;
                if COMPONENTS == 4 {
                    sum_in_a += src_a;
                }
            }

            sp = radius;
            yp = radius;
            if yp > hm {
                yp = hm;
            }
            src_ptr = COMPONENTS * x + yp as usize * stride as usize;
            dst_ptr = COMPONENTS * x;
            for _ in 0..height {
                unsafe {
                    *pixels.get_unchecked_mut(dst_ptr + 0) = ((sum_r * mul_sum) >> shr_sum) as u8;
                    *pixels.get_unchecked_mut(dst_ptr + 1) = ((sum_g * mul_sum) >> shr_sum) as u8;
                    *pixels.get_unchecked_mut(dst_ptr + 2) = ((sum_b * mul_sum) >> shr_sum) as u8;
                    if COMPONENTS == 4 {
                        *pixels.get_unchecked_mut(dst_ptr + 3) =
                            ((sum_a * mul_sum) >> shr_sum) as u8;
                    }
                }
                dst_ptr += stride as usize;

                sum_r -= sum_out_r;
                sum_g -= sum_out_g;
                sum_b -= sum_out_b;
                if COMPONENTS == 4 {
                    sum_a -= sum_out_a;
                }

                stack_start = sp + div - radius;
                if stack_start >= div {
                    stack_start -= div;
                }
                let stack_ptr = unsafe { &mut *stacks.get_unchecked_mut(stack_start as usize) };

                sum_out_r -= stack_ptr.r;
                sum_out_g -= stack_ptr.g;
                sum_out_b -= stack_ptr.b;
                if COMPONENTS == 4 {
                    sum_out_a -= stack_ptr.a;
                }

                if yp < hm {
                    src_ptr += stride as usize; // stride
                    yp += 1;
                }

                let src_r = unsafe { *pixels.get_unchecked(src_ptr + 0) as i32 };
                let src_g = unsafe { *pixels.get_unchecked(src_ptr + 1) as i32 };
                let src_b = unsafe { *pixels.get_unchecked(src_ptr + 2) as i32 };
                let src_a = if COMPONENTS == 4 {
                    unsafe { *pixels.get_unchecked(src_ptr + 3) as i32 }
                } else {
                    0i32
                };

                stack_ptr.r = src_r;
                stack_ptr.g = src_g;
                stack_ptr.b = src_b;
                if COMPONENTS == 4 {
                    stack_ptr.a = src_a;
                }

                sum_in_r += src_r;
                sum_in_g += src_g;
                sum_in_b += src_b;
                if COMPONENTS == 4 {
                    sum_in_a += src_a;
                }

                sum_r += sum_in_r;
                sum_g += sum_in_g;
                sum_b += sum_in_b;
                if COMPONENTS == 4 {
                    sum_a += sum_in_a;
                }
                sp += 1;

                if sp >= div {
                    sp = 0;
                }
                let stack_ptr = unsafe { &mut *stacks.get_unchecked_mut(sp as usize) };

                sum_out_r += stack_ptr.r;
                sum_out_g += stack_ptr.g;
                sum_out_b += stack_ptr.b;
                if COMPONENTS == 4 {
                    sum_out_a += stack_ptr.a;
                }
                sum_in_r -= stack_ptr.r;
                sum_in_g -= stack_ptr.g;
                sum_in_b -= stack_ptr.b;
                if COMPONENTS == 4 {
                    sum_in_a -= stack_ptr.a;
                }
            }
        }
    }
}

#[no_mangle]
pub fn stack_blur(
    in_place: &mut [u8],
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    channels: FastBlurChannels,
) {
    match channels {
        FastBlurChannels::Channels3 => {
            stack_blur_pass::<3>(
                in_place,
                stride,
                width,
                height,
                radius,
                StackBlurPass::HORIZONTAL,
                0,
                1,
            );
            stack_blur_pass::<3>(
                in_place,
                stride,
                width,
                height,
                radius,
                StackBlurPass::VERTICAL,
                0,
                1,
            );
        }
        FastBlurChannels::Channels4 => {
            stack_blur_pass::<4>(
                in_place,
                stride,
                width,
                height,
                radius,
                StackBlurPass::HORIZONTAL,
                0,
                1,
            );
            stack_blur_pass::<4>(
                in_place,
                stride,
                width,
                height,
                radius,
                StackBlurPass::VERTICAL,
                0,
                1,
            );
        }
    }
}
