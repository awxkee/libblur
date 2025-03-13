# Fast blur algorithms library for Rust

There are some very good and blazing fast algorithms that do blurring images.
Also providing convenient api for doing convolution and some edge detection filters.\
Best optimized for NEON, SSE and AVX, partially done WASM.

You may receive gaussian blur in 100 FPS for 4K photo.

Much faster than `image` default blur.

When 4-channels mode is in use that always considered that alpha channel is the last.

Also there are some available options to perform blurring in linear colorspace, or if methods do not fit you `f32`
options also available

# Performance

Most blur algorithms done very good and works at excellent speed. Where appropriate comparison with OpenCV is available.
For measurement was used M3 Pro with NEON feature. On x86_84 OpenCV might be better sometimes since AVX-2 is not fully
supported in library

# Usage

```
cargo add libblur
```

#### Usage with image feature

```rust
let blurred = gaussian_blur_image(
    img,
    61,
    0.,
    EdgeMode::Clamp,
    GaussianPreciseLevel::INTEGRAL,
    ThreadingPolicy::Adaptive,
)
.unwrap();

blurred
.save_with_format("blurred.jpg", ImageFormat::Jpeg)
.unwrap();
```

### Gaussian blur

Excellent results. Have improvements, however, much slower than any approximations. Use when use need gaussian
methods - smoothing, FFT, advanced analysis etc.
There are two methods of convolution, fixed point approximation and exact,
approximation in fixed point adds 1-3% of error. However, it is about two times faster.

Kernel size must be odd. Will panic if kernel size is not odd.

O(R) complexity.

```rust
libblur::gaussian_blur( & bytes, src_stride, & mut dst_bytes, dst_stride, width, height, kernel_size, sigma, FastBlurChannels::Channels3, GaussianPreciseLevel::EXACT);
```

Example comparison time for blurring image 3000x4000 RGB 8-bit in multithreaded mode with 151 kernel size.

|                     | Time(NEON) | Time(AVX) | 
|---------------------|:----------:|:---------:| 
| libblur(Exact)      |  52.39ms   |  43.41ms  | 
| libblur(FixedPoint) |  33.20ms   |  30.72ms  | 
| OpenCV              |  180.56ms  | 182.44ms  | 

Example comparison time for blurring image 2828x4242 RGBA 8-bit in multithreaded mode with 151 kernel size.

|                     | time(NEON) | Time(AVX) |
|---------------------|:----------:|:---------:|
| libblur(Exact)      |  66.13ms   |  54.36ms  |
| libblur(FixedPoint) |  40.13ms   |  38.91ms  |
| OpenCV              |  177.46ms  | 185.30ms  |

Example comparison time for blurring image 3000x4000 single plane 8-bit in multithreaded mode with 151 kernel size.

|                     | time(NEON) | Time(SSE/AVX) |
|---------------------|:----------:|:-------------:|
| libblur(Exact)      |  17.59ms   |    15.51ms    |
| libblur(FixedPoint) |   9.88ms   |    11.45ms    |
| OpenCV              |  74.73ms   |    64.20ms    |

### Stack blur

The fastest with acceptable results. Result are quite close to gaussian and look good. Sometimes noticeable changes
may be
observed. However, if you'll use advanced analysis algorithms non gaussian methods will be detected. Not suitable for
advanced analysis. Results just a little worse than in 'fast gaussian', however it's faster.

O(1) complexity.

```rust
let mut dst_image = BlurImageMut::borrow(&mut src_bytes, dyn_image.width(), dyn_image.height(), FastBlurChannels::Channels3)
libblur::stack_blur( &mut dst_image, 10, ThreadingPolicy::Single).unwrap();
```

Example comparison time for blurring image 3000x4000 RGB 8-bit in multithreaded mode with 77 radius.

|         | time(NEON) | time(SSE) |
|---------|:----------:|:---------:|
| libblur |   5.77ms   |  5.30ms   |
| OpenCV  |   8.43ms   |  10.36ms  |

Example comparison time for blurring image 2828x4242 RGBA 8-bit in multithreaded mode with 77 radius.

|         | time(NEON) | time(SSE) |
|---------|:----------:|:---------:|
| libblur |   5.58ms   |  5.48ms   |
| OpenCV  |   8.00ms   |  8.55ms   |

### Fast gaussian

Very fast. Result are quite close to gaussian and look good. Sometimes noticeable changes
may be
observed. However, if you'll use advanced analysis algorithms non gaussian methods will be detected. Not suitable for
advanced analysis.
Do not use when you need gaussian. Based on binomial filter, generally speed close, might be a little faster than stack
blur , however results are better.

O(log R) complexity.

```rust
let mut dst_image = BlurImageMut::borrow(&mut src_bytes, dyn_image.width(), dyn_image.height(), FastBlurChannels::Channels3)
libblur::fast_gaussian(&mut dst_image,10, ThreadingPolicy::Single, EdgeMode::Wrap).unwrap();
```

Example comparison time for blurring image 3000x4000 RGB 8-bit in multithreaded mode with 77 radius.

|         | time(NEON) | time(SSE) | 
|---------|:----------:|:---------:| 
| libblur |   5.92ms   |  6.38ms   | 
| OpenCV  |     -      |     -     | 

Example comparison time for blurring image 2828x4242 RGBA 8-bit in multithreaded mode with 77 radius.

|         | time(NEON) | time(SSE) |
|---------|:----------:|:---------:|
| libblur |   6.65ms   |  7.18ms   |
| OpenCV  |     --     |    --     |

### Fast gaussian next

Very fast.
Produces very pleasant results close to gaussian. Max radius ~150-180 for u8, for u16 will be less.

O(log R) complexity.

```rust
let mut dst_image = BlurImageMut::borrow(&mut src_bytes, dyn_image.width(), dyn_image.height(), FastBlurChannels::Channels3)
libblur::fast_gaussian_next(&mut dst_image,10, ThreadingPolicy::Single, EdgeMode::Wrap).unwrap();
```

Example comparison time for blurring image 2828x4242 RGBA 8-bit in multithreaded mode with 35 radius.

|         | time(NEON) | time(SSE) |
|---------|:----------:|:---------:| 
| libblur |   4.18ms   |  5.29ms   | 
| OpenCV  |     -      |     -     | 

Example comparison time for blurring image 3000x4000 RGB 8-bit in multithreaded mode with 77 radius.

|         | time(NEON) | time(SSE) |
|---------|:----------:|:---------:|
| libblur |   4.46ms   |  5.39ms   |
| OpenCV  |     -      |     -     |

### Tent blur

2 sequential box blur ( [theory](https://en.wikipedia.org/wiki/Central_limit_theorem) ) that produces a tent filter.
Medium speed, good-looking results with large radius `tents` becoming more noticeable

O(1) complexity.

```rust
let image = BlurImage::borrow(
    &src_bytes,
    dyn_image.width(),
    dyn_image.height(),
    FastBlurChannels::Channels3,
);
let mut dst_image = BlurImageMut::default();
libblur::tent_blur(&image, &mut dst_image,10f32, ThreadingPolicy::Single).unwrap();
```

### Median blur

Median blur ( median filter ). Implementation is fast enough.

O(log R) complexity.

```rust
let image = BlurImage::borrow(
    &src_bytes,
    dyn_image.width(),
    dyn_image.height(),
    FastBlurChannels::Channels3,
);
let mut dst_image = BlurImageMut::default();
libblur::median_blur(&image, &mut dst_image,10, ThreadingPolicy::Single).unwrap();
```

Example comparison time for blurring image 3000x4000 RGB 8-bit in multithreaded mode with 35 radius.

|         | time(NEON) | time(SSE) |
|---------|:----------:|:---------:|
| libblur |  603.51ms  | 872.03ms  |
| OpenCV  |  637.83ms  | 959.07ms  |

Example comparison time for blurring image 2828x4242 RGBA 8-bit in multithreaded mode with 35 radius.

|         | time(NEON) | time(SSE) |
|---------|:----------:|:---------:|
| libblur |  643.22ms  | 695.75ms  |
| OpenCV  |  664.22ms  | 808.21ms  |

### Gaussian box blur

Generally 3 sequential box blurs it is almost gaussian
blur ( [theory](https://en.wikipedia.org/wiki/Central_limit_theorem) ), slow, really pleasant results.
Medium speed.

O(1) complexity.

```rust
let image = BlurImage::borrow(
    &src_bytes,
    dyn_image.width(),
    dyn_image.height(),
    FastBlurChannels::Channels3,
);
let mut dst_image = BlurImageMut::default();
libblur::gaussian_box_blur(&image, &mut dst_image, 10f32, ThreadingPolicy::Single).unwrap();
```

### Box blur

Box blur. Compromise speed with bad looking results.
Medium speed.

O(1) complexity.

```rust
let image = BlurImage::borrow(
    &src_bytes,
    dyn_image.width(),
    dyn_image.height(),
    FastBlurChannels::Channels3,
);
let mut dst_image = BlurImageMut::default();
libblur::box_blur(&image, &mut dst_image,10, ThreadingPolicy::Single).unwrap();
```

Example comparison time for blurring image 3000x4000 RGB 8-bit in multithreaded mode with 77 radius.

|         | time(NEON) | time(SSE) |
|---------|:----------:|:---------:|
| libblur |   6.21ms   |  9.48ms   |
| OpenCV  |  15.73ms   |  43.59ms  |

Example comparison time for blurring image 2828x4242 RGBA 8-bit in multithreaded mode with 77 radius.

|         | Time(NEON) | Time(SSE) |
|---------|:----------:|:---------:|
| libblur |   6.13ms   |  7.99ms   |
| OpenCV  |  15.77ms   |  31.29ms  |

Example comparison time for blurring image 2828x4242 RGBA 8-bit in single-thread mode with 15 radius on MacOS.

|                  | Time(NEON) |
|------------------|:----------:|
| libblur          |  20.64ms   |
| Apple Accelerate |  37.39ms   |

### Fast bilateral blur

This is fast bilateral approximation, note this behaviour significantly differs from OpenCV.
This method has high convergence and will completely blur an image very fast with increasing spatial sigma.
By the nature of this filter the more spatial sigma are the faster method is.

```rust
let image = BlurImage::borrow(
    &src_bytes,
    dyn_image.width(),
    dyn_image.height(),
    FastBlurChannels::Channels3,
);
let mut dst_image = BlurImageMut::default();
libblur::fast_bilateral_filter(&image, &mut dst_image, 25, 7f32, 7f32).unwrap();
```

### Common speed chain

This is arbitrary example for blurring speed for all methods in descending order. 

stack_blur -> fast_gaussian_next -> fast_gaussian -> box_blur -> fast_gaussian_superior -> tent_blur -> gaussian_box_blur -> gaussian_blur -> bilateral -> median

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.
