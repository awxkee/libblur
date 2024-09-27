# Fast blur algorithms library for Rust

There are some very good and blazing fast algorithms that do blurring images.
Best optimized for NEON and SSE, partially AVX, partially done WASM.

You may receive gaussian blur in 100 FPS for 4K photo.

Much faster than `image` default blur.

When 4-channels mode is in use that always considered that alpha channel is the last.

Also there are some available options to perform blurring in linear colorspace, or if methods do not fit you `f32`
options also available

# Performance

Most blur algorithms done very good and works at excellent speed. Where appropriate comparison with OpenCV is available.
For measurement was used M3 Pro with NEON feature. On x86_84 OpenCV might be better sometimes since AVX-2 is not fully supported in library

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

### Stack blur

The fastest with acceptable results. Result are quite close to gaussian and look good. Sometimes noticeable changes
may be
observed. However, if you'll use advanced analysis algorithms non gaussian methods will be detected. Not suitable for
antialias. Results just a little worse than in 'fast gaussian', however it's faster.

O(1) complexity.

```rust
libblur::stack_blur( & mut bytes, stride, width0, height, radius, FastBlurChannels::Channels3);
```

Example comparison time for blurring image 3000x4000 RGB 8-bit in single-threaded mode with 77 radius.

|         |  Time   |
|---------|:-------:|
| libblur | 43.58ms |
| OpenCV  | 89.64ms |

Example comparison time for blurring image 3000x4000 RGB 8-bit in multithreaded mode with 77 radius.

|         |  Time   |
|---------|:-------:|
| libblur | 8.68ms  |
| OpenCV  | 87.99ms |

Example comparison time for blurring image 2828x4242 RGBA 8-bit in multithreaded mode with 77 radius.

|         |  Time   |
|---------|:-------:|
| libblur | 6.73ms  |
| OpenCV  | 93.26ms |

Example comparison time for blurring image 2828x4242 RGBA 8-bit in single-threaded mode with 77 radius.

|         |  Time   |
|---------|:-------:|
| libblur | 31.18ms |
| OpenCV  | 90.82ms |

### Fast gaussian

Very fast. Result are quite close to gaussian and look good. Sometimes noticeable changes
may be
observed. However, if you'll use advanced analysis algorithms non gaussian methods will be detected. Not suitable for
antialias.
Do not use when you need gaussian. Based on binomial filter, generally speed close, might be a little faster than stack
blur ( except NEON or except non multithreaded stack blur, on NEON much faster or overcome non multithreaded
stackblur ), however results better as I see. Max radius ~320 for u8, for u16 will be less.

O(log R) complexity.

```rust
libblur::fast_gaussian( & mut bytes, stride, width0, height, radius, FastBlurChannels::Channels3);
```

Example comparison time for blurring image 3000x4000 RGB 8-bit in single-threaded mode with 77 radius.

|         |  Time   |
|---------|:-------:|
| libblur | 47.40ms |
| OpenCV  |    -    |

Example comparison time for blurring image 3000x4000 RGB 8-bit in multithreaded mode with 77 radius.

|         |  Time  |
|---------|:------:|
| libblur | 9.95ms |
| OpenCV  |   -    |

Example comparison time for blurring image 2828x4242 RGBA 8-bit in multithreaded mode with 77 radius.

|         |  Time  |
|---------|:------:|
| libblur | 9.74ms |
| OpenCV  |   --   |

Example comparison time for blurring image 2828x4242 RGBA 8-bit in single-threaded mode with 77 radius.

|         |  Time   |
|---------|:-------:|
| libblur | 43.60ms |
| OpenCV  |   --    |

### Fast gaussian next

Very fast.
Produces very pleasant results close to gaussian.
If 4K photo blurred in 10 ms this method will be done in 15 ms. Max radius ~150-180 for u8, for u16 will be less.

O(log R) complexity.

```rust
libblur::fast_gaussian_next( & mut bytes, stride, width, height, radius, FastBlurChannels::Channels3);
```

Example comparison time for blurring image 3000x4000 RGB 8-bit in single-threaded mode with 77 radius.

|         |  Time   |
|---------|:-------:|
| libblur | 53.99ms |
| OpenCV  |    -    |

Example comparison time for blurring image 3000x4000 RGB 8-bit in multithreaded mode with 77 radius.

|         |  Time   |
|---------|:-------:|
| libblur | 10.26ms |
| OpenCV  |    -    |

### Tent blur

2 sequential box blur ( [theory](https://en.wikipedia.org/wiki/Central_limit_theorem) ) that produces a tent filter.
Medium speed, good-looking results with large radius `tents` becoming more noticeable

O(1) complexity.

```rust
libblur::tent_blur(bytes, stride, & mut dst_bytes, stride, width, height, radius, FastBlurChannels::Channels3);
```

### Median blur

Median blur ( median filter ). Implementation is fast enough.

O(log R) complexity.

```rust
libblur::median_blur(bytes, stride, & mut dst_bytes, stride, width, height, radius, FastBlurChannels::Channels3);
```

Example comparison time for blurring image 3000x4000 RGB 8-bit in multithreaded mode with 35 radius.

|         |   Time   |
|---------|:--------:|
| libblur | 468.47ms |
| OpenCV  | 725.89ms |

Example comparison time for blurring image 2828x4242 RGBA 8-bit in multithreaded mode with 35 radius.

|         |   Time   |
|---------|:--------:|
| libblur | 643.22ms |
| OpenCV  | 788.93ms |

### Gaussian blur

Excellent results. Have improvements, however, much slower than any approximations slow. Use when use need gaussian
methods - smoothing, anti-alias,
FFT, advanced analysis etc.
There are two methods of convolution, integral approximation and exact,
approximation in integral form is still gaussian with 1-3% of error however about 2x faster.

Kernel size must be odd. Will panic if kernel size is not odd.

O(R) complexity.

```rust
libblur::gaussian_blur(&bytes, src_stride, & mut dst_bytes, dst_stride, width, height, kernel_size, sigma, FastBlurChannels::Channels3, GaussianPreciseLevel::EXACT);
```

Example comparison time for blurring image 3000x4000 RGB 8-bit in multithreaded mode with 151 kernel size.

|         |   Time   |
|---------|:--------:|
| libblur | 122.02ms |
| OpenCV  | 251.10ms |

Example comparison time for blurring image 2828x4242 RGBA 8-bit in multithreaded mode with 151 kernel size.

|         |   Time   |
|---------|:--------:|
| libblur | 131.21ms |
| OpenCV  | 193.67ms |

Example comparison time for blurring image 3000x4000 single plane 8-bit in multithreaded mode with 151 kernel size.

|         |  Time   |
|---------|:-------:|
| libblur | 41.65ms |
| OpenCV  | 75.94ms |

### Gaussian box blur

Generally 3 sequential box blurs it is almost gaussian
blur ( [theory](https://en.wikipedia.org/wiki/Central_limit_theorem) ), slow, really pleasant results.
Medium speed.

O(1) complexity.

```rust
libblur::gaussian_box_blur(bytes, stride, & mut dst_bytes, stride, width, height, radius, FastBlurChannels::Channels3);
```

### Box blur

Box blur. Compromise speed with bad looking results.
Medium speed.

O(1) complexity.

```rust
libblur::box_blur(bytes, stride, & mut dst_bytes, stride, width, height, radius, FastBlurChannels::Channels3);
```

Example comparison time for blurring image 3000x4000 RGB 8-bit in multithreaded mode with 77 radius.

|         |  Time   |
|---------|:-------:|
| libblur | 14.08ms |
| OpenCV  | 96.41ms |

Example comparison time for blurring image 3000x4000 RGB 8-bit in single-threaded mode with 77 radius.

|         |  Time   |
|---------|:-------:|
| libblur | 57.47ms |
| OpenCV  | 92.66ms |

Example comparison time for blurring image 2828x4242 RGBA 8-bit in multithreaded mode with 77 radius.

|         |   Time   |
|---------|:--------:|
| libblur | 12.79ms  |
| OpenCV  | 136.66ms |

Example comparison time for blurring image 2828x4242 RGBA 8-bit in single-threaded mode with 77 radius.

|         |   Time   |
|---------|:--------:|
| libblur | 51.90ms  |
| OpenCV  | 134.28ms |

### Fast bilateral blur

This is fast bilateral approximation, note this behaviour significantly differs from OpenCV.
This method has high convergence and will completely blur an image very fast with increasing spatial sigma

```rust
fast_bilateral_filter(
    src_bytes,
    &mut dst_bytes,
    dimensions.0,
    dimensions.1,
    spatial_sigma,
    range_sigma,
    FastBlurChannels::Channels3,
);
```

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE-APACHE))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.
