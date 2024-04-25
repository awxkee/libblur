# Fast blur algorithms library for Rust

There are some very good ( and sometimes blazing fast ) algorithms that do blurring uint images.
Best optimized for NEON.

You may receive gaussian blur in 100 FPS for 4K photo.

Much faster than `image` default blur.

# Usage

```
cargo add libblur
```

### Fast gaussian

Very fast. Faster than any other blur. Result are quite close to gaussian and look good. Sometimes noticeable changes
may be
observed. However, if you'll use advanced analysis algorithms non gaussian methods will be detected. Not suitable for
antialias.
Do not use when you need gaussian. Based on binomial filter, generally speed close, might be a little faster than stack
blur ( except NEON or except non multithreaded stack blur, on NEON much faster or overcome non multithreaded
stackblur ), however results better as I see. Max radius ~300-500 for u8, for u16 will be less.

On my M3 4K photo blurred in 10ms that means it is 4K 100fps blur :)

O(1) complexity.

```rust
libblur::fast_gaussian( & mut bytes, stride, width0, height, radius, FastBlurChannels::Channels3);
```

### Fast gaussian next

Very fast.
Produces very pleasant results close to gaussian.
If 4K photo blurred in 10 ms this method will be done in 15 ms. Max radius ~150-180 for u8, for u16 will be less.
Faster than stack blur.

O(1) complexity.

```rust
libblur::fast_gaussian_next(& mut bytes, stride, width0, height, radius, FastBlurChannels::Channels3);
```

### Tent blur

2 sequential box blur. Medium speed, good-looking results.

O(1) complexity.

```rust
libblur::tent_blur(bytes, stride, & mut dst_bytes, stride, width, height, radius, FastBlurChannels::Channels3);
```

### Median blur

Median blur ( median filter ). Implementation is fast enough.

O(R) complexity.

```rust
libblur::median_blur(bytes, stride, & mut dst_bytes, stride, width, height, radius, FastBlurChannels::Channels3);
```

### Gaussian blur

Excellent results. Have some improvements, however, slow. Use when use need gaussian methods - smoothing, anti-alias,
FFT analysis etc.

Kernel size must be odd. Will panic if kernel size is not odd.

O(N*K) complexity.

```rust
libblur::gaussian_blur( & bytes, src_stride, & mut dst_bytes, dst_stride, width, height, kernel_size, sigma, FastBlurChannels::Channels3);
```

### Gaussian box blur

generally 3 sequential box blurs it is almost gaussian blur, slow, really pleasant results.
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
