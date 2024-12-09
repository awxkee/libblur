#![no_main]

use libblur::{EdgeMode, FastBlurChannels, ThreadingPolicy};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: (u8, u8, u8)| {
    fuzz_16bit(
        data.0 as usize,
        data.1 as usize,
        data.2 as usize,
        FastBlurChannels::Channels4,
    );
    fuzz_16bit(
        data.0 as usize,
        data.1 as usize,
        data.2 as usize,
        FastBlurChannels::Channels3,
    );
    fuzz_16bit(
        data.0 as usize,
        data.1 as usize,
        data.2 as usize,
        FastBlurChannels::Plane,
    );
});

fn fuzz_16bit(width: usize, height: usize, radius: usize, channels: FastBlurChannels) {
    if width == 0 || height == 0 || radius == 0 {
        return;
    }
    let src_image = vec![15u16; width * height * channels.get_channels()];
    let mut dst_image = vec![0u16; width * height * channels.get_channels()];

    libblur::gaussian_blur_u16(
        &src_image,
        &mut dst_image,
        width as u32,
        height as u32,
        radius as u32 * 2 + 1,
        0.,
        channels,
        EdgeMode::Clamp,
        ThreadingPolicy::Single,
    );
}
