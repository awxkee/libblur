pub(crate) fn split_channels_3<T: Copy>(
    image: &[T],
    width: usize,
    height: usize,
    first: &mut [T],
    second: &mut [T],
    third: &mut [T],
) {
    let mut shift = 0usize;
    let mut shift_plane = 0usize;
    for _ in 0..height {
        let shifted_image = &image[shift..];
        let shifted_first_plane = &mut first[shift_plane..];
        let shifted_second_plane = &mut second[shift_plane..];
        let shifted_third_plane = &mut third[shift_plane..];
        for x in 0..width {
            let px = x * 3;
            shifted_first_plane[x] = shifted_image[px];
            shifted_second_plane[x] = shifted_image[px + 1];
            shifted_third_plane[x] = shifted_image[px + 2];
        }
        shift += width * 3;
        shift_plane += width;
    }
}
