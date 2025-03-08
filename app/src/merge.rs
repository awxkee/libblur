pub(crate) fn merge_channels_3<T: Copy>(
    image: &mut [T],
    width: usize,
    height: usize,
    first: &[T],
    second: &[T],
    third: &[T],
) {
    let mut shift = 0usize;
    let mut shift_plane = 0usize;
    for _ in 0..height {
        let shifted_image = &mut image[shift..];
        let shifted_first_plane = &first[shift_plane..];
        let shifted_second_plane = &second[shift_plane..];
        let shifted_third_plane = &third[shift_plane..];
        for x in 0..width {
            let px = x * 3;
            shifted_image[px] = shifted_first_plane[x];
            shifted_image[px + 1] = shifted_second_plane[x];
            shifted_image[px + 2] = shifted_third_plane[x];
        }
        shift += width * 3;
        shift_plane += width;
    }
}
