#[derive(Clone)]
pub(crate) struct GaussianFilter {
    pub start: usize,
    pub size: usize,
    pub filter: Vec<f32>,
}

impl GaussianFilter {
    pub(crate) fn new(start: usize, size: usize, filter: Vec<f32>) -> GaussianFilter {
        GaussianFilter {
            start,
            size,
            filter,
        }
    }
}

pub(crate) fn create_filter(length: usize, kernel_size: u32, sigma: f32) -> Vec<GaussianFilter> {
    let mut filter: Vec<GaussianFilter> = vec![GaussianFilter::new(0, 0, vec![]); length];
    let filter_radius = (kernel_size / 2) as usize;

    let filter_scale = 1f32 / (f32::sqrt(2f32 * std::f32::consts::PI) * sigma);
    for x in 0..length {
        let start = (x as i64 - filter_radius as i64).max(0) as usize;
        let end = (x + filter_radius).min(length - 1);
        let size = end - start;

        let mut real_filter = vec![];
        let mut filter_sum = 0f32;
        for j in start..end {
            let new_weight =
                f32::exp(-0.5f32 * f32::powf((j as f32 - x as f32) / sigma, 2.0f32)) * filter_scale;
            filter_sum += new_weight;
            real_filter.push(new_weight);
        }

        if filter_sum != 0f32 {
            let scale = 1f32 / filter_sum;
            let new_filter = real_filter.iter().map(|&x| x * scale).collect();
            real_filter = new_filter;
        }
        unsafe {
            *filter.get_unchecked_mut(x) = GaussianFilter::new(start, size, real_filter);
        }
    }
    filter
}
