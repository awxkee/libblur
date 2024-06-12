#[repr(C)]
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
/// Declares thread policy usage
pub enum ThreadingPolicy {
    /// Will use only one thread, current is preferred
    Single,
    /// Computes adaptive thread count between 1...12 for given image bounds
    Adaptive,
    /// Spawn provided threads count
    Fixed(usize),
}

impl ThreadingPolicy {
    pub fn get_threads_count(&self, width: u32, height: u32) -> usize {
        match self {
            ThreadingPolicy::Single => 1,
            ThreadingPolicy::Adaptive => {
                let thread_count =
                    std::cmp::max(std::cmp::min(width * height / (256 * 256), 12), 1);
                thread_count as usize
            }
            ThreadingPolicy::Fixed(fixed) => *fixed,
        }
    }
}
