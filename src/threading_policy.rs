// Copyright (c) Radzivon Bartoshyk. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1.  Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2.  Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3.  Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use std::{num::NonZeroUsize, thread::available_parallelism};

#[repr(C)]
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Default, Hash)]
/// Set threading policy.
pub enum ThreadingPolicy {
    /// Use only one thread, current is preferred.
    Single,
    /// Compute adaptive thread count between 1..available CPUs.
    #[default]
    Adaptive,
    /// Like `Adaptive`, but reserve given amount of threads (i.e. those will not be
    /// used).
    AdaptiveReserve(NonZeroUsize),
    /// Use specified number of threads.
    Fixed(NonZeroUsize),
}

impl ThreadingPolicy {
    /// Returns the number of threads to use for the given image dimensions under the
    /// selected policy variant.
    ///
    /// Must return at least 1.
    pub fn thread_count(&self, width: u32, height: u32) -> usize {
        match self {
            ThreadingPolicy::Single => 1,
            ThreadingPolicy::Adaptive => {
                ((width * height / (256 * 256)) as usize).clamp(1, Self::available_parallelism(2))
            }
            ThreadingPolicy::AdaptiveReserve(reserve) => {
                let reserve = reserve.get();

                let max_threads = {
                    let max_threads = Self::available_parallelism(1);

                    if max_threads <= reserve {
                        1
                    } else {
                        max_threads
                    }
                };

                ((width * height / (256 * 256)) as usize)
                    .clamp(1, max_threads.min(max_threads - reserve))
            }
            ThreadingPolicy::Fixed(fixed) => fixed.get(),
        }
    }

    // Make always return at least some minimal amount of threads, if multi-threading were requested
    // At least on single core CPU have 2 threads is beneficial
    fn available_parallelism(min: usize) -> usize {
        available_parallelism()
            .unwrap_or_else(|_| NonZeroUsize::new(1).unwrap())
            .get()
            .max(min)
    }
}
