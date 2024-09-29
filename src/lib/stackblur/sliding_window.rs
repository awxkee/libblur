/*
 * // Copyright (c) Radzivon Bartoshyk. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use crate::unsafe_slice::UnsafeSlice;
use num_traits::{AsPrimitive, FromPrimitive};
use std::ops::{AddAssign, Mul, Shr, Sub, SubAssign};

#[repr(C)]
#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Copy)]
pub(crate) struct SlidingWindow<const COMPS: usize, J: Copy> {
    pub r: J,
    pub g: J,
    pub b: J,
    pub a: J,
}

impl<const COMPS: usize, J> SlidingWindow<COMPS, J>
where
    J: Copy + Default,
{
    #[inline]
    pub fn new() -> SlidingWindow<COMPS, J> {
        SlidingWindow {
            r: J::default(),
            g: J::default(),
            b: J::default(),
            a: J::default(),
        }
    }

    #[inline]
    pub fn from_components(r: J, g: J, b: J, a: J) -> SlidingWindow<COMPS, J> {
        SlidingWindow { r, g, b, a }
    }

    #[inline]
    pub fn cast<T>(&self) -> SlidingWindow<COMPS, T>
    where
        J: AsPrimitive<T>,
        T: Default + Copy + 'static,
    {
        if COMPS == 1 {
            SlidingWindow::from_components(self.r.as_(), T::default(), T::default(), T::default())
        } else if COMPS == 2 {
            SlidingWindow::from_components(self.r.as_(), self.g.as_(), T::default(), T::default())
        } else if COMPS == 3 {
            SlidingWindow::from_components(self.r.as_(), self.g.as_(), self.b.as_(), T::default())
        } else if COMPS == 4 {
            SlidingWindow::from_components(self.r.as_(), self.g.as_(), self.b.as_(), self.a.as_())
        } else {
            panic!("Not implemented.");
        }
    }
}

impl<const COMPS: usize, J> SlidingWindow<COMPS, J>
where
    J: Copy + FromPrimitive + Default + 'static,
{
    #[inline]
    pub fn from_store<T>(store: &UnsafeSlice<T>, offset: usize) -> SlidingWindow<COMPS, J>
    where
        T: AsPrimitive<J>,
    {
        if COMPS == 1 {
            SlidingWindow {
                r: (*store.get(offset)).as_(),
                g: J::default(),
                b: J::default(),
                a: J::default(),
            }
        } else if COMPS == 2 {
            SlidingWindow {
                r: (*store.get(offset)).as_(),
                g: (*store.get(offset + 1)).as_(),
                b: J::default(),
                a: J::default(),
            }
        } else if COMPS == 3 {
            SlidingWindow {
                r: (*store.get(offset)).as_(),
                g: (*store.get(offset + 1)).as_(),
                b: (*store.get(offset + 2)).as_(),
                a: J::default(),
            }
        } else if COMPS == 4 {
            SlidingWindow {
                r: (*store.get(offset)).as_(),
                g: (*store.get(offset + 1)).as_(),
                b: (*store.get(offset + 2)).as_(),
                a: (*store.get(offset + 3)).as_(),
            }
        } else {
            panic!("Not implemented.")
        }
    }

    #[inline]
    pub fn to_store<T>(&self, store: &UnsafeSlice<T>, offset: usize)
    where
        J: AsPrimitive<T>, T: Copy + 'static
    {
        unsafe {
            store.write(offset, self.r.as_());
            if COMPS > 1 {
                store.write(offset + 1, self.g.as_());
            }
            if COMPS > 2 {
                store.write(offset + 2, self.b.as_());
            }
            if COMPS == 4 {
                store.write(offset + 3, self.a.as_());
            }
        }

    }
}

impl<const COMPS: usize, J> Mul<J> for SlidingWindow<COMPS, J>
where
    J: Copy + Mul<Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: J) -> Self::Output {
        if COMPS == 1 {
            SlidingWindow::from_components(self.r * rhs, self.g, self.b, self.a)
        } else if COMPS == 2 {
            SlidingWindow::from_components(self.r * rhs, self.g * rhs, self.b, self.a)
        } else if COMPS == 3 {
            SlidingWindow::from_components(self.r * rhs, self.g * rhs, self.b * rhs, self.a)
        } else if COMPS == 4 {
            SlidingWindow::from_components(self.r * rhs, self.g * rhs, self.b * rhs, self.a * rhs)
        } else {
            panic!("Not implemented.");
        }
    }
}

impl<const COMPS: usize, J> Sub<J> for SlidingWindow<COMPS, J>
where
    J: Copy + Sub<Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: J) -> Self::Output {
        if COMPS == 1 {
            SlidingWindow::from_components(self.r - rhs, self.g, self.b, self.a)
        } else if COMPS == 2 {
            SlidingWindow::from_components(self.r - rhs, self.g - rhs, self.b, self.a)
        } else if COMPS == 3 {
            SlidingWindow::from_components(self.r - rhs, self.g - rhs, self.b - rhs, self.a)
        } else if COMPS == 4 {
            SlidingWindow::from_components(self.r - rhs, self.g - rhs, self.b - rhs, self.a - rhs)
        } else {
            panic!("Not implemented.");
        }
    }
}

impl<const COMPS: usize, J> Sub<SlidingWindow<COMPS, J>> for SlidingWindow<COMPS, J>
where
    J: Copy + Sub<Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: SlidingWindow<COMPS, J>) -> Self::Output {
        if COMPS == 1 {
            SlidingWindow::from_components(self.r - rhs.r, self.g, self.b, self.a)
        } else if COMPS == 2 {
            SlidingWindow::from_components(self.r - rhs.r, self.g - rhs.g, self.b, self.a)
        } else if COMPS == 3 {
            SlidingWindow::from_components(self.r - rhs.r, self.g - rhs.g, self.b - rhs.b, self.a)
        } else if COMPS == 4 {
            SlidingWindow::from_components(
                self.r - rhs.r,
                self.g - rhs.g,
                self.b - rhs.b,
                self.a - rhs.a,
            )
        } else {
            panic!("Not implemented.");
        }
    }
}

impl<const COMPS: usize, J> Shr<J> for SlidingWindow<COMPS, J>
where
    J: Copy + Shr<J, Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline]
    fn shr(self, rhs: J) -> Self::Output {
        if COMPS == 1 {
            SlidingWindow::from_components(self.r >> rhs, self.g, self.b, self.a)
        } else if COMPS == 2 {
            SlidingWindow::from_components(self.r >> rhs, self.g >> rhs, self.b, self.a)
        } else if COMPS == 3 {
            SlidingWindow::from_components(self.r >> rhs, self.g >> rhs, self.b >> rhs, self.a)
        } else if COMPS == 4 {
            SlidingWindow::from_components(
                self.r >> rhs,
                self.g >> rhs,
                self.b >> rhs,
                self.a >> rhs,
            )
        } else {
            panic!("Not implemented.");
        }
    }
}

impl<const COMPS: usize, J> AddAssign<SlidingWindow<COMPS, J>> for SlidingWindow<COMPS, J>
where
    J: Copy + AddAssign,
{
    #[inline]
    fn add_assign(&mut self, rhs: SlidingWindow<COMPS, J>) {
        if COMPS == 1 {
            self.r += rhs.r;
        } else if COMPS == 2 {
            self.r += rhs.r;
            self.g += rhs.g;
        } else if COMPS == 3 {
            self.r += rhs.r;
            self.g += rhs.g;
            self.b += rhs.b;
        } else if COMPS == 4 {
            self.r += rhs.r;
            self.g += rhs.g;
            self.b += rhs.b;
            self.a += rhs.a;
        }
    }
}

impl<const COMPS: usize, J> SubAssign<SlidingWindow<COMPS, J>> for SlidingWindow<COMPS, J>
where
    J: Copy + SubAssign,
{
    #[inline]
    fn sub_assign(&mut self, rhs: SlidingWindow<COMPS, J>) {
        if COMPS == 1 {
            self.r -= rhs.r;
        } else if COMPS == 2 {
            self.r -= rhs.r;
            self.g -= rhs.g;
        } else if COMPS == 3 {
            self.r -= rhs.r;
            self.g -= rhs.g;
            self.b -= rhs.b;
        } else if COMPS == 4 {
            self.r -= rhs.r;
            self.g -= rhs.g;
            self.b -= rhs.b;
            self.a -= rhs.a;
        }
    }
}

impl<const COMPS: usize, J> Default for SlidingWindow<COMPS, J>
where
    J: Copy + FromPrimitive + Default,
{
    #[inline]
    fn default() -> Self {
        SlidingWindow::new()
    }
}
