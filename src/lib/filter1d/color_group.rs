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
use crate::filter1d::to_approx_storage::ToApproxStorage;
use crate::mlaf::mlaf;
use crate::to_storage::ToStorage;
use crate::unsafe_slice::UnsafeSlice;
use num_traits::{AsPrimitive, FromPrimitive, MulAdd};
use std::ops::{Add, AddAssign, Mul, Shr, Sub, SubAssign};

#[repr(C)]
#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Copy)]
pub(crate) struct ColorGroup<const COMPS: usize, J: Copy> {
    pub r: J,
    pub g: J,
    pub b: J,
    pub a: J,
}

impl<const COMPS: usize, J> ColorGroup<COMPS, J>
where
    J: Copy + Default,
{
    #[inline]
    pub fn new() -> ColorGroup<COMPS, J> {
        ColorGroup {
            r: J::default(),
            g: J::default(),
            b: J::default(),
            a: J::default(),
        }
    }

    #[inline]
    pub fn from_components(r: J, g: J, b: J, a: J) -> ColorGroup<COMPS, J> {
        ColorGroup { r, g, b, a }
    }
}

#[macro_export]
macro_rules! fast_load_color_group_with_offset {
    ($store: expr, $channels: expr, $offset: expr) => {{
        if $channels == 1 {
            ColorGroup {
                r: $store[$offset].as_(),
                g: 0.as_(),
                b: 0.as_(),
                a: 0.as_(),
            }
        } else if $channels == 2 {
            ColorGroup {
                r: $store[$offset].as_(),
                g: $store[$offset + 1].as_(),
                b: 0.as_(),
                a: 0.as_(),
            }
        } else if $channels == 3 {
            ColorGroup {
                r: $store[$offset].as_(),
                g: $store[$offset + 1].as_(),
                b: $store[$offset + 2].as_(),
                a: 0.as_(),
            }
        } else if $channels == 4 {
            ColorGroup {
                r: $store[$offset].as_(),
                g: $store[$offset + 1].as_(),
                b: $store[$offset + 2].as_(),
                a: $store[$offset + 3].as_(),
            }
        } else {
            panic!("Not implemented.")
        }
    }};
}

impl<const COMPS: usize, J> ColorGroup<COMPS, J>
where
    J: Copy + Default + 'static,
{
    #[inline]
    pub fn from_slice<T>(store: &[T], offset: usize) -> ColorGroup<COMPS, J>
    where
        T: AsPrimitive<J>,
    {
        unsafe {
            if COMPS == 1 {
                ColorGroup {
                    r: (*store.get_unchecked(offset)).as_(),
                    g: J::default(),
                    b: J::default(),
                    a: J::default(),
                }
            } else if COMPS == 2 {
                ColorGroup {
                    r: (*store.get_unchecked(offset)).as_(),
                    g: (*store.get_unchecked(offset + 1)).as_(),
                    b: J::default(),
                    a: J::default(),
                }
            } else if COMPS == 3 {
                ColorGroup {
                    r: (*store.get_unchecked(offset)).as_(),
                    g: (*store.get_unchecked(offset + 1)).as_(),
                    b: (*store.get_unchecked(offset + 2)).as_(),
                    a: J::default(),
                }
            } else if COMPS == 4 {
                ColorGroup {
                    r: (*store.get_unchecked(offset)).as_(),
                    g: (*store.get_unchecked(offset + 1)).as_(),
                    b: (*store.get_unchecked(offset + 2)).as_(),
                    a: (*store.get_unchecked(offset + 3)).as_(),
                }
            } else {
                panic!("Not implemented.")
            }
        }
    }

    #[inline]
    pub fn to_approx_store<T>(self, store: &UnsafeSlice<T>, offset: usize)
    where
        J: ToApproxStorage<T>,
        T: Copy + 'static,
    {
        unsafe {
            store.write(offset, self.r.to_approx_());
            if COMPS > 1 {
                store.write(offset + 1, self.g.to_approx_());
            }
            if COMPS > 2 {
                store.write(offset + 2, self.b.to_approx_());
            }
            if COMPS == 4 {
                store.write(offset + 3, self.a.to_approx_());
            }
        }
    }

    #[inline]
    pub fn to_store<T>(self, store: &UnsafeSlice<T>, offset: usize)
    where
        J: ToStorage<T>,
        T: Copy + 'static,
    {
        unsafe {
            store.write(offset, self.r.to_());
            if COMPS > 1 {
                store.write(offset + 1, self.g.to_());
            }
            if COMPS > 2 {
                store.write(offset + 2, self.b.to_());
            }
            if COMPS == 4 {
                store.write(offset + 3, self.a.to_());
            }
        }
    }
}

impl<const COMPS: usize, J> Mul<J> for ColorGroup<COMPS, J>
where
    J: Copy + Mul<Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: J) -> Self::Output {
        if COMPS == 1 {
            ColorGroup::from_components(self.r * rhs, self.g, self.b, self.a)
        } else if COMPS == 2 {
            ColorGroup::from_components(self.r * rhs, self.g * rhs, self.b, self.a)
        } else if COMPS == 3 {
            ColorGroup::from_components(self.r * rhs, self.g * rhs, self.b * rhs, self.a)
        } else if COMPS == 4 {
            ColorGroup::from_components(self.r * rhs, self.g * rhs, self.b * rhs, self.a * rhs)
        } else {
            panic!("Not implemented.");
        }
    }
}

impl<const COMPS: usize, J> Mul<ColorGroup<COMPS, J>> for ColorGroup<COMPS, J>
where
    J: Copy + Mul<Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: ColorGroup<COMPS, J>) -> Self::Output {
        if COMPS == 1 {
            ColorGroup::from_components(self.r * rhs.r, self.g, self.b, self.a)
        } else if COMPS == 2 {
            ColorGroup::from_components(self.r * rhs.r, self.g * rhs.g, self.b, self.a)
        } else if COMPS == 3 {
            ColorGroup::from_components(self.r * rhs.r, self.g * rhs.g, self.b * rhs.b, self.a)
        } else if COMPS == 4 {
            ColorGroup::from_components(
                self.r * rhs.r,
                self.g * rhs.g,
                self.b * rhs.b,
                self.a * rhs.b,
            )
        } else {
            panic!("Not implemented.");
        }
    }
}

impl<const COMPS: usize, J> Sub<J> for ColorGroup<COMPS, J>
where
    J: Copy + Sub<Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: J) -> Self::Output {
        if COMPS == 1 {
            ColorGroup::from_components(self.r - rhs, self.g, self.b, self.a)
        } else if COMPS == 2 {
            ColorGroup::from_components(self.r - rhs, self.g - rhs, self.b, self.a)
        } else if COMPS == 3 {
            ColorGroup::from_components(self.r - rhs, self.g - rhs, self.b - rhs, self.a)
        } else if COMPS == 4 {
            ColorGroup::from_components(self.r - rhs, self.g - rhs, self.b - rhs, self.a - rhs)
        } else {
            panic!("Not implemented.");
        }
    }
}

impl<const COMPS: usize, J> Sub<ColorGroup<COMPS, J>> for ColorGroup<COMPS, J>
where
    J: Copy + Sub<Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: ColorGroup<COMPS, J>) -> Self::Output {
        if COMPS == 1 {
            ColorGroup::from_components(self.r - rhs.r, self.g, self.b, self.a)
        } else if COMPS == 2 {
            ColorGroup::from_components(self.r - rhs.r, self.g - rhs.g, self.b, self.a)
        } else if COMPS == 3 {
            ColorGroup::from_components(self.r - rhs.r, self.g - rhs.g, self.b - rhs.b, self.a)
        } else if COMPS == 4 {
            ColorGroup::from_components(
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

impl<const COMPS: usize, J> Add<ColorGroup<COMPS, J>> for ColorGroup<COMPS, J>
where
    J: Copy + Add<Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: ColorGroup<COMPS, J>) -> Self::Output {
        if COMPS == 1 {
            ColorGroup::from_components(self.r + rhs.r, self.g, self.b, self.a)
        } else if COMPS == 2 {
            ColorGroup::from_components(self.r + rhs.r, self.g + rhs.g, self.b, self.a)
        } else if COMPS == 3 {
            ColorGroup::from_components(self.r + rhs.r, self.g + rhs.g, self.b + rhs.b, self.a)
        } else if COMPS == 4 {
            ColorGroup::from_components(
                self.r + rhs.r,
                self.g + rhs.g,
                self.b + rhs.b,
                self.a + rhs.a,
            )
        } else {
            panic!("Not implemented.");
        }
    }
}

impl<const COMPS: usize, J> Add<J> for ColorGroup<COMPS, J>
where
    J: Copy + Add<Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: J) -> Self::Output {
        if COMPS == 1 {
            ColorGroup::from_components(self.r + rhs, self.g, self.b, self.a)
        } else if COMPS == 2 {
            ColorGroup::from_components(self.r + rhs, self.g + rhs, self.b, self.a)
        } else if COMPS == 3 {
            ColorGroup::from_components(self.r + rhs, self.g + rhs, self.b + rhs, self.a)
        } else if COMPS == 4 {
            ColorGroup::from_components(self.r + rhs, self.g + rhs, self.b + rhs, self.a + rhs)
        } else {
            panic!("Not implemented.");
        }
    }
}

impl<const COMPS: usize, J> Shr<J> for ColorGroup<COMPS, J>
where
    J: Copy + Shr<J, Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline]
    fn shr(self, rhs: J) -> Self::Output {
        if COMPS == 1 {
            ColorGroup::from_components(self.r >> rhs, self.g, self.b, self.a)
        } else if COMPS == 2 {
            ColorGroup::from_components(self.r >> rhs, self.g >> rhs, self.b, self.a)
        } else if COMPS == 3 {
            ColorGroup::from_components(self.r >> rhs, self.g >> rhs, self.b >> rhs, self.a)
        } else if COMPS == 4 {
            ColorGroup::from_components(self.r >> rhs, self.g >> rhs, self.b >> rhs, self.a >> rhs)
        } else {
            panic!("Not implemented.");
        }
    }
}

impl<const COMPS: usize, J> MulAdd<ColorGroup<COMPS, J>, J> for ColorGroup<COMPS, J>
where
    J: Copy + MulAdd<J, Output = J> + Default + 'static + Mul<J, Output = J> + Add<J, Output = J>,
{
    type Output = Self;

    #[inline]
    fn mul_add(self, a: ColorGroup<COMPS, J>, b: J) -> Self::Output {
        if COMPS == 1 {
            ColorGroup::from_components(mlaf(a.r, self.r, b), self.g, self.b, self.a)
        } else if COMPS == 2 {
            ColorGroup::from_components(mlaf(a.r, self.r, b), mlaf(a.g, self.g, b), self.b, self.a)
        } else if COMPS == 3 {
            ColorGroup::from_components(
                mlaf(a.r, self.r, b),
                mlaf(a.g, self.g, b),
                mlaf(a.b, self.b, b),
                self.a,
            )
        } else if COMPS == 4 {
            ColorGroup::from_components(
                mlaf(a.r, self.r, b),
                mlaf(a.g, self.g, b),
                mlaf(a.b, self.b, b),
                mlaf(a.a, self.a, b),
            )
        } else {
            panic!("Not implemented.");
        }
    }
}

impl<const COMPS: usize, J> AddAssign<ColorGroup<COMPS, J>> for ColorGroup<COMPS, J>
where
    J: Copy + AddAssign,
{
    #[inline]
    fn add_assign(&mut self, rhs: ColorGroup<COMPS, J>) {
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

impl<const COMPS: usize, J> SubAssign<ColorGroup<COMPS, J>> for ColorGroup<COMPS, J>
where
    J: Copy + SubAssign,
{
    #[inline]
    fn sub_assign(&mut self, rhs: ColorGroup<COMPS, J>) {
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

impl<const COMPS: usize, J> Default for ColorGroup<COMPS, J>
where
    J: Copy + FromPrimitive + Default,
{
    #[inline]
    fn default() -> Self {
        ColorGroup::new()
    }
}
