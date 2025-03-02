use std::iter::zip;
use std::ops::{Add, AddAssign, Deref, DerefMut, Index, IndexMut, Sub, SubAssign};
use static_assertions::{const_assert, const_assert_eq};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec<const DIM: usize, T>(pub [T; DIM]);

//specialization for usual Vec

// ---- 2D ---- //
pub type Vec2<T> = Vec<2, T>;

#[repr(C)]
pub struct Window2<T> {
    x: T,
    y: T,
}


impl<T> Vec2<T> {
    pub fn new(x: T, y: T) -> Self {
        Self ([x, y])
    }
}

impl<T> Deref for Vec2<T> {
    type Target = Window2<T>;

    fn deref(&self) -> &Self::Target {
        assert_eq!(size_of::<Vec2<T>>(), size_of::<Window2<T>>());
        union Transform2<'a, T> {
            src: &'a Vec2<T>,
            dst: &'a Window2<T>,
        }


        let cast = Transform2 { src: self };
        unsafe { cast.dst } //should we be concerned about this?
    }
}

impl<T> DerefMut for Vec2<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        assert_eq!(size_of::<Vec2<T>>(), size_of::<Window2<T>>());
        union Transform2<'a, T> {
            src: &'a mut Vec2<T>,
            dst: &'a mut Window2<T>,
        }
        let cast = Transform2 { src: self };
        unsafe { cast.dst } // SAFETY: repr(C) guarantees that the fields are in the same order
    }
}

// -- 3 -- //
pub type Vec3<T> = Vec<3, T>;

#[repr(C)]
pub struct Window3<T> {
    x: T,
    y: T,
    z: T,
}

impl<T> Vec3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Self ([x, y, z])
    }
}

impl<T> Deref for Vec3<T> {
    type Target = Window3<T>;
    fn deref(&self) -> &Self::Target {
        union Transform3<'a, T> {
            src: &'a Vec3<T>,
            dst: &'a Window3<T>,
        }
        let cast = Transform3 { src: self };
        unsafe { cast.dst } // SAFETY: repr(C) guarantees that the fields are in the same order
    }
}

impl<T> DerefMut for Vec3<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        union Transform3<'a, T> {
            src: &'a mut Vec3<T>,
            dst: &'a mut Window3<T>,
        }
        let cast = Transform3 { src: self };
        unsafe { cast.dst } //should we be concerned about this?
    }
}

//generic case
impl<T, const D: usize> Index<usize> for Vec<D, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T, const D: usize> IndexMut<usize> for Vec<D, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T, const D: usize> From<[T; D]> for Vec<D, T> {
    fn from(x: [T; D]) -> Self {
        Self(x)
    }
}

impl<T, const D: usize> From<Vec<D, T>> for [T; D] {
    fn from(x: Vec<D, T>) -> Self {
        x.0
    }
}


// heart of most of the operations on Vec
// TODO: Find faster way in debug mode to merge two statics arrays
impl<T, const D: usize> Vec<D, T> {
    fn combine<U, R>(self, other: Vec<D, U>, f: impl Fn(T, U) -> R) -> Vec<D, R> {
        let a = self.0.into_iter();
        let b = other.0.into_iter();
        let mut iter = zip(a, b).map(|(a, b)| f(a, b));
        Vec(std::array::from_fn(|_| unsafe { iter.next().unwrap_unchecked() }))
    }

    fn combine_ref<U, R>(self, other: &Vec<D, U>, f: impl Fn(T, &U) -> R) -> Vec<D, R> {
        let a = self.0.into_iter();
        let b = other.0.iter();
        let mut iter = zip(a, b).map(|(a, b)| f(a, b));
        Vec(std::array::from_fn(|_| unsafe { iter.next().unwrap_unchecked() }))
    }

    fn combine_both_ref<U, R>(&self, other: &Vec<D, U>, f: impl Fn(&T, &U) -> R) -> Vec<D, R> {
        let a = self.0.iter();
        let b = other.0.iter();
        let mut iter = zip(a, b).map(|(a, b)| f(a, b));
        Vec(std::array::from_fn(|_| unsafe { iter.next().unwrap_unchecked() }))
    }

    fn combine_assigne<U>(&mut self, other: Vec<D, U>, f: impl Fn(&mut T, U)) {
        let mut b = other.0.into_iter();
        for a in self.0.iter_mut() {
            unsafe { f(a, b.next().unwrap_unchecked()) }
        }
    }

    fn combine_assigne_ref<U>(&mut self, other: &Vec<D, U>, f: impl Fn(&mut T, &U)) {
        let mut b = other.0.iter();
        for a in self.0.iter_mut() {
            unsafe { f(a, b.next().unwrap_unchecked()) }
        }
    }
}

// addition
impl<T, U, R, const D: usize> Add<Vec<D, U>> for Vec<D, T>
where
    T: Add<U, Output = R>,
{
    type Output = Vec<D, R>;
    fn add(self, rhs: Vec<D, U>) -> Self::Output {
        self.combine(rhs, T::add)
    }
}

impl<T, U, R, const D: usize> Add<&Vec<D, U>> for Vec<D, T>
where
    T: for<'a> Add<&'a U, Output = R>,
{
    type Output = Vec<D, R>;
    fn add(self, rhs: &Vec<D, U>) -> Self::Output {
        self.combine_ref(rhs, | a, b | a + b)
    }
}

impl<T, U, R, const D: usize> Add<Vec<D, U>> for &Vec<D, T>
where
        for<'a> &'a T: Add<U, Output = R>,
{
    type Output = Vec<D, R>;
    fn add(self, rhs: Vec<D, U>) -> Self::Output {
        rhs.combine_ref(self, |a, b | b + a) // swap arguments because we only have one implementation of combine_ref
    }
}

impl<T, U, R, const D: usize> Add<&Vec<D, U>> for &Vec<D, T>
where
    for<'a, 'b> &'a T: Add<&'b U, Output = R>,
{
    type Output = Vec<D, R>;
    fn add(self, rhs: &Vec<D, U>) -> Self::Output {
        self.combine_both_ref(rhs, | a, b | a + b)
    }
}

impl <T, U, const D: usize> AddAssign<Vec<D, U>> for Vec<D, T>
where
    T: AddAssign<U>,
{
    fn add_assign(&mut self, rhs: Vec<D, U>) {
        self.combine_assigne(rhs, |mut a, b | a.add_assign(b));
    }
}

impl <T, U, const D: usize> AddAssign<&Vec<D, U>> for Vec<D, T>
where
    T: for<'a> AddAssign<&'a U>,
{
    fn add_assign(&mut self, rhs: &Vec<D, U>) {
        self.combine_assigne_ref(rhs, |a, b | a.add_assign(b));
    }
}

// subtraction
impl<T, U, R, const D: usize> Sub<Vec<D, U>> for Vec<D, T>
where
    T: Sub<U, Output = R>,
{
    type Output = Vec<D, R>;
    fn sub(self, rhs: Vec<D, U>) -> Self::Output {
        self.combine(rhs, |a, b| a - b)
    }
}

impl<T, U, R, const D: usize> Sub<&Vec<D, U>> for Vec<D, T>
where
    T: for<'a> Sub<&'a U, Output = R>,
{
    type Output = Vec<D, R>;
    fn sub(self, rhs: &Vec<D, U>) -> Self::Output {
        self.combine_ref(rhs, |a, b| a - b)
    }
}

impl<T, U, R, const D: usize> Sub<Vec<D, U>> for &Vec<D, T>
where
    for<'a> &'a T: Sub<U, Output = R>,
{
    type Output = Vec<D, R>;
    fn sub(self, rhs: Vec<D, U>) -> Self::Output {
        rhs.combine_ref(self, |a, b| b - a)
    }
}

impl<T, U, R, const D: usize> Sub<&Vec<D, U>> for &Vec<D, T>
where
    for<'a, 'b> &'a T: Sub<&'b U, Output = R>,
{
    type Output = Vec<D, R>;
    fn sub(self, rhs: &Vec<D, U>) -> Self::Output {
        self.combine_both_ref(rhs, |a, b| a - b)
    }
}

impl<T, U, const D: usize> SubAssign<Vec<D, U>> for Vec<D, T>
where
    T: SubAssign<U>,
{
    fn sub_assign(&mut self, rhs: Vec<D, U>) {
        self.combine_assigne(rhs, |mut a, b| a.sub_assign(b));
    }
}

impl<T, U, const D: usize> SubAssign<&Vec<D, U>> for Vec<D, T>
where
    T: for<'a> SubAssign<&'a U>,
{
    fn sub_assign(&mut self, rhs: &Vec<D, U>) {
        self.combine_assigne_ref(rhs, |a, b| a.sub_assign(b));
    }
}

// This is two view how add is optimized by the compiler
// you're likely looking for ``cargo asm --rust isochro::vector::add``
pub fn add(a: Vec2<i32>, b: Vec2<i32>) -> Vec2<i32> {
    a + b
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec2() {
        let a = Vec2::new(1, 2);
        let b = Vec2::new(3, 4);
        let _ = a + b;
        let _ = &a + b;
        let c = a + &b;
        let _d = a + &b;
        let mut e = a;
        e -= b;
        assert_eq!(c.x, 4);
        assert_eq!(c.y, 6);
    }

    #[test]
    fn test_vec_access() {
        let vec = Vec3::from([1, 2, 3]);
        assert_eq!(vec[0], vec.x);
        assert_eq!(vec[1], vec.y);
        assert_eq!(vec[2], vec.z);
    }
}
