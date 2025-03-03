//! A generic vector type with compile-time dimensionality.
//! This type is a wrapper around a fixed-size array, and provides
//! a number of convenience methods for working with vectors.
//! The dimensionality of the vector is specified as a type parameter.
//! This allows the compiler to catch errors where vectors of different
//! sizes are used incorrectly.
//! # Examples
//! ```
//! use isochro::vector::Vec3;
//! let a = Vec3::new(1.0, 2.0, 3.0);
//! let b = Vec3::new(4.0, 5.0, 6.0);
//! let c = a + b;
//! assert_eq!(c.x, 5.0);
//! assert_eq!(c.y, 7.0);
//! assert_eq!(c.z, 9.0);
//! ```

mod vec2;
mod vec3;
mod vec4;

use core::ops::{Div, DivAssign, Mul, MulAssign};
use std::iter::zip;
use std::ops::{Add, AddAssign, Index, IndexMut, Sub, SubAssign};

pub use vec2::*;
pub use vec3::*;
pub use vec4::*;


/// Like the standard std::ops::* for the dot product.
pub trait DotProduct<Rhs = Self> {
    type Output;

    fn dot(self, other: Rhs) -> Self::Output;
}

/// A generic vector type with compile-time dimensionality.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec<const DIM: usize, T>(pub [T; DIM]);

//generic case
impl<T, const D: usize> Index<usize> for Vec<D, T> {
    type Output = T;

    /// Get the value at the given index.
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T, const D: usize> IndexMut<usize> for Vec<D, T> {
    /// Get a mutable reference to the value at the given index.
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
    /// Create a new vector from two other vectors and a combining function.
    ///
    /// # Example
    /// ```
    /// use isochro::vector::Vec2;
    /// let a = Vec2::new(1, 2);
    /// let b = Vec2::new(3, 4);
    /// let c = Vec2::combine(a, b, |a, b| a.min(b));
    /// assert_eq!(c.x, 1);
    /// assert_eq!(c.y, 2);
    /// ```
    pub fn combine<U, R>(self, other: Vec<D, U>, f: impl Fn(T, U) -> R) -> Vec<D, R> {
        let a = self.0.into_iter();
        let b = other.0.into_iter();
        let mut iter = zip(a, b).map(|(a, b)| f(a, b));

        Vec(std::array::from_fn(|_| unsafe {
            iter.next().unwrap_unchecked()
        }))
    }

    /// Create a new vector from a scalar and a combining function.
    ///
    /// # Example
    /// ```
    /// use isochro::vector::Vec2;
    /// let a = Vec2::new(1, 2);
    /// let b = Vec2::combine_scalar(a, 2, |a, b| a * b);
    /// assert_eq!(b.x, 2);
    /// assert_eq!(b.y, 4);
    /// ```
    pub fn combine_scalar<U: Clone, R>(self, other: U, f: impl Fn(T, U) -> R) -> Vec<D, R> {
        Vec(self.0.map(|a| f(a, other.clone())))
    }

    /// Create a new vector from two other vectors and a combining function.
    ///
    /// # Example
    /// ```
    /// use isochro::vector::Vec2;
    /// let a = Vec2::new(1, 2);
    /// let b = Vec2::new(3, 4);
    /// let c = Vec2::combine_ref(a, &b, |a, b| a.min(*b)); // b is a reference
    /// assert_eq!(c.x, 1);
    /// assert_eq!(c.y, 2);
    /// ```
    pub fn combine_ref<U, R>(self, other: &Vec<D, U>, f: impl Fn(T, &U) -> R) -> Vec<D, R> {
        let a = self.0.into_iter();
        let b = other.0.iter();
        let mut iter = zip(a, b).map(|(a, b)| f(a, b));

        Vec(std::array::from_fn(|_| unsafe {
            iter.next().unwrap_unchecked()
        }))
    }

    /// Create a new vector from a scalar and a combining function.
    ///
    /// # Example
    /// ```
    /// use isochro::vector::Vec2;
    /// let a = Vec2::new(1, 2);
    /// let b = Vec2::combine_scalar_ref(&a, &2, |a, b| a * *b);
    /// assert_eq!(b.x, 2);
    /// assert_eq!(b.y, 4);
    /// ```
    pub fn combine_scalar_ref<U: Copy, R>(&self, other: U, f: impl Fn(&T, U) -> R) -> Vec<D, R> {
        Vec(self.0.each_ref().map(|a| f(a, other)))
    }

    /// Create a new vector from two other vectors and a combining function.
    ///
    /// # Example
    /// ```
    /// use isochro::vector::Vec2;
    /// let a = Vec2::new(1, 2);
    /// let b = Vec2::new(3, 4);
    /// let c = Vec2::combine_both_ref(&a, &b, |a, b| *a.min(b)); // a and b are references
    /// assert_eq!(c, (1, 2));
    /// ```
    pub fn combine_both_ref<U, R>(&self, other: &Vec<D, U>, f: impl Fn(&T, &U) -> R) -> Vec<D, R> {
        let a = self.0.iter();
        let b = other.0.iter();
        let mut iter = zip(a, b).map(|(a, b)| f(a, b));

        Vec(std::array::from_fn(|_| unsafe {
            iter.next().unwrap_unchecked()
        }))
    }

    /// Create a new vector from two other vectors and a combining function.
    /// # Example
    /// ```
    /// use isochro::vector::Vec2;
    /// let mut a = Vec2::new(1, 2);
    /// let b = Vec2::new(3, 4);
    /// a.combine_assign(b, |mut a, b| *a += b);
    /// assert_eq!(a.x, 4);
    /// assert_eq!(a.y, 6);
    /// ```
    pub fn combine_assign<U>(&mut self, other: Vec<D, U>, f: impl Fn(&mut T, U)) {
        let mut b = other.0.into_iter();
        for a in self.0.iter_mut() {
            unsafe { f(a, b.next().unwrap_unchecked()) }
        }
    }

    /// Create a new vector from a scalar and a combining function.
    /// # Example
    /// ```
    /// use isochro::vector::Vec2;
    /// let mut a = Vec2::new(1, 2);
    /// a.combine_assign_scalar(2, |a, b| *a = *a * b);
    /// assert_eq!(a.x, 2);
    /// assert_eq!(a.y, 4);
    /// ```
    pub fn combine_assign_scalar<U: Clone>(&mut self, other: U, f: impl Fn(&mut T, U)) {
        for a in self.0.iter_mut() {
            f(a, other.clone());
        }
    }

    /// Create a new vector from two other vectors and a combining function.
    /// # Example
    /// ```
    /// use isochro::vector::Vec2;
    /// let mut a = Vec2::new(1, 2);
    /// let b = Vec2::new(3, 4);
    /// a.combine_assign_ref(&b, |mut a, b| *a += b);
    /// assert_eq!(a.x, 4);
    /// assert_eq!(a.y, 6);
    /// ```
    pub fn combine_assign_ref<U>(&mut self, other: &Vec<D, U>, f: impl Fn(&mut T, &U)) {
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

    /// Add two vectors together.
    /// # Example
    /// ```
    /// use isochro::vector::Vec2;
    /// let a = Vec2::new(1, 2);
    /// let b = Vec2::new(3, 4);
    /// let c = a + b;
    /// assert_eq!(c.x, 4);
    /// assert_eq!(c.y, 6);
    /// ```
    fn add(self, rhs: Vec<D, U>) -> Self::Output {
        self.combine(rhs, T::add)
    }
}

impl<T, U, R, const D: usize> Add<&Vec<D, U>> for Vec<D, T>
where
    T: for<'a> Add<&'a U, Output = R>,
{
    type Output = Vec<D, R>;

    /// Add two vectors together.
    ///
    /// # Example
    /// ```
    /// use isochro::vector::Vec2;
    /// let a = Vec2::new(1, 2);
    /// let b = Vec2::new(3, 4);
    /// let c = a + &b;
    /// assert_eq!(c.x, 4);
    /// assert_eq!(c.y, 6);
    /// ```
    fn add(self, rhs: &Vec<D, U>) -> Self::Output {
        self.combine_ref(rhs, |a, b| a + b)
    }
}

impl<T, U, R, const D: usize> Add<Vec<D, U>> for &Vec<D, T>
where
    for<'a> &'a T: Add<U, Output = R>,
{
    type Output = Vec<D, R>;

    /// Add two vectors together.
    ///
    /// # Example
    /// ```
    /// use isochro::vector::Vec2;
    /// let a = Vec2::new(1, 2);
    /// let b = Vec2::new(3, 4);
    /// let c = &a + b;
    /// assert_eq!(c.x, 4);
    /// assert_eq!(c.y, 6);
    /// ```
    fn add(self, rhs: Vec<D, U>) -> Self::Output {
        rhs.combine_ref(self, |a, b| b + a) // swap arguments because we only have one implementation of combine_ref
    }
}

impl<T, U, R, const D: usize> Add<&Vec<D, U>> for &Vec<D, T>
where
    for<'a, 'b> &'a T: Add<&'b U, Output = R>,
{
    type Output = Vec<D, R>;

    /// Add two vectors together.
    ///
    /// # Example
    /// ```
    /// use isochro::vector::Vec2;
    /// let a = Vec2::new(1, 2);
    /// let b = Vec2::new(3, 4);
    /// let c = &a + &b;
    /// assert_eq!(c.x, 4);
    /// assert_eq!(c.y, 6);
    /// ```
    fn add(self, rhs: &Vec<D, U>) -> Self::Output {
        self.combine_both_ref(rhs, |a, b| a + b)
    }
}

impl<T, U, const D: usize> AddAssign<Vec<D, U>> for Vec<D, T>
where
    T: AddAssign<U>,
{

    /// Add a vectors to another.
    ///
    /// # Example
    /// ```
    /// use isochro::vector::Vec2;
    /// let mut a = Vec2::new(1, 2);
    /// let b = Vec2::new(3, 4);
    /// a += b;
    /// assert_eq!(a, (4, 6));
    /// ```
    fn add_assign(&mut self, rhs: Vec<D, U>) {
        self.combine_assign(rhs, |a, b| a.add_assign(b));
    }
}

impl<T, U, const D: usize> AddAssign<&Vec<D, U>> for Vec<D, T>
where
    T: for<'a> AddAssign<&'a U>,
{
    /// Add a vector to another.
    ///
    /// # Example
    /// ```
    /// use isochro::vector::Vec2;
    /// let mut a = Vec2::new(1, 2);
    /// let b = Vec2::new(3, 4);
    /// a += &b;
    /// assert_eq!(a, (4, 6));
    /// ```
    fn add_assign(&mut self, rhs: &Vec<D, U>) {
        self.combine_assign_ref(rhs, |a, b| a.add_assign(b));
    }
}

// subtraction
impl<T, U, R, const D: usize> Sub<Vec<D, U>> for Vec<D, T>
where
    T: Sub<U, Output = R>,
{
    type Output = Vec<D, R>;

    /// Subtract one vector from another.
    ///
    /// # Example
    /// ```
    /// use isochro::vector::Vec2;
    /// let a = Vec2::new(1, 2);
    /// let b = Vec2::new(3, 4);
    /// let c = a - b;
    /// assert_eq!(c.x, -2);
    /// assert_eq!(c.y, -2);
    /// ```
    fn sub(self, rhs: Vec<D, U>) -> Self::Output {
        self.combine(rhs, |a, b| a - b)
    }
}

impl<T, U, R, const D: usize> Sub<&Vec<D, U>> for Vec<D, T>
where
    T: for<'a> Sub<&'a U, Output = R>,
{
    type Output = Vec<D, R>;

    /// Subtract one vector from another.
    ///
    /// # Example
    /// ```
    /// use isochro::vector::Vec2;
    /// let a = Vec2::new(1, 2);
    /// let b = Vec2::new(3, 4);
    /// let c = a - &b;
    /// assert_eq!(c.x, -2);
    /// assert_eq!(c.y, -2);
    /// ```
    fn sub(self, rhs: &Vec<D, U>) -> Self::Output {
        self.combine_ref(rhs, |a, b| a - b)
    }
}

impl<T, U, R, const D: usize> Sub<Vec<D, U>> for &Vec<D, T>
where
    for<'a> &'a T: Sub<U, Output = R>,
{
    type Output = Vec<D, R>;

    /// Subtract one vector from another.
    ///
    /// # Example
    /// ```
    /// use isochro::vector::Vec2;
    /// let a = Vec2::new(1, 2);
    /// let b = Vec2::new(3, 4);
    /// let c = &a - b;
    /// assert_eq!(c.x, -2);
    /// assert_eq!(c.y, -2);
    /// ```
    fn sub(self, rhs: Vec<D, U>) -> Self::Output {
        rhs.combine_ref(self, |a, b| b - a)
    }
}

impl<T, U, R, const D: usize> Sub<&Vec<D, U>> for &Vec<D, T>
where
    for<'a, 'b> &'a T: Sub<&'b U, Output = R>,
{
    type Output = Vec<D, R>;

    /// Subtract one vector from another.
    ///
    /// # Example
    /// ```
    /// use isochro::vector::Vec2;
    /// let a = Vec2::new(1, 2);
    /// let b = Vec2::new(3, 4);
    /// let c = &a - &b;
    /// assert_eq!(c.x, -2);
    /// assert_eq!(c.y, -2);
    /// ```
    fn sub(self, rhs: &Vec<D, U>) -> Self::Output {
        self.combine_both_ref(rhs, |a, b| a - b)
    }
}

impl<T, U, const D: usize> SubAssign<Vec<D, U>> for Vec<D, T>
where
    T: SubAssign<U>,
{
    /// Subtract a vector from another.
    ///
    /// # Example
    /// ```
    /// use isochro::vector::Vec2;
    /// let mut a = Vec2::new(1, 2);
    /// let b = Vec2::new(3, 4);
    /// a -= b;
    /// assert_eq!(a, (-2, -2));
    fn sub_assign(&mut self, rhs: Vec<D, U>) {
        self.combine_assign(rhs, |a, b| a.sub_assign(b));
    }
}

impl<T, U, const D: usize> SubAssign<&Vec<D, U>> for Vec<D, T>
where
    T: for<'a> SubAssign<&'a U>,
{
    /// Subtract a vector from another.
    ///
    /// # Example
    /// ```
    /// use isochro::vector::Vec2;
    /// let mut a = Vec2::new(1, 2);
    /// let b = Vec2::new(3, 4);
    /// a -= &b;
    /// assert_eq!(a, (-2, -2));
    /// ```
    fn sub_assign(&mut self, rhs: &Vec<D, U>) {
        self.combine_assign_ref(rhs, |a, b| a.sub_assign(b));
    }
}

// multiplication
impl<T, U, R, const D: usize> Mul<U> for Vec<D, T>
where
    T: Mul<U, Output = R>,
    U: Clone,
{
    type Output = Vec<D, R>;

    /// Multiply a vector by a scalar.
    ///
    /// # Example
    /// ```
    /// use isochro::vector::Vec2;
    /// let a = Vec2::new(1, 2);
    /// let b = a * 2;
    /// assert_eq!(b.x, 2);
    /// assert_eq!(b.y, 4);
    /// ```
    fn mul(self, rhs: U) -> Self::Output {
        self.combine_scalar(rhs, T::mul)
    }
}

impl<T, U, R, const D: usize> Mul<U> for &Vec<D, T>
where
    for<'a> &'a T: Mul<U, Output = R>,
    U: Copy,
{
    type Output = Vec<D, R>;

    /// Multiply a vector by a scalar.
    ///
    /// # Example
    /// ```
    /// use isochro::vector::Vec2;
    /// let a = Vec2::new(1, 2);
    /// let b = &a * 2;
    /// assert_eq!(b.x, 2);
    /// assert_eq!(b.y, 4);
    /// ```
    fn mul(self, rhs: U) -> Self::Output {
        self.combine_scalar_ref(rhs, |a, b| a * b)
    }
}

impl<T, U, const D: usize> MulAssign<U> for Vec<D, T>
where
    T: MulAssign<U>,
    U: Copy,
{
    /// Multiply and assign a vector by a scalar.
    ///
    /// # Example
    /// ```
    /// use isochro::vector::Vec2;
    /// let mut a = Vec2::new(1, 2);
    /// a *= 2;
    /// assert_eq!(a.x, 2);
    /// assert_eq!(a.y, 4);
    /// ```
    fn mul_assign(&mut self, rhs: U) {
        self.combine_assign_scalar(rhs, T::mul_assign);
    }
}

// division
impl<T, U, R, const D: usize> Div<U> for Vec<D, T>
where
    T: Div<U, Output = R>,
    U: Clone,
{
    type Output = Vec<D, R>;

    /// Divide a vector by a scalar.
    ///
    /// # Example
    /// ```
    /// use isochro::vector::Vec2;
    /// let a = Vec2::new(2, 4);
    /// let b = a / 2;
    /// assert_eq!(b.x, 1);
    /// assert_eq!(b.y, 2);
    /// ```
    fn div(self, rhs: U) -> Self::Output {
        self.combine_scalar(rhs, T::div)
    }
}

impl<T, U, R, const D: usize> Div<U> for &Vec<D, T>
where
    for<'a> &'a T: Div<U, Output = R>,
    U: Copy,
{
    type Output = Vec<D, R>;

    /// Divide a vector by a scalar.
    ///
    /// # Example
    /// ```
    /// use isochro::vector::Vec2;
    /// let a = Vec2::new(2, 4);
    /// let b = &a / 2;
    /// assert_eq!(b.x, 1);
    /// assert_eq!(b.y, 2);
    /// ```
    fn div(self, rhs: U) -> Self::Output {
        self.combine_scalar_ref(rhs, |a, b| a / b)
    }
}

impl<T, U, const D: usize> DivAssign<U> for Vec<D, T>
where
    T: DivAssign<U>,
    U: Clone,
{
    /// Divide and assign a vector by a scalar.
    ///
    /// # Example
    /// ```
    /// use isochro::vector::Vec2;
    /// let mut a = Vec2::new(2, 4);
    /// a /= 2;
    /// assert_eq!(a.x, 1);
    /// assert_eq!(a.y, 2);
    /// ```
    fn div_assign(&mut self, rhs: U) {
        self.combine_assign_scalar(rhs, T::div_assign);
    }
}

// dot product
impl<T, U, R, const D: usize> DotProduct<Vec<D, U>> for Vec<D, T>
where
    T: Mul<U, Output = R>,
    R: Add<R, Output = R>,
{
    type Output = R;

    /// Calculate the dot product of two vectors.
    ///
    /// # Example
    /// ```
    /// use isochro::vector::{DotProduct, Vec3};
    /// let a = Vec3::new(1, 2, 3);
    /// let b = Vec3::new(4, 5, 6);
    /// let c = a.dot(b);
    /// assert_eq!(c, 4 + 10 + 18);
    fn dot(self, rhs: Vec<D, U>) -> Self::Output {
        use core::iter::Iterator;
        let result = zip(self.0.into_iter(), rhs.0.into_iter())
            .map(|(a, b)| a * b)
            .reduce(|acc, x| acc + x);
        unsafe { result.unwrap_unchecked() }
    }
}

impl<T, U, R, const D: usize> DotProduct<&Vec<D, U>> for Vec<D, T>
where
    T: for<'a> Mul<&'a U, Output = R>,
    R: Add<R, Output = R>,
{
    type Output = R;

    /// Calculate the dot product of two vectors.
    ///
    /// # Example
    /// ```
    /// use isochro::vector::{DotProduct, Vec3};
    /// let a = Vec3::new(1, 2, 3);
    /// let b = Vec3::new(4, 5, 6);
    /// let c = a.dot(&b);
    /// assert_eq!(c, 4 + 10 + 18);
    /// ```
    fn dot(self, rhs: &Vec<D, U>) -> Self::Output {
        use core::iter::Iterator;
        let result = zip(self.0.into_iter(), rhs.0.iter())
            .map(|(a, b)| a * b)
            .reduce(|acc, x| acc + x);
        unsafe { result.unwrap_unchecked() }
    }
}

impl<T, U, R, const D: usize> DotProduct<Vec<D, U>> for &Vec<D, T>
where
    for<'a> &'a T: Mul<U, Output = R>,
    R: Add<R, Output = R>,
{
    type Output = R;

    /// Calculate the dot product of two vectors.
    ///
    /// # Example
    /// ```
    /// use isochro::vector::{DotProduct, Vec3};
    /// let a = Vec3::new(1, 2, 3);
    /// let b = Vec3::new(4, 5, 6);
    /// let c = (&a).dot(b);
    /// assert_eq!(c, 4 + 10 + 18);
    /// ```
    fn dot(self, rhs: Vec<D, U>) -> Self::Output {
        use core::iter::Iterator;
        let result = zip(self.0.iter(), rhs.0.into_iter())
            .map(|(a, b)| a * b)
            .reduce(|acc, x| acc + x);
        unsafe { result.unwrap_unchecked() }
    }
}

impl<T, U, R, const D: usize> DotProduct<&Vec<D, U>> for &Vec<D, T>
where
    for<'a> &'a T: Mul<&'a U, Output = R>,
    R: Add<R, Output = R>,
{
    type Output = R;

    /// Calculate the dot product of two vectors.
    ///
    /// # Example
    /// ```
    /// use isochro::vector::{DotProduct, Vec3};
    /// let a = Vec3::new(1, 2, 3);
    /// let b = Vec3::new(4, 5, 6);
    /// let c = (&a).dot(&b);
    /// assert_eq!(c, 4 + 10 + 18);
    /// ```
    fn dot(self, rhs: &Vec<D, U>) -> Self::Output {
        use core::iter::Iterator;
        let result = zip(self.0.iter(), rhs.0.iter())
            .map(|(a, b)| a * b)
            .reduce(|acc, x| acc + x);
        unsafe { result.unwrap_unchecked() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_access() {
        let vec = Vec4::from([1, 2, 3, 4]);
        assert_eq!(vec[0], vec.x);
        assert_eq!(vec[1], vec.y);
        assert_eq!(vec[2], vec.z);
        assert_eq!(vec[3], vec.w);
    }
}
