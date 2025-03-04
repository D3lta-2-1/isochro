//! A generic matrix type with compile-time dimensionality.
//! This type is a wrapper around a fixed-size array, and provides
//! a number of convenience methods for working with matrices.
//! The dimensionality of the matrix is specified as a type parameter.
//! This allows the compiler to catch errors where matrices of different
//! sizes are used incorrectly.

use std::iter::zip;
use std::ops::{Add, Index, IndexMut};

use crate::macros::forward_ref_binop;
use crate::vector::Vec;

/// A generic matrix type with compile-time dimensionality.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat<const ROW: usize, const COL: usize, T>(pub [Vec<COL, T>; ROW]);

impl<T, const M: usize, const N: usize> Index<usize> for Mat<M, N, T> {
    type Output = Vec<N, T>;

    /// Get the value at the given index.
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T, const M: usize, const N: usize> IndexMut<usize> for Mat<M, N, T> {
    /// Get a mutable reference to the value at the given index.
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T, const M: usize, const N: usize> Index<(usize, usize)> for Mat<M, N, T> {
    type Output = T;

    /// Get the value at the given index.
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.0[index.0][index.1]
    }
}

impl<T, const M: usize, const N: usize> IndexMut<(usize, usize)> for Mat<M, N, T> {
    /// Get a mutable reference to the value at the given index.
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.0[index.0][index.1]
    }
}

impl<T, U, R, const M: usize, const N: usize> Add<Mat<M, N, U>> for Mat<M, N, T>
where
    T: Add<U, Output = R>,
{
    type Output = Mat<M, N, R>;

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
    fn add(self, rhs: Mat<M, N, U>) -> Self::Output {
        let a = self.0.into_iter();
        let b = rhs.0.into_iter();
        let mut iter = zip(a, b).map(|(a, b)| a + b);

        Mat(std::array::from_fn(|_| unsafe {
            iter.next().unwrap_unchecked()
        }))
    }
}

forward_ref_binop! {
    impl<T, U, R; const M: usize, const N: usize> Add<Mat<M, N, U>>, add for Mat<M, N, T>
    where
        T: Add<U, Output = R> + Copy,
        U: Copy,
}
