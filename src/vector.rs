use std::iter::zip;
use std::ops::{Add, Deref};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec<T, const DIM: usize>(pub [T; DIM]);

//specialization for usual Vec

// ---- 2D ---- //
pub type Vec2<T> = Vec<T, 2>;

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

impl<'a, T> Deref for Vec2<T> {
    type Target = Window2<T>;

    fn deref(&self) -> &Self::Target {
        unsafe { std::mem::transmute(&self.0) } //should we be concerned about this?
    }
}

// -- 3 -- //
pub type Vec3<T> = Vec<T, 3>;

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

impl<'a, T> Deref for Vec3<T> {
    type Target = Window3<T>;

    fn deref(&self) -> &Self::Target {
        unsafe { std::mem::transmute(&self.0) } //should we be concerned about this?
    }
}

//generic case
impl<T, const D: usize> Vec<T, D> {
    fn from(x: [T; D]) -> Self {
        Self(x)
    }

    // TODO: Find faster way in debug mode to merge two statics arrays
    fn combine<U, R>(self, other: Vec<U, D>, f: impl Fn(T, U) -> R) -> Vec<R, D> {
        let a = self.0.into_iter();
        let b = other.0.into_iter();
        let mut iter = zip(a, b).map(|(a, b)| f(a, b));
        Vec(std::array::from_fn(|_| unsafe { iter.next().unwrap_unchecked() }))
    }

    fn combine_ref<U, R>(self, other: &Vec<U, D>, f: impl Fn(T, &U) -> R) -> Vec<R, D> {
        let a = self.0.into_iter();
        let b = other.0.iter();
        let mut iter = zip(a, b).map(|(a, b)| f(a, b));
        Vec(std::array::from_fn(|_| unsafe { iter.next().unwrap_unchecked() }))
    }

    fn combine_both_ref<U, R>(&self, other: &Vec<U, D>, f: impl Fn(&T, &U) -> R) -> Vec<R, D> {
        let a = self.0.iter();
        let b = other.0.iter();
        let mut iter = zip(a, b).map(|(a, b)| f(a, b));
        Vec(std::array::from_fn(|_| unsafe { iter.next().unwrap_unchecked() }))
    }
}

// math related operations
impl<T, U, R, const D: usize> Add<Vec<U, D>> for Vec<T, D>
where
    T: Add<U, Output = R>,
{
    type Output = Vec<R, D>;

    #[inline]
    fn add(self, rhs: Vec<U, D>) -> Self::Output {
        self.combine(rhs, |a, b| a + b)
    }
}

impl<T, U, R, const D: usize> Add<&Vec<U, D>> for Vec<T, D>
where
    T: for<'a> Add<&'a U, Output = R>,
{
    type Output = Vec<R, D>;
    fn add(self, rhs: &Vec<U, D>) -> Self::Output {
        self.combine_ref(rhs, |a, b| a + b)
    }
}

impl<T, U, R, const D: usize> Add<Vec<U, D>> for &Vec<T, D>
where
        for<'a> &'a T: Add<U, Output = R>,
{
    type Output = Vec<R, D>;
    fn add(self, rhs: Vec<U, D>) -> Self::Output {
        rhs.combine_ref(self, |a, b| b + a)
    }
}

impl<T, U, R, const D: usize> Add<&Vec<U, D>> for &Vec<T, D>
where
    for<'a, 'b> &'a T: Add<&'b U, Output = R>,
{
    type Output = Vec<R, D>;
    fn add(self, rhs: &Vec<U, D>) -> Self::Output {
        self.combine_both_ref(rhs, |a, b| a + b)
    }
}

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
        assert_eq!(c.x, 4);
        assert_eq!(c.y, 6);
    }
}
