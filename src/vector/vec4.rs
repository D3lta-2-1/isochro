use std::ops::{Deref, DerefMut};

/// A shorthand for a 4-dimensional vector.
pub type Vec4<T> = crate::vector::Vec<4, T>;

/// This structure isn't used directly, but is used to provide a nicer way to access the fields of a Vec4.
///
/// # Example
/// ```
/// use isochro::vector::Vec4;
///
/// let vec = Vec4::new(1, 2, 3, 4);
/// assert_eq!(vec.x, 1);
/// assert_eq!(vec.y, 2);
/// assert_eq!(vec.z, 3);
/// assert_eq!(vec.w, 4);
/// ```
#[repr(C)]
pub struct Window4<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

impl<T> Vec4<T> {
    pub fn new(x: T, y: T, z: T, w: T) -> Self {
        Self([x, y, z, w])
    }
}

impl<T> Deref for Vec4<T> {
    type Target = Window4<T>;
    fn deref(&self) -> &Self::Target {
        union Transform4<'a, T> {
            src: &'a Vec4<T>,
            dst: &'a Window4<T>,
        }

        let cast = Transform4 { src: self };
        unsafe { cast.dst } // SAFETY: repr(C) guarantees that the fields are in the same order
    }
}

impl<T> DerefMut for Vec4<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        union Transform4<'a, T> {
            src: &'a mut Vec4<T>,
            dst: &'a mut Window4<T>,
        }

        let cast = Transform4 { src: self };
        unsafe { cast.dst } // SAFETY: repr(C) guarantees that the fields are in the same order
    }
}

impl<T: PartialEq> PartialEq<(T, T, T, T)> for Vec4<T> {
    fn eq(&self, other: &(T, T, T, T)) -> bool {
        self[0] == other.0 && self[1] == other.1 && self[2] == other.2 && self[3] == other.3
    }
}

impl<T> From<(T, T, T, T)> for Vec4<T> {
    fn from(tuple: (T, T, T, T)) -> Self {
        Self::new(tuple.0, tuple.1, tuple.2, tuple.3)
    }
}