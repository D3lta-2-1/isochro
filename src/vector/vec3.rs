use std::ops::{Deref, DerefMut};

/// A shorthand for a 3-dimensional vector.
pub type Vec3<T> = crate::vector::Vec<3, T>;

/// This structure isn't used directly, but is used to provide a nicer way to access the fields of a Vec3.
///
/// # Example
/// ```
/// use isochro::vector::Vec3;
///
/// let vec = Vec3::new(1, 2, 3);
/// assert_eq!(vec.x, 1);
/// assert_eq!(vec.y, 2);
/// assert_eq!(vec.z, 3);
/// ```
#[repr(C)]
pub struct Window3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Vec3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Self([x, y, z])
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

impl<T: PartialEq> PartialEq<(T, T, T)> for Vec3<T> {
    fn eq(&self, other: &(T, T, T)) -> bool {
        self[0] == other.0 && self[1] == other.1 && self[2] == other.2
    }
}

impl<T> From<(T, T, T)> for Vec3<T> {
    fn from(tuple: (T, T, T)) -> Self {
        Self::new(tuple.0, tuple.1, tuple.2)
    }
}