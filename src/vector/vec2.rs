use std::ops::{Deref, DerefMut};

/// A shorthand for a 2-dimensional vector.
pub type Vec2<T> = crate::vector::Vec<2, T>;

/// This structure isn't used directly, but is used to provide a nicer way to access the fields of a Vec2.
///
/// # Example
/// ```
/// use isochro::vector::Vec2;
///
/// let vec = Vec2::new(1, 2);
/// assert_eq!(vec.x, 1);
/// assert_eq!(vec.y, 2);
/// ```
#[repr(C)]
pub struct Window2<T> {
    pub x: T,
    pub y: T,
}

impl<T> Vec2<T> {
    pub fn new(x: T, y: T) -> Self {
        Self([x, y])
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

impl<T: PartialEq> PartialEq<(T, T)> for Vec2<T> {
    fn eq(&self, other: &(T, T)) -> bool {
        self[0] == other.0 && self[1] == other.1
    }
}

impl<T> From<(T, T)> for Vec2<T> {
    fn from(value: (T, T)) -> Self {
        Self([value.0, value.1])
    }
}