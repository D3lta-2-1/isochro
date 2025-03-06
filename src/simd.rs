use std::marker::PhantomData;
use std::ops::Add;

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "x86_64")]
mod x86_64;

#[allow(private_bounds)]
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Simd<T, const N: usize>(
    <LaneCount<T, N> as SupportedNativeSimd<T, N>>::RelativeSimdType,
)
where
    LaneCount<T, N>: SupportedNativeSimd<T, N>;

#[allow(private_bounds)]
impl<T, const N: usize> Simd<T, N>
where
    LaneCount<T, N>: SupportedNativeSimd<T, N>,
{
    pub fn from_array(array: [T; N]) -> Self {
        // SAFETY: `&array` is safe to read.
        unsafe { Self(<LaneCount<T, N> as SupportedNativeSimd<T, N>>::load(&array)) }
    }
}

impl<T, const N: usize> Add for Simd<T, N>
where
    LaneCount<T, N>: SupportedNativeSimd<T, N>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        // TODO: add safety guaranty
        Self(unsafe {
            <LaneCount<T, N> as SupportedNativeSimd<T, N>>::add(
                self.0, rhs.0,
            )
        })
    }
}

/// Specifies the number of lanes in a SIMD vector as a type.
pub(crate) struct LaneCount<T, const N: usize>(PhantomData<T>);

/// Trait used as a marker to force the user
pub(crate) trait SupportedNativeSimd<T, const N: usize>: Sized {
    type RelativeSimdType: Copy;

    /// Loads a vector from an array of `T`.
    ///
    /// # Safety
    ///
    /// Reading `ptr` must be safe, as if by [`std::ptr::read`].
    unsafe fn load(ptr: *const [T; N]) -> Self::RelativeSimdType {
        // The safety of reading `ptr` is ensured by the caller.
        unsafe { *ptr.cast() }
    }

    unsafe fn add(lhs: Self::RelativeSimdType, rhs: Self::RelativeSimdType) -> Self::RelativeSimdType;
}

impl SupportedNativeSimd<f32, 1> for LaneCount<f32, 1> {
    type RelativeSimdType = f32;

    #[inline]
    unsafe fn add(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}
