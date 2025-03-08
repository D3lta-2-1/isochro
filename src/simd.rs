use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::{Add, Mul, Sub};
use std::ptr::{addr_of_mut, copy_nonoverlapping};

use macros::generate_simd_support;

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "x86_64")]
mod x86_64;

mod macros;

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
    LaneCount<T, N>: ArithmeticSimdMathOperation<T, N>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        // TODO: add safety guaranty
        Self(unsafe { <LaneCount<T, N> as ArithmeticSimdMathOperation<T, N>>::add(self.0, rhs.0) })
    }
}

impl<T, const N: usize> Sub for Simd<T, N>
where
    LaneCount<T, N>: ArithmeticSimdMathOperation<T, N>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        // TODO: add safety guaranty
        Self(unsafe { <LaneCount<T, N> as ArithmeticSimdMathOperation<T, N>>::sub(self.0, rhs.0) })
    }
}

impl<T, const N: usize> Mul for Simd<T, N>
where
    LaneCount<T, N>: ArithmeticSimdMathOperation<T, N>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        // TODO: add safety guaranty
        Self(unsafe { <LaneCount<T, N> as ArithmeticSimdMathOperation<T, N>>::mul(self.0, rhs.0) })
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
    /// Reading `mem_addr` must be safe, as if by [`std::ptr::read`].
    unsafe fn load(mem_addr: *const [T; N]) -> Self::RelativeSimdType {
        // SAFETY: create a empty destination correctly aligned (like, for __m256
        // that require a alignement of 32)
        let mut dst: Self::RelativeSimdType = unsafe { MaybeUninit::zeroed().assume_init() };
        // SAFETY: the safety contract for `load` must be upheld by the caller
        unsafe {
            copy_nonoverlapping(
                mem_addr as *const u8,
                addr_of_mut!(dst) as *mut u8,
                size_of::<Self::RelativeSimdType>(),
            );
        }
        dst
    }
}

pub(crate) trait ArithmeticSimdMathOperation<T, const N: usize>:
    SupportedNativeSimd<T, N>
{
    unsafe fn add(
        lhs: Self::RelativeSimdType,
        rhs: Self::RelativeSimdType,
    ) -> Self::RelativeSimdType;
    unsafe fn sub(
        lhs: Self::RelativeSimdType,
        rhs: Self::RelativeSimdType,
    ) -> Self::RelativeSimdType;
    unsafe fn mul(
        lhs: Self::RelativeSimdType,
        rhs: Self::RelativeSimdType,
    ) -> Self::RelativeSimdType;
}

macro_rules! gen_single {
    ($($t:ty),* $(,)?) => {
        $(generate_simd_support! {
            for [1 x $t] use $t,
            impl trait ArithmeticSimdMathOperation {
                unsafe fn add(lhs: $t, rhs: $t) -> $t = Add::add,
                unsafe fn sub(lhs: $t, rhs: $t) -> $t = Sub::sub,
                unsafe fn mul(lhs: $t, rhs: $t) -> $t = Mul::mul,
            }
        })*
    };
}

gen_single! {
    i32,
    u32,
    f32,
}
