use std::ops::{BitAnd, BitOr, BitXor, Not};

use crate::macros::assert_unsafe_precondition;
use crate::utils::unlikely;

use super::SimdElement;

pub(crate) trait ArithmeticSimdOperation<const N: usize>: SimdElement<N> {
    /// Function used to define addition for simd vector safely.
    fn add(lhs: Self::NativeType, rhs: Self::NativeType) -> Self::NativeType;

    /// Function used to define subtraction for simd vector safely.
    fn sub(lhs: Self::NativeType, rhs: Self::NativeType) -> Self::NativeType;

    /// Function used to define multiple for simd vector safely.
    fn mul(lhs: Self::NativeType, rhs: Self::NativeType) -> Self::NativeType;
}

pub(crate) trait CheckIntOverflowSimd<const N: usize>: ArithmeticSimdOperation<N> {
    /// Calculates `lhs + rhs`.
    ///
    /// Returns a tuple of the addition along with a boolean indicating whether
    /// an arithmetic overflow would occur. If an overflow would have occurred
    /// then the wrapped value is returned.
    fn overflowing_add(lhs: Self::NativeType, rhs: Self::NativeType) -> (Self::NativeType, bool);

    /// Wrapping (modular) addition. Computes `lhs + rhs`, wrapping around at
    /// the boundary of the type.
    fn wrapping_add(lhs: Self::NativeType, rhs: Self::NativeType) -> Self::NativeType;

    /// Unchecked addition. Computes `lhs + rhs`, assuming overflow cannot occur.
    ///
    /// Calling `Self::unchecked_add(x, y)` is semantically equivalent to calling
    /// `Self::checked_add(x, y).unwrap_unchecked()`.
    ///
    /// If you're just trying to avoid the panic in debug mode, then do not use
    /// this. Instead, you're looking for [`Self::wrapping_add`].
    ///
    /// # Safety
    ///
    /// This results in undefined behavior when `lhs + rhs > T::MAX` or
    /// `lhs + rhs < T::MIN`, i.e. when checked_add would return None.
    #[inline]
    unsafe fn unchecked_add(lhs: Self::NativeType, rhs: Self::NativeType) -> Self::NativeType {
        let (sum, overflow) = Self::overflowing_add(lhs, rhs);
        assert_unsafe_precondition!(
            "Simd::unchecked_add cannot overflow",
            () => !overflow,
        );

        sum
    }

    /// Checked addition. Computes `lhs + rhs`, returning None if overflow occurred.
    #[inline]
    fn checked_add(lhs: Self::NativeType, rhs: Self::NativeType) -> Option<Self::NativeType> {
        if unlikely(Self::overflowing_add(lhs, rhs).1) {
            None
        } else {
            // SAFETY: Just checked it doesn't overflow
            Some(unsafe { Self::unchecked_add(lhs, rhs) })
        }
    }

    /// Calculates `lhs - rhs`.
    ///
    /// Returns a tuple of the subtraction along with a boolean indicating whether
    /// an arithmetic overflow would occur. If an overflow would have occurred
    /// then the wrapped value is returned.
    fn overflowing_sub(lhs: Self::NativeType, rhs: Self::NativeType) -> (Self::NativeType, bool);

    /// Wrapping (modular) subtraction. Computes `lhs - rhs`, wrapping around at
    /// the boundary of the type.
    fn wrapping_sub(lhs: Self::NativeType, rhs: Self::NativeType) -> Self::NativeType;

    /// Unchecked subtraction. Computes `lhs - rhs`, assuming overflow cannot occur.
    ///
    /// Calling `Self::unchecked_sub(x, y)` is semantically equivalent to calling
    /// `Self::checked_sub(x, y).unwrap_unchecked()`.
    ///
    /// If you're just trying to avoid the panic in debug mode, then do not use
    /// this. Instead, you're looking for [`Self::wrapping_sub`].
    ///
    /// # Safety
    ///
    /// This results in undefined behavior when `lhs - rhs > T::MAX` or
    /// `lhs - rhs < T::MIN`, i.e. when checked_sub would return None.
    #[inline]
    unsafe fn unchecked_sub(lhs: Self::NativeType, rhs: Self::NativeType) -> Self::NativeType {
        let (sum, overflow) = Self::overflowing_sub(lhs, rhs);
        assert_unsafe_precondition!(
            "Simd::unchecked_add cannot overflow",
            () => !overflow,
        );

        sum
    }

    /// Checked subtraction. Computes `lhs - rhs`, returning None if overflow occurred.
    #[inline]
    fn checked_sub(lhs: Self::NativeType, rhs: Self::NativeType) -> Option<Self::NativeType> {
        if unlikely(Self::overflowing_sub(lhs, rhs).1) {
            None
        } else {
            // SAFETY: Just checked it doesn't overflow
            Some(unsafe { Self::unchecked_sub(lhs, rhs) })
        }
    }

    /// Calculates `lhs * rhs`.
    ///
    /// Returns a tuple of the multiplication along with a boolean indicating whether
    /// an arithmetic overflow would occur. If an overflow would have occurred
    /// then the wrapped value is returned.
    fn overflowing_mul(lhs: Self::NativeType, rhs: Self::NativeType) -> (Self::NativeType, bool);

    /// Wrapping (modular) multiplication. Computes `lhs * rhs`, wrapping around at
    /// the boundary of the type.
    fn wrapping_mul(lhs: Self::NativeType, rhs: Self::NativeType) -> Self::NativeType;

    /// Unchecked multiplication. Computes `lhs * rhs`, assuming overflow cannot occur.
    ///
    /// Calling `Self::unchecked_mul(x, y)` is semantically equivalent to calling
    /// `Self::checked_mul(x, y).unwrap_unchecked()`.
    ///
    /// If you're just trying to avoid the panic in debug mode, then do not use
    /// this. Instead, you're looking for [`Self::wrapping_mul`].
    ///
    /// # Safety
    ///
    /// This results in undefined behavior when `lhs * rhs > T::MAX` or
    /// `lhs * rhs < T::MIN`, i.e. when checked_sub would return None.
    #[inline]
    unsafe fn unchecked_mul(lhs: Self::NativeType, rhs: Self::NativeType) -> Self::NativeType {
        let (sum, overflow) = Self::overflowing_mul(lhs, rhs);
        assert_unsafe_precondition!(
            "Simd::unchecked_add cannot overflow",
            () => !overflow,
        );

        sum
    }

    /// Checked multiplication. Computes `lhs * rhs`, returning None if overflow occurred.
    #[inline]
    fn checked_mul(lhs: Self::NativeType, rhs: Self::NativeType) -> Option<Self::NativeType> {
        if unlikely(Self::overflowing_mul(lhs, rhs).1) {
            None
        } else {
            // SAFETY: Just checked it doesn't overflow
            Some(unsafe { Self::unchecked_mul(lhs, rhs) })
        }
    }
}

pub(crate) trait ComparisonSimdOperation<const N: usize>: SimdElement<N> {
    // fn cmp(lhs: Self::NativeType, rhs: Self::NativeType) -> Self::NativeType;

    /// Function used to define maximum for simd vector safely.
    fn max(lhs: Self::NativeType, rhs: Self::NativeType) -> Self::NativeType;

    /// Function used to define minimum for simd vector safely.
    fn min(lhs: Self::NativeType, rhs: Self::NativeType) -> Self::NativeType;

    /// Function used to define equality for simd vector safely.
    fn eq(lhs: Self::NativeType, rhs: Self::NativeType) -> Self::Mask;

    /// Function used to define greater-than for simd vector safely.
    fn gt(lhs: Self::NativeType, rhs: Self::NativeType) -> Self::Mask;

    /// Function used to define less-than for simd vector safely.
    fn lt(lhs: Self::NativeType, rhs: Self::NativeType) -> Self::Mask;
}

impl<T, const N: usize> ArithmeticSimdOperation<N> for T
where
    T: CheckIntOverflowSimd<N>,
    T::NativeType: NativeSimd,
{
    fn add(lhs: Self::NativeType, rhs: Self::NativeType) -> Self::NativeType {
        if cfg!(debug_assertions) {
            let (sum, overflow) = Self::overflowing_add(lhs, rhs);
            if overflow {
                panic!("attempt to add with overflow");
            }
            sum
        } else {
            Self::wrapping_add(lhs, rhs)
        }
    }
    fn sub(lhs: Self::NativeType, rhs: Self::NativeType) -> Self::NativeType {
        if cfg!(debug_assertions) {
            let (sum, overflow) = Self::overflowing_sub(lhs, rhs);
            if overflow {
                panic!("attempt to sub with overflow");
            }
            sum
        } else {
            Self::wrapping_sub(lhs, rhs)
        }
    }
    fn mul(lhs: Self::NativeType, rhs: Self::NativeType) -> Self::NativeType {
        if cfg!(debug_assertions) {
            let (sum, overflow) = Self::overflowing_mul(lhs, rhs);
            if overflow {
                panic!("attempt to mul with overflow");
            }
            sum
        } else {
            Self::wrapping_mul(lhs, rhs)
        }
    }
}

pub trait NativeSimd {}

pub unsafe trait MaskElement<const N: usize>:
    Sized
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + BitXor<Output = Self>
    + Not<Output = Self>
    + PartialEq
{
    const FULL_MASK: Self;
}

unsafe impl<const N: usize> MaskElement<N> for bool {
    const FULL_MASK: Self = true;
}

unsafe impl<const N: usize> MaskElement<N> for u8 {
    const FULL_MASK: Self = u8::MAX
        >> (8 - match N {
            0..8 => N as u8,
            _ => 8,
        });
}
