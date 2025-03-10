use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::{Add, BitAnd, BitOr, BitXor, Mul, Not, Sub};
use std::ptr::copy_nonoverlapping;

use boundaries::{ArithmeticSimdOperation, CheckIntOverflowSimd, ComparisonSimdOperation};
use macros::generate_simd_support;

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "x86_64")]
mod x86_64;

mod boundaries;
mod macros;

#[allow(private_bounds)]
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Simd<T, const N: usize>(<LaneCount<T, N> as SupportedNativeSimd<T, N>>::SimdType)
where
    LaneCount<T, N>: SupportedNativeSimd<T, N>;

#[allow(private_bounds)]
impl<T, const N: usize> Simd<T, N>
where
    LaneCount<T, N>: SupportedNativeSimd<T, N>,
{
    /// Converts an array to a SIMD vector.
    #[inline]
    pub fn from_array(array: [T; N]) -> Self {
        // SAFETY: `&array` is safe to read.
        unsafe { Self(<LaneCount<T, N> as SupportedNativeSimd<T, N>>::load(&array)) }
    }

    /// Converts a SIMD vector to an array.
    #[inline]
    pub fn to_array(self) -> [T; N] {
        let mut tmp = MaybeUninit::<[T; N]>::uninit();
        unsafe {
            <LaneCount<T, N> as SupportedNativeSimd<T, N>>::store(tmp.as_mut_ptr(), self.0);
            tmp.assume_init()
        }
    }

    /// Returns an array reference containing the entire SIMD vector.
    #[inline]
    pub fn as_array(&self) -> &[T]
    where
        <LaneCount<T, N> as SupportedNativeSimd<T, N>>::InnerRef: AsRef<[T]>,
    {
        <LaneCount<T, N> as SupportedNativeSimd<T, N>>::as_inner_ref(&self.0).as_ref()
    }

    pub fn partial_eq(self, rhs: Self) -> Simd<bool, N>
    where
        LaneCount<T, N>: ComparisonSimdOperation<T, N>,
        LaneCount<bool, N>: SupportedNativeSimd<
                bool,
                N,
                SimdType = <LaneCount<T, N> as SupportedNativeSimd<T, N>>::Mask,
            >,
    {
        Simd(<LaneCount<T, N> as ComparisonSimdOperation<T, N>>::eq(
            self.0, rhs.0,
        ))
    }

    pub fn partial_gt(self, rhs: Self) -> Simd<bool, N>
    where
        LaneCount<T, N>: ComparisonSimdOperation<T, N>,
        LaneCount<bool, N>: SupportedNativeSimd<
                bool,
                N,
                SimdType = <LaneCount<T, N> as SupportedNativeSimd<T, N>>::Mask,
            >,
    {
        Simd(<LaneCount<T, N> as ComparisonSimdOperation<T, N>>::gt(
            self.0, rhs.0,
        ))
    }

    pub fn partial_lt(self, rhs: Self) -> Simd<bool, N>
    where
        LaneCount<T, N>: ComparisonSimdOperation<T, N>,
        LaneCount<bool, N>: SupportedNativeSimd<
                bool,
                N,
                SimdType = <LaneCount<T, N> as SupportedNativeSimd<T, N>>::Mask,
            >,
    {
        Simd(<LaneCount<T, N> as ComparisonSimdOperation<T, N>>::eq(
            self.0, rhs.0,
        ))
    }

    /// Calculates `self + rhs`.
    ///
    /// Returns a tuple of the addition along with a boolean indicating whether
    /// an arithmetic overflow would occur. If an overflow would have occurred
    /// then the wrapped value is returned.
    #[inline]
    pub fn overflowing_add(self, rhs: Self) -> (Self, bool)
    where
        LaneCount<T, N>: CheckIntOverflowSimd<T, N>,
    {
        let (sum, overflow) =
            <LaneCount<T, N> as CheckIntOverflowSimd<T, N>>::overflowing_add(self.0, rhs.0);
        (Self(sum), overflow)
    }

    /// Wrapping (modular) addition. Computes self + rhs, wrapping around at
    /// the boundary of the type.
    #[inline]
    pub fn wrapping_add(self, rhs: Self) -> Self
    where
        LaneCount<T, N>: CheckIntOverflowSimd<T, N>,
    {
        Self(<LaneCount<T, N> as CheckIntOverflowSimd<T, N>>::wrapping_add(self.0, rhs.0))
    }

    /// Unchecked addition. Computes `self + rhs`, assuming overflow cannot occur.
    ///
    /// Calling `unchecked_add(x, y)` is semantically equivalent to calling `checked_add(x, y).unwrap_unchecked()`.
    ///
    /// If you're just trying to avoid the panic in debug mode, then do not use this. Instead, you're looking for [`Self::wrapping_add`].
    ///
    /// # Safety
    ///
    /// This results in undefined behavior when `self + rhs > T::MAX` or `self + rhs < T::MIN`, i.e. when checked_add would return None.
    #[inline]
    pub unsafe fn unchecked_add(self, rhs: Self) -> Self
    where
        LaneCount<T, N>: CheckIntOverflowSimd<T, N>,
    {
        Self(unsafe {
            <LaneCount<T, N> as CheckIntOverflowSimd<T, N>>::unchecked_add(self.0, rhs.0)
        })
    }

    /// Checked addition. Computes `self + rhs`, returning None if overflow occurred.
    #[inline]
    pub fn checked_add(self, rhs: Self) -> Option<Self>
    where
        LaneCount<T, N>: CheckIntOverflowSimd<T, N>,
    {
        <LaneCount<T, N> as CheckIntOverflowSimd<T, N>>::checked_add(self.0, rhs.0).map(Self)
    }

    /// Calculates `lhs - rhs`.
    ///
    /// Returns a tuple of the subtraction along with a boolean indicating whether
    /// an arithmetic overflow would occur. If an overflow would have occurred
    /// then the wrapped value is returned.
    pub fn overflowing_sub(self, rhs: Self) -> (Self, bool)
    where
        LaneCount<T, N>: CheckIntOverflowSimd<T, N>,
    {
        let (sum, overflow) =
            <LaneCount<T, N> as CheckIntOverflowSimd<T, N>>::overflowing_sub(self.0, rhs.0);
        (Self(sum), overflow)
    }

    /// Wrapping (modular) subtraction. Computes `lhs - rhs`, wrapping around at
    /// the boundary of the type.
    pub fn wrapping_sub(self, rhs: Self) -> Self
    where
        LaneCount<T, N>: CheckIntOverflowSimd<T, N>,
    {
        Self(<LaneCount<T, N> as CheckIntOverflowSimd<T, N>>::wrapping_sub(self.0, rhs.0))
    }

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
    pub unsafe fn unchecked_sub(self, rhs: Self) -> Self
    where
        LaneCount<T, N>: CheckIntOverflowSimd<T, N>,
    {
        Self(unsafe {
            <LaneCount<T, N> as CheckIntOverflowSimd<T, N>>::unchecked_sub(self.0, rhs.0)
        })
    }

    /// Checked subtraction. Computes `lhs - rhs`, returning None if overflow occurred.
    #[inline]
    pub fn checked_sub(self, rhs: Self) -> Option<Self>
    where
        LaneCount<T, N>: CheckIntOverflowSimd<T, N>,
    {
        <LaneCount<T, N> as CheckIntOverflowSimd<T, N>>::checked_sub(self.0, rhs.0).map(Self)
    }

    /// Calculates `lhs * rhs`.
    ///
    /// Returns a tuple of the multiplication along with a boolean indicating whether
    /// an arithmetic overflow would occur. If an overflow would have occurred
    /// then the wrapped value is returned.
    pub fn overflowing_mul(self, rhs: Self) -> (Self, bool)
    where
        LaneCount<T, N>: CheckIntOverflowSimd<T, N>,
    {
        let (sum, overflow) =
            <LaneCount<T, N> as CheckIntOverflowSimd<T, N>>::overflowing_mul(self.0, rhs.0);
        (Self(sum), overflow)
    }

    /// Wrapping (modular) multiplication. Computes `lhs * rhs`, wrapping around at
    /// the boundary of the type.
    pub fn wrapping_mul(self, rhs: Self) -> Self
    where
        LaneCount<T, N>: CheckIntOverflowSimd<T, N>,
    {
        Self(<LaneCount<T, N> as CheckIntOverflowSimd<T, N>>::wrapping_mul(self.0, rhs.0))
    }

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
    pub unsafe fn unchecked_mul(self, rhs: Self) -> Self
    where
        LaneCount<T, N>: CheckIntOverflowSimd<T, N>,
    {
        Self(unsafe {
            <LaneCount<T, N> as CheckIntOverflowSimd<T, N>>::unchecked_mul(self.0, rhs.0)
        })
    }

    /// Checked multiplication. Computes `lhs * rhs`, returning None if overflow occurred.
    #[inline]
    pub fn checked_mul(self, rhs: Self) -> Option<Self>
    where
        LaneCount<T, N>: CheckIntOverflowSimd<T, N>,
    {
        <LaneCount<T, N> as CheckIntOverflowSimd<T, N>>::checked_mul(self.0, rhs.0).map(Self)
    }
}

impl<T: Debug, const N: usize> Debug for Simd<T, N>
where
    LaneCount<T, N>: SupportedNativeSimd<T, N>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <LaneCount<T, N> as SupportedNativeSimd<T, N>>::as_inner_ref(&self.0).fmt(f)
    }
}

impl<T, const N: usize> Add for Simd<T, N>
where
    LaneCount<T, N>: ArithmeticSimdOperation<T, N>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(<LaneCount<T, N> as ArithmeticSimdOperation<T, N>>::add(
            self.0, rhs.0,
        ))
    }
}

impl<T, const N: usize> Sub for Simd<T, N>
where
    LaneCount<T, N>: ArithmeticSimdOperation<T, N>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(<LaneCount<T, N> as ArithmeticSimdOperation<T, N>>::sub(
            self.0, rhs.0,
        ))
    }
}

impl<T, const N: usize> Mul for Simd<T, N>
where
    LaneCount<T, N>: ArithmeticSimdOperation<T, N>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(<LaneCount<T, N> as ArithmeticSimdOperation<T, N>>::mul(
            self.0, rhs.0,
        ))
    }
}

impl<const N: usize> BitAnd for Simd<bool, N>
where
    LaneCount<bool, N>: SupportedNativeSimd<bool, N>,
    <LaneCount<bool, N> as SupportedNativeSimd<bool, N>>::SimdType:
        BitAnd<Output = <LaneCount<bool, N> as SupportedNativeSimd<bool, N>>::SimdType>,
{
    type Output = Simd<bool, N>;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl<const N: usize> BitOr for Simd<bool, N>
where
    LaneCount<bool, N>: SupportedNativeSimd<bool, N>,
    <LaneCount<bool, N> as SupportedNativeSimd<bool, N>>::SimdType:
        BitOr<Output = <LaneCount<bool, N> as SupportedNativeSimd<bool, N>>::SimdType>,
{
    type Output = Simd<bool, N>;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl<const N: usize> BitXor for Simd<bool, N>
where
    LaneCount<bool, N>: SupportedNativeSimd<bool, N>,
    <LaneCount<bool, N> as SupportedNativeSimd<bool, N>>::SimdType:
        BitXor<Output = <LaneCount<bool, N> as SupportedNativeSimd<bool, N>>::SimdType>,
{
    type Output = Simd<bool, N>;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

/// Specifies the number of lanes in a SIMD vector as a type.
pub(crate) struct LaneCount<T, const N: usize>(PhantomData<T>);

/// Trait used as a marker to force the user
pub(crate) trait SupportedNativeSimd<T, const N: usize>: Sized {
    /// Represent the native simd type used internally
    type SimdType: Copy;

    /// Represent the type of the mask when the simd type is cast to a bool
    type Mask: Copy + BitAnd + BitOr + BitXor + Not;

    /// Represent the ref of the simd array
    type InnerRef: Debug + ?Sized;

    /// Returns an array reference containing the entire SIMD vector.
    fn as_inner_ref<'a>(value: &'a Self::SimdType) -> &'a Self::InnerRef;

    /// Store a vector into an array of `T`.
    ///
    /// # Safety
    ///
    /// Writing `mem_addr` must be safe, as if by [`std::ptr::write`].
    unsafe fn store(mem_addr: *mut [T; N], value: Self::SimdType) {
        // SAFETY: the safety contract for `store` must be upheld by the caller
        unsafe {
            copy_nonoverlapping(
                (&raw const value) as *const u8,
                mem_addr as *mut u8,
                size_of::<Self::SimdType>(),
            );
        }
    }

    /// Loads a vector from an array of `T`.
    ///
    /// # Safety
    ///
    /// Reading `mem_addr` must be safe, as if by [`std::ptr::read`].
    unsafe fn load(mem_addr: *const [T; N]) -> Self::SimdType {
        // SAFETY: create a empty destination correctly aligned (like, for __m256
        // that require a alignement of 32)
        let mut dst: Self::SimdType = unsafe { MaybeUninit::zeroed().assume_init() };
        // SAFETY: the safety contract for `load` must be upheld by the caller
        unsafe {
            copy_nonoverlapping(
                mem_addr as *const u8,
                (&raw mut dst) as *mut u8,
                size_of::<Self::SimdType>(),
            );
        }
        dst
    }
}

macro_rules! gen_single_int {
    ($($t:ty),* $(,)?) => {
        $(generate_simd_support! {
            for [1 x $t] use $t : not_native;
            type InnerRef = [$t];
            type Mask = bool;
            impl base {
                fn as_inner_ref<'a>(value: &'a $t) -> &'a [$t] {
                    std::slice::from_ref(value)
                }
            }
            impl trait ArithmeticSimdOperation {
                fn add(lhs: $t, rhs: $t) -> $t { lhs + rhs }
                fn sub(lhs: $t, rhs: $t) -> $t { lhs - rhs }
                fn mul(lhs: $t, rhs: $t) -> $t { lhs * rhs }
            }
            impl trait CheckIntOverflowSimd {
                fn overflowing_add(lhs: $t, rhs: $t) -> ($t, bool) { lhs.overflowing_add(rhs) }
                fn wrapping_add(lhs: $t, rhs: $t) -> $t { lhs.wrapping_add(rhs) }
                unsafe fn unchecked_add(lhs: $t, rhs: $t) -> $t { unsafe {lhs.unchecked_add(rhs)} }
                fn checked_add(lhs: $t, rhs: $t) -> Option<$t> { lhs.checked_add(rhs) }

                fn overflowing_sub(lhs: $t, rhs: $t) -> ($t, bool) { lhs.overflowing_sub(rhs) }
                fn wrapping_sub(lhs: $t, rhs: $t) -> $t { lhs.wrapping_sub(rhs) }
                unsafe fn unchecked_sub(lhs: $t, rhs: $t) -> $t { unsafe {lhs.unchecked_sub(rhs)} }
                fn checked_sub(lhs: $t, rhs: $t) -> Option<$t> { lhs.checked_sub(rhs) }

                fn overflowing_mul(lhs: $t, rhs: $t) -> ($t, bool) { lhs.overflowing_mul(rhs) }
                fn wrapping_mul(lhs: $t, rhs: $t) -> $t { lhs.wrapping_mul(rhs) }
                unsafe fn unchecked_mul(lhs: $t, rhs: $t) -> $t { unsafe {lhs.unchecked_mul(rhs)} }
                fn checked_mul(lhs: $t, rhs: $t) -> Option<$t> { lhs.checked_mul(rhs) }
            }
        })*
    };
}

gen_single_int! {
    i32,
    u32,
}

macro_rules! gen_single_float {
    ($($t:ty),* $(,)?) => {
        $(generate_simd_support! {
            for [1 x $t] use $t;
            type InnerRef = [$t];
            type Mask = bool;
            impl base {
                fn as_inner_ref<'a>(value: &'a $t) -> &'a [$t] {
                    std::slice::from_ref(value)
                }
            }
            impl trait ArithmeticSimdOperation {
                fn add(lhs: $t, rhs: $t) -> $t { lhs + rhs }
                fn sub(lhs: $t, rhs: $t) -> $t { lhs - rhs }
                fn mul(lhs: $t, rhs: $t) -> $t { lhs * rhs }
            }
        })*
    };
}

gen_single_float! {
    f32
}

generate_simd_support! {
    for [1 x bool] use u8 : not_native;
    type InnerRef = u8;
    type Mask = u8;
    impl base {
        fn as_inner_ref<'a>(value: &'a u8) -> &'a u8 {
            value
        }
    }
}
generate_simd_support! {
    for [2 x bool] use u8 : not_native;
    type InnerRef = u8;
    type Mask = u8;
    impl base {
        fn as_inner_ref<'a>(value: &'a u8) -> &'a u8 {
            value
        }
    }
}
generate_simd_support! {
    for [4 x bool] use u8 : not_native;
    type InnerRef = u8;
    type Mask = u8;
    impl base {
        fn as_inner_ref<'a>(value: &'a u8) -> &'a u8 {
            value
        }
    }
}
generate_simd_support! {
    for [8 x bool] use u8 : not_native;
    type InnerRef = u8;
    type Mask = u8;
    impl base {
        fn as_inner_ref<'a>(value: &'a u8) -> &'a u8 {
            value
        }
    }
}
