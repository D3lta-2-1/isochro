use std::fmt::Debug;
use std::mem::MaybeUninit;
use std::ops::{Add, BitAnd, BitOr, BitXor, Mul, Sub};
use std::ptr::copy_nonoverlapping;

use boundaries::MaskElement;
use boundaries::{ArithmeticSimdOperation, CheckIntOverflowSimd, ComparisonSimdOperation};
use macros::generate_simd_support;

use crate::markers::Sealed;

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "x86_64")]
mod x86_64;

mod boundaries;
mod macros;

#[allow(private_bounds)]
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Simd<T, const N: usize>(T::NativeType)
where
    T: SimdElement<N>;

#[allow(private_bounds)]
impl<T, const N: usize> Simd<T, N>
where
    T: SimdElement<N>,
{
    /// Converts an array to a SIMD vector.
    #[inline]
    pub fn splat(value: T) -> Self {
        // SAFETY: `&[value; N]` is safe to read.
        unsafe { Self(T::load(&[value; N])) }
    }

    /// Converts an array to a SIMD vector.
    #[inline]
    pub fn from_array(array: [T; N]) -> Self {
        // SAFETY: `&array` is safe to read.
        unsafe { Self(T::load(&array)) }
    }

    /// Converts a SIMD vector to an array.
    #[inline]
    pub fn to_array(self) -> [T; N] {
        let mut tmp = MaybeUninit::<[T; N]>::uninit();
        unsafe {
            T::store(tmp.as_mut_ptr(), self.0);
            tmp.assume_init()
        }
    }

    /// Returns an array reference containing the entire SIMD vector.
    #[inline]
    pub fn as_array(&self) -> &[T]
    where
        T::InnerRef: AsRef<[T]>,
    {
        T::as_inner_ref(&self.0).as_ref()
    }

    /// Perform `==` operation on each element of 2 vectors.
    #[inline]
    pub fn partial_eq(&self, rhs: &Self) -> Simd<bool, N>
    where
        T: ComparisonSimdOperation<N>,
        bool: SimdElement<N, NativeType = T::Mask>,
    {
        Simd(T::eq(self.0, rhs.0))
    }

    /// Perform `!=` operation on each element of 2 vectors.
    #[inline]
    pub fn partial_ne(&self, rhs: &Self) -> Simd<bool, N>
    where
        T: ComparisonSimdOperation<N>,
        bool: SimdElement<N, NativeType = T::Mask>,
    {
        Simd(!T::eq(self.0, rhs.0))
    }

    /// Perform `>` operation on each element of 2 vectors.
    #[inline]
    pub fn partial_gt(&self, rhs: &Self) -> Simd<bool, N>
    where
        T: ComparisonSimdOperation<N>,
        bool: SimdElement<N, NativeType = T::Mask>,
    {
        Simd(T::gt(self.0, rhs.0))
    }

    /// Perform `<` operation on each element of 2 vectors.
    #[inline]
    pub fn partial_lt(&self, rhs: &Self) -> Simd<bool, N>
    where
        T: ComparisonSimdOperation<N>,
        bool: SimdElement<N, NativeType = T::Mask>,
    {
        Simd(T::lt(self.0, rhs.0))
    }

    /// Compare each elements of 2 vectors and return the max of each comparison,
    /// in the case of a f32, it case the same behaviors has [`f32::max`].
    #[inline]
    pub fn max(self, rhs: Self) -> Simd<T, N>
    where
        T: ComparisonSimdOperation<N>,
    {
        Simd(T::max(self.0, rhs.0))
    }

    /// Compare each elements of 2 vectors and return the min of each comparison,
    /// in the case of a f32, it case the same behaviors has [`f32::max`].
    #[inline]
    pub fn min(self, rhs: Self) -> Simd<T, N>
    where
        T: ComparisonSimdOperation<N>,
    {
        Simd(T::min(self.0, rhs.0))
    }

    /// Restrait the value of self between all value of each element of the
    /// min and max vectors.
    #[inline]
    pub fn clamp(self, min: Self, max: Self) -> Simd<T, N>
    where
        T: ComparisonSimdOperation<N>,
    {
        self.min(max).max(min)
    }

    /// Calculates `self + rhs`.
    ///
    /// Returns a tuple of the addition along with a boolean indicating whether
    /// an arithmetic overflow would occur. If an overflow would have occurred
    /// then the wrapped value is returned.
    #[inline]
    pub fn overflowing_add(self, rhs: Self) -> (Self, bool)
    where
        T: CheckIntOverflowSimd<N>,
    {
        let (sum, overflow) = T::overflowing_add(self.0, rhs.0);
        (Self(sum), overflow)
    }

    /// Wrapping (modular) addition. Computes self + rhs, wrapping around at
    /// the boundary of the type.
    #[inline]
    pub fn wrapping_add(self, rhs: Self) -> Self
    where
        T: CheckIntOverflowSimd<N>,
    {
        Self(T::wrapping_add(self.0, rhs.0))
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
        T: CheckIntOverflowSimd<N>,
    {
        Self(unsafe { T::unchecked_add(self.0, rhs.0) })
    }

    /// Checked addition. Computes `self + rhs`, returning None if overflow occurred.
    #[inline]
    pub fn checked_add(self, rhs: Self) -> Option<Self>
    where
        T: CheckIntOverflowSimd<N>,
    {
        T::checked_add(self.0, rhs.0).map(Self)
    }

    /// Calculates `lhs - rhs`.
    ///
    /// Returns a tuple of the subtraction along with a boolean indicating whether
    /// an arithmetic overflow would occur. If an overflow would have occurred
    /// then the wrapped value is returned.
    pub fn overflowing_sub(self, rhs: Self) -> (Self, bool)
    where
        T: CheckIntOverflowSimd<N>,
    {
        let (sum, overflow) = T::overflowing_sub(self.0, rhs.0);
        (Self(sum), overflow)
    }

    /// Wrapping (modular) subtraction. Computes `lhs - rhs`, wrapping around at
    /// the boundary of the type.
    pub fn wrapping_sub(self, rhs: Self) -> Self
    where
        T: CheckIntOverflowSimd<N>,
    {
        Self(T::wrapping_sub(self.0, rhs.0))
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
        T: CheckIntOverflowSimd<N>,
    {
        Self(unsafe { T::unchecked_sub(self.0, rhs.0) })
    }

    /// Checked subtraction. Computes `lhs - rhs`, returning None if overflow occurred.
    #[inline]
    pub fn checked_sub(self, rhs: Self) -> Option<Self>
    where
        T: CheckIntOverflowSimd<N>,
    {
        T::checked_sub(self.0, rhs.0).map(Self)
    }

    /// Calculates `lhs * rhs`.
    ///
    /// Returns a tuple of the multiplication along with a boolean indicating whether
    /// an arithmetic overflow would occur. If an overflow would have occurred
    /// then the wrapped value is returned.
    pub fn overflowing_mul(self, rhs: Self) -> (Self, bool)
    where
        T: CheckIntOverflowSimd<N>,
    {
        let (sum, overflow) = T::overflowing_mul(self.0, rhs.0);
        (Self(sum), overflow)
    }

    /// Wrapping (modular) multiplication. Computes `lhs * rhs`, wrapping around at
    /// the boundary of the type.
    pub fn wrapping_mul(self, rhs: Self) -> Self
    where
        T: CheckIntOverflowSimd<N>,
    {
        Self(T::wrapping_mul(self.0, rhs.0))
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
        T: CheckIntOverflowSimd<N>,
    {
        Self(unsafe { T::unchecked_mul(self.0, rhs.0) })
    }

    /// Checked multiplication. Computes `lhs * rhs`, returning None if overflow occurred.
    #[inline]
    pub fn checked_mul(self, rhs: Self) -> Option<Self>
    where
        T: CheckIntOverflowSimd<N>,
    {
        T::checked_mul(self.0, rhs.0).map(Self)
    }
}

impl<T: Debug, const N: usize> Debug for Simd<T, N>
where
    T: SimdElement<N>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        T::as_inner_ref(&self.0).fmt(f)
    }
}

impl<T, const N: usize> PartialEq for Simd<T, N>
where
    T: ComparisonSimdOperation<N>,
{
    fn eq(&self, other: &Self) -> bool {
        T::eq(self.0, other.0) == T::Mask::FULL_MASK
    }
}

impl<T, const N: usize> Eq for Simd<T, N> where T: ComparisonSimdOperation<N> {}

impl<T, const N: usize> Add for Simd<T, N>
where
    T: ArithmeticSimdOperation<N>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(T::add(self.0, rhs.0))
    }
}

impl<T, const N: usize> Sub for Simd<T, N>
where
    T: ArithmeticSimdOperation<N>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(T::sub(self.0, rhs.0))
    }
}

impl<T, const N: usize> Mul for Simd<T, N>
where
    T: ArithmeticSimdOperation<N>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(T::mul(self.0, rhs.0))
    }
}

impl<const N: usize> BitAnd for Simd<bool, N>
where
    bool: SimdElement<N>,
    <bool as SimdElement<N>>::NativeType: BitAnd<Output = <bool as SimdElement<N>>::NativeType>,
{
    type Output = Simd<bool, N>;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl<const N: usize> BitOr for Simd<bool, N>
where
    bool: SimdElement<N>,
    <bool as SimdElement<N>>::NativeType: BitOr<Output = <bool as SimdElement<N>>::NativeType>,
{
    type Output = Simd<bool, N>;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl<const N: usize> BitXor for Simd<bool, N>
where
    bool: SimdElement<N>,
    <bool as SimdElement<N>>::NativeType: BitXor<Output = <bool as SimdElement<N>>::NativeType>,
{
    type Output = Simd<bool, N>;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

/// Trait used as a marker to force the user
pub unsafe trait SimdElement<const N: usize>: Sealed + Copy {
    /// Represent the native simd type used internally
    type NativeType: Copy;

    /// Represent the type of the mask when the simd type is cast to a bool
    type Mask: MaskElement<N> + Copy;

    /// Represent the ref of the simd array
    type InnerRef: Debug + ?Sized;

    /// Returns an array reference containing the entire SIMD vector.
    fn as_inner_ref<'a>(value: &'a Self::NativeType) -> &'a Self::InnerRef;

    /// Store a vector into an array of `T`.
    ///
    /// # Safety
    ///
    /// Writing `mem_addr` must be safe, as if by [`std::ptr::write`].
    unsafe fn store<T>(mem_addr: *mut [T; N], value: Self::NativeType) {
        // SAFETY: the safety contract for `store` must be upheld by the caller
        unsafe {
            copy_nonoverlapping(
                (&raw const value) as *const u8,
                mem_addr as *mut u8,
                size_of::<Self::NativeType>(),
            );
        }
    }

    /// Loads a vector from an array of `T`.
    ///
    /// # Safety
    ///
    /// Reading `mem_addr` must be safe, as if by [`std::ptr::read`].
    unsafe fn load<T>(mem_addr: *const [T; N]) -> Self::NativeType {
        // SAFETY: create a empty destination correctly aligned (like, for __m256
        // that require a alignement of 32)
        let mut dst: Self::NativeType = unsafe { MaybeUninit::zeroed().assume_init() };
        // SAFETY: the safety contract for `load` must be upheld by the caller
        unsafe {
            copy_nonoverlapping(
                mem_addr as *const u8,
                (&raw mut dst) as *mut u8,
                size_of::<Self::NativeType>(),
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
        fn as_inner_ref<'a>(value: &'a u8) -> &'a u8 { value }
    }
}
generate_simd_support! {
    for [2 x bool] use u8 : not_native;
    type InnerRef = u8;
    type Mask = u8;
    impl base {
        fn as_inner_ref<'a>(value: &'a u8) -> &'a u8 { value }
    }
}
generate_simd_support! {
    for [4 x bool] use u8 : not_native;
    type InnerRef = u8;
    type Mask = u8;
    impl base {
        fn as_inner_ref<'a>(value: &'a u8) -> &'a u8 { value }
    }
}
generate_simd_support! {
    for [8 x bool] use u8 : not_native;
    type InnerRef = u8;
    type Mask = u8;
    impl base {
        fn as_inner_ref<'a>(value: &'a u8) -> &'a u8 { value }
    }
}

impl Sealed for bool {}
impl Sealed for u32 {}
impl Sealed for i32 {}
impl Sealed for f32 {}
