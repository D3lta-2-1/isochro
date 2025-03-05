#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::marker::PhantomData;
use std::mem::transmute;
use std::ops::Add;

#[cfg(target_arch = "x86_64")]
#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
// We plan to use `repr(simd)` in future versions of Rust (when it is stabilized)
// to allow the compiler to generate optimization and simd usage more easily.
// #[repr(simd)]
#[repr(transparent)]
struct __m64([f32; 2]);

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Simd<T, const N: usize>(<LaneCount<T, N> as SupportedNativeSimd>::RelativeSimdType)
where
    LaneCount<T, N>: SupportedNativeSimd;

impl<T, const N: usize> Simd<T, N>
where
    LaneCount<T, N>: SupportedNativeSimd,
{
    /// Loads a vector from an array of `T`.
    ///
    /// # Safety
    ///
    /// Reading `ptr` must be safe, as if by `<*const [T; N]>::read`.
    #[inline]
    const unsafe fn load(ptr: *const [T; N]) -> Self {
        let mut tmp = core::mem::MaybeUninit::<Self>::uninit();
        // SAFETY: `Simd<T, N>` always contains `N` elements of type `T`.  It may have padding
        // which does not need to be initialized.  The safety of reading `ptr` is ensured by the
        // caller.
        // 
        // FIXME: find a way to force the compiler to ensure the size of a
        // `RelativeSimdType` to be the size of an `[T; N]`
        unsafe {
            core::ptr::copy_nonoverlapping(ptr, tmp.as_mut_ptr().cast(), 1);
            tmp.assume_init()
        }
    }

    pub fn from_array(array: [T; N]) -> Self {
        // SAFETY: `&array` is safe to read.
        unsafe { Self::load(&array) }
    }
}

impl<T, const N: usize> Add for Simd<T, N>
where
    LaneCount<T, N>: SupportedNativeSimd,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(<LaneCount<T, N> as SupportedNativeSimd>::add(self.0, rhs.0))
    }
}

/// Specifies the number of lanes in a SIMD vector as a type.
struct LaneCount<T, const N: usize>(PhantomData<T>);

/// Trait used as a marker to force the user
trait SupportedNativeSimd: Sized {
    type RelativeSimdType: Copy;

    fn add(lhs: Self::RelativeSimdType, rhs: Self::RelativeSimdType) -> Self::RelativeSimdType;
}

impl SupportedNativeSimd for LaneCount<f32, 1> {
    type RelativeSimdType = f32;

    #[inline(always)]
    fn add(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}
#[cfg(target_arch = "x86_64")]
impl SupportedNativeSimd for LaneCount<f32, 2> {
    type RelativeSimdType = __m64;

    #[inline(always)]
    fn add(lhs: __m64, rhs: __m64) -> __m64 {
        // SAFETY: we build a `__m128` vector via `_mm_set_ps` and
        unsafe {
            let xmm0 = _mm_set_ps(0., 0., lhs.0[1], lhs.0[0]);
            let xmm1 = _mm_set_ps(0., 0., rhs.0[1], rhs.0[0]);

            let xmm0 = _mm_add_ps(xmm0, xmm1);

            __m64([
                transmute(_mm_extract_ps::<0>(xmm0)),
                transmute(_mm_extract_ps::<1>(xmm0)),
            ])
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl SupportedNativeSimd for LaneCount<f32, 4> {
    type RelativeSimdType = __m128;

    #[inline(always)]
    fn add(lhs: __m128, rhs: __m128) -> __m128 {
        // Safety: lhs and rhs are vectors
        unsafe { _mm_add_ps(lhs, rhs) }
    }
}

#[cfg(target_arch = "x86_64")]
impl SupportedNativeSimd for LaneCount<f32, 8> {
    type RelativeSimdType = __m256;

    #[inline(always)]
    fn add(lhs: __m256, rhs: __m256) -> __m256 {
        // Safety: lhs and rhs are vectors
        unsafe { _mm256_add_ps(lhs, rhs) }
    }
}

// TODO: add a way to use `_mm512_add_ps` (it's instable)
// #[cfg(target_arch = "x86_64")]
// #[cfg(target_feature = "avx512f")]
// impl SupportedNativeSimd for LaneCount<f32, 16> {
//     type RelativeSimdType = __m512;

//     fn add(lhs: __m512, rhs: __m512) -> __m512 {
//         unsafe {
//             _mm512_add_ps(lhs, rhs)
//         }
//     }
// }
