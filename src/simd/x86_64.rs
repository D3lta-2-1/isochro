use std::arch::x86_64::*;
use std::mem::transmute;

use super::{LaneCount, SupportedNativeSimd};

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
// We plan to use `repr(simd)` in future versions of Rust (when it is stabilized)
// to allow the compiler to generate optimization and simd usage more easily.
// #[repr(simd)]
#[repr(transparent)]
pub(crate) struct __m64([f32; 2]);

impl SupportedNativeSimd<f32, 2> for LaneCount<f32, 2> {
    type RelativeSimdType = __m64;

    #[inline]
    unsafe fn add(lhs: __m64, rhs: __m64) -> __m64 {
        // SAFETY: we build a `__m128` vector via `_mm_set_ps` and `__m64` is a
        // vector correctly defined.
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

impl SupportedNativeSimd<f32, 4> for LaneCount<f32, 4> {
    type RelativeSimdType = __m128;

    #[inline]
    unsafe fn load(ptr: *const [f32; 4]) -> Self::RelativeSimdType {
        // The safety of reading `ptr` is ensured by the caller.
        unsafe { _mm_load_ps(ptr.cast()) }
    }

    #[inline]
    unsafe fn add(lhs: __m128, rhs: __m128) -> __m128 {
        // SAFETY: lhs and rhs are vectors
        unsafe { _mm_add_ps(lhs, rhs) }
    }
}

impl SupportedNativeSimd<f32, 8> for LaneCount<f32, 8> {
    type RelativeSimdType = __m256;

    #[inline]
    unsafe fn load(ptr: *const [f32; 8]) -> Self::RelativeSimdType {
        // The safety of reading `ptr` is ensured by the caller.
        unsafe { _mm256_load_ps(ptr.cast()) }
    }

    #[inline]
    unsafe fn add(lhs: __m256, rhs: __m256) -> __m256 {
        // SAFETY: lhs and rhs are vectors
        unsafe { _mm256_add_ps(lhs, rhs) }
    }
}

// TODO: add a way to use `_mm512_add_ps` (it's instable)
// #[cfg(target_feature = "avx512f")]
// impl SupportedNativeSimd<f32, 16> for LaneCount<f32, 16> {
//     type RelativeSimdType = __m512;

//     unsafe fn add(lhs: __m512, rhs: __m512) -> __m512 {
//         unsafe {
//             _mm512_add_ps(lhs, rhs)
//         }
//     }
// }
