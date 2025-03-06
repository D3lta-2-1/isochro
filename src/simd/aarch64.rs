use std::arch::aarch64::*;

use super::{LaneCount, SupportedNativeSimd};

impl SupportedNativeSimd<f32, 2> for LaneCount<f32, 2> {
    type RelativeSimdType = float32x2_t;

    #[inline]
    unsafe fn add(lhs: float32x2_t, rhs: float32x2_t) -> float32x2_t {
        // SAFETY: lhs and rhs are vectors
        unsafe { vadd_f32(lhs, rhs) }
    }
}

impl SupportedNativeSimd<f32, 4> for LaneCount<f32, 4> {
    type RelativeSimdType = float32x4_t;

    #[inline]
    unsafe fn add(lhs: float32x4_t, rhs: float32x4_t) -> float32x4_t {
        // SAFETY: lhs and rhs are vectors
        unsafe { vaddq_f32(lhs, rhs) }
    }
}

impl SupportedNativeSimd<f32, 8> for LaneCount<f32, 8> {
    // due to the inexistance of float32x8_t, we store it in a float32x4x2_t
    // (it's a storage only type, not a simd type)
    type RelativeSimdType = float32x4x2_t;

    #[inline]
    unsafe fn add(lhs: float32x4x2_t, rhs: float32x4x2_t) -> float32x4x2_t {
        // SAFETY: lhs and rhs are vectors
        unsafe { float32x4x2_t(
            vaddq_f32(lhs.0, rhs.0),
            vaddq_f32(lhs.1, rhs.1),
        ) }
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
