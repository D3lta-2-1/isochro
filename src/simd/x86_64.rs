use std::arch::x86_64::*;

use super::macros::generate_simd_support;
use super::{LaneCount, SupportedNativeSimd};

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
#[repr(align(8))]
pub(crate) struct __m64([f32; 2]);

impl SupportedNativeSimd<f32, 2> for LaneCount<f32, 2> {
    type RelativeSimdType = __m64;

    #[inline]
    unsafe fn add(lhs: __m64, rhs: __m64) -> __m64 {
        __m64([
            lhs.0[0] + rhs.0[0],
            lhs.0[1] + rhs.0[1],
        ])
    }
}

generate_simd_support! {
    for [4 x f32] use __m128,
    fn add(lhs: __m128, rhs: __m128) -> __m128 = _mm_add_ps,
}

generate_simd_support! {
    for [8 x f32] use __m256,
    fn add(lhs: __m256, rhs: __m256) -> __m256 = _mm256_add_ps,
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
