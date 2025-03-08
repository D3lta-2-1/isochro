use std::arch::x86_64::*;

use super::ArithmeticSimdMathOperation;
use super::macros::generate_simd_support;

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
#[repr(align(8))]
pub(crate) struct __m64i([i32; 2]);

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
#[repr(align(8))]
pub(crate) struct __m64([f32; 2]);

#[inline]
unsafe fn _mm64_add_epi32(lhs: __m64i, rhs: __m64i) -> __m64i {
    __m64i([lhs.0[0] + rhs.0[0], lhs.0[1] + rhs.0[1]])
}

#[inline]
unsafe fn _mm64_sub_epi32(lhs: __m64i, rhs: __m64i) -> __m64i {
    __m64i([lhs.0[0] - rhs.0[0], lhs.0[1] - rhs.0[1]])
}

#[inline]
unsafe fn _mm64_mul_epi32(lhs: __m64i, rhs: __m64i) -> __m64i {
    __m64i([lhs.0[0] * rhs.0[0], lhs.0[1] * rhs.0[1]])
}

#[inline]
unsafe fn _mm64_add_ps(lhs: __m64, rhs: __m64) -> __m64 {
    __m64([lhs.0[0] + rhs.0[0], lhs.0[1] + rhs.0[1]])
}

#[inline]
unsafe fn _mm64_sub_ps(lhs: __m64, rhs: __m64) -> __m64 {
    __m64([lhs.0[0] - rhs.0[0], lhs.0[1] - rhs.0[1]])
}

#[inline]
unsafe fn _mm64_mul_ps(lhs: __m64, rhs: __m64) -> __m64 {
    __m64([lhs.0[0] * rhs.0[0], lhs.0[1] * rhs.0[1]])
}

generate_simd_support! {
    for [2 x i32] use __m64i,
    impl trait ArithmeticSimdMathOperation {
        unsafe fn add(lhs: __m64i, rhs: __m64i) -> __m64i = _mm64_add_epi32,
        unsafe fn sub(lhs: __m64i, rhs: __m64i) -> __m64i = _mm64_sub_epi32,
        unsafe fn mul(lhs: __m64i, rhs: __m64i) -> __m64i = _mm64_mul_epi32,
    }
}

generate_simd_support! {
    for [2 x u32] use __m64i,
    impl trait ArithmeticSimdMathOperation {
        unsafe fn add(lhs: __m64i, rhs: __m64i) -> __m64i = _mm64_add_epi32,
        unsafe fn sub(lhs: __m64i, rhs: __m64i) -> __m64i = _mm64_sub_epi32,
        unsafe fn mul(lhs: __m64i, rhs: __m64i) -> __m64i = _mm64_mul_epi32,
    }
}

generate_simd_support! {
    for [2 x f32] use __m64,
    impl trait ArithmeticSimdMathOperation {
        unsafe fn add(lhs: __m64, rhs: __m64) -> __m64 = _mm64_add_ps,
        unsafe fn sub(lhs: __m64, rhs: __m64) -> __m64 = _mm64_sub_ps,
        unsafe fn mul(lhs: __m64, rhs: __m64) -> __m64 = _mm64_mul_ps,
    }
}

generate_simd_support! {
    for [4 x i32] use __m128i,
    impl trait ArithmeticSimdMathOperation {
        unsafe fn add(lhs: __m128i, rhs: __m128i) -> __m128i = _mm_add_epi32,
        unsafe fn sub(lhs: __m128i, rhs: __m128i) -> __m128i = _mm_sub_epi32,
        unsafe fn mul(lhs: __m128i, rhs: __m128i) -> __m128i = _mm_mul_epi32,
    }
}

generate_simd_support! {
    for [4 x u32] use __m128i,
    impl trait ArithmeticSimdMathOperation {
        unsafe fn add(lhs: __m128i, rhs: __m128i) -> __m128i = _mm_add_epi32,
        unsafe fn sub(lhs: __m128i, rhs: __m128i) -> __m128i = _mm_sub_epi32,
        unsafe fn mul(lhs: __m128i, rhs: __m128i) -> __m128i = _mm_mul_epi32,
    }
}

generate_simd_support! {
    for [4 x f32] use __m128,
    impl trait ArithmeticSimdMathOperation {
        unsafe fn add(lhs: __m128, rhs: __m128) -> __m128 = _mm_add_ps,
        unsafe fn sub(lhs: __m128, rhs: __m128) -> __m128 = _mm_sub_ps,
        unsafe fn mul(lhs: __m128, rhs: __m128) -> __m128 = _mm_mul_ps,
    }
}

generate_simd_support! {
    for [8 x i32] use __m256i,
    impl trait ArithmeticSimdMathOperation {
        unsafe fn add(lhs: __m256i, rhs: __m256i) -> __m256i = _mm256_add_epi32,
        unsafe fn sub(lhs: __m256i, rhs: __m256i) -> __m256i = _mm256_sub_epi32,
        unsafe fn mul(lhs: __m256i, rhs: __m256i) -> __m256i = _mm256_mul_epi32,
    }
}

generate_simd_support! {
    for [8 x u32] use __m256i,
    impl trait ArithmeticSimdMathOperation {
        unsafe fn add(lhs: __m256i, rhs: __m256i) -> __m256i = _mm256_add_epi32,
        unsafe fn sub(lhs: __m256i, rhs: __m256i) -> __m256i = _mm256_sub_epi32,
        unsafe fn mul(lhs: __m256i, rhs: __m256i) -> __m256i = _mm256_mul_epi32,
    }
}

generate_simd_support! {
    for [8 x f32] use __m256,
    impl trait ArithmeticSimdMathOperation {
        unsafe fn add(lhs: __m256, rhs: __m256) -> __m256 = _mm256_add_ps,
        unsafe fn sub(lhs: __m256, rhs: __m256) -> __m256 = _mm256_sub_ps,
        unsafe fn mul(lhs: __m256, rhs: __m256) -> __m256 = _mm256_mul_ps,
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
