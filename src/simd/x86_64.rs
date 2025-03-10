use std::arch::x86_64::*;
use std::mem::transmute;

use super::boundaries::ComparisonSimdOperation;
use super::macros::{generate_simd_support, overflowing_check};
use super::{ArithmeticSimdOperation, CheckIntOverflowSimd};

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
#[repr(align(8))]
pub struct __m64i([i32; 2]);

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
#[repr(align(8))]
pub struct __m64u([u32; 2]);

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
#[repr(align(8))]
pub struct __m64([f32; 2]);

generate_simd_support! {
    for [2 x i32] use __m64i;
    type InnerRef = [i32];
    type Mask = u8;
    impl base {
        fn as_inner_ref<'a>(value: &'a __m64i) -> &'a [i32] {
            &value.0
        }
    }
    impl trait CheckIntOverflowSimd {
        fn overflowing_add(lhs: __m64i, rhs: __m64i) -> (__m64i, bool) {
            let (first, o1) = lhs.0[0].overflowing_add(rhs.0[0]);
            let (second, o2) = lhs.0[1].overflowing_add(rhs.0[1]);

            (__m64i([first, second]), o1 || o2)
        }

        fn wrapping_add(lhs: __m64i, rhs: __m64i) -> __m64i {
            __m64i([
                lhs.0[0].wrapping_add(rhs.0[0]),
                lhs.0[0].wrapping_add(rhs.0[0])
            ])
        }

        fn overflowing_sub(lhs: __m64i, rhs: __m64i) -> (__m64i, bool) {
            let (first, o1) = lhs.0[0].overflowing_sub(rhs.0[0]);
            let (second, o2) = lhs.0[1].overflowing_sub(rhs.0[1]);

            (__m64i([first, second]), o1 || o2)
        }

        fn wrapping_sub(lhs: __m64i, rhs: __m64i) -> __m64i {
            __m64i([
                lhs.0[0].wrapping_sub(rhs.0[0]),
                lhs.0[0].wrapping_sub(rhs.0[0])
            ])
        }

        fn overflowing_mul(lhs: __m64i, rhs: __m64i) -> (__m64i, bool) {
            let (first, o1) = lhs.0[0].overflowing_mul(rhs.0[0]);
            let (second, o2) = lhs.0[1].overflowing_mul(rhs.0[1]);

            (__m64i([first, second]), o1 || o2)
        }

        fn wrapping_mul(lhs: __m64i, rhs: __m64i) -> __m64i {
            __m64i([
                lhs.0[0].wrapping_mul(rhs.0[0]),
                lhs.0[0].wrapping_mul(rhs.0[0])
            ])
        }
    }
    impl trait ComparisonSimdOperation {
        fn cmp(lhs: __m64i, rhs: __m64i) -> __m64i {
            __m64i([
                lhs.0[0].cmp(&rhs.0[0]) as i32,
                lhs.0[1].cmp(&rhs.0[1]) as i32,
            ])
        }

        fn eq(lhs: __m64i, rhs: __m64i) -> u8 {
            (lhs.0[0] == rhs.0[0]) as u8 + ((lhs.0[1] == rhs.0[1]) as u8) << 1
        }
        fn gt(lhs: __m64i, rhs: __m64i) -> u8 {
            (lhs.0[0] > rhs.0[0]) as u8 + ((lhs.0[1] > rhs.0[1]) as u8) << 1
        }
        fn lt(lhs: __m64i, rhs: __m64i) -> u8 {
            (lhs.0[0] < rhs.0[0]) as u8 + ((lhs.0[1] < rhs.0[1]) as u8) << 1
        }
    }
}

generate_simd_support! {
    for [2 x u32] use __m64u;
    type InnerRef = [u32];
    type Mask = u8;
    impl base {
        fn as_inner_ref<'a>(value: &'a __m64u) -> &'a [u32] {
            &value.0
        }
    }
    impl trait CheckIntOverflowSimd {
        fn overflowing_add(lhs: __m64u, rhs: __m64u) -> (__m64u, bool) {
            let (first, o1) = lhs.0[0].overflowing_add(rhs.0[0]);
            let (second, o2) = lhs.0[1].overflowing_add(rhs.0[1]);

            (__m64u([first, second]), o1 || o2)
        }

        fn wrapping_add(lhs: __m64u, rhs: __m64u) -> __m64u {
            __m64u([
                lhs.0[0].wrapping_add(rhs.0[0]),
                lhs.0[0].wrapping_add(rhs.0[0])
            ])
        }

        fn overflowing_sub(lhs: __m64u, rhs: __m64u) -> (__m64u, bool) {
            let (first, o1) = lhs.0[0].overflowing_sub(rhs.0[0]);
            let (second, o2) = lhs.0[1].overflowing_sub(rhs.0[1]);

            (__m64u([first, second]), o1 || o2)
        }

        fn wrapping_sub(lhs: __m64u, rhs: __m64u) -> __m64u {
            __m64u([
                lhs.0[0].wrapping_sub(rhs.0[0]),
                lhs.0[0].wrapping_sub(rhs.0[0])
            ])
        }

        fn overflowing_mul(lhs: __m64u, rhs: __m64u) -> (__m64u, bool) {
            let (first, o1) = lhs.0[0].overflowing_mul(rhs.0[0]);
            let (second, o2) = lhs.0[1].overflowing_mul(rhs.0[1]);

            (__m64u([first, second]), o1 || o2)
        }

        fn wrapping_mul(lhs: __m64u, rhs: __m64u) -> __m64u {
            __m64u([
                lhs.0[0].wrapping_mul(rhs.0[0]),
                lhs.0[0].wrapping_mul(rhs.0[0])
            ])
        }
    }
    impl trait ComparisonSimdOperation {
        fn cmp(lhs: __m64u, rhs: __m64u) -> __m64u {
            __m64u([
                lhs.0[0].cmp(&rhs.0[0]) as u32,
                lhs.0[1].cmp(&rhs.0[1]) as u32,
            ])
        }

        fn eq(lhs: __m64u, rhs: __m64u) -> u8 {
            (lhs.0[0] == rhs.0[0]) as u8 + ((lhs.0[1] == rhs.0[1]) as u8) << 1
        }
        fn gt(lhs: __m64u, rhs: __m64u) -> u8 {
            (lhs.0[0] > rhs.0[0]) as u8 + ((lhs.0[1] > rhs.0[1]) as u8) << 1
        }
        fn lt(lhs: __m64u, rhs: __m64u) -> u8 {
            (lhs.0[0] < rhs.0[0]) as u8 + ((lhs.0[1] < rhs.0[1]) as u8) << 1
        }
    }
}

generate_simd_support! {
    for [2 x f32] use __m64;
    type InnerRef = [f32];
    type Mask = u8;
    impl base {
        fn as_inner_ref<'a>(value: &'a __m64) -> &'a [f32] {
            &value.0
        }
    }
    impl trait ArithmeticSimdOperation {
        fn add(lhs: __m64, rhs: __m64) -> __m64 {
            __m64([lhs.0[0] + rhs.0[0], lhs.0[1] + rhs.0[1]])
        }
        fn sub(lhs: __m64, rhs: __m64) -> __m64 {
            __m64([lhs.0[0] - rhs.0[0], lhs.0[1] - rhs.0[1]])
        }
        fn mul(lhs: __m64, rhs: __m64) -> __m64 {
            __m64([lhs.0[0] * rhs.0[0], lhs.0[1] * rhs.0[1]])
        }
    }
    impl trait ComparisonSimdOperation {
        fn cmp(lhs: __m64, rhs: __m64) -> __m64 {
            // SAFETY: i32/u32 is the same size as f32 and we copy the result of
            // the simd x86 comparison
            unsafe { __m64([
                transmute(lhs.0[0].total_cmp(&rhs.0[0]) as i32),
                transmute(lhs.0[1].total_cmp(&rhs.0[1]) as i32),
            ]) }
        }

        fn eq(lhs: __m64, rhs: __m64) -> u8 {
            (lhs.0[0] == rhs.0[0]) as u8 + ((lhs.0[1] == rhs.0[1]) as u8) << 1
        }
        fn gt(lhs: __m64, rhs: __m64) -> u8 {
            (lhs.0[0] > rhs.0[0]) as u8 + ((lhs.0[1] > rhs.0[1]) as u8) << 1
        }
        fn lt(lhs: __m64, rhs: __m64) -> u8 {
            (lhs.0[0] < rhs.0[0]) as u8 + ((lhs.0[1] < rhs.0[1]) as u8) << 1
        }
    }
}

generate_simd_support! {
    for [4 x i32] use __m128i;
    type InnerRef = [i32];
    type Mask = u8;
    impl base {
        fn as_inner_ref<'a>(value: &'a __m128i) -> &'a [i32] {
            // SAFETY: value is valid and correctly aligned and containe the
            // right number of elements
            unsafe { std::slice::from_raw_parts(
                value as *const __m128i as *const i32,
                4
            ) }
        }
    }
    impl trait CheckIntOverflowSimd {
        fn overflowing_add(lhs: __m128i, rhs: __m128i) -> (__m128i, bool) {
            unsafe { overflowing_check!(
                add signed(i32),
                lhs,
                rhs,
                _mm_set1_epi32,
                _mm_add_epi32,
                _mm_sub_epi32,
                _mm_cmpgt_epi32,
                _mm_mask_epi32,
            ) }
        }

        fn wrapping_add(lhs: __m128i, rhs: __m128i) -> __m128i {
            unsafe { _mm_add_epi32(lhs, rhs) }
        }

        fn overflowing_sub(lhs: __m128i, rhs: __m128i) -> (__m128i, bool) {
            unsafe { overflowing_check!(
                sub signed(i32),
                lhs,
                rhs,
                _mm_set1_epi32,
                _mm_add_epi32,
                _mm_sub_epi32,
                _mm_cmpgt_epi32,
                _mm_mask_epi32,
            ) }
        }

        fn wrapping_sub(lhs: __m128i, rhs: __m128i) -> __m128i {
            unsafe { _mm_sub_epi32(lhs, rhs) }
        }

        fn overflowing_mul(lhs: __m128i, rhs: __m128i) -> (__m128i, bool) {
            unsafe { overflowing_check!(
                mul signed(i32),
                lhs,
                rhs,
                _mm_set1_epi32,
                _mm_sign_abs_epi32,
                _mm_add_epi32,
                _mm_mul_epi32,
                _mm_and_si128,
                _mm_srli_epi32,
                _mm_slli_epi32,
                _mm_cmpgt_epi32,
                _mm_mask_epi32,
                _mm_mask_epi32,
            ) }
        }

        fn wrapping_mul(lhs: __m128i, rhs: __m128i) -> __m128i {
            unsafe { _mm_mul_epi32(lhs, rhs) }
        }
    }
    impl trait ComparisonSimdOperation {
        fn cmp(lhs: __m128i, rhs: __m128i) -> __m128i {
            unsafe { _mm_add_epi32(
                _mm_mul_epi32(_mm_cmpgt_epu32(lhs, rhs), _mm_set1_epi32(-1)),
                _mm_cmpgt_epu32(rhs, lhs)
            ) }
        }

        fn eq(lhs: __m128i, rhs: __m128i) -> u8 {
            unsafe { _mm_mask_epi32(_mm_cmpeq_epi32(lhs, rhs)) }
        }
        fn gt(lhs: __m128i, rhs: __m128i) -> u8 {
            unsafe { _mm_mask_epi32(_mm_cmpgt_epu32(lhs, rhs)) }
        }
        fn lt(lhs: __m128i, rhs: __m128i) -> u8 {
            unsafe { _mm_mask_epi32(_mm_cmpgt_epu32(rhs, lhs)) }
        }
    }
}

generate_simd_support! {
    for [4 x u32] use __m128i : not_native;
    type InnerRef = [u32];
    type Mask = u8;
    impl base {
        fn as_inner_ref<'a>(value: &'a __m128i) -> &'a [u32] {
            // SAFETY: value is valid and correctly aligned and containe the
            // right number of elements
            unsafe { std::slice::from_raw_parts(
                value as *const __m128i as *const u32,
                4
            ) }
        }
    }
    impl trait CheckIntOverflowSimd {
        fn overflowing_add(lhs: __m128i, rhs: __m128i) -> (__m128i, bool) {
            // due to the bitly equivalence of i32 and u32, I use
            // the i32 version add, sub, ...
            unsafe { overflowing_check!(
                add unsigned(u32),
                lhs,
                rhs,
                _mm_set1_epu32,
                _mm_add_epi32,
                _mm_sub_epi32,
                _mm_cmpgt_epu32,
                _mm_mask_epi32,
            ) }
        }

        fn wrapping_add(lhs: __m128i, rhs: __m128i) -> __m128i {
            unsafe { _mm_add_epi32(lhs, rhs) }
        }

        fn overflowing_sub(lhs: __m128i, rhs: __m128i) -> (__m128i, bool) {
            unsafe { overflowing_check!(
                sub unsigned(u32),
                lhs,
                rhs,
                _mm_set1_epu32,
                _mm_add_epi32,
                _mm_sub_epi32,
                _mm_cmpgt_epu32,
                _mm_mask_epi32,
            ) }
        }

        fn wrapping_sub(lhs: __m128i, rhs: __m128i) -> __m128i {
            unsafe { _mm_sub_epi32(lhs, rhs) }
        }

        fn overflowing_mul(lhs: __m128i, rhs: __m128i) -> (__m128i, bool) {
            unsafe { overflowing_check!(
                mul unsigned(u32),
                lhs,
                rhs,
                _mm_set1_epu32,
                _mm_add_epi32,
                _mm_mul_epi32,
                _mm_and_si128,
                _mm_srli_epi32,
                _mm_slli_epi32,
                _mm_cmpgt_epu32,
                _mm_mask_epi32,
            ) }
        }

        fn wrapping_mul(lhs: __m128i, rhs: __m128i) -> __m128i {
            unsafe { _mm_mul_epi32(lhs, rhs) }
        }
    }
    impl trait ComparisonSimdOperation {
        fn cmp(lhs: __m128i, rhs: __m128i) -> __m128i {
            unsafe { _mm_add_epi32(
                _mm_mul_epi32(_mm_cmpgt_epu32(lhs, rhs), _mm_set1_epi32(-1)),
                _mm_cmpgt_epu32(rhs, lhs)
            ) }
        }

        fn eq(lhs: __m128i, rhs: __m128i) -> u8 {
            unsafe { _mm_mask_epi32(_mm_cmpeq_epi32(lhs, rhs)) }
        }
        fn gt(lhs: __m128i, rhs: __m128i) -> u8 {
            unsafe { _mm_mask_epi32(_mm_cmpgt_epu32(lhs, rhs)) }
        }
        fn lt(lhs: __m128i, rhs: __m128i) -> u8 {
            unsafe { _mm_mask_epi32(_mm_cmpgt_epu32(rhs, lhs)) }
        }
    }
}

generate_simd_support! {
    for [4 x f32] use __m128;
    type InnerRef = [f32];
    type Mask = u8;
    impl base {
        fn as_inner_ref<'a>(value: &'a __m128) -> &'a [f32] {
            // SAFETY: value is valid and correctly aligned and containe the
            // right number of elements
            unsafe { std::slice::from_raw_parts(
                value as *const __m128 as *const f32,
                4
            ) }
        }
    }
    impl trait ArithmeticSimdOperation {
        fn add(lhs: __m128, rhs: __m128) -> __m128 {
            unsafe { _mm_add_ps(lhs, rhs) }
        }
        fn sub(lhs: __m128, rhs: __m128) -> __m128 {
            unsafe { _mm_sub_ps(lhs, rhs) }
        }
        fn mul(lhs: __m128, rhs: __m128) -> __m128 {
            unsafe { _mm_mul_ps(lhs, rhs) }
        }
    }
    impl trait ComparisonSimdOperation {
        fn cmp(lhs: __m128, rhs: __m128) -> __m128 {
            unsafe {
                // transform all NaN into 1 and 0 into 0 by using: ((v as i32) * -1) as f32
                let v = transmute(_mm_cmpgt_ps(lhs, rhs));
                let greater = transmute(_mm_mul_epi32(v, _mm_set1_epi32(-1)));
                _mm_add_ps(greater, _mm_cmpgt_ps(rhs, lhs))
            }
        }

        fn eq(lhs: __m128, rhs: __m128) -> u8 {
            unsafe { _mm_mask_ps(_mm_cmpeq_ps(lhs, rhs)) }
        }
        fn gt(lhs: __m128, rhs: __m128) -> u8 {
            unsafe { _mm_mask_ps(_mm_cmpgt_ps(lhs, rhs)) }
        }
        fn lt(lhs: __m128, rhs: __m128) -> u8 {
            unsafe { _mm_mask_ps(_mm_cmpgt_ps(rhs, lhs)) }
        }
    }
}

generate_simd_support! {
    for [8 x i32] use __m256i;
    type InnerRef = [i32];
    type Mask = u8;
    impl base {
        fn as_inner_ref<'a>(value: &'a __m256i) -> &'a [i32] {
            // SAFETY: value is valid and correctly aligned and containe the
            // right number of elements
            unsafe { std::slice::from_raw_parts(
                value as *const __m256i as *const i32,
                8
            ) }
        }
    }
    impl trait CheckIntOverflowSimd {
        fn overflowing_add(lhs: __m256i, rhs: __m256i) -> (__m256i, bool) {
            unsafe { overflowing_check!(
                add signed(i32),
                lhs,
                rhs,
                _mm256_set1_epi32,
                _mm256_add_epi32,
                _mm256_sub_epi32,
                _mm256_cmpgt_epi32,
                _mm256_mask_epi32,
            ) }
        }

        fn wrapping_add(lhs: __m256i, rhs: __m256i) -> __m256i {
            unsafe { _mm256_add_epi32(lhs, rhs) }
        }

        fn overflowing_sub(lhs: __m256i, rhs: __m256i) -> (__m256i, bool) {
            unsafe { overflowing_check!(
                sub signed(i32),
                lhs,
                rhs,
                _mm256_set1_epi32,
                _mm256_add_epi32,
                _mm256_sub_epi32,
                _mm256_cmpgt_epi32,
                _mm256_mask_epi32,
            ) }
        }

        fn wrapping_sub(lhs: __m256i, rhs: __m256i) -> __m256i {
            unsafe { _mm256_sub_epi32(lhs, rhs) }
        }

        fn overflowing_mul(lhs: __m256i, rhs: __m256i) -> (__m256i, bool) {
            unsafe { overflowing_check!(
                mul signed(i32),
                lhs,
                rhs,
                _mm256_set1_epi32,
                _mm256_sign_abs_epi32,
                _mm256_add_epi32,
                _mm256_mul_epi32,
                _mm256_and_si256,
                _mm256_srli_epi32,
                _mm256_slli_epi32,
                _mm256_cmpgt_epi32,
                _mm256_mask_epi32,
                _mm256_mask_epi32,
            ) }
        }

        fn wrapping_mul(lhs: __m256i, rhs: __m256i) -> __m256i {
            unsafe { _mm256_mul_epi32(lhs, rhs) }
        }
    }
    impl trait ComparisonSimdOperation {
        fn cmp(lhs: __m256i, rhs: __m256i) -> __m256i {
            unsafe { _mm256_add_epi32(
                _mm256_mul_epi32(_mm256_cmpgt_epu32(lhs, rhs), _mm256_set1_epi32(-1)),
                _mm256_cmpgt_epu32(rhs, lhs)
            ) }
        }

        fn eq(lhs: __m256i, rhs: __m256i) -> u8 {
            unsafe { _mm256_mask_epi32(_mm256_cmpeq_epi32(lhs, rhs)) }
        }
        fn gt(lhs: __m256i, rhs: __m256i) -> u8 {
            unsafe { _mm256_mask_epi32(_mm256_cmpgt_epu32(lhs, rhs)) }
        }
        fn lt(lhs: __m256i, rhs: __m256i) -> u8 {
            unsafe { _mm256_mask_epi32(_mm256_cmpgt_epu32(rhs, lhs)) }
        }
    }
}

generate_simd_support! {
    for [8 x u32] use __m256i : not_native;
    type InnerRef = [u32];
    type Mask = u8;
    impl base {
        fn as_inner_ref<'a>(value: &'a __m256i) -> &'a [u32] {
            // SAFETY: value is valid and correctly aligned and containe the
            // right number of elements
            unsafe { std::slice::from_raw_parts(
                value as *const __m256i as *const u32,
                8
            ) }
        }
    }
    impl trait CheckIntOverflowSimd {
        fn overflowing_add(lhs: __m256i, rhs: __m256i) -> (__m256i, bool) {
            // due to the bitly equivalence of i32 and u32, I use
            // the i32 version add, sub, ...
            unsafe { overflowing_check!(
                add unsigned(u32),
                lhs,
                rhs,
                _mm256_set1_epu32,
                _mm256_add_epi32,
                _mm256_sub_epi32,
                _mm256_cmpgt_epu32,
                _mm256_mask_epi32,
            ) }
        }

        fn wrapping_add(lhs: __m256i, rhs: __m256i) -> __m256i {
            unsafe { _mm256_add_epi32(lhs, rhs) }
        }

        fn overflowing_sub(lhs: __m256i, rhs: __m256i) -> (__m256i, bool) {
            unsafe { overflowing_check!(
                sub unsigned(u32),
                lhs,
                rhs,
                _mm256_set1_epu32,
                _mm256_add_epi32,
                _mm256_sub_epi32,
                _mm256_cmpgt_epu32,
                _mm256_mask_epi32,
            ) }
        }

        fn wrapping_sub(lhs: __m256i, rhs: __m256i) -> __m256i {
            unsafe { _mm256_sub_epi32(lhs, rhs) }
        }

        fn overflowing_mul(lhs: __m256i, rhs: __m256i) -> (__m256i, bool) {
            unsafe { overflowing_check!(
                mul unsigned(u32),
                lhs,
                rhs,
                _mm256_set1_epu32,
                _mm256_add_epi32,
                _mm256_mul_epi32,
                _mm256_and_si256,
                _mm256_srli_epi32,
                _mm256_slli_epi32,
                _mm256_cmpgt_epu32,
                _mm256_mask_epi32,
            ) }
        }

        fn wrapping_mul(lhs: __m256i, rhs: __m256i) -> __m256i {
            unsafe { _mm256_mul_epi32(lhs, rhs) }
        }
    }
    impl trait ComparisonSimdOperation {
        fn cmp(lhs: __m256i, rhs: __m256i) -> __m256i {
            unsafe { _mm256_add_epi32(
                _mm256_mul_epi32(_mm256_cmpgt_epu32(lhs, rhs), _mm256_set1_epi32(-1)),
                _mm256_cmpgt_epu32(rhs, lhs)
            ) }
        }

        fn eq(lhs: __m256i, rhs: __m256i) -> u8 {
            unsafe { _mm256_mask_epi32(_mm256_cmpeq_epi32(lhs, rhs)) }
        }
        fn gt(lhs: __m256i, rhs: __m256i) -> u8 {
            unsafe { _mm256_mask_epi32(_mm256_cmpgt_epu32(lhs, rhs)) }
        }
        fn lt(lhs: __m256i, rhs: __m256i) -> u8 {
            unsafe { _mm256_mask_epi32(_mm256_cmpgt_epu32(rhs, lhs)) }
        }
    }
}

generate_simd_support! {
    for [8 x f32] use __m256;
    type InnerRef = [f32];
    type Mask = u8;
    impl base {
        fn as_inner_ref<'a>(value: &'a __m256) -> &'a [f32] {
            // SAFETY: value is valid and correctly aligned and containe the
            // right number of elements
            unsafe { std::slice::from_raw_parts(
                value as *const __m256 as *const f32,
                8
            ) }
        }
    }
    impl trait ArithmeticSimdOperation {
        fn add(lhs: __m256, rhs: __m256) -> __m256 {
            unsafe { _mm256_add_ps(lhs, rhs) }
        }
        fn sub(lhs: __m256, rhs: __m256) -> __m256 {
            unsafe { _mm256_sub_ps(lhs, rhs) }
        }
        fn mul(lhs: __m256, rhs: __m256) -> __m256 {
            unsafe { _mm256_mul_ps(lhs, rhs) }
        }
    }
    impl trait ComparisonSimdOperation {
        fn cmp(lhs: __m256, rhs: __m256) -> __m256 {
            unsafe {
                // transform all NaN into 1 and 0 into 0 by using: ((v as i32) * -1) as f32
                let v = transmute(_mm256_cmp_ps::<_CMP_GT_OQ>(lhs, rhs));
                let greater = transmute(_mm256_mul_epi32(v, _mm256_set1_epi32(-1)));
                _mm256_add_ps(greater, _mm256_cmp_ps::<_CMP_LT_OQ>(lhs, rhs))
            }
        }

        fn eq(lhs: __m256, rhs: __m256) -> u8 {
            unsafe { _mm256_mask_ps(_mm256_cmp_ps::<_CMP_EQ_OQ>(lhs, rhs)) }
        }
        fn gt(lhs: __m256, rhs: __m256) -> u8 {
            unsafe { _mm256_mask_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(lhs, rhs)) }
        }
        fn lt(lhs: __m256, rhs: __m256) -> u8 {
            unsafe { _mm256_mask_ps(_mm256_cmp_ps::<_CMP_LT_OQ>(lhs, rhs)) }
        }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn _mm_sign_abs_epi32(v: __m128i) -> (__m128i, __m128i) {
    (_mm_sign_epi32(_mm_set1_epi32(-1), v), _mm_abs_epi32(v))
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn _mm256_sign_abs_epi32(v: __m256i) -> (__m256i, __m256i) {
    (
        _mm256_sign_epi32(_mm256_set1_epi32(-1), v),
        _mm256_abs_epi32(v),
    )
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn _mm_mask_epi32(v: __m128i) -> u8 {
    (((_mm_extract_epi32::<3>(v) != 0) as u8) << 3)
        + (((_mm_extract_epi32::<2>(v) != 0) as u8) << 2)
        + (((_mm_extract_epi32::<1>(v) != 0) as u8) << 1)
        + ((_mm_extract_epi32::<0>(v) != 0) as u8)
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn _mm_mask_ps(v: __m128) -> u8 {
    (((_mm_extract_ps::<3>(v) != 0) as u8) << 3)
        + (((_mm_extract_ps::<2>(v) != 0) as u8) << 2)
        + (((_mm_extract_ps::<1>(v) != 0) as u8) << 1)
        + ((_mm_extract_ps::<0>(v) != 0) as u8)
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn _mm256_mask_epi32(v: __m256i) -> u8 {
    (((_mm256_extract_epi32::<7>(v) != 0) as u8) << 7)
        + (((_mm256_extract_epi32::<6>(v) != 0) as u8) << 6)
        + (((_mm256_extract_epi32::<5>(v) != 0) as u8) << 5)
        + (((_mm256_extract_epi32::<4>(v) != 0) as u8) << 4)
        + (((_mm256_extract_epi32::<3>(v) != 0) as u8) << 3)
        + (((_mm256_extract_epi32::<2>(v) != 0) as u8) << 2)
        + (((_mm256_extract_epi32::<1>(v) != 0) as u8) << 1)
        + ((_mm256_extract_epi32::<0>(v) != 0) as u8)
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn _mm256_mask_ps(v: __m256) -> u8 {
    (_mm_mask_ps(_mm256_extractf128_ps::<1>(v)) << 4) + _mm_mask_ps(_mm256_extractf128_ps::<0>(v))
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn _mm_set1_epu32(v: u32) -> __m128i {
    _mm_set1_epi32(transmute(v))
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn _mm256_set1_epu32(v: u32) -> __m256i {
    _mm256_set1_epi32(transmute(v))
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn _mm_cmpgt_epu32(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmpgt_epi32(
        _mm_add_epi32(a, _mm_set1_epi32(i32::MIN)),
        _mm_add_epi32(b, _mm_set1_epi32(i32::MIN)),
    )
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn _mm256_cmpgt_epu32(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmpgt_epi32(
        _mm256_add_epi32(a, _mm256_set1_epi32(i32::MIN)),
        _mm256_add_epi32(b, _mm256_set1_epi32(i32::MIN)),
    )
}
