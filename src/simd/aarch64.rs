use std::arch::aarch64::*;

use super::ArithmeticSimdMathOperation;
use super::macros::generate_simd_support;

generate_simd_support! {
    for [2 x i32] use int32x2_t,
    impl trait ArithmeticSimdMathOperation {
        unsafe fn add(lhs: int32x2_t, rhs: int32x2_t) -> int32x2_t = vadd_s32,
        unsafe fn sub(lhs: int32x2_t, rhs: int32x2_t) -> int32x2_t = vsub_s32,
        unsafe fn mul(lhs: int32x2_t, rhs: int32x2_t) -> int32x2_t = vmul_s32,
    }
}

generate_simd_support! {
    for [2 x u32] use uint32x2_t,
    impl trait ArithmeticSimdMathOperation {
        unsafe fn add(lhs: uint32x2_t, rhs: uint32x2_t) -> uint32x2_t = vadd_u32,
        unsafe fn sub(lhs: uint32x2_t, rhs: uint32x2_t) -> uint32x2_t = vsub_u32,
        unsafe fn mul(lhs: uint32x2_t, rhs: uint32x2_t) -> uint32x2_t = vmul_u32,
    }
}

generate_simd_support! {
    for [2 x f32] use float32x2_t,
    impl trait ArithmeticSimdMathOperation {
        unsafe fn add(lhs: float32x2_t, rhs: float32x2_t) -> float32x2_t = vadd_f32,
        unsafe fn sub(lhs: float32x2_t, rhs: float32x2_t) -> float32x2_t = vsub_f32,
        unsafe fn mul(lhs: float32x2_t, rhs: float32x2_t) -> float32x2_t = vmul_f32,
    }
}

generate_simd_support! {
    for [4 x i32] use int32x4_t,
    impl trait ArithmeticSimdMathOperation {
        unsafe fn add(lhs: int32x4_t, rhs: int32x4_t) -> int32x4_t = vaddq_s32,
        unsafe fn sub(lhs: int32x4_t, rhs: int32x4_t) -> int32x4_t = vsubq_s32,
        unsafe fn mul(lhs: int32x4_t, rhs: int32x4_t) -> int32x4_t = vmulq_s32,
    }
}

generate_simd_support! {
    for [4 x u32] use uint32x4_t,
    impl trait ArithmeticSimdMathOperation {
        unsafe fn add(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t = vaddq_u32,
        unsafe fn sub(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t = vsubq_u32,
        unsafe fn mul(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t = vmulq_u32,
    }
}

generate_simd_support! {
    for [4 x f32] use float32x4_t,
    impl trait ArithmeticSimdMathOperation {
        unsafe fn add(lhs: float32x4_t, rhs: float32x4_t) -> float32x4_t = vaddq_f32,
        unsafe fn sub(lhs: float32x4_t, rhs: float32x4_t) -> float32x4_t = vsubq_f32,
        unsafe fn mul(lhs: float32x4_t, rhs: float32x4_t) -> float32x4_t = vmulq_f32,
    }
}

#[inline]
unsafe fn vaddqx2_s32(lhs: int32x4x2_t, rhs: int32x4x2_t) -> int32x4x2_t {
    #[allow(unsafe_op_in_unsafe_fn)]
    int32x4x2_t(vaddq_s32(lhs.0, rhs.0), vaddq_s32(lhs.1, rhs.1))
}

#[inline]
unsafe fn vsubqx2_s32(lhs: int32x4x2_t, rhs: int32x4x2_t) -> int32x4x2_t {
    #[allow(unsafe_op_in_unsafe_fn)]
    int32x4x2_t(vsubq_s32(lhs.0, rhs.0), vsubq_s32(lhs.1, rhs.1))
}

#[inline]
unsafe fn vmulqx2_s32(lhs: int32x4x2_t, rhs: int32x4x2_t) -> int32x4x2_t {
    #[allow(unsafe_op_in_unsafe_fn)]
    int32x4x2_t(vmulq_s32(lhs.0, rhs.0), vmulq_s32(lhs.1, rhs.1))
}

#[inline]
unsafe fn vaddqx2_u32(lhs: uint32x4x2_t, rhs: uint32x4x2_t) -> uint32x4x2_t {
    #[allow(unsafe_op_in_unsafe_fn)]
    uint32x4x2_t(vaddq_u32(lhs.0, rhs.0), vaddq_u32(lhs.1, rhs.1))
}

#[inline]
unsafe fn vsubqx2_u32(lhs: uint32x4x2_t, rhs: uint32x4x2_t) -> uint32x4x2_t {
    #[allow(unsafe_op_in_unsafe_fn)]
    uint32x4x2_t(vsubq_u32(lhs.0, rhs.0), vsubq_u32(lhs.1, rhs.1))
}

#[inline]
unsafe fn vmulqx2_u32(lhs: uint32x4x2_t, rhs: uint32x4x2_t) -> uint32x4x2_t {
    #[allow(unsafe_op_in_unsafe_fn)]
    uint32x4x2_t(vmulq_u32(lhs.0, rhs.0), vmulq_u32(lhs.1, rhs.1))
}

#[inline]
unsafe fn vaddqx2_f32(lhs: float32x4x2_t, rhs: float32x4x2_t) -> float32x4x2_t {
    #[allow(unsafe_op_in_unsafe_fn)]
    float32x4x2_t(vaddq_f32(lhs.0, rhs.0), vaddq_f32(lhs.1, rhs.1))
}

#[inline]
unsafe fn vsubqx2_f32(lhs: float32x4x2_t, rhs: float32x4x2_t) -> float32x4x2_t {
    #[allow(unsafe_op_in_unsafe_fn)]
    float32x4x2_t(vsubq_f32(lhs.0, rhs.0), vsubq_f32(lhs.1, rhs.1))
}

#[inline]
unsafe fn vmulqx2_f32(lhs: float32x4x2_t, rhs: float32x4x2_t) -> float32x4x2_t {
    #[allow(unsafe_op_in_unsafe_fn)]
    float32x4x2_t(vmulq_f32(lhs.0, rhs.0), vmulq_f32(lhs.1, rhs.1))
}

generate_simd_support! {
    for [8 x i32] use int32x4x2_t,
    impl trait ArithmeticSimdMathOperation {
        unsafe fn add(lhs: int32x4x2_t, rhs: int32x4x2_t) -> int32x4x2_t = vaddqx2_s32,
        unsafe fn sub(lhs: int32x4x2_t, rhs: int32x4x2_t) -> int32x4x2_t = vsubqx2_s32,
        unsafe fn mul(lhs: int32x4x2_t, rhs: int32x4x2_t) -> int32x4x2_t = vmulqx2_s32,
    }
}

generate_simd_support! {
    for [8 x u32] use uint32x4x2_t,
    impl trait ArithmeticSimdMathOperation {
        unsafe fn add(lhs: uint32x4x2_t, rhs: uint32x4x2_t) -> uint32x4x2_t = vaddqx2_u32,
        unsafe fn sub(lhs: uint32x4x2_t, rhs: uint32x4x2_t) -> uint32x4x2_t = vsubqx2_u32,
        unsafe fn mul(lhs: uint32x4x2_t, rhs: uint32x4x2_t) -> uint32x4x2_t = vmulqx2_u32,
    }
}

generate_simd_support! {
    for [8 x f32] use float32x4x2_t,
    impl trait ArithmeticSimdMathOperation {
        unsafe fn add(lhs: float32x4x2_t, rhs: float32x4x2_t) -> float32x4x2_t = vaddqx2_f32,
        unsafe fn sub(lhs: float32x4x2_t, rhs: float32x4x2_t) -> float32x4x2_t = vsubqx2_f32,
        unsafe fn mul(lhs: float32x4x2_t, rhs: float32x4x2_t) -> float32x4x2_t = vmulqx2_f32,
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
