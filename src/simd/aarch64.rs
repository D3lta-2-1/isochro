use std::arch::aarch64::*;
use std::i32;

use super::macros::{generate_simd_support, overflowing_check};
use super::{ArithmeticSimdOperation, CheckIntOverflowSimd};

generate_simd_support! {
    for [2 x i32] use int32x2_t,
    impl trait CheckIntOverflowSimd {
        fn overflowing_add(lhs: int32x2_t, rhs: int32x2_t) -> (int32x2_t, bool) {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { overflowing_check!(
                add signed(i32),
                lhs,
                rhs,
                vmov_n_s32,
                vadd_s32,
                vsub_s32,
                vcgt_s32,
                vmask_u32,
            ) }
        }
        fn wrapping_add(lhs: int32x2_t, rhs: int32x2_t) -> int32x2_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { vadd_s32(lhs, rhs) }
        }
        fn overflowing_sub(lhs: int32x2_t, rhs: int32x2_t) -> (int32x2_t, bool) {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { overflowing_check!(
                sub signed(i32),
                lhs,
                rhs,
                vmov_n_s32,
                vadd_s32,
                vsub_s32,
                vcgt_s32,
                vmask_u32,
            ) }
        }
        fn wrapping_sub(lhs: int32x2_t, rhs: int32x2_t) -> int32x2_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { vsub_s32(lhs, rhs) }
        }
        fn overflowing_mul(lhs: int32x2_t, rhs: int32x2_t) -> (int32x2_t, bool) {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { overflowing_check!(
                mul signed(i32),
                lhs,
                rhs,
                vmov_n_s32,
                vsign_abs_s32,
                vadd_s32,
                vmul_s32,
                vand_s32,
                vshr_n_s32,
                vshl_n_s32,
                vcgt_s32,
                vmask_s32,
                vmask_u32,
            ) }
        }
        fn wrapping_mul(lhs: int32x2_t, rhs: int32x2_t) -> int32x2_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { vmul_s32(lhs, rhs) }
        }
    }
}

generate_simd_support! {
    for [2 x u32] use uint32x2_t,
    impl trait CheckIntOverflowSimd {
        fn overflowing_add(lhs: uint32x2_t, rhs: uint32x2_t) -> (uint32x2_t, bool) {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { overflowing_check!(
                add unsigned(u32),
                lhs,
                rhs,
                vmov_n_u32,
                vadd_u32,
                vsub_u32,
                vcgt_u32,
                vmask_u32,
            ) }
        }
        fn wrapping_add(lhs: uint32x2_t, rhs: uint32x2_t) -> uint32x2_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { vadd_u32(lhs, rhs) }
        }
        fn overflowing_sub(lhs: uint32x2_t, rhs: uint32x2_t) -> (uint32x2_t, bool) {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { overflowing_check!(
                sub unsigned(u32),
                lhs,
                rhs,
                vmov_n_u32,
                vadd_u32,
                vsub_u32,
                vcgt_u32,
                vmask_u32,
            ) }
        }
        fn wrapping_sub(lhs: uint32x2_t, rhs: uint32x2_t) -> uint32x2_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { vsub_u32(lhs, rhs) }
        }
        fn overflowing_mul(lhs: uint32x2_t, rhs: uint32x2_t) -> (uint32x2_t, bool) {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { overflowing_check!(
                mul unsigned(u32),
                lhs,
                rhs,
                vmov_n_u32,
                vadd_u32,
                vmul_u32,
                vand_u32,
                vshr_n_u32,
                vshl_n_u32,
                vcgt_u32,
                vmask_u32,
            ) }
        }
        fn wrapping_mul(lhs: uint32x2_t, rhs: uint32x2_t) -> uint32x2_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { vmul_u32(lhs, rhs) }
        }
    }
}

generate_simd_support! {
    for [2 x f32] use float32x2_t,
    impl trait ArithmeticSimdOperation {
        fn add(lhs: float32x2_t, rhs: float32x2_t) -> float32x2_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { vadd_f32(lhs, rhs) }
        }
        fn sub(lhs: float32x2_t, rhs: float32x2_t) -> float32x2_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { vsub_f32(lhs, rhs) }
        }
        fn mul(lhs: float32x2_t, rhs: float32x2_t) -> float32x2_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { vmul_f32(lhs, rhs) }
        }
    }
}

generate_simd_support! {
    for [4 x i32] use int32x4_t,
    impl trait CheckIntOverflowSimd {
        fn overflowing_add(lhs: int32x4_t, rhs: int32x4_t) -> (int32x4_t, bool) {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { overflowing_check!(
                add signed(i32),
                lhs,
                rhs,
                vmovq_n_s32,
                vaddq_s32,
                vsubq_s32,
                vcgtq_s32,
                vmaskq_u32,
            ) }
        }
        fn wrapping_add(lhs: int32x4_t, rhs: int32x4_t) -> int32x4_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { vaddq_s32(lhs, rhs) }
        }
        fn overflowing_sub(lhs: int32x4_t, rhs: int32x4_t) -> (int32x4_t, bool) {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { overflowing_check!(
                sub signed(i32),
                lhs,
                rhs,
                vmovq_n_s32,
                vaddq_s32,
                vsubq_s32,
                vcgtq_s32,
                vmaskq_u32,
            ) }
        }
        fn wrapping_sub(lhs: int32x4_t, rhs: int32x4_t) -> int32x4_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { vsubq_s32(lhs, rhs) }
        }
        fn overflowing_mul(lhs: int32x4_t, rhs: int32x4_t) -> (int32x4_t, bool) {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { overflowing_check!(
                mul signed(i32),
                lhs,
                rhs,
                vmovq_n_s32,
                vsign_absq_s32,
                vaddq_s32,
                vmulq_s32,
                vandq_s32,
                vshrq_n_s32,
                vshlq_n_s32,
                vcgtq_s32,
                vmaskq_s32,
                vmaskq_u32,
            ) }
        }
        fn wrapping_mul(lhs: int32x4_t, rhs: int32x4_t) -> int32x4_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { vmulq_s32(lhs, rhs) }
        }
    }
}

generate_simd_support! {
    for [4 x u32] use uint32x4_t,
    impl trait CheckIntOverflowSimd {
        fn overflowing_add(lhs: uint32x4_t, rhs: uint32x4_t) -> (uint32x4_t, bool) {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { overflowing_check!(
                add unsigned(u32),
                lhs,
                rhs,
                vmovq_n_u32,
                vaddq_u32,
                vsubq_u32,
                vcgtq_u32,
                vmaskq_u32,
            ) }
        }
        fn wrapping_add(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { vaddq_u32(lhs, rhs) }
        }
        fn overflowing_sub(lhs: uint32x4_t, rhs: uint32x4_t) -> (uint32x4_t, bool) {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { overflowing_check!(
                sub unsigned(u32),
                lhs,
                rhs,
                vmovq_n_u32,
                vaddq_u32,
                vsubq_u32,
                vcgtq_u32,
                vmaskq_u32,
            ) }
        }
        fn wrapping_sub(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { vsubq_u32(lhs, rhs) }
        }
        fn overflowing_mul(lhs: uint32x4_t, rhs: uint32x4_t) -> (uint32x4_t, bool) {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { overflowing_check!(
                mul unsigned(u32),
                lhs,
                rhs,
                vmovq_n_u32,
                vaddq_u32,
                vmulq_u32,
                vandq_u32,
                vshrq_n_u32,
                vshlq_n_u32,
                vcgtq_u32,
                vmaskq_u32,
            ) }
        }
        fn wrapping_mul(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { vmulq_u32(lhs, rhs) }
        }
    }
}

generate_simd_support! {
    for [4 x f32] use float32x4_t,
    impl trait ArithmeticSimdOperation {
        fn add(lhs: float32x4_t, rhs: float32x4_t) -> float32x4_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { vaddq_f32(lhs, rhs) }
        }
        fn sub(lhs: float32x4_t, rhs: float32x4_t) -> float32x4_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { vsubq_f32(lhs, rhs) }
        }
        fn mul(lhs: float32x4_t, rhs: float32x4_t) -> float32x4_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { vmulq_f32(lhs, rhs) }
        }
    }
}

generate_simd_support! {
    for [8 x i32] use int32x4x2_t,
    impl trait CheckIntOverflowSimd {
        fn overflowing_add(lhs: int32x4x2_t, rhs: int32x4x2_t) -> (int32x4x2_t, bool) {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { overflowing_check!(
                add signed(i32),
                lhs,
                rhs,
                vmovqx2_n_s32,
                vaddqx2_s32,
                vsubqx2_s32,
                vcgtqx2_s32,
                vmaskqx2_u32,
            ) }
        }
        fn wrapping_add(lhs: int32x4x2_t, rhs: int32x4x2_t) -> int32x4x2_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { vaddqx2_s32(lhs, rhs) }
        }
        fn overflowing_sub(lhs: int32x4x2_t, rhs: int32x4x2_t) -> (int32x4x2_t, bool) {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { overflowing_check!(
                sub signed(i32),
                lhs,
                rhs,
                vmovqx2_n_s32,
                vaddqx2_s32,
                vsubqx2_s32,
                vcgtqx2_s32,
                vmaskqx2_u32,
            ) }
        }
        fn wrapping_sub(lhs: int32x4x2_t, rhs: int32x4x2_t) -> int32x4x2_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { vsubqx2_s32(lhs, rhs) }
        }
        fn overflowing_mul(lhs: int32x4x2_t, rhs: int32x4x2_t) -> (int32x4x2_t, bool) {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { overflowing_check!(
                mul signed(i32),
                lhs,
                rhs,
                vmovqx2_n_s32,
                vsign_absqx2_s32,
                vaddqx2_s32,
                vmulqx2_s32,
                vandqx2_s32,
                vshrqx2_n_s32,
                vshlqx2_n_s32,
                vcgtqx2_s32,
                vmaskqx2_s32,
                vmaskqx2_u32,
            ) }
        }
        fn wrapping_mul(lhs: int32x4x2_t, rhs: int32x4x2_t) -> int32x4x2_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { vmulqx2_s32(lhs, rhs) }
        }
    }
}

generate_simd_support! {
    for [8 x u32] use uint32x4x2_t,
    impl trait CheckIntOverflowSimd {
        fn overflowing_add(lhs: uint32x4x2_t, rhs: uint32x4x2_t) -> (uint32x4x2_t, bool) {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { overflowing_check!(
                add unsigned(u32),
                lhs,
                rhs,
                vmovqx2_n_u32,
                vaddqx2_u32,
                vsubqx2_u32,
                vcgtqx2_u32,
                vmaskqx2_u32,
            ) }
        }
        fn wrapping_add(lhs: uint32x4x2_t, rhs: uint32x4x2_t) -> uint32x4x2_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { vaddqx2_u32(lhs, rhs) }
        }
        fn overflowing_sub(lhs: uint32x4x2_t, rhs: uint32x4x2_t) -> (uint32x4x2_t, bool) {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { overflowing_check!(
                sub unsigned(u32),
                lhs,
                rhs,
                vmovqx2_n_u32,
                vaddqx2_u32,
                vsubqx2_u32,
                vcgtqx2_u32,
                vmaskqx2_u32,
            ) }
        }
        fn wrapping_sub(lhs: uint32x4x2_t, rhs: uint32x4x2_t) -> uint32x4x2_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { vsubqx2_u32(lhs, rhs) }
        }
        fn overflowing_mul(lhs: uint32x4x2_t, rhs: uint32x4x2_t) -> (uint32x4x2_t, bool) {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { overflowing_check!(
                mul unsigned(u32),
                lhs,
                rhs,
                vmovqx2_n_u32,
                vaddqx2_u32,
                vmulqx2_u32,
                vandqx2_u32,
                vshrqx2_n_u32,
                vshlqx2_n_u32,
                vcgtqx2_u32,
                vmaskqx2_u32,
            ) }
        }
        fn wrapping_mul(lhs: uint32x4x2_t, rhs: uint32x4x2_t) -> uint32x4x2_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { vmulqx2_u32(lhs, rhs) }
        }
    }
}

generate_simd_support! {
    for [8 x f32] use float32x4x2_t,
    impl trait ArithmeticSimdOperation {
        fn add(lhs: float32x4x2_t, rhs: float32x4x2_t) -> float32x4x2_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { float32x4x2_t(vaddq_f32(lhs.0, rhs.0), vaddq_f32(lhs.1, rhs.1)) }
        }
        fn sub(lhs: float32x4x2_t, rhs: float32x4x2_t) -> float32x4x2_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { float32x4x2_t(vsubq_f32(lhs.0, rhs.0), vsubq_f32(lhs.1, rhs.1)) }
        }
        fn mul(lhs: float32x4x2_t, rhs: float32x4x2_t) -> float32x4x2_t {
            // SAFETY: lhs and rhs are correctly aligned and defined
            unsafe { float32x4x2_t(vmulq_f32(lhs.0, rhs.0), vmulq_f32(lhs.1, rhs.1)) }
        }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vsign_abs_s32(v: int32x2_t) -> (int32x2_t, int32x2_t) {
    const BITS: i32 = i32::BITS as i32 - 2;

    // algo to get the sign: ((v & 0b100...0) >> 30) + 1
    // I use `i32::MIN` to get the sign bit because `i32::MIN` only has this
    // bit available, then I shift 30 bits to the right to get -2 or 0, and
    // I add 1 to finally get -1 or 1.
    (
        vadd_s32(
            vshr_n_s32::<BITS>(vand_s32(v, vmov_n_s32(i32::MIN))),
            vmov_n_s32(1),
        ),
        vabs_s32(v),
    )
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vsign_absq_s32(v: int32x4_t) -> (int32x4_t, int32x4_t) {
    const BITS: i32 = i32::BITS as i32 - 2;

    // algo to get the sign: ((v & 0b100...0) >> 30) + 1
    // I use `i32::MIN` to get the sign bit because `i32::MIN` only has this
    // bit available, then I shift 30 bits to the right to get -2 or 0, and
    // I add 1 to finally get -1 or 1.
    (
        vaddq_s32(
            vshrq_n_s32::<BITS>(vandq_s32(v, vmovq_n_s32(i32::MIN))),
            vmovq_n_s32(1),
        ),
        vabsq_s32(v),
    )
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vsign_absqx2_s32(v: int32x4x2_t) -> (int32x4x2_t, int32x4x2_t) {
    let (s1, a1) = vsign_absq_s32(v.0);
    let (s2, a2) = vsign_absq_s32(v.0);

    (int32x4x2_t(s1, s2), int32x4x2_t(a1, a2))
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vmask_s32(v: int32x2_t) -> u8 {
    (((vget_lane_s32::<1>(v) != 0) as u8) << 1) + ((vget_lane_s32::<0>(v) != 0) as u8)
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vmask_u32(v: uint32x2_t) -> u8 {
    (((vget_lane_u32::<1>(v) != 0) as u8) << 1) + ((vget_lane_u32::<0>(v) != 0) as u8)
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vmaskq_s32(v: int32x4_t) -> u8 {
    (((vgetq_lane_s32::<3>(v) != 0) as u8) << 3)
        + (((vgetq_lane_s32::<2>(v) != 0) as u8) << 2)
        + (((vgetq_lane_s32::<1>(v) != 0) as u8) << 1)
        + ((vgetq_lane_s32::<0>(v) != 0) as u8)
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vmaskq_u32(v: uint32x4_t) -> u8 {
    (((vgetq_lane_u32::<3>(v) != 0) as u8) << 3)
        + (((vgetq_lane_u32::<2>(v) != 0) as u8) << 2)
        + (((vgetq_lane_u32::<1>(v) != 0) as u8) << 1)
        + ((vgetq_lane_u32::<0>(v) != 0) as u8)
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vmaskqx2_s32(v: int32x4x2_t) -> u8 {
    vmaskq_s32(v.0) << 4 + vmaskq_s32(v.1)
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vmaskqx2_u32(v: uint32x4x2_t) -> u8 {
    vmaskq_u32(v.0) << 4 + vmaskq_u32(v.1)
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vmovqx2_n_s32(value: i32) -> int32x4x2_t {
    int32x4x2_t(vmovq_n_s32(value), vmovq_n_s32(value))
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vmovqx2_n_u32(value: u32) -> uint32x4x2_t {
    uint32x4x2_t(vmovq_n_u32(value), vmovq_n_u32(value))
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vaddqx2_s32(a: int32x4x2_t, b: int32x4x2_t) -> int32x4x2_t {
    int32x4x2_t(vaddq_s32(a.0, b.0), vaddq_s32(a.1, b.1))
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vaddqx2_u32(a: uint32x4x2_t, b: uint32x4x2_t) -> uint32x4x2_t {
    uint32x4x2_t(vaddq_u32(a.0, b.0), vaddq_u32(a.1, b.1))
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vsubqx2_s32(a: int32x4x2_t, b: int32x4x2_t) -> int32x4x2_t {
    int32x4x2_t(vsubq_s32(a.0, b.0), vsubq_s32(a.1, b.1))
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vsubqx2_u32(a: uint32x4x2_t, b: uint32x4x2_t) -> uint32x4x2_t {
    uint32x4x2_t(vsubq_u32(a.0, b.0), vsubq_u32(a.1, b.1))
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vcgtqx2_s32(a: int32x4x2_t, b: int32x4x2_t) -> uint32x4x2_t {
    uint32x4x2_t(vcgtq_s32(a.0, b.0), vcgtq_s32(a.1, b.1))
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vcgtqx2_u32(a: uint32x4x2_t, b: uint32x4x2_t) -> uint32x4x2_t {
    uint32x4x2_t(vcgtq_u32(a.0, b.0), vcgtq_u32(a.1, b.1))
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vmulqx2_s32(a: int32x4x2_t, b: int32x4x2_t) -> int32x4x2_t {
    int32x4x2_t(vmulq_s32(a.0, b.0), vmulq_s32(a.1, b.1))
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vmulqx2_u32(a: uint32x4x2_t, b: uint32x4x2_t) -> uint32x4x2_t {
    uint32x4x2_t(vmulq_u32(a.0, b.0), vmulq_u32(a.1, b.1))
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vandqx2_s32(a: int32x4x2_t, b: int32x4x2_t) -> int32x4x2_t {
    int32x4x2_t(vandq_s32(a.0, b.0), vandq_s32(a.1, b.1))
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vandqx2_u32(a: uint32x4x2_t, b: uint32x4x2_t) -> uint32x4x2_t {
    uint32x4x2_t(vandq_u32(a.0, b.0), vandq_u32(a.1, b.1))
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vshrqx2_n_s32<const N: i32>(a: int32x4x2_t) -> int32x4x2_t {
    int32x4x2_t(vshrq_n_s32::<N>(a.0), vshrq_n_s32::<N>(a.1))
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vshrqx2_n_u32<const N: i32>(a: uint32x4x2_t) -> uint32x4x2_t {
    uint32x4x2_t(vshrq_n_u32::<N>(a.0), vshrq_n_u32::<N>(a.1))
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vshlqx2_n_s32<const N: i32>(a: int32x4x2_t) -> int32x4x2_t {
    int32x4x2_t(vshlq_n_s32::<N>(a.0), vshlq_n_s32::<N>(a.1))
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vshlqx2_n_u32<const N: i32>(a: uint32x4x2_t) -> uint32x4x2_t {
    uint32x4x2_t(vshlq_n_u32::<N>(a.0), vshlq_n_u32::<N>(a.1))
}
