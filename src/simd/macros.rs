macro_rules! generate_simd_support {
    (
        for [$n:literal x $t:ty] use $simd:ty,
        $(impl base {
            $($(#[$attr:meta])* $($modifier:ident)+ $(<$($lifetime:lifetime),*>)? ($($arg_name:ident: $arg_ty:ty),* $(,)?) -> $return_ty:ty $body:block)*
        })?
        $(impl trait $trait:ident {
            $($(#[$impl_attr:meta])* $($impl_modifier:ident)+ $(<$($impl_lifetime:lifetime),*>)? ($($impl_arg_name:ident: $impl_arg_ty:ty),* $(,)?) -> $impl_return_ty:ty $impl_body:block)*
        })*
    ) => {
        impl $crate::simd::SupportedNativeSimd<$t, $n> for $crate::simd::LaneCount<$t, $n> {
            type SimdType = $simd;
            $($(generate_simd_support! {
                $(#[$attr])* $($modifier)+ $(<$($lifetime),*>)? ($($arg_name: $arg_ty),*) -> $return_ty $body
            })*)?
        }

        $(impl $trait<$t, $n> for $crate::simd::LaneCount<$t, $n> {
            $(generate_simd_support! {
                $(#[$impl_attr])* $($impl_modifier)+ $(<$($impl_lifetime),*>)? ($($impl_arg_name: $impl_arg_ty),*) -> $impl_return_ty $impl_body
            })*
        })*
    };
    // used to generate correctly an unsafe fn
    ($(#[$attr:meta])* unsafe fn $caller:ident $(<$($lifetime:lifetime),*>)?($($arg_name:ident: $arg_ty:ty),* $(,)?) -> $return_ty:ty $body:block) => {
        $(#[$attr])*
        #[inline]
        unsafe fn $caller$(<$($lifetime,)*>)?($($arg_name: $arg_ty),*) -> $return_ty $body
    };
    // used to generate correctly a fn
    ($(#[$attr:meta])* fn $caller:ident $(<$($lifetime:lifetime),*>)?($($arg_name:ident: $arg_ty:ty),* $(,)?) -> $return_ty:ty $body:block) => {
        $(#[$attr])*
        #[inline]
        fn $caller$(<$($lifetime,)*>)?($($arg_name: $arg_ty),*) -> $return_ty $body
    };
}

macro_rules! overflowing_check {
    (
        add signed($t:ty),
        $lhs:expr,
        $rhs:expr,
        $mov:ident,
        $add:ident,
        $sub:ident,
        $cgt:ident,
        $mask:ident,
    ) => {{
        (
            $add($lhs, $rhs),
            $mask($cgt($lhs, $sub($mov(<$t>::MAX), $rhs))) > 0
                || $mask($cgt($sub($mov(<$t>::MIN), $rhs), $lhs)) > 0,
        )
    }};
    (
        add unsigned($t:ty),
        $lhs:expr,
        $rhs:expr,
        $mov:ident,
        $add:ident,
        $sub:ident,
        $cgt:ident,
        $mask:ident,
    ) => {{
        (
            $add($lhs, $rhs),
            $mask($cgt($lhs, $sub($mov(<$t>::MAX), $rhs))) > 0,
        )
    }};
    (
        sub signed($t:ty),
        $lhs:expr,
        $rhs:expr,
        $mov:ident,
        $add:ident,
        $sub:ident,
        $cgt:ident,
        $mask:ident,
    ) => {{
        (
            $sub($lhs, $rhs),
            $mask($cgt($lhs, $add($mov(<$t>::MAX), $rhs))) > 0
                || $mask($cgt($add($mov(<$t>::MIN), $rhs), $lhs)) > 0,
        )
    }};
    (
        sub unsigned($t:ty),
        $lhs:expr,
        $rhs:expr,
        $mov:ident,
        $add:ident,
        $sub:ident,
        $cgt:ident,
        $mask:ident,
    ) => {{
        (
            $sub($lhs, $rhs),
            $mask($cgt($lhs, $add($mov(<$t>::MAX), $rhs))) > 0,
        )
    }};
    (
        mul signed($t:ty),
        $lhs:expr,
        $rhs:expr,
        $mov:ident,
        $sign_abs:ident,
        $add:ident,
        $mul:ident,
        $and:ident,
        $shr:ident,
        $shl:ident,
        $cgt:ident,
        $mask_signed:ident,
        $mask_unsigned:ident,
    ) => {{
        const HALF_SIZE_BITS: i32 = (<$t>::BITS / 2) as i32;

        let (lhs_sign, lhs) = $sign_abs($lhs);
        let (rhs_sign, rhs) = $sign_abs($rhs);
        let sign = $mul(lhs_sign, rhs_sign);

        let half_size_vector = $mov((1 << HALF_SIZE_BITS) - 1);

        let lhs_high = $shr::<HALF_SIZE_BITS>(lhs);
        let lhs_low = $and(lhs, half_size_vector);
        let rhs_high = $shr::<HALF_SIZE_BITS>(rhs);
        let rhs_low = $and(rhs, half_size_vector);

        let lowbits = $mul(lhs_low, rhs_low);
        let lhs_high_set = $mask_signed(lhs_high) > 0;
        let rhs_high_set = $mask_signed(rhs_high) > 0;
        if !(lhs_high_set || rhs_high_set) {
            return (lowbits, false);
        }

        let mut overflowed = lhs_high_set && rhs_high_set;
        let midbits1 = $mul(lhs_low, rhs_high);
        let midbits2 = $mul(lhs_high, rhs_low);
        let midbits = $add(midbits1, midbits2);

        overflowed = overflowed
            || $mask_unsigned($cgt(midbits1, midbits)) > 0
            || $mask_unsigned($cgt(midbits, half_size_vector)) > 0;
        let product = $add(lowbits, $shl::<HALF_SIZE_BITS>(midbits));
        overflowed = overflowed || $mask_unsigned($cgt(lowbits, product)) > 0;

        ($mul(sign, product), overflowed)
    }};
    (
        mul unsigned($t:ty),
        $lhs:expr,
        $rhs:expr,
        $mov:ident,
        $add:ident,
        $mul:ident,
        $and:ident,
        $shr:ident,
        $shl:ident,
        $cgt:ident,
        $mask:ident,
    ) => {{
        const HALF_SIZE_BITS: i32 = (<$t>::BITS / 2) as i32;

        let half_size_vector = $mov((1 << HALF_SIZE_BITS) - 1);

        let lhs_high = $shr::<HALF_SIZE_BITS>($lhs);
        let lhs_low = $and($lhs, half_size_vector);
        let rhs_high = $shr::<HALF_SIZE_BITS>($rhs);
        let rhs_low = $and($rhs, half_size_vector);

        let lowbits = $mul(lhs_low, rhs_low);
        let lhs_high_set = $mask(lhs_high) > 0;
        let rhs_high_set = $mask(rhs_high) > 0;
        if !(lhs_high_set || rhs_high_set) {
            return (lowbits, false);
        }

        let mut overflowed = lhs_high_set && rhs_high_set;
        let midbits1 = $mul(lhs_low, rhs_high);
        let midbits2 = $mul(lhs_high, rhs_low);
        let midbits = $add(midbits1, midbits2);

        overflowed = overflowed
            || $mask($cgt(midbits1, midbits)) > 0
            || $mask($cgt(midbits, half_size_vector)) > 0;
        let product = $add(lowbits, $shl::<HALF_SIZE_BITS>(midbits));
        overflowed = overflowed || $mask($cgt(lowbits, product)) > 0;

        (product, overflowed)
    }};
}

// This trick allow the usage of the macros exported without the inconvence of
// the #[macro_export] that is more like an pub
pub(crate) use {generate_simd_support, overflowing_check};
