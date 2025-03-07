macro_rules! generate_simd_support {
    (
        for [$n:literal x $t:ty] use $simd:ty,
        $(fn $caller:ident($($arg_name:ident: $arg_ty:ty),* $(,)?) -> $return_ty:ty = $callee:ident),* $(,)?
    ) => {
        impl $crate::simd::SupportedNativeSimd<$t, $n> for $crate::simd::LaneCount<$t, $n> {
            type RelativeSimdType = $simd;
        
            $(#[inline]
            unsafe fn $caller($($arg_name: $arg_ty),*) -> $return_ty {
                // SAFETY: need to be correctly defined
                unsafe { $callee($($arg_name),*) }
            })*
        }
    };
}

// This trick allow the usage of the macros exported without the inconvence of
// the #[macro_export] that is more like an pub
pub(crate) use generate_simd_support;
