macro_rules! generate_simd_support {
    (
        for [$n:literal x $t:ty] use $simd:ty,
        $(impl trait $trait:ident {
            $($(#[$attr:meta])*unsafe fn $caller:ident($($arg_name:ident: $arg_ty:ty),* $(,)?) -> $return_ty:ty = $callee:expr),* $(,)?
        })*
    ) => {
        impl $crate::simd::SupportedNativeSimd<$t, $n> for $crate::simd::LaneCount<$t, $n> {
            type RelativeSimdType = $simd;
        }

        $(impl $trait<$t, $n> for $crate::simd::LaneCount<$t, $n> {
            $(#[inline]
            $(#[$attr])*
            unsafe fn $caller($($arg_name: $arg_ty),*) -> $return_ty {
                // SAFETY: need to be correctly defined
                #[allow(unused_unsafe)]
                unsafe { $callee($($arg_name),*) }
            })*
        })*
    };
}

// This trick allow the usage of the macros exported without the inconvence of
// the #[macro_export] that is more like an pub
pub(crate) use generate_simd_support;
