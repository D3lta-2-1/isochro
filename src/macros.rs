/// This macro implement the version of "&U for T", "U for &T" and "&U for &T" for binary operators
macro_rules! forward_ref_binop {
    (impl$(<$($generic:ident),* $(;$(const $const_name:ident : $const_ty:ty),*)?>)? $trait:ident<$u:ty>, $method:ident for $t:ty $(where $($tt:tt)*)?) => {
        impl<$($($generic),*$(,$(const $const_name: $const_ty),*)?)?> $trait<$u> for &$t
        $(where $($tt)*)?
        {
            type Output = <$t as $trait<$u>>::Output;

            #[inline]
            fn $method(self, other: $u) -> Self::Output {
                // TODO: avoid copy (due to big copy of a big vector for example)
                $trait::$method(*self, other)
            }
        }

        impl<$($($generic),*$(,$(const $const_name: $const_ty),*)?)?> $trait<&$u> for $t
        $(where $($tt)*)?
        {
            type Output = <$t as $trait<$u>>::Output;

            #[inline]
            fn $method(self, other: &$u) -> <$t as $trait<$u>>::Output {
                // TODO: avoid copy (due to big copy of a big vector for example)
                $trait::$method(self, *other)
            }
        }

        impl<$($($generic),*$(,$(const $const_name: $const_ty),*)?)?> $trait<&$u> for &$t
        $(where $($tt)*)?
        {
            type Output = <$t as $trait<$u>>::Output;

            #[inline]
            fn $method(self, other: &$u) -> <$t as $trait<$u>>::Output {
                // TODO: avoid copy (due to big copy of a big vector for example)
                $trait::$method(*self, *other)
            }
        }
    };
}

// This trick allow the usage of the macros exported without the inconvence of
// the #[macro_export] that is more like an pub
pub(crate) use forward_ref_binop;
