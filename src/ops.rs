//! Extra maths operators.
//!
//! This module provide a various list of extra operation used inside the
//! library that is not standard in the rust programming language.

/// The dot product operation.
///
/// This trait provide a way to do a dot produit of a given type for the isochro lib.
pub trait DotProduct<Rhs = Self> {
    type Output;

    fn dot(self, other: Rhs) -> Self::Output;
}
