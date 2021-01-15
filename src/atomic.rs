use crate::sealed::Sealed;
use core::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

/// An atomic integer
pub trait Atomic: Sealed {
    /// The maximum allowed refcount for the current [`Atomic`]
    const MAX_REFCOUNT: u64;

    /// Creates an [`Atomic`] with a value of 1
    fn one() -> Self;

    fn load(&self, order: Ordering) -> u64;

    fn fetch_add(&self, val: u64, order: Ordering) -> u64;

    fn fetch_sub(&self, val: u64, order: Ordering) -> u64;
}

macro_rules! impl_atomic {
    ($($atomic:ty => $max_refcount:expr),* $(,)?) => {
        $(
            impl Atomic for $atomic {
                const MAX_REFCOUNT: u64 = $max_refcount as u64;

                #[inline]
                fn one() -> Self {
                    Self::new(1)
                }

                #[inline]
                fn load(&self, order: Ordering) -> u64 {
                    self.load(order) as u64
                }

                #[inline]
                fn fetch_add(&self, val: u64, order: Ordering) -> u64 {
                    self.fetch_add(val as _, order) as u64
                }

                #[inline]
                fn fetch_sub(&self, val: u64, order: Ordering) -> u64 {
                    self.fetch_sub(val as _, order) as u64
                }
            }

            impl Sealed for $atomic {}
        )*
    };
}

impl_atomic! {
    AtomicUsize => isize::max_value(),
    AtomicU64 => i64::max_value(),
    AtomicU32 => i32::max_value(),
    // AtomicU16 => i16::max_value(),
    // AtomicU8 => i8::max_value(),
}
