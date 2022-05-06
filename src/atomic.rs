use crate::sealed::Sealed;
use core::{
    ops::{Add, Sub},
    sync::atomic::{AtomicU16, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering},
};

/// An atomic integer
pub trait Atomic: Sealed {
    /// The non-atomic version of the current type
    type NonAtomic: Copy
        + Eq
        + Add<Self::NonAtomic, Output = Self::NonAtomic>
        + Sub<Self::NonAtomic, Output = Self::NonAtomic>;

    /// The maximum allowed refcount for the current [`Atomic`]
    const MAX_REFCOUNT: u64;

    /// The maximum allowed refcount for the current [`Self::NonAtomic`]
    const MAX_NONATOMIC: u64;

    /// An [`Atomic`] with a value of 1
    const ONE: Self;

    /// An instance of [`Self::NonAtomic`] with a value of 1
    const ONE_NON_ATOMIC: Self::NonAtomic;

    /// Converts a [`Self::NonAtomic`] value to a `u64`
    fn non_atomic_to_u64(value: Self::NonAtomic) -> u64;

    /// Atomically load the current value
    fn load(&self, order: Ordering) -> u64;

    /// Atomically fetch and add to the current value
    fn fetch_add(&self, val: u64, order: Ordering) -> u64;

    /// Atomically fetch and subtract from the current value
    fn fetch_sub(&self, val: u64, order: Ordering) -> u64;

    /// Atomically store to the current value
    fn store(&self, val: u64, order: Ordering);
}

macro_rules! impl_atomic {
    ($($atomic:ty, $non_atomic:ty => $max_refcount:expr),* $(,)?) => {
        $(
            impl Atomic for $atomic {
                type NonAtomic = $non_atomic;

                const MAX_REFCOUNT: u64 = $max_refcount as u64;

                const MAX_NONATOMIC: u64 = <$non_atomic>::MAX as u64;

                #[allow(clippy::declare_interior_mutable_const)]
                const ONE: Self = Self::new(1);

                const ONE_NON_ATOMIC: Self::NonAtomic = 1;

                #[inline]
                fn non_atomic_to_u64(value: Self::NonAtomic) -> u64 {
                    value as u64
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

                #[inline]
                fn store(&self, val: u64, order: Ordering) {
                    self.store(val as _, order);
                }
            }

            impl Sealed for $atomic {}
        )*
    };
}

impl_atomic! {
    AtomicUsize, usize => isize::max_value(),
    AtomicU64, u64 => i64::max_value(),
    AtomicU32, u32 => i32::max_value(),
    AtomicU16, u16 => i16::max_value(),
    AtomicU8, u8 => i8::max_value(),
}
