#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate alloc;

mod abort;
mod arc;
mod atomic;

pub use arc::Arc;
pub use atomic::Atomic;

mod sealed {
    pub trait Sealed {}
}
