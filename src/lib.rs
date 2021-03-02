#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod abort;
mod arc;
mod atomic;
mod header_slice;

pub use arc::Arc;
pub use atomic::Atomic;
pub use header_slice::{HeaderSlice, HeaderSliceWithLength, HeaderWithLength};

mod sealed {
    pub trait Sealed {}
}
