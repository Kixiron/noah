#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

extern crate alloc;

mod abort;
mod arc;
mod atomic;
mod header_slice;
mod hybrid_arc;

pub use arc::Arc;
pub use atomic::Atomic;
pub use header_slice::{HeaderSlice, HeaderSliceWithLength, HeaderWithLength};
#[cfg(feature = "std")]
pub use hybrid_arc::HybridArc;

mod sealed {
    pub trait Sealed {}
}
