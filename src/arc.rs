use crate::{abort::abort, atomic::Atomic};
#[cfg(feature = "std")]
use alloc::boxed::Box;
use core::{
    borrow::Borrow,
    cmp, fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem::{self, ManuallyDrop},
    ops::Deref,
    pin::Pin,
    ptr::NonNull,
    sync::atomic::{AtomicUsize, Ordering},
};
#[cfg(feature = "serde")]
use serde::{
    de::{Deserialize, Deserializer},
    ser::{Serialize, Serializer},
};

#[repr(C)]
pub(crate) struct ArcInner<T: ?Sized, A> {
    pub(crate) count: A,
    pub(crate) data: T,
}

unsafe impl<T: ?Sized + Sync + Send, A: Send + Sync> Send for ArcInner<T, A> {}
unsafe impl<T: ?Sized + Sync + Send, A: Send + Sync> Sync for ArcInner<T, A> {}

/// An atomically reference counted shared pointer
///
/// See the documentation for [`Arc`] in the standard library. Unlike the
/// standard library `Arc`, this `Arc` does not support weak reference counting
///
/// [`Arc`]: std::sync::Arc
// TODO: Remove `Sized` limitations when vtable layouts are specified
// TODO: Add `new_uninit()`-like methods when some alloc stuff goes through
#[repr(transparent)]
pub struct Arc<T: ?Sized, A: Atomic = AtomicUsize> {
    /// A pointer to the heap-allocated stub
    pub(crate) ptr: NonNull<ArcInner<T, A>>,
    /// A PhantomData that conveys ownership of a `T` to the drop checker
    pub(crate) __type: PhantomData<T>,
}

impl<T: ?Sized, A: Atomic> Arc<T, A> {
    /// Construct an [`Arc`]
    #[inline]
    pub fn new(data: T) -> Self
    where
        T: Sized,
    {
        let ptr = Box::new(ArcInner {
            count: A::one(),
            data,
        });

        unsafe {
            Self {
                ptr: NonNull::new_unchecked(Box::into_raw(ptr)),
                __type: PhantomData,
            }
        }
    }

    /// Convert the [`Arc`] to a raw pointer, suitable for use across FFI
    #[inline]
    pub fn into_raw(this: Self) -> *const T {
        let ptr = unsafe { &((*this.ptr()).data) as *const T };
        mem::forget(this);

        ptr
    }

    /// Reconstruct the [`Arc`] from a raw pointer obtained from [`Arc::into_raw`]
    ///
    /// # Safety
    ///
    /// The pointer supplied *must* have come from [`Arc::into_raw`]
    ///
    pub unsafe fn from_raw(ptr: *const T) -> Self
    where
        T: Sized,
    {
        // To find the corresponding pointer to the `ArcInner` we need
        // to subtract the offset of the `data` field from the pointer.
        let ptr = (ptr as *const u8).sub(mem::size_of::<A>());
        Self {
            ptr: NonNull::new_unchecked(ptr as *mut ArcInner<T, A>),
            __type: PhantomData,
        }
    }

    /// Returns the address on the heap of the [`Arc`] itself -- not the T within it
    /// -- for memory reporting.
    #[inline]
    pub fn heap_ptr(&self) -> *const () {
        self.ptr.as_ptr() as *const ArcInner<T, A> as *const ()
    }

    /// Test pointer equality between the two [`Arc`]s, i.e. they must be the *same*
    /// allocation
    #[inline]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        this.ptr() == other.ptr()
    }

    #[inline]
    fn inner(&self) -> &ArcInner<T, A> {
        // This unsafety is ok because while this arc is alive we're guaranteed
        // that the inner pointer is valid. Furthermore, we know that the
        // `ArcInner` structure itself is `Sync` because the inner data is
        // `Sync` as well, so we're ok loaning out an immutable pointer to these
        // contents.
        unsafe { &*self.ptr() }
    }

    // Non-inlined part of `drop`. Just invokes the destructor.
    #[inline(never)]
    #[cold]
    unsafe fn drop_slow(&mut self) {
        let _ = Box::from_raw(self.ptr());
    }

    /// Get the pointer to the current [`Arc`]'s [`ArcInner`]
    #[inline]
    pub(crate) fn ptr(&self) -> *mut ArcInner<T, A> {
        self.ptr.as_ptr()
    }

    /// Makes a mutable reference to the `Arc`, cloning if necessary
    ///
    /// This is functionally equivalent to [`Arc::make_mut`][mm] from the standard library.
    ///
    /// If this `Arc` is uniquely owned, `make_mut()` will provide a mutable
    /// reference to the contents. If not, `make_mut()` will create a _new_ `Arc`
    /// with a copy of the contents, update `this` to point to it, and provide
    /// a mutable reference to its contents.
    ///
    /// This is useful for implementing copy-on-write schemes where you wish to
    /// avoid copying things if your `Arc` is not shared.
    ///
    /// [mm]: https://doc.rust-lang.org/stable/std/sync/struct.Arc.html#method.make_mut
    #[inline]
    pub fn make_mut(this: &mut Self) -> &mut T
    where
        T: Clone,
    {
        if !this.is_unique() {
            // Another pointer exists; clone
            *this = Arc::new((**this).clone());
        }

        unsafe {
            // This unsafety is ok because we're guaranteed that the pointer
            // returned is the *only* pointer that will ever be returned to T. Our
            // reference count is guaranteed to be 1 at this point, and we required
            // the Arc itself to be `mut`, so we're returning the only possible
            // reference to the inner data.
            &mut (*this.ptr()).data
        }
    }

    /// Provides mutable access to the contents _if_ the `Arc` is uniquely owned.
    #[inline]
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        if this.is_unique() {
            unsafe {
                // See make_mut() for documentation of the thread safety here.
                Some(&mut (*this.ptr()).data)
            }
        } else {
            None
        }
    }

    /// Whether or not the `Arc` is uniquely owned (is the refcount 1?).
    #[inline]
    pub fn is_unique(&self) -> bool {
        // See the extensive discussion in [1] for why this needs to be Acquire.
        //
        // [1] https://github.com/servo/servo/issues/21186
        Self::ref_count(self) == 1
    }

    /// Gets the number of pointers to this allocation
    #[inline]
    pub fn ref_count(this: &Self) -> u64 {
        this.inner().count.load(Ordering::Acquire)
    }

    /// Decrements the reference count on the [`Arc`] associated with the provided pointer by one
    ///
    /// # Safety
    ///
    /// The pointer must have been obtained through [`Arc::into_raw`], and the associated [`Arc`]
    /// instance must be valid (i.e. the strong count must be at least 1) for the duration of this method.
    ///
    #[allow(unused_unsafe)]
    #[inline]
    pub unsafe fn incr_ref_count(ptr: *const T)
    where
        T: Sized,
    {
        // Retain Arc, but don't touch refcount by wrapping in a `ManuallyDrop`
        let arc = unsafe { ManuallyDrop::new(Self::from_raw(ptr)) };

        // Now increase refcount, but don't drop new refcount either
        let _arc_clone: ManuallyDrop<_> = arc.clone();
    }

    /// Decrements the reference count on the [`Arc`] associated with the provided pointer by one
    ///
    /// # Safety
    ///
    /// The pointer must have been obtained through [`Arc::into_raw`], and the associated [`Arc`]
    /// instance must be valid (i.e. the strong count must be at least 1) when invoking this method.
    /// This method can be used to release the final [`Arc`] and backing storage, but should not be
    /// called after the final [`Arc`] has been released.
    ///
    #[allow(unused_unsafe)]
    #[inline]
    pub unsafe fn decr_ref_count(ptr: *const T)
    where
        T: Sized,
    {
        unsafe { mem::drop(Self::from_raw(ptr)) };
    }

    /// Constructs a new `Pin<Arc<T>>`. If `T` does not implement [`Unpin`](core::marker::Unpin),
    /// then `data` will be pinned in memory and unable to be moved.
    #[inline]
    pub fn pin(data: T) -> Pin<Self>
    where
        T: Sized,
    {
        unsafe { Pin::new_unchecked(Self::new(data)) }
    }
}

impl<T: ?Sized, A> Clone for Arc<T, A>
where
    A: Atomic,
{
    #[inline]
    fn clone(&self) -> Self {
        // Using a relaxed ordering is alright here, as knowledge of the
        // original reference prevents other threads from erroneously deleting
        // the object.
        //
        // As explained in the [Boost documentation][1], Increasing the
        // reference counter can always be done with memory_order_relaxed: New
        // references to an object can only be formed from an existing
        // reference, and passing an existing reference from one thread to
        // another must already provide any required synchronization.
        //
        // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
        let old_size = self.inner().count.fetch_add(1, Ordering::Relaxed);

        // However we need to guard against massive ref counts in case someone
        // is `mem::forget`ing Arcs. If we don't do this the count can overflow
        // and users will use-after free. We racily saturate to `A::MAX_REFCOUNT` on
        // the assumption that there aren't ~2 billion threads incrementing
        // the reference count at once. This branch will never be taken in
        // any realistic program.
        //
        // We abort because such a program is incredibly degenerate, and we
        // don't care to support it.
        if old_size > A::MAX_REFCOUNT {
            abort();
        }

        unsafe {
            Self {
                ptr: NonNull::new_unchecked(self.ptr()),
                __type: PhantomData,
            }
        }
    }
}

impl<T: ?Sized, A: Atomic> Deref for Arc<T, A> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        &self.inner().data
    }
}

impl<T: ?Sized, A: Atomic> Drop for Arc<T, A> {
    #[inline]
    fn drop(&mut self) {
        // Because `fetch_sub` is already atomic, we do not need to synchronize
        // with other threads unless we are going to delete the object.
        if self.inner().count.fetch_sub(1, Ordering::Release) != 1 {
            return;
        }

        // This load is needed to prevent reordering of use of the data and
        // deletion of the data.  Because it is marked `Release`, the decreasing
        // of the reference count synchronizes with this `Acquire` load. This
        // means that use of the data happens before decreasing the reference
        // count, which happens before this load, which happens before the
        // deletion of the data.
        //
        // As explained in the [Boost documentation][1],
        //
        // > It is important to enforce any possible access to the object in one
        // > thread (through an existing reference) to *happen before* deleting
        // > the object in a different thread. This is achieved by a "release"
        // > operation after dropping a reference (any access to the object
        // > through this reference must obviously happened before), and an
        // > "acquire" operation before deleting the object.
        //
        // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
        let _ = self.inner().count.load(Ordering::Acquire);

        unsafe { self.drop_slow() }
    }
}

impl<T: ?Sized + PartialEq, A: Atomic> PartialEq for Arc<T, A> {
    fn eq(&self, other: &Self) -> bool {
        Self::ptr_eq(self, other) || (**self).eq(&**other)
    }

    #[allow(clippy::partialeq_ne_impl)]
    fn ne(&self, other: &Self) -> bool {
        !Self::ptr_eq(self, other) && (**self).ne(&**other)
    }
}

impl<T: ?Sized + PartialOrd, A: Atomic> PartialOrd for Arc<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        (**self).partial_cmp(&**other)
    }

    fn lt(&self, other: &Self) -> bool {
        (**self).lt(&**other)
    }

    fn le(&self, other: &Self) -> bool {
        (**self).le(&**other)
    }

    fn gt(&self, other: &Self) -> bool {
        (**self).gt(&**other)
    }

    fn ge(&self, other: &Self) -> bool {
        (**self).ge(&**other)
    }
}

impl<T: ?Sized + Ord, A: Atomic> Ord for Arc<T, A> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        (**self).cmp(&**other)
    }
}

impl<T: ?Sized + Eq, A: Atomic> Eq for Arc<T, A> {}

impl<T: ?Sized + fmt::Display, A: Atomic> fmt::Display for Arc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<T: ?Sized + fmt::Debug, A: Atomic> fmt::Debug for Arc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: ?Sized, A: Atomic> fmt::Pointer for Arc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr(), f)
    }
}

impl<T: Default, A: Atomic> Default for Arc<T, A> {
    #[inline]
    fn default() -> Self {
        Arc::new(Default::default())
    }
}

impl<T: ?Sized + Hash, A: Atomic> Hash for Arc<T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<T, A: Atomic> From<T> for Arc<T, A> {
    #[inline]
    fn from(val: T) -> Self {
        Arc::new(val)
    }
}

impl<T: ?Sized, A: Atomic> Borrow<T> for Arc<T, A> {
    #[inline]
    fn borrow(&self) -> &T {
        &**self
    }
}

impl<T: ?Sized, A: Atomic> AsRef<T> for Arc<T, A> {
    #[inline]
    fn as_ref(&self) -> &T {
        &**self
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Deserialize<'de>, A: Atomic> Deserialize<'de> for Arc<T, A> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        T::deserialize(deserializer).map(Arc::new)
    }
}

#[cfg(feature = "serde")]
impl<T: Serialize, A: Atomic> Serialize for Arc<T, A> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        (**self).serialize(serializer)
    }
}

unsafe impl<T: ?Sized + Sync + Send, A: Atomic + Send + Sync> Send for Arc<T, A> {}
unsafe impl<T: ?Sized + Sync + Send, A: Atomic + Send + Sync> Sync for Arc<T, A> {}
