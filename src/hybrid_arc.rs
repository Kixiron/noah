#![cfg(feature = "std")]
//! A hybrid arc implementation based on [Biased Reference Counting](https://dl.acm.org/doi/10.1145/3243176.3243195)
// TODO: We currently only offer this when the `std` feature is enabled because of `ThreadId`, but
//       we could just conditionally elide the `local_id` and `local_count` fields when `std` is disabled,
//       making this nearly identical to a normal `Arc`

use crate::{abort::abort, Atomic};
use core::{
    borrow::Borrow,
    cell::Cell,
    cmp,
    fmt::{self, Debug, Display, Pointer},
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem::{forget, size_of, ManuallyDrop},
    ops::Deref,
    pin::Pin,
    ptr::{addr_of, NonNull},
    sync::atomic::{AtomicUsize, Ordering},
};
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::thread::{self, ThreadId};

#[repr(transparent)]
pub struct HybridArc<T: ?Sized, A: Atomic = AtomicUsize> {
    /// A pointer to the heap-allocated stub
    ptr: NonNull<HybridArcInner<T, A>>,
    /// Convey ownership over a `T`
    __type: PhantomData<T>,
}

impl<T: ?Sized, A: Atomic> HybridArc<T, A> {
    /// Construct a [`HybridArc`]
    #[inline]
    pub fn new(data: T) -> Self
    where
        T: Sized,
    {
        let ptr = Box::new(HybridArcInner {
            header: HybridArcHeader::new(),
            data,
        });

        unsafe {
            Self {
                ptr: NonNull::new_unchecked(Box::into_raw(ptr)),
                __type: PhantomData,
            }
        }
    }

    /// Convert the [`HybridArc`] to a raw pointer, suitable for use across FFI
    #[inline]
    pub fn into_raw(this: Self) -> *const T {
        let ptr = unsafe { addr_of!((*this.ptr()).data) as *const T };
        forget(this);

        ptr
    }

    /// Reconstruct the [`HybridArc`] from a raw pointer obtained from [`HybridArc::into_raw`]
    ///
    /// # Safety
    ///
    /// The pointer supplied *must* have come from `HybridArc::into_raw`
    ///
    #[inline]
    pub unsafe fn from_raw(ptr: *const T) -> Self
    where
        T: Sized,
    {
        // To find the corresponding pointer to the `HybridArcInner` we need
        // to subtract the offset of the `data` field from the pointer.
        let ptr = (ptr as *const u8).sub(size_of::<A>());
        Self {
            ptr: NonNull::new_unchecked(ptr as *mut HybridArcInner<T, A>),
            __type: PhantomData,
        }
    }

    /// Returns the address on the heap of the [`HybridArc`] itself -- not the T within it
    /// -- for memory reporting.
    #[inline]
    pub fn heap_ptr(&self) -> *const () {
        self.ptr.as_ptr() as *const HybridArcInner<T, A> as *const ()
    }

    /// Test pointer equality between the two [`HybridArc`]s, i.e. they must be the *same*
    /// allocation
    #[inline]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        this.ptr() == other.ptr()
    }

    #[inline]
    fn inner(&self) -> &HybridArcInner<T, A> {
        // This unsafety is ok because while this arc is alive we're guaranteed
        // that the inner pointer is valid. Furthermore, we know that the
        // `HybridArcInner` structure itself is `Sync` because the inner data is
        // `Sync` as well, so we're ok loaning out an immutable pointer to these
        // contents.
        unsafe { &*self.ptr() }
    }

    #[inline]
    fn header(&self) -> &HybridArcHeader<A> {
        &self.inner().header
    }

    // Non-inlined part of `drop`. Just invokes the destructor.
    #[inline(never)]
    unsafe fn drop_slow(&mut self) {
        let _ = Box::from_raw(self.ptr());
    }

    /// Get the pointer to the current [`HybridArc`]'s [`HybridArcInner`]
    #[inline]
    fn ptr(&self) -> *mut HybridArcInner<T, A> {
        self.ptr.as_ptr()
    }

    /// Makes a mutable reference to the `HybridArc`, cloning if necessary
    ///
    /// This is functionally equivalent to [`Arc::make_mut`][mm] from the standard library.
    ///
    /// If this `HybridArc` is uniquely owned, `make_mut()` will provide a mutable
    /// reference to the contents. If not, `make_mut()` will create a _new_ `HybridArc`
    /// with a copy of the contents, update `this` to point to it, and provide
    /// a mutable reference to its contents.
    ///
    /// This is useful for implementing copy-on-write schemes where you wish to
    /// avoid copying things if your `HybridArc` is not shared.
    ///
    /// [mm]: https://doc.rust-lang.org/stable/std/sync/struct.Arc.html#method.make_mut
    #[inline]
    pub fn make_mut(this: &mut Self) -> &mut T
    where
        T: Clone,
    {
        if !this.is_unique() {
            // Another pointer exists; clone
            *this = Self::new(T::clone(&**this));
        }

        unsafe {
            // This unsafety is ok because we're guaranteed that the pointer
            // returned is the *only* pointer that will ever be returned to T. Our
            // reference count is guaranteed to be 1 at this point, and we required
            // the Arc itself to be `mut`, so we're returning the only possible
            // reference to the inner data.
            Self::get_mut_unchecked(this)
        }
    }

    /// Provides mutable access to the contents *if* the `HybridArc` is uniquely owned.
    #[inline]
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        if this.is_unique() {
            // This unsafety is ok because we're guaranteed that the pointer
            // returned is the *only* pointer that will ever be returned to T. Our
            // reference count is guaranteed to be 1 at this point, and we required
            // the `Arc` itself to be `mut`, so we're returning the only possible
            // reference to the inner data.
            Some(unsafe { Self::get_mut_unchecked(this) })
        } else {
            None
        }
    }

    /// Returns a mutable reference into the given `HybridArc`, without any checks.
    ///
    /// See also [`HybridArc::get_mut`], which is safe and does appropriate checks.
    ///
    /// # Safety
    ///
    /// Any other `HybridArc` pointers to the same allocation must not be dereferenced
    /// for the duration of the returned borrow.
    /// This is trivially the case if no such pointers exist,
    /// for example immediately after `HybridArc::new`.
    #[allow(unused_unsafe)]
    #[inline]
    pub unsafe fn get_mut_unchecked(this: &mut Self) -> &mut T {
        // We are careful to *not* create a reference covering the "count" fields, as
        // this would alias with concurrent access to the reference counts (e.g. by `Weak`).
        unsafe { &mut (*this.ptr.as_ptr()).data }
    }

    /// Whether or not the `Arc` is uniquely owned (is the refcount 1?).
    #[inline]
    pub fn is_unique(&self) -> bool {
        let header = self.header();

        // Check if the current thread is the local thread and if the local thread's
        // reference count is 1
        let local_unique =
            header.local_id == thread_id() && header.local_count.get() == A::ONE_NON_ATOMIC;

        // This needs to be an `Acquire` to synchronize with the decrement of the `strong`
        // counter in `drop` -- the only access that happens when any but the last reference
        // is being dropped.
        local_unique && header.count.load(Ordering::Acquire) == 1
    }

    /// Gets the number of pointers to this allocation
    #[inline]
    pub fn count(this: &Self) -> u64 {
        let header = this.header();

        let mut count = header.count.load(Ordering::Acquire);
        if header.local_id == thread_id() {
            count += A::non_atomic_to_u64(header.local_count.get());
        }

        count
    }

    /// Decrements the reference count on the [`HybridArc`] associated with the provided pointer by one
    ///
    /// # Safety
    ///
    /// The pointer must have been obtained through [`HybridArc::into_raw`], and the associated `HybridArc`
    /// instance must be valid (i.e. the strong count must be at least 1) for the duration of this method.
    ///
    #[allow(unused_unsafe)]
    #[inline]
    pub unsafe fn inc_count(ptr: *const T)
    where
        T: Sized,
    {
        // Retain Arc, but don't touch refcount by wrapping in a `ManuallyDrop`
        let arc = unsafe { ManuallyDrop::new(Self::from_raw(ptr)) };

        // Now increase refcount, but don't drop new refcount either
        let _arc_clone: ManuallyDrop<_> = arc.clone();
    }

    /// Decrements the reference count on the [`HybridArc`] associated with the provided pointer by one.
    /// This will cause the inner `HybridArc` to be dropped if the reference count is zero which will
    /// create a double free if another clone of the same `HybridArc` is later dropped.
    ///
    /// # Safety
    ///
    /// The pointer must have been obtained through [`HybridArc::into_raw`], and the associated `HybridArc`
    /// instance must be valid (i.e. the reference count must be at least 1) when invoking this method.
    /// This method can be used to release the final `HybridArc` and backing storage, but should not be
    /// called after the final `HybridArc` has been released.
    ///
    #[allow(unused_unsafe)]
    #[inline]
    pub unsafe fn dec_count(ptr: *const T)
    where
        T: Sized,
    {
        unsafe { drop(Self::from_raw(ptr)) };
    }

    /// Constructs a new `Pin<HybridArc<T>>`. If `T` does not implement [`Unpin`](core::marker::Unpin),
    /// then `data` will be pinned in memory and unable to be moved.
    #[inline]
    pub fn pinned(data: T) -> Pin<Self>
    where
        T: Sized,
    {
        unsafe { Pin::new_unchecked(Self::new(data)) }
    }
}

impl<T, A> Clone for HybridArc<T, A>
where
    T: ?Sized,
    A: Atomic,
{
    #[inline]
    fn clone(&self) -> Self {
        let header = self.header();
        let thread_id = thread_id();

        // If this is the parent thread, clone non-atomically
        if thread_id == header.local_id {
            let local_count = header.local_count.get() + A::ONE_NON_ATOMIC;
            header.local_count.set(local_count);

            // Guard against massive refcounts
            if A::non_atomic_to_u64(local_count) > A::MAX_NONATOMIC {
                abort();
            }

        // If this isn't the parent thread, clone atomically
        } else {
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
            let old_size = header.count.fetch_add(1, Ordering::Relaxed);

            // However we need to guard against massive ref counts in case someone
            // is `forget`ing Arcs. If we don't do this the count can overflow
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
        }

        Self {
            ptr: unsafe { NonNull::new_unchecked(self.ptr()) },
            __type: PhantomData,
        }
    }
}

impl<T: ?Sized, A: Atomic> Deref for HybridArc<T, A> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        &self.inner().data
    }
}

impl<T: ?Sized, A: Atomic> Drop for HybridArc<T, A> {
    #[inline]
    fn drop(&mut self) {
        let header = self.header();
        let thread_id = thread_id();

        // If this is the parent thread, decrement non-atomically
        if thread_id == header.local_id {
            let local_count = header.local_count.get() - A::ONE_NON_ATOMIC;
            header.local_count.set(local_count);

            // If we decrement the local refcount and this isn't the last local reference,
            // we're done dropping. Otherwise if this is the last local reference we want to continue
            // on to decrement the arc concurrently
            if A::non_atomic_to_u64(local_count) != 0 {
                return;
            }
        }

        // Because `fetch_sub` is already atomic, we do not need to synchronize
        // with other threads unless we are going to delete the object.
        if header.count.fetch_sub(1, Ordering::Release) != 1 {
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
        let _ = header.count.load(Ordering::Acquire);

        unsafe { self.drop_slow() }
    }
}

impl<T: ?Sized + PartialEq, A: Atomic> PartialEq for HybridArc<T, A> {
    fn eq(&self, other: &Self) -> bool {
        Self::ptr_eq(self, other) || (**self).eq(&**other)
    }

    #[allow(clippy::partialeq_ne_impl)]
    fn ne(&self, other: &Self) -> bool {
        !Self::ptr_eq(self, other) && (**self).ne(&**other)
    }
}

impl<T: ?Sized + PartialOrd, A: Atomic> PartialOrd for HybridArc<T, A> {
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

impl<T: ?Sized + Ord, A: Atomic> Ord for HybridArc<T, A> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        (**self).cmp(&**other)
    }
}

impl<T: ?Sized + Eq, A: Atomic> Eq for HybridArc<T, A> {}

impl<T: ?Sized + Display, A: Atomic> Display for HybridArc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&**self, f)
    }
}

impl<T: ?Sized + Debug, A: Atomic> Debug for HybridArc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&**self, f)
    }
}

impl<T: ?Sized, A: Atomic> Pointer for HybridArc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Pointer::fmt(&self.ptr(), f)
    }
}

impl<T: Default, A: Atomic> Default for HybridArc<T, A> {
    #[inline]
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<T, A> Hash for HybridArc<T, A>
where
    T: ?Sized + Hash,
    A: Atomic,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<T, A: Atomic> From<T> for HybridArc<T, A> {
    #[inline]
    fn from(val: T) -> Self {
        Self::new(val)
    }
}

impl<T: ?Sized, A: Atomic> Borrow<T> for HybridArc<T, A> {
    #[inline]
    fn borrow(&self) -> &T {
        &**self
    }
}

impl<T: ?Sized, A: Atomic> AsRef<T> for HybridArc<T, A> {
    #[inline]
    fn as_ref(&self) -> &T {
        &**self
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Deserialize<'de>, A: Atomic> Deserialize<'de> for HybridArc<T, A> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        T::deserialize(deserializer).map(Self::new)
    }
}

#[cfg(feature = "serde")]
impl<T: Serialize, A: Atomic> Serialize for HybridArc<T, A> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        (**self).serialize(serializer)
    }
}

impl<T: ?Sized, A: Atomic> Unpin for HybridArc<T, A> {}

unsafe impl<T: ?Sized + Sync + Send, A: Atomic + Send + Sync> Send for HybridArc<T, A> {}
unsafe impl<T: ?Sized + Sync + Send, A: Atomic + Send + Sync> Sync for HybridArc<T, A> {}

#[repr(C)]
struct HybridArcInner<T: ?Sized, A: Atomic> {
    /// The arc's header
    header: HybridArcHeader<A>,
    /// The arc's data
    data: T,
}

struct HybridArcHeader<A: Atomic> {
    /// The atomic's concurrent reference count
    count: A,
    /// The id of the thread which can non-atomically modify the reference count
    local_id: ThreadId,
    /// The non-atomic local reference count
    local_count: Cell<A::NonAtomic>,
}

impl<A: Atomic> HybridArcHeader<A> {
    #[inline]
    fn new() -> Self {
        Self {
            count: A::ONE,
            local_id: thread_id(),
            local_count: Cell::new(A::ONE_NON_ATOMIC),
        }
    }
}

#[inline]
fn thread_id() -> ThreadId {
    thread::current().id()
}
