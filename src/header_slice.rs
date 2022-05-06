use crate::{
    arc::{Arc, ArcInner},
    Atomic,
};
use alloc::{alloc::Layout, boxed::Box};
use bytemuck::{allocation::zeroed_slice_box, Zeroable};
use core::{
    marker::PhantomData,
    mem::{self, size_of},
    ptr::{addr_of_mut, NonNull},
    slice,
};

/// Structure to allow Arc-managing some fixed-sized data and a variably-sized
/// slice in a single allocation.
#[derive(Debug, Eq, PartialEq, PartialOrd)]
pub struct HeaderSlice<H, T: ?Sized> {
    /// The fixed-sized data.
    pub header: H,
    /// The dynamically-sized data.
    pub slice: T,
}

/// Header data with an inline length. Consumers that use HeaderWithLength as the
/// Header type in HeaderSlice can take advantage of ThinArc.
#[derive(Debug, Eq, PartialEq, PartialOrd)]
pub struct HeaderWithLength<H> {
    /// The fixed-sized data.
    pub header: H,
    /// The slice length.
    length: usize,
}

impl<H> HeaderWithLength<H> {
    /// Creates a new HeaderWithLength.
    pub fn new(header: H, length: usize) -> Self {
        HeaderWithLength { header, length }
    }
}

pub type HeaderSliceWithLength<H, T> = HeaderSlice<HeaderWithLength<H>, T>;

impl<H, T, A> Arc<HeaderSlice<H, [T]>, A>
where
    A: Atomic,
{
    /// Creates an Arc for a HeaderSlice using the given header struct and
    /// iterator to generate the slice. The resulting Arc will be fat.
    #[inline]
    pub fn from_header_and_iter<I>(header: H, mut items: I) -> Self
    where
        I: Iterator<Item = T> + ExactSizeIterator,
    {
        assert_ne!(size_of::<T>(), 0, "Need to think about ZST");

        // Compute the required size for the allocation
        let num_items = items.len();
        let size = {
            let inner_layout = Layout::new::<ArcInner<HeaderSlice<H, [T; 0]>, A>>();
            let slice_layout = Layout::array::<T>(num_items)
                .expect("arithmetic overflow when trying to create array layout");

            let slice_align = mem::align_of_val::<[T]>(&[]);
            assert_eq!(slice_layout.align(), slice_align);

            let padding = padding_needed_for(&inner_layout, slice_align);
            inner_layout.size() + padding + slice_layout.size()
        };

        let ptr: *mut ArcInner<HeaderSlice<H, [T]>, A>;
        unsafe {
            // Allocate the buffer. We use Vec because the underlying allocation
            // machinery isn't available in stable Rust
            //
            // To avoid alignment issues, we allocate words rather than bytes,
            // rounding up to the nearest word size
            let buffer = if mem::align_of::<T>() <= mem::align_of::<usize>() {
                Self::allocate_buffer::<usize>(size)
            } else if mem::align_of::<T>() <= mem::align_of::<u64>() {
                // On 32-bit platforms <T> may have 8 byte alignment while usize has 4 byte alignment
                // Use u64 to avoid over-alignment
                // This branch will compile away in optimized builds
                Self::allocate_buffer::<u64>(size)
            } else {
                panic!("Over-aligned type not handled");
            };

            // Synthesize the fat pointer. We do this by claiming we have a direct
            // pointer to a [T], and then changing the type of the borrow. The key
            // point here is that the length portion of the fat pointer applies
            // only to the number of elements in the dynamically-sized portion of
            // the type, so the value will be the same whether it points to a [T]
            // or something else with a [T] as its last member
            let fake_slice: &mut [T] = slice::from_raw_parts_mut(buffer.cast(), num_items);
            ptr = fake_slice as *mut [T] as *mut ArcInner<HeaderSlice<H, [T]>, A>;

            // Write the data.
            //
            // Note that any panics here (i.e. from the iterator) are safe, since
            // we'll just leak the uninitialized memory
            addr_of_mut!((*ptr).count).write(A::ONE);
            addr_of_mut!((*ptr).data.header).write(header);

            if let Some(current) = (*ptr).data.slice.get_mut(0) {
                let mut current: *mut T = current;
                for _ in 0..num_items {
                    current.write(
                        items
                            .next()
                            .expect("ExactSizeIterator over-reported length"),
                    );
                    current = current.offset(1);
                }

                assert!(
                    items.next().is_none(),
                    "ExactSizeIterator under-reported length",
                );

                // We should have consumed the buffer exactly
                debug_assert_eq!(current.cast(), buffer.add(size));
            }
        }

        // Return the fat Arc
        assert_eq!(
            size_of::<Self>(),
            size_of::<usize>() * 2,
            "The Arc will be fat",
        );

        Arc {
            // Safety: we have just created the underlying allocation
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            __type: PhantomData,
        }
    }

    #[inline]
    unsafe fn allocate_buffer<W: Zeroable>(size: usize) -> *mut u8 {
        let words_to_allocate = divide_rounding_up(size, size_of::<W>());
        let allocated = zeroed_slice_box::<W>(words_to_allocate);

        Box::into_raw(allocated).cast()
    }
}

#[inline]
const fn divide_rounding_up(dividend: usize, divisor: usize) -> usize {
    (dividend + divisor - 1) / divisor
}

#[inline]
const fn padding_needed_for(layout: &Layout, align: usize) -> usize {
    let len = layout.size();
    let len_rounded_up = len.wrapping_add(align).wrapping_sub(1) & !align.wrapping_sub(1);

    len_rounded_up.wrapping_sub(len)
}
