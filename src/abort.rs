#[cfg(feature = "std")]
pub(crate) use std::process::abort;

/// A `no_std`-compatible abort by forcing a panic while already panicking
#[cfg(not(feature = "std"))]
#[inline(never)]
#[cold]
pub(crate) fn abort() -> ! {
    #[cold]
    fn panic() -> ! {
        panic!()
    }

    struct PanicOnDrop;

    impl Drop for PanicOnDrop {
        fn drop(&mut self) {
            panic()
        }
    }

    let _panic_twice = PanicOnDrop;
    panic()
}
