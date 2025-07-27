# Squashfs_reader

<!-- cargo-rdme start -->

Squashfs_reader is a rust crate offering full read-only access to squashfs archive files. It offers an api similar to `std::io`.

### Features

- **Pure Rust:** Implementation of the entire SquashFS format specification.
- **Full Compression Support:** All possible compression formats (gzip, lzma, xz, lzo, lz4, and zstd) are supported.
- **Thread Safety:** This library is fully thread-safe.
- **Caching:** All accessed metadata and data blocks are cached using [`quick_cache`](https://crates.io/crates/quick_cache) to prevent unnecessary decompressions. The cache size can be configured.
- **Familiar API:** Directory iteration is supported with an API similar to [`std::fs::ReadDir`](https://doc.rust-lang.org/std/fs/struct.ReadDir.html), and files implement the [`std::io::Read`](https://doc.rust-lang.org/std/io/trait.Read.html) and [`std::io::Seek`](https://doc.rust-lang.org/std/io/trait.Seek.html) traits.

### Example

```rust
use std::io;
use squashfs_reader::FileSystem;

fn main() -> io::Result<()> {
    // Open a SquashFS file
    let fs = FileSystem::from_path("example.squashfs")?;
    
    // List contents of root directory
    let root = fs.read_dir("/")?;
    for entry in root {
        println!("{}", entry?.name());
    }
    
    // Read a file
    let mut file = fs.open("path/to/file.txt")?;
    let file_size = file.seek(io::SeekFrom::End(0))?;
    file.rewind()?;

    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    
    Ok(())
}
```

### Compression and features

- `best_performance` (default) - Uses external (non-Rust) libraries when they offer better performance ([`liblzma`](https://crates.io/crates/xz2) and [`zstd-safe`](https://crates.io/crates/zstd-safe)).
- `only_rust`- Only has Rust dependencies, but may offer lower performance when using some compression formats.

If both features are enabled, `only_rust` will be prioritized.

### Safety

This crate is entirely written in safe Rust (it uses `#![forbid(unsafe_code)]`). However, please note that some dependencies may contain unsafe code.

<!-- cargo-rdme end -->
