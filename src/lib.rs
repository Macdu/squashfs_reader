//! Squashfs_reader is a rust crate offering full read-only access to squashfs archive files. It offers an api similar to `std::io`.
//!
//! ## Features
//!
//! - **Pure Rust:** Implementation of the entire SquashFS format specification.
//! - **Full Compression Support:** All possible compression formats (gzip, lzma, xz, lzo, lz4, and zstd) are supported.
//! - **Thread Safety:** This library is fully thread-safe.
//! - **Caching:** All accessed metadata and data blocks are cached using [`quick_cache`](https://crates.io/crates/quick_cache) to prevent unnecessary decompressions. The cache size can be configured.
//! - **Familiar API:** Directory iteration is supported with an API similar to [`std::fs::ReadDir`](https://doc.rust-lang.org/std/fs/struct.ReadDir.html), and files implement the [`std::io::Read`](https://doc.rust-lang.org/std/io/trait.Read.html) and [`std::io::Seek`](https://doc.rust-lang.org/std/io/trait.Seek.html) traits.
//!
//! ## Example
//!
//! ```
//! use std::io;
//! use squashfs_reader::FileSystem;
//!
//! fn main() -> io::Result<()> {
//!     // Open a SquashFS file
//!     let fs = FileSystem::from_path("example.squashfs")?;
//!     
//!     // List contents of root directory
//!     let root = fs.read_dir("/")?;
//!     for entry in root {
//!         println!("{}", entry?.name());
//!     }
//!     
//!     // Read a file
//!     let mut file = fs.open("path/to/file.txt")?;
//!     let file_size = file.seek(io::SeekFrom::End(0))?;
//!     file.rewind()?;
//!
//!     let mut contents = String::new();
//!     file.read_to_string(&mut contents)?;
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Compression and features
//!
//! - `best_performance` (default) - Uses external (non-Rust) libraries when they offer better performance ([`liblzma`](https://crates.io/crates/xz2) and [`zstd-safe`](https://crates.io/crates/zstd-safe)).
//! - `only_rust`- Only has Rust dependencies, but may offer lower performance when using some compression formats.
//!
//! If both features are enabled, `only_rust` will be prioritized.
//!
//! ## Safety
//!
//! This crate is entirely written in safe Rust (it uses `#![forbid(unsafe_code)]`). However, please note that some dependencies may contain unsafe code.

#![forbid(unsafe_code)]

use std::{
    cmp, fs,
    io::{self, Read, Seek},
    ops::Deref,
    path::{self, Path},
    sync::{Arc, OnceLock, Weak},
    time::{Duration, SystemTime},
};

use decompression::decompress_block;
use metadata::MetadataType;
use quick_cache::sync::{Cache, GuardResult};
use readers::{BlockLookupTable, METADATA_BLOCK_SIZE, MetadataReader, ReadAt, SharedReader};
use structs::{CompressorType, FragmentEntry, Superblock};
use zerocopy::{FromBytes, IntoBytes, TryFromBytes, little_endian};

mod decompression;
mod lzo;
mod metadata;
mod readers;
mod structs;

pub use metadata::{DirEntry, Metadata, ReadDir};
pub use readers::FileReader;

#[cfg(not(feature = "any_impl"))]
compile_error!("You need to enable the best_performance (default) or only_rust feature");

pub type Error = io::Error;
pub type Result<T> = io::Result<T>;

/// Compressor used for both data and meta data blocks
#[derive(Clone, Copy)]
pub enum Compression {
    /// zlib deflate (no gzip header)
    Gzip,
    /// LZMA 1 (considered deprecated)
    Lzma,
    Lzo,
    /// LZMA 2
    Xz,
    Lz4,
    Zstd,
}

/// Different types of Files which can be stored in a squashfs
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    Directory,
    File,
    Symlink,
}

/// In the SquashFS archive format, metadata entries (e.g. inodes) are often referenced using a 64 bit integer.
/// The lower 16 bit hold an offset into the uncompressed block and the upper 48 bit point to the on-disk location of the block
/// Relative to the start of the metadata entry (inodes, file listings...)
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct MetadataRef(u64);

impl MetadataRef {
    /// Return the on-disk blocation the block containing the beginning of the metadata is stored to
    fn block_location(self) -> u64 {
        self.0 >> 16
    }

    /// Contain the offset this structure starts from within the block
    fn offset_within_block(self) -> u16 {
        self.0 as u16
    }

    /// Create a metadata ref from the block start and offset within the block to read from
    fn from_block_and_offset(block: u64, offset: u16) -> Self {
        assert!(block < (1 << 48));
        Self((block << 16) | (offset as u64))
    }

    pub fn into_inner(self) -> u64 {
        self.0
    }
}

impl From<u64> for MetadataRef {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

#[derive(Clone)]
struct CachedBlock(Arc<[u8]>);

impl Deref for CachedBlock {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

type BlockCache = quick_cache::sync::Cache<u64, CachedBlock>;
struct Caches {
    metadata_cache: BlockCache,
    file_cache: BlockCache,
}

/// Options which can be used to configure how a squashfs filesystem is opened
/// So far only the cache size can be configured
#[derive(Clone)]
pub struct OpenOptions {
    metadata_cache_size: usize,
    file_cache_size: usize,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenOptions {
    /// Initialize the [OpenOptions] with the default value
    /// Right now, this implies:
    /// - A metadata block cache of size 512KiB
    /// - A file block cache of size 8MiB
    pub const fn new() -> Self {
        Self {
            metadata_cache_size: 512 * 1024,
            file_cache_size: 8 * 1024 * 1024,
        }
    }

    /// Specify how much memory can be used to cache metadata blocks
    /// Metadata blocks are always 8KiB, this value will be rounded up to the
    /// next multiple of 8KiB
    pub fn metadata_cache_size(&mut self, len: usize) -> &mut Self {
        self.metadata_cache_size = len;
        self
    }

    /// Specify how much memory can be used to cache file blocks
    /// File blocks are always the same size, between 4KiB and 1MiB
    /// This is configured in the squashfs header
    /// This value will be rounded up to the next multiple of the block size
    pub fn file_cache_size(&mut self, len: usize) -> &mut Self {
        self.file_cache_size = len;
        self
    }
}

/// The Filesystem structure, this is used to interact with the content of the squashfs file.
/// This is the entrypoint from the library.
/// From there, one can use functions such as read_dir, metadata or open to search and open files
/// inside the squashfs
#[derive(Clone)]
pub struct FileSystem<T: ReadAt>(Arc<FileSystemInner<T>>);

#[cfg(any(windows, unix))]
impl FileSystem<fs::File> {
    #[cfg(any(windows, unix))]
    pub fn from_path<P>(path: P) -> Result<Self>
    where
        P: AsRef<path::Path>,
    {
        Self::from_path_with_options(path, OpenOptions::default())
    }

    pub fn from_path_with_options<P>(path: P, options: OpenOptions) -> Result<Self>
    where
        P: AsRef<path::Path>,
    {
        FileSystemInner::from_path_with_options(path, options).map(|inner| Self(inner))
    }
}

impl<R: Read + Seek> FileSystem<SharedReader<R>> {
    pub fn from_read(reader: R) -> Result<Self> {
        Self::from_with_options(SharedReader::new(reader), OpenOptions::default())
    }
}

impl<T: ReadAt> FileSystem<T> {
    pub fn from(reader: T) -> Result<Self> {
        Self::from_with_options(reader, OpenOptions::default())
    }

    pub fn from_with_options(reader: T, options: OpenOptions) -> Result<Self> {
        Self::from_read_at_with_options(reader, options)
    }

    pub fn from_read_at(reader: T) -> Result<Self> {
        Self::from_read_at_with_options(reader, OpenOptions::default())
    }

    pub fn from_read_at_with_options(reader: T, options: OpenOptions) -> Result<Self> {
        FileSystemInner::from_read_at_with_options(reader, options).map(|inner| Self(inner))
    }

    /// Return the last time the archive was modified
    pub fn modified(&self) -> SystemTime {
        // Cannot overflow
        self.0.modified()
    }

    /// Given an inode number, return the matching metadata ref
    /// Note that this is only supported if the squashfs file comes with an export table
    /// Otherwise [io::ErrorKind::Unsupported] is returned
    pub fn inode_ref_from_number(&self, inode: u32) -> Result<MetadataRef> {
        self.0.inode_ref_from_number(inode)
    }

    /// Given an inode ref, returns its metadata
    pub fn metadata_from_ref(&self, meta_ref: MetadataRef) -> Result<Metadata> {
        self.0.metadata_from_ref(meta_ref)
    }

    /// Given a path inside the archive, return its inode ref if its exists
    pub fn inode_ref_from_path<P: AsRef<Path>>(&self, path: P) -> Result<MetadataRef> {
        self.0.inode_ref_from_path(path.as_ref())
    }

    /// Given the metadata of a directory, returns a structure enumerating entries inside this directory
    pub fn read_dir_from_metadata(&self, metadata: &Metadata) -> Result<ReadDir<T>> {
        self.0.read_dir_from_metadata(metadata)
    }

    /// Given the metadata of a directory, returns a reader for the content of the file
    pub fn open_from_metadata(&self, metadata: &Metadata) -> Result<FileReader<T>> {
        self.0.open_from_metadata(metadata)
    }

    /// Given a path inside the archive, returns its metadata if the entry exists
    pub fn metadata<P: AsRef<Path>>(&self, path: P) -> Result<Metadata> {
        self.0.metadata(path.as_ref())
    }

    /// Given a path inside the archive, returns a structure enumerating entries inside this directory if this is a directory's path
    pub fn read_dir<P: AsRef<Path>>(&self, path: P) -> Result<ReadDir<T>> {
        self.0.read_dir(path.as_ref())
    }

    /// Given a path inside the archive, returns a reader for the content of the file if this is a file's path
    pub fn open<P: AsRef<Path>>(&self, path: P) -> Result<FileReader<T>> {
        self.0.open(path.as_ref())
    }

    /// Given a path, returns if an entry exists for this path
    /// Note that this is a wrapper for [Self::inode_ref_from_path], if you want to do more than check for existence,
    /// You should use [Self::inode_ref_from_path] or [Self::metadata] to avoid the double lookup cost
    pub fn exists<P: AsRef<Path>>(&self, path: P) -> bool {
        self.0.exists(path.as_ref())
    }
}

struct ArchiveInfo {
    /// The size of a data block in bytes. Must be a power of two between 4096 (4k) and 1048576 (1 MiB)
    block_size: u32,
    /// A reference to the inode of the root directory.
    root_ref: MetadataRef,
    /// Compression algorithm used for both data and meta data blocks
    compression: Compression,
    /// The byte offset at which the inode table starts.
    inode_table: u64,
    /// The byte offset at which the directory table starts.
    dir_table: u64,
    /// Last modification time of the archive. Count seconds since 00:00, Jan 1st 1970 UTC
    mod_time: u32,
}

#[doc(hidden)]
pub struct FileSystemInner<T: ReadAt> {
    weak: Weak<Self>,
    reader: T,
    info: ArchiveInfo,
    caches: Caches,
    fragment_lookup: BlockLookupTable<FragmentEntry>,
    id_lookup: BlockLookupTable<little_endian::U32>,
    /// Export lookup table, which can be used to translate an inode id to an inode ref
    /// This field is optional
    export_lookup: Option<BlockLookupTable<little_endian::U64>>,
    /// If a block has size 0, it is a sparse block, we keep a single sparse block for all
    /// encountered full-size file sparse blocks
    file_sparse_block: OnceLock<CachedBlock>,
}

#[cfg(any(windows, unix))]
impl FileSystemInner<fs::File> {
    fn from_path_with_options<P>(path: P, options: OpenOptions) -> Result<Arc<Self>>
    where
        P: AsRef<path::Path>,
    {
        let file = fs::File::open(path)?;
        // On windows/unix, file already implements our ReadAt trait,
        // So we can avoid one level of indirection
        Self::from_read_at_with_options(file, options)
    }
}

impl<T: ReadAt> FileSystemInner<T> {
    fn from_read_at_with_options(reader: T, options: OpenOptions) -> Result<Arc<Self>> {
        let invalid_err = || io::Error::new(io::ErrorKind::InvalidData, "Invalid squashfs header");

        let mut superblock_bytes = [0u8; size_of::<Superblock>()];
        reader.read_exact_at(&mut superblock_bytes, 0)?;
        let superblock =
            Superblock::read_from_bytes(&superblock_bytes).map_err(|_| invalid_err())?;

        // Magic check
        if &superblock.magic != b"hsqs" {
            return Err(invalid_err());
        }

        // block size check
        let block_size = superblock.block_size.get();
        if block_size != (1 << superblock.block_log.get()) {
            return Err(invalid_err());
        }

        let metata_cache_blocks = options
            .metadata_cache_size
            .next_multiple_of(METADATA_BLOCK_SIZE as usize)
            / METADATA_BLOCK_SIZE as usize;
        let file_cache_blocks = options
            .file_cache_size
            .next_multiple_of(block_size as usize)
            / block_size as usize;
        let caches = Caches {
            metadata_cache: Cache::new(metata_cache_blocks.max(1)),
            file_cache: Cache::new(file_cache_blocks.max(1)),
        };

        let root_ref = MetadataRef::from(superblock.root_inode.get());

        let compression =
            match CompressorType::try_read_from_bytes(superblock.compressor.get().as_bytes())
                .map_err(|_| invalid_err())?
            {
                CompressorType::Gzip => Compression::Gzip,
                CompressorType::Lzma => Compression::Lzma,
                CompressorType::Lzo => Compression::Lzo,
                CompressorType::Xz => Compression::Xz,
                CompressorType::Lz4 => Compression::Lz4,
                CompressorType::Zstd => Compression::Zstd,
            };

        let fragment_lookup = BlockLookupTable::new(
            &reader,
            superblock.frag_table.get(),
            superblock.frag_count.get(),
        )?;

        let id_lookup = BlockLookupTable::new(
            &reader,
            superblock.id_table.get(),
            superblock.id_count.get() as u32,
        )?;

        let export_lookup = (superblock.export_table.get() != !0)
            .then(|| {
                BlockLookupTable::new(
                    &reader,
                    superblock.export_table.get(),
                    superblock.inode_count.get(),
                )
            })
            .transpose()?;

        let info = ArchiveInfo {
            block_size,
            root_ref,
            compression,
            inode_table: superblock.inode_table.get(),
            dir_table: superblock.dir_table.get(),
            mod_time: superblock.mod_time.get(),
        };

        Ok(Arc::new_cyclic(|weak| Self {
            weak: weak.clone(),
            reader,
            info,
            caches,
            fragment_lookup,
            id_lookup,
            export_lookup,
            file_sparse_block: OnceLock::new(),
        }))
    }

    fn modified(&self) -> SystemTime {
        // Cannot overflow
        SystemTime::UNIX_EPOCH
            .checked_add(Duration::from_secs(self.info.mod_time as u64))
            .unwrap()
    }

    /// Given an inode number, return the matching metadata ref
    /// Note that this is only supported if the squashfs file comes with an export table
    /// Otherwise [io::ErrorKind::Unsupported] is returned
    fn inode_ref_from_number(&self, inode: u32) -> Result<MetadataRef> {
        let lookup = self
            .export_lookup
            .as_ref()
            .ok_or(io::ErrorKind::Unsupported)?;
        // the lookup table starts at inode number 1
        lookup
            .lookup(inode - 1, self)
            .map(|meta_ref| meta_ref.get().into())
    }

    fn metadata_from_ref(&self, meta_ref: MetadataRef) -> Result<Metadata> {
        let mut reader = MetadataReader::new(self, meta_ref, self.info.inode_table);
        Metadata::read(&mut reader, meta_ref)
    }

    fn inode_ref_from_path(&self, path: &Path) -> Result<MetadataRef> {
        let mut curr_inode = self.info.root_ref;
        for component in path.components() {
            match component {
                path::Component::RootDir => (),
                // We shouldn't have any prefix or start from the current directory in a squashfs path
                path::Component::Prefix(_) | path::Component::CurDir => {
                    return Err(io::ErrorKind::InvalidInput.into());
                }
                // Geting the parent directory is not supported on squashfs archives
                path::Component::ParentDir => Err(io::ErrorKind::InvalidInput)?,
                path::Component::Normal(comp) => {
                    let mut reader = MetadataReader::new(self, curr_inode, self.info.inode_table);
                    let meta = Metadata::read(&mut reader, curr_inode)?;
                    let MetadataType::Directory(meta_dir) = meta.ty else {
                        return Err(io::ErrorKind::NotFound.into());
                    };

                    // sqashfs entries are in ASCIIbetical order, so look at the bytes
                    let comp_bytes = comp.as_encoded_bytes();
                    let (_, comp_ref) = meta_dir
                        .read_dir_for_entry(reader, self, curr_inode, comp_bytes)?
                        .filter_map(|entry| entry.ok())
                        // compare comp and the entry, stop when comp > entry
                        // also, try not to do comparisons twice
                        .map(|entry| {
                            (
                                Ord::cmp(entry.name().as_bytes(), comp_bytes),
                                entry.inode_ref,
                            )
                        })
                        .take_while(|(ord, _)| {
                            matches!(ord, cmp::Ordering::Less | cmp::Ordering::Equal)
                        })
                        .find(|(ord, _)| *ord == cmp::Ordering::Equal)
                        .ok_or(io::ErrorKind::NotFound)?;
                    curr_inode = comp_ref;
                }
            }
        }
        Ok(curr_inode)
    }

    fn read_dir_from_metadata(&self, metadata: &Metadata) -> Result<ReadDir<T>> {
        let MetadataType::Directory(meta_dir) = &metadata.ty else {
            return Err(io::ErrorKind::NotADirectory.into());
        };

        Ok(meta_dir.read_dir(self.weak.upgrade().unwrap(), metadata.meta_ref))
    }

    fn open_from_metadata(&self, metadata: &Metadata) -> Result<FileReader<T>> {
        let MetadataType::File(meta_file) = &metadata.ty else {
            return Err(io::ErrorKind::IsADirectory.into());
        };

        meta_file.read_file(metadata.meta_ref, self.weak.upgrade().unwrap())
    }

    fn metadata(&self, path: &Path) -> Result<Metadata> {
        let inode_ref = self.inode_ref_from_path(path)?;
        self.metadata_from_ref(inode_ref)
    }

    fn read_dir(&self, path: &Path) -> Result<ReadDir<T>> {
        let metadata = self.metadata(path.as_ref())?;
        self.read_dir_from_metadata(&metadata)
    }

    fn open(&self, path: &Path) -> Result<FileReader<T>> {
        let metadata = self.metadata(path.as_ref())?;
        self.open_from_metadata(&metadata)
    }

    fn exists(&self, path: &Path) -> bool {
        self.inode_ref_from_path(path.as_ref()).is_ok()
    }

    /// Read the block at offset and cache it
    /// If the block is already cached, return it
    /// Otherwise, read it using [Self::read_block]
    fn cached_read(
        &self,
        offset: u64,
        is_metadata: bool,
        fill_block: bool,
        compressed_len: u32,
        max_decompressed_len: u32,
        is_compressed: bool,
    ) -> Result<CachedBlock> {
        // Special handling of sparse blocks, so that they don't pollute the cache
        if compressed_len == 0 && fill_block && max_decompressed_len == self.info.block_size {
            let block = self.file_sparse_block.get_or_init(|| {
                let data = vec![0u8; max_decompressed_len as usize];
                CachedBlock(data.into())
            });
            return Ok(block.clone());
        }

        let cache = if is_metadata {
            &self.caches.metadata_cache
        } else {
            &self.caches.file_cache
        };

        let cache_key = if compressed_len == 0 {
            // Multiple sparse (zero-filled) blocks can have the same location and different sizes
            // So use the size as the key in this case
            (1u64 << 63) | max_decompressed_len as u64
        } else {
            offset
        };

        let block = match cache.get_value_or_guard(&cache_key, None) {
            // The block was already in the cache, just use it
            GuardResult::Value(block) => block,
            GuardResult::Guard(placeholder_guard) => {
                // We don't have the block in the cache, therefore read it now
                let block = self.read_block(
                    offset,
                    compressed_len,
                    max_decompressed_len,
                    fill_block,
                    is_compressed,
                )?;
                // This can fail if we use the cache insert or remove function in the mean time
                // We don't and even if it were the case, it wouldn't matter
                let _ = placeholder_guard.insert(block.clone());
                block
            }
            // We don't have a timeout
            GuardResult::Timeout => unreachable!(),
        };

        Ok(block)
    }

    /// Read a block of size compressed_len at the given offset in the archive
    /// If the block is compressed, it will be decompressed according to the compression algorithm
    /// specified in [ArchiveInfo::compression]
    /// If fill_block is set to true, the resulting block will be padded with zeros to be exactly max_decompressed_len
    fn read_block(
        &self,
        offset: u64,
        compressed_len: u32,
        max_decompressed_len: u32,
        fill_block: bool,
        is_compressed: bool,
    ) -> Result<CachedBlock> {
        let mut compressed = vec![0u8; compressed_len as usize];
        self.reader.read_exact_at(&mut compressed, offset)?;

        if is_compressed && compressed_len > 0 {
            decompress_block(
                &compressed,
                max_decompressed_len,
                fill_block,
                self.info.compression,
            )
        } else {
            debug_assert!(compressed_len <= max_decompressed_len);
            if fill_block {
                compressed.resize(max_decompressed_len as usize, 0);
            }
            Ok(CachedBlock(compressed.into()))
        }
    }
}
