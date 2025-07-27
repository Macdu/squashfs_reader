use std::{
    fs,
    io::{self, Read, Seek},
    marker::PhantomData,
    ops::Deref,
    sync::{Arc, Mutex},
};

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, little_endian::U16};

use crate::{CachedBlock, FileSystemInner, Metadata, MetadataRef, Result};

/// Trait providing the function [ReadAt::read_at]
/// This function is thread safe and has the same behavior as seeking and reading at a given location
/// under a mutex. To improve compatibility with different operating systems, the guarantees
/// provided by this function are minimal
pub trait ReadAt {
    /// Similar to [io::Read::read] but instead of using the internal cursor,
    /// Use `offset` as the explicit cursor
    /// To maximise compatibility with different operating systems, the state of
    /// the internal cursor is considered as undefined after a call to this function
    /// The purpose of this library is to implement a squashfs reader, not to expose
    /// some io traits, therefore hide these definitions so as not to clutter up
    /// autocompletion
    #[doc(hidden)]
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize>;

    /// Similar to [io::Read::read_exact], but with the same properties as [ReadAt::read_at]
    #[doc(hidden)]
    fn read_exact_at(&self, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
        while !buf.is_empty() {
            match self.read_at(buf, offset) {
                Ok(0) => break,
                Ok(n) => {
                    buf = &mut buf[n..];
                    offset += n as u64;
                }
                Err(ref e) if matches!(e.kind(), io::ErrorKind::Interrupted) => {}
                Err(e) => return Err(e),
            }
        }

        if !buf.is_empty() {
            Err(io::ErrorKind::UnexpectedEof.into())
        } else {
            Ok(())
        }
    }
}

#[cfg(unix)]
impl ReadAt for &fs::File {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        std::os::unix::fs::FileExt::read_at(*self, buf, offset)
    }

    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        std::os::unix::fs::FileExt::read_exact_at(*self, buf, offset)
    }
}

#[cfg(windows)]
impl ReadAt for &fs::File {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        // [ReadAt::read_at] has no guarantee regarding the value of the cursor
        // at after a call to this function. Therefore, it is valid to implement
        // read_at using seek_read, which modifies the position of the internal cursor
        std::os::windows::fs::FileExt::seek_read(*self, buf, offset)
    }
}

#[cfg(any(unix, windows))]
impl<'a> ReadAt for fs::File {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        (&self).read_at(buf, offset)
    }

    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        (&self).read_exact_at(buf, offset)
    }
}

/// Thread-safe reader which implements [ReadAt] using
/// a mutex with a [Read] and [Seek] type internally
pub struct SharedReader<T: Read + Seek> {
    inner: Mutex<T>,
}

impl<T: Read + Seek> SharedReader<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner: Mutex::new(inner),
        }
    }
}

impl<T: Read + Seek> ReadAt for SharedReader<T> {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        let mut inner = self.inner.lock().unwrap();
        inner.seek(io::SeekFrom::Start(offset))?;
        inner.read(buf)
    }

    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        let mut inner = self.inner.lock().unwrap();
        inner.seek(io::SeekFrom::Start(offset))?;
        inner.read_exact(buf)
    }
}

/// Metadata blocks have a fixed size of 8KiB
pub(crate) const METADATA_BLOCK_SIZE: u32 = 8192;

/// Read a metadata block at offset `offset` and return compressed read size
/// as well as the decompressed block
fn read_metadata_block(
    fs: &FileSystemInner<impl ReadAt>,
    offset: u64,
) -> Result<(usize, CachedBlock)> {
    // Every metadata block starts with a 16-bit value indicating the compressed size of
    // the upcoming block
    // If the MSB is set, it means the upcoming block is uncompressed
    let mut block_info: U16 = U16::ZERO;
    fs.reader.read_exact_at(block_info.as_mut_bytes(), offset)?;

    let block_size = block_info.get() & 0x7FFF;
    if block_size > METADATA_BLOCK_SIZE as u16 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid metadata block size",
        ));
    }

    let is_block_uncompressed = (block_info.get() & 0x8000) != 0;

    let block = fs.cached_read(
        offset + 2,
        true,
        false, // Metada block are not sparse
        block_size as u32,
        METADATA_BLOCK_SIZE,
        !is_block_uncompressed,
    )?;
    Ok((2 + block_size as usize, block))
}

/// Reader used to read Metadata blocks, it only implements [Read] (no seek or read_at)
pub(super) struct MetadataReader<T, P = Arc<FileSystemInner<T>>>
where
    T: ReadAt,
    P: Deref<Target = FileSystemInner<T>>,
{
    /// Pointer to the file system
    pub(crate) fs: P,
    /// Current block being read
    curr_block: Option<CachedBlock>,
    /// Offset within [Self::curr_block]
    curr_offset: u16,
    /// Offset relative to the start of the squashfs of the next block to read
    next_block_offset: u64,
    _phantom: PhantomData<T>,
}

impl<T, P> MetadataReader<T, P>
where
    T: ReadAt,
    P: Deref<Target = FileSystemInner<T>>,
{
    pub fn new(fs: P, location: MetadataRef, offset: u64) -> Self {
        Self {
            fs,
            curr_block: None,
            curr_offset: location.offset_within_block(),
            next_block_offset: offset + location.block_location(),
            _phantom: PhantomData,
        }
    }

    fn ensure_block_available(&mut self) -> Result<()> {
        if self.curr_block.is_some() {
            return Ok(());
        }

        let (read_len, block) = read_metadata_block(&self.fs, self.next_block_offset)?;

        // We can update the next block location with the 16-bit value and the compressed block size
        self.next_block_offset += read_len as u64;
        self.curr_block = Some(block);
        Ok(())
    }

    pub fn skip(&mut self, mut len: u64) -> Result<()> {
        while len > 0 {
            self.ensure_block_available()?;
            let block = self.curr_block.as_ref().unwrap();

            let available = block.len() - self.curr_offset as usize;
            let to_skip = (available as u64).min(len);

            len -= to_skip;
            self.curr_offset += to_skip as u16;

            if self.curr_offset as usize == block.len() {
                // The next read/skip will read the next block
                self.curr_block.take();
                self.curr_offset = 0;
            }
        }
        Ok(())
    }
}

impl<T, P> Read for MetadataReader<T, P>
where
    T: ReadAt,
    P: Deref<Target = FileSystemInner<T>>,
{
    fn read(&mut self, mut buf: &mut [u8]) -> io::Result<usize> {
        // When reading metadata block, we keep going until we reach the end of the archive
        // So if there is no error, we will always read buf.len()
        let read_len = buf.len();

        while !buf.is_empty() {
            self.ensure_block_available()?;
            let block = self.curr_block.as_ref().unwrap();
            let data_available = &block.deref()[self.curr_offset as usize..];

            let copy_len = buf.len().min(data_available.len());
            let (to_copy, leftover) = buf.split_at_mut(copy_len);
            to_copy.copy_from_slice(&data_available[..copy_len]);

            buf = leftover;
            self.curr_offset += copy_len as u16;

            // If we reach the end of our block
            if self.curr_offset as usize == block.len() {
                // The next read will read the next block
                self.curr_block.take();
                self.curr_offset = 0;
            }
        }

        Ok(read_len)
    }
}

pub(crate) struct BlockLookupTable<T> {
    nb_items: u32,
    blocks_offset: Box<[u64]>,
    _marker: PhantomData<T>,
}

impl<T> BlockLookupTable<T> {
    const fn nb_items_per_block() -> u32 {
        const {
            assert!(size_of::<T>().is_power_of_two());
            assert!(size_of::<T>() <= METADATA_BLOCK_SIZE as usize);
        };
        METADATA_BLOCK_SIZE / size_of::<T>() as u32
    }

    pub fn new(reader: &impl ReadAt, offset: u64, nb_items: u32) -> Result<Self> {
        let nb_items_per_block = Self::nb_items_per_block();
        let nb_blocks = nb_items.next_multiple_of(nb_items_per_block) / nb_items_per_block;
        let mut blocks_offset = vec![0u64; nb_blocks as usize];
        reader.read_exact_at(blocks_offset.as_mut_bytes(), offset)?;

        // Convert LE to native endianness
        for offset in &mut blocks_offset {
            *offset = u64::from_le(*offset);
        }

        Ok(Self {
            nb_items,
            blocks_offset: blocks_offset.into(),
            _marker: PhantomData,
        })
    }
}

impl<T: Clone + FromBytes + Immutable + KnownLayout> BlockLookupTable<T> {
    pub fn lookup(&self, entry: u32, fs: &FileSystemInner<impl ReadAt>) -> Result<T> {
        if entry >= self.nb_items {
            return Err(io::ErrorKind::NotFound.into());
        }

        let containing_block = entry / Self::nb_items_per_block();
        let index_within_block = entry % Self::nb_items_per_block();

        // This should never panic as we checked above that entry < self.nb_items
        let block_offset = self.blocks_offset[containing_block as usize];

        let (_, block) = read_metadata_block(fs, block_offset)?;
        let items = <[T]>::ref_from_bytes(&block).map_err(|_| io::ErrorKind::InvalidData)?;
        let item = items
            .get(index_within_block as usize)
            .ok_or(io::ErrorKind::InvalidData)?;

        Ok(item.clone())
    }
}

pub(crate) struct FragmentInfo {
    /// Start of the fragment block
    start: u64,
    /// Size of the fragment block, if bit 24 is set, it means the block is not compressed
    size_and_compressed: u32,
    /// Offset of the file within the fragment block
    offset: u32,
}

impl FragmentInfo {
    pub fn size(&self) -> u32 {
        self.size_and_compressed & ((1 << 24) - 1)
    }

    pub fn is_compressed(&self) -> bool {
        self.size_and_compressed & (1 << 24) == 0
    }
}

impl FragmentInfo {
    pub fn new(start: u64, size_and_compressed: u32, offset: u32) -> Self {
        Self {
            start,
            size_and_compressed,
            offset,
        }
    }
}

/// Data for a file block, the first 63 bits give the start of the block relative to the start of the squashfs file
/// The MSB is set if the following block (not the block at the given offset but the one after) is compressed
#[derive(Clone, Copy)]
pub(crate) struct FileBlockInfo(u64);

impl FileBlockInfo {
    const COMPRESSED_BIT: u64 = 1 << 63;

    pub const fn empty() -> Self {
        Self(0)
    }

    pub const fn from_offset_and_compressed(mut offset: u64, compressed: bool) -> Self {
        debug_assert!(offset < (1 << 63));
        if compressed {
            offset |= Self::COMPRESSED_BIT;
        }
        Self(offset)
    }

    pub const fn offset(self) -> u64 {
        self.0 & !Self::COMPRESSED_BIT
    }

    pub const fn is_compressed(self) -> bool {
        (self.0 & Self::COMPRESSED_BIT) != 0
    }
}

/// Reader for a file in the squashfs filesystem
pub struct FileReader<T: ReadAt> {
    /// Pointer to the file system
    fs: Arc<FileSystemInner<T>>,
    /// Reference to the inode
    inode_ref: MetadataRef,
    /// Current block being read
    curr_block: Option<CachedBlock>,
    /// Is the current block being read a fragment
    block_is_fragment: bool,
    fragment_info: Option<FragmentInfo>,
    /// Current position within the file
    position: u64,
    /// Decompressed size of the file
    file_size: u64,
    /// Array of the start point of each block
    /// It also includes the endpoint of the last block, making
    /// it really easy to compute a given block size
    /// Therefore, it has nb_blocks + 1 entries
    block_starts: Box<[FileBlockInfo]>,
}

impl<T: ReadAt> FileReader<T> {
    pub(crate) fn new(
        fs: Arc<FileSystemInner<T>>,
        inode_ref: MetadataRef,
        frag_info: Option<FragmentInfo>,
        file_size: u64,
        block_starts: Box<[FileBlockInfo]>,
    ) -> Self {
        Self {
            fs,
            inode_ref,
            curr_block: None,
            block_is_fragment: false,
            fragment_info: frag_info,
            position: 0,
            file_size,
            block_starts,
        }
    }

    pub fn metadata(&self) -> Result<Metadata> {
        self.fs.metadata_from_ref(self.inode_ref)
    }

    pub fn read_at(&self, buf: &mut [u8], offset: u64) -> Result<usize> {
        ReadAt::read_at(self, buf, offset)
    }

    /// Return the block containing the given offset as well as if the block is a fragment
    fn get_block_for_offset(&self, offset: u64) -> Result<(CachedBlock, bool)> {
        if offset >= self.file_size {
            return Err(io::ErrorKind::UnexpectedEof)?;
        }

        let block_index = offset / self.fs.info.block_size as u64;
        let offset_aligned = block_index * self.fs.info.block_size as u64;
        let (
            block_offset,
            compressed_size,
            is_compressed,
            is_fragment,
            max_decompressed_size,
            fill_block,
        ) = if block_index == self.block_starts.len() as u64 - 1 {
            // This is a fragment
            let frag_info = self.fragment_info.as_ref().unwrap();
            (
                frag_info.start,
                frag_info.size(),
                frag_info.is_compressed(),
                true,
                self.fs.info.block_size,
                false, // Fragment block can have a size smaller than the block size
            )
        } else {
            let block_starts = &self.block_starts[block_index as usize..];
            let compressed_size = block_starts[1].offset() - block_starts[0].offset();
            // It is the following block (block_starts[1]) which tells if the block at block_index is compressed
            let is_compressed = block_starts[1].is_compressed();
            // All block may be filled (sparse)
            let decompressed_size =
                (self.file_size - offset_aligned).min(self.fs.info.block_size as u64) as u32;
            (
                block_starts[0].offset(),
                compressed_size as u32,
                is_compressed,
                false,
                decompressed_size,
                true,
            )
        };

        let block = self.fs.cached_read(
            block_offset,
            false,
            fill_block,
            compressed_size,
            max_decompressed_size,
            is_compressed,
        )?;
        Ok((block, is_fragment))
    }

    fn ensure_block_available(&mut self) -> Result<()> {
        if self.curr_block.is_some() {
            return Ok(());
        }

        let (block, is_fragment) = self.get_block_for_offset(self.position)?;
        self.curr_block = Some(block);
        self.block_is_fragment = is_fragment;
        Ok(())
    }
}

impl<T: ReadAt> Read for FileReader<T> {
    fn read(&mut self, mut buf: &mut [u8]) -> io::Result<usize> {
        let initial_len = buf.len();
        while !buf.is_empty() && self.position < self.file_size {
            self.ensure_block_available()?;
            let block = self.curr_block.as_ref().unwrap();

            let block_size_available = (self.file_size - self.position).min(
                (self.position + 1).next_multiple_of(self.fs.info.block_size as u64)
                    - self.position,
            ) as u32;
            let copy_len = block_size_available.min(buf.len() as u32);

            let mut src_offset = (self.position % self.fs.info.block_size as u64) as u32;
            if self.block_is_fragment {
                src_offset += self.fragment_info.as_ref().unwrap().offset;
            }

            let src = block
                .get(src_offset as usize..src_offset as usize + copy_len as usize)
                .unwrap();
            let (dst, remaining) = buf.split_at_mut(copy_len as usize);
            buf = remaining;

            dst.copy_from_slice(src);
            self.position += copy_len as u64;

            if copy_len == block_size_available {
                self.curr_block.take();
            }
        }

        Ok(initial_len - buf.len())
    }
}

impl<T: ReadAt> Seek for FileReader<T> {
    fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
        let old_block_index = self.position / self.fs.info.block_size as u64;
        match pos {
            io::SeekFrom::Start(offset) => self.position = offset,
            io::SeekFrom::End(offset) => self.position = self.file_size.wrapping_add_signed(offset),
            io::SeekFrom::Current(offset) => {
                self.position = self.position.wrapping_add_signed(offset)
            }
        };

        let new_block_index = self.position / self.fs.info.block_size as u64;
        if old_block_index != new_block_index {
            self.curr_block.take();
        }

        Ok(self.position)
    }
}

impl<T: ReadAt> ReadAt for FileReader<T> {
    fn read_at(&self, mut buf: &mut [u8], mut offset: u64) -> io::Result<usize> {
        // TODO: This is really similar to the read implementation
        let initial_len = buf.len();
        while !buf.is_empty() && offset < self.file_size {
            let (block, is_fragment) = self.get_block_for_offset(offset)?;

            let block_size_available = (self.file_size - offset)
                .min((offset + 1).next_multiple_of(self.fs.info.block_size as u64) - offset)
                as u32;
            let copy_len = block_size_available.min(buf.len() as u32);

            let mut src_offset = (offset % self.fs.info.block_size as u64) as u32;
            if is_fragment {
                src_offset += self.fragment_info.as_ref().unwrap().offset;
            }

            let src = block
                .get(src_offset as usize..src_offset as usize + copy_len as usize)
                .ok_or(io::ErrorKind::InvalidData)?;
            let (dst, remaining) = buf.split_at_mut(copy_len as usize);
            buf = remaining;

            dst.copy_from_slice(src);
            offset += copy_len as u64;
        }

        Ok(initial_len - buf.len())
    }
}
