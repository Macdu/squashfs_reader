use std::{
    cell::Cell,
    io::{self, Read, Result},
    ops::Deref,
    sync::Arc,
    time::{Duration, SystemTime},
};

use zerocopy::{FromBytes, IntoBytes, TryFromBytes};

use crate::{
    FileSystem, FileSystemInner, FileType, MetadataReader, MetadataRef,
    readers::{FileBlockInfo, FileReader, FragmentInfo, METADATA_BLOCK_SIZE, ReadAt},
    structs::{self, DirectoryEntry, DirectoryHeader},
};

pub struct Metadata {
    permissions: u16,
    /// The unsigned number of seconds (not counting leap seconds) since 00:00, Jan 1st, 1970 UTC when the item described by the inode was last modified.
    mtime: u32,
    /// Unique node number. Must be at least 1 and at most the inode count from the super block.
    inode_number: u32,
    /// UID offset in the index table
    uid_idx: u16,
    /// GID offset in the index table
    gid_idx: u16,
    /// Pointer to this location in the squashfs file
    pub(crate) meta_ref: MetadataRef,
    pub(crate) ty: MetadataType,
}

impl Metadata {
    /// Returns the file type for this metadata.
    pub fn file_type(&self) -> FileType {
        match self.ty {
            MetadataType::File(_) => FileType::File,
            MetadataType::Directory(_) => FileType::Directory,
            MetadataType::Symlink(_) => FileType::Symlink,
        }
    }

    /// Returns true if this metadata is for a directory
    pub fn is_dir(&self) -> bool {
        matches!(self.ty, MetadataType::Directory(_))
    }

    /// Returns true if this metadata is for a regular file
    pub fn is_file(&self) -> bool {
        matches!(self.ty, MetadataType::File(_))
    }

    /// Returns true if this metadata is for a symlink
    pub fn is_symlink(&self) -> bool {
        matches!(self.ty, MetadataType::Symlink(_))
    }

    /// For a symlink, returns the target name
    pub fn target(&self) -> Option<&str> {
        match &self.ty {
            MetadataType::Symlink(sym) => Some(&sym.target),
            _ => None,
        }
    }

    /// Returns the size of the file, in bytes, this metadata is for.
    /// If a symlink, returns 0
    pub fn len(&self) -> u64 {
        match &self.ty {
            MetadataType::File(file) => file.file_size,
            MetadataType::Directory(dir) => dir.file_size.into(),
            MetadataType::Symlink(_) => 0,
        }
    }

    /// Unix file system permissions for the inode
    pub fn permissions(&self) -> u16 {
        self.permissions
    }

    pub fn modified(&self) -> SystemTime {
        // Cannot overflow
        SystemTime::UNIX_EPOCH
            .checked_add(Duration::from_secs(self.mtime.into()))
            .unwrap()
    }

    /// Unique node number. Must be at least 1 and at most the inode count from the super block.
    pub fn inode_number(&self) -> u32 {
        self.inode_number
    }

    /// Return the entry UID
    pub fn uid(&self, fs: &FileSystem<impl ReadAt>) -> Result<u32> {
        fs.0.id_lookup
            .lookup(self.uid_idx as u32, &fs.0)
            .map(|res| res.into())
    }

    /// Return the entry GID
    pub fn gid(&self, fs: &FileSystem<impl ReadAt>) -> Result<u32> {
        fs.0.id_lookup
            .lookup(self.gid_idx as u32, &fs.0)
            .map(|res| res.into())
    }

    /// Return a file reader for the file associated with this metadata
    pub fn read_file<T: ReadAt>(&self, fs: &FileSystem<T>) -> Result<FileReader<T>> {
        let MetadataType::File(file) = &self.ty else {
            return Err(std::io::ErrorKind::IsADirectory.into());
        };
        file.read_file(self.meta_ref, fs.0.clone())
    }

    /// Return a dir reader for the directory associated with this metadata
    pub fn read_dir<T: ReadAt>(&self, fs: &FileSystem<T>) -> Result<ReadDir<T>> {
        let MetadataType::Directory(dir) = &self.ty else {
            return Err(std::io::ErrorKind::NotADirectory.into());
        };
        Ok(dir.read_dir(fs.0.clone(), self.meta_ref))
    }
}

pub(crate) enum MetadataType {
    File(MetadataFile),
    Directory(MetadataDir),
    Symlink(MetadataSymlink),
}

impl From<&'_ MetadataType> for FileType {
    fn from(value: &'_ MetadataType) -> Self {
        match value {
            MetadataType::File(_) => FileType::File,
            MetadataType::Directory(_) => FileType::Directory,
            MetadataType::Symlink(_) => FileType::Symlink,
        }
    }
}

impl From<structs::MetadataType> for FileType {
    fn from(value: structs::MetadataType) -> Self {
        type MetaTy = structs::MetadataType;
        match value {
            MetaTy::BasicDirectory | MetaTy::ExtendedDirectory => FileType::Directory,
            MetaTy::BasicFile | MetaTy::ExtendedFile => FileType::File,
            _ => unimplemented!(),
        }
    }
}

pub(crate) struct MetadataFile {
    /// The offset from the start of the archive to the first data block.
    block_start: u64,
    /// The (uncompressed) size of this file.
    file_size: u64,

    // The number of hard links to this node (only available for an extended file inode)
    // link_count: Option<u32>,
    /// If using a fragment block, contains the index within the fragment table and the offset within the block
    frag_location: Option<(u32, u32)>,
    /// Size of this structure (including ty) on disk decompressed
    struct_size: u8,
}

pub(crate) struct MetadataDir {
    /// Location containing the file listing for this directory
    listing_location: MetadataRef,

    // The number of hard links to this directory.
    // link_count: u32,
    /// Total (uncompressed) size in bytes of the entry listing in the directory table, including headers.
    file_size: u32,

    // The inode number of the parent of this directory. If this is the root directory, this SHOULD be 0.
    // parent_inode: u32,
    /// The number of directory index entries following the inode structure.
    index_count: u16,
}

pub(crate) struct MetadataSymlink {
    /// The target path this symlink points to
    target: String,
}

impl Metadata {
    pub(crate) fn read<T, P>(
        mut reader: &mut MetadataReader<T, P>,
        meta_ref: MetadataRef,
    ) -> Result<Self>
    where
        T: ReadAt,
        P: Deref<Target = FileSystemInner<T>>,
    {
        let header = structs::MetadataHeader::read_from_io(&mut reader)?;
        let raw_ty = structs::MetadataType::try_read_from_bytes(header.ty.as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid inode"))?;

        let meta_ty = match raw_ty {
            structs::MetadataType::BasicDirectory => {
                let raw_dir = structs::BasicDirectory::read_from_io(&mut reader)?;
                // file_size value is 3 bytes larger than the real listing.
                // The Linux kernel creates "." and ".." entries for offsets 0 and 1,
                // and only after 3 looks into the listing, subtracting 3 from the size.
                let file_size = raw_dir.file_size.get().saturating_sub(3);

                let dir = MetadataDir {
                    listing_location: MetadataRef::from_block_and_offset(
                        raw_dir.block_index.get().into(),
                        raw_dir.block_offset.get(),
                    ),
                    // link_count: raw_dir.link_count.get(),
                    file_size: file_size.into(),
                    // parent_inode: raw_dir.parent_inode.get(),
                    index_count: 0,
                };
                MetadataType::Directory(dir)
            }
            structs::MetadataType::ExtendedDirectory => {
                let raw_dir = structs::ExtendedDirectory::read_from_io(&mut reader)?;
                // Same as for BasicDirectory
                let file_size = raw_dir.file_size.get().saturating_sub(3);

                let dir = MetadataDir {
                    listing_location: MetadataRef::from_block_and_offset(
                        raw_dir.block_index.get().into(),
                        raw_dir.block_offset.get(),
                    ),
                    // link_count: raw_dir.link_count.get(),
                    file_size: file_size,
                    // parent_inode: raw_dir.parent_inode.get(),
                    index_count: raw_dir.index_count.get(),
                };
                MetadataType::Directory(dir)
            }
            structs::MetadataType::BasicFile => {
                let raw_file = structs::BasicFile::read_from_io(&mut reader)?;

                let file = MetadataFile {
                    block_start: raw_file.block_index.get().into(),
                    file_size: raw_file.file_size.get().into(),
                    // link_count: None,
                    frag_location: (raw_file.frag_index.get() != !0)
                        .then(|| (raw_file.frag_index.get(), raw_file.frag_offset.get())),
                    struct_size: (size_of_val(&header) + size_of_val(&raw_file)) as u8,
                };
                MetadataType::File(file)
            }
            structs::MetadataType::ExtendedFile => {
                let raw_file = structs::ExtendedFile::read_from_io(&mut reader)?;

                let file = MetadataFile {
                    block_start: raw_file.block_index.get(),
                    file_size: raw_file.file_size.get(),
                    // link_count: Some(raw_file.link_count.get()),
                    frag_location: (raw_file.frag_index.get() != !0)
                        .then(|| (raw_file.frag_index.get(), raw_file.frag_offset.get())),
                    struct_size: (size_of_val(&header) + size_of_val(&raw_file)) as u8,
                };
                MetadataType::File(file)
            }
            structs::MetadataType::BasicSymlink | structs::MetadataType::ExtendedSymlink => {
                let raw_symlink = structs::Symlink::read_from_io(&mut reader)?;
                let mut target = vec![0u8; raw_symlink.target_size.get() as usize];

                reader.read_exact(&mut target)?;
                let target = String::from_utf8(target).map_err(|_| io::ErrorKind::InvalidData)?;
                let symlink = MetadataSymlink { target };
                MetadataType::Symlink(symlink)
            }
            _ => unimplemented!("Type {raw_ty:?} is not implemented"),
        };

        Ok(Self {
            permissions: header.permissions.get(),
            mtime: header.mtime.get(),
            inode_number: header.inode_number.get(),
            uid_idx: header.uid.get(),
            gid_idx: header.gid.get(),
            meta_ref,
            ty: meta_ty,
        })
    }
}

impl MetadataFile {
    pub(crate) fn read_file<T: ReadAt>(
        &self,
        meta_ref: MetadataRef,
        fs: Arc<FileSystemInner<T>>,
    ) -> Result<FileReader<T>> {
        let frag_info = if let Some((index, offset)) = self.frag_location {
            let entry = fs.fragment_lookup.lookup(index, &fs)?;
            Some(FragmentInfo::new(
                entry.start.get(),
                entry.size.get(),
                offset,
            ))
        } else {
            None
        };

        let block_size = fs.info.block_size as u64;
        let nb_blocks = if frag_info.is_some() {
            self.file_size / block_size
        } else {
            self.file_size.next_multiple_of(block_size) / block_size
        };

        let mut compressed_block_sizes = vec![0u32; nb_blocks as usize];
        let mut block_sizes_reader = MetadataReader::new(fs.deref(), meta_ref, fs.info.inode_table);
        block_sizes_reader.skip(self.struct_size as u64)?;
        block_sizes_reader.read_exact(compressed_block_sizes.as_mut_bytes())?;

        let mut block_starts = vec![FileBlockInfo::empty(); nb_blocks as usize + 1];
        block_starts[0] = FileBlockInfo::from_offset_and_compressed(self.block_start, false);

        let block_starts_iter = Cell::from_mut(block_starts.as_mut_slice())
            .as_slice_of_cells()
            .windows(2);
        for (compressed_block_size, cumu_window) in
            compressed_block_sizes.into_iter().zip(block_starts_iter)
        {
            let is_compressed = compressed_block_size & (1 << 24) == 0;
            let size = compressed_block_size & ((1 << 24) - 1);
            let new_data = FileBlockInfo::from_offset_and_compressed(
                cumu_window[0].get().offset() + size as u64,
                is_compressed,
            );
            cumu_window[1].set(new_data);
        }

        Ok(FileReader::new(
            fs,
            meta_ref,
            frag_info,
            self.file_size,
            block_starts.into(),
        ))
    }
}

impl MetadataDir {
    pub(crate) fn read_dir<T, P>(&self, fs: P, inode_ref: MetadataRef) -> ReadDir<T, P>
    where
        T: ReadAt,
        P: Deref<Target = FileSystemInner<T>>,
    {
        let table_offset = fs.info.dir_table;
        ReadDir::new(
            MetadataReader::new(fs, self.listing_location, table_offset),
            inode_ref,
            self.file_size,
        )
    }

    /// This function returns a ReadDir instance that may not enumerate all entries contained in this directory
    /// However, if this directory contains entry, it will be enumerated by the ReadDir instance returned
    /// [Self::read_dir] is a valid implementation for it, but when an index table is available, we use it
    /// to get faster entry lookup
    pub(crate) fn read_dir_for_entry<T, P>(
        &self,
        mut reader: impl Read,
        fs: P,
        inode_ref: MetadataRef,
        entry: &[u8],
    ) -> Result<ReadDir<T, P>>
    where
        T: ReadAt,
        P: Deref<Target = FileSystemInner<T>>,
    {
        if self.index_count == 0 {
            // No index table, ReadDir will enumerate all entries
            return Ok(self.read_dir(fs, inode_ref));
        }

        // Uncompressed byte offset from the directory listing start
        let mut dir_offset = 0;
        // block offset from the beginning of the listing table
        // The block location of a listing can always fit in an u32
        let mut dir_block_location = self.listing_location.block_location() as u32;

        // Directory indices are in ASCIIbetical order
        // Find the first entry strictly greater than entry
        let mut greater_entry = (0..self.index_count)
            .map(|_| -> Result<(structs::DirectoryIndex, Vec<u8>)> {
                let dir_idx = structs::DirectoryIndex::read_from_io(&mut reader)?;
                let mut entry_name = vec![0u8; dir_idx.name_size.get() as usize + 1];
                reader.read_exact(&mut entry_name)?;
                Ok((dir_idx, entry_name))
            })
            .skip_while(|idx_result| {
                if let Ok((dir_idx, name)) = idx_result
                    && name.as_slice() <= entry
                {
                    dir_offset = dir_idx.index.get();
                    dir_block_location = dir_idx.start.get();
                    true
                } else {
                    false
                }
            });

        let dir_upper_bound = greater_entry
            .next()
            .transpose()?
            .map(|(dir_idx, _)| dir_idx.index.get())
            .unwrap_or(self.file_size);

        let block_offset = (self.listing_location.offset_within_block() + dir_offset as u16)
            % METADATA_BLOCK_SIZE as u16;
        let start_metadata =
            MetadataRef::from_block_and_offset(dir_block_location as u64, block_offset);

        let table_offset = fs.info.dir_table;
        Ok(ReadDir::new(
            MetadataReader::new(fs, start_metadata, table_offset),
            inode_ref,
            dir_upper_bound - dir_offset,
        ))
    }
}

/// Structure to iterate over files in a directory
pub struct ReadDir<T, P = Arc<FileSystemInner<T>>>
where
    T: ReadAt,
    P: Deref<Target = FileSystemInner<T>>,
{
    reader: MetadataReader<T, P>,
    /// Reference to the dir inode
    inode_ref: MetadataRef,
    /// Uncompressed size left to read, if 0, means we read everything
    size_left: u32,
    /// Number of entries left to read for the current header
    /// If 0, a new header must be read
    header_count_left: u32,
    /// The location of the metadata block in the inode table where the inodes are stored.
    /// This is relative to the inode table start from the super block.
    inode_block_offset: u32,
    /// An arbitrary inode number. The entries that follow store their inode number as a difference to this.
    inode_base_value: u32,
}

impl<T, P> ReadDir<T, P>
where
    T: ReadAt,
    P: Deref<Target = FileSystemInner<T>>,
{
    pub(crate) fn new(reader: MetadataReader<T, P>, inode_ref: MetadataRef, size: u32) -> Self {
        Self {
            reader,
            inode_ref,
            size_left: size,
            header_count_left: 0,
            // We can set the next 2 fields to 0, they will be updated next read
            // Because header_count_left is 0
            inode_block_offset: 0,
            inode_base_value: 0,
        }
    }

    pub fn metadata(&self) -> Result<Metadata> {
        self.reader.fs.metadata_from_ref(self.inode_ref)
    }

    fn ensure_correct_header(&mut self) -> Result<()> {
        if self.header_count_left > 0 {
            return Ok(());
        }

        let header = DirectoryHeader::read_from_io(&mut self.reader)?;
        // The header count is off by 1 (a header cannot contain 0 entries)
        self.header_count_left = header.count.get() + 1;
        self.inode_block_offset = header.start.get();
        self.inode_base_value = header.inode_number.get();

        self.size_left = self
            .size_left
            .checked_sub(size_of_val(&header) as u32)
            .ok_or(io::ErrorKind::InvalidData)?;
        Ok(())
    }

    fn read_next(&mut self) -> Result<DirEntry> {
        // We should only call this function from Iterator::next which already checks for it
        assert!(self.size_left > 0);

        self.ensure_correct_header()?;
        let raw_entry = DirectoryEntry::read_from_io(&mut self.reader)?;

        // name_size is off by 1 and the string is not null terminated
        let mut name = vec![0u8; raw_entry.name_size.get() as usize + 1];
        self.reader.read_exact(&mut name)?;

        self.size_left = self
            .size_left
            .checked_sub((size_of_val(&raw_entry) + name.len()) as u32)
            .ok_or(io::ErrorKind::InvalidData)?;
        self.header_count_left -= 1;

        let name = String::from_utf8(name)
            // Note: should this be considered as unsupported or invalid data?
            .map_err(|_| io::Error::new(io::ErrorKind::Unsupported, "Using non-UTF8 entry name"))?;

        let ty = structs::MetadataType::try_read_from_bytes(raw_entry.ty.as_bytes())
            .map_err(|_| io::ErrorKind::InvalidData)?;

        let inode_ref = MetadataRef::from_block_and_offset(
            self.inode_block_offset.into(),
            raw_entry.offset.get(),
        );

        Ok(DirEntry {
            ty: ty.into(),
            inode_number: self
                .inode_base_value
                .wrapping_add_signed(raw_entry.inode_offset.get().into()),
            inode_ref,
            name,
        })
    }
}

impl<T, P> Iterator for ReadDir<T, P>
where
    T: ReadAt,
    P: Deref<Target = FileSystemInner<T>>,
{
    type Item = Result<DirEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.size_left == 0 {
            // We read all the entries
            return None;
        }

        Some(self.read_next())
    }
}

/// Represent an entry in a folder
pub struct DirEntry {
    /// basic type of this entry
    ty: FileType,
    /// Inode number of the entry
    inode_number: u32,
    /// Reference to read the entry
    pub(crate) inode_ref: MetadataRef,
    /// Name of the file
    name: String,
}

impl DirEntry {
    /// The entry name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Inode number of the entry
    pub fn inode_number(&self) -> u32 {
        self.inode_number
    }

    /// Inode reference of the entry
    pub fn inode_ref(&self) -> MetadataRef {
        self.inode_ref
    }

    /// Entry type
    pub fn ty(&self) -> FileType {
        self.ty
    }

    /// Read the metadata for this entry
    pub fn metadata(&self, fs: &FileSystem<impl ReadAt>) -> Result<Metadata> {
        fs.metadata_from_ref(self.inode_ref)
    }
}
