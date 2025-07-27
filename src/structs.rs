//! This file defines data structures used by the squashfs file format

use bitflags::bitflags;
use zerocopy::{
    FromBytes, Immutable, KnownLayout, TryFromBytes,
    little_endian::{I16, U16, U32, U64},
};

/// The superblock is the first section of a SquashFS archive.
/// It is always 96 bytes in size and contains important information about the archive, including the locations of other sections.
#[repr(C)]
#[derive(Debug, KnownLayout, FromBytes)]
pub(crate) struct Superblock {
    /// Magic number, must be equal to "hsqs"
    pub(crate) magic: [u8; 4],
    /// The number of inodes stored in the archive.
    pub(crate) inode_count: U32,
    /// Last modification time of the archive. Count seconds since 00:00, Jan 1st 1970 UTC
    pub(crate) mod_time: U32,
    /// The size of a data block in bytes. Must be a power of two between 4096 (4k) and 1048576 (1 MiB)
    pub(crate) block_size: U32,
    /// The number of entries in the fragment table.
    pub(crate) frag_count: U32,
    /// See [CompressorType]
    pub(crate) compressor: U16,
    /// The log2 of the block size. If the two fields do not agree, the archive is considered corrupted.
    pub(crate) block_log: U16,
    /// See [SuperblockFlags]
    pub(crate) flags: U16,
    /// The number of entries in the ID lookup table.
    pub(crate) id_count: U16,
    /// Major version of the format. Must be set to 4.
    pub(crate) version_major: U16,
    /// Minor version of the format. Must be set to 0.
    pub(crate) version_minor: U16,
    /// A reference to the inode of the root directory.
    pub(crate) root_inode: U64,
    /// The number of bytes used by the archive. Because SquashFS archives must be padded to a multiple of the underlying device block size, this can be less than the actual file size.
    pub(crate) bytes_used: U64,
    /// The byte offset at which the id table starts.
    pub(crate) id_table: U64,
    /// The byte offset at which the xattr id table starts.
    pub(crate) xattr_table: U64,
    /// The byte offset at which the inode table starts.
    pub(crate) inode_table: U64,
    /// The byte offset at which the directory table starts.
    pub(crate) dir_table: U64,
    /// The byte offset at which the fragment table starts.
    pub(crate) frag_table: U64,
    /// The byte offset at which the export table starts.
    pub(crate) export_table: U64,
}

const _: () = {
    assert!(size_of::<Superblock>() == 96);
};

/// An ID designating the compressor used for both data and meta data blocks
#[repr(u16)]
#[derive(Debug, Clone, Copy, KnownLayout, TryFromBytes)]
#[allow(unused)] // TODO: the linter does not see TryFromBytes
pub(crate) enum CompressorType {
    Gzip = 1,
    Lzma,
    Lzo,
    Xz,
    Lz4,
    Zstd,
}

/// Flags containing properties of the squashfs system.
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct SuperblockFlags(u16);

bitflags! {
    impl SuperblockFlags : u16 {
        /// Inodes are stored uncompressed.
        const INODES_UNCOMPRESSED = 0x0001;
        /// Data blocks are stored uncompressed.
        const DATA_BLOCKS_UNCOMPRESSED = 0x0002;
        /// Fragments are stored uncompressed.
        const FRAGMENTS_UNCOMPRESSED = 0x0008;
        /// Fragments are not used.
        const FRAGMENTS_UNUSED = 0x0010;
        /// Fragments are always generated.
        const FRAGMENTS_ALWAYS = 0x0020;
        /// Data has been deduplicated.
        const DATA_DEDUPLICATED = 0x0040;
        /// NFS export table exists.
        const NFS_EXPORT_EXISTS = 0x0080;
        /// Xattrs are stored uncompressed.
        const XATTRS_UNCOMPRESSED = 0x0100;
        /// There are no Xattrs in the archive.
        const XATTRS_MISSING = 0x0200;
        ///  Compressor options are present.
        const COMPRESSOR_OPTIONS = 0x0400;
        /// The ID table is uncompressed.
        const ID_TABLE_UNCOMPRESSED = 0x0800;
    }
}

#[repr(C)]
#[derive(FromBytes)]
pub(crate) struct MetadataHeader {
    pub(crate) ty: U16,
    pub(crate) permissions: U16,
    pub(crate) uid: U16,
    pub(crate) gid: U16,
    pub(crate) mtime: U32,
    pub(crate) inode_number: U32,
}

#[repr(u16)]
#[derive(TryFromBytes, Debug)]
#[allow(unused)] // TODO: the linter does not see TryFromBytes
pub(crate) enum MetadataType {
    BasicDirectory = 1,
    BasicFile,
    BasicSymlink,
    BasicBlockDevice,
    BasicCharacterDevice,
    BasicNamedPipe,
    BasicSocket,
    ExtendedDirectory,
    ExtendedFile,
    ExtendedSymlink,
    ExtendedBlockDevice,
    ExtendedCharacterDevice,
    ExtendedNamedPipe,
    ExtendedSocket,
}

#[repr(C)]
#[derive(FromBytes, Debug)]
pub(crate) struct BasicDirectory {
    pub(crate) block_index: U32,
    pub(crate) link_count: U32,
    pub(crate) file_size: U16,
    pub(crate) block_offset: U16,
    pub(crate) parent_inode: U32,
}

#[repr(C)]
#[derive(FromBytes, Debug)]
pub(crate) struct ExtendedDirectory {
    pub(crate) link_count: U32,
    pub(crate) file_size: U32,
    pub(crate) block_index: U32,
    pub(crate) parent_inode: U32,
    pub(crate) index_count: U16,
    pub(crate) block_offset: U16,
    pub(crate) xattr_index: U32,
}

#[repr(C)]
#[derive(FromBytes, Debug)]
pub(crate) struct DirectoryIndex {
    /// This stores a byte offset from the first directory header to the current header,
    /// as if the uncompressed directory metadata blocks were laid out in memory consecutively.
    pub(crate) index: U32,
    /// Start offset of a directory table metadata block, relative to the directory table start.
    pub(crate) start: U32,
    /// One less than the size of the entry name.
    pub(crate) name_size: U32,
}

#[repr(C)]
#[derive(FromBytes)]
pub(crate) struct DirectoryHeader {
    pub(crate) count: U32,
    pub(crate) start: U32,
    pub(crate) inode_number: U32,
}

#[repr(C)]
#[derive(FromBytes)]
pub(crate) struct DirectoryEntry {
    pub(crate) offset: U16,
    pub(crate) inode_offset: I16,
    pub(crate) ty: U16,
    pub(crate) name_size: U16,
}

#[repr(C)]
#[derive(FromBytes, Debug)]
pub(crate) struct BasicFile {
    pub(crate) block_index: U32,
    pub(crate) frag_index: U32,
    pub(crate) frag_offset: U32,
    pub(crate) file_size: U32,
}

#[repr(C)]
#[derive(FromBytes, Debug)]
pub(crate) struct ExtendedFile {
    pub(crate) block_index: U64,
    pub(crate) file_size: U64,
    pub(crate) sparse: U64,
    pub(crate) link_count: U32,
    pub(crate) frag_index: U32,
    pub(crate) frag_offset: U32,
    pub(crate) xattr_index: U32,
}

#[repr(C)]
#[derive(FromBytes, Debug)]
pub(crate) struct Symlink {
    pub(crate) link_count: U32,
    pub(crate) target_size: U32,
}

#[repr(C)]
#[derive(FromBytes, KnownLayout, Immutable, Clone, Copy)]
pub(crate) struct FragmentEntry {
    /// The offset within the archive where the fragment block starts
    pub(crate) start: U64,
    /// The on-disk size of the fragment block. If the block is uncompressed, bit 24 (i.e. 1 << 24) is set.
    pub(crate) size: U32,
    pub(crate) _unused: U32,
}
