use std::io;

use crate::{CachedBlock, Compression, Result};

/// Decompress the data in `input` using the compression scheme specified in `compression`
/// This will output an error if the decompressed data is larger than `block_size`
/// `fill_block` is set, the output block will be padded with zeros to be exactly block_size in length
pub(crate) fn decompress_block(
    input: &[u8],
    block_size: u32,
    fill_block: bool,
    compression: Compression,
) -> Result<CachedBlock> {
    let mut output = vec![0u8; block_size as usize];

    // Call the function from the appropriate crate depending on the compression algorithm used
    // If both the best_performance and only_rust features are set, prioritize only_rust
    match compression {
        Compression::Gzip => {
            let mut decoder = flate2::Decompress::new(true);
            decoder
                .decompress(input, &mut output, flate2::FlushDecompress::Finish)
                .unwrap();

            if !fill_block {
                output.truncate(decoder.total_out() as usize);
            }
        }

        #[cfg(not(feature = "only_rust"))]
        Compression::Xz | Compression::Lzma => {
            let mut decoder = if matches!(compression, Compression::Xz) {
                liblzma::stream::Stream::new_stream_decoder(u64::MAX, 0)
            } else {
                liblzma::stream::Stream::new_lzma_decoder(u64::MAX)
            }
            .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?;
            decoder
                .process(input, &mut output, liblzma::stream::Action::Finish)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;

            if !fill_block {
                output.truncate(decoder.total_out() as usize);
            }
        }

        #[cfg(feature = "only_rust")]
        Compression::Xz | Compression::Lzma => {
            output.clear();
            let mut output_cursor = io::Cursor::new(output);
            let mut input = input;
            if matches!(compression, Compression::Xz) {
                lzma_rs::xz_decompress(&mut input, &mut output_cursor)
            } else {
                lzma_rs::lzma_decompress(&mut input, &mut output_cursor)
            }
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
            output = output_cursor.into_inner();

            if fill_block {
                output.resize(block_size as usize, 0);
            }
        }

        Compression::Lzo => {
            let total_out = crate::lzo::decompress(input, &mut output)
                .map_err(|_| io::ErrorKind::InvalidData)?;

            if !fill_block {
                output.truncate(total_out);
            }
        }

        Compression::Lz4 => {
            let total_out = lz4_flex::decompress_into(input, &mut output)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;

            if !fill_block {
                output.truncate(total_out);
            }
        }

        #[cfg(not(feature = "only_rust"))]
        Compression::Zstd => {
            let total_out = zstd_safe::decompress(&mut output, input)
                .map_err(|_| io::ErrorKind::InvalidData)?;

            if fill_block {
                output.resize(total_out as usize, 0);
            }
        }

        #[cfg(feature = "only_rust")]
        Compression::Zstd => {
            let mut decoder = ruzstd::decoding::FrameDecoder::new();
            let total_out = decoder
                .decode_all(input, &mut output)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;

            if !fill_block {
                output.truncate(total_out);
            }
        }
    }
    Ok(CachedBlock(output.into()))
}
