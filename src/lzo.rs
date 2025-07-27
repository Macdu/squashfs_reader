/// This MAX_255_COUNT is the maximum number of times we can add 255 to a base
/// count without overflowing an integer. The multiply will overflow when
/// multiplying 255 by more than MAXINT/255. The sum will overflow earlier
/// depending on the base count. Since the base count is taken from a u8
/// and a few bits, it is safe to assume that it will always be lower than
/// or equal to 2*255, thus we can always prevent any overflow by accepting
/// two less 255 steps. See Documentation/staging/lzo.rst for more information.
const MAX_255_COUNT: usize = (!0usize) / 255 - 2;
const MIN_ZERO_RUN_LENGTH: usize = 4;

#[derive(Clone, Copy)]
pub(crate) enum Error {
    /// The output buffer is too small
    OutputOverrun,
    /// The input buffer does not contain the full compressed data
    InputOverrun,
    /// The LZO version used is not supported
    UnsupportedVersion,
    /// Error while decompressing: tried to read data which was before
    /// the start of the output buffer
    LookbehindOverrun,
    /// Not all of the input was used while decompressing the data
    InputNotConsumed,
    /// Miscellaneous error while decompressing the data
    Error,
}

/// State while decompressing: number of literals which were copied in the last phase
#[derive(Clone, Copy)]
enum FullState {
    NoLit,
    Lit1,
    Lit2,
    Lit3,
    Lit4OrMore,
}

/// State while decompressing: number of literals which are going to be copied in this phase
/// Note that this state is not used when copying 4 or more literals
#[derive(Clone, Copy)]
enum State {
    NoLit,
    Lit1,
    Lit2,
    Lit3,
}

impl TryFrom<u8> for State {
    type Error = Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::NoLit),
            1 => Ok(Self::Lit1),
            2 => Ok(Self::Lit2),
            3 => Ok(Self::Lit3),
            _ => Err(Error::Error),
        }
    }
}

impl From<State> for FullState {
    fn from(value: State) -> Self {
        match value {
            State::NoLit => FullState::NoLit,
            State::Lit1 => FullState::Lit1,
            State::Lit2 => FullState::Lit2,
            State::Lit3 => FullState::Lit3,
        }
    }
}

/// Decompress LZO-compressed data contained in the input buffer and store the uncompressed data
/// in the output buffer. If decompression is successful, [Ok] is returned with the length of
/// the decompressed data (which is at most `output.len()`), otherwise an error is returned
pub(crate) fn decompress(input: &[u8], output: &mut [u8]) -> Result<usize, Error> {
    if input.len() < 3 {
        return Err(Error::InputOverrun);
    }

    let mut in_pos = 0usize;
    let mut out_pos = 0usize;

    // Previous amount of bytes copied
    let mut state = FullState::NoLit;
    // Amount of bytes to be copied in this loop
    let mut new_state: State;

    // The lookbehind copy will copy from output[out_pos-lb_offset..out_pos-lb_offset+lb_len]
    // to output[output..out_pos+lb_len], these two slices can overlap, in this case copy
    // byte by byte (no what memmove does)
    let mut lb_offset: usize;
    let mut lb_len: usize;

    // Read a single byte (with boundary check) and advance the input cursor
    let read_u8 = |in_pos: &mut usize| -> Result<u8, Error> {
        let val = *input.get(*in_pos).ok_or(Error::InputOverrun)?;
        *in_pos += 1;
        Ok(val)
    };

    // Read a low endian 16-bit value (with boundary check) and advance the input cursor
    let read_u16 = |in_pos: &mut usize| -> Result<u16, Error> {
        let val = input.get(*in_pos..*in_pos + 2).ok_or(Error::InputOverrun)?;
        *in_pos += 2;
        // 16-bit values are low endian
        let val = val[0] as u16 | (val[1] as u16) << 8;
        Ok(val)
    };

    // Read a value zero-length encoded
    // If we have k zero bytes and a non-zero byte m, the final value is 255*k + m
    // Everything is boundary/overflow checked
    let read_zero_byte_length = |in_pos: &mut usize| -> Result<usize, Error> {
        let old_in_pos = *in_pos;
        while matches!(input.get(*in_pos), Some(0)) {
            *in_pos += 1;
        }
        let zero_len = *in_pos - old_in_pos;
        if zero_len > MAX_255_COUNT {
            return Err(Error::Error);
        }

        let additional = read_u8(in_pos)?;
        Ok(zero_len * 255 + additional as usize)
    };

    // Copy len bytes from input to output, with boundary checks
    let copy_in_out = |in_pos: &mut usize,
                       output: &mut [u8],
                       out_pos: &mut usize,
                       len: usize|
     -> Result<(), Error> {
        let src = input
            .get(*in_pos..*in_pos + len)
            .ok_or(Error::InputOverrun)?;
        let dst = output
            .get_mut(*out_pos..*out_pos + len)
            .ok_or(Error::OutputOverrun)?;
        dst.copy_from_slice(src);

        *in_pos += len;
        *out_pos += len;
        Ok(())
    };

    // Does the bitstream support rle (version 1)
    let support_rle = if *input.get(0).ok_or(Error::Error)? == 17 && input.len() >= 5 {
        in_pos += 1;
        // bitstream version. If the first byte is 17, and compressed
        // stream length is at least 5 bytes (length of shortest possible
        // versioned bitstream), the next byte gives the bitstream version
        // (version 1 only).
        // Otherwise, the bitstream version is 0
        let bitstream_version = read_u8(&mut in_pos)?;
        match bitstream_version {
            0 => false,
            1 => true,
            _ => return Err(Error::UnsupportedVersion),
        }
    } else {
        false
    };

    // First byte encoding
    match *input.get(in_pos).ok_or(Error::Error)? {
        inst @ 21.. => {
            // 21..255 : copy literal string
            //           length = (byte - 17) = 4..238
            //           state = 4 [ don't copy extra literals ]
            //           skip byte
            in_pos += 1;
            state = FullState::Lit4OrMore;

            copy_in_out(&mut in_pos, output, &mut out_pos, inst as usize - 17)?;
        }
        inst @ 18..21 => {
            // 18..21 : copy 1..3 literals
            //           state = (byte - 17) = 1..3  [ copy <state> literals ]
            //           skip byte
            new_state = State::try_from(inst - 17)?;
            state = new_state.into();
            in_pos += 1;

            copy_in_out(&mut in_pos, output, &mut out_pos, inst as usize - 17)?;
        }
        _ => {
            // 0..17 : follow regular instruction encoding, see below. It is worth
            // noting that codes 16 and 17 will represent a block copy from
            // the dictionary which is empty, and that they will always be
            // invalid at this place.
        }
    };

    loop {
        let inst = read_u8(&mut in_pos)?;

        if (inst & 0xC0) != 0 {
            // [M2]
            // 1 L L D D D S S  (128..255)
            //   Copy 5-8 bytes from block within 2kB distance
            //   state = S (copy S literals after this block)
            //   length = 5 + L
            // Always followed by exactly one byte : H H H H H H H H
            //   distance = (H << 3) + D + 1
            //
            // 0 1 L D D D S S  (64..127)
            //   Copy 3-4 bytes from block within 2kB distance
            //   state = S (copy S literals after this block)
            //   length = 3 + L
            // Always followed by exactly one byte : H H H H H H H H
            //   distance = (H << 3) + D + 1
            //
            let upper_offset = read_u8(&mut in_pos)? as usize;

            lb_offset = (upper_offset << 3) + ((inst >> 2) & 0x7) as usize + 1;
            lb_len = (inst >> 5) as usize + 1;
            new_state = State::try_from(inst & 0x3)?;
        } else if (inst & 0x20) != 0 {
            // [M3]
            // 0 0 1 L L L L L  (32..63)
            //   Copy of small block within 16kB distance (preferably less than 34B)
            //   length = 2 + (L ?: 31 + (zero_bytes * 255) + non_zero_byte)
            // Always followed by exactly one LE16 :  D D D D D D D D : D D D D D D S S
            //   distance = D + 1
            //   state = S (copy S literals after this block)
            //
            lb_len = 2 + match (inst & 0x1F) as usize {
                0 => 0x1F + read_zero_byte_length(&mut in_pos)?,
                non_zero => non_zero,
            };
            let ds = read_u16(&mut in_pos)?;
            lb_offset = (ds >> 2) as usize + 1;
            new_state = State::try_from(ds as u8 & 0x3)?;
        } else if (inst & 0x10) != 0 {
            // [M4]
            // 0 0 0 1 H L L L  (16..31)
            //   Copy of a block within 16..48kB distance (preferably less than 10B)
            //   length = 2 + (L ?: 7 + (zero_bytes * 255) + non_zero_byte)
            // Always followed by exactly one LE16 :  D D D D D D D D : D D D D D D S S
            //   distance = 16384 + (H << 14) + D
            //   state = S (copy S literals after this block)
            //   End of stream is reached if distance == 16384

            // In version 1 only, this instruction is also used to encode a run of
            // zeros if distance = 0xbfff, i.e. H = 1 and the D bits are all 1.
            // In this case, it is followed by a fourth byte, X.
            // run length = ((X << 3) | (0 0 0 0 0 L L L)) + 4
            let following = input.get(0..2).ok_or(Error::InputOverrun)?;
            let next = following[0] as u16 | (following[1] as u16) << 8;
            if support_rle && (inst & 0x8) == 0x8 && (next & 0xFFFC) == 0xFFFC {
                in_pos += 2;
                let run_length = MIN_ZERO_RUN_LENGTH
                    + ((inst & 0x7) as usize | (read_u8(&mut in_pos)? as usize) << 3);

                // We got a run of zeros, zero fill the appropriate amount in output
                let dst = output
                    .get_mut(out_pos..out_pos + run_length)
                    .ok_or(Error::OutputOverrun)?;
                dst.fill(0);
                out_pos += run_length;

                new_state = State::try_from(next as u8 & 0x3)?;
                // There is no lookbehind copy
                lb_offset = 0;
                lb_len = 0;
            } else {
                lb_len = 2 + match (inst & 0x7) as usize {
                    0 => 0x7 + read_zero_byte_length(&mut in_pos)?,
                    non_zero => non_zero,
                };

                let ds = read_u16(&mut in_pos)?;
                lb_offset = ((inst & 0x8) as usize) << 11 | (ds >> 2) as usize;
                new_state = State::try_from(ds as u8 & 0x3)?;

                if lb_offset == 0 {
                    // Stream finished
                    break;
                }
                lb_offset += 16384;
            }
        } else {
            // [M1] Depends on the number of literals copied by the last instruction.
            match state {
                FullState::NoLit => {
                    // If last instruction did not copy any literal (state == 0), this
                    // encoding will be a copy of 4 or more literal, and must be interpreted
                    // like this :
                    //
                    //    0 0 0 0 L L L L  (0..15)  : copy long literal string
                    //    length = 3 + (L ?: 15 + (zero_bytes * 255) + non_zero_byte)
                    //    state = 4  (no extra literals are copied)
                    //
                    let len = 3 + match inst as usize {
                        0 => 0xF + read_zero_byte_length(&mut in_pos)?,
                        non_zero => non_zero,
                    };

                    copy_in_out(&mut in_pos, output, &mut out_pos, len)?;
                    state = FullState::Lit4OrMore;
                    continue;
                }
                FullState::Lit1 | FullState::Lit2 | FullState::Lit3 => {
                    // If last instruction used to copy between 1 to 3 literals (encoded in
                    // the instruction's opcode or distance), the instruction is a copy of a
                    // 2-byte block from the dictionary within a 1kB distance. It is worth
                    // noting that this instruction provides little savings since it uses 2
                    // bytes to encode a copy of 2 other bytes but it encodes the number of
                    // following literals for free. It must be interpreted like this :
                    //
                    //    0 0 0 0 D D S S  (0..15)  : copy 2 bytes from <= 1kB distance
                    //    length = 2
                    //    state = S (copy S literals after this block)
                    //  Always followed by exactly one byte : H H H H H H H H
                    //    distance = (H << 2) + D + 1

                    new_state = State::try_from(inst & 0x3)?;
                    let h = read_u8(&mut in_pos)?;
                    lb_offset = (inst >> 2) as usize + ((h as usize) << 2) + 1;
                    lb_len = 2;
                }
                FullState::Lit4OrMore => {
                    // If last instruction used to copy 4 or more literals (as detected by
                    // state == 4), the instruction becomes a copy of a 3-byte block from the
                    // dictionary from a 2..3kB distance, and must be interpreted like this :
                    //
                    //    0 0 0 0 D D S S  (0..15)  : copy 3 bytes from 2..3 kB distance
                    //    length = 3
                    //    state = S (copy S literals after this block)
                    //  Always followed by exactly one byte : H H H H H H H H
                    //    distance = (H << 2) + D + 2049

                    new_state = State::try_from(inst & 0x3)?;
                    let h = read_u8(&mut in_pos)?;
                    lb_offset = (inst >> 2) as usize + ((h as usize) << 2) + 2049;
                    lb_len = 3;
                }
            }
        }

        let lb_start = out_pos
            .checked_sub(lb_offset)
            .ok_or(Error::LookbehindOverrun)?;

        // Copy lookbehind
        if out_pos + lb_len > output.len() {
            return Err(Error::OutputOverrun);
        }
        if lb_len <= lb_offset {
            // If the copy sections are not overlapping
            let (src, dst) = output.split_at_mut_checked(out_pos).ok_or(Error::Error)?;
            let src = src.get(lb_start..lb_start + lb_len).ok_or(Error::Error)?;
            let dst = dst.get_mut(..lb_len).ok_or(Error::Error)?;
            dst.copy_from_slice(src);
        } else {
            // copy_within has the same semantics as memmove, which is not what we want
            // Instead convert the output to a slice of cells to have the desired behavior
            // The compiler is able to optimize this pretty well

            let output_cells = core::cell::Cell::from_mut(output).as_slice_of_cells();
            let dst_cells = output_cells
                .get(out_pos..out_pos + lb_len)
                .ok_or(Error::OutputOverrun)?;
            let src_cells = output_cells
                .get(lb_start..lb_start + lb_len)
                .ok_or(Error::Error)?;

            for (src, dst) in src_cells.into_iter().zip(dst_cells) {
                dst.set(src.get());
            }
        }
        out_pos += lb_len;

        state = new_state.into();
        // Copy litteral
        let copy_len = match new_state {
            State::NoLit => 0,
            State::Lit1 => 1,
            State::Lit2 => 2,
            State::Lit3 => 3,
        };
        copy_in_out(&mut in_pos, output, &mut out_pos, copy_len)?;
    }

    if in_pos < input.len() {
        return Err(Error::InputNotConsumed);
    }

    Ok(out_pos)
}
