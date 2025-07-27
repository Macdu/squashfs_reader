use squashfs_reader::{FileSystem, FileType};
use std::{env, fs, io, path::Path};

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <squashfs-file> <output-dir>", args[0]);
        return Ok(());
    }

    let input_file = &args[1];
    let output_dir = &args[2];

    // Open the squashfs file
    let fs = FileSystem::from_path(input_file)?;

    // Create output directory if it doesn't exist
    fs::create_dir_all(output_dir)?;

    // Start recursive extraction from root
    extract_recursive(&fs, Path::new("/"), Path::new(output_dir))?;

    Ok(())
}

/// Recursively extract files from the squashfs filesystem.
fn extract_recursive(
    fs: &FileSystem<fs::File>,
    current_path: &Path,
    output_base: &Path,
) -> io::Result<()> {
    let dir = fs.read_dir(current_path)?;

    for entry in dir {
        let entry = entry?;
        let metadata = entry.metadata(fs)?;
        let rel_path = current_path.join(entry.name());
        let out_path = output_base.join(rel_path.strip_prefix("/").unwrap());

        match metadata.file_type() {
            FileType::Directory => {
                fs::create_dir(&out_path)?;
                extract_recursive(fs, &rel_path, output_base)?;
            }
            FileType::File => {
                println!("Extracting: {}", rel_path.display());
                let mut reader = metadata.read_file(fs)?;
                let mut file = fs::File::create(&out_path)?;
                io::copy(&mut reader, &mut file)?;
            }
            FileType::Symlink => {
                unimplemented!("Symlink extraction is not implemented yet");
            }
        }
    }

    Ok(())
}
