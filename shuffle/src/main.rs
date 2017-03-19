extern crate rand;

use std::{env, fs, io, mem, slice};
use std::io::{ErrorKind, Read, Write};
use rand::Rng;

#[derive(Copy, Clone, PartialEq)]
struct CooccurRec {
    word1: u32,
    word2: u32,
    val: f64,
}

static mut VERBOSE: i32 = 2;
macro_rules! progress {
    ($v:expr, $fmt:expr) => (
        if unsafe { VERBOSE > $v } { write!(io::stderr(), $fmt).unwrap(); }
    );
    ($v:expr, $fmt:expr, $($arg:tt)*) => (
        if unsafe { VERBOSE > $v } { write!(io::stderr(), $fmt, $($arg)*).unwrap(); }
    );
}
macro_rules! log {
    ($v:expr, $fmt:expr) => (
        if unsafe { VERBOSE > $v } { writeln!(io::stderr(), $fmt).unwrap(); }
    );
    ($v:expr, $fmt:expr, $($arg:tt)*) => (
        if unsafe { VERBOSE > $v } { writeln!(io::stderr(), $fmt, $($arg)*).unwrap(); }
    );
}

fn write_chunk<T>(cr: &Vec<CooccurRec>, fout: &mut T) where T: Write + Sized {
    if cr.len() == 0 { return; }
    let mut old = cr[0];

    for x in &cr[1..] {
        if x.word1 == old.word1 && x.word2 == old.word2 {
            old.val += x.val;
            continue;
        }
        fout.write(unsafe {
            slice::from_raw_parts((&old as *const CooccurRec) as *const u8,
            mem::size_of::<CooccurRec>())}).unwrap();
        old = *x;
    }
    fout.write(unsafe {
        slice::from_raw_parts((&old as *const CooccurRec) as *const u8,
        mem::size_of::<CooccurRec>())}).unwrap();
}

fn shuffle_merge(array_size: usize, file_head: &str, n_temp_file: usize) -> i32 {
    let mut array: Vec<CooccurRec> = Vec::with_capacity(array_size);
    let mut fid:  Vec<Option<fs::File>> = Vec::with_capacity(n_temp_file);
    for fidcounter in 0 .. n_temp_file {
        fid.push(Some(fs::File::open(
                    format!("{}_{:>04}.bin", file_head, fidcounter)).unwrap()));
    }
    progress!(0, "Merging temp files: processed 0 lines.");
    let mut rng = rand::thread_rng();
    let mut total = 0usize;
    loop {
        for f in fid.iter_mut() {
            if f.is_none() { continue; }
            for _ in 0 .. array_size / n_temp_file {
                array.push(unsafe { mem::uninitialized::<CooccurRec>() });
                if let Err(e) = f.as_mut().unwrap().read_exact(unsafe {
                    slice::from_raw_parts_mut((array.last_mut().unwrap() as *mut CooccurRec) as *mut u8,
                    mem::size_of::<CooccurRec>())}) {
                    if e.kind() == ErrorKind::UnexpectedEof {
                        array.pop();
                        *f = None;
                        break;
                    }
                    else {
                        panic!(e);
                    }
                }
            }
        }
        if array.len() == 0 { break; }
        total += array.len();
        rng.shuffle(&mut array);
        write_chunk(&array, &mut io::stdout());
        array.clear();
        progress!(0, "\x1b[31G{} lines.", total);
    }
    log!(-1, "\x1b[0GMerging temp files: processed {} lines.", total);
    fid.clear();
    for i in 0 .. n_temp_file {
        if let Err(e) = fs::remove_file(format!("{}_{:>04}.bin", file_head, i)) {
            log!(-1, "Error on fs::remove_file. Error is {:?}", e);
        }
    }
    0
}

// Shuffle large input stream by splitting into chunks
fn shuffle_by_chunks(array_size: usize, file_head: &str) -> i32 {
    log!(-1, "SHUFFLING COOCCURRENCES");
    log!(0, "array size: {}", array_size);

    let mut rng = rand::thread_rng();

    let mut array: Vec<CooccurRec> = Vec::with_capacity(array_size);
    let mut total = 0usize;
    let mut fin = io::stdin();

    let mut fidcounter = 0;
    progress!(1, "Shuffling by chunks: processed 0 lines.");
    loop {
        array.push(unsafe { mem::uninitialized() });
        if let Err(e) = fin.read_exact(unsafe {
            slice::from_raw_parts_mut((array.last_mut().unwrap() as *mut CooccurRec) as *mut u8,
            mem::size_of::<CooccurRec>()) }) {
            if e.kind() == ErrorKind::UnexpectedEof {
                array.pop();
                break;
            }
            else {
                panic!(e);
            }
        }
        if array.len() >= array_size {
            rng.shuffle(&mut array);
            total += array.len();
            progress!(1, "\x1b[22Gprocessed {} lines.", total);
            write_chunk(&array, &mut fs::File::create(
                    format!("{}_{:>04}.bin", file_head, fidcounter)).unwrap());
            fidcounter += 1;
            array.clear();
        }
    }
    rng.shuffle(&mut array);
    total += array.len();
    write_chunk(&array, &mut fs::File::create(
            format!("{}_{:>04}.bin", file_head, fidcounter)).unwrap());
    fidcounter += 1;
    log!(1, "\x1b[22Gprocessed {} lines.", total);
    log!(1, "Wrote {} temporary file(s).", fidcounter);
    shuffle_merge(array_size, file_head, fidcounter)
}

fn main() {
    let array_size: usize;
    let mut temp_file = "temp_shuffle".to_string();
    {
        let args: Vec<String> = env::args().collect();

        if args.len() == 1 {
            println!("Tools to shuffle entries of word-word cooccurrence files
This program is transportation of GloVe shuffle.
Original program (written in C) author is Jeffrey Pennington (jpennin@stanford.edu)
Transpoted to rust by Takumi Kato (takumi.kt@gmail.com)

Usage options:
\t-verbose <int>
\t\tSet verbosity: 0, 1, or 2 (default)
\t-memory <float>
\t\tSoft limit for memory consumption, in GB -- based on simple heuristic, so not extremely accurate; default 4.0
\t-array-size <int>
\t\tLimit to length <int> the buffer  which stores chunks of data to shuffle before writing to disk.
This value overrides that which is automatically produced by '-memory'.
\t-temp-file <file>
\t\tFilename, excluding extension, for temporary file; default temp_shuffle

Example usage:
./shuffle -verbose 2 -memory 8.0 < cooccurrence.bin > cooccurrence.shuf.bin");
            std::process::exit(0);
        }
        let mut is_skip = true;
        let mut it = args.into_iter().peekable();

        let mut array_size_opt: Option<usize> = None;
        let mut memory_limit = 2.0f64;
        loop {
            let arg = match it.next() {
                Some(s) => s,
                None => break,
            };
            if is_skip {
                is_skip = false;
                continue
            }
            match arg.as_ref() {
                "-verbose" => {
                    unsafe { VERBOSE = it.peek().and_then(|v| v.parse().ok())
                        .and_then(|x| if 0 <= x && x <= 2 { Some(x) } else { None })
                        .expect("-verbose <int>, verbosity: 0, 1, or 2 (default)") };
                    is_skip = true;
                },
                "-memory" => {
                    memory_limit = it.peek().and_then(|v| v.parse().ok())
                        .expect("-memory <float>");
                    is_skip = true;
                },
                "-array-size" => {
                    array_size_opt = Some(it.peek().and_then(|v| v.parse().ok())
                        .expect("-array-size <int>"));
                    is_skip = true;
                },
                "-temp-file" => {
                    temp_file = it.peek().expect("-temp-file <file>").to_string();
                    is_skip = true;
                },
                &_ => {},
            }
        }
        array_size = array_size_opt.unwrap_or((0.95 * memory_limit as f64 * 1073741724f64 / mem::size_of::<CooccurRec>() as f64) as usize);
    }
    std::process::exit(shuffle_by_chunks(array_size, &temp_file));
}
