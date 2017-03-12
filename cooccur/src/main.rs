use std::{env, io, mem, num, i32, usize, fs, slice, string};
use std::io::{Read, Write, BufRead, Error, ErrorKind};
use std::cmp::Ordering;
use std::collections::{HashMap,BinaryHeap};

#[derive(Copy, Clone, PartialEq)]
struct CooccurRec {
    word1: u32,
    word2: u32,
    val: f32,
}

#[derive(Copy, Clone, PartialEq)]
struct CRecId {
    crec: CooccurRec,
    id: usize,
}
impl CRecId {
    fn new(c: CooccurRec, n: usize) -> CRecId {
        CRecId { crec: c, id: n}
    }
}
impl Eq for CRecId {}
impl Ord for CRecId {
    fn cmp(&self, other: &CRecId) -> Ordering {
        // For using std::collections::BinaryHeap, ordering is reversed.
        match other.crec.word1.cmp(&self.crec.word1) {
            Ordering::Equal => other.crec.word2.cmp(&self.crec.word2),
            x => x,
        }
    }
}
impl PartialOrd for CRecId {
    fn partial_cmp(&self, other: &CRecId) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

static mut verbose: i32 = 2;

macro_rules! progress {
    ($v:expr, $fmt:expr) => (
        if verbose > $v { write!(io::stderr(), $fmt).unwrap(); }
    );
    ($v:expr, $fmt:expr, $($arg:tt)*) => (
        if verbose > $v { write!(io::stderr(), $fmt, $($arg)*).unwrap(); }
    );
}
macro_rules! log {
    ($v:expr, $fmt:expr) => (
        if verbose > $v { writeln!(io::stderr(), $fmt).unwrap(); }
    );
    ($v:expr, $fmt:expr, $($arg:tt)*) => (
        if verbose > $v { writeln!(io::stderr(), $fmt, $($arg)*).unwrap(); }
    );
}

enum GetWord {
    Word(String),
    EndLine,
    EndOfFile,
    IOError(io::Error),
    Utf8Error(std::string::FromUtf8Error),
}

fn get_word(fin: &io::Read) -> GetWord {
    static mut next_endline: bool = false;
    if next_endline {
        next_endline = false;
        return GetWord::EndLine
    }

    let mut eof: bool = false;
    let mut word: Vec<u8> = vec![];
    let mut byte: [u8; 1];
    loop {
        match fin.read_exact(&mut byte) {
            Ok(_) => match byte[0] {
                b' ' | b'\t' => { break; },
                b'\n' => {
                    next_endline = true;
                    break;
                },
                c => { word.push(c); }
            },
            Err(e) => if e.kind() == ErrorKind::UnexpectedEof {
                eof = true;
                break;
            } else {
                return GetWord::IOError(e);
            }
        }
    }
    if word.len() > 0 {
        match String::from_utf8(word) {
            Ok(x) => GetWord::Word(x),
            Err(e) => GetWord::Utf8Error(e),
        }
    } else if eof {
        GetWord::EndOfFile
    } else {
        get_word(fin)
    }
}

fn merge_files(file_head: &str, num: usize) {
    progress!(1, "Merging cooccurrence files: processed 0 lines.");
    let pq = BinaryHeap::<CRecId>::new();
    let fid = Vec::<fs::File>::new();
    for i in 0..num {
        let mut data: CooccurRec;
        let f = fs::File::open(format!("{}_{:>4}.bin", file_head, i)).unwrap();
        f.read_exact(unsafe {
            slice::from_raw_parts_mut((&mut data as *mut CooccurRec) as *mut u8,
            mem::size_of::<CooccurRec>())}).unwrap();
        pq.push(CRecId::new(data, i));
        fid.push(f);
    }
    // Pop top node, save it in old to see if the next entry is a duplicate
    let mut new = pq.pop().unwrap();
    //...
    //Repeatedly...
}

fn get_cooccurrence(symmetric: bool, window_size: usize,
                    max_product: usize, overflow_length: usize,
                    vocab_file: &str, file_head: &str) -> i32 {
    macro_rules! min {
        ($a: expr, $b: expr) => { if $a <= $b { $a } else { $b } };
    }
    macro_rules! max {
        ($a: expr, $b: expr) => { if $a >= $b { $a } else { $b } };
    }

    log!(-1, "COUNTING COOCCURRENCES");
    log!(0, "window size: {}", window_size);
    log!(0, "context: {}", if symmetric { "symmetric" } else { "asymmetric" });
    log!(1, "max product: {}", max_product);
    log!(1, "overflow length: {}", overflow_length);

    let mut vocab_hash: HashMap<String, i64> = HashMap::new();
    log!(1, "Reading vocab from file \"{}\"...", vocab_file);
    {
        let file = io::BufReader::new(fs::File::open(vocab_file).unwrap());
        for (i, line) in file.lines().enumerate() {
            let line = line.unwrap();
            let vec: Vec<&str> = line.split_whitespace().collect();
            let word = vec[0];

            vocab_hash.insert(word.to_string(), i as i64);
        }
        log!(1, "loaded {} words.", vocab_hash.len());
    }
    let mut table: Vec<Vec<f32>> = Vec::with_capacity(vocab_hash.len());
    {
        let mut size = 1;
        table.push(vec![0.0_f32]);
        for a in 1..vocab_hash.len() {
            size += min!(max_product / a, vocab_hash.len());
            table.push(vec![0.0_f32 ; size]);
        }
    }
    // for each token in input stream, calculate a weighted cooccurrence sum within window_size.
    let value = |j, k| 1.0f32 / (j - k) as f32;
    let write_chunk = |cr: &Vec<CooccurRec>, n| {
        let mut f = fs::File::create(format!("{}_{:>04}.bin", file_head, n)).unwrap();
        let mut old = cr[0];

        for x in &cr[1..] {
            if x.word1 == old.word1 && x.word2 == old.word2 {
                old.val += x.val;
                continue;
            }
            f.write(unsafe {
                slice::from_raw_parts((&old as *const CooccurRec) as *const u8,
                mem::size_of::<CooccurRec>())}).unwrap();
            old = *x;
        }
        f.write(unsafe {
            slice::from_raw_parts((&old as *const CooccurRec) as *const u8,
            mem::size_of::<CooccurRec>())}).unwrap();
    };

    let mut fcounter = 1i32;
    let mut cr: Vec<CooccurRec> = Vec::with_capacity(overflow_length + 1);
    let mut history = vec![0i64 ; window_size as usize];
    let mut n_words = 0usize;
    let mut j = 0i64;
    let stdin = io::BufReader::new(io::stdin());
    progress!(1, "Processing token: 0");
    loop {
        let word = match get_word(&stdin) {
            GetWord::Word(w) => w,
            GetWord::EndLine => { j = 0; continue },
            GetWord::EndOfFile => break,
            GetWord::Utf8Error(_) => continue,
            GetWord::IOError(e) => panic!(format!("{:?}", e)),
        };
        j += 1;
        n_words += 1;
        if n_words % 100000 == 0 { progress!(1, "\x1b[19G{}", j); }
        if let Some(&w2) = vocab_hash.get(&word) {
            for k in (max!(j - window_size as i64, 0) .. j - 1).rev() {
                let w1 = history[k as usize % window_size];
                if (w1 as usize) < max_product / (w2 as usize) {  // Product is small enough to store in a full array
                    table[(w1 - 1) as usize][(w2 - 2) as usize] += value(j, k);
                    if symmetric { table[(w2 - 1) as usize][(w1 - 2) as usize] += value(j, k); }
                }
                else {  // Product is too big, data is likely to be sparse. Store these entries in a temporary buffer to be sorted, merged (accumulated), and written to file when it gets full.
                    cr.push(CooccurRec { word1: w1 as u32, word2: w2 as u32, val: value(j, k)});
                    if symmetric { cr.push(CooccurRec { word1: w2 as u32, word2: w1 as u32, val: value(j, k)}); }
                    if cr.len() >= overflow_length - window_size {
                        cr.sort_by(|lhs, rhs| match lhs.word1.cmp(&rhs.word1) {
                            Ordering::Equal => lhs.word2.cmp(&rhs.word2),
                            x => x,
                        });
                        write_chunk(&cr, fcounter);
                        fcounter += 1;
                        cr.clear();
                    }
                }
                history[j as usize % window_size] = w2;
                j += 1;
            }
        }
    }
    log!(1, "\x1b[0GProcessed {} tokens.", n_words);
    cr.sort_by(|lhs, rhs| match lhs.word1.cmp(&rhs.word1) {
        Ordering::Equal => lhs.word2.cmp(&rhs.word2),
        x => x,
    });
    write_chunk(&cr, fcounter);
    log!(1, "Writing cooccurrences to disk");
    let j = i32::MAX;
    let file = fs::File::create(format!("{}_0000.bin", file_head)).unwrap();
    for (x, &v) in table.iter().enumerate() {
        if ((0.75f64 * (vocab_hash.len() / x) as f64).ln() as i32) < j {
            progress!(1, ".")
        }
        for (y, &r) in v.iter().enumerate() {
            if r != 0f32 {
                file.write(unsafe {mem::transmute::<u32, &[u8; 4]>(x as u32)});
                file.write(unsafe {mem::transmute::<u32, &[u8; 4]>(y as u32)});
                file.write(unsafe {mem::transmute::<f32, &[u8; 4]>(r)});
            }
        }
    }
    log!(1, "{} files in total.", fcounter);
    merge_files(file_head, fcounter + 1);
    0
}

fn main() {
    let mut window_size = 15usize;
    let mut symmetric = true;
    let mut memory_limit = 3.0;
    let mut vocab_file = "vocab.txt".to_string();
    let mut file_head = "overflow".to_string();
    let mut max_product: Option<usize> = None;
    let mut overflow_length: Option<usize> = None;
    {
        let args: Vec<String> = env::args().collect();

        if args.len() == 1 {
            println!("Tool to calculate word-word cooccurrence statistics
This program is transportation of GloVe cooccur.
Original program (written in C) author is Jeffrey Pennington (jpennin@stanford.edu)
Transpoted to rust by Takumi Kato (takumi.kt@gmail.com)

Usage options:
\t-verbose <int>
\t\tSet verbosity: 0, 1, or 2 (default)
\t-symmetric <int>
\t\tIf <int> = 0, only use left context; if <int> = 1 (default), use left and right
\t-window-size <int>
\t\tNumber of context words to the left (and to the right, if symmetric = 1); default 15
\t-vocab-file <file>
\t\tFile containing vocabulary (truncated unigram counts, produced by 'vocab_count'); default vocab.txt
\t-memory <float>
\t\tSoft limit for memory consumption, in GB -- based on simple heuristic, so not extremely accurate; default 4.0
\t-max-product <int>
\t\tLimit the size of dense cooccurrence array by specifying the max product <int> of the frequency counts of the two cooccurring words.
\t\tThis value overrides that which is automatically produced by '-memory'. Typically only needs adjustment for use with very large corpora.
\toverflow-file <file>
\t\tFilename, excluding extension, for temporary file; default overflow

Example usage:
./cooccur -verbose 2 -symmetric 0 -window-size 10 -vocab-file vocab.txt -memory 8.0 -overflow-file tempoverflow < corpus.txt > cooccurrences.bin

");
            std::process::exit(0);
        }
        let mut is_skip = true;
        let mut it = args.into_iter().peekable();
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
                    verbose = it.peek().and_then(|v| v.parse().ok())
                        .and_then(|x| if 0 <= x && x <= 2 { Some(x) } else { None })
                        .expect("-verbose <int>, verbosity: 0, 1, or 2 (default)");
                    is_skip = true;
                },
                "-symmetric" => {
                    symmetric = it.peek().and_then(|v| v.parse().ok())
                        .and_then(|x| if 0 <= x && x <= 1 { Some(x) } else { None })
                        .expect("symmetric <int>, 0 (only use left context) or 1 (use left and right; default)") == 1;
                    is_skip = true;
                },
                "-window-size" => {
                    window_size = it.peek().and_then(|v| v.parse().ok()).expect("-window-size <int>");
                    is_skip = true;
                },
                "-memory" => {
                    memory_limit = it.peek().and_then(|v| v.parse().ok()).expect("-memory <float>");
                    is_skip = true;
                },
                "-vocab-file" => {
                    vocab_file = it.peek().expect("-vocab-file <file>").to_string();
                    is_skip = true;
                },
                "-overflow" => {
                    file_head = it.peek().expect("-overflow <file>").to_string();
                    is_skip = true;
                },
                "-max-product" => {
                    max_product = Some(it.peek().and_then(|v| v.parse().ok()).expect("-max-product <int>"));
                    is_skip = true;
                },
                "-overflow-length" => {
                    overflow_length = Some(it.peek().and_then(|v| v.parse().ok()).expect("-overflow-length <int>"));
                    is_skip = true;
                },
                &_ => {},
            }
            let rlimit = 0.85 * memory_limit * 1073741824_f64 / mem::size_of::<f64>() as f64;
            let mut n = 1e5_f64;
            while (rlimit - n * (n.ln() + 0.1544313298)).abs() > 1e-3 { n = rlimit / (n.ln() + 0.1544313298) }
            max_product = max_product.or_else(|| Some(n as usize));
            overflow_length = overflow_length.or_else(|| Some((rlimit / 6.0) as usize));
        }
        //std::process::exit(get_counts(verbose, max_vocab, min_count));
    }
}
