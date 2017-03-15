use std::{env, io, mem, usize, fs, slice};
use std::io::{Read, Write, BufRead, ErrorKind};
use std::cmp::Ordering;
use std::collections::{HashMap,BinaryHeap};

#[derive(Copy, Clone, PartialEq)]
struct CooccurRec {
    word1: u32,
    word2: u32,
    val: f64,
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

enum GetWord {
    Word(String),
    LastWord(String),
    EndOfFile,
    IOError(io::Error),
    Utf8Error(std::string::FromUtf8Error),
}

fn get_word(fin: &mut io::Read) -> GetWord {
    let mut lastword: bool = false;

    let mut word: Vec<u8> = vec![];
    let mut byte: [u8; 1] = [0];
    loop {
        match fin.read_exact(&mut byte) {
            Ok(_) => match byte[0] {
                b' ' | b'\t' => { break; },
                b'\n' => {
                    lastword = true;
                    break;
                },
                c => { word.push(c); }
            },
            Err(e) => if e.kind() == ErrorKind::UnexpectedEof {
                return GetWord::EndOfFile;
            } else {
                return GetWord::IOError(e);
            }
        }
    }
    if word.len() > 0 {
        match String::from_utf8(word) {
            Ok(x) => if lastword { GetWord::LastWord(x) } else { GetWord::Word(x) },
            Err(e) => GetWord::Utf8Error(e),
        }
    } else {
        get_word(fin)
    }
}

fn merge_write(new: &CRecId, old: &mut CRecId, fout: &mut Write) -> usize {
    if new.crec.word1 == old.crec.word1 && new.crec.word2 == old.crec.word2 {
        old.crec.val += new.crec.val;
        return 0;
    }
    fout.write(unsafe {
            slice::from_raw_parts((&old.crec as *const CooccurRec) as *const u8,
            mem::size_of::<CooccurRec>())}).unwrap();
    *old = *new;
    1
}

fn merge_files(file_head: &str, num: usize) -> i32 {
    progress!(1, "Merging cooccurrence files: processed 0 lines.");
    let mut fout = io::stdout();
    let mut pq = BinaryHeap::<CRecId>::new();
    let mut fid = Vec::<fs::File>::new();
    let mut new: CooccurRec = CooccurRec { word1: 0, word2: 0, val: 0.0 };
    for i in 0..num {
        let mut f = fs::File::open(format!("{}_{:>04}.bin", file_head, i)).unwrap();
        f.read_exact(unsafe {
            slice::from_raw_parts_mut((&mut new as *mut CooccurRec) as *mut u8,
            mem::size_of::<CooccurRec>())}).unwrap();
        pq.push(CRecId::new(new, i));
        fid.push(f);
    }
    // Pop top node, save it in old to see if the next entry is a duplicate
    let mut old = pq.pop().unwrap();
    if fid.get(old.id).unwrap().read_exact(unsafe {
            slice::from_raw_parts_mut((&mut new as *mut CooccurRec) as *mut u8,
            mem::size_of::<CooccurRec>())}).is_ok() {
        pq.push(CRecId::new(new, old.id));
    }
    //Repeatedly pop top node and fill priority queue until files have reached EOF
    let mut counter = 0usize;
    while pq.len() > 0 {
        let crecid = pq.pop().unwrap();
        counter += merge_write(&crecid, &mut old, &mut fout);
        if counter % 100000 == 0 { progress!(1, "\x1b[39G{} lines.", counter); }
        match fid[crecid.id].read_exact(unsafe {
            slice::from_raw_parts_mut((&mut new as *mut CooccurRec) as *mut u8,
            mem::size_of::<CooccurRec>())}) {
            Ok(_) => pq.push(CRecId::new(new, crecid.id)),
            Err(x) => if x.kind() != ErrorKind::UnexpectedEof { panic!(x) }
        }
    }
    fout.write(unsafe {
        slice::from_raw_parts((&old.crec as *const CooccurRec) as *const u8,
        mem::size_of::<CooccurRec>())}).unwrap();
    counter += 1;
    log!(-1, "\x1b[0GMerging coocurrence files: processed {} lines.", counter);
    for i in 0..num {
        if let Err(e) = fs::remove_file(format!("{}_{:>04}.bin", file_head, i)) {
            log!(-1, "Error on fs::remove_file. Error is {:?}", e);
        }
    }
    log!(-1, "");
    0
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
    progress!(1, "Reading vocab from file \"{}\"...", vocab_file);
    {
        let file = io::BufReader::new(fs::File::open(vocab_file).unwrap());
        for (i, line) in file.lines().enumerate() {
            let line = line.unwrap();
            let vec: Vec<&str> = line.split_whitespace().collect();
            let word = vec[0];

            vocab_hash.insert(word.to_string(), (i + 1) as i64);
        }
        log!(1, "loaded {} words.", vocab_hash.len());
    }
    let mut table: Vec<Vec<f64>> = Vec::with_capacity(vocab_hash.len());
    {
        let mut n_elements = 0usize;
        table.push(vec![]);
        for a in 1..vocab_hash.len()+1 {
            table.push(vec![0.0_f64 ; min!(max_product / a, vocab_hash.len())]);
            n_elements += min!(max_product / a, vocab_hash.len());
        }
        log!(1, "Table contains {} elements.", n_elements);
    }
    // for each token in input stream, calculate a weighted cooccurrence sum within window_size.
    let value = |j, k| 1.0f64 / (j - k) as f64;
    let write_chunk = |cr: &Vec<CooccurRec>, n| {
        log!(-1, "write_chunk, n = {}, cr.len = {}", n, cr.len());
        if cr.len() == 0 { return; }
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

    let mut fcounter = 1usize;
    let mut cr: Vec<CooccurRec> = Vec::with_capacity(overflow_length + 1);
    let mut history = vec![0i64 ; window_size as usize];
    let mut n_words = 0usize;
    let mut j = 0i64;
    let mut stdin = io::BufReader::new(io::stdin());
    let mut endline = false;
    progress!(1, "Processing token: 0");
    loop {
        if endline {
            j = 0;
            endline = false;
            continue;
        }
        let word = match get_word(&mut stdin) {
            GetWord::Word(w) => w,
            GetWord::LastWord(w) => { endline = true; w },
            GetWord::EndOfFile => break,
            GetWord::Utf8Error(_) => continue,
            GetWord::IOError(e) => panic!(format!("{:?}", e)),
        };
        n_words += 1;
        if n_words % 100000 == 0 { progress!(1, "\x1b[19G{}", n_words); }
        if let Some(&w2) = vocab_hash.get(&word) {
            for k in (max!(j - window_size as i64, 0) .. j - 1).rev() {
                let w1 = history[k as usize % window_size];
                if (w1 as usize) < max_product / (w2 as usize) {  // Product is small enough to store in a full array
                    table[w1 as usize][(w2 - 1) as usize] += value(j, k);
                    if symmetric { table[w2 as usize][(w1 - 1) as usize] += value(j, k); }
                }
                else {  // Product is too big, data is likely to be sparse. Store these entries in a temporary buffer to be sorted, merged (accumulated), and written to file when it gets full.
                    cr.push(CooccurRec { word1: w1 as u32, word2: w2 as u32, val: value(j, k) });
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
            }
            history[j as usize % window_size] = w2;
            j += 1;
        }
    }
    log!(1, "\x1b[0GProcessed {} tokens.", n_words);
    cr.sort_by(|lhs, rhs| match lhs.word1.cmp(&rhs.word1) {
        Ordering::Equal => lhs.word2.cmp(&rhs.word2),
        x => x,
    });
    write_chunk(&cr, fcounter);
    progress!(1, "Writing cooccurrences to disk");
    let mut j = 1e6 as i64;
    let mut file = fs::File::create(format!("{}_0000.bin", file_head)).unwrap();
    for (x, v) in table.iter().enumerate() {
        if ((0.75f64 * (vocab_hash.len() / (x + 1)) as f64).ln() as i64) < j {
            j = (0.75f64 * (vocab_hash.len() / (x + 1)) as f64).ln() as i64;
            progress!(1, ".")
        }
        for (y, &r) in v.iter().enumerate() {
            if r != 0f64 {
                let x = x as u32 + 1;
                let y = y as u32 + 1;
                file.write(unsafe { mem::transmute::<&u32, &[u8; 4]>(&x) }).unwrap();
                file.write(unsafe { mem::transmute::<&u32, &[u8; 4]>(&y) }).unwrap();
                file.write(unsafe { mem::transmute::<&f64, &[u8; 8]>(&r) }).unwrap();
            }
        }
    }
    log!(1, "{} files in total.", fcounter);
    merge_files(file_head, fcounter + 1)
}

fn main() {
    let mut window_size = 15usize;
    let mut symmetric = true;
    let mut memory_limit = 3.0;
    let mut vocab_file = "vocab.txt".to_string();
    let mut file_head = "overflow".to_string();
    let max_product: usize;
    let overflow_length: usize;
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
        let mut max_product_opt: Option<usize> = None;
        let mut overflow_length_opt: Option<usize> = None;
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
                    let v_ = it.peek().and_then(|v| v.parse().ok())
                        .and_then(|x| if 0 <= x && x <= 2 { Some(x) } else { None })
                        .expect("-verbose <int>, verbosity: 0, 1, or 2 (default)");
                    unsafe { VERBOSE = v_; }
                    is_skip = true;
                },
                "-symmetric" => {
                    symmetric = it.peek().and_then(|v| v.parse().ok())
                        .and_then(|x| if 0 <= x && x <= 1 { Some(x) } else { None })
                        .expect("symmetric <int>, 0 (only use left context) or 1 (use left and right; default)") == 1;
                    is_skip = true;
                },
                "-window-size" => {
                    window_size = it.peek().and_then(|v| v.parse().ok())
                        .expect("-window-size <int>");
                    is_skip = true;
                },
                "-memory" => {
                    memory_limit = it.peek().and_then(|v| v.parse().ok())
                        .expect("-memory <float>");
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
                    max_product_opt = Some(it.peek().and_then(|v| v.parse().ok())
                                           .expect("-max-product <int>"));
                    is_skip = true;
                },
                "-overflow-length" => {
                    overflow_length_opt = Some(it.peek().and_then(|v| v.parse().ok())
                                               .expect("-overflow-length <int>"));
                    is_skip = true;
                },
                &_ => {},
            }
        }
        let rlimit = 0.85 * memory_limit * 1073741824_f64 / mem::size_of::<CooccurRec>() as f64;
        let mut n = 1e5_f64;
        while (rlimit - n * (n.ln() + 0.1544313298)).abs() > 1e-3 { n = rlimit / (n.ln() + 0.1544313298) }
        max_product = max_product_opt.unwrap_or(n as usize);
        overflow_length = overflow_length_opt.unwrap_or((rlimit / 6.0) as usize);
        std::process::exit(get_cooccurrence(symmetric, window_size, max_product,
                                            overflow_length, &vocab_file, &file_head))
    }
}
