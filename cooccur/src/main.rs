use std::env;
use std::io;
use std::mem;
use std::num;
use std::fs;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::io::Read;
use std::io::Write;
use std::io::BufRead;
use std::usize;

struct CooccurRec {
    word1: u32,
    word2: u32,
    val: f32,
}

fn get_filename(file_head: &str, n: i32) -> String {
    return format!("{}_{:>04}.bin", file_head, n);
}

fn get_word(fin: &io::Read) -> Option<String> {
    let mut word: Vec<u8> = vec![];
    let mut byte: [u8; 1];
    while fin.read_exact(&mut byte).is_ok() {
        match byte[0] {
            b' ' | b'\t' | b'\n' => { break; }
            c => { word.push(c); }
        }
    }
    if word.len() > 0 { String::from_utf8(word).ok() } else { None }
}

fn get_cooccurrence(verbose: i32, symmetric: bool, window_size: i32, max_product: usize, overflow_length: usize,
                    vocab_file: &str, file_head: &str) -> i32 {
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
    {
        log!(1, "Reading vocab from file \"{}\"...", vocab_file);
        let file = io::BufReader::new(fs::File::open(vocab_file).expect("Unable to open vocab file."));
        let mut vocab_rank = 1i64;
        for line in file.lines() {
            let line = line.expect("vocab file read error.");
            let vec: Vec<&str> = line.split_whitespace().collect();
            let word = vec[0];
            //let num = vec[1].parse::<u32>().expect("Parse error.");

            vocab_hash.insert(word.to_string(), vocab_rank);
            vocab_rank += 1;
        }
        log!(1, "loaded {} words.\nBuilding lookup table...", vocab_hash.len());
    }
    let mut table: Vec<Vec<f32>> = Vec::with_capacity(vocab_hash.len());
    {
        for a in 0..vocab_hash.len() {
            table.push(vec![0.0_f32 ; min!(max_product / a, vocab_hash.len())]);
        }
    }
    {
        let mut fcounter = 0i32;
        let mut wcounter = 0i64;
        let mut cr: Vec<CooccurRec> = Vec::with_capacity(overflow_length + 1);
        let mut history = vec![0i64 ; window_size];
        let stdin = io::BufReader::new(io::stdin());
        'outer: loop {  // for each file.
            progress!(1, "Processing token: 0");
            let mut file = fs::File::open(get_filename(file_head, fcounter)).expect("File open error.");
            loop {  // for each line.
                let mut n_words = 0;
                while let Some(word) = get_word(&stdin) {
                    wcounter += 1;
                    if counter % 100000 == 0 { progress!(1, "\x1b[19G{}", wcounter); }
                    if let Some(w2) = vocab_hash.get(word) {
                        for k in max!(wcounter - window_size, 0) .. (wcounter - 1).rev() {
                            let w1 = history[k % window_size];
                            if w1 < max_product / w2 {  // Product is small enough to store in a full array
                                table[w1 - 1][w2 - 2] += (1.0 / (wcounter - k) as f64) as f32;
                                if symmetric { table[w2 - 1][w1 - 2] += (1.0 / (wcounter - k) as f64) as f32; }
                            }
                            else {  // Product is too big, data is likely to be sparse. Store these entries in a temporary buffer to be sorted, mered (accumulated), and written to file when it gets full.
                                cr.push(CooccurRec { word1: w1, word2: w2, val: get_val(?)});
                                if symmetric { cr.push(CooccurRec { word1: w2, word2: w1, val: get_val(?)}); }
                            }
                            history[wcounter % window_size] = w2;
                            // TODO: write here...
                        }
                    }
                }
                if n_words == 0 { break; }
            }
            // TODO:
            // sort cr
            // write cr
        }
        // TODO: write out ...
    }
    0
}

fn main() {
    let mut verbose = 1i32;
    let mut window_size = 15i32;
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
