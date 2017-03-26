use std::env;
use std::io;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::io::Write;
use std::io::BufRead;
use std::usize;

struct Vocab {
    word: String,
    count: usize,
}
impl Vocab {
    fn new(word: String, count: usize) -> Vocab { Vocab { word: word, count: count } }
}

fn get_counts(verbose: i32, max_vocab: usize, min_count: usize) -> i32 {
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

    log!(-1, "BUILDING VOCABULALY");
    let mut vocab: Vec<Vocab> = Vec::with_capacity(12500);
    {
        let mut vocab_hash: HashMap<String, usize> = HashMap::new();
        let mut i = 0;
        progress!(1, "Processed {} tokens.", i);
        for word in io::BufReader::new(io::stdin()).split(b' ') {
            match String::from_utf8(word.unwrap()) {
                Ok(s) => {
                    if s == "<unk>" {
                        log!(-1, "\nError, <unk> vector found in corpus.\nPlease remove <unk>s from your corpus (e.g. cat text8 | sed -e 's/<unk>/<raw_unk>/g' > text8.new)");
                        return 1
                    }
                    if s.len() == 0 { continue }
                    *vocab_hash.entry(s).or_insert(0) += 1;
                    i += 1;
                    if i % 100000 == 0 { progress!(1, "\x1b[11G{} tokens.", i) }
                }
                _ => break
            }
        }
        log!(1, "\x1b[0GProcessed {} tokens.", i);
        for (k, v) in vocab_hash {
            vocab.push(Vocab::new(k, v));
        }
        log!(1, "Counted {} unique words.", vocab.len());
    }

    let mut max_vocab = max_vocab;
    if max_vocab > 0 && max_vocab < vocab.len() {
        vocab.sort_by_key(|v| usize::MAX - v.count);
        vocab[..max_vocab].sort_by(|lhs, rhs| match lhs.count.cmp(&rhs.count) {
            Ordering::Less => Ordering::Greater,
            Ordering::Equal => lhs.word.cmp(&rhs.word),
            Ordering::Greater => Ordering::Less,
        });
    }
    else {
        vocab.sort_by(|lhs, rhs| match lhs.count.cmp(&rhs.count) {
            Ordering::Less => Ordering::Greater,
            Ordering::Equal => lhs.word.cmp(&rhs.word),
            Ordering::Greater => Ordering::Less,
        });
        max_vocab = vocab.len();
    }


    let mut n = vocab.len();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    for (i, x) in vocab.iter().enumerate() {
        if i == max_vocab { n = i; break; }
        if x.count < min_count {
            log!(0, "Truncating vocabulary at min count {}.", min_count);
            n = i;
            break;
        }
        writeln!(&mut stdout, "{} {}", x.word, x.count).unwrap();
    }
    if n == max_vocab && max_vocab < vocab.len() {
        log!(0, "Truncating vocabulary at size {}.", max_vocab);
    }
    log!(-1, "Using vocabulary of size {}.\n", n);
    0
}

fn main() {
    let mut verbose = 1i32;
    let mut min_count = 1usize;
    let mut max_vocab = 0usize;
    {
        let args: Vec<String> = env::args().collect();

        if args.len() == 1 {
            println!("Simple tool to extract unigram counts
This program is transportation of GloVe vocab_count.
Original program (written in C) author is Jeffrey Pennington (jpennin@stanford.edu)
Transpoted to rust by Takumi Kato (takumi.kt@gmail.com)

Usage options:
\t-verbose <int>
\t\tSet verbosity: 0, 1, or 2 (default)
\t-max-vocab <int>
\t\tUpper bound on vocabulary size, i.e. keep the <int> most frequent words. The minimum frequency words are randomly sampled so as to obtain an even distribution over the alphabet.
\t-min-count <int>
\t\tLower limit such that words which occur fewer than <int> times are discarded.

Example usage:
./vocab_count -verbose 2 -max-vocab 100000 -min-count 10 < corpus.txt > vocab.txt");
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
                "-max-vocab" => {
                    max_vocab = it.peek().and_then(|v| v.parse().ok()).expect("-max-vocab <int>");
                    is_skip = true;
                },
                "-min-count" => {
                    min_count = it.peek().and_then(|v| v.parse().ok()).expect("-min-count <int>");
                    is_skip = true;
                },
                &_ => {},
            }
        }
    }
    std::process::exit(get_counts(verbose, max_vocab, min_count));
}
