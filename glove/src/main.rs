use std::{env, fs, io, mem, slice};
use std::io::{ErrorKind, Read, Write};

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

fn train_glove() -> i32 {
    log!(-1, "TRAINING MODEL");
    let num_lines = fs::metadata(input_file).unwrap().len() / mem::size_of(CooccurRec);
    log!(-1, "Read {} lines.", num_lines);
    log!(1, "Initializing parameters...");
    initialize_parameters();
    0
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() == 1 {
        println!("Tool to calculate word-word cooccurrence statistics
This program is transportation of GloVe glove.
Original program (written in C) author is Jeffrey Pennington (jpennin@stanford.edu)
Transpoted to rust by Takumi Kato (takumi.kt@gmail.com)

Usage options:
\t-verbose <int>
\t\tSet verbosity: 0, 1, or 2 (default)
\t-vector-size <int>
\t\tDimension of word vector representations (excluding bias term); default 50
\t-threads <int>
\t\tNumber of threads; default 8
\t-iter <int>
\t\tNumber of training iterations; default 25
\t-eta <float>
\t\tInitial leaning rate; default 0.05
\t-alpha <float>
\t\tParameter in exponent of weighting function; default 0.75
\t-x-max <float>
\t\tParameter specifying cutoff in weighting function; default 100.0
\t-binary <int>
\t\tSave output in binary format (0: text, 1: binary, 2: both); default 0
\t-model <int>
\t\tModel for word vector output (for text output only); default 2
\t\t   0: output all data, for both word and context word vectors, including bias terms
\t\t   1: output word vectors, excluding bias terms
\t\t   2: output word vectors + context word vectors, excluding bias terms
\t-input-file <file>
\t\tBinary input file of shuffled cooccurrence data (produced by 'cooccur' and 'shuffle'); default cooccurrence.shuf.bin
\t-vocab-file <file>
\t\tFile containing vocabulary (truncated unigram counts, produced by 'vocab_count'); default vocab.txt
\t-save-file <file>
\t\tFilename, excluding extension, for word vector output; default vectors
\t-gradsq-file <file>
\t\tFilename, excluding extension, for squared gradient output; defaut gradsq
\t-save-gradsq <int>
\t\tSave accumulated squared gradients; default 0 (off); ignored if gradsq-file is specified
\t-checkpoint-every <int>
\t\tCheckpoint a model every <int> iterations; default 0 (off)

Example usage:
./glove -input-file cooccurrence.shuf.bin -vocab-file vocab.txt -save-file vectors -gradsq-file gradsq -verbose 2 -vector-size 100 -threads 16 -alpha 0.75 -x-max 100.0 -eta 0.05 -binary 2 -model 2

");
        std::process::exit(0);
    }
    let mut is_skip = true;
    let mut it = args.into_iter().peekable();
    let mut vector_size = 50usize;
    let mut n_iter = 25usize;
    let mut n_threads = 8usize;
    let mut alpha = 0.75;
    let mut x_max = 100.0;
    let mut eta = 0.05;
    let mut binary = 0i32;
    let mut model = 2i32;
    let mut save_gradsq = false;
    let mut checkpoint_every = false;
    let mut vocab_file = "vocab.txt".to_string();
    let mut input_file = "cooccurrence.shuf.bin".to_string();
    let mut save_file = "vectors".to_string();
    let mut gradsq_file = "gradsq".to_string();
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
            "-vector-size" => {
                vector_size = it.peek().and_then(|v| v.parse().ok())
                    .expect("-vector-size <int>");
                is_skip = true;
            },
            "-iter" => {
                n_iter = it.peek().and_then(|v| v.parse().ok())
                    .expect("-iter <int>");
                is_skip = true;
            },
            "-threads" => {
                n_threads = it.peek().and_then(|v| v.parse().ok())
                    .expect("-threads <int>");
                is_skip = true;
            },
            "-alpha" => {
                alpha = it.peek().and_then(|v| v.parse().ok())
                    .expect("-alpha <float>");
                is_skip = true;
            },
            "-x-max" => {
                x_max = it.peek().and_then(|v| v.parse().ok())
                    .expect("-x-max <float>");
                is_skip = true;
            },
            "-eta" => {
                eta = it.peek().and_then(|v| v.parse().ok())
                    .expect("-eta <float>");
                is_skip = true;
            },
            "-binary" => {
                binary = it.peek().and_then(|v| v.parse().ok())
                    .and_then(|x| if 0 <= x && x <= 2 { Some(x) } else { None })
                    .expect("-binary <int>, 0: text (default), 1: binary, 2: both");
                is_skip = true;
            },
            "-model" => {
                model = it.peek().and_then(|v| v.parse().ok())
                    .and_then(|x| if 0 <= x && x <= 2 { Some(x) } else { None })
                    .expect("-model <int>, 0: All, 1: Word vectors, 2: Word + context word vectors (default)");
                is_skip = true;
            },
            "-save-gradsq" => {
                save_gradsq |= it.peek().and_then(|v| v.parse::<i32>().ok())
                    .expect("-save-gradsq <int>") != 0;
                is_skip = true;
            },
            "-vocab-file" => {
                vocab_file = it.peek().expect("-vocab-file <file>").to_string();
                is_skip = true;
            },
            "-save-file" => {
                save_file = it.peek().expect("-save-file <file>").to_string();
                is_skip = true;
            },
            "-gradsq-file" => {
                gradsq_file = it.peek().expect("-gradsq-file <file>").to_string();
                save_gradsq = true;
                is_skip = true;
            },
            "-input-file" => {
                input_file = it.peek().expect("-input-file <file>").to_string();
                is_skip = true;
            },
            "-checkpoint-every" => {
                checkpoint_every = it.peek().and_then(|v| v.parse::<i32>().ok())
                    .expect("-checkpoint-every <int>") != 0;
                is_skip = true;
            },
            &_ => {},
        }
    }
    std::process::exit(train_glove());
}
