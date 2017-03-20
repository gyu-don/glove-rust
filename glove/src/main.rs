extern crate rand;

use std::{env, f64, fs, io, mem, slice};
use std::io::{BufRead, ErrorKind, Read, Seek, Write};
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

fn initialize_parameters(w: &mut Vec<f64>, gradsq: &mut Vec<f64>,
                         vector_size: usize, vocab_size: usize,
                         input_file: &str) {
    // Allocate space for word vectors and context word vectors, and corresponding gradsq
    w.reserve(2 * vocab_size * (vector_size + 2));
    gradsq.reserve(2 * vocab_size * (vector_size + 2));

    let mut rng = rand::thread_rng();
    for _ in 0 .. 2 * vocab_size * (vector_size + 1) {
        w.push((rng.next_f64() - 0.5) / vector_size as f64);
        gradsq.push(1.0);
    }
}

// Train the Glove model
fn glove_thread(w: &mut Vec<f64>, gradsq: &mut Vec<f64>,
                alpha: f64, eta: f64, x_max: f64,
                input_file: &str, vector_size: usize, vocab_size: usize,
                start: usize, end: usize) {
    let mut cost = 0f64;

    let mut fin = fs::File::open(input_file).unwrap();
    fin.seek(io::SeekFrom::Start(start * mem::size_of::<CooccurRec>()));
    for _ in start .. end {
        let mut cr: CooccurRec = mem::uninitialized();
        fin.read_exact(unsafe {
            slice::from_raw_parts_mut((&mut cr as *mut CooccurRec) as *mut u8,
            mem::size_of::<CooccurRec>())}).unwrap();
        if cr.word1 < 1 || cr.word2 < 1 { continue; }

        let l1: usize = (cr.word1 as usize - 1) * (vector_size + 1);
        let l2: usize = (cr.word2 as usize - 1 + vocab_size) * (vector_size + 1);
        let diff = w[l1 .. l1 + vector_size].zip(w[l2 .. l2 + vector_size]).fold(0.0, |a, &x| a + x.0 * x.1) + w[vector_size + l1] + w[vector_size + l2] - cr.val.ln();
        let fdiff = if cr.val > x_max { diff } else { (cr.val / x_max).powf(alpha) * diff };
        if !diff.is_finite() || !fdiff.is_finite() {
            progress!(-1, "Caught NaN in diff for kdiff for thread. Skipping update");
            continue;
        }
        cost += 0.5 * fdiff * diff;  // weighted squared error

        // Adaptive gradient updates
        fdiff *= eta;  // for ease in calculating gradient
        let w_updates1 = w[l2 .. l2 + vector_size].iter().zip(gradsq[l1..].iter()).map(|a, x| fdiff * x.0 / x.1.sqrt()).collect();
        let w_updates2 = w[l1 .. l1 + vector_size].iter().zip(gradsq[l2..].iter()).map(|a, x| fdiff * x.0 / x.1.sqrt()).collect();
        // gradsq += ... 
        if w_updates1.iter().fold(0f64, |a, x| a + x).is_finite() &&
           w_updates2.iter().fold(0f64, |a, x| a + x).is_finite() {
            for (&mut x, y) in w[l1..].iter_mut().zip(w_updates1.iter()) { x -= y; }
            for (&mut x, y) in w[l2..].iter_mut().zip(w_updates2.iter()) { x -= y; }
        }
        // updates for bias terms
        let check_nan = |x| if !x.is_finite() { progress!(-1, "\ncaught in NaN in update"); 0f64 }
                            else { x };
        w[vector_size + l1] -= check_nan(fdiff / gradsq[vector_size + l1].sqrt());
        w[vector_size + l2] -= check_nan(fdiff / gradsq[vector_size + l2].sqrt());
        fdiff *= fdiff;
        gradsq[vector_size + l1] += fdiff;
        gradsq[vector_size + l2] += fdiff;
    }
}

fn train_glove(vector_size: usize, n_threads: usize, n_iter: usize,
               alpha: f64, x_max: f64, eta: f64,
               binary: i32, model: i32, save_gradsq: bool, checkpoint_every: bool,
               vocab_file: &str, input_file: &str, save_file: &str, gradsq_file: &str) -> i32 {
    log!(-1, "TRAINING MODEL");
    let num_lines = fs::metadata(input_file).unwrap().len() as usize / mem::size_of::<CooccurRec>();
    log!(-1, "Read {} lines.", num_lines);

    log!(1, "Initializing parameters...");
    let mut w: Vec<f64> = vec![];
    let mut gradsq: Vec<f64> = vec![];
    let vocab_size = io::BufReader::new(fs::File::open(vocab_file).unwrap()).lines().count();
    initialize_parameters(&mut w, &mut gradsq, vector_size, vocab_size, input_file);
    log!(1, "done.");
    log!(0, "vector size: {}", vector_size);
    log!(0, "vocab size: {}", vocab_size);
    log!(0, "x_max: {}", x_max);
    log!(0, "alpha: {}", alpha);
    // スレ立ち上げ&join, cost回収
    0
}

fn main() {
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
    std::process::exit(
        train_glove(vector_size, n_threads, n_iter,
                    alpha, x_max, eta,
                    binary, model, save_gradsq, checkpoint_every,
                    &vocab_file, &input_file, &save_file, &gradsq_file));
}
