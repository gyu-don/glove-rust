extern crate rand;
extern crate time;
extern crate crossbeam;
extern crate libc;

#[link(name="m")]
extern {
    fn sqrt(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn pow(x: f64, x: f64) -> f64;
}

use std::{env, f64, fs, io, mem, ops, slice};
use std::io::{BufRead, BufReader, BufWriter, ErrorKind, Read, Seek, Write};
use std::marker::PhantomData;
use rand::Rng;

#[derive(Copy, Clone, Debug)]
struct CooccurRec {
    word1: u32,
    word2: u32,
    val: f64,
}

static USE_UNK_VEC: bool = true;

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

#[allow(dead_code)]
#[derive(Clone, Copy)]
struct UnsafeSlice<'a> {
    p: *mut f64,
    len: usize,
    phantom: PhantomData<&'a Vec<f64>>,
}
unsafe impl<'a> Send for UnsafeSlice<'a> {}
unsafe impl<'a> Sync for UnsafeSlice<'a> {}
#[allow(dead_code)]
impl<'a> UnsafeSlice<'a> {
    fn new(v: &mut Vec<f64>) -> UnsafeSlice {
        UnsafeSlice { p: v.as_mut_ptr(), len: v.len(), phantom: PhantomData }
    }
    #[inline]
    unsafe fn get(self, idx: usize) -> Option<&'a f64> {
        slice::from_raw_parts(self.p, self.len).get(idx)
    }
    #[inline]
    unsafe fn get_slice(self, idx: ops::Range<usize>) -> &'a[f64] {
        &slice::from_raw_parts(self.p, self.len)[idx]
    }
    #[inline]
    unsafe fn get_mut(self, idx: usize) -> Option<&'a mut f64> {
        slice::from_raw_parts_mut(self.p, self.len).get_mut(idx)
    }
    #[inline]
    unsafe fn get_slice_mut(self, idx: ops::Range<usize>) -> &'a mut [f64] {
        &mut slice::from_raw_parts_mut(self.p, self.len)[idx]
    }
    #[inline]
    unsafe fn borrow(self) -> &'a [f64] {
        slice::from_raw_parts(self.p, self.len)
    }
    #[inline]
    unsafe fn borrow_mut(self) -> &'a mut [f64] {
        slice::from_raw_parts_mut(self.p, self.len)
    }
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

fn initialize_parameters(w: &mut Vec<f64>, gradsq: &mut Vec<f64>,
                         vector_size: usize, vocab_size: usize) {
    // Allocate space for word vectors and context word vectors, and corresponding gradsq
    w.reserve(2 * vocab_size * (vector_size + 1));
    gradsq.reserve(2 * vocab_size * (vector_size + 1));

    //let mut rng = rand::thread_rng();
    for _ in 0 .. 2 * vocab_size * (vector_size + 1) {
        //w.push((rng.next_f64() - 0.5) / (vector_size + 1) as f64);
        w.push((unsafe{libc::rand() as f64} / (libc::RAND_MAX as f64) - 0.5) / (vector_size + 1) as f64);
        gradsq.push(1.0);
    }
}

// Train the Glove model
fn glove_thread(w: UnsafeSlice, gradsq: UnsafeSlice,
                alpha: f64, eta: f64, x_max: f64,
                input_file: &str, vector_size: usize, vocab_size: usize,
                start: usize, end: usize) -> f64 {
    let mut cost = 0f64;

    let mut fin = BufReader::new(fs::File::open(&*input_file).unwrap());
    fin.seek(io::SeekFrom::Start((start * mem::size_of::<CooccurRec>()) as u64)).unwrap();

    let mut w_updates1 = vec![0f64 ; vector_size];
    let mut w_updates2 = vec![0f64 ; vector_size];

    for _ in start .. end {
        let mut cr: CooccurRec = unsafe { mem::uninitialized() };
        fin.read_exact(unsafe {
            slice::from_raw_parts_mut((&mut cr as *mut CooccurRec) as *mut u8,
            mem::size_of::<CooccurRec>()) }).unwrap();
        if cr.word1 < 1 || cr.word2 < 1 { continue; }

        let l1 = (cr.word1 as usize - 1) * (vector_size + 1);
        let l2 = (cr.word2 as usize - 1 + vocab_size) * (vector_size + 1);
        //if start==37916540 && i > end-6 { log!(-1, "{:?}", cr); }
        let w1 = unsafe { w.get_slice_mut(l1 .. l1 + vector_size) };
        let w2 = unsafe { w.get_slice_mut(l2 .. l2 + vector_size) };
        let b1 = unsafe { w.get_mut(l1 + vector_size).unwrap() };
        let b2 = unsafe { w.get_mut(l2 + vector_size).unwrap() };
        //let diff = w1.iter().zip(w2.iter()).fold(0.0, |a, (x, y)| a + x * y) + *b1 + *b2 - cr.val.ln();
        //let mut fdiff = if cr.val > x_max { diff } else { (cr.val / x_max).powf(alpha) * diff };
        let diff: f64 = w1.iter().zip(w2.iter()).map(|(x, y)| x * y).sum::<f64>() + *b1 + *b2 - unsafe{log(cr.val)};
        let mut fdiff = if cr.val > x_max { diff } else { diff * (unsafe{pow(cr.val / x_max, alpha)}) };
        //if start == 0 && cr.word1 % 1000 == 1 && cr.word2 % 1000 == 2 { log!(-1, "w1: {}, w2: {}, val: {}, diff: {}, fdiff: {}", cr.word1, cr.word2, cr.val, diff, fdiff); }
        if !diff.is_finite() || !fdiff.is_finite() {
            progress!(-1, "Caught NaN in diff for kdiff for thread. Skipping update");
            continue;
        }
        cost += 0.5 * fdiff * diff;  // weighted squared error

        // Adaptive gradient updates
        let gradsq1 = unsafe { gradsq.get_slice_mut(l1 .. l1 + vector_size) };
        let gradsq2 = unsafe { gradsq.get_slice_mut(l2 .. l2 + vector_size) };
        let gradsq1_b = unsafe { gradsq.get_mut(l1 + vector_size).unwrap() };
        let gradsq2_b = unsafe { gradsq.get_mut(l2 + vector_size).unwrap() };
        fdiff *= eta;  // for ease in calculating gradient
        {
            let mut w1_ = w1.iter();
            let mut w2_ = w2.iter();
            let mut gsq1_ = gradsq1.iter_mut();
            let mut gsq2_ = gradsq2.iter_mut();
            let mut w_updates1_ = w_updates1.iter_mut();
            let mut w_updates2_ = w_updates2.iter_mut();
            for _ in 0 .. vector_size {
                let w1 = w1_.next().unwrap();
                let w2 = w2_.next().unwrap();
                let gsq1 = gsq1_.next().unwrap();
                let gsq2 = gsq2_.next().unwrap();
                let wup1 = w_updates1_.next().unwrap();
                let wup2 = w_updates2_.next().unwrap();

                let temp1 = fdiff * w2;
                let temp2 = fdiff * w1;
                //*wup1 = temp1 / gsq1.sqrt();
                //*wup2 = temp2 / gsq2.sqrt();
                *wup1 = temp1 / unsafe{sqrt(*gsq1)};
                *wup2 = temp2 / unsafe{sqrt(*gsq2)};
                *gsq1 += temp1 * temp1;
                *gsq2 += temp2 * temp2;
            }
        }
        if w_updates1.iter().sum::<f64>().is_finite() &&
           w_updates2.iter().sum::<f64>().is_finite() {
            for (x, y) in w1.iter_mut().zip(w_updates1.iter()) { *x -= *y; }
            for (x, y) in w2.iter_mut().zip(w_updates2.iter()) { *x -= *y; }
        }
        // updates for bias terms
        let check_nan = |x: f64| if !x.is_finite() {
            progress!(-1, "\ncaught in NaN in update"); 0f64
        } else { x };
        //*b1 -= check_nan(fdiff / gradsq1_b.sqrt());
        //*b2 -= check_nan(fdiff / gradsq2_b.sqrt());
        *b1 -= check_nan(fdiff / unsafe{sqrt(*gradsq1_b)});
        *b2 -= check_nan(fdiff / unsafe{sqrt(*gradsq2_b)});
        fdiff *= fdiff;
        *gradsq1_b += fdiff;
        *gradsq2_b += fdiff;
        if start == 0 && cr.word1 % 1000 == 1 && cr.word2 % 1000 == 2 { for x in w2.iter() { progress!(-1, "{:.6} ", *x); }log!(-1, "{:.6}", *b2);}
    }
    cost
}

fn save_params_bin(w: &[f64], save_file: &str, nb_iter: usize) -> i32 {
    let output_file: String;
    if nb_iter <= 0 {
        output_file = format!("{}.bin", save_file);
    } else {
        output_file = format!("{}.{:>03}.bin", save_file, nb_iter);
    }
    let mut fout = BufWriter::new(fs::File::create(output_file).unwrap());
    for x in w.iter() {
        fout.write(unsafe {
            slice::from_raw_parts((x as *const f64) as *const u8, 8)
        }).unwrap();
    }
    0
}

fn save_params_txt(w: &[f64], save_file: &str, vocab_file: &str,
                   vector_size: usize, vocab_size: usize,
                   nb_iter: usize, model: i32) -> i32 {
    let output_file: String;
    if nb_iter <= 0 {
        output_file = format!("{}.txt", save_file);
    } else {
        output_file = format!("{}.{:>03}.txt", save_file, nb_iter);
    }
    let mut fout = BufWriter::new(fs::File::create(output_file).unwrap());
    let mut fid = BufReader::new(fs::File::open(vocab_file).unwrap());
    for a in 0 .. vocab_size {
        let word = match get_word(&mut fid) {
            GetWord::Word(w) => w,
            GetWord::LastWord(w) => w,
            _ => { return 1; }
        };
        if word == "<unk>" { return 1; }
        fout.write(word.as_bytes()).unwrap();
        match model {
            0 => {
                for b in 0 .. vector_size + 1 {
                    fout.write_fmt(format_args!(" {:.5}", w[a * (vector_size + 1) + b])).unwrap();
                }
                for b in 0 .. vector_size + 1 {
                    fout.write_fmt(format_args!(" {:.5}", w[(vocab_size + a) * (vector_size + 1) + b])).unwrap();
                }
            },
            1 => {
                for b in 0 .. vector_size {
                    fout.write_fmt(format_args!(" {:.5}", w[a * (vector_size + 1) + b])).unwrap();
                }
            },
            2 => {
                for b in 0 .. vector_size {
                    fout.write_fmt(format_args!(" {:.5}", w[a * (vector_size + 1) + b] + w[(vocab_size + a) * (vector_size + 1) + b])).unwrap();
                }
            },
            _ => unreachable!()
        }
        fout.write(b"\n").unwrap();
        match get_word(&mut fid) {
            GetWord::Word(_) => {},
            GetWord::LastWord(_) => {},
            _ => { return 1; }
        };
    }
    if USE_UNK_VEC {
        let mut unk_vec = vec![0f64 ; vector_size + 1];
        let mut unk_context = vec![0f64 ; vector_size + 1];
        let word = "<unk>";

        let num_rare_words = if vocab_size < 100 { vocab_size } else { 100 };
        for a in vocab_size - num_rare_words .. vocab_size {
            for (x, y) in unk_vec.iter_mut().zip(w[a * (vector_size + 1)..].iter()) {
                *x += y / num_rare_words as f64;
            }
            for (x, y) in unk_context.iter_mut().zip(w[(vocab_size + a) * (vector_size + 1)..].iter()) {
                *x += y / num_rare_words as f64;
            }
        }
        fout.write(word.as_bytes()).unwrap();
        match model {
            0 => {
                for x in unk_vec.iter() { fout.write_fmt(format_args!(" {:.5}", x)).unwrap(); }
                for x in unk_context.iter() { fout.write_fmt(format_args!(" {:.5}", x)).unwrap(); }
            },
            1 => {
                for x in unk_vec[..vector_size].iter() {
                    fout.write_fmt(format_args!(" {:.5}", x)).unwrap();
                }
            },
            2 => {
                for (x, y) in unk_vec[..vector_size].iter().zip(unk_context.iter()) {
                    fout.write_fmt(format_args!(" {:.5}", x + y)).unwrap();
                }
            },
            _ => unreachable!()
        }
    }
    0
}

fn save_gsq_bin(gradsq: &[f64], save_file: &str, nb_iter: usize) -> i32 {
    let output_file_gsq: String;
    if nb_iter <= 0 {
        output_file_gsq = format!("{}.bin", save_file);
    } else {
        output_file_gsq = format!("{}.{:>03}.bin", save_file, nb_iter);
    }
    let mut fgs = BufWriter::new(fs::File::create(output_file_gsq).unwrap());
    for x in gradsq.iter() {
        fgs.write(unsafe {
            slice::from_raw_parts((x as *const f64) as *const u8, 8)
        }).unwrap();
    }
    0
}

fn save_gsq_txt(gradsq: &[f64], save_file: &str, vocab_file: &str,
                vector_size: usize, vocab_size: usize, nb_iter: usize) -> i32 {
    let output_file_gsq: String;
    if nb_iter <= 0 {
        output_file_gsq = format!("{}.txt", save_file);
    } else {
        output_file_gsq = format!("{}.{:>03}.txt", save_file, nb_iter);
    }
    let mut fgs = BufWriter::new(fs::File::create(output_file_gsq).unwrap());
    let mut fid = BufReader::new(fs::File::open(vocab_file).unwrap());
    for a in 0 .. vocab_size {
        let word = match get_word(&mut fid) {
            GetWord::Word(w) => w,
            GetWord::LastWord(w) => w,
            _ => { return 1; }
        };
        if word == "<unk>" { return 1; }
        fgs.write(word.as_bytes()).unwrap();
        for x in gradsq[a * (vector_size + 1) .. (a + 1) * (vector_size + 1)].iter() {
            fgs.write_fmt(format_args!(" {.5}", x)).unwrap();
        }
        for x in gradsq[(vocab_size + a) * (vector_size + 1) .. (vocab_size + a + 1) * (vector_size + 1)].iter() {
            fgs.write_fmt(format_args!(" {.5}", x)).unwrap();
        }
    }
    0
}

fn train_glove(vector_size: usize, n_threads: usize, n_iter: usize,
               alpha: f64, x_max: f64, eta: f64,
               binary: i32, model: i32, save_gradsq: bool, checkpoint_every: usize,
               vocab_file: &str, input_file: &str, save_file: &str, gradsq_file: &str) -> i32 {
    log!(-1, "TRAINING MODEL");
    let num_lines = fs::metadata(input_file).unwrap().len() as usize / mem::size_of::<CooccurRec>();
    log!(-1, "Read {} lines.", num_lines);

    progress!(1, "Initializing parameters...");
    let mut w: Vec<f64> = vec![];
    let mut gradsq: Vec<f64> = vec![];
    let vocab_size = BufReader::new(fs::File::open(vocab_file).unwrap()).lines().count();
    initialize_parameters(&mut w, &mut gradsq, vector_size, vocab_size);
    log!(1, "done.");
    log!(0, "vector size: {}", vector_size);
    log!(0, "vocab size: {}", vocab_size);
    log!(0, "x_max: {}", x_max);
    log!(0, "alpha: {}", alpha);
    let input_file = input_file.to_string();
    for i in 0 .. n_iter {
        {
            let w_slice = UnsafeSlice::new(&mut w);
            let gradsq_slice = UnsafeSlice::new(&mut gradsq);
            crossbeam::scope(|scope| {
                let threads: Vec<_> = (0 .. n_threads).map(|j| {
                    let start = num_lines / n_threads * j;
                    let end = if j != n_threads - 1 { num_lines / n_threads * (j + 1) } else { num_lines };
                    let input_file = &input_file;
                    scope.spawn(move || {
                        glove_thread(
                            w_slice, gradsq_slice, alpha, eta, x_max,
                            input_file, vector_size, vocab_size, start, end
                        )
                    })
                }).collect();
                let total_cost = threads.into_iter().map(|e| e.join()).fold(0f64, |a, b| a + b);
                log!(-1, "{}, iter: {:>03}, cost: {}", time::strftime("%x - %I:%M.%S%p", &time::now()).unwrap(), i+1, total_cost / num_lines as f64);
            });
        }
        if checkpoint_every > 0 && (i + 1) % checkpoint_every == 0 {
            progress!(-1, "    saving intermediate parameters for iter {:>03}...", i+1);
            if binary > 0 {
                save_params_bin(&w, save_file, i + 1);
                if save_gradsq {
                    save_gsq_bin(&gradsq, gradsq_file, i + 1);
                }
            }
            if binary != 1 {
                save_params_txt(&w, save_file, vocab_file, vector_size, vocab_size, i + 1, model);
                if save_gradsq {
                    save_gsq_txt(&gradsq, gradsq_file, vocab_file, vector_size, vocab_size, i + 1);
                }
            }
        }
    }
    let mut retval = 0i32;
    if binary > 0 {
        retval |= save_params_bin(&w, save_file, 0);
        if save_gradsq {
            retval |= save_gsq_bin(&gradsq, gradsq_file, 0);
        }
    }
    if binary != 1 {
        retval |= save_params_txt(&w, save_file, vocab_file, vector_size, vocab_size, 0, model);
        if save_gradsq {
            retval |= save_gsq_txt(&gradsq, gradsq_file, vocab_file, vector_size, vocab_size, 00000000);
        }
    }
    retval
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
    let mut checkpoint_every = 0usize;
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
                checkpoint_every = it.peek().and_then(|v| v.parse().ok())
                    .expect("-checkpoint-every <int>");
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
