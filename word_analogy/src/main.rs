extern crate clap;

use std::{cmp, f64, fs, io};
use std::collections::HashMap;
use std::io::{BufRead, Error, ErrorKind};
use clap::{App, Arg};

macro_rules! some {
    ( $e : expr ) => { match $e { Some(x) => x, None => return None } }
}

fn generate(vectors_file: &str) -> Result<HashMap<String, Vec<f64>>, Error> {
    let mut map = HashMap::<String, Vec<f64>>::new();
    let f = io::BufReader::new(fs::File::open(vectors_file)?);
    for line in f.lines() {
        let line = line?;
        let s = line.trim();
        if s.len() == 0 { break; }
        let mut it = s.split_whitespace();
        let w = it.next().ok_or(Error::new(ErrorKind::InvalidData, "Bad line"))?;
        let mut v = Vec::<f64>::new();
        for x in it {
            match x.parse::<f64>() {
                Ok(val) => v.push(val),
                Err(e) => return Err(Error::new(ErrorKind::InvalidData, e)),
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in v.iter_mut() { *x /= norm; }
        map.insert(w.to_string(), v);
    }
    // normalize: pending
    Ok(map)
}

fn word_analogy<'a>(map: &'a HashMap<String, Vec<f64>>, w1: &str, w2: &str, w3: &str) -> Option<Vec<(&'a str, f64)>> {
    let mut vec = some!(map.get(w2)).clone();
    let it2 = some!(map.get(w1)).iter();
    for (x, y) in vec.iter_mut().zip(it2) { *x -= *y; }
    let it3 = some!(map.get(w3)).iter();
    for (x, y) in vec.iter_mut().zip(it3) { *x += *y; }
    let norm = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in vec.iter_mut() { *x /= norm; }
    let mut wv: Vec<(&'a str, f64)> = map.iter().map(|(w, v)| {
        if w != w1 && w != w2 && w != w3 {
            // inner product
            (w.as_str(), vec.iter().zip(v).map(|(x, y)| x * y).sum())
        }
        else { (w.as_str(), f64::NEG_INFINITY) }
    }).collect();
    wv.sort_by(|x,y| (y.1).partial_cmp(&x.1).unwrap_or(cmp::Ordering::Less));
    Some(wv)
}

fn main() {
    let matches = App::new("distance")
        .about("Get distance between 2 vectors.")
        .arg(Arg::with_name("vectors_file")
             .default_value("vectors.txt"))
        .get_matches();

    println!("vectors_file: {}", matches.value_of("vectors_file").unwrap());
    let word_vector = generate(matches.value_of("vectors_file").unwrap()).unwrap();
    println!("Enter 3 words (EXIT to break)");
    loop {
        let mut line = String::new();
        if io::stdin().read_line(&mut line).is_err() { break; }
        let line = line.trim().to_string();
        if line == "EXIT" { break; }
        let words: Vec<_> = line.split_whitespace().collect();
        if words.len() == 3 {
            let v = word_analogy(&word_vector, words[0], words[1], words[2]).unwrap_or(vec![]);
            for (x, _) in v.iter().zip(0 .. 10) {
                println!("{:?}", x);
            }
        }
        println!("Enter 3 words (EXIT to break)");
    }
}
