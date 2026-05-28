use clap::{ArgAction, Parser};
use rayon::ThreadPoolBuilder;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Error, ErrorKind::InvalidData, Write};
use std::process;
use symscan::{
    get_hamming_neighbors_across, get_hamming_neighbors_within, get_neighbors_across,
    get_neighbors_within, NeighborPairs,
};

/// Minimal CLI utility for fast discovery of nearest neighbour strings that fall within a
/// threshold edit distance.
///
/// If you provide symscan with a path to a [FILE_QUERY], it will read its contents for input. If
/// no path is supplied, symscan will read from the standard input until it receives an EOF signal.
/// Symscan will then look for pairs of similar strings within its input, where each line of text
/// is treated as an individual string. You can also supply symscan with two paths -- a
/// [FILE_QUERY] and [FILE_REFERENCE], in which case the program will look for pairs of similar
/// strings across the contents of the two files. Currently, only valid ASCII input is supported.
///
/// By default, the threshold (Levenshtein) edit distance at or below which a pair of strings are
/// considered similar is set at 1. This can be changed by setting the --max-distance option. You
/// can also limit the neighbor search to only consider substitutions (i.e. no insertions /
/// deletions) by setting the --hamming option.
///
/// Symscan's output is plain text, where each line encodes a detected pair of similar input
/// strings. Each line is comprised of three integers separated by commas, which represent, in
/// respective order: the (1-indexed) line number of the string from the primary input (i.e. stdin
/// or [FILE_QUERY]), the (1-indexed) line number of the string from the secondary input (i.e.
/// stdin or [FILE_QUERY] if one input, or [FILE_REFERENCE] if two inputs), and the (Levenshtein)
/// edit distance between the similar strings.
#[derive(Debug, Parser)]
#[command(version)]
struct Args {
    /// The maximum edit distance away to check for neighbours.
    #[arg(short = 'd', long, default_value_t = 1)]
    max_distance: u8,

    /// Limit the neighbour search to only consider substitutions (i.e. use Hamming distance).
    #[arg(long, action = ArgAction::SetTrue)]
    hamming: bool,

    /// The number of OS threads the program spawns (if 0 spawns one thread per CPU core).
    #[arg(short, long, default_value_t = 0)]
    num_threads: usize,

    /// 0-index line numbers in the output.
    #[arg(short, long, action = ArgAction::SetTrue)]
    zero_index: bool,

    /// Primary input (if absent program reads from stdin until EOF).
    file_query: Option<String>,

    /// If provided, searches for pairs of similar strings between the query file and the reference
    /// file.
    file_reference: Option<String>,
}

/// Reads (blocking) all lines from in_stream until EOF, and converts the data into a vector of
/// Strings where each String is a line from in_stream. Performs symdel to look for String
/// pairs within <MAX_DISTANCE> (as read from the CLI arguments, defaults to 1) edit distance.
/// Outputs the detected pairs from symdel into out_stream, where each new line written encodes a
/// detected pair as a pair of 1-indexed line numbers of the input strings involved separated by a
/// comma, and the lower line number is always first.
fn main() {
    let mut stdout = BufWriter::new(io::stdout().lock());
    let args = Args::parse();

    ThreadPoolBuilder::new()
        .num_threads(args.num_threads)
        .build_global()
        .unwrap_or_else(|_| {
            eprintln!("global thread pool cannot be initialised more than once");
            process::exit(1);
        });

    let query = match args.file_query {
        Some(path) => {
            let reader = get_file_bufreader(&path);
            get_input_lines_as_ascii(reader).unwrap_or_else(|e| {
                eprintln!("(from {}) {}", &path, e);
                process::exit(1);
            })
        }
        None => {
            let stdin = io::stdin().lock();
            get_input_lines_as_ascii(stdin).unwrap_or_else(|e| {
                eprintln!("(from stdin) {}", e);
                process::exit(1);
            })
        }
    };

    match args.file_reference {
        Some(path) => {
            let ref_reader = get_file_bufreader(&path);
            let ref_input = get_input_lines_as_ascii(ref_reader).unwrap_or_else(|e| {
                eprintln!("(from {}) {}", &path, e);
                process::exit(1);
            });

            let hits = if args.hamming {
                get_hamming_neighbors_across(&query, &ref_input, args.max_distance).unwrap_or_else(
                    |e| {
                        eprintln!("{}", e);
                        process::exit(1)
                    },
                )
            } else {
                get_neighbors_across(&query, &ref_input, args.max_distance).unwrap_or_else(|e| {
                    eprintln!("{}", e);
                    process::exit(1)
                })
            };

            write_true_hits(hits, args.zero_index, &mut stdout);
        }
        None => {
            let hits = if args.hamming {
                get_hamming_neighbors_within(&query, args.max_distance).unwrap_or_else(|e| {
                    eprintln!("{}", e);
                    process::exit(1)
                })
            } else {
                get_neighbors_within(&query, args.max_distance).unwrap_or_else(|e| {
                    eprintln!("{}", e);
                    process::exit(1)
                })
            };

            write_true_hits(hits, args.zero_index, &mut stdout);
        }
    };
}

/// Get a buffered reader to a file at path.
fn get_file_bufreader(path: &str) -> BufReader<File> {
    let file = File::open(path).unwrap_or_else(|e| {
        eprintln!("failed to open {}: {}", &path, e);
        process::exit(1)
    });
    BufReader::new(file)
}

/// Read lines from in_stream until EOF and collect into vector of byte vectors. Return any
/// errors if trouble reading, or if the input text contains non-ASCII data. The returned vector
/// is guaranteed to only contain ASCII bytes.
fn get_input_lines_as_ascii(in_stream: impl BufRead) -> Result<Vec<String>, Error> {
    let mut strings = Vec::new();

    for (idx, line) in in_stream.lines().enumerate() {
        let line_unwrapped = line?;

        if !line_unwrapped.is_ascii() {
            let err_msg = format!(
                "non-ASCII data is currently unsupported (\"{}\" from input line {})",
                line_unwrapped,
                idx + 1
            );
            return Err(Error::new(InvalidData, err_msg));
        }

        strings.push(line_unwrapped);
    }

    Ok(strings)
}

/// Write to stdout
fn write_true_hits(hits: NeighborPairs, zero_index: bool, writer: &mut impl Write) {
    for idx in 0..hits.len() {
        if zero_index {
            writeln!(
                writer,
                "{},{},{}",
                hits.row[idx], hits.col[idx], hits.dists[idx]
            )
            .unwrap();
        } else {
            writeln!(
                writer,
                "{},{},{}",
                hits.row[idx] + 1,
                hits.col[idx] + 1,
                hits.dists[idx]
            )
            .unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_input_lines_as_ascii() {
        let strings = get_input_lines_as_ascii(&mut "foo\nbar\nbaz\n".as_bytes())
            .expect("input is valid ASCII");
        let expected: Vec<String> = vec!["foo".into(), "bar".into(), "baz".into()];
        assert_eq!(strings, expected);
    }

    #[test]
    fn test_get_input_lines_as_ascii_rejects_non_ascii() {
        let strings = get_input_lines_as_ascii(&mut "foo\nbar\nバズ\n".as_bytes());
        assert!(strings.is_err());
    }

    #[test]
    fn test_write_true_hits() {
        let cases = [
            (
                NeighborPairs {
                    row: vec![0, 1],
                    col: vec![1, 2],
                    dists: vec![1, 1],
                },
                "0,1,1\n1,2,1\n",
            ),
            (
                NeighborPairs {
                    row: vec![0, 0, 0, 1],
                    col: vec![1, 2, 3, 2],
                    dists: vec![1, 2, 2, 1],
                },
                "0,1,1\n0,2,2\n0,3,2\n1,2,1\n",
            ),
        ];
        let mut test_output_stream = Vec::new();

        for (hits, expected) in cases {
            write_true_hits(hits, true, &mut test_output_stream);
            assert_eq!(test_output_stream, expected.as_bytes());
            test_output_stream.clear();
        }
    }
}
