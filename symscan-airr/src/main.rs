// TODO: support locus field
// TODO: support using V/J calls

use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter};

use clap::{ArgAction, Parser};
use rayon::ThreadPoolBuilder;
use thiserror::Error;

use crate::parsing::AirrData;

mod analysis;
mod parsing;
mod serialization;

/// CLI tool for fast comparison between different adaptive immune receptor repertoires (AIRRs),
/// powered by the symscan algorithm.
///
/// You can provide symscan-airr with an AIRR-compliant TSV containing AIR data from repertoires of
/// interest. The tool will then compute the level of overlap between all pairs of input
/// repertoires. Overlap is defined as the total number, accounting for duplicate count, of AIR
/// pairs between the repertoires that fall within the target similarity threshold. If you provide
/// the program with a path to a [FILE_QUERY], it will read its contents for input. Otherwise, it
/// will read from standard input until reaching an EOF signal.
///
/// If you provide the program with both [FILE_QUERY] and [FILE_REFERENCE], then it will compute the
/// overlap scores between repertoires across the two files. Repertoires from the same file will not
/// be compared to one another.
///
/// The output is a TSV where every row represets a pair of repertoires. The first two columns
/// contain the names of two repertoires, and the third column contains the overlap quantity between
/// them.
#[derive(Debug, Parser)]
#[command(version)]
struct Args {
    /// The maximum edit distance away to check for neighbours.
    #[arg(short = 'd', long, default_value_t = 1)]
    max_distance: u8,

    /// Limit the neighbour search to only consider substitutions (i.e. use Hamming distance).
    #[arg(long, action = ArgAction::SetTrue)]
    hamming: bool,

    /// Use the amino acid sequence from the cdr3_aa column.
    ///
    /// By default, the program uses the junction_aa column.
    #[arg(long, action = ArgAction::SetTrue)]
    cdr3: bool,

    /// The number of OS threads the program spawns for computations (if 0 spawns one thread per CPU core).
    #[arg(short, long, default_value_t = 0)]
    num_threads: usize,

    /// Path to input AIRR-compliant TSV (if absent program reads from stdin until EOF).
    file_query: Option<String>,

    /// If provided, compares repertoires in the reference file against the repertoires in the query
    /// file.
    ///
    /// This must also be a path to an AIRR-compliant TSV.
    file_reference: Option<String>,
}

#[derive(Debug, Error)]
enum Error {
    #[error(transparent)]
    ThreadPool(#[from] rayon::ThreadPoolBuildError),

    #[error("while parsing from {input_name}, got the following error:\n{error}")]
    Parsing {
        input_name: String,
        error: parsing::Error,
    },

    #[error(transparent)]
    Processing(#[from] symscan::Error),

    #[error(transparent)]
    Io(#[from] io::Error),
}

struct FileReaderWithSizeHint {
    reader: BufReader<File>,
    num_rows_hint: Option<u32>,
}

fn main() -> Result<(), Error> {
    let args = Args::parse();

    ThreadPoolBuilder::new()
        .num_threads(args.num_threads)
        .build_global()?;

    let data_query = match args.file_query.as_ref() {
        Some(path) => {
            let reader_with_hint = get_file_reader_with_size_hint(path)?;
            parsing::parse_airr_tsv(
                reader_with_hint.reader,
                args.cdr3,
                reader_with_hint.num_rows_hint,
            )
            .map_err(|e| Error::Parsing {
                input_name: path.to_string(),
                error: e,
            })?
        }
        None => {
            let stdin = io::stdin().lock();
            parsing::parse_airr_tsv(stdin, args.cdr3, None).map_err(|e| Error::Parsing {
                input_name: "stdin".to_string(),
                error: e,
            })?
        }
    };

    if let Some(ref_path) = &args.file_reference {
        if ref_path
            != args
                .file_query
                .as_ref()
                .expect("query file must be specified if reference file is specified")
        {
            let reader_ref = get_file_reader_with_size_hint(ref_path)?;
            let data_ref =
                parsing::parse_airr_tsv(reader_ref.reader, args.cdr3, reader_ref.num_rows_hint)
                    .map_err(|e| Error::Parsing {
                        input_name: ref_path.to_string(),
                        error: e,
                    })?;

            return run_analysis_across(&data_query, &data_ref, &args);
        }
    };

    run_analysis_within(&data_query, &args)
}

fn run_analysis_within(data: &AirrData, args: &Args) -> Result<(), Error> {
    let overlap_matrix =
        analysis::compute_overlap_matrix_within(data, args.max_distance, args.hamming)?;

    let mut writer = BufWriter::new(io::stdout().lock());
    serialization::om_as_triplet_tsv_upper(&overlap_matrix, data, &mut writer)?;

    Ok(())
}

fn run_analysis_across(
    data_query: &AirrData,
    data_ref: &AirrData,
    args: &Args,
) -> Result<(), Error> {
    let overlap_matrix = analysis::compute_overlap_matrix_across(
        data_query,
        data_ref,
        args.max_distance,
        args.hamming,
    )?;

    let mut writer = BufWriter::new(io::stdout().lock());
    serialization::om_as_triplet_tsv_full(&overlap_matrix, data_query, data_ref, &mut writer)?;

    Ok(())
}

/// Get a buffered reader to a file at path, with a hint on the number of rows.
fn get_file_reader_with_size_hint(path: &str) -> io::Result<FileReaderWithSizeHint> {
    let file = File::open(path)?;
    let num_bytes_in_file = file.metadata().ok().map(|m| m.len());
    let mut reader = BufReader::with_capacity(1 << 16, file);
    let num_rows_hint = get_num_rows_hint(&mut reader, num_bytes_in_file)?;

    Ok(FileReaderWithSizeHint {
        reader,
        num_rows_hint,
    })
}

fn get_num_rows_hint(
    reader: &mut BufReader<File>,
    num_bytes_in_file: Option<u64>,
) -> io::Result<Option<u32>> {
    match num_bytes_in_file {
        None => Ok(None),
        Some(num_bytes_in_file) => {
            let buf = reader.fill_buf()?;

            let Some(last_nl_pos) = buf.iter().rposition(|&b| b == b'\n') else {
                return Ok(None);
            };
            let buf = &buf[0..=last_nl_pos];

            let num_lines_in_preview = buf.iter().filter(|&&b| b == b'\n').count();
            if num_lines_in_preview == 0 {
                return Ok(None);
            }

            let bytes_per_line = buf.len() as f64 / num_lines_in_preview as f64;
            let estimate = num_bytes_in_file as f64 / bytes_per_line;

            Ok(Some(estimate as u32))
        }
    }
}
