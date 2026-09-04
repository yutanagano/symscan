// TODO: support locus field
// TODO: support using V/J calls

use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::process;

use clap::{ArgAction, Parser};
use rayon::ThreadPoolBuilder;
use thiserror::Error;

use crate::parsing::AirrData;

mod analysis;
mod parsing;
mod serialization;

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

fn main() -> Result<(), Error> {
    let args = Args::parse();

    ThreadPoolBuilder::new()
        .num_threads(args.num_threads)
        .build_global()?;

    let data_query = match args.file_query.as_ref() {
        Some(path) => {
            let reader = get_file_bufreader(path);
            parsing::parse_airr_tsv(reader).map_err(|e| Error::Parsing {
                input_name: path.to_string(),
                error: e,
            })?
        }
        None => {
            let stdin = io::stdin().lock();
            parsing::parse_airr_tsv(stdin).map_err(|e| Error::Parsing {
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
            let ref_reader = get_file_bufreader(ref_path);
            let data_ref = parsing::parse_airr_tsv(ref_reader).map_err(|e| Error::Parsing {
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

/// Get a buffered reader to a file at path.
fn get_file_bufreader(path: &str) -> BufReader<File> {
    let file = File::open(path).unwrap_or_else(|e| {
        eprintln!("failed to open {}: {}", path, e);
        process::exit(1)
    });
    BufReader::new(file)
}
