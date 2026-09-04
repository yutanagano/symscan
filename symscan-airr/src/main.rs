// TODO: support locus field
// TODO: support using V/J calls

use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::process;

use clap::Parser;
use rayon::ThreadPoolBuilder;
use thiserror::Error;

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
/// pairs between the repertoires that fall within the target similarity threshold.
///
/// If you provide the program with a path to a [FILE_QUERY], it will read its contents for input.
/// Otherwise, it will read form standard input until reaching an EOF signal.
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

    /// The number of OS threads the program spawns for computations (if 0 spawns one thread per CPU core).
    #[arg(short, long, default_value_t = 0)]
    num_threads: usize,

    /// Path to input AIRR-compliant TSV (if absent program reads from stdin until EOF).
    file_query: Option<String>,
}

fn main() -> Result<(), Error> {
    let args = Args::parse();

    ThreadPoolBuilder::new()
        .num_threads(args.num_threads)
        .build_global()?;

    let mut writer = BufWriter::new(io::stdout().lock());

    let parsed = match args.file_query {
        Some(path) => {
            let reader = get_file_bufreader(&path);
            parsing::parse_airr_tsv(reader).map_err(|e| Error::Parsing {
                input_name: path,
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

    let overlap_matrix = analysis::compute_overlap_matrix(&parsed)?;

    serialization::write_overlap_matrix_as_tsv(&overlap_matrix, &parsed, &mut writer)?;

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
