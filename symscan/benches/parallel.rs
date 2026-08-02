use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use rayon::ThreadPoolBuilder;
use std::env;
use std::fs::File;
use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Duration;
use symscan::get_neighbors_across;

fn default_n() -> usize {
    200_000
}

fn bench_n() -> usize {
    env::var("SYMSCAN_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(default_n)
}

fn test_files_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../test_files")
}

fn load_lines(path: PathBuf, n: usize) -> Vec<String> {
    let file = File::open(&path).unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display()));
    BufReader::new(file)
        .lines()
        .take(n)
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

fn configure_criterion() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(15)
}

fn thread_counts() -> Vec<usize> {
    let max = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mut counts = vec![1, 2, 4, 8];
    if max > 8 && !counts.contains(&max) {
        counts.push(max);
    }
    counts.retain(|&c| c <= max);
    counts.sort_unstable();
    counts.dedup();
    counts
}

fn setup_parallel_benchmarks(c: &mut Criterion) {
    let n = bench_n();
    let dir = test_files_dir();
    let query = load_lines(dir.join("cdr3b_1m_a.txt"), n);
    let reference = load_lines(dir.join("cdr3b_1m_b.txt"), n);
    assert_eq!(query.len(), n, "expected {n} query lines");
    assert_eq!(reference.len(), n, "expected {n} reference lines");

    let mut group = c.benchmark_group(format!("parallel_across_{n}"));
    group.sampling_mode(SamplingMode::Flat);
    group.throughput(Throughput::Elements(n as u64));

    for threads in thread_counts() {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap_or_else(|e| panic!("failed to build rayon pool with {threads} threads: {e}"));

        group.bench_with_input(BenchmarkId::from_parameter(threads), &threads, |b, _| {
            b.iter(|| {
                pool.install(|| {
                    black_box(get_neighbors_across(&query, &reference, 1).expect("bench input ok"))
                })
            });
        });
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = setup_parallel_benchmarks
}
criterion_main!(benches);
