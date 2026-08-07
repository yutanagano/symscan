use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use rayon::ThreadPoolBuilder;
use std::fs::File;
use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Duration;
use symscan::{get_neighbors_across, CachedRef};

const NUM_STRINGS: usize = 1_000_000;

fn test_files_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../test_files")
}

fn load_lines(path: PathBuf) -> Vec<String> {
    let file = File::open(&path).unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display()));
    BufReader::new(file)
        .lines()
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

fn load_inputs() -> (Vec<String>, Vec<String>) {
    let dir = test_files_dir();
    let query = load_lines(dir.join("cdr3b_1m_a.txt"));
    let reference = load_lines(dir.join("cdr3b_1m_b.txt"));
    assert_eq!(query.len(), NUM_STRINGS, "expected {NUM_STRINGS} query lines");
    assert_eq!(
        reference.len(),
        NUM_STRINGS,
        "expected {NUM_STRINGS} reference lines"
    );
    (query, reference)
}

fn setup_parallel_across(c: &mut Criterion) {
    let (query, reference) = load_inputs();

    let mut group = c.benchmark_group("parallel_across");
    group.sampling_mode(SamplingMode::Flat);
    group.throughput(Throughput::Elements(NUM_STRINGS as u64));

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

fn setup_parallel_cached_new(c: &mut Criterion) {
    let (_, reference) = load_inputs();

    let mut group = c.benchmark_group("parallel_cached_new");
    group.sampling_mode(SamplingMode::Flat);
    group.throughput(Throughput::Elements(NUM_STRINGS as u64));

    for threads in thread_counts() {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap_or_else(|e| panic!("failed to build rayon pool with {threads} threads: {e}"));

        group.bench_with_input(BenchmarkId::from_parameter(threads), &threads, |b, _| {
            b.iter(|| {
                pool.install(|| black_box(CachedRef::new(&reference, 1).expect("bench input ok")))
            });
        });
    }

    group.finish();
}

fn setup_parallel_cached_across(c: &mut Criterion) {
    let (query, reference) = load_inputs();
    let cached = CachedRef::new(&reference, 1).expect("bench input ok");

    let mut group = c.benchmark_group("parallel_cached_across");
    group.sampling_mode(SamplingMode::Flat);
    group.throughput(Throughput::Elements(NUM_STRINGS as u64));

    for threads in thread_counts() {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap_or_else(|e| panic!("failed to build rayon pool with {threads} threads: {e}"));

        group.bench_with_input(BenchmarkId::from_parameter(threads), &threads, |b, _| {
            b.iter(|| {
                pool.install(|| {
                    black_box(cached.get_neighbors_across(&query, 1).expect("bench input ok"))
                })
            });
        });
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = setup_parallel_across, setup_parallel_cached_new, setup_parallel_cached_across
}
criterion_main!(benches);
