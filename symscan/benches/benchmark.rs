use criterion::{criterion_group, criterion_main, Bencher, Criterion, SamplingMode};
use rayon::ThreadPoolBuilder;
use std::hint::black_box;
use std::io::{self, BufRead, Cursor};
use std::time::Duration;
use symscan::{
    get_hamming_neighbors_across, get_hamming_neighbors_within, get_neighbors_across,
    get_neighbors_within, CachedRef, CachedRefHamming,
};

static QUERY_BYTES: &[u8] = include_bytes!("../../test_files/cdr3b_10k_a.txt");
static REFERENCE_BYTES: &[u8] = include_bytes!("../../test_files/cdr3b_10k_b.txt");

fn bytes_as_ascii_lines(bytes: &[u8]) -> Vec<String> {
    Cursor::new(bytes)
        .lines()
        .collect::<io::Result<Vec<String>>>()
        .expect("test files have valid lines")
}

fn configure_criterion() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(10))
        .sample_size(40)
}

fn bench_on_pool<T: Send>(
    pool: &rayon::ThreadPool,
    b: &mut Bencher,
    mut routine: impl FnMut() -> T + Send,
) {
    b.iter(|| pool.install(|| black_box(routine())));
}

fn setup_benchmarks(c: &mut Criterion) {
    let pool = ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("failed to build single-thread rayon pool");

    let query = bytes_as_ascii_lines(QUERY_BYTES);
    let reference = bytes_as_ascii_lines(REFERENCE_BYTES);
    let cached_query = CachedRef::new(&query, 1).expect("short input");
    let cached_reference = CachedRef::new(&reference, 1).expect("short input");
    let cached_query_hamming = CachedRefHamming::new(&query, 1).expect("short input");
    let cached_reference_hamming = CachedRefHamming::new(&reference, 1).expect("short input");

    let mut group = c.benchmark_group("symscan");
    group.sampling_mode(SamplingMode::Flat);

    group.bench_function("get_neighbors_within", |b| {
        bench_on_pool(&pool, b, || get_neighbors_within(&query, 1).unwrap());
    });

    group.bench_function("get_hamming_neighbors_within", |b| {
        bench_on_pool(&pool, b, || get_hamming_neighbors_within(&query, 1).unwrap());
    });

    group.bench_function("get_neighbors_across", |b| {
        bench_on_pool(&pool, b, || get_neighbors_across(&query, &reference, 1).unwrap());
    });

    group.bench_function("get_hamming_neighbors_across", |b| {
        bench_on_pool(&pool, b, || {
            get_hamming_neighbors_across(&query, &reference, 1).unwrap()
        });
    });

    group.bench_function("get_neighbors_within (cached)", |b| {
        bench_on_pool(&pool, b, || cached_reference.get_neighbors_within(1).unwrap());
    });

    group.bench_function("get_hamming_neighbors_within (cached)", |b| {
        bench_on_pool(&pool, b, || {
            cached_reference_hamming.get_neighbors_within(1).unwrap()
        });
    });

    group.bench_function("get_neighbors_cross (partially cached)", |b| {
        bench_on_pool(&pool, b, || {
            cached_reference.get_neighbors_across(&query, 1).unwrap()
        });
    });

    group.bench_function("get_neighbors_cross (fully cached)", |b| {
        bench_on_pool(&pool, b, || {
            cached_reference
                .get_neighbors_across_cached(&cached_query, 1)
                .unwrap()
        });
    });

    group.bench_function("get_hamming_neighbors_cross (partially cached)", |b| {
        bench_on_pool(&pool, b, || {
            cached_reference_hamming
                .get_neighbors_across(&query, 1)
                .unwrap()
        });
    });

    group.bench_function("get_hamming_neighbors_cross (fully cached)", |b| {
        bench_on_pool(&pool, b, || {
            cached_reference_hamming
                .get_neighbors_across_cached(&cached_query_hamming, 1)
                .unwrap()
        });
    });

    group.bench_function("cached instantiation", |b| {
        bench_on_pool(&pool, b, || CachedRef::new(&reference, 1).unwrap());
    });

    group.finish();
}

criterion_group! {
    name = bench;
    config = configure_criterion();
    targets = setup_benchmarks
}
criterion_main!(bench);
