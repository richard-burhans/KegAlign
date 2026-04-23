# KegAlign — Codebase Knowledge Document

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Architecture](#2-architecture)
3. [Feature-by-Feature Analysis](#3-feature-by-feature-analysis)
4. [Data Flow](#4-data-flow)
5. [Key Data Structures](#5-key-data-structures)
6. [GPU Kernel Analysis](#6-gpu-kernel-analysis)
7. [Python Pipeline Layer](#7-python-pipeline-layer)
8. [MPS/MIG Multi-GPU Orchestration](#8-mpsmig-multi-gpu-orchestration)
9. [Build System](#9-build-system)
10. [Nuances, Subtleties & Gotchas](#10-nuances-subtleties--gotchas)
11. [Technical Reference & Glossary](#11-technical-reference--glossary)
12. [File Index](#12-file-index)

---

## 1. High-Level Overview

### What KegAlign Is

KegAlign is a **GPU-accelerated pairwise genome alignment tool**. It is a Galaxy Project fork of [SegAlign](https://github.com/gsneha26/SegAlign), designed to find homologous regions between two DNA sequences (a "target" / reference genome and a "query" genome) using NVIDIA GPUs to massively parallelize the computationally expensive seeding and ungapped extension phases.

**KegAlign does NOT produce final alignments itself.** It generates **High-Scoring Segment Pairs (HSPs)** on the GPU and outputs them as **LASTZ commands** (with segment files). The actual gapped alignment is then performed by [LASTZ](https://lastz.github.io/lastz/), a well-established CPU-based genome aligner. KegAlign acts as a high-performance front-end that replaces LASTZ's slowest phases (seeding + ungapped extension) with GPU-accelerated equivalents.

### Business Purpose

- **Domain**: Computational genomics / bioinformatics
- **Target users**: Genome researchers, bioinformatics engineers, Galaxy platform users
- **Problem solved**: Whole-genome alignment (WGA) is computationally expensive. A human-vs-human alignment can take days on CPUs. KegAlign offloads the O(n*m) seeding/filtering work to GPUs, reducing wall-clock time from days to hours.
- **Integration**: Ships as a Conda package (`kegalign-full`), and as Galaxy tool wrappers (`richard-burhans:kegalign`, `richard-burhans:batched_lastz`) on usegalaxy.org.

### Key Changes from Original SegAlign

1. Advanced runner script with MIG/MPS support for better GPU utilization
2. Updated to compile with TBB 2020.2
3. Fixed `--scoring` option to correctly read LASTZ scoring files
4. Added `--num_threads` option for CPU thread control
5. Added `--segment_size` option for CPU load balancing
6. Cleaned up build and addressed compiler warnings

### License

MIT License. Original copyright (2020) by Goenka, Turakhia, Paten, Horowitz. Fork copyright (2025) by Gulhan, Burhans, Harris, Kandemir, Haeussler, Nekrutenko.

---

## 2. Architecture

### Architecture Type

**Three-tier pipeline architecture:**

1. **GPU Core** (C++/CUDA) — seeding, seed lookup, ungapped X-drop extension on GPU
2. **TBB Flow Graph** (C++) — orchestrates data movement between CPU and GPU using Intel TBB's dataflow graph
3. **Python/Bash Pipeline** — coordinates the end-to-end workflow: input conversion, KegAlign execution, diagonal partitioning, LASTZ gapped extension, output merging

### Component Map

```
┌─────────────────────────────────────────────────────────────┐
│                    Python/Bash Pipeline                     │
│  runner.py / run_kegalign / run_lastz_tarball.py            │
│  diagonal_partition.py / package_output.py                  │
│  mps-mig/run_mig.py / split_input.py                        │
├─────────────────────────────────────────────────────────────┤
│                     KegAlign C++ Binary                     │
│  ┌──────────┐   ┌───────────┐   ┌────────────────┐          │
│  │ main.cpp │──>│ TBB Flow  │──>│ segment_printer│──> stdout│
│  │ (I/O +   │   │  Graph    │   │  (LASTZ cmds)  │          │
│  │ config)  │   │           │   └────────────────┘          │
│  └──────────┘   │ source    │                               │
│                 │ node      │   ┌──────────────┐            │
│                 │    │      │   │  seeder.cpp  │            │
│                 │    ▼      │   │  (k-mer +    │            │
│                 │ gatekeeper│──>│   GPU call)  │            │
│                 │ (join)    │   └──────┬───────┘            │
│                 └───────────┘          │                    │
│                                        ▼                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │               CUDA GPU Layer                        │    │
│  │  seed_filter.cu     seed_filter_interface.cu        │    │
│  │  seed_pos_table.cu                                  │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐   │    │
│  │  │ find_num_hits│  │  find_hits   │  │ find_hsps│   │    │
│  │  │  (kernel)    │  │  (kernel)    │  │ (kernel) │   │    │
│  │  └──────────────┘  └──────────────┘  └──────────┘   │    │
│  │  ┌────────────────────┐  ┌────────────────────┐     │    │
│  │  │compress_string     │  │compress_string_    │     │    │
│  │  │  (kernel)          │  │  rev_comp (kernel) │     │    │
│  │  └────────────────────┘  └────────────────────┘     │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                    Common / Shared                          │
│  DRAM.cpp/h  ntcoding.cpp/h  scoring.c/h  dna_utilities.c/h │
│  utilities.c/h  parameters.h  cuda_utils.h  kseq.h          │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology |
|-------|-----------|
| GPU compute | CUDA (C), Thrust library |
| CPU parallelism | Intel TBB (Threading Building Blocks) 2020.2, TBB Flow Graph |
| CLI parsing | Boost.Program_Options |
| FASTA reading | kseq.h (gzip-aware FASTA/FASTQ parser) + zlib |
| Scoring file parsing | LASTZ's own scoring code (from Robert S. Harris, MIT licensed) |
| Memory allocation | TBB scalable allocator (6 GB pre-allocated buffers) |
| Python scripts | Python 3.12, bashlex, pynvml, configparser, concurrent.futures |
| Build system | CMake >= 3.10 |
| Package management | Conda (conda-forge / bioconda channels) |

---

## 3. Feature-by-Feature Analysis

### Feature 1: Spaced-Seed Indexing on GPU

**Purpose**: Build a hash table (seed position table) of all k-mer positions in the reference sequence, then look up query k-mers to find "seed hits" (potential alignment anchors).

**Business need**: Finding matching subsequences between two genomes is the most time-consuming step. GPU parallelism turns an O(n) scan into a massively parallel operation.

**Technical details**:

- **Entry point**: `GenerateSeedPosTable()` in `common/seed_pos_table.cu`
- **Seed shapes**: Spaced seeds like "12of19" (`TTT0T00TT00T0T0TTTT`) where `T` = match position, `0` = don't-care. Configurable via `--seed`.
- **Index table**: A `2^(2*kmer_size)` prefix-sum array mapping each possible k-mer to its position list in the pos_table
- **Parallel construction**: Uses `tbb::parallel_for` on CPU to count k-mer occurrences, then `InclusivePrefixScan` (via Thrust on GPU) to build the cumulative index, then parallel fill of the position table
- **Transition seeds**: Optionally allows one transition mutation (A↔G or C↔T) in seed matches, increasing sensitivity at the cost of more seed hits. Controlled by `--notransition` flag.
- **Key files**: `common/seed_pos_table.cu`, `common/ntcoding.cpp`, `common/ntcoding.h`

### Feature 2: GPU Seed-and-Filter (Ungapped X-drop Extension)

**Purpose**: For each seed hit, perform ungapped extension in both directions using the X-drop algorithm to find High-Scoring Segment Pairs (HSPs).

**Business need**: Most seed hits are spurious. The X-drop extension filters them down to statistically significant HSPs that represent real homology.

**Technical details**:

- **Entry point**: `SeedAndFilter()` in `src/seed_filter.cu`
- **GPU kernels**:
  - `find_num_hits`: counts how many reference positions match each query seed
  - `find_hits`: populates the hit array with (ref_pos, query_pos) pairs
  - `find_hsps`: performs bidirectional X-drop ungapped extension using warp-level parallelism (each warp of 32 threads extends one hit)
  - `compress_output`: stream-compacts the results (removes failed extensions)
- **Warp-level parallelism**: Each warp (4 warps per block = `NUM_WARPS`) handles one seed hit. Within a warp, threads use `__shfl_up_sync` for prefix-sum computation of alignment scores.
- **Entropy filter**: HSPs near the threshold (score between 1x and 3x `hspthresh`) are penalized by an entropy factor to suppress low-complexity matches.
- **Deduplication**: After extension, HSPs are sorted and deduplicated using Thrust's `stable_sort` + `unique_copy` with custom comparators (`hspComp`, `hspEqual`).
- **Multi-GPU support**: A mutex-protected GPU pool (`available_gpus` vector) distributes work across GPUs. Each call to `SeedAndFilter` acquires a GPU, does the work, and releases it.
- **Key files**: `src/seed_filter.cu`, `common/seed_filter_interface.cu`, `common/store_gpu.h`

### Feature 3: TBB Flow Graph Pipeline

**Purpose**: Orchestrate the CPU↔GPU data flow, managing reference blocks, query blocks, and seed intervals through a producer-consumer pipeline.

**Business need**: Genomes are too large to fit entirely in GPU memory. The pipeline processes them in blocks, overlapping GPU computation with I/O and CPU work.

**Technical details**:

- **Entry point**: `main()` in `src/main.cpp`, lines 585-773
- **Graph structure**:
  ```
  source_node (reader) → join_node (gatekeeper) → function_node (seeder) → multifunction_node (printer)
                              ↑                                                        │
                              └── buffer_node (ticketer) ←─────────────────────────────┘
  ```
- **Ticket system**: A `buffer_node<size_t>` holds `cfg.num_threads` tokens. Each seeder invocation consumes a token; each printer completion returns one. This limits GPU concurrency.
- **Double buffering**: `BUFFER_DEPTH = 2` allows two query blocks to be resident on the GPU simultaneously, enabling overlap of GPU computation and host-to-device transfer.
- **Block management**: Sequences are split into blocks of `cfg.seq_block_size` (default 500 MB). Within each block, seed intervals of `cfg.lastz_interval_size` (default 10 MB) are dispatched to the seeder.
- **Key files**: `src/main.cpp`, `src/graph.h`

### Feature 4: Segment File Generation & LASTZ Command Output

**Purpose**: Convert GPU-computed HSPs into segment files and corresponding LASTZ command lines for gapped extension.

**Business need**: KegAlign replaces only the seeding/filtering phases. The actual gapped alignment (with insertions/deletions) is delegated to LASTZ.

**Technical details**:

- **Entry point**: `segment_printer_body::operator()` in `src/segment_printer.cpp`
- **Output format**: Tab-separated segment files with columns: `ref_chr ref_start ref_end query_chr query_start query_end strand score`
- **Coordinate translation**: The printer maps global buffer positions back to per-chromosome coordinates using binary search (`std::upper_bound`) on the chromosome start position arrays.
- **LASTZ command generation**: For each segment file, a complete LASTZ command is printed to stdout, including all relevant options (`--format`, `--ydrop`, `--gappedthresh`, `--strand`, `--ambiguous`, `--notrivial`, `--scores`, `--inner`, `--segments`, `--output`).
- **Thread safety**: `io_lock` mutex protects stdout writes.
- **2bit files**: LASTZ reads sequences from `.2bit` format files (created by `faToTwoBit` before KegAlign runs).
- **Key files**: `src/segment_printer.cpp`, `src/store.h`

### Feature 5: Diagonal Partitioning

**Purpose**: Split large segment files into smaller chunks sorted by diagonal, improving LASTZ's cache behavior and enabling better CPU load balancing.

**Business need**: LASTZ processes segments sequentially. Sorting by diagonal and chunking prevents any single LASTZ process from being a bottleneck.

**Technical details**:

- **Entry point**: `scripts/diagonal_partition.py`
- **Algorithm**: 
  - For forward (+) strand: sort by `(seq1_mid + seq2_mid, seq1_mid)` — groups segments along anti-diagonals
  - For reverse (-) strand: sort by `(seq1_mid - seq2_mid, seq1_mid)` — groups segments along diagonals
- **Chunk sizing**: Configurable via the first argument. `-1` = auto-estimate using quartile statistics of existing segment file sizes. `0` = skip partitioning.
- **Small-pair optimization**: Chromosome pairs with segment count ≤ chunk_size are kept unsorted in aggregated files, respecting LASTZ's requirement that query sequence names appear in input file order.
- **File naming**: Split files get `.split{N}` inserted before `.segments`
- **Key files**: `scripts/diagonal_partition.py`

### Feature 6: Galaxy Integration Pipeline (runner.py + package_output.py)

**Purpose**: Provide a Galaxy-compatible wrapper that orchestrates the full pipeline: KegAlign → diagonal partition → LASTZ → output merging.

**Business need**: Galaxy is a widely-used web-based platform for reproducible bioinformatics. Integration requires specific I/O conventions and packaging.

**Technical details**:

- **runner.py** (`scripts/runner.py`):
  - Parses Galaxy-style arguments, runs KegAlign as a subprocess
  - Optionally runs diagonal partitioners in parallel via `concurrent.futures.ProcessPoolExecutor`
  - Supports three output modes: `commands` (just the LASTZ command list), `output` (run LASTZ and produce alignment), `tarball` (package for deferred execution)
  - Uses `multiprocessing.Manager().Queue()` for inter-process communication

- **package_output.py** (`scripts/package_output.py`):
  - Packages KegAlign output into a tarball ("keg") for portable execution
  - Parses LASTZ commands using `bashlex`, rewrites paths for the tarball structure
  - Includes segment files, 2bit sequence files, name subset files, scoring files
  - Writes a `commands.json` manifest and `format.txt` metadata file
  - Tarball structure: `galaxy/commands.json`, `galaxy/format.txt`, `galaxy/files/*`

- **run_lastz_tarball.py** (`scripts/run_lastz_tarball.py`):
  - Extracts and executes a keg tarball
  - Runs LASTZ commands in parallel using `ProcessPoolExecutor`
  - Handles LASTZ truncation warnings gracefully
  - Produces final output in the Galaxy-expected format with metadata (`galaxy.json`)

- **Key files**: `scripts/runner.py`, `scripts/package_output.py`, `scripts/run_lastz_tarball.py`, `scripts/lastz-cmd.ini`

### Feature 7: Shell Runner Scripts

**Purpose**: Bash scripts that wrap the KegAlign binary with input conversion, output concatenation, and LASTZ execution.

**Technical details**:

- **run_kegalign** (`scripts/run_kegalign`): Standalone runner. Converts FASTA to 2bit, runs KegAlign piped through diagonal_partition.py, executes LASTZ commands in parallel (rate-limited by `num_threads`), concatenates output.
- **run_kegalign_symlink_sort** (`scripts/mps-mig/run_kegalign_symlink_sort`): MPS/MIG variant. Uses symlinks to pre-existing 2bit files instead of converting. Writes a UID file on GPU completion for process tracking. Uses `reallynice` for process priority management.
- **Error codes**: Both scripts define custom exit codes (4=file not found, 5=permissions, 6=LASTZ error, 7=unable to open, 9=not enough DRAM, 10-14=CUDA errors).

### Feature 8: MPS/MIG Multi-GPU Orchestration

**Purpose**: Run multiple KegAlign instances on the same GPU using NVIDIA MPS (Multi-Process Service) and/or MIG (Multi-Instance GPU) for up to 20% faster alignments.

**Business need**: Large GPUs (like A100) are underutilized by a single KegAlign instance. MPS/MIG allows sharing GPU resources.

**Technical details**:

- **split_input.py** (`scripts/mps-mig/split_input.py`):
  - Splits a genome FASTA into `N` approximately equal-sized chunks using Longest-Processing-Time-First bin packing
  - Optionally converts chunks to 2bit format
  - Supports `--goal_bp` for automatic bin count optimization using MSE

- **run_mig.py** (`scripts/mps-mig/run_mig.py`):
  - Manages a queue of (query, target) pairs across multiple GPU/MIG devices
  - Uses `CUDA_VISIBLE_DEVICES` and `CUDA_MPS_PIPE_DIRECTORY` to route each KegAlign instance to a specific device/MPS server
  - Tracks GPU completion via UID sentinel files
  - Handles memory failures with optional resubmission
  - Uses `pynvml` for NVIDIA management library access

---

## 4. Data Flow

### End-to-End Pipeline

```
Input FASTA (target.fa.gz, query.fa.gz)
    │
    ▼
faToTwoBit ──> ref.2bit, query.2bit   (for LASTZ later)
    │
    ▼
KegAlign binary (src/main.cpp)
    │
    ├─ Read target FASTA ──> ref_DRAM buffer (6 GB, TBB scalable alloc)
    ├─ Read query FASTA  ──> query_DRAM + query_rc_DRAM buffers
    │   └─ RevComp() generates reverse complement
    │
    ├─ For each reference block:
    │   ├─ SendRefWriteRequest() ──> compress to numeric encoding on GPU
    │   ├─ GenerateSeedPosTable() ──> build k-mer index table on GPU
    │   │
    │   └─ For each query block:
    │       ├─ SendQueryWriteRequest() ──> compress + rev-comp on GPU
    │       │
    │       └─ For each seed interval:
    │           ├─ seeder_body: extract k-mers from query
    │           ├─ SeedAndFilter() on GPU:
    │           │   ├─ find_num_hits kernel
    │           │   ├─ inclusive_scan (Thrust)
    │           │   ├─ find_hits kernel
    │           │   ├─ find_hsps kernel (X-drop extension)
    │           │   ├─ inclusive_scan + compress_output
    │           │   ├─ sort + unique (dedup)
    │           │   └─ return HSPs to CPU
    │           │
    │           └─ segment_printer_body:
    │               ├─ Write segment file (HSPs as tab-separated coords)
    │               └─ Print LASTZ command to stdout
    │
    ▼
LASTZ commands on stdout
    │
    ▼
diagonal_partition.py (optional)
    ├─ Sort segments by diagonal
    ├─ Split into chunks
    └─ Rewrite LASTZ commands with new segment files
    │
    ▼
LASTZ (gapped extension)
    ├─ Read ref.2bit + query.2bit
    ├─ Read segment file (HSP anchors)
    ├─ Perform Y-drop gapped extension
    └─ Output alignment (MAF, SAM, AXT, etc.)
    │
    ▼
Final alignment file (e.g., output.maf)
```

### DNA Encoding

Characters are encoded as integer values throughout the GPU pipeline:

| Character | Code | Constant |
|-----------|------|----------|
| A | 0 | `A_NT` |
| C | 1 | `C_NT` |
| G | 2 | `G_NT` |
| T | 3 | `T_NT` |
| a,c,g,t (lowercase/masked) | 4 | `L_NT` |
| N (ambiguous) | 5 | `N_NT` |
| other IUPAC | 6 | `X_NT` |
| & (block separator) | 7 | `E_NT` |

The substitution matrix is 8x8 (`NUC=8`, `NUC2=64`), stored as a flat array indexed by `[row*NUC + col]`.

---

## 5. Key Data Structures

### Configuration (`src/graph.h:30-75`)

```cpp
struct Configuration {
    string reference_filename, query_filename, data_folder;
    string strand;                    // "plus", "minus", "both"
    string scoring_file;              // LASTZ-format scoring file
    string ambiguous;                 // "n", "iupac", or "x" with penalties
    int sub_mat[NUC2];               // 8x8 substitution matrix (flat)
    Seed_config seed;                 // seed shape, size, kmer_size, transition
    uint32_t step;                    // seed step size
    int xdrop, hspthresh;            // ungapped extension params
    bool noentropy;                   // skip entropy filter
    bool gapped;                      // do gapped extension
    int ydrop, gappedthresh;         // gapped extension params (passed to LASTZ)
    int inner;                        // inner alignment param
    string output_format;             // "maf-", "sam", etc.
    uint32_t wga_chunk_size;         // GPU work chunk (default 250K)
    uint32_t lastz_interval_size;    // seed interval size (default 10M)
    uint32_t seq_block_size;         // sequence block size (default 500M)
    int num_gpu, num_threads;
    bool debug;
};
```

### segmentPair (`src/graph.h:23-28`)

```cpp
struct segmentPair {
    uint32_t ref_start;
    uint32_t query_start;
    uint32_t len;
    int score;
};
```
16 bytes per HSP. Used both on CPU and GPU. The `score` field is reused: in the first element of `SeedAndFilter`'s return vector, it stores the total hit count; in subsequent elements, it stores the HSP alignment score.

### DRAM (`common/DRAM.h`)

```cpp
class DRAM {
    char* buffer;          // 6 GB pre-allocated via TBB scalable_aligned_malloc
    size_t size;           // 6 GB
    size_t seqSize;        // actual sequence data size
    size_t bufferPosition; // current write position
};
```

Three DRAM instances: `ref_DRAM`, `query_DRAM`, `query_rc_DRAM`. Each allocates 6 GB at startup. This is a hard memory limit — sequences exceeding 6 GB will cause an exit.

### TBB Flow Graph Types (`src/graph.h:96-114`)

```
seeder_payload = tuple<seq_block, seed_interval>
printer_payload = tuple<seeder_payload, vector<segmentPair>, vector<segmentPair>>
seeder_input = tuple<seeder_payload, size_t>    // payload + ticket token
printer_input = tuple<printer_payload, size_t>  // payload + ticket token
```

### Chromosome Tracking (Global Vectors in `src/main.cpp`)

```cpp
// Forward query
vector<string>   q_chr_name;       // chromosome names
vector<uint32_t> q_chr_file_name;  // chromosome index
vector<size_t>   q_chr_start;      // start offset in DRAM buffer
vector<uint32_t> q_chr_len;        // chromosome length

// Reverse-complement query (reversed order within each block)
vector<string>   rc_q_chr_name;
vector<size_t>   rc_q_chr_start;
...

// Reference
vector<string>   r_chr_name;
vector<size_t>   r_chr_start;
...
```

---

## 6. GPU Kernel Analysis

### Kernel: `compress_string` (`common/seed_filter_interface.cu:19-47`)

Converts ASCII DNA characters to integer encoding on GPU. Simple grid-stride loop.

### Kernel: `compress_string_rev_comp` (`src/seed_filter.cu:112-156`)

Same as above but also produces the reverse complement simultaneously in a second output buffer (reversed index: `dst_seq_rc[len-1-i]`).

### Kernel: `find_num_hits` (`src/seed_filter.cu:159-183`)

For each seed, looks up how many reference positions share that k-mer via the index table. Grid-stride loop, one thread per seed.

### Kernel: `find_hits` (`src/seed_filter.cu:186-231`)

For each seed, populates the hit array with (ref_pos, query_pos) pairs. One block per seed, multiple warps iterate over matching reference positions.

### Kernel: `find_hsps` (`src/seed_filter.cu:234-653`)

The core computational kernel. For each seed hit, performs bidirectional X-drop ungapped extension:

1. **Right extension**: Starting from the seed position, extends rightward using warp-level prefix sums of substitution scores. Tracks running max score and position. Stops when X-drop condition is met or sequence boundary reached.
2. **Left extension**: Same algorithm extending leftward.
3. **Entropy filter**: If score is between `hspthresh` and `3*hspthresh`, computes Shannon entropy of matching bases. Adjusts score by entropy factor. Low-entropy (repetitive) regions are penalized.
4. **Output**: Writes final HSP (ref_start, query_start, len, score) or zeros it out if below threshold.

Key optimization: uses `__shfl_up_sync` for warp-level prefix scans instead of shared memory, reducing synchronization overhead.

### Kernel: `compress_output` (`src/seed_filter.cu:656-681`)

Stream compaction: uses prefix-summed `d_done` array to pack surviving HSPs into contiguous memory.

---

## 7. Python Pipeline Layer

### runner.py — Main Galaxy Orchestrator

**Classes**:
- `LastzCommands`: Collection of parsed LASTZ commands with associated segments
- `LastzCommand`: Parses a LASTZ command string via regex, extracting all parameters
- `KegAlignSegments` / `KegAlignSegment`: Tracks and sorts segment files by (strand, tmp, block, r, split)

**Flow**:
1. Parse args, pass unknown args through to KegAlign
2. Run KegAlign subprocess, collect LASTZ commands from stdout
3. If `--diagonal-partition`: run diagonal partitioners in parallel on the commands
4. Write commands to file
5. If `--output-type=output`: run LASTZ workers in parallel, concatenate outputs
6. If `--output-type=tarball`: package for later execution

### diagonal_partition.py — Load Balancer

**Algorithm**:
1. Parse LASTZ command to find segment file
2. If segment count ≤ chunk_size or ≤ MIN_CHUNK_SIZE (5000): pass through unchanged
3. Otherwise, group HSPs by (ref_chr, query_chr) pair
4. Sort each pair's segments by diagonal coordinate
5. Split into chunks of `chunk_size` segments
6. Small pairs (≤ chunk_size) are aggregated into combined files, maintaining LASTZ's required query name order
7. Output new LASTZ commands with updated file references

### package_output.py — Tarball Packager

Creates a portable tarball containing:
- `galaxy/commands.json`: JSON-lines file with parsed LASTZ commands
- `galaxy/format.txt`: output format identifier
- `galaxy/files/`: segment files, 2bit files, name files, scoring files

### run_lastz_tarball.py — Tarball Executor

Extracts tarball, validates command structure, runs LASTZ in parallel, concatenates output, writes Galaxy metadata.

---

## 8. MPS/MIG Multi-GPU Orchestration

### split_input.py

**Algorithm**: Longest-Processing-Time-First (LPT) bin packing
1. Read FASTA, sort sequences by length (descending)
2. Use a min-heap to assign each sequence to the lightest bin
3. If `--goal_bp` is set: try all bin counts 1..max_chunks, select the one minimizing MSE from goal

### run_mig.py

**Architecture**:
- `GPU_queue`: Tracks which processes run on which GPU/MIG device, using UID files as completion signals
- `Process_List`: Manages subprocess lifecycle, detects failures (core dumps, CUDA errors, bad_alloc)
- Main loop: iterates over all (query, target) pairs, assigns each to a free GPU slot, monitors completion

**GPU assignment**: Uses `CUDA_VISIBLE_DEVICES` to restrict each KegAlign instance to one GPU/MIG device, and `CUDA_MPS_PIPE_DIRECTORY` to route through the correct MPS server.

---

## 9. Build System

### CMakeLists.txt

- **Languages**: C, C++, CUDA
- **C++ standard**: C++14
- **CUDA architectures**: Auto-detected via `scripts/get-cuda-arches.bash` which maps CUDA compiler version to supported SM architectures (from SM 3.5 through SM 12.1)
- **Dependencies**: TBB (tbbmalloc, tbb_preview), Boost (program_options), ZLIB, CUDA
- **Compile flags**: `-O4 -std=c++14`, CUDA separable compilation enabled
- **Suppressed warnings**: `TBB_SUPPRESS_DEPRECATED_MESSAGES`

### Source files compiled into `kegalign` binary:

```
common/DRAM.cpp
common/ntcoding.cpp
common/seed_filter_interface.cu
common/seed_pos_table.cu
common/utilities.c
common/dna_utilities.c
common/scoring.c
src/seed_filter.cu
src/seeder.cpp
src/segment_printer.cpp
src/main.cpp
```

### Conda Environment

`scripts/make-conda-env.bash` installs either:
- **Production**: `kegalign-full` from bioconda
- **Development** (`-dev`): bashlex, cmake, gxx 13, libboost, tbb-devel 2020.2, zlib, Python 3.12, mypy, black, flake8

### faToTwoBit Build

`make-faToTwoBit.bash` downloads UCSC userApps v470, applies two patches (`include.patch` for build system fixes, `add-option.patch` for `--namePrefix` option), and builds just the `faToTwoBit` utility.

---

## 10. Nuances, Subtleties & Gotchas

### Things You Must Know Before Changing Code

1. **6 GB hard memory limit**: Each DRAM buffer (`ref_DRAM`, `query_DRAM`, `query_rc_DRAM`) allocates exactly 6 GB at startup via `scalable_aligned_malloc`. Input sequences exceeding this will cause `exit(9)`. This is not configurable at runtime.

2. **Block separator character `&`**: Between chromosomes within a block, a `&` character (encoded as `E_NT = 7`) is inserted with a penalty of `-10*xdrop`. This prevents seed extensions from crossing chromosome boundaries. The segment printer must account for this extra byte when computing chromosome offsets.

3. **Double-buffering complexity**: The `BUFFER_DEPTH = 2` system allows two query blocks on GPU simultaneously. The `q_buffer` vector tracks which physical buffer (0 or 1) each logical query block uses. The `num_seeded_regions` atomic array tracks completion per buffer. Getting this wrong causes GPU memory corruption.

4. **`gapped` flag is inverted**: `cfg.gapped = !cfg.gapped` (line 219 of main.cpp). The command-line flag is `--nogapped` which sets `cfg.gapped = true`, then it's flipped. So `cfg.gapped == true` means gapped extension IS enabled.

5. **`transition` flag is also inverted**: `cfg.seed.transition = !cfg.seed.transition` (line 193). The CLI flag `--notransition` sets it to `true`, then it's flipped. `cfg.seed.transition == true` means transitions ARE allowed.

6. **MAX_HITS scales with GPU memory**: `MAX_HITS = MAX_HITS_PER_GB * global_mem_gb` where `MAX_HITS_PER_GB = 4194304` (4M). On a 16 GB GPU, that's ~64M hits. If a single seed interval generates more hits than `MAX_HITS`, the work is split into multiple iterations (`num_iter`).

7. **Segment file naming convention**: `tmp{N}.block{B}.r{R}.{plus|minus}[.split{S}].segments`. This naming is parsed by regex in multiple places (runner.py's `LastzCommand`, diagonal_partition.py, run_kegalign scripts). Changing the format requires updating all parsers.

8. **Query name order requirement**: LASTZ requires query sequence names in segment files to appear in the same order as in the query 2bit file. `diagonal_partition.py` takes care of this when aggregating small chromosome pairs (`query_key_order_table`). Violating this causes silent incorrect output.

9. **Reverse complement chromosome order**: Within each query block, reverse-complement chromosome names are stored in reverse order (`for(int i = block_chrs.size()-1; i >= 0; i--)` in main.cpp). This matches the reversed memory layout after `RevComp()`.

10. **GPU pool mutex**: The `mu` mutex and `cv` condition variable in `common/store_gpu.h` / `common/seed_filter_interface.cu` protect GPU device selection. Each `SeedAndFilter` call and `InclusivePrefixScan` call acquires a GPU from the pool and releases it when done. Deadlock is possible if all GPUs are held and a blocking operation is introduced.

11. **Seed position table uses host-side prefix scan**: Despite using Thrust, `InclusivePrefixScan` uses `thrust::host` execution policy (line 24 of seed_pos_table.cu). This is intentional — it still acquires a GPU from the pool (for device context) but runs the scan on CPU. This is likely a historical artifact.

12. **The `first_el` hack in SeedAndFilter return**: The first element of the returned vector is NOT an HSP — it encodes metadata: `first_el.len = total_anchors`, `first_el.score = num_hits`. The seeder must skip index 0 and start processing from index 1.

13. **LASTZ scoring file compatibility**: The `load_scoring_matrix()` function in `common/scoring.c` uses LASTZ's own `read_score_set_by_name()` from `dna_utilities.c` (a large ~1400-line file from the original LASTZ codebase). This requires the full LASTZ scoring file format including gap penalties, not just a 4x4 matrix.

14. **kseq.h is a header-only FASTA parser**: `KSEQ_INIT2(, gzFile, gzread)` macro instantiates the parser for gzip files. It handles both compressed and uncompressed FASTA. The `kseq_read()` function reads one sequence at a time.

15. **Concurrent LASTZ execution in shell scripts**: The `run_kegalign` script uses bash-level parallelism (`eval "$line" &`) with manual PID tracking to limit concurrency. The `run_kegalign_symlink_sort` variant uses `reallynice` for autogroup-aware priority management and `pv` (pipe viewer) for buffering instead of `mbuffer`.

16. **Patch files**: `add-option.patch` adds a `--namePrefix` option to `faToTwoBit` (from UCSC tools) to prefix sequence names in 2bit files. `include.patch` fixes build paths for Conda environments. Both are applied during `make-faToTwoBit.bash`.

---

## 11. Technical Reference & Glossary

### Domain Glossary

| Term | Definition |
|------|-----------|
| **HSP** | High-Scoring Segment Pair — a maximal-scoring ungapped alignment between two sequences |
| **X-drop** | Extension termination criterion: stop when the current score drops more than X below the maximum seen |
| **Y-drop** | Similar to X-drop but for gapped extension (used by LASTZ) |
| **Spaced seed** | A pattern of match (T/1) and don't-care (0) positions, e.g., "12of19" has 12 match positions in a 19-base window |
| **Transition** | A purine↔purine (A↔G) or pyrimidine↔pyrimidine (C↔T) mutation. More common than transversions in biology. |
| **Entropy filter** | Penalizes HSPs in low-complexity (repetitive) regions using Shannon entropy |
| **2bit format** | UCSC compact format storing DNA as 2 bits per base with masking information |
| **MAF** | Multiple Alignment Format — standard output for pairwise/multiple genome alignments |
| **MIG** | Multi-Instance GPU — NVIDIA technology to partition a physical GPU into isolated instances |
| **MPS** | Multi-Process Service — NVIDIA technology allowing multiple CUDA processes to share a GPU |
| **WGA** | Whole-Genome Alignment |
| **Diagonal** | In a dot plot, alignments between two sequences appear as diagonals. Forward matches on `y = x + offset`, reverse on `y = -x + offset`. |
| **Segment file** | Tab-separated file of HSPs (anchors) passed to LASTZ via `--segments` flag |
| **Keg / Tarball** | Packaged bundle of LASTZ commands + data files for portable execution |

### Key Constants (`common/parameters.h`)

| Constant | Value | Meaning |
|----------|-------|---------|
| `VERSION` | `"v0.1.2.8"` | Current KegAlign version |
| `NUC` | 8 | Nucleotide alphabet size (A,C,G,T,lower,N,X,&) |
| `NUC2` | 64 | Substitution matrix size (8*8) |
| `MAX_BLOCKS` | 1024 | CUDA grid size for utility kernels |
| `MAX_THREADS` | 1024 | CUDA block size for utility kernels |
| `BLOCK_SIZE` | 128 | CUDA block size for find_hits and find_hsps |
| `NUM_WARPS` | 4 | Warps per block in find_hsps (128/32=4) |
| `TRANSITION_MASK` | 2 | XOR mask for transition mutations (flips bit 1) |

### Default Configuration Values (`src/graph.h`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_WGA_CHUNK` | 250,000 | GPU work chunk size for seed processing |
| `DEFAULT_LASTZ_INTERVAL` | 10,000,000 | Query interval size per seeder dispatch |
| `DEFAULT_SEQ_BLOCK_SIZE` | 500,000,000 | Sequence block size (500 MB) |
| `BUFFER_DEPTH` | 2 | Double-buffered query blocks on GPU |

### Default Alignment Parameters (from main.cpp CLI)

| Parameter | Default | CLI Flag |
|-----------|---------|----------|
| Seed shape | 12of19 | `--seed` |
| Step | 1 | `--step` |
| Transitions | allowed | `--notransition` to disable |
| X-drop | 910 | `--xdrop` |
| HSP threshold | 3000 | `--hspthresh` |
| Entropy | enabled | `--noentropy` to disable |
| Gapped extension | enabled | `--nogapped` to disable |
| Y-drop | 9430 | `--ydrop` |
| Gapped threshold | same as hspthresh | `--gappedthresh` |
| Output format | maf- | `--format` |
| Strand | both | `--strand` |

### Default Scoring Matrix (HOXD70)

```
       A     C     G     T
A    91  -114   -31  -123
C  -114   100  -125   -31
G   -31  -125   100  -114
T  -123   -31  -114    91
```

Gap open: 400, Gap extend: 30, Bad score: -1000, Fill score: -100.

### Error Exit Codes (from run_kegalign scripts)

| Code | Meaning |
|------|---------|
| 4 | File not found |
| 5 | No permissions to create directory |
| 6 | Error with LASTZ gapped extension |
| 7 | Unable to open file |
| 9 | Not enough allocated CPU DRAM (>6 GB) |
| 10 | Requested GPUs greater than available GPUs |
| 11 | `cudaSetDevice` error |
| 12 | `cudaMalloc` error |
| 13 | `cudaMemcpy` error |
| 14 | `cudaFree` error |

### Function Pointer Interface

GPU functions are accessed via function pointers, allowing potential runtime dispatch:

```cpp
// seed_filter_interface.h
g_InitializeInterface   → InitializeInterface()   // detect GPUs
g_SendRefWriteRequest   → SendRefWriteRequest()    // upload ref to GPU
g_ClearRef              → ClearRef()               // free ref GPU memory

// seed_filter.h
g_InitializeProcessor   → InitializeProcessor()    // alloc GPU buffers
g_SendQueryWriteRequest → SendQueryWriteRequest()   // upload query to GPU
g_SeedAndFilter         → SeedAndFilter()           // main GPU computation
g_ClearQuery            → ClearQuery()              // free query GPU memory
g_ShutdownProcessor     → ShutdownProcessor()       // cleanup
```

---

## 12. File Index

### Core C++/CUDA (`src/`)

| File | Type | Purpose |
|------|------|---------|
| `src/main.cpp` | C++ | Entry point, CLI parsing, sequence loading, TBB flow graph setup |
| `src/seeder.cpp` | C++ | Seeder node: k-mer extraction, calls GPU SeedAndFilter |
| `src/segment_printer.cpp` | C++ | Printer node: writes segment files, generates LASTZ commands |
| `src/seed_filter.cu` | CUDA | GPU kernels: find_hits, find_hsps, SeedAndFilter orchestrator |
| `src/graph.h` | Header | All type definitions: Configuration, segmentPair, TBB graph types |
| `src/store.h` | Header | Extern declarations for global DRAM and chromosome vectors |
| `src/seed_filter.h` | Header | Function pointer typedefs for GPU interface |

### Common Library (`common/`)

| File | Type | Purpose |
|------|------|---------|
| `common/DRAM.cpp` / `.h` | C++ | 6 GB pre-allocated sequence buffer class |
| `common/ntcoding.cpp` / `.h` | C++ | K-mer encoding, seed shape parsing, reverse complement |
| `common/seed_filter_interface.cu` / `.h` | CUDA | GPU initialization, reference upload, device management |
| `common/seed_pos_table.cu` | CUDA | Seed position table generation (parallel k-mer indexing) |
| `common/parameters.h` | Header | Version, nucleotide constants, CUDA kernel dimensions |
| `common/cuda_utils.h` | Header | CUDA error-checking wrappers |
| `common/store_gpu.h` | Header | GPU pool mutex, device count, GPU memory pointers |
| `common/scoring.c` / `.h` | C | LASTZ scoring file parser |
| `common/dna_utilities.h` / `.c` | C | LASTZ DNA utility library (scoring types, score sets, entropy) |
| `common/utilities.h` / `.c` | C | LASTZ utility library (sized types, memory wrappers, string ops) |
| `common/kseq.h` | Header | Header-only gzip-aware FASTA/FASTQ parser |

### Python Scripts (`scripts/`)

| File | Type | Purpose |
|------|------|---------|
| `scripts/runner.py` | Python | Galaxy orchestrator: KegAlign → partition → LASTZ pipeline |
| `scripts/diagonal_partition.py` | Python | Sorts segment files by diagonal, splits into chunks |
| `scripts/package_output.py` | Python | Packages output as Galaxy-compatible tarball |
| `scripts/run_lastz_tarball.py` | Python | Extracts and executes tarball, parallel LASTZ |
| `scripts/lastz-cmd.ini` | Config | LASTZ argument type definitions for command parsing |
| `scripts/mypy.ini` | Config | MyPy type-checking configuration |

### Shell Scripts (`scripts/`)

| File | Type | Purpose |
|------|------|---------|
| `scripts/run_kegalign` | Bash | Standalone runner: convert → kegalign → partition → LASTZ |
| `scripts/make-conda-env.bash` | Bash | Conda environment setup (production + dev) |
| `scripts/get-cuda-arches.bash` | Bash | Auto-detect CUDA SM architectures for CMake |

### MPS/MIG Scripts (`scripts/mps-mig/`)

| File | Type | Purpose |
|------|------|---------|
| `scripts/mps-mig/split_input.py` | Python | Split genome FASTA into balanced chunks |
| `scripts/mps-mig/run_mig.py` | Python | Multi-GPU/MIG orchestrator with MPS support |
| `scripts/mps-mig/run_kegalign_symlink_sort` | Bash | MIG-aware KegAlign runner with UID tracking |

### Build & Config

| File | Type | Purpose |
|------|------|---------|
| `CMakeLists.txt` | CMake | Build configuration |
| `add-option.patch` | Patch | Adds `--namePrefix` to faToTwoBit |
| `include.patch` | Patch | Fixes UCSC build paths for Conda |
| `make-faToTwoBit.bash` | Bash | Downloads and builds faToTwoBit with patches |
| `.gitignore` | Git | Ignores `build/`, `test/`, `bin/` |

### Test Data (`test-data/`)

| File | Purpose |
|------|---------|
| `test-data/apple.fasta.gz` | Sample target genome |
| `test-data/orange.fasta.gz` | Sample query genome |
| `test-data/apple_orange.maf.gz` | Expected alignment output for validation |

---

*Document generated from KegAlign repository at commit f76e36c on branch `experiment`.*
