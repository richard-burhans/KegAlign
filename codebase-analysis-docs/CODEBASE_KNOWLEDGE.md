# KegAlign Codebase Knowledge Document

> **Version:** v0.1.2.8  
> **Generated:** 2026-04-21  
> **Repo root:** `KegAlign/`  
> **License:** MIT  
> **Citation:** A.B. Gulhan et al. 2024 bioRxiv

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [System Architecture](#2-system-architecture)
3. [Directory Structure & File Map](#3-directory-structure--file-map)
4. [Data Structures Reference](#4-data-structures-reference)
5. [Feature-by-Feature Analysis](#5-feature-by-feature-analysis)
   - [5.1 Seed Generation & Indexing](#51-seed-generation--indexing)
   - [5.2 GPU Seed Filtering & X-drop Extension](#52-gpu-seed-filtering--x-drop-extension)
   - [5.3 TBB Flow Graph Concurrency](#53-tbb-flow-graph-concurrency)
   - [5.4 LASTZ Command Generation (Segment Printer)](#54-lastz-command-generation-segment-printer)
   - [5.5 Diagonal Partitioning](#55-diagonal-partitioning)
   - [5.6 Galaxy / Runner Orchestration](#56-galaxy--runner-orchestration)
   - [5.7 MPS/MIG GPU Scheduling](#57-mpsmig-gpu-scheduling)
   - [5.8 Genome Input Partitioning](#58-genome-input-partitioning)
   - [5.9 LASTZ Tarball Execution](#59-lastz-tarball-execution)
   - [5.10 Output Packaging](#510-output-packaging)
6. [Cross-Feature Interaction Map](#6-cross-feature-interaction-map)
7. [GPU Memory Management](#7-gpu-memory-management)
8. [Nucleotide Encoding & Scoring](#8-nucleotide-encoding--scoring)
9. [Nuances, Subtleties & Gotchas](#9-nuances-subtleties--gotchas)
10. [Command-Line Reference](#10-command-line-reference)
11. [Build System](#11-build-system)
12. [Glossary](#12-glossary)
13. [Key Functions Index](#13-key-functions-index)
14. [Assumptions Table](#14-assumptions-table)

---

## 1. High-Level Overview

### What is KegAlign?

KegAlign is a **GPU-accelerated whole-genome pairwise sequence aligner**. It is a research tool in the field of comparative genomics, designed to align large genomes (hundreds of megabases to gigabases) by offloading the computationally expensive seeding and ungapped extension phase to NVIDIA GPUs.

KegAlign is a **modified fork of SegAlign** (Goenka et al. 2020), developed by the Galaxy Project team (Gulhan, Burhans, Harris, Kandemir, Haeussler, Nekrutenko, 2025).

### Business/Research Purpose

| Goal | How KegAlign Addresses It |
|------|--------------------------|
| Align two large genomes | Spaced-seed + X-drop filter on GPU, then LASTZ gapped alignment on CPU |
| Speed up whole-genome alignment | GPU parallelism replaces CPU-only SegAlign, ~20% faster with MPS/MIG |
| Galaxy workflow integration | Python wrapper scripts for Galaxy Tool Shed deployment |
| Load-balanced LASTZ execution | Diagonal partition of HSPs for LASTZ cache locality |
| Multi-GPU support | MIG/MPS orchestration with per-device scheduling |

### Primary Users

- **Computational biologists** performing whole-genome alignments
- **Galaxy platform users** using the packaged Galaxy tools
- **HPC administrators** running on GPU clusters
- **Bioinformatics developers** integrating into alignment pipelines

### Three-Stage Pipeline (High Level)

```
[Input Preparation]    [KegAlign - GPU Seeding]    [LASTZ - Gapped Alignment]
FASTA → .2bit    →    Seed+Filter+XDrop        →   Gapped extension → MAF/SAM/AXT
```

1. **Input Preparation:** Convert FASTA to UCSC `.2bit` binary format for random-access reading
2. **GPU Seeding (KegAlign binary):** Generate spaced k-mer seeds, find matches in reference, perform ungapped X-drop extension → produce HSP (High Scoring Pair) segment files + LASTZ commands
3. **Gapped Alignment (LASTZ):** Run each LASTZ command to perform Smith-Waterman gapped alignment on the filtered HSPs

---

## 2. System Architecture

### Architecture Diagram

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                        KegAlign System                                        │
│                                                                               │
│  ┌────────────────┐   ┌────────────────────────────────────────────────────┐  │
│  │  Orchestration │   │              KegAlign Binary (C++/CUDA)            │  │
│  │  Layer         │   │                                                    │  │
│  │                │   │  main.cpp ──► DRAM (6GB×3 buffers)                 │  │
│  │  runner.py     │──►│     │                                              │  │
│  │  run_mig.py    │   │     ├──► seed_pos_table.cu                         │  │
│  │  split_in.py   │   │     │    (Build kmer index → GPU)                  │  │
│  │                │   │     │                                              │  │
│  │                │   │     └──► TBB Flow Graph (graph.h)                  │  │
│  │                │   │          │                                         │  │
│  │                │   │          ├── seeder.cpp (k-mer extraction)         │  │
│  │                │   │          ├── seed_filter.cu (GPU kernels)          │  │
│  │                │   │          └── segment_printer.cpp (LASTZ cmds)      │  │
│  └────────────────┘   └────────────────────────────────────────────────────┘  │
│                                      │                                        │
│                                      ▼ LASTZ commands (stdout)                │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │  Post-Processing Layer (Python)                                         │  │
│  │                                                                         │  │
│  │  diagonal_partition.py ──► LASTZ binary ──► MAF/SAM/AXT/PAF output      │  │
│  │  package_output.py     ──► Tarball for Galaxy                           │  │
│  │  run_lastz_tarball.py  ──► Execute from tarball                         │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────┘
```

See also: [`assets/architecture.mmd`](assets/architecture.mmd) (Mermaid source)

### TBB Flow Graph Topology

The concurrent execution model inside the KegAlign binary is a **token-throttled TBB flow graph**:

```
reader(source_node) ──────────────────────────────────────────────►
                                                                    │
ticketer(buffer_node) ──► gatekeeper(join_node) ──► seeder(fn) ──► printer(multifunction_node)
     ▲                                                                    │
     └────────────────────────────────────────────────────────────────────┘
                            (ticket returned via output_port<0>)
```

- **reader** emits `seeder_payload` items (ref block × query interval pairs)
- **ticketer** holds N tokens (N = `num_threads`), gating concurrency
- **gatekeeper** (join_node) requires both a payload AND a token before forwarding to seeder
- **seeder** performs GPU seed-and-filter on one payload, emitting `printer_payload`
- **printer** writes LASTZ command + segment file, then returns the ticket to **ticketer**

This prevents more than `num_threads` GPU tasks from executing simultaneously. See [`assets/tbb_flow_graph.mmd`](assets/tbb_flow_graph.mmd).

### Data Flow (Sequence Diagram)

See [`assets/data_flow.mmd`](assets/data_flow.mmd) for the full sequence diagram.

Abbreviated:
1. `main.cpp` reads FASTA (gzip-compressed) → `DRAM` 6GB CPU buffers (ref, query, query_rc)
2. `seed_pos_table.cu` builds kmer index from reference → uploads to all GPUs
3. `seed_filter_interface.cu` uploads compressed reference to all GPUs
4. TBB flow graph iterates over ref_block × query_interval pairs:
   - `seeder.cpp` extracts k-mer seeds from query chunk
   - `seed_filter.cu` runs GPU kernels: match seeds → X-drop extend → return HSPs
   - `segment_printer.cpp` writes `.seg` file + prints LASTZ command to stdout
5. Caller (runner.py or user) executes LASTZ commands in parallel

---

## 3. Directory Structure & File Map

```
KegAlign/
├── CMakeLists.txt              # Build system (CMake 3.10+)
├── README.md                   # User documentation
├── LICENSE                     # MIT license
├── add-option.patch            # Patch for faToTwoBit -namePrefix option
├── include.patch               # Conda/SSL/Python3 patches
├── make-faToTwoBit.bash        # Build helper for faToTwoBit
├── kegalign_logo.{png,webp,svg}# Brand assets
│
├── src/                        # Main application source
│   ├── main.cpp                # Entry point, argument parsing, TBB graph setup
│   ├── graph.h                 # All shared data structures + TBB node types
│   ├── seeder.cpp              # seeder_body::operator() — k-mer generation
│   ├── seed_filter.cu          # CUDA GPU kernels (seed matching, X-drop)
│   ├── seed_filter.h           # Function pointer interface to seed_filter.cu
│   ├── segment_printer.cpp     # segment_printer_body::operator() — LASTZ cmd output
│   └── store.h                 # extern declarations for global DRAM + chromosome metadata
│
├── common/                     # Shared utilities and GPU infrastructure
│   ├── DRAM.cpp / DRAM.h       # 6GB aligned CPU buffer class (TBB scalable_aligned_malloc)
│   ├── cuda_utils.h            # CUDA error checking wrappers (exit codes 11-14)
│   ├── dna_utilities.c/.h      # DNA scoring matrices, HOXD70, ambiguity codes
│   ├── kseq.h                  # FASTA/FASTQ reader (macro-template, header-only)
│   ├── ntcoding.cpp/.h         # Spaced seed encoding, kmer extraction, reverse complement
│   ├── parameters.h            # Constants: VERSION, NUC/NUC2, GPU params (BLOCK_SIZE etc.)
│   ├── scoring.c / scoring.h   # load_scoring_matrix() wrapper
│   ├── seed_filter_interface.cu/.h  # Multi-GPU init + reference distribution
│   ├── seed_pos_table.cu       # Seed position table builder (TBB parallel + thrust prefix sum)
│   ├── store_gpu.h             # extern GPU variable declarations (mutex, cv, d_ref_seq etc.)
│   └── utilities.c / utilities.h   # String/memory/bit utilities (Robert S. Harris)
│
├── scripts/                    # Python and bash orchestration
│   ├── runner.py               # Galaxy-style orchestration wrapper
│   ├── diagonal_partition.py   # Diagonal sort + chunk HSP segment files
│   ├── run_lastz_tarball.py    # Execute LASTZ from tarball (Galaxy containerized)
│   ├── package_output.py       # Package LASTZ results into tarball for Galaxy
│   ├── lastz-cmd.ini           # LASTZ argument schema (INI format)
│   ├── mypy.ini                # Python type checking config
│   ├── make-conda-env.bash     # Conda environment setup
│   ├── get-cuda-arches.bash    # CUDA architecture auto-detection
│   └── run_kegalign            # Shell wrapper for kegalign execution
│       │
│       └── mps-mig/            # Multi-GPU utilization scripts
│           ├── run_mig.py      # MPS/MIG scheduling with UID-based completion tracking
│           ├── split_input.py  # Genome partitioning (LPT heap bin-packing)
│           ├── run_kegalign_symlink_sort  # Shell wrapper for MIG mode
│           └── mypy.ini
│
└── test-data/
    ├── apple.fasta.gz          # Reference genome (test)
    ├── orange.fasta.gz         # Query genome (test)
    └── apple_orange.maf.gz     # Expected MAF output (test)
```

### FILE INDEX (Priority Order)

| # | Priority | Path | Type | Notes |
|---|----------|------|------|-------|
| 1 | ★★★ | `src/main.cpp` | C++ | Entry point, TBB graph, argument parsing |
| 2 | ★★★ | `src/graph.h` | C++ header | ALL shared data structures |
| 3 | ★★★ | `src/seed_filter.cu` | CUDA | Core GPU kernels |
| 4 | ★★★ | `common/seed_pos_table.cu` | CUDA | Seed index building |
| 5 | ★★★ | `common/seed_filter_interface.cu` | CUDA | Multi-GPU init |
| 6 | ★★★ | `src/seeder.cpp` | C++ | CPU-side k-mer generation |
| 7 | ★★★ | `src/segment_printer.cpp` | C++ | LASTZ command output |
| 8 | ★★★ | `scripts/runner.py` | Python | Main Galaxy orchestration |
| 9 | ★★ | `common/DRAM.cpp/.h` | C++ | Memory buffer management |
| 10 | ★★ | `common/ntcoding.cpp/.h` | C++ | Nucleotide encoding |
| 11 | ★★ | `scripts/diagonal_partition.py` | Python | HSP optimization |
| 12 | ★★ | `scripts/mps-mig/run_mig.py` | Python | Multi-GPU scheduling |
| 13 | ★★ | `scripts/mps-mig/split_input.py` | Python | Genome partitioning |
| 14 | ★ | `common/dna_utilities.c/.h` | C | Scoring matrix library |
| 15 | ★ | `scripts/run_lastz_tarball.py` | Python | LASTZ execution |
| 16 | ★ | `scripts/package_output.py` | Python | Output packaging |
| 17 | ★ | `common/parameters.h` | C header | Constants |
| 18 | ★ | `common/cuda_utils.h` | C++ header | CUDA error helpers |
| 19 | ★ | `common/kseq.h` | C header | FASTA parser |
| 20 | ★ | `common/utilities.c/.h` | C | General utilities |

---

## 4. Data Structures Reference

All core data structures are defined in `src/graph.h`.

### `Seed_config` (graph.h:16–21)

```cpp
struct Seed_config {
    std::string shape;   // e.g. "TTT0T00TT00T0T0TTTT" (12of19) or "TTTT0T0TT0011T0T0TTTTT" (14of22)
    int size;            // Total seed span (19 or 22)
    int kmer_size;       // Number of matching positions (12 or 14)
    bool transition;     // Allow one A↔G or C↔T transition in hits
};
```

### `segmentPair` (graph.h:23–28)

```cpp
struct segmentPair {
    uint32_t ref_start;    // Position in reference sequence
    uint32_t query_start;  // Position in query sequence
    uint32_t len;          // Length of ungapped segment
    int score;             // X-drop alignment score
};
```
This is the **primary output** of the GPU pipeline. Every `segmentPair` represents one HSP.

### `Configuration` (graph.h:30–75)

```cpp
struct Configuration {
    // Input
    std::string reference_filename, query_filename, data_folder;
    // Scoring
    std::string scoring_file;
    bool ambiguous_set; std::string ambiguous;
    int sub_mat[NUC2];   // 64-entry substitution matrix (8×8 nucleotide codes)
    // Seeding
    Seed_config seed;
    std::string seed_shape;
    int step;            // Stride through reference for k-mer extraction
    // Filter
    int xdrop;           // X-drop threshold for ungapped extension (default 910)
    int hspthresh;       // Minimum score for HSPs (default 3000)
    bool noentropy;      // Skip entropy-based score adjustment
    // Gapped extension
    bool gapped;
    int ydrop, gappedthresh, inner;
    bool notrivial;
    // Output
    std::string output_format, output;
    std::string target_prefix, query_prefix;
    bool markend;
    // System
    int wga_chunk_size;     // GPU chunk size for X-drop (default 250,000)
    int lastz_interval_size; // LASTZ interval size (default 10,000,000)
    int seq_block_size;     // Sequence block size (default 500,000,000)
    int num_gpu;            // Number of GPUs (-1 = all)
    int num_threads;        // Thread limit (-1 = all)
    bool debug;
};
```

One **global instance** `Configuration cfg` is declared in `graph.h` and populated in `main.cpp`.

### `seq_block` (graph.h:79–86)

```cpp
struct seq_block {
    int r_index;       // Reference block index
    int q_index;       // Query block index
    size_t r_start;    // Reference block start (absolute byte offset in DRAM)
    size_t q_start;    // Query block start
    uint32_t r_len;    // Reference block length in bytes
    uint32_t q_len;    // Query block length in bytes
};
```

### `seed_interval` (graph.h:88–94)

```cpp
struct seed_interval {
    uint32_t start;           // Query interval start (within block)
    uint32_t end;             // Query interval end
    uint32_t num_invoked;     // Current invocation index for this block
    uint32_t num_intervals;   // Total intervals in this block
    uint32_t buffer;          // GPU buffer slot ID (0..BUFFER_DEPTH-1)
};
```

### TBB Type Aliases (graph.h:95–102)

```cpp
using hsp_output        = std::vector<segmentPair>;
using seeder_payload    = std::tuple<seq_block, seed_interval>;
using printer_payload   = std::tuple<seeder_payload, hsp_output, hsp_output>;  // fwd + rc HSPs
using seeder_input      = std::tuple<seeder_payload, size_t>;   // payload + ticket token
using printer_input     = std::tuple<printer_payload, size_t>;
using printer_node      = tbb::flow::multifunction_node<printer_input, std::tuple<size_t>>;
```

### `seeder_body` (graph.h:103–110)

```cpp
struct seeder_body {
    // Atomic statistics counters (shared across all TBB threads)
    static std::atomic<long long> num_seed_hits;
    static std::atomic<long long> num_seeds;
    static std::atomic<long long> num_hsps;
    static std::atomic<long long> total_xdrop;
    static std::atomic<long long> num_seeded_regions[BUFFER_DEPTH];
    
    printer_input operator()(seeder_input input);
};
```

### `DRAM` class (common/DRAM.h)

```cpp
class DRAM {
public:
    char* buffer;              // 64-byte aligned allocation via TBB scalable_aligned_malloc
    std::size_t size;          // Fixed at ~6 GB
    std::size_t seqSize;       // Used portion size
    std::size_t bufferPosition; // Current write position
};
```

Three global instances:
- `DRAM *ref_DRAM` — reference (target) sequence
- `DRAM *query_DRAM` — query sequence (forward strand)
- `DRAM *query_rc_DRAM` — query reverse complement

---

## 5. Feature-by-Feature Analysis

### 5.1 Seed Generation & Indexing

**Business Purpose:** Build a fast lookup table mapping k-mer seeds to their positions in the reference genome, enabling O(1) seed hit lookup on the GPU.

**Entry Point:** Called from `main.cpp` after reading the reference sequence:
```cpp
GenerateSeedPosTable(ref_DRAM->buffer, cfg.seed.shape, ref_len, cfg.step, ...);
SendSeedPosTable();  // Upload index + pos tables to all GPUs
```

**Implementation File:** `common/seed_pos_table.cu`

**Algorithm:**

1. **Index Table Size:** `2^(2 × kmer_size) + 1` entries (one per possible kmer value)
   - For 12of19: kmer_size=12 → 2^24 + 1 = 16,777,217 entries
   - For 14of22: kmer_size=14 → 2^28 + 1 = 268,435,457 entries

2. **TBB Parallel Histogram (atomic increments):**
   ```cpp
   tbb::parallel_for(size_t(0), num_steps, [&](size_t i) {
       uint32_t index = GetKmerIndexAtPos(buffer, shape, i * step, ...);
       if (index != INVALID_KMER)
           __sync_fetch_and_add(&index_table[index + 1], 1);  // lock-free atomic
   });
   ```

3. **Prefix Sum:** `InclusivePrefixScan(index_table)` converts histogram to cumulative offsets. After this, `index_table[k]` = start position in `pos_table` for kmer value `k`.

4. **Position Assignment:**
   ```cpp
   for each position i at step:
       pos_table[index_table[kmer] + atomic_offset++] = i * step;
   ```

5. **Upload to all GPUs:** `SendSeedPosTable()` in `seed_filter_interface.cu` copies both `index_table` and `pos_table` to each device.

**Key Constants:**
- `GRAIN_SIZE = 262144` (1<<18) — TBB grain size for parallel_for
- `INVALID_KMER = 0x80000000` — returned for positions with ambiguous nucleotides (N, X)

**Interaction with Other Features:**
- Built once per reference sequence, shared across all query interval iterations
- `d_index_table[device]` and `d_pos_table[device]` on GPU are queried in every `find_hits` kernel launch

---

### 5.2 GPU Seed Filtering & X-drop Extension

**Business Purpose:** Rapidly find all seed hits between reference and query, then extend them into longer ungapped alignments (HSPs) using GPU parallelism, filtering out low-quality matches.

**Entry Point:** Called from `seeder.cpp` via function pointer:
```cpp
g_SeedAndFilter(seed_offset_vector, rev=false, buffer);  // forward strand
g_SeedAndFilter(seed_offset_vector, rev=true,  buffer);  // reverse complement
```

**Implementation File:** `src/seed_filter.cu`

**GPU Pipeline (5 CUDA kernels):**

```
seed_offsets ──► compress_string_rev_comp ──► find_num_hits ──► find_hits ──► find_hsps ──► compress_output
```

See [`assets/gpu_pipeline.mmd`](assets/gpu_pipeline.mmd) for visual diagram.

#### Kernel 1: `compress_string_rev_comp`
- **Input:** Raw query sequence (ASCII)
- **Output:** 2-bit encoded query + reverse complement in `d_query_seq`, `d_query_rc_seq`
- **Encoding:** A→0, C→1, G→2, T→3, N→4 (ambiguous), lowercase→5, &→6 (separator), others→invalid
- **Thread assignment:** Grid-stride loop, one thread per position

#### Kernel 2: `find_num_hits`
- **Input:** `d_seed_offsets` (packed as `kmer_index << 32 | query_pos`)
- **Output:** `d_hit_num_array` — count of reference positions matching each seed
- **Algorithm:** For each seed, lookup `index_table[kmer_index+1] - index_table[kmer_index]`

#### Thrust: `inclusive_scan` on `d_hit_num_array`
- Converts per-seed counts to cumulative offsets for efficient parallel scatter

#### Kernel 3: `find_hits`
- **Output:** `d_hsp` array with raw (ref_pos, query_pos) candidates
- **Warp parallelism:** Each warp handles one seed; threads distribute over reference hits
- **Transition seeds:** For each seed with `transition=true`, also check one-base-off positions (A↔G, C↔T)

#### Kernel 4: `find_hsps` — Core X-drop Extension
- **Warp parallelism:** NUM_WARPS warps per block; each warp handles one HSP candidate
- **Right extension algorithm (tiles of warp_size=32 positions):**
  ```
  for each tile of 32 positions:
      warp thread i computes score for position tile_start + i
      warp_reduce via __shfl_up to accumulate running score
      track max_score and current_score
      if max_score - current_score > xdrop: terminate
  ```
- **Left extension:** Same logic, backwards from hit position
- **Entropy filtering:** If `!noentropy`, compute nucleotide composition and scale score by Shannon entropy (reduces repetitive sequence HSPs)
- **Threshold:** HSP kept if `score >= hspthresh`

#### Kernel 5: `compress_output`
- Compacts sparse HSP array by removing invalid entries (score < hspthresh)
- Input: `d_done` flags (1=keep) + `d_hsp` values
- Output: Dense `d_hsp_reduced` array

#### Post-Processing (Thrust on GPU):
1. `thrust::sort` by diagonal (ref_start - query_start), then ref_start
2. `thrust::unique_copy` using `hspEqual` comparator — removes duplicate HSPs

**Multi-GPU Resource Pooling:**
```cpp
std::mutex mu;
std::condition_variable cv;
std::vector<int> available_gpus;  // Global pool

// In SeedAndFilter():
std::unique_lock<std::mutex> lk(mu);
cv.wait(lk, []{ return !available_gpus.empty(); });
int device = available_gpus.back(); available_gpus.pop_back();
// ... do GPU work ...
available_gpus.push_back(device);
cv.notify_one();
```

**Key Limits:**
- `MAX_HITS_PER_GB = 4,194,304` — HSP array allocation scales with GPU memory
- Overflow handling: If hits exceed capacity, iterates in batches

---

### 5.3 TBB Flow Graph Concurrency

**Business Purpose:** Maximize CPU/GPU utilization by processing multiple reference × query block pairs concurrently while preventing GPU memory exhaustion.

**Entry Point:** `main.cpp` lines ~585–650

**Graph Nodes:**

| Node | Type | Role |
|------|------|------|
| `reader` | `source_node<seeder_payload>` | Iterates all ref_block × query_interval combinations |
| `ticketer` | `buffer_node<size_t>` | Token pool with N=`num_threads` tokens |
| `gatekeeper` | `join_node<tuple<seeder_payload, size_t>>` | Gates execution until token available |
| `seeder` | `function_node<seeder_input, printer_input>` | GPU seed+filter, one token consumed |
| `printer` | `multifunction_node<printer_input, tuple<size_t>>` | Output + return token |

**Token Lifecycle:**
1. `ticketer` is pre-loaded with `num_threads` size_t tokens (values 0..N-1)
2. When both a payload and a token are available, `gatekeeper` forwards to `seeder`
3. `seeder` processes the GPU task (may block waiting for a free GPU device)
4. `printer` writes output, then pushes token back to `ticketer`

**Concurrency invariant:** At most `num_threads` seeder tasks execute simultaneously.

**Block Iteration Strategy (in `reader`):**
```cpp
// For each reference block r:
//   For each query block q:
//     compute num_intervals = ceil(q_len / lastz_interval_size)
//     for i in 0..num_intervals:
//       emit seeder_payload{r_block, q_interval_i}
```

The query is processed in intervals of `lastz_interval_size` (default 10M bases) to match LASTZ's expected segment granularity.

---

### 5.4 LASTZ Command Generation (Segment Printer)

**Business Purpose:** Translate GPU-computed HSPs into LASTZ-compatible command lines and segment files, which LASTZ then uses to perform gapped alignment within each HSP region.

**Entry Point:** `segment_printer_body::operator()` in `src/segment_printer.cpp`

**Inputs (from printer_payload):**
- `seq_block` — reference and query block indices/positions
- `seed_interval` — query interval start/end
- `hsp_output fwd_hsps` — forward strand HSPs
- `hsp_output rc_hsps` — reverse complement HSPs

**Segment File Format (tab-delimited):**
```
ref_chr_name  ref_start  ref_end  query_chr_name  query_start  query_end  strand  score
```
- Strand is `f` (forward) or `r` (reverse)
- Positions are absolute genome coordinates (not relative to blocks)

**Coordinate Mapping:**
- Block-relative → absolute: uses `upper_bound` binary search over chromosome start arrays (`r_chr_start`, `q_chr_start`)
- Reverse complement: `query_start = chr_len - (rc_start + len)` to convert RC coordinates back to forward

**LASTZ Command Construction:**
```
lastz ref.2bit[namestart..nameend] query.2bit[namestart..nameend]
  --segments=path/to/tmp.seg
  [--format=FORMAT] [--ydrop=VALUE] [--gappedthresh=VALUE]
  [--scoring=FILE] [--ambiguous=TYPE] [--noentropy] [--notrivial]
  [--strand=plus|minus|both] [--inner=VALUE]
  > path/to/output.maf-
  2> path/to/output.err
```

**File naming convention:**
- Segment files: `{data_folder}/tmp{block_index}.{q_block}.r{r_block}.{strand}.seg`
- Output files: `{data_folder}/tmp{block_index}.{q_block}.r{r_block}.{strand}.maf-`

**Thread safety:** All `printf` calls are protected by a global mutex to prevent interleaved output lines.

---

### 5.5 Diagonal Partitioning

**Business Purpose:** Improve LASTZ execution speed by sorting and chunking HSP segment files so that adjacent HSPs share diagonal proximity, maximizing LASTZ's internal cache reuse.

**Entry Point:** `scripts/diagonal_partition.py` — invoked as a subprocess for each LASTZ command

**Algorithm:**

1. **Parse segment file:** Read HSP records, compute midpoint:
   ```python
   seq1_mid = (seq1_start + seq1_end) / 2
   seq2_mid = (seq2_start + seq2_end) / 2
   ```

2. **Group by chromosome pair:** `(seq1_name, seq2_name)` → list of HSPs

3. **Sort by diagonal:**
   - Forward (`direction='f'`): Sort key = `seq1_mid + seq2_mid` (main diagonal)
   - Reverse (`direction='r'`): Sort key = `seq1_mid - seq2_mid` (anti-diagonal)
   
   **Why:** LASTZ sweeps through diagonals of the alignment matrix. Presenting HSPs in diagonal order allows it to load each genome region once instead of repeatedly.

4. **Chunking:** Split sorted HSPs into chunks of `chunk_size` (default 50,000):
   - Each chunk → new `.seg.splitN` file
   - Modified LASTZ command: `--segments=file.seg.splitN --output=out.maf-.splitN`

5. **Adaptive chunk sizing:** `runner.py` calls `estimate_chunk_size()` which uses `statistics.quantiles()` on existing segment file sizes to predict an appropriate chunk size.

**Key Constants:**
```python
MIN_CHUNK_SIZE = 5000
MAX_CHUNK_SIZE = 50000
DELETE_AFTER_CHUNKING = True  # Remove original .seg file
```

**Interaction:** `runner.py` creates `diagonal_partition_worker` threads that each invoke `diagonal_partition.py` on one LASTZ command.

---

### 5.6 Galaxy / Runner Orchestration

**Business Purpose:** Provide a Galaxy-compatible wrapper that handles the full KegAlign → Diagonal Partition → LASTZ pipeline with Galaxy tool calling conventions, error handling, and output file management.

**Entry Point:** `scripts/runner.py` — invoked by Galaxy or command-line

**Key Arguments:**
```
runner.py [options] target.fasta.gz query.fasta.gz
  --diagonal-partition   Enable HSP partitioning before LASTZ
  --format FORMAT        Output format (maf-, axt, sam, etc.)
  --num-cpu N            Worker process count
  --num-gpu N            GPU count passed to kegalign
  --output-file FILE     Output file path
  --output-type tarball|output  Packaging mode
  --tool_directory PATH  Location of scripts
```

**Pipeline Steps:**
1. Run `kegalign` binary (via `run_kegalign()` subprocess)
2. Parse stdout → `LastzCommands` registry (deduplicates via set)
3. If `--diagonal-partition`: spawn `ProcessPoolExecutor` workers running `diagonal_partition.py`
4. If `--output-type=output`: run LASTZ workers in parallel
5. Concatenate MAF results; write output file

**Key Classes in runner.py:**

| Class | Purpose |
|-------|---------|
| `LastzCommands` | Deduplicating registry of LASTZ command strings |
| `LastzCommand` | Parser for individual LASTZ command lines (regex-based) |
| `KegAlignSegments` | Dict of unique segment files (by filename) |
| `KegAlignSegment` | Parses segment file naming convention |

**Segment Filename Parsing (KegAlignSegment):**
Pattern: `tmp{tmp}_{block}.r{r}.{strand}[.split{split}].seg`
- `tmp` — reference block group index
- `block` — query block index
- `r` — reference block index within group
- `strand` — `plus` or `minus`
- `split` — optional split index (from diagonal partition)

---

### 5.7 MPS/MIG GPU Scheduling

**Business Purpose:** Maximize GPU utilization on multi-GPU or partitioned-GPU (MIG) systems by running multiple KegAlign instances concurrently with smart scheduling and failure recovery.

**Entry Point:** `scripts/mps-mig/run_mig.py`

**Two Modes:**

**MIG (Multi-Instance GPU):**
- NVIDIA MIG divides one physical GPU into isolated GPU instances
- Each instance has dedicated memory and compute
- `--MIG "GPU0,MIG1g.5gb,MIG1g.5gb"` — list of device names

**MPS (Multi-Process Service):**
- Multiple KegAlign processes share one GPU via CUDA MPS daemon
- `--MPS "2,2"` — processes per device
- One MPS daemon per device, with separate pipe directories

**UID-based Completion Tracking:**
- Each KegAlign invocation gets a unique ID (UID) = timestamp + random suffix
- External script writes `{uid_folder}/{uid}` file on completion
- `gpu_queue.check_completion()` scans for UID files to detect completion
- **Why UID files instead of process PIDs:** PID reuse can cause false completion detection; file existence is unambiguous

**Scheduling Algorithm:**
```python
while pairs_remaining:
    free_devices = gpu_queue.get_free_device_list()
    for device in free_devices:
        while free_slots[device] > 0:
            query, target = pairs.pop()
            uid = generate_uid()
            cmd = build_kegalign_cmd(query, target, device, uid, ...)
            process_list.append(Popen(cmd))
            gpu_queue.submit(uid, device)
            free_slots[device] -= 1
    
    completed = gpu_queue.check_completion()
    failed = process_list.get_fails_and_check_completion()
    if memory_failed and --resubmit_fails:
        pairs.add_back(failed_pairs)
```

**Error Classification:**
- `mem_err`: CUDA OOM errors → resubmit if `--resubmit_fails`
- `other_err`: Core dumps, bad_alloc, open file errors → fatal

**Cartesian Product Task Generation:**
```python
pairs = list(product(query_block_files, target_block_files))
```
All query chunks × all target chunks = N×M alignment tasks.

---

### 5.8 Genome Input Partitioning

**Business Purpose:** Split large genomes into balanced chunks for MPS/MIG parallel processing, minimizing wall-clock time by equalizing work across GPU instances.

**Entry Point:** `scripts/mps-mig/split_input.py`

**Algorithm: Longest Processing Time First (LPT) Bin Packing**

1. Sort sequences by length (descending) — longest first
2. Initialize min-heap of N bins (each starts at size 0)
3. For each sequence:
   ```python
   size, bin_id = heappop(bins)
   assign sequence to bin_id
   heappush(bins, (size + seq.length, bin_id))
   ```
4. **Rationale:** LPT is a well-known approximation algorithm for makespan minimization. It achieves ≤ 4/3 × optimal makespan.

**Adaptive Bin Count (MSE optimization):**
```python
for num_bins in 1..max_chunks:
    bin_sizes = split_chr(fasta, num_bins)
    loss = mean_squared_error(bin_sizes, goal_bp)  # goal_bp = target bp per bin
    if loss < best_loss:
        best_bins = num_bins
```

**Output:**
- `{out}/chunk_0`, `{out}/chunk_1`, ... — FASTA format
- If `--to_2bit`: Each chunk also converted to `.2bit` via `faToTwoBit` (parallel)

---

### 5.9 LASTZ Tarball Execution

**Business Purpose:** Enable Galaxy-containerized execution of LASTZ commands packaged in a tarball, decoupling the KegAlign seeding stage from LASTZ alignment in Galaxy workflows.

**Entry Point:** `scripts/run_lastz_tarball.py`

**Tarball Format (created by package_output.py):**
```
tarball.tgz/
├── galaxy/
│   ├── commands.json     # JSONL: one command per line
│   ├── format.txt        # Output format string
│   └── files/
│       ├── seq1.2bit
│       ├── seq2.2bit
│       └── segments/*.seg
```

**Execution:**
```python
with ProcessPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(run_command, cmd) for cmd in batch_commands()]
```

Each `run_command`:
1. Runs `lastz --allocate:traceback=1.99G [args]`
2. Accepts return codes 0 and 1 (LASTZ returns 1 for empty output — benign)
3. Validates stderr: only "truncating" messages are acceptable
4. Returns elapsed time

**Output Handling:**
- If format is MAF: prepend `##maf version=1` header to concatenated output
- Writes `galaxy.json` with output metadata for Galaxy

---

### 5.10 Output Packaging

**Business Purpose:** Bundle LASTZ commands and all referenced data files into a portable tarball for Galaxy distribution or delayed execution.

**Entry Point:** `scripts/package_output.py`

**Key Classes:**

`PackageFile` — Creates `galaxy/commands.json` + `galaxy/format.txt` + referenced data files in tarball:
```python
class PackageFile:
    def add_file(path):  # Copy file → galaxy/files/
    def add_config(cmd_dict):  # Append to commands.json
    def add_format(fmt):  # Write format.txt
    def close():  # Finalize tar archive
```

`bashCommandLineFile` — Parses raw LASTZ command lines using `bashlex`:
- Extracts target/query 2bit files, segment files, scoring files
- Detects output format: BAM, MAF, interval/differences, tabular
- Produces `commands.json` with relative file paths inside package

**LASTZ Argument Schema:** `scripts/lastz-cmd.ini` defines:
- `flag_args`: boolean flags (no value)
- `str_args`: string-valued arguments
- `int_args`: integer-valued arguments
- `bool_str_args`, `bool_int_args`: dual-mode arguments

---

## 6. Cross-Feature Interaction Map

```
                    ┌──────────────────────────────────────────┐
                    │          Feature Interactions            │
                    └──────────────────────────────────────────┘

split_input.py ──────────────────────────────────────────────► run_mig.py
(partition genomes)                                            (schedule tasks)
                                                                     │
                                                            ┌────────▼─────────────────────┐
                                                            │  kegalign binary             │
                                                            │  (one instance per GPU task) │
                                                            └──────────┬───────────────────┘
                                                                       │ stdout: LASTZ cmds
                                                       ┌───────────────▼──────────────┐
runner.py ───────────────────────────────────────────► │  diagonal_partition.py       │
(orchestrate, Galaxy API)                              │  (optimize HSP ordering)     │
                                                       └───────────────┬──────────────┘
                                                                       │
                                          ┌────────────────────────────▼──────────────┐
                                          │  LASTZ binary                             │
                                          │  (gapped alignment on segments)           │
                                          └────────────────────────────┬──────────────┘
                                                                        │ .maf- output
                                    ┌───────────────────────────────────▼─────────────┐
                                    │  package_output.py / run_lastz_tarball.py       │
                                    │  (tarball for Galaxy / direct execution)        │
                                    └─────────────────────────────────────────────────┘

Internal to kegalign binary:

main.cpp ──► seed_pos_table.cu ──► (index uploaded to GPUs) ──┐
    │                                                         │
    └──► TBB Flow Graph ──► seeder.cpp ──► seed_filter.cu ────┘
                                    (uses GPU index to find hits)
                                               │
                                    segment_printer.cpp
                                    (writes .seg + LASTZ command)
```

**Shared Resources:**
- `Configuration cfg` (global): read by seeder.cpp, segment_printer.cpp, seed_pos_table.cu
- `DRAM *ref_DRAM / query_DRAM / query_rc_DRAM` (global): written by main.cpp, read by seeder.cpp
- Chromosome metadata vectors in `store.h` (global): written by main.cpp, read by segment_printer.cpp
- GPU device pool `available_gpus` (global mutex): used by all SeedAndFilter calls
- `sub_mat[NUC2]` in Configuration: loaded in main.cpp from dna_utilities, passed to GPU via InitializeProcessor

---

## 7. GPU Memory Management

### Host Memory (CPU)

Three `DRAM` instances, each 6GB, allocated via `TBB scalable_aligned_malloc` (64-byte alignment):

| Buffer | Content | Size |
|--------|---------|------|
| `ref_DRAM` | Reference (target) sequence in 8-bit nucleotide codes | up to 6GB |
| `query_DRAM` | Query sequence (forward strand) | up to 6GB |
| `query_rc_DRAM` | Query reverse complement | up to 6GB |

The 6GB limit matches typical reference genome sizes. If genomes exceed 6GB, they must be blocked (via `seq_block_size`).

### GPU Memory (Device)

Per-GPU allocations in `seed_filter.cu`:

| Variable | Size | Purpose |
|----------|------|---------|
| `d_ref_seq[device]` | `ref_len` bytes | Compressed reference (2-bit) |
| `d_query_seq[buffer][device]` | `wga_chunk_size` | Compressed query chunk |
| `d_query_rc_seq[buffer][device]` | `wga_chunk_size` | Compressed RC query chunk |
| `d_index_table[device]` | `2^(2k)+1` × 4 bytes | Seed → offset mapping |
| `d_pos_table[device]` | `ref_len/step` × 4 bytes | Seed → positions |
| `d_seed_offsets` | `MAX_SEEDS × 8` bytes | Input k-mer pairs |
| `d_hsp[buffer]` | `MAX_HITS × 16` bytes | HSP output |
| `d_sub_mat[device]` | `NUC2 × 4` bytes = 256 bytes | Substitution matrix |

`MAX_HITS_PER_GB = 4,194,304`: GPU memory is detected at runtime and used to scale the HSP array size appropriately.

### Memory Error Recovery

In `run_mig.py`, CUDA OOM errors (`mem_err`) can trigger task resubmission:
```python
if failure_type == "mem_err" and args.resubmit_fails:
    pairs.append(failed_pair)  # Add back to queue
```

### CUDA Exit Codes (cuda_utils.h)

| Exit Code | Error |
|-----------|-------|
| 11 | `cudaSetDevice` failed |
| 12 | `cudaMalloc` failed |
| 13 | `cudaMemcpy` failed |
| 14 | `cudaFree` failed |

---

## 8. Nucleotide Encoding & Scoring

### Nucleotide Encoding (parameters.h + ntcoding.h)

8 nucleotide codes (NUC=8, NUC2=NUC×NUC=64):

| Index | Constant | Meaning |
|-------|----------|---------|
| 0 | `A_NT` | Adenine |
| 1 | `C_NT` | Cytosine |
| 2 | `G_NT` | Guanine |
| 3 | `T_NT` | Thymine |
| 4 | `L_NT` | Lowercase (soft-masked) threshold |
| 5 | `N_NT` | Ambiguous N |
| 6 | `X_NT` | IUPAC ambiguity codes |
| 7 | `E_NT` | End/error marker |

The substitution matrix `sub_mat[NUC2]` is indexed as `sub_mat[nuc_a * NUC + nuc_b]` — a full 8×8 symmetric scoring matrix.

### Spaced Seeds (ntcoding.cpp)

**Default Seeds:**

| Name | Pattern | Span | Matches |
|------|---------|------|---------|
| `12of19` | `TTT0T00TT00T0T0TTTT` | 19 | 12 |
| `14of22` | `TTTT0T0TT0011T0T0TTTTT` | 22 | 14 |
| custom | binary string | variable | count of '1' |

**`T` = must match, `0` = don't care (transition allowed if `--transition`)**

**`GetKmerIndexAtPos()` algorithm:**
1. Read `shape_size` characters from position
2. For positions where shape='T': extract nucleotide, left-shift accumulator, add nucleotide
3. If any character maps to invalid: return `INVALID_KMER`
4. Return accumulated `kmer_index` (up to 28 bits for 14of22)

**Transition Matching:**
- `TRANSITION_MASK = 2` (bit 1 of nucleotide code)
- A↔G: bits 00↔10 (differ only in bit 1)
- C↔T: bits 01↔11 (differ only in bit 1)
- `IsTransitionAtPos()`: returns true if transition would occur
- When `transition=true`, for each seed with a transition position, an alternative kmer is generated where that position flips bit 1

### Scoring Matrix (dna_utilities.c)

**Default: HOXD70** (Chiaramonte et al. 2002, Human-Mouse comparison):
- Gap open penalty: 400
- Gap extension penalty: 30
- Substitution matrix: empirically derived from highly conserved human-rodent regions

**Custom scoring:** Use `--scoring path/to/file` in LASTZ format. `load_scoring_matrix()` in `scoring.c` reads via `read_score_set()` from dna_utilities.

**Entropy adjustment** (`find_hsps` kernel):
- Compute nucleotide frequencies A/C/G/T in extended region
- Shannon entropy: `H = -Σ p_i * log2(p_i)`
- If `score < hspthresh + some_delta`: scale `score *= entropy / 2.0`
- Prevents low-complexity repeats (AAAA...) from generating false HSPs even if score passes threshold

---

## 9. Nuances, Subtleties & Gotchas

### G1: Global Variable Architecture
`Configuration cfg`, `DRAM *ref_DRAM`, chromosome metadata vectors — all are defined in `main.cpp` as **global non-const variables** and declared extern in `store.h`/`graph.h`. There is no dependency injection. Any function in any translation unit can read these at any time. This is safe because they are all written in `main.cpp` during initialization, before TBB graph execution begins, and are read-only during graph execution.

**Implication for changes:** If you add new global state, ensure it is fully populated before `g.wait_for_all()` in main.cpp.

### G2: Buffer Depth (BUFFER_DEPTH)
Multiple GPU buffers exist (indexed `0..BUFFER_DEPTH-1`). The `seed_interval.buffer` field selects which buffer slot the query upload occupies. This enables pipelining: one buffer can be uploaded while another is being processed. **If you change BUFFER_DEPTH, adjust all arrays indexed by it.**

### G3: Ticket Token Values
Ticket tokens in the TBB graph are `size_t` values (0..N-1). They are **not used for data** — only for flow control. The value itself has no meaning; only its presence/absence matters for the join_node.

### G4: Reverse Complement Coordinate Conversion
RC coordinates require careful inversion. In `seeder.cpp`:
```cpp
rc_q_inter_start = q_block_len - q_inter_end;
rc_q_inter_end   = q_block_len - q_inter_start;
```
In `segment_printer.cpp`, when printing RC HSPs:
```cpp
// rc_start in RC buffer → forward coordinates
query_start = chr_len - (rc_start + hsp.len)
query_end   = chr_len - rc_start
```
**Getting these wrong produces silently incorrect coordinates in output.**

### G5: LASTZ Return Code Tolerance
`run_lastz_tarball.py` accepts LASTZ return code **1** as success (in addition to 0). LASTZ returns 1 when it produces no alignments for a segment — this is a valid outcome, not an error.

### G6: Segment File Deduplication
`LastzCommands.add()` in `runner.py` uses a Python set to deduplicate LASTZ commands. This prevents processing the same segment pair twice if the KegAlign pipeline emits duplicates (which can happen at block boundaries).

### G7: kseq.h Macro Template
`kseq.h` uses C macro metaprogramming to generate type-specific FASTA/FASTQ readers. In `main.cpp`:
```cpp
KSEQ_INIT(gzFile, gzread)  // Generates kseq_t, kseq_init, kseq_read for gzFile
```
This is a header-only approach common in C bioinformatics tools. Do not call `kseq_init` without first calling `KSEQ_INIT` with matching types.

### G8: TBB 2020.2 API
KegAlign uses **oneTBB 2020.2**, which has some deprecated APIs. The `TBB_SUPPRESS_DEPRECATED_MESSAGES` compile flag suppresses related warnings. The TBB flow graph API changed significantly between TBB 4.x and 2020.x — `source_node` was introduced in 2020.x. Do not assume older TBB API compatibility.

### G9: Thrust on CPU vs GPU
`seed_pos_table.cu` uses Thrust for `inclusive_scan` **on CPU** (host vectors). `seed_filter.cu` uses Thrust **on GPU** (device vectors). Both use the same Thrust API but very different execution backends. Check the namespace/type (`thrust::host_vector` vs `thrust::device_vector`) to distinguish.

### G10: faToTwoBit `-namePrefix` Patch
`add-option.patch` adds a `-namePrefix=XX.` option to the UCSC `faToTwoBit` utility. This is applied to the local build of faToTwoBit. The patch is necessary for cases where chromosome names in 2bit files need a prefix (e.g., `hg38.chr1` instead of `chr1`). Without this patch, the option doesn't exist.

### G11: MPS Pipe Directory Isolation
In `run_mig.py`, each MIG device gets its own MPS pipe directory:
```python
pipe_dir = f"{args.mps_pipe_dir}/{device_name}"
```
This is required by CUDA MPS — multiple daemons cannot share a pipe directory. If you add devices, ensure each has a unique subdirectory.

### G12: HSP Overflow Batching
In `seed_filter.cu`, if the number of seed hits exceeds `MAX_HITS` (scaled by GPU memory), the hit processing iterates in batches:
```cpp
while (remaining_hits > 0):
    process min(remaining_hits, MAX_HITS) hits
    remaining_hits -= processed
```
**Performance implication:** Batching adds latency. Setting `wga_chunk_size` too large can trigger this.

### G13: Diagonal vs Anti-diagonal Sorting
`diagonal_partition.py` sorts:
- Forward HSPs by `seq1_mid + seq2_mid` (runs along main diagonal, constant slope +1)
- Reverse HSPs by `seq1_mid - seq2_mid` (runs along anti-diagonal, constant slope -1)

This matches LASTZ's internal memory access pattern. Grouping HSPs on the same diagonal means LASTZ loads each genome region once and processes multiple HSPs in one pass.

### G14: Segment File Cleanup
`diagonal_partition.py` sets `DELETE_AFTER_CHUNKING = True`, which **deletes the original `.seg` file** after splitting it into `.split0`, `.split1`, etc. If the partition script fails mid-way, you may have missing segment files. Recovery requires re-running KegAlign.

### G15: Scoring Matrix Dimensions
`sub_mat` is declared as `int sub_mat[NUC2]` where `NUC2 = NUC * NUC = 64`. However, `dna_utilities.h` uses a different encoding for full ambiguity support. The `load_scoring_matrix()` function bridges between the two systems. When modifying scoring, ensure consistency between `parameters.h` constants and `dna_utilities.h` definitions.

---

## 10. Command-Line Reference

### `kegalign` (compiled binary)

```
Usage: kegalign <target> <query> <data_folder> [options]

Positional:
  target              Target/reference FASTA.gz file path
  query               Query FASTA.gz file path
  data_folder         Output folder for segment files (must end with /)

Sequence Options:
  --strand STRAND     Strand to search: plus | minus | both (default: both)

Scoring Options:
  --scoring FILE      LASTZ-format scoring file
  --ambiguous TYPE    Ambiguous nucleotide handling: n | iupac

Seeding Options:
  --seed PATTERN      Seed pattern: 12of19 | 14of22 | custom_binary (default: 12of19)
  --step N            Stride between seed positions (default: 1)
  --notransition      Disable one-transition seed hits (default: transitions allowed)

Ungapped Extension:
  --xdrop N           X-drop threshold (default: 910)
  --hspthresh N       Minimum HSP score threshold (default: 3000)
  --noentropy         Disable entropy-based score scaling

Gapped Extension:
  --nogapped          Disable gapped extension stage
  --ydrop N           Y-drop threshold (default: 9430)
  --gappedthresh N    Gapped alignment score threshold (default: hspthresh)
  --notrivial         Exclude trivial self-alignment blocks
  --inner N           Inner alignment search interval (-1 = disabled)

Output Options:
  --format FORMAT     Output format (default: maf-)
                      Options: lav, lav+text, axt, axt+, maf, maf+, maf-,
                               sam, softsam, sam-, softsam-, cigar, BLASTN,
                               differences, rdotplot, text
  --output FILE       Output filename
  --target_prefix P   Prefix for target sequence names
  --query_prefix P    Prefix for query sequence names
  --markend           Write marker line before completion

System Options:
  --wga_chunk_size N  GPU chunk size (default: 250000)
  --lastz_interval_size N  LASTZ interval size (default: 10000000)
  --seq_block_size N  Sequence block size (default: 500000000)
  --num_gpu N         Number of GPUs (-1 = all, default: -1)
  --num_threads N     Number of CPU threads (-1 = all, default: -1)
  --debug             Enable debug output
  --version           Print version (v0.1.2.8)
  --help              Print help
```

### `runner.py`

```
Usage: runner.py [options] target.fasta.gz query.fasta.gz

  --diagonal-partition    Enable HSP diagonal partitioning
  --format FORMAT         Output format (default: maf-)
  --num-cpu N             Worker process count
  --num-gpu N             GPU count
  --output-file FILE      Output file
  --output-type TYPE      tarball | output
  --tool_directory PATH   Path to scripts directory
  --seed PATTERN          Seed pattern (passed to kegalign)
  --step N                Seed step (passed to kegalign)
  --scoring FILE          Scoring file (passed to kegalign)
  --debug                 Debug mode (load cached results)
```

### `run_mig.py`

```
Usage: run_mig.py [options]

  --MIG DEVICES       Comma-separated MIG device names (e.g. "GPU0,MIG1g.5gb")
  --MPS COUNTS        Comma-separated processes per device (e.g. "2,3")
  --query DIR         Directory with query chunk files
  --target DIR        Directory with target chunk files
  --output FILE       Output MAF file
  --format FORMAT     Output format
  --num_threads N     Threads per KegAlign instance
  --segment_size N    HSP segment size for diagonal partition
  --refresh SECS      Polling interval (default: 0.2)
  --resubmit_fails    Resubmit OOM failures
```

### `split_input.py`

```
Usage: split_input.py [options]

  --input FILE        Input FASTA or FASTA.gz
  --out DIR           Output directory
  --goal_bp N         Target basepairs per partition (0 = use max_chunks)
  --max_chunks N      Maximum number of partitions
  --to_2bit           Convert chunks to 2bit format
```

---

## 11. Build System

**File:** `KegAlign/CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.10)
project(kegalign LANGUAGES C CXX CUDA)

# CUDA architecture detection (via get-cuda-arches.bash)
# Policies: CMP0104, CMP0128, CMP0146, CMP0167

# Dependencies
find_package(TBB REQUIRED COMPONENTS tbbmalloc tbbmalloc_proxy tbb_preview)
find_package(ZLIB REQUIRED)
find_package(Boost 1.70 REQUIRED COMPONENTS program_options)

# Executable
add_executable(kegalign
    common/DRAM.cpp common/ntcoding.cpp common/seed_filter_interface.cu
    common/seed_pos_table.cu common/utilities.c common/dna_utilities.c
    common/scoring.c src/seed_filter.cu src/seeder.cpp
    src/segment_printer.cpp src/main.cpp
)

# Compilation settings
target_compile_options(kegalign PRIVATE -O4 -std=c++14 -DTBB_SUPPRESS_DEPRECATED_MESSAGES)
set_property(TARGET kegalign PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET kegalign PROPERTY CUDA_STANDARD 14)
```

**Build Steps:**
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

**CUDA Architecture Detection:**
`scripts/get-cuda-arches.bash` queries installed GPUs and sets appropriate `-arch` flags. This ensures the binary is compiled for the actual GPUs present.

**Conda Installation (preferred for end-users):**
```bash
conda install conda-forge::kegalign           # C++ binary only
conda install bioconda::kegalign-full         # Binary + LASTZ + faToTwoBit
```

---

## 12. Glossary

| Term | Definition |
|------|-----------|
| **HSP** | High Scoring Pair — an ungapped alignment region with score ≥ hspthresh |
| **Spaced seed** | A k-mer pattern with "don't care" positions (e.g. `12of19` = 12 match positions within 19-base span) |
| **X-drop** | Ungapped extension heuristic: stop extending when score drops more than X below maximum seen so far |
| **Y-drop** | Gapped extension heuristic (used by LASTZ): stop when score drops more than Y below maximum |
| **kmer** | k-character subsequence extracted at seed positions only (not all consecutive positions) |
| **Diagonal** | Line in the alignment matrix where `ref_pos - query_pos = constant`; HSPs on the same diagonal are collinear |
| **Anti-diagonal** | For reverse complement alignments: `ref_pos + query_pos = constant` |
| **DRAM** | In this codebase: 6GB aligned CPU memory buffer (not hardware DRAM) |
| **MIG** | Multi-Instance GPU: NVIDIA feature partitioning one GPU into isolated instances |
| **MPS** | Multi-Process Service: NVIDIA CUDA feature enabling multiple processes to share a GPU |
| **2bit** | UCSC binary genome format enabling O(1) random access to any sequence position |
| **MAF** | Multiple Alignment Format: standard bioinformatics format for genome alignments |
| **AXT** | Blastz-compatible pairwise alignment format |
| **HOXD70** | Empirical substitution matrix for mammalian genome comparison (Chiaramonte et al. 2002) |
| **LPT** | Longest Processing Time — bin-packing algorithm for balanced genome partitioning |
| **TBB** | Intel Threading Building Blocks: C++ parallelism library with flow graph API |
| **LASTZ** | Pairwise sequence aligner by Robert Harris; handles gapped alignment given seed segments |
| **SegAlign** | Original GPU-accelerated aligner (Goenka et al. 2020) that KegAlign is forked from |
| **Galaxy** | Web-based bioinformatics workflow platform; KegAlign includes Galaxy Tool Shed wrappers |
| **Segment file** | Tab-delimited file listing HSP coordinates; passed to LASTZ via `--segments=` |
| **seed_interval** | A subdivision of a query block processed in one TBB flow graph token |
| **seq_block** | A reference × query block pair, one unit of work in the TBB graph |
| **warp** | 32-thread CUDA execution unit; KegAlign uses warp-level parallelism in HSP extension |
| **BUFFER_DEPTH** | Number of parallel GPU buffer slots for pipelining query uploads |
| **Transition** | Single-base mutation: A↔G (purines) or C↔T (pyrimidines); allowed in seeds with `--transition` |

---

## 13. Key Functions Index

### C++/CUDA (`src/` and `common/`)

| Function | File | Purpose |
|----------|------|---------|
| `main()` | `src/main.cpp` | Entry point: parse args, read sequences, build TBB graph |
| `available_cpus()` | `src/main.cpp` | Get CPU count via sched_getaffinity |
| `seeder_body::operator()` | `src/seeder.cpp` | K-mer extraction → GPU seed filter → return HSPs |
| `segment_printer_body::operator()` | `src/segment_printer.cpp` | Write .seg file + print LASTZ command |
| `SeedAndFilter()` | `src/seed_filter.cu` | Host: orchestrate all GPU kernels, return HSPs |
| `InitializeProcessor()` | `src/seed_filter.cu` | Allocate GPU memory, load substitution matrix |
| `SendQueryWriteRequest()` | `src/seed_filter.cu` | Upload compressed query chunk to GPU |
| `ClearQuery()` | `src/seed_filter.cu` | Free query GPU memory for buffer slot |
| `ShutdownProcessor()` | `src/seed_filter.cu` | Release all GPU resources |
| `compress_string_rev_comp` | `src/seed_filter.cu` | CUDA kernel: ASCII → 2-bit + RC |
| `find_num_hits` | `src/seed_filter.cu` | CUDA kernel: count hits per seed |
| `find_hits` | `src/seed_filter.cu` | CUDA kernel: generate (ref,query) hit pairs |
| `find_hsps` | `src/seed_filter.cu` | CUDA kernel: X-drop extension with warp parallelism |
| `compress_output` | `src/seed_filter.cu` | CUDA kernel: compact HSP array |
| `InitializeInterface()` | `common/seed_filter_interface.cu` | Detect GPUs, allocate per-device arrays |
| `SendRefWriteRequest()` | `common/seed_filter_interface.cu` | Upload reference to all GPUs |
| `ClearRef()` | `common/seed_filter_interface.cu` | Free reference GPU memory |
| `compress_string` | `common/seed_filter_interface.cu` | CUDA kernel: ASCII → 2-bit (reference) |
| `GenerateSeedPosTable()` | `common/seed_pos_table.cu` | Build kmer index + pos table (TBB parallel) |
| `SendSeedPosTable()` | `common/seed_pos_table.cu` | Upload index + pos tables to all GPUs |
| `InclusivePrefixScan()` | `common/seed_pos_table.cu` | Thrust prefix sum on host |
| `GetKmerIndexAtPos()` | `common/ntcoding.cpp` | Extract kmer index at position using spaced seed |
| `IsTransitionAtPos()` | `common/ntcoding.cpp` | Check if transition mutation at position |
| `RevComp()` | `common/ntcoding.cpp` | Generate reverse complement string |
| `GenerateShapePos()` | `common/ntcoding.cpp` | Parse seed shape string |
| `load_scoring_matrix()` | `common/scoring.c` | Load LASTZ-format scoring matrix |
| `DRAM::DRAM()` | `common/DRAM.cpp` | Allocate 6GB aligned CPU buffer |

### Python (`scripts/`)

| Function/Class | File | Purpose |
|----------------|------|---------|
| `main()` | `runner.py` | Full pipeline orchestration |
| `run_kegalign()` | `runner.py` | Subprocess: run kegalign binary |
| `run_diagonal_partitioners()` | `runner.py` | Spawn partition workers |
| `diagonal_partition_worker()` | `runner.py` | Call diagonal_partition.py |
| `run_lastz()` | `runner.py` | Run LASTZ commands in parallel |
| `estimate_chunk_size()` | `runner.py` | Quantile-based chunk size estimation |
| `LastzCommands` | `runner.py` | Deduplicated LASTZ command registry |
| `LastzCommand` | `runner.py` | Parse individual LASTZ command line |
| `KegAlignSegment` | `runner.py` | Parse segment filename convention |
| `parse_line()` | `diagonal_partition.py` | Strip bash redirects, extract args |
| `sort_and_partition()` | `diagonal_partition.py` | Diagonal sort + chunk HSPs |
| `chunks()` | `diagonal_partition.py` | Generator: yield N-sized chunks |
| `main()` | `run_mig.py` | MPS/MIG GPU scheduling loop |
| `GPU_queue` | `run_mig.py` | Per-device process tracker |
| `Process_List` | `run_mig.py` | Subprocess lifecycle manager |
| `split_chr()` | `split_input.py` | LPT bin-packing for genome partitioning |
| `FastaFile` | `split_input.py` | FASTA reader + partitioner |
| `BatchTar` | `run_lastz_tarball.py` | Tarball extractor + command parser |
| `TarRunner` | `run_lastz_tarball.py` | Parallel LASTZ execution from tarball |
| `run_command()` | `run_lastz_tarball.py` | Execute single LASTZ command |
| `PackageFile` | `package_output.py` | Create output tarball for Galaxy |
| `bashCommandLineFile` | `package_output.py` | Parse LASTZ commands via bashlex |

---

## 14. Assumptions Table

| # | Assumption | Confidence | Basis |
|---|-----------|-----------|-------|
| A1 | Sequences must be ≤ 6GB each (per DRAM buffer size) | High | DRAM constructor allocates fixed 6GB |
| A2 | kmer_size must be ≤ 15 (kmer values ≤ 2^30) | High | Comment in seed_pos_table.cu |
| A3 | GPU must have sufficient memory for index table (2^(2k+2) bytes for k=14: ~1GB) | High | Calculated from index_table_size formula |
| A4 | LASTZ binary must be version 1.04.22 for compatibility | Medium | README specifies exact version |
| A5 | faToTwoBit must be patched with add-option.patch for namePrefix support | Medium | Patch file present; README implies usage |
| A6 | Python ≥ 3.6 required (f-strings, statistics.quantiles) | High | Code uses f-strings + quantiles |
| A7 | CUDA Toolkit ≥ 10.0 assumed (for warp shuffle functions) | Medium | __shfl_up used without qualifier |
| A8 | TBB exactly 2020.2 (API changed between versions) | High | README and CMakeLists.txt specify 2020.2 |
| A9 | Sequences stored as null-byte-separated chromosomes in DRAM | High | Separator character '&' (code 6) used in encoding |
| A10 | BUFFER_DEPTH ≥ 2 for pipelining to work | Medium | Logic in seeder_body uses buffer IDs |

---

## STATE BLOCK (Final)

```
INDEX_VERSION: v1.0
FILE_MAP_SUMMARY: 77 files in KegAlign/ — all key files analyzed
OPEN_QUESTIONS:
  - Exact BUFFER_DEPTH constant value (not found in reviewed files)
  - Full segment_printer.cpp coordinate logic (partially inferred)
  - strand handling for "both" vs "plus"/"minus" in printer
KNOWN_RISKS:
  - Global mutable state (cfg, DRAM*, chromosome vectors) is thread-safe only because
    all writes complete before TBB graph starts — this is a fragile invariant
  - LASTZ return-code 1 tolerance could mask real errors in edge cases
  - HSP deduplication via thrust::unique may miss near-duplicates at block boundaries
GLOSSARY_DELTA: All terms added in Section 12
```

---

*End of KegAlign Codebase Knowledge Document*

*Generated by automated codebase analysis. All file references are relative to the repository root `KegAlign/`.*
