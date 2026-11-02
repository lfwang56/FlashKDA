#pragma once

#include <cstdint>

// Private varlen eligibility; sequence boundaries are always read on device.
// Fixed-layout calls retain C16 and the allocator reserves both upper bounds.
// The centered path has a measured advantage at this shape. All layouts,
// ragged boundaries and empty sequences remain valid within the selected path.
inline bool flash_kda_ppu_reserve_centered32(int64_t tokens, int64_t heads, int64_t sequences) {
    return heads == 6 && sequences == 8 && tokens == 16384;
}

// Controlled C16 dispatch experiment: host tile counts and device launch agree.
// Preserve both workspace upper bounds so allocator size is not the experiment.
inline bool flash_kda_ppu_use_centered32(int64_t, int64_t, int64_t) {
    return false;
}

// Single-request H6 uses time-parallel recurrence for long and irregular
// token counts. The host allocator and device launcher share this predicate.
// Per-segment bounds are whole-chunk partitions; device metadata still gates
// the full-sequence fast path so padded/empty varlen calls retain fallback.
inline bool flash_kda_ppu_reserve_segment_scan(int64_t tokens, int64_t heads, int64_t sequences) {
    if (sequences != 1)
        return false;
    if (heads == 6)
        return tokens == 1024 || tokens == 2048 || (tokens >= 4096 && tokens <= 262144);
    return heads == 12 && (tokens == 4096 || tokens == 8192 || tokens == 16384 || tokens == 32768 ||
                           tokens == 65536 || tokens == 131072);
}

inline constexpr int flash_kda_ppu_general_serial_tokens = 3072;

// A single-request launch combines parallel recurrence and native row-split
// serial fallback; the device reads the current sequence length every call.
inline bool flash_kda_ppu_fuse_general_serial(int, int heads, int sequences) {
    return sequences == 1 && heads <= 13 && heads != 12;
}

// Prepare can benefit even when the recurrence keeps its native row geometry.
inline bool flash_kda_ppu_single_prepare_only(int64_t tokens, int64_t heads, int64_t sequences) {
    return sequences == 1 && heads == 12 && tokens >= 3072 &&
           !flash_kda_ppu_reserve_segment_scan(tokens, heads, sequences);
}

// The native recurrence retains its standalone prefix. Reuse that table
// in bounded prepare to reduce empty-block startup without per-tile shuffles.
inline bool flash_kda_ppu_compact_prepare_only(int64_t tokens, int64_t heads, int64_t sequences) {
    if (sequences <= 1 || sequences > 32 || heads <= 0 || heads * sequences > 26 ||
        tokens / sequences < 3072)
        return false;
    if ((heads == 6 && (sequences == 4 || sequences == 8) && tokens == 16384) ||
        (heads == 12 && sequences == 4 && tokens == 8192))
        return false;
    return true;
}

// Keep a finite trip bound so the compiler retains the zero-spill tile body.
// The launcher uses the measured resident capacity, expanding the grid when
// this bound is needed to cover long physical allocations.
template <bool SinglePrefix> inline constexpr int flash_kda_ppu_general_prepare_tiles = 64;

// Select time parallelism from work per sequence and the existing CTA grid.
// Keep previously qualified paths intact; this is the general alternative for
// shapes that otherwise execute a long, underfilled serial recurrence.
// The same result defines the launch geometry and the allocator's upper bound.
inline int flash_kda_ppu_general_scan_segments(int64_t tokens, int64_t heads, int64_t sequences) {
    if (tokens <= 0 || heads <= 0 || sequences <= 0)
        return 0;
    // TP8 has a dedicated two-MMA/four-row-slice serial geometry. Keep that
    // complete path: widening its block to share the four-MMA scan makes
    // mutable short ranges slower even when the state itself is packed.
    if (heads == 12)
        return 0;
    if (flash_kda_ppu_reserve_segment_scan(tokens, heads, sequences))
        return 0;
    if ((heads == 6 && (sequences == 4 || sequences == 8) && tokens == 16384) ||
        (heads == 12 && sequences == 4 && tokens == 8192))
        return 0;
    int64_t const grid = heads * sequences;
    int64_t const average_tokens = tokens / sequences;
    // Summary/replay and state staging need substantial idle capacity to
    // amortize their extra transforms. Preserve ordinary recurrence once its
    // two row slices already occupy more than two thirds of the39-CU chip.
    int64_t const minimum_tokens = 3072;
    if (grid > 13 || average_tokens < minimum_tokens)
        return 0;
    int64_t limit = (156 + grid - 1) / grid;
    limit = limit < 16 ? limit : 16;
    limit = limit < 64 / sequences ? limit : 64 / sequences;
    int result = 0;
    constexpr int candidates[] = {4, 6, 8, 13, 16};
    for (int p : candidates)
        if (p <= limit)
            result = p;
    return result;
}
