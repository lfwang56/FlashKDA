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
