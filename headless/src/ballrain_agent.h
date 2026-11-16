#pragma once
// Ballrain headless agent mode (sim/ballrain_agent.cpp): its entry, called by SimTool's main (patches/0002) with
// SimTool's own arguments as its parser read them. The frame recipe is SimTool's --tas one, the flags and their
// defaults included (the verified Player.exe state: legacy edit on, --tas-movement-check 4, --tas-seed 3560118427,
// no per-level precision reset, --tas-offset 34), and its legacy edit is SimTool's apply_tas_legacy_edit.

#include <string>

namespace bmmo::sim {
class headless_engine;
}

struct ballrain_agent_args {
    std::string root;      // --root
    std::string sock;      // --sock (else BALLRAIN_SOCK)
    int level;             // --level (0: BALLRAIN_LEVEL)
    int boot_ticks;        // --boot-ticks
    int offset;            // --tas-offset: fixed ticks between Load Level and frame 0
    int reset_frame;       // --tas-reset-frame: the physics clock reset before frame R
    bool legacy;           // not --tas-no-legacy: TASSupport legacy mode's explosion-script edit
    int movement_check;    // --tas-movement-check: next_movement_check at the reset (< 0: kept)
    unsigned seed;         // --tas-seed: the ivp_rand seed at the reset
    bool reset_precision;  // --tas-reset-precision: BMMO's per-level x87 precision reset
    bool orig_random;      // --tas-orig-random: the engine's own Random blocks
    void (*legacy_edit)(bmmo::sim::headless_engine &engine);   // SimTool's apply_tas_legacy_edit
};

int ballrain_agent_main(const ballrain_agent_args &args);
