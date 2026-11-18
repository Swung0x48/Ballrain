// The env's reward (ballrain/rewards.py, SR mode only: the pool exports SR policies only).
#pragma once

#include <optional>

#include "ballrain/json.h"
#include "ballrain/state.h"

namespace ballrain {

struct RewardConfig {
    double gamma = 0.99, progress_weight = 0.1, time_weight = 1.0, finish_bonus = 10.0, death_penalty = 5.0;
    double shaping_weight = 0.0, backward_scale = 1.0;
    bool hold_unsupported = true;
    double death_refund = 1.0, checkpoint_bonus = 0.0;
    bool pay_jumps = true;
    static RewardConfig from_json(const Json &j);  // the manifest's "reward"; Error unless mode "sr"
};

class Reward {
public:
    explicit Reward(RewardConfig c) : cfg(c) {}
    void begin(const GameTracker &t, std::optional<double> s);
    // rewards.Reward.step: the reward of one env step (terms summed in the Python order)
    double step(const GameTracker &t, std::optional<double> s, bool died, bool finished, bool terminal, bool rebase,
                std::optional<bool> supported);

    RewardConfig cfg;
    std::optional<double> s;
    double held = 0, sector_paid = 0, total = 0;
    std::optional<double> sr;
    int checkpoints = 0;
};

}  // namespace ballrain
