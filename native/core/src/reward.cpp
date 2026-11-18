#include "ballrain/reward.h"

#include <algorithm>

namespace ballrain {

RewardConfig RewardConfig::from_json(const Json &j) {
    RewardConfig c;
    if (j.get_string("mode", "sr") != "sr") throw Error("only SR rewards are supported");
    c.gamma = j.get_double("gamma", c.gamma);
    c.progress_weight = j.get_double("progress_weight", c.progress_weight);
    c.time_weight = j.get_double("time_weight", c.time_weight);
    c.finish_bonus = j.get_double("finish_bonus", c.finish_bonus);
    c.death_penalty = j.get_double("death_penalty", c.death_penalty);
    c.shaping_weight = j.get_double("shaping_weight", c.shaping_weight);
    c.backward_scale = j.get_double("backward_scale", c.backward_scale);
    c.hold_unsupported = j.get_bool("hold_unsupported", c.hold_unsupported);
    c.death_refund = j.get_double("death_refund", c.death_refund);
    c.checkpoint_bonus = j.get_double("checkpoint_bonus", c.checkpoint_bonus);
    c.pay_jumps = j.get_bool("pay_jumps", c.pay_jumps);
    return c;
}

void Reward::begin(const GameTracker &t, std::optional<double> s0) {
    s = s0;
    held = sector_paid = total = 0;
    auto v = t.sr_ms();
    sr = v ? *v : 0.0;
    checkpoints = t.checkpoints;
}

double Reward::step(const GameTracker &t, std::optional<double> s1, bool died, bool finished, bool terminal,
                    bool rebase, std::optional<bool> supported) {
    const RewardConfig &c = cfg;
    auto tsr = t.sr_ms();
    double srn = tsr ? *tsr : (sr ? *sr : 0.0);
    double dt_s = sr ? (srn - *sr) / 1000.0 : 0.0;
    double ds = (s1 && s && (c.pay_jumps || !rebase)) ? (*s1 - *s) : 0.0;
    if (ds < 0) ds *= c.backward_scale;
    if (t.checkpoints > checkpoints) sector_paid = 0.0;
    double pay = ds;
    if (c.hold_unsupported) {
        held += ds;
        pay = 0.0;
        if (died) pay = std::min(held, 0.0), held = 0.0;
        else if (finished || !supported || *supported) pay = held, held = 0.0;
    }
    double r = 0.0;
    double progress = c.progress_weight * pay;
    r += progress;
    sector_paid += progress;
    r += -c.time_weight * dt_s;
    if (c.shaping_weight && s1 && s) {
        double phi0 = c.shaping_weight * *s, phi1 = terminal ? 0.0 : c.shaping_weight * *s1;
        r += c.gamma * phi1 - phi0;
    }
    if (c.checkpoint_bonus && t.checkpoints > checkpoints) r += c.checkpoint_bonus * (t.checkpoints - checkpoints);
    if (finished) r += c.finish_bonus;
    if (died) {
        r += -c.death_penalty;
        if (c.death_refund) r += -c.death_refund * std::max(sector_paid, 0.0);
    }
    sr = srn;
    if (s1) s = s1;
    checkpoints = t.checkpoints;
    total += r;
    return r;
}

}  // namespace ballrain
