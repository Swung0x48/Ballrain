#include "ballrain/task_runner.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ballrain {

static const double PI = 3.141592653589793;

// ---------------------------------------------------------------------------------------------- config

Task Task::from_json(const Json &t) {
    Task k;
    k.task_id = t.at("task_id").as_string();
    k.kind = t.at("kind").as_string();
    k.run = t.at("run").as_string();
    k.policy_sha = t.at("policy_sha").as_string();
    k.level = int(t.at("level").as_int());
    k.record = t.at("record").as_string();
    k.record_sha256 = t.at("record_sha256").as_string();
    k.start_frame = t.at("start_frame").as_int();
    k.ball_order_seed = t.at("ball_order_seed").as_int();
    k.engine = t.at("engine").as_string();
    k.chunk_steps = t.at("chunk_steps").as_int();
    k.max_steps = t.at("max_steps").as_int();
    k.max_frames = t.at("max_frames").as_int();
    k.deadline_s = t.at("deadline_s").as_int();
    k.issued_at = t.at("issued_at").as_int();
    k.greedy = t.at("greedy").as_bool();
    k.sample_seed = uint64_t(t.at("sample_seed").as_int());
    k.asset_starts = t.at("assets").at("starts").as_string();
    k.asset_path = t.at("assets").at("path").as_string();
    k.rest = RestStart::from_json(t.find("rest"));
    if (const Json *c = t.find("life_seed")) k.life_seed = uint64_t(c->as_int());
    if (const Json *c = t.find("respawn_frontier"); c && !c->is_null()) k.respawn_frontier = c->as_int();
    if (k.chunk_steps < 1 || k.max_steps < 1 || k.max_frames < 1) throw Error("task limits");
    return k;
}

RolloutSpec RolloutSpec::from_json(const Json &r) {
    RolloutSpec s;
    for (auto &kv : r.members()) {
        const std::string &k = kv.first;
        const Json &v = kv.second;
        if (k == "chunk") s.chunk = v.as_int();
        else if (k == "gamma") s.gamma = v.as_double();
        else if (k == "cap_mult") s.cap_mult = v.as_double();
        else if (k == "cap_extra_s") s.cap_extra_s = v.as_double();
        else if (k == "stall_s") s.stall_s = v.as_double();
        else if (k == "respawn_stall_s") s.respawn_stall_s = v.is_null() ? std::nullopt : std::optional<double>(v.as_double());
        else if (k == "stall_progress") s.stall_progress = v.as_double();
        else if (k == "stall_terminal") s.stall_terminal = v.as_bool();
        else if (k == "stall_extra") s.stall_extra = v.as_double();
        else if (k == "stall_compensate") s.stall_compensate = v.as_bool();
        else if (k == "stall_speed") s.stall_speed = v.is_null() ? std::nullopt : std::optional<double>(v.as_double());
        else if (k == "stall_max_s") s.stall_max_s = v.as_double();
        else if (k == "route_probes") { /* bookkeeping only: no reward */ }
        else if (k == "gates") s.gates = v;
        else if (k == "split_k") s.split_k = v.as_double();
        else if (k == "split_min_ref_s") s.split_min_ref_s = v.as_double();
        else if (k == "split_rest_allow_s") s.split_rest_allow_s = v.as_double();
        else if (k == "split_extra")
            for (auto &e : v.items()) s.split_extra.push_back({e.at(size_t(0)).as_double(), e.at(size_t(1)).as_double(), e.at(size_t(2)).as_double()});
        else if (k == "counter_rand") s.counter_rand = v.is_null() ? Json::object() : v;
        else if (k == "p_early_respawn") s.p_early_respawn = v.as_double();
        else if (k == "respawn_level_starts") s.respawn_level_starts = v.as_bool();
        else throw Error("rollout field " + k + " unknown to this client");
    }
    return s;
}

// ---------------------------------------------------------------------------------------------- counters

CounterRand::CounterRand(const Json &spec, const FeatureExtractor &fx) {
    int at = 0;
    for (auto &l : fx.layout()) {
        if (l.first == "lives") i_lives_ = at;
        else if (l.first == "sr_s") i_sr_ = at;
        else if (l.first == "points") i_points_ = at;
        at += l.second;
    }
    if (!spec.is_object()) return;
    if (const Json *l = spec.find("lives"); l && i_lives_ >= 0) {
        for (auto &v : l->at("values").items()) values_.push_back(v.as_double());
        if (values_.empty()) throw Error("counter_rand.lives.values is empty");
        std::vector<double> p;
        const Json *pj = l->find("p");
        if (!pj || pj->is_null()) {
            p.assign(values_.size(), 1.0 / double(values_.size()));
        } else {
            for (auto &v : pj->items()) p.push_back(v.as_double());
            if (p.size() != values_.size()) throw Error("counter_rand.lives.p");
            double sum = numpy_sum(p.data(), p.size());
            for (auto &x : p) x /= sum;
        }
        double c = 0.0;
        for (double x : p) c += x, cdf_.push_back(c);  // np.cumsum
        has_lives_ = true;
    }
    if (const Json *r = spec.find("sr_s"); r && i_sr_ >= 0)
        sr_lo_ = r->at(size_t(0)).as_double(), sr_hi_ = r->at(size_t(1)).as_double(), has_sr_ = true;
    if (const Json *r = spec.find("points"); r && i_points_ >= 0)
        pt_lo_ = r->at(size_t(0)).as_double(), pt_hi_ = r->at(size_t(1)).as_double(), has_pt_ = true;
    any_ = !spec.members().empty();
}

void CounterRand::draw(uint64_t seed, int64_t life) {
    ops_.clear();
    uint64_t b = 4 * uint64_t(life);
    if (has_lives_) {
        double u = sample_u(seed, b);
        size_t i = size_t(std::upper_bound(cdf_.begin(), cdf_.end(), u) - cdf_.begin());  // searchsorted side=right
        i = std::min(i, values_.size() - 1);
        ops_.push_back({i_lives_, true, values_[i]});
    }
    if (has_sr_) ops_.push_back({i_sr_, false, sr_lo_ + (sr_hi_ - sr_lo_) * sample_u(seed, b + 1)});
    if (has_pt_) ops_.push_back({i_points_, false, pt_lo_ + (pt_hi_ - pt_lo_) * sample_u(seed, b + 2)});
}

void CounterRand::apply(std::vector<float> &o) const {
    for (auto &op : ops_) {  // numpy: float32 array element with a Python float (NEP 50: float32 arithmetic)
        float &x = o[size_t(op.index)];
        x = op.set ? float(op.value) : x + float(op.value);
    }
}

// ---------------------------------------------------------------------------------------------- runner

TaskRunner::TaskRunner(BallanceEnv &env, const TasStarts *starts, const Task &task, const RolloutSpec &spec,
                       double delta_ms)
    : env_(env), st_(starts), task_(task), sp_(spec), delta_ms_(delta_ms), cr_(spec.counter_rand, env.features()) {
    env_.set_digest(&dg_);
    if (st_ && !st_->valid.empty()) {  // rollout._Shaping
        lf_ = st_->levelfinish_frame && *st_->levelfinish_frame ? *st_->levelfinish_frame : st_->valid.back();
        rec_ms_ = std::isfinite(st_->sr_final) ? st_->sr_final / double(std::max<int64_t>(lf_ - st_->valid[0], 1))
                                               : 1000.0 / 264.0;
    }
    stall_all_ = int64_t(1000 * sp_.stall_s / delta_ms_);
    stall_rsp_ = int64_t(1000 * (sp_.respawn_stall_s && *sp_.respawn_stall_s ? *sp_.respawn_stall_s : sp_.stall_s) / delta_ms_);
    stall_max_ = int64_t(1000 * sp_.stall_max_s / delta_ms_);
}

double TaskRunner::tas_igt(int64_t k) const {
    const auto &v = st_->valid;
    int64_t i = int64_t(std::lower_bound(v.begin(), v.end(), k) - v.begin());
    i = std::min<int64_t>(std::max<int64_t>(i, 0), int64_t(v.size()) - 1);
    return st_->sr_double[size_t(i)];
}

std::optional<double> TaskRunner::tas_cross(int sector) const {
    if (!st_) return std::nullopt;
    if (sector == 0) return st_->sr_final;
    for (size_t i = 0; i < st_->sector.size(); ++i)
        if (st_->sector[i] == sector) return st_->sr_double[i];
    return std::nullopt;
}

std::optional<int64_t> TaskRunner::tas_cross_frame(int sector) const {
    for (size_t i = 0; i < st_->sector.size(); ++i)
        if (st_->sector[i] == sector) return st_->valid[i];
    return std::nullopt;
}

std::optional<int64_t> TaskRunner::cap_frames(const StepInfo &info) const {
    if (!st_) return std::nullopt;
    const StartInfo &s = env_.start_info();
    int64_t k;
    if (s.kind == "respawn") {
        auto f = tas_cross_frame(info.sector);
        k = f ? *f : st_->valid[0];
    } else {
        k = s.frame;
    }
    double rem_ms = double(std::max<int64_t>(lf_ - k, 0)) * rec_ms_;
    return int64_t((sp_.cap_mult * rem_ms + 1000 * sp_.cap_extra_s) / delta_ms_);
}

void TaskRunner::split_start(Episode &ep) {
    if (!sp_.split_k || !st_) return;
    std::optional<double> ref;
    if (ep.kind == "tas_rest" && ep.start_frame > 0 && !st_->index_of(ep.start_frame)) {
        ep.has_seg = false;  // handed over in an excluded range (off the TAS's route): no reference time
        return;
    }
    if (ep.kind == "tas" && ep.start_frame > 0) ref = tas_igt(ep.start_frame);
    else if (ep.kind == "tas_rest") ref = tas_igt(ep.start_frame) - 1000 * sp_.split_rest_allow_s;
    else if (ep.kind == "respawn") {
        auto c = tas_cross(ep.start_sector);
        if (c) ref = *c - 1000 * sp_.split_rest_allow_s;
    } else ref = tas_cross(1);
    std::optional<int64_t> k0;
    if (ep.kind == "respawn") k0 = tas_cross_frame(ep.start_sector);
    else k0 = ep.start_frame > 0 ? ep.start_frame : st_->valid[0];
    ep.has_seg = ref.has_value();
    if (ref) ep.seg_agent = ep.start_sr_ms, ep.seg_ref = *ref, ep.seg_k0 = k0;
}

double TaskRunner::split_bonus(Episode &ep, const StepInfo &info, bool finished) {
    if (!ep.has_seg || !sp_.split_k) return 0.0;
    int sec = finished ? 0 : info.sector;
    auto end_ref = tas_cross(sec);
    if (!end_ref || !info.sr_ms) return 0.0;
    double sr = *info.sr_ms;
    double t_agent = (sr - ep.seg_agent) / 1000.0;
    double t_ref = (*end_ref - ep.seg_ref) / 1000.0;
    std::optional<int64_t> end_frame;
    if (sec) end_frame = tas_cross_frame(sec);
    else end_frame = st_->levelfinish_frame && *st_->levelfinish_frame ? *st_->levelfinish_frame : int64_t(1000000000);
    for (auto &e : sp_.split_extra)
        if (ep.seg_k0 && end_frame && double(*ep.seg_k0) < e[0] && double(*end_frame) >= e[1]) t_ref += e[2];
    double k = sp_.split_k;
    double b = k * (t_ref - t_agent) / std::max(t_ref, sp_.split_min_ref_s);
    b = std::min(std::max(b, -k), k);
    ep.seg_agent = sr, ep.seg_ref = *end_ref, ep.seg_k0 = end_frame;
    return b;
}

double TaskRunner::gate(Episode &ep, const double pos[3], const double vel[3]) {
    double bonus = 0.0;
    for (auto &kv : sp_.gates.members()) {
        const std::string &name = kv.first;
        const Json &g = kv.second;
        if (ep.gates_hit.count(name)) continue;
        if (const Json *boxes = g.find("boxes")) {
            bool any = false;
            for (auto &b : boxes->items())
                any = any || (b.at(size_t(0)).as_double() <= pos[0] && pos[0] <= b.at(size_t(1)).as_double() &&
                              b.at(size_t(2)).as_double() <= pos[2] && pos[2] <= b.at(size_t(3)).as_double() &&
                              pos[1] > b.at(size_t(4)).as_double());
            if (!any) continue;
        } else {
            const Json &c = g.at("center");
            double d[3] = {pos[0] - c.at(size_t(0)).as_double(), pos[1] - c.at(size_t(1)).as_double(),
                           pos[2] - c.at(size_t(2)).as_double()};
            if (std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) > g.get_double("radius", 2.0)) continue;
        }
        double vh = std::hypot(vel[0], vel[2]);
        if (vh < g.get_double("min_speed", 0.0)) continue;
        if (g.has("heading")) {
            double hd = std::atan2(vel[2], vel[0]) * (180.0 / PI);
            double x = std::fmod(hd - g.at("heading").as_double() + 180, 360.0);
            if (x < 0) x += 360.0;  // Python %: the sign of the divisor
            if (std::abs(x - 180) > g.get_double("heading_tol", 180)) continue;
        }
        ep.gates_hit.insert(name);
        bonus += g.get_double("bonus", 0.0);
    }
    return bonus;
}

void TaskRunner::new_episode(const StepInfo &info) {
    Episode ep;
    const StartInfo &s = env_.start_info();
    ep.kind = s.kind;
    ep.start_frame = s.frame;
    ep.start_sector = info.sector;
    ep.start_sr_ms = info.sr_ms ? *info.sr_ms : 0.0;
    ep.cap_frames = cap_frames(info);
    auto rs = env_.reward().s;
    ep.mark_s = rs ? *rs : -std::numeric_limits<double>::infinity();
    ep.max_sector_seen = ep.start_sector;
    split_start(ep);
    ep_ = std::move(ep);
    ++episodes;
}

std::vector<float> TaskRunner::new_life(std::vector<float> obs) {
    // as Worker._reset / rollout.TaskRunner._new_life_counters: the reset observation was computed with the true
    // counters, the new life's set applies from it on
    cr_.clear();
    if (cr_.active(task_.start_frame)) cr_.draw(task_.life_seed, lives_);
    ++lives_;
    cr_.apply(obs);
    return obs;
}

// counters.skip_respawn (Worker._skip_respawn): the respawn after the current life's death in `sector` is not played
bool TaskRunner::skip_respawn(int sector) const {
    if (sp_.p_early_respawn >= 1.0 || !task_.respawn_frontier || !st_) return false;
    if (sp_.respawn_level_starts && task_.start_frame == 0) return false;
    auto cross = tas_cross_frame(sector);
    if (!cross || *cross >= *task_.respawn_frontier) return false;
    return sample_u(task_.life_seed, 4 * uint64_t(lives_ - 1) + 3) >= sp_.p_early_respawn;
}

std::vector<float> TaskRunner::start() {
    std::optional<std::vector<float>> o;
    try {
        o = env_.reset(true, &info_);
    } catch (const std::exception &e) {
        throw EngineFailure(std::string("start: ") + e.what());
    }
    digest = dg_.take();
    f_hand_ = info_.frames_done;
    new_episode(info_);
    return new_life(std::move(*o));
}

static std::string end_name(const StepInfo &info, const std::string &worker_end) {
    if (worker_end == "cap" || worker_end == "stall") return worker_end;
    bool lf = false, go = false;
    for (auto &e : info.events) lf = lf || e.code == Ev::LevelFinish, go = go || e.code == Ev::GameOver;
    if (info.episode_end == "LevelFinish" || lf) return "LevelFinish";
    if (info.episode_end == "GameOver" || go) return "GameOver";
    return "truncated";
}

StepResult TaskRunner::step(int a, const std::function<std::string()> &stop) {
    if (!end.empty()) throw Error("task ended: " + end);
    Session *sess = env_.session();
    uint32_t f0 = sess ? sess->frames_done() : 0;
    bool term, trunc;
    StepInfo info2;
    double r = 0;
    std::vector<float> obs2;
    try {
        obs2 = env_.step(a, term, trunc, info2, &r);
        cr_.apply(obs2);
    } catch (const std::exception &e) {
        throw EngineFailure(std::string("step: ") + e.what());
    }
    if (info2.error) throw EngineFailure("step: " + *info2.error);
    double env_r = r;
    int64_t frames_step = int64_t(info2.frames_done) - int64_t(f0);
    Episode &ep = ep_;
    if (sp_.split_k && (info2.sector > ep.max_sector_seen || info2.episode_end == "LevelFinish"))
        r += split_bonus(ep, info2, info2.episode_end == "LevelFinish");
    ep.max_sector_seen = std::max(ep.max_sector_seen, info2.sector);
    sess = env_.session();
    if (!sp_.gates.members().empty() && sess) {
        const Observation *o = sess->obs();
        if (o && o->ball && o->ball->found && !term) {
            double vel[3];
            for (int k = 0; k < 3; ++k) vel[k] = double(o->ball->speed[k]) + double(o->ball->speed_change[k]);
            r += gate(ep, o->ball->pos, vel);
        }
    }
    std::string wend = (term || trunc) ? info2.episode_end : "";
    if (!(term || trunc)) {
        int64_t ef = env_.ep_frames();
        auto s = env_.reward().s;
        if (s && *s > ep.mark_s + sp_.stall_progress) ep.mark_s = *s, ep.mark_frame = ef;
        int64_t win = ep.kind == "respawn" ? stall_rsp_ : stall_all_;
        int64_t since = ef - ep.mark_frame;
        if (sp_.stall_speed) {
            const Observation *o = sess ? sess->obs() : nullptr;
            if (o && o->ball) {
                const float *v = o->ball->speed;
                float n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);  // np.linalg.norm of float32
                if (double(n) > *sp_.stall_speed) ep.move_frame = ef;
            }
            since = std::min(since, ef - ep.move_frame);
            if (ef - ep.mark_frame >= stall_max_) since = win;
        }
        if (ep.cap_frames && ef >= *ep.cap_frames) {
            trunc = true, wend = "cap";
        } else if (s && since >= win) {
            trunc = true, wend = "stall";
            if (sp_.stall_terminal) {
                const RewardConfig &rc = env_.reward().cfg;
                double pen = -(rc.death_penalty + rc.death_refund * std::max(env_.reward().sector_paid, 0.0) + sp_.stall_extra);
                if (sp_.stall_compensate) pen /= std::pow(sp_.gamma, double(win) / double(std::max(env_.config().action_repeat, 1)));
                r += pen;
                term = true, trunc = false;
            }
        }
    }
    ++steps;
    bool respawn = term && wend != "stall" && env_.after_death();
    uint8_t flags = term ? FLAG_TERMINATED : (trunc ? FLAG_TRUNCATED : 0);
    std::string task_end;
    if (respawn && wend == "BallOff" && skip_respawn(info2.sector)) {
        task_end = "respawn_skipped";  // the learner's early-respawn rule: its process would end here
    } else if (trunc || (term && !respawn)) {
        task_end = end_name(info2, wend);
    } else {
        if (steps >= task_.max_steps) task_end = "max_steps";
        else if (int64_t(info2.frames_done) - f_hand_ >= task_.max_frames) task_end = "max_frames";
        else if (stop) task_end = stop();
        if (!task_end.empty() && !term) flags = FLAG_TRUNCATED;
    }
    frames = int64_t(info2.frames_done) - f_hand_;
    std::vector<float> nxt = obs2;
    FeatureInfo last_feat = info2.feat;
    if (task_end.empty() && term) {  // a respawn in the same process (or the process ends during it)
        std::optional<std::vector<float>> on;
        StepInfo in;
        cr_.clear();
        try {
            on = env_.reset(false, &in);
        } catch (const std::exception &e) {
            throw EngineFailure(std::string("respawn: ") + e.what());
        }
        if (!on) {
            task_end = "GameOver";
        } else {
            nxt = new_life(std::move(*on));
            last_feat = in.feat;
            new_episode(in);
        }
    }
    StepResult res;
    res.action = a;
    res.reward = float(r);
    res.env_reward = float(env_r);
    res.flags = flags;
    res.frames = frames_step;
    res.sector = info2.sector;
    res.sr_ms = info2.sr_ms ? *info2.sr_ms : 0.0;
    res.arc = last_feat.path ? last_feat.path->s : std::numeric_limits<double>::quiet_NaN();
    res.next_obs = std::move(nxt);
    end = task_end;
    res.end = task_end;
    digest = dg_.take();
    res.digest = digest;
    return res;
}

// ---------------------------------------------------------------------------------------------- chunks

ChunkAssembler::ChunkAssembler(const Task &task, int n_obs, int n_act, const std::string &policy_sha)
    : task_(task), n_obs_(n_obs), n_act_(n_act), sha_(policy_sha), prev_(chain_seed(task.task_id)) {}

void ChunkAssembler::begin(const std::vector<float> &obs, float value, const Bytes &digest) {
    rows_ = {obs};
    digests_ = {digest};
    values_ = {value};
}

void ChunkAssembler::add(float logp, const StepResult &st, float next_value) {
    actions_.push_back(uint8_t(st.action));
    rewards_.push_back(st.reward);
    logp_.push_back(float(std::min(std::max(double(logp), MIN_LOGP), 0.0)));
    flags_.push_back(st.flags);
    env_rewards_.push_back(st.env_reward);
    sector_.push_back(uint8_t(std::min(std::max(st.sector, 0), 255)));
    frames_.push_back(uint16_t(std::min<int64_t>(std::max<int64_t>(st.frames, 0), 65535)));
    sr_ms_.push_back(st.sr_ms);
    arc_.push_back(float(st.arc));
    rows_.push_back(st.next_obs);
    digests_.push_back(st.digest);
    values_.push_back(next_value);
}

Bytes ChunkAssembler::cut(const std::string &end) {
    size_t T = pending();
    if (!T) throw Error("no steps to cut");
    ChunkData c;
    c.task_id = task_.task_id, c.policy_sha = sha_, c.seq = seq, c.step0 = step0, c.n_obs = n_obs_, c.n_act = n_act_;
    c.chain_prev = prev_;
    c.final = !end.empty(), c.end = end;
    c.actions = actions_, c.rewards = rewards_, c.logp = logp_, c.values = values_, c.flags = flags_;
    if (!end.empty() && !c.flags.back()) c.flags.back() = FLAG_TRUNCATED;
    for (auto &r : rows_) c.obs.insert(c.obs.end(), r.begin(), r.end());
    Bytes h = prev_;
    for (auto &d : digests_) h = chain_next(h, d), c.obs_hash.push_back(h);
    c.env_rewards = env_rewards_, c.sector = sector_, c.frames = frames_, c.sr_ms = sr_ms_, c.arc = arc_;
    Bytes data = encode_chunk(c);
    prev_ = c.obs_hash[c.obs_hash.size() - 2];
    ++seq;
    step0 += int64_t(T);
    rows_ = {rows_.back()}, digests_ = {digests_.back()}, values_ = {values_.back()};
    actions_.clear(), rewards_.clear(), logp_.clear(), flags_.clear(), env_rewards_.clear(), sector_.clear();
    frames_.clear(), sr_ms_.clear(), arc_.clear();
    return data;
}

std::string sample_task(TaskRunner &runner, const Policy &policy, const Task &task,
                        const std::function<void(const Bytes &, bool)> &on_chunk, const std::function<std::string()> &stop,
                        const std::function<int(int64_t, const std::vector<float> &)> &choose) {
    ChunkAssembler asm_(task, policy.n_obs, policy.n_act, policy.sha);
    std::vector<float> obs = runner.start();
    std::vector<float> x = policy.normalize(obs);
    asm_.begin(obs, policy.value(x), runner.digest);
    std::string end;
    while (end.empty()) {
        std::vector<float> lp = policy.log_probs(x);
        int a;
        if (task.greedy) a = int(std::max_element(lp.begin(), lp.end()) - lp.begin());
        else a = sample_action(lp, sample_u(task.sample_seed, uint64_t(runner.steps)));
        if (choose) a = choose(runner.steps, lp);
        StepResult st;
        try {
            st = runner.step(a, stop);
        } catch (const EngineFailure &) {
            end = "error";
            break;
        }
        x = policy.normalize(st.next_obs);
        asm_.add(lp[size_t(a)], st, policy.value(x));
        end = st.end;
        if (end.empty() && asm_.ready()) on_chunk(asm_.cut(), false);
    }
    if (asm_.pending()) on_chunk(asm_.cut(end), true);
    return end;
}

}  // namespace ballrain
