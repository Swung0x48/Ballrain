#include "ballrain/env.h"

#include <cmath>
#include <limits>

#include "ballrain/npz.h"
#include "ballrain/policy.h"
#include "ballrain/sha256.h"
#include "platform.h"

namespace ballrain {

const uint32_t ARROWS9[9] = {0,        KEY_UP,           KEY_DOWN,           KEY_LEFT,          KEY_RIGHT,
                             KEY_UP | KEY_LEFT, KEY_UP | KEY_RIGHT, KEY_DOWN | KEY_LEFT, KEY_DOWN | KEY_RIGHT};
static const uint32_t ARROW_BITS[4] = {KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT};
static const uint32_t ARROW_MASK = KEY_UP | KEY_DOWN | KEY_LEFT | KEY_RIGHT;
static const double PI = 3.141592653589793;
static uint64_t episode_stop() { return event_mask({Ev::BallOff, Ev::LevelFinish, Ev::GameOver}); }

static double norm3(const double *d) { return std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]); }

static std::optional<int64_t> opt_int(const Json &env, const char *k, std::optional<int64_t> dflt) {
    const Json *j = env.find(k);
    if (!j) return dflt;
    if (j->is_null()) return std::nullopt;
    return j->as_int();
}

EnvConfig EnvConfig::from_manifest(const Json &e) {
    EnvConfig c;
    c.level = int(e.get_int("level", 1));
    if (e.get_string("mode", "sr") != "sr") throw Error("only SR-mode policies are supported");
    c.delta = float(e.get_double("delta_ms", double(c.delta)));
    c.action_repeat = int(e.get_int("action_repeat", 1));
    c.camera_actions = e.get_bool("camera_actions", false);
    c.world_actions = e.get_bool("world_actions", false);
    if (const Json *r = e.find("repress_stale_keys"); r && !r->is_null()) c.repress_stale_keys = r->as_bool();
    c.camera_quarter_tol = e.get_double("camera_quarter_tol", c.camera_quarter_tol);
    c.camera_turn_grace_ms = e.get_double("camera_turn_grace_ms", c.camera_turn_grace_ms);
    const Json *esc = e.find("esc_action");
    if ((esc && esc->is_bool() && esc->as_bool()) || e.get_bool("collect_action", false))
        throw Error("Esc / collect actions are not supported");
    if (c.camera_actions) throw Error("camera actions are not supported");
    c.esc_frames = int(e.get_int("esc_frames", 4));
    c.mask_keys_while_paused = e.get_bool("mask_keys_while_paused", true);
    c.pause_keys = uint32_t(e.get_int("pause_keys", 0));
    c.max_pause_frames = opt_int(e, "max_pause_frames", c.max_pause_frames);
    c.max_episode_pause_frames = opt_int(e, "max_episode_pause_frames", c.max_episode_pause_frames);
    c.per_life = e.get_bool("per_life", true);
    c.max_episode_frames = opt_int(e, "max_episode_frames", std::nullopt);
    c.max_process_frames = opt_int(e, "max_process_frames", std::nullopt);
    c.idle_keys = uint32_t(e.get_int("idle_keys", 0));
    c.skip_tutorial = e.get_bool("skip_tutorial", true);
    c.timeout_ms = uint32_t(e.get_int("timeout_ms", c.timeout_ms));
    c.obs_timeout_s = e.get_double("obs_timeout_s", c.obs_timeout_s);
    c.hello_timeout_s = e.get_double("hello_timeout_s", c.hello_timeout_s);
    c.sr_tail = int(e.get_int("sr_tail", 32));
    c.respawn_limit_frames = uint32_t(e.get_int("respawn_limit_frames", 20000));
    c.body_radius = e.get_double("body_radius", 60.0);
    c.skip_render = e.get_bool("skip_render", true);
    c.start_check = e.get_bool("start_check", true);
    c.start_retries = int(e.get_int("start_retries", 3));
    if (const Json *x = e.find("exclude_starts"); x && x->is_array())
        for (auto &r : x->items()) c.exclude_starts.push_back({r.at(size_t(0)).as_int(), r.at(size_t(1)).as_int()});
    c.rest_speed = e.get_double("rest_speed", c.rest_speed);
    c.rest_max_frames = e.get_int("rest_max_frames", c.rest_max_frames);
    return c;
}

std::optional<RestStart> RestStart::from_json(const Json *r) {
    if (!r || r->is_null()) return std::nullopt;
    RestStart s;
    const Json &t = r->at("target");
    if (!t.is_null()) {
        if (t.items().size() != 2) throw Error("task rest.target must be [x, z]");
        s.drive = true, s.x = t.at(size_t(0)).as_double(), s.z = t.at(size_t(1)).as_double();
    }
    if (const Json *b = r->find("branch_s"); b && !b->is_null()) {
        if (s.drive) throw Error("task rest: a target or branch_s, not both");
        s.branch_s = b->as_double();
    }
    if (const Json *id = r->find("branch_id"); id && !id->is_null()) {
        if (!s.branch_s) throw Error("task rest: branch_id needs branch_s");
        s.branch_id = id->as_string();
    }
    if (const Json *m = r->find("max_frames"); m && !m->is_null()) {
        s.max_frames = m->as_int();
        if (*s.max_frames < 1) throw Error("task rest.max_frames");
    }
    return s;
}

// ---------------------------------------------------------------------------------------------- world actions

uint32_t world_to_camera_keys(uint32_t keys, const float m[12]) {
    if (!(keys & ARROW_MASK)) return keys;
    double w[3] = {double(bool(keys & KEY_RIGHT)) - double(bool(keys & KEY_LEFT)), 0.0,
                   double(bool(keys & KEY_UP)) - double(bool(keys & KEY_DOWN))};
    double wn = norm3(w);
    for (auto &x : w) x /= wn;
    // key_push_dirs: UP / DOWN along +- the z row, RIGHT / LEFT along +- the x row, horizontal and normalised
    double dirs[4][3];
    const int row[4] = {2, 2, 0, 0};
    const double sign[4] = {1.0, -1.0, -1.0, 1.0};  // UP, DOWN, LEFT, RIGHT
    for (int i = 0; i < 4; ++i) {
        double v[3] = {sign[i] * double(m[3 * row[i]]), sign[i] * double(m[3 * row[i] + 1]), sign[i] * double(m[3 * row[i] + 2])};
        v[1] = 0.0;
        double n = norm3(v);
        for (int k = 0; k < 3; ++k) dirs[i][k] = n > 1e-9 ? v[k] / n : 0.0;
    }
    uint32_t best = 0;
    double best_c = -std::numeric_limits<double>::infinity();
    for (int a = 1; a < 9; ++a) {
        uint32_t ks = ARROWS9[a];
        double v[3] = {0.0, 0.0, 0.0};
        for (int b = 0; b < 4; ++b)
            if (ks & ARROW_BITS[b])
                for (int k = 0; k < 3; ++k) v[k] = v[k] + dirs[b][k];
        double n = norm3(v);
        double c = n > 1e-9 ? (v[0] * w[0] + v[1] * w[1] + v[2] * w[2]) / n : -std::numeric_limits<double>::infinity();
        if (c > best_c + 1e-9) best = ks, best_c = c;
    }
    return (keys & ~ARROW_MASK) | best;
}

// ---------------------------------------------------------------------------------------------- env

BallanceEnv::BallanceEnv(EnvConfig cfg, FeatureConfig features, EngineConfig engine, std::string run_root,
                         StreamWriter *rec, const TasStarts *starts, int64_t start_frame, RewardConfig reward,
                         std::optional<RestStart> rest)
    : cfg_(std::move(cfg)), eng_(std::move(engine)), run_root_(std::move(run_root)), rec_(rec), fx_(std::move(features)),
      starts_(starts), start_frame_(start_frame), rest_(rest), reward_(reward) {
    // an at-rest start with a drive target may hand over in an excluded range (TasStarts.expected knows it)
    bool any = rest_ && (rest_->drive || rest_->branch_s);
    if (start_frame_ > 0 && (!starts_ || !(any ? starts_->expected(start_frame_).has_value()
                                                : starts_->index_of(start_frame_).has_value())))
        throw Error("start frame " + std::to_string(start_frame_) + " is not a valid start frame of the record");
    if (rest_ && start_frame_ <= 0) throw Error("an at-rest start needs a TAS start frame");
    for (auto k : ARROWS9) actions_.push_back(k);
    if (cfg_.esc_frames < 3) throw Error("esc_frames must be >= 3");
    // env.BallanceEnv: the reach block's 8 directions are the action keys (world actions need camera "world",
    // camera-relative keys a camera frame)
    if (fx_.config().reach && (fx_.config().camera == "world") != cfg_.world_actions)
        throw Error("FeatureConfig.reach with camera " + fx_.config().camera + " and world_actions=" +
                    (cfg_.world_actions ? "true" : "false") + ": the reach directions would not be the action directions");
    repress_ = cfg_.repress_stale_keys ? *cfg_.repress_stale_keys : cfg_.world_actions;
    ms_since_turn_ = std::numeric_limits<double>::infinity();
    // env.branch_rules: the installed bundle's rules (the task is refused when they do not parse)
    // (wire v2: the defaults until a member is active; env._info then follows the active member's rules)
    if (!fx_.config().branch_dir.empty() && !file_exists(fx_.config().branch_dir + "/branches.json"))
        branch_rules_ = load_branch_rules(fx_.config().branch_dir, cfg_.level);
}

BallanceEnv::~BallanceEnv() { close(); }

void BallanceEnv::close() {
    if (sess_) finish_process();
}

void BallanceEnv::finish_process() {
    if (!sess_) return;
    try {
        sess_->stop();
    } catch (...) {
        sess_->close(true);
        sess_.reset();
        after_death_ = false;
        throw;
    }
    sess_->close();
    sess_.reset();
    after_death_ = false;
}

bool BallanceEnv::launch(int64_t k) {
    SessionConfig sc;
    sc.level = cfg_.level;
    sc.out_dir = run_root_ + "/run-" + std::to_string(platform::self_pid()) + "-" + std::to_string(processes++);
    sc.timeout_ms = cfg_.timeout_ms;
    if (cfg_.max_process_frames) sc.max_frames = uint32_t(*cfg_.max_process_frames);
    sc.body_radius = cfg_.body_radius;
    sc.sr_tail = cfg_.sr_tail;
    sc.skip_render = cfg_.skip_render;
    sc.obs_timeout_s = cfg_.obs_timeout_s;
    sc.hello_timeout_s = cfg_.hello_timeout_s;
    if (k > 0) {  // the record's first k frames, replayed by the engine before the hand-over (BALLRAIN_PREFIX)
        if (!platform::make_dirs(sc.out_dir)) throw Error("cannot create " + sc.out_dir);
        sc.prefix = sc.out_dir + "/prefix.bin";
        Bytes raw;
        for (auto &f : starts_->prefix(k)) put<float>(raw, f.delta), put<uint32_t>(raw, f.keys);
        write_file(sc.prefix, raw);
    }
    if (digest_) digest_->reset();  // a process discarded by the hand-over check leaves nothing in the digest
    sess_ = std::make_unique<Session>(sc, eng_, rec_);
    sess_->set_digest(digest_);
    sess_->start();
    try {
        intro();
    } catch (...) {
        sess_->close(true);
        sess_.reset();
        throw;
    }
    return !(k > 0 && cfg_.start_check) || handover_ok(k);
}

// env._handover_check: the hand-over state against the record's (frame, sector, checkpoints, points, lives, SR, ball
// position: exact)
bool BallanceEnv::handover_ok(int64_t k) const {
    auto exp = starts_->expected(k);
    const Observation *o = sess_->obs();
    if (!exp || !o || !o->ball || sess_->ended()) return false;
    const GameTracker &t = sess_->tracker();
    auto sr = t.sr_ms();
    return int64_t(sess_->frames_done()) == exp->frame && t.sector() == exp->sector && t.checkpoints == exp->checkpoints &&
           t.points() == exp->points && t.lives() == exp->lives && (sr ? *sr : 0.0) == exp->sr_double &&
           o->ball->pos[0] == exp->pos[0] && o->ball->pos[1] == exp->pos[1] && o->ball->pos[2] == exp->pos[2];
}

// a failed stone rest start (its own checks: assets, range, ball lost, ball type / sector, not at rest on the line):
// start_process falls back to the TAS start; engine and session faults are not caught
struct BranchRestFailed : Error {
    explicit BranchRestFailed(const std::string &m) : Error(m) {}
};
// the brake lost the ball (env._brake's RuntimeError; the process is already stopped): a game outcome, not a fault
struct BallLostBraking : Error {
    explicit BallLostBraking(const std::string &m) : Error(m) {}
};

int BallanceEnv::launch_checked(int64_t k) {
    int mismatches = 0;
    for (int attempt = 0;; ++attempt) {
        if (launch(k)) break;
        ++mismatches;
        finish_process();
        if (attempt >= std::max(0, cfg_.start_retries))
            throw Error("hand-over at frame " + std::to_string(k) + " differs from the record " +
                        std::to_string(mismatches) + " times");
    }
    return mismatches;
}

void BallanceEnv::start_process() {
    int64_t k = start_frame_;
    int mismatches = launch_checked(k);
    start_ = StartInfo();
    start_.kind = "tas";
    start_.frame = k;
    start_.new_process = true;
    start_.handover_mismatches = mismatches;
    if (rest_ && k > 0 && rest_->branch_s) {  // env._prepare: the stone branch's measured drive, then a closed loop
        std::string err;
        try {
            branch_rest(*rest_->branch_s, rest_->branch_id);
        } catch (const BranchRestFailed &e) {  // the drive's own checks; engine and session faults propagate
            err = e.what();
            if (sess_) sess_->close(true), sess_.reset();
            after_death_ = false;
            ++branch_rest_failures_;
            // env._rules_for_rest: an unknown / ambiguous member refuses here (RouteError: the task fails)
            if (rest_rules(rest_->branch_id).rest_fallback == "error") throw;  // the rules ask for no relaunch
        } catch (...) {
            if (sess_) sess_->close(true), sess_.reset();
            after_death_ = false;
            throw;
        }
        if (err.empty()) {
            start_.kind = "tas_rest";
            start_.branch_s = rest_->branch_s;
            start_.branch_id = rest_branch_id_, start_.branch_phase = rest_branch_phase_;
            init_held(0);
            return;
        }
        // the failed simulation is not a valid hand-over: launch once without the drive, at the same TAS frame
        mismatches = launch_checked(k);
        start_.handover_mismatches = mismatches;
        start_.branch_rest_failures = branch_rest_failures_;
        start_.branch_rest_target = rest_->branch_s;
        start_.branch_rest_error = err;
        init_held(k);
        return;
    }
    if (rest_ && k > 0) {  // env._prepare: drive (with a target), then brake; _init_held then holds nothing
        uint32_t f0 = sess_->frames_done();
        if (rest_->drive) drive(*rest_);
        start_.drive_frames = int64_t(sess_->frames_done()) - int64_t(f0);
        uint32_t f1 = sess_->frames_done();
        brake();
        start_.rest_frames = int64_t(sess_->frames_done()) - int64_t(f1);
        start_.kind = "tas_rest";
        init_held(0);
        return;
    }
    init_held(k);
}

// The arrow combination whose world direction (x = right - left, z = up - down, normalised) has the largest dot product
// with w (Python max(): the first of equal ones, in ARROWS9 order)
static uint32_t best_arrows(double wx, double wz) {
    uint32_t best = 0;
    double bc = -std::numeric_limits<double>::infinity();
    for (int a = 1; a < 9; ++a) {
        uint32_t k = ARROWS9[a];
        double kx = double(bool(k & KEY_RIGHT)) - double(bool(k & KEY_LEFT));
        double kz = double(bool(k & KEY_UP)) - double(bool(k & KEY_DOWN));
        double c = (wx * kx + wz * kz) / std::sqrt(kx * kx + kz * kz);
        if (c > bc) bc = c, best = k;
    }
    return best;
}

static bool episode_event(const GameTracker &t, size_t ev0) {
    for (size_t i = ev0; i < t.events.size(); ++i)
        if (t.events[i].code == Ev::BallOff || t.events[i].code == Ev::LevelFinish || t.events[i].code == Ev::GameOver)
            return true;
    return false;
}

// env._branch_rest: replay the measured drive through the trafo, then close the loop on an arc target of the branch
// line. Never a synthesized transform or a teleport; a paper ball or an unsupported hand-over is refused. A private
// extractor (its projection state is not the episode's)
const BranchRules &BallanceEnv::rest_rules(const std::string &branch_id) {
    // env._rules_for_rest: wire v2: the addressed member's rules (from the installed manifest and its child
    // description; an unknown / ambiguous member is a RouteError); v1: the bundle's
    const std::string &dir = fx_.config().branch_dir;
    Bytes raw;
    if (dir.empty() || !read_file(dir + "/branches.json", raw)) return branch_rules_;
    std::vector<BranchEntry> entries = parse_branch_manifest(raw);
    int found = -1, n = 0;
    for (size_t i = 0; i < entries.size(); ++i)
        if (branch_id.empty() ? entries[i].legacy : entries[i].id == branch_id) found = int(i), ++n;
    if (n != 1) throw RouteError("unknown or ambiguous rest branch '" + branch_id + "'; specify branch:<id>:<s>");
    rest_rules_cache_ = load_branch_rules(dir + "/" + entries[size_t(found)].id, cfg_.level);
    return rest_rules_cache_;
}

void BallanceEnv::branch_rest(double station, const std::string &branch_id) {
    const FeatureConfig &fc = fx_.config();
    rest_branch_id_.clear(), rest_branch_phase_ = -1;
    if (fc.branch_dir.empty()) throw BranchRestFailed("branch rest start needs branch_dir");
    Session &s = *sess_;
    FeatureExtractor fx(fc);
    fx.reference_path(s.tracker());
    const RouteFollower *rf = fx.branch();
    if (!rf) throw BranchRestFailed("no matching branch at handover");
    // wire v2: the addressed member (rest_member: an id, or the unique legacy member) and its child directory
    MultiFollower *multi = dynamic_cast<MultiFollower *>(fx.branch_mut());
    const BranchFollower *br = dynamic_cast<const BranchFollower *>(rf);
    const BranchEntry *entry = nullptr;
    std::string root = fc.branch_dir, key;
    if (multi) {
        int i;
        try {
            i = multi->rest_member(branch_id);
        } catch (const RouteError &e) {
            throw BranchRestFailed(e.what());
        }
        entry = &multi->entries[size_t(i)];
        br = multi->members[size_t(i)].get();
        key = entry->id;
        root += "/" + key;
    }
    auto z = load_npz(root + "/drive.npz");
    char ln[16];
    std::snprintf(ln, sizeof ln, "/L%02d.npz", cfg_.level);
    if (z.at("record_sha256").unicode() != starts_->record_sha256 ||
        z.at("branch_sha256").unicode() != file_sha256(root + ln))
        throw BranchRestFailed("drive replay does not match TAS recording or branch asset");
    const NpyArray &sa = z.at("samples");
    std::vector<double> samples = sa.f64();
    int64_t prefix = z.at("prefix_frame").ints().at(0);
    size_t ns = samples.size() / 2;
    if (sa.shape.size() != 2 || sa.shape[1] != 2 || ns == 0) throw BranchRestFailed("drive.npz: samples");
    double lo = samples[1], hi = samples[1];
    for (size_t i = 0; i < ns; ++i) lo = std::min(lo, samples[2 * i + 1]), hi = std::max(hi, samples[2 * i + 1]);
    if (start_frame_ != prefix || !(lo <= station && station <= hi))
        throw BranchRestFailed("prefix frame or target outside the recorded branch drive");
    size_t best = 0;  // np.argmin(abs(samples[:, 1] - station)): the first minimum
    for (size_t i = 1; i < ns; ++i)
        if (std::abs(samples[2 * i + 1] - station) < std::abs(samples[2 * best + 1] - station)) best = i;
    int64_t offset = int64_t(samples[2 * best]);
    const NpyArray &fr = z.at("frames");
    size_t nf = fr.data.size() / 8;
    if (offset < 0 || size_t(offset) > nf) throw BranchRestFailed("drive.npz: offset outside the frames");
    // the material and sector the replay must produce: a phased member's phase at the station (the newer phase owns a
    // boundary), its when.ball and when.sector (else the line's sector at the station)
    int phase = 0, required = br->ball_type, required_sector = br->sector;
    if (multi && !entry->legacy) {
        int found = -1;
        for (size_t i = 0; i < entry->phases.size(); ++i)
            if (entry->phases[i].lo <= station && station <= entry->phases[i].hi) found = int(i);
        if (found < 0) throw BranchRestFailed("the rest station is in no phase of the branch");
        phase = found;
        const BranchGate &cond = entry->phases[size_t(phase)].when;
        required = *cond.ball;
        int64_t i0, i1;
        br->alt->segments(station, station, i0, i1);
        required_sector = cond.sector ? *cond.sector : int(br->alt->seg_sector[size_t(i0)]);
    }
    std::vector<Frame> frames(static_cast<size_t>(offset));
    for (size_t i = 0; i < frames.size(); ++i)
        std::memcpy(&frames[i].delta, fr.data.data() + 8 * i, 4), std::memcpy(&frames[i].keys, fr.data.data() + 8 * i + 4, 4);
    for (size_t b = 0; b < frames.size(); b += 512) {
        std::vector<Frame> blk(frames.begin() + std::ptrdiff_t(b), frames.begin() + std::ptrdiff_t(std::min(frames.size(), b + 512)));
        s.act(blk, 1, episode_stop());
        if (s.ended() || s.tracker().flags.ball_off) throw BranchRestFailed("ball lost during branch drive replay");
    }
    V3 target = br->alt->point_at(station);
    if (s.tracker().ball_type() != required || s.tracker().sector() != required_sector)
        throw BranchRestFailed("branch drive did not produce the required ball type and sector");
    const BranchRules &rules = br->rules;
    RestStart r;
    r.drive = true, r.x = target[0], r.z = target[2];
    drive(r, rules);
    try {
        brake(rules);
    } catch (const BallLostBraking &e) {  // env._prepare catches it like the drive's own checks: the TAS fallback
        throw BranchRestFailed(e.what());
    }
    if (multi) multi->restore(key, phase, station);  // the verified cursor for the hand-over check
    FeatureInfo fi;
    fx.compute(s.tracker(), &fi);
    const Observation *o = s.obs();
    double d = 1e9;
    if (o && o->ball)
        d = std::sqrt((o->ball->pos[0] - target[0]) * (o->ball->pos[0] - target[0]) +
                      (o->ball->pos[1] - target[1]) * (o->ball->pos[1] - target[1]) +
                      (o->ball->pos[2] - target[2]) * (o->ball->pos[2] - target[2]));
    double vxz = std::sqrt(fi.vel[0] * fi.vel[0] + fi.vel[2] * fi.vel[2]);
    double stop = rules.rest_stop_speed ? *rules.rest_stop_speed : cfg_.rest_speed;
    if ((rules.rest_require_support && !fi.supported) || !(fi.has_branch && fi.branch) ||
        d > rules.rest_position_tolerance || vxz > stop)
        throw BranchRestFailed("branch rest target not reached at rest on the correct surface");
    if (multi) {
        if (s.tracker().ball_type() != required || s.tracker().sector() != required_sector || fi.branch_id != key ||
            fi.branch_phase != phase)
            throw BranchRestFailed("branch phase mismatch");
        rest_branch_id_ = key, rest_branch_phase_ = phase;
    }
}

// env._drive: push the ball with world arrows towards (x, z), aiming at a velocity proportional to the remaining
// distance (capped), until it is within 1 unit and slower than rest_speed or max_frames passed
void BallanceEnv::drive(const RestStart &r, const BranchRules &rules) {
    Session &s = *sess_;
    uint32_t n0 = s.frames_done();
    int64_t max_frames = r.max_frames ? *r.max_frames
                                      : (rules.rest_max_drive_frames ? *rules.rest_max_drive_frames : cfg_.rest_max_frames);
    double stop = rules.rest_stop_speed ? *rules.rest_stop_speed : cfg_.rest_speed;
    while (int64_t(s.frames_done() - n0) < max_frames && !s.ended()) {
        const Observation *o = s.obs();
        if (!o || !o->ball) break;
        double px = o->ball->pos[0], pz = o->ball->pos[2];
        double vx = double(o->ball->speed[0]) + double(o->ball->speed_change[0]);
        double vz = double(o->ball->speed[2]) + double(o->ball->speed_change[2]);
        double dx = r.x - px, dz = r.z - pz;
        if (std::sqrt(dx * dx + dz * dz) < rules.drive_arrival_radius && std::sqrt(vx * vx + vz * vz) < stop) break;
        double wx = dx * rules.drive_gain, wz = dz * rules.drive_gain;
        double wn = std::sqrt(wx * wx + wz * wz);
        if (wn > rules.drive_speed_cap) {  // want *= cap / |want|
            double f = rules.drive_speed_cap / wn;
            wx *= f, wz *= f;
        }
        wx -= vx, wz -= vz;
        uint32_t keys = 0;
        if (std::sqrt(wx * wx + wz * wz) > rules.drive_deadband) {
            keys = best_arrows(wx, wz);
            if (o->cams) keys = world_to_camera_keys(keys, o->cams->cams[0].matrix);
        }
        size_t ev0 = s.tracker().events.size();
        s.act({{cfg_.delta, keys}}, uint32_t(rules.rest_control_repeat), episode_stop());
        if (episode_event(s.tracker(), ev0)) break;
    }
}

// env._brake: push against the ball's horizontal velocity until it is slower than rest_speed, release the arrows
void BallanceEnv::brake(const BranchRules &rules) {
    Session &s = *sess_;
    uint32_t n0 = s.frames_done();
    int64_t max_frames = rules.rest_max_brake_frames ? *rules.rest_max_brake_frames : cfg_.rest_max_frames;
    double stop = rules.rest_stop_speed ? *rules.rest_stop_speed : cfg_.rest_speed;
    while (int64_t(s.frames_done() - n0) < max_frames && !s.ended()) {
        const Observation *o = s.obs();
        double vx = 0.0, vz = 0.0;
        if (o && o->ball) {
            vx = double(o->ball->speed[0]) + double(o->ball->speed_change[0]);
            vz = double(o->ball->speed[2]) + double(o->ball->speed_change[2]);
        }
        double n = std::sqrt(vx * vx + 0.0 + vz * vz);
        if (n < stop) break;
        uint32_t keys = best_arrows(-vx / n, -vz / n);
        if (o && o->cams) keys = world_to_camera_keys(keys, o->cams->cams[0].matrix);
        size_t ev0 = s.tracker().events.size();
        s.act({{cfg_.delta, keys}}, uint32_t(rules.rest_control_repeat), episode_stop());
        if (episode_event(s.tracker(), ev0)) break;
    }
    s.act({{cfg_.delta, 0u}}, 1, episode_stop());  // release the arrows
    const GameTracker &t = s.tracker();
    if (s.ended() || t.flags.ball_off || !t.flags.ball_nav) {
        finish_process();
        throw BallLostBraking("at-rest start at frame " + std::to_string(start_frame_) + ": the ball was lost while braking");
    }
}

// env._init_held: none from level load; after a TAS prefix the keys of its last frame, each with the record's camera
// quarter when it went down; the camera-turn clock from the record's last turn press
void BallanceEnv::init_held(int64_t k) {
    held_ = 0;
    for (int b = 0; b < 4; ++b) press_known_[b] = false, press_q_[b].reset();
    ms_since_turn_ = std::numeric_limits<double>::infinity();
    if (k <= 0 || !starts_) return;
    const std::vector<Frame> &fr = starts_->frames;
    size_t n = size_t(std::min<int64_t>(k, int64_t(fr.size())));
    auto turn = [&](size_t i) { return (fr[i].keys & KEY_SHIFT) && (fr[i].keys & (KEY_LEFT | KEY_RIGHT)); };
    std::optional<size_t> e;
    for (size_t i = 0; i < n; ++i)
        if (turn(i) && (i == 0 || !turn(i - 1))) e = i;
    if (e) {
        std::vector<double> d;
        for (size_t i = *e; i < n; ++i) d.push_back(double(fr[i].delta));
        ms_since_turn_ = numpy_sum(d.data(), d.size());
    }
    if (!n) return;
    uint32_t last = fr[n - 1].keys & ARROW_MASK;
    if (!last) return;
    const std::vector<int8_t> &cq = starts_->camera_quarter;
    for (int b = 0; b < 4; ++b) {
        uint32_t bit = ARROW_BITS[b];
        if (!(last & bit)) continue;
        size_t j = n - 1;
        while (j > 0 && (fr[j - 1].keys & bit)) --j;
        press_known_[b] = true;
        if (j < cq.size() && cq[j] >= 0) press_q_[b] = int(cq[j]);
        else press_q_[b].reset();
    }
    held_ = last;
}

void BallanceEnv::intro() {
    Session &s = *sess_;
    GameTracker &t = s.tracker();
    if (!t.flags.started) s.run_until(cfg_.idle_keys, event_mask({Ev::StartLevel}), cfg_.delta);
    if (cfg_.level == 1 && cfg_.skip_tutorial && !t.flags.cam_nav && !s.ended()) {
        std::vector<Frame> pulse{{cfg_.delta, KEY_Q}, {cfg_.delta, 0}};
        for (int i = 0; i < 200; ++i) {
            if (s.ended() || t.flags.cam_nav || t.flags.ball_nav) break;
            s.act(pulse, 1, event_mask({Ev::CamNavActive, Ev::BallNavActive, Ev::GameOver}));
        }
    }
    if (!t.flags.ball_nav && !s.ended())
        s.run_until(cfg_.idle_keys, event_mask({Ev::BallNavActive, Ev::GameOver, Ev::LevelFinish}), cfg_.delta);
}

std::vector<float> BallanceEnv::features(StepInfo *info) {
    FeatureInfo fi;
    std::vector<float> v = fx_.compute(sess_->tracker(), &fi);
    if (rec_) rec_->put(K_FEAT, v.data(), v.size() * 4);
    if (info) {
        // env._info: the route's branch fields, else (no IVP ball: transforms, finish) the branch's last timing, so
        // the route clock is never compared with the absolute TAS IGT
        if (fi.has_branch) {
            info->has_branch = true, info->branch = fi.branch, info->local_s = fi.local_s, info->has_local_s = true;
            info->ref_elapsed_ms = fi.ref_elapsed_ms, info->ref_remaining_ms = fi.ref_remaining_ms;
            if (fi.branch_multi) {  // env._info: the active member's rules (the defaults on the main line)
                info->branch_multi = true, info->branch_id = fi.branch_id, info->branch_phase = fi.branch_phase;
                const auto *m = dynamic_cast<const MultiFollower *>(fx_.branch());
                int i = m && !fi.branch_id.empty() ? m->index_of(fi.branch_id) : -1;
                branch_rules_ = i >= 0 ? m->members[size_t(i)]->rules : BranchRules();
            }
        } else if (const RouteFollower *b = fx_.branch(); b && b->timing.set) {
            info->has_branch = true, info->branch = b->timing.branch;
            info->ref_elapsed_ms = b->timing.ref_elapsed_ms, info->ref_remaining_ms = b->timing.ref_remaining_ms;
        }
        info->feat = std::move(fi);
    }
    return v;
}

std::optional<std::vector<float>> BallanceEnv::reset(bool relaunch, StepInfo *info) {
    Session *s = sess_.get();
    bool had = s != nullptr;
    if (s && !s->ended() && after_death_) {
        uint32_t f0 = s->frames_done();
        s->run_until(cfg_.idle_keys, event_mask({Ev::BallNavActive, Ev::GameOver, Ev::LevelFinish}), cfg_.delta, 4096,
                     cfg_.respawn_limit_frames);
        ms_since_turn_ += double(s->frames_done() - f0) * double(cfg_.delta);
        set_held(cfg_.idle_keys);
        invalidate_held();
        after_death_ = false;
        if (s->ended() || !s->tracker().flags.ball_nav) finish_process();
    } else if (s) {
        finish_process();
    }
    if (!sess_ && had && !relaunch) {
        if (info) info->process_ended = true;
        return std::nullopt;
    }
    if (!sess_) {
        start_process();
        if (sess_->ended() || !sess_->tracker().flags.ball_nav) throw Error("run ended before the player got control");
    } else {  // a respawn in the same process
        start_ = StartInfo();
        start_.kind = "respawn";
        start_.frame = sess_->frames_done();
        start_.new_process = false;
    }
    ep_steps_ = ep_frames_ = ep_pause_frames_ = 0;
    fx_.reset_episode();
    if (!start_.branch_id.empty()) {  // env.reset: the verified rest start's route cursor
        fx_.reference_path(sess_->tracker());
        auto *m = dynamic_cast<MultiFollower *>(fx_.branch_mut());
        if (!m) throw Error("a multi-branch rest start without the multi-branch routes");
        m->restore(start_.branch_id, start_.branch_phase, *start_.branch_s);
    }
    if (rec_) rec_->put(K_RESET, Bytes());
    StepInfo tmp;
    StepInfo *pi = info ? info : &tmp;
    std::vector<float> o = features(pi);
    reward_.begin(sess_->tracker(), pi->feat.path ? std::optional<double>(pi->feat.path->s) : std::nullopt);
    if (info) {
        info->frames_done = sess_->frames_done();
        info->sr_ms = sess_->tracker().sr_ms();
        info->sector = sess_->tracker().sector();
    }
    return o;
}

std::optional<int> BallanceEnv::cam_quarter() const {
    const Observation *o = sess_ ? sess_->obs() : nullptr;
    if (!o || !o->cams) return std::nullopt;
    const float *m = o->cams->cams[0].matrix;
    double yaw = std::atan2(double(m[6]), double(m[8]));
    int64_t q = int64_t(std::nearbyint(yaw / (PI / 2)));  // Python round(): half to even
    return int(((q % 4) + 4) % 4);
}

double BallanceEnv::camera_dev() const {
    const Observation *o = sess_ ? sess_->obs() : nullptr;
    if (!o || !o->cams) return 0.0;
    const float *m = o->cams->cams[0].matrix;
    return std::abs(std::remainder(std::atan2(double(m[6]), double(m[8])), PI / 2));
}

bool BallanceEnv::camera_off_quarter() const {
    return camera_dev() > cfg_.camera_quarter_tol && ms_since_turn_ > cfg_.camera_turn_grace_ms;
}

void BallanceEnv::invalidate_held() {
    for (int b = 0; b < 4; ++b)
        if (held_ & ARROW_BITS[b]) press_known_[b] = true, press_q_[b].reset();
}

void BallanceEnv::set_held(uint32_t keys) {
    keys &= ARROW_MASK;
    std::optional<int> q = cam_quarter();
    for (int b = 0; b < 4; ++b) {
        uint32_t bit = ARROW_BITS[b];
        if ((keys & bit) && !(held_ & bit)) press_known_[b] = true, press_q_[b] = q;
        else if (!(keys & bit)) press_known_[b] = false, press_q_[b].reset();
    }
    held_ = keys;
}

uint32_t BallanceEnv::stale(uint32_t keys) const {
    std::optional<int> q = cam_quarter();
    uint32_t out = 0;
    for (int b = 0; b < 4; ++b) {
        uint32_t bit = ARROW_BITS[b];
        if ((keys & bit) && (held_ & bit) && q) {
            std::optional<int> p = press_known_[b] ? press_q_[b] : q;  // _press_q.get(b, q)
            if (p != q) out |= bit;
        }
    }
    return out;
}

bool BallanceEnv::pause_budget_spent() const {
    return cfg_.max_episode_pause_frames && ep_pause_frames_ >= *cfg_.max_episode_pause_frames;
}

std::string BallanceEnv::esc() {
    Session &s = *sess_;
    bool was = s.tracker().flags.paused;
    std::vector<Frame> f;
    for (int i = 0; i < cfg_.esc_frames; ++i) f.push_back({cfg_.delta, i == 0 ? KEY_ESC : 0u});
    s.act(f, 1, episode_stop());
    bool now = s.tracker().flags.paused;
    return now == was ? "no_toggle" : (now ? "paused" : "unpaused");
}

std::vector<float> BallanceEnv::step(int action, bool &terminated, bool &truncated, StepInfo &info, double *reward_out) {
    Session *s = sess_.get();
    if (!s || s->ended()) throw Error("step() after the run ended; call reset()");
    if (action < 0 || action >= int(actions_.size())) throw Error("action out of range");
    uint32_t a = actions_[size_t(action)];
    GameTracker &t = s->tracker();
    uint32_t n0 = s->frames_done();
    size_t ev0 = t.events.size();
    bool paused0 = t.flags.paused;
    info = StepInfo();
    try {
        uint32_t keys = a;
        if (cfg_.world_actions) {
            const Observation *o = s->obs();
            if (o && o->cams) keys = world_to_camera_keys(a, o->cams->cams[0].matrix);
        }
        if (t.flags.paused && cfg_.mask_keys_while_paused) keys = cfg_.pause_keys;
        uint32_t st = repress_ ? stale(keys) : 0;
        int R = cfg_.action_repeat;
        if (st && R > 1) {
            std::vector<Frame> fr{{cfg_.delta, keys & ~st}};
            for (int i = 1; i < R; ++i) fr.push_back({cfg_.delta, keys});
            s->act(fr, 1, episode_stop());
            uint32_t done = s->frames_done() - n0;
            if (done >= 1) set_held(keys & ~st);
            if (done >= 2) set_held(keys);
        } else {
            s->act({{cfg_.delta, keys}}, uint32_t(R), episode_stop());
            if (s->frames_done() > n0) set_held(keys);
        }
        if (paused0 || t.flags.paused) ep_pause_frames_ += s->frames_done() - n0;
        if (paused0 && !t.flags.paused) invalidate_held();
        if (!s->ended() && t.flags.paused &&
            ((cfg_.max_pause_frames && int64_t(t.paused_frames()) >= *cfg_.max_pause_frames) || pause_budget_spent())) {
            uint32_t n1 = s->frames_done();
            esc();
            ep_pause_frames_ += s->frames_done() - n1;
        }
    } catch (const PeerGone &e) {
        info.error = std::string("PeerGone: ") + e.what();
    } catch (const Timeout &e) {
        info.error = std::string("Timeout: ") + e.what();
    }
    if (info.error) {
        // BallanceEnv.step: the game died or hung; truncate (s.ended = True)
        s->close(true);
    }
    bool ended = s->ended() || info.error.has_value();
    ++ep_steps_;
    ep_frames_ += s->frames_done() - n0;
    bool died = false, finished = false, game_over = false;
    for (size_t i = ev0; i < t.events.size(); ++i) {
        info.events.push_back(t.events[i]);
        died = died || t.events[i].code == Ev::BallOff;
        finished = finished || t.events[i].code == Ev::LevelFinish;
        game_over = game_over || t.events[i].code == Ev::GameOver;
    }
    terminated = died || finished || game_over;
    truncated = false;
    if (!terminated && (ended || (cfg_.max_episode_frames && ep_frames_ >= *cfg_.max_episode_frames))) truncated = true;
    ms_since_turn_ += double(s->frames_done() - n0) * double(cfg_.delta);
    bool cam_bad = !terminated && !ended && camera_off_quarter();
    if (cam_bad) terminated = true, truncated = false;
    std::vector<float> obs = features(&info);
    const auto &pr = info.feat.path;
    double r = reward_.step(t, pr ? std::optional<double>(pr->s) : std::nullopt, died, finished, terminated,
                            pr && pr->jump, std::optional<bool>(info.feat.supported));
    if (cam_bad) {  // no credit for the episode: its return becomes -death_penalty
        double pen = -reward_.total - reward_.cfg.death_penalty;
        r += pen;
        reward_.total += pen;
    }
    if (reward_out) *reward_out = r;
    info.frames_done = s->frames_done();
    info.sr_ms = t.sr_ms();
    info.sector = t.sector();
    info.camera_off_quarter = cam_bad;
    if (terminated || truncated) {
        info.episode_end = cam_bad ? "camera_off_quarter"
                                   : died ? "BallOff" : finished ? "LevelFinish" : game_over ? "GameOver" : "truncated";
        after_death_ = died && cfg_.per_life && !ended;
        if (!after_death_) {
            if (info.error) sess_.reset();
            else finish_process();
        }
    }
    return obs;
}

}  // namespace ballrain
