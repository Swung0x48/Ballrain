#include "ballrain/env.h"

#include <cmath>
#include <limits>

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
    return c;
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
                         StreamWriter *rec)
    : cfg_(std::move(cfg)), eng_(std::move(engine)), run_root_(std::move(run_root)), rec_(rec), fx_(std::move(features)) {
    for (auto k : ARROWS9) actions_.push_back(k);
    if (cfg_.esc_frames < 3) throw Error("esc_frames must be >= 3");
    repress_ = cfg_.repress_stale_keys ? *cfg_.repress_stale_keys : cfg_.world_actions;
    ms_since_turn_ = std::numeric_limits<double>::infinity();
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

void BallanceEnv::start_process() {
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
    sess_ = std::make_unique<Session>(sc, eng_, rec_);
    sess_->start();
    try {
        intro();
    } catch (...) {
        sess_->close(true);
        sess_.reset();
        throw;
    }
    // _init_held for a start from level load
    held_ = 0;
    for (int b = 0; b < 4; ++b) press_known_[b] = false, press_q_[b].reset();
    ms_since_turn_ = std::numeric_limits<double>::infinity();
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
    if (info) info->feat = std::move(fi);
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
    }
    ep_steps_ = ep_frames_ = ep_pause_frames_ = 0;
    fx_.reset_episode();
    if (rec_) rec_->put(K_RESET, Bytes());
    std::vector<float> o = features(info);
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

std::vector<float> BallanceEnv::step(int action, bool &terminated, bool &truncated, StepInfo &info) {
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
