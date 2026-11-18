#include "ballrain/session.h"

#include <algorithm>
#include <cctype>
#include <cstring>

#include "platform.h"
#include "ballrain/json.h"

namespace ballrain {

static const uint32_t BR_PING = 0x100, BR_PONG = 0x101, BR_STATE = 0x102, BR_OBS = 0x103, BR_ACT = 0x104;

// ---------------------------------------------------------------------------------------------- level sources

std::optional<Bytes> FileLevelSource::load_full(bool level_loaded, bool full) {
    if (level_loaded && !full) return std::nullopt;
    Bytes b;
    if (!read_file(path_, b)) return std::nullopt;
    return b;
}

std::optional<Bytes> FileLevelSource::load_inc(uint64_t inc_pos) {
    Bytes tail;
    uint64_t size = 0;
    if (!read_file_from(path_ + ".inc", inc_pos, tail, &size)) return std::nullopt;
    if (size <= inc_pos) return std::nullopt;
    return tail;
}

// ---------------------------------------------------------------------------------------------- ingest

void Ingest::load_level(bool full) {
    auto b = src_->load_full(level_loaded, full);
    if (rec_) rec_->put(K_LVL_FULL, b ? Bytes(1, '\1') + *b : Bytes(1, '\0'));
    if (!b) return;
    size_t len = 0;
    LevelMessage lm = parse_level_message(b->data(), b->size(), &len);
    if (digest) digest->level('F', b->data(), b->size());
    tracker.merge_level(lm);
    level_loaded = true;
}

void Ingest::load_increments() {
    auto tail = src_->load_inc(inc_pos);
    if (rec_) rec_->put(K_LVL_INC, tail ? Bytes(1, '\1') + *tail : Bytes(1, '\0'));
    if (!tail) return;
    size_t used = 0;
    auto msgs = parse_level_stream(tail->data(), tail->size(), &used);
    if (digest) digest->level('I', tail->data(), used);
    for (auto &lm : msgs) tracker.merge_level(lm);
    inc_pos += used;
}

void Ingest::mark_ended(const ChanHeader &h) {
    ended = true;
    end_reason = h.end_reason;
}

void Ingest::ingest(Observation obs, const ChanHeader &h) {
    if (obs.frames_done < frames_done) throw Error("frames_done went back");
    frames_done = obs.frames_done;
    if (obs.hash) trace_hash = obs.hash->trace_hash;
    bool start = obs.has_event(Ev::StartLevel);
    if (start || !level_loaded) load_level(start);
    if (level_loaded && (!obs.phys.empty() || start)) load_increments();
    bool lf = obs.has_event(Ev::LevelFinish);
    std::optional<double> sren_finish, time_sr;
    if (obs.sren && obs.sren->level_finished) sren_finish = obs.sren->sr_double_at_finish;
    if (obs.time) time_sr = obs.time->sr_double;
    bool end = h.mod_state == ModState::ENDED || h.mod_state == ModState::ERROR || obs.has_event(Ev::RunEnd) ||
               obs.final();
    tracker.update(std::move(obs));
    if (lf && !sr_at_levelfinish) {
        if (sren_finish) sr_at_levelfinish = sren_finish;
        else if (time_sr) sr_at_levelfinish = time_sr;
    }
    if (end) mark_ended(h);
}

// ---------------------------------------------------------------------------------------------- socket channel

SocketChannel::SocketChannel() : listener_(platform::Listener::listen_loopback(0)) {}
SocketChannel::~SocketChannel() { close(); }

std::string SocketChannel::name() const { return "tcp:127.0.0.1:" + std::to_string(listener_ ? listener_->port() : 0); }

bool SocketChannel::recv(double deadline, uint32_t &type, Bytes &payload) {
    if (closed_peer_) return false;
    uint32_t h[2];
    try {
        if (!conn_->recv_exact(h, 8, deadline)) {
            closed_peer_ = true;
            return false;
        }
        type = h[0];
        payload.assign(h[1], '\0');
        if (h[1] && !conn_->recv_exact(&payload[0], h[1], deadline)) {
            closed_peer_ = true;
            return false;
        }
    } catch (const Error &e) {
        if (std::string(e.what()) == "timeout") throw Timeout("no message from the game");
        throw;
    }
    return true;
}

void SocketChannel::send(uint32_t type, const Bytes &payload) {
    uint32_t h[2] = {type, uint32_t(payload.size())};
    Bytes b(reinterpret_cast<const char *>(h), 8);
    b += payload;
    if (!conn_ || !conn_->send_all(b.data(), b.size())) throw PeerGone("send failed");
}

void SocketChannel::apply_state(const Bytes &p) {
    Reader r(p);
    hdr_.mod_state = r.u32(), hdr_.end_reason = r.u32(), hdr_.frames_done = r.u32(), hdr_.last_error = r.u32();
}

ChanHeader SocketChannel::wait_hello(double timeout_s, const std::function<bool()> &alive) {
    double deadline = platform::monotonic_s() + timeout_s;
    while (!conn_) {
        double left = deadline - platform::monotonic_s();
        if (left <= 0) throw Timeout("the game did not connect");
        conn_ = listener_->accept(alive ? std::min(left, 0.25) : left);
        if (!conn_ && alive && !alive()) throw PeerGone("the game process exited before it connected");
    }
    send(BR_PING, Bytes());
    for (;;) {
        uint32_t t;
        Bytes p;
        if (!recv(deadline, t, p)) throw PeerGone("game process exited before hello");
        if (t == BR_STATE) {
            apply_state(p);
            if (hdr_.mod_state == ModState::ERROR)
                throw Error("game refused to run (last_error " + std::to_string(hdr_.last_error) + ", see its log)");
        } else if (t == BR_PONG) {
            Reader r(p);
            hdr_.mod_pid = r.u32(), hdr_.obs_versions = r.u32(), hdr_.mod_state = r.u32();
            Bytes v = r.raw(180);
            hdr_.mod_version = v.substr(0, v.find('\0'));
            return hdr_;
        }
    }
}

std::optional<Bytes> SocketChannel::wait_obs(double timeout_s) {
    double deadline = timeout_s > 0 ? platform::monotonic_s() + timeout_s : 0.0;
    for (;;) {
        uint32_t t;
        Bytes p;
        if (!recv(deadline, t, p)) {
            if (hdr_.mod_state == ModState::ENDED || hdr_.mod_state == ModState::ERROR) return std::nullopt;
            throw PeerGone("game process closed the connection");
        }
        if (t == BR_STATE) {
            apply_state(p);
        } else if (t == BR_OBS) {
            Reader r(p);
            hdr_.mod_state = r.u32(), hdr_.end_reason = r.u32(), hdr_.frames_done = r.u32(), hdr_.last_error = r.u32();
            hdr_.mod_seq = r.u64();
            return p;
        }
    }
}

Bytes SocketChannel::send_action(const Action &a) {
    ++py_seq_;
    Bytes b = encode_action(a, py_seq_);
    send(BR_ACT, b);
    return b;
}

void SocketChannel::close() {
    if (conn_) conn_->close();
    conn_.reset();
    listener_.reset();
}

// ---------------------------------------------------------------------------------------------- launch

std::string py_float_str(double v) { return py_float_repr(v); }

std::vector<std::string> engine_argv(const EngineConfig &e) {
    std::vector<std::string> a{platform::absolute_path(e.binary), "--ballrain-agent", "--root", platform::absolute_path(e.root)};
    auto opt = [&](const char *flag, const std::optional<int64_t> &v) {
        if (v) a.push_back(flag), a.push_back(std::to_string(*v));
    };
    opt("--boot-ticks", e.boot_ticks);
    opt("--tas-offset", e.hidden_ticks);
    opt("--tas-movement-check", e.reset_nmc);
    opt("--tas-seed", e.reset_seed);
    opt("--tas-reset-frame", e.reset_frame);
    if (!e.legacy_edit) a.push_back("--tas-no-legacy");
    if (e.reset_precision) a.push_back("--tas-reset-precision");
    if (e.orig_random) a.push_back("--tas-orig-random");
    return a;
}

static bool starts_ballrain(const std::string &k) {
    static const char *p = "BALLRAIN_";
    if (k.size() < 9) return false;
    for (size_t i = 0; i < 9; ++i)
        if (char(std::toupper(static_cast<unsigned char>(k[i]))) != p[i]) return false;
    return true;
}

std::vector<std::pair<std::string, std::string>> engine_env(const SessionConfig &s, const EngineConfig &e,
                                                            const std::string &sock) {
    std::vector<std::pair<std::string, std::string>> out;
    for (auto &kv : platform::environment())
        if (!starts_ballrain(kv.first)) out.push_back(kv);
    auto set = [&](const std::string &k, const std::string &v) {
        for (auto &kv : out)
            if (kv.first == k) {
                kv.second = v;
                return;
            }
        out.emplace_back(k, v);
    };
    std::string d = platform::absolute_path(s.out_dir);
    set("BALLRAIN_MODE", "agent");
    set("BALLRAIN_LEVEL", std::to_string(s.level));
    set("BALLRAIN_SOCK", sock);
    set("BALLRAIN_PARENT_PID", std::to_string(platform::self_pid()));
    set("BALLRAIN_TIMEOUT_MS", std::to_string(s.timeout_ms));
    set("BALLRAIN_LEVELMSG", d + "/level.msg");
    set("BALLRAIN_FRAMELOG", d + "/framelog.bin");
    set("BALLRAIN_BODY_RADIUS", py_float_str(s.body_radius));
    set("BALLRAIN_SR_TAIL", std::to_string(s.sr_tail));
    if (s.skip_render) set("BALLRAIN_SKIP_RENDER", "1");
    if (s.exit_game) set("BALLRAIN_EXIT", "1");
    set("BALLRAIN_LOADS", "1");
    if (s.max_frames) set("BALLRAIN_MAX_FRAMES", std::to_string(*s.max_frames));
    if (!s.prefix.empty()) set("BALLRAIN_PREFIX", platform::absolute_path(s.prefix));
    if (s.events_txt) set("BALLRAIN_EVENTS", d + "/events.txt");
    for (auto &kv : e.extra_env) set(kv.first, kv.second);
    bool have_order = false;
    for (auto &kv : out) have_order = have_order || kv.first == "BALLRAIN_HL_BALL_ORDER";
    if (!have_order) set("BALLRAIN_HL_BALL_ORDER", e.ball_order);
    return out;
}

// ---------------------------------------------------------------------------------------------- session

Session::Session(SessionConfig cfg, EngineConfig eng, StreamWriter *rec)
    : cfg_(std::move(cfg)), eng_(std::move(eng)), rec_(rec) {
    levelmsg_ = platform::absolute_path(cfg_.out_dir) + "/level.msg";
    src_ = std::make_unique<FileLevelSource>(levelmsg_);
    ing_ = std::make_unique<Ingest>(cfg_.level, src_.get(), rec_);
}

Session::~Session() { close(true); }

int Session::pid() const { return proc_ ? proc_->pid() : 0; }

const Observation &Session::start() {
    if (!platform::make_dirs(cfg_.out_dir)) throw Error("cannot create " + cfg_.out_dir);
    std::string d = platform::absolute_path(cfg_.out_dir);
    for (const char *n : {"framelog.bin", "level.msg", "trace.txt", "events.txt"}) {
        platform::remove_file(d + "/" + n);
        platform::remove_file(d + "/" + n + ".inc");
    }
    if (eng_.binary.empty() || !file_exists(eng_.binary)) throw Error("headless binary not found: " + eng_.binary);
    if (eng_.root.empty() || !file_exists(eng_.root + "/base.cmo")) throw Error("no base.cmo in the game root " + eng_.root);
    chan_ = std::make_unique<SocketChannel>();
    platform::Process::Spec spec;
    spec.argv = engine_argv(eng_);
    spec.env = engine_env(cfg_, eng_, chan_->name());
    if (eng_.log) spec.stderr_path = d + "/headless.log";
    spec.background = eng_.background;
    try {
        timing.launch = platform::monotonic_s();
        proc_ = platform::Process::spawn(spec);
        hello_ = chan_->wait_hello(cfg_.hello_timeout_s, [this] { return proc_->running(); });
        timing.hello = platform::monotonic_s();
        if (rec_) rec_->put(K_HELLO, hello_.mod_version);
        if (hello_.mod_version.find("headless") == std::string::npos)
            throw Error("hello '" + hello_.mod_version + "' is not a headless one");
        const Observation *o = next_obs();
        timing.first_obs = platform::monotonic_s();
        if (!o) throw Error("game ended before the first observation");
        return *o;
    } catch (...) {
        close(true);
        throw;
    }
}

const Observation *Session::next_obs() {
    for (;;) {
        auto data = chan_->wait_obs(cfg_.obs_timeout_s);
        if (!data) {
            ing_->mark_ended(chan_->header());
            if (chan_->header().mod_state == ModState::ERROR)
                throw Error("engine error (last_error " + std::to_string(chan_->header().last_error) + ")");
            return nullptr;
        }
        if (rec_) rec_->put(K_OBS, *data);
        if (ing_->digest) ing_->digest->obs(data->data() + 24, data->size() - 24);
        Observation o = parse_observation(data->data() + 24, data->size() - 24);
        bool prefix = (o.flags & OBS_PREFIX) && !o.final();
        ing_->ingest(std::move(o), chan_->header());
        if (prefix && !ing_->ended) {
            Action ack;
            Bytes b = chan_->send_action(ack);
            if (rec_) rec_->put(K_ACT, b);
            continue;
        }
        return obs();
    }
}

const Observation *Session::step(const Action &a) {
    if (ing_->ended) throw Error("session has ended");
    Bytes b = chan_->send_action(a);
    if (rec_) rec_->put(K_ACT, b);
    try {
        return next_obs();
    } catch (const PeerGone &) {
        ing_->mark_ended(chan_->header());
        throw;
    } catch (const Timeout &) {
        ing_->mark_ended(chan_->header());
        throw;
    }
}

const Observation *Session::act(const std::vector<Frame> &frames, uint32_t repeat, uint64_t stop_mask, uint16_t flags,
                                uint32_t obs_mask) {
    Action a;
    a.frames = frames;
    a.repeat = repeat;
    a.stop_mask = stop_mask;
    a.flags = flags;
    a.obs_mask = obs_mask;
    return step(a);
}

const Observation *Session::run_until(uint32_t keys, uint64_t mask, float delta, uint32_t chunk, uint32_t limit) {
    std::vector<Frame> frames{{delta, keys}};
    uint32_t done = 0;
    const Observation *o = obs();
    while (!ing_->ended && done < limit) {
        uint32_t n0 = ing_->frames_done;
        o = act(frames, std::min(chunk, limit - done), mask);
        done += ing_->frames_done - n0;
        if (!o) break;
        bool hit = false;
        for (auto &e : o->events)
            if (e.code < 64 && ((mask >> e.code) & 1)) hit = true;
        if (hit) break;
    }
    return o;
}

void Session::stop(double wait_s) {
    if (!chan_) return;
    if (!ing_->ended) {
        try {
            Action a;
            a.flags = ACT_STOP;
            Bytes b = chan_->send_action(a);
            if (rec_) rec_->put(K_ACT, b);
            const Observation *o = nullptr;
            try {
                o = next_obs();
            } catch (const Timeout &) {
            } catch (const PeerGone &) {
            }
            if (o && !ing_->ended) ing_->mark_ended(chan_->header());
        } catch (const PeerGone &) {
        }
        ing_->ended = true;
    }
    finish_process(wait_s);
}

void Session::finish_process(double wait_s) {
    if (proc_) {
        if (!proc_->wait(wait_s)) {
            proc_->kill();
            proc_->wait(5.0);
        }
        proc_.reset();
    }
}

void Session::close(bool kill) {
    if (proc_) finish_process(kill ? 0.0 : 30.0);
    if (chan_) chan_->close(), chan_.reset();
}

}  // namespace ballrain
