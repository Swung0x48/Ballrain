#include "ballrain/starts.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <set>

#include "ballrain/npz.h"
#include "ballrain/sha256.h"

namespace ballrain {

static const size_t MAX_ASSET_BYTES = 64 * 1024 * 1024;
static const int STARTS_VERSION = 1;

TasStarts TasStarts::load_pack(const std::string &path, const std::string &expect_sha) {
    Bytes file;
    if (!read_file(path, file)) throw Error("cannot read starts pack " + path);
    if (file.size() > MAX_ASSET_BYTES) throw Error("starts pack too large");
    if (Sha256::hex(file) != expect_sha) throw Error("starts pack sha256 differs from " + expect_sha.substr(0, 12));
    auto z = load_npz_bytes(file);
    // _STARTS_ARRAYS: name -> (dtype, ndim)
    static const std::map<std::string, std::pair<std::string, size_t>> spec = {
        {"record_frames", {"[('delta', '<f4'), ('keys', '<u4')]", 1}}, {"starts_version", {"<i8", 0}},
        {"starts_frames_done", {"<i8", 1}}, {"starts_sector", {"<i8", 1}}, {"starts_sr_double", {"<f8", 1}},
        {"starts_points", {"<i8", 1}}, {"starts_lives", {"<i8", 1}}, {"starts_pos", {"<f8", 2}},
        {"starts_checkpoints", {"<i8", 1}}, {"starts_levelfinish_frame", {"<i8", 0}}, {"starts_sr_final", {"<f8", 0}},
        {"camera_quarter", {"|i1", 1}}, {"camera_dev", {"<f4", 1}}, {"meta_json", {"|u1", 1}}};
    if (z.size() != spec.size()) throw Error("starts pack: unexpected / missing arrays");
    for (auto &kv : spec) {
        auto it = z.find(kv.first);
        if (it == z.end()) throw Error("starts pack: missing " + kv.first);
        if (it->second.descr != kv.second.first || it->second.shape.size() != kv.second.second)
            throw Error("starts pack: " + kv.first + " has dtype " + it->second.descr);
    }
    TasStarts s;
    Json m = Json::parse(z.at("meta_json").data, 4);
    if (m.get_string("format", "") != "ballrain-pool-starts/1") throw Error("starts pack: format");
    s.record = m.at("record").as_string();
    s.record_sha256 = m.at("record_sha256").as_string();
    s.level = int(m.at("level").as_int());
    if (z.at("starts_version").ints()[0] != STARTS_VERSION) throw Error("starts pack version");
    const NpyArray &rf = z.at("record_frames");
    size_t nf = size_t(rf.count());
    if (rf.data.size() != nf * 8) throw Error("starts pack: record frames size");
    s.frames.resize(nf);
    for (size_t i = 0; i < nf; ++i) std::memcpy(&s.frames[i].delta, rf.data.data() + 8 * i, 4), std::memcpy(&s.frames[i].keys, rf.data.data() + 8 * i + 4, 4);
    s.valid = z.at("starts_frames_done").ints();
    s.sector = z.at("starts_sector").ints();
    s.points = z.at("starts_points").ints();
    s.lives = z.at("starts_lives").ints();
    s.checkpoints = z.at("starts_checkpoints").ints();
    s.sr_double = z.at("starts_sr_double").f64();
    auto pos = z.at("starts_pos").f64();
    size_t n = s.valid.size();
    if (s.sector.size() != n || s.points.size() != n || s.lives.size() != n || s.checkpoints.size() != n ||
        s.sr_double.size() != n || z.at("starts_pos").shape != std::vector<int64_t>{int64_t(n), 3})
        throw Error("starts pack: lengths");
    for (size_t i = 0; i < n; ++i) s.pos.push_back({pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]});
    for (double v : s.sr_double)
        if (!std::isfinite(v)) throw Error("starts pack: non-finite sr_double");
    for (double v : pos)
        if (!std::isfinite(v)) throw Error("starts pack: non-finite pos");
    for (float v : z.at("camera_dev").f32())
        if (!std::isfinite(v)) throw Error("starts pack: non-finite camera_dev");
    if (n == 0 || s.valid[0] < 0 || s.valid[n - 1] > int64_t(nf)) throw Error("starts pack: start frames outside the record");
    for (size_t i = 1; i < n; ++i)
        if (s.valid[i] <= s.valid[i - 1]) throw Error("starts pack: start frames not increasing");
    int64_t lf = z.at("starts_levelfinish_frame").ints()[0];
    if (lf >= 0) s.levelfinish_frame = lf;
    s.sr_final = z.at("starts_sr_final").f64()[0];
    for (auto q : z.at("camera_quarter").ints()) s.camera_quarter.push_back(int8_t(q));
    s.camera_dev = z.at("camera_dev").f32();
    if (s.camera_quarter.size() != s.camera_dev.size()) throw Error("starts pack: camera shapes");
    return s;
}

void TasStarts::restrict(const std::vector<bool> &mask) {
    Arrays a;
    for (size_t i = 0; i < valid.size(); ++i)
        if (mask[i]) {
            a.valid.push_back(valid[i]), a.sector.push_back(sector[i]), a.points.push_back(points[i]);
            a.lives.push_back(lives[i]), a.checkpoints.push_back(checkpoints[i]), a.sr_double.push_back(sr_double[i]);
            a.pos.push_back(pos[i]);
        }
    valid = a.valid, sector = a.sector, points = a.points, lives = a.lives, checkpoints = a.checkpoints;
    sr_double = a.sr_double, pos = a.pos;
}

void TasStarts::apply_camera_filter(double tol, double grace_ms) {
    // ms = float(np.median(frames["delta"])) (float32 median: the mean of the two middle values in float32)
    std::vector<float> d;
    for (auto &f : frames) d.push_back(f.delta);
    float ms = 1000.0f / 264.0f;
    if (!d.empty()) {
        std::sort(d.begin(), d.end());
        size_t n = d.size();
        ms = n % 2 ? d[n / 2] : (d[n / 2 - 1] + d[n / 2]) / 2.0f;
    }
    double msd = d.empty() ? 1000.0 / 264.0 : double(ms);
    int64_t grace_frames = int64_t(grace_ms / msd);
    // camera_ok
    size_t nk = frames.size();
    std::vector<int64_t> last(nk);
    int64_t cur = -1000000000;
    bool prev_turn = false;
    for (size_t i = 0; i < nk; ++i) {
        uint32_t k = frames[i].keys;
        bool turn = (k & KEY_SHIFT) && (k & (KEY_LEFT | KEY_RIGHT));
        if (turn && !prev_turn) cur = std::max<int64_t>(cur, int64_t(i));
        prev_turn = turn;
        last[i] = cur;
    }
    float ftol = float(tol);  // numpy compares the float32 dev array with the scalar in float32
    std::vector<bool> ok(valid.size());
    size_t dropped = 0;
    for (size_t i = 0; i < valid.size(); ++i) {
        int64_t k = valid[i];
        float dv = 0.0f;
        if (!camera_dev.empty()) dv = camera_dev[size_t(std::min<int64_t>(std::max<int64_t>(k, 0), int64_t(camera_dev.size()) - 1))];
        bool recent = false;
        if (nk) recent = (k - last[size_t(std::min<int64_t>(std::max<int64_t>(k - 1, 0), int64_t(nk) - 1))]) <= grace_frames;
        ok[i] = dv <= ftol || recent;
        dropped += !ok[i];
    }
    if (dropped) restrict(ok);
}

void TasStarts::exclude(const std::vector<std::pair<int64_t, int64_t>> &ranges) {
    if (ranges.empty()) return;
    std::vector<bool> keep(valid.size(), true);
    for (size_t i = 0; i < valid.size(); ++i)
        for (auto &r : ranges)
            if (valid[i] >= r.first && valid[i] < r.second) keep[i] = false;
    if (!all_) all_ = Arrays{valid, sector, points, lives, checkpoints, sr_double, pos};
    restrict(keep);
}

std::optional<size_t> TasStarts::index_of(int64_t k) const {
    auto it = std::lower_bound(valid.begin(), valid.end(), k);
    if (it == valid.end() || *it != k) return std::nullopt;
    return size_t(it - valid.begin());
}

std::optional<StartState> TasStarts::expected(int64_t k) const {
    auto mk = [&](const std::vector<int64_t> &v, const std::vector<int64_t> &sec, const std::vector<int64_t> &cps,
                  const std::vector<int64_t> &pts, const std::vector<int64_t> &lv, const std::vector<double> &sr,
                  const std::vector<std::array<double, 3>> &p) -> std::optional<StartState> {
        auto it = std::lower_bound(v.begin(), v.end(), k);
        if (it == v.end() || *it != k) return std::nullopt;
        size_t i = size_t(it - v.begin());
        return StartState{k, sec[i], cps[i], pts[i], lv[i], sr[i], p[i]};
    };
    if (auto s = mk(valid, sector, checkpoints, points, lives, sr_double, pos)) return s;
    if (all_) return mk(all_->valid, all_->sector, all_->checkpoints, all_->points, all_->lives, all_->sr_double, all_->pos);
    return std::nullopt;
}

std::vector<Frame> TasStarts::prefix(int64_t k) const {
    return std::vector<Frame>(frames.begin(), frames.begin() + std::min<int64_t>(k, int64_t(frames.size())));
}

}  // namespace ballrain
