// The record stream -> tracker part of the session (no platform dependency: the feature module builds it too)
#include "ballrain/session.h"

#include <algorithm>
#include <cstring>

namespace ballrain {

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

}  // namespace ballrain
