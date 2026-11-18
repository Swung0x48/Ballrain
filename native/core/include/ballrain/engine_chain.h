// The pool's engine-state hash chain (chain v2; ballrain/pool/engine_chain.py is the reference): row digests over the
// canonical OBS1 / LVL1 messages received between two observation rows, SCOR / SREN / IVPO left out.
#pragma once

#include <string>

#include "ballrain/sha256.h"
#include "ballrain/util.h"

namespace ballrain {

class EngineDigest {
public:
    EngineDigest() { reset(); }
    void reset();
    void obs(const char *msg, size_t n);  // one OBS1 message (after the 24-byte socket header)
    void level(char kind, const char *data, size_t n);  // 'F': level.msg as read; 'I': consumed increments
    Bytes take();  // 32-byte digest of the records since the last take / reset; starts the next row
    int records = 0;

private:
    Sha256 h_;
};

Bytes chain_seed(const std::string &task_id);
Bytes chain_next(const Bytes &prev, const Bytes &row_digest);

}  // namespace ballrain
