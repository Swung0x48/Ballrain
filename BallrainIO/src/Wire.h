#pragma once

// Little-endian section writer for the agent-mode messages (docs/agent-protocol.md section 2.1 / 3).
// A message is: header, then sections {u32 tag, u16 sec_version, u16 sec_flags, u32 payload_len, payload},
// each section starting 4-byte aligned (payload_len excludes the padding).
// Writes into a fixed buffer (no allocation) or, for the level message, into a growing std::vector.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

class Wire {
public:
    Wire(uint8_t *buf, size_t cap) : m_Buf(buf), m_Cap(cap) {}
    explicit Wire(std::vector<uint8_t> *grow) : m_Grow(grow) { m_Grow->clear(); }

    size_t Size() const { return m_Len; }
    uint8_t *Data() { return m_Grow ? m_Grow->data() : m_Buf; }
    uint32_t Sections() const { return m_Sections; }
    bool Dropped() const { return m_Dropped; }

    void Raw(const void *data, size_t n) {
        uint8_t *dst = Reserve(n);
        if (dst && n)
            std::memcpy(dst, data, n);
    }
    void Zeros(size_t n) {
        uint8_t *dst = Reserve(n);
        if (dst && n)
            std::memset(dst, 0, n);
    }
    template <typename T>
    void Put(T v) {
        Raw(&v, sizeof(T));
    }
    void U8(uint32_t v) { Put<uint8_t>((uint8_t) v); }
    void U16(uint32_t v) { Put<uint16_t>((uint16_t) v); }
    void U32(uint32_t v) { Put<uint32_t>(v); }
    void I32(int32_t v) { Put<int32_t>(v); }
    void F32(float v) { Put<float>(v); }
    void F64(double v) { Put<double>(v); }

    // Patch a u16/u32 written earlier at absolute offset `at`.
    void PatchU16(size_t at, uint16_t v) {
        if (at + 2 <= m_Len)
            std::memcpy(Data() + at, &v, 2);
    }
    void PatchU32(size_t at, uint32_t v) {
        if (at + 4 <= m_Len)
            std::memcpy(Data() + at, &v, 4);
    }

    void Begin(const char tag[4], uint16_t version) {
        m_SecStart = m_Len;
        m_Overflow = false;
        Raw(tag, 4);
        U16(version);
        U16(0);
        U32(0);
    }
    void SetSectionFlags(uint16_t flags) { PatchU16(m_SecStart + 6, flags); }
    // Ends the section: patches payload_len and pads to 4. A section that did not fit is rolled back.
    bool End() {
        if (m_Overflow) {
            m_Len = m_SecStart;
            m_Overflow = false;
            m_Dropped = true;
            return false;
        }
        const size_t payload = m_Len - (m_SecStart + 12);
        PatchU32(m_SecStart + 8, (uint32_t) payload);
        while (m_Len & 3)
            U8(0);
        if (m_Overflow) {   // padding did not fit
            m_Len = m_SecStart;
            m_Overflow = false;
            m_Dropped = true;
            return false;
        }
        ++m_Sections;
        return true;
    }

private:
    uint8_t *Reserve(size_t n) {
        if (m_Grow) {
            m_Grow->resize(m_Len + n);
            uint8_t *p = m_Grow->data() + m_Len;
            m_Len += n;
            return p;
        }
        if (m_Overflow || m_Len + n > m_Cap) {
            m_Overflow = true;
            return nullptr;
        }
        uint8_t *p = m_Buf + m_Len;
        m_Len += n;
        return p;
    }

    uint8_t *m_Buf = nullptr;
    size_t m_Cap = 0;
    std::vector<uint8_t> *m_Grow = nullptr;
    size_t m_Len = 0;
    size_t m_SecStart = 0;
    uint32_t m_Sections = 0;
    bool m_Overflow = false;
    bool m_Dropped = false;
};
