#include "ballrain/json.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace ballrain {

Json Json::boolean(bool b) {
    Json j;
    j.t_ = Bool;
    j.b_ = b;
    return j;
}
Json Json::number(double d) {
    Json j;
    j.t_ = Number;
    j.d_ = d;
    j.i_ = 0;
    return j;
}
Json Json::integer(int64_t i) {
    Json j;
    j.t_ = Number;
    j.int_ = true;
    j.i_ = i;
    j.d_ = double(i);
    return j;
}
Json Json::string(std::string s) {
    Json j;
    j.t_ = String;
    j.s_ = std::move(s);
    return j;
}
Json Json::array() {
    Json j;
    j.t_ = Array;
    return j;
}
Json Json::object() {
    Json j;
    j.t_ = Object;
    return j;
}

bool Json::as_bool() const {
    if (t_ != Bool) throw Error("json: not a bool");
    return b_;
}
double Json::as_double() const {
    if (t_ != Number) throw Error("json: not a number");
    return d_;
}
int64_t Json::as_int() const {
    if (t_ != Number) throw Error("json: not a number");
    if (int_) return i_;
    if (std::floor(d_) != d_ || std::fabs(d_) > 9.007199254740992e15) throw Error("json: not an integer");
    return int64_t(d_);
}
const std::string &Json::as_string() const {
    if (t_ != String) throw Error("json: not a string");
    return s_;
}
const std::vector<Json> &Json::items() const {
    if (t_ != Array) throw Error("json: not an array");
    return a_;
}
std::vector<Json> &Json::items() {
    if (t_ != Array) throw Error("json: not an array");
    return a_;
}
const std::vector<std::pair<std::string, Json>> &Json::members() const {
    if (t_ != Object) throw Error("json: not an object");
    return o_;
}
const Json *Json::find(const std::string &key) const {
    if (t_ != Object) return nullptr;
    for (auto &kv : o_)
        if (kv.first == key) return &kv.second;
    return nullptr;
}
const Json &Json::at(const std::string &key) const {
    const Json *j = find(key);
    if (!j) throw Error("json: missing key " + key);
    return *j;
}
Json &Json::set(const std::string &key, Json v) {
    if (t_ == Null) t_ = Object;
    if (t_ != Object) throw Error("json: not an object");
    for (auto &kv : o_)
        if (kv.first == key) {
            kv.second = std::move(v);
            return kv.second;
        }
    o_.emplace_back(key, std::move(v));
    return o_.back().second;
}
const Json &Json::at(size_t i) const {
    if (t_ != Array || i >= a_.size()) throw Error("json: index out of range");
    return a_[i];
}
size_t Json::size() const {
    if (t_ == Array) return a_.size();
    if (t_ == Object) return o_.size();
    return 0;
}
void Json::push(Json v) {
    if (t_ == Null) t_ = Array;
    if (t_ != Array) throw Error("json: not an array");
    a_.push_back(std::move(v));
}
double Json::get_double(const std::string &key, double dflt) const {
    const Json *j = find(key);
    return (j && j->is_number()) ? j->as_double() : dflt;
}
int64_t Json::get_int(const std::string &key, int64_t dflt) const {
    const Json *j = find(key);
    return (j && j->is_number()) ? j->as_int() : dflt;
}
bool Json::get_bool(const std::string &key, bool dflt) const {
    const Json *j = find(key);
    return (j && j->is_bool()) ? j->as_bool() : dflt;
}
std::string Json::get_string(const std::string &key, const std::string &dflt) const {
    const Json *j = find(key);
    return (j && j->is_string()) ? j->as_string() : dflt;
}

// ------------------------------------------------------------------------------------------------ parser

class JsonParser {
public:
    JsonParser(const std::string &s, size_t max_depth) : s_(s), max_depth_(max_depth) {}
    Json run() {
        ws();
        Json v = value(0);
        ws();
        if (i_ != s_.size()) fail("trailing data");
        return v;
    }

private:
    [[noreturn]] void fail(const char *m) { throw Error(std::string("json: ") + m + " at " + std::to_string(i_)); }
    void ws() {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) ++i_;
    }
    bool lit(const char *w) {
        size_t n = std::strlen(w);
        if (s_.compare(i_, n, w) == 0) {
            i_ += n;
            return true;
        }
        return false;
    }
    Json value(size_t depth) {
        if (depth > max_depth_) fail("too deep");
        if (i_ >= s_.size()) fail("unexpected end");
        char c = s_[i_];
        if (c == '{') return object(depth);
        if (c == '[') return array(depth);
        if (c == '"') return Json::string(str());
        if (lit("true")) return Json::boolean(true);
        if (lit("false")) return Json::boolean(false);
        if (lit("null")) return Json();
        if (c == '-' || (c >= '0' && c <= '9')) return num();
        fail("unexpected character");
    }
    Json object(size_t depth) {
        Json o = Json::object();
        ++i_;
        ws();
        if (i_ < s_.size() && s_[i_] == '}') {
            ++i_;
            return o;
        }
        for (;;) {
            ws();
            if (i_ >= s_.size() || s_[i_] != '"') fail("expected key");
            std::string k = str();
            ws();
            if (i_ >= s_.size() || s_[i_] != ':') fail("expected ':'");
            ++i_;
            ws();
            Json v = value(depth + 1);
            for (auto &kv : o.o_)
                if (kv.first == k) fail("duplicate key");
            o.o_.emplace_back(std::move(k), std::move(v));
            ws();
            if (i_ < s_.size() && s_[i_] == ',') {
                ++i_;
                continue;
            }
            if (i_ < s_.size() && s_[i_] == '}') {
                ++i_;
                return o;
            }
            fail("expected ',' or '}'");
        }
    }
    Json array(size_t depth) {
        Json a = Json::array();
        ++i_;
        ws();
        if (i_ < s_.size() && s_[i_] == ']') {
            ++i_;
            return a;
        }
        for (;;) {
            ws();
            a.a_.push_back(value(depth + 1));
            ws();
            if (i_ < s_.size() && s_[i_] == ',') {
                ++i_;
                continue;
            }
            if (i_ < s_.size() && s_[i_] == ']') {
                ++i_;
                return a;
            }
            fail("expected ',' or ']'");
        }
    }
    static void utf8(std::string &out, uint32_t cp) {
        if (cp < 0x80) {
            out += char(cp);
        } else if (cp < 0x800) {
            out += char(0xC0 | (cp >> 6));
            out += char(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += char(0xE0 | (cp >> 12));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        } else {
            out += char(0xF0 | (cp >> 18));
            out += char(0x80 | ((cp >> 12) & 0x3F));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        }
    }
    uint32_t hex4() {
        if (i_ + 4 > s_.size()) fail("short \\u escape");
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            char c = s_[i_++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= uint32_t(c - '0');
            else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
            else fail("bad \\u escape");
        }
        return v;
    }
    std::string str() {
        ++i_;
        std::string out;
        for (;;) {
            if (i_ >= s_.size()) fail("unterminated string");
            char c = s_[i_++];
            if (c == '"') return out;
            if (static_cast<unsigned char>(c) < 0x20) fail("control character in string");
            if (c != '\\') {
                out += c;
                continue;
            }
            if (i_ >= s_.size()) fail("bad escape");
            char e = s_[i_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && i_ + 6 <= s_.size() && s_[i_] == '\\' && s_[i_ + 1] == 'u') {
                        size_t save = i_;
                        i_ += 2;
                        uint32_t lo = hex4();
                        if (lo >= 0xDC00 && lo < 0xE000) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        else i_ = save;
                    }
                    utf8(out, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
    }
    Json num() {
        size_t b = i_;
        bool is_int = true;
        if (s_[i_] == '-') ++i_;
        if (i_ >= s_.size()) fail("bad number");
        if (s_[i_] == '0') {
            ++i_;
        } else if (s_[i_] >= '1' && s_[i_] <= '9') {
            while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
        } else {
            fail("bad number");
        }
        if (i_ < s_.size() && s_[i_] == '.') {
            is_int = false;
            ++i_;
            if (i_ >= s_.size() || !(s_[i_] >= '0' && s_[i_] <= '9')) fail("bad fraction");
            while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
        }
        if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
            is_int = false;
            ++i_;
            if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-')) ++i_;
            if (i_ >= s_.size() || !(s_[i_] >= '0' && s_[i_] <= '9')) fail("bad exponent");
            while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
        }
        std::string t = s_.substr(b, i_ - b);
        if (is_int && t.size() <= 18) return Json::integer(std::strtoll(t.c_str(), nullptr, 10));
        Json j = Json::number(std::strtod(t.c_str(), nullptr));
        if (is_int) {  // large integer literal: keep it integral when exact
            j.int_ = std::fabs(j.d_) <= 9.007199254740992e15;
            j.i_ = j.int_ ? int64_t(j.d_) : 0;
        }
        return j;
    }
    const std::string &s_;
    size_t max_depth_;
    size_t i_ = 0;
};

Json Json::parse(const std::string &text, size_t max_depth) { return JsonParser(text, max_depth).run(); }

// ------------------------------------------------------------------------------------------------ dump

std::string py_float_repr(double v) {
    if (std::isnan(v) || std::isinf(v)) throw Error("json: non-finite number");
    if (v == 0.0) return std::signbit(v) ? "-0.0" : "0.0";
    char buf[64];
    auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);  // shortest round-trip
    std::string sci(buf, r.ptr);
    // sci = [-]d[.ddd]e(+|-)XX
    bool neg = sci[0] == '-';
    if (neg) sci.erase(0, 1);
    size_t e = sci.find('e');
    std::string mant = sci.substr(0, e);
    int exp10 = std::atoi(sci.c_str() + e + 1);
    std::string digits;
    for (char c : mant)
        if (c != '.') digits += c;
    std::string out;
    if (exp10 >= -4 && exp10 < 16) {
        int point = exp10 + 1;  // digits before the decimal point
        if (point <= 0) {
            out = "0." + std::string(size_t(-point), '0') + digits;
        } else if (size_t(point) >= digits.size()) {
            out = digits + std::string(size_t(point) - digits.size(), '0') + ".0";
        } else {
            out = digits.substr(0, size_t(point)) + "." + digits.substr(size_t(point));
        }
    } else {
        out = digits.substr(0, 1);
        if (digits.size() > 1) out += "." + digits.substr(1);
        char eb[16];
        std::snprintf(eb, sizeof eb, "e%c%02d", exp10 < 0 ? '-' : '+', std::abs(exp10));
        out += eb;
    }
    return neg ? "-" + out : out;
}

static void dump_str(std::string &out, const std::string &s) {
    // json.dumps(ensure_ascii=False): UTF-8 kept as is; quote, backslash and control characters escaped
    out += '"';
    for (char ch : s) {
        unsigned char c = static_cast<unsigned char>(ch);
        switch (c) {
            case 0x22: out += "\\\""; break;
            case 0x5C: out += "\\\\"; break;
            case 0x0A: out += "\\n"; break;
            case 0x0D: out += "\\r"; break;
            case 0x09: out += "\\t"; break;
            case 0x08: out += "\\b"; break;
            case 0x0C: out += "\\f"; break;
            default:
                if (c < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof b, "\\u%04x", unsigned(c));
                    out += b;
                } else {
                    out += ch;
                }
        }
    }
    out += '"';
}

static void dump_rec(std::string &out, const Json &j, bool sort_keys) {
    switch (j.type()) {
        case Json::Null: out += "null"; break;
        case Json::Bool: out += j.as_bool() ? "true" : "false"; break;
        case Json::Number:
            if (j.is_int()) out += std::to_string(j.as_int());
            else out += py_float_repr(j.as_double());
            break;
        case Json::String: dump_str(out, j.as_string()); break;
        case Json::Array: {
            out += '[';
            bool first = true;
            for (auto &x : j.items()) {
                if (!first) out += ',';
                first = false;
                dump_rec(out, x, sort_keys);
            }
            out += ']';
            break;
        }
        case Json::Object: {
            std::vector<const std::pair<std::string, Json> *> m;
            for (auto &kv : j.members()) m.push_back(&kv);
            if (sort_keys)
                std::sort(m.begin(), m.end(), [](auto *a, auto *b) { return a->first < b->first; });
            out += '{';
            bool first = true;
            for (auto *kv : m) {
                if (!first) out += ',';
                first = false;
                dump_str(out, kv->first);
                out += ':';
                dump_rec(out, kv->second, sort_keys);
            }
            out += '}';
            break;
        }
    }
}

std::string Json::dump(bool sort_keys) const {
    std::string out;
    dump_rec(out, *this, sort_keys);
    return out;
}

}  // namespace ballrain
