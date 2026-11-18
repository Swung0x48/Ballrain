// Minimal JSON value + strict parser (manifests, reference-line meta, pool messages). No exceptions escape except
// ballrain::Error on malformed input or a type mismatch.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ballrain/util.h"

namespace ballrain {

class Json {
public:
    enum Type { Null, Bool, Number, String, Array, Object };

    Json() = default;
    static Json boolean(bool b);
    static Json number(double d);
    static Json integer(int64_t i);
    static Json string(std::string s);
    static Json array();
    static Json object();

    Type type() const { return t_; }
    bool is_null() const { return t_ == Null; }
    bool is_bool() const { return t_ == Bool; }
    bool is_number() const { return t_ == Number; }
    bool is_int() const { return t_ == Number && int_; }  // written without fraction / exponent
    bool is_string() const { return t_ == String; }
    bool is_array() const { return t_ == Array; }
    bool is_object() const { return t_ == Object; }

    bool as_bool() const;
    double as_double() const;
    int64_t as_int() const;
    const std::string &as_string() const;
    const std::vector<Json> &items() const;
    std::vector<Json> &items();
    const std::vector<std::pair<std::string, Json>> &members() const;

    // object access
    const Json *find(const std::string &key) const;
    const Json &at(const std::string &key) const;  // Error when missing
    bool has(const std::string &key) const { return find(key) != nullptr; }
    Json &set(const std::string &key, Json v);  // replaces an existing key in place, else appends
    // array access
    const Json &at(size_t i) const;
    size_t size() const;
    void push(Json v);

    // convenience with defaults
    double get_double(const std::string &key, double dflt) const;
    int64_t get_int(const std::string &key, int64_t dflt) const;
    bool get_bool(const std::string &key, bool dflt) const;
    std::string get_string(const std::string &key, const std::string &dflt) const;

    static Json parse(const std::string &text, size_t max_depth = 64);
    // Compact JSON; sort_keys orders object keys by their UTF-8 bytes (Python json.dumps(sort_keys=True) on ASCII keys).
    std::string dump(bool sort_keys = false) const;

private:
    Type t_ = Null;
    bool b_ = false;
    bool int_ = false;
    double d_ = 0.0;
    int64_t i_ = 0;
    std::string s_;
    std::vector<Json> a_;
    std::vector<std::pair<std::string, Json>> o_;
    friend class JsonParser;
};

// Python repr(float) formatting (shortest round-trip digits, Python's choice of fixed / exponent notation): the
// number formatting json.dumps uses.
std::string py_float_repr(double v);

}  // namespace ballrain
