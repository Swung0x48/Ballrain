// numpy .npy / .npz reading (np.save / np.savez / np.savez_compressed / task.save_npz_deterministic: zip64, stored or
// deflated). Arrays keep their raw little-endian bytes; typed accessors check the dtype.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ballrain/util.h"

namespace ballrain {

struct NpyArray {
    std::string descr;           // numpy dtype str: "<f4", "<f8", "|u1", "<U1055", or a structured-dtype repr
    std::vector<int64_t> shape;  // () for 0-d
    bool fortran = false;
    Bytes data;                  // C-order raw bytes

    int64_t count() const;
    size_t itemsize() const;     // from descr (simple dtypes only; structured: data.size() / count())
    std::vector<float> f32() const;     // "<f4"
    std::vector<double> f64() const;    // "<f4" / "<f8" converted to double
    std::vector<int64_t> ints() const;  // "|u1" "|i1" "<u2" "<i2" "<u4" "<i4" "<i8" "<u8"
    std::string unicode() const;        // "<U<n>" 0-d or 1-element: UTF-32LE -> UTF-8 (trailing NULs stripped)
};

NpyArray parse_npy(const Bytes &file);
// Every array of an npz (names without ".npy"). Error on a malformed file.
std::map<std::string, NpyArray> load_npz(const std::string &path);
std::map<std::string, NpyArray> load_npz_bytes(const Bytes &data);

}  // namespace ballrain
