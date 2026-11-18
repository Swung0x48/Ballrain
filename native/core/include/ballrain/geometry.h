// Collision geometry (a port of ballrain/geometry.py): triangle soup with an (x, z) grid, vertical rays, closest
// points, walkable boundary edges. float64 throughout, the same formulas and tolerances as the numpy code.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

namespace ballrain {

using V3 = std::array<double, 3>;
constexpr double UP_WALKABLE = 0.5;
constexpr double WELD = 1e-3;

void quat_matrix(const double q[4], double R[3][3]);
// CK row-vector transform of one vertex with a (4,4) (stride 4) or (4,3) (stride 3) float matrix
V3 ck_to_world(const float v[3], const float *m, int stride);
// moving body: world = R(q) (R(q_cfo) v + shift) + pos
V3 body_to_world(const float v[3], const double R[3][3], const double Rc[3][3], const double shift[3], const double pos[3]);

class Grid2D {
public:
    Grid2D() = default;
    Grid2D(const std::vector<V3> &lo, const std::vector<V3> &hi, double cell);
    bool empty() const { return items_.empty() && start_.size() <= 1; }
    // sorted unique items of the cells overlapping the (x, z) box
    void query_box(double xlo, double zlo, double xhi, double zhi, std::vector<int64_t> &out) const;
    int64_t point_cell(double x, double z) const;  // -1 outside
    const int64_t *cell_items(int64_t cell, size_t &n) const;

private:
    double ox_ = 0, oz_ = 0, cell_ = 4.0;
    int64_t nx_ = 1, nz_ = 1;
    std::vector<int64_t> start_{0, 0};
    std::vector<int64_t> items_;
};

struct Closest {
    double d;
    V3 q;
    int region;
    int64_t tri;
    int64_t owner;
};

class TriangleSet {
public:
    std::vector<V3> A, B, C, N, lo, hi;
    std::vector<int64_t> owner;
    std::optional<Grid2D> grid;

    struct Item {
        int64_t owner;
        const std::vector<V3> *V;
        const uint32_t *faces;
        size_t n_faces;
    };
    static TriangleSet from_items(const std::vector<Item> &items, double grid_cell);  // grid_cell <= 0: no grid
    void finish(double grid_cell);
    size_t size() const { return A.size(); }

    void candidates(const V3 &p, double r, std::vector<int64_t> &out) const;
    std::optional<Closest> closest(const V3 &p, double r) const;
    // highest hit at or below P (vertical ray down); NaN = miss
    double ray_down(const V3 &P) const;
};

// Ericson 5.1.5 with the numpy implementation's case order (later cases overwrite earlier ones)
V3 closest_point_triangle(const V3 &p, const V3 &a, const V3 &b, const V3 &c, int *region);
double ray_down_pair(const V3 &P, const V3 &a, const V3 &b, const V3 &c);  // NaN = miss
double merge_ray(double y1, double y2);  // the higher hit (NaN = miss)

class BoundaryEdges {
public:
    BoundaryEdges() = default;
    // walkable faces (|n_y| > up) whose owner passes keep(owner)
    template <typename F>
    BoundaryEdges(const TriangleSet &tris, F keep, double up = UP_WALKABLE) {
        std::vector<size_t> idx;
        for (size_t i = 0; i < tris.size(); ++i)
            if (std::abs(tris.N[i][1]) > up && keep(tris.owner[i])) idx.push_back(i);
        build(tris, idx);
    }
    double distance(const V3 &q, double r = 12.0) const;  // inf when nothing is near
    size_t size() const { return S0.size(); }
    std::vector<V3> S0, S1;

private:
    void build(const TriangleSet &tris, const std::vector<size_t> &idx);
    std::optional<Grid2D> grid_;
};

double seg_dist(const V3 &p, const V3 &s0, const V3 &s1);

}  // namespace ballrain
