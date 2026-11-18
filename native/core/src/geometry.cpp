#include "ballrain/geometry.h"

#include <algorithm>
#include <limits>

namespace ballrain {

static const double NaN = std::numeric_limits<double>::quiet_NaN();

void quat_matrix(const double q[4], double R[3][3]) {
    double x = q[0], y = q[1], z = q[2], w = q[3];
    R[0][0] = 1 - 2 * (y * y + z * z), R[0][1] = 2 * (x * y - z * w), R[0][2] = 2 * (x * z + y * w);
    R[1][0] = 2 * (x * y + z * w), R[1][1] = 1 - 2 * (x * x + z * z), R[1][2] = 2 * (y * z - x * w);
    R[2][0] = 2 * (x * z - y * w), R[2][1] = 2 * (y * z + x * w), R[2][2] = 1 - 2 * (x * x + y * y);
}

V3 ck_to_world(const float v[3], const float *m, int stride) {
    V3 out;
    for (int c = 0; c < 3; ++c)
        out[c] = double(v[0]) * m[0 * stride + c] + double(v[1]) * m[1 * stride + c] + double(v[2]) * m[2 * stride + c] +
                 double(m[3 * stride + c]);
    return out;
}

V3 body_to_world(const float v[3], const double R[3][3], const double Rc[3][3], const double shift[3],
                 const double pos[3]) {
    double w[3];
    for (int i = 0; i < 3; ++i) w[i] = double(v[0]) * Rc[i][0] + double(v[1]) * Rc[i][1] + double(v[2]) * Rc[i][2] + shift[i];
    V3 out;
    for (int i = 0; i < 3; ++i) out[i] = w[0] * R[i][0] + w[1] * R[i][1] + w[2] * R[i][2] + pos[i];
    return out;
}

// ---------------------------------------------------------------------------------------------- Grid2D

Grid2D::Grid2D(const std::vector<V3> &lo, const std::vector<V3> &hi, double cell) : cell_(cell) {
    size_t n = lo.size();
    if (!n) return;
    ox_ = lo[0][0], oz_ = lo[0][2];
    for (auto &l : lo) ox_ = std::min(ox_, l[0]), oz_ = std::min(oz_, l[2]);
    ox_ -= 1e-6, oz_ -= 1e-6;
    std::vector<int64_t> c0x(n), c0z(n), c1x(n), c1z(n);
    int64_t mx = 0, mz = 0;
    for (size_t i = 0; i < n; ++i) {
        c0x[i] = int64_t(std::floor((lo[i][0] - ox_) / cell_));
        c0z[i] = int64_t(std::floor((lo[i][2] - oz_) / cell_));
        c1x[i] = int64_t(std::floor((hi[i][0] - ox_) / cell_));
        c1z[i] = int64_t(std::floor((hi[i][2] - oz_) / cell_));
        if (i == 0 || c1x[i] > mx) mx = c1x[i];
        if (i == 0 || c1z[i] > mz) mz = c1z[i];
    }
    nx_ = mx + 1, nz_ = mz + 1;
    std::vector<int64_t> count(size_t(nx_ * nz_) + 1, 0);
    for (size_t i = 0; i < n; ++i)
        for (int64_t ix = c0x[i]; ix <= c1x[i]; ++ix)
            for (int64_t iz = c0z[i]; iz <= c1z[i]; ++iz) ++count[size_t(ix * nz_ + iz) + 1];
    start_.assign(size_t(nx_ * nz_) + 1, 0);
    for (size_t c = 1; c < start_.size(); ++c) start_[c] = start_[c - 1] + count[c];
    items_.assign(size_t(start_.back()), 0);
    std::vector<int64_t> fill(start_.begin(), start_.end() - 1);
    for (size_t i = 0; i < n; ++i)  // items in increasing index per cell (numpy's stable argsort)
        for (int64_t ix = c0x[i]; ix <= c1x[i]; ++ix)
            for (int64_t iz = c0z[i]; iz <= c1z[i]; ++iz) items_[size_t(fill[size_t(ix * nz_ + iz)]++)] = int64_t(i);
}

void Grid2D::query_box(double xlo, double zlo, double xhi, double zhi, std::vector<int64_t> &out) const {
    out.clear();
    int64_t c0x = int64_t(std::floor((xlo - ox_) / cell_)), c0z = int64_t(std::floor((zlo - oz_) / cell_));
    int64_t c1x = int64_t(std::floor((xhi - ox_) / cell_)), c1z = int64_t(std::floor((zhi - oz_) / cell_));
    c0x = std::max<int64_t>(c0x, 0), c0z = std::max<int64_t>(c0z, 0);
    c1x = std::min<int64_t>(c1x, nx_ - 1), c1z = std::min<int64_t>(c1z, nz_ - 1);
    if (c1x < c0x || c1z < c0z) return;
    for (int64_t ix = c0x; ix <= c1x; ++ix)
        for (int64_t iz = c0z; iz <= c1z; ++iz) {
            size_t c = size_t(ix * nz_ + iz);
            out.insert(out.end(), items_.begin() + start_[c], items_.begin() + start_[c + 1]);
        }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
}

int64_t Grid2D::point_cell(double x, double z) const {
    int64_t cx = int64_t(std::floor((x - ox_) / cell_)), cz = int64_t(std::floor((z - oz_) / cell_));
    if (cx < 0 || cx >= nx_ || cz < 0 || cz >= nz_) return -1;
    return cx * nz_ + cz;
}

const int64_t *Grid2D::cell_items(int64_t cell, size_t &n) const {
    n = size_t(start_[size_t(cell) + 1] - start_[size_t(cell)]);
    return items_.data() + start_[size_t(cell)];
}

// ---------------------------------------------------------------------------------------------- triangles

TriangleSet TriangleSet::from_items(const std::vector<Item> &items, double grid_cell) {
    TriangleSet t;
    for (auto &it : items)
        for (size_t f = 0; f < it.n_faces; ++f) {
            const uint32_t *F = it.faces + 3 * f;
            t.A.push_back((*it.V)[F[0]]);
            t.B.push_back((*it.V)[F[1]]);
            t.C.push_back((*it.V)[F[2]]);
            t.owner.push_back(it.owner);
        }
    t.finish(grid_cell);
    return t;
}

void TriangleSet::finish(double grid_cell) {
    size_t n = A.size();
    N.resize(n), lo.resize(n), hi.resize(n);
    for (size_t i = 0; i < n; ++i) {
        double u[3], v[3];
        for (int k = 0; k < 3; ++k) u[k] = B[i][k] - A[i][k], v[k] = C[i][k] - A[i][k];
        double c[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
        double ln = std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
        double dv = ln == 0 ? 1.0 : ln;
        for (int k = 0; k < 3; ++k) {
            N[i][k] = c[k] / dv;
            lo[i][k] = std::min(std::min(A[i][k], B[i][k]), C[i][k]);
            hi[i][k] = std::max(std::max(A[i][k], B[i][k]), C[i][k]);
        }
    }
    if (grid_cell > 0 && n) grid = Grid2D(lo, hi, grid_cell);
    else grid.reset();
}

void TriangleSet::candidates(const V3 &p, double r, std::vector<int64_t> &out) const {
    out.clear();
    if (!grid) {
        for (size_t i = 0; i < A.size(); ++i) {
            bool ok = true;
            for (int k = 0; k < 3; ++k) ok = ok && lo[i][k] <= p[k] + r && hi[i][k] >= p[k] - r;
            if (ok) out.push_back(int64_t(i));
        }
        return;
    }
    std::vector<int64_t> idx;
    grid->query_box(p[0] - r, p[2] - r, p[0] + r, p[2] + r, idx);
    for (auto i : idx)
        if (lo[size_t(i)][1] <= p[1] + r && hi[size_t(i)][1] >= p[1] - r) out.push_back(i);
}

V3 closest_point_triangle(const V3 &p, const V3 &A, const V3 &B, const V3 &C, int *region) {
    double ab[3], ac[3], ap[3], bp[3], cp[3];
    for (int k = 0; k < 3; ++k)
        ab[k] = B[k] - A[k], ac[k] = C[k] - A[k], ap[k] = p[k] - A[k], bp[k] = p[k] - B[k], cp[k] = p[k] - C[k];
    auto dot = [](const double *a, const double *b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
    double d1 = dot(ab, ap), d2 = dot(ac, ap), d3 = dot(ab, bp), d4 = dot(ac, bp), d5 = dot(ab, cp), d6 = dot(ac, cp);
    double va = d3 * d6 - d5 * d4;
    double vb = d5 * d2 - d1 * d6;
    double vc = d1 * d4 - d3 * d2;
    double den = va + vb + vc;
    if (std::abs(den) < 1e-300) den = 1e-300;
    double v = vb / den, w = vc / den;
    V3 Q;
    for (int k = 0; k < 3; ++k) Q[k] = A[k] + ab[k] * v + ac[k] * w;
    int reg = 2;
    auto sdiv = [](double a, double b) { return a / (b == 0 ? 1.0 : b); };
    if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
        double t = sdiv(d4 - d3, (d4 - d3) + (d5 - d6));
        for (int k = 0; k < 3; ++k) Q[k] = B[k] + (C[k] - B[k]) * t;
        reg = 1;
    }
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        double t = sdiv(d2, d2 - d6);
        for (int k = 0; k < 3; ++k) Q[k] = A[k] + ac[k] * t;
        reg = 1;
    }
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        double t = sdiv(d1, d1 - d3);
        for (int k = 0; k < 3; ++k) Q[k] = A[k] + ab[k] * t;
        reg = 1;
    }
    if (d6 >= 0 && d5 <= d6) Q = C, reg = 0;
    if (d3 >= 0 && d4 <= d3) Q = B, reg = 0;
    if (d1 <= 0 && d2 <= 0) Q = A, reg = 0;
    if (region) *region = reg;
    return Q;
}

static double norm3(const double *d) { return std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]); }

std::optional<Closest> TriangleSet::closest(const V3 &p, double r) const {
    std::vector<int64_t> idx;
    candidates(p, r, idx);
    if (idx.empty()) return std::nullopt;
    Closest best{};
    bool any = false;
    for (auto i : idx) {
        int reg;
        V3 q = closest_point_triangle(p, A[size_t(i)], B[size_t(i)], C[size_t(i)], &reg);
        double d[3] = {p[0] - q[0], p[1] - q[1], p[2] - q[2]};
        double dist = norm3(d);
        if (!any || dist < best.d) best = Closest{dist, q, reg, i, owner[size_t(i)]}, any = true;  // argmin: first
    }
    if (best.d > r) return std::nullopt;
    return best;
}

double ray_down_pair(const V3 &P, const V3 &A, const V3 &B, const V3 &C) {
    double x = P[0], z = P[2];
    double x0 = A[0], z0 = A[2];
    double e1x = B[0] - x0, e1z = B[2] - z0;
    double e2x = C[0] - x0, e2z = C[2] - z0;
    double det = e1x * e2z - e2x * e1z;
    bool ok = std::abs(det) > 1e-9;
    if (!ok) det = 1.0;
    double px = x - x0, pz = z - z0;
    double u = (px * e2z - e2x * pz) / det;
    double v = (e1x * pz - px * e1z) / det;
    bool hit = ok && u >= -1e-9 && v >= -1e-9 && u + v <= 1 + 1e-9;
    double y = A[1] + u * (B[1] - A[1]) + v * (C[1] - A[1]);
    hit = hit && y <= P[1] + 1e-9;
    return hit ? y : NaN;
}

double TriangleSet::ray_down(const V3 &P) const {
    double best = NaN;
    auto test = [&](size_t i) {
        double y = ray_down_pair(P, A[i], B[i], C[i]);
        if (!std::isnan(y) && (std::isnan(best) || y > best)) best = y;
    };
    if (A.empty()) return best;
    if (!grid) {
        for (size_t i = 0; i < A.size(); ++i) test(i);
        return best;
    }
    int64_t c = grid->point_cell(P[0], P[2]);
    if (c < 0) return best;
    size_t n;
    const int64_t *it = grid->cell_items(c, n);
    for (size_t k = 0; k < n; ++k) test(size_t(it[k]));
    return best;
}

double merge_ray(double y1, double y2) {
    double a = std::isnan(y1) ? -std::numeric_limits<double>::infinity() : y1;
    double b = std::isnan(y2) ? -std::numeric_limits<double>::infinity() : y2;
    return b > a ? y2 : y1;
}

// ---------------------------------------------------------------------------------------------- boundary edges

void BoundaryEdges::build(const TriangleSet &tris, const std::vector<size_t> &idx) {
    if (idx.empty()) return;
    size_t n = idx.size();
    // vertices: A of every walkable triangle, then B, then C (numpy concatenation order; only the set matters)
    std::vector<std::array<int64_t, 3>> key(3 * n);
    for (size_t j = 0; j < 3; ++j)
        for (size_t i = 0; i < n; ++i) {
            const V3 &P = j == 0 ? tris.A[idx[i]] : (j == 1 ? tris.B[idx[i]] : tris.C[idx[i]]);
            for (int k = 0; k < 3; ++k) key[j * n + i][k] = int64_t(std::nearbyint(P[k] / WELD));  // np.round: half-even
        }
    std::vector<std::array<int64_t, 3>> uniq(key);
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    auto id = [&](const std::array<int64_t, 3> &k) {
        return int64_t(std::lower_bound(uniq.begin(), uniq.end(), k) - uniq.begin());
    };
    std::vector<std::pair<int64_t, int64_t>> edges;
    edges.reserve(3 * n);
    for (size_t i = 0; i < n; ++i) {
        int64_t a = id(key[i]), b = id(key[n + i]), c = id(key[2 * n + i]);
        int64_t pr[3][2] = {{a, b}, {b, c}, {c, a}};
        for (auto &e : pr) {
            int64_t u = std::min(e[0], e[1]), v = std::max(e[0], e[1]);
            if (u != v) edges.emplace_back(u, v);
        }
    }
    std::sort(edges.begin(), edges.end());
    std::vector<V3> lo, hi;
    for (size_t i = 0; i < edges.size();) {
        size_t j = i;
        while (j < edges.size() && edges[j] == edges[i]) ++j;
        if (j - i == 1) {
            V3 s0, s1;
            for (int k = 0; k < 3; ++k)
                s0[k] = double(uniq[size_t(edges[i].first)][k]) * WELD, s1[k] = double(uniq[size_t(edges[i].second)][k]) * WELD;
            S0.push_back(s0), S1.push_back(s1);
            V3 l, h;
            for (int k = 0; k < 3; ++k) l[k] = std::min(s0[k], s1[k]), h[k] = std::max(s0[k], s1[k]);
            lo.push_back(l), hi.push_back(h);
        }
        i = j;
    }
    if (!S0.empty()) grid_ = Grid2D(lo, hi, 4.0);
}

double seg_dist(const V3 &p, const V3 &s0, const V3 &s1) {
    double d[3] = {s1[0] - s0[0], s1[1] - s0[1], s1[2] - s0[2]};
    double L2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    double pr[3] = {p[0] - s0[0], p[1] - s0[1], p[2] - s0[2]};
    double t = (pr[0] * d[0] + pr[1] * d[1] + pr[2] * d[2]) / (L2 == 0 ? 1.0 : L2);
    t = std::min(std::max(t, 0.0), 1.0);
    double e[3];
    for (int k = 0; k < 3; ++k) e[k] = p[k] - (s0[k] + d[k] * t);
    return norm3(e);
}

double BoundaryEdges::distance(const V3 &q, double r) const {
    const double inf = std::numeric_limits<double>::infinity();
    if (!grid_) return inf;
    std::vector<int64_t> idx;
    grid_->query_box(q[0] - r, q[2] - r, q[0] + r, q[2] + r, idx);
    if (idx.empty()) return inf;
    double best = inf;
    for (auto i : idx) best = std::min(best, seg_dist(q, S0[size_t(i)], S1[size_t(i)]));
    return best;
}

}  // namespace ballrain
