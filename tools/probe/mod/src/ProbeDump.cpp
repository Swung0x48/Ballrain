// Read-only probe dumps. Every raw IVP read goes through SafeRead (SEH); no IVP or physics call is made.
// CK parameter values are read with update=FALSE (no parameter-operation pull).

#include "Probe.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

#include "ivp_retail.h"

typedef unsigned char u8;

static bool SafeRead(const void *p, void *out, size_t n) {
    if (!p)
        return false;
    __try {
        memcpy(out, p, n);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

template <typename T>
static bool Rd(const void *base, ptrdiff_t off, T &out) {
    return SafeRead(static_cast<const u8 *>(base) + off, &out, sizeof(T));
}

static void Hex(FILE *f, const void *p, size_t n) {
    std::vector<u8> buf(n);
    if (!SafeRead(p, buf.data(), n)) {
        std::fputs("??", f);
        return;
    }
    static const char d[] = "0123456789abcdef";
    for (u8 b : buf) {
        std::fputc(d[b >> 4], f);
        std::fputc(d[b & 15], f);
    }
}

static const char *Nm(CKObject *o) { return o && o->GetName() ? o->GetName() : "-"; }

// IVP_Real_Object / IVP_Core offsets from the reference header (layout reported to match the 2002 DLL).
static const ptrdiff_t RO_TYPE = offsetof(IVP_Object, object_type);
static const ptrdiff_t RO_NEXT = offsetof(IVP_Object, next_in_cluster);
static const ptrdiff_t RO_PREV = offsetof(IVP_Object, prev_in_cluster);
static const ptrdiff_t RO_NAME = offsetof(IVP_Object, name);
static const ptrdiff_t RO_EXACT = offsetof(IVP_Real_Object, exact_synapses);
static const ptrdiff_t RO_INVALID = offsetof(IVP_Real_Object, invalid_synapses);
static const ptrdiff_t RO_FRIC = offsetof(IVP_Real_Object, friction_synapses);
static const ptrdiff_t RO_QCF = offsetof(IVP_Real_Object, q_core_f_object);
static const ptrdiff_t RO_SHIFT = offsetof(IVP_Real_Object, shift_core_f_object);
static const ptrdiff_t RO_MAT = offsetof(IVP_Real_Object, l_default_material);
static const ptrdiff_t RO_XRAD = offsetof(IVP_Real_Object, extra_radius);
static const ptrdiff_t RO_CORE = offsetof(IVP_Real_Object, physical_core);
static const ptrdiff_t RO_CLIENT = offsetof(IVP_Real_Object, client_data);
static const ptrdiff_t RO_FLAGS = offsetof(IVP_Real_Object, flags);

static const ptrdiff_t CO_TLPSI = offsetof(IVP_Core, time_of_last_psi);
static const ptrdiff_t CO_IDT = offsetof(IVP_Core, i_delta_time);
static const ptrdiff_t CO_RSC = offsetof(IVP_Core, rot_speed_change);
static const ptrdiff_t CO_SC = offsetof(IVP_Core, speed_change);
static const ptrdiff_t CO_RS = offsetof(IVP_Core, rot_speed);
static const ptrdiff_t CO_S = offsetof(IVP_Core, speed);
static const ptrdiff_t CO_POS = offsetof(IVP_Core, pos_world_f_core_last_psi);
static const ptrdiff_t CO_DPOS = offsetof(IVP_Core, delta_world_f_core_psis);
static const ptrdiff_t CO_QL = offsetof(IVP_Core, q_world_f_core_last_psi);
static const ptrdiff_t CO_QN = offsetof(IVP_Core, q_world_f_core_next_psi);
static const ptrdiff_t CO_M = offsetof(IVP_Core, m_world_f_core_last_psi);
static const ptrdiff_t CO_AXIS = offsetof(IVP_Core, rotation_axis_world_space);
static const ptrdiff_t CO_CURSPEED = offsetof(IVP_Core, current_speed);
static const ptrdiff_t CO_ROTI = offsetof(IVP_Core, rot_inertia);
static const ptrdiff_t CO_PSIFLAGS = sizeof(IVP_Core_Fast_Static);

static unsigned CoreFlags(const u8 *core) {
    unsigned v = 0;
    Rd(core, 0, v);
    return v;
}
static bool CoreUnmoveable(const u8 *core) { return ((CoreFlags(core) >> 2) & 3) != 0; }

// ---------------------------------------------------------------- events / physicalize

void BallrainProbe::ProbeEvent(const char *name) {
    if (m_PF)
        std::fprintf(m_PF, "EV %u %s playing=%d\n", (unsigned) m_FrameIndex, name, (int) m_Playing);
}

void BallrainProbe::OnPhysicalize(CK3dEntity *target, CKBOOL fixed, float friction, float elasticity, float mass,
                                  const char *collGroup, CKBOOL startFrozen, CKBOOL enableColl, CKBOOL calcMassCenter,
                                  float linearDamp, float rotDamp, const char *collSurface, VxVector massCenter,
                                  int convexCnt, CKMesh **convexMesh, int ballCnt, VxVector *ballCenter,
                                  float *ballRadius, int concaveCnt, CKMesh **concaveMesh) {
    if (!m_Active || !target)
        return;
    PendingPhys p;
    p.entity = target->GetID();
    if (m_PP) {
        std::fprintf(m_PP, "PHYS f=%u play=%d ent=%d %s fixed=%d friction=%.9g elasticity=%.9g mass=%.9g group=%s "
                           "frozen=%d coll=%d calcmc=%d lindamp=%.9g rotdamp=%.9g surface=%s mc=%.9g,%.9g,%.9g",
                     (unsigned) m_FrameIndex, (int) m_Playing, (int) target->GetID(), Nm(target), (int) fixed,
                     friction, elasticity, mass, collGroup ? collGroup : "-", (int) startFrozen, (int) enableColl,
                     (int) calcMassCenter, linearDamp, rotDamp, collSurface ? collSurface : "-", massCenter.x,
                     massCenter.y, massCenter.z);
        std::fprintf(m_PP, " convex=%d", convexCnt);
        for (int i = 0; i < convexCnt; ++i) {
            CKMesh *m = convexMesh ? convexMesh[i] : nullptr;
            std::fprintf(m_PP, " [%d:%s:%d:%d]", m ? (int) m->GetID() : 0, Nm(m), m ? m->GetVertexCount() : 0,
                         m ? m->GetFaceCount() : 0);
            if (m)
                p.meshes.emplace_back(m->GetID(), 0);
        }
        std::fprintf(m_PP, " balls=%d", ballCnt);
        for (int i = 0; i < ballCnt; ++i)
            std::fprintf(m_PP, " [%.9g,%.9g,%.9g r=%.9g]", ballCenter[i].x, ballCenter[i].y, ballCenter[i].z,
                         ballRadius[i]);
        std::fprintf(m_PP, " concave=%d", concaveCnt);
        for (int i = 0; i < concaveCnt; ++i) {
            CKMesh *m = concaveMesh ? concaveMesh[i] : nullptr;
            std::fprintf(m_PP, " [%d:%s:%d:%d]", m ? (int) m->GetID() : 0, Nm(m), m ? m->GetVertexCount() : 0,
                         m ? m->GetFaceCount() : 0);
            if (m)
                p.meshes.emplace_back(m->GetID(), 1);
        }
        std::fputc('\n', m_PP);
    }
    m_PhysEntities.insert(p.entity);
    if (fixed)
        m_FixedEntities.insert(p.entity);
    m_PendingPhys.push_back(std::move(p));
}

void BallrainProbe::OnUnphysicalize(CK3dEntity *target) {
    if (!m_Active || !target)
        return;
    if (m_PP)
        std::fprintf(m_PP, "UNPHYS f=%u play=%d ent=%d %s\n", (unsigned) m_FrameIndex, (int) m_Playing,
                     (int) target->GetID(), Nm(target));
}

void BallrainProbe::ExportPendingMeshes() {
    if (m_PendingPhys.empty())
        return;
    CKContext *ctx = m_BML->GetCKContext();
    for (const PendingPhys &p : m_PendingPhys) {
        auto *ent = (CK3dEntity *) ctx->GetObject(p.entity);
        if (!ent || !m_PM)
            continue;
        const VxMatrix &w = ent->GetWorldMatrix();
        for (const auto &mk : p.meshes) {
            if (!m_ExportedMeshes.insert({p.entity, mk.first}).second)
                continue;
            auto *mesh = (CKMesh *) ctx->GetObject(mk.first);
            if (!mesh)
                continue;
            const int nv = mesh->GetVertexCount(), nf = mesh->GetFaceCount();
            std::fprintf(m_PM, "MESH f=%u ent=%d %s mesh=%d %s kind=%s nv=%d nf=%d fixed=%d\nW", (unsigned) m_FrameIndex,
                         (int) p.entity, Nm(ent), (int) mk.first, Nm(mesh), mk.second ? "concave" : "convex", nv, nf,
                         (int) m_FixedEntities.count(p.entity));
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    std::fprintf(m_PM, " %.9g", w[r][c]);
            std::fputc('\n', m_PM);
            for (int i = 0; i < nv; ++i) {
                VxVector v;
                mesh->GetVertexPosition(i, &v);
                std::fprintf(m_PM, "V %.9g %.9g %.9g\n", v.x, v.y, v.z);
            }
            for (int i = 0; i < nf; ++i) {
                int a = 0, b = 0, c = 0;
                mesh->GetFaceVertexIndex(i, a, b, c);
                std::fprintf(m_PM, "T %d %d %d\n", a, b, c);
            }
        }
    }
    m_PendingPhys.clear();
    RebuildEntitySet();
}

// ---------------------------------------------------------------- IVP object discovery

void BallrainProbe::RebuildEntitySet() {
    m_EntitySet.clear();
    CKContext *ctx = m_BML->GetCKContext();
    const XObjectPointerArray &list = ctx->GetObjectListByType(CKCID_3DENTITY, TRUE);
    for (int k = 0; k < list.Size(); ++k)
        if (CKObject *o = list[k])
            m_EntitySet.insert(o);
}

void BallrainProbe::WalkCluster(u8 *anyReal) {
    m_RealOf.clear();
    for (int dir = 0; dir < 2; ++dir) {
        u8 *o = anyReal;
        for (int guard = 0; o && guard < 20000; ++guard) {
            int type = 0;
            void *client = nullptr;
            if (!Rd(o, RO_TYPE, type))
                break;
            if (type != IVP_CLUSTER && Rd(o, RO_CLIENT, client) && m_EntitySet.count(client))
                m_RealOf[client] = o;
            u8 *nx = nullptr;
            if (!Rd(o, dir == 0 ? RO_NEXT : RO_PREV, nx) || nx == anyReal)
                break;
            o = nx;
        }
    }
}

u8 *BallrainProbe::FindRealObject(CK3dEntity *ent) {
    if (!ent)
        return nullptr;
    auto it = m_RealOf.find(ent);
    if (it != m_RealOf.end()) {
        void *client = nullptr;
        if (Rd(it->second, RO_CLIENT, client) && client == ent)
            return it->second;
    }
    if (m_EntitySet.empty())
        RebuildEntitySet();
    // Scan the manager for IVP_U_Vector<IVP_Real_Object> (memsize, n_elems, elems) holding this entity.
    auto *mgr = reinterpret_cast<u8 *>(m_IpionManager);
    for (int off = 0x10; off < 0x400; off += 4) {
        unsigned short memsize = 0, n = 0;
        void **elems = nullptr;
        if (!Rd(mgr, off, memsize) || !Rd(mgr, off + 2, n) || !Rd(mgr, off + 4, elems))
            continue;
        if (n == 0 || n > memsize || n > 8000 || !elems)
            continue;
        for (int i = 0; i < n; ++i) {
            u8 *ro = nullptr;
            void *client = nullptr;
            if (!SafeRead(elems + i, &ro, 4) || !ro || !Rd(ro, RO_CLIENT, client))
                break;
            if (client == ent) {
                static bool logged = false;
                if (!logged && m_PF) {
                    std::fprintf(m_PF, "INFO real object of %s found in manager vector +0x%x (n=%d) at index %d\n",
                                 Nm(ent), off, n, i);
                    logged = true;
                }
                WalkCluster(ro);
                m_RealOf[ent] = ro;
                return ro;
            }
        }
    }
    return nullptr;
}

// ---------------------------------------------------------------- triangle decode

struct Tri {
    const u8 *ledge = nullptr;
    int triIndex = -1, edgeIndex = -1, nTri = 0;
    float v[3][3] = {};
};

static bool DecodeTri(const u8 *edge, Tri &t) {
    if (!edge)
        return false;
    const u8 *tri = reinterpret_cast<const u8 *>(reinterpret_cast<uintptr_t>(edge) & ~uintptr_t(15));
    unsigned hdr = 0;
    if (!Rd(tri, 0, hdr))
        return false;
    t.triIndex = (int) (hdr & 0xFFF);
    const u8 *first = tri - 16 * t.triIndex;
    t.ledge = first - 16;
    int pointOffset = 0;
    short nTri = 0;
    if (!Rd(t.ledge, 0, pointOffset) || !Rd(t.ledge, 0xC, nTri))
        return false;
    t.nTri = nTri;
    if (nTri <= 0 || t.triIndex >= nTri)
        return false;
    t.edgeIndex = (int) ((reinterpret_cast<uintptr_t>(edge) & 0xC) >> 2) - 1;
    const u8 *points = t.ledge + pointOffset;
    for (int k = 0; k < 3; ++k) {
        unsigned e = 0;
        if (!Rd(tri, 4 + 4 * k, e))
            return false;
        if (!SafeRead(points + 16 * (e & 0xFFFF), t.v[k], 12))
            return false;
    }
    return true;
}

static const u8 *OppositeEdge(const u8 *edge) {
    unsigned e = 0;
    if (!Rd(edge, 0, e))
        return nullptr;
    const int opp = ((int) (e << 1)) >> 17;   // signed 15 bits at 16..30
    return edge + 4 * opp;
}

static void PrintTri(FILE *f, const Tri &t) {
    std::fprintf(f, " ledge=%p tri=%d/%d e=%d", t.ledge, t.triIndex, t.nTri, t.edgeIndex);
    for (int k = 0; k < 3; ++k)
        std::fprintf(f, " %.9g %.9g %.9g", t.v[k][0], t.v[k][1], t.v[k][2]);
}

// ---------------------------------------------------------------- per-frame

void BallrainProbe::ProbeFrame() {
    if (!m_PF)
        return;
    ++m_ProbeFrames;
    CK3dEntity *ball = GetActiveBall();
    std::fprintf(m_PF, "F %u %s %d\n", (unsigned) m_FrameIndex, Nm(ball), ball ? (int) ball->GetID() : 0);
    {
        auto *mgr = reinterpret_cast<u8 *>(m_IpionManager);
        u8 *env = nullptr;
        Rd(mgr, 0xC0, env);
        double cur = -1, next = -1, last = -1;
        if (env) {
            Rd(env, 0x120, cur);
            Rd(env, 0x128, next);
            Rd(env, 0x130, last);
        }
        float c8 = 0, cc = 0, d0 = 0;
        Rd(mgr, 0xC8, c8);
        Rd(mgr, 0xCC, cc);
        Rd(mgr, 0xD0, d0);
        std::fprintf(m_PF, "T %.17g %.17g %.17g %.9g %.9g %.9g d=%.9g abs=%.9g sr=%.9g hs=%d\n", cur, next, last, c8, cc,
                     d0, m_TimeManager->GetLastDeltaTime(), m_TimeManager->GetAbsoluteTime(), m_BML->GetSRScore(),
                     m_BML->GetHSScore());
    }
    if (ball) {
        const VxMatrix &m = ball->GetWorldMatrix();
        std::fprintf(m_PF, "BW");
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 3; ++c)
                std::fprintf(m_PF, " %.9g", m[r][c]);
        std::fputc('\n', m_PF);
        ProbeIvpBall(ball);
        ProbeBodies(ball);
    }
    ProbeArrays();
    ProbeWatch();
    ProbeBehaviors();
}

void BallrainProbe::ProbeIvpBall(CK3dEntity *ball) {
    u8 *real = FindRealObject(ball);
    if (!real) {
        std::fprintf(m_PF, "K none\n");
        return;
    }
    WalkCluster(real);   // every frame: bodies come and go
    m_RealOf[ball] = real;
    u8 *core = nullptr;
    Rd(real, RO_CORE, core);
    float xr = 0;
    Rd(real, RO_XRAD, xr);
    int type = 0;
    Rd(real, RO_TYPE, type);
    unsigned flags = 0;
    Rd(real, RO_FLAGS, flags);
    std::fprintf(m_PF, "K real=%p core=%p type=%d xrad=%.9g roflags=%08x", real, core, type, xr, flags);
    if (core) {
        double t = 0, pos[3] = {}, ql[4] = {}, qn[4] = {};
        float s[3] = {}, sc[3] = {}, rs[3] = {}, rsc[3] = {}, dp[3] = {}, axis[3] = {}, cs[3] = {}, idt = 0, ri[4] = {};
        unsigned psif = 0;
        Rd(core, CO_TLPSI, t);
        SafeRead(core + CO_S, s, 12);
        SafeRead(core + CO_SC, sc, 12);
        SafeRead(core + CO_RS, rs, 12);
        SafeRead(core + CO_RSC, rsc, 12);
        SafeRead(core + CO_POS, pos, 24);
        SafeRead(core + CO_DPOS, dp, 12);
        SafeRead(core + CO_QL, ql, 32);
        SafeRead(core + CO_QN, qn, 32);
        SafeRead(core + CO_AXIS, axis, 12);
        SafeRead(core + CO_CURSPEED, cs, 12);
        SafeRead(core + CO_ROTI, ri, 16);
        Rd(core, CO_IDT, idt);
        Rd(core, CO_PSIFLAGS, psif);
        std::fprintf(m_PF, " cflags=%08x psif=%08x tpsi=%.17g idt=%.9g", CoreFlags(core), psif, t, idt);
        std::fprintf(m_PF, " s=%.9g,%.9g,%.9g sc=%.9g,%.9g,%.9g rs=%.9g,%.9g,%.9g rsc=%.9g,%.9g,%.9g", s[0], s[1],
                     s[2], sc[0], sc[1], sc[2], rs[0], rs[1], rs[2], rsc[0], rsc[1], rsc[2]);
        std::fprintf(m_PF, " pos=%.17g,%.17g,%.17g dp=%.9g,%.9g,%.9g", pos[0], pos[1], pos[2], dp[0], dp[1], dp[2]);
        std::fprintf(m_PF, " ql=%.17g,%.17g,%.17g,%.17g qn=%.17g,%.17g,%.17g,%.17g", ql[0], ql[1], ql[2], ql[3], qn[0],
                     qn[1], qn[2], qn[3]);
        std::fprintf(m_PF, " axis=%.9g,%.9g,%.9g cur=%.9g,%.9g,%.9g mass=%.9g", axis[0], axis[1], axis[2], cs[0], cs[1],
                     cs[2], ri[3]);
    }
    std::fputc('\n', m_PF);

    // Friction synapses -> contact points.
    u8 *s = nullptr;
    Rd(real, RO_FRIC, s);
    if (s && m_FrL < 0) {
        for (int L = 8; L <= 16; L += 4) {
            u8 *lo = nullptr;
            if (Rd(s, L, lo) && lo == real) {
                m_FrL = L;
                break;
            }
        }
        std::fprintf(m_PF, "INFO friction synapse l_obj offset %d\n", m_FrL);
    }
    for (int n = 0; s && m_FrL >= 0 && n < 32; ++n) {
        const int L = m_FrL, SZ = L + 12;
        short cpOff = 0, st = 0;
        u8 *edge = nullptr;
        Rd(s, L + 4, cpOff);
        Rd(s, L + 6, st);
        Rd(s, L + 8, edge);
        u8 *cp = s + cpOff;
        int self = (s == cp + 8) ? 0 : (s == cp + 8 + SZ) ? 1 : -1;
        u8 *ps = self == 0 ? cp + 8 + SZ : self == 1 ? cp + 8 : nullptr;
        u8 *po = nullptr;
        short pst = -1;
        u8 *pedge = nullptr;
        void *pclient = nullptr;
        if (ps) {
            Rd(ps, L, po);
            Rd(ps, L + 6, pst);
            Rd(ps, L + 8, pedge);
            if (po)
                Rd(po, RO_CLIENT, pclient);
        }
        CKObject *pent = m_EntitySet.count(pclient) ? (CKObject *) pclient : nullptr;
        std::fprintf(m_PF, "C %d cp=%p off=%d self=%d sst=%d pst=%d partner=%s pid=%d pobj=%p pedge=%p raw=", n, cp,
                     (int) cpOff, self, st & 0xFF, pst & 0xFF, Nm(pent), pent ? (int) pent->GetID() : 0, po, pedge);
        Hex(m_PF, cp, 0xA0);
        {
            u8 *lt = nullptr;   // tmp_contact_info (IVP_Impact_Solver_Long_Term*, begins with IVP_Contact_Situation)
            if (Rd(cp, 0x40, lt) && lt) {
                std::fputs(" lt=", m_PF);
                Hex(m_PF, lt, 0xC0);
            }
        }
        std::fputc('\n', m_PF);
        Tri t;
        if (pedge && (pst & 0xFF) <= 2 && DecodeTri(pedge, t)) {
            std::fprintf(m_PF, "CT %d", n);
            PrintTri(m_PF, t);
            std::fputc('\n', m_PF);
        }
        if (po && !m_PartnersSeen.count(po)) {
            m_PartnersSeen.insert(po);
        }
        u8 *nx = nullptr;
        if (!Rd(s, L - 8, nx) || !nx || nx == s)
            break;
        u8 *head = nullptr;
        Rd(real, RO_FRIC, head);
        if (nx == head)
            break;
        s = nx;
    }

    // Exact synapses -> mindists.
    s = nullptr;
    Rd(real, RO_EXACT, s);
    if (s && m_SynL < 0) {
        for (int L = 8; L <= 20; L += 4) {
            u8 *lo = nullptr;
            if (Rd(s, L, lo) && lo == real) {
                m_SynL = L;
                break;
            }
        }
        std::fprintf(m_PF, "INFO exact synapse l_obj offset %d\n", m_SynL);
    }
    std::set<u8 *> partners;
    for (int n = 0; s && m_SynL >= 0 && n < 64; ++n) {
        const int L = m_SynL, SZ = L + 12;
        short mdOff = 0, st = 0;
        u8 *edge = nullptr;
        Rd(s, L + 8, mdOff);
        Rd(s, L + 10, st);
        Rd(s, L + 4, edge);
        u8 *md = s + mdOff;
        // The partner synapse is the neighbour (s + SZ or s - SZ) whose l_obj is a known real object != real.
        u8 *ps = nullptr, *po = nullptr;
        void *pclient = nullptr;
        for (int sign : {1, -1}) {
            u8 *cand = s + sign * SZ, *lo = nullptr;
            void *cl = nullptr;
            short coff = 0;
            if (Rd(cand, L, lo) && lo && lo != real && Rd(lo, RO_CLIENT, cl) && m_EntitySet.count(cl) &&
                Rd(cand, L + 8, coff) && cand + coff == md) {
                ps = cand;
                po = lo;
                pclient = cl;
                break;
            }
        }
        short pst = -1;
        u8 *pedge = nullptr;
        if (ps) {
            Rd(ps, L + 10, pst);
            Rd(ps, L + 4, pedge);
        }
        CKObject *pent = (CKObject *) pclient;
        std::fprintf(m_PF, "D %d md=%p off=%d sst=%d pst=%d partner=%s pid=%d pobj=%p pedge=%p raw=", n, md, (int) mdOff,
                     st, pst, Nm(pent), pent ? (int) pent->GetID() : 0, po, pedge);
        Hex(m_PF, md, 0x90);
        std::fputc('\n', m_PF);
        Tri t;
        if (pedge && pst >= 0 && pst <= 2 && DecodeTri(pedge, t)) {
            std::fprintf(m_PF, "DT %d", n);
            PrintTri(m_PF, t);
            Tri o;
            const u8 *oe = OppositeEdge(pedge);
            if (pst == 1 && oe && DecodeTri(oe, o)) {
                std::fputs(" | opp", m_PF);
                PrintTri(m_PF, o);
            }
            std::fputc('\n', m_PF);
        }
        if (po)
            partners.insert(po);
        u8 *nx = nullptr;
        if (!Rd(s, L - 8, nx) || !nx || nx == s)
            break;
        u8 *head = nullptr;
        Rd(real, RO_EXACT, head);
        if (nx == head)
            break;
        s = nx;
    }
    // Invalid synapses (count only).
    {
        u8 *iv = nullptr;
        Rd(real, RO_INVALID, iv);
        int cnt = 0;
        for (; iv && m_SynL >= 0 && cnt < 64; ++cnt) {
            u8 *nx = nullptr;
            if (!Rd(iv, m_SynL - 8, nx) || !nx || nx == iv)
                break;
            iv = nx;
        }
        if (iv)
            std::fprintf(m_PF, "INV %d\n", cnt + 1);
    }
    // Partner object transforms: first time for static partners, every frame for movable ones.
    for (u8 *po : partners) {
        u8 *pcore = nullptr;
        Rd(po, RO_CORE, pcore);
        const bool movable = pcore && !CoreUnmoveable(pcore);
        if (!movable && m_PartnersSeen.count(po))
            continue;
        m_PartnersSeen.insert(po);
        void *cl = nullptr;
        Rd(po, RO_CLIENT, cl);
        int type = 0;
        Rd(po, RO_TYPE, type);
        float xr = 0;
        Rd(po, RO_XRAD, xr);
        double mat[16] = {};
        if (pcore)
            SafeRead(pcore + CO_M, mat, sizeof(mat));
        IVP_U_Quat *qp = nullptr;
        double q[4] = {0, 0, 0, 1};
        if (Rd(po, RO_QCF, qp) && qp)
            SafeRead(qp, q, 32);
        float shift[4] = {};
        SafeRead(po + RO_SHIFT, shift, 16);
        u8 *mat0 = nullptr;
        Rd(po, RO_MAT, mat0);
        std::fprintf(m_PF, "P obj=%p %s id=%d type=%d movable=%d xrad=%.9g core=", po, Nm((CKObject *) cl),
                     cl ? (int) ((CKObject *) cl)->GetID() : 0, type, (int) movable, xr);
        for (double d : mat)
            std::fprintf(m_PF, " %.17g", d);
        std::fprintf(m_PF, " qcf=%.17g,%.17g,%.17g,%.17g shift=%.9g,%.9g,%.9g mat=", q[0], q[1], q[2], q[3], shift[0],
                     shift[1], shift[2]);
        Hex(m_PF, mat0, 0x20);
        std::fputc('\n', m_PF);
    }
}

void BallrainProbe::ProbeBodies(CK3dEntity *ball) {
    const VxVector bp = ball->GetWorldMatrix()[3];
    int movable = 0, total = 0;
    for (const auto &kv : m_RealOf) {
        ++total;
        if (kv.first == ball)
            continue;
        u8 *core = nullptr;
        if (!Rd(kv.second, RO_CORE, core) || !core || CoreUnmoveable(core))
            continue;
        ++movable;
        auto *ent = (CK3dEntity *) kv.first;
        const VxMatrix &w = ent->GetWorldMatrix();
        const VxVector p = w[3];
        const float dx = p.x - bp.x, dy = p.y - bp.y, dz = p.z - bp.z;
        if (dx * dx + dy * dy + dz * dz > 60.0f * 60.0f)
            continue;
        double pos[3] = {}, q[4] = {};
        float s[3] = {}, rs[3] = {};
        unsigned psif = 0;
        SafeRead(core + CO_POS, pos, 24);
        SafeRead(core + CO_QL, q, 32);
        SafeRead(core + CO_S, s, 12);
        SafeRead(core + CO_RS, rs, 12);
        Rd(core, CO_PSIFLAGS, psif);
        std::fprintf(m_PF, "B %s %d psif=%08x pos=%.9g,%.9g,%.9g q=%.9g,%.9g,%.9g,%.9g v=%.9g,%.9g,%.9g w=%.9g,%.9g,%.9g W",
                     Nm(ent), (int) ent->GetID(), psif, pos[0], pos[1], pos[2], q[0], q[1], q[2], q[3], s[0], s[1], s[2],
                     rs[0], rs[1], rs[2]);
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 3; ++c)
                std::fprintf(m_PF, " %.9g", w[r][c]);
        std::fputc('\n', m_PF);
    }
    std::fprintf(m_PF, "BN total=%d movable=%d\n", total, movable);
}

// Array cell as a string, never pulling parameter operations.
static std::string Cell(CKDataArray *a, int r, int c) {
    static char buf[1 << 16];
    buf[0] = 0;
    const CK_ARRAYTYPE t = a->GetColumnType(c);
    if (t == CKARRAYTYPE_PARAMETER) {
        CKDWORD *e = a->GetElement(r, c);
        auto *p = e ? (CKParameter *) *e : nullptr;
        if (p && p->GetStringValue(nullptr, FALSE) < (int) sizeof(buf))
            p->GetStringValue(buf, FALSE);
        return buf;
    }
    if (a->GetElementStringValue(r, c, nullptr) < (int) sizeof(buf))
        a->GetElementStringValue(r, c, buf);
    return buf;
}

void BallrainProbe::ProbeArrays() {
    CKContext *ctx = m_BML->GetCKContext();
    const int n = ctx->GetObjectsCountByClassID(CKCID_DATAARRAY);
    CK_ID *ids = ctx->GetObjectsListByClassID(CKCID_DATAARRAY);
    for (int k = 0; k < n; ++k) {
        auto *a = (CKDataArray *) ctx->GetObject(ids[k]);
        if (!a || !a->GetName())
            continue;
        const std::string key = std::string(a->GetName()) + "#" + std::to_string(a->GetID());
        auto &last = m_ArrayLast[key];
        const int rows = a->GetRowCount(), cols = a->GetColumnCount();
        if ((int) last.size() != rows * cols) {
            std::fprintf(m_PF, "AR %s %d %d %d\n", a->GetName(), (int) a->GetID(), rows, cols);
            last.assign(rows * cols, std::string("\x01"));
        }
        if (rows * cols > 400)
            continue;
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                std::string v = Cell(a, r, c);
                std::string &old = last[r * cols + c];
                if (v != old) {
                    std::fprintf(m_PF, "A %s %d %d %s\n", a->GetName(), r, c, v.c_str());
                    old = std::move(v);
                }
            }
    }
}

void BallrainProbe::ProbeWatch() {
    CKContext *ctx = m_BML->GetCKContext();
    for (CK_ID id : m_Watch) {
        auto *e = (CK3dEntity *) ctx->GetObject(id);
        if (!e)
            continue;
        const VxVector p = e->GetWorldMatrix()[3];
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%d %.3f %.3f %.3f", (int) e->IsVisible(), p.x, p.y, p.z);
        std::string &old = m_WatchLast[id];
        if (old != buf) {
            std::fprintf(m_PF, "W %d %s %s\n", (int) id, Nm(e), buf);
            old = buf;
        }
    }
    for (CK_ID id : m_Cams) {
        auto *e = (CK3dEntity *) ctx->GetObject(id);
        if (!e)
            continue;
        const VxMatrix &w = e->GetWorldMatrix();
        std::fprintf(m_PF, "M %s %d", Nm(e), (int) e->IsVisible());
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 3; ++c)
                std::fprintf(m_PF, " %.9g", w[r][c]);
        std::fputc('\n', m_PF);
    }
    if (CKRenderContext *rc = ctx->GetPlayerRenderContext())
        std::fprintf(m_PF, "MA %s\n", Nm(rc->GetAttachedCamera()));
}

static CKParameter *LocalByName(CKBehavior *b, const char *name) {
    for (int i = 0; i < b->GetLocalParameterCount(); ++i) {
        CKParameter *p = b->GetLocalParameter(i);
        if (p && p->GetName() && !strcmp(p->GetName(), name))
            return p;
    }
    return nullptr;
}

void BallrainProbe::ProbeBehaviors() {
    CKContext *ctx = m_BML->GetCKContext();
    for (CK_ID id : m_ContactBBs) {
        auto *b = (CKBehavior *) ctx->GetObject(id);
        if (!b)
            continue;
        unsigned mask = 0;
        for (int i = 0; i < b->GetOutputCount() && i < 32; ++i)
            if (b->IsOutputActive(i))
                mask |= 1u << i;
        CKParameter *h = LocalByName(b, "IVP_Handle");
        void *data = nullptr;
        if (h && h->GetReadDataPtr(FALSE))
            SafeRead(h->GetReadDataPtr(FALSE), &data, 4);
        std::fprintf(m_PF, "CC %d act=%d mask=%04x data=%p raw=", (int) id, (int) b->IsActive(), mask, data);
        if (data) {
            Hex(m_PF, data, 0x40);
            for (int off = 8; off < 0x20; off += 4) {
                void *q = nullptr;
                if (Rd(data, off, q) && q) {
                    std::fprintf(m_PF, " *%x=", off);
                    Hex(m_PF, q, 0x30);
                }
            }
        }
        std::fputc('\n', m_PF);
    }
    for (CK_ID id : m_CollBBs) {
        auto *b = (CKBehavior *) ctx->GetObject(id);
        if (!b)
            continue;
        char buf[512];
        int off = std::snprintf(buf, sizeof(buf), "act=%d out=%d", (int) b->IsActive(),
                                b->GetOutputCount() > 0 ? (int) b->IsOutputActive(0) : -1);
        for (int i = 0; i < b->GetOutputParameterCount() && i < 4; ++i) {
            CKParameterOut *p = b->GetOutputParameter(i);
            if (!p)
                continue;
            const int sz = std::min(p->GetDataSize(), 12);
            float v[3] = {};
            SafeRead(p->GetReadDataPtr(FALSE), v, sz);
            if (i == 0)
                off += std::snprintf(buf + off, sizeof(buf) - off, " ent=%u", *(unsigned *) v);
            else
                off += std::snprintf(buf + off, sizeof(buf) - off, " %.9g %.9g %.9g", v[0], sz > 4 ? v[1] : 0.f,
                                     sz > 8 ? v[2] : 0.f);
        }
        std::string &old = m_CollLast[id];
        if (old != buf || (b->GetOutputCount() > 0 && b->IsOutputActive(0))) {
            std::fprintf(m_PF, "CD %d %s\n", (int) id, buf);
            old = buf;
        }
    }
}

// ---------------------------------------------------------------- static dump

static bool StartsWith(const char *s, const char *p) { return s && !strncmp(s, p, strlen(p)); }
static bool ContainsI(const char *s, const char *p) {
    if (!s)
        return false;
    std::string a(s), b(p);
    std::transform(a.begin(), a.end(), a.begin(), ::tolower);
    std::transform(b.begin(), b.end(), b.begin(), ::tolower);
    return a.find(b) != std::string::npos;
}

void BallrainProbe::DumpStatic(const char *tag) {
    ++m_StaticDumps;
    if (m_ProbeDir.empty())
        return;
    FILE *f = std::fopen((m_ProbeDir + "\\static_" + tag + ".txt").c_str(), "w");
    if (!f)
        return;
    std::setvbuf(f, nullptr, _IONBF, 0);
    RebuildEntitySet();
    CKContext *ctx = m_BML->GetCKContext();
    std::fprintf(f, "FRAME %u\n", (unsigned) m_FrameIndex);
    static char buf[1 << 16];

    CKAttributeManager *am = ctx->GetAttributeManager();
    for (int t = 0; t < am->GetAttributeCount(); ++t)
        std::fprintf(f, "ATTRTYPE %d %s\n", t, am->GetAttributeNameByType(t) ? am->GetAttributeNameByType(t) : "-");

    {
        const int n = ctx->GetObjectsCountByClassID(CKCID_GROUP);
        CK_ID *ids = ctx->GetObjectsListByClassID(CKCID_GROUP);
        const bool fillWatch = m_Watch.empty();
        for (int k = 0; k < n; ++k) {
            auto *g = (CKGroup *) ctx->GetObject(ids[k]);
            if (!g)
                continue;
            std::fprintf(f, "GROUP %d %s %d\n", (int) g->GetID(), Nm(g), g->GetObjectCount());
            const char *gn = g->GetName();
            const bool watch = StartsWith(gn, "All_P_Extra") || StartsWith(gn, "All_P_Trafo") || StartsWith(gn, "All_PC_") || StartsWith(gn, "All_PS_") || StartsWith(gn, "All_PE_") || StartsWith(gn, "LifeBalls") || StartsWith(gn, "Trafos") || StartsWith(gn, "P_Extra") || StartsWith(gn, "P_Trafo") || StartsWith(gn, "PC_") ||
                               StartsWith(gn, "PR_") || StartsWith(gn, "PS_") || StartsWith(gn, "PE_") ||
                               ContainsI(gn, "Depth");
            for (int i = 0; i < g->GetObjectCount(); ++i) {
                CKBeObject *o = g->GetObject(i);
                if (!o)
                    continue;
                std::fprintf(f, "  - %d %d %s\n", (int) o->GetID(), (int) o->GetClassID(), Nm(o));
                if (ContainsI(gn, "Depth") && CKIsChildClassOf(o, CKCID_3DENTITY)) {
                    auto *e = (CK3dEntity *) o;
                    CKMesh *mesh = e->GetCurrentMesh();
                    if (mesh && mesh->GetVertexCount() > 0) {
                        VxVector lo(1e30f, 1e30f, 1e30f), hi(-1e30f, -1e30f, -1e30f);
                        const VxMatrix &w = e->GetWorldMatrix();
                        for (int vi = 0; vi < mesh->GetVertexCount(); ++vi) {
                            VxVector v, t;
                            mesh->GetVertexPosition(vi, &v);
                            Vx3DMultiplyMatrixVector(&t, w, &v);
                            lo.x = std::min(lo.x, t.x); lo.y = std::min(lo.y, t.y); lo.z = std::min(lo.z, t.z);
                            hi.x = std::max(hi.x, t.x); hi.y = std::max(hi.y, t.y); hi.z = std::max(hi.z, t.z);
                        }
                        std::fprintf(f, "    bbox %.9g %.9g %.9g %.9g %.9g %.9g\n", lo.x, lo.y, lo.z, hi.x, hi.y, hi.z);
                    }
                }
                if (fillWatch && watch && CKIsChildClassOf(o, CKCID_3DENTITY))
                    m_Watch.push_back(o->GetID());
            }
        }
        std::sort(m_Watch.begin(), m_Watch.end());
        m_Watch.erase(std::unique(m_Watch.begin(), m_Watch.end()), m_Watch.end());
    }
    {
        const int n = ctx->GetObjectsCountByClassID(CKCID_DATAARRAY);
        CK_ID *ids = ctx->GetObjectsListByClassID(CKCID_DATAARRAY);
        for (int k = 0; k < n; ++k) {
            auto *a = (CKDataArray *) ctx->GetObject(ids[k]);
            if (!a)
                continue;
            std::fprintf(f, "ARRAY %d %s %dx%d\n", (int) a->GetID(), Nm(a), a->GetRowCount(), a->GetColumnCount());
            for (int c = 0; c < a->GetColumnCount(); ++c)
                std::fprintf(f, "  col %d %s type=%d\n", c, a->GetColumnName(c) ? a->GetColumnName(c) : "-",
                             (int) a->GetColumnType(c));
            if (a->GetRowCount() * a->GetColumnCount() > 20000)
                continue;
            for (int r = 0; r < a->GetRowCount(); ++r)
                for (int c = 0; c < a->GetColumnCount(); ++c)
                    std::fprintf(f, "  [%d,%d] %s\n", r, c, Cell(a, r, c).c_str());
        }
    }
    {
        const bool fillCams = m_Cams.empty();
        const XObjectPointerArray &list = ctx->GetObjectListByType(CKCID_3DENTITY, TRUE);
        for (int k = 0; k < list.Size(); ++k) {
            auto *e = (CK3dEntity *) list[k];
            if (!e)
                continue;
            const VxMatrix &m = e->GetWorldMatrix();
            CKMesh *mesh = e->GetCurrentMesh();
            std::fprintf(f, "ENT %d %d %s vis=%d parent=%s mesh=%s", (int) e->GetID(), (int) e->GetClassID(), Nm(e),
                         (int) e->IsVisible(), Nm(e->GetParent()), Nm(mesh));
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 3; ++c)
                    std::fprintf(f, " %.9g", m[r][c]);
            static const char *const kAttrs[] = {"Continuous Contact ID", "Coll Detection ID", "Floor", "Sound Obstacle",
                                                 "PhysicsScript"};
            for (const char *an : kAttrs) {
                const int t = am->GetAttributeTypeByName((CKSTRING) an);
                if (t < 0 || !e->HasAttribute(t))
                    continue;
                CKParameterOut *p = e->GetAttributeParameter(t);
                buf[0] = 0;
                const CKGUID g = p ? p->GetGUID() : CKGUID();
                if (p && (g == CKPGUID_INT || g == CKPGUID_FLOAT || g == CKPGUID_BOOL)) {
                    if (p->GetStringValue(nullptr, FALSE) < (int) sizeof(buf))
                        p->GetStringValue(buf, FALSE);
                    std::fprintf(f, " @%s=%s", am->GetAttributeNameByType(t) ? am->GetAttributeNameByType(t) : "?", buf);
                } else {
                    std::fprintf(f, " @%s=raw:", am->GetAttributeNameByType(t) ? am->GetAttributeNameByType(t) : "?");
                    if (p && p->GetDataSize() > 0)
                        Hex(f, p->GetReadDataPtr(FALSE), std::min(p->GetDataSize(), 32));
                }
            }
            void *real = m_RealOf.count(e) ? m_RealOf[e] : nullptr;
            if (real)
                std::fprintf(f, " ivp=%p", real);
            std::fputc('\n', f);
            if (fillCams && m_Cams.size() < 16 &&
                (CKIsChildClassOf(e, CKCID_CAMERA) || StartsWith(e->GetName(), "Cam")))
                m_Cams.push_back(e->GetID());
        }
    }
    {
        const bool fillBB = m_ContactBBs.empty() && m_CollBBs.empty();
        const int n = ctx->GetObjectsCountByClassID(CKCID_BEHAVIOR);
        CK_ID *ids = ctx->GetObjectsListByClassID(CKCID_BEHAVIOR);
        for (int k = 0; k < n; ++k) {
            auto *b = (CKBehavior *) ctx->GetObject(ids[k]);
            if (!b)
                continue;
            CKBehavior *owner = b->GetOwnerScript();
            if (b->GetType() == CKBEHAVIORTYPE_SCRIPT && !b->GetParent()) {
                std::fprintf(f, "SCRIPT %d %s owner=%s active=%d\n", (int) b->GetID(), Nm(b), Nm(b->GetOwner()),
                             (int) b->IsActive());
            }
            const char *proto = b->GetPrototypeName();
            const bool physics = ContainsI(proto, "physic") || ContainsI(proto, "contact") || ContainsI(proto, "coll") ||
                                 ContainsI(proto, "sound") || ContainsI(proto, "ray");
            const bool gameplay = owner && (StartsWith(owner->GetName(), "Gameplay_") ||
                                            StartsWith(owner->GetName(), "Ball_") ||
                                            StartsWith(owner->GetName(), "Cam"));
            if (!physics && !gameplay)
                continue;
            std::fprintf(f, "BB %d proto=%s name=%s script=%s obj=%s active=%d in=%d out=%d\n", (int) b->GetID(),
                         proto ? proto : "-", Nm(b), Nm(owner), Nm(b->GetOwner()), (int) b->IsActive(),
                         b->GetInputCount(), b->GetOutputCount());
            for (int p = 0; p < b->GetLocalParameterCount(); ++p) {
                CKParameter *param = b->GetLocalParameter(p);
                if (!param)
                    continue;
                buf[0] = 0;
                if (param->GetStringValue(nullptr, FALSE) < (int) sizeof(buf))
                    param->GetStringValue(buf, FALSE);
                std::fprintf(f, "  local %s = %s\n", Nm(param), buf);
            }
            if (physics) {
                for (int p = 0; p < b->GetInputParameterCount(); ++p) {
                    CKParameterIn *pin = b->GetInputParameter(p);
                    CKParameter *src = pin ? pin->GetRealSource() : nullptr;
                    buf[0] = 0;
                    if (src && src->GetStringValue(nullptr, FALSE) < (int) sizeof(buf))
                        src->GetStringValue(buf, FALSE);
                    std::fprintf(f, "  in %s = %s\n", Nm(pin), buf);
                }
                for (int p = 0; p < b->GetOutputParameterCount(); ++p)
                    std::fprintf(f, "  outparam %s\n", Nm(b->GetOutputParameter(p)));
                for (int p = 0; p < b->GetOutputCount(); ++p)
                    std::fprintf(f, "  output %d %s\n", p, Nm(b->GetOutput(p)));
                if (fillBB && ContainsI(proto, "contact"))
                    m_ContactBBs.push_back(b->GetID());
                else if (fillBB && ContainsI(proto, "coll") && ContainsI(proto, "physic"))
                    m_CollBBs.push_back(b->GetID());
            }
        }
    }
    {
        // IVP objects known by now.
        CK3dEntity *ball = GetActiveBall();
        if (ball)
            FindRealObject(ball);
        for (const auto &kv : m_RealOf) {
            u8 *core = nullptr;
            Rd(kv.second, RO_CORE, core);
            float xr = 0;
            Rd(kv.second, RO_XRAD, xr);
            int type = 0;
            Rd(kv.second, RO_TYPE, type);
            const char *nm = nullptr;
            Rd(kv.second, RO_NAME, nm);
            char name[64] = {};
            if (nm)
                SafeRead(nm, name, 63);
            u8 *mat = nullptr;
            Rd(kv.second, RO_MAT, mat);
            std::fprintf(f, "IVPOBJ %s ivpname=%s type=%d unmoveable=%d xrad=%.9g mat=", Nm((CKObject *) kv.first),
                         name, type, core ? (int) CoreUnmoveable(core) : -1, xr);
            Hex(f, mat, 0x20);
            std::fputc('\n', f);
        }
    }
    std::fprintf(f, "WATCH %d CAMS %d CONTACTBB %d COLLBB %d\n", (int) m_Watch.size(), (int) m_Cams.size(),
                 (int) m_ContactBBs.size(), (int) m_CollBBs.size());
    std::fclose(f);
}
