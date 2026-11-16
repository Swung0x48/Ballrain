#pragma once

// Raw memory layout of the 2002 physics_RT.dll (sha256 e72e4afc..., checked by BallrainIO::VerifyPhysicsDll).
// Every offset here is valid for that exact file only; no read happens unless the hash check passed.
// Sources: docs/state-spec.md sections 2 and 3 (verified with a read-only probe on the retail game).
// All reads go through SafeRead (SEH + memcpy): no IVP function is called and nothing is written.

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ivp2002 {

// CKIpionManager (physics manager).
constexpr ptrdiff_t MGR_MOVABLE_VEC = 0x28;   // IVP_U_Vector<IVP_Real_Object>: u16 memsize, u16 n, void **elems
constexpr ptrdiff_t MGR_ENV = 0xC0;           // IVP_Environment *
constexpr ptrdiff_t MGR_DELTA = 0xC8;         // f32 smoothed delta (ms)
constexpr ptrdiff_t MGR_PHYS_DELTA = 0xCC;    // f32 physics step (s)
constexpr ptrdiff_t MGR_TIME_FACTOR = 0xD0;   // f32 time factor

// IVP_Environment.
constexpr ptrdiff_t ENV_CURRENT_TIME = 0x120;   // f64
constexpr ptrdiff_t ENV_NEXT_PSI = 0x128;       // f64
constexpr ptrdiff_t ENV_LAST_PSI = 0x130;       // f64

// IVP_Object / IVP_Real_Object.
constexpr ptrdiff_t RO_TYPE = 0x04;     // i32 object_type (1 = cluster, 2 = polygon, 3 = ball)
constexpr ptrdiff_t RO_NEXT = 0x08;     // next_in_cluster
constexpr ptrdiff_t RO_PREV = 0x0C;     // prev_in_cluster
constexpr ptrdiff_t RO_EXACT = 0x20;    // IVP_Synapse_Real * exact_synapses
constexpr ptrdiff_t RO_FRIC = 0x28;     // IVP_Synapse_Friction * friction_synapses
constexpr ptrdiff_t RO_QCF = 0x2C;      // IVP_U_Quat * q_core_f_object (4 x f64)
constexpr ptrdiff_t RO_SHIFT = 0x30;    // IVP_U_Float_Point shift_core_f_object (4 x f32)
constexpr ptrdiff_t RO_MAT = 0x98;      // IVP_Material * l_default_material
constexpr ptrdiff_t RO_XRAD = 0xA0;     // f32 extra_radius
constexpr ptrdiff_t RO_CORE = 0xA4;     // IVP_Core * physical_core
constexpr ptrdiff_t RO_CLIENT = 0xB0;   // void * client_data (the CK3dEntity)
constexpr int TYPE_CLUSTER = 1;

// IVP_Core.
constexpr ptrdiff_t CO_FLAGS = 0x00;      // u32 bit field; bits 2-3 physical_unmoveable
constexpr ptrdiff_t CO_PSIFLAGS = 0x60;   // u32 bit field; low byte movement_state
constexpr ptrdiff_t CO_TLPSI = 0x68;      // f64 time_of_last_psi
constexpr ptrdiff_t CO_RSC = 0x74;        // f32 x3 rot_speed_change
constexpr ptrdiff_t CO_SC = 0x84;         // f32 x3 speed_change
constexpr ptrdiff_t CO_RS = 0x94;         // f32 x3 rot_speed
constexpr ptrdiff_t CO_S = 0xA4;          // f32 x3 speed
constexpr ptrdiff_t CO_POS = 0xB8;        // f64 x3 pos_world_f_core_last_psi
constexpr ptrdiff_t CO_QL = 0xE8;         // f64 x4 q_world_f_core_last_psi (x y z w)
constexpr ptrdiff_t CO_QN = 0x108;        // f64 x4 q_world_f_core_next_psi
constexpr ptrdiff_t CO_AXIS = 0x1A8;      // f32 x3 rotation_axis_world_space
constexpr ptrdiff_t CO_CURSPEED = 0x1B8;  // f32 current_speed
constexpr ptrdiff_t CO_ABSOMEGA = 0x1BC;  // f32 abs_omega

// IVP_Synapse_Friction (0x14 bytes) and its contact point.
constexpr ptrdiff_t SF_NEXT = 0x00;
constexpr ptrdiff_t SF_LOBJ = 0x08;
constexpr ptrdiff_t SF_CPOFF = 0x0C;   // i16 contact_point_offset
constexpr ptrdiff_t SF_STATUS = 0x0E;  // low byte
constexpr ptrdiff_t SF_SIZE = 0x14;
constexpr ptrdiff_t CP_SYN0 = 0x08;
constexpr ptrdiff_t CP_SYN1 = 0x1C;
constexpr ptrdiff_t CP_LT = 0x40;          // IVP_Impact_Solver_Long_Term * tmp_contact_info (transaction memory)
constexpr ptrdiff_t CP_FRICTION = 0x44;    // f32 real_friction_factor
constexpr ptrdiff_t CP_GAP = 0x58;         // f32 last_gap_len
constexpr ptrdiff_t CP_STATUS = 0x60;      // low byte cp_status
constexpr ptrdiff_t LT_NORMAL = 0x00;      // f32 x3 surf_normal
constexpr ptrdiff_t LT_POINT = 0x20;       // f64 x3 contact_point_ws

// IVP_Synapse_Real (0x1C bytes) and its mindist.
constexpr ptrdiff_t SR_NEXT = 0x08;
constexpr ptrdiff_t SR_LOBJ = 0x10;
constexpr ptrdiff_t SR_EDGE = 0x14;     // tagged pointer to the feature (edge / triangle)
constexpr ptrdiff_t SR_MDOFF = 0x18;    // i16 mindist_offset
constexpr ptrdiff_t SR_STATUS = 0x1A;   // i16 status (0 point, 1 edge, 2 triangle, 3 ball)
constexpr ptrdiff_t MD_SYN0 = 0x18;
constexpr ptrdiff_t MD_SYN1 = 0x34;
constexpr ptrdiff_t MD_SUMXRAD = 0x50;   // f32 sum_extra_radius
constexpr ptrdiff_t MD_LEN = 0x54;       // f32 len_numerator (surface gap)
constexpr ptrdiff_t MD_PLANE = 0x68;     // f32 x4 contact_plane (normal from the partner surface to the ball, hesse)

// IVP_Compact_Ledge / triangle / edge.
constexpr ptrdiff_t LEDGE_POINT_OFF = 0x00;   // i32 c_point_offset (points: 16 bytes, xyz + pad, object space)
constexpr ptrdiff_t LEDGE_NTRI = 0x0C;        // i16 n_triangles

// The physics DLL's IVP_RAND_SEED (RVA) - read only by the reload code.
constexpr ptrdiff_t RVA_RAND_SEED = 0x685B4;

}  // namespace ivp2002

// SEH-guarded copy. Kept in its own function (no C++ objects with destructors in the guarded frame).
bool SafeRead(const void *p, void *out, size_t n);

template <typename T>
inline bool Rd(const void *base, ptrdiff_t off, T &out) {
    return base && SafeRead(static_cast<const unsigned char *>(base) + off, &out, sizeof(T));
}
