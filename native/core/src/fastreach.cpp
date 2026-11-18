// The fastreach query kernel: a port of ballrain/fastreach.py _perceive and every @_jit helper it calls
// (docs/f7-native-port.md). Bit-identical to the numba kernel: IEEE double in the same operation order, Python's
// max/min (first argument kept unless the second compares strictly greater / smaller: NaN and -0.0 behave as in
// Python), floor division / modulo for negative ints, round() half to even, and the libm-dependent tables (S1/S2,
// the probe and key directions) embedded from Python (tools/native/reach_diff.py tables). The live log / pow calls
// (brake stop, rolling displacement) are the only libm dependence (set_libm below).
#include "ballrain/fastreach.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "ballrain/json.h"
#include "ballrain/npz.h"
#include "ballrain/util.h"
#include "miniz.h"


namespace ballrain {
namespace fastreach {

namespace {

const double NaN = std::numeric_limits<double>::quiet_NaN();
const double INF = std::numeric_limits<double>::infinity();

// fastreach.py constants (literal expressions as in Python: the compiler folds them to the same doubles)
constexpr double GRAVITY = 20.0;
constexpr double PSI_HZ = 66.0;
constexpr double ROLL_INERTIA = 1.4;
constexpr double UP_FACE = 0.5;
constexpr double MOUNT_CAP = 1.95;
constexpr double MOUNT_FREE = 0.1;
constexpr double RES = 0.5;
constexpr int64_t TL = 8;
constexpr int KH = 3;
constexpr int KR = 4;
constexpr double VRES = 0.5;
constexpr int64_t BRICK = 8;
constexpr double SDF_BAND = 3.0;
constexpr double SDF_SCALE = 4096.0;
constexpr int64_t NPSI = 396;
constexpr double GPSI = GRAVITY / PSI_HZ;
constexpr double DEATH_MARGIN = 2.0;
constexpr double GROUND_TOL = 0.3;
constexpr double DROP_TOL = 1.0;
constexpr double UP_TOL = 0.15;
constexpr double PEN_TOL = 0.2;
constexpr double DS = 0.5;
constexpr double L_MAX = 20.0;
constexpr double GAP_MAX = 40.0;
constexpr double GAP_DROP = 40.0;
constexpr double CONTACT_TOL = 0.1;
constexpr double V_MAX = 60.0;
constexpr double CLIP_V = 20.0;
constexpr double FALL_TOL = 3.0;
constexpr double EXCL_R = 2.5;
constexpr double FAN_R_IN = 1.95;
constexpr double FAN_R_OUT = 3.6;
constexpr double FAN_R_NOM = 2.85;
constexpr int FAN_COLS = 18;
constexpr int N_HEAD = 19;
constexpr int NDBG = 33;
constexpr int NPDD = 13;

// DBG_NAMES indices
enum {
    DBG_GROUNDED, DBG_SUP_REST, DBG_SUP_FLAG, DBG_AIR_ANY, DBG_N_APEX, DBG_N_STOP, DBG_N_FIRST, DBG_FIRST_X,
    DBG_FIRST_H, DBG_FIRST_Z, DBG_N_NEAR, DBG_NEAR_X, DBG_NEAR_H, DBG_NEAR_Z, DBG_NEAR_EFFORT, DBG_LAND_CX,
    DBG_LAND_CY, DBG_LAND_CZ, DBG_LAND_VX, DBG_LAND_VZ, DBG_LAND_BRAKE, DBG_LAND_EDGE, DBG_CLEAR_MIN,
    DBG_DOOM_REASON, DBG_BRAKE_OK, DBG_BRAKE_TAKEOFF, DBG_JUMP_TAKEOFF, DBG_JUMP_EDGE, DBG_JUMP_V, DBG_VISITED,
    DBG_SPEED, DBG_FAN, DBG_FAN_T
};
static_assert(DBG_FAN_T + 1 == NDBG, "DBG_NAMES");

// Python builtins on two floats (numba reproduces CPython: the accumulator is replaced only on a strict compare)
inline double pmax(double a, double b) { return b > a ? b : a; }
inline double pmin(double a, double b) { return b < a ? b : a; }
inline int64_t imax(int64_t a, int64_t b) { return b > a ? b : a; }
inline int64_t imin(int64_t a, int64_t b) { return b < a ? b : a; }
inline int64_t floordiv(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}
inline int64_t pymod(int64_t a, int64_t b) {
    int64_t m = a % b;
    if (m != 0 && ((m < 0) != (b < 0))) m += b;
    return m;
}
inline int64_t ifloor(double x) { return (int64_t)std::floor(x); }  // int(math.floor(x))
inline int64_t iceil(double x) { return (int64_t)std::ceil(x); }    // int(math.ceil(x))
inline bool isnan_(double x) { return x != x; }

// The live libm calls (brake stop PSI, rolling displacement): std::log / std::pow unless set_libm installs others.
// Python and numba call the C runtime's; on Windows that is the system ucrtbase.dll, whose pow differs from the static
// CRT's (libucrt.lib) by 1 ulp on ~0.04 % of the arguments (both within 0.501 ulp). In the kernel that changes 31 of
// 55745 test states by at most 4e-16 scaled, no flag (tools/native/reach_diff.py run --crt); nc_reach_diff installs
// the DLL's to show bit-identity.
using LogFn = double (*)(double);
using PowFn = double (*)(double, double);
double crt_log(double x) { return std::log(x); }
double crt_pow(double x, double y) { return std::pow(x, y); }
LogFn g_log = crt_log;
PowFn g_pow = crt_pow;
inline double c_log(double x) { return g_log(x); }
inline double c_pow(double x, double y) { return g_pow(x, y); }

// BEGIN GENERATED TABLES (python tools/native/reach_diff.py tables; do not edit by hand)
// per ball: air lambda, air accel, ground lambda, ground accel, radius, fan dv (reach.BALLS, fan_dv)
const double BALL_K[3][6] = {{
        0x1.f904a7904a790p-1, 0x1.cf7ea712dcf7fp-3, 0x1.faca956384c3fp-1,
        0x1.4b1152c454b12p-3, 0x1.0000000000000p+1, 0x1.af286bca1af29p-5}, {
        0x1.f45d1745d1746p-1, 0x1.4cccccccccccdp-2, 0x1.f777777777777p-1,
        0x1.db6db6db6db6fp-3, 0x1.0083126e978d5p+1, 0x1.0000000000000p-1}, {
        0x1.fdac37dac37dbp-1, 0x1.78d4fdf3b645ap-4, 0x1.fe1db34f92106p-1,
        0x1.0d2a6c405d9f7p-4, 0x1.0000000000000p+1, 0x0.0p+0}};
// fastreach.air_tables(BALLS[b]) (numpy lam ** k)
const double S1T[3][397] = {{
        0x0.0p+0, 0x1.f07c1f07c1f08p-7, 0x1.ed19878bc704fp-6, 0x1.6f4f1ca48c3d0p-5,
        0x1.e66be656479b6p-5, 0x1.2df4703d8ec24p-4, 0x1.67e5db50e34f0p-4, 0x1.a10d0046dd1b5p-4,
        0x1.d96ca13e23c97p-4, 0x1.0883bb5a2f844p-3, 0x1.23f017d3e9d38p-3, 0x1.3efcb8dc45aafp-3,
        0x1.59aaeca48454ap-3, 0x1.73fbfccf4476fp-3, 0x1.8df12e806ab50p-3, 0x1.a78bc26cd2c13p-3,
        0x1.c0ccf4e9c9accp-3, 0x1.d9b5fdfc522c7p-3, 0x1.f248116833930p-3, 0x1.05422f5f6a1d2p-2,
        0x1.113608b6f0096p-2, 0x1.1d002866de062p-2, 0x1.28a12017e387ap-2, 0x1.34197f763498bp-2,
        0x1.3f69d43878ebfp-2, 0x1.4a92aa26a2b72p-2, 0x1.55948b20adaf2p-2, 0x1.606fff254677ap-2,
        0x1.6b258c585ad70p-2, 0x1.75b5b7099301bp-2, 0x1.802101bab44c9p-2, 0x1.8a67ed25ed930p-2,
        0x1.948af8440da19p-2, 0x1.9e8aa052a3f13p-2, 0x1.a86760da0bfc3p-2, 0x1.b221b3b3637b7p-2,
        0x1.bbba110e6bd2ap-2, 0x1.c530ef7756f55p-2, 0x1.ce86c3dc800d2p-2, 0x1.d7bc01941025dp-2,
        0x1.e0d11a618f26cp-2, 0x1.e9c67e7b615e8p-2, 0x1.f29c9c9031e25p-2, 0x1.fb53e1cc4a070p-2,
        0x1.01f65cef6b1a6p-1, 0x1.0633c77f8c29fp-1, 0x1.0a6264f8c40bep-1, 0x1.0e82690670b1bp-1,
        0x1.1294069f9083dp-1, 0x1.16977009380c2p-1, 0x1.1a8cd6d8ff0a6p-1, 0x1.1e746bf76510ep-1,
        0x1.224e5fa22dc5bp-1, 0x1.261ae16eb4e7ep-1, 0x1.29da204c3a324p-1, 0x1.2d8c4a86253b9p-1,
        0x1.31318dc6416dap-1, 0x1.34ca1716f22f3p-1, 0x1.385612e55f5dep-1, 0x1.3bd5ad039a300p-1,
        0x1.3f4910aaba9b2p-1, 0x1.42b0687cf558bp-1, 0x1.460bde87aaa18p-1, 0x1.495b9c456dbc0p-1,
        0x1.4c9fcaa005732p-1, 0x1.4fd891f26591dp-1, 0x1.53061a0aa1783p-1, 0x1.56288a2bd7e5ep-1,
        0x1.59400910180e8p-1, 0x1.5c4cbcea40117p-1, 0x1.5f4ecb67d4ea6p-1, 0x1.624659b2d3f39p-1,
        0x1.65338c737e0e2p-1, 0x1.681687d21c87cp-1, 0x1.6aef6f78bfd36p-1, 0x1.6dbe6694f82aep-1,
        0x1.70838fd9882d7p-1, 0x1.733f0d801191ap-1, 0x1.75f1014abbff0p-1, 0x1.78998c85d622ap-1,
        0x1.7b38d00971150p-1, 0x1.7dceec3af6224p-1, 0x1.805c010eb70b4p-1, 0x1.82e02e0978d08p-1,
        0x1.855b9241f91b2p-1, 0x1.87ce4c626e570p-1, 0x1.8a387aaa028e7p-1, 0x1.8c9a3aee491d9p-1,
        0x1.8ef3aa9caf4b4p-1, 0x1.9144e6bbe7dc7p-1, 0x1.938e0bed51b17p-1, 0x1.95cf366e59802p-1,
        0x1.98088219d6ba7p-1, 0x1.9a3a0a6963b40p-1, 0x1.9c63ea76b1154p-1, 0x1.9e863cfcd4ae2p-1,
        0x1.a0a11c5993b83p-1, 0x1.a2b4a28ea8970p-1, 0x1.a4c0e94304286p-1, 0x1.a6c609c40ab1dp-1,
        0x1.a8c41d06cc7d0p-1, 0x1.aabb3ba93a2f4p-1, 0x1.acab7df354ee3p-1, 0x1.ae94fbd85a5dcp-1,
        0x1.b077ccf7ec871p-1, 0x1.b254089f35b6fp-1, 0x1.b429c5ca08604p-1, 0x1.b5f91b23fb12fp-1,
        0x1.b7c21f0980923p-1, 0x1.b984e788fc1a5p-1, 0x1.bb418a63d1e08p-1, 0x1.bcf81d0f73dd0p-1,
        0x1.bea8b4b66ae9fp-1, 0x1.c05366395c449p-1, 0x1.c1f846300b7eap-1, 0x1.c39768ea58e9dp-1,
        0x1.c530e2713c8cap-1, 0x1.c6c4c687bda9cp-1, 0x1.c85328abe6e82p-1, 0x1.c9dc1c17b726fp-1,
        0x1.cb5fb3c20f07cp-1, 0x1.ccde025f9b3cdp-1, 0x1.ce571a63bba44p-1, 0x1.cfcb0e01673d8p-1,
        0x1.d139ef2c0d022p-1, 0x1.d2a3cf9871adep-1, 0x1.d408c0bd8a813p-1, 0x1.d568d3d555073p-1,
        0x1.d6c419ddabeadp-1, 0x1.d81aa39918e4ep-1, 0x1.d96c818fa3cd8p-1, 0x1.dab9c40f9edaap-1,
        0x1.dc027b2e7014ep-1, 0x1.dd46b6c9580e6p-1, 0x1.de86868635e2cp-1, 0x1.dfc1f9d4488b7p-1,
        0x1.e0f91feced915p-1, 0x1.e22c07d45d23ep-1, 0x1.e35ac05a63a0bp-1, 0x1.e485581b18927p-1,
        0x1.e5abdd7f9331cp-1, 0x1.e6ce5ebe9c6f8p-1, 0x1.e7ece9dd5e91ap-1, 0x1.e9078cb0126b3p-1,
        0x1.ea1e54daaa374p-1, 0x1.eb314fd17a201p-1, 0x1.ec408ad9de78dp-1, 0x1.ed4c130adfb47p-1,
        0x1.ee53f54dd41efp-1, 0x1.ef583e5eff62cp-1, 0x1.f058face2fe10p-1, 0x1.f15636ff59e48p-1,
        0x1.f24fff2b30b70p-1, 0x1.f3465f5fbda06p-1, 0x1.f4396380f4d5bp-1, 0x1.f529174948617p-1,
        0x1.f615864a390adp-1, 0x1.f6febbece5421p-1, 0x1.f7e4c372961bap-1, 0x1.f8c7a7f54a5d1p-1,
        0x1.f9a774683fa70p-1, 0x1.fa84339879bdfp-1, 0x1.fb5df02d47fc2p-1, 0x1.fc34b4a8c8f15p-1,
        0x1.fd088b686c36fp-1, 0x1.fdd97ea5727f2p-1, 0x1.fea798756be48p-1, 0x1.ff72e2cab4819p-1,
        0x1.001db3ba77aa3p+0, 0x1.00809810bfbaep+0, 0x1.00e2232dffd4ap+0, 0x1.014259c75c25ep+0,
        0x1.01a1408189d31p+0, 0x1.01fedbf108554p+0, 0x1.025b309a5a0fdp+0, 0x1.02b642f23c217p+0,
        0x1.0310175ddd71ap+0, 0x1.0368b23314fe7p+0, 0x1.03c017b8976c4p+0, 0x1.04164c262bdaep+0,
        0x1.046b53a4e0022p+0, 0x1.04bf324f3b98ap+0, 0x1.0511ec317306ap+0, 0x1.056385499968ap+0,
        0x1.05b40187d1e2fp+0, 0x1.060364ce8048fp+0, 0x1.0651b2f2791b0p+0, 0x1.069eefbb30dc2p+0,
        0x1.06eb1ee2eac36p+0, 0x1.07364416e6c93p+0, 0x1.078062f78f14ep+0, 0x1.07c97f18a4ca8p+0,
        0x1.08119c016c3dap+0, 0x1.0858bd2cd888dp+0, 0x1.089ee609b68d1p+0, 0x1.08e419fad75bep+0,
        0x1.09285c573a0bep+0, 0x1.096bb06a34fd0p+0, 0x1.09ae19739e8b5p+0, 0x1.09ef9aa7f534ep+0,
        0x1.0a30373087323p+0, 0x1.0a6ff22b9985bp+0, 0x1.0aaeceac8e81ep+0, 0x1.0aeccfbc0bc9dp+0,
        0x1.0b29f8581fcc2p+0, 0x1.0b664b7466bbbp+0, 0x1.0ba1cbfa2f078p+0, 0x1.0bdc7cc89d524p+0,
        0x1.0c1660b4cfed7p+0, 0x1.0c4f7a8a01d7fp+0, 0x1.0c87cd09ad42dp+0, 0x1.0cbf5aebad9d0p+0,
        0x1.0cf626de61293p+0, 0x1.0d2c3386ca1d6p+0, 0x1.0d618380af4f6p+0, 0x1.0d96195ebc6f4p+0,
        0x1.0dc9f7aaa1d10p+0, 0x1.0dfd20e533c79p+0, 0x1.0e2f978689921p+0, 0x1.0e615dfe1bdd4p+0,
        0x1.0e9276b2e2daap+0, 0x1.0ec2e40373ee2p+0, 0x1.0ef2a8461ef49p+0, 0x1.0f21c5c90b23ap+0,
        0x1.0f503ed253854p+0, 0x1.0f7e15a0230f7p+0, 0x1.0fab4c68d05a2p+0, 0x1.0fd7e55af8f40p+0,
        0x1.1003e29d9c58ap+0, 0x1.102f465036878p+0, 0x1.105a128ada3f0p+0, 0x1.1084495e4adbbp+0,
        0x1.10adecd415dc9p+0, 0x1.10d6feeeac0f7p+0, 0x1.10ff81a97a649p+0, 0x1.112776f9026bbp+0,
        0x1.114ee0caf27c1p+0, 0x1.1175c1063d878p+0, 0x1.119c198b329b0p+0, 0x1.11c1ec33940c8p+0,
        0x1.11e73ad2ae586p+0, 0x1.120c07356eaedp+0, 0x1.123053227932ap+0, 0x1.1254205a3eea2p+0,
        0x1.127770971363dp+0, 0x1.129a458d420f1p+0, 0x1.12bca0eb234aap+0, 0x1.12de845931296p+0,
        0x1.12fff17a1bef0p+0, 0x1.1320e9eade449p+0, 0x1.13416f42d126fp+0, 0x1.13618313bf8f7p+0,
        0x1.138126e9f9d86p+0, 0x1.13a05c4c68dcdp+0, 0x1.13bf24bca0d6bp+0, 0x1.13dd81b6f3fa8p+0,
        0x1.13fb74b284d24p+0, 0x1.1418ff215857ep+0, 0x1.1436227067d13p+0, 0x1.1452e007b26d0p+0,
        0x1.146f394a4ea2ap+0, 0x1.148b2f967b552p+0, 0x1.14a6c445b0badp+0, 0x1.14c1f8acb108ep+0,
        0x1.14dcce1b98e64p+0, 0x1.14f745ddefa37p+0, 0x1.1511613ab73b3p+0, 0x1.152b21747c1a3p+0,
        0x1.154487c964b06p+0, 0x1.155d957340cb6p+0, 0x1.15764ba798bc3p+0, 0x1.158eab97bc46ep+0,
        0x1.15a6b670d15ebp+0, 0x1.15be6d5be2af3p+0, 0x1.15d5d17dedf1ep+0, 0x1.15ece3f7f211dp+0,
        0x1.1603a5e6fd1edp+0, 0x1.161a18643a0e8p+0, 0x1.16303c84fe4eap+0, 0x1.1646135ad7278p+0,
        0x1.165b9df396ef8p+0, 0x1.1670dd596210bp+0, 0x1.1685d292bbe12p+0, 0x1.169a7ea2934e2p+0,
        0x1.16aee2884f5b6p+0, 0x1.16c2ff3fdb768p+0, 0x1.16d6d5c1b39f2p+0, 0x1.16ea6702f0658p+0,
        0x1.16fdb3f552be1p+0, 0x1.1710bd874faccp+0, 0x1.172384a41bc6bp+0, 0x1.17360a33b68c8p+0,
        0x1.17484f1af59c7p+0, 0x1.175a543b8fbd9p+0, 0x1.176c1a7427c4bp+0, 0x1.177da2a05752bp+0,
        0x1.178eed98b96dep+0, 0x1.179ffc32f4f61p+0, 0x1.17b0cf41c6f44p+0, 0x1.17c167950cc63p+0,
        0x1.17d1c5f9ce26dp+0, 0x1.17e1eb3a4712ep+0, 0x1.17f1d81df18c0p+0, 0x1.18018d698f390p+0,
        0x1.18110bdf32e4ep+0, 0x1.1820543e49dc5p+0, 0x1.182f6743a52b1p+0, 0x1.183e45a982b8ep+0,
        0x1.184cf0279645ep+0, 0x1.185b67731248cp+0, 0x1.1869ac3eb0ad6p+0, 0x1.1877bf3abb74cp+0,
        0x1.1885a11515370p+0, 0x1.1893527941881p+0, 0x1.18a0d4106d3e7p+0, 0x1.18ae2681769d6p+0,
        0x1.18bb4a70f562ap+0, 0x1.18c8408142b7cp+0, 0x1.18d5095281084p+0, 0x1.18e1a582a3bbbp+0,
        0x1.18ee15ad76d57p+0, 0x1.18fa5a6ca6795p+0, 0x1.19067457c6565p+0, 0x1.1912640458f73p+0,
        0x1.191e2a05d6fa1p+0, 0x1.1929c6edb62dbp+0, 0x1.19353b4b70975p+0, 0x1.194087ac8b5f0p+0,
        0x1.194bac9c9da3ep+0, 0x1.1956aaa557395p+0, 0x1.1961824e874b8p+0, 0x1.196c341e22edcp+0,
        0x1.1976c0984b915p+0, 0x1.1981283f5565dp+0, 0x1.198b6b93cda3ap+0, 0x1.19958b1480c01p+0,
        0x1.199f873e808bcp+0, 0x1.19a9608d2a3c4p+0, 0x1.19b3177a2c5fep+0, 0x1.19bcac7d8cbd3p+0,
        0x1.19c6200dae1dfp+0, 0x1.19cf729f56057p+0, 0x1.19d8a4a5b253cp+0, 0x1.19e1b6925ed42p+0,
        0x1.19eaa8d56ab93p+0, 0x1.19f37bdd5e050p+0, 0x1.19fc30173edefp+0, 0x1.1a04c5ee96d67p+0,
        0x1.1a0d3dcd7813ep+0, 0x1.1a15981c8276fp+0, 0x1.1a1dd542e8a30p+0, 0x1.1a25f5a674fadp+0,
        0x1.1a2df9ab8e893p+0, 0x1.1a35e1b53dda6p+0, 0x1.1a3dae2531c2dp+0, 0x1.1a455f5bc416ap+0,
        0x1.1a4cf5b7fe4fbp+0, 0x1.1a5471979e23fp+0, 0x1.1a5bd3571a0b6p+0, 0x1.1a631b51a5b6bp+0,
        0x1.1a6a49e13675cp+0, 0x1.1a715f5e878f4p+0, 0x1.1a785c211e890p+0, 0x1.1a7f407f4f612p+0,
        0x1.1a860cce40b8ap+0, 0x1.1a8cc161efef8p+0, 0x1.1a935e8d35327p+0, 0x1.1a99e4a1c77a4p+0,
        0x1.1aa053f0407dap+0, 0x1.1aa6acc82094cp+0, 0x1.1aacef77d28f9p+0, 0x1.1ab31c4caf7efp+0,
        0x1.1ab93393026f9p+0, 0x1.1abf35960c197p+0, 0x1.1ac522a006811p+0, 0x1.1acafafa288c8p+0,
        0x1.1ad0beeca98c2p+0, 0x1.1ad66ebec4b6ep+0, 0x1.1adc0ab6bc99fp+0, 0x1.1ae19319de7d7p+0,
        0x1.1ae7082c85bc2p+0, 0x1.1aec6a321f101p+0, 0x1.1af1b96d2bd3bp+0, 0x1.1af6f61f45379p+0,
        0x1.1afc20891f6ccp+0, 0x1.1b0138ea8cc41p+0, 0x1.1b063f8280c2fp+0, 0x1.1b0b348f132cep+0,
        0x1.1b10184d8302cp+0, 0x1.1b14eafa39777p+0, 0x1.1b19acd0ccda4p+0, 0x1.1b1e5e0c03771p+0,
        0x1.1b22fee5d66c9p+0, 0x1.1b278f9774784p+0, 0x1.1b2c105944b94p+0, 0x1.1b308162e9692p+0,
        0x1.1b34e2eb428b0p+0}, {
        0x0.0p+0, 0x1.f07c1f07c1f08p-7, 0x1.ead7cd391fbc8p-6, 0x1.6bf7063808b86p-5,
        0x1.dfd070be84a48p-5, 0x1.2883e983e1c8ap-4, 0x1.5fd63f5c09420p-4, 0x1.95e6b62975a76p-4,
        0x1.cabc9ea0c27edp-4, 0x1.fe5f1ee6cd972p-4, 0x1.186a99c42c377p-3, 0x1.3112d836046c8p-3,
        0x1.492ba0df74ceep-3, 0x1.60b8366dc58c0p-3, 0x1.77bbc895f18aep-3, 0x1.8e3974830543ep-3,
        0x1.a4344541fb72cp-3, 0x1.b9af342b2637fp-3, 0x1.cead294932ed0p-3, 0x1.e330fbbdd69e0p-3,
        0x1.f73d722430c50p-3, 0x1.056aa17878ccep-2, 0x1.0efd8a68287aap-2, 0x1.1858bf80f1418p-2,
        0x1.217d84d929d56p-2, 0x1.2a6d17298fabbp-2, 0x1.3328abf821bd2p-2, 0x1.3bb171c201f10p-2,
        0x1.4408902462de1p-2, 0x1.4c2f280487688p-2, 0x1.542653b6d9aa0p-2, 0x1.5bef27251e6a0p-2,
        0x1.638aaff3ca544p-2, 0x1.6af9f5a67df93p-2, 0x1.723df9c3ad88ep-2, 0x1.7957b7f7791b0p-2,
        0x1.80482635ba3b6p-2, 0x1.871034db4b553p-2, 0x1.8db0cece8d7fep-2, 0x1.942ad99f31127p-2,
        0x1.9a7f35a5454a4p-2, 0x1.a0aebe1f933afp-2, 0x1.a6ba495148207p-2, 0x1.aca2a89ef317fp-2,
        0x1.b268a8aada2ccp-2, 0x1.b80d1170aa8cbp-2, 0x1.bd90a66087a4ap-2, 0x1.c2f426797ccd7p-2,
        0x1.c8384c63551e2p-2, 0x1.cd5dce87dce16p-2, 0x1.d2655f2b901a5p-2, 0x1.d74fac85b96f2p-2,
        0x1.dc1d60d804bc2p-2, 0x1.e0cf228588817p-2, 0x1.e565942949485p-2, 0x1.e9e154ac3a0a8p-2,
        0x1.ee42ff5abc99dp-2, 0x1.f28b2bf9a4ebap-2, 0x1.f6ba6edac2246p-2, 0x1.fad158f0f0276p-2,
        0x1.fed077e3b464ap-2, 0x1.015c2b113444ap-1, 0x1.0344bd7b7ad67p-1, 0x1.0522354c11420p-1,
        0x1.06f4d31dbb884p-1, 0x1.08bcd6135c1e7p-1, 0x1.0a7a7be07edfbp-1, 0x1.0c2e00d1b24b0p-1,
        0x1.0dd79fd4b02e4p-1, 0x1.0f77928056d7dp-1, 0x1.110e111c73e66p-1, 0x1.129b52a961c02p-1,
        0x1.141f8ce778bdbp-1, 0x1.159af45e55070p-1, 0x1.170dbc63f2200p-1, 0x1.187817239d1b7p-1,
        0x1.19da35a4be629p-1, 0x1.1b3447d17bff5p-1, 0x1.1c867c7d354d0p-1, 0x1.1dd1016ad8f01p-1,
        0x1.1f14035315f26p-1, 0x1.204fadea68d78p-1, 0x1.21842be7057d5p-1, 0x1.22b1a7069e93bp-1,
        0x1.23d848140b753p-1, 0x1.24f836eccd230p-1, 0x1.26119a8673243p-1, 0x1.272498f3e1026p-1,
        0x1.2831576a751b8p-1, 0x1.2937fa471179fp-1, 0x1.2a38a5130761ep-1, 0x1.2b337a88e63ebp-1,
        0x1.2c289c992e968p-1, 0x1.2d182c6ee9a68p-1, 0x1.2e024a7426478p-1, 0x1.2ee716565bb66p-1,
        0x1.2fc6af0ab2d71p-1, 0x1.30a132d236888p-1, 0x1.3176bf3deb98dp-1, 0x1.32477132d0e8cp-1,
        0x1.331364edc8485p-1, 0x1.33dab60768944p-1, 0x1.349d7f77b998ap-1, 0x1.355bdb99da3a1p-1,
        0x1.3615e42f91636p-1, 0x1.36cbb264ca348p-1, 0x1.377d5ed2fbe9ap-1, 0x1.382b01847df1fp-1,
        0x1.38d4b1f7c8a89p-1, 0x1.397a8722a320ep-1, 0x1.3a1c97753e738p-1, 0x1.3abaf8dd3ef8bp-1,
        0x1.3b55c0c8b3d7ep-1, 0x1.3bed0428fd54fp-1, 0x1.3c80d775a23dap-1, 0x1.3d114eaf14db4p-1,
        0x1.3d9e7d6167c6dp-1, 0x1.3e2876a6f2fe7p-1, 0x1.3eaf4d2ae997ap-1, 0x1.3f33132be0679p-1,
        0x1.3fb3da7e45f8bp-1, 0x1.4031b48ecc237p-1, 0x1.40acb264c39edp-1, 0x1.4124e4a469d75p-1,
        0x1.419a5b9129600p-1, 0x1.420d270fcd489p-1, 0x1.427d56a8a7a5ep-1, 0x1.42eaf989ab987p-1,
        0x1.43561e887b16fp-1, 0x1.43bed42468c6fp-1, 0x1.442528886e273p-1, 0x1.4489298d16510p-1,
        0x1.44eae4ba5d911p-1, 0x1.454a6749861b6p-1, 0x1.45a7be26e216fp-1, 0x1.4602f5f393411p-1,
        0x1.465c1b0740647p-1, 0x1.46b33971c0de5p-1, 0x1.47085cfcbe6cap-1, 0x1.475b912d4d7d9p-1,
        0x1.47ace1457c36ep-1, 0x1.47fc5845d86bep-1, 0x1.484a00eeecb70p-1, 0x1.4895e5c2b4e93p-1,
        0x1.48e011060a02fp-1, 0x1.49288cc205e7bp-1, 0x1.496f62c55efbap-1, 0x1.49b49ca5bbdadp-1,
        0x1.49f843c0ff57cp-1, 0x1.4a3a613e8cef0p-1, 0x1.4a7afe1085da1p-1, 0x1.4aba22f4feee5p-1,
        0x1.4af7d8772f70bp-1, 0x1.4b3426f099075p-1, 0x1.4b6f168a28f1dp-1, 0x1.4ba8af3d52ae5p-1,
        0x1.4be0f8d524321p-1, 0x1.4c17faef53dbap-1, 0x1.4c4dbcfd48415p-1, 0x1.4c8246451a01cp-1,
        0x1.4cb59de28fb81p-1, 0x1.4ce7cac81435dp-1, 0x1.4d18d3bfa7251p-1, 0x1.4d48bf6bc831dp-1,
        0x1.4d7794485cdb6p-1, 0x1.4da558ab910ccp-1, 0x1.4dd212c6b29a0p-1, 0x1.4dfdc8a707c12p-1,
        0x1.4e288036a0ca5p-1, 0x1.4e523f3d24e4dp-1, 0x1.4e7b0b609a5bbp-1, 0x1.4ea2ea262a3e8p-1,
        0x1.4ec9e0f2df965p-1, 0x1.4eeff50c62436p-1, 0x1.4f152b99ad9b0p-1, 0x1.4f3989a3c2df4p-1,
        0x1.4f5d141657a7cp-1, 0x1.4f7fcfc080548p-1, 0x1.4fa1c15556a03p-1, 0x1.4fc2ed6c9c6a1p-1,
        0x1.4fe358835acc9p-1, 0x1.500306fc7d957p-1, 0x1.5021fd216b36fp-1, 0x1.50403f2299434p-1,
        0x1.505dd1181d898p-1, 0x1.507ab7023be56p-1, 0x1.5096f4c9f0d68p-1, 0x1.50b28e4178f0bp-1,
        0x1.50cd8724d538dp-1, 0x1.50e7e31a4c7f4p-1, 0x1.5101a5b2e9c9fp-1, 0x1.511ad26af7deap-1,
        0x1.51336caa79fe6p-1, 0x1.514b77c5a1e34p-1, 0x1.5162f6fd430eap-1, 0x1.5179ed7f43790p-1,
        0x1.51905e6709b25p-1, 0x1.51a64cbde881ap-1, 0x1.51bbbb7b88122p-1, 0x1.51d0ad864cbc6p-1,
        0x1.51e525b3bb7a1p-1, 0x1.51f926c8dc108p-1, 0x1.520cb37a9900ap-1, 0x1.521fce6e1d486p-1,
        0x1.52327a393002ep-1, 0x1.5244b9628df34p-1, 0x1.52568e6241070p-1, 0x1.5267fba1f5da3p-1,
        0x1.5279037d4f4b8p-1, 0x1.5289a8423828ep-1, 0x1.5299ec3133012p-1, 0x1.52a9d17da8260p-1,
        0x1.52b95a4e31e71p-1, 0x1.52c888bce7125p-1, 0x1.52d75ed7a3c27p-1, 0x1.52e5dea05085cp-1,
        0x1.52f40a0d27e79p-1, 0x1.5301e308fa644p-1, 0x1.530f6b7370d28p-1, 0x1.531ca5214d49dp-1,
        0x1.532991dcaa8fep-1, 0x1.533633653a146p-1, 0x1.53428b7080847p-1, 0x1.534e9baa10fd8p-1,
        0x1.535a65b3c6e84p-1, 0x1.5365eb25fe821p-1, 0x1.53712d8fcc1e2p-1, 0x1.537c2e7732253p-1,
        0x1.5386ef5955dabp-1, 0x1.539171aab2eefp-1, 0x1.539bb6d74de5ap-1, 0x1.53a5c042e556cp-1,
        0x1.53af8f492210ep-1, 0x1.53b9253dc623ep-1, 0x1.53c2836cdad96p-1, 0x1.53cbab1adda20p-1,
        0x1.53d49d84ebfd6p-1, 0x1.53dd5be0ee625p-1, 0x1.53e5e75dc22dap-1, 0x1.53ee4123629d1p-1,
        0x1.53f66a5310db7p-1, 0x1.53fe64077b241p-1, 0x1.54062f54e3025p-1, 0x1.540dcd4942b2cp-1,
        0x1.54153eec71aadp-1, 0x1.541c8540484bcp-1, 0x1.5423a140c2c63p-1, 0x1.542a93e423323p-1,
        0x1.54315e1b12e18p-1, 0x1.543800d0c2efdp-1, 0x1.543e7ceb0c150p-1, 0x1.5444d34a8dbf3p-1,
        0x1.544b04cacc770p-1, 0x1.545112424f935p-1, 0x1.5456fc82be408p-1, 0x1.545cc458fbde1p-1,
        0x1.54626a8d43b81p-1, 0x1.5467efe3441e9p-1, 0x1.546d551a38dfdp-1, 0x1.54729aed05285p-1,
        0x1.5477c2124ccc3p-1, 0x1.547ccb3c8cfdep-1, 0x1.5481b71a34744p-1, 0x1.54868655bb050p-1,
        0x1.548b3995b8b56p-1, 0x1.548fd17cfc44bp-1, 0x1.54944eaaa1339p-1, 0x1.5498b1ba254bap-1,
        0x1.549cfb437da90p-1, 0x1.54a12bdb2b49fp-1, 0x1.54a544124f275p-1, 0x1.54a94476bdd6fp-1,
        0x1.54ad2d9312bccp-1, 0x1.54b0ffeec2cbdp-1, 0x1.54b4bc0e2edaap-1, 0x1.54b86272b58bfp-1,
        0x1.54bbf39ac4d09p-1, 0x1.54bf7001eb021p-1, 0x1.54c2d820e79b4p-1, 0x1.54c62c6dbb8e1p-1,
        0x1.54c96d5bb93b5p-1, 0x1.54cc9b5b940d6p-1, 0x1.54cfb6db6fb7cp-1, 0x1.54d2c046ef1e4p-1,
        0x1.54d5b80742e55p-1, 0x1.54d89e8337adbp-1, 0x1.54db741f43fd2p-1, 0x1.54de393d95d67p-1,
        0x1.54e0ee3e20021p-1, 0x1.54e3937ea709cp-1, 0x1.54e6295acde8ap-1, 0x1.54e8b02c2270ep-1,
        0x1.54eb284a296a8p-1, 0x1.54ed920a6a69fp-1, 0x1.54efedc07b63bp-1, 0x1.54f23bbe0bfaap-1,
        0x1.54f47c52f08e3p-1, 0x1.54f6afcd2d071p-1, 0x1.54f8d678ff65fp-1, 0x1.54faf0a0ea142p-1,
        0x1.54fcfe8dbdf88p-1, 0x1.54ff0086a451fp-1, 0x1.5500f6d12857dp-1, 0x1.5502e1b140a36p-1,
        0x1.5504c1695861ap-1, 0x1.5506963a584ffp-1, 0x1.55086063af847p-1, 0x1.550a20235c035p-1,
        0x1.550bd5b5f3225p-1, 0x1.550d8156a9bacp-1, 0x1.550f233f5c2cdp-1, 0x1.5510bba896338p-1,
        0x1.55124ac99a8b9p-1, 0x1.5513d0d86a6d4p-1, 0x1.55154e09ccdb4p-1, 0x1.5516c29155c6dp-1,
        0x1.55182ea16d0a1p-1, 0x1.5519926b553a5p-1, 0x1.551aee1f32523p-1, 0x1.551c41ec10353p-1,
        0x1.551d8dffe90d3p-1, 0x1.551ed287ab833p-1, 0x1.55200faf40d3ap-1, 0x1.552145a192bf5p-1,
        0x1.55227488915a0p-1, 0x1.55239c8d38b70p-1, 0x1.5524bdd79674dp-1, 0x1.5525d88ecf288p-1,
        0x1.5526ecd923a97p-1, 0x1.5527fadbf63eep-1, 0x1.552902bbcfadfp-1, 0x1.552a049c642bfp-1,
        0x1.552b00a09832bp-1, 0x1.552bf6ea85394p-1, 0x1.552ce79b7e513p-1, 0x1.552dd2d414a8ap-1,
        0x1.552eb8b41bf25p-1, 0x1.552f995aaeb4ap-1, 0x1.553074e6327e1p-1, 0x1.55314b745c030p-1,
        0x1.55321d223321fp-1, 0x1.5532ea0c16d1ap-1, 0x1.5533b24dc0f78p-1, 0x1.553476024a284p-1,
        0x1.553535442d520p-1, 0x1.5535f02d4b521p-1, 0x1.5536a6d6ee751p-1, 0x1.55375959cde2fp-1,
        0x1.553807ce10f70p-1, 0x1.5538b24b5284cp-1, 0x1.553958e8a4098p-1, 0x1.5539fbbc90cb4p-1,
        0x1.553a9add20e5ap-1, 0x1.553b365fdc454p-1, 0x1.553bce59cd914p-1, 0x1.553c62df85045p-1,
        0x1.553cf4051b34cp-1, 0x1.553d81de33cccp-1, 0x1.553e0c7e0032cp-1, 0x1.553e93f742221p-1,
        0x1.553f185c4e34cp-1, 0x1.553f99bf0e5e3p-1, 0x1.5540183104583p-1, 0x1.554093c34c00ep-1,
        0x1.55410c869dab8p-1, 0x1.5541828b5063cp-1, 0x1.5541f5e15c236p-1, 0x1.554266985bfbdp-1,
        0x1.5542d4bf90324p-1, 0x1.55434065e0502p-1, 0x1.5543a999dd278p-1, 0x1.55441069c2cb8p-1,
        0x1.554474e37a7d2p-1, 0x1.5544d7149c8dcp-1, 0x1.5545370a72352p-1, 0x1.554594d1f75ebp-1,
        0x1.5545f077dc6a1p-1, 0x1.55464a0887e3dp-1, 0x1.5546a19018321p-1, 0x1.5546f71a6538bp-1,
        0x1.55474ab301f38p-1, 0x1.55479c653e072p-1, 0x1.5547ec3c27490p-1, 0x1.55483a428b3dap-1,
        0x1.55488682f88fap-1, 0x1.5548d107c07cep-1, 0x1.554919daf83bfp-1, 0x1.554961067a59ap-1,
        0x1.5549a693e80dfp-1, 0x1.5549ea8caa89bp-1, 0x1.554a2cf9f43cfp-1, 0x1.554a6de4c214cp-1,
        0x1.554aad55dcb34p-1, 0x1.554aeb55d99fbp-1, 0x1.554b27ed1c6f8p-1, 0x1.554b6323d7e91p-1,
        0x1.554b9d020f1ffp-1}, {
        0x0.0p+0, 0x1.f07c1f07c1f08p-7, 0x1.ef5b41de6e4b2p-6, 0x1.72ac7398cb750p-5,
        0x1.ed1c270c13752p-5, 0x1.337eb0e9e3ab4p-4, 0x1.702864dc6a01bp-4, 0x1.ac8b81e19e466p-4,
        0x1.e8a85a1d7d5d1p-4, 0x1.123f9faa379bcp-3, 0x1.30084175db0aap-3, 0x1.4dae3af4ee9a5p-3,
        0x1.6b31b47b93875p-3, 0x1.8892d62efd86ep-3, 0x1.a5d1c805a95dbp-3, 0x1.c2eeb1c79343bp-3,
        0x1.dfe9bb0e6cf8cp-3, 0x1.fcc30b45d3a6dp-3, 0x1.0cbd64d5c2c05p-2, 0x1.1b088ea7cb8fdp-2,
        0x1.2943168a54559p-2, 0x1.376d0fd80d5b5p-2, 0x1.45868dd521776p-2, 0x1.538fa3af503f6p-2,
        0x1.6188647e08203p-2, 0x1.6f70e34280547p-2, 0x1.7d4932e7d2c0ep-2, 0x1.8b11664315acap-2,
        0x1.98c990137561cp-2, 0x1.a671c3024dac0p-2, 0x1.b40a11a3433c3p-2, 0x1.c1928e745cea9p-2,
        0x1.cf0b4bde1ce20p-2, 0x1.dc745c3399a5ep-2, 0x1.e9cdd1b29702dp-2, 0x1.f717be839edb6p-2,
        0x1.02291a5d0cef2p-1, 0x1.08bea32a340d2p-1, 0x1.0f4c829dfcbc7p-1, 0x1.15d2c1a2b30dep-1,
        0x1.1c516918434e8p-1, 0x1.22c881d4461a8p-1, 0x1.293814a20c610p-1, 0x1.2fa02a42ab5a6p-1,
        0x1.3600cb6d0870dp-1, 0x1.3c5a00cde51b0p-1, 0x1.42abd307eaa88p-1, 0x1.48f64ab3b6010p-1,
        0x1.4f39705fe3558p-1, 0x1.55754c9119c48p-1, 0x1.5ba9e7c216ef6p-1, 0x1.61d74a63ba842p-1,
        0x1.67fd7cdd11b78p-1, 0x1.6e1c878b62b3ep-1, 0x1.743472c237f9cp-1, 0x1.7a4546cb6bb25p-1,
        0x1.804f0be732f70p-1, 0x1.8651ca4c29095p-1, 0x1.8c4d8a275a7f8p-1, 0x1.9242539c5063fp-1,
        0x1.98302ec51b45fp-1, 0x1.9e1723b25e40bp-1, 0x1.a3f73a6b59f21p-1, 0x1.a9d07aedf7671p-1,
        0x1.afa2ed2ed2fb5p-1, 0x1.b56e991947295p-1, 0x1.bb33868f7752dp-1, 0x1.c0f1bd6a5a774p-1,
        0x1.c6a94579c5e2ap-1, 0x1.cc5a268477cc2p-1, 0x1.d204684821eb6p-1, 0x1.d7a8127973ff0p-1,
        0x1.dd452cc42647fp-1, 0x1.e2dbbecb03f93p-1, 0x1.e86bd027f59abp-1, 0x1.edf5686c0b5f7p-1,
        0x1.f3788f1f87715p-1, 0x1.f8f54bc1e82f2p-1, 0x1.fe6ba5c9f260ap-1, 0x1.01edd252ddae2p+0,
        0x1.04a2a7dd5991ep+0, 0x1.07545732d7392p+0, 0x1.0a02e3fd7860dp+0, 0x1.0cae51e31b20ep+0,
        0x1.0f56a4855ee24p+0, 0x1.11fbdf81a9501p+0, 0x1.149e06712b423p+0, 0x1.173d1ce8e5a1cp+0,
        0x1.19d92679ae48ep+0, 0x1.1c7226b034dccp+0, 0x1.1f08211507a0ep+0, 0x1.219b192c9845cp+0,
        0x1.242b127740b21p+0, 0x1.26b8107147c56p+0, 0x1.29421692e6166p+0, 0x1.2bc928504aaa7p+0,
        0x1.2e4d49199fa8ep+0, 0x1.30ce7c5b0f07ep+0, 0x1.334cc57cc7348p+0, 0x1.35c827e2ffb4bp+0,
        0x1.3840a6edfdc49p+0, 0x1.3ab645fa18edep+0, 0x1.3d29085fbf9aap+0, 0x1.3f98f1737ba14p+0,
        0x1.42060485f6cd3p+0, 0x1.447044e3ff606p+0, 0x1.46d7b5d68c913p+0, 0x1.493c5aa2c301dp+0,
        0x1.4b9e3689f9332p+0, 0x1.4dfd4cc9bbf22p+0, 0x1.5059a09bd2c09p+0, 0x1.52b335364437ep+0,
        0x1.550a0dcb5a67dp+0, 0x1.575e2d89a72ecp+0, 0x1.59af979c088e5p+0, 0x1.5bfe4f29acfa4p+0,
        0x1.5e4a575617a24p+0, 0x1.6093b34124b6ep+0, 0x1.62da66070da9ap+0, 0x1.651e72c06d682p+0,
        0x1.675fdc824491ep+0, 0x1.699ea65dfdaa1p+0, 0x1.6bdad36171434p+0, 0x1.6e146696ea278p+0,
        0x1.704b6305297a4p+0, 0x1.727fcbaf6ad73p+0, 0x1.74b1a395686a7p+0, 0x1.76e0edb35f052p+0,
        0x1.790dad02122d0p+0, 0x1.7b37e476d0276p+0, 0x1.7d5f970375fefp+0, 0x1.7f84c79673852p+0,
        0x1.81a7791acf4f6p+0, 0x1.83c7ae782aaeep+0, 0x1.85e56a92c5a40p+0, 0x1.8800b04b82cd8p+0,
        0x1.8a19827feb52ep+0, 0x1.8c2fe40a32c98p+0, 0x1.8e43d7c13b170p+0, 0x1.90556078984cep+0,
        0x1.9264810094812p+0, 0x1.94713c2633a2dp+0, 0x1.967b94b33747bp+0, 0x1.98838d6e22789p+0,
        0x1.9a89291a3d774p+0, 0x1.9c8c6a7799807p+0, 0x1.9e8d544314899p+0, 0x1.a08be9365cfa3p+0,
        0x1.a2882c07f5610p+0, 0x1.a4821f6b38248p+0, 0x1.a679c6105b2f2p+0, 0x1.a86f22a473981p+0,
        0x1.aa6237d179466p+0, 0x1.ac53083e4a910p+0, 0x1.ae41968eafda0p+0, 0x1.b02de5635f263p+0,
        0x1.b217f759ffafdp+0, 0x1.b3ffcf0d2d752p+0, 0x1.b5e56f147cc43p+0, 0x1.b7c8da047dc03p+0,
        0x1.b9aa126ebfe50p+0, 0x1.bb891ae1d584ep+0, 0x1.bd65f5e957437p+0, 0x1.bf40a60de78bcp+0,
        0x1.c1192dd536024p+0, 0x1.c2ef8fc202f36p+0, 0x1.c4c3ce5422be7p+0, 0x1.c695ec08813a7p+0,
        0x1.c865eb592519fp+0, 0x1.ca33cebd3348dp+0, 0x1.cbff98a8f2469p+0, 0x1.cdc94b8dcd7d5p+0,
        0x1.cf90e9da58947p+0, 0x1.d15675fa52bf4p+0, 0x1.d319f256aa081p+0, 0x1.d4db61557e978p+0,
        0x1.d69ac55a25f7ep+0, 0x1.d85820c52e544p+0, 0x1.da1375f461b4dp+0, 0x1.dbccc742c936bp+0,
        0x1.dd841708b0402p+0, 0x1.df39679ba7b12p+0, 0x1.e0ecbb4e89103p+0, 0x1.e29e147179b37p+0,
        0x1.e44d7551ede63p+0, 0x1.e5fae03aac0acp+0, 0x1.e7a65773cfb89p+0, 0x1.e94fdd42ccd65p+0,
        0x1.eaf773ea72b1bp+0, 0x1.ec9d1daaef11ep+0, 0x1.ee40dcc1d147bp+0, 0x1.efe2b36a0d39ep+0,
        0x1.f182a3dbfe6d1p+0, 0x1.f320b04d6b0a3p+0, 0x1.f4bcdaf186de7p+0, 0x1.f65725f8f65a5p+0,
        0x1.f7ef9391d18c4p+0, 0x1.f98625e7a717ap+0, 0x1.fb1adf237f284p+0, 0x1.fcadc16bde629p+0,
        0x1.fe3ecee4c8d11p+0, 0x1.ffce09afc4cd1p+0, 0x1.00adb9f5eef28p+1, 0x1.017387dad3df9p+1,
        0x1.02386f93a07c7p+1, 0x1.02fc722c2b014p+1, 0x1.03bf90af11fc4p+1, 0x1.0481cc25bdbbep+1,
        0x1.0543259861b85p+1, 0x1.06039e0dfdfaap+1, 0x1.06c3368c60827p+1, 0x1.0781f01826aa3p+1,
        0x1.083fcbb4be899p+1, 0x1.08fcca6468569p+1, 0x1.09b8ed2837c45p+1, 0x1.0a74350015614p+1,
        0x1.0b2ea2eabff30p+1, 0x1.0be837e5cdd10p+1, 0x1.0ca0f4edae3d1p+1, 0x1.0d58dafdaabb4p+1,
        0x1.0e0feb0fe8675p+1, 0x1.0ec6261d69498p+1, 0x1.0f7b8d1e0da82p+1, 0x1.10302108955a4p+1,
        0x1.10e3e2d2a115fp+1, 0x1.1196d370b3bf5p+1, 0x1.1248f3d633b4dp+1, 0x1.12fa44f56c1a0p+1,
        0x1.13aac7bf8e219p+1, 0x1.145a7d24b2557p+1, 0x1.15096613d9dcep+1, 0x1.15b7837aefc1dp+1,
        0x1.1664d646ca350p+1, 0x1.17115f632bcf1p+1, 0x1.17bd1fbac4d28p+1, 0x1.1868183734698p+1,
        0x1.191249c109e53p+1, 0x1.19bbb53fc5f8cp+1, 0x1.1a645b99dbf4ep+1, 0x1.1b0c3db4b3016p+1,
        0x1.1bb35c74a754cp+1, 0x1.1c59b8bd0b6b4p+1, 0x1.1cff5370293bdp+1, 0x1.1da42d6f436c1p+1,
        0x1.1e48479a96828p+1, 0x1.1eeba2d15a179p+1, 0x1.1f8e3ff1c205ap+1, 0x1.20301fd8ff96cp+1,
        0x1.20d1436342b19p+1, 0x1.2171ab6bbb055p+1, 0x1.221158cc99337p+1, 0x1.22b04c5f0ff8cp+1,
        0x1.234e86fb55544p+1, 0x1.23ec0978a3ae4p+1, 0x1.2488d4ad3afc3p+1, 0x1.2524e96e61e4ap+1,
        0x1.25c0489066e0dp+1, 0x1.265af2e6a15e4p+1, 0x1.26f4e94372dd7p+1, 0x1.278e2c7848107p+1,
        0x1.2826bd5599f79p+1, 0x1.28be9caaeefd0p+1, 0x1.2955cb46dc0f4p+1, 0x1.29ec49f705b9ep+1,
        0x1.2a821988213d4p+1, 0x1.2b173ac5f5a56p+1, 0x1.2babae7b5cde7p+1, 0x1.2c3f757244c94p+1,
        0x1.2cd29073b04dcp+1, 0x1.2d650047b86c8p+1, 0x1.2df6c5b58d4e9p+1, 0x1.2e87e1837754cp+1,
        0x1.2f185476d8257p+1, 0x1.2fa81f542bb8cp+1, 0x1.303742df0963bp+1, 0x1.30c5bfda24e2ep+1,
        0x1.315397074f62ap+1, 0x1.31e0c92778870p+1, 0x1.326d56faaf724p+1, 0x1.32f9414023c9cp+1,
        0x1.338488b626ba9p+1, 0x1.340f2e1a2bfbap+1, 0x1.34993228cad03p+1, 0x1.3522959dbf07ep+1,
        0x1.35ab5933e9fe3p+1, 0x1.36337da553989p+1, 0x1.36bb03ab2b43ap+1, 0x1.3741ebfdc8ef3p+1,
        0x1.37c83754ae088p+1, 0x1.384de66686746p+1, 0x1.38d2f9e929873p+1, 0x1.395772919afcfp+1,
        0x1.39db51140bee9p+1, 0x1.3a5e9623dbc7ep+1, 0x1.3ae14273993b0p+1, 0x1.3b6356b503335p+1,
        0x1.3be4d39909c7ap+1, 0x1.3c65b9cfcf2a8p+1, 0x1.3ce60a08a8999p+1, 0x1.3d65c4f21f4cfp+1,
        0x1.3de4eb39f1637p+1, 0x1.3e637d8d12cffp+1, 0x1.3ee17c97ae442p+1, 0x1.3f5ee905261adp+1,
        0x1.3fdbc3801540ap+1, 0x1.40580cb2501d0p+1, 0x1.40d3c544e5788p+1, 0x1.414eede01f62ap+1,
        0x1.41c9872b84175p+1, 0x1.424391cdd6e27p+1, 0x1.42bd0e6d1902dp+1, 0x1.4335fdae8a8bbp+1,
        0x1.43ae6036ab461p+1, 0x1.442636a93b8fdp+1, 0x1.449d81a93d3b0p+1, 0x1.451441d8f46b3p+1,
        0x1.458a77d9e8728p+1, 0x1.4600244ce4ad0p+1, 0x1.467547d1f95b8p+1, 0x1.46e9e3087c7dap+1,
        0x1.475df68f0aaa1p+1, 0x1.47d1830387e69p+1, 0x1.48448903207ecp+1, 0x1.48b7092a49d9bp+1,
        0x1.49290414c34f0p+1, 0x1.499a7a5d96fa9p+1, 0x1.4a0b6c9f1a8f1p+1, 0x1.4a7bdb72f028ep+1,
        0x1.4aebc772071ddp+1, 0x1.4b5b31349cce3p+1, 0x1.4bca19523d734p+1, 0x1.4c388061c4edbp+1,
        0x1.4ca666f95f92bp+1, 0x1.4d13cdae8af80p+1, 0x1.4d80b51616bfdp+1, 0x1.4ded1dc42562ap+1,
        0x1.4e59084c2cf90p+1, 0x1.4ec47540f8046p+1, 0x1.4f2f6534a6360p+1, 0x1.4f99d8b8ad369p+1,
        0x1.5003d05dd96b7p+1, 0x1.506d4cb44ebc1p+1, 0x1.50d64e4b89559p+1, 0x1.513ed5b25e6e6p+1,
        0x1.51a6e376fd085p+1, 0x1.520e7826eeb1cp+1, 0x1.5275944f1846bp+1, 0x1.52dc387bbaafep+1,
        0x1.5342653873a1ep+1, 0x1.53a81b103e5afp+1, 0x1.540d5a8d74600p+1, 0x1.54722439ce38dp+1,
        0x1.54d6789e642b5p+1, 0x1.553a5843aef67p+1, 0x1.559dc3b1888b0p+1, 0x1.5600bb6f2cc55p+1,
        0x1.566340033a24bp+1, 0x1.56c551f3b2826p+1, 0x1.5726f1c5fbc86p+1, 0x1.57881ffee0a68p+1,
        0x1.57e8dd2291471p+1, 0x1.584929b4a402fp+1, 0x1.58a9063816140p+1, 0x1.5908732f4c47ep+1,
        0x1.5967711c13b10p+1, 0x1.59c6007fa2577p+1, 0x1.5a2421da97e82p+1, 0x1.5a81d5acfe646p+1,
        0x1.5adf1c764acfcp+1, 0x1.5b3bf6b55ddd8p+1, 0x1.5b9864e8849d5p+1, 0x1.5bf4678d7926cp+1,
        0x1.5c4fff2163447p+1, 0x1.5cab2c20d91e8p+1, 0x1.5d05ef07dfe3cp+1, 0x1.5d604851ec72bp+1,
        0x1.5dba3879e400ep+1, 0x1.5e13bffa1cc2ep+1, 0x1.5e6cdf4c5e91ep+1, 0x1.5ec596e9e391fp+1,
        0x1.5f1de74b58d6dp+1, 0x1.5f75d0e8df077p+1, 0x1.5fcd543a0b027p+1, 0x1.602471b5e67fbp+1,
        0x1.607b29d2f0b30p+1, 0x1.60d17d071eed3p+1, 0x1.61276bc7dd3c1p+1, 0x1.617cf68a0f0a8p+1,
        0x1.61d21dc20fbfap+1, 0x1.6226e1e3b35cap+1, 0x1.627b4362471acp+1, 0x1.62cf42b092079p+1,
        0x1.6322e040d5a1fp+1, 0x1.63761c84ce74ap+1, 0x1.63c8f7edb4b15p+1, 0x1.641b72ec3ccafp+1,
        0x1.646d8df0980e4p+1}};
const double S2T[3][397] = {{
        0x0.0p+0, 0x0.0p+0, 0x1.f07c1f07c1a08p-7, 0x1.72abcb87d3d8dp-5,
        0x1.70fd7416302f0p-4, 0x1.3219b3a0a9f42p-3, 0x1.c913ebbf7159fp-3, 0x1.3e836cb3f17bdp-2,
        0x1.a6c6acc5a8ca8p-2, 0x1.0e90ea8a98dc3p-1, 0x1.50b1d96124bcfp-1, 0x1.99addf561f336p-1,
        0x1.e96d0d8d309a5p-1, 0x1.1febe45b28d9bp+0, 0x1.4e6b63f51167dp+0, 0x1.802989c51ebe9p+0,
        0x1.b51b0212b916bp+0, 0x1.ed34a0aff24b0p+0, 0x1.1435b037be481p+1, 0x1.335a314e41823p+1,
        0x1.5402773a2ec57p+1, 0x1.762938510cc6fp+1, 0x1.99c93d5de8872p+1, 0x1.bedd6160e4f92p+1,
        0x1.e560914fab8bfp+1, 0x1.06a6e5eb5d545p+2, 0x1.1b50108dc7801p+2, 0x1.30a9593fd25aep+2,
        0x1.46b0593226c25p+2, 0x1.5d62b1f7ac703p+2, 0x1.74be0d6845a03p+2, 0x1.8cc01d83f0e48p+2,
        0x1.a5669c564fbdep+2, 0x1.beaf4bda9097fp+2, 0x1.d897f5dfbad76p+2, 0x1.f31e6bed5b970p+2,
        0x1.0720439448e75p+3, 0x1.14fe141cbc45fp+3, 0x1.23279b9876fd9p+3, 0x1.319bd1b75afdep+3,
        0x1.4059b1c3fb7f5p+3, 0x1.4f603a9707f86p+3, 0x1.5eae6e8ae3032p+3, 0x1.6e43536f64928p+3,
        0x1.7e1df27dc6e2ap+3, 0x1.8e3d584cbd941p+3, 0x1.9ea094c4b656cp+3, 0x1.af46bb1442978p+3,
        0x1.c02ee1a4a9a2bp+3, 0x1.d158220ea2aadp+3, 0x1.e2c1990f362bbp+3, 0x1.f46a667cc61c2p+3,
        0x1.0328d69e1e36cp+4, 0x1.0c3b499b2fa4dp+4, 0x1.156c20a6a54c2p+4, 0x1.1ebaf1a9071ddp+4,
        0x1.282753fd38478p+4, 0x1.31b0e06b6a530p+4, 0x1.3b57312421e48p+4, 0x1.4519e1bb4cdf6p+4,
        0x1.4ef88f2369b0fp+4, 0x1.58f2d7a8bf85bp+4, 0x1.63085aeca7308p+4, 0x1.6d38b9e0e4859p+4,
        0x1.778396c30ff39p+4, 0x1.81e89518101f0p+4, 0x1.8c6759a7a34b9p+4, 0x1.96ff8a77f8574p+4,
        0x1.a1b0cec95716ap+4, 0x1.ac7acf11d7d6fp+4, 0x1.b75d34f929d79p+4, 0x1.c257ab54687eep+4,
        0x1.cd69de21ff1e7p+4, 0x1.d8937a859b0edp+4, 0x1.e3d42ec42bf33p+4, 0x1.ef2baa3ff1f1ep+4,
        0x1.fa999d7499b2fp+4, 0x1.030edcf9b2fa4p+5, 0x1.08dbd92fb3408p+5, 0x1.0eb39d34de309p+5,
        0x1.14960366f5890p+5, 0x1.1a82e6a71b4d5p+5, 0x1.207a22580725ep+5, 0x1.267b925c42022p+5,
        0x1.2c87131467e57p+5, 0x1.329c815d6fc9bp+5, 0x1.38bbba8ef9831p+5, 0x1.3ee49c79a18d4p+5,
        0x1.451705655ab1dp+5, 0x1.4b52d40fcd6f0p+5, 0x1.5197e7aabd0e7p+5, 0x1.57e61fda72553p+5,
        0x1.5e3d5cb42bbb4p+5, 0x1.649d7ebc93163p+5, 0x1.6b0666e638a4fp+5, 0x1.7177f69013694p+5,
        0x1.77f20f8406bc0p+5, 0x1.7e7493f56d0afp+5, 0x1.84ff667fa7ad2p+5, 0x1.8b926a24b3bdep+5,
        0x1.922d824bc3e8bp+5, 0x1.98d092bfdf1a9p+5, 0x1.9f7b7fae84035p+5, 0x1.a62e2da651570p+5,
        0x1.ace88195b2c09p+5, 0x1.b3aa60c99272ap+5, 0x1.ba73b0ec0f497p+5, 0x1.c1445803376b0p+5,
        0x1.c81c3c6fc7575p+5, 0x1.cefb44ebed598p+5, 0x1.d5e1588a114a0p+5, 0x1.dcce5eb3a0918p+5,
        0x1.e3c23f27de60fp+5, 0x1.eabce1fab80cap+5, 0x1.f1be2f939d7d9p+5, 0x1.f8c610ac5dabap+5,
        0x1.ffd46e50070f6p+5, 0x1.037498ece600cp+6, 0x1.07022279f57c2p+6, 0x1.0a92c8cb4d49ep+6,
        0x1.0e2681037cb83p+6, 0x1.11bd406b00d64p+6, 0x1.1556fc6fc00ccp+6, 0x1.18f3aaa487840p+6,
        0x1.1c9340c08a527p+6, 0x1.2035b49ee26c9p+6, 0x1.23dafc3e134fep+6, 0x1.27830dbf8e64ep+6,
        0x1.2b2ddf67390efp+6, 0x1.2edb679af466dp+6, 0x1.328b9ce226989p+6, 0x1.363e75e545e02p+6,
        0x1.39f3e96d651dep+6, 0x1.3dabee63c1fe1p+6, 0x1.41667bd154ae3p+6, 0x1.452388de6119fp+6,
        0x1.48e30cd209ab0p+6, 0x1.4ca4ff11e3862p+6, 0x1.506957218c406p+6, 0x1.54300ca24107bp+6,
        0x1.57f917527738ep+6, 0x1.5bc46f0d765f3p+6, 0x1.5f920bcaf3981p+6, 0x1.6361e59eae553p+6,
        0x1.6733f4b80e7a1p+6, 0x1.6b083161c3ce8p+6, 0x1.6ede940166c2dp+6, 0x1.72b715171a7fbp+6,
        0x1.7691ad3d303f1p+6, 0x1.7a6e5527cbe76p+6, 0x1.7e4d05a489e62p+6, 0x1.822db79a2645ep+6,
        0x1.8610640824f9bp+6, 0x1.89f504067b5b1p+6, 0x1.8ddb90c53ad65p+6, 0x1.91c4038c3cc00p+6,
        0x1.95ae55bacf50dp+6, 0x1.999a80c763c2ep+6, 0x1.9d887e3f3d8d5p+6, 0x1.a17847c622b99p+6,
        0x1.a569d7160d4e5p+6, 0x1.a95d25feddcdap+6, 0x1.ad522e660ec11p+6, 0x1.b148ea4669510p+6,
        0x1.b54153afbae30p+6, 0x1.b93b64c68bbb5p+6, 0x1.bd3717c3d6a06p+6, 0x1.c13466f4c1782p+6,
        0x1.c5334cba56e12p+6, 0x1.c933c38940bfdp+6, 0x1.cd35c5e983becp+6, 0x1.d1394e763bbe1p+6,
        0x1.d53e57dd592ebp+6, 0x1.d944dcdf5f55fp+6, 0x1.dd4cd84f23774p+6, 0x1.e15645118cdf8p+6,
        0x1.e5611e1d55d01p+6, 0x1.e96d5e7acd45ep+6, 0x1.ed7b01439999dp+6, 0x1.f18a01a27bf78p+6,
        0x1.f59a5ad314a6fp+6, 0x1.f9ac0821a826fp+6, 0x1.fdbf04eae5155p+6, 0x1.00e9a64dd570cp+7,
        0x1.02f46d5868a39p+7, 0x1.04ffd55b78475p+7, 0x1.070bdc251547ep+7, 0x1.09187f8afa3a1p+7,
        0x1.0b25bd6a709bdp+7, 0x1.0d3393a836715p+7, 0x1.0f420030643efp+7, 0x1.115100f6535d0p+7,
        0x1.136093f484a6bp+7, 0x1.1570b72c877f2p+7, 0x1.178168a6e1304p+7, 0x1.1992a672f49d4p+7,
        0x1.1ba46ea6ea4c0p+7, 0x1.1db6bf5f98c02p+7, 0x1.1fc996c06d2a1p+7, 0x1.21dcf2f354673p+7,
        0x1.23f0d228a4519p+7, 0x1.2605329705600p+7, 0x1.281a127b5c930p+7, 0x1.2a2f7018b5b01p+7,
        0x1.2c4549b82dc7ap+7, 0x1.2e5b9da8de074p+7, 0x1.30726a3fc6d4bp+7, 0x1.3289add7bb32dp+7,
        0x1.34a166d14c6d6p+7, 0x1.36b99392b60d3p+7, 0x1.38d23287ca10fp+7, 0x1.3aeb4221dd6b6p+7,
        0x1.3d04c0d7b4c6bp+7, 0x1.3f1ead2571890p+7, 0x1.4139058c7f1d3p+7, 0x1.4353c893807bep+7,
        0x1.456ef4c63df4dp+7, 0x1.478a88b593387p+7, 0x1.49a682f75d9fep+7, 0x1.4bc2e2266ab30p+7,
        0x1.4ddfa4e266eacp+7, 0x1.4ffcc9cfccb08p+7, 0x1.521a4f97d3985p+7, 0x1.543834e85fd64p+7,
        0x1.56567873f1ec8p+7, 0x1.587518f196939p+7, 0x1.5a94151cd6d9ap+7, 0x1.5cb36bb5a87a7p+7,
        0x1.5ed31b805e6c5p+7, 0x1.60f3234599a50p+7, 0x1.631381d23a121p+7, 0x1.653435f74fc69p+7,
        0x1.67553e8a0c5c4p+7, 0x1.69769a63b487fp+7, 0x1.6b98486191e01p+7, 0x1.6dba4764e4d4fp+7,
        0x1.6fdc9652d6d9bp+7, 0x1.71ff34146cbecp+7, 0x1.74221f967939cp+7, 0x1.764557c98f9f0p+7,
        0x1.7868dba1f6c71p+7, 0x1.7a8caa179c23cp+7, 0x1.7cb0c22607011p+7, 0x1.7ed522cc4bf38p+7,
        0x1.80f9cb0d00715p+7, 0x1.831eb9ee2e983p+7, 0x1.8543ee79491c3p+7, 0x1.876967bb1f62dp+7,
        0x1.898f24c3d1c52p+7, 0x1.8bb524a6c5fd0p+7, 0x1.8ddb667a9bb99p+7, 0x1.9001e959215bdp+7,
        0x1.9228ac5f48db0p+7, 0x1.944faead1ccebp+7, 0x1.9676ef65b5a07p+7, 0x1.989e6daf2ee22p+7,
        0x1.9ac628b29cca1p+7, 0x1.9cee1f9c01d3bp+7, 0x1.9f16519a44846p+7, 0x1.a13ebddf2553fp+7,
        0x1.a367639f34b8ep+7, 0x1.a5904211c9563p+7, 0x1.a7b95870f64cdp+7, 0x1.a9e2a5f981ae4p+7,
        0x1.ac0c29eadb105p+7, 0x1.ae35e38712423p+7, 0x1.b05fd212ce217p+7, 0x1.b289f4d5438fep+7,
        0x1.b4b44b182c881p+7, 0x1.b6ded427bf517p+7, 0x1.b9098f52a5d31p+7, 0x1.bb347be9f5048p+7,
        0x1.bd5f9941247d0p+7, 0x1.bf8ae6ae061fcp+7, 0x1.c1b66388bde51p+7, 0x1.c3e20f2bb9c10p+7,
        0x1.c60de8f3a9a52p+7, 0x1.c839f03f779f6p+7, 0x1.ca66247040139p+7, 0x1.cc9284e94a103p+7,
        0x1.cebf110fffbe8p+7, 0x1.d0ebc84be6ec5p+7, 0x1.d318aa0699b07p+7, 0x1.d545b5abbf285p+7,
        0x1.d772eaa9044eep+7, 0x1.d9a0486e14ed8p+7, 0x1.dbcdce6c94a48p+7, 0x1.ddfb7c18180bbp+7,
        0x1.e02950e61dec9p+7, 0x1.e2574c4e08920p+7, 0x1.e4856dc917315p+7, 0x1.e6b3b4d25f68fp+7,
        0x1.e8e220e6c6d61p+7, 0x1.eb10b184fcc13p+7, 0x1.ed3f662d73e0cp+7, 0x1.ef6e3e625c304p+7,
        0x1.f19d39a79cdefp+7, 0x1.f3cc5782ce51bp+7, 0x1.f5fb977b343bap+7, 0x1.f82af919b7c99p+7,
        0x1.fa5a7be8e1e32p+7, 0x1.fc8a1f74d57f6p+7, 0x1.feb9e34b4a0d9p+7, 0x1.0074e37dc2f86p+8,
        0x1.018ce50b2c879p+8, 0x1.02a4f6170bba7p+8, 0x1.03bd166b4a045p+8, 0x1.04d545d28da97p+8,
        0x1.05ed8418372c3p+8, 0x1.0705d1085ec28p+8, 0x1.081e2c6fd1d4dp+8, 0x1.0936961c10857p+8,
        0x1.0a4f0ddb4b40ep+8, 0x1.0b67937c60562p+8, 0x1.0c8026ced997ap+8, 0x1.0d98c7a2ea04fp+8,
        0x1.0eb175c96b7b7p+8, 0x1.0fca3113dc70ep+8, 0x1.10e2f9545db39p+8, 0x1.11fbce5db034bp+8,
        0x1.1314b00332d86p+8, 0x1.142d9e18e04f3p+8, 0x1.154698734cf5bp+8, 0x1.165f9ee7a4bc0p+8,
        0x1.1778b14ba9150p+8, 0x1.1891cf75aeebfp+8, 0x1.19aaf93c9ca23p+8, 0x1.1ac42e77e812cp+8,
        0x1.1bdd6eff949e2p+8, 0x1.1cf6baac313bcp+8, 0x1.1e101156d692fp+8, 0x1.1f2972d9251a4p+8,
        0x1.2042df0d433d2p+8, 0x1.215c55cddb88cp+8, 0x1.2275d6f61ade2p+8, 0x1.238f6261aeabdp+8,
        0x1.24a8f7ecc32c9p+8, 0x1.25c2977401ad2p+8, 0x1.26dc40d48ed74p+8, 0x1.27f5f3ec0903cp+8,
        0x1.290fb09886906p+8, 0x1.2a2976b8943eap+8, 0x1.2b43462b33949p+8, 0x1.2c5d1ecfd946fp+8,
        0x1.2d7700866ba5bp+8, 0x1.2e90eb2f41107p+8, 0x1.2faadeab1e6e7p+8, 0x1.30c4dadb35ad6p+8,
        0x1.31dedfa124443p+8, 0x1.32f8ecdef1bc5p+8, 0x1.341302770e3ecp+8, 0x1.352d204c51276p+8,
        0x1.36474641f79c5p+8, 0x1.3761743ba32aep+8, 0x1.387baa1d5868bp+8, 0x1.3995e7cb7d9a8p+8,
        0x1.3ab02d2ad95e9p+8, 0x1.3bca7a20915cfp+8, 0x1.3ce4ce9228fb0p+8, 0x1.3dff2a6580150p+8,
        0x1.3f198d80d1bacp+8, 0x1.4033f7cab2f13p+8, 0x1.414e692a1178dp+8, 0x1.4268e18632975p+8,
        0x1.438360c6b1e6bp+8, 0x1.449de6d380277p+8, 0x1.45b87394e2176p+8, 0x1.46d306f36f4cap+8,
        0x1.47eda0d811141p+8, 0x1.4908412c01548p+8, 0x1.4a22e7d8c9752p+8, 0x1.4b3d94c84147ap+8,
        0x1.4c5847e48df73p+8, 0x1.4d73011820f9ap+8, 0x1.4e8dc04db705cp+8, 0x1.4fa88570570c3p+8,
        0x1.50c3506b5134dp+8, 0x1.51de212a3dde5p+8, 0x1.52f8f798fca30p+8, 0x1.5413d3a3b35f9p+8,
        0x1.552eb536cd3e2p+8, 0x1.56499c3ef9c3ep+8, 0x1.576488a92be2fp+8, 0x1.587f7a62990ecp+8,
        0x1.599a7158b8540p+8, 0x1.5ab56d7941735p+8, 0x1.5bd06eb22c002p+8, 0x1.5ceb74f1ae80fp+8,
        0x1.5e0680263d942p+8, 0x1.5f21903e8b171p+8, 0x1.603ca5298550ap+8, 0x1.6157bed6561d7p+8,
        0x1.6272dd346220ep+8, 0x1.638e003347f74p+8, 0x1.64a927c2df6bdp+8, 0x1.65c453d338b08p+8,
        0x1.66df84549b99ep+8}, {
        0x0.0p+0, 0x0.0p+0, 0x1.f07c1f07c1e08p-7, 0x1.718aee5e80506p-5,
        0x1.6ec0fa4b44886p-4, 0x1.2f549955436dap-3, 0x1.c3968e1734547p-3, 0x1.39c0d6e29c79ap-2,
        0x1.9f3a846cf9e31p-2, 0x1.08f4d60a9540fp-1, 0x1.48c0b9e76ef3ap-1, 0x1.8edb60587a01bp-1,
        0x1.db201665fb1d2p-1, 0x1.16b57f4eec28fp+0, 0x1.42cc861ca4da0p+0, 0x1.71c3ff2f630b0p+0,
        0x1.a38b2dbfc3b41p+0, 0x1.d811b66803222p+0, 0x1.07a3ce76b3f4ap+1, 0x1.248ea10b47231p+1,
        0x1.42c1b0c7248d5p+1, 0x1.623587e96799bp+1, 0x1.82e2dc1876b36p+1, 0x1.a4c28d657bc2cp+1,
        0x1.c7cda55599eadp+1, 0x1.ebfd55f0bf257p+1, 0x1.08a57c6af88d7p+2, 0x1.1bd8072a7aa97p+2,
        0x1.2f931e469ac85p+2, 0x1.43d3a748e0f65p+2, 0x1.589699c9296cep+2, 0x1.6dd8ff0497078p+2,
        0x1.8397f176e8ee1p+2, 0x1.99d09c7625931p+2, 0x1.b0803bd08d72cp+2, 0x1.c7a41b6cc84b7p+2,
        0x1.df3996ec3fdd2p+2, 0x1.f73e194f9b80ep+2, 0x1.07d78e4ea81b1p+3, 0x1.144514c51c86fp+3,
        0x1.20e66b92160f8p+3, 0x1.2dba653f4039fp+3, 0x1.3abfdb303cd3bp+3, 0x1.47f5ad7ac714cp+3,
        0x1.555ac2bfbead8p+3, 0x1.62ee0805157efp+3, 0x1.70ae70909ad35p+3, 0x1.7e9af5c39f109p+3,
        0x1.8cb296f76af70p+3, 0x1.9af4595a859fdp+3, 0x1.a95f47cec486fp+3, 0x1.b7f272c821079p+3,
        0x1.c6acf02c4ed31p+3, 0x1.d58ddb330ef92p+3, 0x1.e49454473b3d0p+3, 0x1.f3bf80e885876p+3,
        0x1.018745c6f3abdp+4, 0x1.094051c45e9e4p+4, 0x1.110a7e744531fp+4, 0x1.18e5682fb03a9p+4,
        0x1.20d0ad9373fb1p+4, 0x1.28cbef7302ccap+4, 0x1.30d6d0cb8c6edp+4, 0x1.38f0f6b768459p+4,
        0x1.411a0861c8cfap+4, 0x1.4951aefab6abdp+4, 0x1.519795ab518ccp+4, 0x1.59eb698a5583dp+4,
        0x1.624cd990e3162p+4, 0x1.6abb968f88979p+4, 0x1.733753238b4e5p+4, 0x1.7bbfc3ac6eed8p+4,
        0x1.84549e41b9fb9p+4, 0x1.8cf59aa8f5c17p+4, 0x1.95a2724be869cp+4, 0x1.9e5ae02f07fabp+4,
        0x1.a71ea0e824e39p+4, 0x1.afed72954ad6bp+4, 0x1.b8c714d3d6b6ap+4, 0x1.c1ab48b7c0610p+4,
        0x1.ca99d0c317289p+4, 0x1.d39270ddafd82p+4, 0x1.dc94ee4d031edp+4, 0x1.e5a10fac3b4acp+4,
        0x1.eeb69ce4703f5p+4, 0x1.f7d55f25109b0p+4, 0x1.007e906e3b821p+5, 0x1.0516d6d8554eap+5,
        0x1.09b3693c24d2bp+5, 0x1.0e542e99cea71p+5, 0x1.12f90e82eaed0p+5, 0x1.17a1f117370a9p+5,
        0x1.1c4ebf015aa38p+5, 0x1.20ff6173bf5ddp+5, 0x1.25b3c2257b047p+5, 0x1.2a6bcb4f4b9d9p+5,
        0x1.2f2767a8a50c7p+5, 0x1.33e68264cfd7dp+5, 0x1.38a9073018b1fp+5, 0x1.3d6ee22d10605p+5,
        0x1.4237fff1dba3fp+5, 0x1.47044d8592c51p+5, 0x1.4bd3b85db0676p+5, 0x1.50a62e5b8f4dcp+5,
        0x1.557b9dc9f6b6bp+5, 0x1.5a53f55ab4fc4p+5, 0x1.5f2f242448250p+5, 0x1.640d199f9414bp+5,
        0x1.68edc5a5a60c9p+5, 0x1.6dd1186d852f1p+5, 0x1.72b7028a0fbbap+5, 0x1.779f74e7e4b56p+5,
        0x1.7c8a60cb59b14p+5, 0x1.8177b7ce7c80cp+5, 0x1.86676bdf20761p+5, 0x1.8b596f3cf6ff0p+5,
        0x1.904db477b3526p+5, 0x1.95442e6d38f17p+5, 0x1.9a3cd047d4bd7p+5, 0x1.9f378d7c8063ep+5,
        0x1.a43459c92fe56p+5, 0x1.a933293328fd6p+5, 0x1.ae33f005642dfp+5, 0x1.b336a2cef73c7p+5,
        0x1.b83b366188e3cp+5, 0x1.bd419fcfcd894p+5, 0x1.c249d46c0cbe7p+5, 0x1.c753c9c6af5d0p+5,
        0x1.cc5f75acd60b6p+5, 0x1.d16cce26f7f7bp+5, 0x1.d67bc977899adp+5, 0x1.db8c5e19ab537p+5,
        0x1.e09e82bfdfaccp+5, 0x1.e5b22e52c9230p+5, 0x1.eac757efef3b6p+5, 0x1.efddf6e88ac3cp+5,
        0x1.f4f602c05910ep+5, 0x1.fa0f732c76125p+5, 0x1.ff2a40123d15dp+5, 0x1.022330c31807cp+6,
        0x1.04b1e7e572a2cp+6, 0x1.074141a7fd9b2p+6, 0x1.09d13a58894c0p+6, 0x1.0c61ce5a67257p+6,
        0x1.0ef2fa25ec8f4p+6, 0x1.1184ba47f8a34p+6, 0x1.14170b617caf1p+6, 0x1.16a9ea27076d1p+6,
        0x1.193d536052e4dp+6, 0x1.1bd143e7d4e37p+6, 0x1.1e65b8aa51fd5p+6, 0x1.20faaea673090p+6,
        0x1.239022ec5d06ep+6, 0x1.2626129d4b65cp+6, 0x1.28bc7aeb2c97dp+6, 0x1.2b53591840e9bp+6,
        0x1.2deaaa76bb8f1p+6, 0x1.30826c6865d77p+6, 0x1.331a9c5e447f2p+6, 0x1.35b337d83f0fbp+6,
        0x1.384c3c64c943bp+6, 0x1.3ae5a7a08e632p+6, 0x1.3d7f77361e8b9p+6, 0x1.4019a8dd9dd9ep+6,
        0x1.42b43a5c756a4p+6, 0x1.454f29850623fp+6, 0x1.47ea74365d461p+6, 0x1.4a86185beaab3p+6,
        0x1.4d2213ed38badp+6, 0x1.4fbe64eda5fc5p+6, 0x1.525b096c20463p+6, 0x1.54f7ff82e17aep+6,
        0x1.579545572dcf5p+6, 0x1.5a32d919138e8p+6, 0x1.5cd0b9032c530p+6, 0x1.5f6ee35a5fae4p+6,
        0x1.620d566da7340p+6, 0x1.64ac1095d3e35p+6, 0x1.674b103554e3fp+6, 0x1.69ea53b7ff913p+6,
        0x1.6c89d992d8ca0p+6, 0x1.6f29a043df7fbp+6, 0x1.71c9a651d87aep+6, 0x1.7469ea4c1b513p+6,
        0x1.770a6aca6083cp+6, 0x1.79ab266c90beep+6, 0x1.7c4c1bda9536ap+6, 0x1.7eed49c429185p+6,
        0x1.818eaee0ac0a3p+6, 0x1.843049eef5b4ap+6, 0x1.86d219b52a4dap+6, 0x1.89741d0090213p+6,
        0x1.8c1652a566110p+6, 0x1.8eb8b97ebb04fp+6, 0x1.915b506e4648cp+6, 0x1.93fe165c40cedp+6,
        0x1.96a10a373f55cp+6, 0x1.99442af40d692p+6, 0x1.9be7778d893a3p+6, 0x1.9e8aef04804a5p+6,
        0x1.a12e905f8ce3cp+6, 0x1.a3d25aaaf45acp+6, 0x1.a6764cf88612fp+6, 0x1.a91a665f7b44fp+6,
        0x1.abbea5fc577f6p+6, 0x1.ae630af0c9df6p+6, 0x1.b10794638efb5p+6, 0x1.b3ac4180537d7p+6,
        0x1.b651117797691p+6, 0x1.b8f6037e9207cp+6, 0x1.bb9b16cf16780p+6, 0x1.be404aa778de0p+6,
        0x1.c0e59e4a742e6p+6, 0x1.c38b10ff10922p+6, 0x1.c630a2108a605p+6, 0x1.c8d650ce39a7cp+6,
        0x1.cb7c1c8b7a48ep+6, 0x1.ce22049f9498ap+6, 0x1.d0c80865a68d7p+6, 0x1.d36e273c8d6f2p+6,
        0x1.d6146086d009ap+6, 0x1.d8bab3aa895edp+6, 0x1.db61201153d30p+6, 0x1.de07a52834d3fp+6,
        0x1.e0ae425f88f5fp+6, 0x1.e354f72af083dp+6, 0x1.e5fbc3013c80dp+6, 0x1.e8a2a55c5c190p+6,
        0x1.eb499db94a7d5p+6, 0x1.edf0ab97fd290p+6, 0x1.f097ce7b528efp+6, 0x1.f33f05e9012abp+6,
        0x1.f5e6516986f55p+6, 0x1.f88db08819397p+6, 0x1.fb3522d294c5bp+6, 0x1.fddca7d96e7b8p+6,
        0x1.00421f97d21b6p+7, 0x1.0195f43557075p+7, 0x1.02e9d19137f5cp+7, 0x1.043db77895b7fp+7,
        0x1.0591a5b9b91a9p+7, 0x1.06e59c240c2b6p+7, 0x1.08399a8813a68p+7, 0x1.098da0b768899p+7,
        0x1.0ae1ae84b1cc4p+7, 0x1.0c35c3c39e3dfp+7, 0x1.0d89e048de863p+7, 0x1.0ede03ea1f490p+7,
        0x1.10322e7e036c3p+7, 0x1.11865fdc1e7f0p+7, 0x1.12da97dcef420p+7, 0x1.142ed659da4e1p+7,
        0x1.15831b2d24dbdp+7, 0x1.16d76631efa84p+7, 0x1.182bb74431f7ep+7, 0x1.19800e40b4b62p+7,
        0x1.1ad46b050db20p+7, 0x1.1c28cd6f9af5bp+7, 0x1.1d7d355f7e39dp+7, 0x1.1ed1a2b49872bp+7,
        0x1.2026154f8577ep+7, 0x1.217a8d1197c4bp+7, 0x1.22cf09dcd451ap+7, 0x1.24238b93ee862p+7,
        0x1.2578121a44411p+7, 0x1.26cc9d53d9f9dp+7, 0x1.28212d2556f61p+7, 0x1.2975c17401974p+7,
        0x1.2aca5a25bbbcap+7, 0x1.2c1ef720ff3a4p+7, 0x1.2d73984cda658p+7, 0x1.2ec83d90ecb4cp+7,
        0x1.301ce6d563729p+7, 0x1.31719402f6854p+7, 0x1.32c64502e5481p+7, 0x1.341af9bef376fp+7,
        0x1.356fb221662c8p+7, 0x1.36c46e1500f15p+7, 0x1.38192d8502dc5p+7, 0x1.396df05d23c3ep+7,
        0x1.3ac2b689917f7p+7, 0x1.3c177ff6ed38bp+7, 0x1.3d6c4c9248cccp+7, 0x1.3ec11c49243c7p+7,
        0x1.4015ef096b2b9p+7, 0x1.416ac4c1726e8p+7, 0x1.42bf9d5ff5a62p+7, 0x1.441478d414ea2p+7,
        0x1.4569570d52800p+7, 0x1.46be37fb90a00p+7, 0x1.48131b8f0f470p+7, 0x1.496801b86a14fp+7,
        0x1.4abcea6896376p+7, 0x1.4c11d590e060dp+7, 0x1.4d66c322eacb3p+7, 0x1.4ebbb310ab46ap+7,
        0x1.5010a54c6952ap+7, 0x1.516599c8bc431p+7, 0x1.52ba907889703p+7, 0x1.540f894f026f8p+7,
        0x1.5564843fa359bp+7, 0x1.56b9813e31179p+7, 0x1.580e803eb7bbfp+7, 0x1.5963813588e44p+7,
        0x1.5ab884173a24fp+7, 0x1.5c0d88d8a37d4p+7, 0x1.5d628f6eddd59p+7, 0x1.5eb797cf41851p+7,
        0x1.600ca1ef64e11p+7, 0x1.6161adc51ad44p+7, 0x1.62b6bb46717e0p+7, 0x1.640bca69b0da3p+7,
        0x1.6560db2559706p+7, 0x1.66b5ed70230afp+7, 0x1.680b0140fb755p+7, 0x1.6960168f05423p+7,
        0x1.6ab52d519697fp+7, 0x1.6c0a458038050p+7, 0x1.6d5f5f12a35a3p+7, 0x1.6eb47a00c28c9p+7,
        0x1.70099642ae9cdp+7, 0x1.715eb3d0ae85cp+7, 0x1.72b3d2a336315p+7, 0x1.7408f2b2e5722p+7,
        0x1.755e13f88704ep+7, 0x1.76b3366d0f965p+7, 0x1.78085a099ccf0p+7, 0x1.795d7ec774656p+7,
        0x1.7ab2a4a003349p+7, 0x1.7c07cb8cdc583p+7, 0x1.7d5cf387b84e8p+7, 0x1.7eb21c8a741e2p+7,
        0x1.8007468f10826p+7, 0x1.815c718fb11a9p+7, 0x1.82b19d869b9fbp+7, 0x1.8406ca6e371e1p+7,
        0x1.855bf8410b32cp+7, 0x1.86b126f9bf4ecp+7, 0x1.8806569319fd6p+7, 0x1.895b8708002fep+7,
        0x1.8ab0b853748bep+7, 0x1.8c05ea7096bf0p+7, 0x1.8d5b1d5aa2d5ep+7, 0x1.8eb0510cf096dp+7,
        0x1.90058582f2e10p+7, 0x1.915abab8370e5p+7, 0x1.92aff0a864599p+7, 0x1.9405274f3b481p+7,
        0x1.955a5ea895160p+7, 0x1.96af96b06326fp+7, 0x1.9804cf62ae796p+7, 0x1.995a08bb971d7p+7,
        0x1.9aaf42b753ae4p+7, 0x1.9c047d5230cf2p+7, 0x1.9d59b88890ab6p+7, 0x1.9eaef456ea790p+7,
        0x1.a00430b9c9fe0p+7, 0x1.a1596dadcf193p+7, 0x1.a2aeab2fad4d0p+7, 0x1.a403e93c2b4d4p+7,
        0x1.a55927d0228f5p+7, 0x1.a6ae66e87edd9p+7, 0x1.a803a6823debfp+7, 0x1.a958e69a6ef04p+7,
        0x1.aaae272e323c4p+7, 0x1.ac03683ab8d9fp+7, 0x1.ad58a9bd442a6p+7, 0x1.aeadebb325868p+7,
        0x1.b0032e19bde27p+7, 0x1.b15870ee7d72bp+7, 0x1.b2adb42ee352fp+7, 0x1.b402f7d87d302p+7,
        0x1.b5583be8e6f2fp+7, 0x1.b6ad805dca6d7p+7, 0x1.b802c534df09fp+7, 0x1.b9580a6be97c3p+7,
        0x1.baad5000bb738p+7, 0x1.bc0295f1334ffp+7, 0x1.bd57dc3b3bd7ep+7, 0x1.bead22dccbf00p+7,
        0x1.c00269d3e6554p+7, 0x1.c157b11e99573p+7, 0x1.c2acf8bafe954p+7, 0x1.c40240a73abc9p+7,
        0x1.c55788e17d47cp+7, 0x1.c6acd16800405p+7, 0x1.c8021a390800dp+7, 0x1.c9576352e2f90p+7,
        0x1.caacacb3e9736p+7, 0x1.cc01f65a7d5b7p+7, 0x1.cd5740450a05fp+7, 0x1.ceac8a7203fa3p+7,
        0x1.d001d4dfe8bc5p+7, 0x1.d1571f8d3e990p+7, 0x1.d2ac6a789472ap+7, 0x1.d401b5a0818f0p+7,
        0x1.d5570103a5670p+7}, {
        0x0.0p+0, 0x0.0p+0, 0x1.f07c1f07c42ddp-7, 0x1.73cca8b1276fbp-5,
        0x1.733c8e24f9010p-4, 0x1.34e550d581bf4p-3, 0x1.cea4a94a73484p-3, 0x1.435c6ddc54396p-2,
        0x1.ae7f4e54bbad6p-2, 0x1.1454b26e0d8b1p-1, 0x1.58e49a589b658p-1, 0x1.a4e6aab612315p-1,
        0x1.f85239734defep-1, 0x1.298f5349195afp+0, 0x1.5aa1ae0ef91c3p+0, 0x1.8f5be70fae3cep+0,
        0x1.c7b9bd48a0a04p+0, 0x1.01db7a553721ap+1, 0x1.21a7ab0994598p+1, 0x1.433f57a44cb9bp+1,
        0x1.66a0697946265p+1, 0x1.8bc8cc4a90b2ep+1, 0x1.b2b66e45925a2p+1, 0x1.db67400036890p+1,
        0x1.02ec9a3b10472p+2, 0x1.19052082f0ce5p+2, 0x1.2ffc2eb718d0fp+2, 0x1.47d0c1e595fccp+2,
        0x1.6081d849c7576p+2, 0x1.7a0e714afeb01p+2, 0x1.94758d7b23871p+2, 0x1.afb62e9557bf1p+2,
        0x1.cbcf577c9d8a1p+2, 0x1.e8c00c3a7f5a4p+2, 0x1.0343a8fedc792p+3, 0x1.1292178c7132cp+3,
        0x1.224ad5808e298p+3, 0x1.326d67265ef7cp+3, 0x1.42f9515902389p+3, 0x1.53ee1982e2035p+3,
        0x1.654b459d0d35cp+3, 0x1.77105c2e916b9p+3, 0x1.893ce44bd5cd2p+3, 0x1.9bd06595f692dp+3,
        0x1.aeca683a21493p+3, 0x1.c22a74f0f1cf3p+3, 0x1.d5f014fdd0210p+3, 0x1.ea1ad22e4eca5p+3,
        0x1.feaa36d98a2b6p+3, 0x1.09cee6efc4303p+4, 0x1.147a91544cfeap+4, 0x1.1f57e0925db58p+4,
        0x1.2a669ae57b899p+4, 0x1.35a686cc6417fp+4, 0x1.41176b08bf2d0p+4, 0x1.4cb90e9ed0edap+4,
        0x1.588b38d52c4a9p+4, 0x1.648db13465e22p+4, 0x1.70c03f86c72b0p+4, 0x1.7d22abd801fe0p+4,
        0x1.89b4be74e481ap+4, 0x1.96763feb0d5b5p+4, 0x1.a366f908a04d2p+4, 0x1.b086b2dbfb1e0p+4,
        0x1.bdd536b36ad84p+4, 0x1.cb524e1ce1716p+4, 0x1.d8fdc2e5aba9dp+4, 0x1.e6d75f1a27647p+4,
        0x1.f4deed057a37ep+4, 0x1.018a1b98a433fp+5, 0x1.08bb8432b6132p+5, 0x1.100395d3d69aap+5,
        0x1.1762361dbc6a8p+5, 0x1.1ed74ad0cd03dp+5, 0x1.2662b9cbf9138p+5, 0x1.2e04690c98e9dp+5,
        0x1.35bc3eae49170p+5, 0x1.3d8a20eac7354p+5, 0x1.456df619ced60p+5, 0x1.4d67a4b0f69fbp+5,
        0x1.557713438d8cap+5, 0x1.5d9c288278594p+5, 0x1.65d6cb3c0f138p+5, 0x1.6e26e25bfad5dp+5,
        0x1.768c54eb13af0p+5, 0x1.7f070a0f3ea6bp+5, 0x1.8796e90b4bf0ep+5, 0x1.903bd93ed54a8p+5,
        0x1.98f5c2261c787p+5, 0x1.a1c48b59e9ea8p+5, 0x1.aaa81c8f6b90ep+5, 0x1.b3a05d9813cdep+5,
        0x1.bcad366178900p+5, 0x1.c5ce8ef53295dp+5, 0x1.cf044f78bcd3ep+5, 0x1.d84e602d5404bp+5,
        0x1.e1aca96fd65a4p+5, 0x1.eb1f13b8a357ap+5, 0x1.f4a5879b7bcfap+5, 0x1.fe3fedc762093p+5,
        0x1.03f717833d035p+6, 0x1.08d81a1ef4fabp+6, 0x1.0dc2f336dd5e4p+6, 0x1.12b797585c5ccp+6,
        0x1.17b5fb1e2a4b0p+6, 0x1.1cbe133042269p+6, 0x1.21cfd443d2240p+6, 0x1.26eb331b2c565p+6,
        0x1.2c102485b7621p+6, 0x1.313e9d5fdf470p+6, 0x1.367692930636dp+6, 0x1.3bb7f9157581fp+6,
        0x1.4102c5ea4e928p+6, 0x1.4656ee217bfc2p+6, 0x1.4bb466d7a298fp+6, 0x1.511b253612bb4p+6,
        0x1.568b1e72b96f2p+6, 0x1.5c0447d011cdbp+6, 0x1.6186969d16607p+6, 0x1.6712003532970p+6,
        0x1.6ca67a00344cep+6, 0x1.7243f9723d5f1p+6, 0x1.77ea740bb555dp+6, 0x1.7d99df593b1aap+6,
        0x1.835230f396c39p+6, 0x1.89135e7fab696p+6, 0x1.8edd5dae6914cp+6, 0x1.94b0243cbeb63p+6,
        0x1.9a8ba7f38c328p+6, 0x1.a06fdea7947b3p+6, 0x1.a65cbe396fbbcp+6, 0x1.ac523c957d93bp+6,
        0x1.b2504fb3d761dp+6, 0x1.b856ed98429f1p+6, 0x1.be660c522349cp+6, 0x1.c47da1fc6e607p+6,
        0x1.ca9da4bd9c6b7p+6, 0x1.d0c60ac79c18fp+6, 0x1.d6f6ca57c4e3dp+6, 0x1.dd2fd9b6c9d03p+6,
        0x1.e3712f38ac31cp+6, 0x1.e9bac13cae838p+6, 0x1.f00c862d47520p+6, 0x1.f6667480142f8p+6,
        0x1.fcc882b5ccb93p+6, 0x1.019953ad1ad79p+7, 0x1.04d26c820a0a9p+7, 0x1.080f872a9033bp+7,
        0x1.0b509efcfced9p+7, 0x1.0e95af550cd86p+7, 0x1.11deb393e348bp+7, 0x1.152ba72003ff1p+7,
        0x1.187c85654ce63p+7, 0x1.1bd149d4efd89p+7, 0x1.1f29efe56c6ddp+7, 0x1.2286731289cd9p+7,
        0x1.25e6cedd508bbp+7, 0x1.294afecc048b3p+7, 0x1.2cb2fe6a1ee61p+7, 0x1.301ec94847dfap+7,
        0x1.338e5afc50db1p+7, 0x1.3701af212e5b0p+7, 0x1.3a78c156f2061p+7, 0x1.3df38d42c4b48p+7,
        0x1.41720e8ee0838p+7, 0x1.44f440ea8aefcp+7, 0x1.487a200a0ef57p+7, 0x1.4c03a7a6b73b0p+7,
        0x1.4f90d37ec83d9p+7, 0x1.53219f557a87bp+7, 0x1.56b606f2f4ee4p+7, 0x1.5a4e062446d2dp+7,
        0x1.5de998bb626dcp+7, 0x1.6188ba8f171eep+7, 0x1.652b677b0bc47p+7, 0x1.68d19b5fb9189p+7,
        0x1.6c7b522264159p+7, 0x1.702887ad18618p+7, 0x1.73d937eea2be5p+7, 0x1.778d5eda8b81cp+7,
        0x1.7b44f86911143p+7, 0x1.7f0000972274ap+7, 0x1.82be736659c40p+7, 0x1.86804cdcf6d61p+7,
        0x1.8a458905d9c98p+7, 0x1.8e0e23f07da56p+7, 0x1.91da19b0f2fd5p+7, 0x1.95a9665fda9cfp+7,
        0x1.997c061a60369p+7, 0x1.9d51f502351bep+7, 0x1.a12b2f3d8afa1p+7, 0x1.a507b0f70e9c9p+7,
        0x1.a8e7765de2b73p+7, 0x1.acca7ba59ab3ep+7, 0x1.b0b0bd063589dp+7, 0x1.b49a36bc18979p+7,
        0x1.b886e5080a847p+7, 0x1.bc76c42f2e278p+7, 0x1.c069d07afd75bp+7, 0x1.c460063944743p+7,
        0x1.c85961bc1c30ep+7, 0x1.cc55df59e5c27p+7, 0x1.d0557b6d454c1p+7, 0x1.d45832551d07cp+7,
        0x1.d85e007488574p+7, 0x1.dc66e232d6d95p+7, 0x1.e072d3fb87854p+7, 0x1.e481d23e43cd3p+7,
        0x1.e893d96edac44p+7, 0x1.eca8e6053c4b0p+7, 0x1.f0c0f47d7442dp+7, 0x1.f4dc0157a5c4ep+7,
        0x1.f8fa0918065fap+7, 0x1.fd1b0846d959bp+7, 0x1.009f7db8357d9p+8, 0x1.02b2ef9285ed2p+8,
        0x1.04c7d7fc8617fp+8, 0x1.06de35425b97cp+8, 0x1.08f605b227335p+8, 0x1.0b0f479c028fdp+8,
        0x1.0d29f951fde57p+8, 0x1.0f4619281db61p+8, 0x1.1163a5745888dp+8, 0x1.13829c8e94a40p+8,
        0x1.15a2fcd0a5cecp+8, 0x1.17c4c4964b10fp+8, 0x1.19e7f23d2c786p+8, 0x1.1c0c8424d8dfcp+8,
        0x1.1e3278aec3b82p+8, 0x1.2059ce3e42d45p+8, 0x1.228283388c38fp+8, 0x1.24ac9604b3eccp+8,
        0x1.26d8050ba9cc3p+8, 0x1.2904ceb83760ap+8, 0x1.2b32f176fdb82p+8, 0x1.2d626bb67341ep+8,
        0x1.2f933be6e1aaap+8, 0x1.31c5607a63be7p+8, 0x1.33f8d7e4e34a8p+8, 0x1.362da09c17024p+8,
        0x1.3863b91780686p+8, 0x1.3a9b1fd069b70p+8, 0x1.3cd3d341e3cdep+8, 0x1.3f0dd1e8c4205p+8,
        0x1.41491a43a2a71p+8, 0x1.4385aad2d7d43p+8, 0x1.45c382187a886p+8, 0x1.48029e985e0c6p+8,
        0x1.4a42fed8100b9p+8, 0x1.4c84a15ed6910p+8, 0x1.4ec784b5ae070p+8, 0x1.510ba76747395p+8,
        0x1.5351080005595p+8, 0x1.5597a50dfc040p+8, 0x1.57df7d20ed4b6p+8, 0x1.5a288eca47c14p+8,
        0x1.5c72d89d24853p+8, 0x1.5ebe592e4552fp+8, 0x1.610b0f141295ap+8, 0x1.6358f8e6997b5p+8,
        0x1.65a8153f8a0b6p+8, 0x1.67f862ba353f6p+8, 0x1.6a49dff38b1d6p+8, 0x1.6c9c8b8a18d58p+8,
        0x1.6ef0641e06e10p+8, 0x1.7145685117236p+8, 0x1.739b96c6a30ecp+8, 0x1.75f2ee2399c87p+8,
        0x1.784b6d0e7e521p+8, 0x1.7aa5122f65b2ap+8, 0x1.7cffdc2ff5236p+8, 0x1.7f5bc9bb603e2p+8,
        0x1.81b8d97e672ccp+8, 0x1.84170a2754dd0p+8, 0x1.86765a65fd348p+8, 0x1.88d6c8ebbb474p+8,
        0x1.8b38546b6f910p+8, 0x1.8d9afb997e2fcp+8, 0x1.8ffebd2bcd20ep+8, 0x1.926397d9c27fcp+8,
        0x1.94c98a5c42c74p+8, 0x1.9730936daf14cp+8, 0x1.9998b1c9e36cdp+8, 0x1.9c01e42e35027p+8,
        0x1.9e6c295970807p+8, 0x1.a0d7800bd8548p+8, 0x1.a343e70722fbcp+8, 0x1.a5b15d0e79523p+8,
        0x1.a81fe0e674e42p+8, 0x1.aa8f71551e401p+8, 0x1.ad000d21eb4d0p+8, 0x1.af71b315bda01p+8,
        0x1.b1e461fae0d61p+8, 0x1.b458189d08edep+8, 0x1.b6ccd5c950a57p+8, 0x1.b942984e37d7ep+8,
        0x1.bbb95efba1de6p+8, 0x1.be3128a2d3f1dp+8, 0x1.c0a9f41673904p+8, 0x1.c323c02a84e15p+8,
        0x1.c59e8bb4691ffp+8, 0x1.c81a558add02ep+8, 0x1.ca971c85f7288p+8, 0x1.cd14df7f2684fp+8,
        0x1.cf939d5130d13p+8, 0x1.d21354d830fbcp+8, 0x1.d49404f1959bep+8, 0x1.d715ac7c1f66ep+8,
        0x1.d9984a57dfa59p+8, 0x1.dc1bdd6636adep+8, 0x1.dea06489d25b9p+8, 0x1.e125dea6ac8dap+8,
        0x1.e3ac4aa209a2bp+8, 0x1.e633a76276f93p+8, 0x1.e8bbf3cfc9705p+8, 0x1.eb452ed31beaep+8,
        0x1.edcf5756cdd3bp+8, 0x1.f05a6c4681a49p+8, 0x1.f2e66c8f1b6ddp+8, 0x1.f573571ebf60ap+8,
        0x1.f8012ae4d0599p+8, 0x1.fa8fe6d1ee6efp+8, 0x1.fd1f89d7f57edp+8, 0x1.ffb012e9fbbfcp+8,
        0x1.0120c07e2829bp+9, 0x1.0269e9823ced0p+9, 0x1.03b383fc9a840p+9, 0x1.04fd8f69399e9p+9,
        0x1.06480b44ac8ecp+9, 0x1.0792f70c1e95dp+9, 0x1.08de523d5332ap+9, 0x1.0a2a1c56a5701p+9,
        0x1.0b7654d707350p+9, 0x1.0cc2fb3e0094ap+9, 0x1.0e100f0baf1f8p+9, 0x1.0f5d8fc0c5365p+9,
        0x1.10ab7cde895bbp+9, 0x1.11f9d5e6d588bp+9, 0x1.13489a5c1680bp+9, 0x1.1497c9c14b26fp+9,
        0x1.15e7639a03d41p+9, 0x1.1737676a61ad8p+9, 0x1.1887d4b715fc4p+9, 0x1.19d8ab056185ap+9,
        0x1.1b29e9db13e40p+9, 0x1.1c7b90be8ae10p+9, 0x1.1dcd9f36b1cfcp+9, 0x1.1f2014cb00e80p+9,
        0x1.2072f1037ca2cp+9, 0x1.21c63368b5166p+9, 0x1.2319db83c554bp+9, 0x1.246de8de52c90p+9,
        0x1.25c25b028c974p+9, 0x1.2717317b2afb7p+9, 0x1.286c6bd36eaa7p+9, 0x1.29c2099720330p+9,
        0x1.2b180a528f5fbp+9, 0x1.2c6e6d929299ep+9, 0x1.2dc532e4864c7p+9, 0x1.2f1c59d64c483p+9,
        0x1.3073e1f64b28ep+9, 0x1.31cbcad36dba2p+9, 0x1.332413fd225e2p+9, 0x1.347cbd035a743p+9,
        0x1.35d5c57689c08p+9, 0x1.372f2ce7a5d43p+9, 0x1.3888f2e825768p+9, 0x1.39e3170a000e7p+9,
        0x1.3b3d98dfad0cdp+9, 0x1.3c9877fc2357ap+9, 0x1.3df3b3f2d8b57p+9, 0x1.3f4f4c57c13a1p+9,
        0x1.40ab40bf4eb33p+9, 0x1.420790be70168p+9, 0x1.43643bea90efap+9, 0x1.44c141d998cf8p+9,
        0x1.461ea221eabc0p+9, 0x1.477c5c5a649ffp+9, 0x1.48da701a5ebccp+9, 0x1.4a38dcf9ab1b5p+9,
        0x1.4b97a29094feep+9, 0x1.4cf6c077e057bp+9, 0x1.4e563648c936cp+9, 0x1.4fb6039d0341bp+9,
        0x1.5116280eb9285p+9, 0x1.5276a3388c190p+9, 0x1.53d774b59337ep+9, 0x1.55389c215b152p+9,
        0x1.569a1917e5243p+9, 0x1.57fbeb35a733fp+9, 0x1.595e12178ae73p+9, 0x1.5ac08d5aed2e6p+9,
        0x1.5c235c9d9dc06p+9, 0x1.5d867f7dde960p+9, 0x1.5ee9f59a63648p+9, 0x1.604dbe9251193p+9,
        0x1.61b1da053d560p+9}};
// FastReach._directions(q): key combo world x, z (unit) and push; camera right / forward (reach.key_push)
const double DIRS[4][8][3] = {{{0x0.0p+0, 0x1.0000000000000p+0, 0x1.0000000000000p+0}, {0x0.0p+0, -0x1.0000000000000p+0, 0x1.0000000000000p+0}, {-0x1.0000000000000p+0, 0x0.0p+0, 0x1.0000000000000p+0}, {0x1.0000000000000p+0, 0x0.0p+0, 0x1.0000000000000p+0}, {-0x1.6a09e667f3bccp-1, 0x1.6a09e667f3bccp-1, 0x1.6a09e667f3bcdp+0}, {0x1.6a09e667f3bccp-1, 0x1.6a09e667f3bccp-1, 0x1.6a09e667f3bcdp+0}, {-0x1.6a09e667f3bccp-1, -0x1.6a09e667f3bccp-1, 0x1.6a09e667f3bcdp+0}, {0x1.6a09e667f3bccp-1, -0x1.6a09e667f3bccp-1, 0x1.6a09e667f3bcdp+0}}, {{0x1.0000000000000p+0, 0x1.1a62633145c07p-54, 0x1.0000000000000p+0}, {-0x1.0000000000000p+0, -0x1.1a62633145c07p-54, 0x1.0000000000000p+0}, {-0x1.1a62633145c07p-54, 0x1.0000000000000p+0, 0x1.0000000000000p+0}, {0x1.1a62633145c07p-54, -0x1.0000000000000p+0, 0x1.0000000000000p+0}, {0x1.6a09e667f3bccp-1, 0x1.6a09e667f3bcdp-1, 0x1.6a09e667f3bccp+0}, {0x1.6a09e667f3bcdp-1, -0x1.6a09e667f3bccp-1, 0x1.6a09e667f3bccp+0}, {-0x1.6a09e667f3bcdp-1, 0x1.6a09e667f3bccp-1, 0x1.6a09e667f3bccp+0}, {-0x1.6a09e667f3bccp-1, -0x1.6a09e667f3bcdp-1, 0x1.6a09e667f3bccp+0}}, {{0x1.1a62633145c07p-53, -0x1.0000000000000p+0, 0x1.0000000000000p+0}, {-0x1.1a62633145c07p-53, 0x1.0000000000000p+0, 0x1.0000000000000p+0}, {0x1.0000000000000p+0, 0x1.1a62633145c07p-53, 0x1.0000000000000p+0}, {-0x1.0000000000000p+0, -0x1.1a62633145c07p-53, 0x1.0000000000000p+0}, {0x1.6a09e667f3bcep-1, -0x1.6a09e667f3bcbp-1, 0x1.6a09e667f3bcdp+0}, {-0x1.6a09e667f3bcbp-1, -0x1.6a09e667f3bcep-1, 0x1.6a09e667f3bcdp+0}, {0x1.6a09e667f3bcbp-1, 0x1.6a09e667f3bcep-1, 0x1.6a09e667f3bcdp+0}, {-0x1.6a09e667f3bcep-1, 0x1.6a09e667f3bcbp-1, 0x1.6a09e667f3bcdp+0}}, {{-0x1.0000000000000p+0, -0x1.a79394c9e8a0ap-53, 0x1.0000000000000p+0}, {0x1.0000000000000p+0, 0x1.a79394c9e8a0ap-53, 0x1.0000000000000p+0}, {0x1.a79394c9e8a0ap-53, -0x1.0000000000000p+0, 0x1.0000000000000p+0}, {-0x1.a79394c9e8a0ap-53, 0x1.0000000000000p+0, 0x1.0000000000000p+0}, {-0x1.6a09e667f3bcbp-1, -0x1.6a09e667f3bcep-1, 0x1.6a09e667f3bcdp+0}, {-0x1.6a09e667f3bcep-1, 0x1.6a09e667f3bcbp-1, 0x1.6a09e667f3bcdp+0}, {0x1.6a09e667f3bcep-1, -0x1.6a09e667f3bcbp-1, 0x1.6a09e667f3bcdp+0}, {0x1.6a09e667f3bcbp-1, 0x1.6a09e667f3bcep-1, 0x1.6a09e667f3bcdp+0}}};
const double RIGHTV[4][2] = {{0x1.0000000000000p+0, 0x0.0p+0}, {0x1.1a62633145c07p-54, -0x1.0000000000000p+0}, {-0x1.0000000000000p+0, -0x1.1a62633145c07p-53}, {-0x1.a79394c9e8a0ap-53, 0x1.0000000000000p+0}};
const double FWDV[4][2] = {{0x0.0p+0, 0x1.0000000000000p+0}, {0x1.0000000000000p+0, 0x1.1a62633145c07p-54}, {0x1.1a62633145c07p-53, -0x1.0000000000000p+0}, {-0x1.0000000000000p+0, -0x1.a79394c9e8a0ap-53}};
// _perceive's side probes: math.cos / math.sin of q * (pi / 4)
const double PROBE_COS[8] = {
    0x1.0000000000000p+0, 0x1.6a09e667f3bcdp-1, 0x1.1a62633145c07p-54, -0x1.6a09e667f3bccp-1,
    -0x1.0000000000000p+0, -0x1.6a09e667f3bcep-1, -0x1.a79394c9e8a0ap-53, 0x1.6a09e667f3bcbp-1};
const double PROBE_SIN[8] = {
    0x0.0p+0, 0x1.6a09e667f3bcdp-1, 0x1.0000000000000p+0, 0x1.6a09e667f3bcdp-1,
    0x1.1a62633145c07p-53, -0x1.6a09e667f3bccp-1, -0x1.0000000000000p+0, -0x1.6a09e667f3bcep-1};
// END GENERATED TABLES

}  // namespace

// ------------------------------------------------------------------------------------------------ the fields
struct Fields::Impl {
    std::vector<int32_t> tiles;
    int64_t tiles0 = 0, tiles1 = 0;
    std::vector<float> Hh, P1, P2, Pd, Rr, Rl;
    std::vector<int64_t> Poff, Pdim;
    int64_t npdim = 0;
    std::vector<int8_t> Rn;
    std::vector<uint8_t> Rf;
    std::vector<int32_t> bidx;
    int64_t b0 = 0, b1 = 0, b2 = 0;
    std::vector<int16_t> bricks;
    std::vector<double> seg;
    int64_t nseg = 0;
    std::vector<int32_t> rstart, ritems;
    std::vector<double> boxes, fans;
    int64_t nboxes = 0, nfans = 0;
    double ox = 0, oz = 0, sx0 = 0, sy0 = 0, sz0 = 0, void_y = 0;
    int64_t ntz = 0;

    // element accessors (C order)
    inline double hh(int64_t t, int64_t i, int64_t j, int k) const { return (double)Hh[((t * TL + i) * TL + j) * KH + k]; }
    inline int64_t tile(int64_t i, int64_t j) const { return tiles[i * tiles1 + j]; }
    inline const double *segp(int64_t s) const { return &seg[s * 7]; }
    inline const double *fanp(int64_t f) const { return &fans[f * FAN_COLS]; }
    inline const double *boxp(int64_t b) const { return &boxes[b * 6]; }
};

namespace {

template <class T>
std::vector<T> typed(const std::map<std::string, NpyArray> &z, const char *name, const char *descr,
                     std::vector<int64_t> *shape_out, size_t ndim) {
    auto it = z.find(name);
    if (it == z.end()) throw Error(std::string("fastreach fields: missing array ") + name);
    const NpyArray &a = it->second;
    if (a.descr != descr) throw Error(std::string("fastreach fields: ") + name + " has dtype " + a.descr);
    if (a.fortran) throw Error(std::string("fastreach fields: ") + name + " is Fortran order");
    if (a.shape.size() != ndim) throw Error(std::string("fastreach fields: ") + name + " has the wrong rank");
    if (a.data.size() != size_t(a.count()) * sizeof(T)) throw Error(std::string("fastreach fields: ") + name + " size");
    std::vector<T> v(size_t(a.count()));
    if (!v.empty()) std::memcpy(v.data(), a.data.data(), v.size() * sizeof(T));
    if (shape_out) *shape_out = a.shape;
    return v;
}

void need_shape(const std::vector<int64_t> &s, std::initializer_list<int64_t> tail, const char *name) {
    size_t k = s.size() - tail.size();
    size_t i = 0;
    for (int64_t d : tail) {
        if (s[k + i] != d) throw Error(std::string("fastreach fields: ") + name + " has the wrong shape");
        ++i;
    }
}


// The members the kernel reads (load_npz would refuse the file: tile_ij is stored in Fortran order).
std::map<std::string, NpyArray> load_members(const std::string &path) {
    static const char *names[] = {"info", "tiles", "Hh", "P1", "P2", "Pd", "Poff", "Pdim", "Rr", "Rl", "Rn", "Rf",
                                  "bidx", "bricks", "seg", "rstart", "ritems", "boxes", "fans", "meta"};
    Bytes data;
    if (!read_file(path, data)) throw Error("fastreach fields: cannot read " + path);
    mz_zip_archive zip;
    std::memset(&zip, 0, sizeof zip);
    if (!mz_zip_reader_init_mem(&zip, data.data(), data.size(), 0)) throw Error("fastreach fields: not a zip file");
    std::map<std::string, NpyArray> out;
    try {
        for (const char *nm : names) {
            int i = mz_zip_reader_locate_file(&zip, (std::string(nm) + ".npy").c_str(), nullptr, 0);
            if (i < 0) continue;
            size_t sz = 0;
            void *p = mz_zip_reader_extract_to_heap(&zip, mz_uint(i), &sz, 0);
            if (!p) throw Error(std::string("fastreach fields: cannot extract ") + nm);
            Bytes b(static_cast<const char *>(p), sz);
            mz_free(p);
            out[nm] = parse_npy(b);
        }
    } catch (...) {
        mz_zip_reader_end(&zip);
        throw;
    }
    mz_zip_reader_end(&zip);
    return out;
}

}  // namespace

std::shared_ptr<const Fields> Fields::load(const std::string &npz_path) {
    auto z = load_members(npz_path);
    auto im = std::make_shared<Impl>();
    Impl &f = *im;
    std::vector<int64_t> sh, nt_sh;
    f.tiles = typed<int32_t>(z, "tiles", "<i4", &sh, 2);
    f.tiles0 = sh[0], f.tiles1 = sh[1];
    f.Hh = typed<float>(z, "Hh", "<f4", &nt_sh, 4);
    need_shape(nt_sh, {TL, TL, KH}, "Hh");
    const int64_t nt = nt_sh[0];
    f.P1 = typed<float>(z, "P1", "<f4", &sh, 4);
    if (sh[0] != nt) throw Error("fastreach fields: P1 tiles");
    need_shape(sh, {4, 4, 2}, "P1");
    f.P2 = typed<float>(z, "P2", "<f4", &sh, 4);
    if (sh[0] != nt) throw Error("fastreach fields: P2 tiles");
    need_shape(sh, {2, 2, 2}, "P2");
    f.Pd = typed<float>(z, "Pd", "<f4", &sh, 2);
    need_shape(sh, {2}, "Pd");
    f.Poff = typed<int64_t>(z, "Poff", "<i8", &sh, 1);
    f.Pdim = typed<int64_t>(z, "Pdim", "<i8", &sh, 2);
    need_shape(sh, {2}, "Pdim");
    f.npdim = sh[0];
    if (f.npdim < 1 || (int64_t)f.Poff.size() < f.npdim) throw Error("fastreach fields: pyramid levels");
    f.Rr = typed<float>(z, "Rr", "<f4", &sh, 4);
    if (sh[0] != nt) throw Error("fastreach fields: Rr tiles");
    need_shape(sh, {TL, TL, KR}, "Rr");
    f.Rl = typed<float>(z, "Rl", "<f4", &sh, 4);
    if (sh[0] != nt) throw Error("fastreach fields: Rl tiles");
    f.Rn = typed<int8_t>(z, "Rn", "|i1", &sh, 5);
    if (sh[0] != nt) throw Error("fastreach fields: Rn tiles");
    need_shape(sh, {TL, TL, KR, 2}, "Rn");
    f.Rf = typed<uint8_t>(z, "Rf", "|u1", &sh, 4);
    if (sh[0] != nt) throw Error("fastreach fields: Rf tiles");
    f.bidx = typed<int32_t>(z, "bidx", "<i4", &sh, 3);
    f.b0 = sh[0], f.b1 = sh[1], f.b2 = sh[2];
    f.bricks = typed<int16_t>(z, "bricks", "<i2", &sh, 4);
    need_shape(sh, {BRICK + 1, BRICK + 1, BRICK + 1}, "bricks");
    f.seg = typed<double>(z, "seg", "<f8", &sh, 2);
    need_shape(sh, {7}, "seg");
    f.nseg = sh[0];
    f.rstart = typed<int32_t>(z, "rstart", "<i4", &sh, 1);
    f.ritems = typed<int32_t>(z, "ritems", "<i4", &sh, 1);
    f.boxes = typed<double>(z, "boxes", "<f8", &sh, 2);
    need_shape(sh, {6}, "boxes");
    f.nboxes = sh[0];
    if (z.count("fans")) {
        f.fans = typed<double>(z, "fans", "<f8", &sh, 2);
        if (f.fans.size() % FAN_COLS) throw Error("fastreach fields: fans");
        f.nfans = int64_t(f.fans.size() / FAN_COLS);
    }
    std::vector<double> meta = typed<double>(z, "meta", "<f8", &sh, 1);
    if (meta.size() != 8) throw Error("fastreach fields: meta");
    if (meta[7] != double(BUILD_VERSION))
        throw Error("fastreach fields: BUILD_VERSION " + std::to_string(meta[7]) + ", the kernel is " +
                    std::to_string(BUILD_VERSION));
    f.ox = meta[0], f.oz = meta[1], f.sx0 = meta[2], f.sy0 = meta[3], f.sz0 = meta[4], f.void_y = meta[5];
    f.ntz = f.tiles1;
    if (f.ntz < 1) throw Error("fastreach fields: tiles");
    for (int32_t t : f.tiles)
        if (t >= nt) throw Error("fastreach fields: tile index out of range");
    for (int32_t b : f.bidx)
        if (b >= 0 && int64_t(b) * 729 >= int64_t(f.bricks.size())) throw Error("fastreach fields: brick index");
    for (int32_t s : f.ritems)
        if (s < 0 || s >= f.nseg) throw Error("fastreach fields: rail item");
    for (size_t i = 0; i < f.rstart.size(); ++i)
        if (f.rstart[i] < 0 || f.rstart[i] > (int32_t)f.ritems.size() || (i && f.rstart[i] < f.rstart[i - 1]))
            throw Error("fastreach fields: rail CSR");
    auto out = std::make_shared<Fields>();
    auto it = z.find("info");
    if (it != z.end()) {
        Json info = Json::parse(it->second.unicode());
        out->key = info.get_string("key", "");
    }
    out->impl = im;
    return out;
}

// ------------------------------------------------------------------------------------------------ workspace
struct Workspace::Impl {
    std::vector<double> yy, xf, zf, D, fins, yl, ya;
    std::vector<int64_t> stack;  // (rows, 3)
    int64_t stack_rows = 0, stack_max = 0;
    std::vector<int64_t> stamp;
    int64_t stamp_id = 0;
    double res[16], res0[16], mres[9];
    double dbg_s[NDBG], pdd_s[8 * NPDD];
    Impl()
        : yy(NPSI + 1), xf(NPSI + 1), zf(NPSI + 1), D(NPSI + 1), fins(NPSI + 1, 0.0), yl(NPSI + 1, 0.0),
          ya(NPSI + 1, 0.0), stack(512 * 3, 0), stack_rows(512) {}
    inline void push(int64_t &sp, int64_t lev, int64_t a, int64_t b) {
        if (sp >= stack_rows) {
            stack_rows *= 2;
            stack.resize(size_t(stack_rows) * 3);
        }
        stack[sp * 3 + 0] = lev, stack[sp * 3 + 1] = a, stack[sp * 3 + 2] = b;
        ++sp;
        if (sp > stack_max) stack_max = sp;
    }
};

Workspace::Workspace() : impl(new Impl()) {}
Workspace::~Workspace() = default;

namespace {

using F = Fields::Impl;
using W = Workspace::Impl;

struct Ball {
    const double *S1, *S2;
    double lam_a, a_air, lam_g, a_g, r, fdv;
};

// ======================================================================================== scalar kernels

double mount_speed(double h, double r) {
    if (h <= MOUNT_FREE) return 0.0;
    if (h >= MOUNT_CAP || h >= r) return INF;
    double f = 1.0 - 5.0 * h / (7.0 * r);
    if (f < 1e-6) f = 1e-6;
    return std::sqrt(GRAVITY * h / (0.7 * f * f));
}

inline int64_t cidx(double x, double o) { return ifloor((x - o) / RES + 0.5); }

inline int64_t tile_of(const F &f, int64_t i, int64_t j) {
    if (i < 0 || j < 0) return -1;
    int64_t ti = floordiv(i, TL), tj = floordiv(j, TL);
    if (ti >= f.tiles0 || tj >= f.tiles1) return -1;
    return f.tile(ti, tj);
}

double h_top(const F &f, double x, double z, double lo, double hi) {
    int64_t i = cidx(x, f.ox), j = cidx(z, f.oz);
    int64_t t = tile_of(f, i, j);
    if (t < 0) return NaN;
    int64_t li = pymod(i, TL), lj = pymod(j, TL);
    for (int k = 0; k < KH; ++k) {
        double h = f.hh(t, li, lj, k);
        if (h != h) break;
        if (lo <= h && h <= hi) return h;
    }
    return NaN;
}

struct AxisNear {
    double d, x, y, z, r;
};

AxisNear rail_axis_near(const F &f, double x, double y, double z) {
    int64_t ti = ifloor((x - f.ox + 0.5 * RES) / (RES * TL));
    int64_t tj = ifloor((z - f.oz + 0.5 * RES) / (RES * TL));
    AxisNear b{INF, 0.0, 0.0, 0.0, 0.0};
    if (ti < 0 || tj < 0 || tj >= f.ntz || (ti * f.ntz + tj + 1) >= (int64_t)f.rstart.size()) return b;
    int64_t c = ti * f.ntz + tj;
    for (int64_t q = f.rstart[c]; q < f.rstart[c + 1]; ++q) {
        const double *s = f.segp(f.ritems[q]);
        double ax = s[0], ay = s[1], az = s[2];
        double dx = s[3] - ax, dy = s[4] - ay, dz = s[5] - az;
        double L2 = dx * dx + dy * dy + dz * dz;
        double t = 0.0;
        if (L2 > 1e-12) {
            t = ((x - ax) * dx + (y - ay) * dy + (z - az) * dz) / L2;
            t = pmin(pmax(t, 0.0), 1.0);
        }
        double px = ax + t * dx, py = ay + t * dy, pz = az + t * dz;
        double ex = x - px, ey = y - py, ez = z - pz;
        double d = std::sqrt(ex * ex + ey * ey + ez * ez);
        if (d < b.d) b = AxisNear{d, px, py, pz, s[6]};
    }
    return b;
}

double rail_top_near(const F &f, double x, double z, double rad, double lo, double hi) {
    double sx = x - f.ox + 0.5 * RES;
    double sz = z - f.oz + 0.5 * RES;
    int64_t ti = ifloor(sx / (RES * TL)), tj = ifloor(sz / (RES * TL));
    double best = NaN;
    if (ti < 0 || tj < 0 || tj >= f.ntz || (ti * f.ntz + tj + 1) >= (int64_t)f.rstart.size()) return best;
    int64_t c = ti * f.ntz + tj;
    for (int64_t q = f.rstart[c]; q < f.rstart[c + 1]; ++q) {
        const double *s = f.segp(f.ritems[q]);
        double ax = s[0], az = s[2];
        double dx = s[3] - ax, dz = s[5] - az;
        double L2 = dx * dx + dz * dz;
        double t = 0.0;
        if (L2 > 1e-12) {
            t = ((x - ax) * dx + (z - az) * dz) / L2;
            t = pmin(pmax(t, 0.0), 1.0);
        }
        double px = ax + t * dx, pz = az + t * dz;
        double ex = x - px, ez = z - pz;
        if (ex * ex + ez * ez <= rad * rad) {
            double top = s[1] + t * (s[4] - s[1]) + s[6];
            if (lo <= top && top <= hi && (best != best || top > best)) best = top;
        }
    }
    return best;
}

struct Vec4 {
    double d, x, y, z;
};

Vec4 rail_dist(const F &f, double x, double y, double z) {
    AxisNear a = rail_axis_near(f, x, y, z);
    if (a.d == INF) return Vec4{INF, 0.0, 1.0, 0.0};
    double nx = x - a.x, ny = y - a.y, nz = z - a.z;
    double ln = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (ln < 1e-9) return Vec4{a.d - a.r, 0.0, 1.0, 0.0};
    return Vec4{a.d - a.r, nx / ln, ny / ln, nz / ln};
}

Vec4 sdf(const F &f, double x, double y, double z) {
    double fx = (x - f.sx0) / VRES, fy = (y - f.sy0) / VRES, fz = (z - f.sz0) / VRES;
    int64_t ix = ifloor(fx), iy = ifloor(fy), iz = ifloor(fz);
    int64_t bx = floordiv(ix, BRICK), by = floordiv(iy, BRICK), bz = floordiv(iz, BRICK);
    if (bx < 0 || by < 0 || bz < 0 || bx >= f.b0 || by >= f.b1 || bz >= f.b2) return Vec4{SDF_BAND, 0.0, 0.0, 0.0};
    int64_t b = f.bidx[(bx * f.b1 + by) * f.b2 + bz];
    if (b < 0) return Vec4{SDF_BAND, 0.0, 0.0, 0.0};
    int64_t lx = ix - bx * BRICK, ly = iy - by * BRICK, lz = iz - bz * BRICK;
    double tx = fx - (double)ix, ty = fy - (double)iy, tz = fz - (double)iz;
    const int16_t *B = &f.bricks[size_t(b) * 729];
    auto at = [&](int64_t i, int64_t j, int64_t k) { return (double)B[(i * 9 + j) * 9 + k] / SDF_SCALE; };
    double c000 = at(lx, ly, lz);
    double c100 = at(lx + 1, ly, lz);
    double c010 = at(lx, ly + 1, lz);
    double c110 = at(lx + 1, ly + 1, lz);
    double c001 = at(lx, ly, lz + 1);
    double c101 = at(lx + 1, ly, lz + 1);
    double c011 = at(lx, ly + 1, lz + 1);
    double c111 = at(lx + 1, ly + 1, lz + 1);
    double c00 = c000 + (c100 - c000) * tx;
    double c10 = c010 + (c110 - c010) * tx;
    double c01 = c001 + (c101 - c001) * tx;
    double c11 = c011 + (c111 - c011) * tx;
    double c0 = c00 + (c10 - c00) * ty;
    double c1 = c01 + (c11 - c01) * ty;
    double d = c0 + (c1 - c0) * tz;
    double gx = (c100 - c000) * (1 - ty) * (1 - tz) + (c110 - c010) * ty * (1 - tz) + (c101 - c001) * (1 - ty) * tz +
                (c111 - c011) * ty * tz;
    double gy = (c010 - c000) * (1 - tx) * (1 - tz) + (c110 - c100) * tx * (1 - tz) + (c011 - c001) * (1 - tx) * tz +
                (c111 - c101) * tx * tz;
    double gz = c1 - c0;
    double ln = std::sqrt(gx * gx + gy * gy + gz * gz);
    if (ln > 1e-12) {
        gx /= ln;
        gy /= ln;
        gz /= ln;
    }
    return Vec4{d, gx, gy, gz};
}

Vec4 udist(const F &f, double x, double y, double z) {
    Vec4 s = sdf(f, x, y, z);
    if (s.d < 0.0) return Vec4{-s.d, -s.x, -s.y, -s.z};
    return s;
}

struct RRead {
    double rest, low, nx, ny, nz;
    int64_t fl;
};

inline RRead r_read(const F &f, int64_t t, int64_t li, int64_t lj, int k) {
    size_t o = size_t(((t * TL + li) * TL + lj) * KR + k);
    RRead r;
    r.rest = (double)f.Rr[o];
    r.low = (double)f.Rl[o];
    r.nx = (double)f.Rn[o * 2 + 0] / 127.0;
    r.nz = (double)f.Rn[o * 2 + 1] / 127.0;
    r.ny = std::sqrt(pmax(0.0, 1.0 - r.nx * r.nx - r.nz * r.nz));
    r.fl = int64_t(f.Rf[o]) & 3;
    return r;
}

struct Ground {
    double rest, nx, ny, nz;
    int64_t fl;
};

Ground r_ground(const F &f, double x, double z, double y, double tol) {
    int64_t i = cidx(x, f.ox), j = cidx(z, f.oz);
    int64_t t = tile_of(f, i, j);
    if (t < 0) return Ground{NaN, 0.0, 1.0, 0.0, -1};
    int64_t li = pymod(i, TL), lj = pymod(j, TL);
    for (int k = 0; k < KR; ++k) {
        RRead r = r_read(f, t, li, lj, k);
        if (r.rest != r.rest) break;
        if (std::fabs(r.rest - y) <= tol) return Ground{r.rest, r.nx, r.ny, r.nz, r.fl};
    }
    return Ground{NaN, 0.0, 1.0, 0.0, -1};
}

double r_below(const F &f, double x, double z, double y) {
    int64_t i = cidx(x, f.ox), j = cidx(z, f.oz);
    int64_t t = tile_of(f, i, j);
    if (t < 0) return NaN;
    int64_t li = pymod(i, TL), lj = pymod(j, TL);
    for (int k = 0; k < KR; ++k) {
        double rest = (double)f.Rr[size_t(((t * TL + li) * TL + lj) * KR + k)];
        if (rest != rest) break;
        if (rest <= y + GROUND_TOL) return rest;
    }
    return NaN;
}

struct Hold {
    double hold, nx, ny, nz;
    int64_t fl;
    double block;
    bool wall;
};

Hold r_hold(const F &f, double x, double z, double y, double drop_tol, double ux, double uz, double step) {
    int64_t i = cidx(x, f.ox), j = cidx(z, f.oz);
    int64_t t = tile_of(f, i, j);
    if (t < 0) return Hold{NaN, 0.0, 1.0, 0.0, -1, NaN, false};
    int64_t li = pymod(i, TL), lj = pymod(j, TL);
    double hold = NaN, hnx = 0.0, hny = 1.0, hnz = 0.0;
    int64_t hfl = -1;
    bool hwall = false;
    for (int k = 0; k < KR; ++k) {
        RRead r = r_read(f, t, li, lj, k);
        if (r.rest != r.rest) break;
        double allow;
        if (r.fl == 1)
            allow = UP_TOL + step * pmax(0.0, -(r.nx * ux + r.nz * uz)) / pmax(r.ny, 0.2);
        else if (r.fl == 2)
            allow = UP_TOL + 0.5 * step;
        else
            allow = UP_TOL;
        if (y - drop_tol <= r.rest && r.rest <= y + allow) {
            if (hold != hold || r.rest > hold) {
                hold = r.rest, hnx = r.nx, hny = r.ny, hnz = r.nz, hfl = r.fl;
                hwall = (int64_t(f.Rf[size_t(((t * TL + li) * TL + lj) * KR + k)]) & 4) != 0;
            }
        }
    }
    double ynew = hold == hold ? hold : y;
    double block = NaN;
    for (int k = 0; k < KR; ++k) {
        RRead r = r_read(f, t, li, lj, k);
        if (r.rest != r.rest) break;
        double allow;
        if (r.fl == 1)
            allow = UP_TOL + step * pmax(0.0, -(r.nx * ux + r.nz * uz)) / pmax(r.ny, 0.2);
        else if (r.fl == 2)
            allow = UP_TOL + 0.5 * step;
        else
            allow = UP_TOL;
        if (r.rest > ynew + allow && r.low < ynew - 1e-3) {
            if (block != block || r.rest < block) block = r.rest;
        }
    }
    return Hold{hold, hnx, hny, hnz, hfl, block, hwall};
}

double h_top_disc(const F &f, double x, double z, double rad, double lo, double hi) {
    int64_t m = iceil(rad / RES);
    int64_t i0 = cidx(x, f.ox), j0 = cidx(z, f.oz);
    double best = NaN;
    for (int64_t di = -m; di < m + 1; ++di) {
        for (int64_t dj = -m; dj < m + 1; ++dj) {
            if ((double)(di * di + dj * dj) * RES * RES > rad * rad + 1e-9) continue;
            int64_t t = tile_of(f, i0 + di, j0 + dj);
            if (t < 0) continue;
            int64_t li = pymod(i0 + di, TL), lj = pymod(j0 + dj, TL);
            for (int k = 0; k < KH; ++k) {
                double h = f.hh(t, li, lj, k);
                if (h != h) break;
                if (lo <= h && h <= hi && (best != best || h > best)) best = h;
            }
        }
    }
    double rt = rail_top_near(f, x, z, rad, lo, hi);
    if (rt == rt && (best != best || rt > best)) best = rt;
    return best;
}

double death_y(const F &f, double x, double z) {
    double d = f.void_y;
    for (int64_t b = 0; b < f.nboxes; ++b) {
        const double *B = f.boxp(b);
        if (B[0] <= x && x <= B[3] && B[2] <= z && z <= B[5]) d = pmax(d, B[4] + DEATH_MARGIN);
    }
    return d;
}

double death_floor_box(const F &f, double x0, double x1, double z0, double z1) {
    double d = INF;
    bool inside = false;
    for (int64_t b = 0; b < f.nboxes; ++b) {
        const double *B = f.boxp(b);
        if (x1 >= B[0] && x0 <= B[3] && z1 >= B[2] && z0 <= B[5]) {
            d = pmin(d, B[4] + DEATH_MARGIN);
            if (x0 >= B[0] && x1 <= B[3] && z0 >= B[2] && z1 <= B[5]) inside = true;
        }
    }
    if (!inside) d = pmin(d, f.void_y);
    return d;
}

double fan_dist(const F &f, double x, double y, double z) {
    double best = INF;
    for (int64_t k = 0; k < f.nfans; ++k) {
        const double *A = f.fanp(k);
        double dx = x - A[0], dy = y - A[1], dz = z - A[2];
        double s = 0.0;
        for (int i = 0; i < 3; ++i) {
            double loc = dx * A[6 + 3 * i] + dy * A[7 + 3 * i] + dz * A[8 + 3 * i];
            double ex = std::fabs(loc) - A[3 + i];
            if (ex > 0.0) s += ex * ex;
        }
        double d = std::sqrt(s);
        if (d < best) best = d;
    }
    return best;
}

double fan_col_dist(const F &f, double x, double y, double z) {
    double best = INF;
    for (int64_t k = 0; k < f.nfans; ++k) {
        const double *A = f.fanp(k);
        double dx = x - A[0], dy = y - A[1], dz = z - A[2];
        double s = 0.0;
        for (int i = 0; i < 3; i += 2) {
            double loc = dx * A[6 + 3 * i] + dy * A[7 + 3 * i] + dz * A[8 + 3 * i];
            double ex = std::fabs(loc) - A[3 + i];
            if (ex > 0.0) s += ex * ex;
        }
        double d = std::sqrt(s);
        if (d < best) best = d;
    }
    return best;
}

int64_t fan_touch(const F &f, const double *yy, const double *xf, const double *zf, const double *D, int64_t n_end,
                  double rad) {
    const int64_t N = NPSI;
    if (n_end > N) n_end = N;
    if (n_end < 0 || f.nfans == 0) return -1;
    double ylo = INF, yhi = -INF;
    for (int64_t n = 0; n < n_end + 1; ++n) {
        ylo = pmin(ylo, yy[n]);
        yhi = pmax(yhi, yy[n]);
    }
    double xl = pmin(xf[0], xf[n_end]) - D[n_end];
    double xh = pmax(xf[0], xf[n_end]) + D[n_end];
    double zl = pmin(zf[0], zf[n_end]) - D[n_end];
    double zh = pmax(zf[0], zf[n_end]) + D[n_end];
    int64_t best = -1;
    double r2 = rad * rad;
    for (int64_t k = 0; k < f.nfans; ++k) {
        const double *A = f.fanp(k);
        double cx = A[0], cy = A[1], cz = A[2], hx = A[15], hy = A[16], hz = A[17];
        if (xh < cx - hx - rad || xl > cx + hx + rad || zh < cz - hz - rad || zl > cz + hz + rad ||
            yhi < cy - hy - rad || ylo > cy + hy + rad)
            continue;
        for (int64_t n = 0; n < n_end + 1; ++n) {
            if (best >= 0 && n >= best) break;
            double ex = pmax(0.0, std::fabs(xf[n] - cx) - D[n] - hx);
            double ey = pmax(0.0, std::fabs(yy[n] - cy) - hy);
            double ez = pmax(0.0, std::fabs(zf[n] - cz) - D[n] - hz);
            if (ex * ex + ey * ey + ez * ez <= r2) {
                best = n;
                break;
            }
        }
    }
    return best;
}

int64_t fan_profile(const F &f, const double *xf, const double *zf, double y0, double vy, double lam, double dv,
                    double rnom, int64_t n_max, double *yy, double *ins) {
    double y = y0, v = vy;
    yy[0] = y0;
    ins[0] = fan_dist(f, xf[0], y0, zf[0]) <= rnom ? 1.0 : 0.0;
    int64_t cnt = 0;
    for (int64_t n = 1; n < n_max + 1; ++n) {
        y += v / PSI_HZ;
        yy[n] = y;
        double a = -GPSI;
        if (fan_dist(f, xf[n], y, zf[n]) <= rnom) {
            a += dv;
            ins[n] = 1.0;
            cnt += 1;
        } else {
            ins[n] = 0.0;
        }
        v = lam * v + a;
    }
    return cnt;
}

void fan_upper(double y0, double vy, double lam, double dv, int64_t n_from, int64_t n_max, double *yy) {
    double y = y0, v = vy;
    yy[0] = y0;
    for (int64_t n = 1; n < n_max + 1; ++n) {
        y += v / PSI_HZ;
        yy[n] = y;
        double a = -GPSI;
        if (n >= n_from) a += dv;
        v = lam * v + a;
    }
}

int64_t first_le(const double *y, int64_t a, int64_t b, double v) {
    if (a > b || y[b] > v) return b + 1;
    int64_t lo = a, hi = b;
    while (lo < hi) {
        int64_t m = floordiv(lo + hi, 2);
        if (y[m] <= v)
            hi = m;
        else
            lo = m + 1;
    }
    return lo;
}

int64_t last_ge(const double *y, int64_t a, int64_t b, double v) {
    if (a > b || y[a] < v) return a - 1;
    int64_t lo = a, hi = b;
    while (lo < hi) {
        int64_t m = floordiv(lo + hi + 1, 2);
        if (y[m] >= v)
            lo = m;
        else
            hi = m - 1;
    }
    return lo;
}

struct Brake {
    double d, v, kstop;
};

Brake brake_axis(double v0, double g_t, double a_g, double lam, double t) {
    if (std::fabs(v0) < 1e-9) return Brake{0.0, 0.0, 0.0};
    double s = v0 > 0 ? 1.0 : -1.0;
    double a = -s * a_g + g_t;
    double one = 1.0 - lam;
    double k, kstop;
    if (a * s >= 0.0) {
        k = t;
        kstop = INF;
    } else {
        double A = a / one;
        kstop = c_log(-A / (v0 - A)) / c_log(lam);
        k = pmin(t, kstop);
    }
    double lk = c_pow(lam, k);
    double geo = (1.0 - lk) / one;
    double d = (v0 * geo + a * (k - geo) / one) / PSI_HZ;
    double v = lk * v0 + a * geo;
    if (k >= kstop) v = 0.0;
    return Brake{d, v, kstop};
}

inline void disp1(double v0, double a, double lam, double k, double &d, double &v) {
    double one = 1.0 - lam;
    double lk = c_pow(lam, k);
    double geo = (1.0 - lk) / one;
    d = (v0 * geo + a * (k - geo) / one) / PSI_HZ;
    v = lk * v0 + a * geo;
}

double v_reach(double v0, double a, double lam, double L) {
    if (L <= 0.0) return pmax(v0, 0.0);
    double one = 1.0 - lam;
    double lo, hi, d, v;
    if (a < 0.0 || (a == 0.0 && v0 > 0.0)) {
        if (v0 <= 0.0) return 0.0;
        double ks;
        if (a < 0.0) {
            double A = a / one;
            ks = c_log(-A / (v0 - A)) / c_log(lam);
        } else {
            ks = 1e6;
        }
        disp1(v0, a, lam, ks, d, v);
        if (d < L) return 0.0;
        lo = 0.0;
        hi = ks;
    } else if (a == 0.0) {
        return 0.0;
    } else {
        hi = 1.0;
        while (true) {
            disp1(v0, a, lam, hi, d, v);
            if (d >= L) break;
            hi *= 2.0;
            if (hi > 1e6) return 0.0;
        }
        lo = hi > 1.0 ? hi * 0.5 : 0.0;
    }
    double k = hi;
    for (int it = 0; it < 40; ++it) {
        disp1(v0, a, lam, k, d, v);
        double fk = d - L;
        if (std::fabs(fk) < 1e-7) break;
        if (fk > 0.0)
            hi = k;
        else
            lo = k;
        double kn = v > 1e-9 ? k - fk * PSI_HZ / v : -1.0;
        if (!(lo < kn && kn < hi)) kn = 0.5 * (lo + hi);
        k = kn;
    }
    disp1(v0, a, lam, k, d, v);
    return v;
}

bool clear_ok(const double *S1, const double *S2, int64_t N, double v, double k, double vy_abs, double a_p,
              double x_need, double dy_t) {
    if (v * S1[N] + a_p * S2[N] < x_need) return false;
    int64_t lo = 0, hi = N;
    while (lo < hi) {
        int64_t m = floordiv(lo + hi, 2);
        if (v * S1[m] + a_p * S2[m] >= x_need)
            hi = m;
        else
            lo = m + 1;
    }
    double yv = (k * v + vy_abs) * S1[lo] - GPSI * S2[lo];
    return yv >= dy_t;
}

double v_need(const double *S1, const double *S2, int64_t N, double k, double vy_abs, double a_p, double x_need,
              double dy_t) {
    if (clear_ok(S1, S2, N, 0.0, k, vy_abs, a_p, x_need, dy_t)) return 0.0;
    if (!clear_ok(S1, S2, N, V_MAX, k, vy_abs, a_p, x_need, dy_t)) return INF;
    double lo = 0.0, hi = V_MAX;
    for (int it = 0; it < 30; ++it) {
        double m = 0.5 * (lo + hi);
        if (clear_ok(S1, S2, N, m, k, vy_abs, a_p, x_need, dy_t))
            hi = m;
        else
            lo = m;
    }
    return hi;
}

// --------------------------------------------------------------------------------------------- the march
void march(const F &f, double r, double x0, double y0, double z0, double ux, double uz, double lmax, bool want_far,
           double k0, double *out) {
    double y = y0;
    double edge_s = INF, y_edge = NaN, k_edge = 0.0;
    double cur_k = k0 == k0 ? k0 : 0.0;
    double y_face = y0;
    double obst_s = INF, obst_h = NaN;
    double s_near = NaN, s_far = INF, h_far = NaN;
    int64_t nsteps = (int64_t)(lmax / DS + 1e-9);
    for (int64_t st = 1; st < nsteps + 1; ++st) {
        double s = (double)st * DS;
        double x = x0 + ux * s;
        double z = z0 + uz * s;
        Hold h = r_hold(f, x, z, y, DROP_TOL, ux, uz, DS);
        double ynew = h.hold == h.hold ? h.hold : y;
        bool obst = h.block == h.block;
        if (!obst && h.wall) {
            Vec4 u = udist(f, x, ynew, z);
            if (u.d < r - PEN_TOL && u.x * ux + u.z * uz < -0.3) obst = true;
        }
        if (obst) {
            obst_s = s - DS;
            double ht = h_top_disc(f, x, z, r, y - r + 0.05, y - r + 8.0);
            obst_h = ht == ht ? ht - (y - r) : INF;
            break;
        }
        if (h.hold != h.hold) {
            edge_s = s;
            y_edge = y_face;
            k_edge = cur_k;
            break;
        }
        y = h.hold;
        if (h.fl != 0) y_face = y;
        if (h.fl == 1)
            cur_k = -(h.nx * ux + h.nz * uz) / pmax(h.ny, 0.2);
        else if (h.fl == 2)
            cur_k = 0.0;
    }
    out[0] = edge_s;
    out[1] = y_edge;
    out[2] = k_edge;
    out[3] = obst_s;
    out[4] = obst_h;
    out[8] = y;
    if (want_far && edge_s < INF) {
        double hb = y_edge - r;
        s_near = edge_s - 1.73;
        double sb = edge_s;
        while (sb >= pmax(0.0, edge_s - 3.0)) {
            double h = h_top(f, x0 + ux * sb, z0 + uz * sb, hb - 0.8, hb + 0.8);
            if (h == h) {
                s_near = sb;
                break;
            }
            sb -= DS;
        }
        double sf = s_near + DS;
        while (sf <= edge_s + GAP_MAX) {
            double xx = x0 + ux * sf, zz = z0 + uz * sf;
            double h = h_top(f, xx, zz, hb - GAP_DROP, hb + 4.5);
            double rt = rail_top_near(f, xx, zz, DS + 0.05, hb - GAP_DROP, hb + 4.5);
            if (rt == rt && (h != h || rt > h)) h = rt;
            if (h == h) {
                s_far = sf;
                h_far = h;
                break;
            }
            sf += 2.0 * DS;
        }
    }
    out[5] = s_near;
    out[6] = s_far;
    out[7] = h_far;
}

// ------------------------------------------------------------------------------------- airborne landing scan
inline void pyr_node(const F &f, int64_t lev, int64_t a, int64_t b, double &mn, double &mx) {
    if (lev >= 3) {
        int64_t L = lev - 3;
        if (a < 0 || b < 0 || a >= f.Pdim[L * 2 + 0] || b >= f.Pdim[L * 2 + 1]) {
            mn = INF, mx = -INF;
            return;
        }
        int64_t o = f.Poff[L] + a * f.Pdim[L * 2 + 1] + b;
        mn = (double)f.Pd[o * 2 + 0], mx = (double)f.Pd[o * 2 + 1];
        return;
    }
    int64_t sh = 3 - lev;
    int64_t ti = a >> sh, tj = b >> sh;  // arithmetic shift = floor (Python >>)
    if (ti < 0 || tj < 0 || ti >= f.tiles0 || tj >= f.tiles1) {
        mn = INF, mx = -INF;
        return;
    }
    int64_t t = f.tile(ti, tj);
    if (t < 0) {
        mn = INF, mx = -INF;
        return;
    }
    int64_t la = a - (ti << sh), lb = b - (tj << sh);
    if (lev == 2) {
        size_t o = size_t(((t * 2 + la) * 2 + lb) * 2);
        mn = (double)f.P2[o], mx = (double)f.P2[o + 1];
        return;
    }
    if (lev == 1) {
        size_t o = size_t(((t * 4 + la) * 4 + lb) * 2);
        mn = (double)f.P1[o], mx = (double)f.P1[o + 1];
        return;
    }
    mn = INF, mx = -INF;
    for (int k = 0; k < KH; ++k) {
        double h = f.hh(t, la, lb, k);
        if (h != h) break;
        mn = pmin(mn, h);
        mx = pmax(mx, h);
    }
}

struct Contact {
    int64_t first;
    double emin;
    int64_t ne;
};

Contact contact_window(const F &f, const double *yy, const double *yl, const double *ya, const double *lp, int cmode,
                       const double *xf, const double *zf, const double *D, int64_t na, int64_t ns, double cx,
                       double cz, double h, double rc, double hc) {
    int64_t n0 = first_le(yl, na, ns, h + rc);
    int64_t n1 = last_ge(yy, na, ns, h);
    int64_t first = -1, ne = -1;
    double emin = INF;
    double dcol = NaN;
    for (int64_t n = n0; n < n1 + 1; ++n) {
        double dy;
        if (cmode == 2) {
            double ys = pmax(h, yl[n]);
            if (ys > yy[n] || ys > h + rc) continue;
            dy = ys - h;
        } else if (cmode == 1) {
            dy = ya[n] - h;
            if (dy < 0.0) continue;
        } else {
            dy = yy[n] - h;
        }
        double rho2 = rc * rc - dy * dy;
        if (rho2 < 0.0) continue;
        double ddx = pmax(std::fabs(cx - xf[n]) - D[n] - hc, 0.0);
        double ddz = pmax(std::fabs(cz - zf[n]) - D[n] - hc, 0.0);
        if (ddx * ddx + ddz * ddz > rho2) continue;
        if (cmode == 1 && lp[n] >= 0.0) {
            if (dcol != dcol) dcol = fan_col_dist(f, cx, h, cz) - FAN_R_NOM - rc;
            if (dcol > 0.0) {
                int64_t m = (int64_t)lp[n];
                if (m >= n) continue;
                double tx = xf[n] - xf[m], tz = zf[n] - zf[m];
                if (dcol > std::sqrt(tx * tx + tz * tz) + D[n - m]) continue;
            }
        }
        if (first < 0) first = n;
        double ex = cx - xf[n], ez = cz - zf[n];
        double e = pmax(0.0, std::sqrt(ex * ex + ez * ez) - std::sqrt(rho2));
        if (e < emin) {
            emin = e;
            ne = n;
        }
    }
    return Contact{first, emin, ne};
}

// _air_scan with the empty collect (the kernel never collects here). filt (4), excl (3) as in Python.
void air_scan(const F &f, W &w, const double *S1, const double *S2, double x0, double y0, double z0, double vx,
              double vy, double vz, double a_air, double r, const double *filt, const double *excl, int mode,
              double *res, int given) {
    const int64_t N = NPSI;
    double *yy = w.yy.data(), *xf = w.xf.data(), *zf = w.zf.data(), *D = w.D.data();
    double *yl = w.yl.data(), *ya = w.ya.data(), *fins = w.fins.data();
    for (int64_t n = 0; n < N + 1; ++n) {
        if (given == 0) yy[n] = y0 + vy * S1[n] - GPSI * S2[n];
        xf[n] = x0 + vx * S1[n];
        zf[n] = z0 + vz * S1[n];
        D[n] = a_air * S2[n];
    }
    if (given == 1) {
        double last = -1.0;
        for (int64_t n = 0; n < N + 1; ++n) {
            ya[n] = yy[n];
            if (fins[n] > 0.5) last = (double)n;
            fins[n] = last;
        }
    }
    int64_t na;
    if (given == 0) {
        na = 0;
        while (na < N && yy[na + 1] >= yy[na]) na += 1;
        na = imax(na, 1);
    } else if (given == 2) {
        na = 0;
        while (na < N && yl[na + 1] >= yl[na]) na += 1;
        na = imax(na, 1);
    } else {
        na = 1;
    }
    int64_t ns = N;
    bool crossed = false;
    for (int64_t n = na; n < N + 1; ++n) {
        double dfl = death_floor_box(f, xf[n] - D[n] - r, xf[n] + D[n] + r, zf[n] - D[n] - r, zf[n] + D[n] + r);
        if (yy[n] < dfl) {
            ns = n - 1;
            crossed = true;
            break;
        }
    }
    if (given != 0) {
        if (given == 1) {
            na = 0;
            while (na < ns && yy[na + 1] >= yy[na]) na += 1;
            na = imax(na, 1);
            double m = INF;
            for (int64_t n = na; n < ns + 1; ++n) {
                if (ya[n] < m) m = ya[n];
                yl[n] = m;
            }
        }
        double m = -INF;
        for (int64_t n = ns; n > na - 1; --n) {
            if (yy[n] > m) m = yy[n];
            yy[n] = m;
        }
    }
    int64_t best_n = N + 1;
    double best_e = INF;
    double fx = NaN, fh = NaN, fz = NaN;
    double ex_ = NaN, eh_ = NaN, ez_ = NaN;
    int64_t en_ = -1;
    double erc_ = NaN, eyc_ = NaN;
    int64_t visited = 0;
    const double rc = r + CONTACT_TOL;
    const double hc = 0.5 * RES;
    const bool use_f = filt[0] <= filt[1];
    const bool use_x = excl[2] > 0.0;
    bool done = false;
    const double *ylw = given == 0 ? yy : yl;
    if (ns >= na) {
        // ---- pyramid walk
        const int64_t top = f.npdim - 1 + 3;
        int64_t sp = 0;
        const int64_t na0 = f.Pdim[(f.npdim - 1) * 2 + 0], nb0 = f.Pdim[(f.npdim - 1) * 2 + 1];
        for (int64_t a = 0; a < na0; ++a)
            for (int64_t b = 0; b < nb0; ++b) w.push(sp, top, a, b);
        while (sp > 0 && !done) {
            sp -= 1;
            const int64_t lev = w.stack[sp * 3 + 0], a = w.stack[sp * 3 + 1], b = w.stack[sp * 3 + 2];
            visited += 1;
            double hmin, hmax;
            pyr_node(f, lev, a, b, hmin, hmax);
            if (hmin > hmax) continue;
            const int64_t span = int64_t(1) << lev;
            double bx0 = f.ox + ((double)(a * span) - 0.5) * RES;
            double bx1 = f.ox + ((double)((a + 1) * span) - 0.5) * RES;
            double bz0 = f.oz + ((double)(b * span) - 0.5) * RES;
            double bz1 = f.oz + ((double)((b + 1) * span) - 0.5) * RES;
            if (use_f && (bx1 < filt[0] || bx0 > filt[1] || bz1 < filt[2] || bz0 > filt[3])) continue;
            int64_t n_lo = first_le(ylw, na, ns, hmax + rc);
            int64_t n_hi = last_ge(yy, na, ns, hmin);
            if (n_lo > n_hi) continue;
            double Dm = D[n_hi] + rc + hc;
            double xl = pmin(xf[n_lo], xf[n_hi]), xh = pmax(xf[n_lo], xf[n_hi]);
            double zl = pmin(zf[n_lo], zf[n_hi]), zh = pmax(zf[n_lo], zf[n_hi]);
            if (xh + Dm < bx0 || xl - Dm > bx1 || zh + Dm < bz0 || zl - Dm > bz1) continue;
            if (mode == 0) {  // (and the empty collect)
                double gx = pmax(0.0, pmax(bx0 - xh, xl - bx1));
                double gz = pmax(0.0, pmax(bz0 - zh, zl - bz1));
                double e_lb = pmax(0.0, std::sqrt(gx * gx + gz * gz) - rc - hc);
                if (n_lo >= best_n && en_ >= 0 && (e_lb > best_e || (e_lb == best_e && n_lo >= en_))) continue;
            }
            if (lev > 0) {
                for (int64_t ca = 0; ca < 2; ++ca)
                    for (int64_t cb = 0; cb < 2; ++cb) w.push(sp, lev - 1, 2 * a + ca, 2 * b + cb);
                continue;
            }
            // ---- a cell: its layers
            int64_t t = tile_of(f, a, b);
            double cx = f.ox + (double)a * RES;
            double cz = f.oz + (double)b * RES;
            if (use_x) {
                double ex = cx - excl[0], ez = cz - excl[1];
                if (ex * ex + ez * ez <= excl[2] * excl[2]) continue;
            }
            for (int k = 0; k < KH; ++k) {
                double h = f.hh(t, pymod(a, TL), pymod(b, TL), k);
                if (h != h) break;
                Contact c = contact_window(f, yy, ylw, ya, fins, given, xf, zf, D, na, ns, cx, cz, h, rc, hc);
                if (c.first < 0) continue;
                if (yy[c.first] < death_y(f, cx, cz)) continue;
                if (c.first < best_n) {
                    best_n = c.first;
                    fx = cx, fh = h, fz = cz;
                }
                if (c.emin < best_e || (c.emin == best_e && c.ne < en_)) {
                    best_e = c.emin;
                    ex_ = cx, eh_ = h, ez_ = cz;
                    en_ = c.ne;
                    erc_ = rc;
                    eyc_ = h;
                }
                if (mode == 1) {
                    done = true;
                    break;
                }
            }
        }
        // ---- rails: axis samples every <= 0.5 with contact radius r + tube radius
        if (!done && f.nseg > 0) {
            double Dm = D[ns] + r + 1.0;
            double xl = pmin(xf[na], xf[ns]) - Dm;
            double xh = pmax(xf[na], xf[ns]) + Dm;
            double zl = pmin(zf[na], zf[ns]) - Dm;
            double zh = pmax(zf[na], zf[ns]) + Dm;
            if (use_f) {
                xl = pmax(xl, filt[0] - 1.0);
                xh = pmin(xh, filt[1] + 1.0);
                zl = pmax(zl, filt[2] - 1.0);
                zh = pmin(zh, filt[3] + 1.0);
            }
            int64_t ti0 = ifloor((xl - f.ox + 0.5 * RES) / (RES * TL));
            int64_t ti1 = ifloor((xh - f.ox + 0.5 * RES) / (RES * TL));
            int64_t tj0 = ifloor((zl - f.oz + 0.5 * RES) / (RES * TL));
            int64_t tj1 = ifloor((zh - f.oz + 0.5 * RES) / (RES * TL));
            const int64_t ntx = floordiv((int64_t)f.rstart.size() - 1, f.ntz);
            for (int64_t ti = imax(ti0, 0); ti < imin(ti1, ntx - 1) + 1; ++ti) {
                if (done) break;
                for (int64_t tj = imax(tj0, 0); tj < imin(tj1, f.ntz - 1) + 1; ++tj) {
                    if (done) break;
                    int64_t c = ti * f.ntz + tj;
                    for (int64_t q = f.rstart[c]; q < f.rstart[c + 1]; ++q) {
                        int64_t s = f.ritems[q];
                        if (w.stamp[s] == w.stamp_id) continue;
                        w.stamp[s] = w.stamp_id;
                        const double *S = f.segp(s);
                        double e0 = S[3] - S[0], e1 = S[4] - S[1], e2 = S[5] - S[2];
                        double L = std::sqrt(e0 * e0 + e1 * e1 + e2 * e2);
                        int64_t m = iceil(L / RES);
                        double rcr = r + S[6] + CONTACT_TOL;
                        for (int64_t u = 0; u < m + 1; ++u) {
                            double fr = (double)u / (double)imax(m, 1);
                            double px = S[0] + fr * (S[3] - S[0]);
                            double py = S[1] + fr * (S[4] - S[1]);
                            double pz = S[2] + fr * (S[5] - S[2]);
                            if (use_f && (px < filt[0] || px > filt[1] || pz < filt[2] || pz > filt[3])) continue;
                            if (use_x) {
                                double ex = px - excl[0], ez = pz - excl[1];
                                if (ex * ex + ez * ez <= excl[2] * excl[2]) continue;
                            }
                            Contact cw =
                                contact_window(f, yy, ylw, ya, fins, given, xf, zf, D, na, ns, px, pz, py, rcr, 0.0);
                            if (cw.first < 0) continue;
                            if (yy[cw.first] < death_y(f, px, pz)) continue;
                            if (cw.first < best_n) {
                                best_n = cw.first;
                                fx = px, fh = py + S[6], fz = pz;
                            }
                            if (cw.emin < best_e || (cw.emin == best_e && cw.ne < en_)) {
                                best_e = cw.emin;
                                ex_ = px, eh_ = py + S[6], ez_ = pz;
                                en_ = cw.ne;
                                erc_ = rcr;
                                eyc_ = py;
                            }
                            if (mode == 1) {
                                done = true;
                                break;
                            }
                        }
                        if (done) break;
                    }
                }
            }
        }
    }
    bool found = best_n <= N;
    res[0] = found ? 1.0 : 0.0;
    res[1] = found ? (double)best_n : -1.0;
    res[2] = fx;
    res[3] = fh;
    res[4] = fz;
    res[5] = (double)en_;
    res[6] = ex_;
    res[7] = eh_;
    res[8] = ez_;
    res[9] = best_e;
    res[10] = (double)na;
    res[11] = (double)ns;
    res[12] = (double)visited;
    res[13] = erc_;
    res[14] = eyc_;
    res[15] = crossed ? 1.0 : 0.0;
}

// ------------------------------------------------------------------------------------------- the full query
int64_t perceive_kernel(const F &f, W &w, const double *uzones, int64_t nuz, const Ball &B, const double (*dirs)[3],
                        const double *rightv, const double *fwdv, double px, double py, double pz, double vx,
                        double vy, double vz, double *out, double *dbg, double *pdd) {
    const int64_t N = NPSI;
    const double *S1 = B.S1, *S2 = B.S2;
    const double lam_a = B.lam_a, a_air = B.a_air, lam_g = B.lam_g, a_g = B.a_g, r = B.r, fdv = B.fdv;
    double *yy = w.yy.data(), *xf = w.xf.data(), *zf = w.zf.data(), *D = w.D.data();
    double *yl = w.yl.data(), *ya = w.ya.data(), *fins = w.fins.data();
    double *res = w.res, *res0 = w.res0, *mres = w.mres;
    int64_t stamp_id = w.stamp_id;
    if ((int64_t)w.stamp.size() < std::max<int64_t>(f.nseg, 1)) w.stamp.resize(size_t(std::max<int64_t>(f.nseg, 1)), 0);
    const bool fan_on = fdv > 0.0 && f.nfans > 0;
    for (int i = 0; i < N_FEATURES; ++i) out[i] = 0.0;
    for (int i = 0; i < NDBG; ++i) dbg[i] = NaN;
    for (int i = 0; i < 8 * NPDD; ++i) pdd[i] = NaN;
    const double nofilt[4] = {1.0, 0.0, 0.0, 0.0};
    const double noex[3] = {0.0, 0.0, 0.0};
    double speed = std::sqrt(vx * vx + vz * vz);
    dbg[DBG_SPEED] = speed;
    // ---- unknown zones (moving bodies)
    double unk = 0.0;
    for (int64_t u = 0; u < nuz; ++u) {
        const double *Z = uzones + u * 6;
        if (Z[0] <= px && px <= Z[3] && Z[1] <= py && py <= Z[4] && Z[2] <= pz && pz <= Z[5]) unk = 1.0;
    }
    out[2] = unk;
    // ---- support
    Ground g = r_ground(f, px, pz, py, GROUND_TOL);
    double rest = g.rest, snx = g.nx, sny = g.ny, snz = g.nz;
    const int64_t sfl = g.fl;
    bool grounded = rest == rest;
    if (grounded) {
        double vy_plane = -(snx * vx + snz * vz) / pmax(sny, 0.2);
        if (vy < vy_plane - FALL_TOL) {
            grounded = false;
            rest = NaN;
        }
    }
    dbg[DBG_GROUNDED] = grounded ? 1.0 : 0.0;
    dbg[DBG_SUP_REST] = rest;
    dbg[DBG_SUP_FLAG] = (double)sfl;
    out[0] = grounded ? 1.0 : 0.0;
    if (grounded) {
        out[13] = snx * rightv[0] + snz * rightv[1];
        out[14] = sny;
        out[15] = snx * fwdv[0] + snz * fwdv[1];
    }
    // ---- side contact: SDF probes 1.2 from the centre in 8 directions, rails
    double best = INF, bnx = 0.0, bny = 0.0, bnz = 0.0;
    for (int q = 0; q < 8; ++q) {
        double qx = px + 1.2 * PROBE_COS[q];
        double qz = pz + 1.2 * PROBE_SIN[q];
        Vec4 u = udist(f, qx, py, qz);
        double gap = u.d - (r - 1.2);
        if (gap < 0.15 && std::fabs(u.y) < UP_FACE && gap < best) {
            best = gap;
            bnx = u.x, bny = u.y, bnz = u.z;
        }
    }
    Vec4 rd = rail_dist(f, px, py, pz);
    if (rd.d - r < 0.15 && rd.y < UP_FACE && rd.d - r < best) {
        best = rd.d - r;
        bnx = rd.x, bny = rd.y, bnz = rd.z;
    }
    if (best < INF) {
        double hn = std::sqrt(bnx * bnx + bny * bny + bnz * bnz);
        if (hn > 1e-9) {
            out[16] = (bnx * rightv[0] + bnz * rightv[1]) / hn;
            out[17] = bny / hn;
            out[18] = (bnx * fwdv[0] + bnz * fwdv[1]) / hn;
        }
    }
    // ---- airborne landing reachability
    out[4] = (double)N / 132.0;
    if (fan_on) dbg[DBG_FAN] = 0.0;
    if (!grounded) {
        stamp_id += 1;
        w.stamp_id = stamp_id;
        air_scan(f, w, S1, S2, px, py, pz, vx, vy, vz, a_air, r, nofilt, noex, 0, res, 0);
        bool fan_unk = false;
        if (fan_on) {
            int64_t ft = fan_touch(f, yy, xf, zf, D, res[15] > 0.0 ? (int64_t)res[11] + 1 : N, FAN_R_OUT);
            dbg[DBG_FAN_T] = (double)ft;
            if (ft >= 0 && !(res[0] > 0.0 && pmax(res[1], res[5]) < (double)ft)) {
                bool found0 = res[0] > 0.0;
                std::memcpy(res0, res, sizeof(double) * 16);
                int64_t pushed = fan_profile(f, xf, zf, py, vy, lam_a, fdv, FAN_R_NOM, N, yy, fins);
                int64_t n_push = N + 1;
                for (int64_t n = 0; n < N + 1; ++n) {
                    if (fins[n] > 0.5) {
                        n_push = n;
                        break;
                    }
                }
                if (found0 && pmax(res0[1], res0[5]) < (double)n_push) pushed = 0;
                if (pushed > 0) {
                    stamp_id += 1;
                    w.stamp_id = stamp_id;
                    air_scan(f, w, S1, S2, px, py, pz, vx, vy, vz, a_air, r, nofilt, noex, 0, res, 1);
                }
                if (pushed > 0 && res[0] > 0.0) {
                    dbg[DBG_FAN] = 1.0;
                    for (int64_t n = 0; n < N + 1; ++n) yy[n] = ya[n];
                } else if (found0) {
                    dbg[DBG_FAN] = 2.0;
                    std::memcpy(res, res0, sizeof(double) * 16);
                    for (int64_t n = 0; n < N + 1; ++n) yy[n] = py + vy * S1[n] - GPSI * S2[n];
                } else {
                    fan_upper(py, vy, lam_a, fdv, ft, N, yy);
                    for (int64_t n = 0; n < N + 1; ++n) yl[n] = py + vy * S1[n] - GPSI * S2[n];
                    stamp_id += 1;
                    w.stamp_id = stamp_id;
                    air_scan(f, w, S1, S2, px, py, pz, vx, vy, vz, a_air, r, nofilt, noex, 1, res, 2);
                    if (res[0] > 0.0 || res[15] == 0.0) {
                        dbg[DBG_FAN] = 3.0;
                        fan_unk = true;
                    } else {
                        dbg[DBG_FAN] = 7.0;
                    }
                    res[0] = 0.0;
                }
            }
        }
        dbg[DBG_AIR_ANY] = res[0];
        dbg[DBG_N_APEX] = res[10];
        dbg[DBG_N_STOP] = res[11];
        dbg[DBG_VISITED] = res[12];
        if (res[0] > 0.0) {
            out[3] = 1.0;
            int64_t nf = (int64_t)res[1];
            out[4] = (double)nf / 132.0;
            out[5] = yy[nf] - py;
            dbg[DBG_N_FIRST] = (double)nf;
            dbg[DBG_FIRST_X] = res[2];
            dbg[DBG_FIRST_H] = res[3];
            dbg[DBG_FIRST_Z] = res[4];
            int64_t nn = (int64_t)res[5];
            double cx = res[6], ch = res[7], cz = res[8];
            dbg[DBG_N_NEAR] = (double)nn;
            dbg[DBG_NEAR_X] = cx;
            dbg[DBG_NEAR_H] = ch;
            dbg[DBG_NEAR_Z] = cz;
            dbg[DBG_NEAR_EFFORT] = res[9];
            double dyc = yy[nn] - res[14];
            double rho = std::sqrt(pmax(res[13] * res[13] - dyc * dyc, 0.0));
            double ddx = cx - xf[nn], ddz = cz - zf[nn];
            double dd = std::sqrt(ddx * ddx + ddz * ddz);
            double need = pmax(0.0, dd - rho);
            double ofx = dd > 1e-9 ? ddx / dd * need : 0.0;
            double ofz = dd > 1e-9 ? ddz / dd * need : 0.0;
            double tx = xf[nn] + pmin(pmax(ofx, -D[nn]), D[nn]);
            double tz = zf[nn] + pmin(pmax(ofz, -D[nn]), D[nn]);
            double ux_ = D[nn] > 1e-9 ? (tx - xf[nn]) / D[nn] : 0.0;
            double uz_ = D[nn] > 1e-9 ? (tz - zf[nn]) / D[nn] : 0.0;
            double dxl = tx - px, dzl = tz - pz;
            out[6] = dxl * rightv[0] + dzl * rightv[1];
            out[7] = dxl * fwdv[0] + dzl * fwdv[1];
            out[8] = yy[nn] - py;
            out[9] = (double)nn / 132.0;
            out[10] = pmax(std::fabs(ux_), std::fabs(uz_));
            dbg[DBG_LAND_CX] = tx;
            dbg[DBG_LAND_CY] = yy[nn];
            dbg[DBG_LAND_CZ] = tz;
            double g66 = PSI_HZ * S1[nn];
            double lamn = 1.0 - (1.0 - lam_a) * g66;
            double lvx = lamn * vx + a_air * ux_ * g66;
            double lvz = lamn * vz + a_air * uz_ * g66;
            dbg[DBG_LAND_VX] = lvx;
            dbg[DBG_LAND_VZ] = lvz;
            double lsp = std::sqrt(lvx * lvx + lvz * lvz);
            double sx_ = brake_axis(lvx, 0.0, a_g, lam_g, 1e9).d;
            double sz_ = brake_axis(lvz, 0.0, a_g, lam_g, 1e9).d;
            double bl = std::sqrt(sx_ * sx_ + sz_ * sz_);
            dbg[DBG_LAND_BRAKE] = bl;
            double can_stop = 1.0;
            if (lsp > 0.05) {
                double ysup = r_below(f, tx, tz, yy[nn] + 1.0);
                if (ysup != ysup) ysup = yy[nn];
                march(f, r, tx, ysup, tz, sx_ / pmax(bl, 1e-9), sz_ / pmax(bl, 1e-9), pmin(bl + 2.0, 40.0), false,
                      NaN, mres);
                dbg[DBG_LAND_EDGE] = mres[0];
                if (mres[0] < INF && mres[0] < bl && !(mres[3] < mres[0])) can_stop = 0.0;
            }
            out[11] = can_stop;
            double cmin = INF;
            for (int64_t j = 1; j < 11; ++j) {
                int64_t k = (int64_t)std::nearbyint((double)(j * (nn - 2)) / 10.0);  // round(): half to even
                if (k < 1) continue;
                double qx = xf[k] + ux_ * D[k];
                double qz = zf[k] + uz_ * D[k];
                double d = udist(f, qx, yy[k], qz).d;
                double d2 = rail_dist(f, qx, yy[k], qz).d;
                cmin = pmin(pmin(cmin, d), d2);
            }
            dbg[DBG_CLEAR_MIN] = cmin;
            out[12] = cmin >= r - 0.15 ? 1.0 : 0.0;
        } else if (res[15] > 0.0 && !fan_unk) {
            out[1] = 1.0;
            dbg[DBG_DOOM_REASON] = dbg[DBG_FAN] == 7.0 ? 6.0 : 1.0;
        } else {
            out[2] = 1.0;
            dbg[DBG_DOOM_REASON] = fan_unk ? 5.0 : 4.0;
        }
    }
    // ---- ground: 8 directions
    for (int q = 0; q < 8; ++q) {
        int base = N_HEAD + q * 5;
        out[base + 0] = L_MAX;
        out[base + 1] = L_MAX;
        out[base + 2] = -CLIP_V;
        out[base + 3] = L_MAX;
        out[base + 4] = -CLIP_V;
    }
    if (grounded) {
        double gtx = (GPSI / ROLL_INERTIA) * sny * snx;
        double gtz = (GPSI / ROLL_INERTIA) * sny * snz;
        Brake bx = brake_axis(vx, gtx, a_g, lam_g, 1e9);
        Brake bz = brake_axis(vz, gtz, a_g, lam_g, 1e9);
        const double bxd = bx.d, kbx = bx.kstop, bzd = bz.d, kbz = bz.kstop;
        for (int q = 0; q < 8; ++q) {
            double ux = dirs[q][0], uz = dirs[q][1], pmag = dirs[q][2];
            int base = N_HEAD + q * 5;
            double *pq = pdd + q * NPDD;
            double k_sup = sfl == 1 ? -(snx * ux + snz * uz) / pmax(sny, 0.2) : 0.0;
            march(f, r, px, rest, pz, ux, uz, L_MAX, true, k_sup, mres);
            double edge_s = mres[0], obst_s = mres[3], obst_h = mres[4];
            double brake_along;
            if (kbx == INF || kbz == INF)
                brake_along = INF;
            else
                brake_along = pmax(0.0, bxd * ux + bzd * uz);
            double v0d = vx * ux + vz * uz;
            double a_eff = a_g * pmag + gtx * ux + gtz * uz;
            pq[0] = edge_s;
            pq[1] = obst_s;
            pq[2] = obst_h;
            pq[4] = brake_along;
            if (edge_s < INF) {
                out[base + 0] = pmin(edge_s, L_MAX);
                out[base + 1] = pmax(-L_MAX, pmin(L_MAX, edge_s - brake_along));
                double s_near = mres[5], s_far = mres[6];
                pq[7] = s_near;
                pq[8] = s_far;
                pq[9] = mres[7];
                pq[12] = mres[2];
                double vcan = v_reach(v0d, a_eff, lam_g, pmax(s_near, 0.0));
                pq[5] = vcan;
                if (s_far < INF) {
                    double x_need = s_far - s_near;
                    double dy_t = mres[7] - mres[1];
                    double vn = v_need(S1, S2, N, mres[2], 0.0, a_air * pmag, x_need, dy_t);
                    pq[10] = vn;
                    pq[11] = v_need(S1, S2, N, mres[2], 0.0, 0.0, x_need, dy_t);
                    out[base + 2] = vn == INF ? CLIP_V : pmax(-CLIP_V, pmin(CLIP_V, vn - vcan));
                } else {
                    out[base + 2] = CLIP_V;
                }
            } else if (brake_along == INF) {
                out[base + 1] = -L_MAX;
            }
            if (obst_s < INF) {
                out[base + 3] = pmin(obst_s, L_MAX);
                double mv = obst_h == obst_h ? mount_speed(obst_h, r) : INF;
                pq[3] = mv;
                double vco = v_reach(v0d, a_eff, lam_g, pmax(obst_s, 0.0));
                pq[6] = vco;
                out[base + 4] = mv == INF ? CLIP_V : pmax(-CLIP_V, pmin(CLIP_V, mv - vco));
            }
        }
        // ---- grounded doomed: brake path, else a takeoff with a reachable landing
        dbg[DBG_DOOM_REASON] = 0.0;
        bool fan_in = false;
        if (fan_on) {
            double dfan = fan_dist(f, px, py, pz);
            if (dfan <= FAN_R_OUT) {
                fan_in = true;
                dbg[DBG_FAN] = 4.0;
                if (dfan <= FAN_R_IN && fdv > GPSI) out[2] = 1.0;
            }
        }
        if (sfl != 2 && speed > 0.5 && !fan_in) {
            bool safe = false;
            bool tko = false;
            if (kbx < INF && kbz < INF) {
                double kmax = pmin(pmax(kbx, kbz), 1320.0);
                double yb = rest;
                bool ok = true;
                double lx = px, lz = pz, lvx_ = vx, lvz_ = vz;
                for (int64_t j = 1; j < 13; ++j) {
                    double t = kmax * (double)j / 12.0;
                    Brake ax = brake_axis(vx, gtx, a_g, lam_g, t);
                    Brake az = brake_axis(vz, gtz, a_g, lam_g, t);
                    double qx = px + ax.d, qz = pz + az.d;
                    double ex = qx - lx, ez = qz - lz;
                    double stl = std::sqrt(ex * ex + ez * ez);
                    double mx = (qx - lx) / pmax(stl, 1e-9);
                    double mz = (qz - lz) / pmax(stl, 1e-9);
                    Hold h = r_hold(f, qx, qz, yb, DROP_TOL, mx, mz, stl);
                    if (h.block == h.block) break;
                    if (h.hold != h.hold) {
                        ok = false;
                        lx = qx, lz = qz, lvx_ = ax.v, lvz_ = az.v;
                        break;
                    }
                    yb = h.hold;
                    lx = qx, lz = qz;
                }
                if (ok) {
                    safe = true;
                } else {
                    tko = true;
                    dbg[DBG_BRAKE_OK] = 0.0;
                    double sp_ = std::sqrt(lvx_ * lvx_ + lvz_ * lvz_);
                    const double excl[3] = {lx, lz, EXCL_R};
                    stamp_id += 1;
                    w.stamp_id = stamp_id;
                    air_scan(f, w, S1, S2, lx, yb, lz, lvx_, 0.0, lvz_, a_air, r, nofilt, excl, 1, res, 0);
                    dbg[DBG_BRAKE_TAKEOFF] = res[0];
                    if (res[0] > 0.0 || res[15] == 0.0 || sp_ < 0.05) {
                        safe = true;
                    } else if (fan_on) {
                        int64_t ftk = fan_touch(f, yy, xf, zf, D, (int64_t)res[11] + 1, FAN_R_OUT);
                        if (ftk >= 0) {
                            for (int64_t n = 0; n < N + 1; ++n) yl[n] = yy[n];
                            fan_upper(yb, 0.0, lam_a, fdv, ftk, N, yy);
                            stamp_id += 1;
                            w.stamp_id = stamp_id;
                            air_scan(f, w, S1, S2, lx, yb, lz, lvx_, 0.0, lvz_, a_air, r, nofilt, excl, 1, res, 2);
                            safe = res[0] > 0.0 || res[15] == 0.0;
                            dbg[DBG_FAN] = safe ? 5.0 : 8.0;
                        }
                    }
                }
            }
            if (!safe) {
                double ux = vx / speed, uz = vz / speed;
                double k_sup = sfl == 1 ? -(snx * ux + snz * uz) / pmax(sny, 0.2) : 0.0;
                march(f, r, px, rest, pz, ux, uz, 40.0, false, k_sup, mres);
                dbg[DBG_JUMP_EDGE] = mres[0];
                if (mres[0] == INF || mres[3] < mres[0]) {
                    safe = true;
                } else {
                    double pm = std::fabs(ux) + std::fabs(uz);
                    double a_eff = a_g * pm + gtx * ux + gtz * uz;
                    double s_to = pmax(mres[0] - DS, 0.0);
                    double vc = v_reach(speed, a_eff, lam_g, s_to);
                    dbg[DBG_JUMP_V] = vc;
                    const double excl[3] = {px + ux * s_to, pz + uz * s_to, EXCL_R};
                    stamp_id += 1;
                    w.stamp_id = stamp_id;
                    air_scan(f, w, S1, S2, excl[0], mres[1], excl[1], vc * ux, vc * mres[2], vc * uz, a_air, r, nofilt,
                             excl, 1, res, 0);
                    dbg[DBG_JUMP_TAKEOFF] = res[0];
                    if (res[0] > 0.0 || res[15] == 0.0) {
                        safe = true;
                    } else if (fan_on) {
                        int64_t ftk = fan_touch(f, yy, xf, zf, D, (int64_t)res[11] + 1, FAN_R_OUT);
                        if (ftk >= 0) {
                            for (int64_t n = 0; n < N + 1; ++n) yl[n] = yy[n];
                            fan_upper(mres[1], vc * mres[2], lam_a, fdv, ftk, N, yy);
                            stamp_id += 1;
                            w.stamp_id = stamp_id;
                            air_scan(f, w, S1, S2, excl[0], mres[1], excl[1], vc * ux, vc * mres[2], vc * uz, a_air,
                                     r, nofilt, excl, 1, res, 2);
                            safe = res[0] > 0.0 || res[15] == 0.0;
                            dbg[DBG_FAN] = safe ? 6.0 : 9.0;
                        }
                    }
                }
            }
            if (!safe) {
                out[1] = 1.0;
                dbg[DBG_DOOM_REASON] = tko ? 2.0 : 3.0;
            }
        }
    }
    w.stamp_id = stamp_id;
    return stamp_id;
}

Ball ball_of(int ball_type) {
    if (ball_type < 0 || ball_type > 2) throw Error("fastreach: ball type " + std::to_string(ball_type));
    const double *k = BALL_K[ball_type];
    return Ball{S1T[ball_type], S2T[ball_type], k[0], k[1], k[2], k[3], k[4], k[5]};
}

}  // namespace

void perceive(const Fields &fields, Workspace &ws, const std::vector<double> &uzones, int ball_type, int quarter,
              const double pos[3], const double vel[3], double *out, double *dbg, double *pdd) {
    if (!fields.impl) throw Error("fastreach: fields not loaded");
    if (uzones.size() % 6) throw Error("fastreach: uzones must be k x 6");
    Ball b = ball_of(ball_type);
    int64_t q = pymod(quarter, 4);
    W &w = *ws.impl;
    perceive_kernel(*fields.impl, w, uzones.data(), int64_t(uzones.size() / 6), b, DIRS[q], RIGHTV[q], FWDV[q], pos[0],
                    pos[1], pos[2], vel[0], vel[1], vel[2], out, dbg ? dbg : w.dbg_s, pdd ? pdd : w.pdd_s);
}

int n_dbg() { return NDBG; }
int n_pdd() { return NPDD; }

void set_libm(double (*log_fn)(double), double (*pow_fn)(double, double)) {
    g_log = log_fn ? log_fn : crt_log;
    g_pow = pow_fn ? pow_fn : crt_pow;
}

int64_t stamp_id(const Workspace &ws) { return ws.impl->stamp_id; }
int64_t stack_high_water(const Workspace &ws) { return ws.impl->stack_max; }

// --------------------------------------------------------------------------------------- leaf functions
bool leaf(const Fields &fields, Workspace &ws, const std::string &name, const double *a, size_t na,
          std::vector<double> &out) {
    const F &f = *fields.impl;
    W &w = *ws.impl;
    out.clear();
    auto need = [&](size_t k) {
        if (na < k) throw Error("fastreach leaf " + name + ": " + std::to_string(k) + " arguments");
    };
    if (name == "h_top") {
        need(4);
        out = {h_top(f, a[0], a[1], a[2], a[3])};
    } else if (name == "sdf") {
        need(3);
        Vec4 v = sdf(f, a[0], a[1], a[2]);
        out = {v.d, v.x, v.y, v.z};
    } else if (name == "rail_dist") {
        need(3);
        Vec4 v = rail_dist(f, a[0], a[1], a[2]);
        out = {v.d, v.x, v.y, v.z};
    } else if (name == "rail_top_near") {
        need(5);
        out = {rail_top_near(f, a[0], a[1], a[2], a[3], a[4])};
    } else if (name == "r_ground") {
        need(4);
        Ground g = r_ground(f, a[0], a[1], a[2], a[3]);
        out = {g.rest, g.nx, g.ny, g.nz, (double)g.fl};
    } else if (name == "r_hold") {
        need(7);
        Hold h = r_hold(f, a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
        out = {h.hold, h.nx, h.ny, h.nz, (double)h.fl, h.block, h.wall ? 1.0 : 0.0};
    } else if (name == "h_top_disc") {
        need(5);
        out = {h_top_disc(f, a[0], a[1], a[2], a[3], a[4])};
    } else if (name == "march") {  // r, x0, y0, z0, ux, uz, lmax, want_far, k0
        need(9);
        out.assign(9, 0.0);
        march(f, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7] != 0.0, a[8], out.data());
    } else if (name == "fan_dist") {
        need(3);
        out = {fan_dist(f, a[0], a[1], a[2]), fan_col_dist(f, a[0], a[1], a[2])};
    } else if (name == "brake_axis") {
        need(5);
        Brake b = brake_axis(a[0], a[1], a[2], a[3], a[4]);
        out = {b.d, b.v, b.kstop};
    } else if (name == "v_reach") {
        need(4);
        out = {v_reach(a[0], a[1], a[2], a[3])};
    } else if (name == "v_need") {  // ball, k, vy_abs, a_p, x_need, dy_t
        need(6);
        Ball b = ball_of((int)a[0]);
        out = {v_need(b.S1, b.S2, NPSI, a[1], a[2], a[3], a[4], a[5])};
    } else if (name == "mount_speed") {
        need(2);
        out = {mount_speed(a[0], a[1])};
    } else if (name == "log") {
        need(1);
        out = {c_log(a[0])};
    } else if (name == "pow") {
        need(2);
        out = {c_pow(a[0], a[1])};
    } else if (name == "log_crt") {  // the executable's own C runtime (not used by the kernel)
        need(1);
        out = {std::log(a[0])};
    } else if (name == "pow_crt") {
        need(2);
        out = {std::pow(a[0], a[1])};
    } else if (name == "air_scan") {  // ball, x0, y0, z0, vx, vy, vz, mode, excl x, z, radius -> res (16), yy (N+1)
        need(11);
        Ball b = ball_of((int)a[0]);
        if ((int64_t)w.stamp.size() < std::max<int64_t>(f.nseg, 1)) w.stamp.resize(size_t(std::max<int64_t>(f.nseg, 1)), 0);
        const double nofilt[4] = {1.0, 0.0, 0.0, 0.0};
        const double excl[3] = {a[8], a[9], a[10]};
        w.stamp_id += 1;
        double res[16];
        air_scan(f, w, b.S1, b.S2, a[1], a[2], a[3], a[4], a[5], a[6], b.a_air, b.r, nofilt, excl, (int)a[7], res, 0);
        out.assign(res, res + 16);
        out.insert(out.end(), w.yy.begin(), w.yy.end());
    } else if (name == "fan_touch") {  // ball, x0, y0, z0, vx, vy, vz, n_end -> ft, profile count, n_push
        need(8);
        Ball b = ball_of((int)a[0]);
        for (int64_t n = 0; n < NPSI + 1; ++n) {
            w.yy[n] = a[2] + a[5] * b.S1[n] - GPSI * b.S2[n];
            w.xf[n] = a[1] + a[4] * b.S1[n];
            w.zf[n] = a[3] + a[6] * b.S1[n];
            w.D[n] = b.a_air * b.S2[n];
        }
        int64_t ft = fan_touch(f, w.yy.data(), w.xf.data(), w.zf.data(), w.D.data(), (int64_t)a[7], FAN_R_OUT);
        int64_t cnt = fan_profile(f, w.xf.data(), w.zf.data(), a[2], a[5], b.lam_a, b.fdv, FAN_R_NOM, NPSI, w.yy.data(),
                                  w.fins.data());
        out = {(double)ft, (double)cnt};
        out.insert(out.end(), w.yy.begin(), w.yy.end());
    } else {
        return false;
    }
    return true;
}

}  // namespace fastreach
}  // namespace ballrain
