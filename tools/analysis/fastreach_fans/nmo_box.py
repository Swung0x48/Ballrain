"""Mesh vertex extents from a Virtools .nmo file (minimal reader after libcmo21: CKFileReader, CKStateChunk,
CKMesh::Load): the source of fastreach.FAN_LOCAL_* and of the ball boxes in fan_fit.py.

    .venv/Scripts/python.exe -I nmo_box.py "<game>/3D Entities/PH/P_Modul_18.nmo" Kollisionsquader
    .venv/Scripts/python.exe -I nmo_box.py "<game>/3D Entities/Balls.nmo" Ball_

Prints every object whose name contains the pattern (index, id, class id, name) and, for meshes (class 32), the vertex
count and local min / max. Handles whole-file compressed files (CKFILE_WHOLECOMPRESSED, as the Ballance entity files
are) with state chunks of version 6-7.
"""
import struct
import sys
import zlib

import numpy as np

CKCID_MESH = 32
CK_STATESAVE_MESHVERTICES = 0x00020000


def read_nmo(path):
    b = open(path, "rb").read()
    if b[:8] != b"Nemo Fi\0":
        raise ValueError("not a Virtools file")
    (crc, ckver, filever, zero, mode, hdr1pack, dpack, dunpack, mgrc, objc, maxid, prodver, prodbuild,
     hdr1unpack) = struct.unpack_from("<14I", b, 8)
    hdr = b[64:64 + hdr1pack]
    if hdr1pack != hdr1unpack:
        hdr = zlib.decompress(hdr)
    objs, o = [], 0
    for _ in range(objc):
        oid, cid, fidx, nl = struct.unpack_from("<iIII", hdr, o)
        o += 16
        objs.append((oid, cid, hdr[o:o + nl].rstrip(b"\0").decode("latin-1")))
        o += nl
    data = b[64 + hdr1pack:64 + hdr1pack + dpack]
    if mode & 0x9:  # CKFILE_CHUNKCOMPRESSED_OLD | CKFILE_WHOLECOMPRESSED
        data = zlib.decompress(data)
    o = 0
    for _ in range(mgrc):  # manager chunks: guid (8), size, chunk
        ln, = struct.unpack_from("<I", data, o + 8)
        o += 12 + ln
    chunks = []
    for _ in objs:
        ln, = struct.unpack_from("<I", data, o)
        chunks.append(data[o + 4:o + 4 + ln])
        o += 4 + ln
    return objs, chunks


def chunk_dwords(ch):
    cv = ch[2]  # chunk version in the 3rd byte
    dw = np.frombuffer(ch[:len(ch) // 4 * 4], "<u4")
    if 6 <= cv <= 7:  # CHUNK_VERSION3/4: (data ver, class, chunk ver, options), data size, data
        return dw[2:2 + int(dw[1])]
    n = int(dw[2])
    return dw[(6 if cv < 5 else 7):][:n]


def seek(d, ident):
    """Position after an identifier (identifier, next position) list entry, None when absent."""
    pos = 0
    if len(d) < 2:
        return None
    while d[pos] != ident:
        pos = int(d[pos + 1])
        if pos == 0 or pos + 1 >= len(d):
            return None
    return pos + 2


def mesh_vertices(ch):
    d = chunk_dwords(ch)
    q = seek(d, CK_STATESAVE_MESHVERTICES)
    if q is None:
        return None
    n = int(d[q])
    if n == 0:
        return np.zeros((0, 3))
    if int(d[q + 1]) & 0x10:  # VertexSaveFlags::NoPos
        return None
    # count, save flags, size in dwords, then positions (3 floats per vertex) first
    return np.frombuffer(d[q + 3:q + 3 + 3 * n].tobytes(), "<f4").reshape(n, 3).astype(np.float64)


if __name__ == "__main__":
    objs, chunks = read_nmo(sys.argv[1])
    pat = sys.argv[2] if len(sys.argv) > 2 else ""
    for i, (oid, cid, name) in enumerate(objs):
        if pat in name:
            line = f"{i:5d} id {oid:5d} class {cid:3d} {name}"
            if cid == CKCID_MESH and chunks[i]:
                V = mesh_vertices(chunks[i])
                if V is not None and len(V):
                    line += f"  verts {len(V)} min {np.round(V.min(0), 4).tolist()} max {np.round(V.max(0), 4).tolist()}"
            print(line)
