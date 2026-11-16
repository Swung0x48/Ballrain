"""Wire format tests. Layouts are checked against hand-packed bytes in the field order of
docs/agent-protocol.md (not only encoder/decoder round trips)."""
import struct

import numpy as np
import pytest

from ballrain import protocol as P


def test_magics_and_header_offsets():
    assert P.SHM_MAGIC == 0x4D535242
    assert struct.pack("<I", P.OBS_MAGIC) == b"OBS1"
    assert struct.pack("<I", P.ACT_MAGIC) == b"ACT1"
    assert struct.pack("<I", P.LVL_MAGIC) == b"LVL1"
    assert (P.H_MOD_SEQ, P.H_PY_SEQ, P.H_MOD_STATE, P.H_END_REASON) == (0x20, 0x28, 0x30, 0x34)
    assert (P.H_FRAMES_DONE, P.H_OBS_VERSIONS, P.H_MOD_VERSION) == (0x40, 0x48, 0x4C)
    assert P.SHM_HEADER_SIZE == 0x100
    assert P.SHM_OBS_OFFSET == 0x1000


def test_event_codes_match_table():
    assert P.Ev.PreLoadLevel == 1 and P.Ev.BallOff == 12 and P.Ev.LevelFinish == 21
    assert P.Ev.GameOver == 22 and P.Ev.RunEnd == 30 and P.Ev.ActiveBallChanged == 27
    assert P.event_mask("BallOff", 21) == (1 << 12) | (1 << 21)
    with pytest.raises(ValueError):
        P.event_mask(64)


def _section(tag, payload, ver=1):
    return struct.pack("<4sHHI", tag.encode(), ver, 0, len(payload)) + payload + b"\0" * (-len(payload) % 4)


def _message(frames_done, sections, magic=b"OBS1"):
    body = b"".join(sections)
    return struct.pack("<4sHHIIII", magic, 1, 0, 24 + len(body), frames_done, len(sections), 0) + body


def test_time_section_hand_packed():
    payload = struct.pack("<dddffffdfBI", 1.5, 1.515, 1.5, 7.5, 0.015, 0.002, 7.5757575, 1234.5, 1234.5, 1, 3)
    o = P.parse_observation(_message(42, [_section("TIME", payload)]))
    assert o.frames_done == 42
    t = o.time
    assert t["env_time"] == 1.5 and t["time_factor"] == np.float32(0.002)
    assert t["sr_double"] == 1234.5 and t["sr_running"] == 1 and t["frames_covered"] == 3


def test_ball_section_hand_packed():
    m12 = list(range(12))
    payload = (struct.pack("<IBB", 77, 1, 1) + struct.pack("<12f", *m12) + struct.pack("<3d", 1, 2, 3)
               + struct.pack("<4d", 0, 0, 0, 1) + struct.pack("<4d", 0, 0, 1, 0) + struct.pack("<d", 9.0)
               + struct.pack("<15f", *range(100, 115)) + struct.pack("<ff", 3.5, 4.5) + struct.pack("<II", 0xC, 1)
               + struct.pack("<f", 0.0))
    assert len(payload) == P.BALL_DT.itemsize
    b = P.parse_observation(_message(1, [_section("BALL", payload)])).ball
    assert b["ckid"] == 77 and b["type"] == 1 and b["found"] == 1
    assert b["ck_matrix"][3].tolist() == [9, 10, 11]
    assert b["pos"].tolist() == [1, 2, 3] and b["q_last"][3] == 1 and b["tpsi"] == 9
    assert b["speed"].tolist() == [100, 101, 102] and b["rot_axis"].tolist() == [112, 113, 114]
    assert b["current_speed"] == 3.5 and b["core_flags"] == 0xC and b["core_state"] == 1


def test_mind_and_fric_lists_hand_packed():
    tri = [float(i) for i in range(9)]
    item = (struct.pack("<IBBHf", 501, 3, 2, 0, 0.0136) + struct.pack("<4f", 0, 1, 0, -4.5)
            + struct.pack("<fIHbB", 2.0, 0xABC0, 7, -1, 1) + struct.pack("<9f", *tri) + struct.pack("<9f", *[0.0] * 9))
    mind = struct.pack("<H", 2) + item + item
    fitem = struct.pack("<IBBBBff3f3dB", 501, 3, 2, 1, 0, 0.013, 0.56, 0, -1, 0, 1, 2, 3, 1)
    fric = struct.pack("<H", 1) + fitem
    o = P.parse_observation(_message(5, [_section("MIND", mind), _section("FRIC", fric)]))
    assert len(o.mind) == 2 and o.mind["partner"][1] == 501
    assert o.mind["gap"][0] == np.float32(0.0136)
    assert o.mind["plane"][0].tolist() == [0, 1, 0, -4.5]
    assert o.mind["edge_index"][0] == -1 and o.mind["n_tris"][0] == 1
    assert o.mind["tri"][0][2].tolist() == [6, 7, 8]
    assert o.fric["friction"][0] == np.float32(0.56) and o.fric["lt_point"][0].tolist() == [1, 2, 3]


def test_unknown_sections_skipped_and_alignment():
    ev = struct.pack("<H", 1) + struct.pack("<IHH", 141, 4, 0)  # 10 bytes -> padded to 12
    msg = _message(141, [_section("XXXX", b"abc"), _section("EVTS", ev), _section("PHYS", struct.pack("<HII", 2, 7, 8))])
    o = P.parse_observation(msg)
    assert o.unknown == ["XXXX"]
    assert o.event_codes() == [4] and o.events["frame"][0] == 141
    assert o.phys.tolist() == [7, 8]
    assert o.has_event("CamNavActive") and not o.has_event("BallOff")


def test_fixed_section_tolerates_appended_fields_and_newer_list_items():
    payload = struct.pack("<dddffffdfBI", 0, 0, 0, 0, 0, 0, 7.5, 1.0, 1.0, 0, 1) + b"NEWFIELD"
    item_v2 = struct.pack("<IBB4f", 9, 1, 1, 0, 0, 0, 0) + b"xx"  # WTCH v1 item (17 bytes) + 2 new bytes
    wt = struct.pack("<H", 2) + item_v2 + item_v2
    o = P.parse_observation(_message(3, [_section("TIME", payload), _section("WTCH", wt, ver=2)]))
    assert o.time["delta_ms"] == 7.5
    assert o.wtch["ckid"].tolist() == [9, 9] and o.wtch["visible"].tolist() == [1, 1]


def test_truncated_message_rejected():
    msg = _message(1, [_section("TIME", b"\0" * 57)])
    with pytest.raises(P.ProtocolError):
        P.parse_observation(msg[:-8])
    with pytest.raises(P.ProtocolError):
        P.parse_observation(b"OBS2" + msg[4:])
    with pytest.raises(P.ProtocolError):
        P.parse_observation(_message(1, [_section("MIND", struct.pack("<H", 3) + b"\0" * 112)]))


def test_arrs_cells():
    cells = struct.pack("<H", 2) + struct.pack("<IHHBB", 35782, 0, 0, 1, 4) + struct.pack("<i", 999) \
        + struct.pack("<IHHBB", 35779, 0, 0, 4, 4) + struct.pack("<I", 4242)
    o = P.parse_observation(_message(1, [_section("ARRS", cells)]))
    assert [(c.array, c.row, c.col, c.col_type) for c in o.arrs] == [(35782, 0, 0, 1), (35779, 0, 0, 4)]
    assert P.decode_cell(o.arrs[0].data, 1) == 999 and P.decode_cell(o.arrs[1].data, 4) == 4242
    assert P.pack_arrs(o.arrs) == cells


def test_action_encoding_hand_checked():
    frames = np.zeros(2, P.FRAME_ACT_DT)
    frames["delta"] = np.float32(1000 / 132)
    frames["keys"] = [1, 1 | 16]
    a = P.Action(frames, repeat=4, stop_mask=P.event_mask("BallOff", "LevelFinish"), flags=P.ACT_WANT_TRAC, obs_mask=1)
    b = P.encode_action(a, 7)
    hdr = struct.unpack_from("<4sHHIIIIIII", b)
    assert hdr == (b"ACT1", 1, 2, 7, 2, 4, (1 << 12) | (1 << 21), 0, 1, 0)
    assert struct.unpack_from("<fIfI", b, 36) == (np.float32(1000 / 132), 1, np.float32(1000 / 132), 17)
    back, seq = P.decode_action(b)
    assert seq == 7 and back.repeat == 4 and back.stop_mask == a.stop_mask
    assert back.expanded()["keys"].tolist() == [1] * 4 + [17] * 4
    assert a.total_frames == 8


def _level():
    lm = P.LevelMessage(frames_done=104)
    lm.entities = [P.Entity(42914, "A01_Floor_02", 41, 0, True, np.arange(12, dtype=np.float32).reshape(4, 3), 1, 1, b"Faces;FALSE")]
    lm.groups = [P.Group(44152, "Phys_Floors", np.array([42914], np.uint32))]
    lm.arrays = [P.ArrayDef(35782, "Energy", [P.ArrayColumn("Points", 1, None), P.ArrayColumn("Timefactor", 5, "Time")],
                            [[struct.pack("<i", 1000), struct.pack("<f", 500.0)]])]
    lm.phys = [P.PhysRecord(3, 22, 42914, False, True, 0.7, 0.3, 0.0, "Floor", False, True, False, 0, 0, "srf",
                            (0, 0, 0), [], [(np.zeros(3, np.float32), 2.0)], [8001])]
    lm.meshes = [P.Mesh(42914, 8001, 1, np.eye(4, dtype=np.float32), np.zeros((3, 3), np.float32),
                        np.array([[0, 1, 2]], np.uint32))]
    lm.ivp = [P.IvpObject(42914, 2, True, 0.0, b"m" * 32)]
    lm.depth = [(1, np.zeros(3, np.float32), np.ones(3, np.float32))]
    lm.watch = [(5, "All_P_Extra_Point")]
    return lm


def test_level_message_round_trip_and_hand_layout():
    lm = _level()
    data = P.build_level_message(lm)
    back, n = P.parse_level_message(data)
    assert n == len(data) and back.frames_done == 104
    e = back.entities[0]
    assert (e.ckid, e.name, e.class_id, e.contact_id, e.coll_id, e.floor) == (42914, "A01_Floor_02", 41, 1, 1, b"Faces;FALSE")
    assert e.matrix[3].tolist() == [9, 10, 11]
    assert back.groups[0].name == "Phys_Floors" and back.groups[0].members.tolist() == [42914]
    a = back.arrays[0]
    assert a.name == "Energy" and a.columns[1].param_type == "Time" and P.decode_cell(a.cells[0][1], 5, "Time") == 500.0
    p = back.phys[0]
    assert (p.seq, p.frame, p.fixed, p.collision_group, p.concave) == (3, 22, True, "Floor", [8001])
    assert p.balls[0][1] == 2.0
    assert back.meshes[0].faces.tolist() == [[0, 1, 2]] and back.ivp[0].material == b"m" * 32
    assert back.watch == [(5, "All_P_Extra_Point")]
    # hand layout of the ENTS item: u32 ckid, u32 name idx, u32 class, u32 parent, u8 vis, 12f, i32, i32, u16 len, bytes
    msg = P.split_message(data, 0, P.LVL_MAGIC)
    ents = msg.get("ENTS").payload
    strs = msg.get("STRS").payload
    n_str = struct.unpack_from("<I", strs)[0]
    names = []
    pos = 4
    for _ in range(n_str):
        ln = struct.unpack_from("<H", strs, pos)[0]
        names.append(strs[pos + 2:pos + 2 + ln].decode())
        pos += 2 + ln
    cnt, ck, name_i, cls, parent, vis = struct.unpack_from("<IIIIIB", ents)
    assert (cnt, ck, names[name_i], cls, parent, vis) == (1, 42914, "A01_Floor_02", 41, 0, 1)
    cid, coll, flen = struct.unpack_from("<iiH", ents, 21 + 48)
    assert (cid, coll, flen) == (1, 1, 11)


def test_level_stream_leaves_partial_message():
    a = P.build_level_message(_level())
    b = P.build_level_message(P.LevelMessage(frames_done=200, phys=[P.PhysRecord(4, 200, 1, True)]))
    msgs, used = P.parse_level_stream(a + b[:30])
    assert len(msgs) == 1 and used == len(a)
    msgs, used = P.parse_level_stream(a + b)
    assert [m.frames_done for m in msgs] == [104, 200] and msgs[1].phys[0].unphysicalize


def test_decode_cell_types():
    m = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert np.array_equal(P.decode_cell(m.tobytes(), P.ARR_PARAMETER), m)
    assert P.decode_cell(struct.pack("<i", 1), P.ARR_PARAMETER, "Boolean") is True
    assert P.decode_cell(b"Ball_Wood", P.ARR_STRING) == "Ball_Wood"
    assert P.decode_cell(struct.pack("<f", 0.5), P.ARR_FLOAT) == 0.5


def test_mod_extension_sections_hand_packed():
    """TIME appended fields, SCOR and SREN as BallrainIO's Agent.cpp writes them (field by field)."""
    time_payload = struct.pack("<dddffffdfBI", 0, 0, 0, 0, 0, 0, 3.75, 100.0, 100.0, 1, 1) + struct.pack("<fII", 99.5, 812, 2)
    hs = struct.pack("<B6BB6IIii", 1, 1, 1, 1, 1, 5, 1, 1, 950, 3, 1000, 3, 0, 200, 300, 1550, 1850)
    scor = hs + struct.pack("<III", 2, 700, 900)
    sren = struct.pack("<dffIBBBBdf", 6553.5, 6553.25, 6553.5, 3, 1, 1, 0, 0, 6538.0, 6538.0) + hs
    o = P.parse_observation(_message(812, [_section("TIME", time_payload), _section("SCOR", scor), _section("SREN", sren)]))
    assert o.time["delta_ms"] == 3.75 and o.time_ext["frame_counter"] == 812 and o.time_ext["tail_frames"] == 2
    assert o.scor["energy"].tolist() == [950, 3, 1000, 3, 0, 200] and o.scor["hs"] == 1850 and o.scor["last_ballnav"] == 900
    assert o.sren["sr_double"] == 6553.5 and o.sren["sr_double_at_finish"] == 6538.0 and o.sren["level_finished"] == 1
    assert o.sren["hs_valid"] == 1 and o.sren["hs"] == 1850 and o.sren["bml_hs"] == 1550


def test_level_extensions_and_empty_entity_record():
    lm = _level()
    lm.entity_meshes = [(42914, 8001)]
    lm.tables = [{"cams": [1, 2, 3, 4, 5], "contact_bbs": [77, 78]}]
    lm.entities.append(P.Entity(0, None, 0, 0, False, np.zeros((4, 3), np.float32), -1, -1, b""))
    back, _ = P.parse_level_message(P.build_level_message(lm))
    assert back.entity_meshes == [(42914, 8001)] and back.tables[0]["contact_bbs"] == [77, 78]
    assert back.entities[1].ckid == 0 and back.entities[1].name is None
    from ballrain.state import LevelData
    ld = LevelData()
    ld.merge(back)
    assert 0 not in ld.entities and ld.by_name["A01_Floor_02"] == 42914
