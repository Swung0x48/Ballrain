"""Client capabilities (server side): which policies a client may get tasks of, and what the player is told otherwise.

The gateway gates by capability, not by version number (the user's principle: upgrade the volunteers' program as
rarely as possible). A client lists what it can run in hello.capabilities (absent = none); a policy lists what its
tasks need in manifest.requires (absent = nothing: every client). task() hands a policy's tasks only to sessions that
have all of its requires, so an old client keeps working on every policy that needs nothing new.

A client that lacks something gets no_task with the EXISTING reason code client_too_old (old clients check reason codes
against their own list, so a new code would break them), whose message (shown verbatim by every client) says what was
added, why the running program cannot take part, and what to update to. Messages are built here, server side.
"""
from __future__ import annotations

from .reasons import Reason, payload

# capability -> (the first client version that has it, what it is: one Chinese sentence for the player)
CAPABILITIES = {
    "task_format3": ("0.3.0", "任务格式 3：训练器的完整开局分布（复习开局、静止开局、每条命的计数器随机化、提前重生规则）"),
    "branch_v1": ("0.4.0", "第 2 关小节 4 的石头分支：策略在这里改走另一条参考线（换成石球的路线），并多了从这条路线上出发的静止开局"),
    "reach_f7": ("0.4.0", "f7 观测：特征末尾多了一块“可达性”信息（球往各个方向能不能停住、会不会掉下去、能落到哪里），"
                          "客户端要用和训练器相同的算法和场数据把它算出来"),
}
WHY = ("当前运行的主控程序不认识这些新内容：它仍会按旧规则跑任务，算出的观测和奖励与训练器不一致，验证不会通过，"
       "所以服务端暂时不给它发这个训练策略的任务。已经交上来的数据和你的令牌都不受影响。")
MAX_MESSAGE = 1000  # protocol REASON.message


def missing(requires, capabilities) -> list:
    """The policy's requires the client does not have (in the policy's order)."""
    have = set(capabilities or ())
    return [r for r in (requires or ()) if r not in have]


def refusal(client_version: str, lacking: list) -> dict:
    """The reason object of a no_task for a client lacking capabilities: code client_too_old (params got / min_version
    as the code's own template has them, plus the missing capability names), message explaining what changed."""
    lacking = list(lacking)
    need = max((CAPABILITIES.get(c, ("", ""))[0] for c in lacking), default="",
               key=lambda v: tuple(int(x) if x.isdigit() else 0 for x in v.split("."))) or "最新版本"
    what = "；".join(CAPABILITIES[c][1] if c in CAPABILITIES else f"新功能 {c}" for c in lacking)
    msg = (f"主控程序版本 {client_version} 不支持训练现在用到的新内容：{what}。{WHY}"
           f"请把主控程序更新到 {need} 或更新的版本后再试。")
    if len(msg) > MAX_MESSAGE:
        msg = msg[:MAX_MESSAGE - 1] + "…"
    r = payload(Reason.CLIENT_TOO_OLD, got=client_version, min_version=need, missing=",".join(lacking)[:200])
    r["message"] = msg
    return r
