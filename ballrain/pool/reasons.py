"""Rejection / revocation reason codes of the pool, with the Chinese text the controller shows the player verbatim.

Every refusal the gateway sends (a rejected chunk, a refused hello / task, a revoked token, a refused login) carries
{"code": <Reason value>, "message": <rendered Chinese text>, "params": {...}}; the controller prints "message" as is
(docs/volunteer-compute.md 11.6) and may use "code" for its own logic (e.g. stop asking for tasks after a revocation).
Templates take only the named parameters below; render() never formats anything else into them.
"""
from __future__ import annotations

from enum import Enum


class Reason(str, Enum):
    # ---- login / eligibility (forum account; docs/volunteer-compute.md 11.2)
    ACCOUNT_TOO_NEW = "account_too_new"
    EMAIL_NOT_VERIFIED = "email_not_verified"
    TOO_FEW_REPLIES = "too_few_replies"
    LOGIN_FAILED = "login_failed"
    OAUTH_STATE_MISMATCH = "oauth_state_mismatch"
    # ---- token
    TOKEN_MISSING = "token_missing"
    TOKEN_INVALID = "token_invalid"
    TOKEN_REVOKED = "token_revoked"  # by an admin (free-text note in params.note)
    TOKEN_REVOKED_REPLAY = "token_revoked_replay"  # automatic: a replay mismatch
    # ---- client / engine / game files
    PROTOCOL_UNSUPPORTED = "protocol_unsupported"
    CLIENT_TOO_OLD = "client_too_old"
    ENGINE_UNKNOWN = "engine_unknown"
    ENGINE_MISMATCH = "engine_mismatch"
    GAME_FILES_MISMATCH = "game_files_mismatch"
    TOO_MANY_INSTANCES = "too_many_instances"
    SESSION_EXPIRED = "session_expired"  # unknown / expired X-Pool-Session: the controller sends hello again
    # ---- tasks / chunks
    NO_TASK = "no_task"
    TASK_UNKNOWN = "task_unknown"
    TASK_EXPIRED = "task_expired"
    TASK_NOT_YOURS = "task_not_yours"
    CHUNK_DUPLICATE = "chunk_duplicate"
    CHUNK_OUT_OF_ORDER = "chunk_out_of_order"
    STALE_POLICY = "stale_policy"
    POLICY_MISMATCH = "policy_mismatch"
    BAD_MESSAGE = "bad_message"
    BAD_CHUNK = "bad_chunk"
    CHUNK_TOO_LARGE = "chunk_too_large"
    CHAIN_BROKEN = "chain_broken"
    LOGP_INVALID = "logp_invalid"
    REPLAY_MISMATCH = "replay_mismatch"
    REPLAY_REWARD_MISMATCH = "replay_reward_mismatch"
    REPLAY_LOGP_MISMATCH = "replay_logp_mismatch"
    REPLAY_VALUE_MISMATCH = "replay_value_mismatch"
    REPLAY_ACTION_MISMATCH = "replay_action_mismatch"  # the action is not the task's sample_seed draw
    REPLAY_FAILED = "replay_failed"  # the verifier could not run the replay (not the client's fault; chunk held)
    # ---- rate / server
    RATE_LIMITED = "rate_limited"
    SERVER_BUSY = "server_busy"
    INTERNAL_ERROR = "internal_error"


# Chinese user-facing text; {name} placeholders are filled by render() from params (missing ones read "?")
MESSAGES: dict[Reason, str] = {
    Reason.ACCOUNT_TOO_NEW: "论坛账号注册未满 {min_days} 天（已注册 {days} 天），暂时不能领取算力池令牌。",
    Reason.EMAIL_NOT_VERIFIED: "论坛账号的邮箱尚未验证，验证后再登录即可领取令牌。",
    Reason.TOO_FEW_REPLIES: "论坛账号的回复数不足 {min_replies} 条（当前 {replies} 条），暂时不能领取令牌。",
    Reason.LOGIN_FAILED: "论坛登录失败，请重试。（{detail}）",
    Reason.OAUTH_STATE_MISMATCH: "登录请求已过期或不匹配，请回到登录页重新开始。",
    Reason.TOKEN_MISSING: "请求没有带算力池令牌，请先在主控程序里登录。",
    Reason.TOKEN_INVALID: "算力池令牌无效，请重新登录。",
    Reason.TOKEN_REVOKED: "你的算力池令牌已被管理员吊销。说明：{note}",
    Reason.TOKEN_REVOKED_REPLAY: "你的算力池令牌已被停用：任务 {task_id} 的第 {chunk} 段数据在第 {step} 步重放结果与提交不一致。"
                                 "这个令牌还没进入训练的数据已全部隔离。",
    Reason.PROTOCOL_UNSUPPORTED: "主控程序的协议版本 {got} 不受支持（服务端支持 {supported}），请更新主控程序。",
    Reason.CLIENT_TOO_OLD: "主控程序版本 {got} 过旧（最低 {min_version}），请更新后再试。",
    Reason.ENGINE_UNKNOWN: "无头引擎（{engine}，sha256 {sha}）不是已发布的版本，请重新下载主控程序附带的引擎。",
    Reason.ENGINE_MISMATCH: "任务要求 {want} 引擎，而本机运行的是 {got} 引擎。",
    Reason.GAME_FILES_MISMATCH: "游戏文件校验不通过：{file} 与原版不一致（sha256 {got}）。请检查 Ballance 安装目录。",
    Reason.TOO_MANY_INSTANCES: "同时运行的实例数超过上限 {max_instances}，请在主控程序里调低实例数。",
    Reason.SESSION_EXPIRED: "会话已失效，主控程序会自动重新连接。",
    Reason.NO_TASK: "当前没有可分配的任务，请稍后再试。",
    Reason.TASK_UNKNOWN: "任务 {task_id} 不存在或已被服务端撤回。",
    Reason.TASK_EXPIRED: "任务 {task_id} 已超时（期限 {deadline_s} 秒），数据不再接收。",
    Reason.TASK_NOT_YOURS: "任务 {task_id} 不是分配给这个令牌的。",
    Reason.CHUNK_DUPLICATE: "任务 {task_id} 的第 {chunk} 段数据已经提交过。",
    Reason.CHUNK_OUT_OF_ORDER: "任务 {task_id} 的数据段顺序不对：收到第 {chunk} 段，应为第 {expected} 段。",
    Reason.STALE_POLICY: "数据使用的策略版本 {got} 已过时（服务端要求 {want}），这段数据未被接收。",
    Reason.POLICY_MISMATCH: "数据使用的策略版本 {got} 与任务指定的 {want} 不一致。",
    Reason.BAD_MESSAGE: "请求格式错误：{detail}",
    Reason.BAD_CHUNK: "数据段格式错误：{detail}",
    Reason.CHUNK_TOO_LARGE: "数据段太大（{size} 字节，上限 {limit} 字节）。",
    Reason.CHAIN_BROKEN: "任务 {task_id} 的第 {chunk} 段观测哈希链在第 {step} 步断开。",
    Reason.LOGP_INVALID: "任务 {task_id} 的第 {chunk} 段在第 {step} 步的 behaviour log-prob 无效。",
    Reason.REPLAY_MISMATCH: "任务 {task_id} 的第 {chunk} 段数据在第 {step} 步重放结果与提交不一致。",
    Reason.REPLAY_REWARD_MISMATCH: "任务 {task_id} 的第 {chunk} 段数据在第 {step} 步重放得到的奖励与提交不一致。",
    Reason.REPLAY_LOGP_MISMATCH: "任务 {task_id} 的第 {chunk} 段数据在第 {step} 步的 log-prob 与策略重算结果不一致。",
    Reason.REPLAY_VALUE_MISMATCH: "任务 {task_id} 的第 {chunk} 段数据在第 {step} 步的价值估计与策略重算结果不一致。",
    Reason.REPLAY_ACTION_MISMATCH: "任务 {task_id} 的第 {chunk} 段数据在第 {step} 步的动作不是按任务的采样种子从策略抽出的。",
    Reason.REPLAY_FAILED: "服务端暂时无法重放任务 {task_id}，数据先保留，稍后再验证（不影响你的令牌）。",
    Reason.RATE_LIMITED: "请求太频繁，请 {retry_s} 秒后再试。",
    Reason.SERVER_BUSY: "服务端繁忙，请稍后再试。",
    Reason.INTERNAL_ERROR: "服务端内部错误，请稍后再试。",
}

# Reasons after which the controller must stop working with its token (show the message, ask for a new login)
TOKEN_FATAL = frozenset({Reason.TOKEN_MISSING, Reason.TOKEN_INVALID, Reason.TOKEN_REVOKED, Reason.TOKEN_REVOKED_REPLAY})
# Reasons that come with an HTTP status other than 400 / 403 (the gateway's mapping; default 400 for chunk / message
# errors, 403 for eligibility / token)
HTTP_STATUS = {Reason.RATE_LIMITED: 429, Reason.SERVER_BUSY: 503, Reason.INTERNAL_ERROR: 500, Reason.NO_TASK: 200,
               Reason.TASK_UNKNOWN: 404, Reason.TASK_EXPIRED: 410, Reason.CHUNK_DUPLICATE: 409,
               Reason.CHUNK_OUT_OF_ORDER: 409, Reason.CHUNK_TOO_LARGE: 413, Reason.SESSION_EXPIRED: 409, Reason.REPLAY_FAILED: 202,
               **{r: 403 for r in (Reason.ACCOUNT_TOO_NEW, Reason.EMAIL_NOT_VERIFIED, Reason.TOO_FEW_REPLIES,
                                   Reason.TOKEN_REVOKED, Reason.TOKEN_REVOKED_REPLAY, Reason.TASK_NOT_YOURS)},
               **{r: 401 for r in (Reason.TOKEN_MISSING, Reason.TOKEN_INVALID, Reason.LOGIN_FAILED,
                                   Reason.OAUTH_STATE_MISMATCH)}}

_PARAM_MAX = 200  # characters per rendered parameter


class _Params(dict):
    def __missing__(self, key):
        return "?"


def _clean(v) -> str:
    s = str(v)
    s = "".join(ch for ch in s if ch.isprintable())
    return s[:_PARAM_MAX]


def render(code: Reason | str, **params) -> str:
    """The Chinese message of a reason with its parameters (non-printable characters dropped, each cut to 200)."""
    r = Reason(code)
    p = _Params({k: _clean(v) for k, v in params.items()})
    return MESSAGES[r].format_map(p)


def payload(code: Reason | str, **params) -> dict:
    """The reason object of a refusal: {"code", "message", "params"} (params: str / int / float / bool only)."""
    r = Reason(code)
    clean = {}
    for k, v in params.items():
        if isinstance(v, bool) or isinstance(v, (int, float)):
            clean[k] = v
        else:
            clean[k] = _clean(v)
    return {"code": r.value, "message": render(r, **params), "params": clean}


def http_status(code: Reason | str) -> int:
    return HTTP_STATUS.get(Reason(code), 400)
