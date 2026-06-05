from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class HandlerAuth(_message.Message):
    __slots__ = ("caller_pubkey", "signature")
    CALLER_PUBKEY_FIELD_NUMBER: _ClassVar[int]
    SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    caller_pubkey: bytes
    signature: bytes
    def __init__(self, caller_pubkey: _Optional[bytes] = ..., signature: _Optional[bytes] = ...) -> None: ...

class HandlerAck(_message.Message):
    __slots__ = ("control_global_commit_seq", "l_ctrl_chain_hash", "effective_tick")
    CONTROL_GLOBAL_COMMIT_SEQ_FIELD_NUMBER: _ClassVar[int]
    L_CTRL_CHAIN_HASH_FIELD_NUMBER: _ClassVar[int]
    EFFECTIVE_TICK_FIELD_NUMBER: _ClassVar[int]
    control_global_commit_seq: int
    l_ctrl_chain_hash: bytes
    effective_tick: int
    def __init__(self, control_global_commit_seq: _Optional[int] = ..., l_ctrl_chain_hash: _Optional[bytes] = ..., effective_tick: _Optional[int] = ...) -> None: ...

class SpawnAgentRequest(_message.Message):
    __slots__ = ("auth", "agent_id", "agent_pubkey", "role", "theta", "stub_policy")
    AUTH_FIELD_NUMBER: _ClassVar[int]
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    AGENT_PUBKEY_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    THETA_FIELD_NUMBER: _ClassVar[int]
    STUB_POLICY_FIELD_NUMBER: _ClassVar[int]
    auth: HandlerAuth
    agent_id: str
    agent_pubkey: bytes
    role: str
    theta: _containers.RepeatedScalarFieldContainer[float]
    stub_policy: str
    def __init__(self, auth: _Optional[_Union[HandlerAuth, _Mapping]] = ..., agent_id: _Optional[str] = ..., agent_pubkey: _Optional[bytes] = ..., role: _Optional[str] = ..., theta: _Optional[_Iterable[float]] = ..., stub_policy: _Optional[str] = ...) -> None: ...

class SpawnAgentResponse(_message.Message):
    __slots__ = ("agent_id", "spawn_tick")
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    SPAWN_TICK_FIELD_NUMBER: _ClassVar[int]
    agent_id: str
    spawn_tick: int
    def __init__(self, agent_id: _Optional[str] = ..., spawn_tick: _Optional[int] = ...) -> None: ...

class KillAgentRequest(_message.Message):
    __slots__ = ("auth", "agent_id")
    AUTH_FIELD_NUMBER: _ClassVar[int]
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    auth: HandlerAuth
    agent_id: str
    def __init__(self, auth: _Optional[_Union[HandlerAuth, _Mapping]] = ..., agent_id: _Optional[str] = ...) -> None: ...

class RetuneRequest(_message.Message):
    __slots__ = ("auth", "agent_id", "theta")
    AUTH_FIELD_NUMBER: _ClassVar[int]
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    THETA_FIELD_NUMBER: _ClassVar[int]
    auth: HandlerAuth
    agent_id: str
    theta: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, auth: _Optional[_Union[HandlerAuth, _Mapping]] = ..., agent_id: _Optional[str] = ..., theta: _Optional[_Iterable[float]] = ...) -> None: ...

class OpenChannelRequest(_message.Message):
    __slots__ = ("auth", "channel_id", "member_agent_ids")
    AUTH_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_ID_FIELD_NUMBER: _ClassVar[int]
    MEMBER_AGENT_IDS_FIELD_NUMBER: _ClassVar[int]
    auth: HandlerAuth
    channel_id: str
    member_agent_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, auth: _Optional[_Union[HandlerAuth, _Mapping]] = ..., channel_id: _Optional[str] = ..., member_agent_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class OpenChannelResponse(_message.Message):
    __slots__ = ("channel_id", "open_tick")
    CHANNEL_ID_FIELD_NUMBER: _ClassVar[int]
    OPEN_TICK_FIELD_NUMBER: _ClassVar[int]
    channel_id: str
    open_tick: int
    def __init__(self, channel_id: _Optional[str] = ..., open_tick: _Optional[int] = ...) -> None: ...

class CloseChannelRequest(_message.Message):
    __slots__ = ("auth", "channel_id")
    AUTH_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_ID_FIELD_NUMBER: _ClassVar[int]
    auth: HandlerAuth
    channel_id: str
    def __init__(self, auth: _Optional[_Union[HandlerAuth, _Mapping]] = ..., channel_id: _Optional[str] = ...) -> None: ...

class AssignTaskRequest(_message.Message):
    __slots__ = ("auth", "agent_ids", "task_kind", "task_blob")
    AUTH_FIELD_NUMBER: _ClassVar[int]
    AGENT_IDS_FIELD_NUMBER: _ClassVar[int]
    TASK_KIND_FIELD_NUMBER: _ClassVar[int]
    TASK_BLOB_FIELD_NUMBER: _ClassVar[int]
    auth: HandlerAuth
    agent_ids: _containers.RepeatedScalarFieldContainer[str]
    task_kind: str
    task_blob: bytes
    def __init__(self, auth: _Optional[_Union[HandlerAuth, _Mapping]] = ..., agent_ids: _Optional[_Iterable[str]] = ..., task_kind: _Optional[str] = ..., task_blob: _Optional[bytes] = ...) -> None: ...

class InjectExternalMessageRequest(_message.Message):
    __slots__ = ("auth", "agent_id", "claimed_source", "content")
    AUTH_FIELD_NUMBER: _ClassVar[int]
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    CLAIMED_SOURCE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    auth: HandlerAuth
    agent_id: str
    claimed_source: str
    content: str
    def __init__(self, auth: _Optional[_Union[HandlerAuth, _Mapping]] = ..., agent_id: _Optional[str] = ..., claimed_source: _Optional[str] = ..., content: _Optional[str] = ...) -> None: ...

class PauseRequest(_message.Message):
    __slots__ = ("auth",)
    AUTH_FIELD_NUMBER: _ClassVar[int]
    auth: HandlerAuth
    def __init__(self, auth: _Optional[_Union[HandlerAuth, _Mapping]] = ...) -> None: ...

class ResumeRequest(_message.Message):
    __slots__ = ("auth",)
    AUTH_FIELD_NUMBER: _ClassVar[int]
    auth: HandlerAuth
    def __init__(self, auth: _Optional[_Union[HandlerAuth, _Mapping]] = ...) -> None: ...

class StepRequest(_message.Message):
    __slots__ = ("auth", "ticks")
    AUTH_FIELD_NUMBER: _ClassVar[int]
    TICKS_FIELD_NUMBER: _ClassVar[int]
    auth: HandlerAuth
    ticks: int
    def __init__(self, auth: _Optional[_Union[HandlerAuth, _Mapping]] = ..., ticks: _Optional[int] = ...) -> None: ...
