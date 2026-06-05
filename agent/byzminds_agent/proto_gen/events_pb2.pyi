from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class EventEnvelope(_message.Message):
    __slots__ = ("emitter_pubkey", "tick", "sequence_per_ledger", "signature", "event_type", "payload")
    EMITTER_PUBKEY_FIELD_NUMBER: _ClassVar[int]
    TICK_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_PER_LEDGER_FIELD_NUMBER: _ClassVar[int]
    SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    EVENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    emitter_pubkey: bytes
    tick: int
    sequence_per_ledger: int
    signature: bytes
    event_type: str
    payload: bytes
    def __init__(self, emitter_pubkey: _Optional[bytes] = ..., tick: _Optional[int] = ..., sequence_per_ledger: _Optional[int] = ..., signature: _Optional[bytes] = ..., event_type: _Optional[str] = ..., payload: _Optional[bytes] = ...) -> None: ...

class SigningInput(_message.Message):
    __slots__ = ("emitter_pubkey", "tick", "sequence_per_ledger", "event_type", "payload")
    EMITTER_PUBKEY_FIELD_NUMBER: _ClassVar[int]
    TICK_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_PER_LEDGER_FIELD_NUMBER: _ClassVar[int]
    EVENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    emitter_pubkey: bytes
    tick: int
    sequence_per_ledger: int
    event_type: str
    payload: bytes
    def __init__(self, emitter_pubkey: _Optional[bytes] = ..., tick: _Optional[int] = ..., sequence_per_ledger: _Optional[int] = ..., event_type: _Optional[str] = ..., payload: _Optional[bytes] = ...) -> None: ...

class Speak(_message.Message):
    __slots__ = ("channel_id", "content")
    CHANNEL_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    channel_id: str
    content: str
    def __init__(self, channel_id: _Optional[str] = ..., content: _Optional[str] = ...) -> None: ...

class Vote(_message.Message):
    __slots__ = ("option",)
    OPTION_FIELD_NUMBER: _ClassVar[int]
    option: str
    def __init__(self, option: _Optional[str] = ...) -> None: ...

class OpenChannelReq(_message.Message):
    __slots__ = ("proposed_members",)
    PROPOSED_MEMBERS_FIELD_NUMBER: _ClassVar[int]
    proposed_members: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, proposed_members: _Optional[_Iterable[str]] = ...) -> None: ...

class CloseChannelReq(_message.Message):
    __slots__ = ("channel_id",)
    CHANNEL_ID_FIELD_NUMBER: _ClassVar[int]
    channel_id: str
    def __init__(self, channel_id: _Optional[str] = ...) -> None: ...

class RequestCapability(_message.Message):
    __slots__ = ("cap_id", "justification")
    CAP_ID_FIELD_NUMBER: _ClassVar[int]
    JUSTIFICATION_FIELD_NUMBER: _ClassVar[int]
    cap_id: str
    justification: str
    def __init__(self, cap_id: _Optional[str] = ..., justification: _Optional[str] = ...) -> None: ...

class DropCapability(_message.Message):
    __slots__ = ("cap_id",)
    CAP_ID_FIELD_NUMBER: _ClassVar[int]
    cap_id: str
    def __init__(self, cap_id: _Optional[str] = ...) -> None: ...

class Yield(_message.Message):
    __slots__ = ("reason",)
    REASON_FIELD_NUMBER: _ClassVar[int]
    reason: str
    def __init__(self, reason: _Optional[str] = ...) -> None: ...

class DeclareIntent(_message.Message):
    __slots__ = ("content",)
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    content: str
    def __init__(self, content: _Optional[str] = ...) -> None: ...

class CogIndSnapshot(_message.Message):
    __slots__ = ("agent_id", "theta")
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    THETA_FIELD_NUMBER: _ClassVar[int]
    agent_id: str
    theta: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, agent_id: _Optional[str] = ..., theta: _Optional[_Iterable[float]] = ...) -> None: ...

class HandlerControlEvent(_message.Message):
    __slots__ = ("handler_rpc_name", "handler_request_bytes", "effective_tick")
    HANDLER_RPC_NAME_FIELD_NUMBER: _ClassVar[int]
    HANDLER_REQUEST_BYTES_FIELD_NUMBER: _ClassVar[int]
    EFFECTIVE_TICK_FIELD_NUMBER: _ClassVar[int]
    handler_rpc_name: str
    handler_request_bytes: bytes
    effective_tick: int
    def __init__(self, handler_rpc_name: _Optional[str] = ..., handler_request_bytes: _Optional[bytes] = ..., effective_tick: _Optional[int] = ...) -> None: ...

class TickTimeoutIncident(_message.Message):
    __slots__ = ("agent_id", "tick", "budget_nanos")
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    TICK_FIELD_NUMBER: _ClassVar[int]
    BUDGET_NANOS_FIELD_NUMBER: _ClassVar[int]
    agent_id: str
    tick: int
    budget_nanos: int
    def __init__(self, agent_id: _Optional[str] = ..., tick: _Optional[int] = ..., budget_nanos: _Optional[int] = ...) -> None: ...

class ElicitationRequest(_message.Message):
    __slots__ = ("agent_id", "tick", "action_event_type", "action_summary", "action_global_commit_seq")
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    TICK_FIELD_NUMBER: _ClassVar[int]
    ACTION_EVENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    ACTION_SUMMARY_FIELD_NUMBER: _ClassVar[int]
    ACTION_GLOBAL_COMMIT_SEQ_FIELD_NUMBER: _ClassVar[int]
    agent_id: str
    tick: int
    action_event_type: str
    action_summary: str
    action_global_commit_seq: int
    def __init__(self, agent_id: _Optional[str] = ..., tick: _Optional[int] = ..., action_event_type: _Optional[str] = ..., action_summary: _Optional[str] = ..., action_global_commit_seq: _Optional[int] = ...) -> None: ...

class ContextTruncation(_message.Message):
    __slots__ = ("agent_id", "tick", "channel_id", "dropped_count", "kept_count")
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    TICK_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_ID_FIELD_NUMBER: _ClassVar[int]
    DROPPED_COUNT_FIELD_NUMBER: _ClassVar[int]
    KEPT_COUNT_FIELD_NUMBER: _ClassVar[int]
    agent_id: str
    tick: int
    channel_id: str
    dropped_count: int
    kept_count: int
    def __init__(self, agent_id: _Optional[str] = ..., tick: _Optional[int] = ..., channel_id: _Optional[str] = ..., dropped_count: _Optional[int] = ..., kept_count: _Optional[int] = ...) -> None: ...

class MalformedSubmission(_message.Message):
    __slots__ = ("agent_id", "tick", "raw_output", "failure")
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    TICK_FIELD_NUMBER: _ClassVar[int]
    RAW_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    FAILURE_FIELD_NUMBER: _ClassVar[int]
    agent_id: str
    tick: int
    raw_output: bytes
    failure: str
    def __init__(self, agent_id: _Optional[str] = ..., tick: _Optional[int] = ..., raw_output: _Optional[bytes] = ..., failure: _Optional[str] = ...) -> None: ...
