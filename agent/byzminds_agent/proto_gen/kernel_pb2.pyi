import events_pb2 as _events_pb2
import ledger_pb2 as _ledger_pb2
import view_pb2 as _view_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CommitReceipt(_message.Message):
    __slots__ = ("committed", "ledger_id", "ledger_channel_id", "sequence_per_ledger", "global_commit_seq", "chain_hash", "kernel_signature", "rejection_reason")
    COMMITTED_FIELD_NUMBER: _ClassVar[int]
    LEDGER_ID_FIELD_NUMBER: _ClassVar[int]
    LEDGER_CHANNEL_ID_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_PER_LEDGER_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_COMMIT_SEQ_FIELD_NUMBER: _ClassVar[int]
    CHAIN_HASH_FIELD_NUMBER: _ClassVar[int]
    KERNEL_SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    REJECTION_REASON_FIELD_NUMBER: _ClassVar[int]
    committed: bool
    ledger_id: _ledger_pb2.LedgerID
    ledger_channel_id: str
    sequence_per_ledger: int
    global_commit_seq: int
    chain_hash: bytes
    kernel_signature: bytes
    rejection_reason: str
    def __init__(self, committed: bool = ..., ledger_id: _Optional[_Union[_ledger_pb2.LedgerID, str]] = ..., ledger_channel_id: _Optional[str] = ..., sequence_per_ledger: _Optional[int] = ..., global_commit_seq: _Optional[int] = ..., chain_hash: _Optional[bytes] = ..., kernel_signature: _Optional[bytes] = ..., rejection_reason: _Optional[str] = ...) -> None: ...

class ViewRequest(_message.Message):
    __slots__ = ("reader_pubkey", "from_tick", "signature")
    READER_PUBKEY_FIELD_NUMBER: _ClassVar[int]
    FROM_TICK_FIELD_NUMBER: _ClassVar[int]
    SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    reader_pubkey: bytes
    from_tick: int
    signature: bytes
    def __init__(self, reader_pubkey: _Optional[bytes] = ..., from_tick: _Optional[int] = ..., signature: _Optional[bytes] = ...) -> None: ...

class EventView(_message.Message):
    __slots__ = ("event",)
    EVENT_FIELD_NUMBER: _ClassVar[int]
    event: _ledger_pb2.CommittedEvent
    def __init__(self, event: _Optional[_Union[_ledger_pb2.CommittedEvent, _Mapping]] = ...) -> None: ...

class SubscribeRequest(_message.Message):
    __slots__ = ("agent_pubkey", "from_tick", "signature")
    AGENT_PUBKEY_FIELD_NUMBER: _ClassVar[int]
    FROM_TICK_FIELD_NUMBER: _ClassVar[int]
    SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    agent_pubkey: bytes
    from_tick: int
    signature: bytes
    def __init__(self, agent_pubkey: _Optional[bytes] = ..., from_tick: _Optional[int] = ..., signature: _Optional[bytes] = ...) -> None: ...
