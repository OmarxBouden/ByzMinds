import events_pb2 as _events_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class LedgerID(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LEDGER_ID_UNSPECIFIED: _ClassVar[LedgerID]
    LEDGER_ID_L_PUB: _ClassVar[LedgerID]
    LEDGER_ID_L_PRV: _ClassVar[LedgerID]
    LEDGER_ID_L_COG_IND: _ClassVar[LedgerID]
    LEDGER_ID_L_COG_ELI: _ClassVar[LedgerID]
    LEDGER_ID_L_CTRL: _ClassVar[LedgerID]
LEDGER_ID_UNSPECIFIED: LedgerID
LEDGER_ID_L_PUB: LedgerID
LEDGER_ID_L_PRV: LedgerID
LEDGER_ID_L_COG_IND: LedgerID
LEDGER_ID_L_COG_ELI: LedgerID
LEDGER_ID_L_CTRL: LedgerID

class CommittedEvent(_message.Message):
    __slots__ = ("envelope", "ledger_id", "ledger_channel_id", "global_commit_seq", "commit_unix_nanos", "prev_chain_hash", "chain_hash", "kernel_signature")
    ENVELOPE_FIELD_NUMBER: _ClassVar[int]
    LEDGER_ID_FIELD_NUMBER: _ClassVar[int]
    LEDGER_CHANNEL_ID_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_COMMIT_SEQ_FIELD_NUMBER: _ClassVar[int]
    COMMIT_UNIX_NANOS_FIELD_NUMBER: _ClassVar[int]
    PREV_CHAIN_HASH_FIELD_NUMBER: _ClassVar[int]
    CHAIN_HASH_FIELD_NUMBER: _ClassVar[int]
    KERNEL_SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    envelope: _events_pb2.EventEnvelope
    ledger_id: LedgerID
    ledger_channel_id: str
    global_commit_seq: int
    commit_unix_nanos: int
    prev_chain_hash: bytes
    chain_hash: bytes
    kernel_signature: bytes
    def __init__(self, envelope: _Optional[_Union[_events_pb2.EventEnvelope, _Mapping]] = ..., ledger_id: _Optional[_Union[LedgerID, str]] = ..., ledger_channel_id: _Optional[str] = ..., global_commit_seq: _Optional[int] = ..., commit_unix_nanos: _Optional[int] = ..., prev_chain_hash: _Optional[bytes] = ..., chain_hash: _Optional[bytes] = ..., kernel_signature: _Optional[bytes] = ...) -> None: ...

class ChainInput(_message.Message):
    __slots__ = ("envelope", "ledger_id", "ledger_channel_id", "global_commit_seq", "prev_chain_hash")
    ENVELOPE_FIELD_NUMBER: _ClassVar[int]
    LEDGER_ID_FIELD_NUMBER: _ClassVar[int]
    LEDGER_CHANNEL_ID_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_COMMIT_SEQ_FIELD_NUMBER: _ClassVar[int]
    PREV_CHAIN_HASH_FIELD_NUMBER: _ClassVar[int]
    envelope: _events_pb2.EventEnvelope
    ledger_id: LedgerID
    ledger_channel_id: str
    global_commit_seq: int
    prev_chain_hash: bytes
    def __init__(self, envelope: _Optional[_Union[_events_pb2.EventEnvelope, _Mapping]] = ..., ledger_id: _Optional[_Union[LedgerID, str]] = ..., ledger_channel_id: _Optional[str] = ..., global_commit_seq: _Optional[int] = ..., prev_chain_hash: _Optional[bytes] = ...) -> None: ...
