import events_pb2 as _events_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class View(_message.Message):
    __slots__ = ("agent_id", "tick", "channel_memberships", "loaded_capabilities", "scenario", "task_artifact", "phase", "round", "total_rounds", "channel_histories", "external_messages", "available_tools", "elicit_request")
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    TICK_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_MEMBERSHIPS_FIELD_NUMBER: _ClassVar[int]
    LOADED_CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    SCENARIO_FIELD_NUMBER: _ClassVar[int]
    TASK_ARTIFACT_FIELD_NUMBER: _ClassVar[int]
    PHASE_FIELD_NUMBER: _ClassVar[int]
    ROUND_FIELD_NUMBER: _ClassVar[int]
    TOTAL_ROUNDS_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_HISTORIES_FIELD_NUMBER: _ClassVar[int]
    EXTERNAL_MESSAGES_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_TOOLS_FIELD_NUMBER: _ClassVar[int]
    ELICIT_REQUEST_FIELD_NUMBER: _ClassVar[int]
    agent_id: str
    tick: int
    channel_memberships: _containers.RepeatedScalarFieldContainer[str]
    loaded_capabilities: _containers.RepeatedScalarFieldContainer[str]
    scenario: ScenarioRef
    task_artifact: str
    phase: str
    round: int
    total_rounds: int
    channel_histories: _containers.RepeatedCompositeFieldContainer[ChannelHistory]
    external_messages: _containers.RepeatedCompositeFieldContainer[ExternalMsg]
    available_tools: _containers.RepeatedScalarFieldContainer[str]
    elicit_request: _events_pb2.ElicitationRequest
    def __init__(self, agent_id: _Optional[str] = ..., tick: _Optional[int] = ..., channel_memberships: _Optional[_Iterable[str]] = ..., loaded_capabilities: _Optional[_Iterable[str]] = ..., scenario: _Optional[_Union[ScenarioRef, _Mapping]] = ..., task_artifact: _Optional[str] = ..., phase: _Optional[str] = ..., round: _Optional[int] = ..., total_rounds: _Optional[int] = ..., channel_histories: _Optional[_Iterable[_Union[ChannelHistory, _Mapping]]] = ..., external_messages: _Optional[_Iterable[_Union[ExternalMsg, _Mapping]]] = ..., available_tools: _Optional[_Iterable[str]] = ..., elicit_request: _Optional[_Union[_events_pb2.ElicitationRequest, _Mapping]] = ...) -> None: ...

class ScenarioRef(_message.Message):
    __slots__ = ("scenario_name", "scenario_yaml_hash")
    SCENARIO_NAME_FIELD_NUMBER: _ClassVar[int]
    SCENARIO_YAML_HASH_FIELD_NUMBER: _ClassVar[int]
    scenario_name: str
    scenario_yaml_hash: str
    def __init__(self, scenario_name: _Optional[str] = ..., scenario_yaml_hash: _Optional[str] = ...) -> None: ...

class ChannelHistory(_message.Message):
    __slots__ = ("channel_id", "messages")
    CHANNEL_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    channel_id: str
    messages: _containers.RepeatedCompositeFieldContainer[Message]
    def __init__(self, channel_id: _Optional[str] = ..., messages: _Optional[_Iterable[_Union[Message, _Mapping]]] = ...) -> None: ...

class Message(_message.Message):
    __slots__ = ("sender_id", "content", "tick", "global_commit_seq")
    SENDER_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    TICK_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_COMMIT_SEQ_FIELD_NUMBER: _ClassVar[int]
    sender_id: str
    content: str
    tick: int
    global_commit_seq: int
    def __init__(self, sender_id: _Optional[str] = ..., content: _Optional[str] = ..., tick: _Optional[int] = ..., global_commit_seq: _Optional[int] = ...) -> None: ...

class ExternalMsg(_message.Message):
    __slots__ = ("claimed_source", "content", "inject_tick")
    CLAIMED_SOURCE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    INJECT_TICK_FIELD_NUMBER: _ClassVar[int]
    claimed_source: str
    content: str
    inject_tick: int
    def __init__(self, claimed_source: _Optional[str] = ..., content: _Optional[str] = ..., inject_tick: _Optional[int] = ...) -> None: ...
