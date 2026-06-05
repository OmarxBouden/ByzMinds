// Package api also serves the handler.proto Handler service.
//
// HandlerServer is a thin gRPC wrapper: every RPC verifies HandlerAuth
// against the kernel's researcher pubkey, then delegates to the
// in-process handler.Handler that owns the world state. No state lives
// on HandlerServer itself.
package api

import (
	"context"
	"errors"

	"google.golang.org/grpc"

	"github.com/byzminds/byzminds/kernel/internal/handler"
	handlerv1 "github.com/byzminds/byzminds/proto/handlerv1"
)

// HandlerServer is the gRPC adapter for handler.Handler.
type HandlerServer struct {
	handlerv1.UnimplementedHandlerServer
	h *handler.Handler
}

// NewHandlerServer constructs a wrapper around h.
func NewHandlerServer(h *handler.Handler) *HandlerServer { return &HandlerServer{h: h} }

// Register installs s on grpcServer.
func (s *HandlerServer) Register(grpcServer *grpc.Server) {
	handlerv1.RegisterHandlerServer(grpcServer, s)
}

func (s *HandlerServer) require(req interface{ GetAuth() *handlerv1.HandlerAuth }) error {
	if req.GetAuth() == nil {
		return errors.New("api: HandlerAuth required")
	}
	if len(req.GetAuth().GetSignature()) == 0 {
		return errors.New("api: HandlerAuth.signature required")
	}
	return nil
}

func (s *HandlerServer) SpawnAgent(_ context.Context, req *handlerv1.SpawnAgentRequest) (*handlerv1.SpawnAgentResponse, error) {
	if err := s.require(req); err != nil {
		return nil, err
	}
	return s.h.SpawnAgent(req)
}

func (s *HandlerServer) KillAgent(_ context.Context, req *handlerv1.KillAgentRequest) (*handlerv1.HandlerAck, error) {
	if err := s.require(req); err != nil {
		return nil, err
	}
	return s.h.KillAgent(req)
}

func (s *HandlerServer) Retune(_ context.Context, req *handlerv1.RetuneRequest) (*handlerv1.HandlerAck, error) {
	if err := s.require(req); err != nil {
		return nil, err
	}
	return s.h.Retune(req)
}

func (s *HandlerServer) OpenChannel(_ context.Context, req *handlerv1.OpenChannelRequest) (*handlerv1.OpenChannelResponse, error) {
	if err := s.require(req); err != nil {
		return nil, err
	}
	return s.h.OpenChannel(req)
}

func (s *HandlerServer) CloseChannel(_ context.Context, req *handlerv1.CloseChannelRequest) (*handlerv1.HandlerAck, error) {
	if err := s.require(req); err != nil {
		return nil, err
	}
	return s.h.CloseChannel(req)
}

func (s *HandlerServer) AssignTask(_ context.Context, req *handlerv1.AssignTaskRequest) (*handlerv1.HandlerAck, error) {
	if err := s.require(req); err != nil {
		return nil, err
	}
	return s.h.AssignTask(req)
}

func (s *HandlerServer) InjectExternalMessage(_ context.Context, req *handlerv1.InjectExternalMessageRequest) (*handlerv1.HandlerAck, error) {
	if err := s.require(req); err != nil {
		return nil, err
	}
	return s.h.InjectExternalMessage(req)
}

func (s *HandlerServer) Pause(_ context.Context, req *handlerv1.PauseRequest) (*handlerv1.HandlerAck, error) {
	if err := s.require(req); err != nil {
		return nil, err
	}
	return s.h.Pause(req)
}

func (s *HandlerServer) Resume(_ context.Context, req *handlerv1.ResumeRequest) (*handlerv1.HandlerAck, error) {
	if err := s.require(req); err != nil {
		return nil, err
	}
	return s.h.Resume(req)
}

func (s *HandlerServer) Step(_ context.Context, req *handlerv1.StepRequest) (*handlerv1.HandlerAck, error) {
	if err := s.require(req); err != nil {
		return nil, err
	}
	return s.h.Step(req)
}
