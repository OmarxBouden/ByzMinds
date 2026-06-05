package handler

import (
	"fmt"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	"github.com/byzminds/byzminds/kernel/internal/crypto"
	handlerv1 "github.com/byzminds/byzminds/proto/handlerv1"
)

// canonicalRequestBytes marshals req deterministically with the Auth
// field's signature cleared, producing the bytes the researcher signs
// (and the server verifies).
//
// Implementation: clone req, find a top-level field named "auth" of
// message type HandlerAuth, blank its signature, marshal canonically.
func canonicalRequestBytes(req proto.Message) ([]byte, error) {
	if req == nil {
		return nil, fmt.Errorf("handler: nil request")
	}
	clone := proto.Clone(req)
	clearAuthSignature(clone.ProtoReflect())
	return crypto.CanonicalBytes(clone)
}

// clearAuthSignature walks the message's fields, finds the singular
// HandlerAuth field, and clears its signature.
func clearAuthSignature(m protoreflect.Message) {
	fields := m.Descriptor().Fields()
	for i := 0; i < fields.Len(); i++ {
		fd := fields.Get(i)
		if fd.Kind() != protoreflect.MessageKind {
			continue
		}
		if fd.Message().FullName() != (&handlerv1.HandlerAuth{}).ProtoReflect().Descriptor().FullName() {
			continue
		}
		if !m.Has(fd) {
			continue
		}
		auth := m.Mutable(fd).Message()
		authFields := auth.Descriptor().Fields()
		sigField := authFields.ByName("signature")
		if sigField != nil {
			auth.Clear(sigField)
		}
	}
}

// SignRequest computes the HandlerAuth signature for req and writes it
// into req.Auth.Signature. Exported because the scenario loader uses it
// to sign requests that pass through the gRPC layer in tests.
//
// req must have a singular "auth" HandlerAuth field; this helper takes
// the field pointer via reflection.
func SignRequest(req proto.Message, callerPubkey, callerPriv []byte) error {
	authFD := req.ProtoReflect().Descriptor().Fields().ByName("auth")
	if authFD == nil || authFD.Kind() != protoreflect.MessageKind {
		return fmt.Errorf("handler: request has no auth field")
	}
	auth := req.ProtoReflect().Mutable(authFD).Message()
	pkFD := auth.Descriptor().Fields().ByName("caller_pubkey")
	sigFD := auth.Descriptor().Fields().ByName("signature")
	if pkFD == nil || sigFD == nil {
		return fmt.Errorf("handler: HandlerAuth missing expected fields")
	}
	auth.Set(pkFD, protoreflect.ValueOfBytes(append([]byte(nil), callerPubkey...)))
	auth.Clear(sigFD)
	bytes, err := crypto.CanonicalBytes(req)
	if err != nil {
		return err
	}
	sig, err := crypto.SignBytes(callerPriv, bytes)
	if err != nil {
		return err
	}
	auth.Set(sigFD, protoreflect.ValueOfBytes(sig))
	return nil
}
