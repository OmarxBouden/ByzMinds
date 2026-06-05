// byzminds-kernel boots the deterministic core and serves the gRPC
// surface defined in proto/kernel.proto.
//
// Step 1 scope: the binary loads a researcher pubkey + kernel keypair
// from disk, opens the five ledgers (with optional pre-declared private
// channels), starts the gRPC server, and runs until SIGINT/SIGTERM.
// Tick advancement and the dispatch loop arrive in Step 2.
package main

import (
	"context"
	"crypto/ed25519"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"google.golang.org/grpc"

	"github.com/byzminds/byzminds/kernel/api"
	"github.com/byzminds/byzminds/kernel/internal/crypto"
	"github.com/byzminds/byzminds/kernel/internal/handler"
	"github.com/byzminds/byzminds/kernel/internal/ledger"
	"github.com/byzminds/byzminds/kernel/internal/scheduler"
)

// version is the kernel build version stamp. Override at build time:
//
//	go build -ldflags "-X main.version=<sha>" ./cmd/byzminds-kernel
var version = "dev"

func main() {
	addr := flag.String("addr", "127.0.0.1:7777", "gRPC listen address")
	researcherHex := flag.String("researcher-pubkey-hex", "", "hex-encoded researcher Ed25519 public key (32 B)")
	kernelKeyHex := flag.String("kernel-priv-hex", "", "hex-encoded kernel Ed25519 private key (64 B)")
	channels := flag.String("private-channels", "", "comma-separated channel ids to open at boot (members supplied via --channel-members for now)")
	flag.Parse()

	rPub, err := decodePubkey(*researcherHex)
	if err != nil {
		log.Fatalf("--researcher-pubkey-hex: %v", err)
	}
	kPriv, err := decodePrivkey(*kernelKeyHex)
	if err != nil {
		log.Fatalf("--kernel-priv-hex: %v", err)
	}

	var prv []ledger.PrivateChannelConfig
	for _, ch := range splitCSV(*channels) {
		prv = append(prv, ledger.PrivateChannelConfig{ChannelID: ch})
	}

	ls, err := ledger.New(ledger.Config{
		Researcher:      rPub,
		KernelPriv:      kPriv,
		PrivateChannels: prv,
	})
	if err != nil {
		log.Fatalf("ledger.New: %v", err)
	}

	h := handler.New(ls)
	sch := scheduler.New(ls, h, 30*time.Second)
	kernelSrv := api.New(ls)
	kernelSrv.AttachStep2(h, sch)
	handlerSrv := api.NewHandlerServer(h)

	grpcServer := grpc.NewServer()
	kernelSrv.Register(grpcServer)
	handlerSrv.Register(grpcServer)

	// Run the scheduler in the background; advance ticks freely until
	// shutdown. Cmd-line scenarios will switch to driven mode via
	// byzminds-run; standalone kernel runs a no-op scheduler that just
	// awaits agents.
	schedCtx, schedCancel := context.WithCancel(context.Background())
	defer schedCancel()
	go func() {
		if err := sch.RunUntil(schedCtx, ^uint64(0)); err != nil && !errors.Is(err, context.Canceled) {
			log.Printf("scheduler stopped: %v", err)
		}
	}()

	lis, err := net.Listen("tcp", *addr)
	if err != nil {
		log.Fatalf("listen %s: %v", *addr, err)
	}

	log.Printf("byzminds-kernel %s listening on %s", version, *addr)
	errCh := make(chan error, 1)
	go func() { errCh <- grpcServer.Serve(lis) }()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	select {
	case err := <-errCh:
		if err != nil && !errors.Is(err, grpc.ErrServerStopped) {
			log.Fatalf("grpc serve: %v", err)
		}
	case <-ctx.Done():
		log.Printf("shutdown signal received; stopping server")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		stopped := make(chan struct{})
		go func() { grpcServer.GracefulStop(); close(stopped) }()
		select {
		case <-stopped:
		case <-shutdownCtx.Done():
			log.Printf("graceful stop timed out; forcing")
			grpcServer.Stop()
		}
	}
}

func decodePubkey(s string) (ed25519.PublicKey, error) {
	if s == "" {
		return nil, errors.New("required")
	}
	b, err := hex.DecodeString(s)
	if err != nil {
		return nil, fmt.Errorf("decode hex: %w", err)
	}
	if len(b) != crypto.PublicKeySize {
		return nil, fmt.Errorf("expected %d bytes, got %d", crypto.PublicKeySize, len(b))
	}
	return ed25519.PublicKey(b), nil
}

func decodePrivkey(s string) (ed25519.PrivateKey, error) {
	if s == "" {
		return nil, errors.New("required")
	}
	b, err := hex.DecodeString(s)
	if err != nil {
		return nil, fmt.Errorf("decode hex: %w", err)
	}
	if len(b) != crypto.PrivateKeySize {
		return nil, fmt.Errorf("expected %d bytes, got %d", crypto.PrivateKeySize, len(b))
	}
	return ed25519.PrivateKey(b), nil
}

func splitCSV(s string) []string {
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := parts[:0]
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}
