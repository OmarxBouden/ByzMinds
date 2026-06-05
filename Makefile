.PHONY: tools proto proto-python agent-venv agent-test build test clean

AGENT_VENV ?= agent/.venv
AGENT_PYTHON ?= $(AGENT_VENV)/bin/python
AGENT_PIP ?= $(AGENT_VENV)/bin/pip

# Install protoc plugins into $(go env GOBIN) (or $GOPATH/bin).
tools:
	go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.35.2
	go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.5.1

# Generate Go bindings for every .proto file. Output lands next to each
# source under proto/<name>v1/ per the go_package option.
proto:
	@mkdir -p proto/eventsv1 proto/ledgerv1 proto/kernelv1 proto/viewv1 proto/handlerv1
	protoc \
		--proto_path=proto \
		--go_out=. --go_opt=module=github.com/byzminds/byzminds \
		--go-grpc_out=. --go-grpc_opt=module=github.com/byzminds/byzminds \
		proto/events.proto proto/ledger.proto proto/view.proto \
		proto/kernel.proto proto/handler.proto

# Build the kernel + scenario binaries.
build:
	cd kernel && go build -o ../bin/byzminds-kernel ./cmd/byzminds-kernel
	cd kernel && go build -o ../bin/byzminds-panel ./cmd/byzminds-panel
	cd kernel && go build -o ../bin/byzminds-stub-agent ./cmd/byzminds-stub-agent
	cd kernel && go build -o ../bin/byzminds-run ./cmd/byzminds-run

# Run all tests (kernel module only; proto module has no tests).
test:
	cd kernel && go test ./...

clean:
	rm -rf bin proto/eventsv1 proto/ledgerv1 proto/kernelv1 proto/viewv1 proto/handlerv1 \
	       agent/byzminds_agent/proto_gen/*_pb2.py agent/byzminds_agent/proto_gen/*_pb2_grpc.py \
	       agent/byzminds_agent/proto_gen/*_pb2.pyi

# ---- Python agent ---------------------------------------------------------

# Create an isolated venv at agent/.venv (idempotent).
agent-venv:
	@if [ ! -d "$(AGENT_VENV)" ]; then \
		python3.11 -m venv "$(AGENT_VENV)"; \
		$(AGENT_PIP) install --upgrade pip setuptools wheel; \
	fi

# Generate Python protobuf + gRPC bindings into
# agent/byzminds_agent/proto_gen/. Requires grpcio-tools in the venv.
proto-python: agent-venv
	$(AGENT_PIP) install --quiet "grpcio-tools>=1.68.0"
	$(AGENT_PYTHON) -m grpc_tools.protoc \
		--proto_path=proto \
		--python_out=agent/byzminds_agent/proto_gen \
		--pyi_out=agent/byzminds_agent/proto_gen \
		--grpc_python_out=agent/byzminds_agent/proto_gen \
		proto/events.proto proto/ledger.proto proto/view.proto \
		proto/kernel.proto proto/handler.proto
	@# Patch generated absolute imports to package-relative so
	@# `import byzminds_agent.proto_gen.kernel_pb2` works.
	@$(AGENT_PYTHON) agent/scripts/fix_proto_imports.py agent/byzminds_agent/proto_gen
	@touch agent/byzminds_agent/proto_gen/__init__.py
	@echo "proto-python: bindings regenerated under agent/byzminds_agent/proto_gen/"

# Install the agent package in editable mode with dev extras.
agent-install: agent-venv
	$(AGENT_PIP) install -e "agent[dev]"

# Run pytest against the agent package.
agent-test: agent-install
	$(AGENT_VENV)/bin/pytest agent/tests -q
