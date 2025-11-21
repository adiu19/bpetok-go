GO       := go
BINDIR   := bin
GOMAXPROCS := 1

CMDS := fetch_gpt2_tokenizer

.PHONY: all
all: build

.PHONY: build
build: $(CMDS:%=$(BINDIR)/%)

# Pattern rule:
# Build cmd/<name> -> bin/<name>
# We FORCE external link mode to satisfy macOS dyld LC_UUID requirement.
$(BINDIR)/%: cmd/%/*
	@mkdir -p $(BINDIR)
	$(GO) build -ldflags="-linkmode=external" -o $@ ./cmd/$*

# Fetch GPT-2 assets (vocab.json, merges.txt) into testdata/gpt2/
.PHONY: fetch-gpt2
fetch-gpt2: $(BINDIR)/fetch_gpt2_tokenizer
	$(BINDIR)/fetch_gpt2_tokenizer

.PHONY: test-vocab
test-vocab:
	@echo "Checking vocab.json integrity.........."
	$(GO) run ./cmd/test_vocab_load

.PHONY: test-offline
test-offline:
	go test -v ./internal/tokenizer -run TestOffline -count=1

.PHONY: test-decode
test-decode:
	go test -v ./internal/tokenizer -run TestDecode -count=1


.PHONY: test-streaming
test-streaming:
	go test -v ./internal/tokenizer -run TestStreaming -count=1

.PHONY: bench
bench:
	go test -run '^$$' -bench Benchmark -benchmem -benchtime=3x ./internal/tokenizer

.PHONY: bench-cpu
bench-cpu:
	go test -run '^$$' -bench BenchmarkEncodeOffline -benchmem -benchtime=10x -cpuprofile=cpu.out ./internal/tokenizer
	@echo "Profile saved to cpu.out. View with: go tool pprof cpu.out"

.PHONY: bench-naive-streaming-trace-whole-chunk
bench-naive-streaming-trace-whole-chunk:
	GOMAXPROCS=$(GOMAXPROCS)go test -run '^$$' -bench BenchmarkNaiveEncodeStreaming_WholeChunk -benchmem -benchtime=10x -trace=trace.out -cpuprofile=cpu.out ./internal/tokenizer
	@echo "Trace saved to trace.out. View with: go tool trace trace.out"
	@echo "CPU profile saved to cpu.out. View with: go tool pprof cpu.out"

.PHONY: bench-naive-streaming-trace-4kb-chunk
bench-naive-streaming-trace-4kb-chunk:
	GOMAXPROCS=$(GOMAXPROCS) go test -run '^$$' -bench BenchmarkNaiveEncodeStreaming_4KBChunks -benchmem -benchtime=10x -trace=trace.out -cpuprofile=cpu.out ./internal/tokenizer
	@echo "Trace saved to trace.out. View with: go tool trace trace.out"
	@echo "CPU profile saved to cpu.out. View with: go tool pprof cpu.out"
# Clean build artifacts
.PHONY: clean
clean:
	@echo "🧹 Cleaning up..."
	rm -rf $(BINDIR)
	find . -name '*.test' -type f -delete
	find . -name '*.out'  -type f -delete
	find . -name 'coverage.*' -type f -delete
	@echo "Done."
