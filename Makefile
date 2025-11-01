GO       := go
BINDIR   := bin

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

.PHONY: test
test:
	$(GO) test ./...

# Clean build artifacts
.PHONY: clean
clean:
	@echo "🧹 Cleaning up..."
	rm -rf $(BINDIR)
	find . -name '*.test' -type f -delete
	find . -name '*.out'  -type f -delete
	find . -name 'coverage.*' -type f -delete
	@echo "Done."