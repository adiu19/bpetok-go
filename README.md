# bpetok-go
BPE Training discovers the order, encoding replays it. We're going to build an encoder.
## Goals

- Streaming input, unbounded length
- Zero allocations per token on the hot path (under steady load)
- Exact UTF-8 round trip (input bytes → tokenize → detokenize → same bytes)
- Deterministic output IDs for {a known vocab}
- Throughput target: ≥1 GB/s on 8 cores
- Latency target: P99 <5ms for 64KB chunks
- Works on arbitrarily split chunks, even in the middle of a multibyte rune

## API Guarantees

1. Encoder.Feed() accepts arbitrary byte chunks, including splits in the middle of multi-byte UTF-8 sequences. The encoder internally carries incomplete state across calls and only emits finalized token IDs.
2. Encoder.Flush() returns all remaining token IDs and resets the encoder so we can reuse it without allocating a new one.
3. The tuple (Encoder → Decoder) is guaranteed to reconstruct the exact original byte stream when we:
    - feed the same stream of bytes into the Encoder (via Feed, then Flush),
    - collect all emitted token IDs in order,
    - feed those IDs to the Decoder (via Feed),
    - and concatenate all returned byte slices.
4. Both Feed() methods may return slices backed by internal memorybuffer. Caller must treat return values as ephemeral.


## TODOs
1. Impose a max working set per encoder
2. Replace [2]int in the tokenizer with bit-packing for faster lookups
3. Remove sync.Pool and use per-stream, pre-allocated slices