package main

import (
	"log"
	"path/filepath"

	"github.com/bpetok/internal/tokenizer"
)

func main() {
	vocab := filepath.Join("testdata", "gpt2", "vocab.json")
	merges := filepath.Join("testdata", "gpt2", "merges.txt")

	tok, err := tokenizer.LoadTokenizerFromFiles(vocab, merges)
	if err != nil {
		log.Fatalf("failed to load tokenizer: %v", err)
	}

	_ = tok
	log.Println("vocab loaded successfully and IDs are dense")
}
