package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

var files = map[string]string{
	"vocab.json": "https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json",
	"merges.txt": "https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt",
}

func download(url, destPath string) error {
	// 1. GET
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("GET %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: unexpected status %s", url, resp.Status)
	}

	// 2. create dest file
	out, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("create %s: %w", destPath, err)
	}
	defer out.Close()

	// 3. copy body -> file
	n, err := io.Copy(out, resp.Body)
	if err != nil {
		return fmt.Errorf("write %s: %w", destPath, err)
	}
	if n == 0 {
		return fmt.Errorf("download %s: got 0 bytes", url)
	}

	return nil
}

func main() {
	targetDir := filepath.Join("testdata", "gpt2")

	// Make sure testdata/gpt2 exists.
	if err := os.MkdirAll(targetDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "mkdir %s: %v\n", targetDir, err)
		os.Exit(1)
	}

	for name, url := range files {
		destPath := filepath.Join(targetDir, name)
		fmt.Printf("-> downloading %s\n", name)

		if err := download(url, destPath); err != nil {
			fmt.Fprintf(os.Stderr, "error downloading %s: %v\n", name, err)
			os.Exit(1)
		}
	}

	fmt.Println("done. files in testdata/gpt2/")
}
