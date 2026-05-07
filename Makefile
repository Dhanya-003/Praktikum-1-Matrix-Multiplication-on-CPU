# ─── Makefile for Lab 1: Matrix Multiplication on CPU ───────────────────────
CC = gcc
CFLAGS = -O3 -march=native -ffast-math
LDFLAGS = -lm

all: matmul

matmul: matmul.c
	$(CC) $(CFLAGS) matmul.c -o matmul $(LDFLAGS)

clean:
	rm -f matmul *.o
