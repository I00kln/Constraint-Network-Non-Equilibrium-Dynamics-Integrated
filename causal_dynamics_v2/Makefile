CC = gcc
CFLAGS = -Wall -O2 -I./include -fopenmp
LDFLAGS = -lm -fopenmp -Wl,--stack,67108864

SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = .

SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
TARGET = $(BINDIR)/causal_sim_v2

all: directories $(TARGET)

directories:
	@mkdir -p $(OBJDIR) $(BINDIR)/data

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) ./causal_sim_v2

test: all
	./causal_sim_v2 test

single: all
	./causal_sim_v2 single -N 500 -steps 1000

null: all
	./causal_sim_v2 null

# 大规模测试
large: all
	./causal_sim_v2 single -N 1000 -steps 2000 -eta 0.1 -lambda 1.0 -kc 0.1 -alpha 0.5

# 超大规模测试
xlarge: all
	./causal_sim_v2 single -N 1500 -steps 3000 -eta 0.1 -lambda 1.0 -kc 0.1 -alpha 0.5

phase0: all
	@echo "=== Phase 0: Implementation and Debugging ==="
	./causal_sim_v2 test
	@echo ""
	@echo "Phase 0 Complete: All unit tests passed"

phase1: all
	@echo "=== Phase 1: Small-scale Exploration ==="
	python3 parameter_scan.py
	@echo ""
	@echo "Phase 1 Complete: Parameter scan finished"

.PHONY: all clean test single null large xlarge phase0 phase1 directories
