CC = gcc
CFLAGS = -O2 -Wall -Wextra -Iinclude
LDFLAGS = -lm -Wl,--stack,67108864

SRCDIR = src
OBJDIR = obj
TARGET = phase1_verify

SRCS = $(SRCDIR)/main.c $(SRCDIR)/causal_capacity.c $(SRCDIR)/propagation_analysis.c
OBJS = $(OBJDIR)/main.o $(OBJDIR)/causal_capacity.o $(OBJDIR)/propagation_analysis.o

all: $(OBJDIR) $(TARGET)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -rf $(OBJDIR) $(TARGET).exe $(TARGET)

run_quick: $(TARGET)
	./$(TARGET).exe 0

run_scan: $(TARGET)
	./$(TARGET).exe 1

run_topo: $(TARGET)
	./$(TARGET).exe 2

run_time: $(TARGET)
	./$(TARGET).exe 3

run_scan200: $(TARGET)
	./$(TARGET).exe 4

run_single200: $(TARGET)
	./$(TARGET).exe 5

.PHONY: all clean run_quick run_scan run_topo run_time run_scan200 run_single200
