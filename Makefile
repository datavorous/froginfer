CC = gcc
CFLAGS = -Wall -g
LIBS = -lm
TARGET = llm
SOURCE = main.c

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET) *.o
.PHONY: all clean
