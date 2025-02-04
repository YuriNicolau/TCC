CC = clang++
#CFLAGS = -Wall -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_61 -L/opt/cuda/lib64 -I/opt/cuda/include -lm -lcudart
CFLAGS = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_61 -Wall -lm
LDFLAGS = -lm -lcudart
OBJFILES = wave.o
TARGET = wave

all: $(TARGET)

$(TARGET) : $(OBJFILES)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJFILES) $(LDFLAGS)

clean:
	rm -f $(OBJFILES) $(TARGET) *~
