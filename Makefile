CFLAGS = -std=c++11 -O3 -Xcompiler -ansi -Xcompiler -Ofast -Xcompiler -std=c++11 -Wno-deprecated-gpu-targets
LDFLAGS = -lGL -lglut -lGLU
OUTF = build/
MKDIR_P = mkdir -p

objects = main.o SPH_cuda/SPH_cuda.o Math3D/m3Matrix.o Math3D/m9Matrix.o

all: $(objects) $(OUTF) 
		nvcc $(CFLAGS) -arch=sm_20 $(objects) -o $(OUTF)app $(LDFLAGS)

$(OUTF):
	$(MKDIR_P) $(OUTF)

%.o: %.cpp
		nvcc $(CFLAGS) -x cu -arch=sm_20 -I Math3D/ -I SPH_cuda/ -I ${CUDA_HOME}/include/ -I cuda_common/ -dc $< -o $@

clean:
		rm -f *.o Math3D/*.o SPH_cuda/*.o $(OUTF)app