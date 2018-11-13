CC = nvcc
CFLAGS = -std=c++11 
INCLUDES = 
LDFLAGS = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -D_MWAITXINTRIN_H_INCLUDED
SOURCES = GPU.cu
OUTF = GPU

$(OUTF): GPU.o
	$(CC) $(CFLAGS) -o $(OUTF) $< $(LDFLAGS)

GPU.o: $(SOURCES)
	$(CC) $(CFLAGS) -c $<

rebuild: clean $(OUTF)

clean:
	rm *.o GPU