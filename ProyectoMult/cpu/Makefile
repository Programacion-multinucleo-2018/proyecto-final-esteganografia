CC = g++
CFLAGS = -std=c++11 
INCLUDES = 
LDFLAGS = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
SOURCES = CPU.cpp
OUTF = CPU

$(OUTF): CPU.o
	$(CC) $(CFLAGS) -o $(OUTF) $< $(LDFLAGS)

CPU.o: $(SOURCES)
	$(CC) $(CFLAGS) -c $<

rebuild: clean $(OUTF)

clean:
	rm *.o CPU