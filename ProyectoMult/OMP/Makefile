CC = g++
CFLAGS = -std=c++11 
INCLUDES = 
LDFLAGS = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lpthread -fopenmp
SOURCES = CPUOMP.cpp
OUTF = CPUOMP

$(OUTF): CPUOMP.o
	$(CC) $(CFLAGS) -o $(OUTF) $< $(LDFLAGS)

CPUOMP.o: $(SOURCES)
	$(CC) $(CFLAGS) -c $<

rebuild: clean $(OUTF)

clean:
	rm *.o CPUOMP