
HEADER_DIR=headers

CPP = g++
CUDA = nvcc
CFLAGS = -O3 -Wall -Wextra -g -std=c++17 -march=native 
INCLUDE = -I./$(HEADER_DIR) -I/usr/include/eigen3

HEADERS = disc.h interpolate.h monte_carlo_model.h ciesla_model.h thomson_model.h \
		particles.h domain.h histogram.h
HEADERS := $(addprefix $(HEADER_DIR)/, $(HEADERS))

OBJ = diffusion_model.o 

ALL = $(HEADERS) src/front_model.cpp  tests/test_disc.cpp tests/test_MC.cpp \
	diffusive_disc

.PHONY: tests tidy clean tar

tests : test_disc test_MC

tar: ../MC_dust.tar

%.o: src/%.cpp headers/%.h Makefile
	$(CPP) $(CFLAGS) $(INCLUDE) -c $<


residence: src/residence.cpp $(HEADERS) Makefile 
	$(CPP) $(CFLAGS) $(INCLUDE) $< -o $@

front: src/front_model.cpp $(HEADERS) Makefile 
	$(CPP) $(CFLAGS) $(INCLUDE) $< -o $@

full_disc: src/full_disc_model.cpp $(HEADERS) Makefile 
	$(CPP) $(CFLAGS) $(INCLUDE) $< -o $@

diffusive_disc: src/diffusive_disc.cpp $(OBJ) $(HEADERS) Makefile 
	$(CPP) $(CFLAGS) $(INCLUDE) $< $(OBJ) -o $@

test_disc : tests/test_disc.cpp $(HEADERS) Makefile 
	$(CPP) $(CFLAGS) $(INCLUDE) $< -o $@

test_MC : tests/test_MC.cpp $(HEADERS) Makefile 
	$(CPP) $(CFLAGS) $(INCLUDE) $< -o $@ -pthread

tidy: 
	rm -rf discs.dat settling.dat output

clean: 
	rm -f *.o test_disc test_MC front residence

../MC_dust.tar: $(ALL) Makefile
	tar cvf $@ --exclude=.git --exclude=output --exclude=full_disc_sim --exclude=test_results *
