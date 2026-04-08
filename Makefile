CXX = g++
CXXFLAGS = -O2 -std=c++17

# For OpenMP (if available)
OPENMP = -fopenmp

all: dem

dem:
	$(CXX) $(CXXFLAGS) dem_solver.cpp -o dem

parallel:
	$(CXX) $(CXXFLAGS) $(OPENMP) dem_solver.cpp -o dem

run:
	./dem

clean:
	rm -f dem *.csv