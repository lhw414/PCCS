CXX     := g++
CXXFLAGS := -std=c++17 -O2 -Wall
LDFLAGS := 

TARGET  := main
SRC     := main.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	@rm -f $(TARGET)
