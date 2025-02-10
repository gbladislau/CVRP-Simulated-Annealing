all: compile

compile: clean
	g++ -o main $(wildcard ./src/*.cpp) -lm


clean:
	rm -f main trabalhocg_d *.o

debug: clean
	g++ -o main $(wildcard ./src/*.cpp) -lm -g
	gdb ./main

run: clean compile
	./main