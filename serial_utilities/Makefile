#variable for my compiler
CC = gcc

#enumerate the object files
objects = test_radixsort.o sonic_sort.o

#all rule to generate sonic_sort
all: sonic_sort

sonic_sort: test_radixsort.o sonic_sort.o
	$(CC) $(objects) -o radix

test_radixsort.o: test_radixsort.c sonic_sort.h sonic_sort.c
	$(CC) -c test_radixsort.c 

sonic_sort.o: sonic_sort.h sonic_sort.c
	$(CC) -c sonic_sort.c

clean:
	rm $(objects) radix