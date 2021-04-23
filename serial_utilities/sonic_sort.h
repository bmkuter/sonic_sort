#ifndef SONIC_SORT_H
#define SONIC_SORT_H

void radixsort(unsigned int *input_array, int num_elements);

void merge_adjacent_arrays(unsigned int *leftSubArray, unsigned int *rightSubArray, const unsigned int sizeLeft, const unsigned int sizeRight);

void initializeArray1D(unsigned int *arr, int len, int seed);

#endif /* End SONIC_SORT_H */

