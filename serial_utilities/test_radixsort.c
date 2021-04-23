#include <stdlib.h>
#include <stdio.h>
#include "sonic_sort.h"

//void radixsort(unsigned int *input_array, int num_elements);

unsigned int state = 123;

unsigned int xorshift32()
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

void GenerateRandomData_max(unsigned int *arr, int count, int seed, int max)
{
    int i;
    
    state = seed;
    for (i = 0; i < count; i++)
    {
        arr[i] = xorshift32() % max;
    }
}

void GenerateRandomData(unsigned int *arr, int count, int seed)
{
    int i;
    
    state = seed;
    for (i = 0; i < count; i++)
    {
        arr[i] = xorshift32();
    }
}



int main()
{   
    int COUNT = 10;
    int r;

    unsigned int * arr = (unsigned int *) calloc(COUNT, sizeof(unsigned int));

    //GenerateRandomData_max(arr, COUNT, 123, 10000);

    initializeArray1D(arr, COUNT, 420);

    for (r=0; r < COUNT; r++)
    {
        printf("%d\n", arr[r]);
    }

    radixsort(arr, COUNT);

    printf("\n\n");

    for (r=0; r < COUNT;r++)
    {
        printf("%d\n", arr[r]);
    }

    

    for (r = 0; r < COUNT-1; r++)
    {
        if (arr[r] > arr[r+1]) 
        {
            printf("List not sorted");
            break;
        }
    }

    printf("list sorted\n");


    unsigned int left[10] = {1,3,5,7,9,0,2,4,6,8};
    unsigned int right[5] = {1,3,5,7,9};

    merge_adjacent_arrays(left,left+5,5,5);

    for (r = 0; r < 10; r++)
    {
        printf("%d\n", left[r]);
    }

    return 0;
}

