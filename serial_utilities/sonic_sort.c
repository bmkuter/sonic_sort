#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "sonic_sort.h"

/* RadixSort implementation using Counting Sort and a radix of 256 or 1-byte */

void radixsort(unsigned int *input_array, int num_elements)
{
    int shift, s, i, index;
    unsigned int * tmp;
    
    /* base case: if array is empty or its one element then it is sorted. */
    if (num_elements <= 1) return;

    /* create the output array for count sort */
    unsigned int *array_b = (unsigned int *) calloc(num_elements, sizeof(unsigned int));
   
   /* initialize array that counts within counting sort*/
    int *count = (int*) calloc(256, sizeof(unsigned int));

    /* this will allow us to know if we did even number of swaps
     between input_array pointer and output pointer */
    unsigned int* original_array = input_array;

    /* if the radix is 256 (total numbers in a 1 byte), then sorting 32-bit numbers will take 4 iteration */
    for (shift = 0, s = 0; shift < 4; shift++, s+=8)
    {
        /* reset the count array */
        for (i = 0; i < 256; i++)
        {
            count[i] = 0;
        }

        /* counting occurances of a number and incrementing elements in count array */
        for (i = 0; i < num_elements; i++)
        {
            /* access element input, shift right 's' times, then do bit-wise and with value 255 */
            count[(input_array[i] >> s) &0xff]++;
        }

        /* do prefix sum so that count[i] indicates where a digit belongs in the output array. */
        for (i = 1; i < 256; i++)
        {
            count[i] += count[i-1];
        }

        /* build the output array */
        for (i = num_elements-1; i >= 0; i--)
        {
            index = (input_array[i] >> s) &0xff;

            /* decrement element within count array to figure out input_array[i]'s place in output array */
            array_b[--count[index]] = input_array[i];
        }

        /* input array is now sorted according to current digit.
           swap input_array's and array_b's pointers to simulate copying data
           from one array to other.
        */
       tmp = input_array;
       input_array = array_b;
       array_b = tmp;
    }

    /* if odd number of swaps happened with the pointers,
        then copy over data once more before finishing
    */
   if (original_array == array_b)
   {
       tmp = input_array;
       input_array = array_b;
       array_b = tmp;
   }

   free(array_b);
   free(count);

}

/*
  merge algorithm for two sorted arrays: https://www.geeksforgeeks.org/merge-two-sorted-arrays/
*/
void merge_adjacent_arrays(unsigned int *leftSubArray, unsigned int *rightSubArray, const unsigned int sizeLeft, const unsigned int sizeRight)
{
    /* pointers that will help iterate throught the lists */
    int i=0, j=0, k=0;

    /* create array that holds merged list */
    unsigned int *result = (unsigned int *) malloc((sizeLeft + sizeRight)*sizeof(unsigned int));

    //place the elements of the left and right arrays in the correct place
    while ( i < sizeLeft && j < sizeRight)
    {
        
        if (leftSubArray[i] < rightSubArray[j]) 
        {
            result[k++] = leftSubArray[i++];
        }
        else
        {
            result[k++] = rightSubArray[j++];
        }
    }

    //merge remaining elements
    while ( i < sizeLeft)
    {
        result[k++] = leftSubArray[i++];
    }

    //merge remaining elements
    while ( j < sizeRight)
    {
        result[k++] = rightSubArray[j++];
    }

    memcpy(leftSubArray, result, (sizeLeft + sizeRight)*sizeof(unsigned int));

}

void initializeArray1D(unsigned int *arr, int len, int seed)
{
    int i;
    unsigned int randNum;
    srand(seed);

    for (i = 0; i < len; i++)
    {
        randNum = (unsigned int) rand();
        arr[i] = randNum;
    }
}