/*
nvcc -arch sm_35 sonic_sort.cu -o sonic_sort
module load cuda
*/
#include "cuPrintf.cu"
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>
#include <unistd.h>


// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char *)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
                                       cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}
//16777216
#define BIG_SERIAL_INT              4096
#define NUM_THREADS_PER_SORT        4

#define SM_ARR_LEN                  4096
#define THREADS_PER_BLOCK_MERGE     1024
#define BLOCKS                      8

#define PRINT_TIME                  1

#define THRESHOLD 2 //log2(NUM_THREADS_PER_SORT)

#define CPNS 3.0

#define BSIZE 1

#define NUM_TESTS 1
#define OPTIONS 1

#define MINVAL   1
#define MAXVAL  20

typedef unsigned int data_t;

/* Prototypes */
void printArray(data_t *array, int rowlen);
void initializeArray1D(data_t *arr, long unsigned int len, int seed);
void sort(data_t *arrayA, data_t *arrayB,data_t *arrayC,long int rowlen);
__global__ void sonic_sort(data_t *input_array, data_t *array_b, data_t arr_len, data_t num_elements_per_thread);
__global__ void mega_merge(data_t *arrayA, data_t *arrayB,data_t *arrayC,long int array_len);
__device__ int binary_search(data_t *array, int L, int R, int X, int thread_id, int array_len);
void radixsort(unsigned int *input_array, int num_elements);
void merge_adjacent_arrays(unsigned int *leftSubArray, unsigned int *rightSubArray, const unsigned int sizeLeft, const unsigned int sizeRight);
int compare_lists(data_t *array1, data_t *array2, long int length);

int clock_gettime(clockid_t clk_id, struct timespec *tp);

int clock_gettime(clockid_t clk_id, struct timespec *tp);

double interval(struct timespec start, struct timespec end)
{
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}

/* This routine "wastes" a little time to make sure the machine gets
   out of power-saving mode (800 MHz) and switches to normal speed. */
double wakeup_delay()
{
  double meas = 0; int i, j;
  struct timespec time_start, time_stop;
  double quasi_random = 0;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
  j = 100;
  while (meas < 1.0) {
    for (i=1; i<j; i++) {
      /* This iterative calculation uses a chaotic map function, specifically
         the complex quadratic map (as in Julia and Mandelbrot sets), which is
         unpredictable enough to prevent compiler optimisation. */
      quasi_random = quasi_random*quasi_random - 1.923432;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    meas = interval(time_start, time_stop);
    j *= 2; /* Twice as much delay next time, until we've taken 1 second */
  }
  return quasi_random;
}

int main(int argc, char **argv){
  data_t arrLen = 0;
  struct timespec time_start, time_stop;
  printf("Test\n");

  wakeup_delay();

  if (argc > 1) {
    arrLen  = (atoi(argv[1]));
  }
  else {
    arrLen = (SM_ARR_LEN);
  }

  cudaPrintfInit();


  // GPU Timing variables
  cudaEvent_t start, stop;
  float elapsed_gpu;

  // Arrays on GPU global memory
  data_t *d_arrayA;
  data_t *d_arrayB;
  data_t *d_arrayC;

  // Arrays on the host memory
  data_t *h_arrayA;
  data_t *h_arrayB;
  data_t *h_arrayC;
  data_t *h_arraySerial;
  data_t *h_arrayC_gold;

  //int errCount = 0, zeroCount = 0;

  size_t allocSize = arrLen*sizeof(data_t);

  // Allocate arrays on host memory
  h_arrayA                   = (data_t *) calloc(arrLen,sizeof(data_t));
  h_arrayB                   = (data_t *) calloc(arrLen,sizeof(data_t));
  h_arrayC                   = (data_t *) calloc(arrLen,sizeof(data_t));    //Output for Device
  h_arrayC_gold              = (data_t *) calloc(2*arrLen,sizeof(data_t));    //Validation array
  h_arraySerial              = (data_t *) calloc(BIG_SERIAL_INT,sizeof(data_t));

  if (!h_arrayC) {
      free((void *) h_arrayC);
      printf("\n COULDN'T ALLOCATE STORAGE FOR h_array \n");
      return -1;  /* Couldn't allocate storage */
  }
  printf("Host Init\n");
  // Initialize the host arrays
  // Arrys are initialized with a known seed for reproducability
  initializeArray1D(h_arrayA, arrLen, 123);
  initializeArray1D(h_arraySerial, BIG_SERIAL_INT, 123);
  //initializeArray1D(h_arrayB, arrLen, 351);
  //initializeArray1D(h_arrayC, arrLen, 12);
  //data_t h_arrayA[SM_ARR_LEN] = {19, 14, 3, 16, 7, 2, 8, 20, 4, 15};
  //data_t h_arrayB[SM_ARR_LEN] = {8, 11, 10, 4, 7, 1, 17, 20, 5, 14};

  //memcpy(h_arraySerial,h_arrayA,arrLen*sizeof(data_t));

  //printf("arrayA: ");
  //printArray(h_arrayA,SM_ARR_LEN);
  //printf ("\n");

  printf("arraySerial: ");
  //printArray(h_arraySerial,SM_ARR_LEN);
  printf ("\n");
  /*
  printf("arrayB: ");
  printArray(h_arrayB,SM_ARR_LEN);
  printf ("\n");

  printf("Serial Sorting. . .\n");
  */
  //radixsort(h_arrayA, SM_ARR_LEN);
  //radixsort(h_arrayB, SM_ARR_LEN);
 /*
  printf("arrayA: ");
  printArray(h_arrayA,SM_ARR_LEN);
  printf ("\n");

  printf("arrayB: ");
  printArray(h_arrayB,SM_ARR_LEN);
  printf ("\n");
  */

  printf("Host Init Finished\n");

  // Select GPU
  CUDA_SAFE_CALL(cudaSetDevice(0));
  printf("Alloc\n");
  // Allocate GPU memory

  CUDA_SAFE_CALL(cudaMalloc((void **)&d_arrayA, allocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_arrayB, allocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_arrayC, allocSize));

  // Transfer the arrays to the GPU memory
  CUDA_SAFE_CALL(cudaMemcpy(d_arrayA, h_arrayA, allocSize, cudaMemcpyHostToDevice));
  //CUDA_SAFE_CALL(cudaMemcpy(d_arrayB, h_arrayB, allocSize, cudaMemcpyHostToDevice));
  printf("Kernel Launch\n");

  dim3 dimBlock(THREADS_PER_BLOCK_MERGE, 1);
  dim3 dimGrid(BLOCKS, 1); //arrLen/1024

  printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n", dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

#if PRINT_TIME
  // Create the cuda events
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // Record event on the default stream
  cudaEventRecord(start, 0);
#endif
      /* https://forums.developer.nvidia.com/t/size-limitation-for-1d-arrays-in-cuda/31066 */


  sonic_sort<<<1, NUM_THREADS_PER_SORT>>>(d_arrayA, d_arrayB, arrLen, arrLen/NUM_THREADS_PER_SORT); //4

  int i,j;

  unsigned int half_array_length = arrLen >> 1; //32 / 2 = 16

  for ( i = log2((float)NUM_THREADS_PER_SORT); i > 0; i >>= 1)
  //for ( i = 1; i < NUM_THREADS_PER_SORT ; i <<= 1)
  {
      for ( j = 0; j <= i; j += 2)
      //for ( j = 0, k = 0; k < num_inner_loops; j = j + (2*i),k++)
      {
          mega_merge<<<dimGrid, THREADS_PER_BLOCK_MERGE >>>(d_arrayA+(j*(half_array_length/i)), d_arrayA+((j+1)*(half_array_length/i)), d_arrayC+(j*(half_array_length/i)),half_array_length/i);
          //mega_merge<<<dimGrid, (arrLen/i)>>>(d_arrayA+j, d_arrayA+(j+i), d_arrayC+j,half_array_length/i);
          cudaDeviceSynchronize();

      }
      //num_inner_loops >>= 1;

      data_t *temp = d_arrayA;
      d_arrayA = d_arrayC;
      d_arrayC = temp;
  }


#if PRINT_TIME
  /* Stop and destroy the timer */
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_gpu, start, stop);
  printf("\nGPU time: %f (msec)\n", elapsed_gpu);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
#endif

  cudaPrintfDisplay(stdout, true);
  printf("\t... done\n\n");

  // Check for errors during launch
  CUDA_SAFE_CALL(cudaPeekAtLastError());

  // Transfer the results back to the host
  CUDA_SAFE_CALL(cudaMemcpy(h_arrayC, d_arrayC, allocSize, cudaMemcpyDeviceToHost));

  CUDA_SAFE_CALL(cudaMemcpy(h_arrayA, d_arrayA, allocSize, cudaMemcpyDeviceToHost));
//HOST SOR. Use copied SOR function here.

  printf("Cuda Array (arrayA): \n");
  //printArray(h_arrayA,SM_ARR_LEN);
  printf ("\n");

/* Serial Land */

  wakeup_delay();
   printf("Host Launch\n");
  double time_stamp;
  clock_gettime(CLOCK_REALTIME, &time_start);

      radixsort(h_arraySerial, BIG_SERIAL_INT);

      //merge_adjacent_arrays(h_arraySerial, h_arraySerial + half_array_length, half_array_length, half_array_length);

  clock_gettime(CLOCK_REALTIME, &time_stop);
  time_stamp = interval(time_start, time_stop);

  printf("CPU Time: %.6f (msec)\n",1000*time_stamp);

  printf("Cuda Array (arraySerial): \n");
  //printArray(h_arraySerial,SM_ARR_LEN);
  printf ("\n");

  if (compare_lists(h_arraySerial,h_arrayA,SM_ARR_LEN) != 0) printf("List comparision failed!\n");
  else printf("Lists are the same!\n");

//Change to compare SOR'd matrices using difference

/*
printf("Compare\n");
  // Compare the results

  if (errCount > 0) {
    printf("\n@ERROR: TEST FAILED: %d results did not match\n", errCount);
  }
  else if (zeroCount > 0){
    printf("\n@ERROR: TEST FAILED: %d results (from GPU) are zero\n", zeroCount);
  }
  else {
    printf("\nTEST PASSED: All results matched\n");
  }
*/

  // Free-up device and host memory
  CUDA_SAFE_CALL(cudaFree(d_arrayA));
  CUDA_SAFE_CALL(cudaFree(d_arrayB));
  CUDA_SAFE_CALL(cudaFree(d_arrayC));


  free(h_arrayA);
  free(h_arrayB);
  free(h_arrayC);
  free(h_arrayC_gold);
  free(h_arraySerial);

  cudaPrintfEnd();


  return 0;
}



/************************************/

void sort(data_t *arrayA, data_t *arrayB, data_t *arrayC, long int rowlen)
{

}


__global__ void sonic_sort(data_t *input_array, data_t *array_b, data_t arr_len, data_t num_elements_per_thread)
{
    //int bx = blockIdx.x; /* Can we use this block as a way to work on different sections? */
    int tx = threadIdx.x;



    int thread_starting_point = tx*num_elements_per_thread;
    int thread_end_point = thread_starting_point + num_elements_per_thread;
    int shift, s, i, index;
    unsigned int *tmp;


    /* base case: if array is empty or its one element then it is sorted. */
    if (num_elements_per_thread <= 1) return;

    /* create the output array for count sort */
    //unsigned int *array_b= (unsigned int *) malloc(arr_len*sizeof(unsigned int));


   /* initialize array that counts within counting sort*/
    int *count= (int*) malloc(256*sizeof(unsigned int));

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
        for (i = thread_starting_point; i < thread_end_point; i++)
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
        for (i = thread_end_point-1; i >= thread_starting_point; i--)
        {
            index = (input_array[i] >> s) &0xff;

            /* decrement element within count array to figure out input_array[i]'s place in output array */
            array_b[thread_starting_point + (--count[index])] = input_array[i];
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

__global__ void mega_merge(data_t *left_array, data_t *right_array, data_t *arrayC, long int array_len)
{
  /* Input arrays are subdivisions of the original matrix, being operated on in parallel.
   * Output has to be some shared array the same size as the original array.
   * array_len is half of the output array size, i.e. the size of each input array.
   */

  int bx = blockIdx.x; /* Can we use this block as a way to work on different sections? */
  int tx = threadIdx.x;
  //int array_side = (tx < array_len/BLOCKS) ? 0 : 1; /* Left (0) or Right (1) array */
  //unsigned long int element = 0;
  unsigned long int smaller_than_me = 0;

  /**** New Variables ***/
  int merge_number = bx / ((2*array_len)/THREADS_PER_BLOCK_MERGE);
  long int absolute_element = tx + (bx * THREADS_PER_BLOCK_MERGE);
  int left_or_right = ( (absolute_element - (merge_number * (2 * array_len))) < array_len ) ? 0 : 1; /* Left (0) or Right (1) array */
  int relative_block = bx % (array_len/THREADS_PER_BLOCK_MERGE);
  int what_element_am_I_in_my_list = tx + (relative_block * THREADS_PER_BLOCK_MERGE);
  /*******/

  //cuPrintf("List size: %d\nMerge Number: %d\nLocal Element: %d\nLeft or Right: %d\nRelative Block: %d\n\n",array_len, merge_number,what_element_am_I_in_my_list,left_or_right,relative_block);

  /* Setting my element number if I'm right array */
  //if(array_side) element = (tx - array_len/BLOCKS) + (bx * THREADS_PER_BLOCK_MERGE >> 1);
  //if(array_side) element = tx - array_len;
  /* Setting my element number if I'm left array */
  //else element = tx + (bx * THREADS_PER_BLOCK_MERGE >> 1);
  //else element = tx;

  /* Left Side, so we want to search right array */
  //if (!array_side) smaller_than_me = element + binary_search(right_array,0,array_len-1,left_array[element],array_side,array_len);
  /* Right Side, so we want to search left array */
  //else if (array_side) smaller_than_me = element + binary_search(left_array,0,array_len-1,right_array[element],array_side,array_len);

  /* New Code */
  /* Left Side, so we want to search right array */
  if (!left_or_right) smaller_than_me = what_element_am_I_in_my_list + binary_search(right_array,0,array_len-1,left_array[what_element_am_I_in_my_list],left_or_right,array_len);
  /* Right Side, so we want to search left array */
  else if (left_or_right) smaller_than_me = what_element_am_I_in_my_list + binary_search(left_array,0,array_len-1,right_array[what_element_am_I_in_my_list],left_or_right,array_len);

  //cuPrintf("Element: %d.%lu || How many smaller than me: %lu || Value: %d\n",array_side,element ,smaller_than_me,(array_side) ? right_array[what_element_am_I_in_my_list] : left_array[what_element_am_I_in_my_list]);

  arrayC[smaller_than_me] = (left_or_right) ? right_array[what_element_am_I_in_my_list] : left_array[what_element_am_I_in_my_list];

}

//Add input for left or right array, which is determined by the thread through threadID & array_len
__device__ int binary_search(data_t *array, int L, int R, int X, int which_array, int array_len)
{
  /*
  Variables:
    *array := array to be searched
    L := Leftmost element, usually 0
    R := Rightmost element, array_len - 1.
    X := The variable to search for. "Us" in this context.
    which_array := || 0 = left, 1 = right || Indicates if we are left or right array when comparing ourselves to an identical value. Left array goes first on a tie, right array goes second on a tie.
    array_len := Length of the input array.

 The overall goal here is to determine how many elements in the other array are smaller than me (X). On ties, the item with the smaller threadID goes first in the output list. The number of smaller items can be represented by M, or our current place in the array. */
  int left_value = 0, center_value = 0, right_value = 0;

  while(L <=  R)
  {
    int M = L + (R-L)/2; /*Division will probably be compiled away, so no need for i>>1 */
    left_value = (M == (0)) ? (MINVAL-1) : array[M-1];
    center_value = array[M];
    right_value = (M == (array_len-1)) ? (MAXVAL+1) : array[M+1];

    //printf("\n\nMe: %d\nPosition: %d\nLeft: %d\nCenter: %d\nRight: %d\n\n",X,M,left_value,center_value,right_value);

    /* If we are equal to the center value, we have to look at which SIDE we are */
    if(center_value == X)
    {
      /* First lets consider left array values */
      if(!which_array)
      {
        /* If left value is less than us, return our current place */
        if(left_value < X) return M;
        /* If left value is equal to us, we need to bin_hop leftward */
        else if (left_value == X) R = M - 1;
        /* Error condition */
        else return -1;
      }
      /* Now lets consider right array values */
      else if (which_array)
      {
        /* If right value is greater than us, move right one and return */
        if(right_value > X) return (++M);
        /* If right value is equal to us, we need to bin_hop rightward */
        else if (right_value == X) L = M + 1;
        /* Error condition */
        else return -1;
      }
      /* Error condition */
      else return -1;
    }

    /* If we are less than the center value, we will want to move left */
    else if(center_value > X)
    {
      /* Look left */
      /* If left value is less than us, return our current place */
      if(left_value < X) return M;
      /* If left value is equal to us, we need to check SIDE */
      else if(left_value == X)
      {
        /* If we are the left array, we need to bin_hop leftward */
        if(!which_array) R = M - 1;
        /* If we are right array, we see that we are less than center, and same as left. But we want to be to the right of any identical values from left, so we can return current place. */
        else if(which_array) return M;
        /* Error condition */
        else return -1;
      }
      /* If the left value is also greater than us, we need to bin_hop further leftward in the array. */
      else if(left_value > X) R = M - 1;
      /* Error condition */
      else return -1;
    }

    /* If we are greater than center value, we will want to move to the right */
    else if(center_value < X)
    {
      /* Look right. If its greater than us, move right 1 and return */
      if(right_value > X) return (++M);
      /* If right value is equal to us, check SIDE */
      else if(right_value == X)
      {
        /* If left array, we will come before any equal values so we can shift right 1 and return */
        if(!which_array) return (++M);
        /* If we are right array, we will need to bin_hop right since we need to be to the right of any equal values */
        else if (which_array) L = M + 1;
        /* Error condition */
        else return -1;
      }
      /* If right is still smaller than us, we need to bin_hop even further right */
      else if(right_value < X) L = M + 1;
      /* Error condition */
      else return -1;
    }

    /* Error condition */
    else return -1;
  }
  /* Need flag to indicate if we're bigger or smaller than everything in the other list */

  cuPrintf("\nOutside array boundaries\n\n");

  if(X > array[array_len - 1]) return array_len; /* We're bigger, so we return array length, since we're larger than every element in the other array. */
  else if (X < array[0]) return 0; /* We're smaller, so we return 0 because there are no elements in the other array smaller than us. */
  else return -1; /*Error condition */

}

void initializeArray1D(unsigned int *arr, long unsigned int len, int seed) {
  long int i;
  srand(seed);
  //data_t fRand(data_t fMin, data_t fMax);

  for (i = 0; i < len; i++)
  {
    arr[i] = (rand() % (MAXVAL - MINVAL + 1)) + MINVAL;
  }
  printf("\n");
}

int compare_lists(data_t *array1, data_t *array2, long int length)
{
  for (int i = 0; i < length; i++)
  {
    if(array1[i]-array2[i] != 0) return 1;
  }
  return 0;
}

void printArray(data_t *array, int rowlen)
{
  int i;
  for (i=0; i < rowlen; i++)
  {
    printf("%d   ", array[i]);
  }
}

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
