/*
   Minimal CUDA program, intended just to test ability
   to compile and run a CUDA program


     
     nvcc -arch sm_35 sonic_sort.cu -o sonic_sort
      module load cuda
 

   You need to follow instructions provided elsewhere, such as in the
   "SCC-for-EC527" slides, both of the SCC_Cheatsheet PDFs, and
   SCC_Getting_Started PDFs, to get onto the system where you can
   compile and run this.

   To understand the program, of course you should read the lecture notes
   (slides) that have "GPU" in the name.
*/
#include "cuPrintf.cu"
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>

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

#define NUM_THREADS_PER_BLOCK   16    //sqr(256) = 16
#define NUM_BLOCKS         128          //sqr(16) = 4
#define PRINT_TIME         1
#define SM_ARR_LEN        2048
#define COMPARE_TOL         .05
#define TILE_WIDTH          16

#define THRESHOLD 10

#define CPNS 3.0

#define BSIZE 1

#define NUM_TESTS 1
#define OPTIONS 1

#define MINVAL   0.0
#define MAXVAL  10.0

typedef unsigned int data_t;

/* Prototypes */
void printArray(data_t *array, int rowlen);
void initializeArray1D(data_t *arr, int len, int seed);
void sort(data_t *arrayA, data_t *arrayB,data_t *arrayC,long int rowlen);
__global__ void sonic_sort(data_t *arrayA, data_t *arrayB,data_t *arrayC,long int rowlen);
__global__ void mega_merge(data_t *arrayA, data_t *arrayB,data_t *arrayC,long int rowlen);
__device__ int binary_search(data_t *array, int L, int R, int X, int thread_id, int array_len);
data_t fRand(data_t fMin, data_t fMax);

/* Prototypes */
void printArray(data_t *array, int rowlen);

int clock_gettime(clockid_t clk_id, struct timespec *tp);

data_t interval(struct timespec start, struct timespec end)
{
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((data_t)temp.tv_sec) + ((data_t)temp.tv_nsec)*1.0e-9);
}

/* This routine "wastes" a little time to make sure the machine gets
   out of power-saving mode (800 MHz) and switches to normal speed. */
data_t wakeup_delay()
{
  data_t meas = 0; int i, j;
  struct timespec time_start, time_stop;
  data_t quasi_random = 0;
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
  long int arrLen = 0;
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
  data_t *h_arrayC_gold;

  int errCount = 0, zeroCount = 0;

  size_t allocSize = arrLen*sizeof(data_t);
  
  // Allocate arrays on host memory
  h_arrayA                   = (data_t *) malloc(allocSize);
  h_arrayB                   = (data_t *) malloc(allocSize);
  h_arrayC                   = (data_t *) calloc(arrLen,sizeof(data_t));    //Output for Device
  h_arrayC_gold              = (data_t *) calloc(arrLen,sizeof(data_t));    //Validation array
  
  if (!h_arrayA) {
      free((void *) h_arrayA);
      printf("\n COULDN'T ALLOCATE STORAGE FOR h_array \n");
      return -1;  /* Couldn't allocate storage */
  }
  printf("Host Init\n");
  // Initialize the host arrays
  // Arrys are initialized with a known seed for reproducability
  initializeArray1D(h_arrayA, arrLen, 123);
  initializeArray1D(h_arrayB, arrLen, 351);
  initializeArray1D(h_arrayC, arrLen, 12);
  printf("Host Init Finished\n");

  // Select GPU
  CUDA_SAFE_CALL(cudaSetDevice(0));
  printf("Alloc\n");
  // Allocate GPU memory
  
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_arrayA, allocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_arrayC, allocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_arrayB, allocSize));

  // Transfer the arrays to the GPU memory
  CUDA_SAFE_CALL(cudaMemcpy(d_arrayA, h_arrayA, allocSize, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_arrayB, h_arrayB, allocSize, cudaMemcpyHostToDevice));
  printf("Kernel Launch\n");
   //Launch the kernel(nuke)   SM_ARR_LEN
     //define block geometry? 
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 dimGrid(arrLen/TILE_WIDTH, arrLen/TILE_WIDTH); 
  
#if PRINT_TIME
  // Create the cuda events
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // Record event on the default stream
  cudaEventRecord(start, 0);
#endif
  for (int i = 0; i < THRESHOLD; i++)
  {
       sonic_sort<<<dimGrid, dimBlock>>>(d_arrayA,d_arrayB,d_arrayC,arrLen);
       mega_merge<<<dimGrid, dimBlock>>>(d_arrayA,d_arrayB,d_arrayC,arrLen);
       cudaDeviceSynchronize();
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
  //printArray(h_array_result,SM_ARR_LEN);



//HOST SOR. Use copied SOR function here.
  
  wakeup_delay();
   printf("Host Launch\n");
  data_t time_stamp;
  clock_gettime(CLOCK_REALTIME, &time_start);
    sort(h_arrayA,h_arrayB,h_arrayC_gold,arrLen);
  clock_gettime(CLOCK_REALTIME, &time_stop);
  time_stamp = interval(time_start, time_stop);
  
  printf("CPU Time: %.2f (msec)\n",1000*time_stamp);

//Change to compare SOR'd matrices using difference

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

  // Free-up device and host memory
  CUDA_SAFE_CALL(cudaFree(d_arrayA));
  CUDA_SAFE_CALL(cudaFree(d_arrayB));
  CUDA_SAFE_CALL(cudaFree(d_arrayC));


  free(h_arrayA);
  free(h_arrayB);
  free(h_arrayC);
  free(h_arrayC_gold);

  cudaPrintfEnd();


  return 0;
}



/************************************/

void sort(data_t *arrayA, data_t *arrayB, data_t *arrayC, long int rowlen)
{
  
}


__global__ void sonic_sort(data_t *arrayA, data_t *arrayB,data_t *arrayC,long int rowlen) 
{
   
}

__global__ void mega_merge(data_t *arrayA, data_t *arrayB,data_t *arrayC,long int rowlen) 
{
  int bx = blockIdx.x; 
  int tx = threadIdx.x; 
  
  
}

__device__ int binary_search(data_t *array, int L, int R, int X, int thread_id, int array_len)
{
  /* 
  Variables:
    *array := array to be searched
    L := Leftmost element, usually 0
    R := Rightmost element, array_len - 1.
    X := The variable to search for. "Us" in this context.
    thread_ID := Our CUDA thread number within the block. Correlates to an element number.
    array_len := Length of the input array.

 The overall goal here is to determine how many elements in the other array are smaller than me (X). On ties, the item with the smaller threadID goes first in the output list. The number of smaller items can be represented by M, or our current place in the array. */
  int left_value = 0, center_value = 0, right_value = 0;
  
  while(L <=  R)
  {
    int M = L + (R-L)/2;
    
    if (array[M] == X) /* If I'm the same size as center. Now we compare threadIDs, and lower threadID will come first always */
    {
      /* If we're the same, look at thread ID. If smaller than array_len, I'm first. Otherwise second.*/
      if (thread_id < array_len)
      {
        /* We come first, but need to check what's left, in case its also equal to us. If it is, we bin_jump */
        if (array[M-1] < X ) return M;
        else
        {
          /* We're the same size as center. We have a smaller threadID so we look left and compare. We aren't greater than it, so we need to binary hope to the left to find how many elements come before us in output array. */
          R = M - 1; 
        }
      }
      /* Our threadID is larger, so we need to move right. */
      else if (thread_id > array_len)
      {
        /* We come second, but we need to look right and see if anything else is the same as us. If not, we have to move one element to the right and return. If there's something the same size as us to the right, we need to bin_hop rightwards. */
        if (array[M+1] > X) return M++; 
        else
        {
          L = M + 1;
        }
      }
    }
    /* If I'm bigger than center... */
    if (array[M] < X) 
    {
      /* ...check right.*/
      /* If right is greater than us or equal to us...*/ 
      if (array[M+1] > X)
      { 
        /* Check to see if right is the same as me */
        if (array[M+1] = X)
        {
          /* Check my threadID to see if I go first */
          if (thread_id < array_len) return M++;
          /* Otherwise we need to bin_hop to the right */
          else
          {
            L = M + 1;
          }  
        }
        /* I'm bigger than center, but smaller than R */
        return M++; 
      }
      /* If right isn't larger or equal, we bin_hop right. */
      else
      {
        L = M + 1;
      }
    }
    else  /* If I'm smaller than center */
    {
      /* Look Left and compare */
      /* If we're bigger than left and less than center, we return our current place, as it gives how many elements are smaller than us. */
      if (array[M-1] < X) return M;
      /* If left is the same as us, we need to check thread_id to see who goes first */
      else if (array[M-1] = X)
      {
        /* We check if our threadID is less, indicating we go first. So we bin_hop left, as we don't care about right. */
        if(thread_id < array_len) R = M - 1;
        /* Otherwise we can return M, since center is larger than us. */
        else return M;
      }
    }
  }
  
  /* Need flag to indicate if we're bigger or smaller than everything in the other list */
  
  if(X > array[array_len - 1]) return array_len; /* We're bigger, so we return array length, since we're larger than every element in the other array. */
  else if (X < array[0]) return 0; /* We're smaller, so we return 0 because there are no elements in the other array smaller than us. */
  else return -1; /*Error condition */
  
}

void initializeArray1D(data_t *arr, int len, int seed) {
  long int i;
  srand(seed);
  data_t fRand(data_t fMin, data_t fMax);

  for (i = 0; i < len; i++) 
  {
    arr[i] = (fRand((data_t)(MINVAL),(data_t)(MAXVAL)));  
  }
}

void printArray(data_t *array, int rowlen)
{
  int i;
  for (i=0; i < rowlen; i++)
  {
    printf("%-5.3f   ", array[i]);
  }
 
}

data_t fRand(data_t fMin, data_t fMax)
{
  data_t f = (data_t)random() / RAND_MAX;
  return fMin + f * (fMax - fMin);
}
