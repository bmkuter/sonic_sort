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

typedef float data_t;

/* Prototypes */
void printArray(float *array, int rowlen);
void initializeArray1D(float *arr, int len, int seed);
void sort(float *arrayA, float *arrayB,float *arrayC,long int rowlen);
__global__ void sonic_sort(float *arrayA, float *arrayB,float *arrayC,long int rowlen);
__global__ void mega_merge(float *arrayA, float *arrayB,float *arrayC,long int rowlen);
double fRand(double fMin, double fMax);

/* Prototypes */
void printArray(float *array, int rowlen);

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
  float *d_arrayA;
  float *d_arrayB;
  float *d_arrayC;

  // Arrays on the host memory
  float *h_arrayA;
  float *h_arrayB;
  float *h_arrayC;
  float *h_arrayC_gold;

  int errCount = 0, zeroCount = 0;

  size_t allocSize = arrLen*sizeof(float);
  
  // Allocate arrays on host memory
  h_arrayA                   = (float *) malloc(allocSize);
  h_arrayB                   = (float *) malloc(allocSize);
  h_arrayC                   = (float *) calloc(arrLen,sizeof(float));    //Output for Device
  h_arrayC_gold              = (float *) calloc(arrLen,sizeof(float));    //Validation array
  
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
  for (i = 0; i < THRESHOLD; i++)
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
  double time_stamp;
  clock_gettime(CLOCK_REALTIME, &time_start);
    sort(h_arrayA,h_arrayB,h_arrayC_gold,arrLen);
  clock_gettime(CLOCK_REALTIME, &time_stop);
  time_stamp = interval(time_start, time_stop);
  
  printf("CPU Time: %.2f (msec)\n",1000*time_stamp);

//Change to compare SOR'd matrices using difference

printf("Compare\n");
  // Compare the results
  for(int i = 0; i < arrLen; i++) {
      for(int j = 0; j < arrLen; j++) {    //FIX TOLERANCE??
        if (abs(h_arrayC_gold[i*arrLen+j] - h_arrayC[i*arrLen+j])/((h_arrayC_gold[i*arrLen+j] + h_arrayC[i*arrLen+j])*.5) > COMPARE_TOL) {
          errCount++;
        }
        if (h_arrayC[i*arrLen+j] == 0) {
          zeroCount++;
        }
      }
  }

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

void sort(float *arrayA, float *arrayB, float *arrayC, long int rowlen)
{

}


__global__ void sonic_sort(float *arrayA, float *arrayB,float *arrayC,long int rowlen) 
{
 
}

__global__ void mega_merge(float *arrayA, float *arrayB,float *arrayC,long int rowlen) 
{
 
}

void initializeArray1D(float *arr, int len, int seed) {
  long int i;
  srand(seed);
  double fRand(double fMin, double fMax);

  for (i = 0; i < len; i++) 
  {
    arr[i] = (fRand((double)(MINVAL),(double)(MAXVAL)));  
  }
}

void printArray(float *array, int rowlen)
{
  int i;
  for (i=0; i < rowlen; i++)
  {
    printf("%-5.3f   ", array[i]);
  }

}

double fRand(double fMin, double fMax)
{
  double f = (double)random() / RAND_MAX;
  return fMin + f * (fMax - fMin);
}
