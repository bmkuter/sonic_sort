/*
   Minimal CUDA program, intended just to test ability
   to compile and run a CUDA program


     
     nvcc -arch sm_35 mmm_shared.cu -o mmm_shared
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

#define CPNS 3.0

#define BSIZE 1

#define NUM_TESTS 1
#define OPTIONS 1

#define MINVAL   0.0
#define MAXVAL  10.0

typedef float data_t;

void initializeArray2D(float *arr, int len, int seed);
void MMM(float *matrixA, float *matrixB,float *matrixC,long int rowlen);
__global__ void communist_kernel_MMM(float *matrixA, float *matrixB,float *matrixC,long int rowlen);
double fRand(double fMin, double fMax);

/* Prototypes */
void initializeArray2D(float *arr, int len, int seed);
void printMatrix(float *matrix, int rowlen);

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
  long int sideLength;
  struct timespec time_start, time_stop;
  printf("Test\n");
  
  wakeup_delay();
  
  if (argc > 1) {
    arrLen  = (atoi(argv[1]))*(atoi(argv[1]));
    sideLength = atoi(argv[1]);
  }
  else {
    arrLen = (SM_ARR_LEN)*(SM_ARR_LEN);
    sideLength = SM_ARR_LEN;
  }

  printf("Side Length = %d\n", sideLength);

  cudaPrintfInit();


  // GPU Timing variables
  cudaEvent_t start, stop;
  float elapsed_gpu;

  // Arrays on GPU global memory
  float *d_matrixA;
  float *d_matrixB;
  float *d_matrixC;

  // Arrays on the host memory
  float *h_matrixA;
  float *h_matrixB;
  float *h_matrixC;
  float *h_matrixC_gold;

  int errCount = 0, zeroCount = 0;

  size_t allocSize = arrLen*sizeof(float);


  
  // Allocate arrays on host memory
  h_matrixA                   = (float *) malloc(allocSize);
  h_matrixB                   = (float *) malloc(allocSize);
  h_matrixC                   = (float *) calloc(arrLen,sizeof(float));    //Output for Device
  h_matrixC_gold              = (float *) calloc(arrLen,sizeof(float));    //Validation Matrix
  
  if (!h_matrixA) {
      free((void *) h_matrixA);
      printf("\n COULDN'T ALLOCATE STORAGE FOR h_matrix \n");
      return -1;  /* Couldn't allocate storage */
  }
  printf("Host Init\n");
  // Initialize the host arrays
  // Arrys are initialized with a known seed for reproducability
  initializeArray2D(h_matrixA, sideLength, 123);
  initializeArray2D(h_matrixB, sideLength, 351);
  initializeArray2D(h_matrixC, sideLength, 12);
  printf("Host Init Finished\n");
  //printMatrix(h_matrixA,sideLength);
  //printf("\n");
  //printMatrix(h_matrixB,sideLength);
  //printf("\n");
  //printMatrix(h_matrixC,sideLength);





  // Select GPU
  CUDA_SAFE_CALL(cudaSetDevice(0));
  printf("Alloc\n");
  // Allocate GPU memory
  
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_matrixA, allocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_matrixC, allocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_matrixB, allocSize));

  // Transfer the arrays to the GPU memory
  CUDA_SAFE_CALL(cudaMemcpy(d_matrixA, h_matrixA, allocSize, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_matrixB, h_matrixB, allocSize, cudaMemcpyHostToDevice));
  printf("Kernel Launch\n");
   //Launch the kernel(nuke)   SM_ARR_LEN
     //define block geometry? 
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 dimGrid(sideLength/TILE_WIDTH, sideLength/TILE_WIDTH); 
  
  #if PRINT_TIME
  // Create the cuda events
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // Record event on the default stream
  cudaEventRecord(start, 0);
#endif
      //Communist because shared resources  
       communist_kernel_MMM<<<dimGrid, dimBlock>>>(d_matrixA,d_matrixB,d_matrixC,sideLength);
       cudaDeviceSynchronize();
    
    #if PRINT_TIME
  // Stop and destroy the timer
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
  CUDA_SAFE_CALL(cudaMemcpy(h_matrixC, d_matrixC, allocSize, cudaMemcpyDeviceToHost));
  //printMatrix(h_matrix_result,SM_ARR_LEN);



//HOST SOR. Use copied SOR function here.
  
  wakeup_delay();
   printf("Host Launch\n");
  double time_stamp;
  clock_gettime(CLOCK_REALTIME, &time_start);
    MMM(h_matrixA,h_matrixB,h_matrixC_gold,sideLength);
  clock_gettime(CLOCK_REALTIME, &time_stop);
  time_stamp = interval(time_start, time_stop);
  
  printf("CPU Time: %.2f (msec)\n",1000*time_stamp);

//Change to compare SOR'd matrices using difference

printf("Compare\n");
  // Compare the results
  for(int i = 0; i < sideLength; i++) {
      for(int j = 0; j < sideLength; j++) {    //FIX TOLERANCE??
        if (abs(h_matrixC_gold[i*sideLength+j] - h_matrixC[i*sideLength+j])/((h_matrixC_gold[i*sideLength+j] + h_matrixC[i*sideLength+j])*.5) > COMPARE_TOL) {
          errCount++;
        }
        if (h_matrixC[i*sideLength+j] == 0) {
          zeroCount++;
        }
      }
  }

  //printMatrix(h_matrixC_gold,sideLength);
  //printf("^CPU   vGPU\n");
  //printMatrix(h_matrixC,sideLength);

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
  CUDA_SAFE_CALL(cudaFree(d_matrixA));
  CUDA_SAFE_CALL(cudaFree(d_matrixB));
  CUDA_SAFE_CALL(cudaFree(d_matrixC));


  free(h_matrixA);
  free(h_matrixB);
  free(h_matrixC);
  free(h_matrixC_gold);

  cudaPrintfEnd();


  return 0;
}



/************************************/

void MMM(float *matrixA, float *matrixB, float *matrixC, long int rowlen)
{//Blocking kij
  long int i, j, k;
  
  for (k = 0; k < rowlen; k++){
    for (i = 0; i < rowlen; i++){
      for (j = 0; j < rowlen; j++){
        matrixC[i*rowlen+j] += matrixA[i*rowlen + k]*matrixB[k*rowlen + j];
        //printf("Sum: %f\n", sum);
      }
    }
  }
}


__global__ void communist_kernel_MMM(float *matrixA, float *matrixB,float *matrixC,long int rowlen) 
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH]; // Shared memory
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH]; // declarations
    
    int bx = blockIdx.x; int by = blockIdx.y; // ID thread
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    // Identify the row and column of the Pd element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    
    float sum = 0; // REGISTER!
    
    // Loop over the Md and Nd tiles required to compute the Pd element
    for (int m = 0; m < rowlen/TILE_WIDTH; ++m)
    {
        // Collaborative loading of Md and Nd tiles into shared memory
        Mds[ty][tx] = matrixA[Row*rowlen + (m*TILE_WIDTH + tx)];
        Nds[ty][tx] = matrixB[Col + (m*TILE_WIDTH + ty)*rowlen];
        
        __syncthreads();
        
        for (int k = 0; k < TILE_WIDTH; ++k)
        sum += Mds[ty][k] * Nds[k][tx];
        
        __syncthreads();
    }
    matrixC[Row*rowlen+Col] = sum;
}

void initializeArray2D(float *arr, int len, int seed) {
  long int i,j;
  //float randNum;
  srand(seed);
  double fRand(double fMin, double fMax);

  for (i = 0; i < len; i++) {
    for (j= 0; j < len; j++)
    {
        arr[i*len+j] = (fRand((double)(MINVAL),(double)(MAXVAL)));  
    }
  }
}

void printMatrix(float *matrix, int rowlen)
{
  int i,j;
  for (i=0; i < rowlen; i++)
  {
    for (j=0; j < rowlen; j++)
    {
      printf("%-5.3f   ", matrix[i*rowlen+j]);
    }
    
    printf("\n");
  }

}

double fRand(double fMin, double fMax)
{
  double f = (double)random() / RAND_MAX;
  return fMin + f * (fMax - fMin);
}
















