#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "my_timer.h"

#define NI 4096
#define NJ 4096
#define NK 4096

/* Array initialization. */
static
void init_array(float C[NI*NJ], float A[NI*NK], float B[NK*NJ])
{
  int i, j;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      C[i*NJ+j] = (float)((i*j+1) % NI) / NI;
  for (i = 0; i < NI; i++)
    for (j = 0; j < NK; j++)
      A[i*NK+j] = (float)(i*(j+1) % NK) / NK;
  for (i = 0; i < NK; i++)
    for (j = 0; j < NJ; j++)
      B[i*NJ+j] = (float)(i*(j+2) % NJ) / NJ;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(float C[NI*NJ])
{
  int i, j;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      printf("C[%d][%d] = %f\n", i, j, C[i*NJ+j]);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array_sum(float C[NI*NJ])
{
  int i, j;

  float sum = 0.0;
  
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      sum += C[i*NJ+j];

  printf("sum of C array = %f\n", sum);
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemm(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta, int num_threads)
{

/*===========================GENERAL COMMENTS======================================*/
//pragma omp parrallel for private(variable) <-- makes a certain variable "private", and it wont be treated as shared memory. 
//NOTE: any variables defined BEFORE the #pragma statement is read as a SHARED variable
//once a variable is privatized, it is UNDEFINED until you give it a value
//pragma omp master --> only the master thread works
//pragma omp single nowait --> idle threads do not have to join with working threads after task finishes
/*=================================================================================*/

/*===========================OPTIMIZATION COMMENTS=================================*/
//parallelizing the outer loop (i loop) is the most efficient way to parallelize this
//because each iteration of the i loop is independent of each other (no data dependency)
//so there is no need for synchronization between threads
//private(i,j,k) makes i,j,k private variables for each thread
//num_threads(num_threads) sets the number of threads to use
/*=================================================================================*/

/*===========================TILING COMMENTS=======================================*/
int tile_size = 32; 
/*=================================================================================*/


int i, j, k;
// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
#pragma omp parallel num_threads(num_threads) private (i,j,k)
{
  #pragma omp for 
  for (int i = 0; i < NI/tile_size; i++) {
      // for (int j = 0; j < NJ/tile_size; j++) {
      //     for (int ii = 0; ii < tile_size; ii++) {
      //         for (int jj = 0; jj < tile_size; jj++) {
      //             C[(i*tile_size+ii)*NJ+(j*tile_size+jj)] *= beta;
      //         }
      //     }
      //     for (int k = 0; k < NK; k++) {
      //         for (int ii = 0; ii < tile_size; ii++) {
      //             for (int jj = 0; jj < tile_size; jj++) {
      //                 C[(i*tile_size+ii)*NJ+(j*tile_size+jj)] += alpha * A[(i*tile_size+ii)*NK+k] * B[k*NJ+(j*tile_size+jj)];
      //             }
      //         }
      //     }
      // }


      for (int j = 0; j < NJ/tile_size; j++) {
        for (int ii )
          C[i*NJ+j] *= beta;
          for (int k = 0; k < NK; k++) {
              C[i*NJ+j] += alpha * A[i*NK+k] * B[k*NJ+j];
          }
      }
  }
}


int main(int argc, char** argv)
{
  /* Variable declaration/allocation. */
  float *A = (float *)malloc(NI*NK*sizeof(float));
  float *B = (float *)malloc(NK*NJ*sizeof(float));
  float *C = (float *)malloc(NI*NJ*sizeof(float));

  bool parallel_outer = false;
  int choice;
 
  int num_threads = 1;
  while (1){
    printf("Enter number of threads: "); 
    scanf("%d", &num_threads);
    if (num_threads > 0){
      break;
    }
    else{
      printf("Invalid input. The number of threads cannot be less than or equal to 0\n\n");
    }

  }

  
  float sum = 0;
  int trials = 3;
  for (int i = 0; i<trials; i++){
    /* Initialize array(s). */
    init_array (C, A, B);

    /* Start timer. */
    timespec timer = tic();

    /* Run kernel. */
    kernel_gemm (C, A, B, 1.5, 2.5, num_threads);
    
    /* Stop and print timer. */
    sum += toc(&timer, "kernel execution");
  }
  
  /* Print results. */
  print_array_sum (C);

  //printing the average time: 
  printf("average time cost: %f \n", sum / trials);
  
  /* free memory for A, B, C */
  free(A);
  free(B);
  free(C);

  return 0;
}
