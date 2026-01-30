#include <stdio.h>
#include <stdlib.h>
#include "my_timer.h"

#include <x86intrin.h>
#include <omp.h>
#include <math.h> 

#define NI 4096
#define NJ 4096
#define NK 4096

#define TILE_N 128
#define Unroll 8

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

// Default function
static
void kernel_gemm(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k;

// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C[i*NJ+j] *= beta;
    }
    for (j = 0; j < NJ; j++) {
      for (k = 0; k < NK; ++k) {
	C[i*NJ+j] += alpha * A[i*NK+k] * B[k*NJ+j];
      }
    }
  }
}

/* Main computational kernel with tiling. */
static
void kernel_gemm_t(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k, ii, jj, kk;

  // Performance for C *= beta is negligible
  for(i = 0; i < NI; i++){
    for(j = 0; j < NJ; j++){
      C[i*NJ+j] *= beta;
    }
  }
  
  // Tiling
  for (ii =  0 ; ii < NI; ii += TILE_N){
    for (kk = 0; kk < NK; kk += TILE_N){
      for(jj = 0; jj < NJ; jj += TILE_N){
      
        // Inside tiles
        for (i = ii; i < ii + TILE_N; i+= Unroll){
          for (k = kk; k < kk + TILE_N; k++){

            // Unrolling a * A
            float A0 = alpha * A[(i + 0) * NK + k];
            float A1 = alpha * A[(i + 1) * NK + k];
            float A2 = alpha * A[(i + 2) * NK + k];
            float A3 = alpha * A[(i + 3) * NK + k];
            float A4 = alpha * A[(i + 4) * NK + k]; 
            float A5 = alpha * A[(i + 5) * NK + k];
            float A6 = alpha * A[(i + 6) * NK + k];
            float A7 = alpha * A[(i + 7) * NK + k];

            for(j = jj; j < jj + TILE_N; j++){
              // Load a chunk of B
              float load_B = B[k*NJ+j];
              // Unrolling A_updated * B
              C[(i + 0) * NJ + j] += A0 * load_B;
              C[(i + 1) * NJ + j] += A1 * load_B;
              C[(i + 2) * NJ + j] += A2 * load_B;
              C[(i + 3) * NJ + j] += A3 * load_B;
              C[(i + 4) * NJ + j] += A4 * load_B;
              C[(i + 5) * NJ + j] += A5 * load_B;
              C[(i + 6) * NJ + j] += A6 * load_B;
              C[(i + 7) * NJ + j] += A7 * load_B;


            }
          }
        }
      }
    }
  }
}

/* Main computational kernel with tiling and vectorization. */
static
void kernel_gemm_tv(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k, ii, jj, kk;

  // Performance for C *= beta is negligible
  for(i = 0; i < NI; i++){
    for(j = 0; j < NJ; j++){
      C[i*NJ+j] *= beta;
    }
  }
  
  // Tiling
  for (ii =  0 ; ii < NI; ii += TILE_N){
    for (kk = 0; kk < NK; kk += TILE_N){
      for(jj = 0; jj < NJ; jj += TILE_N){
      
        // Inside tiles
        for (i = ii; i < ii + TILE_N; i+= Unroll){
          for (k = kk; k < kk + TILE_N; k++){

            // Unrolling a * A
            float A0 = alpha * A[(i + 0) * NK + k];
            float A1 = alpha * A[(i + 1) * NK + k];
            float A2 = alpha * A[(i + 2) * NK + k];
            float A3 = alpha * A[(i + 3) * NK + k];
            float A4 = alpha * A[(i + 4) * NK + k]; 
            float A5 = alpha * A[(i + 5) * NK + k];
            float A6 = alpha * A[(i + 6) * NK + k];
            float A7 = alpha * A[(i + 7) * NK + k];

            // Vectorization
            #pragma omp simd
            for(j = jj; j < jj + TILE_N; j++){
              // Load a chunk of B
              float load_B = B[k*NJ+j];
              // Unrolling A_updated * B
              C[(i + 0) * NJ + j] += A0 * load_B;
              C[(i + 1) * NJ + j] += A1 * load_B;
              C[(i + 2) * NJ + j] += A2 * load_B;
              C[(i + 3) * NJ + j] += A3 * load_B;
              C[(i + 4) * NJ + j] += A4 * load_B;
              C[(i + 5) * NJ + j] += A5 * load_B;
              C[(i + 6) * NJ + j] += A6 * load_B;
              C[(i + 7) * NJ + j] += A7 * load_B;


            }
          }
        }
      }
    }
  }
}

/* Main computational kernel: with tiling, simd, and parallelization optimizations. */
static
void kernel_gemm_tvp(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k, ii, jj, kk;

  // Performance for C *= beta is negligible
  for(i = 0; i < NI; i++){
    for(j = 0; j < NJ; j++){
      C[i*NJ+j] *= beta;
    }
  }
  
  // Parallel for on outer loop
  #pragma omp parallel for private (i,j,k,ii,jj,kk)
  // Tiling
  for (ii =  0 ; ii < NI; ii += TILE_N){
    for (kk = 0; kk < NK; kk += TILE_N){
      for(jj = 0; jj < NJ; jj += TILE_N){
      
        // Inside tiles
        for (i = ii; i < ii + TILE_N; i+= Unroll){
          for (k = kk; k < kk + TILE_N; k++){

            // Unrolling a * A
            float A0 = alpha * A[(i + 0) * NK + k];
            float A1 = alpha * A[(i + 1) * NK + k];
            float A2 = alpha * A[(i + 2) * NK + k];
            float A3 = alpha * A[(i + 3) * NK + k];
            float A4 = alpha * A[(i + 4) * NK + k]; 
            float A5 = alpha * A[(i + 5) * NK + k];
            float A6 = alpha * A[(i + 6) * NK + k];
            float A7 = alpha * A[(i + 7) * NK + k];

            // Vectorization
            #pragma omp simd
            for(j = jj; j < jj + TILE_N; j++){
              // Load a chunk of B
              float load_B = B[k*NJ+j];
              // Unrolling A_updated * B
              C[(i + 0) * NJ + j] += A0 * load_B;
              C[(i + 1) * NJ + j] += A1 * load_B;
              C[(i + 2) * NJ + j] += A2 * load_B;
              C[(i + 3) * NJ + j] += A3 * load_B;
              C[(i + 4) * NJ + j] += A4 * load_B;
              C[(i + 5) * NJ + j] += A5 * load_B;
              C[(i + 6) * NJ + j] += A6 * load_B;
              C[(i + 7) * NJ + j] += A7 * load_B;


            }
          }
        }
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

  /* Initialize array(s). */
  init_array (C, A, B);

  /* Time tiling. */
  timespec timer = tic();

  /* Run kernel. */
  // REPLACE THIS FUNCTION WITH [t/tv/tvp] FOR DIFFERENT EXECUTION
  kernel_gemm_tvp(C, A, B, 1.5, 2.5);

  printf("testing\n");
  /* Stop and print timer. */
  toc(&timer, "kernel execution");
  
  /* Print results. */
  print_array_sum (C);

  /* free memory for A, B, C */
  free(A);
  free(B);
  free(C);
  
  return 0;
}
