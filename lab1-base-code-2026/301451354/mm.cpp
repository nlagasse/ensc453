#include <stdio.h>
#include <stdlib.h>
#include "my_timer.h"

#include <omp.h>

#define NI 2048
#define NJ 2048
#define NK 2048

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
void kernel_gemm(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k, ii, jj, kk;

// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ

// printf("number of devices = %d\n", omp_get_max_threads());

// Tiling
int TILE_SIZE = 16;

omp_set_num_threads(6);
#pragma omp parallel for private (j)
  for (i = 0; i < NI; i+=TILE_SIZE) {
    // J
    for (j = 0; j < NJ; j+=TILE_SIZE) {
      // I
      for(int row = i; row < TILE_SIZE +i; row++){
        for(int col = j; col < TILE_SIZE +j; col++){
          C[row*NJ+col] *= beta;
        }
      }
      // K 
      for(k = 0; k< NK; k+= TILE_SIZE){
        for (ii=i; ii< TILE_SIZE+i && ii<NI; ii++){
          for (jj=j; jj< TILE_SIZE+j && jj<NJ; jj++){              
            for (kk=k; kk< TILE_SIZE+k && kk<NK; kk++){ 
              C[ii*NJ+jj] += alpha * A[ii*NK+kk] * B[kk*NJ+jj];
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

  /* Start timer. */
  timespec timer = tic();

  /* Run kernel. */
  kernel_gemm (C, A, B, 1.5, 2.5);

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
