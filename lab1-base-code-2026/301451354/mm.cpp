#include <stdio.h>
#include <stdlib.h>
#include "my_timer.h"

#include <x86intrin.h>
#include <omp.h>
#include <math.h> 


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

//helper function for transposing a matrix
float* transposeMatrix(const float B[NK*NJ], float transposed[NK*NJ]) {
    for (int i = 0; i < NK; i++) {
        for (int j = 0; j < NJ; j++) {
            transposed[j*NI + i] = B[i*NJ + j];
        }
    }
  return transposed;
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
int TILE_SIZE_i = 16;
int TILE_SIZE_j = 16;
int TILE_SIZE_k = 2048;

__m256 A_row ;
__m256 B_row;
__m256 alpha_vec;

omp_set_num_threads(20);

float *B_trans = (float*)malloc(NK*NJ*sizeof(float));
transposeMatrix(B, B_trans);

#pragma omp parallel for
  for (i = 0; i < NI; i+=TILE_SIZE_i) {
    for (j = 0; j < NJ; j+=TILE_SIZE_j) {
      for(int row = i; row < TILE_SIZE_i +i; row++){
        for(int col = j; col < TILE_SIZE_j +j; col++){
          C[row*NJ+col] *= beta;
        }
      }
      // K 
      for(k = 0; k< NK; k+= TILE_SIZE_k){
        for (ii=i; ii< TILE_SIZE_i+i && ii<NI; ii++){
          for (jj=j; jj< TILE_SIZE_i+j && jj<NJ; jj++){

            __m256 sum = _mm256_setzero_ps();           
            
            for (kk=k; kk< TILE_SIZE_k+k; kk+=16){

              __m256 alpha_vec = _mm256_set1_ps(alpha);

              __m256 A_row_1 = _mm256_loadu_ps(&A[ii*NK+kk]);
              __m256 B_row_1 = _mm256_loadu_ps(&B_trans[jj*NJ+kk]); 

              __m256 A_row_2 = _mm256_loadu_ps(&A[ii*NK+kk+8]);
              __m256 B_row_2 = _mm256_loadu_ps(&B_trans[jj*NJ+kk+8]); 

              sum =_mm256_add_ps(sum, _mm256_mul_ps((_mm256_mul_ps(alpha_vec, A_row_1)), B_row_1));
              sum =_mm256_add_ps(sum, _mm256_mul_ps((_mm256_mul_ps(alpha_vec, A_row_2)), B_row_2));                      
              // C[ii*NJ+jj] += alpha * A[ii*NK+kk] * B[kk*NJ+jj];
            }

            float tempResult[8];
            _mm256_storeu_ps(tempResult, sum);

            for (int index = 0; index < 1; index++) {
              C[ii*NJ+jj] += tempResult[index];
              C[ii*NJ+jj] += tempResult[index + 1];
              C[ii*NJ+jj] += tempResult[index + 2];
              C[ii*NJ+jj] += tempResult[index + 3];
              C[ii*NJ+jj] += tempResult[index + 4];
              C[ii*NJ+jj] += tempResult[index + 5];
              C[ii*NJ+jj] += tempResult[index + 6];
              C[ii*NJ+jj] += tempResult[index + 7];
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
