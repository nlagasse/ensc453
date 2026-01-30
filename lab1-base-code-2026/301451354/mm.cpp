#include <stdio.h>
#include <stdlib.h>
#include "my_timer.h"

#include <x86intrin.h>
#include <omp.h>
#include <math.h> 

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

/* Main computational kernel with tiling. The whole function will be timed, 
including the call and return. */
static
void kernel_gemm_T(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k, ii, jj, kk;

  const int tileSize_i = 128;
  const int tileSize_j = 256;
  const int tileSize_k = 64;

  for(i = 0; i < NI; i += tileSize_i){
    for(j = 0; j < NJ; j += tileSize_j){

      // beta tiled
      for(ii = i; ii < i+tileSize_i && ii < NI; ii++){
        for(jj = j; jj < j+tileSize_j && jj < NJ; jj+=8){

          C[ii*NJ + jj] *= beta;

        }
      }

      // gemm tiled
      for(k = 0; k < NK; k += tileSize_k){
        for(ii = i; ii < i+tileSize_i && ii < NI; ii+=4){
          for(jj = j; jj < j+tileSize_j && jj < NJ; jj+=16){

            float sum = 0.0f;

            for (kk = k; kk < k + tileSize_k && kk < NK; kk++) {
              sum += A[ii*NK + kk] * B[kk*NJ + jj];
            }

            C[ii*NJ + jj] += alpha * sum;

          }
        }
      }

    }
  }

}

/* Main computational kernel with tiling and vectorization. The whole function will be timed,
   including the call and return. */
static
__attribute__((target("avx2,fma")))
void kernel_gemm_TV(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k, ii, jj, kk;

  const int tileSize_i = 128;
  const int tileSize_j = 256;
  const int tileSize_k = 64;

  for(i = 0; i < NI; i += tileSize_i){
    for(j = 0; j < NJ; j += tileSize_j){
    
      //beta
      for(ii = i; ii < i+tileSize_i && ii < NI; ii++){
        for(jj = j; jj < j+tileSize_j && jj < NJ; jj+=8){

          __m256 cV = _mm256_loadu_ps(&C[(ii << 12) + jj]);
          cV = _mm256_mul_ps(cV, _mm256_set1_ps(beta));
          _mm256_storeu_ps(&C[((ii+0)<<12) + jj], cV);

        }
      }
      for(k = 0; k < NK; k += tileSize_k){
        for(ii = i; ii < i+tileSize_i && ii < NI; ii+=4){
          for(jj = j; jj < j+tileSize_j && jj < NJ; jj+=16){
           
            // unrolling
            __m256 sumV[8];
            sumV[0] = _mm256_setzero_ps();
            sumV[1] = _mm256_setzero_ps();
            sumV[2] = _mm256_setzero_ps();
            sumV[3] = _mm256_setzero_ps();
            sumV[4] = _mm256_setzero_ps();
            sumV[5] = _mm256_setzero_ps();
            sumV[6] = _mm256_setzero_ps();
            sumV[7] = _mm256_setzero_ps();

            for(kk = k; kk < k+tileSize_k && kk < NK; kk++){
              __m256 aV[4];
              aV[0] = _mm256_set1_ps(A[((ii+0)<<12)+kk]);
              aV[1] = _mm256_set1_ps(A[((ii+1)<<12)+kk]);
              aV[2] = _mm256_set1_ps(A[((ii+2)<<12)+kk]);
              aV[3] = _mm256_set1_ps(A[((ii+3)<<12)+kk]);

              __m256 bV[2];
              bV[0] = _mm256_loadu_ps(&B[(kk<<12) + (jj)]);
              bV[1] = _mm256_loadu_ps(&B[(kk<<12) + (jj+8)]);
              
              sumV[0] = _mm256_add_ps(sumV[0], _mm256_mul_ps(aV[0], bV[0]));
              sumV[1] = _mm256_add_ps(sumV[1], _mm256_mul_ps(aV[0], bV[1]));
              sumV[2] = _mm256_add_ps(sumV[2], _mm256_mul_ps(aV[1], bV[0]));
              sumV[3] = _mm256_add_ps(sumV[3], _mm256_mul_ps(aV[1], bV[1]));
              sumV[4] = _mm256_add_ps(sumV[4], _mm256_mul_ps(aV[2], bV[0]));
              sumV[5] = _mm256_add_ps(sumV[5], _mm256_mul_ps(aV[2], bV[1]));
              sumV[6] = _mm256_add_ps(sumV[6], _mm256_mul_ps(aV[3], bV[0]));
              sumV[7] = _mm256_add_ps(sumV[7], _mm256_mul_ps(aV[3], bV[1]));           
            }

            //MULTIPLICATION BY ALPHA:  
            //alpha = 1.5, therefore sumV * alpha = 0.5*sumV + sumV
            __m256 half = _mm256_set1_ps(0.5f);

            sumV[0] = _mm256_add_ps(sumV[0], _mm256_mul_ps(sumV[0], half));
            sumV[1] = _mm256_add_ps(sumV[1], _mm256_mul_ps(sumV[1], half));
            sumV[2] = _mm256_add_ps(sumV[2], _mm256_mul_ps(sumV[2], half));
            sumV[3] = _mm256_add_ps(sumV[3], _mm256_mul_ps(sumV[3], half));
            sumV[4] = _mm256_add_ps(sumV[4], _mm256_mul_ps(sumV[4], half));
            sumV[5] = _mm256_add_ps(sumV[5], _mm256_mul_ps(sumV[5], half));
            sumV[6] = _mm256_add_ps(sumV[6], _mm256_mul_ps(sumV[6], half));
            sumV[7] = _mm256_add_ps(sumV[7], _mm256_mul_ps(sumV[7], half));


            __m256 cV[8];
            cV[0] = _mm256_loadu_ps(&C[((ii+0)<<12) + (jj)]);
            cV[1] = _mm256_loadu_ps(&C[((ii+0)<<12) + (jj+8)]);
            cV[2] = _mm256_loadu_ps(&C[((ii+1)<<12) + (jj)]);
            cV[3] = _mm256_loadu_ps(&C[((ii+1)<<12) + (jj+8)]);
            cV[4] = _mm256_loadu_ps(&C[((ii+2)<<12) + (jj)]);
            cV[5] = _mm256_loadu_ps(&C[((ii+2)<<12) + (jj+8)]);
            cV[6] = _mm256_loadu_ps(&C[((ii+3)<<12) + (jj)]);
            cV[7] = _mm256_loadu_ps(&C[((ii+3)<<12) + (jj+8)]);

            cV[0] = _mm256_add_ps(cV[0], sumV[0]);
            cV[1] = _mm256_add_ps(cV[1], sumV[1]);
            cV[2] = _mm256_add_ps(cV[2], sumV[2]);
            cV[3] = _mm256_add_ps(cV[3], sumV[3]);
            cV[4] = _mm256_add_ps(cV[4], sumV[4]);
            cV[5] = _mm256_add_ps(cV[5], sumV[5]);
            cV[6] = _mm256_add_ps(cV[6], sumV[6]);
            cV[7] = _mm256_add_ps(cV[7], sumV[7]);

            _mm256_storeu_ps(&C[((ii+0)<<12) + (jj)], cV[0]);
            _mm256_storeu_ps(&C[((ii+0)<<12) + (jj+8)], cV[1]);
            _mm256_storeu_ps(&C[((ii+1)<<12) + (jj)], cV[2]);
            _mm256_storeu_ps(&C[((ii+1)<<12) + (jj+8)], cV[3]);
            _mm256_storeu_ps(&C[((ii+2)<<12) + (jj)], cV[4]);
            _mm256_storeu_ps(&C[((ii+2)<<12) + (jj+8)], cV[5]);
            _mm256_storeu_ps(&C[((ii+3)<<12) + (jj)], cV[6]);
            _mm256_storeu_ps(&C[((ii+3)<<12) + (jj+8)], cV[7]);
          }
        }
      }
    }
  }
}

/* Main computational kernel: with tiling, simd, and parallelization optimizations. */
static
__attribute__((target("avx2,fma")))
void kernel_gemm_TVP(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k, ii, jj, kk;

// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
  const int tileSize_i = 128;
  const int tileSize_j = 256;
  const int tileSize_k = 64;

  // __m256 alphaV = _mm256_set1_ps(alpha);
  // __m256 betaV  = _mm256_set1_ps(beta);

  omp_set_num_threads(20);
  #pragma omp parallel for collapse(2) private(i,j,k,ii,jj,kk)
  for(i = 0; i < NI; i += tileSize_i){
    for(j = 0; j < NJ; j += tileSize_j){
      for(ii = i; ii < i+tileSize_i && ii < NI; ii++){
        for(jj = j; jj < j+tileSize_j && jj < NJ; jj+=8){

          //multiplying by BETA here: 
          __m256 cV = _mm256_loadu_ps(&C[(ii << 12) + jj]);
          cV = _mm256_mul_ps(cV, _mm256_set1_ps(beta));
          _mm256_storeu_ps(&C[((ii+0)<<12) + jj], cV);
        }
      }
      for(k = 0; k < NK; k += tileSize_k){
        for(ii = i; ii < i+tileSize_i && ii < NI; ii+=4){
          for(jj = j; jj < j+tileSize_j && jj < NJ; jj+=16){
           
            // unrolling
            __m256 sumV[8];
            sumV[0] = _mm256_setzero_ps();
            sumV[1] = _mm256_setzero_ps();
            sumV[2] = _mm256_setzero_ps();
            sumV[3] = _mm256_setzero_ps();
            sumV[4] = _mm256_setzero_ps();
            sumV[5] = _mm256_setzero_ps();
            sumV[6] = _mm256_setzero_ps();
            sumV[7] = _mm256_setzero_ps();

            for(kk = k; kk < k+tileSize_k && kk < NK; kk++){
              __m256 aV[4];
              aV[0] = _mm256_set1_ps(A[((ii+0)<<12)+kk]);
              aV[1] = _mm256_set1_ps(A[((ii+1)<<12)+kk]);
              aV[2] = _mm256_set1_ps(A[((ii+2)<<12)+kk]);
              aV[3] = _mm256_set1_ps(A[((ii+3)<<12)+kk]);

              __m256 bV[2];
              bV[0] = _mm256_loadu_ps(&B[(kk<<12) + (jj)]);
              bV[1] = _mm256_loadu_ps(&B[(kk<<12) + (jj+8)]);
              
              sumV[0] = _mm256_add_ps(sumV[0], _mm256_mul_ps(aV[0], bV[0]));
              sumV[1] = _mm256_add_ps(sumV[1], _mm256_mul_ps(aV[0], bV[1]));
              sumV[2] = _mm256_add_ps(sumV[2], _mm256_mul_ps(aV[1], bV[0]));
              sumV[3] = _mm256_add_ps(sumV[3], _mm256_mul_ps(aV[1], bV[1]));
              sumV[4] = _mm256_add_ps(sumV[4], _mm256_mul_ps(aV[2], bV[0]));
              sumV[5] = _mm256_add_ps(sumV[5], _mm256_mul_ps(aV[2], bV[1]));
              sumV[6] = _mm256_add_ps(sumV[6], _mm256_mul_ps(aV[3], bV[0]));
              sumV[7] = _mm256_add_ps(sumV[7], _mm256_mul_ps(aV[3], bV[1]));           
            }

            //MULTIPLICATION BY ALPHA:  
            //alpha = 1.5, therefore sumV * alpha = 0.5*sumV + sumV
            __m256 half = _mm256_set1_ps(0.5f);

            sumV[0] = _mm256_add_ps(sumV[0], _mm256_mul_ps(sumV[0], half));
            sumV[1] = _mm256_add_ps(sumV[1], _mm256_mul_ps(sumV[1], half));
            sumV[2] = _mm256_add_ps(sumV[2], _mm256_mul_ps(sumV[2], half));
            sumV[3] = _mm256_add_ps(sumV[3], _mm256_mul_ps(sumV[3], half));
            sumV[4] = _mm256_add_ps(sumV[4], _mm256_mul_ps(sumV[4], half));
            sumV[5] = _mm256_add_ps(sumV[5], _mm256_mul_ps(sumV[5], half));
            sumV[6] = _mm256_add_ps(sumV[6], _mm256_mul_ps(sumV[6], half));
            sumV[7] = _mm256_add_ps(sumV[7], _mm256_mul_ps(sumV[7], half));


            __m256 cV[8];
            cV[0] = _mm256_loadu_ps(&C[((ii+0)<<12) + (jj)]);
            cV[1] = _mm256_loadu_ps(&C[((ii+0)<<12) + (jj+8)]);
            cV[2] = _mm256_loadu_ps(&C[((ii+1)<<12) + (jj)]);
            cV[3] = _mm256_loadu_ps(&C[((ii+1)<<12) + (jj+8)]);
            cV[4] = _mm256_loadu_ps(&C[((ii+2)<<12) + (jj)]);
            cV[5] = _mm256_loadu_ps(&C[((ii+2)<<12) + (jj+8)]);
            cV[6] = _mm256_loadu_ps(&C[((ii+3)<<12) + (jj)]);
            cV[7] = _mm256_loadu_ps(&C[((ii+3)<<12) + (jj+8)]);

            cV[0] = _mm256_add_ps(cV[0], sumV[0]);
            cV[1] = _mm256_add_ps(cV[1], sumV[1]);
            cV[2] = _mm256_add_ps(cV[2], sumV[2]);
            cV[3] = _mm256_add_ps(cV[3], sumV[3]);
            cV[4] = _mm256_add_ps(cV[4], sumV[4]);
            cV[5] = _mm256_add_ps(cV[5], sumV[5]);
            cV[6] = _mm256_add_ps(cV[6], sumV[6]);
            cV[7] = _mm256_add_ps(cV[7], sumV[7]);

            _mm256_storeu_ps(&C[((ii+0)<<12) + (jj)], cV[0]);
            _mm256_storeu_ps(&C[((ii+0)<<12) + (jj+8)], cV[1]);
            _mm256_storeu_ps(&C[((ii+1)<<12) + (jj)], cV[2]);
            _mm256_storeu_ps(&C[((ii+1)<<12) + (jj+8)], cV[3]);
            _mm256_storeu_ps(&C[((ii+2)<<12) + (jj)], cV[4]);
            _mm256_storeu_ps(&C[((ii+2)<<12) + (jj+8)], cV[5]);
            _mm256_storeu_ps(&C[((ii+3)<<12) + (jj)], cV[6]);
            _mm256_storeu_ps(&C[((ii+3)<<12) + (jj+8)], cV[7]);
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
  // REPLACE THIS FUNCTION WITH T, TV, OR TVP FOR DIFFERENT EXECUTION
  kernel_gemm_T(C, A, B, 1.5, 2.5);

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
