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
// static
// __attribute__((target("avx2")))
// void kernel_gemm(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
// {
//   int i, j, k, ii, jj, kk, row, col;

// // => Form C := alpha*A*B + beta*C,
// //A is NIxNK
// //B is NKxNJ
// //C is NIxNJ

// // printf("number of devices = %d\n", omp_get_max_threads());

// // Tiling
// int TILE_SIZE_i = 32;
// int TILE_SIZE_j = 32;
// int TILE_SIZE_k = 4096;

// __m256 A_row;
// __m256 B_row;
// __m256 alpha_vec;

// omp_set_num_threads(20);

// float *B_trans = (float*)malloc(NK*NJ*sizeof(float));
// transposeMatrix(B, B_trans);

// #pragma omp parallel for private (j)
//   for (i = 0; i < NI; i+=TILE_SIZE_i) {
//     for (j = 0; j < NJ; j+=TILE_SIZE_j) {
//       for(int row = i; row < TILE_SIZE_i +i; row++){
//         for(int col = j; col < TILE_SIZE_j +j; col++){
//           C[row*NJ+col] *= beta;
//         }
//       }
//       // K
//       // #pragma omp parallel for
//       for(k = 0; k< NK; k+= TILE_SIZE_k){
//         for (ii=i; ii< TILE_SIZE_i+i && ii<NI; ii++){
//           for (jj=j; jj< TILE_SIZE_i+j && jj<NJ; jj++){

//             __m256 sum = _mm256_setzero_ps();           
            
//             for (kk=k; kk< TILE_SIZE_k+k; kk+=16){

//               __m256 alpha_vec = _mm256_set1_ps(alpha);

//               __m256 A_row_1 = _mm256_loadu_ps(&A[ii*NK+kk]);
//               __m256 B_row_1 = _mm256_loadu_ps(&B_trans[jj*NJ+kk]); 

//               __m256 A_row_2 = _mm256_loadu_ps(&A[ii*NK+kk+8]);
//               __m256 B_row_2 = _mm256_loadu_ps(&B_trans[jj*NJ+kk+8]); 

//               sum =_mm256_add_ps(sum, _mm256_mul_ps((_mm256_mul_ps(alpha_vec, A_row_1)), B_row_1));
//               sum =_mm256_add_ps(sum, _mm256_mul_ps((_mm256_mul_ps(alpha_vec, A_row_2)), B_row_2));                      
//               // C[ii*NJ+jj] += alpha * A[ii*NK+kk] * B[kk*NJ+jj];
//             }

//             float tempResult[8];
//             _mm256_storeu_ps(tempResult, sum);

//             for (int index = 0; index < 1; index++) {
//               C[ii*NJ+jj] += tempResult[index];
//               C[ii*NJ+jj] += tempResult[index + 1];
//               C[ii*NJ+jj] += tempResult[index + 2];
//               C[ii*NJ+jj] += tempResult[index + 3];
//               C[ii*NJ+jj] += tempResult[index + 4];
//               C[ii*NJ+jj] += tempResult[index + 5];
//               C[ii*NJ+jj] += tempResult[index + 6];
//               C[ii*NJ+jj] += tempResult[index + 7];
//             }
//           }
//         }
//       }
//     }
//   }
// }

/* Main computational kernel: with tiling, simd, and parallelization optimizations. */
static
__attribute__((target("avx2")))
void kernel_gemm(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k, ii, jj, kk;

// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
  
  const int tileSize = 64;

  __m256 alphaV = _mm256_set1_ps(alpha);
  __m256 betaV = _mm256_set1_ps(beta);

  // float *B_trans = (float*)malloc(NK*NJ*sizeof(float));
  // transposeMatrix(B, B_trans);

  omp_set_num_threads(20);
  #pragma omp parallel for
  for(i = 0; i < NI; i += tileSize){
    for(j = 0; j < NJ; j += tileSize){

      for(ii = i; ii < i+tileSize && ii < NI; ii++){
        for(jj = j; jj < j+tileSize && jj < NJ; jj+=8){
          __m256 cV = _mm256_loadu_ps(&C[ii*NJ + jj]);
          cV = _mm256_mul_ps(cV, betaV);
          _mm256_storeu_ps(&C[ii*NJ + jj], cV);
        }
      }

      for(k = 0; k < NK; k += tileSize){

        for(ii = i; ii < i+tileSize && ii < NI; ii+=4){
          for(jj = j; jj < j+tileSize && jj < NJ; jj+=2*8){
           
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

            for(kk = k; kk < k+tileSize && kk < NK; kk++){
              __m256 aV[4];
              aV[0] = _mm256_set1_ps(A[(ii+0)*NK+kk]);
              aV[1] = _mm256_set1_ps(A[(ii+1)*NK+kk]);
              aV[2] = _mm256_set1_ps(A[(ii+2)*NK+kk]);
              aV[3] = _mm256_set1_ps(A[(ii+3)*NK+kk]);

              __m256 bV[2];
              bV[0] = _mm256_loadu_ps(&B[kk*NJ + (jj+0*8)]);
              bV[1] = _mm256_loadu_ps(&B[kk*NJ + (jj+1*8)]);

              sumV[0] = _mm256_add_ps(sumV[0], _mm256_mul_ps(alphaV, _mm256_mul_ps(aV[0], bV[0])));
              sumV[1] = _mm256_add_ps(sumV[1], _mm256_mul_ps(alphaV, _mm256_mul_ps(aV[0], bV[1])));
              sumV[2] = _mm256_add_ps(sumV[2], _mm256_mul_ps(alphaV, _mm256_mul_ps(aV[1], bV[0])));
              sumV[3] = _mm256_add_ps(sumV[3], _mm256_mul_ps(alphaV, _mm256_mul_ps(aV[1], bV[1])));
              sumV[4] = _mm256_add_ps(sumV[4], _mm256_mul_ps(alphaV, _mm256_mul_ps(aV[2], bV[0])));
              sumV[5] = _mm256_add_ps(sumV[5], _mm256_mul_ps(alphaV, _mm256_mul_ps(aV[2], bV[1])));
              sumV[6] = _mm256_add_ps(sumV[6], _mm256_mul_ps(alphaV, _mm256_mul_ps(aV[3], bV[0])));
              sumV[7] = _mm256_add_ps(sumV[7], _mm256_mul_ps(alphaV, _mm256_mul_ps(aV[3], bV[1])));

            }

            __m256 cV[8];
            cV[0] = _mm256_loadu_ps(&C[(ii+0)*NJ + (jj+0*8)]);
            cV[1] = _mm256_loadu_ps(&C[(ii+0)*NJ + (jj+1*8)]);
            cV[2] = _mm256_loadu_ps(&C[(ii+1)*NJ + (jj+0*8)]);
            cV[3] = _mm256_loadu_ps(&C[(ii+1)*NJ + (jj+1*8)]);
            cV[4] = _mm256_loadu_ps(&C[(ii+2)*NJ + (jj+0*8)]);
            cV[5] = _mm256_loadu_ps(&C[(ii+2)*NJ + (jj+1*8)]);
            cV[6] = _mm256_loadu_ps(&C[(ii+3)*NJ + (jj+0*8)]);
            cV[7] = _mm256_loadu_ps(&C[(ii+3)*NJ + (jj+1*8)]);

            cV[0] = _mm256_add_ps(cV[0], sumV[0]);
            cV[1] = _mm256_add_ps(cV[1], sumV[1]);
            cV[2] = _mm256_add_ps(cV[2], sumV[2]);
            cV[3] = _mm256_add_ps(cV[3], sumV[3]);
            cV[4] = _mm256_add_ps(cV[4], sumV[4]);
            cV[5] = _mm256_add_ps(cV[5], sumV[5]);
            cV[6] = _mm256_add_ps(cV[6], sumV[6]);
            cV[7] = _mm256_add_ps(cV[7], sumV[7]);

            _mm256_storeu_ps(&C[(ii+0)*NJ + (jj+0*8)], cV[0]);
            _mm256_storeu_ps(&C[(ii+0)*NJ + (jj+1*8)], cV[1]);
            _mm256_storeu_ps(&C[(ii+1)*NJ + (jj+0*8)], cV[2]);
            _mm256_storeu_ps(&C[(ii+1)*NJ + (jj+1*8)], cV[3]);
            _mm256_storeu_ps(&C[(ii+2)*NJ + (jj+0*8)], cV[4]);
            _mm256_storeu_ps(&C[(ii+2)*NJ + (jj+1*8)], cV[5]);
            _mm256_storeu_ps(&C[(ii+3)*NJ + (jj+0*8)], cV[6]);
            _mm256_storeu_ps(&C[(ii+3)*NJ + (jj+1*8)], cV[7]);
            
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