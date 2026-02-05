#include <stdio.h>
#include <stdlib.h>
#include "my_timer.h"

#include <x86intrin.h>
#include <omp.h>
#include <math.h> 

#define NI 4096
#define NJ 4096
#define NK 4096

#define TILE_SIZE 64
#define Unroll 8
#define NUM_THREADS 20

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
omp_set_num_threads(NUM_THREADS);
  #pragma omp parallel for
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
  for (ii =  0 ; ii < NI; ii += TILE_SIZE){
    for (kk = 0; kk < NK; kk += TILE_SIZE){
      for(jj = 0; jj < NJ; jj += TILE_SIZE){
      
        // Inside tiles
        for (i = ii; i < ii + TILE_SIZE; i+= Unroll){
          for (k = kk; k < kk + TILE_SIZE; k++){

            // Unrolling a * A
            float A0 = alpha * A[(i + 0) * NK + k];
            float A1 = alpha * A[(i + 1) * NK + k];
            float A2 = alpha * A[(i + 2) * NK + k];
            float A3 = alpha * A[(i + 3) * NK + k];
            float A4 = alpha * A[(i + 4) * NK + k]; 
            float A5 = alpha * A[(i + 5) * NK + k];
            float A6 = alpha * A[(i + 6) * NK + k];
            float A7 = alpha * A[(i + 7) * NK + k];

            for(j = jj; j < jj + TILE_SIZE; j++){
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
__attribute__((target("avx2")))
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
  for (ii =  0 ; ii < NI; ii += TILE_SIZE){
    for (kk = 0; kk < NK; kk += TILE_SIZE){
      for(jj = 0; jj < NJ; jj += TILE_SIZE){
      
        // Inside tiles
        for (i = ii; i < ii + TILE_SIZE; i+= Unroll){
          for (k = kk; k < kk + TILE_SIZE; k++){

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
            for(j = jj; j < jj + TILE_SIZE; j++){
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

/* Main computational kernel: with tiling, simd, parallelization, and intrinsic optimizations. */
static
__attribute__((target("avx2")))
void kernel_gemm_tvp(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k, ii, jj, kk;
  omp_set_num_threads(NUM_THREADS);

  #pragma omp parallel default(none) shared(A,B,C,alpha,beta) private(i,j,k,ii,jj,kk)
  {
    // C *= beta
    #pragma omp for schedule(static)
    for(i = 0; i < NI; i++){
      float *row_C = &C[i*NJ];
      #pragma omp simd
      for(j = 0; j < NJ; j++) row_C[j] *= beta;
    }

    // Tiling and parallelization
    #pragma omp for schedule(static)
    for (ii = 0; ii < NI; ii += TILE_SIZE) {
      for (kk = 0; kk < NK; kk += TILE_SIZE) {
        for (jj = 0; jj < NJ; jj += TILE_SIZE) {

          // Inside tiles
          for (i = ii; i < ii + TILE_SIZE; i += 4) { 

            // Load C
            float *C0 = &C[(i + 0) * NJ];
            float *C1 = &C[(i + 1) * NJ];
            float *C2 = &C[(i + 2) * NJ];
            float *C3 = &C[(i + 3) * NJ];
            
            for (k = kk; k < kk + TILE_SIZE; k++) {

              // Load alpha * A
              __m256 va0 = _mm256_set1_ps(alpha * A[(i + 0) * NK + k]);
              __m256 va1 = _mm256_set1_ps(alpha * A[(i + 1) * NK + k]);
              __m256 va2 = _mm256_set1_ps(alpha * A[(i + 2) * NK + k]);
              __m256 va3 = _mm256_set1_ps(alpha * A[(i + 3) * NK + k]);

              // Unrolling
              for (j = jj; j < jj + TILE_SIZE; j += 8) {
                  // Load row of B
                  __m256 vb = _mm256_loadu_ps(&B[k * NJ + j]);

                  // Load current C values
                  __m256 vc0 = _mm256_loadu_ps(&C[(i + 0) * NJ + j]);
                  __m256 vc1 = _mm256_loadu_ps(&C[(i + 1) * NJ + j]);
                  __m256 vc2 = _mm256_loadu_ps(&C[(i + 2) * NJ + j]);
                  __m256 vc3 = _mm256_loadu_ps(&C[(i + 3) * NJ + j]);

                  // C += A * B
                  vc0 = _mm256_add_ps(vc0, _mm256_mul_ps(va0, vb));
                  vc1 = _mm256_add_ps(vc1, _mm256_mul_ps(va1, vb));
                  vc2 = _mm256_add_ps(vc2, _mm256_mul_ps(va2, vb));
                  vc3 = _mm256_add_ps(vc3, _mm256_mul_ps(va3, vb));

                  // Store results back
                  _mm256_storeu_ps(&C[(i + 0) * NJ + j], vc0);
                  _mm256_storeu_ps(&C[(i + 1) * NJ + j], vc1);
                  _mm256_storeu_ps(&C[(i + 2) * NJ + j], vc2);
                  _mm256_storeu_ps(&C[(i + 3) * NJ + j], vc3);
              }
            }
          }
        }
      }
    }
  }
}


static
__attribute__((target("avx2")))
void kernel_gemm_tvp_bitshifting(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k, ii, jj, kk;
  omp_set_num_threads(NUM_THREADS);
  
  const int bitshift_NI = log2(NI);
  const int bitshift_NJ = log2(NI);
  const int bitshift_NK = log2(NI);

  const int TILESIZE_I = 256;
  const int TILESIZE_J = 512;
  const int TILESIZE_K = 64;

  __m256 betaV = _mm256_set1_ps(beta);

  #pragma omp parallel default(none) shared(A,B,C,alpha,beta, betaV) private(i,j,k,ii,jj,kk)
  {
    // C *= beta
    #pragma omp for schedule(static)
    for (int ii = 0; ii < NI; ii += TILESIZE_I) {
        int IImax = (ii + TILESIZE_I < NI ? ii + TILESIZE_I : NI);
        for (int jj = 0; jj < NJ; jj += TILESIZE_J) {
            int JJmax = (jj + TILESIZE_J < NJ ? jj + TILESIZE_J : NJ);

            for (int i = ii; i < IImax; i++) {
                float *row_C = &C[i << bitshift_NJ];

                #pragma omp simd
                for (int j = jj; j < JJmax; j+= 8) {
                    // row_C[j] *= beta;
                    __m256 cV = _mm256_loadu_ps(&row_C[j]); 
                    cV = _mm256_mul_ps(cV, betaV); 
                    _mm256_storeu_ps(&row_C[j], cV);
                }
            }
        }
    }


    // Tiling and parallelization
    #pragma omp for schedule(static)
    for (ii = 0; ii < NI; ii += TILESIZE_I) {
      for (kk = 0; kk < NK; kk += TILESIZE_K) {
        for (jj = 0; jj < NJ; jj += TILESIZE_J) {

          // Inside tiles
          for (i = ii; i < ii + TILESIZE_I; i += 4) { 

            // Load C
            // float *C0 = &C[((i + 0) << bitshift_NJ)];
            // float *C1 = &C[((i + 1) << bitshift_NJ)];
            // float *C2 = &C[((i + 2) << bitshift_NJ)];
            // float *C3 = &C[((i + 3) << bitshift_NJ)];
            
            for (k = kk; k < kk + TILESIZE_K; k++) {

              // Load alpha * A
              __m256 va0 = _mm256_set1_ps(alpha * A[((i + 0) << bitshift_NK) + k]);
              __m256 va1 = _mm256_set1_ps(alpha * A[((i + 1) << bitshift_NK) + k]);
              __m256 va2 = _mm256_set1_ps(alpha * A[((i + 2) << bitshift_NK) + k]);
              __m256 va3 = _mm256_set1_ps(alpha * A[((i + 3) << bitshift_NK) + k]);

              // Unrolling
              for (j = jj; j < jj + TILESIZE_J; j += 8) {
                  // Load row of B
                  __m256 vb = _mm256_loadu_ps(&B[(k << bitshift_NJ) + j]);

                  // Load current C values
                  __m256 vc0 = _mm256_loadu_ps(&C[((i + 0) << bitshift_NJ) + j]);
                  __m256 vc1 = _mm256_loadu_ps(&C[((i + 1) << bitshift_NJ) + j]);
                  __m256 vc2 = _mm256_loadu_ps(&C[((i + 2) << bitshift_NJ) + j]);
                  __m256 vc3 = _mm256_loadu_ps(&C[((i + 3) << bitshift_NJ) + j]);

                  // C += A * B
                  vc0 = _mm256_add_ps(vc0, _mm256_mul_ps(va0, vb));
                  vc1 = _mm256_add_ps(vc1, _mm256_mul_ps(va1, vb));
                  vc2 = _mm256_add_ps(vc2, _mm256_mul_ps(va2, vb));
                  vc3 = _mm256_add_ps(vc3, _mm256_mul_ps(va3, vb));

                  // Store results back
                  _mm256_storeu_ps(&C[((i + 0) << bitshift_NJ) + j], vc0);
                  _mm256_storeu_ps(&C[((i + 1) << bitshift_NJ) + j], vc1);
                  _mm256_storeu_ps(&C[((i + 2) << bitshift_NJ) + j], vc2);
                  _mm256_storeu_ps(&C[((i + 3) << bitshift_NJ) + j], vc3);
              }
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
  kernel_gemm_tvp_bitshifting(C, A, B, 1.5, 2.5);

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
