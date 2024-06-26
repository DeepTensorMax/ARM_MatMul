#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>
#include <pthread.h>
#include <string.h>
#include "armpmu_lib_pmu.h"

#define N 1024         // Matrix dimensions (NxN)
#define TILE_SIZE 32    // Size of each sub-matrix for parallel processing
#define NUM_THREADS 4   // Number of worker threads
#define UNROLL_FACTOR 128 // Loop unrolling factor

// Align matrices to 16-byte boundaries for NEON optimization
int A[N][N] __attribute__((aligned(16)));
int B[N][N] __attribute__((aligned(16)));
int C[N][N] __attribute__((aligned(16)));

// Structure to represent a task (sub-matrix to multiply)
typedef struct {
    int startRow, startCol;
    int endRow, endCol;
} Task;

Task tasks[(N / TILE_SIZE) * (N / TILE_SIZE)];  // Array of tasks for threads
int taskIndex = 0;                              // Global task index (shared by threads)
pthread_mutex_t taskMutex = PTHREAD_MUTEX_INITIALIZER; // Mutex for taskIndex synchronization

// Thread function: multiplies assigned tiles using NEON SIMD instructions
void *multiplyTilesNEON(void *arg) {
    Task task;             
    const int16_t *pA, *pB; // Pointers for efficient matrix access
    int32x4_t acc0, acc1, acc2, acc3;  // Accumulators for 4x4 block multiplication
    int32x2_t result0, result1, result2, result3; // Partial results for two rows

    while (1) {
        // Atomically acquire a task from the task queue
        pthread_mutex_lock(&taskMutex);
        if (taskIndex < (N / TILE_SIZE) * (N / TILE_SIZE)) {
            int i = taskIndex / (N / TILE_SIZE);  // Calculate row index of the task
            int j = taskIndex % (N / TILE_SIZE);  // Calculate column index of the task
            
            // Define the boundaries of the task (sub-matrix)
            task = (Task){i * TILE_SIZE, j * TILE_SIZE, (i + 1) * TILE_SIZE, (j + 1) * TILE_SIZE};
            if (task.endRow > N) task.endRow = N;  // Ensure task doesn't exceed matrix dimensions
            if (task.endCol > N) task.endCol = N;
            taskIndex++;  
        } else {
            pthread_mutex_unlock(&taskMutex);
            break; // No more tasks left
        }
        pthread_mutex_unlock(&taskMutex);

        // Iterate over rows of the sub-matrix (2 rows at a time)
        for (int i = task.startRow; i < task.endRow; i += 2) {  
            // Iterate over columns of the sub-matrix (4 columns at a time)
            for (int j = task.startCol; j < task.endCol; j += 4) {
                pA = (const int16_t*)&A[i][0];    // Reset pointers to start of rows
                pB = (const int16_t*)&B[0][j];

                // Initialize accumulators to zero
                acc0 = acc1 = acc2 = acc3 = vdupq_n_s32(0);  

                // Iterate over k (inner dimension for matrix multiplication)
                for (int k = 0; k < N; k += 8 * UNROLL_FACTOR) { 
                    // Prefetch data into L1 cache for upcoming iterations (software prefetching)
                    __builtin_prefetch(&A[i][k + 64], 0, 3);  
                    __builtin_prefetch(&B[k + 64][j], 0, 3); 
                    __builtin_prefetch(&A[i + 1][k + 64], 0, 3);

                    // Load 8 elements (16-bit) from matrices A and B into NEON registers
                    int16x8_t a0 = vld1q_s16(pA); pA += 8; 
                    int16x8_t b0 = vld1q_s16(pB); pB += 8;
                    int16x8_t a1 = vld1q_s16(pA); pA += 8;
                    int16x8_t b1 = vld1q_s16(pB); pB += 8;
                    int16x8_t a2 = vld1q_s16(pA); pA += 8;
                    int16x8_t b2 = vld1q_s16(pB); pB += 8;
                    int16x8_t a3 = vld1q_s16(pA); pA += 8;
                    int16x8_t b3 = vld1q_s16(pB); pB += 8;

                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                    // UNROLLED LOOP 2
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 3
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 4
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                    // UNROLLED LOOP 5
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 6
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 7
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 8
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 9
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                    // UNROLLED LOOP 10
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 11
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 12
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 13
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 14
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                    // UNROLLED LOOP 15
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 16
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                       // UNROLLED LOOP 17
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 18
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                    // UNROLLED LOOP 19
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 20
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 21
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 22
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 23
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                    // UNROLLED LOOP 24
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 25
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));
                                       // UNROLLED LOOP 26
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 27
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                    // UNROLLED LOOP 28
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 29
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 30
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 31
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 32
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));
                                        // UNROLLED LOOP 33
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                       // UNROLLED LOOP 34
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 35
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                    // UNROLLED LOOP 36
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 37
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 38
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 39
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 40
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                    // UNROLLED LOOP 41
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 42
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));
                                       // UNROLLED LOOP 43
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 44
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                    // UNROLLED LOOP 45
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 46
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 47
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 48
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 49
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                                            // UNROLLED LOOP 50
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                       // UNROLLED LOOP 51
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 52
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                    // UNROLLED LOOP 53
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 54
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 55
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 56
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 57
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                    // UNROLLED LOOP 58
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 59
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));
                                       // UNROLLED LOOP 60
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 61
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                    // UNROLLED LOOP 62
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 63
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 64
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                                            // UNROLLED LOOP 65
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));
                                        // UNROLLED LOOP 66
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                       // UNROLLED LOOP 67
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 68
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                    // UNROLLED LOOP 69
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 70
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 71
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 72
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 73
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                    // UNROLLED LOOP 74
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 75
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));
                                       // UNROLLED LOOP 76
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 77
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                    // UNROLLED LOOP 78
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 79
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 80
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 81
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 82
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                                            // UNROLLED LOOP 83
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                       // UNROLLED LOOP 84
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 85
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                    // UNROLLED LOOP 86
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 87
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 88
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 89
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 90
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                    // UNROLLED LOOP 91
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 92
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));
                                       // UNROLLED LOOP 93
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 94
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                    // UNROLLED LOOP 95
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 96
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 97
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                                            // UNROLLED LOOP 98
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));
                                        // UNROLLED LOOP 99
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                       // UNROLLED LOOP 100
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 101
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                    // UNROLLED LOOP 102
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 103
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 104
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 105
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 106
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                    // UNROLLED LOOP 107
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 108
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));
                                       // UNROLLED LOOP 109
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 110
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                    // UNROLLED LOOP 111
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 112
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 113
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 114
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 115
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                                            // UNROLLED LOOP 116
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                       // UNROLLED LOOP 117
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 118
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                    // UNROLLED LOOP 119
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 120
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 121
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 122
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 123
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                    // UNROLLED LOOP 124
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 125
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));
                                       // UNROLLED LOOP 126
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                                        // UNROLLED LOOP 127
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                    // UNROLLED LOOP 128
                    pA = (const int16_t*)&A[i][k + 16]; // Update pA for the next row
                    pB = (const int16_t*)&B[k + 16][j]; // Update pB for the next row
                    
                    // First 16 vmlal_s16 operations (for a0, a1, b0, b1)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b0));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b0));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b0));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b0));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_low_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_low_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_low_s16(b1));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_high_s16(b1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a0), vget_high_s16(b1));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a1), vget_high_s16(b1));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a1), vget_high_s16(b1));


                    // Second 16 vmlal_s16 operations (for a2, a3, b2, b3)
                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b2));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b2));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b2));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b2));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_low_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_low_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_low_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_low_s16(b3));

                    acc0 = vmlal_s16(acc0, vget_low_s16(a2), vget_high_s16(b3));
                    acc1 = vmlal_s16(acc1, vget_high_s16(a2), vget_high_s16(b3));
                    acc2 = vmlal_s16(acc2, vget_low_s16(a3), vget_high_s16(b3));
                    acc3 = vmlal_s16(acc3, vget_high_s16(a3), vget_high_s16(b3));

                    
                    
                }
                    result0 = vpadd_s32(vget_low_s32(acc0), vget_high_s32(acc0));
                    result1 = vpadd_s32(vget_low_s32(acc1), vget_high_s32(acc1));
                    result2 = vpadd_s32(vget_low_s32(acc2), vget_high_s32(acc2));
                    result3 = vpadd_s32(vget_low_s32(acc3), vget_high_s32(acc3));

                    vst1q_s32(&C[i][j], vcombine_s32(result0, result1)); // Store results for first row
                    vst1q_s32(&C[i + 1][j], vcombine_s32(result2, result3)); // Store results for second row
            }
        }
    }
    pthread_exit(NULL);
}

void print_matrix(const char *name, int matrix[N][N])
{
    printf("%s:\n", name);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf(" %d ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void reset_matrix(int C[][N])
{
    memset(C, 0, N * N * sizeof(int));
}

int main() {
    // Initialize variables for cycle and time measurements
    uint64_t start_cycles, end_cycles;
    uint64_t elapsed_cycles;
    uint64_t count_start, count_end;
    struct timespec start_time, end_time;
    long elapsed_time;
    
    // Matrix initialization (fill A with 10, transpose B and fill with 20)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = 10;
            B[j][i] = 20; // Note: B is transposed for more efficient cache access during multiplication
        }
    }

    // Time measuring using PMU and clock_gettime
    enable_pmu(ARM_PMU_CPU_CYCLES);
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    start_cycles = read_pmu();
    
    // Create and launch worker threads for parallel matrix multiplication
    pthread_t threads[NUM_THREADS]; 
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, multiplyTilesNEON, NULL); 
    }

    // Wait for all worker threads to complete their tasks
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    end_cycles = read_pmu();
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    disable_pmu(ARM_PMU_CPU_CYCLES);

    //print_matrix("Matrix C = AxB", C);
    reset_matrix(C);        // Resetting matrix

    // Instruction measuring using PMU
    enable_pmu(ARM_PMU_INST_RETIRED);
    count_start = read_pmu();

    // Create and launch worker threads for parallel matrix multiplication
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, multiplyTilesNEON, NULL); 
    }

    // Wait for all worker threads to complete their tasks
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    count_end = read_pmu();
    disable_pmu(ARM_PMU_INST_RETIRED);

    elapsed_cycles = end_cycles - start_cycles;
    elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000000000L + (end_time.tv_nsec - start_time.tv_nsec);
    printf("multiply_opt; retired instructions = %u, elapsed CPU cycles = %u, elapsed time = %ld ns\n", count_end - count_start, elapsed_cycles, elapsed_time);

    return 0; // Exit successfully
}

//gcc -O3 -march=armv8-a+crc -mtune=cortex-a53 -pthread ex03matmul.c -o exe