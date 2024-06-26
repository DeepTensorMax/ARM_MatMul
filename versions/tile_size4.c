#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>
#include <pthread.h>
#include <time.h>
#include <string.h>
#include "armpmu_lib_pmu.h"

#define N 1024         // Matrix dimensions (NxN)
#define TILE_SIZE 4    // Size of each sub-matrix for parallel processing
#define NUM_THREADS 4   // Number of worker threads
#define UNROLL_FACTOR 2 // Loop unrolling factor

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
                    //_builtin_prefetch(&A[i][k + 64], 0, 3);  
                    //__builtin_prefetch(&B[k + 64][j], 0, 3); 
                    //__builtin_prefetch(&A[i + 1][k + 64], 0, 3);

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
    enable_pmu(ARM_PMU_L1D_CACHE);
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
    disable_pmu(ARM_PMU_L1D_CACHE);

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