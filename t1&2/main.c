#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 480

int main(int argc, char *argv[]) {
    int rank, size;
    int array[N] = {7, 3, 7, 9, 6, 2, 1, 3, 2, 5, 4, 1, 6, 9, 9, 1, 0, 3,
        5, 6, 2, 9, 8, 2, 5, 5, 1, 2, 6, 5, 7, 3, 7, 9, 6, 2, 1, 3, 2, 5, 4, 1, 6, 9, 9, 1, 0, 3,
        5, 6, 2, 9, 8, 2, 5, 5, 1, 2, 6, 5, 7, 3, 7, 9, 6, 2, 1, 3, 2, 5, 4, 1, 6, 9, 9, 1, 0, 3,
        5, 6, 2, 9, 8, 2, 5, 5, 1, 2, 6, 5, 7, 3, 7, 9, 6, 2, 1, 3, 2, 5, 4, 1, 6, 9, 9, 1, 0, 3,
        5, 6, 2, 9, 8, 2, 5, 5, 1, 2, 6, 5, 7, 3, 7, 9, 6, 2, 1, 3, 2, 5, 4, 1, 6, 9, 9, 1, 0, 3,
        5, 6, 2, 9, 8, 2, 5, 5, 1, 2, 6, 5, 7, 3, 7, 9, 6, 2, 1, 3, 2, 5, 4, 1, 6, 9, 9, 1, 0, 3,
        5, 6, 2, 9, 8, 2, 5, 5, 1, 2, 6, 5, 7, 3, 7, 9, 6, 2, 1, 3, 2, 5, 4, 1, 6, 9, 9, 1, 0, 3,
        5, 6, 2, 9, 8, 2, 5, 5, 1, 2, 6, 5, 7, 3, 7, 9, 6, 2, 1, 3, 2, 5, 4, 1, 6, 9, 9, 1, 0, 3,
        5, 6, 2, 9, 8, 2, 5, 5, 1, 2, 6, 5, 7, 3, 7, 9, 6, 2, 1, 3, 2, 5, 4, 1, 6, 9, 9, 1, 0, 3,
        5, 6, 2, 9, 8, 2, 5, 5, 1, 2, 6, 5, 7, 3, 7, 9, 6, 2, 1, 3, 2, 5, 4, 1, 6, 9, 9, 1, 0, 3,
        5, 6, 2, 9, 8, 2, 5, 5, 1, 2, 6, 5, 7, 3, 7, 9, 6, 2, 1, 3, 2, 5, 4, 1, 6, 9, 9, 1, 0, 3,
        5, 6, 2, 9, 8, 2, 5, 5, 1, 2, 6, 5, 7, 3, 7, 9, 6, 2, 1, 3, 2, 5, 4, 1, 6, 9, 9, 1, 0, 3,
        5, 6, 2, 9, 8, 2, 5, 5, 1, 2, 6, 5, 7, 3, 7, 9, 6, 2, 1, 3, 2, 5, 4, 1, 6, 9, 9, 1, 0, 3,
        5, 6, 2, 9, 8, 2, 5, 5, 1, 2, 6, 5, 7, 3, 7, 9, 6, 2, 1, 3, 2, 5, 4, 1, 6, 9, 9, 1, 0, 3,
        5, 6, 2, 9, 8, 2, 5, 5, 1, 2, 6, 5};
    int num;
    int repetitions = 0;
    int local_repetitions = 0;
    int chunk_size;
    int *recv_counts = NULL;
    int *displacements = NULL;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    chunk_size = N / size;

    // root process    
    if (rank == 0) {
        printf("Enter a number between 1-10: ");
        scanf("%d", &num);
    }
    
    MPI_Bcast(&num, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    int *local_array = (int *)malloc(chunk_size * sizeof(int));

    MPI_Scatter(array, chunk_size, MPI_INT, local_array, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    
    for (int i = 0; i < chunk_size; i++) {
        if (local_array[i] == num) {
            local_repetitions++;
        }
    }

    MPI_Reduce(&local_repetitions, &repetitions, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("The number %d was repeated %d times in the array.\n", num, repetitions);
    }
    
    free(local_array);
    MPI_Finalize();
    
    return 0;
}