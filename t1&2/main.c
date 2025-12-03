#include <stdio.h>
#define N 30

int main() {
    int array[N] = {7, 3, 7, 9, 6, 2, 1, 3, 2, 5, 4, 1, 6, 9, 9, 1, 0, 3, 5, 6, 2, 9, 8, 2, 5, 5, 1, 2, 6, 5};
    int num;
    int repetitions = 0;

   printf("Enter number between 1-10: ");
   scanf("%d", &num);

   for(int i = 0; i < N; i++) {
        if (array[i] == num) {
            repetitions++;
        }
   }

   printf("The number of times %d is repited is: %d", num, repetitions);

   return 0;
}