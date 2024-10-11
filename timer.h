#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>
#include <stdio.h>

// Color codes for console output
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"

typedef struct {
    struct timeval start;
    struct timeval end;
} Timer;

void startTime(Timer* timer) {
    gettimeofday(&(timer->start), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->end), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.end.tv_sec - timer.start.tv_sec) * 1000.0 + 
                    (timer.end.tv_usec - timer.start.tv_usec) / 1000.0));
}

void printElapsedTime(Timer timer, const char* message, const char* color) {
    printf("%s%s: %f ms%s\n", color, message, elapsedTime(timer), RESET);
}

#endif // TIMER_H