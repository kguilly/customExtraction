#ifndef SHARED_TEST_OBJS_H
#define SHARED_TEST_OBJS_H

typedef struct Station{
    float lat;
    float lon;
    double **values;
    int closestPoint;
} station_t;

#endif