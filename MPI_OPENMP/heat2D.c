#include "mpi.h"
#include "omp.h"
#include "functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

struct Parms {
    float cx;
    float cy;
} parms = {0.1, 0.1};

int main(int argc, char** argv) {

    int rank, size, offset,ix,iy ,averageColumns, extra, nColumns, start, end, width, height, XblockSize, YblockSize, NXblocks, NYblocks;
    double startTime, endTime;
    MPI_File file;
    MPI_Status status;

    MPI_Datatype column, row;

    if ( argc != 3 ) {
        printf("Error! Wrong number of arguments\n");
        exit(1);
    }

    width = atoi(argv[1]);
    height = atoi(argv[2]);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    XblockSize = width / sqrt(size);
    YblockSize = height / sqrt(size);

    NXblocks = sqrt(size);
    NYblocks = sqrt(size);




    MPI_Type_vector(1, YblockSize, XblockSize * NXblocks, MPI_FLOAT, &column);
    MPI_Type_commit(&column);
    MPI_Type_vector(XblockSize, 1, YblockSize * NYblocks, MPI_FLOAT, &row);
    MPI_Type_commit(&row);

    int dim[2];
    dim[0] = dim[1] = 0;

    int ndims = 2;

    MPI_Dims_create(size, ndims, dim);

    MPI_Comm comm;


    int period[2];
    period[0] = period[1] = 0;

    int reorder = 0;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &comm);

    int coord[2];
    MPI_Cart_coords(comm, rank, ndims, coord);
    printf("Rank %d coordinates are %d %d\n", rank, coord[0], coord[1]);
    fflush(stdout);


    int direction = 1, displacement = 1;
    int neighborRanks[4];
    MPI_Cart_shift(comm, direction, displacement, &neighborRanks[LEFT], &neighborRanks[RIGHT]);

    direction = 0;
    MPI_Cart_shift(comm, direction, displacement, &neighborRanks[UP], &neighborRanks[DOWN]);


    printf("Rank %d neighbours are right:%d left:%d up:%d down:%d\n", rank, neighborRanks[RIGHT], neighborRanks[LEFT],
           neighborRanks[UP], neighborRanks[DOWN]);
    fflush(stdout);
    int i,j;

    //float u[2][width][height];

    float (*u)[XblockSize][YblockSize] = calloc(2, sizeof(*u));


    for ( i = 0; i < YblockSize; i++ ) {
        for ( j = 0; j < XblockSize; j++ ) {
            u[0][j][i] = 0.0f;
            u[1][j][i] = 0.0f;
        }
    }



    int dimensions[2] = {width, height};
    int sub_array_dimensions[2] = {XblockSize, YblockSize};

    inidat(coord, dimensions, sub_array_dimensions, (float*) u);


    //parallelWriteToFile("initial.dat",coord, rank, XblockSize, YblockSize, NXblocks, NYblocks, (float*) u);
    Buffer buffers[4];

    if ( neighborRanks[UP] > -1 ) {
        buffers[UP].buf = malloc(XblockSize * sizeof(float));
    }

    if ( neighborRanks[DOWN] > -1 ) {
        buffers[DOWN].buf = malloc(XblockSize * sizeof(float));
    }

    if ( neighborRanks[RIGHT] > -1 ) {
        buffers[RIGHT].buf = malloc(YblockSize * sizeof(float));
    }

    if ( neighborRanks[LEFT] > -1 ) {
        buffers[LEFT].buf = malloc(YblockSize * sizeof(float));
    }


    int currentArray = 0, Yoffset, Xoffset;


    MPI_Request leftRequest[2], rightRequest[2], upRequest[2], downRequest[2];
    MPI_Status leftStatus[2], rightStatus[2], upStatus[2], downStatus[2];

    MPI_Barrier(comm);

    startTime = MPI_Wtime();

#pragma parallel num_threads(4) shared(u, parms, neighborRanks,buffers) private(i,ix, iy, currentArray)

    for ( i = 0; i < STEPS; i++ ) {
        int changed = 0;

        if ( neighborRanks[UP] > -1 ) {
            MPI_Isend(&u[currentArray][0][0], 1, row, neighborRanks[UP], MSGTAG,
                      MPI_COMM_WORLD, &upRequest[SEND]);


            MPI_Irecv(buffers[UP].buf, XblockSize, MPI_FLOAT, neighborRanks[UP], MSGTAG, MPI_COMM_WORLD,
                      &upRequest[RECEIVE]);

        }

        if ( neighborRanks[DOWN] > -1 ) {
            MPI_Isend(&u[currentArray][0][YblockSize - 1], 1, row,
                      neighborRanks[DOWN], MSGTAG,
                      MPI_COMM_WORLD, &downRequest[SEND]);

            MPI_Irecv(buffers[DOWN].buf, XblockSize, MPI_FLOAT, neighborRanks[DOWN], MSGTAG, MPI_COMM_WORLD,
                      &downRequest[RECEIVE]);
        }

        if ( neighborRanks[RIGHT] > -1 ) {
            MPI_Isend(&u[currentArray][XblockSize - 1][0], 1, column,
                      neighborRanks[RIGHT], MSGTAG, comm, &rightRequest[SEND]);

            MPI_Irecv(buffers[RIGHT].buf, YblockSize, MPI_FLOAT, neighborRanks[RIGHT], MSGTAG, comm,
                      &rightRequest[RECEIVE]);
        }

        if ( neighborRanks[LEFT] > -1 ) {
            MPI_Isend(&u[currentArray][0][0], 1, column, neighborRanks[LEFT],
                      MSGTAG, comm, &leftRequest[SEND]);

            MPI_Irecv(buffers[LEFT].buf, YblockSize, MPI_FLOAT, neighborRanks[LEFT], MSGTAG, comm,
                      &leftRequest[RECEIVE]);
        }

#pragma omp for schedule(static, 4) collapse(2)
        for ( ix =  1; ix < XblockSize - 1; ix++ )
            for ( iy =  1; iy < YblockSize - 1; iy++ )  {
                u[1 - currentArray][ix][iy] = u[currentArray][ix][iy] +
                                              parms.cx * (u[currentArray][ix + 1][iy] +
                                                          u[currentArray][ix - 1][iy] -
                                                          2.0 * u[currentArray][ix][iy]) +
                                              parms.cy * (u[currentArray][ix][iy + 1] +
                                                          u[currentArray][ix][iy - 1] -
                                                          2.0 * u[currentArray][ix][iy]);
                if (u[1 - currentArray][ix][iy] == u[currentArray][ix][iy])
                    changed++;
            }
        if ( neighborRanks[UP] > -1 )
            MPI_Wait(&upRequest[RECEIVE], &upStatus[RECEIVE]);

        if ( neighborRanks[DOWN] > -1 )
            MPI_Wait(&downRequest[RECEIVE], &downStatus[RECEIVE]);

        if ( neighborRanks[RIGHT] > -1 )
            MPI_Wait(&rightRequest[RECEIVE], &rightStatus[RECEIVE]);

        if ( neighborRanks[LEFT] > -1 )
            MPI_Wait(&leftRequest[RECEIVE], &leftStatus[RECEIVE]);

    /*Up row*/
        if ( neighborRanks[UP] > -1 ) {

            iy = 0;
#pragma omp for schedule(static, 4)
            for ( ix =  1; ix < XblockSize - 1; ix++ ) {
                u[1 - currentArray][ix][iy] = u[currentArray][ix][iy] +
                                              parms.cx * (u[currentArray][ix + 1][iy] +
                                                          u[currentArray][ix - 1][iy] -
                                                          2.0 * u[currentArray][ix][iy]) +
                                              parms.cy * (u[currentArray][ix][iy + 1] +
                                                          buffers[UP].buf[ix ] -
                                                          2.0 * u[currentArray][ix][iy]);                                   
            }
            /*Up left corner*/
            if ( neighborRanks[LEFT] > -1 ) {
                ix = 0;
                u[1 - currentArray][ix][iy] = u[currentArray][ix][iy] +
                                              parms.cx * (u[currentArray][ix + 1][iy] +
                                                          buffers[UP].buf[0] -
                                                          2.0 * u[currentArray][ix][iy]) +
                                              parms.cy * (u[currentArray][ix][iy + 1] +
                                                          buffers[LEFT].buf[0] -
                                                          2.0 * u[currentArray][ix][iy]);                                       
            }
            /*Up right corner*/
            if ( neighborRanks[RIGHT] > -1 ) {
                ix =  XblockSize - 1;
                u[1 - currentArray][ix][iy] = u[currentArray][ix][iy] +
                                              parms.cx * (buffers[RIGHT].buf[0] +
                                                          u[currentArray][ix - 1][iy] -
                                                          2.0 * u[currentArray][ix][iy]) +
                                              parms.cy * (u[currentArray][ix][iy + 1] +
                                                          buffers[UP].buf[XblockSize - 1] -
                                                          2.0 * u[currentArray][ix][iy]);                                       
            }
        }

        /*Down row*/
        if ( neighborRanks[DOWN] > -1 ) {
            iy = YblockSize - 1;
#pragma omp for schedule(static, 4)
            for ( ix =  1; ix <  XblockSize - 1; ix++ ) {
                u[1 - currentArray][ix][iy] = u[currentArray][ix][iy] +
                                              parms.cx * (u[currentArray][ix + 1][iy] +
                                                          u[currentArray][ix - 1][iy] -
                                                          2.0 * u[currentArray][ix][iy]) +
                                              parms.cy * (buffers[DOWN].buf[ix ] +
                                                          u[currentArray][ix][iy - 1] -
                                                          2.0 * u[currentArray][ix][iy]);                                     
            }
            /*Down left corner*/
            if ( neighborRanks[LEFT] > -1 ) {
                ix = 0;
                u[1 - currentArray][ix][iy] = u[currentArray][ix][iy] +
                                              parms.cx * (u[currentArray][ix + 1][iy] +
                                                          buffers[LEFT].buf[YblockSize - 1] -
                                                          2.0 * u[currentArray][ix][iy]) +
                                              parms.cy * (buffers[DOWN].buf[0] +
                                                          u[currentArray][ix][iy - 1] -
                                                          2.0 * u[currentArray][ix][iy]);                                       
            }
            /*Down right corner*/
            if ( neighborRanks[RIGHT] > -1 ) {
                ix = XblockSize - 1;
                u[1 - currentArray][ix][iy] = u[currentArray][ix][iy] +
                                              parms.cx * (buffers[RIGHT].buf[YblockSize - 1] +
                                                          u[currentArray][ix - 1][iy] -
                                                          2.0 * u[currentArray][ix][iy]) +
                                              parms.cy * (buffers[DOWN].buf[XblockSize - 1] +
                                                          u[currentArray][ix][iy - 1] -
                                                          2.0 * u[currentArray][ix][iy]);                                       
            }
        }

        /*Left column*/
        if ( neighborRanks[LEFT] > -1 ) {
            ix = 0;
#pragma omp for schedule(static, 4)
            for ( iy =  1; iy <  YblockSize - 1; iy++ ) {
                u[1 - currentArray][ix][iy] = u[currentArray][ix][iy] +
                                              parms.cx * (u[currentArray][ix + 1][iy] +
                                                          buffers[LEFT].buf[iy] -
                                                          2.0 * u[currentArray][ix][iy]) +
                                              parms.cy * (u[currentArray][ix][iy + 1] +
                                                          u[currentArray][ix][iy - 1] -
                                                          2.0 * u[currentArray][ix][iy]);                              
            }
        }

        /*Right column*/
        if ( neighborRanks[RIGHT] > -1 ) {
            ix =  XblockSize - 1;
#pragma omp for schedule(static, 4)
            for ( iy =  1; iy <  YblockSize - 1; iy++ ) {
                u[1 - currentArray][ix][iy] = u[currentArray][ix][iy] +
                                              parms.cx * (buffers[RIGHT].buf[iy] +
                                                          u[currentArray][ix - 1][iy] -
                                                          2.0 * u[currentArray][ix][iy]) +
                                              parms.cy * (u[currentArray][ix][iy + 1] +
                                                          u[currentArray][ix][iy - 1] -
                                                          2.0 * u[currentArray][ix][iy]);                                       
            }
        }

        if ( neighborRanks[UP] > -1 )
            MPI_Wait(&upRequest[SEND], &upStatus[SEND]);

        if ( neighborRanks[DOWN] > -1 )
            MPI_Wait(&downRequest[SEND], &downStatus[SEND]);

        if ( neighborRanks[RIGHT] > -1 )
            MPI_Wait(&rightRequest[SEND], &rightStatus[SEND]);

        if ( neighborRanks[LEFT] > -1 )
            MPI_Wait(&leftRequest[SEND], &leftStatus[SEND]);


        currentArray = 1 - currentArray;
    }

    endTime = MPI_Wtime();



    //parallelWriteToFile("final.dat", coord, rank, XblockSize, YblockSize, NXblocks, NYblocks, (float*) u);

    if (rank == 0 ){
        printf(" DURATION: %lf\n",endTime - startTime);
    }
    free(u);

    MPI_Finalize();

}