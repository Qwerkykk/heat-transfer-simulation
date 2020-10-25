#include "functions.h"
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>


void inidat(int* c, int* d, int* sd, float* u) {
    size_t i,j;
    for ( i = 0; i < sd[0]; i++ ){
        for ( j = 0; j < sd[1] ; j++ ) {
                    *(u + i * sd[1] + j) = (float) ((i + c[1]*sd[0]) * (d[0] - (i + c[1]*sd[0]) - 1) * (j + c[0]*sd[1]) * (d[1] - (j + c[0]*sd[1]) - 1));
        }
    }
}



void parallelWriteToFile(char* fileName,int* coord, int rank, int XblockSize, int YblockSize, int NXblocks, int NYblocks,
                         float* heatMap) {

    int Xoffset, Yoffset, fileOffset;
    char buf[25];

    MPI_File fhw;
    MPI_Status status;

    MPI_File_open(MPI_COMM_WORLD, fileName, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fhw);

    if ( rank < NXblocks ) {
        Yoffset = 0;
        Xoffset = rank;
    }
    else {
        Yoffset = rank / NXblocks;
        Xoffset = rank % NXblocks;
    }

    fileOffset = XblockSize * Xoffset * 17 + Yoffset * YblockSize * XblockSize * NXblocks * 17;
    int i,j;
    for ( i = 0; i < YblockSize; i++ ) {
        for ( j = 0; j < XblockSize; j++ ) {

            if ( j + coord[1] * XblockSize == XblockSize * NXblocks - 1 )
                sprintf(buf, "%6.1f          \n", *(heatMap + j * YblockSize  + i));
            else
                sprintf(buf, "%6.1f           ", *(heatMap + j * YblockSize + i));


            MPI_File_write_at(fhw, fileOffset, buf, 17, MPI_CHAR, &status);

            fileOffset += 17;
        }
        fileOffset += (NXblocks - 1) * XblockSize * 17;
    }

    MPI_File_close(&fhw);
}

