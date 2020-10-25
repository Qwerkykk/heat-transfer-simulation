#define STEPS    100            /* number of time steps */
#define NONE     0              /* indicates no neighbor */
#define MSGTAG   999
#define SEND     0
#define RECEIVE  1
#define UP 0
#define DOWN 1
#define RIGHT 2
#define LEFT 3

typedef struct buffer {
    float* buf;
} Buffer;

void inidat(int* s, int* d, int* sd, float* u);
float* get_sub_array(int*, int, int, float*, int);


void parallelWriteToFile(char* fileName, int* coord, int rank, int XblockSize, int YblockSize, int NXblocks, int NYblocks,
                         float* heatMap);

void insideUpdate(int Xoffset, int XblockSize, int Yoffset, int YblockSize, int ny, float* u1, float* u2);

void outerUpdate(int* neighborRanks, int NXblocks, int NYblocks, int Xoffset, int XblockSize, int Yoffset,
                 int YblockSize, float* u1, float* u2, Buffer* buffers);