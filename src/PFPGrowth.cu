//
// Created by rafael on 01/09/18.
//
#include "cudaHeaders.h"
#include "Kernel.h"
#include "PFPArray.h"
#include "PFPGrowth.cu.h"
#include "PFPArray.h"
#include "PFPArray.h"
#include "../include/PFPArray.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

PFPGrowth::PFPGrowth(ArrayMap *arrayMap, Elo *eloMap, size_t arrayMapSize, size_t eloPosMapSize) {
    ArrayMap *device_ArrayMap;
    EloVector *device_pointer_elo_kx, *host_pointer_elo_kx, *data_host_pointer_elo_kx;

    int *deviceEloVectorSize;
    int hostEloVectorSize;

//    hostEloVectorSize=(int *)malloc(sizeof(int));
    data_host_pointer_elo_kx = (EloVector *)malloc(sizeof(EloVector)*eloPosMapSize);
    for (int j = 0; j < eloPosMapSize; ++j) {
        data_host_pointer_elo_kx[j].eloArray=(Elo *)malloc(sizeof(Elo)*eloPosMapSize);
    }
    data_host_pointer_elo_kx->eloArray=eloMap;
    data_host_pointer_elo_kx->size=eloPosMapSize;
    hostEloVectorSize=0;

    host_pointer_elo_kx = (EloVector*)malloc(eloPosMapSize * sizeof(EloVector));
    memcpy(host_pointer_elo_kx, data_host_pointer_elo_kx, eloPosMapSize * sizeof(EloVector));

    for (int i=0; i<eloPosMapSize; i++){
        cudaMalloc(&(host_pointer_elo_kx[i].eloArray), eloPosMapSize*sizeof(Elo));
        cudaMemcpy(host_pointer_elo_kx[i].eloArray, data_host_pointer_elo_kx[i].eloArray, eloPosMapSize*sizeof(Elo), cudaMemcpyHostToDevice);
    }

    cudaMalloc((void **)&device_pointer_elo_kx, sizeof(EloVector)*eloPosMapSize);
    cudaMemcpy(device_pointer_elo_kx,host_pointer_elo_kx,sizeof(EloVector)*eloPosMapSize,cudaMemcpyHostToDevice);

    gpuErrchk(cudaMalloc((void **) &device_ArrayMap, sizeof(ArrayMap) * arrayMapSize));

    gpuErrchk(cudaMalloc((void **) &deviceEloVectorSize, sizeof(int)));

    gpuErrchk(cudaMemcpy(device_ArrayMap, arrayMap, sizeof(ArrayMap) * arrayMapSize, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(deviceEloVectorSize,&hostEloVectorSize, sizeof(int), cudaMemcpyHostToDevice));

    pfp_growth << < 1,eloPosMapSize >>>
                  (device_pointer_elo_kx,
                    deviceEloVectorSize,
                    device_ArrayMap,
                    arrayMapSize);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    cudaMemcpy(host_pointer_elo_kx,device_pointer_elo_kx,sizeof(EloVector)*eloPosMapSize,cudaMemcpyDeviceToHost);
    cudaMemcpy(&hostEloVectorSize,deviceEloVectorSize,sizeof(int),cudaMemcpyDeviceToHost);

    printf("Valor de Size GrideMap %d",hostEloVectorSize);
    for (int k = 0; k < 1; ++k) {
        for (int j = 0; j <  1; ++j) {
            printf("VALOR %s",host_pointer_elo_kx[k].eloArray[j].ItemId);
        }
    }


    cudaFree(device_ArrayMap);


}
