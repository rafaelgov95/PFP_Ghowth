/*
   Copyright 2016 Rafael Viana 01/09/18.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

#include "../include/Kernel.h"
#include "../include/PFPArray.h"
#include "../include/PFPGrowth.cu.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

PFPGrowth::PFPGrowth(ArrayMap *_arrayMap, Elo *_eloMap, size_t arrayMapSize, size_t eloPosMapSize, int minimo_suporte):eloPos(_eloMap),arrayMap(_arrayMap){
    ArrayMap *device_ArrayMap;
    Elo *device_pointer_elo_vector;
    Elo *device_elo_inicial;
    int *device_elosize_inical;
    int *deviceEloVectorSize;
    int *device_minimo_suporte;
    int hostEloVectorSize = 0;

    Elo  *host_eloMap = (Elo *) malloc(eloPosMapSize * 1048576 * sizeof(Elo));
    cudaMalloc((void **) &device_pointer_elo_vector, sizeof(Elo) * 331238);

    gpuErrchk(cudaMalloc((void **) &device_elo_inicial, sizeof(Elo) * 90000));

    gpuErrchk(cudaMalloc((void **) &device_ArrayMap, sizeof(ArrayMap) * arrayMapSize));
    gpuErrchk(cudaMalloc((void **) &deviceEloVectorSize, sizeof(int)));

    gpuErrchk(cudaMalloc((void **) &device_elosize_inical, sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &device_minimo_suporte, sizeof(int)));

    gpuErrchk(cudaMemcpy(device_ArrayMap, arrayMap, sizeof(ArrayMap) * arrayMapSize, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(device_elo_inicial, _eloMap, sizeof(Elo) * eloPosMapSize, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(deviceEloVectorSize, &hostEloVectorSize, sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(device_elosize_inical, &eloPosMapSize, sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(device_minimo_suporte, &minimo_suporte, sizeof(int), cudaMemcpyHostToDevice));


    runKernel << < 1, 1 >> >
                      (device_pointer_elo_vector,
                              deviceEloVectorSize,
                              device_ArrayMap,
                              arrayMapSize,
                              device_elo_inicial,
                              device_elosize_inical,
                              device_minimo_suporte);

    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaMemcpy(&hostEloVectorSize, deviceEloVectorSize, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Valor Final do Elo_k1 %d\n",hostEloVectorSize);
    gpuErrchk(cudaMemcpy(host_eloMap, device_pointer_elo_vector,sizeof(Elo) * hostEloVectorSize, cudaMemcpyDeviceToHost));
//
        for (int i = 0; i <hostEloVectorSize; ++i) {
        	printf("%d = %s\n",i,host_eloMap[i].ItemId);
        }
//
//
//    for (int i = 0; i < eloPosMapSize; ++i) {
//        gpuErrchk(cudaMemcpy(host_elos[i], host_elos_vector_and_memory_pointer_elos[i].eloArray,
//                             sizeof(Elo) * eloPosMapSize * 10000,
//                             cudaMemcpyDeviceToHost)); //Tamanho ficou pequeno para o final
//
//    }
//    cudaEventSynchronize(stop);
//    float milliseconds = 0;
//    cudaEventElapsedTime(&milliseconds, start, stop);
//
//    SetMap *setMap = (SetMap *) malloc(sizeof(SetMap) * eloPosMapSize);
//    int intdex = 0;
//    memset(setMap, 0, sizeof(SetMap) * eloPosMapSize);
//
//    for (int k = 0; k < eloPosMapSize; ++k) {
//        strcmp(setMap[k].elo.ItemId, " ");
//    }
//    for (int k = 0; k < eloPosMapSize; ++k) {
//        int i = 0;
//        bool flag = true;
//        while (i < eloPosMapSize && flag) {
//            if (0 == strcmp(setMap[i].elo.ItemId, "")) {
//                setMap[intdex].elo = eloMap[k];
//                intdex++;
//                flag = false;
//            } else {
//                if (0 == strcmp(eloMap[k].ItemId, setMap[i].elo.ItemId)) {
//                    flag = false;
//                    setMap[i].elo.suporte += eloMap[k].suporte;
//                }
//            }
//            i++;
//        }
//    }
//    for (int l = 0; l < intdex; ++l) {
//        host_elos[0][l] = setMap[l].elo;
//    }
//    host_elos_vector_and_memory_pointer_elos[0].size = intdex;
//    free(setMap);
//
//    printf("Total de Gerações de Frequência %d\n", hostEloVectorSize);
//    int i = 0;
//    for (int k = 0; k < hostEloVectorSize; ++k) {
//        for (int j = 0; j < host_elos_vector_and_memory_pointer_elos[k].size; ++j) {
//
//            printf("Index %d | %s;%d;%d \n", i, host_elos[k][j].ItemId, host_elos[k][j].indexArrayMap,
//                   host_elos[k][j].suporte);
//            i++;
//        }
//    }
//    printf("Tempo em Millisegundos  %ld\n", milliseconds);
//
//
//    for (int i = 0; i < eloPosMapSize; i++) {
//        gpuErrchk(cudaFree(host_elos_vector_and_memory_pointer_elos[i].eloArray));
//        free(host_elos[i]);
//        free(data_host_elos_vector[i].eloArray);
//
//    }
    gpuErrchk(cudaFree(device_pointer_elo_vector));
    gpuErrchk(cudaFree(device_elo_inicial));
    gpuErrchk(cudaFree(device_ArrayMap));
    gpuErrchk(cudaFree(deviceEloVectorSize));
    gpuErrchk(cudaFree(device_elosize_inical));
    gpuErrchk(cudaFree(device_minimo_suporte));
    free(host_eloMap);
//    free(data_host_elos_vector);

}
