/*
   Copyright 2016 Rafael Viana 20/08/18.

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

#include <cudaHeaders.h>
#include "Kernel.h"
#include "PFPTree.h"
#include "PFPArray.h"
#include <cuda_runtime_api.h>
#include <cstdio>
#include "cuda.h"
#include "../include/PFPArray.h"
#include "../include/cudaHeaders.h"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


__device__ volatile int sem = 0;

__device__ void acquire_semaphore(volatile int *lock) {
    while (atomicCAS((int *) lock, 0, 1) != 0);
}

__device__ void release_semaphore(volatile int *lock) {
    *lock = 0;
    __threadfence();
}


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


__device__ int compare(char *String_1, char *String_2) {
    char TempChar_1,
            TempChar_2;

    do {
        TempChar_1 = *String_1++;
        TempChar_2 = *String_2++;
    } while (TempChar_1 && TempChar_1 == TempChar_2);

    return TempChar_1 - TempChar_2;
}

__device__ bool my_strcmp(char *array1, char *array2) {
    int i = 0;
    while (array1[i] != '\0') {
        if (array1[i] != array2[i]) {
            return false;
        }
        i++;
    }
    return true;
}

__device__ char *my_strcpy(char *dest, const char *src) {
    int i = 0;
    do {
        dest[i] = src[i];
    } while (src[i++] != 0);
    return dest;
}

__device__ char *my_strcat(char *dest, const char *src) {
    int i = 0;
    while (dest[i] != 0) i++;
    my_strcpy(dest + i, src);
    return dest;
}

__device__ char *my_cpcat(const char *array1, const char *array2, char *src) {
    my_strcat(src, array1);
    my_strcat(src, array2);
    return src;
}

__device__ unsigned int count = 0;
__shared__ bool isLastBlockDone;
__device__ int index_elo_put;
__device__ unsigned int indexSetMap = 0;


__global__ void


frequencia_x2(__volatile__ EloVector *elo_k1, __volatile__ int elo_cur, Elo *set_elo, int *eloMapSizePointer,
              int minimo) {
    auto indexAtual = blockIdx.x * blockDim.x + threadIdx.x; //PC
    int eloMapSize = (*eloMapSizePointer);

    if (indexAtual < eloMapSize) {
        printf("Thread ");
        bool newFlag = true;
        int indexSetMap = 0;
        while (newFlag && indexSetMap < eloMapSize) {
            if ((0 == compare(elo_k1[elo_cur].eloArray[indexAtual].ItemId, set_elo[indexSetMap].ItemId)) &&
                (set_elo[indexSetMap].suporte >= minimo)) {
//                elo_new_put[0] = elo_k1[elo_k1_current].eloArray[indexAtual];
                printf("Thread %d Elo size %d AQUI %s %d\n",threadIdx.x,indexAtual, set_elo[indexAtual].ItemId,set_elo[indexAtual].suporte);
                newFlag = false;
            }
            indexSetMap++;
        }
    }
}

__global__ void frequencia_x1(Elo *set_elo, int *eloMapSizePointer, Elo *eloSetTemp, int *eloSetTempSize) {
    auto indexAtual = blockIdx.x * blockDim.x + threadIdx.x; //PC
    int eloMapSize = (*eloMapSizePointer);index_elo_put
    if (indexAtual < eloMapSize) {

        bool newFlag = true;

        int index = 0;

        while (newFlag && index < eloMapSize) {
            acquire_semaphore(&sem);
            if (0 == compare(eloSetTemp[index].ItemId, "")) {
//
                newFlag = false;
                (*eloSetTempSize) = (*eloSetTempSize) + 1;
                eloSetTemp[index] = set_elo[indexAtual];
//                printf("Sou Thread %d do Bloco %d Elo_k1 %s elox %s\n", indexAtual, blockIdx.x,
//                       elo_k1[elo_cur].eloArray[index].ItemId, set_elo[indexAtual].ItemId);

            } else if (0 == compare(eloSetTemp[index].ItemId,
                                    set_elo[indexAtual].ItemId)) {
                newFlag = false;
                eloSetTemp[index].suporte += set_elo[indexAtual].suporte;
//                printf("Sou Igual Thread %d do Bloco %d Elo_k1 %s elox %s\n", indexAtual, blockIdx.x,
//                       elo_k1[elo_cur].eloArray[index].ItemId, set_elo[indexAtual].ItemId);
            } else {
                index++;
            }
            release_semaphore(&sem);
        }

    }

}

__global__ void
pfp_growth(__volatile__ EloVector *elo_k1, __volatile__ int *elo_curr, ArrayMap *arrayMap, size_t arrayMapSize,
           Elo *elo_x, int *elo_int_x, int *minimo_suporte) {
    if (threadIdx.x == 0) {
        unsigned int value = atomicInc(&count, gridDim.x);
        isLastBlockDone = (value == (gridDim.x - 1));
    }

    auto indexAtual = blockIdx.x * blockDim.x + threadIdx.x;
    int elo_x_size = (*elo_int_x);
    if (indexAtual < elo_x_size) {
        int elo_cur = (*elo_curr);
        int xxx = 0;
        index_elo_put = 0;
        bool flag = true;
        Elo *Elo_k1 = (Elo *) malloc(sizeof(Elo) * elo_x_size);

        auto indexThreadArrayMap = elo_x[indexAtual].indexArrayMap;
        auto indexParentArrayMap = elo_x[indexAtual].indexArrayMap;
        while (flag) {
            indexParentArrayMap = arrayMap[indexParentArrayMap].indexP;
            if (arrayMap[indexThreadArrayMap].indexP != -1 &&
                arrayMap[indexParentArrayMap].indexP != -1) {
                my_cpcat(elo_x[indexAtual].ItemId,
                         arrayMap[indexParentArrayMap].ItemId, Elo_k1[xxx].ItemId);

                Elo_k1[xxx].indexArrayMap = indexParentArrayMap;
                Elo_k1[xxx].suporte = elo_x[indexAtual].suporte;
            } else {
                flag = false;
            }
            xxx++;
        }
        int temp;
        for (int i = 0; i < (xxx - 1); ++i) {
            temp = atomicAdd(&index_elo_put, 1);
            elo_k1[elo_cur].eloArray[temp] = Elo_k1[i];
        }

        __syncthreads();

        if (isLastBlockDone) {
            if (temp == (index_elo_put - 1)) {
                (*elo_int_x) = temp;
                memset(elo_x, 0, sizeof(Elo) * index_elo_put);
                memcpy(elo_x, elo_k1[elo_cur].eloArray, sizeof(Elo) * index_elo_put);
                memset(elo_k1[elo_cur].eloArray, 0, sizeof(SetMap) * index_elo_put);

//     for (int i = 0; i < (*elo_int_x); ++i) {
//                    printf("%d CANDIDATO VAI PARA FREQUENCIA  Round :%d  | ELO :%s | IndexArray :%d | Suporte :%d\n",
//                           blockIdx.x,
//                           elo_cur, elo_x[i].ItemId, elo_x[i].indexArrayMap,
//                           elo_x[i].suporte);
//                }
                Elo *eloSetTemp = (Elo *) malloc(sizeof(Elo));
                eloSetTemp = (Elo *) malloc(sizeof(Elo) * index_elo_put);
                int *ss = (int *) malloc(sizeof(int));
                int block_size = 16;
                int blocks_per_row = (index_elo_put / block_size) + (index_elo_put % block_size > 0 ? 1 : 0);
                printf("Quantidade de Blocos %d Total %d\n", blocks_per_row, index_elo_put);
                frequencia_x1 << < blocks_per_row, block_size >> >
                                                   (elo_x, elo_int_x, eloSetTemp, ss);
                cudaDeviceSynchronize();


                blocks_per_row = ( (*ss) / block_size) + ( (*ss) % block_size > 0 ? 1 : 0);

                frequencia_x2 << < blocks_per_row, block_size >> >
                                                   (elo_k1, elo_cur, eloSetTemp, ss, (*minimo_suporte));
                cudaDeviceSynchronize();
                printf("AQUI DEPOIS SETMAP TESTE = %d\n", (*ss));
                for (int i = 0; i < (*ss); ++i) {
                    printf("VOLTA DA FREQUENCIA SEMAP  Round :%d  | ELO :%s | IndexArray :%d | Suporte :%d\n", elo_cur,
                           elo_k1[elo_cur].eloArray[i].ItemId, eloSetTemp[i].indexArrayMap,
                           elo_k1[elo_cur].eloArray[i].suporte);
                }
//                free(Elo_k1);
//                free(eloSetTemp->eloArray);
//                free(eloSetTemp);

            }
        }








//        if (temp == (index_elo_put - 1)) {
//
//            printf("Quem Sou eu ? %d bloco %d TesteTotal %d\n",threadIdx.x,blockIdx.x, atomicAdd(&teste, index_elo_put));
//
////            Elo *set_elo = (Elo *) malloc(sizeof(Elo) * index_elo_put);
////            memset(set_elo, 0, sizeof(Elo) * index_elo_put);
////            memset(elo_x,0,sizeof(Elo) * temp);
////            memcpy(elo_x,elo_k1[elo_cur].eloArray, sizeof(Elo) * temp);
//              int eloSize = 0;
////            for (int k = 0; k < index_elo_put; ++k) {
////
////      int i = 0;
////                bool flag = true;
////                while (i < index_elo_put && flag) {
////                    if (0 == compare(set_elo[i].ItemId, "")) {
////                        set_elo[i] = elo_k1[elo_cur].eloArray[k];
////                        eloSize++;
////                        flag = false;
////                    } else {
////                        if (0 == compare(elo_k1[elo_cur].eloArray[k].ItemId, set_elo[i].ItemId)) {
////                            flag = false;
////                            set_elo[i].suporte += elo_k1[elo_cur].eloArray[k].suporte;
////                        }
////                    }
////                    i++;
////                }
////            }
////

////
////            int x_threads = (*elo_int_x);
////            printf("AQUI CURR %d\n", (*elo_curr));
////            if(x_threads>0) {
////                *(elo_curr) = *(elo_curr) +1;
//////                printf("Chamando denovo com %d threads \n", x_threads);
////                pfp_growth << < 1, x_threads,x_threads*sizeof(Elo)*22>> >
////                                              (elo_k1, elo_curr, arrayMap, arrayMapSize,elo_xx,elo_int_x,minimo_suporte);
////////                cudaDeviceSynchronize();
////            }
//////                free(elo_x);
////
//        }

    }
}


