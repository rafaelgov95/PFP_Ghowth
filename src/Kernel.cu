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
__device__ unsigned index_elo_put = 0;
__device__ unsigned int indexSetMap = 0;
__device__ unsigned int indexEloFim = 0;


__global__ void
frequencia_x3(__volatile__ EloVector *elo_k1, __volatile__ int elo_cur, Elo *elo_x_temp, int set_map_elo_size,
              Elo *elo_x, int minimo) {
    auto indexAtual = blockIdx.x * blockDim.x + threadIdx.x; //PC
    if (indexAtual < set_map_elo_size) {
        bool flag = true;
        int index = 0;
        while (flag && index < elo_k1[elo_cur].size) {
            if (0 == compare(elo_k1[elo_cur].eloArray[index].ItemId, elo_x_temp[indexAtual].ItemId)) {
                elo_x[atomicAdd(&indexEloFim, 1)] = elo_x_temp[indexAtual];
//                printf("  IndexAll %d Item %s suporte %d\n",  indexAtual, elo_x[indexSetMap].ItemId,
//                       elo_x[indexSetMap].suporte);
                flag = false;
            } else {
                index++;
            }
        }
    }
}

__global__ void
frequencia_x2(__volatile__ EloVector *elo_k1, __volatile__ int elo_cur, Elo *set_elo, int eloMapSizePointer,
              int minimo) {
    auto indexAtual = blockIdx.x * blockDim.x + threadIdx.x; //PC

    if (indexAtual < eloMapSizePointer) {
//        printf("frequencia_x2 teste  set_map_elo %s \n",set_elo[indexAtual].ItemId);

        if (set_elo[indexAtual].suporte >= minimo) {
            int temp = atomicAdd(&indexSetMap, 1);

            elo_k1[elo_cur].eloArray[temp] = set_elo[indexAtual];
//            printf("index %d Thread %d ITEM %s %d\n", temp , indexAtual, set_elo[indexAtual].ItemId,
//                   set_elo[indexAtual].suporte);

        }

    }
}


__global__ void
geracao_de_candidatos(__volatile__ EloVector *elo_k1, __volatile__ int *elo_curr, ArrayMap *arrayMap,
                      size_t arrayMapSize,
                      Elo *elo_x, int *elo_int_x, int *minimo_suporte) {
    auto indexAtual = blockIdx.x * blockDim.x + threadIdx.x;
    if (indexAtual < (*elo_int_x)) {
        int elo_x_size = (*elo_int_x);
        int elo_cur = (*elo_curr);
//        if (elo_cur == 5)
//            printf("Thread %d Elo_x %s\n", indexAtual,elo_x[indexAtual].ItemId);

        int xxx = 0;
        index_elo_put = 0;
        bool flag = true;
//        Elo *Elo_k1[elo_x_size];
        Elo* Elo_k1= (Elo *) malloc(sizeof(Elo) * arrayMapSize);
        memset(Elo_k1,0,sizeof(Elo) * arrayMapSize);
        printf("Meu pointer Thread %d pointer %p\n", indexAtual,Elo_k1);

        auto indexThreadArrayMap = elo_x[indexAtual].indexArrayMap;
        auto indexParentArrayMap = elo_x[indexAtual].indexArrayMap;
        while (flag ) {
            indexParentArrayMap = arrayMap[indexParentArrayMap].indexP;
            if (arrayMap[indexThreadArrayMap].indexP != -1 &&
                arrayMap[indexParentArrayMap].indexP != -1) {
//                Elo_k1[xxx].ItemId=(char *) malloc(sizeof(char)*16);
//                if (elo_cur != 6) {
                    my_cpcat(elo_x[indexAtual].ItemId,
                             arrayMap[indexParentArrayMap].ItemId,Elo_k1[xxx].ItemId);
                    Elo_k1[xxx].indexArrayMap = indexParentArrayMap;
                    Elo_k1[xxx].suporte = elo_x[indexAtual].suporte;

//                } else {
//                    my_cpcat(elo_x[indexAtual].ItemId,
//                             arrayMap[indexParentArrayMap].ItemId, Elo_k1[xxx].ItemId);
//                    printf("Thread %d  | %s IndexArrayMap %d suporte\n",indexAtual,Elo_k1[xxx].ItemId, elo_x[indexAtual].suporte);
//                    Elo_k1[xxx].indexArrayMap;
//                    indexParentArrayMap;
//                    Elo_k1[xxx].suporte;
//                             elo_x[indexAtual].suporte;
//                    printf("INT Thread %d pai %d xxx %d \n",xxx);
//                    printf("INT Thread %d pai %d xxx %d \n",indexAtual,indexParentArrayMap,xxx);
//                    printf("STRING Thread %s pai %s elo_x %s\n",elo_x[indexAtual].ItemId,arrayMap[indexParentArrayMap].ItemId,Elo_k1[xxx].ItemId);

//                }

            } else {
                flag = false;
            }

            xxx++;
            printf("Thread %d XXX %d\n", indexAtual,xxx);

        }
//        if (elo_cur != 6) {
            for (int i = 0; i < (xxx - 1); ++i) {
//                printf("%d %d\n",atomicAdd(&index_elo_put, 1),indexAtual);
                elo_k1[elo_cur].eloArray[atomicAdd(&index_elo_put, 1)] = Elo_k1[i];
//                free(&Elo_k1[i]);
            }
//            free(Elo_k1);
//            __syncthreads();
//            free(Elo_k1);
//        }

//        free(Elo_k1);


    }
}


__global__ void
runKernel(__volatile__ EloVector *elo_k1, __volatile__ int *elo_curr, ArrayMap *arrayMap, size_t arrayMapSize,
          Elo *elo_x, int *elo_int_x, int *minimo_suporte) {
    bool flag = true;
    int block_size = 32;
    int blocks_per_row = 0;
    while (flag) {
        int elo_cur = (*elo_curr);
        blocks_per_row = ((*elo_int_x) / block_size) + ((*elo_int_x) % block_size > 0 ? 1 : 0);
        printf("pfp_growth new Blocos %d Total %d\n", blocks_per_row, (*elo_int_x));
        geracao_de_candidatos << < blocks_per_row, block_size >> >
                                                   (elo_k1, elo_curr, arrayMap, arrayMapSize, elo_x, elo_int_x, minimo_suporte);
        cudaDeviceSynchronize();
        printf("Round %d  Value Elo_x = %d \n", elo_cur, index_elo_put);
        memset(elo_x, 0, sizeof(Elo) * index_elo_put);
        memcpy(elo_x, elo_k1[elo_cur].eloArray, sizeof(Elo) * index_elo_put);
        memset(elo_k1[elo_cur].eloArray, 0, sizeof(Elo) * index_elo_put);
        Elo *eloSetTemp = (Elo *) malloc(sizeof(Elo) * index_elo_put);
        blocks_per_row = (index_elo_put / block_size) + (index_elo_put % block_size > 0 ? 1 : 0);
        printf("Frequencia 1 Quantidade de Blocos %d Total %d\n", 1, 1);
        int elo_set_map_size = 0;
        for (int i = 0; i < index_elo_put; ++i) {
            int index = 0;
            bool newFlag = true;
            while (newFlag) {
                if (0 == compare(eloSetTemp[index].ItemId, "")) {
                    elo_set_map_size++;
                    eloSetTemp[index] = elo_x[i];
                    newFlag = false;
                } else if (0 == compare(eloSetTemp[index].ItemId, elo_x[i].ItemId)) {
                    eloSetTemp[index].suporte += elo_x[i].suporte;
                    newFlag = false;
                } else {
                    index++;
                }
            }
        }

        blocks_per_row = (elo_set_map_size / block_size) + (elo_set_map_size % block_size > 0 ? 1 : 0);
        printf("Frequancia 2 Quantidade de Blocos %d Total %d\n", blocks_per_row, elo_set_map_size);
        frequencia_x2 << < blocks_per_row, block_size >> >
                                           (elo_k1, elo_cur, eloSetTemp, elo_set_map_size, (*minimo_suporte));
        cudaDeviceSynchronize();
        free(eloSetTemp);

        elo_k1[elo_cur].size = indexSetMap;
        Elo *elo_x_temp = (Elo *) malloc(sizeof(Elo) * index_elo_put);
        memcpy(elo_x_temp, elo_x, index_elo_put * sizeof(Elo));
        memset(elo_x, 0, sizeof(Elo) * index_elo_put);
        blocks_per_row = (index_elo_put / block_size) + (index_elo_put % block_size > 0 ? 1 : 0);
        printf("Frequancia 3 Quantidade de Blocos %d Total %d\n", blocks_per_row, index_elo_put);
        frequencia_x3 << < blocks_per_row, block_size >> >
                                           (elo_k1, elo_cur, elo_x_temp, index_elo_put, elo_x, (*minimo_suporte));
        cudaDeviceSynchronize();
        free(elo_x_temp);
        if (indexEloFim > 0) {
            (*elo_curr) = (*elo_curr) + 1;
            (*elo_int_x) = indexEloFim;
            count = 0;
            indexEloFim = 0;
            index_elo_put = 0;
            indexSetMap = 0;
        } else {
            flag = false;
        }
    }


}