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





__device__ int compare(char* String_1, char* String_2)
{
    char TempChar_1,
            TempChar_2;

    do
    {
        TempChar_1 = *String_1++;
        TempChar_2 = *String_2++;
    } while(TempChar_1 && TempChar_1 == TempChar_2);

    return TempChar_1 - TempChar_2;
}

__device__ bool my_strcmp( char *array1, char *array2) {
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

__device__ int index_elo_setmap;
__device__ int index_new_elo_setmap;
__device__ int index_elo_put;




__global__ void frequencia_x(__volatile__ EloVector *elo_k1,__volatile__ int elo_k1_current, Elo *elo_x,  int *eloMapSizePointer, int minimo) {
    extern __shared__ SetMap setMap[];
    auto indexAtual = blockIdx.x * blockDim.x + threadIdx.x; //PC
    bool newFlag = true;
    int indexSetMap = 0;
    int eloSize = 0;
    int eloMapSize =(*eloMapSizePointer);
    Elo *elo_new_put = (Elo*)malloc(sizeof(Elo));
//    Elo *new_elo_x = (Elo*)malloc(sizeof(Elo)());

    memset(setMap, 0, sizeof(SetMap) * eloMapSize);
    index_elo_setmap =0;
    index_new_elo_setmap=0;

    __syncthreads();
        if (threadIdx.x == 0) {
            for (int k = 0; k < eloMapSize; ++k) {
                int i = 0;
                bool flag = true;
                while (i < eloMapSize && flag) {
                    if (0 == compare(setMap[i].elo.ItemId, "")) {

                        setMap[i].elo = elo_x[k];

                        eloSize++;
                        flag = false;
                    } else {
                        if (0 == compare(elo_x[k].ItemId, setMap[i].elo.ItemId)) {
                            flag = false;
                            setMap[i].elo.suporte += elo_x[k].suporte;
                        }
                    }
                    i++;
                }

            }
            atomicAdd(&index_elo_setmap, eloSize-1);
        }

        __syncthreads();
        while (newFlag && indexSetMap <=index_elo_setmap) {
            if ((0 == compare(elo_x[indexAtual].ItemId, setMap[indexSetMap].elo.ItemId)) &&
                (setMap[indexSetMap].elo.suporte >= minimo)) {
                elo_new_put[0] = elo_x[indexAtual];
//                printf("Thread %d Elo size %d AQUI %s %d\n",threadIdx.x,indexAtual, elo_x[indexAtual].ItemId,elo_x[indexAtual].suporte);
                newFlag = false;
            }
            indexSetMap++;

        }
    __syncthreads();
            memset(elo_x, 0, sizeof(Elo) * eloMapSize);
    __syncthreads();

    if(elo_new_put[0].suporte!=0) {
        elo_x[atomicAdd(&index_new_elo_setmap, 1)] = elo_new_put[0];
    }
    __syncthreads();
    (*eloMapSizePointer)=index_new_elo_setmap;
    index_new_elo_setmap=0;
//    __syncthreads();
     if (indexAtual <= index_elo_setmap && setMap[indexAtual].elo.suporte >= minimo && 0 != compare(setMap[indexAtual].elo.ItemId, "")) {
                    elo_k1[elo_k1_current].eloArray[atomicAdd(&index_new_elo_setmap, 1)] = setMap[indexAtual].elo;
                printf("Thread %d Elo size %d AQUI %s %d\n",indexAtual,index_new_elo_setmap, elo_x[indexAtual].ItemId,elo_x[indexAtual].suporte);
                   elo_k1[elo_k1_current].size=index_new_elo_setmap;

           }



}

__global__ void pfp_growth(__volatile__ EloVector *elo_k1, __volatile__ int *elo_curr ,ArrayMap *arrayMap,size_t arrayMapSize, Elo *elo_x, int *elo_int_x, int *minimo_suporte) {
    extern __shared__ Elo eloMap[];
    auto indexAtual = blockIdx.x * blockDim.x + threadIdx.x;
    int elo_cur= (*elo_curr);
    int elo_x_size= (*elo_int_x);
    int xxx = 0;
    bool flag = true;
    Elo *Elo_k1 = (Elo * ) malloc(sizeof(Elo) *elo_x_size);

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

        for (int i = 0; i < (xxx - 1); ++i) {
            eloMap[atomicAdd(&index_elo_put, 1)] = Elo_k1[i];
        }

    __syncthreads();
        if (threadIdx.x == 0) {
            (*elo_int_x)=index_elo_put;
           Elo *elo_xx =(Elo*)malloc(sizeof(Elo) * index_elo_put);

            memset(elo_xx, 0, sizeof(Elo) * index_elo_put);
            memcpy(elo_xx,eloMap,sizeof(Elo) * index_elo_put);
//                printf("AQUI ANTES %d\n", (*elo_int_x));
//            for (int i = 0; i < (*elo_int_x); ++i) {
//                printf("CANDIDATO VAI PARA FREQUENCIA  Round :%d  | ELO :%s | IndexArray :%d | Suporte :%d\n",elo_cur,elo_xx[i].ItemId,elo_xx[i].indexArrayMap,elo_xx[i].suporte);
//            }
                frequencia_x << < 1,  (*elo_int_x), sizeof(SetMap) *  (*elo_int_x) >> >
                                                (elo_k1, elo_cur, elo_xx, elo_int_x , (*minimo_suporte));
                            cudaDeviceSynchronize();

//            printf("AQUI DEPOIS %d\n", (*elo_int_x));
//            for (int i = 0; i < (*elo_int_x); ++i) {
//                printf("VOLTA DA FREQUENCIA   Round :%d  | ELO :%s | IndexArray :%d | Suporte :%d\n",elo_cur,elo_xx[i].ItemId,elo_xx[i].indexArrayMap,elo_xx[i].suporte);
//            }
                index_elo_put = 0;

            int x_threads = (*elo_int_x);

//            printf("AQUI CURR %d\n", (*elo_curr));

            if(x_threads>0) {
                *(elo_curr) = *(elo_curr) +1;
//                printf("Chamando denovo com %d threads \n", x_threads);
                pfp_growth << < 1, x_threads,x_threads*sizeof(Elo)*4>> >
                                              (elo_k1, elo_curr, arrayMap, arrayMapSize,elo_xx,elo_int_x,minimo_suporte);
//                cudaDeviceSynchronize();
            }
//                free(elo_x);

            }
            __syncthreads();


        }

//}


