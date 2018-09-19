//
// Created by rafael on 20/08/18.
//

#include <cudaHeaders.h>
#include "Kernel.h"
#include "PFPTree.h"
#include "PFPArray.h"
#include <cuda_runtime_api.h>
#include <cstdio>
#include "cuda.h"
#include "../include/PFPArray.h"


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




__global__ void frequencia_x(EloVector *elo_k1,int elo_k1_current,Elo *elo_x,int *eloMapSizePointer, int minimo) {
    extern __shared__ SetMap setMap[];
    __shared__ Elo elo_new_put[199];
    auto indexAtual = blockIdx.x * blockDim.x + threadIdx.x; //PC
    bool newFlag = true;
    int indexSetMap = 0;
    int eloSize = 0;
    int eloMapSize =(*eloMapSizePointer);
    memset(elo_new_put, 0, sizeof(Elo) * eloMapSize);
    memset(setMap, 0, sizeof(SetMap) * eloMapSize);
    index_elo_setmap =0;
    index_new_elo_setmap=0;

    __syncthreads();
        if (threadIdx.x == 0) {
            for (int k = 0; k < eloMapSize; ++k) {

                my_strcpy(setMap[k].elo.ItemId, " ");
            }

        }
    __syncthreads();

        if (indexAtual == 0) {

            for (int k = 0; k < eloMapSize; ++k) {
                int i = 0;
                bool flag = true;
                while (i < eloMapSize && flag) {
                    if (0 == compare(setMap[i].elo.ItemId, " ")) {

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
                elo_new_put[atomicAdd(&index_new_elo_setmap, 1)] = elo_x[indexAtual];
//                if(elo_k1_current==2) {
//                    printf("Thread %d Elo size %d AQUI %s %d\n",threadIdx.x,index_new_elo_setmap, elo_new_put[0].ItemId,elo_new_put[0].suporte);
////                    printf("ELozine  Elo size %d AQUI %s %d\n",eloMapSize, elo_x[0].ItemId,elo_x[0].suporte);
//                }
                newFlag = false;
            }
            indexSetMap++;
        }
        __syncthreads();

    if (threadIdx.x == eloMapSize - 1) {
        (*eloMapSizePointer)=index_new_elo_setmap;
        memset(elo_x, 0, sizeof(Elo) * eloMapSize);

        for (int i = 0; i <index_new_elo_setmap; ++i) {
            elo_x[i]=elo_new_put[i];

        }



        int valux=0;
            for (int j = 0; j <=index_elo_setmap; ++j) {
                if( (setMap[j].elo.suporte >= minimo)){
                    elo_k1[elo_k1_current].eloArray[valux] = setMap[j].elo;
                    valux++;
                }


            }
        elo_k1[elo_k1_current].size=valux;
    }

}

__global__ void pfp_growth(EloVector *elo_k1, int *elo_curr ,ArrayMap *arrayMap,size_t arrayMapSize, Elo *elo_x, int *elo_int_x) {
    extern __shared__ Elo elo[];
    auto indexAtual = blockIdx.x * blockDim.x + threadIdx.x;


    int  elo_cur= (*elo_curr);
    int elo_x_size= (*elo_int_x);
//    printf("AQUI ANATES %d \n",elo_x_size);
        int xxx = 0;
        bool flag = true;
        Elo *Elo_k1 = (Elo * ) malloc(sizeof(Elo) *elo_x_size);
//        printf("Sou o %d  :%s\n",indexAtual,elo_k1[elo_cur].eloArray[indexAtual].ItemId);


    auto indexThreadArrayMap = elo_x[indexAtual].indexArrayMap; // indexAtual 13 =  c | parent 4
    auto indexParentArrayMap = elo_x[indexAtual].indexArrayMap;
    while (flag) { //
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

// Algoritmo 1 End;
// Algoritmo 2 Begin;
    __syncthreads();
        for (int i = 0; i < (xxx - 1); ++i) {
            elo_x[atomicAdd(&index_elo_put, 1)] = Elo_k1[i];


        }

        if (threadIdx.x == elo_x_size - 1) {
            (*elo_int_x)=index_elo_put;

//            printf("AQUI ANATES %d\n",  (*elo_int_x));

            frequencia_x << < 1, index_elo_put, sizeof(SetMap) * index_elo_put >> >
                                                (elo_k1, elo_cur, elo_x,elo_int_x , 3);

            cudaDeviceSynchronize();
//            printf("AQUI DEPOIS %d\n", (*elo_int_x));
//            for (int i = 0; i < (*elo_int_x); ++i) {
//                printf("VOLTA DA MORTE  Round :%d  | ELO :%s | IndexArray :%d | Suporte :%d\n",elo_cur,elo_x[i].ItemId,elo_x[i].indexArrayMap,elo_x[i].suporte);
//            }
            index_elo_put = 0;
            if ( (*elo_int_x) > 0) {
                int x_threads = (*elo_int_x);
                *(elo_curr) = *(elo_curr) + 1;
//                printf("Chamando denovo com %d threads \n", x_threads);
                pfp_growth << < 1, x_threads, (*elo_int_x) * 4 * sizeof(Elo) >> >
                                              (elo_k1, elo_curr, arrayMap, arrayMapSize,elo_x,elo_int_x);
                cudaPeekAtLastError();
                cudaDeviceSynchronize();
            }
        }

}


