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
#include "../../../../../../usr/include/form.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

typedef struct {
    Elo elo;
    int size;
}SetMap;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
__device__ char *counter1;
__device__ int index_new_elo;
__device__ int index_elo_setmap;
__shared__ Elo elo[256];
__device__ int index_elo_put;


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

__global__ void frequencia_x(EloVector *elo_k1,int elo_k1_current,Elo *elo_x,int eloMapSize, int minimo) {
    extern __shared__ SetMap
    setMap[];
    __shared__
    Elo elo_new_put[199];
    int eloSize = 0;
    memset(elo_new_put, 0, sizeof(Elo) * eloMapSize);
    memset(setMap, 0, sizeof(SetMap) * eloMapSize);
    index_elo_setmap =0;



        if (threadIdx.x == 0) {
            for (int k = 0; k < eloMapSize; ++k) {
                my_strcpy(setMap[k].elo.ItemId, " ");
            }

        }

        __syncthreads();
        if (threadIdx.x == 0) {

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
            atomicAdd(&index_elo_setmap, eloSize);

        }

        __syncthreads();
        bool newFlag = true;
        int indexSetMap = 0;
        index_new_elo=0;


        while (newFlag && indexSetMap < index_elo_setmap) {
            if ((0 == compare(elo_x[threadIdx.x].ItemId, setMap[indexSetMap].elo.ItemId)) &&
                (setMap[indexSetMap].elo.suporte >= minimo)) {
                elo_new_put[atomicAdd(&index_new_elo, 1)] = elo_x[threadIdx.x];
                newFlag = false;
            }
            indexSetMap++;
        }


        if (threadIdx.x == eloMapSize - 1) {
//        for(int i =0;i<index_elo_setmap;++i ){
//            printf("SetMAP Thread %d valor MAP %s Suporte %d \n",threadIdx.x,setMap[i].elo.ItemId,setMap[i].elo.suporte);
//        }
//        for(int i =0;i<inde_new_elo;++i ){
//        printf("Elo_new_PUT Thread %d valor MAP %s Suporte %d \n",threadIdx.x,elo_new_put[i].ItemId,elo_new_put[i].suporte);
//        }
            for (int i = 0; i < index_new_elo; ++i) {
                elo_k1[elo_k1_current].eloArray[i] = elo_new_put[i];
            }
            elo_k1[elo_k1_current].size = index_new_elo;
        }

}

__global__ void pfp_growth(EloVector *elo_k1, int *elo_curr ,ArrayMap *arrayMap,size_t arrayMapSize) {

// Algoritmo 1 Begin;
  int  elo_cur = *elo_curr;
//  printf("%d",elo_cur);
auto indexAtual = threadIdx.x; //PC
    int xxx = 0;
    bool flag = true;
    Elo *Elo_k1 = (Elo *) malloc(sizeof(Elo) * elo_k1[elo_cur].size);
    printf("QUEM SOU EU Round :%d  | ELO :%s | IndexArray :%d | Suporte :%d\n",elo_cur,elo_k1[elo_cur].eloArray[threadIdx.x].ItemId,elo_k1[elo_cur].eloArray[threadIdx.x].indexArrayMap,elo_k1[elo_cur].eloArray[threadIdx.x].suporte);
    while (flag && (indexAtual + xxx) <=elo_k1[elo_cur].size) { //
        auto indexThreadArrayMap = elo_k1[elo_cur].eloArray[indexAtual].indexArrayMap; // indexAtual 13 =  c | parent 4
        auto indexParentArrayMap = arrayMap[elo_k1[elo_cur].eloArray[indexAtual + xxx].indexArrayMap].indexP;
        if (arrayMap[indexThreadArrayMap].indexP != -1 &&
                arrayMap[indexParentArrayMap].indexP != -1) {
             my_cpcat(elo_k1[elo_cur].eloArray[indexAtual].ItemId,
                     arrayMap[indexParentArrayMap].ItemId, Elo_k1[xxx].ItemId);
                Elo_k1[xxx].indexArrayMap = indexParentArrayMap;
                Elo_k1[xxx].suporte = elo_k1[elo_cur].eloArray[indexAtual].suporte;
            }
            else{
               flag = false;
            }
            xxx++;
        }




// Algoritmo 1 End;
// Algoritmo 2 Begin;
    for (int i = 0; i < (xxx-1); ++i)
    elo[atomicAdd(&index_elo_put,1)]=Elo_k1[i];

    if (threadIdx.x == elo_k1[elo_cur].size-1 ) {
        Elo *elo_x= (Elo *)malloc(sizeof(Elo)*index_elo_put);
        for (int i = 0; i < index_elo_put; ++i){
                    elo_x[i]= elo[i];
//                    printf("Sem SetMap round :%d  valor %s  index %d  suporte %d\n",elo_cur,elo_x[i].ItemId,elo_x[i].indexArrayMap,elo_x[i].suporte);
        }


        frequencia_x<<<1,index_elo_put,sizeof(SetMap)*index_elo_put>>>(elo_k1,elo_cur+1,elo_x,index_elo_put,3);
        cudaDeviceSynchronize();
//        for (int i = 0; i < elo_k1[elo_cur+1].size; ++i){
//                printf("VOLTA  SETMAP Round :%d  | ELO :%s | IndexArray :%d | Suporte :%d\n",elo_cur,elo_k1[elo_cur+1].eloArray[i].ItemId,elo_k1[elo_cur+1].eloArray[i].indexArrayMap,elo_k1[elo_cur+1].eloArray[i].suporte);
//        }
        index_elo_put=0;
        if(elo_k1[elo_cur+1].size > 0){
            int x_threads =(elo_k1[elo_cur+1].size);
//            printf("Chamando denovo com %d threads \n",x_threads);
            *(elo_curr)=*(elo_curr)+1;
            pfp_growth<<<1,x_threads>>>(elo_k1,elo_curr,arrayMap, arrayMapSize);
            cudaDeviceSynchronize();

        }
       }

}


