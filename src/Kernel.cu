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

#include "../include/Kernel.h"
#include "../include/PFPTree.h"
#include "../include/PFPArray.h"
#include <cuda_runtime_api.h>
#include <cstdio>
#include "cuda.h"
const int block_size=64;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


__device__ char atomicMinChar(char* address, char val)

{

    unsigned int *base_address = (unsigned int *)((size_t)address & ~3);

    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};

    unsigned int sel = selectors[(size_t)address & 3];

    unsigned int old, assumed, min_, new_;



    old = *base_address;

    do {
        assumed = old;
        min_ = min(val, (char)__byte_perm(old, 0, ((size_t)address & 3)));
        new_ = __byte_perm(old, min_, sel);
        old = atomicCAS(base_address, assumed, new_);
    } while (assumed != old);

    return old;

}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__device__ int compare(const char *String_1, const char *String_2) {
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
__device__ volatile char *my_strcpy(volatile char *dest, volatile char *src) {
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
//__device__ unsigned int count = 0;
__shared__ bool isLastBlockDone;
__device__ unsigned int count = 0;
__device__ unsigned index_elo_put = 0;
__device__ unsigned int indexSetMap = 0;
__device__ unsigned int indexEloFim = 0;


__global__ void
frequencia_x3( Elo **new_canditatos_suporte, int new_canditatos_suporte_size, Elo **new_canditatos, int new_canditatos_size,
              Elo *elo_x, int minimo) {
    auto indexAtual = blockIdx.x * blockDim.x + threadIdx.x; //PC

    if (indexAtual < new_canditatos_size) {
        bool flag = true;
        int index = 0;
//        printf(" X %d\n",x);
        while (flag && index < new_canditatos_suporte_size) {
//        	printf("%s\n",elo_k1[elo_cur+index].ItemId);
            if (0 == compare(new_canditatos_suporte[index]->ItemId, new_canditatos[indexAtual]->ItemId)) {
            	int temp=atomicAdd(&indexEloFim, 1);
            	my_strcpy(elo_x[temp].ItemId, new_canditatos[indexAtual]->ItemId);
            	elo_x[temp].indexArrayMap = new_canditatos[indexAtual]->indexArrayMap;
            	elo_x[temp].suporte = new_canditatos[indexAtual]->suporte;
//                printf("IndexAll %d Item %s suporte %d\n", indexAtual, elo_x[temp].ItemId,
//                       elo_x[temp].suporte);
                flag = false;
//                new_canditatos_suporte
            } else {
                index++;
            }
        }
    }
}

__global__ void
frequencia_x2( Elo *elo_k1, int elo_cur, Elo **set_elo, int eloMapSizePointer,
              int minimo) {
    auto indexAtual = blockIdx.x * blockDim.x + threadIdx.x; //PC

    if (indexAtual < eloMapSizePointer) {
//        printf("frequencia_x2 set_map_elo %s \n",set_elo[indexAtual].ItemId);

        if (set_elo[indexAtual]->suporte >= minimo) {
            int temp = atomicAdd(&indexSetMap, 1);
//            printf("Index Frequenia 2 %d \n",elo_cur+temp);
            my_strcpy(elo_k1[elo_cur+temp].ItemId,set_elo[indexAtual]->ItemId);
            elo_k1[elo_cur+temp].indexArrayMap = set_elo[indexAtual]->indexArrayMap;
            elo_k1[elo_cur+temp].suporte = set_elo[indexAtual]->suporte;

//            printf("index %d Thread %d ITEM %s %d\n", temp , indexAtual, elo_k1[elo_cur+temp].ItemId,
//            		elo_k1[elo_cur+temp].suporte);

        }

    }
}
__device__ unsigned int magical = 0;

__device__ void frequencia_x1_device( Elo** new_canditatos_cont_suporte, Elo* new_canditatos){
	            int index = 0;
	            bool newFlag = true;
	            while (newFlag ) {
	            	if(0 == compare(new_canditatos_cont_suporte[atomicAdd(&magical,0)]->ItemId,"")){
	            	        int tempMagical = atomicAdd(&magical,1);
	            		    my_strcpy(new_canditatos_cont_suporte[tempMagical]->ItemId,new_canditatos->ItemId);
	            		    new_canditatos_cont_suporte[tempMagical]->indexArrayMap = new_canditatos->indexArrayMap;
	            		    new_canditatos_cont_suporte[tempMagical]->suporte = new_canditatos->suporte;
	            		    newFlag = false;
	            	}else if (0 == compare(new_canditatos_cont_suporte[atomicAdd(&magical,0)]->ItemId, new_canditatos->ItemId)) {
	            	        new_canditatos_cont_suporte[atomicAdd(&magical,0)]->suporte += new_canditatos->suporte;
	            	        newFlag = false;
	                   }else{
	                      index++;
	                    }
	               }
}

__global__ void
//geracao_de_candidatos(Elo** new_canditatos_cont_suporte,Elo **temp_elo_x, ArrayMap *arrayMap,
geracao_de_candidatos(Elo **temp_elo_x, ArrayMap *arrayMap,
                       int arrayMapSize,
                      Elo *elo_x, int *elo_int_x, int *minimo_suporte) {
    auto indexAtual = blockIdx.x * blockDim.x + threadIdx.x;
    if (indexAtual < (*elo_int_x)) {
        bool flag = true;
        auto indexThreadArrayMap = elo_x[indexAtual].indexArrayMap;
        auto indexParentArrayMap = elo_x[indexAtual].indexArrayMap;
        while (flag ) {
            indexParentArrayMap = arrayMap[indexParentArrayMap].indexP;
            if (arrayMap[indexThreadArrayMap].indexP != -1 &&
                arrayMap[indexParentArrayMap].indexP != -1) {
            	int temp= atomicAdd(&index_elo_put, 1);
            	 temp_elo_x[temp]=(Elo*)malloc(sizeof(Elo));
            	 memset(temp_elo_x[temp],0,sizeof(Elo));
                 my_cpcat(elo_x[indexAtual].ItemId,arrayMap[indexParentArrayMap].ItemId,temp_elo_x[temp]->ItemId);
                 temp_elo_x[temp]->indexArrayMap = indexParentArrayMap;
                 temp_elo_x[temp]->suporte = elo_x[indexAtual].suporte;
//                 frequencia_x1_device(new_canditatos_cont_suporte,temp_elo_x[temp]);

            } else {
                flag = false;
            }
        }
   }

}

__global__ void aloc(Elo** elos, int x){
	 auto indexAtual = blockIdx.x * blockDim.x + threadIdx.x;
	    if (indexAtual < x ) {
	      elos[indexAtual]=(Elo*)malloc(sizeof(Elo));
	      memset(elos[indexAtual],0,sizeof(Elo));

	    }
}
//__global__ void mmenset(Elo** elos, int x){
//	 auto indexAtual = blockIdx.x * blockDim.x + threadIdx.x;
//	    if (indexAtual < x ) {
//	      memset(elos[indexAtual],0,sizeof(Elo));
//	    }
//}


__global__ void freee(Elo** elos, int x){
	 auto indexAtual = blockIdx.x * blockDim.x + threadIdx.x;
	    if (indexAtual < x ) {
	     free(elos[indexAtual]);
	    }
}


__global__ void frequencia_x1_paralela(Elo** new_canditatos_cont_suporte, int *new_canditatos_cont_suporte_size , Elo** new_canditatos, int new_canditatos_size){
	 auto i = blockIdx.x * blockDim.x + threadIdx.x;
	  if(i<new_canditatos_size){
	            int index = 0;
	            printf("%s \n",new_canditatos[i]->ItemId);
//	            atomicAdd(&(new_canditatos_cont_suporte["312312"]->suporte),1);

//	            while (newFlag and index != magical ) {
//	            	 if (0 == compare(new_canditatos_cont_suporte[index]->ItemId, new_canditatos[i]->ItemId)) {
//	            		                new_canditatos_cont_suporte[index]->suporte += new_canditatos[i]->suporte;
//	            		                newFlag = false;
//	                   }else{
//	                      index++;
//	                    }
//	                 }
//	                 if(index == magical){
//	            	         int tempMagical = atomicAdd(&magical,1);
//	            			 my_strcpy(new_canditatos_cont_suporte[tempMagical]->ItemId,new_canditatos[i]->ItemId);
//	            			 new_canditatos_cont_suporte[tempMagical]->indexArrayMap = new_canditatos[i]->indexArrayMap;
//	            			 new_canditatos_cont_suporte[tempMagical]->suporte = new_canditatos[i]->suporte;
//	            			 newFlag = false;
//	                 }
	      }
}

__global__ void frequencia_x1(Elo** new_canditatos_cont_suporte, int *new_canditatos_cont_suporte_size , Elo** new_canditatos, int new_canditatos_size){
	        for (int i = 0; i < new_canditatos_size; ++i){
	        	printf("%d \n",i);
	            int index = 0;
	            bool newFlag = true;
	            while (newFlag) {
	            	 if (0 == compare(new_canditatos_cont_suporte[index]->ItemId, "")) {
	            		                	(*new_canditatos_cont_suporte_size)++;
//	            		                	printf(" %s \n",new_canditatos[i]->ItemId);
	            		                	my_strcpy(new_canditatos_cont_suporte[index]->ItemId,new_canditatos[i]->ItemId);
	            		                	new_canditatos_cont_suporte[index]->indexArrayMap = new_canditatos[i]->indexArrayMap;
	            		                	new_canditatos_cont_suporte[index]->suporte = new_canditatos[i]->suporte;
	            		                    newFlag = false;
	                } else if (0 == compare(new_canditatos_cont_suporte[index]->ItemId, new_canditatos[i]->ItemId)) {
	            	    	                new_canditatos_cont_suporte[index]->suporte += new_canditatos[i]->suporte;
	            		                    newFlag = false;
	                }else{
	           	               index++;
	                }
	            }
	   }
}



__global__ void
runKernel( Elo *elo_k1,  int *elo_curr, ArrayMap *arrayMap, int arrayMapSize,
          Elo *elo_x, int *elo_int_x, int *minimo_suporte) {
    bool flag = true;
    int roud=0;
    int blocks_per_row = 0;
    while(flag){
    	roud++;
    	blocks_per_row = ((*elo_int_x) / block_size) + ((*elo_int_x) % block_size > 0 ? 1 : 0);
        printf("Round %d pfp_growth new Blocos %d Total %d\n",roud, blocks_per_row, (*elo_int_x));
        printf("arrayMapSize  %d\n",sizeof(Elo));

        Elo **new_canditatos=(Elo**)malloc(sizeof(Elo)*(*elo_int_x)*arrayMapSize);
//        memset(new_canditatos,0,sizeof(Elo)*331238);
//                for(int i=0;i<(*elo_int_x)*arrayMapSize;++i){
//                	new_canditatos[i]=(Elo*)malloc(sizeof(Elo));
//                }
        geracao_de_candidatos << < blocks_per_row, block_size >> >
                                                   (new_canditatos, arrayMap, arrayMapSize, elo_x, elo_int_x, minimo_suporte);
        cudaDeviceSynchronize();

//        thrust::device_vector < int > D;

        Elo **new_canditatos_cont_suporte=(Elo**)malloc(sizeof(Elo*)*index_elo_put);

        blocks_per_row = (index_elo_put / block_size) + (index_elo_put % block_size > 0 ? 1 : 0);
        aloc<<<blocks_per_row,block_size>>>(new_canditatos_cont_suporte,index_elo_put);
//        cudaDeviceSynchronize();
//
//
//
        printf("Frequencia 1 Quantidade de Blocos %d Total %d\n", 1, 1);
        int *new_canditatos_cont_suporte_size=(int*)malloc(sizeof(int));
        (*new_canditatos_cont_suporte_size)=0;
        frequencia_x1<<<1,1>>>(new_canditatos_cont_suporte,new_canditatos_cont_suporte_size,new_canditatos,index_elo_put);
        cudaDeviceSynchronize();
        printf("Frequencia 1 %d\n",(*new_canditatos_cont_suporte_size));
//
//
//        for(int i =0;i<3;++i){
//            printf(" ID=%d ITEM=%s S=%d \n",i,new_canditatos_cont_suporte[i]->ItemId,new_canditatos_cont_suporte[i]->suporte);
//        }


        blocks_per_row = ((*new_canditatos_cont_suporte_size) / block_size) + ((*new_canditatos_cont_suporte_size) % block_size > 0 ? 1 : 0);
        printf("Frequancia 2 Quantidade de Blocos %d Total %d\n", blocks_per_row, (*new_canditatos_cont_suporte_size));
        frequencia_x2 << < blocks_per_row, block_size >> >
                                           (elo_k1, (*elo_curr), new_canditatos_cont_suporte, (*new_canditatos_cont_suporte_size), (*minimo_suporte));

//
//
        blocks_per_row = (index_elo_put / block_size) + (index_elo_put % block_size > 0 ? 1 : 0);
        printf("Frequancia 3 Quantidade de Blocos %d Total %d\n", blocks_per_row, index_elo_put);
        frequencia_x3 << < blocks_per_row, block_size >> >
                                           (new_canditatos_cont_suporte,(*new_canditatos_cont_suporte_size), new_canditatos,index_elo_put, elo_x, (*minimo_suporte));
        cudaDeviceSynchronize();
        freee<< < blocks_per_row, block_size >> >(new_canditatos,index_elo_put);
        blocks_per_row = ((*new_canditatos_cont_suporte_size) / block_size) + ((*new_canditatos_cont_suporte_size) % block_size > 0 ? 1 : 0);
        freee<< < blocks_per_row, block_size >> >(new_canditatos_cont_suporte,(*new_canditatos_cont_suporte_size));
        free(new_canditatos_cont_suporte_size);
        (*elo_curr)=(*elo_curr)+indexSetMap;
        (*elo_int_x) = indexEloFim;
        printf("Elo_x Restante %d\n",(*elo_int_x));
        if (indexEloFim > 0) {
        	   count = 0;
        	   indexEloFim = 0;
        	   index_elo_put = 0;
        	   indexSetMap = 0;

        } else {
            flag = false;
        }
    }
}
