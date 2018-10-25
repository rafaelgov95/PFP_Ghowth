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
#include "../include/PFPArray.cu.h"
#include <cuda_runtime_api.h>
#include <cstdio>
#include "cuda.h"
const int block_size=1024;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
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
__shared__ bool isLastBlockDone;
__device__ unsigned int count = 0;
__device__ unsigned index_elo_put = 0;
__device__ unsigned int indexSetMap = 0;
__device__ unsigned int indexEloFim = 0;
__device__ EloArray elo_teste_array;

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


__global__ void
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

__global__ void freee(Elo** elos, int x){
	 auto indexAtual = blockIdx.x * blockDim.x + threadIdx.x;
	    if (indexAtual < x ) {
	     free(elos[indexAtual]);
	    }
}


__global__ void frequencia_x1_paralela(EloArray new_canditatos_cont_suporte, Elo** new_canditatos, int new_canditatos_size){
	 auto i = blockIdx.x * blockDim.x + threadIdx.x;
	  if(i<new_canditatos_size){
		  atomicAdd(&(new_canditatos_cont_suporte[new_canditatos[i]]->suporte),new_canditatos[i]->suporte);
	  }
}

__global__ void frequencia_x1(Elo** new_canditatos_cont_suporte, int *new_canditatos_cont_suporte_size , Elo** new_canditatos, int new_canditatos_size){
	        for (int i = 0; i < new_canditatos_size; ++i){
	        	new_canditatos_cont_suporte[(*new_canditatos_cont_suporte_size)]=(Elo*)malloc(sizeof(Elo));
	            int index = 0;
	            bool newFlag = true;
	            while (newFlag ) {
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

        Elo **new_canditatos=(Elo**)malloc(sizeof(Elo*)*(*elo_int_x)*(*elo_int_x));

        geracao_de_candidatos << < blocks_per_row, block_size >> >
                                                   (new_canditatos, arrayMap, arrayMapSize, elo_x, elo_int_x, minimo_suporte);
        cudaDeviceSynchronize();

//        EloArray elo_teste_array;
//        elo_teste_array.elo=(Elo**)malloc(sizeof(Elo)*index_elo_put);
//        elo_teste_array.size=(int*)malloc(sizeof(int));
//        blocks_per_row = (index_elo_put / block_size) + (index_elo_put % block_size > 0 ? 1 : 0);
//        printf("Frequencia 1 Quantide de Blocos %d Total %d\n", blocks_per_row, index_elo_put);
//        frequencia_x1_paralela<<<block_size,block_size>>>(elo_teste_array,new_canditatos,index_elo_put);
//        cudaDeviceSynchronize();
//        for(int i =0;i<(*elo_teste_array.size);++i){
//          printf(" ID=%d ITEM=%s S=%d \n",i,elo_teste_array.elo[i]->ItemId,elo_teste_array.elo[i]->suporte);
//        }

        printf("%d \n",index_elo_put);
        Elo**  new_canditatos_cont_suporte = (Elo**)malloc(sizeof(Elo*)*index_elo_put);
        int *new_canditatos_cont_suporte_size=(int*)malloc(sizeof(int));
        (*new_canditatos_cont_suporte_size)=0;
        printf("Frequencia 1\n");
        frequencia_x1<<<1,1>>>(new_canditatos_cont_suporte,new_canditatos_cont_suporte_size,new_canditatos,index_elo_put);
        cudaDeviceSynchronize();
        for(int i =0;i<(*new_canditatos_cont_suporte_size);++i){
               printf(" ID=%d ITEM=%s S=%d \n",i,new_canditatos_cont_suporte[i]->ItemId,new_canditatos_cont_suporte[i]->suporte);
             }
//
//
        blocks_per_row = ((*new_canditatos_cont_suporte_size) / block_size) + ((*new_canditatos_cont_suporte_size) % block_size > 0 ? 1 : 0);
        printf("Frequancia 2 Quantidade de Blocos %d Total %d\n", blocks_per_row, (*new_canditatos_cont_suporte_size));
        frequencia_x2 << < blocks_per_row, block_size >> >
                                           (elo_k1, (*elo_curr), new_canditatos_cont_suporte, (*new_canditatos_cont_suporte_size), (*minimo_suporte));

        blocks_per_row = (index_elo_put / block_size) + (index_elo_put % block_size > 0 ? 1 : 0);
        printf("Frequancia 3 Quantidade de Blocos %d Total %d\n", blocks_per_row, index_elo_put);
        frequencia_x3 << < blocks_per_row, block_size >> >
                                           (new_canditatos_cont_suporte,(*new_canditatos_cont_suporte_size), new_canditatos,index_elo_put, elo_x, (*minimo_suporte));
        cudaDeviceSynchronize();
        freee<< < blocks_per_row, block_size >> >(new_canditatos,index_elo_put);
        cudaDeviceSynchronize();

        blocks_per_row = ((*new_canditatos_cont_suporte_size) / block_size) + ((*new_canditatos_cont_suporte_size) % block_size > 0 ? 1 : 0);
        freee<< < blocks_per_row, block_size >> >(new_canditatos_cont_suporte,(*new_canditatos_cont_suporte_size));
        cudaDeviceSynchronize();

        free(new_canditatos_cont_suporte_size);

        (*elo_curr)=(*elo_curr)+indexSetMap;
        (*elo_int_x) = indexEloFim;
        printf("Elo_x Restante %d\n",(*elo_int_x));
        if (indexEloFim > 0) {
        	   count = 0;
        	   indexEloFim = 0;
        	   index_elo_put = 0;
        	   indexSetMap = 0;
               (*new_canditatos_cont_suporte_size)=0;


        } else {
            flag = false;
        }
    }
}
