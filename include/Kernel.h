//
// Created by rafael on 20/08/18.
//

#ifndef FP_GROWTH_GPU_KERNEL_PFP_H
#define FP_GROWTH_GPU_KERNEL_PFP_H


#include <cuda_runtime_api.h>
#include <cstdio>
#include "cuda.h"
#include "cudaHeaders.h"
#include "PFPArray.h"

__global__ void pfp_growth(EloVector *Elo_k1,int *Elo_k1_size,ArrayMap *arrayMap, size_t arrayMapSize,Elo *elo ,int *elosize, int *minimo_suporte);


#endif //FP_GROWTH_GPU_KERNEL_PFP_H
