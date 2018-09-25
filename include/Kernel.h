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

#ifndef FP_GROWTH_GPU_KERNEL_PFP_H
#define FP_GROWTH_GPU_KERNEL_PFP_H


#include <cuda_runtime_api.h>
#include <cstdio>
#include "cuda.h"
#include "cudaHeaders.h"
#include "PFPArray.h"

__global__ void pfp_growth(EloVector *Elo_k1,int *Elo_k1_size,ArrayMap *arrayMap, size_t arrayMapSize,Elo *elo ,int *elosize, int *minimo_suporte);


#endif //FP_GROWTH_GPU_KERNEL_PFP_H
