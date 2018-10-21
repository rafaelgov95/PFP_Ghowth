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


#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include "PFPArray.h"

#ifndef PFP_GROWTH_PFPGROWTH_H
#define PFP_GROWTH_PFPGROWTH_H




class PFPGrowth {
    ArrayMap*  arrayMap;
    Elo*  eloPos;
public:
    PFPGrowth(ArrayMap *_arrayMap,Elo *_eloPos,size_t _arrayMapSize,size_t _eloPosSize, int _minimo_suporte);

};


#endif //PFP_GROWTH_PFPGROWTH_H
