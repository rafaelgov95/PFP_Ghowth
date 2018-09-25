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

#ifndef PFP_ARRAY_HP
#define PFP_ARRAY_HP

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <utility>
#include "PFPTree.h"

#include <iostream>
#include <thrust/device_vector.h>
#define MAX_STR_SIZE 32


struct PFPArrayMap {
    PFPNode *ItemId;
    int indexP;
    int suporte;
    PFPArrayMap(PFPNode *, const int);
    PFPArrayMap(PFPNode *, const int, const int);

};
typedef struct {
     char ItemId[MAX_STR_SIZE];
    cuda_int indexArrayMap;
    cuda_int suporte;
} Elo;

typedef struct {
     char ItemId[MAX_STR_SIZE];
    cuda_int indexP;
    cuda_int suporte;
} ArrayMap;

typedef struct {
    Elo *elo;
    size_t size;
} EloMap;


typedef struct {
    EloMap *eloMap;
    cuda_int size;
} EloGrid;

typedef struct{
    Elo *eloArray;
    int size;
}EloVector;

typedef struct {
    Elo elo;
    int size;
}SetMap;

using HashMap = std::vector<std::pair<PFPArrayMap, int >>;

class PFPArray {
public:
    PFPArray(const PFPTree &fptree);
    HashMap hashMap;
    ArrayMap*  _arrayMap;
    Elo*  _eloMap;
    std::vector<PFPArrayMap> arrayMap;
private:

    void eloPosStapOne();

    int recur_is_parent_array(PFPNode *a);

    void create_array_and_elepos(const PFPTree &fptree);

};

#endif  // PFP_ARRAY_HP
