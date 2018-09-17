#include <algorithm>
#include <cassert>
#include <iostream>
#include <cstdint>
#include <utility>
#include <cublas_v2.h>
#include <PFPArray.h>
#include "PFPArray.h"
#include "Kernel.h"


void cstringcpy(char *src, char * dest)
{
    while (*src) {
        *(dest++) = *(src++);
    }
    *dest = '\0';
}

PFPArrayMap::PFPArrayMap(PFPNode *i, const int p, const int s) : ItemId(i), indexP(p), suporte(s) {
}


PFPArray::PFPArray(const PFPTree &fptree) {
    fptree.root.get()->frequency = -1;
    create_array_and_elepos(fptree);
    eloPosStapOne();
}


void PFPArray::eloPosStapOne() {
    size_t size = sizeof(Elo) * (arrayMap.size()-1);
    _eloMap = (Elo *) malloc(size);
    int b =0;
    for (int i=0; i < arrayMap.size();i++) {
        if(_arrayMap[i].indexP > -1) {
            strcpy(_eloMap[b].ItemId,_arrayMap[i].ItemId);
            _eloMap[b].indexArrayMap = i;
            _eloMap[b].suporte = _arrayMap[i].suporte;
            b++;
        }
    }
}


int PFPArray::recur_is_parent_array(PFPNode *current_leaf) {
    auto k = std::find_if(hashMap.begin(), hashMap.end(), [&](std::pair<PFPArrayMap, int> const &ref) {
        return ref.first.ItemId == current_leaf;
    });
    if (k != hashMap.cend()) {
        return k->second;
    } else {

            int parent_pos =(int)arrayMap.size() + 1;
            return parent_pos;

    }
}


void PFPArray::create_array_and_elepos(const PFPTree &fptree) {
    auto rootFolha = fptree.rootFolhas;
    while (rootFolha) {
        auto current_leaf = rootFolha.get()->value;
        bool returnn = false;
        while (current_leaf.get()) {
            auto bb = std::find_if(hashMap.begin(), hashMap.end(), [&](std::pair<PFPArrayMap, uint64_t> const &ref) {
                return ref.first.ItemId == current_leaf.get();
            });
            if (bb == hashMap.cend() && !current_leaf.get()->is_visit) {
                int parent_pos = 0;
                if (current_leaf.get()->parent) {
                    parent_pos = recur_is_parent_array(current_leaf.get()->parent.get());
                }else{
                    parent_pos =-1;
                }
                current_leaf.get()->is_visit = true;
                const PFPArrayMap a(current_leaf.get(), parent_pos, current_leaf.get()->frequency);
                arrayMap.push_back(a);
                size_t pos = arrayMap.size() - 1;
                hashMap.push_back(std::make_pair(a, int(pos)));
            } else {
                if (current_leaf.get()->is_visit && !current_leaf.get()->children.empty()) {
                    returnn = true;
                } else {
                    auto Id_parent = recur_is_parent_array(current_leaf.get()->parent.get());
                    current_leaf.get()->is_visit = true;
                    const PFPArrayMap a(current_leaf.get(), Id_parent, current_leaf.get()->frequency);
                    arrayMap.push_back(a);
                }
            }
            if (!returnn) {
                current_leaf = current_leaf.get()->parent;
            } else {
                current_leaf = nullptr;
            }

        }
        rootFolha = rootFolha.get()->next;
    }
    size_t size = sizeof(ArrayMap) * arrayMap.size();
    _arrayMap = (ArrayMap *) malloc(size);
    for (auto it = arrayMap.begin(); it != arrayMap.end(); ++it) {
        long index = std::distance(arrayMap.begin(), it);
        _arrayMap[index].suporte = (*it).suporte;
        _arrayMap[index].indexP = (*it).indexP;
        std::strcpy(_arrayMap[index].ItemId, (*it).ItemId->item.c_str());
    }
}




