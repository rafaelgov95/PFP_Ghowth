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

#ifndef PFPTREE_HPP
#define PFPTREE_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <utility>
#include <vector>
#include <thrust/device_vector.h>
using cuda_int = int;
using cuda_uint = unsigned int;
using cuda_real = float;

using Item = std::string;
using Items = std::vector<Item>;
using DItems = thrust::device_vector<Item>;
using Transaction = std::vector<Item>;
using TransformedPrefixPath = std::pair<std::vector<Item>, uint64_t>;
using Pattern = std::pair<std::set<Item>, uint64_t>;

struct PFPNode {
    const Item item;
    bool is_visit ;
    int frequency;
    std::shared_ptr<PFPNode> parent;
    std::vector<std::shared_ptr<PFPNode>> children;
    PFPNode(const Item&, const std::shared_ptr<PFPNode>&);
};


struct PFPLeaf{
    std::shared_ptr<PFPNode> value;
    std::shared_ptr<PFPLeaf> next;
    PFPLeaf(const std::shared_ptr<PFPNode> value);
    PFPLeaf();
};


struct PFPTree {
    std::shared_ptr<PFPNode> root;
    std::shared_ptr<PFPLeaf> rootFolhas;
    int minimum_support_threshold;

    PFPTree(const std::vector<Transaction>&, int);

};


#endif  // PFPTREE_HPP
