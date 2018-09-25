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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>
#include <iostream>
#include "PFPTree.h"


struct frequency_comparator {
    bool operator()(const std::pair<Item, int> &lhs, const std::pair<Item, int> &rhs) const {
        return lhs.second > rhs.second || lhs.second == rhs.second && lhs.first < rhs.first;
    }
};

PFPLeaf::PFPLeaf(const std::shared_ptr<PFPNode> &) :
        value(value), next(next) {
}

PFPNode::PFPNode(const Item &item, const std::shared_ptr<PFPNode> &parent) :
        item(item), frequency(1), parent(parent), children(children), is_visit(false) {
}

PFPTree::PFPTree(const std::vector<Transaction> &transactions, int minimum_support_threshold) :
        root(std::make_shared<PFPNode>("{}", nullptr)),
        minimum_support_threshold(minimum_support_threshold),
        rootFolhas(std::make_shared<PFPLeaf>(nullptr)) {

    std::map<Item, int> frequency_by_item;
    for (const Transaction &transaction : transactions) {
        for (const Item &item : transaction) {
            ++frequency_by_item[item];
        }
    }

    for (auto it = frequency_by_item.cbegin(); it != frequency_by_item.cend();) {
        const int item_frequency = (*it).second;
        if (item_frequency < minimum_support_threshold) { frequency_by_item.erase(it++); }
        else { ++it; }
    }




//    std::set<std::pair<Item, int>> items_ordered_by_frequency(frequency_by_item.cbegin(), frequency_by_item.cend());


    // um problema esta aqui.
//        std::set<std::pair<Item, int>, frequency_comparator> items_ordered_by_frequency(frequency_by_item.cbegin(), frequency_by_item.cend());


    //Apelacao

//    std::vector<std::pair<Item, uint64_t>> items_ordered_by_frequency;
//    std::pair<Item, uint64_t> a = std::make_pair(("F"), uint64_t(4));
//    std::pair<Item, uint64_t> b = std::make_pair("C", uint64_t(4));
//    std::pair<Item, uint64_t> c = std::make_pair("A", uint64_t(3));
//    std::pair<Item, uint64_t> d = std::make_pair("B", uint64_t(3));
//    std::pair<Item, uint64_t> e = std::make_pair("M", uint64_t(3));
//    std::pair<Item, uint64_t> f = std::make_pair("P", uint64_t(3));
//
//    items_ordered_by_frequency.push_back(a);
//    items_ordered_by_frequency.push_back(b);
//    items_ordered_by_frequency.push_back(c);
//    items_ordered_by_frequency.push_back(d);
//    items_ordered_by_frequency.push_back(e);
//    items_ordered_by_frequency.push_back(f);

    auto curr_rootFolhas = rootFolhas;
//
    curr_rootFolhas.get()->next = std::make_shared<PFPLeaf>(nullptr);
//
//    for (const Transaction &transaction : transactions) {
//        auto curr_fpnode = root;
//
//        for (const auto &pair : items_ordered_by_frequency) {
//            const Item &item = pair.first;
//
//            if (std::find(transaction.cbegin(), transaction.cend(), item) != transaction.cend()) {
//
//                 auto it = std::find_if(
//                        curr_fpnode->children.cbegin(), curr_fpnode->children.cend(),
//                        [item](const std::shared_ptr<PFPNode> &fpnode) {
//                            return fpnode->item == item;
//                        });
//                if (it == curr_fpnode->children.cend()) {
//                    auto curr_fpnode_new_child = std::make_shared<PFPNode>(item, curr_fpnode);
//                    curr_fpnode->children.push_back(curr_fpnode_new_child);
//                    curr_fpnode = curr_fpnode_new_child;
//                    curr_rootFolhas.get()->value = curr_fpnode;
//                } else {
//                    auto curr_fpnode_child = *it;
//                    ++curr_fpnode_child->frequency;
//                    curr_fpnode = curr_fpnode_child;
//                }
//            }
//
//        }
//
//        //corrigir criando atua no final
//        curr_rootFolhas.get()->next = std::make_shared<PFPLeaf>(nullptr);
//        curr_rootFolhas = curr_rootFolhas.get()->next;
//
//    }
}

