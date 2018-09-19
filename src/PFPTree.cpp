#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>
#include <iostream>
#include "PFPTree.h"

PFPLeaf::PFPLeaf(const std::shared_ptr<PFPNode> &) :
        value(value), next(next) {
}

PFPNode::PFPNode(const Item &item, const std::shared_ptr<PFPNode> &parent) :
        item(item), frequency(1), parent(parent), children(children), is_visit(false) {
}

PFPTree::PFPTree(const std::vector<Transaction> &transactions, uint64_t minimum_support_threshold) :
        root(std::make_shared<PFPNode>("{}", nullptr)),
        minimum_support_threshold(minimum_support_threshold),
        rootFolhas(std::make_shared<PFPLeaf>(nullptr)) {

    std::map<Item, uint64_t> frequency_by_item;
    for (const Transaction &transaction : transactions) {
        for (const Item &item : transaction) {
            ++frequency_by_item[item];
        }
    }

    // keep only items which have a frequency greater or equal than the minimum support threshold
    for (auto it = frequency_by_item.cbegin(); it != frequency_by_item.cend();) {
        const uint64_t item_frequency = (*it).second;
        if (item_frequency < minimum_support_threshold) { frequency_by_item.erase(it++); }
        else { ++it; }
    }


    struct frequency_comparator {
        bool operator()(const std::pair<Item, uint64_t> &lhs, const std::pair<Item, uint64_t> &rhs) const {
            return lhs.second > rhs.second || lhs.second == rhs.second && lhs.first < rhs.first;
        }
    };
    std::set<std::pair<Item, uint64_t>, frequency_comparator> items_ordered_by_frequency(frequency_by_item.cbegin(), frequency_by_item.cend());


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
    // scan the transactions again
    curr_rootFolhas.get()->next = std::make_shared<PFPLeaf>(nullptr);
    for (const Transaction &transaction : transactions) {
        auto curr_fpnode = root;

        // select and sort the frequent items in transaction according to the order of items_ordered_by_frequency
        for (const auto &pair : items_ordered_by_frequency) {
            const Item &item = pair.first;

//            printf("%s",item.c_str());
            // check if item is contained in the current transaction
            if (std::find(transaction.cbegin(), transaction.cend(), item) != transaction.cend()) {
                // insert item in the tree

                // check if curr_fpnode has a child curr_fpnode_child such that curr_fpnode_child.item = item
                const auto it = std::find_if(
                        curr_fpnode->children.cbegin(), curr_fpnode->children.cend(),
                        [item](const std::shared_ptr<PFPNode> &fpnode) {
                            return fpnode->item == item;
                        });
                if (it == curr_fpnode->children.cend()) {
                    // the child doesn't exist, create a new node
                    const auto curr_fpnode_new_child = std::make_shared<PFPNode>(item, curr_fpnode);
                    curr_fpnode->children.push_back(curr_fpnode_new_child);
                    curr_fpnode = curr_fpnode_new_child;
                    curr_rootFolhas.get()->value = curr_fpnode;
                } else {
                    // the child exist, increment its frequency
                    auto curr_fpnode_child = *it;
                    ++curr_fpnode_child->frequency;
                    curr_fpnode = curr_fpnode_child;
                }
            }

        }

        //corrigir criando atua no final
        curr_rootFolhas.get()->next = std::make_shared<PFPLeaf>(nullptr);
        curr_rootFolhas = curr_rootFolhas.get()->next;

    }
}

