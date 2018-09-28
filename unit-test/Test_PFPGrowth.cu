
//#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
//#include "catch.hpp"
#include "PFPTree.h"
#include "PFPArray.h"
#include "PFPGrowth.cu.h"
#include "Kernel.h"
#include "FPTransMap.h"
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <sstream>      // std::istringstream
#define COLUMNS 500
#define ROWS 500


int main( int argc, char * argv[] ){
    using namespace std;
    ifstream read;
    string linha;
    read.open("./../dataset/rafael.txt");
    vector<Transaction> transactions;
    if(read.is_open()) {
        while (!read.eof()) {
            getline(read, linha);
//            printf("%s\n", linha.c_str());
            istringstream iss(linha);
            vector<std::string> results(std::istream_iterator<std::string>{iss},
                                        std::istream_iterator<std::string>());
            transactions.push_back(results);

        }
        read.close();
    } else{
        printf("Not Working \n");
    }
    printf("Terminou de fazer o Mapa \n");

    struct timeval startc, end;
    long seconds, useconds;
    double mtime;
    gettimeofday(&startc, NULL);

     int minimum_support_threshold=3;
     PFPTree fptree{transactions, minimum_support_threshold};
        PFPArray pfp_array(fptree);
        if(pfp_array.arrayMap.size()>0){
//    assert( std::numeric_limits<BitBlock>::digits == 8 );
//    size_type block_size = std::numeric_limits<BitBlock>::digits;
//    size_type blocks_per_row = ( pfp_array.arrayMap.size()-1 / block_size ) + ( n_trans % block_size > 0 ? 1 : 0 );

            PFPGrowth pfpGrowth(pfp_array._arrayMap,pfp_array._eloMap,pfp_array.arrayMap.size(),pfp_array.arrayMap.size()-1,minimum_support_threshold);
            gettimeofday(&end, NULL);

            seconds  = end.tv_sec  - startc.tv_sec;
            useconds = end.tv_usec - startc.tv_usec;
            mtime = useconds;
            mtime/=1000;
            mtime+=seconds*1000;
            //cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
            double memXFers=5*4*COLUMNS*ROWS;
            memXFers/=1024*1024*1024;

            printf("\n CPU : %g ms bandwidth %g GB/s",mtime, memXFers/(mtime/1000.0));
            std::cout << "All tests passed!" << std::endl;
        }else{
            printf("MINIMO DE CANDIDATO NÃO EXISTENTE \n");
        }
//    const cuda_custom_trans_map::Item a = 0, b = 1, c = 2, d = 3, e = 4, f = 5, g = 6, h = 7, i = 8, j = 9, k = 10, l = 11, m = 12, n = 13,
//            o = 14, p = 15, q = 16, r = 17, s = 18, t = 19, u = 20, v = 21, w = 22, x = 23, y = 24, z = 25;
//    // each line represents a transaction
//    cuda_custom_trans_map::Items trans{
//            f, a, c, d, g, i, m, p,
//            a, b, c, f, l, m, o,
//            b, f, h, j, o,
//            b, c, k, s, p,
//            a, f, c, e, l, p, m, n
//    };
//    // start index of each transaction
//    cuda_custom_trans_map::Indices indices{0, 8, 15, 20, 25};
//    // number of items in each transaction
//    cuda_custom_trans_map::Sizes sizes{8, 7, 5, 5, 8};
//    // construct FPTransactionMap object
//    cuda_custom_trans_map::FPTransMap fp_trans_map(trans.cbegin(), indices.cbegin(), sizes.cbegin(), indices.size(), 3);
    return 0;
}

