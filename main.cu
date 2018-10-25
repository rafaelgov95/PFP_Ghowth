#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "include/PFPArray.cu.h"
#include "include/PFPTree.h"
#include "include/PFPGrowth.cu.h"

#define COLUMNS 500
#define ROWS 500
int main( int argc, char * argv[] ){
    using namespace std;
    ifstream read;
    string linha;
    read.open("./dataset/chess.data");
    vector<Transaction> transactions;
    if(read.is_open()) {
        while (!read.eof()) {
            getline(read, linha);
            istringstream iss(linha);
            vector<string> results(std::istream_iterator<std::string>{iss},
                                        std::istream_iterator<std::string>());
            for(auto const& value: results) {
         	   printf("%s ",value.c_str());
            }
      	    printf("\n");

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
     printf("Terminou de fazer o FPTree \n");
     PFPArray pfp_array(fptree);
     printf("Terminou de fazer o PFPArray \n");
    if(pfp_array.arrayMap.size()>0){
            PFPGrowth pfpGrowth(pfp_array._arrayMap,pfp_array._eloMap,pfp_array.arrayMap.size(),pfp_array.arrayMap.size()-1,minimum_support_threshold);
            gettimeofday(&end, NULL);

            seconds  = end.tv_sec  - startc.tv_sec;
            useconds = end.tv_usec - startc.tv_usec;
            mtime = useconds;
            mtime/=1000;
            mtime+=seconds*1000;
            double memXFers=5*4*COLUMNS*ROWS;
            memXFers/=1024*1024*1024;

            printf("\n CPU : %g ms bandwidth %g GB/s",mtime, memXFers/(mtime/1000.0));
            std::cout << "All tests passed!" << std::endl;
        }else{
            printf("MINIMO DE CANDIDATO NÃO EXISTENTE \n");
        }

    return 0;
}
