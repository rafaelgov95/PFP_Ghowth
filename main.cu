



#ifdef UNIT_TEST
#include "unit-test/Test_PFPGrowth.cu"


//

#else
//

//#include <FPRadixTree.h>
//#include "FPTransMap.h"
//int  main() {
//    const cuda_fp_growth::Item a = 0, b = 1, c = 2, d = 3, e = 4, f = 5, g = 6, h = 7, i = 8, j = 9, k = 10, l = 11, m = 12, n = 13,
//            o = 14, p = 15, q = 16, r = 17, s = 18, t = 19, u = 20, v = 21, w = 22, x = 23, y = 24, z = 25;
//
//    // each line represents a transaction
//    cuda_fp_growth::Items trans{
//            f, a, c, d, g, i, m, p,
//            a, b, c, f, l, m, o,
//            b, f, h, j, o,
//            b, c, k, s, p,
//            a, f, c, e, l, p, m, n
//    };
//
//    // start index of each transaction
//    cuda_fp_growth::Indices indices{0, 8, 15, 20, 25};
//
//    // number of items in each transaction
//    cuda_fp_growth::Sizes sizes{8, 7, 5, 5, 8};
//
//    // construct FPTransactionMap object
//    cuda_fp_growth::FPTransMap fp_trans_map(trans.cbegin(), indices.cbegin(), sizes.cbegin(), indices.size(), 3);
//    cuda_fp_growth::FPRadixTree fp_radix_tree(fp_trans_map);
//    return 0;
}
//extern "C"
//{
//}

#endif  // UNIT_TEST