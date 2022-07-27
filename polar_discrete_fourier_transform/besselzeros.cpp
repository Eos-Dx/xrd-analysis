/*
 * Code to compute Jn zeros
 * Adapted from: https://www.boost.org/doc/libs/1_79_0/libs/math/doc/html/math_toolkit/bessel/bessel_root.html
 */

#include <iostream>
#include <fstream>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/special_functions/bessel.hpp>

#define OUTFILEPATH ""


int main()
{

    // for single zeros

    int max_n = 2; // Maximum Jn order
    unsigned zero_count = 10; // The 1-indexed zeros to compute
    std::vector<double> jn_zeros; // Vector to store Jn zeros


    try {
		// Loop over Jn orders and number of zeros we are interested in
        for (int n = 0; n < max_n; n++) {
            for (int z = 0; z < zero_count; z++) {
				// Use boost library function to calculate Jn zero
                double root = boost::math::cyl_bessel_j_zero(n*1.0, z+1);
				// Store results in the vector
                jn_zeros.push_back(root);
            }
        }

        // Save the vector to file
        std::ofstream outfile(OUTFILEPATH, std::ofstream::binary);
        std::ostream_iterator<double> output_iterator(outfile, "\n");
        std::copy(jn_zeros.begin(), jn_zeros.end(), output_iterator);

    }
    catch (std::exception& ex)
    {
        std::cout << "Thrown exception " << ex.what() << std::endl;
    }

    return 0;
}
