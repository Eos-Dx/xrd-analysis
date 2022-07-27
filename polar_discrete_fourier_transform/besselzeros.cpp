//
// Code to compute Jn zeros
// Adapted from: https://www.boost.org/doc/libs/1_79_0/libs/math/doc/html/math_toolkit/bessel/bessel_root.html
// 5001 x 5001 in research paper
//
// Want: bessel_j_zero_array(float [], int [])
//
// template <class T, class OutputIterator>
// OutputIterator cyl_bessel_j_zero(
//                      T v,                       // Floating-point value for Jv.
//                      int start_index,           // 1-based index of first zero.
//                      unsigned number_of_zeros,  // How many zeros to generate.
//                      OutputIterator out_it);    // Destination for zeros.
//

#include <iostream>
#include <fstream>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/special_functions/bessel.hpp>

#define OUTFILEPATH ""

/*
 * TODO: Clean this up for use with getting an array of zeros
template <class T>
struct output_iterator
{
    output_iterator(T* p) : q(p)
    {}
    output_iterator& operator*()
    { return *this; }
    output_iterator& operator++()
    { return *this; }
    output_iterator& operator++(int)
    { return *this; }
    output_iterator& operator=(T const& val)
    { return *this; }

    private:
       T* q;
};
*/

void saveVector(std::string path, const std::vector<std::vector<int> >& myVector)
{
    std::ofstream FILE(path, std::ios::out | std::ofstream::binary);

    // Store size of the outer vector
    int s1 = myVector.size();
    FILE.write(reinterpret_cast<const char *>(&s1), sizeof(s1));

    // Now write each vector one by one
    for (auto& v : myVector) {
        // Store its size
        int size = v.size();
        FILE.write(reinterpret_cast<const char *>(&size), sizeof(size));

        // Store its contents
        FILE.write(reinterpret_cast<const char *>(&v[0]), v.size()*sizeof(float));
    }
    FILE.close();
}


int main()
{

    /*
    // for multiple zeros

    float v = 0.0;
    int start_index = 1;
    unsigned number_of_zeros = 1;
    std::vector<double> root_vec;

//     std::size_t sz = sizeof(v);
//     std::vector<std::uint8_t> root_iter(sz);
//     std::vector<std::uint8_t>::const_iterator b = root_iter.begin();
//     std::vector<std::uint8_t>::const_iterator e = root_iter.end();
    output_iterator<double> root_iter(&root_vec);


    try {
        boost::math::cyl_bessel_j_zero(v, start_index, number_of_zeros, root_iter);
    }
    catch (std::exception& ex)
    {
        std::cout << "Thrown exception " << ex.what() << std::endl;
    }
    */

    // for single zeros

    int max_n = 2;
    unsigned zero_count = 10; // The 1-indexed zeros to compute
    // Initialize vector of size max_n by zero_count
    std::vector<double> jn_zeros;


    try {
        for (int n = 0; n < max_n; n++) {
            for (int z = 0; z < zero_count; z++) {
                double root = boost::math::cyl_bessel_j_zero(n*1.0, z+1);
                // Displaying with default precision of 6 decimal digits:
                // std::cout << n << ", " << z << ": " << root << std::endl;
                jn_zeros.push_back(root);
            }
        }
        /*
        // And with all the guaranteed (15) digits:
        std::cout.precision(std::numeric_limits<double>::digits10);
        std::cout << "boost::math::cyl_bessel_j_zero(0.0, 1) " << root << std::endl; // 2.40482555769577
        */

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
