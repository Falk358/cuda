
#include <iostream>
#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>


int main()
{
    
    Kokkos::initialize();
    
    {
        int m = 256;

        int p = 256;
        Kokkos::View<double**, Kokkos::LayoutLeft> d_matrix("m",m, m);
        Kokkos::View<double*, Kokkos::LayoutLeft> d_vector_in("i", p), d_vector_out("o",p), d_vector_out_gemv("g",p);
        
        auto h_matrix = Kokkos::create_mirror_view(d_matrix);
        auto h_vector_in = Kokkos::create_mirror_view(d_vector_in);
        auto h_vector_out = Kokkos::create_mirror_view(d_vector_out);
        auto h_vector_out_gemv = Kokkos::create_mirror_view(d_vector_out_gemv);
        for (int i = 0; i < p; i++) 
        {
            h_vector_in(i) = 1.0;
        }

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < m; j++)
            {
                h_matrix(i,j) = 3.0;
            }
        }
            
            Kokkos::deep_copy(d_vector_in, h_vector_in);
            Kokkos::deep_copy(d_matrix, h_matrix);
            
            Kokkos::parallel_for("MatVecMult_Vanilla", m, KOKKOS_LAMBDA(const int i) {
                    double sum = 0.0;
                    for (int j = 0; j < p; ++j) 
                    {
                        sum += d_matrix(i, j) * d_vector_in(j);
                    }
                    d_vector_out(i) = sum;
        });
        
        KokkosBlas::gemv("N",1.0, d_matrix, d_vector_in, 0.0, d_vector_out_gemv);
        
        Kokkos::deep_copy(h_vector_out, d_vector_out);
        Kokkos::deep_copy(h_vector_out_gemv, d_vector_out_gemv);
        double correct_res = 3.0 * double(p);
        for (int i = 0; i < p; i++) 
        {
            if (h_vector_out(i) != correct_res)
                std::cerr << "Error! vector out of vanilla computation is incorrect at index " << i << ": res is " <<  h_vector_out(i) << std::endl;
        }
        for (int i = 0; i < p; i++) 
        {
            if (h_vector_out_gemv(i) != correct_res)
                std::cerr << "Error! vector out of gemv computation is incorrect at index " << i << ": res is " <<  h_vector_out_gemv(i) << std::endl;
        }
    }
    Kokkos::finalize();
}