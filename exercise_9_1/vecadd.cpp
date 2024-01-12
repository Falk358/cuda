#include<iostream>
#include<Kokkos_Core.hpp>


int main(int argc, char* argv[]){
    Kokkos::initialize(argc, argv);
    int n = 1024;

    {
        Kokkos::View<double*> d_x("x", n), d_y("y", n), d_res("res", n);
        
        auto h_x = Kokkos::create_mirror_view(d_x);
        auto h_y = Kokkos::create_mirror_view(d_y);
        auto h_res = Kokkos::create_mirror_view(d_res);

        for (int i = 0; i < n; i++)
        {
            h_x(i) = 1.0;
            h_y(i) = 2.0;
            h_res(i) = 0.0;
        }

        Kokkos::deep_copy(d_x, h_x);
        Kokkos::deep_copy(d_y, h_y);
        Kokkos::deep_copy(d_res, h_res);
        
        Kokkos::parallel_for("vecadd", n, KOKKOS_LAMBDA(int i){
            d_res(i) = d_x(i) + d_y(i);
        });
        
        Kokkos::deep_copy(h_res, d_res);
        
        for (int i = 0; i < n; i++)
        {
            if (h_res(i) != 3.0)
                std::cout << "verification failed at index " << i << std::endl;
        }
        
        std::cout << "verification successful!" << std::endl;

        
    }




    Kokkos::finalize();

}