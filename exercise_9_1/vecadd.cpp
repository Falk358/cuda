#include<iostream>
#include<Kokkos_Core.hpp>


int main(int argc, char* argv[]){
    Kokkos::initialize(argc, argv);
    int n = 1024;

    {
        Kokkos::View<double***, Kokkos::LayoutLeft> d_left_x("x", n, n, n), d_left_y("y", n, n, n), d_left_res("res", n, n, n);
        Kokkos::View<double***, Kokkos::LayoutRight> d_right_x("x", n, n, n), d_right_y("y", n, n, n), d_right_res("res", n, n, n);
        
        
        auto h_left_x = Kokkos::create_mirror_view(d_left_x);
        auto h_left_y = Kokkos::create_mirror_view(d_left_y);
        auto h_left_res = Kokkos::create_mirror_view(d_left_res);

        auto h_right_x = Kokkos::create_mirror_view(d_right_x);
        auto h_right_y = Kokkos::create_mirror_view(d_right_y);
        auto h_right_res = Kokkos::create_mirror_view(d_right_res);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    h_left_x(i,j,k) = 1.0;
                    h_left_y(i,j,k) = 2.0;
                    h_left_res(i,j,k) = 0.0;
                }
            }
        }
        
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    h_right_x(i,j,k) = 1.0;
                    h_right_y(i,j,k) = 2.0;
                    h_right_res(i,j,k) = 0.0;
                }
            }
        }

        Kokkos::deep_copy(d_left_x, h_left_x);
        Kokkos::deep_copy(d_left_y, h_left_y);
        Kokkos::deep_copy(d_left_res, h_left_res);
        
        Kokkos::deep_copy(d_right_x, h_right_x);
        Kokkos::deep_copy(d_right_y, h_right_y);
        Kokkos::deep_copy(d_right_res, h_right_res);

        Kokkos::parallel_for("vecadd", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{n,n,n}), KOKKOS_LAMBDA(int i,int j,int k){
            d_left_res(i,j,k) = d_left_x(i,j,k) + d_left_y(i,j,k);
        });
        
        Kokkos::parallel_for("vecadd", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {n,n,n}), KOKKOS_LAMBDA(int i,int j,int k){
            d_right_res(i,j,k) = d_right_x(i,j,k) + d_right_y(i,j,k);
        });
        
        Kokkos::deep_copy(h_left_res, d_left_res);
        Kokkos::deep_copy(h_right_res, d_right_res);
        
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    if (h_left_res(i,j,k) != 3.0)
                        std::cout << "verification failed at index " << i << ", " << j << ", " << k << std::endl;
                }
            }
        }
        
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    if (h_right_res(i,j,k) != 3.0)
                        std::cout << "verification failed at index " << i << ", " << j << ", " << k << std::endl;
                }
            }
        }

        std::cout << "verification successful!" << std::endl;

        
    }




    Kokkos::finalize();

}