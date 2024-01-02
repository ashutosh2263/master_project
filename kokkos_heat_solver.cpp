#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <chrono>
#include <Kokkos_Core.hpp>

//Creating view type in default Kokkos memory space
using view_type = Kokkos::View<double*>;

//Creating mirror of view type residing in host memory space
using host_view_type =  view_type::HostMirror;


view_type central_diff_1D(const view_type& r, double dx) {
    int n = r.extent(0);
    view_type ar("ar", n);
    Kokkos::View<double> dx2("dx2");
    Kokkos::View<double>::HostMirror h_dx2 = Kokkos::create_mirror_view(dx2);
    h_dx2() = dx*dx;
    Kokkos::deep_copy(dx2, h_dx2);

    Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
        if (i > 0 && i < n - 1) {
            ar(i) = (r(i - 1) - 2.0 * r(i) + r(i + 1))/dx2();
        }
        else {
            ar(i) = r(i);
        }
    });
    return ar;
}

void conjugate(view_type& sol, view_type(*Ar)(const view_type&, double), const view_type& b, view_type x0, double dx, double tol = 1e-6, int maxiter = 200) {
    int n = x0.extent(0);
    view_type r0("r0", n), p_k("p_k", n), r_k("r_k", n), x_k("x_k", n), Ap("Ap", n), r_k1("r_k1", n), x_k1("x_k1", n), p_k1("p_k1", n);
    Ap = Ar(x0,dx);
    Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
        r0(i) = b(i) - Ap(i);
    });
    double r0_norm = 0.0;
    Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(const int i, double& update) {
        update += r0(i) * r0(i);
    }, r0_norm);

    if (std::sqrt(r0_norm) < tol) {
        std::cout << "tol reached already" << std::endl;
        sol = x0;
    }

    p_k = r0;
    r_k = r0;
    x_k = x0;

    int j;
    for (j = 0; j < maxiter; j++) {
        Ap = Ar(p_k, dx);
        double alpha_num = 0.0;
        double alpha_denom = 0.0;
        Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(const int i, double& update_num, double& update_denom) {
            update_num += r_k(i) * r_k(i);
            update_denom += p_k(i) * Ap(i);
        }, alpha_num, alpha_denom);
        double alpha = alpha_num / alpha_denom;
        double beta_denom = alpha_num;
        Kokkos::View<double> d_alpha("d_alpha");
        Kokkos::View<double>::HostMirror h_alpha = Kokkos::create_mirror_view(d_alpha);
        h_alpha() = alpha;
        Kokkos::deep_copy(d_alpha, h_alpha);
    
        Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
            x_k1(i) = x_k(i) + d_alpha() * p_k(i);
            r_k1(i) = r_k(i) - d_alpha() * Ap(i);
        });

        double r_k1_norm = 0.0;
        Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(const int i, double& update) {
            update += r_k1(i) * r_k1(i);
        }, r_k1_norm);

        if (std::sqrt(r_k1_norm) < tol) {
            std::cout << "tolerance reached " << j << std::endl;
            sol = x_k;
            return;
        }
        double beta_num = 0.0;
        Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(const int i, double& update_num) {
            update_num += r_k1(i) * r_k1(i);
        }, beta_num);
        double beta = beta_num / beta_denom;
        Kokkos::View<double> d_beta("beta");
        Kokkos::View<double>::HostMirror h_beta = Kokkos::create_mirror_view(d_beta);
        h_beta() = beta;
        Kokkos::deep_copy(d_beta, h_beta);
        Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
            p_k1(i) = r_k1(i) + d_beta() * p_k(i);
        });
        x_k = x_k1;
        p_k = p_k1;
        r_k = r_k1;
    }

    if (j == maxiter) {
        std::cout << "Max iterations reached\n"<< std::endl;
    }
    sol = x_k;
}

int main() {
    Kokkos::initialize();
    {
        Kokkos::Timer timer;
        int N = 10001;
        double dx = 1.0 / (N - 1);
        //Creating views in default memory space
        view_type x("x", N), phi("phi", N), b("b", N), sol("sol", N);

        //Create mirror view in host memory space
        host_view_type h_x = Kokkos::create_mirror_view(x);
        host_view_type h_phi = Kokkos::create_mirror_view(phi);
        host_view_type h_b = Kokkos::create_mirror_view(b);
        host_view_type h_sol = Kokkos::create_mirror_view(sol);
       
       //Initializing points in default memory space
        Kokkos::parallel_for(N, KOKKOS_LAMBDA(const int i) {
            x(i) = i * dx;
        });

        //Initializing values in host memory space, and then deep copying to default memory space
        h_phi(0) = 1.0;
        h_phi(N - 1) = 0.0;
        h_b(0) = 1.0;
        h_b(N - 1) = 0.0;
        Kokkos::deep_copy(phi, h_phi);
        Kokkos::deep_copy(b, h_b);
        conjugate(sol, central_diff_1D, b, phi, dx);

        //Deep copying from default memory space to host memory space
        Kokkos::deep_copy(h_sol, sol);
        Kokkos::deep_copy(h_x, x);
        h_sol(0) = 1.0;
        // for (int i = 0; i < N; ++i) {
        //     std::cout<<h_x(i)<<" "<<h_sol(i)<<std::endl;
        // }
        double time = timer.seconds();
        std::cout << "Execution Time: " << time << " seconds" << std::endl;
        }
    Kokkos::finalize();
}
