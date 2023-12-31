#include<Kokkos_Core.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <chrono>

using Real = double;
using View1D = Kokkos::View<Real*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;
View1D central_diff_1D(const View1D& r, Real dx){
    int n = r.size();
    View1D ar("ar",n);
    Kokkos::deep_copy(ar, 0.0);
    Real dx2=dx*dx;
    Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
    if (i > 0 && i < n - 1) {
        ar(i) = (r(i - 1) - 2.0 * r(i) + r(i + 1)) / dx2;
    }
    else {
        ar(i) = r(i);
    }
    });
    
    return ar;
}

View1D conjugate(View1D (*Ar)(const View1D&, Real), const View1D& b, View1D x0, Real dx, Real tol = 1e-6, int maxiter = 200) {
    int n = x0.size();
    View1D r0("r0", n), p_k("p_k", n), r_k("r_k", n), x_k("x_k", n), Ap("Ap", n), r_k1("r_k1", n), x_k1("x_k1", n), p_k1("p_k1", n);
    Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
        r0(i) = b(i) - Ar(x0, dx)(i);
    });
    Real r0_norm = 0.0;
    Kokkos::parallel_reduce(
    "InnerProduct",
    Kokkos::RangePolicy<>(0, n),
    KOKKOS_LAMBDA(int i, Real& result) {
        result += r0(i) * r0(i);
    },
    r0_norm // Initial value of the result
);
    r0_norm=Kokkos::sqrt(r0_norm);
    
    //Real r0_norm = Kokkos::sqrt(std::inner_product(r0.begin(), r0.end(), r0.begin(), 0.0));
    if (r0_norm < tol) {
        std::cout << "tol reached already" << std::endl;
        return x0;
    }

    Kokkos::deep_copy(p_k,r0);
    Kokkos::deep_copy(r_k,r0);
    Kokkos::deep_copy(x_k,x0);
    
    
    int j;
    for (j = 0; j < maxiter; j++) {
        Ap = Ar(p_k, dx);
        Real alpha_1=0.0;
        Kokkos::parallel_reduce("Alpha_1",Kokkos::RangePolicy<>(0, n),KOKKOS_LAMBDA(int i, Real& result) {
            result += r_k(i) * r_k(i);},alpha_1);
        
        Real alpha_2=0.0;
        Kokkos::parallel_reduce("Alpha_2",Kokkos::RangePolicy<>(0, n),KOKKOS_LAMBDA(int i, Real& result) {
            result += p_k(i) * Ap(i);},alpha_2);
        
        Real alpha=alpha_1/alpha_2;
        Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
            x_k1(i) = x_k(i) + alpha * p_k(i);r_k1(i) = r_k(i)-alpha * Ap(i);
        });
        
        Real r_k1_product=0.0;
        Kokkos::parallel_reduce("R_K_1_Product",Kokkos::RangePolicy<>(0, n),KOKKOS_LAMBDA(int i, Real& result) {
            result += r_k1(i) * r_k1(i);},r_k1_product);
        r_k1_product=Kokkos::sqrt(r_k1_product);

        if (r_k1_product < tol) {
            std::cout << "tolerance reached " << j << std::endl;
            break;
        }
        
        Real beta_1=0.0;
        Kokkos::parallel_reduce("Beta_1",Kokkos::RangePolicy<>(0, n),KOKKOS_LAMBDA(int i, Real& result) {
            result += r_k1(i) * r_k1(i);},beta_1);
        
        Real beta_2=0.0;
        Kokkos::parallel_reduce("Beta_2",Kokkos::RangePolicy<>(0, n),KOKKOS_LAMBDA(int i, Real& result) {
            result += r_k(i) * r_k(i);},beta_2);
        
        Real beta=beta_1/beta_2;
        
        Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
            p_k1(i) = r_k1(i) + beta * p_k(i);
        });
        
        Kokkos::deep_copy(x_k,x_k1);
        Kokkos::deep_copy(p_k,p_k1);
        Kokkos::deep_copy(r_k,r_k1);

    }

    if (j == maxiter) {
        std::cout << "Max iterations reached" << std::endl;
    }
    
    return x_k;
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc,argv);
    int N = 10001;
    Real dx = 1.0 / (N - 1);
    //std::vector<double> x(N), phi(N), b(N), sol(N);
    auto start_time = std::chrono::high_resolution_clock::now();
    
    Kokkos::View<double*> x("x", N);
    Kokkos::View<double*> phi("phi", N);
    Kokkos::View<double*> b("b", N);
    Kokkos::View<double*> sol("sol", N);
    Kokkos::parallel_for(N, KOKKOS_LAMBDA(const int i) {
        x(i) = i * dx;
        phi(i)=0.0;
    });

    phi(0) = 1.0;
    phi(N - 1) = 0.0;
    b(0) = phi(0);
    b(N - 1) = phi(N - 1);


    sol = conjugate(central_diff_1D, b, phi, dx);

    sol(0) = 1.0;
    for (int i = 0; i < N; ++i) {
        std::cout<<x(i)<<" "<<sol(i)<<std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time);

    std::cout << "Execution Time: " << duration.count() << " microseconds" << std::endl;

    Kokkos::finalize();
    
    return 0;
}
