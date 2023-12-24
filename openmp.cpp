#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <chrono>

using Real = double;
using View1D = std::vector<Real>;

View1D central_diff_1D(const View1D& r, Real dx) {
    int n = r.size();
    View1D ar(n, 0.0);
    Real dx2 = dx * dx;

    #pragma omp parallel for
    for (int i = 1; i < n - 1; ++i) {
        ar[i] = (r[i - 1] - 2.0 * r[i] + r[i + 1]) / dx2;
    }

        

    return ar;
}

View1D conjugate(View1D (*Ar)(const View1D&, Real), const View1D& b, View1D x0, Real dx, Real tol = 1e-6, int maxiter = 200) {
    int n = x0.size();
    View1D r0(n), p_k(n), r_k(n), x_k(n), Ap(n), r_k1(n), x_k1(n), p_k1(n);
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        r0[i] = b[i] - Ar(x0,dx)[i];
    }

    Real r0_norm = sqrt(std::inner_product(r0.begin(), r0.end(), r0.begin(), 0.0));
    if (r0_norm < tol) {
        std::cout << "tol reached already" << std::endl;
        return x0;
    }

    
        p_k = r0;
        r_k = r0;
        x_k = x0;
    

    int j;
    for (j = 0; j < maxiter; j++) {
    	View1D Ap_k=Ar(p_k,dx);
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            Ap[i] = Ap_k[i];
        }

        Real alpha_1 = std::inner_product(r_k.begin(), r_k.end(), r_k.begin(), 0.0);
        Real alpha_2 = std::inner_product(p_k.begin(), p_k.end(), Ap.begin(), 0.0);
        Real alpha = alpha_1 / alpha_2;

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            x_k1[i] = x_k[i] + alpha * p_k[i];
            r_k1[i] = r_k[i] - alpha * Ap[i];
        }

        Real r_k1_product = sqrt(std::inner_product(r_k1.begin(), r_k1.end(), r_k1.begin(), 0.0));

        if (r_k1_product < tol) {
            std::cout << "tolerance reached " << j << std::endl;
            break;
        }

        Real beta_1 = std::inner_product(r_k1.begin(), r_k1.end(), r_k1.begin(), 0.0);
        Real beta_2 = std::inner_product(r_k.begin(), r_k.end(), r_k.begin(), 0.0);
        Real beta = beta_1 / beta_2;

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            p_k1[i] = r_k1[i] + beta * p_k[i];
        }

        
            x_k = x_k1;
            p_k = p_k1;
            r_k = r_k1;
        
    }

    if (j == maxiter) {
        std::cout << "Max iterations reached" << std::endl;
    }

    return x_k;
}

int main() {
    int N = 10001;
    Real dx = 1.0 / (N - 1);

    View1D x(N), phi(N), b(N), sol(N);
    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        x[i] = i * dx;
    }

    phi[0] = 1.0;
    phi[N - 1] = 0.0;
    b[0] = phi[0];
    b[N - 1] = phi[N - 1];

    sol = conjugate(central_diff_1D, b, phi, dx);

    sol[0] = 1.0;

    for (int i = 0; i < N; ++i) {
        std::cout << x[i] << " " << sol[i] << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    std::cout << "Execution Time: " << duration.count() << " microseconds" << std::endl;

    return 0;
}
