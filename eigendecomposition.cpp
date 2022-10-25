#include "mkl.h"
#include "mkl_lapacke.h"
#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

void dgeminv(vector<double>& A, vector<double>& A_inv, int m, int n)
{
    lapack_int info_getrf, info_getri;
    lapack_int ipiv[m];

    double* a = (double*)malloc(m*n*sizeof(double));

    cblas_dcopy(m*n, &*A.begin(), 1, a, 1);

    info_getrf = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, a, n, ipiv);
    info_getri = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, a, n, ipiv);

    cblas_dcopy(m*n, a, 1, &*A_inv.begin(), 1);
}

void dgemiv(double* A, double* A_inv, int m, int n)
{
    lapack_int info_getrf, info_getri;
    lapack_int ipiv[m];

    double* a = (double*)malloc(m*n*sizeof(double));

    cblas_dcopy(m*n, A, 1, a, 1);

    info_getrf = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, a, n, ipiv);
    info_getri = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, a, n, ipiv);

    cblas_dcopy(m*n, a, 1, A_inv, 1);
}

void eigendecomposition(vector<double>& M, vector<double>& Q, vector<double>& A, vector<double>& Q_inv )
{
    int N = (int)pow((double)M.size(),0.5);
    vector<double> M_copy (M.size());
    cblas_dcopy(M.size(), &*M.begin(), 1, &*M_copy.begin(), 1);
    // cout << N << endl;

    double wr[N], wi[N], vl[N*N], vr[N*N];

    lapack_int result;
    result = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', N, &*M_copy.begin(), N, wr, wi, vl, N, vr, N);

    for (int i = 0; i < N*N; i++)
    {
        Q[i] = vr[i];
    }
    
    dgemiv(vr, &*Q_inv.begin(), N, N);

    for (int i = 0; i < N; i++)
    {
        A[i] = wr[i];
    }
}

void vec_ptr(vector<double>& v)
{
    for (int i = 0; i < v.size(); i++)
    {
        cout << v[i] << ' ';
    }
    cout << endl;
}

void matrix_ptr(vector<double>& mat, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << mat[i*n+j] << ' ';
        }
        cout << '\n';
    }
}


int main(void)
{
    vector<double> M {0, -3, -2, 1, -4, -2, -3, 4, 1};
    vector<double> Q(9);
    vector<double> A(3);
    vector<double> Q_inv(9);

    eigendecomposition(M, Q, A, Q_inv);

    matrix_ptr(M, 3, 3);
    matrix_ptr(Q, 3, 3);
    vec_ptr(A);
    matrix_ptr(Q_inv, 3, 3);

    return 0;
}