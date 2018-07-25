
#include <iostream>
#include <omp.h>

#include <Eigen/Eigenvalues>
#include <GenEigsSolver.h>
#include <MatOp/DenseGenMatProd.h>
#include <SymEigsSolver.h>
#include <mpi/MPI_SymEigsSolver.h>
#include <mpi/MPI_GenEigsSolver.h>
#include <mpi/MatOp/MPI_MatProd.h>


#include "dd_real.h"
#include "qd_real.h"
#include "dd_traits.h"
#include "qd_traits.h"



namespace Spectra {
template<>  struct numTraits<dd_real>
{
    typedef dd_real RealType;
    static int Datatype;
    static int MPI_OP_SUM;
    static int MPI_OP_MAX;

    static inline dd_real min()
    {
        return std::numeric_limits<dd_real>::min();
    }
};
template<> struct numTraits<qd_real>
{
    typedef qd_real RealType;
    static int Datatype;
    static int MPI_OP_SUM;
    static int MPI_OP_MAX;

    static inline qd_real min()
    {
        return std::numeric_limits<qd_real>::min();
    }

};
}//namespace Spectra

int Spectra::numTraits<dd_real>::Datatype;
int Spectra::numTraits<dd_real>::MPI_OP_SUM;
int Spectra::numTraits<dd_real>::MPI_OP_MAX;

int Spectra::numTraits<qd_real>::Datatype;
int Spectra::numTraits<qd_real>::MPI_OP_SUM;
int Spectra::numTraits<qd_real>::MPI_OP_MAX;



struct SetUpMPI
{
    //MPI::Comm m_comm(MPI::COMM_WORLD);
    int m_size;
    int m_rank;
    char m_name[MPI_MAX_PROCESSOR_NAME];


    SetUpMPI()
    {
        int namelen;
        MPI::Init();

        m_size = MPI::COMM_WORLD.Get_size();
        m_rank = MPI::COMM_WORLD.Get_rank();
        MPI::Get_processor_name(m_name, namelen);

        Init_MPI();

    }

    ~SetUpMPI()
    {
        MPI::Finalize();
    }

    void Init_MPI()
    {

        //using namespace Spectra;

        MPI_Type_contiguous(2, MPI_DOUBLE, &Spectra::numTraits<dd_real>::Datatype);
        MPI_Type_contiguous(4, MPI_DOUBLE, &Spectra::numTraits<qd_real>::Datatype);


        MPI_Type_commit(&Spectra::numTraits<dd_real>::Datatype);
        MPI_Type_commit( &Spectra::numTraits<qd_real>::Datatype);


        MPI_Op_create(DD_Sum, true, &Spectra::numTraits<dd_real>::MPI_OP_SUM);
        MPI_Op_create(DD_Max, true, &Spectra::numTraits<dd_real>::MPI_OP_MAX);

        MPI_Op_create(QD_Sum, true, &Spectra::numTraits<qd_real>::MPI_OP_SUM);
        MPI_Op_create(QD_Max, true, &Spectra::numTraits<qd_real>::MPI_OP_MAX);


    }

    static void DD_Prod(void  *in_voidptr, void  *inout_voidptr, int *len,
                        MPI_Datatype *dptr)
    {
        typedef dd_real Real;
        Real *in = (Real*)in_voidptr;
        Real *inout = (Real *)inout_voidptr;
        for (int i = 0; i< *len; ++i) {
            inout[i]*=in[i];
        }
    }

    static void DD_Sum(void  *in_voidptr, void  *inout_voidptr, int *len,
                       MPI_Datatype *dptr)
    {
        typedef dd_real Real;
        Real *in = (Real*)in_voidptr;
        Real *inout = (Real *)inout_voidptr;
        for (int i = 0; i< *len; ++i) {
            inout[i]+=in[i];
        }
    }

    static void DD_Max(void  *in_voidptr, void  *inout_voidptr, int *len,
                       MPI_Datatype *dptr)
    {
        typedef dd_real Real;
        Real *in = (Real *)in_voidptr;
        Real *inout = (Real*)inout_voidptr;

        for (int i = 0; i< *len; ++i) {
            if(inout[i]<=in[i])
                inout[i]=in[i];
        }
    }


    static void QD_Prod(void  *in_voidptr, void  *inout_voidptr, int *len,
                        MPI_Datatype *dptr)
    {
        typedef qd_real Real;
        Real *in = (Real*)in_voidptr;
        Real *inout = (Real *)inout_voidptr;
        for (int i = 0; i< *len; ++i) {
            inout[i]*=in[i];
        }
    }

    static void QD_Sum(void  *in_voidptr, void  *inout_voidptr, int *len,
                       MPI_Datatype *dptr)
    {
        typedef qd_real Real;
        Real *in = (Real*)in_voidptr;
        Real *inout = (Real *)inout_voidptr;
        for (int i = 0; i< *len; ++i) {
            inout[i]+=in[i];
        }
    }

    static void QD_Max(void  *in_voidptr, void  *inout_voidptr, int *len,
                       MPI_Datatype *dptr)
    {
        typedef qd_real Real;
        Real *in = (Real *)in_voidptr;
        Real *inout = (Real*)inout_voidptr;

        for (int i = 0; i< *len; ++i) {
            if(inout[i]<=in[i])
                inout[i]=in[i];
        }
    }
};


static SetUpMPI myMPI;


using namespace std;
namespace TEST
{
typedef dd_real Real;
//typedef qd_real Real;
typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> Matrix;


void frank_matrix_n(Matrix &mat, int n)
{
    mat.resize(n, n);

    int i, j;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            mat(i, j) = Real(n - max(i + 1, j + 1) + 1);
        }
    }
}

void test_spectra_1()
{
    using namespace std;
    using namespace Eigen;
    using namespace Spectra;

    Matrix mat;
    int n = 400;
    frank_matrix_n(mat, n);

    DenseGenMatProd<Real> op(mat);

    SymEigsSolver<Real, 0, DenseGenMatProd<Real>> eigs(&op, 10, 70);
    double time0, time1;
    eigs.init();


    time0 = omp_get_wtime();
    int nconv = eigs.compute(500, NumTraits<Real>::epsilon());
    time1 = omp_get_wtime();
    cout << "eigs.init(), time used " << time1 - time0 << endl;


    int niter = eigs.num_iterations();
    int nops = eigs.num_operations();

    auto evals = eigs.eigenvalues();
    auto evecs = eigs.eigenvectors();

    auto resid = mat * evecs - evecs * evals.asDiagonal();
    Real err = resid.array().abs().maxCoeff();


    cout << "nconv=" << nconv << endl;
    cout << "niter=" << niter << endl;
    cout << "nops=" << nops << endl;
    cout << "err=" << err.to_string() << endl;

    for (int i = 0; i < nconv; i++)
    {
        cout << evals(i).to_string() << endl;
    }
}

int test_MPI_SymEigsSolver()
{

    Matrix mat;
    int n = 400;
    frank_matrix_n(mat, n);


    using namespace Spectra;

    MPI::Comm &comm=MPI::COMM_WORLD;
    MPI_MatProd<Real> op(mat,comm);

    MPI_SymEigsSolver<Real,LARGEST_MAGN, MPI_MatProd<Real>>
            eigs(&op,10,70);

    eigs.init();

    int nconv=eigs.compute(100,Eigen::NumTraits<Real>::epsilon());

    using namespace std;

    if(myMPI.m_rank==0)
    {
        if(eigs.info()==SUCCESSFUL)
        {
            auto evals=eigs.eigenvalues();
            //cout<<evalues<<endl;
            for (int i = 0; i < nconv; i++)
            {
                cout << evals(i).to_string() << endl;
            }
        }
        cout<<"nconv="<<nconv<<endl;
    }

}

int test_MPI_GenEigsSolver()
{

    Matrix mat;
    int n = 400;
    frank_matrix_n(mat, n);

    mat(0,100)=n;

    mat(100,0)=-n;

    using namespace Spectra;

    MPI::Comm &comm=MPI::COMM_WORLD;
    MPI_MatProd<Real> op(mat,comm);

    MPI_GenEigsSolver<Real,LARGEST_MAGN, MPI_MatProd<Real>>
            eigs(&op,10,70);

    eigs.init();

    int nconv=eigs.compute(100,Eigen::NumTraits<Real>::epsilon());

    using namespace std;

    if(myMPI.m_rank==myMPI.m_size-1)
    {
        if(eigs.info()==SUCCESSFUL)
        {
            auto evals=eigs.eigenvalues();
            //cout<<evalues<<endl;
            for (int i = 0; i < nconv; i++)
            {
                Real re=evals(i).real();
                Real im=evals(i).imag();
                cout <<"("<<re.to_string() <<","<<im.to_string()<<")"<<endl;
            }
        }
        cout<<"nconv="<<nconv<<endl;
    }

}

}

int main()
{
    //TEST::test_MPI_SymEigsSolver();
    TEST::test_MPI_GenEigsSolver();
}
