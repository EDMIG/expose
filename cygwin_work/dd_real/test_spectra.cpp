#include <iostream>

#include "mpi/MPI_SymEigsSolver.h"
#include "mpi/MPI_GenEigsSolver.h"
#include "mpi/MatOp/MPI_MatProd.h"


static const int nglobal=400;
static const int mglobal=100;

Eigen::MatrixXd A400x400(nglobal,nglobal);
Eigen::MatrixXd V400x100(nglobal,mglobal);

void init_GenEigsSolver()
{
    FILE *fp=fopen("../A400x400.bin","rb+");
    assert(fp);
    int N=nglobal*nglobal;
    auto nread=fread(A400x400.data(),sizeof(double),N,fp);
    assert(nread==N);
    fclose(fp);

    fp=fopen("../V400x100.bin","rb+");
    assert(fp);
    N=nglobal*mglobal;
    nread = fread(V400x100.data(),sizeof(double),N,fp);
    assert(nread==N);
    fclose(fp);
}

Eigen::MatrixXd Frank400x400(400,400);

void init_SymEigsSolver()
{
    int i,j;
    int n=Frank400x400.rows();
    for(j=0; j<n; j++)
    {
        for(i=0; i<n; i++)
        {
            Frank400x400(i,j)=n-std::max(i,j);
        }
    }
}

//A*X=lambda*X
//A real general matrix
int test_SymEigsSolver()
{
    MPI::Init();
    MPI::Comm &comm=MPI::COMM_WORLD;
    int rank,size;
    rank=comm.Get_rank();
    size=comm.Get_size();

    init_SymEigsSolver();

    using namespace Spectra;

    MPI_MatProd<double> op(Frank400x400,comm);

    MPI_SymEigsSolver<double,LARGEST_MAGN, MPI_MatProd<double>>
            eigs(&op,6,50);

    eigs.init();

    int nconv=eigs.compute(100);

    using namespace std;

    if(rank==0)
    {
        if(eigs.info()==SUCCESSFUL)
        {
            auto evalues=eigs.eigenvalues();
            cout<<evalues<<endl;
        }
        cout<<"nconv="<<nconv<<endl;
    }

    MPI::Finalize();

}


//A*X=lambda*X
//A real general matrix
int test_GenEigsSolver()
{
    MPI::Init();
    MPI::Comm &comm=MPI::COMM_WORLD;
    int rank,size;
    rank=comm.Get_rank();
    size=comm.Get_size();

    init_GenEigsSolver();

    using namespace Spectra;

    MPI_MatProd<double> op(A400x400,comm);

    MPI_GenEigsSolver<double,LARGEST_MAGN, MPI_MatProd<double>>
            eigs(&op,6,50);

    eigs.init();
    int nconv=eigs.compute(100);

    using namespace std;

    if(rank==0)
    {
        if(eigs.info()==SUCCESSFUL)
        {
            auto evalues=eigs.eigenvalues();
            cout<<evalues<<endl;
        }
        cout<<"nconv="<<nconv<<endl;
    }
    MPI::Finalize();

}

int main()
{
    //test_GenEigsSolver();

    test_SymEigsSolver();
}
