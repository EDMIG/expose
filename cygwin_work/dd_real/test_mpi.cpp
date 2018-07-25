
#include<cstdio>
#include <iostream>
#include<cassert>
#include <Eigen/Core>
#include "matrix_io.h"

#include "mpi.h"

static const int nglobal=400;
static const int mglobal=100;

Eigen::MatrixXd A400x400(nglobal,nglobal);
Eigen::MatrixXd V400x100(nglobal,mglobal);
//Eigen::Map<Eigen::MatrixXd> CC;


struct dd_real
{
    double  real,imag;
};

struct SetUpTest
{
    SetUpTest()
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
    ~SetUpTest()
    {

    }
};

static SetUpTest myTest;

struct SetUpMPI
{
    SetUpMPI()
    {
        int namelen;
        MPI::Init();

        m_size = MPI::COMM_WORLD.Get_size();
        m_rank = MPI::COMM_WORLD.Get_rank();
        MPI::Get_processor_name(m_name, namelen);

        //DD,QD,DD_COMPLEX, QD_COMPLEX���Ͷ���
        MPI_Type_contiguous(2, MPI_DOUBLE, &QDTraits.MPI_DD_Type);
        MPI_Type_contiguous(4, MPI_DOUBLE, &QDTraits.MPI_QD_Type);
        MPI_Type_contiguous(4, MPI_DOUBLE, &QDTraits.MPI_DD_Complex_Type);
        MPI_Type_contiguous(8, MPI_DOUBLE, &QDTraits.MPI_QD_Complex_Type);

        MPI_Type_commit(&QDTraits.MPI_DD_Type);
        MPI_Type_commit(&QDTraits.MPI_QD_Type);
        MPI_Type_commit(&QDTraits.MPI_DD_Complex_Type);
        MPI_Type_commit(&QDTraits.MPI_QD_Complex_Type);

        //DD MPI user defined operations
        MPI_Op_create(DD_Sum, true, &QDTraits.MPI_DD_SUM);
        MPI_Op_create(DD_Max, true, &QDTraits.MPI_DD_MAX);
        MPI_Op_create(DD_Prod, true, &QDTraits.MPI_DD_PROD);

        //------------------------------------------------------
        MPI_Type_contiguous(2, MPI_FLOAT, &MPI_FLOAT_COMPLEX_Type);
        MPI_Type_contiguous(2, MPI_DOUBLE, &MPI_DOUBLE_COMPLEX_Type);
        MPI_Type_commit(&MPI_FLOAT_COMPLEX_Type);
        MPI_Type_commit(&MPI_DOUBLE_COMPLEX_Type);

        //DD MPI user defined operations
        MPI_Op_create(FLOAT_COMPLEX_Sum, true, &MPI_FLOAT_COMPLEX_SUM);
        MPI_Op_create(DOUBLE_COMPLEX_Sum, true, &MPI_DOUBLE_COMPLEX_SUM);

        //-------------------------------------------------------
    }

    void partition(int nglobal_)
    {
        nglobal=nglobal_;

        mlocal=mglobal;
        nlocal=(nglobal+m_size-1)/m_size;

        noffset=nlocal*m_rank;

        if(m_rank==m_size-1)
        {
            nlocal=nglobal-noffset;
        }

        fprintf(stderr, "size=%d,rank=%d,nglobal=%d, nlocal=%d, noffset=%d\n", m_size,m_rank,nglobal,nlocal,noffset);

    }

    ~SetUpMPI()
    {
        MPI::Finalize();
    }

    static void DD_Prod(void  *in_voidptr, void  *inout_voidptr, int *len,
                        MPI_Datatype *dptr)
    {

        dd_real *in = (dd_real*)in_voidptr;
        dd_real *inout = (dd_real*)inout_voidptr;

        dd_real c;
        for (int i = 0; i< *len; ++i) {
            c.real = inout->real*in->real -
                     inout->imag*in->imag;

            c.imag = inout->real*in->imag +
                     inout->imag*in->real;

            *inout = c;
            in++;
            inout++;
        }
    }

    static void DD_Sum(void  *in_voidptr, void  *inout_voidptr, int *len,
                       MPI_Datatype *dptr)
    {

        dd_real *in = (dd_real*)in_voidptr;
        dd_real *inout = (dd_real*)inout_voidptr;

        dd_real c;
        for (int i = 0; i< *len; ++i) {
            c.real = inout->real+in->real;
            c.imag = inout->imag+in->imag;
            *inout = c;
            in++;
            inout++;
        }
    }

    static void DD_Max(void  *in_voidptr, void  *inout_voidptr, int *len,
                       MPI_Datatype *dptr)
    {

        dd_real *in = (dd_real*)in_voidptr;
        dd_real *inout = (dd_real*)inout_voidptr;

        dd_real c;
        for (int i = 0; i< *len; ++i) {
            c.real = std::max(inout->real, in->real);
            c.imag = std::max(inout->imag, in->imag);
            *inout = c;
            in++;
            inout++;
        }
    }

    //MPI::Comm m_comm(MPI::COMM_WORLD);
    int m_size;
    int m_rank;
    char m_name[MPI_MAX_PROCESSOR_NAME];

//====================================================

    MPI_Datatype MPI_FLOAT_COMPLEX_Type;
    MPI_Datatype MPI_DOUBLE_COMPLEX_Type;

    MPI_Op MPI_FLOAT_COMPLEX_SUM;
    MPI_Op MPI_DOUBLE_COMPLEX_SUM;

    static void FLOAT_COMPLEX_Sum(void  *in_voidptr, void  *inout_voidptr, int *len,
                                  MPI_Datatype *dptr)
    {

        typedef std::complex<float> complex_t;
        complex_t *in = (complex_t*)in_voidptr;
        complex_t *inout = (complex_t *)inout_voidptr;

        dd_real c;
        for (int i = 0; i< *len; ++i) {
            inout[i]+=in[i];
        }
    }

    static void DOUBLE_COMPLEX_Sum(void  *in_voidptr, void  *inout_voidptr, int *len,
                                   MPI_Datatype *dptr)
    {

        typedef std::complex<double> complex_t;
        complex_t *in = (complex_t*)in_voidptr;
        complex_t *inout = (complex_t *)inout_voidptr;

        dd_real c;
        for (int i = 0; i< *len; ++i) {
            inout[i]+=in[i];
        }
    }
//========================================================

    struct _QDTraits
    {
        MPI_Datatype MPI_DD_Type;
        MPI_Datatype MPI_QD_Type;
        MPI_Datatype MPI_DD_Complex_Type;
        MPI_Datatype MPI_QD_Complex_Type;


        MPI_Op MPI_DD_SUM;
        MPI_Op MPI_DD_MAX;
        MPI_Op MPI_DD_PROD;



        MPI_Op MPI_QD_SUM;
        MPI_Op MPI_DD_COMPLEX_SUM;
        MPI_Op MPI_QD_COMPLEX_SUM;


        MPI_Op MPI_QD_MAX;
        MPI_Op MPI_DD_COMPLEX_MAX;
        MPI_Op MPI_QD_COMPLEX_MAX;

    } QDTraits;

    int nglobal;
    int nlocal, mlocal;
    int noffset;
};

static SetUpMPI MyMPI;

#include <iomanip>

namespace MyUtil
{

//max(abs(x))
//template<typename Scalar> Scalar pabsmax(int n, Scalar *x, MPI::Comm &comm);

//sqrt(dot(x,x))
//template<typename Scalar> Scalar pnorm(int nloc, Scalar *x, MPI::Comm &comm);
//dot(x,x)
//template<typename Scalar> Scalar pnorm2(int nloc, Scalar *x, MPI::Comm &comm);
//dot(x,y)
//template<typename Scalar> Scalar pdot(int n, Scalar *x, Scalar *y, const MPI::Comm &comm);
//y=A*x
//template<typename Scalar> void pgemv(int mloc, int nloc, Scalar *A, const Scalar *x, Scalar *y, MPI::Comm & comm);
//f=f-v*(v'*f)
template<typename Scalar> void pf_orth_v(int n, int nv, Scalar *v, Scalar *f, const MPI::Comm &comm, Scalar *work=nullptr);
//f=f-v*vf
template<typename Scalar> void pf_orth_v(int n, int nv, Scalar *v, Scalar *vf, Scalar *f, const MPI::Comm &comm);


template<typename T> struct numTraits
{
    enum {
        IsInteger = 0,
        IsSigned = 0,
        IsComplex = 0,
    };

    static const int Datatype =-1;
    static const int MPI_OP_SUM = -1;
    static const int MPI_OP_MAX = -1;
};

template<>struct numTraits<dd_real>
{
    MPI_Datatype Datatype = MyMPI.QDTraits.MPI_DD_Type;
    MPI_Op MPI_OP_SUM = MyMPI.QDTraits.MPI_DD_SUM;
    MPI_Op MPI_OP_MAX = MyMPI.QDTraits.MPI_DD_MAX;
};


struct MPI_Common_OpSet
{
    static const int MPI_OP_SUM = MPI_SUM;
    static const int MPI_OP_MAX = MPI_MAX;
};

struct MPI_Float_OpSet :MPI_Common_OpSet
{
};

struct MPI_Int_OpSet :MPI_Common_OpSet
{
};

template<> struct numTraits<float>: MPI_Float_OpSet
{
    typedef float RealType;
    typedef float NonItegerType;
    typedef float NestedType;

    enum {
        IsInteger = 0,
        IsSigned = 1,
        IsComplex = 0,
    };

    static const int Datatype = MPI_FLOAT;
};

template<> struct numTraits<double>:MPI_Float_OpSet
{
    typedef double  RealType;
    typedef double  NonItegerType;
    typedef double  NestedType;

    enum {
        IsInteger = 0,
        IsSigned = 1,
        IsComplex = 0,
    };

    static const int Datatype = MPI_DOUBLE;
};

template<> struct numTraits<std::complex<double>>
{
    typedef double  RealType;
    typedef std::complex<double>  NonItegerType;
    typedef double NestedType;

    enum {
        IsInteger = 0,
        IsSigned = 1,
        IsComplex = 1,
    };

    static  MPI_Datatype Datatype;
    static  MPI_Op MPI_OP_SUM;
    //static  MPI_Datatype Datatype = MyMPI.MPI_DOUBLE_COMPLEX_Type;
    //static  MPI_Op MPI_OP_SUM = MyMPI.MPI_DOUBLE_COMPLEX_SUM;

};

MPI_Datatype numTraits<std::complex<double>>::Datatype=MyMPI.MPI_DOUBLE_COMPLEX_Type;
MPI_Op numTraits<std::complex<double>>::MPI_OP_SUM = MyMPI.MPI_DOUBLE_COMPLEX_SUM;


template<> struct numTraits<std::complex<float>>
{
    typedef float  RealType;
    typedef std::complex<float>  NonItegerType;
    typedef float NestedType;

    enum {
        IsInteger = 0,
        IsSigned = 1,
        IsComplex = 1,
    };

    MPI_Datatype Datatype = MyMPI.MPI_FLOAT_COMPLEX_Type;
    MPI_Op MPI_OP_SUM = MyMPI.MPI_FLOAT_COMPLEX_SUM;
};


template<> struct numTraits<int>:MPI_Int_OpSet
{
    typedef int NestedType;

    enum {
        IsInteger = 1,
        IsSigned = 1,
        IsComplex = 0,
    };
    static const int Datatype = MPI_INT;
};




template<typename Scalar>
auto pabsmax(int n, Scalar *x, MPI::Comm &comm)
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef typename numTraits<Scalar>::NestedType _T;

    Eigen::Map<Vector> vec_x(x, n);
    _T local_absmax=vec_x.cwiseAbs().maxCoeff();
    _T absmax;
    comm.Allreduce(&local_absmax, &absmax, 1, numTraits<_T>::Datatype, numTraits<_T>::MPI_OP_MAX);
    return absmax;
}



template<typename Scalar>
auto pdot(int n, Scalar *x, Scalar *y, const MPI::Comm &comm)
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef typename numTraits<Scalar>::NestedType _T;

    Eigen::Map<Vector> vec_x(x, n);
    Eigen::Map<Vector> vec_y(y, n);
    Scalar d = vec_x.dot(vec_y);
    Scalar sum;
    comm.Allreduce(&d, &sum, 1, numTraits<Scalar>::Datatype, numTraits<Scalar>::MPI_OP_SUM);
    return sum;
}


//dot(x,x)
template<typename Scalar>
typename numTraits<Scalar>::NestedType pnorm2(int n, Scalar *x, MPI::Comm &comm)
{
    //using namespace Eigen;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef typename numTraits<Scalar>::NestedType _T;

    Eigen::Map<Vector> vec_x(x, n);


    _T norm2= vec_x.stableNorm();
    _T max_norm2;

    comm.Allreduce(&norm2, &max_norm2, 1, numTraits<_T>::Datatype, numTraits<_T>::MPI_OP_MAX);
    if (max_norm2 == 0) return 0;

    norm2 /=  max_norm2;
    norm2 *= norm2;

    _T sum;
    comm.Allreduce(&norm2, &sum, 1, numTraits<_T>::Datatype, numTraits<_T>::MPI_OP_SUM);

    return max_norm2*std::sqrt(std::abs(sum));
}




template<>
void pf_orth_v<double>(int n, int nv, double *v, double *f, const MPI::Comm &comm, double *work)
{
    //using namespace Eigen;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;

    Eigen::Map<Matrix> mV(v, n, nv);
    Eigen::Map<Vector> vec_f(f, n);
    Eigen::Map<Vector> vec_w(work, nv);

    vec_w = mV.transpose()*vec_f;
    double *sum = work + nv;

    comm.Allreduce(work, sum, nv, MPI::DOUBLE, MPI::SUM);

    Eigen::Map<Vector> vec_s(sum, nv);

    vec_f.noalias() -= mV*vec_s;

    return;
}


template<>
void pf_orth_v<double>(int n, int nv, double *v, double *vf, double *f, const MPI::Comm &comm)
{
    using namespace Eigen;
    Map<Matrix<double, Dynamic, Dynamic>> mV(v, n, nv);
    Map<Matrix<double, Dynamic, 1>> vec_vf(vf, nv);
    Map<Matrix<double, Dynamic, 1>> vec_f(f, n);
    vec_f.noalias() -=mV* vec_vf;
}

/*
A->A(nloc, n)
x->x(nloc)
y->y(nloc)

A_global(n,n)
x_global(n)
y_global(n)

y_global(:)=A_global(:,:)*x_global(:)
*/
template<typename Scalar>
void pgemv(int n, int nlocal, Scalar *A, Scalar *x, Scalar *y, const MPI::Comm &comm, bool transposeA=false)
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;


    assert(A && x && y);


    Eigen::Map<Vector> vec_x(x, nlocal);
    Eigen::Map<Vector> vec_y(y, nlocal);


    int size=comm.Get_size();
    int rank=comm.Get_rank();


    if(size>1)
    {

        int nlocal_array[100];
        int noffset_array[100];


        comm.Allgather(&nlocal,1,MPI_INT,nlocal_array,1,MPI_INT);

        using namespace std;

        //cout<<"rank#"<<rank<<endl;

        int offset=0;
        int max_nlocal=0;
        for(int i=0; i<size; i++)
        {
            noffset_array[i]=offset;
            offset+=nlocal_array[i];
            max_nlocal=std::max(max_nlocal, nlocal_array[i]);
            //cout<<"(nlocal, noffset)="<<nlocal_array[i]<<","<<noffset_array[i]<<endl;
        }

        int prev,next;
        prev=rank-1;
        next=rank+1;
        if(prev<0) prev=size-1;
        if(next==size) next=0;

        Vector vec_x_recv(max_nlocal);
        Scalar *x_recv_buf=vec_x_recv.data();


        MPI::Request req_send, req_recv;

        MPI::Datatype Dtype= numTraits<Scalar>::Datatype;
        req_recv=comm.Irecv(x_recv_buf,max_nlocal,Dtype,
                            prev,MPI::ANY_TAG);

        int tag=rank;
        comm.Send(x,nlocal,Dtype, next,tag);
        //compute local A*x
        //A(nloc,n)
        //A=[A0,A1,A2,...,Arank]

        Scalar *ptr_A=A+nlocal*noffset_array[rank];
        Eigen::Map<Matrix> mA(ptr_A,nlocal,nlocal);
        vec_y.noalias()=mA*vec_x;

        int numRecv=0;
        while(1)
        {
            MPI::Status status;
            req_recv.Wait(status);
            numRecv++;

            int recv_rank, nrecv,tag;
            recv_rank=status.Get_tag();
            tag = status.Get_tag();
            nrecv=status.Get_count(Dtype);

            assert(nrecv==nlocal_array[recv_rank]);

            int noffset=noffset_array[recv_rank];

            assert(recv_rank!=rank);


            /*
            if(rank==size-1)
            {
            	cout<<"recv_rank="<<recv_rank<<endl;
            	cout<<"nrecv="<<nrecv<<endl;
            }
            */

            if(recv_rank!=next)
                req_send=comm.Isend(x_recv_buf,nrecv,Dtype,next,tag);

            Scalar *ptr_A=A+nlocal*noffset;
            Eigen::Map<Matrix> mA(ptr_A,nlocal, nrecv);
            Eigen::Map<Vector> vec_x(x_recv_buf, nrecv);

            vec_y.noalias()+=mA*vec_x;

            if(recv_rank==next) break;

            req_send.Wait(status);

            req_recv=comm.Irecv(x_recv_buf,max_nlocal,Dtype,
                                prev,MPI_ANY_TAG);
        }
    }
    else
    {
        Scalar *ptr_A=A;
        Eigen::Map<Matrix> mA(ptr_A,nlocal,nlocal);

        vec_y.noalias()=mA*vec_x;
    }

}

template<int N=100>
void test_all()
{
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
    typedef std::complex<double> Complex;

    int rank, size;

    size = MyMPI.m_size;
    rank = MyMPI.m_rank;

    int nlen=N;

    MyMPI.partition(nlen);

    int nlocal=MyMPI.nlocal;
    int noffset=MyMPI.noffset;


    using namespace std;

    double *ptr_A=A400x400.data();
    double *ptr_V=V400x100.data();

    Complex *ptr_CA=(Complex*)ptr_A;
    Complex *ptr_CV=(Complex*)ptr_V;


    cout<<setprecision(20);


    /*
    		cout<<"Test pabsmax:\n";

    		{
    			auto absmax=pabsmax(nlocal,ptr_A+noffset,MPI::COMM_WORLD);

    			cout<<"\tTest double pabsmax: absmax="<<absmax<<endl;
    		}
    		{

    			auto absmax=pabsmax(nlocal,ptr_CA+noffset,MPI::COMM_WORLD);

    			cout<<"\tTest complex pabsmax: absmax="<<absmax<<endl;
    		}


    		cout<<"Test pdot:\n";

    		{
    			auto dot=pdot(nlocal, ptr_A+noffset,  ptr_V+noffset,  MPI::COMM_WORLD);

    			cout<<"\tTest double pdot: dot="<<dot<<endl;
    		}
    		{

    			auto dot=pdot(nlocal,ptr_CA+noffset,ptr_CV+noffset, MPI::COMM_WORLD);

    			cout<<"\tTest complex pdot: dot="<<dot<<endl;
    		}


    		cout<<"Test pnorm2:\n";

    		{
    			auto norm2=pnorm2(nlocal, ptr_A+noffset, MPI::COMM_WORLD);

    			cout<<"\tTest double pnorm2: norm2="<<norm2<<endl;
    		}
    		{

    			auto norm2=pnorm2(nlocal,ptr_CA+noffset, MPI::COMM_WORLD);

    			cout<<"\tTest complex pnorm2: norm2="<<norm2<<endl;
    		}
    */

    cout<<"Test pgemv:\n";

    {
        Vector vec_y(nlocal);
        double *ptr_y=vec_y.data();
        //double *ptr_x= ptr_V;

        //pgemv<double>(nlen, nlocal, ptr_A, ptr_x, ptr_y, MPI::COMM_WORLD);

        using namespace Eigen;
        MatrixXd A;
        VectorXd X;
        A=MatrixXd::Random(nlocal,nlen);
        X=VectorXd::Random(nlocal);
        A.array()+=rank;

        double *ptr_x=X.data();
        double *ptr_A=A.data();

        char fname[100];
        sprintf(fname, "A%d.bin",rank);
        FILE *fp=fopen(fname,"wb+");
        assert(fp);
        assert(fwrite(ptr_A,sizeof(double),nlocal*nlen,fp)==nlocal*nlen);
        fclose(fp);

        sprintf(fname, "X%d.bin",rank);
        fp=fopen(fname,"wb+");
        assert(fp);
        assert(fwrite(ptr_x,sizeof(double),nlocal,fp)==nlocal);
        fclose(fp);


        pgemv<double>(nlen, nlocal, ptr_A, ptr_x, ptr_y, MPI::COMM_WORLD);

        sprintf(fname, "Y%d.bin",rank);
        fp=fopen(fname,"wb+");
        assert(fp);
        assert(fwrite(ptr_y,sizeof(double),nlocal,fp)==nlocal);
        fclose(fp);

        cout<<"y(0)="<<vec_y(0)<<endl;
    }

    {

    }


}


}

int mpi_main(int argc, char *argv[])
{
    int i, rank, size, namelen;
    char name[MPI_MAX_PROCESSOR_NAME];
    MPI::Status stat;

    //MPI::Init(argc, argv);

    //size = MPI::COMM_WORLD.Get_size();
    //rank = MPI::COMM_WORLD.Get_rank();
    //MPI::Get_processor_name(name, namelen);
    size = MyMPI.m_size;
    rank = MyMPI.m_rank;

    if (rank == 0) {

        std::cout << "Hello world: rank " << rank << " of " << size << " running on " << name << "\n";

        for (i = 1; i < size; i++) {
            MPI::COMM_WORLD.Recv(&rank, 1, MPI_INT, i, 1, stat);
            MPI::COMM_WORLD.Recv(&size, 1, MPI_INT, i, 1, stat);
            MPI::COMM_WORLD.Recv(&namelen, 1, MPI_INT, i, 1, stat);
            MPI::COMM_WORLD.Recv(name, namelen + 1, MPI_CHAR, i, 1, stat);
            std::cout << "Hello world: rank " << rank << " of " << size << " running on " << name << "\n";
        }

    }
    else {

        MPI::COMM_WORLD.Send(&rank, 1, MPI_INT, 0, 1);
        MPI::COMM_WORLD.Send(&size, 1, MPI_INT, 0, 1);
        MPI::COMM_WORLD.Send(&namelen, 1, MPI_INT, 0, 1);
        MPI::COMM_WORLD.Send(name, namelen + 1, MPI_CHAR, 0, 1);

    }

    //MPI::Finalize();

    return (0);
}

int main(int argc, char **argv)
{
    //test_Eigen();
    //MyUtil::test_pabsmax<float>(argc, argv);
    //MyUtil::test_pabsmax<double>(argc, argv);
    //MyUtil::test_all<100>();
    //MyUtil::test_all<123>();
    //MyUtil::test_all<234>();
    MyUtil::test_all<231>();

}

/*
#include <boost/mpi.hpp>
#include <iostream>
#include <string>
#include <boost/serialization/string.hpp>

namespace mpi = boost::mpi;

int main()
{
	mpi::environment env;
	mpi::communicator world;

	if (world.rank() == 0) {
		world.send(1, 0, std::string("Hello"));
		std::string msg;
		world.recv(1, 1, msg);
		std::cout << msg << "!" << std::endl;
	}
	else {
		std::string msg;
		world.recv(0, 0, msg);
		std::cout << msg << ", ";
		std::cout.flush();
		world.send(0, 1, std::string("world"));
	}

	return 0;
}

*/
