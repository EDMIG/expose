#pragma once 

#include <iostream>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include "mpi.h"

#include "../Util/numTraits.h"

namespace Spectra {

template < typename Scalar>
class MPI_MatProd
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Map<const Matrix> MapConstMat;
    typedef Eigen::Map<const Vector> MapConstVec;
    typedef Eigen::Map<Vector> MapVec;

    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;

	typedef typename numTraits<Scalar>::RealType Real;

	
    const MapConstMat m_mat;
	int m_rows;
	int m_global_rows; 
	
	const MPI::Comm &m_comm;
	int m_rank;
	int m_size;
	int m_rows_array[100];
	int m_rows_offset[100];
	
	int m_prev;
	int m_next;
	
private:
	/*
	row方向进行一维划分，每个MPI进程处理一部分A*X
	*/
	void partition()
	{
		int n=m_global_rows;
		int n1=n/m_size;
		int n2=n%m_size;
		
		int offset=0;
		for(int i=0; i<m_size; i++)
		{
			if(i<n2)
			{
				m_rows_array[i]=n1+1;
			}
			else 
			{
				m_rows_array[i]=n1;
			}
			m_rows_offset[i]=offset;
			offset+=m_rows_array[i];
			
		}
		assert(offset==m_global_rows);
		m_rows=m_rows_array[m_rank];

		if(m_rank==m_size-1)
		{
			assert(m_rows+m_rows_offset[m_rank]==m_global_rows);
		}
		else 
		{
			assert(m_rows+m_rows_offset[m_rank]==m_rows_offset[m_rank+1]);
		}
		
	}
public:

    MPI_MatProd(ConstGenericMatrix& mat_, MPI::Comm &comm_) :
        m_mat(mat_.data(),mat_.rows(),mat_.cols()), 
		m_comm(comm_)
    {
		assert(mat_.rows()==mat_.cols());
		
		m_global_rows=mat_.rows();
		
		m_rank=m_comm.Get_rank();
		m_size=m_comm.Get_size();

		m_prev=m_rank-1;
		m_next=m_rank+1;
		
		if(m_rank==0) m_prev=m_size-1;
		if(m_rank==m_size-1) m_next=0;		
		
		partition();
	}

public:	
	int global_rows() const { return m_mat.rows(); }
    int cols() const { return m_mat.cols(); }
	int rows()const{return m_rows;}
	const MPI::Comm &comm() const{return m_comm;}
	int rank()const{return m_rank;}
	int size()const{return m_size;}

public:	
    // y_out = A * x_in
    void perform_op(const Scalar* x_in, Scalar* y_out) const
    {

#if 1
		perform_op_impl(x_in,y_out); return;
#else		
		MapConstVec x(x_in, m_rows);
        MapVec      y(y_out,m_rows);
		
		MPI::Datatype Dtype=numTraits<Scalar>::Datatype;
		
		Vector gx(m_global_rows);
		m_comm.Allgatherv(x_in,m_rows,Dtype,gx.data(),m_rows_array,m_rows_offset,Dtype);
		y.noalias()=m_mat.block(m_rows_offset[m_rank],0, m_rows, m_global_rows)*gx;
#endif		
    }
    void perform_op_impl(const Scalar* x_in, Scalar* y_out) const
    {
		MapConstVec x(x_in, m_rows);
        MapVec      y(y_out,m_rows);
		
		MPI::Datatype Dtype=numTraits<Scalar>::Datatype;
		
		
		if(m_size==1)
		{
			//计算本地
			y.noalias()=
				m_mat.block(m_rows_offset[m_rank],m_rows_offset[m_rank], m_rows, m_rows)*x;
			return;
		}
		
		/*
		   A=[A0 A1 A3]
		   X=[X0;X1;X2]
		   Y=A*X=A0*X0+A1*X1+A2*X2
		*/
		
		//recv buffer: m_rows+1 is max count by recv
		Vector x_recv(m_rows+1);
		
		MPI::Request req_send,req_recv;
		
		req_recv=m_comm.Irecv(x_recv.data(),m_rows+1,
				Dtype,m_prev,MPI::ANY_TAG);
		
		int tag=m_rank;
		m_comm.Send(x_in,m_rows,Dtype,m_next,tag);
		
		//计算本地
		y.noalias()=m_mat.block(m_rows_offset[m_rank],m_rows_offset[m_rank], m_rows, m_rows)*x;
		
		int numRecv=0;
		while(1)
		{
			/*
			recv from prev, then send to next！
			*/
			MPI::Status status;
			req_recv.Wait(status);
			
			numRecv++;
			
			int recv_rank,nrecv,tag;
			recv_rank=status.Get_tag();
			tag = recv_rank;
			nrecv=status.Get_count(Dtype);
			
			assert(nrecv==m_rows_array[recv_rank]);
			assert(recv_rank!=m_rank);
			assert(recv_rank>=0 && recv_rank<m_size);
			
			if(recv_rank!=m_next)
				req_send=m_comm.Isend(x_recv.data(),nrecv,
						Dtype,m_next,tag);
			
			
			assert(m_rows_offset[m_rank]+m_rows<=m_mat.rows());
			assert(m_rows_offset[recv_rank]+nrecv<=m_mat.cols());
			assert(m_rows>=0 && nrecv>=0);
			assert(m_rows_offset[m_rank]>=0 && m_rows_offset[recv_rank]>=0);
			
			//计算prev进程传过来的数据
			y.noalias()+=m_mat.block(m_rows_offset[m_rank], m_rows_offset[recv_rank],m_rows,nrecv)*x_recv.head(nrecv);
			
			if(recv_rank==m_next) break;

			req_send.Wait(status);
			req_recv=m_comm.Irecv(x_recv.data(),m_rows+1,Dtype,
				m_prev,MPI::ANY_TAG);
		
		}

		assert(numRecv==m_size-1);
		
		
		//using namespace std;
		//if(0&&m_rank==0) cout<<"numRecv="<<numRecv<<endl;
		
	}	
public:
	Scalar pdot(const Vector &vec_x, const Vector &vec_y) const
	{
		assert(vec_x.rows()==vec_y.rows());
		
		int n;
		n=vec_x.rows();
		const Scalar *px=vec_x.data();
		const Scalar *py=vec_y.data();
		
		Scalar myDot=vec_x.dot(vec_y);
		Scalar sum;
		m_comm.Allreduce(&myDot,&sum,1,numTraits<Scalar>::Datatype,numTraits<Scalar>::MPI_OP_SUM);
		return sum;
	}
	
	Real pnorm2(const Vector &vec_x) const
	{
		
		int n;
		n=vec_x.rows();
		const Scalar *px=vec_x.data();
		Real myNorm=vec_x.stableNorm();
		Real maxNorm;
		
		m_comm.Allreduce(&myNorm,&maxNorm,1,numTraits<Real>::Datatype, numTraits<Real>::MPI_OP_MAX);
		if(maxNorm==0) return 0;
		myNorm/=maxNorm;
		myNorm*=myNorm;
		
		Real sum;
		m_comm.Allreduce(&myNorm, &sum, 1, numTraits<Real>::Datatype, numTraits<Real>::MPI_OP_SUM);
		return maxNorm*std::sqrt(std::abs(sum));
	}
	
	Real pabsmax(const Vector &vec_x) const
	{
		int n;
		n=vec_x.rows();
		const Scalar *px=vec_x.data();
		Real myAbsmax=vec_x.cwiseAbs().maxCoeff();
		Real absmax;
		m_comm.Allreduce(&myAbsmax, &absmax, 1, numTraits<Real>::Datatype, numTraits<Real>::MPI_OP_MAX);
		return absmax;	
	}
};

} // namespace Spectra
