#pragma once

#include <Eigen/Core>
#include <limits>
#include <mpi.h>

namespace Spectra {

template <typename Scalar>
struct numTraits
{
    typedef Scalar RealType;

    static inline Scalar min()
    {
        return Eigen::numext::pow(Eigen::NumTraits<Scalar>::epsilon(), Scalar(3));
    }
};

// Full specialization
template <>
struct numTraits<float>
{
    typedef float RealType;
    static const int Datatype=MPI_FLOAT;

    static const int MPI_OP_SUM=MPI_SUM;
    static const int MPI_OP_MAX=MPI_MAX;

    static inline float min()
    {
        return std::numeric_limits<float>::min();
    }
};

// Full specialization

/*
template <>
struct numTraits<std::complex<float>>
{
	typedef float RealType;
	static const int Datatype=MPI_2FLOAT;
	static const int MPI_OP_SUM=MPI_SUM;

    static inline float min()
    {
        return std::numeric_limits<float>::min();
    }
};
*/


template <>
struct numTraits<double>
{
    typedef double RealType;
    static const int Datatype=MPI_DOUBLE;
    static const int MPI_OP_SUM=MPI_SUM;
    static const int MPI_OP_MAX=MPI_MAX;
    static inline double min()
    {
        return std::numeric_limits<double>::min();
    }
};

/*
template <>
struct numTraits<std::complex<double>>
{
	typedef double RealType;
	static const int Datatype=MPI_DOUBLE;
	static const int MPI_OP_SUM=MPI_SUM;
	  static inline double min()
    {
        return std::numeric_limits<double>::min();
    }
};
*/

template <>
struct numTraits<long double>
{
    typedef long double RealType;
    static const int Datatype=MPI_LONG_DOUBLE;
    static const int MPI_OP_SUM=MPI_SUM;
    static const int MPI_OP_MAX=MPI_MAX;
    static inline long double min()
    {
        return std::numeric_limits<long double>::min();
    }
};


} // namespace Spectra

