#pragma once

#include <Eigen/Core>
#include "qd_real.h"

namespace Eigen {

template<> struct NumTraits<qd_real>
    : GenericNumTraits<qd_real>
{
    typedef qd_real Real;
    typedef qd_real NonInteger;
    typedef qd_real Nested;

    enum {
        IsInteger = 0,
        IsSigned = 1,
        IsComplex = 0,
        RequireInitialization = 1,
        ReadCost = 4,
        AddCost = 32,
        MulCost = 64
    };


    static inline Real highest() {
        return Real::_max;
    }
    static inline Real lowest() {
        return -Real::_max;
    }

    // Constants
    static inline Real Pi() {
        return Real::_pi;
    }
    static inline Real Euler() {
        return Real::_e;
    }
    static inline Real Log2() {
        return  Real::_log2;
    }
    //static inline Real Catalan() { return Real::_catalan; }

    static inline Real epsilon() {
        return Real::_eps;
    }
    static inline Real epsilon(const Real& x) {
        return Real::_eps;
    }
    static inline int digits() {
        return std::numeric_limits<Real>::digits;
    }
    static inline int digits(const Real& x) {
        return std::numeric_limits<Real>::digits;
    }
    static inline int digits10() {
        return std::numeric_limits<Real>::digits10;
    }
    static inline int digits10(const  Real& x) {
        return std::numeric_limits<Real>::digits10;
    }

    static inline Real dummy_precision()
    {
        return Real::_eps;
    }
};

namespace internal {


template<>
inline qd_real random< qd_real >()
{
    return qd_real::rand();
}

template<> inline qd_real random<qd_real>(const qd_real& a, const qd_real& b)
{
    return a + (b - a) * random<qd_real>();
}


inline bool isMuchSmallerThan(const qd_real& a, const qd_real& b, const qd_real& eps)
{
    return abs(a) <= abs(b) * eps;
}

inline bool isApprox(const qd_real& a, const qd_real& b, const qd_real& eps)
{
    return abs(a - b) <= eps;
}

inline bool isApproxOrLessThan(const qd_real& a, const qd_real& b, const qd_real& eps)
{
    return a <= b || abs(a - b) <= eps;
}

template<> inline long double cast<qd_real, long double>(const qd_real& x)
{
    return  x.x[0];
}

template<> inline double cast<qd_real, double>(const qd_real& x)
{
    return x.x[0];
}

template<> inline long cast<qd_real, long>(const qd_real& x)
{
    return long(x.x[0]);
}

template<> inline int cast<qd_real, int>(const qd_real& x)
{
    return int(x.x[0]);
}

} // end namespace internal


/*
namespace numext {
inline const qd_real& conj(const qd_real& x)  { return x; }
inline const qd_real& real(const qd_real& x)  { return x; }
inline qd_real imag(const qd_real&)    { return 0.; }
inline qd_real abs(const qd_real&  x)  { return abs(x); }
inline qd_real abs2(const qd_real& x)  { return sqr(x); }
}
*/

}
