#pragma once

#include <Eigen/Core>
#include "dd_real.h"

namespace Eigen {
		
	template<> struct NumTraits<dd_real>
		: GenericNumTraits<dd_real>
	{
		typedef dd_real Real;
		typedef dd_real NonInteger;
		typedef dd_real Nested;

		enum {
			IsInteger = 0,
			IsSigned = 1,
			IsComplex = 0,
			RequireInitialization = 1,
			ReadCost = 2,
			AddCost = 8,
			MulCost = 18
		};


		static inline Real highest() { return Real::_max; }
		static inline Real lowest() { return -Real::_max; }

		// Constants
		static inline Real Pi() { return Real::_pi; }
		static inline Real Euler() { return Real::_e; }
		static inline Real Log2() { return  Real::_log2; }
		//static inline Real Catalan() { return Real::_catalan; }

		static inline Real epsilon() { return Real::_eps; }
		static inline Real epsilon(const Real& x) { return Real::_eps; }
		static inline int digits() { return std::numeric_limits<Real>::digits; }
		static inline int digits(const Real& x) { return std::numeric_limits<Real>::digits; }
		static inline int digits10() { return std::numeric_limits<Real>::digits10; }
		static inline int digits10(const  Real& x) { return std::numeric_limits<Real>::digits10; }

		static inline Real dummy_precision()
		{
			return Real::_eps;
		}
	};

		namespace internal {

			
			template<> 
			inline dd_real random< dd_real >()
			{
				return dd_real::rand();
			}

			template<> inline dd_real random<dd_real>(const dd_real& a, const dd_real& b)
			{
				return a + (b - a) * random<dd_real>();
			}
			

			inline bool isMuchSmallerThan(const dd_real& a, const dd_real& b, const dd_real& eps)
			{
				return abs(a) <= abs(b) * eps;
			}

			inline bool isApprox(const dd_real& a, const dd_real& b, const dd_real& eps)
			{
				return abs(a - b) <= eps;
			}

			inline bool isApproxOrLessThan(const dd_real& a, const dd_real& b, const dd_real& eps)
			{
				return a <= b ||  abs(a-b)<=eps;
			}

			template<> inline long double cast<dd_real, long double>(const dd_real& x)
			{
				return  x.x[0];
			}

			template<> inline double cast<dd_real, double>(const dd_real& x)
			{
				return x.x[0];
			}

			template<> inline long cast<dd_real, long>(const dd_real& x)
			{
				return long(x.x[0])+long(x.x[1]);
			}

			template<> inline int cast<dd_real, int>(const dd_real& x)
			{
				return int(x.x[0])+int(x.x[1]);
			}

		} // end namespace internal


		/*
		namespace numext {
			inline const dd_real& conj(const dd_real& x)  { return x; }
			inline const dd_real& real(const dd_real& x)  { return x; }
			inline dd_real imag(const dd_real&)    { return 0.; }
			inline dd_real abs(const dd_real&  x)  { return abs(x); }
			inline dd_real abs2(const dd_real& x)  { return sqr(x); }
		}
		*/

}
