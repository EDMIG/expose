#include <iostream>

#include "dd_real.h"
#include "qd_real.h"
#include "dd_traits.h"
#include "qd_traits.h"

using namespace std;

#if 0
bool flag_verbose = true;
const int double_digits = 6;

template <class T>
bool test1() {
	cout << endl;
	cout << "Test 1.  (Polynomial)." << endl;

	static const int n = 8;
	T *c = new T[n];
	T x, y;

	for (int i = 0; i < n; i++)
		c[i] = static_cast<double>(i + 1);

	x = polyroot(c, n - 1, T(0.0));
	y = polyeval(c, n - 1, x);

	if (flag_verbose) {
		cout.precision(T::_ndigits);
		cout << "Root Found:  x  = " << x << endl;
		cout << "           p(x) = " << y << endl;
	}

	delete[] c;
	return (to_double(y) < 4.0 * T::_eps);
}

/* Test 2.  Machin's Formula for Pi. */
template <class T>
bool test2() {

	cout << endl;
	cout << "Test 2.  (Machin's Formula for Pi)." << endl;

	/* Use the Machin's arctangent formula:

	pi / 4  =  4 arctan(1/5) - arctan(1/239)

	The arctangent is computed based on the Taylor series expansion

	arctan(x) = x - x^3 / 3 + x^5 / 5 - x^7 / 7 + ...
	*/

	T s1, s2, t, r;
	int k;
	int sign;
	double d;
	double err;

	/* Compute arctan(1/5) */
	d = 1.0;
	t = T(1.0) / 5.0;
	r = sqr(t);
	s1 = 0.0;
	k = 0;

	sign = 1;
	while (t > T::_eps) {
		k++;
		if (sign < 0)
			s1 -= (t / d);
		else
			s1 += (t / d);

		d += 2.0;
		t *= r;
		sign = -sign;
	}

	if (flag_verbose)
		cout << k << " Iterations" << endl;

	/* Compute arctan(1/239) */
	d = 1.0;
	t = T(1.0) / 239.0;
	r = sqr(t);
	s2 = 0.0;
	k = 0;

	sign = 1;
	while (t > T::_eps) {
		k++;
		if (sign < 0)
			s2 -= (t / d);
		else
			s2 += (t / d);

		d += 2.0;
		t *= r;
		sign = -sign;
	}

	if (flag_verbose)
		cout << k << " Iterations" << endl;

	T p = 4.0 * s1 - s2;

	p *= 4.0;
	err = abs(to_double(p - T::_pi));

	if (flag_verbose) {
		cout.precision(T::_ndigits);
		cout << "   pi = " << p << endl;
		cout << "  _pi = " << T::_pi << endl;

		cout.precision(double_digits);
		cout << "error = " << err << " = " << err / T::_eps << " eps" << endl;
	}

	return (err < 8.0 * T::_eps);
}

/* Test 6.  Taylor Series Formula for log 2.*/
template <class T>
bool test6() {
	cout << endl;
	cout << "Test 6.  (Taylor Series Formula for Log 2)." << endl;
	cout.precision(T::_ndigits);

	/* Use the Taylor series

	-log(1-x) = x + x^2/2 + x^3/3 + x^4/4 + ...

	with x = 1/2 to get  log(1/2) = -log 2.
	*/

	T s = 0.5;
	T t = 0.5;
	double delta;
	double n = 1.0;
	double i = 0;

	while (abs(t) > T::_eps) {
		i++;
		n += 1.0;
		t *= 0.5;
		s += (t / n);
	}

	delta = abs(to_double(s - T::_log2));

	if (flag_verbose) {
		cout << " log2 = " << s << endl;
		cout << "_log2 = " << T::_log2 << endl;

		cout.precision(double_digits);
		cout << "error = " << delta << " = " << (delta / T::_eps)
			<< " eps" << endl;
		cout << i << " iterations." << endl;
	}

	return (delta < 4.0 * T::_eps);
}

template <class T>
bool test8() {
	cout << endl;
	cout << "Test 8.  (Sanity check for sin / cos)." << endl;
	cout.precision(T::_ndigits);

	/* Do simple sanity check
	*
	*  sin(x) = sin(5x/7)cos(2x/7) + cos(5x/7)sin(2x/7)
	*
	*  cos(x) = cos(5x/7)cos(2x/7) - sin(5x/7)sin(2x/7);
	*/

	T x = T::_pi / 3.0;
	T x1 = 5.0 * x / 7.0;
	T x2 = 2.0 * x / 7.0;

	T r1 = sin(x1)*cos(x2) + cos(x1)*sin(x2);
	T r2 = cos(x1)*cos(x2) - sin(x1)*sin(x2);
	T t1 = sqrt(T(3.0)) / 2.0;
	T t2 = 0.5;

	double delta = std::max(abs(to_double(t1 - r1)), abs(to_double(t2 - r2)));

	if (flag_verbose) {
		cout << "  r1 = " << r1 << endl;
		cout << "  t1 = " << t1 << endl;
		cout << "  r2 = " << r2 << endl;
		cout << "  t2 = " << t2 << endl;

		cout.precision(double_digits);
		cout << " error = " << delta << " = " << (delta / T::_eps)
			<< " eps" << endl;
	}

	return (delta < 4.0 * T::_eps);
}

void test_quadt_main();


void test_qd()
{
	double t1, t2;
	double s, err;
	t1 = std::sqrt(2.1);
	t2 = std::sqrt(3.1);

	s = qd::two_sum(t1, t2, err);
	cout.precision(15);
	cout << "s=" << s << endl;
	cout << "err=" << err << endl;

}

#endif


//20180628
/*
	Eigen结合dd_real、qd_real实验！
*/


#include <Eigen/core>
#include <Eigen/SVD>
#include <Eigen/Dense>


void test_eigen_1()
{
	using namespace Eigen;

	Eigen::Matrix<dd_real, 2, 2> x;
	x(0, 0) = sqrt(dd_real(2.1));
	x(0, 1) = sqrt(dd_real(3.0));
	x(1, 0) = sin(dd_real(4.0));
	x(1, 1) = cos(dd_real(5.0));
	
	Eigen::Matrix<dd_real, 2, 2> y;

	y(0, 0) = cos(dd_real(2.1));
	y(0, 1) = cos(dd_real(3.1));
	y(1, 0) = cos(dd_real(4.1));
	y(1, 1) = cos(dd_real(5.1));

	//矩阵相乘:
	auto z = x*y;

	//SVD成功
	JacobiSVD<Matrix<dd_real,2,2> > svd(z, ComputeFullU|ComputeFullV);

	auto &U = svd.singularValues();
	auto Cp = svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();
	auto diff = Cp - z;

	cout << diff(0, 0).to_string() << endl;
	cout << diff(0, 1).to_string() << endl;
	cout << diff(1, 0).to_string() << endl;
	cout << diff(1, 1).to_string() << endl;

	cout << U(0,0).to_string() << endl;
	cout << U(1, 0).to_string() << endl;
	//cout << U(2, 0).to_string() << endl;

	//solver
	Matrix<dd_real, 2, 1> v;
	v(0, 0) = z(0, 0);
	v(1, 0) = z(1, 0);
	
	Eigen::Index in;
	/*
		需要增加dd_traits.h，特性类声明!
	*/
	Eigen::Matrix<dd_real, 2, 2> s = x.colPivHouseholderQr().solve(z);
	
	auto diff1 = s - y;
	
	cout << "s:" << endl;
	cout << s(0, 0).to_string() << endl;
	cout << s(0, 1).to_string() << endl;
	cout << s(1, 0).to_string() << endl;
	cout << s(1, 1).to_string() << endl;
	
	cout << "y:" << endl;
	cout << y(0, 0).to_string() << endl;
	cout << y(0, 1).to_string() << endl;
	cout << y(1, 0).to_string() << endl;
	cout << y(1, 1).to_string() << endl;



	/* Python Decimal Module:
	
		from decimal import *
		getcontext().prec=34
		x00=Decimal('1.4491376746189438880169333150257')
		x01=Decimal('1.7320508075688772935274463415058')
		y00=Decimal('-0.50484610459985752828933163777342')
		y10=Decimal('-0.57482394653326920224545723244855')

		z00=Decimal('-1.7272157908631477151639325868808')

		t=x00*y00+x01*y10

		tt=x00/x01+y00/y10+(Decimal('10')*z00).exp()

		print t
		print tt
	*/

	
	cout << "x00: " << x(0, 0).to_string() << endl;
	cout << "x01: " << x(0, 1).to_string() << endl;
	cout << "y00: " << y(0, 0).to_string() << endl;
	cout << "y10: " << y(1, 0).to_string() << endl;
	
	cout << "z00： "<<z(0,0).to_string() << endl;

	auto t = x(0,0)/ x(0,1) + y(0,0) / y(1,0) + exp(10*z(0,0));

	cout << t.to_string() << endl;

	//cout << x(0, 0).to_string() << endl;
	//x *= 3;
	//cout << x(0,0).to_string() << endl;
	
#if 0
	for (int n = 1; n < 100; n++)
	{

		//matlab: vpa(cos(sqrt(sym(79))/sym(2)))
		dd_real x=sqrt(dd_real(n));
		//dd_real y = sqrt(x/2);
		dd_real y = cos(x/2);
		cout <<n<<": " <<  y.to_string() << endl;
	}


	/*
	dd_real a, b;
	a = cos(dd_real::_pi/7.1);
	b = 3.0;
	cout << (a*b).to_string() << endl;
	cout << a.to_string() << endl;

	//matlab内常数需要用sym声明，否者会调用double函数处理！
	//matlab: vpa(sqrt(sym(2))*sym(pi))
	cout << (dd_real::_pi*sqrt(dd_real(2))).to_string() << endl;
	
	dd_real c = dd_real::_pi*dd_real::_e;

	cout << c.to_string() << endl;
	cout.precision(20);
	cout << c.x[0] << endl;
	cout << c.x[1] << endl;

	double t1, t2;
	t1=qd::two_sum(c.x[0], c.x[1], t2);
	cout << t1 << endl;
	cout << t2 << endl;
	*/

#endif


}

#include <Eigen/Eigenvalues>

template<typename REAL>
void test_eigen_2()
{
	using namespace Eigen;
	using namespace std;

	Matrix<REAL, 2, 2> x;
	x(0, 0) = sqrt(REAL(2.1));
	x(0, 1) = sqrt(REAL(3.0));
	x(1, 0) = sin(REAL(4.0));
	x(1, 1) = cos(REAL(5.0));

	Matrix<REAL, 3, 3> y;

	y(0, 0) = -10+3*cos(REAL(2.1));
	y(0, 1) = 2+cos(REAL(3.1));
	y(0, 2) = 3+cos(REAL(4.1));

	y(1, 0) = 4+cos(REAL(4.1));
	y(1, 1) = 5+cos(REAL(5.1));
	y(1, 2) = 6+cos(REAL(6.1));

	y(2, 0) = 7+cos(REAL(7.1));
	y(2, 1) = 8+cos(REAL(8.1));
	y(2, 2) = 9+cos(REAL(9.1));

	EigenSolver<Matrix<REAL, 3, 3>> es(y, true);

	size_t i, j;
	for (i = 0; i<3; i++)
	{
		auto &e = es.eigenvalues()[i];
		auto re= e.real();
		auto im= e.imag();
		cout << "("<<re.to_string() <<","<<im.to_string()<<")"<< endl;
	}
	
	Matrix<REAL, 4, 4> z;
	z(0, 0) = 0.964888535199277;
	z(0, 1) = 0.485375648722841;
	z(0, 2) = 0.915735525189067;
	z(0, 3) = 0.035711678574190;
	
	z(1, 0) = 0.157613081677548;
	z(1, 1) = 0.800280468888800;
	z(1, 2) = 0.792207329559554;
	z(1, 3) = 0.849129305868777;
	
	z(2, 0) = 0.970592781760616;
	z(2, 1) = 0.141886338627215;
	z(2, 2) = 0.959492426392903;
	z(2, 3) = 0.933993247757551;

	z(3, 0) = 0.957166948242946;
	z(3, 1) = 0.421761282626275;
	z(3, 2) = 0.655740699156587;
	z(3, 3) = 0.678735154857773;

	EigenSolver<Matrix<REAL, 4, 4>> zes(z, true);


	for (i = 0; i<4; i++)
	{
		auto &e = zes.eigenvalues()[i];
		auto re = e.real();
		auto im = e.imag();
		cout << "(" << re.to_string() << "," << im.to_string() << ")" << endl;
	}
	
	//(2.6913538095811144811336344340067e+00, 0.0000000000000000000000000000000e+00)
	//(7.4195689716536474535931799182253e-02, 4.7663997267413587765467477013036e-01)
	//(7.4195689716536474535931799182253e-02, -4.7663997267413587765467477013036e-01)
	//(5.6365139632456556963312047468520e-01, 0.0000000000000000000000000000000e+00)

}

void test_eigen_3()
{
	using namespace Eigen;
	typedef Matrix<dd_real, Dynamic, Dynamic> MatrixXq;
	MatrixXq A = MatrixXq::Random(10, 10);

	EigenSolver<MatrixXq> es(A, true);
	
	Index i;

	for (i = 0; i<10; i++)
	{
		auto &e = es.eigenvalues()[i];
		auto re = e.real();
		auto im = e.imag();
		cout << "(" << re.to_string() << "," << im.to_string() << ")" << endl;
	}

	//(5.0558411595984598369069960217078e+00, 0.0000000000000000000000000000000e+00)
	//(-1.1751034255570295153258409976742e+00, 0.0000000000000000000000000000000e+00)
	//(8.8869467317301398028199382178394e-01, 2.1101286677440598921131509354487e-01)
	//(8.8869467317301398028199382178394e-01, -2.1101286677440598921131509354487e-01)
	//(3.0197135804762833692771778536557e-01, 6.0445272233235290383626430398528e-01)
	//(3.0197135804762833692771778536557e-01, -6.0445272233235290383626430398528e-01)
	//(3.7412870758467994805803232989322e-03, 2.2585801342080580244229880609912e-01)
	//(3.7412870758467994805803232989322e-03, -2.2585801342080580244229880609912e-01)
	//(5.0774616607545050221746782122480e-01, 0.0000000000000000000000000000000e+00)
	//(-2.3578439969965619404960929501024e-01, 0.0000000000000000000000000000000e+00)

}

#include <GenEigsSolver.h>
#include <MatOp/DenseGenMatProd.h>
#include <omp.h>

typedef Eigen::Matrix<dd_real, Eigen::Dynamic, Eigen::Dynamic> MatrixXq;



void test_spectra_1()
{
	using namespace std;
	using namespace Eigen;
	using namespace Spectra;

	MatrixXq A = MatrixXq::Random(1500, 1500);
	DenseGenMatProd<dd_real> op(A);

	GenEigsSolver<dd_real, 0, DenseGenMatProd<dd_real>> eigs(&op, 6, 50);
	double time0, time1;
	time0 = omp_get_wtime();
	eigs.init();
	time1 = omp_get_wtime();

	cout << "eigs.init(), time used " << time1 - time0 << endl;


	time0 = omp_get_wtime();
	int nconv = eigs.compute(500, NumTraits<dd_real>::epsilon());
	time1 = omp_get_wtime();
	cout << "eigs.init(), time used " << time1 - time0 << endl;

	int niter = eigs.num_iterations();
	int nops = eigs.num_operations();

	auto evals = eigs.eigenvalues();
	auto evecs = eigs.eigenvectors();
	
	//相对误差的意义更大！
	auto resid = A * evecs - evecs * evals.asDiagonal();
	dd_real err = resid.array().abs().maxCoeff();


	cout << "nconv=" << nconv << endl;
	cout << "niter=" << niter << endl;
	cout << "nops=" << nops << endl;
	cout << "err=" << err.to_string() << endl;

}

#include <SymEigsSolver.h>
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

	void test_spectra_2()
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
		time0 = omp_get_wtime();
		eigs.init();
		time1 = omp_get_wtime();

		cout << "eigs.init(), time used " << time1 - time0 << endl;


		time0 = omp_get_wtime();

		int nconv = eigs.compute(500, NumTraits<Real>::epsilon());
		time1 = omp_get_wtime();
		cout << "eigs.init(), time used " << time1 - time0 << endl;

		int niter = eigs.num_iterations();
		int nops = eigs.num_operations();

		auto evals = eigs.eigenvalues();
		auto evecs = eigs.eigenvectors();

		//相对误差的意义更大！
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
}
/*
#include "dd_complex.h"
void test_dd_complex()
{
	dd_real a, b;
	dd_complex c, d;

	a = sqrt(dd_real(2));
	b = sqrt(dd_real(3));

	c = dd_complex(a, b);
	d = dd_complex(2 * a, 3 * b);

	dd_complex t,x,y,w;

	t = c*d;
	x = c / d;
	y = c*a;
	w = c / a;
	cout << a.to_string() << endl;
	cout << b.to_string() << endl;
	cout << endl;

	cout << c.to_string() << endl;
	cout << d.to_string() << endl;
	cout << endl;

	cout << t.to_string() << endl;
	cout << x.to_string() << endl;
	cout << y.to_string() << endl;
	cout << w.to_string() << endl;


}

*/

#include <complex>
void test_complex()
{
	//std::complex<dd_real>测试通过！
	typedef std::complex<dd_real> complex_t;

	dd_real a, b;

	/*matlab code
	a=vpa(sqrt(2));
	b=vpa(sqrt(3));
	c1=a,c2=b;
	d1=a+b,d2=a-b;

	y1=(c1*d1+c2*d2)/(d1*d1+d2*d2)
	y2=(c2*d1-c1*d2)/(d1*d1+d2*d2)
	*/
	
	a = sqrt(dd_real(2));
	b = sqrt(dd_real(3));

	complex_t c = complex_t(a, b);
	complex_t d = complex_t(a + b, a - b);
	complex_t x, y, z;
	
	x = c*d;
	y = c / d;
	z = sqrt(c);
	cout << real(x).to_string() << endl;
	cout << real(y).to_string() << endl;
	cout << real(z).to_string() << endl;
	cout << abs(z).to_string() << endl;

}

int main(int argc, char **argv)
{
	
	/*
	test1<dd_real>();
	test2<dd_real>();
	test6<dd_real>();
	test8<dd_real>();
	
	test1<qd_real>();
	test2<qd_real>();
	test6<qd_real>();
	test8<qd_real>();
	*/

	//test_qd();
	//test_quadt_main();

	//test_eigen_1();
	//test_eigen_2<dd_real>();
	//test_eigen_2<qd_real>();

	//test_eigen_3();


	//test_spectra_1();
	TEST::test_spectra_2();
	//test_dd_complex();
	//test_complex();


	

	return 0;
}



