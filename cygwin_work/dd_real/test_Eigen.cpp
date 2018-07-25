#include <iostream>
#include <Eigen/Core>
#include "matrix_io.h"


static const int nglobal=400;
static const int mglobal=100;

Eigen::MatrixXd A400x400(nglobal,nglobal);
Eigen::MatrixXd V400x100(nglobal,mglobal);
void test_Eigen_1()
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

    auto y0=A400x400.block(0,0,100,400)*V400x100;
    auto y1=A400x400.block(100,0,100,400)*V400x100;
    auto y2=A400x400.block(200,0,100,400)*V400x100;
    auto y3=A400x400.block(300,0,100,400)*V400x100;
    auto y=A400x400*V400x100;

    using namespace std;

    cout<<y(0)-y0(0)<<endl;
    cout<<y(100)-y1(0)<<endl;
    cout<<y(200)-y2(0)<<endl;
    cout<<y(300)-y3(0)<<endl;



}


void test_Eigen()
{
    using namespace std;
    using namespace Eigen;

    VectorXd a, b;

    int n = 100;
    a=VectorXd::Random(n);
    b=VectorXd::Random(n);


    double stable= a.stableNorm();
    double blue = a.blueNorm();
    double squred= a.squaredNorm();
    double norm = a.norm();
    double hypot = a.hypotNorm();
    double t = 0;

    double absmax=a.cwiseAbs().maxCoeff();

    for (size_t i = 0; i < a.size(); i++)
    {
        t += a(i)*a(i);
    }

    cout<<"real norm:\n";
    cout << "stableNorm=" <<stable<< endl;
    cout << "blueNorm=" << blue << endl;
    cout << "squaredNorm=" << squred << endl;
    cout << "norm=" << norm << endl;
    cout << "hypotnorm=" << hypot << endl;
    cout << t << endl;

    double c = a.dot(b);
    cout<<"real dot:\n";
    cout<<a.dot(b)<<endl;

    VectorXcd c1,c2;
    c1=VectorXcd::Random(n);
    c2=VectorXcd::Random(n);

    cout<<"complex dot:\n";
    cout<<"c1.dot(c2): "<<c1.dot(c2)<<endl;
    cout<<"c1.adjoint()*c2: "<<c1.adjoint()*c2<<endl;
    cout<<"c1.stableNorm(): "<<c1.stableNorm()<<endl;
    cout<<"c1.squreNorm(): "<<c1.squaredNorm()<<endl;
    cout<<"maxCoeff: " << c1.cwiseAbs().maxCoeff()<<endl;;

}

int main()
{
    //test_Eigen();
    test_Eigen_1();
}
