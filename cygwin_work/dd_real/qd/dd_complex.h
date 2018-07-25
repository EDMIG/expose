#pragma once
#include <complex>
#include "dd_real.h"


struct dd_complex {
    dd_real re, im;
    //constructor
    dd_complex();
    dd_complex(const dd_complex & a);
    dd_complex(const dd_real & a, const dd_real & b);
    dd_complex(const dd_real & a);
    dd_complex(const std::complex<double> & a);

    //extraction of real and imaginary parts
    const dd_real & real() const
    {
        return re;
    }
    const dd_real & imag() const
    {
        return im;
    }
    void real(const dd_real r)
    {
        this->re = r;
    }
    void imag(const dd_real r)
    {
        this->im = r;
    }

    //comparison
    friend bool operator==(const dd_complex & a, const dd_complex & b);
    friend bool operator==(const dd_complex & a, const dd_real & b);
    friend bool operator==(const dd_real & a, const dd_complex & b);

    friend bool operator!=(const dd_complex & a, const dd_complex & b);
    friend bool operator!=(const dd_complex & a, const dd_real & b);
    friend bool operator!=(const dd_real & a, const dd_complex & b);

    //subsututtion
    //difficult to implement; dd_complex& operator=(const dd_complex& b);
    dd_complex & operator=(const std::complex < double >b);
    dd_complex & operator=(const dd_real b);
    dd_complex & operator=(const double b);

    //addition
    dd_complex & operator+=(const dd_complex & b);
    dd_complex & operator+=(const dd_real & b);
    const dd_complex operator+() const;

    //subtraction
    dd_complex & operator-=(const dd_complex & b);
    dd_complex & operator-=(const dd_real & b);
    const dd_complex operator-() const;

    //multiplication
    dd_complex & operator*=(const dd_complex & b);
    dd_complex & operator*=(const dd_real & b);

    //division
    dd_complex & operator/=(const dd_complex & b);
    dd_complex & operator/=(const dd_real & b);

    //operators
    operator std::complex<double>() const;

    std::string to_string(int precision = dd_real::_ndigits, int width = 0,
                          std::ios_base::fmtflags fmt = static_cast<std::ios_base::fmtflags>(0),
                          bool showpos = false, bool uppercase = false, char fill = ' ') const;
};

const dd_complex operator+(const dd_complex & a, const dd_complex & b);
const dd_complex operator+(const dd_complex & a, const dd_real & b);
const dd_complex operator+(const dd_real & a, const dd_complex & b);
const dd_complex operator+(const dd_complex & a, const std::complex < double >&b);
const dd_complex operator+(const std::complex < double >&a, const dd_complex & b);

const dd_complex operator-(const dd_complex & a, const dd_complex & b);
const dd_complex operator-(const dd_complex & a, const dd_real & b);
const dd_complex operator-(const dd_real & a, const dd_complex & b);
const dd_complex operator-(const dd_complex & a, std::complex < double >&b);
const dd_complex operator-(std::complex < double >&a, const dd_complex & b);

const dd_complex operator*(const dd_complex & a, const dd_complex & b);
const dd_complex operator*(const dd_complex & a, const dd_real & b);
const dd_complex operator*(const dd_real & a, const dd_complex & b);
const dd_complex operator*(const dd_complex & a, std::complex < double >&b);
const dd_complex operator*(std::complex < double >&a, const dd_complex & b);

const dd_complex operator/(const dd_complex & a, const dd_complex & b);
const dd_complex operator/(const dd_complex & a, const dd_real & b);
const dd_complex operator/(const dd_real & a, const dd_complex & b);
const dd_complex operator/(const dd_complex & a, std::complex < double >&b);
const dd_complex operator/(std::complex < double >&a, const dd_complex & b);

//constructor
inline dd_complex::dd_complex()
{
    re = 0.0;
    im = 0.0;
}

inline dd_complex::dd_complex(const dd_complex & a)
{
    re = a.re;
    im = a.im;
}

inline dd_complex::dd_complex(const dd_real & a, const dd_real & b)
{
    re = a;
    im = b;
}

inline dd_complex::dd_complex(const dd_real & a)
{
    re = a;
    im = 0.0;
}

inline dd_complex::dd_complex(const std::complex<double>& a)
{
    re = a.real();
    im = a.imag();
}

//comparison
inline bool operator==(const dd_complex & a, const dd_complex & b)
{
    return (a.re == b.re) && (a.im == b.im);
}

inline bool operator==(const dd_real & a, const dd_complex & b)
{
    return (a == b.re) && (b.im == 0.0);
}

inline bool operator==(const dd_complex & a, const dd_real & b)
{
    return (a.re == b) && (a.im == 0.0);
}

inline bool operator!=(const dd_complex & a, const dd_complex & b)
{
    return (a.re != b.re) || (a.im != b.im);
}

inline bool operator!=(const dd_real & a, const dd_complex & b)
{
    return (a != b.re) || (b.im != 0.0);
}

inline bool operator!=(const dd_complex & a, const dd_real & b)
{
    return (a.re != b) || (a.im != 0.0);
}

inline dd_complex & dd_complex::operator=(const std::complex < double >b)
{
    re = b.real();
    im = b.imag();
    return *this;
}

inline dd_complex & dd_complex::operator=(const dd_real b)
{
    re = b;
    im = 0.0;
    return *this;
}

inline dd_complex & dd_complex::operator=(const double b)
{
    re = b;
    im = 0.0;
    return *this;
}

inline dd_complex & dd_complex::operator+=(const dd_complex & b)
{
    re += b.re;
    im += b.im;
    return *this;
}

inline dd_complex & dd_complex::operator+=(const dd_real & b)
{
    re += b;
    return *this;
}

inline const dd_complex dd_complex::operator+() const
{
    return dd_complex(*this);
}

inline const dd_complex operator+(const dd_complex & a, const dd_complex & b)
{
    dd_complex tmp(a);
    tmp += b;
    return tmp;
}

inline const dd_complex operator+(const dd_complex & a, const std::complex<double> & b)
{
    dd_complex tmp(b);
    tmp += a;
    return tmp;
}

inline const dd_complex operator+(const std::complex<double> & a, const dd_complex &b)
{
    dd_complex tmp(a);
    tmp += b;
    return tmp;
}

inline const dd_complex operator+(const dd_complex & a, const dd_real & b)
{
    dd_complex tmp(b);
    tmp += a;
    return tmp;
}

inline const dd_complex operator+(const dd_real & a, const dd_complex & b)
{
    dd_complex tmp(a);
    tmp += b;
    return tmp;
}

inline dd_complex & dd_complex::operator-=(const dd_complex & b)
{
    re -= b.re;
    im -= b.im;
    return *this;
}

inline dd_complex & dd_complex::operator-=(const dd_real & b)
{
    re -= b;
    return *this;
}

inline const dd_complex dd_complex::operator-() const
{
    dd_complex tmp;
    tmp.re = -re;
    tmp.im = -im;
    return tmp;
}

inline const dd_complex operator-(const dd_complex & a, const dd_complex & b)
{
    dd_complex tmp(a);
    return tmp -= b;
}

//XXX can overflow
inline dd_complex & dd_complex::operator*=(const dd_complex & b)
{
    dd_complex tmp(*this);
    re = tmp.re * b.re - tmp.im * b.im;
    im = tmp.re * b.im + tmp.im * b.re;
    return (*this);
}

inline dd_complex & dd_complex::operator*=(const dd_real & b)
{
    dd_complex tmp(*this);
    re = tmp.re * b;
    im = tmp.im * b;
    return (*this);
}

inline const dd_complex operator*(const dd_complex & a, const dd_complex & b)
{
    dd_complex tmp(a);
    tmp *= b;
    return tmp;
}

inline dd_complex & dd_complex::operator/=(const dd_complex & b)
{
    dd_complex tmp(*this);
    dd_real abr, abi, ratio, den;

    if ((abr = b.re) < 0.)
        abr = -abr;
    if ((abi = b.im) < 0.)
        abi = -abi;
    if (abr <= abi) {
        if (abi == 0) {
            if (tmp.im != 0 || tmp.re != 0)
                abi = 1.;
            tmp.im = tmp.re = abi / abr;
            return (*this);
        }
        ratio = b.re / b.im;
        den = b.im * (1.0 + ratio * ratio);
        re = (tmp.re * ratio + tmp.im) / den;
        im = (tmp.im * ratio - tmp.re) / den;
    }
    else {
        ratio = b.im / b.re;
        den = b.re * (1.0 + ratio * ratio);
        re = (tmp.re + tmp.im * ratio) / den;
        im = (tmp.im - tmp.re * ratio) / den;
    }
    return (*this);
}

inline dd_complex & dd_complex::operator/=(const dd_real & b)
{
    dd_complex tmp(*this);

    //20180629,CXG,Ô­³ÌÐòÓÐ´í£¡
    //re = (tmp.re * b);
    //im = (tmp.im * b);

    re = tmp.re /b;
    im = tmp.im/ b;

    return (*this);
}

inline dd_complex::operator std::complex<double>() const
{
    std::complex<double> p;
    p.real(re.x[0]);
    p.imag(im.x[0]);
    return p;
}

inline const dd_complex operator/(const dd_complex & a, const dd_complex & b)
{
    dd_complex tmp(a);
    tmp /= b;
    return tmp;
}
inline const dd_complex operator/(const dd_complex & a, const dd_real & b)
{
    dd_complex tmp;
    tmp.real(a.real() / b);
    tmp.imag(a.imag() / b);
    return tmp;
}
inline const dd_complex operator/(const dd_real & a, const dd_complex & b)
{
    dd_complex tmp(a);
    tmp /= b;
    return tmp;
}

inline const dd_complex operator*(const dd_complex & a, const dd_real & b)
{
    dd_complex p;
    p.real(a.real() * b);
    p.imag(a.imag() * b);
    return p;
}
inline const dd_complex operator*(const dd_real & a, const dd_complex & b)
{
    dd_complex p;
    p.real(a * b.real());
    p.imag(a * b.imag());
    return p;
}

inline const dd_complex operator-(const dd_complex & a, const dd_real & b)
{
    dd_complex p;
    p.real(a.real() - b);
    p.imag(a.imag());
    return p;
}
inline const dd_complex operator-(const dd_real & a, const dd_complex & b)
{
    dd_complex p;
    p.real(a - b.real());
    p.imag(-b.imag());
    return p;
}

//XXX can overflow
inline dd_real abs(dd_complex ctmp)
{
    return sqrt(ctmp.real() * ctmp.real() + ctmp.imag() * ctmp.imag());
}

inline dd_complex sqrt(dd_complex z)
{
    dd_real mag;
    dd_complex r;

    mag = abs(z);
    if (abs(mag) == 0.0) {
        r.real(0.0), r.imag(0.0);
    }
    else if (z.real() > 0.0) {
        r.real(sqrt(0.5 * (mag + z.real())));
        r.imag(z.imag() / (2.0 * r.real()));
    }
    else {
        r.imag(sqrt(0.5 * (mag - z.real())));
        if (z.imag() < 0.0)
            r.imag(-r.imag());
        r.real(z.imag() / (2.0 * r.imag()));
    }
    return r;
}

inline dd_complex conj(dd_complex ctmp)
{
    dd_complex ctmp2;
    ctmp2.real(ctmp.real());
    ctmp2.imag(-ctmp.imag());
    return ctmp2;
}

string dd_complex::to_string(int precision, int width, ios_base::fmtflags fmt,
                             bool showpos, bool uppercase, char fill) const {

    return "(" + re.to_string(precision, width, fmt,
                              showpos, uppercase, fill) + "," + im.to_string(precision, width, fmt,
                                      showpos, uppercase, fill) + ")";

}
