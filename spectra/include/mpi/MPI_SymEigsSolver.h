#pragma once

#include <Eigen/Core>
#include <vector>     // std::vector
#include <cmath>      // std::abs, std::pow, std::sqrt
#include <algorithm>  // std::min, std::copy
#include <stdexcept>  // std::invalid_argument

#include "../Util/TypeTraits.h"
#include "../Util/SelectionRule.h"
#include "../Util/CompInfo.h"
#include "../Util/SimpleRandom.h"
#include "../LinAlg/UpperHessenbergQR.h"
#include "../LinAlg/TridiagEigen.h"

#include "Util/numTraits.h"

namespace Spectra {


template < typename Scalar,
           int SelectionRule,
           typename OpType>
class MPI_SymEigsSolver
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> Array;
    typedef Eigen::Array<bool, Eigen::Dynamic, 1> BoolArray;
    typedef Eigen::Map<Matrix> MapMat;
    typedef Eigen::Map<Vector> MapVec;

    typedef typename numTraits<Scalar>::RealType Real;
    typedef std::complex<Real> Complex;
    typedef Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> ComplexMatrix;
    typedef Eigen::Matrix<Complex, Eigen::Dynamic, 1> ComplexVector;


protected:
    OpType*      m_op;         // object to conduct matrix operation,
    // e.g. matrix-vector product
    const int    m_n;          // dimension of matrix A
    const int    m_nev;        // number of eigenvalues requested
    const int    m_ncv;        // dimension of Krylov subspace in the Lanczos method
    int          m_nmatop;     // number of matrix operations CALLed
    int          m_niter;      // number of restarting iterations

    Matrix       m_fac_V;      // V matrix in the Lanczos factorization
    Matrix       m_fac_H;      // H matrix in the Lanczos factorization
    Vector       m_fac_f;      // residual in the Lanczos factorization
    Vector       m_ritz_val;   // Ritz values

private:
    Matrix       m_ritz_vec;   // Ritz vectors
    Vector       m_ritz_est;   // last row of m_ritz_vec, also CALLed the Ritz estimates
    BoolArray    m_ritz_conv;  // indicator of the convergence of Ritz values
    int          m_info;       // status of the computation

    const Scalar m_near_0;     // a very small value, but 1.0 / m_near_0 does not overflow
    // ~= 1e-307 for the "double" type
    const Scalar m_eps;        // the machine precision, ~= 1e-16 for the "double" type
    const Scalar m_eps23;      // m_eps^(2/3), used to test the convergence

    // Given orthonormal basis functions V, find a nonzero vector f such that V'f = 0
    // Assume that f has been properly allocated
    void expand_basis(const MapMat& V, const int seed, Vector& f, Scalar& fnorm)
    {
        using std::sqrt;

        const Scalar thresh = m_eps * sqrt(Scalar(m_n));
        for(int iter = 0; iter < 5; iter++)
        {
            // Randomly generate a new vector and orthogonalize it against V
            SimpleRandom<Scalar> rng(seed + 123 * (iter+m_op->rank()));
            f.noalias() = rng.random_vec(m_n);
            // f <- f - V * V' * f, so that f is orthogonal to V
            int cols=V.cols();
            for(int idx=0; idx<cols; idx++)
            {
                //CALL MPI
                Scalar a=m_op->pdot(V.col(idx),f);
                f-=a*V.col(idx);
            }
            // fnorm <- ||f||
            //CALL MPI
            fnorm = m_op->pnorm2(m_fac_f);

            // If fnorm is too close to zero, we try a new random vector,
            // otherwise return the result
            if(fnorm >= thresh)
                return;
        }
    }

    // Lanczos factorization starting from step-k
    void factorize_from(int from_k, int to_m, const Vector& fk)
    {
        using std::sqrt;

        if(to_m <= from_k) return;

        const Scalar beta_thresh = m_eps * sqrt(Scalar(m_n));
        m_fac_f.noalias() = fk;

        // Pre-allocate Vf
        Vector Vf(to_m);
        Vector w(m_n);

        //MPI
        Scalar beta = m_op->pnorm2(m_fac_f);

        Scalar Hii = Scalar(0);
        // Keep the upperleft k x k submatrix of H and set other elements to 0
        m_fac_H.rightCols(m_ncv - from_k).setZero();
        m_fac_H.block(from_k, 0, m_ncv - from_k, from_k).setZero();
        for(int i = from_k; i <= to_m - 1; i++)
        {
            bool restart = false;
            // If beta = 0, then the next V is not full rank
            // We need to generate a new residual vector that is orthogonal
            // to the current V, which we CALL a restart
            if(beta < m_near_0)
            {
                MapMat V(m_fac_V.data(), m_n, i); // The first i columns
                expand_basis(V, 2 * i, m_fac_f, beta);
                restart = true;
            }

            // v <- f / ||f||
            MapVec v(&m_fac_V(0, i), m_n); // The (i+1)-th column
            v.noalias() = m_fac_f / beta;

            // Note that H[i+1, i] equals to the unrestarted beta
            m_fac_H(i, i - 1) = restart ? Scalar(0) : beta;

            // w <- A * v
            m_op->perform_op(v.data(), w.data());
            m_nmatop++;

            //CALL MPI
            Hii = m_op->pdot(v, w);

            m_fac_H(i - 1, i) = m_fac_H(i, i - 1); // Due to symmetry
            m_fac_H(i, i) = Hii;

            // f <- w - V * V' * w = w - H[i+1, i] * V{i} - H[i+1, i+1] * V{i+1}
            // If restarting, we know that H[i+1, i] = 0
            if(restart)
                m_fac_f.noalias() = w - Hii * v;
            else
                m_fac_f.noalias() = w - m_fac_H(i, i - 1) * m_fac_V.col(i - 1) - Hii * v;

            //CALL MPI
            beta = m_op->pnorm2(m_fac_f);

            // f/||f|| is going to be the next column of V, so we need to test
            // whether V' * (f/||f||) ~= 0
            const int i1 = i + 1;
            MapMat V(m_fac_V.data(), m_n, i1); // The first (i+1) columns

            for(int idx=0; idx<i1; idx++)
            {
                //CALL MPI
                Vf(idx)= m_op->pdot(V.col(idx), m_fac_f);
            }

            Scalar ortho_err = Vf.head(i1).cwiseAbs().maxCoeff();
            // If not, iteratively correct the residual
            int count = 0;
            while(count < 5 && ortho_err > m_eps * beta)
            {
                // There is an edge case: when beta=||f|| is close to zero, f mostly consists
                // of noises of rounding errors, so the test [ortho_err < eps * beta] is very
                // likely to fail. In particular, if beta=0, then the test is ensured to fail.
                // Hence when this happens, we force f to be zero, and then restart in the
                // next iteration.
                if(beta < beta_thresh)
                {
                    m_fac_f.setZero();
                    beta = Scalar(0);
                    break;
                }

                // f <- f - V * Vf
                m_fac_f.noalias() -= V * Vf.head(i1);
                // h <- h + Vf
                m_fac_H(i - 1, i) += Vf[i - 1];
                m_fac_H(i, i - 1) = m_fac_H(i - 1, i);
                m_fac_H(i, i) += Vf[i];
                // beta <- ||f||

                //CALL MPI
                beta = m_op->pnorm2(m_fac_f);


                for(int idx=0; idx<V.cols(); idx++)
                {
                    //CALL MPI
                    Vf(idx) =m_op->pdot(V.col(idx), m_fac_f);
                }
                ortho_err = Vf.head(i1).cwiseAbs().maxCoeff();
                count++;
            }
        }
    }

    // Implicitly restarted Lanczos factorization
    void restart(int k)
    {
        if(k >= m_ncv)
            return;

        TridiagQR<Scalar> decomp;
        Matrix Q = Matrix::Identity(m_ncv, m_ncv);

        for(int i = k; i < m_ncv; i++)
        {
            // QR decomposition of H-mu*I, mu is the shift
            m_fac_H.diagonal().array() -= m_ritz_val[i];
            decomp.compute(m_fac_H);

            // Q -> Q * Qi
            decomp.apply_YQ(Q);
            // H -> Q'HQ
            // Since QR = H - mu * I, we have H = QR + mu * I
            // and therefore Q'HQ = RQ + mu * I
            decomp.matrix_RQ(m_fac_H);
            m_fac_H.diagonal().array() += m_ritz_val[i];
        }
        // V -> VQ, only need to update the first k+1 columns
        // Q has some elements being zero
        // The first (ncv - k + i) elements of the i-th column of Q are non-zero
        Matrix Vs(m_n, k + 1);
        for(int i = 0; i < k; i++)
        {
            const int nnz = m_ncv - k + i + 1;
            MapVec q(&Q(0, i), nnz);
            Vs.col(i).noalias() = m_fac_V.leftCols(nnz) * q;
        }
        Vs.col(k).noalias() = m_fac_V * Q.col(k);
        m_fac_V.leftCols(k + 1).noalias() = Vs;

        const Vector fk = m_fac_f * Q(m_ncv - 1, k - 1) + m_fac_V.col(k) * m_fac_H(k, k - 1);
        factorize_from(k, m_ncv, fk);
        retrieve_ritzpair();
    }

    // Calculates the number of converged Ritz values
    int num_converged(Scalar tol)
    {
        // thresh = tol * max(m_eps23, abs(theta)), theta for Ritz value
        Array thresh = tol * m_ritz_val.head(m_nev).array().abs().max(m_eps23);
        Array resid =  m_ritz_est.head(m_nev).array().abs() * norm(m_fac_f);
        // Converged "wanted" Ritz values
        m_ritz_conv = (resid < thresh);

        return m_ritz_conv.cast<int>().sum();
    }

    // Returns the adjusted nev for restarting
    int nev_adjusted(int nconv)
    {
        using std::abs;

        int nev_new = m_nev;
        for(int i = m_nev; i < m_ncv; i++)
            if(abs(m_ritz_est[i]) < m_near_0)  nev_new++;

        // Adjust nev_new, according to dsaup2.f line 677~684 in ARPACK
        nev_new += std::min(nconv, (m_ncv - nev_new) / 2);
        if(nev_new == 1 && m_ncv >= 6)
            nev_new = m_ncv / 2;
        else if(nev_new == 1 && m_ncv > 2)
            nev_new = 2;

        if(nev_new > m_ncv - 1)
            nev_new = m_ncv - 1;

        return nev_new;
    }

    // Retrieves and sorts Ritz values and Ritz vectors
    void retrieve_ritzpair()
    {
        TridiagEigen<Scalar> decomp(m_fac_H);
        const Vector& evals = decomp.eigenvalues();
        const Matrix& evecs = decomp.eigenvectors();

        SortEigenvalue<Scalar, SelectionRule> sorting(evals.data(), evals.size());
        std::vector<int> ind = sorting.index();

        // For BOTH_ENDS, the eigenvalues are sorted according
        // to the LARGEST_ALGE rule, so we need to move those smallest
        // values to the left
        // The order would be
        // Largest => Smallest => 2nd largest => 2nd smallest => ...
        // We keep this order since the first k values will always be
        // the wanted collection, no matter k is nev_updated (used in restart())
        // or is nev (used in sort_ritzpair())
        if(SelectionRule == BOTH_ENDS)
        {
            std::vector<int> ind_copy(ind);
            for(int i = 0; i < m_ncv; i++)
            {
                // If i is even, pick values from the left (large values)
                // If i is odd, pick values from the right (small values)
                if(i % 2 == 0)
                    ind[i] = ind_copy[i / 2];
                else
                    ind[i] = ind_copy[m_ncv - 1 - i / 2];
            }
        }

        // Copy the Ritz values and vectors to m_ritz_val and m_ritz_vec, respectively
        for(int i = 0; i < m_ncv; i++)
        {
            m_ritz_val[i] = evals[ind[i]];
            m_ritz_est[i] = evecs(m_ncv - 1, ind[i]);
        }
        for(int i = 0; i < m_nev; i++)
        {
            m_ritz_vec.col(i).noalias() = evecs.col(ind[i]);
        }
    }

protected:
    // In generalized eigenvalue problem Ax=lambda*Bx, define the inner product to be <x, y> = x'By
    // For regular eigenvalue problems, it is the usual inner product <x, y> = x'y
    virtual Scalar inner_product(const Vector& x, const Vector& y) {
        return x.dot(y);
    }
    virtual Scalar inner_product(const MapVec& x, const Vector& y) {
        return x.dot(y);
    }
    virtual Vector inner_product(const MapMat& x, const Vector& y) {
        return x.transpose() * y;
    }

    // B-norm of a vector. For regular eigenvalue problems it is simply the L2 norm
    virtual Scalar norm(const Vector& x) {
        return x.norm();
    }

    // Sorts the first nev Ritz pairs in the specified order
    // This is used to return the final results
    virtual void sort_ritzpair(int sort_rule)
    {
        // First make sure that we have a valid index vector
        SortEigenvalue<Scalar, LARGEST_ALGE> sorting(m_ritz_val.data(), m_nev);
        std::vector<int> ind = sorting.index();

        switch(sort_rule)
        {
        case LARGEST_ALGE:
            break;
        case LARGEST_MAGN:
        {
            SortEigenvalue<Scalar, LARGEST_MAGN> sorting(m_ritz_val.data(), m_nev);
            ind = sorting.index();
        }
        break;
        case SMALLEST_ALGE:
        {
            SortEigenvalue<Scalar, SMALLEST_ALGE> sorting(m_ritz_val.data(), m_nev);
            ind = sorting.index();
        }
        break;
        case SMALLEST_MAGN:
        {
            SortEigenvalue<Scalar, SMALLEST_MAGN> sorting(m_ritz_val.data(), m_nev);
            ind = sorting.index();
        }
        break;
        default:
            throw std::invalid_argument("unsupported sorting rule");
        }

        Vector new_ritz_val(m_ncv);
        Matrix new_ritz_vec(m_ncv, m_nev);
        BoolArray new_ritz_conv(m_nev);

        for(int i = 0; i < m_nev; i++)
        {
            new_ritz_val[i] = m_ritz_val[ind[i]];
            new_ritz_vec.col(i).noalias() = m_ritz_vec.col(ind[i]);
            new_ritz_conv[i] = m_ritz_conv[ind[i]];
        }

        m_ritz_val.swap(new_ritz_val);
        m_ritz_vec.swap(new_ritz_vec);
        m_ritz_conv.swap(new_ritz_conv);
    }

public:
    ///
    /// Constructor to create a solver object.
    ///
    /// \param op   Pointer to the matrix operation object, which should implement
    ///             the matrix-vector multiplication operation of \f$A\f$:
    ///             calculating \f$Av\f$ for any vector \f$v\f$. Users could either
    ///             create the object from the wrapper class such as DenseSymMatProd, or
    ///             define their own that impelements all the public member functions
    ///             as in DenseSymMatProd.
    /// \param nev  Number of eigenvalues requested. This should satisfy \f$1\le nev \le n-1\f$,
    ///             where \f$n\f$ is the size of matrix.
    /// \param ncv  Parameter that controls the convergence speed of the algorithm.
    ///             TypiCALLy a larger `ncv` means faster convergence, but it may
    ///             also result in greater memory use and more matrix operations
    ///             in each iteration. This parameter must satisfy \f$nev < ncv \le n\f$,
    ///             and is advised to take \f$ncv \ge 2\cdot nev\f$.
    ///
    MPI_SymEigsSolver(OpType* op, int nev, int ncv) :
        m_op(op),
        m_n(m_op->rows()),
        m_nev(nev),
        m_ncv(ncv > m_n ? m_n : ncv),
        m_nmatop(0),
        m_niter(0),
        m_info(NOT_COMPUTED),
        m_near_0(TypeTraits<Scalar>::min() * Scalar(10)),
        m_eps(Eigen::NumTraits<Scalar>::epsilon()),
        m_eps23(Eigen::numext::pow(m_eps, Scalar(2.0) / 3))
    {
        if(nev < 1 || nev > m_n - 1)
            throw std::invalid_argument("nev must satisfy 1 <= nev <= n - 1, n is the size of matrix");

        if(ncv <= nev || ncv > m_n)
            throw std::invalid_argument("ncv must satisfy nev < ncv <= n, n is the size of matrix");
    }

    ///
    /// Virtual destructor
    ///
    virtual ~MPI_SymEigsSolver() {}

    ///
    /// Initializes the solver by providing an initial residual vector.
    ///
    /// \param init_resid Pointer to the initial residual vector.
    ///
    /// **Spectra** (and also **ARPACK**) uses an iterative algorithm
    /// to find eigenvalues. This function allows the user to provide the initial
    /// residual vector.
    ///
    void init(const Scalar* init_resid)
    {
        // Reset all matrices/vectors to zero
        m_fac_V.resize(m_n, m_ncv);
        m_fac_H.resize(m_ncv, m_ncv);
        m_fac_f.resize(m_n);
        m_ritz_val.resize(m_ncv);
        m_ritz_vec.resize(m_ncv, m_nev);
        m_ritz_est.resize(m_ncv);
        m_ritz_conv.resize(m_nev);

        m_fac_V.setZero();
        m_fac_H.setZero();
        m_fac_f.setZero();
        m_ritz_val.setZero();
        m_ritz_vec.setZero();
        m_ritz_est.setZero();
        m_ritz_conv.setZero();

        m_nmatop = 0;
        m_niter = 0;

        // Set the initial vector
        Vector v(m_n);
        std::copy(init_resid, init_resid + m_n, v.data());

        //CALL MPI
        const Scalar vnorm = m_op->pnorm2(v);
        if(vnorm < m_near_0)
            throw std::invalid_argument("initial residual vector cannot be zero");
        v /= vnorm;

        Vector w(m_n);
        m_op->perform_op(v.data(), w.data());
        m_nmatop++;

        //CALL MPI
        m_fac_H(0, 0) = m_op->pdot(v, w);

        m_fac_f.noalias() = w - v * m_fac_H(0, 0);
        m_fac_V.col(0).noalias() = v;

        // In some cases f is zero in exact arithmetics, but due to rounding errors
        // it may contain tiny fluctuations. When this happens, we force f to be zero.
        //CALL MPI
        if(m_op->pabsmax(m_fac_f) < m_eps)
            m_fac_f.setZero();
    }

    ///
    /// Initializes the solver by providing a random initial residual vector.
    ///
    /// This overloaded function generates a random initial residual vector
    /// (with a fixed random seed) for the algorithm. Elements in the vector
    /// follow independent Uniform(-0.5, 0.5) distribution.
    ///
    void init()
    {
        SimpleRandom<Scalar> rng(0);
        Vector init_resid = rng.random_vec(m_n);
        init(init_resid.data());
    }

    ///
    /// Conducts the major computation procedure.
    ///
    /// \param maxit      Maximum number of iterations allowed in the algorithm.
    /// \param tol        Precision parameter for the calculated eigenvalues.
    /// \param sort_rule  Rule to sort the eigenvalues and eigenvectors.
    ///                   Supported values are
    ///                   `Spectra::LARGEST_ALGE`, `Spectra::LARGEST_MAGN`,
    ///                   `Spectra::SMALLEST_ALGE` and `Spectra::SMALLEST_MAGN`,
    ///                   for example `LARGEST_ALGE` indicates that largest eigenvalues
    ///                   come first. Note that this argument is only used to
    ///                   **sort** the final result, and the **selection** rule
    ///                   (e.g. selecting the largest or smallest eigenvalues in the
    ///                   full spectrum) is specified by the template parameter
    ///                   `SelectionRule` of SymEigsSolver.
    ///
    /// \return Number of converged eigenvalues.
    ///
    int compute(int maxit = 1000, Scalar tol = 1e-10, int sort_rule = LARGEST_ALGE)
    {
        // The m-step Arnoldi factorization
        factorize_from(1, m_ncv, m_fac_f);

        retrieve_ritzpair();
        // Restarting
        int i, nconv = 0, nev_adj;
        for(i = 0; i < maxit; i++)
        {
            nconv = num_converged(tol);
            if(nconv >= m_nev)
                break;

            nev_adj = nev_adjusted(nconv);
            restart(nev_adj);
        }
        // Sorting results
        sort_ritzpair(sort_rule);

        m_niter += i + 1;
        m_info = (nconv >= m_nev) ? SUCCESSFUL : NOT_CONVERGING;

        return std::min(m_nev, nconv);
    }

    ///
    /// Returns the status of the computation.
    /// The full list of enumeration values can be found in \ref Enumerations.
    ///
    int info() const {
        return m_info;
    }

    ///
    /// Returns the number of iterations used in the computation.
    ///
    int num_iterations() const {
        return m_niter;
    }

    ///
    /// Returns the number of matrix operations used in the computation.
    ///
    int num_operations() const {
        return m_nmatop;
    }

    ///
    /// Returns the converged eigenvalues.
    ///
    /// \return A vector containing the eigenvalues.
    /// Returned vector type will be `Eigen::Vector<Scalar, ...>`, depending on
    /// the template parameter `Scalar` defined.
    ///
    Vector eigenvalues() const
    {
        const int nconv = m_ritz_conv.cast<int>().sum();
        Vector res(nconv);

        if(!nconv)
            return res;

        int j = 0;
        for(int i = 0; i < m_nev; i++)
        {
            if(m_ritz_conv[i])
            {
                res[j] = m_ritz_val[i];
                j++;
            }
        }

        return res;
    }

    ///
    /// Returns the eigenvectors associated with the converged eigenvalues.
    ///
    /// \param nvec The number of eigenvectors to return.
    ///
    /// \return A matrix containing the eigenvectors.
    /// Returned matrix type will be `Eigen::Matrix<Scalar, ...>`,
    /// depending on the template parameter `Scalar` defined.
    ///
    virtual Matrix eigenvectors(int nvec) const
    {
        const int nconv = m_ritz_conv.cast<int>().sum();
        nvec = std::min(nvec, nconv);
        Matrix res(m_n, nvec);

        if(!nvec)
            return res;

        Matrix ritz_vec_conv(m_ncv, nvec);
        int j = 0;
        for(int i = 0; i < m_nev && j < nvec; i++)
        {
            if(m_ritz_conv[i])
            {
                ritz_vec_conv.col(j).noalias() = m_ritz_vec.col(i);
                j++;
            }
        }

        res.noalias() = m_fac_V * ritz_vec_conv;

        return res;
    }

    ///
    /// Returns all converged eigenvectors.
    ///
    virtual Matrix eigenvectors() const
    {
        return eigenvectors(m_nev);
    }
};


} // namespace Spectra
