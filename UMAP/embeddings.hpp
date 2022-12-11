#ifndef EMBEDDINGS_HPP
#define EMBEDDINGS_HPP

#include <random>
#include <algorithm>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>

#include "utils.hpp"

#include <ezarpack/arpack_solver.hpp>
#include <ezarpack/storages/eigen.hpp>

/// For initialize_embedding()
class Spectral
{

};

class Random
{

};

auto spectral_layout(const Eigen::SparseMatrix<float> &graph, int embed_dim) -> Eigen::MatrixXf
{
    /// The operation "Eigen::VectorXf D_ = graph.rowwise().sum()" doesn't work, so we have to do...
    Eigen::VectorXf D_ = graph * Eigen::VectorXf::Ones(graph.cols());

    /// inv(sqrt(D_))
    Eigen::VectorXf D = D_.array().sqrt().inverse();

    /// I - D * graph * D
    Eigen::SparseMatrix<float> L = Eigen::SparseMatrix<float>(Eigen::VectorXf::Ones(graph.cols()).asDiagonal()) - D.asDiagonal() * graph * D.asDiagonal();

    int k = embed_dim + 1;
    int num_lanczos_vectors = std::max(2 * k + 1, static_cast<int>(std::round(std::sqrt(L.rows()))));

    using solver_t = ezarpack::arpack_solver<ezarpack::Symmetric, ezarpack::eigen_storage>;
    solver_t solver(graph.cols());

    using params_t = solver_t::params_t;
    params_t params(k, params_t::SmallestMagnitude, true);
    params.ncv = num_lanczos_vectors;
    params.random_residual_vector = false;
    params.tolerance = 1e-4;
    params.max_iter = L.rows() * 5;

    auto rv = solver.residual_vector();
    for (int i = 0; i < L.rows(); ++i)
        rv[i] = 1.0;

    using vector_view_t = solver_t::vector_view_t;
    using vector_const_view_t = solver_t::vector_const_view_t;

    Eigen::SparseMatrix<double> L_d = L.cast<double>();

    auto Aop = [&L_d](vector_const_view_t in, vector_view_t out)
    {
        //out = (L * in.cast<float>()).cast<double>();
        out = L_d * in;
    };

    solver(Aop, params);

    Eigen::VectorXf lambda = solver.eigenvalues().cast<float>();
    //Eigen::MatrixXf vecs = solver.eigenvectors().cast<float>();
    Eigen::MatrixXf vecs(graph.rows(), embed_dim);

    for (int i = 0; i < embed_dim; ++i)
        vecs.col(i) = solver.eigenvectors().cast<float>().col(i + 1);

    Eigen::MatrixXf layout = vecs.transpose();

    return layout;

}

auto initialize_embedding(Eigen::SparseMatrix<float> &graph, int n_components, Spectral spectral) -> Eigen::MatrixXf
{

    Eigen::MatrixXf embed = spectral_layout(graph, n_components);

    float expansion = 10.0f / embed.maxCoeff();

    /// Random seed for normal distributed values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> normal_values(0.0f, 1.0f);

    Eigen::MatrixXf normalValues(embed.rows(), embed.cols());
    for (int i = 0; i < embed.rows(); ++i)
    {
        for (int j = 0; j < embed.cols(); ++j)
        {
            normalValues(i, j) = normal_values(gen);
        }
    }

    embed = embed * expansion + (1.0f / 10000.0f) * normalValues;

    return embed;

}

/*auto initialize_embedding(Eigen::SparseMatrix<float> graph, int n_components, Random random) -> Eigen::MatrixXf
{

}*/

auto square_euclidean(Eigen::Ref<Eigen::VectorXf> x, Eigen::Ref<Eigen::VectorXf> y) -> float
{

    float distance = 0.0f;

    for (int i = 0; i < x.rows(); ++i)
    {
        distance += std::pow(x(i) - y(i), 2.0f);
    }

    return distance;
}

auto optimize_embedding(const Eigen::SparseMatrix<float> &graph, Eigen::MatrixXf &query_embedding, Eigen::MatrixXf &ref_embedding, int n_epochs, float initial_alpha, float min_dist, float spread, float gamma, int neg_sample_rate, std::optional<float> _a = {}, std::optional<float> _b = {}, bool move_ref = false) -> Eigen::MatrixXf
{

    auto [a, b] = fit_ab(min_dist, spread, _a, _b);


    bool self_reference = &query_embedding == &ref_embedding;
    float alpha = initial_alpha;

    for (int e = 0; e < n_epochs; ++e)
    {
        /// TODO: verify that we are iterating correctly
        for (int i = 0; i < graph.outerSize(); ++i)
        {
            for (Eigen::SparseMatrix<float>::InnerIterator it(graph, i); it; ++it)
            {
                int j = it.row();
                float p = it.value();

                /// Random seed for uniform distributed values
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<float> uniform_values(0.0f, 1.0f);

                if (p > uniform_values(gen))
                {
                    float sdist = square_euclidean(query_embedding.col(i), ref_embedding.col(j));

                    float delta = (-2.0f * a * b * std::pow(sdist, b - 1.0f)) / (1.0f + a * std::pow(sdist, b));

                    for (int d = 0; d < query_embedding.rows(); ++d)
                    {
                        float grad = std::clamp(delta * (query_embedding(d, i) - ref_embedding(d, j)), -4.0f, 4.0f);

                        query_embedding(d, i) += alpha * grad;

                        if (move_ref == true)
                            ref_embedding(d, j) -= alpha * grad;
                    }

                    for (int sample_rate = 0; sample_rate < neg_sample_rate; ++sample_rate)
                    {
                        std::uniform_int_distribution<> uniform_range(0, ref_embedding.cols() - 1);

                        int k = uniform_range(gen);

                        if (i == k && self_reference == true)
                            continue;

                        sdist = square_euclidean(query_embedding.col(i), ref_embedding.col(k));

                        delta = (2.0 * gamma * b) / ((1.0f / 1000.0f + sdist) * (1.0f + a * std::pow(sdist, b)));

                        for (int d = 0; d < query_embedding.rows(); ++d)
                        {
                            float grad = 4.0f;

                            if (delta > 0.0f)
                                grad = std::clamp(delta * (query_embedding(d, i) - ref_embedding(d, k)), -4.0f, 4.0f);

                            query_embedding(d, i) += alpha * grad;
                        }
                    }
                }
            }
        }

        alpha = initial_alpha * (1.0f - e / static_cast<float>(n_epochs));
    }

    return query_embedding;

}

#endif // EMBEDDINGS_HPP
