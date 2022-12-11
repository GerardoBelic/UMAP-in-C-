#ifndef UMAP__HPP
#define UMAP__HPP

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <optional>
#include <tuple>
#include <cmath>
#include <variant>
#include <limits>
#include <chrono>

#include "utils.hpp"
#include "embeddings.hpp"

#define SMOOTH_K_TOLERANCE 1e-5f
#define OFFSET 1

auto smooth_knn_dists(Eigen::Ref<Eigen::VectorXf> dists, float rho, float k, float bandwidth, int niter) -> float
{

    float target = std::log2(k) * bandwidth;

    float lo = 0.0f;
    float mid = 1.0f;
    float hi = std::numeric_limits<float>::max();

    for (int n = 0; n < niter; ++n)
    {
        Eigen::VectorXf max_dist_zero = (dists.array() - rho).array().max(0.0f);
        Eigen::VectorXf dist_mid = (-max_dist_zero / mid);
        Eigen::VectorXf exp_dist = dist_mid.array().exp();
        float psum = exp_dist.sum();

        if (std::abs(psum - target) < SMOOTH_K_TOLERANCE)
            break;

        if (psum > target)
        {
            hi = mid;
            mid = (lo + hi) / 2.0f;
        }
        else
        {
            lo = mid;

            if (hi == std::numeric_limits<float>::max())
                mid *= 2.0f;
            else
                mid = (lo + hi) / 2.0f;
        }
    }

    return mid;

}

auto smooth_knn_dists(Eigen::MatrixXf &knn_dists, int k, float local_connectivity, int niter = 64, float bandwidth = 1.0f) -> std::tuple<std::vector<float>, std::vector<float>>
{

    std::vector<float> rhos(knn_dists.cols());
    std::vector<float> sigmas(knn_dists.cols());

    for (int i = 0; i < knn_dists.cols(); ++i)
    {
        Eigen::VectorXf nz_dists = knn_dists.col(i);

        if (nz_dists.rows() >= static_cast<int>(local_connectivity))
        {
            int index = static_cast<int>(std::floor(local_connectivity)) - OFFSET;

            float interpolation = local_connectivity - static_cast<float>(index + OFFSET);

            if (index >= 0)
            {
                rhos[i] = nz_dists(index);

                if (interpolation > SMOOTH_K_TOLERANCE)
                {
                    rhos[i] += interpolation * (nz_dists[index + 1] - nz_dists[index]);
                }
            }
            else
            {
                rhos[i] = interpolation * nz_dists[0];
            }
        }
        else if (nz_dists.rows() > 0)
        {
            /// Since distances are sorted, in theory we can select the last value
            rhos[i] = nz_dists.maxCoeff();
        }

        sigmas[i] = smooth_knn_dists(knn_dists.col(i), rhos[i], k, bandwidth, niter);
    }

    return {rhos, sigmas};

}

auto compute_membership_strenghs(Eigen::MatrixXi &knns, Eigen::MatrixXf &dists, std::vector<float> &rhos, std::vector<float> &sigmas) -> std::tuple<std::vector<int>, std::vector<int>, std::vector<float>>
{

    std::vector<int> rows(knns.rows() * knns.cols());
    std::vector<int> cols(knns.rows() * knns.cols());
    std::vector<float> vals(knns.rows() * knns.cols());

    for (int i = 0; i < knns.cols(); ++i)
    {
        for (int j = 0; j < knns.rows(); ++j)
        {
            /// Here we should have performed a condition to check if a knn column contains itself,
            /// but in knn_search we alredy took care of that case
            double d = std::exp(-std::max(dists(j, i) - rhos[i], 0.0f) / sigmas[i]);

            rows[i * knns.rows() + j] = knns(j, i);
            cols[i * knns.rows() + j] = i;
            vals[i * knns.rows() + j] = d;
        }
    }

    return {rows, cols, vals};

}

auto fuzzy_simplicial_set(Eigen::MatrixXi &knns, Eigen::MatrixXf &dists, int n_neighbors, int n_points, float local_connectivity, float set_operation_ratio, bool apply_fuzzy_combine = true) -> Eigen::SparseMatrix<float>
{

    auto [rhos, sigmas] = smooth_knn_dists(dists, n_neighbors, local_connectivity);

    /// TODO: In the original, it passes first sigmas and then rhos, but the function expects first rhos and then sigmas, so what is the correct answer
    auto [rows, cols, vals] = compute_membership_strenghs(knns, dists, rhos, sigmas);

    /// TODO: fill with triplets https://eigen.tuxfamily.org/dox/group__TutorialSparse.html
    Eigen::SparseMatrix<float> fs_set(n_points, n_points);
    fs_set.reserve(vals.size());
    for (int i = 0; i < rows.size(); ++i)
    {
        fs_set.insert(rows[i], cols[i]) = vals[i];
    }
    fs_set.makeCompressed();

    /// TODO: Check if we should prune the resulting sparse matrix for performance
    if (apply_fuzzy_combine)
    {
        Eigen::SparseMatrix<float> res = combine_fuzzy_sets(fs_set, set_operation_ratio);

        return res;
    }
    else
    {
        return fs_set;
    }

}

struct UMAP_Arguments
{
    std::optional<float> a = {};
    std::optional<float> b = {};

    int n_components = 2;
	int n_neighbors = 15;
    int n_epochs = 300;

    float learning_rate = 1.0f;
    float min_dist = 0.1f;
    float spread = 1.0f;
    float set_operation_ratio = 1.0f;
    float local_connectivity = 1.0f;
    float repulsion_strength = 1.0f;
    int neg_sample_rate = 5;
};

auto UMAP_(Eigen::MatrixXf &X, int n_components = 2, int n_neighbors = 15, auto metric = square_euclidean,
           int n_epochs = 300, float learning_rate = 1.0f, std::variant<Spectral, Random> init = Spectral(), float min_dist = 0.1f,
           float spread = 1.0f, float set_operation_ratio = 1.0f, float local_connectivity = 1.0f, float repulsion_strength = 1.0f,
           int neg_sample_rate = 5, std::optional<float> a = {}, std::optional<float> b = {}) -> Eigen::MatrixXf
{

    auto start = std::chrono::steady_clock::now();
    std::cout << "Starting UMAP algorithm" << "\n\n";

    auto [knns, dists] = knn_search(X, n_neighbors, metric);

    auto knn_search_time = std::chrono::steady_clock::now();
    std::cout << "KNN Search time = " << std::chrono::duration_cast<std::chrono::microseconds>(knn_search_time - start).count()/1e3 << "[ms]" << "\n";

    auto graph = fuzzy_simplicial_set(knns, dists, n_neighbors, X.cols(), local_connectivity, set_operation_ratio);

    auto fuzzy_simplicial_set_time = std::chrono::steady_clock::now();
    std::cout << "Fuzzy simplicial set time = " << std::chrono::duration_cast<std::chrono::microseconds>(fuzzy_simplicial_set_time - knn_search_time).count()/1e3f  << "[ms]" << "\n";

    auto embedding = initialize_embedding(graph, n_components, std::get<0>(init));

    auto initialize_embedding_time = std::chrono::steady_clock::now();
    std::cout << "Embedding initialization time = " << std::chrono::duration_cast<std::chrono::microseconds>(initialize_embedding_time - fuzzy_simplicial_set_time).count()/1e3f  << "[ms]" << "\n";

    auto embedding_optimized = optimize_embedding(graph, embedding, embedding, n_epochs, learning_rate, min_dist, spread, repulsion_strength, neg_sample_rate);

    auto optimize_embedding_time = std::chrono::steady_clock::now();
    std::cout << "Optimize embedding time = " << std::chrono::duration_cast<std::chrono::microseconds>(optimize_embedding_time - initialize_embedding_time).count()/1e3f  << "[ms]" << "\n\n";
    std::cout << "UMAP algorithm finished in " << std::chrono::duration_cast<std::chrono::microseconds>(optimize_embedding_time - start).count()/1e3f  << "[ms]" << "\n";

    return embedding_optimized;

}

auto UMAP_(Eigen::MatrixXf &X, UMAP_Arguments &args) -> Eigen::MatrixXf
{
    return UMAP_(X, args.n_components, args.n_neighbors, square_euclidean, args.n_epochs,
                 args.learning_rate, Spectral(), args.min_dist, args.spread, args.set_operation_ratio,
                 args.local_connectivity, args.repulsion_strength, args.neg_sample_rate, args.a, args.b);
}

#endif // UMAP__HPP
