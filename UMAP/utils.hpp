#ifndef UTILS_HPP
#define UTILS_HPP

#include <tuple>
#include <vector>
#include <optional>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "curve_fit.hpp"

auto insertion_sort(std::vector<float> &distances, std::vector<int> &indexes, int num_points, int n_neighbors) -> void
{

    for (int i = 1; i < num_points; ++i)
    {
        float curr_distance = distances[i];
        int curr_index = i;

        if (i >= n_neighbors && curr_distance > distances[n_neighbors - 1])
            continue;

        int j = std::min(i, n_neighbors - 1);
        while (j > 0 && distances[j - 1] > curr_distance)
        {
            distances[j] = distances[j - 1];
            indexes[j] = indexes[j - 1];
            --j;
        }

        distances[j] = curr_distance;
        indexes[j] = curr_index;
    }

}

auto knn_search(Eigen::MatrixXf &X, int n_neighbors, auto metric) -> std::tuple<Eigen::MatrixXi, Eigen::MatrixXf>
{

    /// Since we are measuring the nearest neighbors of every element agains every other element,
    /// the reference and query points are the same size (and comes from the same matrix)
    int number_reference_points = X.cols();
    int number_query_points = X.cols();

    /// Matrixes that store the nearest neighbors and its distances to each query point
    Eigen::MatrixXf distances(n_neighbors, number_query_points);
    Eigen::MatrixXi indexes(n_neighbors, number_query_points);

    for (int i = 0; i < number_query_points; ++i)
    {
        /// For every query we calculate the nearest neighbors
        std::vector<float> temp_distances(number_reference_points);
        std::vector<int> temp_indexes(number_reference_points);

        for (int j = 0; j < number_reference_points; ++j)
        {
            temp_distances[j] = metric(X.col(i), X.col(j));
            temp_indexes[j] = j;
        }

        /// Sort from nearest to farthest;
        /// the first nearest neighbor is the point itself, so we won't consider it
        insertion_sort(temp_distances, temp_indexes, number_reference_points, n_neighbors + 1);

        for (int j = 0; j < n_neighbors; ++j)
        {
            distances(j, i) = temp_distances[j + 1];
            indexes(j, i) = temp_indexes[j + 1];
        }
    }

    return {indexes, distances};

}

auto fuzzy_set_union(Eigen::SparseMatrix<float> &fs_set) -> Eigen::SparseMatrix<float>
{

    /// I can't do "fs_set + fs_set.transpose() .- (fs_set .* fs_set.transpose())" directly, so we do...
    Eigen::SparseMatrix<float> fs_set_transposed = fs_set.transpose();

    return fs_set + fs_set_transposed - (fs_set.cwiseProduct(fs_set_transposed));

}

auto fuzzy_set_intersection(Eigen::SparseMatrix<float> &fs_set) -> Eigen::SparseMatrix<float>
{

    Eigen::SparseMatrix<float> fs_set_transposed = fs_set.transpose();

    return fs_set.cwiseProduct(fs_set_transposed);

}

auto combine_fuzzy_sets(Eigen::SparseMatrix<float> &fs_set, float set_op_ratio) -> Eigen::SparseMatrix<float>
{

    return set_op_ratio * fuzzy_set_union(fs_set) + (1.0f - set_op_ratio) * fuzzy_set_intersection(fs_set);

}

auto fit_ab(float min_dist, float spread, std::optional<float> a = {}, std::optional<float> b = {}) -> std::tuple<float, float>
{

    if (a && b)
    {
        return {*a, *b};
    }

    /// TODO: change double to float
    std::vector<double> xs(300);
    for (int i = 0; i < 300; ++i)
        xs[i] = i * 1.0/300.0 * 3.0;

    std::vector<double> ys (300);
    for (int i = 0; i < 300; ++i)
    {
        if (xs[i] >= 0.12)
            ys[i] = std::exp(-(xs[i] - 0.12) / 1.0);
        else
            ys[i] = 1;
    }

    auto psi = [](double x, double a, double b) -> double
    {
        return 1.0 / (1.0 + a * std::pow(x, 2.0 * b));
    };

    std::vector<double> result = curve_fit(psi, {1.0, 1.0}, xs, ys);

    return {static_cast<float>(result[0]), static_cast<float>(result[1])};

}

#endif // UTILS_HPP
