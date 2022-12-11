#include <iostream>
#include <fstream>
#include <optional>

#include <Eigen/Dense>

#include "CLI/CLI.hpp"

#include "umap_.hpp"

template<typename Matrix_Type>
auto load_csv (const std::string & path) -> Matrix_Type
{
    std::ifstream indata;
    indata.open(path);

    std::string line;
    std::vector<typename Matrix_Type::Scalar> values;
    unsigned rows = 0;

    while (std::getline(indata, line))
    {
        std::stringstream lineStream(line);
        std::string cell;

        while (std::getline(lineStream, cell, ','))
            values.push_back(std::stof(cell));

        ++rows;
    }

    return Eigen::Map<const Eigen::Matrix<typename Matrix_Type::Scalar, Matrix_Type::RowsAtCompileTime, Matrix_Type::ColsAtCompileTime, Eigen::RowMajor>>(values.data(), rows, values.size() / rows);
}

auto write_output(const Eigen::MatrixXf &mat, const std::string &output) -> void
{
    std::ofstream csv_file(output);

    for (int i = 0; i < mat.rows(); ++i)
    {
        for (int j = 0; j < mat.cols() - 1; ++j)
        {
            csv_file << mat(i,j) << ",";
        }

        csv_file << mat(i, mat.cols() - 1) << std::endl;
    }
}

auto parse_arguments(int argc, char **argv) -> std::tuple<std::string, std::string, UMAP_Arguments>
{
	/// Parse arguments to form the particle set and the computation info
	CLI::App app{"UMAP Algorithm"};

	std::string input_dir;
	std::string output_dir = "output.csv";

	UMAP_Arguments umap_args;

    app.add_option("--input", input_dir, "CSV file path to input data (REQUIRED)") -> required();
	app.add_option("--output", output_dir, "File path to ouput data (with name and extension included; ie., \"output/file.csv\" )");

	app.add_option("--n_components", umap_args.n_components, "Final number of the data dimesion");
	app.add_option("--n_neighbors", umap_args.n_neighbors, "The number of neighbors to consider as locally connected. Larger values capture more global structure in the data, while small values capture more local structure.");
	app.add_option("--n_epochs", umap_args.n_epochs, "The number of training epochs for embedding optimization");
	app.add_option("--learning_rate", umap_args.learning_rate, "The initial learning rate during optimization");
	app.add_option("--min_dist", umap_args.min_dist, "The minimum spacing of points in the output embedding");
	app.add_option("--spread", umap_args.spread, "The effective scale of embedded points. Determines how clustered embedded points are in combination with `min_dist`.");
	app.add_option("--set_operation_ratio", umap_args.set_operation_ratio, "Interpolates between fuzzy set union and fuzzy set intersection when constructing the UMAP graph (global fuzzy simplicial set). The value of this parameter should be between 1.0 and 0.0: 1.0 indicates pure fuzzy union, while 0.0 indicates pure fuzzy intersection.");
	app.add_option("--local_connectivity", umap_args.local_connectivity, "The number of nearest neighbors that should be assumed to be locally connected. The higher this value, the more connected the manifold becomes. This should not be set higher than the intrinsic dimension of the manifold.");
	app.add_option("--repulsion_strength", umap_args.repulsion_strength, "The weighting of negative samples during the optimization process.");
	app.add_option("--neg_sample_rate", umap_args.neg_sample_rate, "The number of negative samples to select for each positive sample. Higher values will increase computational cost but result in slightly more accuracy.");
	app.add_option("--a", umap_args.a, "This controls the embedding. By default, this is determined automatically by `min_dist` and `spread`.");
	app.add_option("--b", umap_args.b, "This controls the embedding. By default, this is determined automatically by `min_dist` and `spread`.");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        std::exit(app.exit(e));
    }

    std::cout << "UMAP Algorithm in C++ (Author: Gerardo Medina Arellano)" << "\n\n";

    std::cout << "Input file: " << input_dir << "\n";
    std::cout << "Output file: " << output_dir << "\n\n";

    std::cout << "Parameters:" << "\n";
    std::cout << "\t-Number of components: " << umap_args.n_components << "\n";
    std::cout << "\t-Number of neighbors: " << umap_args.n_neighbors << "\n";
    std::cout << "\t-Number of epochs: " << umap_args.n_epochs << "\n";
    std::cout << "\t-Learning rate: " << umap_args.learning_rate << "\n";
    std::cout << "\t-Minimum distance: " << umap_args.min_dist << "\n";
    std::cout << "\t-Spread: " << umap_args.spread << "\n";
    std::cout << "\t-Operation ratio: " << umap_args.set_operation_ratio << "\n";
    std::cout << "\t-Local connectivity: " << umap_args.local_connectivity << "\n";
    std::cout << "\t-Repulsion strength: " << umap_args.repulsion_strength << "\n";
    std::cout << "\t-Negative sample rate: " << umap_args.neg_sample_rate << "\n";
    std::cout << "\t-a: " << ((!umap_args.a) ? "None" : std::to_string(*umap_args.a)) << "\n";
    std::cout << "\t-b: " << ((!umap_args.b) ? "None" : std::to_string(*umap_args.b)) << "\n\n";

	return {input_dir, output_dir, umap_args};

}

int main(int argc, char **argv)
{

    auto [input, output, umap_args] = parse_arguments(argc, argv);

    /// Load csv
    Eigen::MatrixXf dataframe = load_csv<Eigen::MatrixXf>(input).transpose();

    /// Distance function
    auto euclidean = [](const Eigen::Ref<Eigen::VectorXf> &x, const Eigen::Ref<Eigen::VectorXf> &y) -> float
    {
        float distance = 0.0f;

        for (int i = 0; i < x.rows(); ++i)
        {
            distance += std::pow(x(i) - y(i), 2.0f);
        }

        return std::sqrt(distance);
    };

    /// UMAP algorithm that returns a new dataframe with its dimension reduced
    auto embedding = UMAP_(dataframe, umap_args);
    //auto embedding = UMAP_(dataframe, 3, 15, euclidean, 300, 1.0f, Spectral(), 0.1f, 1.0f, 1.0f, 1.0f, 1.0f, 5, {}, {});

    /// Write the output to a file
    Eigen::MatrixXf embedding_trans = embedding.transpose();
    write_output(embedding_trans, output);

}
