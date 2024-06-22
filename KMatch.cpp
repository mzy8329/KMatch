#include "KMatch.h"

std::vector<int> KMatch(Eigen::MatrixXf _adj_matrix, int _type, float _disconnect_value)
{
    int X_origin_size = _adj_matrix.rows();
    int Y_origin_size = _adj_matrix.cols();

    if (_type = LOSS)
    {
        float max_element = _adj_matrix.maxCoeff();

                _adj_matrix *= -1;
    }

    std::vector<float> X_bench;
    std::vector<float> Y_bench;
    for (int row = 0; row < X_origin_size; row++)
    {
        float max = _adj_matrix.row(row).maxCoeff();
        float min = _adj_matrix.row(row).minCoeff();
    }
}