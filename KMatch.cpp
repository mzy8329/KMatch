#include "KMatch.h"

bool rDepthFirstSearch(int _i,
                       Eigen::MatrixXf &_adj_matrix,
                       std::vector<float> &_X_label,
                       std::vector<float> &_Y_label,
                       std::vector<int> &_Y_match_index,
                       std::vector<int> &_X_visited,
                       std::vector<int> &_Y_visited,
                       std::vector<float> &_minimum_d)
{
    _X_visited[_i] = 1;

    for (int j = 0; j < _adj_matrix.cols(); j++)
    {
        if (_Y_visited[j] == 1 || _adj_matrix(_i, j) == 0)
        {
            continue;
        }

        if (_X_label[_i] + _Y_label[j] == _adj_matrix(_i, j))
        {
            _Y_visited[j] = 1;

            if (_Y_match_index[j] == -1 ||
                rDepthFirstSearch(_Y_match_index[j],
                                  _adj_matrix,
                                  _X_label,
                                  _Y_label,
                                  _Y_match_index,
                                  _X_visited,
                                  _Y_visited,
                                  _minimum_d))
            {
                _Y_match_index[j] = _i;
                return 1;
            }
        }
        else
        {
            _minimum_d[j] = std::min(_minimum_d[j], _X_label[_i] + _Y_label[j] - _adj_matrix(_i, j));
        }
    }

    return 0;
}

std::vector<int> KMatch(Eigen::MatrixXf _adj_matrix, int _type, float _disconnect_value)
{
    if (_type == LOSS)
    {
        float max_element = _adj_matrix.maxCoeff();
        _adj_matrix *= -1;
        if (max_element == _disconnect_value)
        {
            _adj_matrix = _adj_matrix.array() + max_element;
        }
        else
        {
            _adj_matrix = _adj_matrix.array() + max_element + 1;
        }
    }
    _disconnect_value = 0;

    int size = std::max(_adj_matrix.rows(), _adj_matrix.cols());
    Eigen::MatrixXf adj_matrix(size, size);
    adj_matrix.fill(-1);
    adj_matrix.block(0, 0, _adj_matrix.rows(), _adj_matrix.cols()) = _adj_matrix;
    std::cout << adj_matrix << std::endl;

    std::vector<float> X_label(size, 0.0);
    std::vector<float> Y_label(size, 0.0);

    for (int i = 0; i < size; i++)
    {
        Eigen::VectorXf row_vec = adj_matrix.row(i);
        X_label[i] = row_vec.maxCoeff();
    }

    std::vector<int> Y_match_index(size, -1);

    std::vector<int> X_visited(size, 0);
    std::vector<int> Y_visited(size, 0);
    std::vector<float> minimum_d(size, 65536.0);

    for (int i = 0; i < size; i++)
    {
        while (1)
        {
            std::fill(X_visited.begin(), X_visited.end(), 0);
            std::fill(Y_visited.begin(), Y_visited.end(), 0);
            std::fill(minimum_d.begin(), minimum_d.end(), 65536.0);

            if (rDepthFirstSearch(i,
                                  adj_matrix,
                                  X_label,
                                  Y_label,
                                  Y_match_index,
                                  X_visited,
                                  Y_visited,
                                  minimum_d))
            {
                break;
            }

            float delta = 65536.0;
            for (int j = 0; j < size; j++)
            {
                delta = std::min(delta, minimum_d[j]);
            }

            // 没有可增加的边
            if (std::abs(delta - 65536) < 1.0)
            {
                break;
            }

            // 加边
            for (int n = 0; n < size; n++)
            {
                if (X_visited[n])
                {
                    X_label[n] -= delta;
                }
                if (Y_visited[n])
                {
                    Y_label[n] += delta;
                }
            }
        }
    }

    return Y_match_index;
}
