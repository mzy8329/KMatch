#include "KMatch.h"

typedef enum {
    KM,
    Hungarian
}Mtype_e;

bool rDepthFirstSearch(const Mtype_e& _type,
    int _i,
    Eigen::MatrixXf& _adj_matrix,
    std::vector<float>& _X_label,
    std::vector<float>& _Y_label,
    std::vector<int>& _Y_match_index,
    std::vector<int>& _X_visited,
    std::vector<int>& _Y_visited,
    std::vector<float>& _minimum_d)
{
    _X_visited[_i] = 1;

    if (_type == KM) {
        for (int j = 0; j < _adj_matrix.cols(); j++) {
            if (_Y_visited[j] == 1 || _adj_matrix(_i, j) == 0) {
                continue;
            }

            if (_X_label[_i] + _Y_label[j] == _adj_matrix(_i, j)) {
                _Y_visited[j] = 1;

                if (_Y_match_index[j] == -1 ||
                    rDepthFirstSearch(_type,
                        _Y_match_index[j],
                        _adj_matrix,
                        _X_label,
                        _Y_label,
                        _Y_match_index,
                        _X_visited,
                        _Y_visited,
                        _minimum_d)) {
                    _Y_match_index[j] = _i;
                    return 1;
                }
            }
            else {
                _minimum_d[j] = std::min(_minimum_d[j], _X_label[_i] + _Y_label[j] - _adj_matrix(_i, j));
            }
        }
    }
    else {
        for (int j = 0; j < _adj_matrix.cols(); j++) {
            if (_Y_visited[j] == 1 || _adj_matrix(_i, j) == 0) {
                continue;
            }

            _Y_visited[j] = 1;
            if (_Y_match_index[j] == -1 ||
                rDepthFirstSearch(_type,
                    _Y_match_index[j],
                    _adj_matrix,
                    _X_label,
                    _Y_label,
                    _Y_match_index,
                    _X_visited,
                    _Y_visited,
                    _minimum_d)) {
                _Y_match_index[j] = _i;
                return 1;
            }
        }
    }


    return 0;
}

std::vector<int> KMatch(Eigen::MatrixXf _adj_matrix, int _type, float _disconnect_value)
{
    if (_type == LOSS) {
        float max_element = _adj_matrix.maxCoeff();
        _adj_matrix *= -1;
        if (std::abs(max_element - _disconnect_value) < 0.01f) {
            _adj_matrix = _adj_matrix.array() + max_element;
        }
        else {
            _adj_matrix = _adj_matrix.array() + max_element + 1;
        }
    }
    _disconnect_value = 0;

    int size = std::max(_adj_matrix.rows(), _adj_matrix.cols());
    Eigen::MatrixXf adj_matrix(size, size);
    adj_matrix.fill(-1);
    adj_matrix.block(0, 0, _adj_matrix.rows(), _adj_matrix.cols()) = _adj_matrix;

    std::vector<float> X_label(size, 0.0);
    std::vector<float> Y_label(size, 0.0);

    for (int i = 0; i < size; i++) {
        Eigen::VectorXf row_vec = adj_matrix.row(i);
        X_label[i] = row_vec.maxCoeff();
    }

    std::vector<int> Y_match_index(size, -1);

    std::vector<int> X_visited(size, 0);
    std::vector<int> Y_visited(size, 0);
    std::vector<float> minimum_d(size, MATCH_INF);
    for (int i = 0; i < size; i++) {
        while (1) {
            std::fill(X_visited.begin(), X_visited.end(), 0);
            std::fill(Y_visited.begin(), Y_visited.end(), 0);
            std::fill(minimum_d.begin(), minimum_d.end(), MATCH_INF);

            if (rDepthFirstSearch(KM,
                i,
                adj_matrix,
                X_label,
                Y_label,
                Y_match_index,
                X_visited,
                Y_visited,
                minimum_d)) {
                break;
            }

            float delta = MATCH_INF;
            for (int j = 0; j < size; j++) {
                delta = std::min(delta, minimum_d[j]);
            }

            // no edge
            if (std::abs(delta - MATCH_INF) < 1.0 || std::abs(delta) < 1e-3) {
                break;
            }

            // add edge
            for (int n = 0; n < size; n++) {
                if (X_visited[n]) {
                    X_label[n] -= delta;
                }
                if (Y_visited[n]) {
                    Y_label[n] += delta;
                }
            }
        }
    }
    return Y_match_index;
}

std::vector<int> HungarianMatch(Eigen::MatrixXf _adj_matrix)
{
    int size = std::max(_adj_matrix.rows(), _adj_matrix.cols());
    Eigen::MatrixXf adj_matrix(size, size);
    adj_matrix.fill(-1);
    adj_matrix.block(0, 0, _adj_matrix.rows(), _adj_matrix.cols()) = _adj_matrix;

    std::vector<int> Y_match_index(size, -1);
    std::vector<int> X_visited(size, 0);
    std::vector<int> Y_visited(size, 0);
    std::vector<float> place_holder;
    for (int i = 0; i < size; i++) {
        if (rDepthFirstSearch(Hungarian,
            i,
            adj_matrix,
            place_holder,
            place_holder,
            Y_match_index,
            X_visited,
            Y_visited,
            place_holder)) {
            break;
        }
    }
    return Y_match_index;
}
