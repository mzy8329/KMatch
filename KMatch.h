#ifndef _KMATCH_H_
#define _KMATCH_H_

#include <vector>
#include <eigen3/Eigen/Dense>

#define REWARD 0
#define LOSS 1

/**
 * @brief get match by KM algorithm
 *
 * @param _adj_matrix: The adjacency matrix
 * @return std::vector<int>: return row-based matching result
 */
std::vector<int> KMatch(Eigen::MatrixXf _adj_matrix, int _type, float _disconnect_value);

#endif