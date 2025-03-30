#ifndef _KMATCH_H_
#define _KMATCH_H_

#include <vector>
#include <list>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <pybind11_catkin/pybind11/pybind11.h>
#include <pybind11_catkin/pybind11/eigen.h>
#include <pybind11_catkin/pybind11/stl.h>
#include <pybind11_catkin/pybind11/functional.h>
#include <pybind11_catkin/pybind11/chrono.h>
#include <pybind11_catkin/pybind11/complex.h>


#define REWARD 0
#define LOSS 1
#define MATCH_INF 65535.0 // the max value in adjacency matrix should less than MATCH_INF

/**
 * @brief get match by KM algorithm
 *
 * @param _adj_matrix: The adjacency matrix
 * @return std::vector<int>: return col-based matching result
 */
std::vector<int> KMatch(Eigen::MatrixXf _adj_matrix, int _type, float _disconnect_value);

std::vector<int> HungarianMatch(Eigen::MatrixXf _adj_matrix);

#endif