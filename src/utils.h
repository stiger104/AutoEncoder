#ifndef AUTOENCODER_UTILS_H_
#define AUTOENCODER_UTILS_H_

#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>
#include "boost/lexical_cast.hpp"
#include <boost/algorithm/string.hpp>


double uniform(double min, double max);
int binomial(int n, double p);
double sigmoid(double x);
int read_data(std::vector<std::vector<int> >& data, std::vector<int>& goal, std::string input_path);
int trans_array(int** train_X, std::vector<std::vector<int> >& data_, int train_N, int n_visible);


#endif //