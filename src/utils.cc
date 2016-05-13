//
// Created by zhuangxk on 4/19/16.
//
#include "utils.h"


double uniform(double min, double max) {
    return rand() / (RAND_MAX + 1.0) * (max - min) + min;
}

int binomial(int n, double p) {
    if (p < 0 || p > 1) return 0;
    int c = 0;
    double r;
    for (int i = 0; i < n; i++) {
        r = rand() / (RAND_MAX + 1.0);
        if (r < p) c++;
    }
    return c;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

int read_data(std::vector<std::vector<int> >& data_, std::vector<int>& goal,std::string input_path) {
    std::ifstream f_in(input_path.c_str());
    if (f_in.fail()) {
        std::cout << "Open file failed, check the path!" << std::endl;
        return -1;
    }
    int size = 0;
    std::string line;
    while (getline(f_in, line)) {
        std::vector<std::string> str_vec;
        boost::algorithm::trim_if(line, boost::algorithm::is_any_of("\r\n "));
        boost::algorithm::split(str_vec, line, boost::algorithm::is_any_of(" "));
        int label = atoi(str_vec[0].c_str());
        if(99 != label) {
            goal.push_back(label);
            std::vector<int> values;
            for (size_t i=1; i<str_vec.size(); i++) {
                std::vector<std::string> str_vec_2;
                boost::algorithm::split(str_vec_2, str_vec[i], boost::algorithm::is_any_of(":"));
                values.push_back(atoi(str_vec_2[1].c_str()));
            }
            data_.push_back(values);
            size++;
        }
    }
    std::cout << "Read data from " << input_path << " successfully !" << std::endl;
    return 0;
}

int trans_array(int** train_X, std::vector<std::vector<int> >& data_, int train_N, int n_visible) {
    for(int i=0; i<train_N; i++) {
        train_X[i] = new int[n_visible];
        for(int j=0; j<n_visible; j++){
            train_X[i][j] = data_[i][j];
        }
    }
    return 0;
}




