#ifndef AUTOENCODER_DA_H_
#define AUTOENCODER_DA_H_

#include <iostream>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>


class deAutoEncoder {

    public:
        int N;
        int n_visible;
        int n_hidden;
        double** W;
        double* hbias;
        double* vbias;

    public:
        deAutoEncoder(int n_v, int n_h, double** w, double* hb, double* vb);
        ~deAutoEncoder();
        int load_model(std::string model_file);
        int save_model(std::string model_file);
        int encode_hidden_layer(std::string input_file, std::string output_file, std::string model_file);
        int train_file(std::string input_file,
                       std::string model_file,
                       double learning_rate,
                       double corruption_level,
                       int epoch_num);

    public:
        void get_corrupted_input(int* x, int* tilde_x, double p);
        void get_hidden_values(int* x, double* y);
        void get_reconstructed_input(double* y, double* z);
        void train(int* x, double lr, double corruption_level);
        void reconstruct(int* x, double* z);

};

#endif //
