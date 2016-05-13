//
// Created by zhuangxk on 4/19/16.
//
#include "utils.h"
#include "de_auto_encoder.h"


using boost::property_tree::ptree;
using boost::property_tree::read_json;
using boost::property_tree::write_json;


deAutoEncoder::deAutoEncoder(int n_v, int n_h, double** w, double* hb, double* vb) {
    n_visible = n_v;
    n_hidden = n_h;

    if (w == NULL) {
        W = new double* [n_hidden];
        for (int i = 0; i < n_hidden; i++) W[i] = new double[n_visible];
        double a = 1.0 / n_visible;

        for (int i = 0; i < n_hidden; i++) {
            for (int j = 0; j < n_visible; j++) {
                W[i][j] = uniform(-a, a);
            }
        }
    } else {
        W = w;
    }

    if (hb == NULL) {
        hbias = new double[n_hidden];
        for (int i = 0; i < n_hidden; i++) hbias[i] = 0;
    } else {
        hbias = hb;
    }

    if (vb == NULL) {
        vbias = new double[n_visible];
        for (int i = 0; i < n_visible; i++) vbias[i] = 0;
    } else {
        vbias = vb;
    }
}

deAutoEncoder::~deAutoEncoder() {
    for (int i = 0; i < n_hidden; i++)
        delete[] W[i];
    delete[] W;
    delete[] hbias;
    delete[] vbias;
}

void deAutoEncoder::get_corrupted_input(int* x, int* tilde_x, double p) {
    for (int i = 0; i < n_visible; i++) {
        if (x[i] == 0) {
            tilde_x[i] = 0;
        } else {
            tilde_x[i] = binomial(1, p);
        }
    }
}

// Encode
void deAutoEncoder::get_hidden_values(int* x, double* y) {
    for (int i = 0; i < n_hidden; i++) {
        y[i] = 0;
        for (int j = 0; j < n_visible; j++) {
            y[i] += W[i][j] * x[j];
        }
        y[i] += hbias[i];
        y[i] = sigmoid(y[i]);
    }
}

// Decode
void deAutoEncoder::get_reconstructed_input(double* y, double* z) {
    for (int i = 0; i < n_visible; i++) {
        z[i] = 0;
        for (int j = 0; j < n_hidden; j++) {
            z[i] += W[j][i] * y[j];
        }
        z[i] += vbias[i];
        z[i] = sigmoid(z[i]);
    }
}

void deAutoEncoder::train(int* x, double lr, double corruption_level) {
    int* tilde_x = new int[n_visible];
    double* y = new double[n_hidden];
    double* z = new double[n_visible];

    double* v_error = new double[n_visible];
    double* h_error = new double[n_hidden];

    double p = 1 - corruption_level;

    get_corrupted_input(x, tilde_x, p);
    get_hidden_values(tilde_x, y);
    get_reconstructed_input(y, z);

    // vbias
    for (int i = 0; i < n_visible; i++) {
        v_error[i] = x[i] - z[i];
        vbias[i] = vbias[i] + lr * v_error[i] / N;
    }

    // hbias
    for (int i = 0; i < n_hidden; i++) {
        h_error[i] = 0;
        for (int j = 0; j < n_visible; j++) {
            h_error[i] = h_error[i] + W[i][j] * v_error[j];
        }
        h_error[i] = h_error[i]*y[i] * (1 - y[i]);
        hbias[i] = hbias[i] + lr * h_error[i] / N;
    }

    // W
    for (int i = 0; i < n_hidden; i++) {
        for (int j = 0; j < n_visible; j++) {
            W[i][j] = W[i][j] + lr * (h_error[i] * tilde_x[j] + v_error[j] * y[i]) / N;
        }
    }

    delete[] h_error;
    delete[] v_error;
    delete[] z;
    delete[] y;
    delete[] tilde_x;
}

void deAutoEncoder::reconstruct(int* x, double* z) {
    double* y = new double[n_hidden];
    get_hidden_values(x, y);
    get_reconstructed_input(y, z);
    delete[] y;
}

int deAutoEncoder::load_model(std::string model_file) {
    try {
        ptree pt;
        read_json(model_file, pt);

        N = pt.get<int>("train_samples", 0);
        n_visible = pt.get<int>("input_neuron", 0);
        n_hidden = pt.get<int>("hidden_neuron", 0);

        // load vbias
        int i = 0;
        vbias = new double[n_visible];
        BOOST_FOREACH(boost::property_tree::ptree::value_type& v, pt.get_child("visible_bias")) {
                        assert(v.first.empty()); // array elements have no names
                        vbias[i] = v.second.get_value<double>();
                        i++;
                    }

        // load hbias
        i = 0;
        hbias = new double[n_hidden];
        BOOST_FOREACH(boost::property_tree::ptree::value_type& v, pt.get_child("hidden_bias")) {
                        assert(v.first.empty()); // array elements have no names
                        hbias[i] = v.second.get_value<double>();
                        i++;
                    }

        // load W
        i = 0;
        W = new double* [n_hidden];
        BOOST_FOREACH (boost::property_tree::ptree::value_type& row_pair, pt.get_child("weight")) {
                        assert(row_pair.first.empty()); // array elements have no names
                        int j = 0;
                        W[i] = new double[n_visible];
                        BOOST_FOREACH (boost::property_tree::ptree::value_type& item_pair, row_pair.second) {
                                        assert(item_pair.first.empty()); // array elements have no names
                                        W[i][j] = item_pair.second.get_value<double>();
                                        j++;
                                    }
                        i++;
                    }
    }
    catch (std::exception const& e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    return 0;

}

int deAutoEncoder::save_model(std::string model_file) {
    try {
        ptree pt;
        pt.put("model_type", "Denosing Auto Encoder");
        pt.put("model_version", "1.0");
        pt.put("train_samples", N);
        pt.put("input_neuron", n_visible);
        pt.put("hidden_neuron", n_hidden);

        // add hidden bias
        ptree children_hbias;
        ptree array_element_hbias;
        for(int i=0; i<n_hidden; i++){
            array_element_hbias.put_value(hbias[i]);
            children_hbias.push_back(std::make_pair("", array_element_hbias));
        }
        pt.add_child("hidden_bias", children_hbias);

        // add visible bias
        ptree children_vbias;
        ptree array_element_vbias;
        for(int i=0; i<n_visible; i++){
            array_element_vbias.put_value(vbias[i]);
            children_vbias.push_back(std::make_pair("", array_element_vbias));
        }
        pt.add_child("visible_bias", children_vbias);

        // add W
        ptree children_W;
        for(int i=0; i<n_hidden; i++){
            ptree children_hidden_w;
            ptree array_element_w;
            for(int j=0; j<n_visible; j++){
                array_element_w.put_value(W[i][j]);
                children_hidden_w.push_back(std::make_pair("", array_element_w));
            }
            children_W.push_back(std::make_pair("", children_hidden_w));
        }
        pt.add_child("weight", children_W);

        write_json(model_file, pt);
        std::cout << "Modle file is saved to " + model_file + " successfully !" << std::endl;
    }
    catch (std::exception const& e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    return 0;
}

int deAutoEncoder::encode_hidden_layer(std::string input_file, std::string output_file, std::string model_file) {

    if(load_model(model_file) < 0) {
        std::cout << "Load model failed !" << std::endl;
    }

    // load data
    std::vector<std::vector<int> > data_;
    std::vector<int> goal;
    read_data(data_, goal, input_file);

    // trans array
    int** input_X = new int*[data_.size()];
    trans_array(input_X, data_, data_.size(), n_visible);
    std::cout << "+" <<std::endl;


    // get hidden values
    std::ofstream f_out(output_file.c_str());
    for (int i = 0; i < data_.size(); i++) {
        f_out << goal[i];
        double y[n_hidden];
        get_hidden_values(input_X[i], y);
        for (int j = 0; j < n_hidden; j++) {
            f_out <<  " " << j+1 << ":" << std::setprecision(6) << y[j];
        }
        f_out << std::endl;
    }

    // release mem
    for (int i = 0; i < data_.size(); i++)
        delete[] input_X[i];
    delete[] input_X;

    std::cout <<"Get hidden layer output to " << output_file << " successfully !" <<std::endl;
}

int deAutoEncoder::train_file(std::string input_file,
                                std::string model_file,
                                double learning_rate,
                                double corruption_level,
                                int epoch_num){
    // load data
    std::vector<std::vector<int> > data_;
    std::vector<int> goal;
    read_data(data_, goal, input_file);

    // trans array
    int** train_X = new int*[data_.size()];
    trans_array(train_X, data_, data_.size(), n_visible);
    std::cout << "+" <<std::endl;

    std::cout << "Training started !" << std::endl;
    deAutoEncoder da(n_visible, n_hidden, NULL, NULL, NULL);
    da.N = data_.size();

    // train
    for (int epoch = 1; epoch <= epoch_num; epoch++) {
        for (int i = 0; i < data_.size(); i++) {
            da.train(train_X[i], learning_rate, corruption_level);
        }
        std::cout << "Epoch " << epoch << " is done ... ..." << std::endl;
    }

    da.save_model(model_file);

    std::cout << "Trained model successfully !" << std::endl;
    return 0;
}