//
// Created by zhuangxk on 4/19/16.
//
#include "../src/de_auto_encoder.h"
#include "gtest/gtest.h"


std::string data_root = "/home/zhuangxk/ClionProjects/AutoEncoder/data/";
std::string model_file = data_root + "model.json";
std::string input_file = data_root + "test.encode";
std::string output_file = data_root + "test.encode.hidden";


double learning_rate = 0.1;
double corruption_level = 0;
int training_epochs = 1e4;

int train_N = 4;
int test_N = 4;
int n_visible = 4;
int n_hidden = 2;

// construct de_auto_encoder
deAutoEncoder da(4, 2, NULL, NULL, NULL);


TEST(AutoEncoder, main) {

    srand(0);

    // training data
    int train_X[train_N][n_visible] = {
        {0, 0, 0, 1},
        {0, 0, 1, 0},
        {0, 1, 0, 0},
        {1, 0, 0, 0},
    };

    // train
    for (int epoch = 0; epoch < training_epochs; epoch++) {
        for (int i = 0; i < train_N; i++) {
            da.train(train_X[i], learning_rate, corruption_level);
        }
    }

    EXPECT_EQ(0, da.save_model(model_file));

    // test data
    int test_X[test_N][n_visible] = {
        {0, 0, 0, 1},
        {0, 0, 1, 0},
        {0, 1, 0, 0},
        {1, 0, 0, 0},
    };
    double reconstructed_X[test_N][n_visible];

    // test
    for (int i = 0; i < test_N; i++) {
        da.reconstruct(test_X[i], reconstructed_X[i]);
        for (int j = 0; j < n_visible; j++) {
            printf("%.4f ", reconstructed_X[i][j]);
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;

    // get hidden values
    for (int i = 0; i < test_N; i++) {
        double y[n_hidden];
        da.get_hidden_values(test_X[i], y);
        for (int j = 0; j < n_hidden; j++) {
            printf("%.4f ", y[j]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

}

TEST(AutoEncoder, initialize) {

    srand(0);

    EXPECT_EQ(0, da.load_model(model_file));

    // test data
    int test_X[test_N][n_visible] = {
        {0, 0, 0, 1},
        {0, 0, 1, 0},
        {0, 1, 0, 0},
        {1, 0, 0, 0},
    };
    double reconstructed_X[test_N][n_visible];
    std::cout << std::endl;

    // test
    for (int i = 0; i < test_N; i++) {
        da.reconstruct(test_X[i], reconstructed_X[i]);
        for (int j = 0; j < n_visible; j++) {
            printf("%.4f ", reconstructed_X[i][j]);
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;

    // get hidden values
    for (int i = 0; i < test_N; i++) {
        double y[n_hidden];
        da.get_hidden_values(test_X[i], y);
        for (int j = 0; j < n_hidden; j++) {
            printf("%.6f ", y[j]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

TEST(AutoEncoder, encode_hidden_layer) {

    da.encode_hidden_layer(input_file, output_file, model_file);

}

TEST(AutoEncoder, train_file) {
    srand(0);

    da.train_file(input_file, model_file, learning_rate, corruption_level, training_epochs);

}



