//
// Created by zhuangxk on 4/19/16.
//
#include "de_auto_encoder.h"
#include "utils.h"
#include "interface.h"


struct parameter param;


void exit_with_help() {
    printf(
        "Usage: AutoEncoder [options] input_set_file output_file model_file\n"
            "options:\n"
            "-m mode : set work mode of dAE (default 1)\n"
            "	     1 -- train mode\n"
            "	     2 -- encode hidden layer mode\n"
            "-c corruption level (default 0.0)\n"
            "-r learning rate (default 0.1)\n"
            "-n max epoch num (default 100)\n"
            "-h hidden neurons (default 10)\n"
            "-v visible neurons (default 100)\n"
    );
    exit(1);
}

void parse_command_line(int argc, char** argv, std::string& input_file_name, std::string& model_file_name, std::string& output_file_name) {
    int i;
    void (*print_func)(const char *) = NULL;    // default printing to stdout

    // parse options
    for (i = 1; i < argc; i++) {
        if (argv[i][0] != '-') break;
        if (++i >= argc)
            exit_with_help();
        switch (argv[i-1][1]) {
            case 'm':
                param.work_mode = atoi(argv[i]);
                break;
            case 'c':
                param.corruption_level = atof(argv[i]);
                break;
            case 'r':
                param.learning_rate = atof(argv[i]);
                break;
            case 'n':
                param.epoch_num = atoi(argv[i]);
                break;
            case 'h':
                param.n_hidden = atoi(argv[i]);
                break;
            case 'v':
                param.n_visible = atoi(argv[i]);
                break;
            default:
                fprintf(stderr, "unknown option: -%c\n", argv[i-1][1]);
                exit_with_help();
                break;
        }
    }
    if(i>=argc)
        exit_with_help();
    input_file_name = std::string(argv[i]);
    model_file_name = std::string(argv[++i]);
    if(2 == param.work_mode) {
        output_file_name = std::string(argv[++i]);
    }

}


int main(int argc, char **argv) {

    srand(0);
    std::string input_file_name;
    std::string model_file_name;
    std::string output_file_name;
    parse_command_line(argc, argv, input_file_name, model_file_name, output_file_name);

    deAutoEncoder da(param.n_visible, param.n_hidden, NULL, NULL, NULL);

    if(1 == param.work_mode) {
        da.train_file(input_file_name, model_file_name, param.learning_rate, param.corruption_level, param.epoch_num);
    }

    if(2 == param.work_mode) {
        da.encode_hidden_layer(input_file_name, output_file_name, model_file_name);
    }

}