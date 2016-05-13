//
// Created by zhuangxk on 4/19/16.
//

#ifndef AUTOENCODER_INTERFACE_H
#define AUTOENCODER_INTERFACE_H

#endif //AUTOENCODER_INTERFACE_H


struct parameter
{
  int work_mode = 1;
  int epoch_num = 100;
  double learning_rate = 0.1;
  double corruption_level = 0.0;
  int n_hidden = 10;
  int n_visible = 100;
};


void exit_with_help();
void parse_command_line(int argc, char** argv, char* input_file_name, char* output_file_name, char* model_file_name);







