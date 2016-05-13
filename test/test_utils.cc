//
// Created by zhuangxk on 4/19/16.
//

#include "../src/utils.h"
#include "gtest/gtest.h"


TEST(Utils, read_data) {
    std::string data_root = "/home/zhuangxk/ClionProjects/AutoEncoder/data/";
    std::vector<std::vector<int> > data;
    std::vector<int> goal;
    std::string path = data_root + "test.encode";
    EXPECT_EQ(0, read_data(data, goal, path));
    EXPECT_EQ(1, data[0][3]);
}