//
// Created by zhuangxk on 12/23/15.
//


#include <stdio.h>
#include "gtest/gtest.h"


int main_(int argc, char **argv) {
    printf("Running google test from main.cc !\n\n");
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

