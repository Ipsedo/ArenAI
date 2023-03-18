//
// Created by samuel on 18/03/2023.
//

#include <sstream>
#include "string_utils.h"

std::vector<std::string> split_string(const std::string &input, char delimiter) {
    std::stringstream ss(input);
    std::string item;
    std::vector<std::string> elems;

    while (std::getline(ss, item, delimiter)) {
        elems.push_back(item);
    }

    return elems;
}
