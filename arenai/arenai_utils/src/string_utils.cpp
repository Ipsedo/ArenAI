//
// Created by samuel on 18/03/2023.
//

#include <sstream>

#include <arenai_utils/string_utils.h>

std::vector<std::string> split_string(const std::string &input, const char delimiter) {
    std::stringstream ss(input);
    std::string item;
    std::vector<std::string> elements;

    while (std::getline(ss, item, delimiter)) { elements.push_back(item); }

    return elements;
}
