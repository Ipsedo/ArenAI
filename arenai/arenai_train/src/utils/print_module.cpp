//
// Created by samuel on 14/02/2026.
//

#include "./print_module.h"

void dump_module_tree(
    const std::shared_ptr<torch::nn::Module> &m, std::ostream &out, const int indent,
    const std::string &name) {
    const std::string pad(indent, ' ');

    out << pad << name << ": " << m->name();

    {
        std::ostringstream tmp;
        m->pretty_print(tmp);
        if (const auto repr = tmp.str(); !repr.empty() && repr != m->name()) {
            out << "  " << repr;
        }
    }
    out << "\n";

    for (const auto &child: m->named_children()) {
        dump_module_tree(child.value(), out, indent + 2, child.key());
    }
}
