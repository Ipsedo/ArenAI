//
// Created by samuel on 02/08/2026.
//

#include "./adapters.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <utility>

namespace arenai::desktop::gui {

    double MenuSystemInterface::GetElapsedTime() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    }

    ReaderBackedFileInterface::ReaderBackedFileInterface(
        std::shared_ptr<utils::AbstractResourceFileReader> reader)
        : reader_(std::move(reader)) {}

    Rml::FileHandle ReaderBackedFileInterface::Open(const Rml::String &path) {
        try {
            auto file = std::make_unique<OpenedFile>(OpenedFile{reader_->read_text(path)});
            return reinterpret_cast<Rml::FileHandle>(file.release());
        } catch (const std::exception &e) {
            std::cerr << "RmlUi asset open failed: " << e.what() << std::endl;
            return 0;
        }
    }

    void ReaderBackedFileInterface::Close(const Rml::FileHandle file) {
        delete reinterpret_cast<OpenedFile *>(file);
    }

    size_t
    ReaderBackedFileInterface::Read(void *buffer, const size_t size, const Rml::FileHandle file) {
        auto *opened = reinterpret_cast<OpenedFile *>(file);
        const size_t nb_read = std::min(size, opened->content.size() - opened->position);
        std::memcpy(buffer, opened->content.data() + opened->position, nb_read);
        opened->position += nb_read;
        return nb_read;
    }

    bool ReaderBackedFileInterface::Seek(
        const Rml::FileHandle file, const long offset, const int origin) {
        auto *opened = reinterpret_cast<OpenedFile *>(file);
        const long base = origin == SEEK_CUR   ? static_cast<long>(opened->position)
                          : origin == SEEK_END ? static_cast<long>(opened->content.size())
                                               : 0L;
        const long target = base + offset;
        if (target < 0 || target > static_cast<long>(opened->content.size())) return false;
        opened->position = static_cast<size_t>(target);
        return true;
    }

    size_t ReaderBackedFileInterface::Tell(const Rml::FileHandle file) {
        return reinterpret_cast<OpenedFile *>(file)->position;
    }

}// namespace arenai::desktop::gui
