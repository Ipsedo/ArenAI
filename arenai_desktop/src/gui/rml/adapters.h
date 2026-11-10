#ifndef ARENAI_DESKTOP_GUI_RML_ADAPTERS_H
#define ARENAI_DESKTOP_GUI_RML_ADAPTERS_H

#include <chrono>
#include <memory>
#include <string>

#include <RmlUi/Core.h>

#include <arenai_utils/file_reader.h>

namespace arenai::desktop::gui {

    class MenuSystemInterface final : public Rml::SystemInterface {
    public:
        double GetElapsedTime() override;

    private:
        std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
    };

    class ReaderBackedFileInterface final : public Rml::FileInterface {
    public:
        explicit ReaderBackedFileInterface(
            std::shared_ptr<utils::AbstractResourceFileReader> reader);

        Rml::FileHandle Open(const Rml::String &path) override;
        void Close(Rml::FileHandle file) override;
        size_t Read(void *buffer, size_t size, Rml::FileHandle file) override;
        bool Seek(Rml::FileHandle file, long offset, int origin) override;
        size_t Tell(Rml::FileHandle file) override;

    private:
        struct OpenedFile {
            std::string content;
            size_t position = 0;
        };

        std::shared_ptr<utils::AbstractResourceFileReader> reader_;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_ADAPTERS_H
