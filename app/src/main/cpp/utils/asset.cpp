//
// Created by samuel on 18/03/2023.
//

#include "asset.h"

#include <android/asset_manager.h>

#include "logging.h"

std::string read_asset(AAssetManager *mgr, const std::string &file_name) {
    // Open your file
    AAsset *file = AAssetManager_open(mgr, file_name.c_str(), AASSET_MODE_BUFFER);
    // Get the file length
    off_t file_length = AAsset_getLength(file);

    // Allocate memory to read your file
    char *file_content = new char[file_length + 1];

    // Read your file
    AAsset_read(file, file_content, size_t(file_length));
    // For safety you can add a 0 terminating character at the end of your file ...
    file_content[file_length] = '\0';

    // Do whatever you want with the content of the file
    AAsset_close(file);

    std::string res = std::string(file_content);

    delete[] file_content;

    return res;
}

void user_read_data(png_structp pngPtr, png_bytep data, png_size_t length) {
    png_voidp a = png_get_io_ptr(pngPtr);
    AAsset_read((AAsset *) a, (char *) data, length);
}

libpng_image read_png(AAssetManager *mgr, const std::string &png_file_path) {
    AAsset *file = AAssetManager_open(mgr, png_file_path.c_str(), AASSET_MODE_BUFFER);

    char header[8];
    AAsset_read(file, header, 8);
    if (png_sig_cmp((png_byte *) header, 0, 8)) {
        LOG_ERROR("Unrecognized png sig %s", png_file_path.c_str());
        exit(1);
    }

    // http://www.piko3d.net/tutorials/libpng-tutorial-loading-png-files-from-streams/
    png_structp png_ptr = nullptr;
    png_infop info_ptr = nullptr;
    png_infop end_info = nullptr;
    png_bytep row = nullptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    info_ptr = png_create_info_struct(png_ptr);
    end_info = png_create_info_struct(png_ptr);
    png_set_read_fn(png_ptr, (png_voidp) file, user_read_data);

    png_set_sig_bytes(png_ptr, 8);

    png_read_info(png_ptr, info_ptr);

    png_uint_32 img_width = png_get_image_width(png_ptr, info_ptr);
    png_uint_32 img_height = png_get_image_height(png_ptr, info_ptr);
    png_uint_32 bit_depth = png_get_bit_depth(png_ptr, info_ptr);
    png_uint_32 channels = png_get_channels(png_ptr, info_ptr);
    png_uint_32 color_type = png_get_color_type(png_ptr, info_ptr);

    // color palete -> RGB
    // gray -> gray 1 octet
    switch (color_type) {
        case PNG_COLOR_TYPE_PALETTE:
            png_set_palette_to_rgb(png_ptr);
            channels = 3;
            break;

        case PNG_COLOR_TYPE_GRAY:
            if (bit_depth < 8)
                png_set_expand_gray_1_2_4_to_8(png_ptr);
            bit_depth = 8;
            break;
        default:
            break;
    }

    // full alpha conversion
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) {
        png_set_tRNS_to_alpha(png_ptr);
        channels += 1;
    }
    if (color_type & PNG_COLOR_MASK_ALPHA) {
        png_set_strip_alpha(png_ptr);
        channels -= 1;
    }

    // Pass to 8 bits (1 byte) depth
    if (bit_depth < 8) {
        png_set_packing(png_ptr);
        bit_depth = 8;
    } else if (bit_depth == 16) {
        png_set_strip_16(png_ptr);
        bit_depth = 8;
    }

    png_read_update_info(png_ptr, info_ptr);

    auto *row_ptrs = new png_bytep[img_height];
    char *data = new char[img_width * img_height * bit_depth * channels / 8];
    const unsigned int stride = img_width * bit_depth * channels / 8u;

    for (unsigned int i = 0u; i < img_height; i++) {
        png_uint_32 q = (img_height - i - 1u) * stride;
        row_ptrs[i] = (png_bytep) data + q;
    }

    png_read_image(png_ptr, row_ptrs);

    png_read_end(png_ptr, end_info);

    png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
    AAsset_close(file);

    return {img_width,
            img_height,
            bit_depth,
            channels,
            data,
            row_ptrs};
}
