//
// Created by samuel on 25/05/18.
//
#include "assets.h"

#include <android/log.h>

std::string getFileText(AAssetManager *mgr, std::string fileName) {
    // Open your file
    AAsset *file = AAssetManager_open(mgr, fileName.c_str(), AASSET_MODE_BUFFER);
    // Get the file length
    off_t fileLength = AAsset_getLength(file);

    // Allocate memory to read your file
    char *fileContent = new char[fileLength + 1];

    // Read your file
    AAsset_read(file, fileContent, size_t(fileLength));
    // For safety you can add a 0 terminating character at the end of your file
    // ...
    fileContent[fileLength] = '\0';

    // Do whatever you want with the content of the file
    AAsset_close(file);

    std::string res = std::string(fileContent);

    delete[] fileContent;

    return res;
}

void user_read_data(png_structp pngPtr, png_bytep data, png_size_t length) {
    png_voidp a = png_get_io_ptr(pngPtr);
    AAsset_read((AAsset *) a, (char *) data, length);
}

bool validatePNG(AAssetManager *mgr, std::string pngName) {
    AAsset *file = AAssetManager_open(mgr, pngName.c_str(), AASSET_MODE_BUFFER);

    png_byte header[8];
    AAsset_read(file, header, size_t(8));
    AAsset_close(file);
    return !png_sig_cmp(header, 0, 8);
}

libpng_image readPNG(AAssetManager *mgr, std::string pngName) {
    AAsset *file = AAssetManager_open(mgr, pngName.c_str(), AASSET_MODE_BUFFER);

    char header[8];
    AAsset_read(file, header, 8);
    if (png_sig_cmp((png_byte *) header, 0, 8)) {
        __android_log_print(ANDROID_LOG_DEBUG, "PhyVR", "unrecognize png sig %s", pngName.c_str());
        exit(667);
    }

    // http://www.piko3d.net/tutorials/libpng-tutorial-loading-png-files-from-streams/
    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;
    png_infop end_info = NULL;
    png_bytep row = NULL;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    info_ptr = png_create_info_struct(png_ptr);
    end_info = png_create_info_struct(png_ptr);
    png_set_read_fn(png_ptr, (png_voidp) file, userReadData);

    png_set_sig_bytes(png_ptr, 8);

    png_read_info(png_ptr, info_ptr);

    png_uint_32 imgWidth = png_get_image_width(png_ptr, info_ptr);
    png_uint_32 imgHeight = png_get_image_height(png_ptr, info_ptr);
    png_uint_32 bitdepth = png_get_bit_depth(png_ptr, info_ptr);
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
            if (bitdepth < 8) png_set_expand_gray_1_2_4_to_8(png_ptr);
            bitdepth = 8;
            break;
        default: break;
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
    if (bitdepth < 8) {
        png_set_packing(png_ptr);
        bitdepth = 8;
    } else if (bitdepth == 16) {
        png_set_strip_16(png_ptr);
        bitdepth = 8;
    }

    png_read_update_info(png_ptr, info_ptr);

    png_bytep *rowPtrs = new png_bytep[imgHeight];
    char *data = new char[imgWidth * imgHeight * bitdepth * channels / 8];
    const unsigned int stride = imgWidth * bitdepth * channels / 8u;

    for (unsigned int i = 0u; i < imgHeight; i++) {
        png_uint_32 q = (imgHeight - i - 1u) * stride;
        rowPtrs[i] = (png_bytep) data + q;
    }

    png_read_image(png_ptr, rowPtrs);

    png_read_end(png_ptr, end_info);

    png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
    AAsset_close(file);

    return {imgWidth, imgHeight, bitdepth, channels, data, rowPtrs};
}
