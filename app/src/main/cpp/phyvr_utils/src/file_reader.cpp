//
// Created by samuel on 28/09/2025.
//

#include <android/imagedecoder.h>
#include <phyvr_utils/file_reader.h>

/*
 * Common
 */

img_grey AbstractFileReader::to_img_grey(img_rgb image) {
  img_grey res{};

  res.pixels = new float[image.width * image.height];

  for (int row = 0; row < image.height; row++)
    for (int col = 0; col < image.width; col++)
      // image.pixels -> RGBA -> * 4
      res.pixels[row * image.width + col] =
          (float(image.pixels[4 * row * image.width + col * 4]) +
           float(image.pixels[4 * row * image.width + col * 4 + 1]) +
           float(image.pixels[4 * row * image.width + col * 4 + 2])) /
          3.f / 255.f;

  res.width = int(image.width);
  res.height = int(image.height);

  return res;
}

/*
 * Android
 */

AndroidFileReader::AndroidFileReader(AAssetManager *mgr) : mgr(mgr) {}

std::string AndroidFileReader::read_text(const std::string &file_name) {
  AAsset *file = AAssetManager_open(mgr, file_name.c_str(), AASSET_MODE_BUFFER);
  off_t file_length = AAsset_getLength(file);

  char *file_content = new char[file_length + 1];

  AAsset_read(file, file_content, size_t(file_length));
  file_content[file_length] = '\0';

  AAsset_close(file);

  std::string res(file_content);

  delete[] file_content;

  return res;
}

img_rgb AndroidFileReader::read_png(const std::string &png_file_path) {
  AAsset *file =
      AAssetManager_open(mgr, png_file_path.c_str(), AASSET_MODE_BUFFER);

  AImageDecoder *decoder;
  AImageDecoder_createFromAAsset(file, &decoder);

  auto decoder_cleanup = [&decoder]() { AImageDecoder_delete(decoder); };

  const AImageDecoderHeaderInfo *header_info =
      AImageDecoder_getHeaderInfo(decoder);

  int bitmap_format =
      AImageDecoderHeaderInfo_getAndroidBitmapFormat(header_info);
  if (bitmap_format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
    decoder_cleanup();
    throw std::runtime_error("Only RGBA_8888 is accepted");
  }

  int channels = 4;
  int width = AImageDecoderHeaderInfo_getWidth(header_info);
  int height = AImageDecoderHeaderInfo_getHeight(header_info);

  size_t stride = AImageDecoder_getMinimumStride(decoder);

  int size = width * height * channels;
  char *pixels = new char[size];

  if (AImageDecoder_decodeImage(decoder, pixels, stride, size) !=
      ANDROID_IMAGE_DECODER_SUCCESS) {
    decoder_cleanup();
    throw std::runtime_error("Error in image decoding");
  }

  decoder_cleanup();

  return {width, height, pixels};
}
