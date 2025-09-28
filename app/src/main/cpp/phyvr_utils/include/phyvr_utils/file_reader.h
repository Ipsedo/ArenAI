//
// Created by samuel on 28/09/2025.
//

#ifndef PHYVR_FILE_READER_H
#define PHYVR_FILE_READER_H

#include <string>

struct img_rgb {
  int width;
  int height;
  char *pixels;
};

struct img_grey {
  int width;
  int height;
  float *pixels;
};

class AbstractFileReader {
public:
  virtual std::string read_text(const std::string &file_name) = 0;
  virtual img_rgb read_png(const std::string &png_file_path) = 0;

  static img_grey to_img_grey(img_rgb image);
};

#endif // PHYVR_FILE_READER_H
