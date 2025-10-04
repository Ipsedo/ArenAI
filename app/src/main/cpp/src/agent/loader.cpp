//
// Created by samuel on 03/10/2025.
//

#include "./loader.h"
#include <vector>

std::string copy_asset_to_files(AAssetManager *mgr, const char *asset_name,
                                const char *dst_dir) {
  AAsset *asset = AAssetManager_open(mgr, asset_name, AASSET_MODE_BUFFER);
  const size_t len = AAsset_getLength(asset);
  std::vector<char> buf(len);

  AAsset_read(asset, buf.data(), len);
  AAsset_close(asset);

  std::string out_path = std::string(dst_dir) + "/" + asset_name;
  FILE *f = fopen(out_path.c_str(), "wb");

  fwrite(buf.data(), 1, buf.size(), f);
  fclose(f);

  return out_path;
}
