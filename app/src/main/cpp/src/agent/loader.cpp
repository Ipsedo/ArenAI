//
// Created by samuel on 03/10/2025.
//

#include "./loader.h"

#include <filesystem>
#include <fstream>
#include <vector>

std::string get_cache_dir(android_app *app) {
  JNIEnv *env = nullptr;
  JavaVM *vm = app->activity->vm;
  vm->AttachCurrentThread(&env, nullptr);

  jobject activityObj = app->activity->clazz;

  jclass activityClass = env->GetObjectClass(activityObj);

  jmethodID midGetCacheDir = env->GetMethodID(activityClass, "getCacheDir", "()Ljava/io/File;");
  jobject fileObj = env->CallObjectMethod(activityObj, midGetCacheDir);

  jclass fileClass = env->GetObjectClass(fileObj);
  jmethodID midGetPath = env->GetMethodID(fileClass, "getAbsolutePath", "()Ljava/lang/String;");
  auto pathStr = (jstring) env->CallObjectMethod(fileObj, midGetPath);

  const char *pathCStr = env->GetStringUTFChars(pathStr, nullptr);
  std::string result(pathCStr);
  env->ReleaseStringUTFChars(pathStr, pathCStr);

  vm->DetachCurrentThread();
  return result;
}

std::string
copy_asset_to_files(AAssetManager *mgr, const std::string &asset_name, const std::string &dst_dir) {
  AAsset *asset = AAssetManager_open(mgr, asset_name.c_str(), AASSET_MODE_BUFFER);
  const size_t len = AAsset_getLength(asset);
  std::vector<char> buf(len);

  AAsset_read(asset, buf.data(), len);
  AAsset_close(asset);

  std::string out_path = std::filesystem::path(dst_dir) / asset_name;

  std::ofstream output_file(out_path, std::ofstream::out);

  output_file.write(buf.data(), static_cast<std::streamsize>(buf.size()));
  output_file.close();

  return out_path;
}
