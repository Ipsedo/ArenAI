//
// Created by samuel on 25/05/18.
//
#include "assets.h"

std::string getFileText(AAssetManager *mgr, std::string fileName) {
	// Open your file
	AAsset *file = AAssetManager_open(mgr, fileName.c_str(), AASSET_MODE_BUFFER);
	// Get the file length
	off_t fileLength = AAsset_getLength(file);

	// Allocate memory to read your file
	char *fileContent = new char[fileLength + 1];

	// Read your file
	AAsset_read(file, fileContent, size_t(fileLength));
	// For safety you can add a 0 terminating character at the end of your file ...
	fileContent[fileLength] = '\0';

	// Do whatever you want with the content of the file

	return std::string(fileContent);
}

