#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <cstdio>
#include <iostream>
#include <matrix.hpp>
#include <fcntl.h>
// #include <io.h>

#pragma once

using namespace std;

struct Converter {

	//convert rgb to tone of gray
	static uint8_t grayOf(int const& r, int const& g, int const& b) {
		return static_cast<uint8_t>(0.299f * r + 0.587f * g + 0.114f * b + 0.5f);
	}


    //read stdin
    static vector<unsigned char> getIn() {
        vector<unsigned char> in;


        //set stdin to binary
        // _setmode(_fileno(stdin), _O_BINARY);

        char buf[4096];

        size_t n;
        while ((n = fread(buf, 1, sizeof buf, stdin)) > 0) in.insert(in.end(), buf, buf + n);

        //test for input
        if (in.empty()) cerr << "Error: no input on stdin\n";

        return in;
    }


    //read file
    static vector<unsigned char> getIn(string const& filename) {
        vector<unsigned char> in;

        FILE* file = fopen(filename.c_str(), "rb");

        if (!file) {
            cerr << "Error: cannot open file: " << filename << "\n";
            return {};
        }

        unsigned char buf[4096];
        size_t n;

        while ((n = fread(buf, 1, sizeof(buf), file)) > 0) {
            in.insert(in.end(), buf, buf + n);
        }

        fclose(file);

        return in;
    }


    //decodes the image and converts it to gray value scale
    static Matrix<float> decodeAndConvert(vector<unsigned char> const& in) {
        //decode (regardless of source bit depth/color, get RGBA, 8-bit)
        int w = 0, h = 0, channels = 0;
        stbi_uc* px = stbi_load_from_memory(in.data(), (int)in.size(), &w, &h, &channels, 4);

        //test if empty
        if (!px) cerr << "Error: cannot identify image file\n";

        //convert RGBA to grayscale
        Matrix<float> gray;
        gray.addRow(vector<float>((size_t)w * h));
        for (int i = 0; i < w * h; ++i) {
            const stbi_uc* p = px + 4 * i;
            gray[0][i] = (float)Converter::grayOf(p[0], p[1], p[2]) / 255.0f;     //ignore alpha
        }

        //free memory
        stbi_image_free(px);

        return gray;
    }
};
