#pragma once

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace guide_image {

struct Image {
    int width = 0;
    int height = 0;
    int channels = 0; // 1 for P5, 3 for P6
    std::vector<std::uint8_t> pixels; // compact interleaved rows
};

inline std::string next_token(std::istream& input) {
    for (;;) {
        const int c = input.peek();
        if (c == EOF) throw std::runtime_error("unexpected end of PNM header");
        if (std::isspace(static_cast<unsigned char>(c))) { input.get(); continue; }
        if (c == '#') { std::string comment; std::getline(input, comment); continue; }
        break;
    }
    std::string token;
    for (;;) {
        const int c = input.peek();
        if (c == EOF || std::isspace(static_cast<unsigned char>(c))) break;
        token.push_back(static_cast<char>(input.get()));
    }
    if (token.empty()) throw std::runtime_error("empty PNM token");
    return token;
}

inline int decimal_token(const std::string& token) {
    std::size_t consumed = 0;
    const long long value = std::stoll(token, &consumed, 10);
    if (consumed != token.size() || value <= 0 || value > std::numeric_limits<int>::max())
        throw std::runtime_error("invalid PNM dimension or max value");
    return static_cast<int>(value);
}

inline std::size_t byte_count(int width, int height, int channels) {
    if (width <= 0 || height <= 0 || (channels != 1 && channels != 3))
        throw std::runtime_error("invalid image shape");
    const std::size_t w = static_cast<std::size_t>(width);
    const std::size_t h = static_cast<std::size_t>(height);
    const std::size_t c = static_cast<std::size_t>(channels);
    constexpr std::size_t max_bytes = 256u * 1024u * 1024u;
    if (w > max_bytes / c || h > max_bytes / (w * c))
        throw std::runtime_error("PNM image exceeds 256 MiB teaching limit");
    return w * h * c;
}

inline Image read_pnm(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open PNM input: " + path);
    const std::string magic = next_token(input);
    if (magic != "P5" && magic != "P6") throw std::runtime_error("only binary P5/P6 supported");
    Image image;
    image.channels = magic == "P5" ? 1 : 3;
    image.width = decimal_token(next_token(input));
    image.height = decimal_token(next_token(input));
    if (decimal_token(next_token(input)) != 255)
        throw std::runtime_error("only 8-bit PNM max value 255 supported");
    char separator = 0;
    if (!input.get(separator) || !std::isspace(static_cast<unsigned char>(separator)))
        throw std::runtime_error("missing PNM header/data separator");
    if (separator == '\r' && input.peek() == '\n') input.get();
    image.pixels.resize(byte_count(image.width, image.height, image.channels));
    input.read(reinterpret_cast<char*>(image.pixels.data()),
               static_cast<std::streamsize>(image.pixels.size()));
    if (input.gcount() != static_cast<std::streamsize>(image.pixels.size()))
        throw std::runtime_error("truncated PNM pixel data");
    return image;
}

inline void write_pnm(const std::string& path, const Image& image) {
    if (image.pixels.size() != byte_count(image.width, image.height, image.channels))
        throw std::runtime_error("PNM pixel count does not match shape");
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot open PNM output: " + path);
    output << (image.channels == 1 ? "P5\n" : "P6\n")
           << image.width << ' ' << image.height << "\n255\n";
    output.write(reinterpret_cast<const char*>(image.pixels.data()),
                 static_cast<std::streamsize>(image.pixels.size()));
    if (!output) throw std::runtime_error("failed to write PNM output: " + path);
}

} // namespace guide_image
