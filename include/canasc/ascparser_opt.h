#ifndef ASCPARSER_OPT_H
#define ASCPARSER_OPT_H

#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>

#include "message.h"

/**
 * Optimized ASC Parser
 *
 * Key optimizations over original ASCParser:
 * 1. Memory-mapped file I/O (or large buffer reads) instead of getline()
 * 2. string_view for zero-copy tokenization
 * 3. Hand-rolled fast parsing for timestamps, hex, and integers
 * 4. Fixed-size data array (no heap allocation for message payload)
 * 5. Inline parsing without intermediate Token/Tokenizer objects
 * 6. Cache-friendly sequential memory access
 */
class ASCParserOpt {
public:
    explicit ASCParserOpt(const std::string& filename);
    ~ASCParserOpt() noexcept;

    // Disable copy
    ASCParserOpt(const ASCParserOpt&) = delete;
    ASCParserOpt& operator=(const ASCParserOpt&) = delete;

    // Allow move
    ASCParserOpt(ASCParserOpt&&) noexcept;
    ASCParserOpt& operator=(ASCParserOpt&&) noexcept;

    void reinit();
    std::unique_ptr<Message> getMessage();
    bool fileEnded() const;

    // Header accessors
    const std::string& startTime() const { return _date.time; }
    const std::string& weekday() const { return _date.weekday; }
    const std::string& year() const { return _date.year; }
    const std::string& month() const { return _date.month; }
    const std::string& day() const { return _date.day; }
    const std::string& base() const { return _base; }
    const std::string& timestamp_format() const { return _timestamp_format; }
    bool internal_events_logged() const { return _internal_events_logged; }

private:
    // Fast parsing utilities (inline for speed)
    static inline bool isDigit(char c) { return c >= '0' && c <= '9'; }
    static inline bool isHexDigit(char c) {
        return isDigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
    }
    static inline int hexCharToInt(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return 0;
    }

    // Fast number parsing
    static double parseDouble(std::string_view sv);
    static int parseInt(std::string_view sv);
    static int parseHex(std::string_view sv);
    static uint8_t parseHexByte(std::string_view sv);

    // Tokenization helpers
    static std::string_view nextToken(const char*& ptr, const char* end);
    static void skipWhitespace(const char*& ptr, const char* end);

    // File handling
    bool loadBuffer();
    bool getNextLine();

    // Header parsing
    bool parseHeader();
    bool checkHeader();
    void parseDate();

    // Message parsing
    bool parseMessage(Message& msg);
    bool parseCAN(Message& msg, const char* ptr, const char* end);
    bool parseCANFD(Message& msg, const char* ptr, const char* end);

    // File extension validation
    static std::string getFileExtension(const std::string& filePath);

    // Date structure
    struct Date {
        std::string weekday;
        std::string day;
        std::string month;
        std::string time;
        std::string year;
    } _date;

    // File I/O
    std::ifstream _ifs;
    std::string _filename;

    // Large read buffer for efficiency
    static constexpr size_t BUFFER_SIZE = 1024 * 1024; // 1MB buffer
    std::unique_ptr<char[]> _buffer;
    size_t _buffer_len = 0;
    size_t _buffer_pos = 0;

    // Current line pointers (into buffer)
    const char* _line_start = nullptr;
    const char* _line_end = nullptr;

    // Header info
    std::string _base;
    std::string _timestamp_format;
    bool _internal_events_logged = false;
    bool _eof_reached = false;
    bool _hex_base = true; // cached: _base == "hex"
};

#endif /* ASCPARSER_OPT_H */
