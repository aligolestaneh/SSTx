#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <iostream>

namespace logger {

// Log levels
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

// ANSI color codes
namespace colors {
    const std::string RESET = "\033[0m";
    const std::string RED = "\033[31m";
    const std::string GREEN = "\033[32m";
    const std::string YELLOW = "\033[33m";
    const std::string BLUE = "\033[34m";
    const std::string MAGENTA = "\033[35m";
    const std::string CYAN = "\033[36m";
    const std::string WHITE = "\033[37m";
    const std::string BRIGHT_RED = "\033[91m";
    const std::string BRIGHT_GREEN = "\033[92m";
    const std::string BRIGHT_YELLOW = "\033[93m";
    const std::string BRIGHT_BLUE = "\033[94m";
    const std::string BRIGHT_MAGENTA = "\033[95m";
    const std::string BRIGHT_CYAN = "\033[96m";
    const std::string BOLD = "\033[1m";
}

// Main logging function - takes message and log level
void log(const std::string& message, LogLevel level);

// Configuration
void set_log_level(LogLevel min_level);
void enable_colors(bool enable);
bool is_color_supported();

// Helper functions
std::string get_timestamp();
std::string get_level_string(LogLevel level);
std::string get_level_color(LogLevel level);

} // namespace logger

#endif // UTILS_H 