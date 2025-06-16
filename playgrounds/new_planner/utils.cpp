#include "utils.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif

namespace logger {

// Global configuration
static LogLevel g_min_log_level = LogLevel::DEBUG;
static bool g_colors_enabled = true;
static bool g_color_support_checked = false;
static bool g_color_supported = false;

// Check if colors are supported
bool is_color_supported() {
    if (!g_color_support_checked) {
        g_color_support_checked = true;
        
#ifdef _WIN32
        // Windows 10 version 1511 and later support ANSI colors
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        if (hOut != INVALID_HANDLE_VALUE) {
            DWORD dwMode = 0;
            if (GetConsoleMode(hOut, &dwMode)) {
                dwMode |= 0x0004; // ENABLE_VIRTUAL_TERMINAL_PROCESSING
                g_color_supported = SetConsoleMode(hOut, dwMode);
            }
        }
#else
        // Check if stdout is a terminal and supports colors
        g_color_supported = isatty(fileno(stdout)) && 
                           (getenv("TERM") != nullptr) && 
                           (std::string(getenv("TERM")) != "dumb");
#endif
    }
    return g_color_supported;
}

// Get current timestamp
std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%H:%M:%S");
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

// Get level string
std::string get_level_string(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::INFO:    return "INFO ";
        case LogLevel::WARNING: return "WARN ";
        case LogLevel::ERROR:   return "ERROR";
        default:                return "UNKNOWN";
    }
}

// Get color for log level
std::string get_level_color(LogLevel level) {
    if (!g_colors_enabled || !is_color_supported()) {
        return "";
    }
    
    switch (level) {
        case LogLevel::DEBUG:   return colors::WHITE;
        case LogLevel::INFO:    return colors::BRIGHT_CYAN;
        case LogLevel::WARNING: return colors::BRIGHT_YELLOW;
        case LogLevel::ERROR:   return colors::BRIGHT_RED;
        default:                return colors::RESET;
    }
}

// Main logging function - takes message and log level
void log(const std::string& message, LogLevel level) {
    // Check if we should log this level
    if (level < g_min_log_level) {
        return;
    }
    
    std::string timestamp = get_timestamp();
    std::string level_str = get_level_string(level);
    std::string color = get_level_color(level);
    std::string reset = (g_colors_enabled && is_color_supported()) ? colors::RESET : "";
    
    // Format: [HH:MM:SS.mmm] [LEVEL] message
    std::cout << color 
              << "[" << timestamp << "] "
              << "[" << level_str << "] "
              << message 
              << reset 
              << std::endl;
}

// Configuration functions
void set_log_level(LogLevel min_level) {
    g_min_log_level = min_level;
}

void enable_colors(bool enable) {
    g_colors_enabled = enable;
}

} // namespace logger 