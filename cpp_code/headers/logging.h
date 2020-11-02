
#ifndef _LOGGING_H
#define _LOGGING_H

#include <iostream>
#include <fstream>
#include <string>

class Logger {
   public:
    Logger(){};
    Logger(std::string log_file, bool write_to_cout = true)
     : _log(log_file), _write_to_stdout(write_to_cout){};

    template <class T>
    Logger& operator<<(const T& val) {
        if (_write_to_stdout) std::cout << val;
        if (_log.is_open()) _log << val;
        return *this;
    }

    void flush() {
        if (_write_to_stdout) std::cout.flush();
        if (_log.is_open()) _log.flush();
    }

   private:
    std::ofstream _log;
    bool _write_to_stdout = true;
};

#endif