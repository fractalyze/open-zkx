/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Copyright 2025 The ZKX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef ZKX_BASE_LOGGING_H_
#define ZKX_BASE_LOGGING_H_

#include <sstream>
#include <string>

#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"

namespace zkx::base {

// LogSink support adapted from absl/log/log.h
//
// `LogSink` is an interface which can be extended to intercept and process
// all log messages. LogSink implementations must be thread-safe. A single
// instance will be called from whichever thread is performing a logging
// operation.
class LogEntry {
 public:
  explicit LogEntry(absl::LogSeverity severity, std::string_view message)
      : severity_(severity), message_(message) {}

  explicit LogEntry(absl::LogSeverity severity, std::string_view fname,
                    int line, std::string_view message)
      : severity_(severity), fname_(fname), line_(line), message_(message) {}

  absl::LogSeverity log_severity() const { return severity_; }
  std::string FName() const { return fname_; }
  int Line() const { return line_; }
  std::string ToString() const { return message_; }
  std::string_view text_message() const { return message_; }

  // Returning similar result as `text_message` as there is no prefix in this
  // implementation.
  std::string_view text_message_with_prefix() const { return message_; }

 private:
  const absl::LogSeverity severity_;
  const std::string fname_;
  int line_ = -1;
  const std::string message_;
};

class LogSink {
 public:
  virtual ~LogSink() = default;

  // `Send` is called synchronously during the log statement.  The logging
  // module guarantees not to call `Send` concurrently on the same log sink.
  // Implementations should be careful not to call`LOG` or `CHECK` or take
  // any locks that might be held by the `LOG` caller, to avoid deadlock.
  //
  // `e` is guaranteed to remain valid until the subsequent call to
  // `WaitTillSent` completes, so implementations may store a pointer to or
  // copy of `e` (e.g. in a thread local variable) for use in `WaitTillSent`.
  virtual void Send(const LogEntry& entry) = 0;

  // `WaitTillSent` blocks the calling thread (the thread that generated a log
  // message) until the sink has finished processing the log message.
  // `WaitTillSent` is called once per log message, following the call to
  // `Send`. This may be useful when log messages are buffered or processed
  // asynchronously by an expensive log sink.
  // The default implementation returns immediately. Like `Send`,
  // implementations should be careful not to call `LOG` or `CHECK or take any
  // locks that might be held by the `LOG` caller, to avoid deadlock.
  virtual void WaitTillSent() {}
};

// This is the default log sink. This log sink is used if there are no other
// log sinks registered. To disable the default log sink, set the
// "no_default_logger" Bazel config setting to true or define a
// NO_DEFAULT_LOGGER preprocessor symbol. This log sink will always log to
// stderr.
class DefaultLogSink : public LogSink {
 public:
  void Send(const LogEntry& entry) override;
};

// Add or remove a `LogSink` as a consumer of logging data. Thread-safe.
void AddLogSink(LogSink* sink);
void RemoveLogSink(LogSink* sink);

// Get all the log sinks. Thread-safe.
std::vector<LogSink*> GetLogSinks();

template <typename T>
T&& CheckNotNull(const char* file, int line, const char* exprtext, T&& t) {
  if (t == nullptr) {
    absl::log_internal::LogMessageFatal(file, line) << exprtext;
  }
  return std::forward<T>(t);
}

// Emit "message" as a log message to the log for the specified
// "severity" as if it came from a LOG call at "fname:line"
void LogString(const char* fname, int line, absl::LogSeverity severity,
               const std::string& message);

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char* fname, int line, absl::LogSeverity severity);
  ~LogMessage() override;

  // Change the location of the log message.
  LogMessage& AtLocation(const char* fname, int line);

  // Returns the maximum log level for VLOG statements.
  // E.g., if MaxVLogLevel() is 2, then VLOG(2) statements will produce output,
  // but VLOG(3) will not. Defaults to 0.
  static int MaxVLogLevel();

  // Returns whether VLOG level lvl is activated for the file fname.
  //
  // E.g. if the environment variable TF_CPP_VMODULE contains foo=3 and fname is
  // foo.cc and lvl is <= 3, this will return true. It will also return true if
  // the level is lower or equal to TF_CPP_MAX_VLOG_LEVEL (default zero).
  //
  // It is expected that the result of this query will be cached in the VLOG-ing
  // call site to avoid repeated lookups. This routine performs a hash-map
  // access against the VLOG-ing specification provided by the env var.
  static bool VmoduleActivated(const char* fname, int level);

 protected:
  void GenerateLogMessage();

 private:
  const char* fname_;
  int line_;
  absl::LogSeverity severity_;
};

// Split the text into multiple lines and log each line with the given
// severity, filename, and line number.
void LogLines(absl::LogSeverity sev, std::string_view text, const char* fname,
              int lineno);
inline void LogLinesINFO(std::string_view text, const char* fname, int lineno) {
  return LogLines(absl::LogSeverity::kInfo, text, fname, lineno);
}
inline void LogLinesWARNING(std::string_view text, const char* fname,
                            int lineno) {
  return LogLines(absl::LogSeverity::kWarning, text, fname, lineno);
}
inline void LogLinesERROR(std::string_view text, const char* fname,
                          int lineno) {
  return LogLines(absl::LogSeverity::kError, text, fname, lineno);
}
inline void LogLinesFATAL(std::string_view text, const char* fname,
                          int lineno) {
  return LogLines(absl::LogSeverity::kFatal, text, fname, lineno);
}

}  // namespace zkx::base

#define CHECK_NOTNULL(val)                                                     \
  ::zkx::base::CheckNotNull(__FILE__, __LINE__, "'" #val "' Must be non NULL", \
                            (val))

// Note that STRING is evaluated regardless of whether it will be logged.
#define ZKX_LOG_LINES(SEV, STRING) \
  ::zkx::base::LogLines##SEV(STRING, __FILE__, __LINE__)

// Like LOG_LINES, but only logs if VLOG is enabled for the given level.
// STRING is evaluated only if it will be logged.
#define ZKX_VLOG_LINES(LEVEL, STRING)                   \
  do {                                                  \
    if (VLOG_IS_ON(LEVEL)) ZKX_LOG_LINES(INFO, STRING); \
  } while (false)

#endif  // ZKX_BASE_LOGGING_H_
