#pragma once
#include <chrono>

namespace krans::utils {

/** Simple steady-clock timer with reset and seconds-precision elapsed(). */
class Timer {
public:
  /** Clock alias. */
  using clock = std::chrono::steady_clock;
  /** Construct and start the timer. */
  Timer() : t0_(clock::now()) {}
  /** Reset start time to now. */
  void reset() { t0_ = clock::now(); }
  /** Elapsed wall time in seconds since last reset/construct. */
  double elapsed_seconds() const {
    return std::chrono::duration<double>(clock::now() - t0_).count();
  }

private:
  /** Start time point. */
  clock::time_point t0_;
};

} // namespace krans::utils
