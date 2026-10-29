env ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 ctest --test-dir build-s01-reference/sanitize --verbose 
