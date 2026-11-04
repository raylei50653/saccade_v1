bash -c cmake\ -S\ .\ -B\ build/cpu\ -DCMAKE_BUILD_TYPE=Release\ \&\&\ cmake\ --build\ build/cpu\ --parallel\ 8\ \&\&\ ctest\ --test-dir\ build/cpu\ --output-on-failure 
