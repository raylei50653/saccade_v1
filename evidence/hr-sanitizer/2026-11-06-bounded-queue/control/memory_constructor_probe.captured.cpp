#include "saccade/tracking_session.hpp"
#include <chrono>
#include <iostream>
#include <sys/resource.h>
#include <stdexcept>
int main(int argc,char**argv){try{
 if(argc!=3)throw std::invalid_argument("usage: constructor_probe ENGINE REPEATS");
 const unsigned repeats=std::stoul(argv[2]);
 if(!repeats||repeats>20)throw std::invalid_argument("repeats1..20");
 for(unsigned i=0;i<repeats;++i){
  rusage r{};getrusage(RUSAGE_SELF,&r);
  std::cout<<"phase=before_constructor iteration="<<i<<" peak_rss_KiB="<<r.ru_maxrss<<std::endl;
  saccade::TrackingSessionConfig cfg;cfg.profile=saccade::ModelProfile::Hr1280x768;
  cfg.original_width=1920;cfg.original_height=1080;
  saccade::TrackingSession session(argv[1],cfg);
  getrusage(RUSAGE_SELF,&r);std::cout<<"phase=after_constructor iteration="<<i<<" peak_rss_KiB="<<r.ru_maxrss<<std::endl;
  session.close();getrusage(RUSAGE_SELF,&r);std::cout<<"phase=after_close iteration="<<i<<" peak_rss_KiB="<<r.ru_maxrss<<std::endl;
 }
 std::cout<<"constructor_probe complete repeats="<<repeats<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<"constructor_probe: "<<e.what()<<std::endl;return 1;}}
