/*
重新设计
使用第三方库：
boost: filesystem
spdlog:日志
json:工作卡解码
*/

#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

using namespace std;

int
test_boost_filesystem(int argc, char **argv)
{
  boost::filesystem::path path("./Makefile");

  if(boost::filesystem::exists(path))
    {
      cout<<"exists"<<endl;
    }

  return(0);

}

// https://github.com/nlohmann/json

#include<string>
#include <nlohmann/json.hpp>

int
test_jsoncpp(int argc, char **argv)
{
  using json = nlohmann::json;
  json j;
  j["pi"] = 3.141;
  j["happy"] = true;
  j["name"] = "Niels";
  j["nothing"] = nullptr;
  j["answer"]["everything"] = 42;
  j["list"] = { 1, 0, 2 };
  j["object"] = { {"currency", "USD"}, {"value", 42.99} };

  json j2 = {
    {"pi", 3.141},
    {"happy", true},
    {"name", "Niels"},
    {"nothing", nullptr},
    {"answer", {
      {"everything", 42}
    }},
    {"list", {1, 0, 2}},
    {"object", {
      {"currency", "USD"},
      {"value", 42.99}
    }}
  };

  auto j3 = R"(
    {
      "happy": true,
      "pi": 3.141
    }
  )"_json;

  // explicit conversion to string
  std::string s = j.dump();    // {\"happy\":true,\"pi\":3.141}

  // serialization with pretty printing
  // pass in the amount of spaces to indent
  std::cout << j.dump(4) << std::endl;


  // read a JSON file
  std::ifstream i("file.json");
  i >> j;

  // write prettified JSON to another file
  std::ofstream o("pretty.json");
  o <<j<< std::endl;

  return(0);
}


// https://github.com/gabime/spdlog

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/daily_file_sink.h"

int
test_spdlog(int argc, char **argv)
{

  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  console_sink->set_level(spdlog::level::warn);
  console_sink->set_pattern("[multi_sink_example] [%^%l%$] %v");

  auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/multisink.txt", true);
  file_sink->set_level(spdlog::level::trace);

  spdlog::logger logger("multi_sink", {console_sink, file_sink});
  logger.set_level(spdlog::level::debug);
  logger.warn("this should appear in both console and file");
  logger.info("this message should not appear in the console, only in the file");
  return(0);
}

#include "xmig/constant.hpp"
#include "xmig/head.hpp"

int test_xmig(int agc,char **argv)
{
  XMIG::NTRACE_PER_LOOP=100;
  XMIG::INV_KPKP[100]=100;

  return(0);
}

int
main(int argc, char **argv)
{

  test_boost_filesystem(argc,argv);
  test_spdlog(argc,argv);
  test_jsoncpp(argc,argv);

  test_xmig(argc,argv);

  return(0);
}
