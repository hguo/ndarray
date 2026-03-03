#include <iostream>
#include <ndarray/config.hh>
#include <ndarray/ndarray_stream.hh>
#include "cxxopts.hpp"

#if NDARRAY_HAVE_MPI
#include <mpi.h>
#endif

std::string input_yaml_filename;
std::string data_path_prefix;

int main(int argc, char **argv)
{
#if NDARRAY_HAVE_MPI
  MPI_Init(&argc, &argv);
#endif
  ftk::ndarray_init();

  cxxopts::Options options(argv[0]);
  options.add_options()
    ("input,i", "Input yaml file", cxxopts::value<std::string>(input_yaml_filename))
    ("prefix,p", "Data path prefix", cxxopts::value<std::string>(data_path_prefix))
    ("help,h", "Print this information");

  options.parse_positional({"input"});
  auto results = options.parse(argc, argv);

  if (results.count("help")) {
    std::cerr << options.help() << '\n';
    exit(0);
  }

  if (!results.count("input")) {
    std::cerr << options.help() << '\n';
    exit(1);
  }

  auto stream = std::make_shared<ftk::stream<>>();

  if (!data_path_prefix.empty())
    stream->set_path_prefix( data_path_prefix );

  stream->parse_yaml(input_yaml_filename);

  auto gs = stream->read_static();
  gs->print_info(std::cerr);

  for (int i = 0; i < stream->total_timesteps(); i ++)  {
    auto g = stream->read(i);
    g->print_info(std::cerr);
  }

  ftk::ndarray_finalize();

#if NDARRAY_HAVE_MPI
  MPI_Finalize();
#endif

  return 0;
}
