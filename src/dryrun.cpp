#include <iostream>
#include <ndarray/ndarray_group_stream.hh>
#include "cxxopts.hpp"

std::string input_yaml_filename;

int main(int argc, char **argv)
{
#if NDARRAY_HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  cxxopts::Options options(argv[0]);
  options.add_options()
    ("input,i", "Input yaml file", cxxopts::value<std::string>(input_yaml_filename))
    ("help,h", "Print this information");

  options.parse_positional({"input"});
  auto results = options.parse(argc, argv);

  if (!results.count("input") || results.count("help")) {
    std::cerr << options.help() << std::endl;
    exit(0);
  }

  fprintf(stderr, "input yaml file: %s\n", input_yaml_filename.c_str());

  std::shared_ptr<ndarray::stream> stream(new ndarray::stream);
  stream->parse_yaml(input_yaml_filename);

  auto gs = stream->read_static();
  gs->print_info(std::cerr);

  for (int i = 0; i < stream->total_timesteps(); i ++)  {
    auto g = stream->read(i);
    g->print_info(std::cerr);
  }

#if NDARRAY_HAVE_MPI
  MPI_Finalize();
#endif

  return 0;
}
