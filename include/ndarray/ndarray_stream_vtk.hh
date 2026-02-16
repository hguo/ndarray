#ifndef _NDARRAY_STREAM_VTK_HH
#define _NDARRAY_STREAM_VTK_HH

#include <ndarray/config.hh>

#if NDARRAY_HAVE_YAML && NDARRAY_HAVE_VTK

#include <ndarray/ndarray_stream.hh>
#include <vtkXMLPUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkXMLImageDataReader.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkResampleToImage.h>

namespace ftk {

/**
 * @brief VTI (VTK ImageData) input substream
 *
 * Reads variables from VTI files.
 */
template <typename StoragePolicy = native_storage>
struct substream_vti : public substream<StoragePolicy> {
  using stream_type = stream<StoragePolicy>;
  using group_type = ndarray_group<StoragePolicy>;
  substream_vti(stream_type& s) : substream<StoragePolicy>(s) {}
  bool require_input_files() { return true; }
  bool require_dimensions() { return false; }
  int direction() { return SUBSTREAM_DIR_INPUT;}

  void initialize(YAML::Node);
  void read(int, std::shared_ptr<group_type>);
};

/**
 * @brief VTI (VTK ImageData) output substream
 *
 * Writes variables to VTI files.
 */
template <typename StoragePolicy = native_storage>
struct substream_vti_o : public substream<StoragePolicy> {
  using stream_type = stream<StoragePolicy>;
  using group_type = ndarray_group<StoragePolicy>;
  substream_vti_o(stream_type& s) : substream<StoragePolicy>(s) {}
  bool require_input_files() { return false; }
  bool require_dimensions() { return false; }
  int direction() { return SUBSTREAM_DIR_OUTPUT;}

  void initialize(YAML::Node);
  void read(int, std::shared_ptr<group_type>);
};

/**
 * @brief VTU resample substream
 *
 * Reads VTU (unstructured grid) files and resamples them to regular grids.
 */
template <typename StoragePolicy = native_storage>
struct substream_vtu_resample : public substream<StoragePolicy> {
  using stream_type = stream<StoragePolicy>;
  using group_type = ndarray_group<StoragePolicy>;
  substream_vtu_resample(stream_type& s) : substream<StoragePolicy>(s) {}
  bool require_input_files() { return true; }
  bool require_dimensions() { return true; }
  int direction() { return SUBSTREAM_DIR_INPUT;}

  void initialize(YAML::Node);
  void read(int, std::shared_ptr<group_type>);

public:
  bool has_bounds = false;
  std::array<double, 3> lb, ub;
};

///////////
// Implementation
///////////

template <typename StoragePolicy>
inline void substream_vti<StoragePolicy>::initialize(YAML::Node y)
{
  this->total_timesteps = this->filenames.size();
}

template <typename StoragePolicy>
inline void substream_vti<StoragePolicy>::read(int i, std::shared_ptr<group_type> g)
{
  const auto f = this->filenames[i]; // assume each vti has only one timestep

  vtkSmartPointer<vtkXMLImageDataReader> reader = vtkXMLImageDataReader::New();
  reader->SetFileName(f.c_str());
  reader->Update();
  vtkSmartPointer<vtkImageData> vti = reader->GetOutput();

  for (const auto &var : this->variables) {
    std::shared_ptr<ndarray_base> p = ndarray_base::new_from_vtk_image_data(vti, var.name);
    g->set(var.name, p);
  }
}

template <typename StoragePolicy>
inline void substream_vti_o<StoragePolicy>::initialize(YAML::Node y)
{
}

template <typename StoragePolicy>
inline void substream_vti_o<StoragePolicy>::read(int i, std::shared_ptr<group_type> g)
{
  const auto f = series_filename(this->filename_pattern, i);
  fprintf(stderr, "writing step %d, f=%s\n", i, f.c_str());

  vtkSmartPointer<vtkImageData> vti = vtkImageData::New();

  // assuming all variables have the same shape
  bool first = true;
  for (const auto &var : this->variables) {
    auto arr = g->get(var.name);

    if (first) {
      if (arr->multicomponents()) {
        if (arr->nd() == 3) vti->SetDimensions(arr->shapef(1), arr->shapef(2), 1);
        else vti->SetDimensions(arr->shapef(1), arr->shapef(2), arr->shapef(3)); // nd == 4
      } else {
        if (arr->nd() == 2) vti->SetDimensions(arr->shapef(0), arr->shapef(1), 1);
        else vti->SetDimensions(arr->shapef(0), arr->shapef(1), arr->shapef(2)); // nd == 3
      }
      first = false;
    }

    auto da = arr->to_vtk_data_array();
    da->SetName(var.name.c_str());
    vti->GetPointData()->AddArray(da);
  }

  vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkXMLImageDataWriter::New();
  writer->SetFileName(f.c_str());
  writer->SetInputData(vti);
  writer->Write();
}

template <typename StoragePolicy>
inline void substream_vtu_resample<StoragePolicy>::initialize(YAML::Node y)
{
  this->total_timesteps = this->filenames.size();

  if (!this->has_dimensions())
    nd::fatal("missing dimensions for vtu_resample");
}

template <typename StoragePolicy>
inline void substream_vtu_resample<StoragePolicy>::read(int i, std::shared_ptr<group_type> g)
{
  const auto f = this->filenames[i];

  vtkSmartPointer<vtkUnstructuredGrid> grid;

  const int ext = file_extension(f);
  if (ext == FILE_EXT_VTU) {
    vtkSmartPointer<vtkXMLUnstructuredGridReader> reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
    reader->SetFileName(f.c_str());
    reader->Update();
    grid = reader->GetOutput();
  } else if (ext == FILE_EXT_PVTU) {
    vtkSmartPointer<vtkXMLPUnstructuredGridReader> reader = vtkSmartPointer<vtkXMLPUnstructuredGridReader>::New();
    reader->SetFileName(f.c_str());
    reader->Update();
    grid = reader->GetOutput();
  }

  vtkSmartPointer<vtkResampleToImage> resample = vtkSmartPointer<vtkResampleToImage>::New();
  std::array<int, 3> dims = {1, 1, 1};
  for (int i = 0; i < this->dimensions.size(); i ++)
    dims[i] = this->dimensions[i];
  resample->SetSamplingDimensions(dims[0], dims[1], dims[2]);

  if (this->has_bounds) {
    resample->SetUseInputBounds(false);
    resample->SetSamplingBounds(this->lb[0], this->lb[1], this->lb[2], this->ub[0], this->ub[1], this->ub[2]);
  } else
    resample->SetUseInputBounds(true);

  resample->SetInputDataObject(grid);
  resample->Update();

  vtkSmartPointer<vtkImageData> vti = resample->GetOutput();
  for (const auto &var : this->variables) {
    std::shared_ptr<ndarray_base> p = ndarray_base::new_from_vtk_image_data(vti, var.name);
    g->set(var.name, p);
  }
}

} // namespace ftk

#endif // NDARRAY_HAVE_YAML && NDARRAY_HAVE_VTK

#endif // _NDARRAY_STREAM_VTK_HH
