#include <cstdint>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "codelibrary/base/array.h"
#include "codelibrary/geometry/kernel/point_3d.h"
#include "codelibrary/util/tree/kd_tree.h"

#include "vccs_knn_supervoxel.h"
#include "vccs_supervoxel.h"

namespace py = pybind11;

namespace {

cl::Array<cl::RPoint3D> ToPointArray(const py::array_t<double, py::array::c_style | py::array::forcecast>& pts) {
    if (pts.ndim() != 2 || pts.shape(1) != 3) {
        throw py::value_error("pts must have shape (N, 3)");
    }

    const auto n_points = static_cast<int>(pts.shape(0));
    if (n_points <= 0) {
        throw py::value_error("pts must contain at least one point");
    }

    cl::Array<cl::RPoint3D> points;
    points.reserve(n_points);

    auto p = pts.unchecked<2>();
    for (int i = 0; i < n_points; ++i) {
        points.emplace_back(p(i, 0), p(i, 1), p(i, 2));
    }

    return points;
}

void ValidateColors(const py::array& colors, int n_points) {
    if (colors.ndim() != 2 || colors.shape(1) != 3) {
        throw py::value_error("colors must have shape (N, 3)");
    }

    if (colors.shape(0) != n_points) {
        throw py::value_error("colors and pts must have the same number of rows");
    }
}

py::array_t<std::int32_t> LabelsToNumpy(const cl::Array<int>& labels) {
    py::array_t<std::int32_t> out(labels.size());
    auto out_view = out.mutable_unchecked<1>();

    for (int i = 0; i < labels.size(); ++i) {
        out_view(i) = static_cast<std::int32_t>(labels[i]);
    }
    return out;
}

py::array_t<std::int32_t> SegmentVccs(const cl::Array<cl::RPoint3D>& points,
                                      double voxel_resolution,
                                      double seed_resolution,
                                      double spatial_importance,
                                      double normal_importance) {
    if (voxel_resolution <= 0.0) {
        throw py::value_error("voxel_resolution must be > 0");
    }
    if (seed_resolution < 2.0 * voxel_resolution) {
        throw py::value_error("seed_resolution must be >= 2 * voxel_resolution for method='vccs'");
    }

    VCCSSupervoxel segmenter(points.begin(), points.end(), voxel_resolution, seed_resolution);
    segmenter.set_spatial_importance(spatial_importance);
    segmenter.set_normal_importance(normal_importance);

    cl::Array<int> labels;
    cl::Array<VCCSSupervoxel::Supervoxel> supervoxels;
    segmenter.Segment(&labels, &supervoxels);

    return LabelsToNumpy(labels);
}

py::array_t<std::int32_t> SegmentVccsKnn(const cl::Array<cl::RPoint3D>& points,
                                         double seed_resolution,
                                         double spatial_importance,
                                         double normal_importance) {
    if (seed_resolution <= 0.0) {
        throw py::value_error("seed_resolution must be > 0 for method='vccs_knn'");
    }
    if (points.size() < 20) {
        throw py::value_error("method='vccs_knn' requires at least 20 points");
    }

    cl::KDTree<cl::RPoint3D> kd_tree(points.begin(), points.end());
    VCCSKNNSupervoxel segmenter(kd_tree, seed_resolution);
    segmenter.set_spatial_importance(spatial_importance);
    segmenter.set_normal_importance(normal_importance);

    cl::Array<int> labels;
    cl::Array<VCCSKNNSupervoxel::Supervoxel> supervoxels;
    segmenter.Segment(&labels, &supervoxels);

    return LabelsToNumpy(labels);
}

py::array_t<std::int32_t> SegmentPcd(
        const py::array_t<double, py::array::c_style | py::array::forcecast>& pts,
        const py::array& colors,
        const std::string& method,
        double voxel_resolution,
        double seed_resolution,
        double spatial_importance,
        double normal_importance) {
    cl::Array<cl::RPoint3D> points = ToPointArray(pts);
    ValidateColors(colors, points.size());

    if (spatial_importance < 0.0) {
        throw py::value_error("spatial_importance must be >= 0");
    }
    if (normal_importance < 0.0) {
        throw py::value_error("normal_importance must be >= 0");
    }

    if (method == "vccs") {
        return SegmentVccs(points, voxel_resolution, seed_resolution,
                           spatial_importance, normal_importance);
    }
    if (method == "vccs_knn") {
        return SegmentVccsKnn(points, seed_resolution,
                              spatial_importance, normal_importance);
    }

    throw py::value_error("method must be one of: 'vccs', 'vccs_knn'");
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "Python bindings for supervoxel segmentation";

    m.def(
        "segment_pcd",
        &SegmentPcd,
        py::arg("pts"),
        py::arg("colors"),
        py::kw_only(),
        py::arg("method") = "vccs",
        py::arg("voxel_resolution") = 0.03,
        py::arg("seed_resolution") = 1.0,
        py::arg("spatial_importance") = 0.4,
        py::arg("normal_importance") = 1.0,
        R"pbdoc(
Segment a point cloud into supervoxels.

Parameters
----------
pts : numpy.ndarray
    Point coordinates with shape (N, 3).
colors : numpy.ndarray
    Color matrix with shape (N, 3). Used for input validation only.
method : str, optional
    Segmentation backend: "vccs" (default) or "vccs_knn".
voxel_resolution : float, optional
    Voxel resolution used by VCCS backend.
seed_resolution : float, optional
    Seed resolution used by selected backend.
spatial_importance : float, optional
    Spatial distance weight.
normal_importance : float, optional
    Normal similarity weight.

Returns
-------
numpy.ndarray
    Integer labels of shape (N,), where each value is a supervoxel index.
)pbdoc");
}
