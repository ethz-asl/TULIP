#include "../extern/pybind11/include/pybind11/eigen.h"
#include "../extern/pybind11/include/pybind11/stl.h"
#include <Eigen/Dense>

#include <unordered_set>
#include <omp.h>


struct Grid3DKey {
public:
    typedef uint16_t key_type;

public:
    Grid3DKey() { k[0] = k[1] = k[2] = 0; }
    Grid3DKey(key_type _kx, key_type _ky, key_type _kz) {
        k[0] = _kx;
        k[1] = _ky;
        k[2] = _kz;
    }
    Grid3DKey(const Grid3DKey& _key) {
        k[0] = _key.k[0];
        k[1] = _key.k[1];
        k[2] = _key.k[2];
    }

    bool operator==(const Grid3DKey& _key) const { return (k[0] == _key.k[0]) && (k[1] == _key.k[1]) && (k[2] == _key.k[2]); }
    bool operator!=(const Grid3DKey& _key) const { return (k[0] != _key.k[0]) || (k[1] != _key.k[1]) || (k[2] != _key.k[2]); }

    const key_type& operator[] (unsigned int _idx) const { return k[_idx]; }
          key_type& operator[] (unsigned int _idx)       { return k[_idx]; }

public:
    key_type k[3];

    struct KeyHash {
        std::size_t operator()(const Grid3DKey& key) const {
            return static_cast<std::size_t>(key.k[0]) + 1447*static_cast<std::size_t>(key.k[1]) + 345637*static_cast<std::size_t>(key.k[2]);
        }
    };
};


struct Grid3DSet {
public:
    typedef std::unordered_set<Grid3DKey, Grid3DKey::KeyHash> Grid3D;

public:
    /**
     * Constructors and destructors
     *
     * @param resolution: voxel size; [m]
     */
    explicit Grid3DSet(double _resolution) : RESOLUTION(_resolution), RESOLUTION_FACTOR(1.0/_resolution) { }
    ~Grid3DSet() { grid_set.clear(); }

    /**
     * Update the grid set given point cloud.
     * Assume that the points within the sensor's field-of-view.
     *
     * @param points: a set of points; [3xN] matrix
     */
    void update_grid_from_points(const Eigen::MatrixXf& _points) {
        for(int n = 0; n < _points.cols(); n++) {
            grid_set.insert(coordinate_to_key(_points.col(n)));
        }
    }

    /**
     * Update the grid set given range image.
     *
     * @param range_image: a range image; [WxH] matrix
     * @param lidar: lidar specification for converting the range image to the points
     */
    void update_grid_from_image(const Eigen::MatrixXf& _range_image, const std::unordered_map<std::string, double>& _lidar) {
        Eigen::VectorXf v_dir = Eigen::VectorXf::LinSpaced(_range_image.rows(), (float)_lidar.at("min_v"), (float)_lidar.at("max_v"));
        Eigen::VectorXf h_dir = Eigen::VectorXf::LinSpaced(_range_image.cols()+1, (float)_lidar.at("min_h"), (float)_lidar.at("max_h"));

        for(unsigned int h = 0; h < h_dir.rows() - 1; h++) {
            for(unsigned int v = 0; v < v_dir.rows(); v++) {
                double r = _range_image(v, h);
                if(r < _lidar.at("min_r") || r > _lidar.at("max_r"))
                    continue;

                double x = std::sin(h_dir[h]) * std::cos(v_dir[v]) * r;
                double y = std::cos(h_dir[h]) * std::cos(v_dir[v]) * r;
                double z = std::sin(v_dir[v]) * r;

                grid_set.insert(coordinate_to_key(x, y, z));
            }
        }
    }

    /// Convert types of data between Euclidean coordinate and Discretized space.
    inline Grid3DKey::key_type coordinate_to_key(double _coordinate) const {
        return ((int)floor(RESOLUTION_FACTOR * _coordinate)) + GRID_MAX_VAL;
    }
    inline Grid3DKey coordinate_to_key(double _x, double _y, double _z) const {
        return Grid3DKey(coordinate_to_key(_x), coordinate_to_key(_y), coordinate_to_key(_z));
    }
    inline Grid3DKey coordinate_to_key(const Eigen::Vector3f& _coordinate) const {
        return coordinate_to_key(_coordinate(0), _coordinate(1), _coordinate(2));
    }
    inline double key_to_coordinate(Grid3DKey::key_type _key) const {
        return (double((int)_key - GRID_MAX_VAL) + 0.5) * RESOLUTION;
    }
    inline Eigen::Vector3f key_to_coordinate(Grid3DKey::key_type _kx, Grid3DKey::key_type _ky, Grid3DKey::key_type _kz) const {
        return { (float)key_to_coordinate(_kx), (float)key_to_coordinate(_ky), (float)key_to_coordinate(_kz) };
    }
    inline Eigen::Vector3f key_to_coordinate(const Grid3DKey& _key) const {
        return key_to_coordinate(_key[0], _key[1], _key[2]);
    }

    /// Get size of grid set
    unsigned int size() const { return grid_set.size(); }

    /// Get grid structure
    const Grid3D& get_grid() const { return grid_set; }

    /// Find the voxel having given key
    bool find(const Grid3DKey& _key) const { return grid_set.find(_key) != grid_set.end(); }

protected:
    Grid3D grid_set;

    // Parameters
    double RESOLUTION;          // cell size [m]
    double RESOLUTION_FACTOR;   // 1.0/RESOLUTION

    const int GRID_MAX_VAL = 32768;
};


namespace py = pybind11;
typedef std::unordered_map<std::string, double> Dictionary;

/**
 * Compute the voxel IoUs of range image pairs.
 *
 * @param gt_rimg: ground truth range images (batched); [Nx(W*H)] matrix
 * @param pred_rimg: predicted range images (batched); [Nx(W*H)] matrix
 * @param voxel_size: cell size for voxelization; [m]
 * @param lidar: lidar specification
 * @return evaluation results: IoU, Precision, Recall, F1; [Nx4] matrix
 */
Eigen::MatrixXf compute_voxel_iou_of_images(Eigen::MatrixXf& _gt_rimg, Eigen::MatrixXf& _pred_rimg,
                                            float _voxel_size, Dictionary& _lidar)
{
    assert(_gt_rimg.rows() == _pred_rimg.rows() && _gt_rimg.cols() == _pred_rimg.cols());

    /// Parse the parameters
    const int N = (int)_gt_rimg.rows();
    const int IMG_HEIGHT = (int)_lidar["channels"];
    const int IMG_WIDTH = (int)_lidar["points_per_ring"];

    /// Set the multi-threading
    omp_set_num_threads(std::min(omp_get_max_threads(), N));

    /// Compute the metrics: IoU, Precision, Recall, F1
    Eigen::MatrixXf eval_results(4, N);

#pragma omp parallel for
    for(int i = 0; i < N; i++) {
        /// 0. Voxelize the range images
        Grid3DSet gt_grid(_voxel_size);
        Grid3DSet pred_grid(_voxel_size);

        Eigen::VectorXf gt_rimg = _gt_rimg.row(i);
        Eigen::VectorXf pred_rimg = _pred_rimg.row(i);
        Eigen::MatrixXf gt_rimg_in = Eigen::Map<Eigen::MatrixXf>(gt_rimg.data(), IMG_WIDTH, IMG_HEIGHT);
        Eigen::MatrixXf pred_rimg_in = Eigen::Map<Eigen::MatrixXf>(pred_rimg.data(), IMG_WIDTH, IMG_HEIGHT);

        gt_grid.update_grid_from_image(gt_rimg_in.transpose(), _lidar);
        pred_grid.update_grid_from_image(pred_rimg_in.transpose(), _lidar);

        /// 1. Compute the evaluation metrics: IoU, Precision & Recall
        unsigned int num_of_unions = gt_grid.size();
        unsigned int num_of_intersections = 0;
        for(const auto& pred_voxel : pred_grid.get_grid())
            gt_grid.find(pred_voxel) ? num_of_intersections++ : num_of_unions++;

        double iou = (double)num_of_intersections / (double)num_of_unions;
        double precision = (double)num_of_intersections / (double)pred_grid.size();
        double recall = (double)num_of_intersections / (double)gt_grid.size();
        double f1 = (2.0 * precision * recall) / (precision + recall);

        eval_results.col(i) << (float)iou, (float)precision, (float)recall, (float)f1;
    }

    return eval_results.transpose();
}

/**
 * Compute the voxel IoUs of points pair.
 *
 * @param gt_samples: ground truth points [P, 3]
 * @param pred_samples: predicted points [P, 3]
 * @param voxel_size: cell size for voxelization [m]
 * @return evaluation results [N, [IoU, Precision, Recall, F1]]
 */
Dictionary compute_voxel_iou_of_points(Eigen::MatrixXf& _gt_points, Eigen::MatrixXf& _pred_points,
                                       float _voxel_size) {
    /// 0. Voxelize the points
    Grid3DSet gt_grid(_voxel_size);
    Grid3DSet pred_grid(_voxel_size);

    gt_grid.update_grid_from_points(_gt_points);
    pred_grid.update_grid_from_points(_pred_points);

    /// 1. Compute the evaluation metrics: IoU, Precision & Recall
    unsigned int num_of_unions = gt_grid.size();
    unsigned int num_of_intersections = 0;
    for(const auto& pred_voxel : pred_grid.get_grid())
        gt_grid.find(pred_voxel) ? num_of_intersections++ : num_of_unions++;

    Dictionary results;
    results["iou"] = (double)num_of_intersections / (double)num_of_unions;
    results["precision"] = (double)num_of_intersections / (double)pred_grid.size();
    results["recall"] = (double)num_of_intersections / (double)gt_grid.size();
    results["f1"] = (2.0 * results["precision"] * results["recall"]) / (results["precision"] + results["recall"]);

    return results;
}

/**
 * Get the voxelization results (voxel centers) of the range image
 *
 * @param range_image: range image; [HxW] matrix
 * @param voxel_size: cell size for voxelization [m]
 * @param lidar: lidar specification
 * @return voxel centers: [Cx3] matrix
 */
Eigen::MatrixXf get_voxel_centers_from_image(Eigen::MatrixXf& _range_image, float _voxel_size, Dictionary& _lidar)
{
    /// 0. Voxelize the range image
    Grid3DSet grid(_voxel_size);
    grid.update_grid_from_image(_range_image, _lidar);

    /// 1. Get the voxel centers
    Eigen::MatrixXf centers(3, grid.size());
    unsigned int n = 0;
    for(const auto& voxel : grid.get_grid())
        centers.col(n++) = grid.key_to_coordinate(voxel);

    return centers.transpose();
}

/**
 * Get the voxelization results (voxel centers) of the points
 *
 * @param points: points; [Px3] matrix
 * @param voxel_size: cell size for voxelization [m]
 * @return voxel centers: [Cx3] matrix
 */
Eigen::MatrixXf get_voxel_centers_from_points(Eigen::MatrixXf& _points, float _resolution)
{
    /// 0. Voxelize the points
    Grid3DSet grid(_resolution);
    grid.update_grid_from_points(_points.transpose());

    /// 1. Get the voxel centers
    Eigen::MatrixXf centers(3, grid.size());
    unsigned int n = 0;
    for(const auto& voxel : grid.get_grid())
        centers.col(n++) = grid.key_to_coordinate(voxel);

    return centers.transpose();
}

PYBIND11_MODULE(voxelizer, m) {
    m.doc() = "utility functions of voxelization";
    m.def("compute_voxel_iou_of_images", &compute_voxel_iou_of_images, "Compute the voxel IoUs of range image pairs",
          py::arg("gt_rimg"), py::arg("pred_rimg"),
          py::arg("resolution"), py::arg("lidar"));
    m.def("compute_voxel_iou_of_points", &compute_voxel_iou_of_points, "Compute the IoU from points",
          py::arg("gt_samples"), py::arg("pred_samples"),
          py::arg("resolution"));
    m.def("get_voxel_centers_from_image", &get_voxel_centers_from_image, "Get the voxel centers from range image",
          py::arg("range_image"), py::arg("resolution"), py::arg("lidar"));
    m.def("get_voxel_centers_from_points", &get_voxel_centers_from_points, "Get the voxel centers from points",
          py::arg("range_samples"), py::arg("resolution"));
}