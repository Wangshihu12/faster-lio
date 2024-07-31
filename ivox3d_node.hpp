#include <pcl/common/centroid.h>
#include <algorithm>
#include <cmath>
#include <list>
#include <vector>

#include "hilbert.hpp"

namespace faster_lio {

// squared distance of two pcl points
// 计算两个 PointT 类型点的平方距离
template <typename PointT>
inline double distance2(const PointT& pt1, const PointT& pt2) {
    Eigen::Vector3f d = pt1.getVector3fMap() - pt2.getVector3fMap();
    return d.squaredNorm();
}

// convert from pcl point to eigen
// 将 pointT 类型的点转换为 Eigen
template <typename T, int dim, typename PointType>
inline Eigen::Matrix<T, dim, 1> ToEigen(const PointType& pt) {
    return Eigen::Matrix<T, dim, 1>(pt.x, pt.y, pt.z);
}
// 将 PointXYZ 类型的点转换为 Eigen
template <>
inline Eigen::Matrix<float, 3, 1> ToEigen<float, 3, pcl::PointXYZ>(const pcl::PointXYZ& pt) {
    return pt.getVector3fMap();
}

template <>
inline Eigen::Matrix<float, 3, 1> ToEigen<float, 3, pcl::PointXYZI>(const pcl::PointXYZI& pt) {
    return pt.getVector3fMap();
}

template <>
inline Eigen::Matrix<float, 3, 1> ToEigen<float, 3, pcl::PointXYZINormal>(const pcl::PointXYZINormal& pt) {
    return pt.getVector3fMap();
}

template <typename PointT, int dim = 3>
class IVoxNode {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    struct DistPoint;

    IVoxNode() = default;
    IVoxNode(const PointT& center, const float& side_length) {}  /// same with phc

    // 插入点到 points_
    void InsertPoint(const PointT& pt);

    // 判断 points_ 是否为空
    inline bool Empty() const;

    // 返回 points_.size()
    inline std::size_t Size() const;

    // 输入索引,获取对应的点,返回 points_[idx]
    inline PointT GetPoint(const std::size_t idx) const;

    // 寻找距离 cur_pt 最近的 k 个点存入 dis_points，最大距离不超过 max_range，返回的是最近的个数
    int KNNPointByCondition(std::vector<DistPoint>& dis_points, const PointT& point, const int& K,
                            const double& max_range);

   private:
    std::vector<PointT> points_;    // IVOX 中 PointT 类型的点
};

// 基于哈希码的空间索引机制
template <typename PointT, int dim = 3>
class IVoxNodePhc {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    struct DistPoint;
    // 哈希桶相关的点列表
    struct PhcCube;

    IVoxNodePhc() = default;

    // 构造函数,根据 phc_order 预先分配 phc_cubes_ 的大小
    IVoxNodePhc(const PointT& center, const float& side_length, const int& phc_order = 6);

    // 插入 cube 到 phc_cubes_
    void InsertPoint(const PointT& pt);

    // 根据阈值 erase_distance_th_ 从 phc_cubes_ 中释放 cube
    void ErasePoint(const PointT& pt, const double erase_distance_th_);

    inline bool Empty() const;

    inline std::size_t Size() const;

    PointT GetPoint(const std::size_t idx) const;

    bool NNPoint(const PointT& cur_pt, DistPoint& dist_point) const;

    // 寻找距离 cur_pt 最近的 k 个点存入 dis_points，最大距离不超过 max_range，返回的是最近的个数
    int KNNPointByCondition(std::vector<DistPoint>& dis_points, const PointT& cur_pt, const int& K = 5,
                            const double& max_range = 5.0);

   private:
   // 纯虚函数,根据点的坐标计算其哈希索引
    uint32_t CalculatePhcIndex(const PointT& pt) const;

   private:
    std::vector<PhcCube> phc_cubes_;

    PointT center_;
    float side_length_ = 0;
    int phc_order_ = 6;
    float phc_side_length_ = 0;
    float phc_side_length_inv_ = 0;
    Eigen::Matrix<float, dim, 1> min_cube_;
};

template <typename PointT, int dim>
struct IVoxNode<PointT, dim>::DistPoint {
    double dist = 0;
    IVoxNode* node = nullptr;
    int idx = 0;

    DistPoint() = default;
    DistPoint(const double d, IVoxNode* n, const int i) : dist(d), node(n), idx(i) {}

    // 返回节点内和索引对应的点
    PointT Get() { return node->GetPoint(idx); }
    // 友元函数或运算符重载，根据 dist 排序
    inline bool operator()(const DistPoint& p1, const DistPoint& p2) { return p1.dist < p2.dist; }

    inline bool operator<(const DistPoint& rhs) { return dist < rhs.dist; }
};

template <typename PointT, int dim>
void IVoxNode<PointT, dim>::InsertPoint(const PointT& pt) {
    points_.template emplace_back(pt);
}

template <typename PointT, int dim>
bool IVoxNode<PointT, dim>::Empty() const {
    return points_.empty();
}

template <typename PointT, int dim>
std::size_t IVoxNode<PointT, dim>::Size() const {
    return points_.size();
}

template <typename PointT, int dim>
PointT IVoxNode<PointT, dim>::GetPoint(const std::size_t idx) const {
    return points_[idx];
}

template <typename PointT, int dim>
int IVoxNode<PointT, dim>::KNNPointByCondition(std::vector<DistPoint>& dis_points, const PointT& point, const int& K,
                                               const double& max_range) {
    std::size_t old_size = dis_points.size();
// #define INNER_TIMER
#ifdef INNER_TIMER
    static std::unordered_map<std::string, std::vector<int64_t>> stats;
    if (stats.empty()) {
        stats["dis"] = std::vector<int64_t>();
        stats["put"] = std::vector<int64_t>();
        stats["nth"] = std::vector<int64_t>();
    }
#endif

    for (const auto& pt : points_) {
#ifdef INNER_TIMER
        auto t0 = std::chrono::high_resolution_clock::now();
#endif
        double d = distance2(pt, point);
#ifdef INNER_TIMER
        auto t1 = std::chrono::high_resolution_clock::now();
#endif
        if (d < max_range * max_range) {
            dis_points.template emplace_back(DistPoint(d, this, &pt - points_.data()));
        }
#ifdef INNER_TIMER
        auto t2 = std::chrono::high_resolution_clock::now();

        auto dis = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        stats["dis"].emplace_back(dis);
        auto put = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        stats["put"].emplace_back(put);
#endif
    }

#ifdef INNER_TIMER
    auto t1 = std::chrono::high_resolution_clock::now();
#endif
    // sort by distance
    if (old_size + K >= dis_points.size()) {
    } else {
        std::nth_element(dis_points.begin() + old_size, dis_points.begin() + old_size + K - 1, dis_points.end());
        dis_points.resize(old_size + K);
    }

#ifdef INNER_TIMER
    auto t2 = std::chrono::high_resolution_clock::now();
    auto nth = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    stats["nth"].emplace_back(nth);

    constexpr int STAT_PERIOD = 100000;
    if (!stats["nth"].empty() && stats["nth"].size() % STAT_PERIOD == 0) {
        for (auto& it : stats) {
            const std::string& key = it.first;
            std::vector<int64_t>& stat = it.second;
            int64_t sum_ = std::accumulate(stat.begin(), stat.end(), 0);
            int64_t num_ = stat.size();
            stat.clear();
            std::cout << "inner_" << key << "(ns): sum=" << sum_ << " num=" << num_ << " ave=" << 1.0 * sum_ / num_
                      << " ave*n=" << 1.0 * sum_ / STAT_PERIOD << std::endl;
        }
    }
#endif

    return dis_points.size();
}

template <typename PointT, int dim>
struct IVoxNodePhc<PointT, dim>::DistPoint {
    double dist = 0;    // 距离
    IVoxNodePhc* node = nullptr;    // 节点
    int idx = 0;    // 节点内的索引

    DistPoint() {}
    DistPoint(const double d, IVoxNodePhc* n, const int i) : dist(d), node(n), idx(i) {}

    PointT Get() { return node->GetPoint(idx); }

    inline bool operator()(const DistPoint& p1, const DistPoint& p2) { return p1.dist < p2.dist; }

    inline bool operator<(const DistPoint& rhs) { return dist < rhs.dist; }
};

template <typename PointT, int dim>
struct IVoxNodePhc<PointT, dim>::PhcCube {
    uint32_t idx = 0;
    pcl::CentroidPoint<PointT> mean;    // 计算点的均值

    PhcCube(uint32_t index, const PointT& pt) { mean.add(pt); }

    void AddPoint(const PointT& pt) { mean.add(pt); }

    // 返回所有点均值的点
    PointT GetPoint() const {
        PointT pt;
        mean.get(pt);
        return std::move(pt);
    }
};

template <typename PointT, int dim>
IVoxNodePhc<PointT, dim>::IVoxNodePhc(const PointT& center, const float& side_length, const int& phc_order)
    : center_(center), side_length_(side_length), phc_order_(phc_order) {
    assert(phc_order <= 8);
    phc_side_length_ = side_length_ / (std::pow(2, phc_order_));    // 2 的 phc_order_ 次幂
    phc_side_length_inv_ = (std::pow(2, phc_order_)) / side_length_;
    min_cube_ = center_.getArray3fMap() - side_length / 2.0;
    phc_cubes_.reserve(64); // 初始化 64 个值
}

template <typename PointT, int dim>
void IVoxNodePhc<PointT, dim>::InsertPoint(const PointT& pt) {
    uint32_t idx = CalculatePhcIndex(pt);   // 计算哈希索引

    PhcCube cube{idx, pt};
    // 在已排序的范围内找到第一个不小于给定 cube 索引的元素
    auto it = std::lower_bound(phc_cubes_.begin(), phc_cubes_.end(), cube,
                               [](const PhcCube& a, const PhcCube& b) { return a.idx < b.idx; });

    /* 如果没有找到就将 cube 加入 phc_cubes_
       找到了，如果索引相等就计算均值
              不相等就插入 */
    if (it == phc_cubes_.end()) {
        phc_cubes_.emplace_back(cube);
    } else {
        if (it->idx == idx) {
            it->AddPoint(pt);
        } else {
            phc_cubes_.insert(it, cube);
        }
    }
}

template <typename PointT, int dim>
void IVoxNodePhc<PointT, dim>::ErasePoint(const PointT& pt, const double erase_distance_th_) {
    uint32_t idx = CalculatePhcIndex(pt);   // 计算哈希索引

    PhcCube cube{idx, pt};
    auto it = std::lower_bound(phc_cubes_.begin(), phc_cubes_.end(), cube,
                               [](const PhcCube& a, const PhcCube& b) { return a.idx < b.idx; });

    // erase_distance_th_ 阈值似乎没有作用
    if (erase_distance_th_ > 0) {
    }
    // 如果索引一样，就释放
    if (it != phc_cubes_.end() && it->idx == idx) {
        phc_cubes_.erase(it);
    }
}

// 判断 phc_cubes_ 是否为空
template <typename PointT, int dim>
bool IVoxNodePhc<PointT, dim>::Empty() const {
    return phc_cubes_.empty();
}

// 返回 phc_cubes_ 的 size
template <typename PointT, int dim>
std::size_t IVoxNodePhc<PointT, dim>::Size() const {
    return phc_cubes_.size();
}

// 返回索引所在 cube 内均值的点
template <typename PointT, int dim>
PointT IVoxNodePhc<PointT, dim>::GetPoint(const std::size_t idx) const {
    return phc_cubes_[idx].GetPoint();
}

// 查找当前点 cur_pt 的最近邻，并存储在 DistPoint，返回是否成功
template <typename PointT, int dim>
bool IVoxNodePhc<PointT, dim>::NNPoint(const PointT& cur_pt, DistPoint& dist_point) const {
    if (phc_cubes_.empty()) {
        return false;
    }
    uint32_t cur_idx = CalculatePhcIndex(cur_pt);
    PhcCube cube{cur_idx, cur_pt};
    auto it = std::lower_bound(phc_cubes_.begin(), phc_cubes_.end(), cube,
                               [](const PhcCube& a, const PhcCube& b) { return a.idx < b.idx; });

    if (it == phc_cubes_.end()) {
        it--;
        dist_point = DistPoint(distance2(cur_pt, it->GetPoint()), this, it - phc_cubes_.begin());
    } else if (it == phc_cubes_.begin()) {
        dist_point = DistPoint(distance2(cur_pt, it->GetPoint()), this, it - phc_cubes_.begin());
    } else {
        auto last_it = it;
        last_it--;
        double d1 = distance2(cur_pt, it->GetPoint());
        double d2 = distance2(cur_pt, last_it->GetPoint());
        if (d1 > d2) {
            dist_point = DistPoint(d2, this, it - phc_cubes_.begin());
        } else {
            dist_point = DistPoint(d1, this, it - phc_cubes_.begin());
        }
    }

    return true;
}

// 查找 k 个最近邻
template <typename PointT, int dim>
int IVoxNodePhc<PointT, dim>::KNNPointByCondition(std::vector<DistPoint>& dis_points, const PointT& cur_pt,
                                                  const int& K, const double& max_range) {
    uint32_t cur_idx = CalculatePhcIndex(cur_pt);
    PhcCube cube{cur_idx, cur_pt};
    auto it = std::lower_bound(phc_cubes_.begin(), phc_cubes_.end(), cube,
                               [](const PhcCube& a, const PhcCube& b) { return a.idx < b.idx; });

    const int max_search_cube_side_length = std::pow(2, std::ceil(std::log2(max_range * phc_side_length_inv_)));
    const int max_search_idx_th =
        8 * max_search_cube_side_length * max_search_cube_side_length * max_search_cube_side_length;

    auto create_dist_point = [&cur_pt, this](typename std::vector<PhcCube>::const_iterator forward_it) {
        double d = distance2(forward_it->GetPoint(), cur_pt);
        return DistPoint(d, this, forward_it - phc_cubes_.begin());
    };

    typename std::vector<PhcCube>::const_iterator forward_it(it);
    typename std::vector<PhcCube>::const_reverse_iterator backward_it(it);
    if (it != phc_cubes_.end()) {
        dis_points.emplace_back(create_dist_point(it));
        forward_it++;
    }
    if (backward_it != phc_cubes_.rend()) {
        backward_it++;
    }

    auto forward_reach_boundary = [&]() {
        return forward_it == phc_cubes_.end() || forward_it->idx - cur_idx > max_search_idx_th;
    };
    auto backward_reach_boundary = [&]() {
        return backward_it == phc_cubes_.rend() || cur_idx - backward_it->idx > max_search_idx_th;
    };

    while (!forward_reach_boundary() && !backward_reach_boundary()) {
        if (forward_it->idx - cur_idx > cur_idx - backward_it->idx) {
            dis_points.emplace_back(create_dist_point(forward_it));
            forward_it++;
        } else {
            dis_points.emplace_back(create_dist_point(backward_it.base()));
            backward_it++;
        }
        if (dis_points.size() > K) {
            break;
        }
    }

    if (forward_reach_boundary()) {
        while (!backward_reach_boundary() && dis_points.size() < K) {
            dis_points.emplace_back(create_dist_point(backward_it.base()));
            backward_it++;
        }
    }

    if (backward_reach_boundary()) {
        while (!forward_reach_boundary() && dis_points.size() < K) {
            dis_points.emplace_back(create_dist_point(forward_it));
            forward_it++;
        }
    }

    return dis_points.size();
}

// 计算给定点的 PHC 索引
template <typename PointT, int dim>
uint32_t IVoxNodePhc<PointT, dim>::CalculatePhcIndex(const PointT& pt) const {
    Eigen::Matrix<float, dim, 1> eposf = (pt.getVector3fMap() - min_cube_) * phc_side_length_inv_;
    Eigen::Matrix<int, dim, 1> eposi = eposf.template cast<int>();
    for (int i = 0; i < dim; ++i) {
        if (eposi(i, 0) < 0) {
            eposi(i, 0) = 0;
        }
        if (eposi(i, 0) > std::pow(2, phc_order_)) {
            eposi(i, 0) = std::pow(2, phc_order_) - 1;
        }
    }
    std::array<uint8_t, 3> apos{eposi(0), eposi(1), eposi(2)};
    std::array<uint8_t, 3> tmp = hilbert::v2::PositionToIndex(apos);

    uint32_t idx = (uint32_t(tmp[0]) << 16) + (uint32_t(tmp[1]) << 8) + (uint32_t(tmp[2]));
    return idx;
}

}  // namespace faster_lio
