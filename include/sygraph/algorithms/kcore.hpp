/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "sygraph/operators/config.hpp"
#include <sycl/sycl.hpp>

#include <sygraph/frontier/frontier.hpp>
#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/advance/advance.hpp>
#include <sygraph/operators/for/for.hpp>
#ifdef ENABLE_PROFILING
#include <sygraph/utils/profiler.hpp>
#endif
#include <memory>
#include <set>

/**
 * @namespace sygraph
 * @brief Namespace for the SYgraph library.
 *
 * The sygraph namespace contains classes and functions for graph algorithms and data structures.
 */
namespace sygraph{
namespace algorithms {

enum class kcore_direction { push, pull };

struct KCORERunDetails {
  size_t iterations = 0;
  int max_core = 0;
};

namespace detail {
/**
* @brief Represents an instance of the K-Core Decomposition algorithm on a graph.
*
* The KCoreInstance struct encapsulates the necessary and operations for performing the K-Core algorithm on graph.
* It stores the graph and arrays for degrees, flags, and core numbers.
*/
template<typename GraphType>
struct KCOREInstance {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;

  GraphType &G; // The graph on which the K-Core algorithm will be performed.
  edge_t *degree; // Array to store the current degree of each vertex.
  int *removed; // Array to store whether a vertex has been removed (0 = active, 1 = removed). */
  edge_t *core; // Array to store the core number of each vertex. */

  /**
   * @brief Constructs a KCOREInstance object.
   * 
   * @param G The graph on which the K-Core algorithm will be performed.
   */
  KCOREInstance(GraphType &G) : G(G) {
    sycl::queue &queue = G.getQueue();
    size_t size = G.getVertexCount();

    // Initialize degree
    degree = memory::detail::memoryAlloc<edge_t, memory::space::shared>(size, queue);
    queue.fill(degree, static_cast<edge_t>(0), size).wait();

    // Initialize removed array
    removed = memory::detail::memoryAlloc<int, memory::space::shared>(size, queue);
    queue.fill(removed, 0, size).wait();

    // Initialize core number
    core = memory::detail::memoryAlloc<edge_t, memory::space::shared>(size, queue);
    queue.fill(core, static_cast<edge_t>(0), size).wait();
  }

  /**
   * @brief Destroys the KCOREInstance object and frees the allocated memory.
   */
  ~KCOREInstance() {
    memory::detail::releaseUSM(degree, G.getQueue());
    memory::detail::releaseUSM(removed, G.getQueue());
    memory::detail::releaseUSM(core, G.getQueue());
  }
};
} // namespace detail

/**
 * @brief Represents the K-Core Decomposition algorithm.
 * 
 * The K-Core algorithm iteratively peels the graph layer by layer, assigning a core number to each vertex.
 * 
 * @tparam GraphType The type of the graph on which the algorithm will be performed.
 */
template <typename GraphType>
class KCORE {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;

public:
  /**
   * @brief Constructs a KCORE object.
   */
  KCORE(GraphType &g) : _g(g) {};

  /**
   * @brief Initializes the KCORE algorithm with the given graph and source vertex.
   * 
   * @param G The graph on which the KCORE algorithm will be performed.
   */
  void init() { _instance = std::make_unique<detail::KCOREInstance<GraphType>>(_g); }

  /**
   * @brief Resets the KCORE algorithm.
   */
  void reset() { _instance.reset(); }

  /**
   * @brief Runs the KCORE algorithm.
   * 
   * @param direction The direction of the KCORE traversal (push, pull, or hybrid).
   * @tparam EnableProfiling A boolean flag to enable profiling.
   * @throws std::runtime_error if the KCORE instance is not initialized.
   */
  KCORERunDetails run(kcore_direction direction = kcore_direction::push) {
    KCORERunDetails details;
    if (!_instance) { throw std::runtime_error("KCORE instance not initialized"); }

    auto &G = _instance->G;
    auto &degree = _instance->degree;
    auto &removed = _instance->removed;
    auto &core = _instance->core;

    sycl::queue &queue = G.getQueue();
    
    using load_balance_t = sygraph::operators::load_balancer;
    using direction_t = sygraph::operators::direction;
    using frontier_view_t = sygraph::frontier::frontier_view;
    using frontier_impl_t = sygraph::frontier::frontier_type;

    auto in_frontier = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_impl_t::mlb>(queue, G);
    auto out_frontier = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_impl_t::mlb>(queue, G);

    size_t n = G.getVertexCount();
    auto g_device = G.getDeviceGraph();
    int iter = 0;
    int k = 1;

    // Compute initial degree of each vertex on GPU
    auto e_init = queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        degree[i[0]] = g_device.getDegree(i[0]);
      });
    });
    e_init.wait();
#ifdef ENABLE_PROFILING
sygraph::Profiler::addEvent(e_init, "e_init_degree");
#endif

    // Compute max degree of the graph with SYCL reduction for max number
    edge_t max_degree = 0;
    sycl::buffer<edge_t, 1> max_buf(&max_degree, sycl::range<1>(1));

    auto e_max_deg = queue.submit([&](sycl::handler& cgh) {
      auto max_reduction = sycl::reduction(max_buf, cgh, sycl::maximum<edge_t>());
      cgh.parallel_for(sycl::range<1>(n), max_reduction, [=](sycl::id<1> i, auto& max){
        max.combine(degree[i[0]]);
      });
    });
    e_max_deg.wait();
#ifdef ENABLE_PROFILING
sygraph::Profiler::addEvent(e_max_deg, "e_max_deg");
#endif

    max_degree = max_buf.get_host_access()[0];

    sygraph::Event e;

    // compute operator for mark vertex as removed, set core number of v
    auto compute_step = [&]() {
      return sygraph::operators::compute::execute<frontier_view_t::vertex>(
        G,
        in_frontier,
        [=](auto v){
          removed[v] = 1;
          core[v] = k;
        });
    };

    auto push_step = [&]() {
      return sygraph::operators::advance::frontier<load_balance_t::workgroup_mapped, frontier_view_t::vertex, frontier_view_t::vertex>(
        G,
        in_frontier,
        out_frontier,
        [=](auto src, auto dst, auto edge, auto weight) -> bool {
          if(removed[dst]){
            return false;
          }
          sycl::atomic_ref<edge_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(degree[dst]);
          auto old_deg = ref.fetch_sub(1);
          if((old_deg - 1) <= k){
            return true;
          }
          else{
            return false;
          }
        },
        sygraph::frontier::size::fetch_from_memory);
    };

    auto pull_step = [&]() {
      return sygraph::operators::advance::
              frontier<direction_t::pull, load_balance_t::workgroup_mapped, frontier_view_t::vertex, frontier_view_t::vertex>(
        G,
        in_frontier,
        out_frontier,
        [=](auto src, auto dst, auto edge, auto weight) -> bool {
          //TODO
          return false;
        },
        sygraph::frontier::size::fetch_from_memory);
    };

    bool push = direction != kcore_direction::pull;

    while (k <= max_degree) {
      auto device_frontier = in_frontier.getDeviceFrontier();
      auto e_insert = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i){
          if (!removed[i[0]] && degree[i[0]] <= k) {
            device_frontier.insert(i[0]);
          }
        });
      });
      e_insert.wait();
#ifdef ENABLE_PROFILING
sygraph::Profiler::addEvent(e_insert, "e_insert");
#endif

      while (!in_frontier.empty()) {

        e = compute_step();
        e.waitAndThrow();
#ifdef ENABLE_PROFILING
sygraph::Profiler::addEvent(e, "compute");
#endif
        if(push) {
          e = push_step();
        }
        else {
          e = pull_step();
        }
        e.waitAndThrow();
#ifdef ENABLE_PROFILING
sygraph::Profiler::addEvent(e, "advance");
#endif
        sygraph::frontier::swap(in_frontier, out_frontier);
        out_frontier.clear();
        iter++;
      }

      edge_t next_k = max_degree + 1;
      sycl::buffer<edge_t, 1> min_buf(&next_k, sycl::range<1>(1));
      auto e_min = queue.submit([&](sycl::handler& cgh) {
        auto min_reduction = sycl::reduction(min_buf, cgh, sycl::minimum<edge_t>());
        cgh.parallel_for(sycl::range<1>(n), min_reduction, [=](sycl::id<1> i, auto& min){
          if(!removed[i[0]]){
            min.combine(degree[i[0]]);
          }
        });
      });
      e_min.wait();
#ifdef ENABLE_PROFILING
sygraph::Profiler::addEvent(e_min, "e_min");
#endif
      k = static_cast<int>(min_buf.get_host_access()[0]);
    }
    details.iterations = iter;
    
    edge_t max_core_val = 0;
    sycl::buffer<edge_t, 1> max_core_buf(&max_core_val, sycl::range<1>(1));

    auto e_maxcore = queue.submit([&](sycl::handler& cgh) {
      auto max_reduction = sycl::reduction(max_core_buf, cgh, sycl::maximum<edge_t>());
      cgh.parallel_for(sycl::range<1>(n), max_reduction, [=](sycl::id<1> i, auto& max){
        max.combine(core[i[0]]);
      });
    });
    e_maxcore.wait();
#ifdef ENABLE_PROFILING
sygraph::Profiler::addEvent(e_maxcore, "e_maxcore");
#endif

    details.max_core = static_cast<int>(max_core_buf.get_host_access()[0]);
    return details;
  }

  /**
   * @brief Returns the the core number of a vertex.
   * 
   * @param vertex The vertex for which to get the core number.
   * @return A core number.
   */
  edge_t getCoreNumber(size_t vertex) const {
    return _instance->core[vertex];
  }
  
  /**
   * @brief Returns the core numbers of all vertices.
   * 
   * @return A vector of core numbers.
   */
  std::vector<edge_t> getCoreNumbers() const {
    std::vector<edge_t> cores(_instance->G.getVertexCount());
    sycl::queue& queue = _instance->G.getQueue();
    auto e_copy = queue.copy(_instance->core, cores.data(), cores.size());
    e_copy.wait();
#ifdef ENABLE_PROFILING
sygraph::Profiler::addEvent(e_copy, "e_copy");
#endif
    return cores;
  }

private:
    GraphType &_g;
    std::unique_ptr<detail::KCOREInstance<GraphType>> _instance;
};
} // namespace algorithms
} // namespace sygraph