/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../include/utils.hpp"
#include <CLI/CLI.hpp>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

template<typename GraphT, typename KCoreT>
bool validate(const GraphT& graph, KCoreT& kcore){
  using vertex_t = typename GraphT::vertex_t;
  using edge_t = typename GraphT::edge_t;
  std::vector<vertex_t> in_frontier;
  std::vector<vertex_t> out_frontier;

  auto* row_offsets = graph.getRowOffsets();
  auto* col_indices = graph.getColumnIndices();

  auto host_core_numbers = kcore.getCoreNumbers();

  size_t n = graph.getVertexCount();
  size_t iter = 0;
  size_t mismatches = 0;
  int k = 1;
  edge_t max_degree = 0;

  std::vector<edge_t> degree(n);
  std::vector<edge_t> cpu_core(n, 0);
  std::vector<int> removed (n, 0);

  for (size_t v = 0; v < n; v++) {
    degree[v] = row_offsets[v + 1] - row_offsets[v];
    max_degree = std::max(max_degree, degree[v]);
  }

  while (k <= static_cast<int>(max_degree)){

    for (size_t v = 0; v < n; v++) {
      if(!removed[v] && degree[v] <= k){
        in_frontier.push_back(v);
      }
    }

    while (in_frontier.size()) {

      for (size_t v = 0; v < in_frontier.size(); v++) {
        auto vertex = in_frontier[v];

        if(removed[vertex])
          continue;

        removed[vertex] = 1;
        cpu_core[vertex] = k;

        auto start = row_offsets[vertex];
        auto end = row_offsets[vertex + 1];

        for (size_t j = start; j < end; j++) {
          auto neighbor = col_indices[j];
          if(!removed[neighbor]){
            degree[neighbor] --;
            if (degree[neighbor] <= k){
              out_frontier.push_back(neighbor);
            }
          }
        }
      }

      std::swap(in_frontier, out_frontier);
      out_frontier.clear();
      iter++;
    }
    k++;
  }

  for(size_t v = 0; v < n; v++){
    if(cpu_core[v] != host_core_numbers[v]){
      mismatches++;
    }
  }

  if (mismatches) { std::cerr << "Mismatches: " << mismatches << std::endl; }
  return mismatches == 0;
}

std::string directionToString(sygraph::algorithms::kcore_direction direction) {
  switch (direction) {
    case sygraph::algorithms::kcore_direction::push: return "push";
    case sygraph::algorithms::kcore_direction::pull: return "pull";
    default: return "push";
  }
}

sygraph::algorithms::kcore_direction parseAdvanceDirection(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (value == "pull") { return sygraph::algorithms::kcore_direction::pull; }
  return sygraph::algorithms::kcore_direction::push;
}

int main(int argc, char** argv) {
  using type_t = unsigned int;
  GraphOptions opts;
  CLI::App app{"SYgraph example"};
  auto* source_option = configureBaseCLI(app, opts);
  std::string advance_mode = "push";

  app.add_option("--advance", advance_mode, "Select KCORE advance strategy (push|pull)")
      ->check(CLI::IsMember({"push", "pull"}, CLI::ignore_case));
  CLI11_PARSE(app, argc, argv);
  finalizeGraphOptions(opts, source_option);
  auto advance_direction = parseAdvanceDirection(advance_mode);

  std::cerr << "[*] Reading CSR" << std::endl;
  sygraph::graph::Properties properties;
  auto csr = readCSR<float, type_t, type_t>(opts, &properties);

#ifdef ENABLE_PROFILING
  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling()};
#else
  sycl::queue q{sycl::gpu_selector_v};
#endif

  printDeviceInfo(q, "[*] ");

  std::cerr << "[*] Building Graph" << std::endl;
  auto G = sygraph::graph::build::fromCSR<graph_location>(q, csr, properties);
  printGraphInfo(G);
  size_t size = G.getVertexCount();

  sygraph::algorithms::KCORE kcore{G};
  kcore.init();

  std::cout << "[*] Running KCORE (" << directionToString(advance_direction) << " advance)" << std::endl;
  auto start_timer = std::chrono::high_resolution_clock::now();
  auto details = kcore.run(advance_direction);
  auto end_timer = std::chrono::high_resolution_clock::now();

  std::cerr << "[!] Done" << std::endl;

  std::cerr << "Iterations: " << details.iterations << std::endl;
  std::cerr << "Max core: " << details.max_core << std::endl;

  if (opts.validate) {
    std::cout << "Validation: [";
    auto validation_start = std::chrono::high_resolution_clock::now();
    if (!validate(G, kcore)) {
      std::cout << failString();
    } else {
      std::cout << successString();
    }
    std::cout << "] | ";
    auto validation_end = std::chrono::high_resolution_clock::now();
    std::cout << "Validation Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(validation_end - validation_start).count() << " ms"
              << std::endl;
  }

  printProfilingOutput(opts);
  std::cout << "Total Host Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_timer - start_timer).count() << " ms" << std::endl;
  return 0;
}
