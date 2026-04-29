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

  printProfilingOutput(opts);
  std::cout << "Total Host Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_timer - start_timer).count() << " ms" << std::endl;
  return 0;
}
