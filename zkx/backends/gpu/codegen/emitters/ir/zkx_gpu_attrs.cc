/* Copyright 2024 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <optional>

#include "zkx/backends/gpu/codegen/emitters/ir/zkx_gpu_ops.h"
#include "zkx/hlo/analysis/indexing_map.h"
#include "zkx/hlo/analysis/indexing_map_serialization.h"

namespace zkx::gpu {

mlir::Attribute LayoutAttr::parse(mlir::AsmParser& parser, mlir::Type) {
  mlir::StringAttr memory_space_str;
  if (failed(parser.parseLess()) ||
      failed(parser.parseAttribute(memory_space_str)) ||
      failed(parser.parseComma())) {
    return {};
  }
  std::optional<MemorySpace> memspace =
      symbolizeMemorySpace(memory_space_str.getValue());
  if (!memspace.has_value()) {
    return {};
  }
  std::optional<IndexingMap> indexing_map =
      parseChainOfStringsAsIndexingMap(parser);
  if (!indexing_map.has_value() || failed(parser.parseGreater())) {
    return {};
  }
  auto* context = parser.getContext();
  context->getOrLoadDialect<ZkxDialect>();
  return LayoutAttr::get(context, MemorySpaceAttr::get(context, *memspace),
                         IndexingMapAttr::get(context, *indexing_map));
}

void LayoutAttr::print(mlir::AsmPrinter& printer) const {
  printer << "<\"" << stringifyMemorySpace(getMemorySpace().getValue())
          << "\", \"" << ToString(getThreadMap().getIndexingMap()) << "\">";
}

}  // namespace zkx::gpu
