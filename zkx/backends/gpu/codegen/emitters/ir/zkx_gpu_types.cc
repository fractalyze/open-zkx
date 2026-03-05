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

#include <stdint.h>

#include "mlir/Support/LLVM.h"

#include "zkx/backends/gpu/codegen/emitters/ir/zkx_gpu_ops.h"

namespace zkx::gpu {

mlir::Type IndexedVectorType::parse(mlir::AsmParser& parser) {
  mlir::SmallVector<int64_t, 4> shape;
  mlir::Type type;
  IndexingMapAttr indexing_map_attr;
  if (failed(parser.parseLess()) ||
      failed(parser.parseDimensionList(shape, /*allowDynamic=*/false)) ||
      failed(parser.parseType(type)) || failed(parser.parseComma()) ||
      failed(parser.parseAttribute(indexing_map_attr)) ||
      failed(parser.parseGreater())) {
    return {};
  }
  return IndexedVectorType::get(parser.getContext(), shape, type,
                                indexing_map_attr);
}

void IndexedVectorType::print(mlir::AsmPrinter& printer) const {
  printer << "<";
  printer.printDimensionList(getShape());
  printer << "x" << getElementType() << ", " << getIndexingMapAttr() << ">";
}

}  // namespace zkx::gpu
