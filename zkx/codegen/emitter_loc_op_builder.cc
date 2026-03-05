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

#include "zkx/codegen/emitter_loc_op_builder.h"

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"

namespace zkx {

mlir::Location EmitterLocOpBuilder::Loc(
    EmitterLocOpBuilder::SourceLocation location) const {
  if (!annotate_loc_ || location.line() == 0) {
    return current_loc_;
  }
  std::vector<std::string> file_name =
      absl::StrSplit(location.file_name(), '/');
  std::string previous_loc;
  if (mlir::isa<mlir::NameLoc>(current_loc_)) {
    auto name_loc = mlir::cast<mlir::NameLoc>(current_loc_);
    previous_loc = name_loc.getName().str();
  }

  const std::string text = absl::StrCat(previous_loc, " -> ", file_name.back(),
                                        ":", location.line());
  return mlir::NameLoc::get(mlir::StringAttr::get(getContext(), text));
}

}  // namespace zkx
