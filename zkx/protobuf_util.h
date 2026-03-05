/* Copyright 2017 The OpenXLA Authors.
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

#ifndef ZKX_PROTOBUF_UTIL_H_
#define ZKX_PROTOBUF_UTIL_H_

#include <stddef.h>

#include "google/protobuf/message.h"

namespace zkx::protobuf_util {

// Returns true if `m1` is equal to `m2`.
//
// WARNING: We use protocol buffer serialization and then check for
// equality of the serialized representation, which may miss some
// cases of equality.  However, for the purposes of the ZKX code
// base, this form of equality checking is sufficient.
bool ProtobufEquals(const google::protobuf::Message& m1,
                    const google::protobuf::Message& m2);

// Return the hash of the message `m`.
//
// WARNING: This uses the same serialization approach used by ProtobufEquals,
// so the WARNING for that function applies here.
size_t ProtobufHash(const google::protobuf::Message& m);

// Wrappers for above methods so that they can be used in containers.
class ProtobufEqualsWrapper {
 public:
  bool operator()(const google::protobuf::Message& m1,
                  const google::protobuf::Message& m2) const {
    return ProtobufEquals(m1, m2);
  }
};

class ProtobufHashWrapper {
 public:
  size_t operator()(const google::protobuf::Message& m) const {
    return ProtobufHash(m);
  }
};

}  // namespace zkx::protobuf_util

#endif  // ZKX_PROTOBUF_UTIL_H_
