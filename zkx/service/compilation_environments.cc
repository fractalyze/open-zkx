/* Copyright 2022 The OpenXLA Authors.
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

#include "zkx/service/compilation_environments.h"

#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/debugging/leak_check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace zkx {
namespace {

ABSL_CONST_INIT absl::Mutex g_process_new_env_fns_mu(absl::kConstInit);
absl::flat_hash_map<const google::protobuf::Descriptor*,
                    CompilationEnvironments::ProcessNewEnvFn>*
    g_process_new_env_fns ABSL_GUARDED_BY(g_process_new_env_fns_mu) = nullptr;

// A global singleton stats object for implementing CompilationEnvironments::{
// DefaultEnvCreatedByCompilationEnvironments(), EnvAdded()}.
class GlobalCompEnvStats {
 public:
  static GlobalCompEnvStats& GetSingleton() {
    static GlobalCompEnvStats* singleton =
        absl::IgnoreLeak(new GlobalCompEnvStats());

    return *singleton;
  }

  void DefaultEnvCreatedByCompilationEnvironments(std::string_view env_type)
      ABSL_LOCKS_EXCLUDED(mu_) {
    {
      absl::MutexLock l(&mu_);
      ++stats_[std::string(env_type)]
            .default_env_created_by_compilation_environments;
    }
    VLOG(1) << "New GlobalCompEnvStats value: " << ToString();
  }

  void EnvAdded(std::string_view env_type) ABSL_LOCKS_EXCLUDED(mu_) {
    {
      absl::MutexLock l(&mu_);
      ++stats_[std::string(env_type)].env_added;
    }
    VLOG(1) << "New GlobalCompEnvStats value: " << ToString();
  }

  std::string ToString() const ABSL_LOCKS_EXCLUDED(mu_) {
    absl::ReaderMutexLock l(&mu_);
    return absl::StrJoin(
        stats_, "; ",
        [](std::string* out, const StatMap::value_type& env_stats_pair) {
          absl::StrAppend(out, env_stats_pair.first, ": { ",
                          env_stats_pair.second.ToString(), " }");
        });
  }

 private:
  struct PerEnvStats {
    std::string ToString() const {
      return absl::StrCat(
          "# default envs created by CompilationEnvironments: ",
          default_env_created_by_compilation_environments, " ",
          "# envs added to CompilationEnvironments: ", env_added);
    }

    unsigned default_env_created_by_compilation_environments = 0;
    unsigned env_added = 0;
  };

  using StatMap = absl::flat_hash_map<std::string, PerEnvStats>;

  GlobalCompEnvStats() = default;
  GlobalCompEnvStats(const GlobalCompEnvStats&) = delete;
  GlobalCompEnvStats& operator=(const GlobalCompEnvStats&) = delete;
  GlobalCompEnvStats(GlobalCompEnvStats&&) = delete;
  GlobalCompEnvStats& operator=(GlobalCompEnvStats&&) = delete;

  mutable absl::Mutex mu_;
  StatMap stats_ ABSL_GUARDED_BY(mu_);
};

}  // namespace

CompilationEnvironments& CompilationEnvironments::operator=(
    const CompilationEnvironments& rhs) {
  Clear();
  for (const auto& descriptor_message_pair : rhs.environments_) {
    auto env = absl::WrapUnique(descriptor_message_pair.second->New());
    env->CopyFrom(*descriptor_message_pair.second);
    environments_.insert({descriptor_message_pair.first, std::move(env)});
  }
  return *this;
}

// static
absl::StatusOr<std::unique_ptr<CompilationEnvironments>>
CompilationEnvironments::CreateFromProto(
    const CompilationEnvironmentsProto& proto) {
  auto envs = std::make_unique<CompilationEnvironments>();

  const google::protobuf::DescriptorPool* const pool =
      google::protobuf::DescriptorPool::generated_pool();

  for (const auto& env_proto : proto.environments()) {
    std::string fullname;
    if (!google::protobuf::Any::ParseAnyTypeUrl(env_proto.type_url(),
                                                &fullname)) {
      return absl::DataLossError(
          absl::StrFormat("Invalid CompilationEnvironment message type url: %s",
                          env_proto.type_url()));
    }

    const google::protobuf::Descriptor* const descriptor =
        pool->FindMessageTypeByName(fullname);
    if (descriptor == nullptr) {
      return absl::DataLossError(absl::StrFormat(
          "Unknown CompilationEnvironment message type: %s", fullname));
    }

    const google::protobuf::Message* const prototype =
        google::protobuf::MessageFactory::generated_factory()->GetPrototype(
            descriptor);
    if (prototype == nullptr) {
      return absl::InternalError(absl::StrFormat(
          "Unsupported CompilationEnvironment message type: %s", fullname));
    }

    std::unique_ptr<google::protobuf::Message> env(prototype->New());
    if (!env_proto.UnpackTo(env.get())) {
      return absl::DataLossError(absl::StrFormat(
          "Unable to unpack CompilationEnvironment message of type '%s'",
          fullname));
    }

    TF_RETURN_IF_ERROR(envs->AddEnv(std::move(env)));
  }

  return envs;
}

// static
void CompilationEnvironments::RegisterProcessNewEnvFn(
    const google::protobuf::Descriptor* descriptor,
    ProcessNewEnvFn process_new_env) {
  absl::MutexLock l(&g_process_new_env_fns_mu);
  if (g_process_new_env_fns == nullptr) {
    g_process_new_env_fns =
        new absl::flat_hash_map<const google::protobuf::Descriptor*,
                                CompilationEnvironments::ProcessNewEnvFn>();
  }
  const bool inserted =
      g_process_new_env_fns->insert({descriptor, std::move(process_new_env)})
          .second;
  CHECK(inserted) << "ProcessNewEnvFn for ZKX compilation environment '"
                  << descriptor->full_name() << "' has already been registered";
}

absl::Status CompilationEnvironments::AddEnv(
    std::unique_ptr<google::protobuf::Message> env) {
  if (!env) {
    return absl::InvalidArgumentError(
        "Can not add a null compilation environment.");
  }
  const google::protobuf::Descriptor& descriptor = *env->GetDescriptor();
  return AddEnvImpl(descriptor, std::move(env));
}

CompilationEnvironmentsProto CompilationEnvironments::ToProto() const {
  // Sort the environments by their message types' full names so that the
  // proto fields are deterministically ordered.
  std::vector<const google::protobuf::Descriptor*> descriptors;
  descriptors.reserve(environments_.size());
  for (const auto& [descriptor, message] : environments_) {
    descriptors.push_back(descriptor);
  }
  absl::c_sort(descriptors, [](const google::protobuf::Descriptor* lhs,
                               const google::protobuf::Descriptor* rhs) {
    return lhs->full_name() < rhs->full_name();
  });

  CompilationEnvironmentsProto proto;
  for (const auto* const descriptor : descriptors) {
    proto.add_environments()->PackFrom(*environments_.at(descriptor));
  }
  return proto;
}

// static
CompilationEnvironments::ProcessNewEnvFn
CompilationEnvironments::GetProcessNewEnvFn(
    const google::protobuf::Descriptor& descriptor) {
  absl::MutexLock l(&g_process_new_env_fns_mu);
  if (g_process_new_env_fns == nullptr) {
    return nullptr;
  }
  const auto it = g_process_new_env_fns->find(&descriptor);
  if (it == g_process_new_env_fns->end()) {
    return nullptr;
  }
  return it->second;
}

// static
void CompilationEnvironments::DefaultEnvCreatedByCompilationEnvironments(
    std::string_view env_type) {
  GlobalCompEnvStats::GetSingleton().DefaultEnvCreatedByCompilationEnvironments(
      env_type);
}

// static
void CompilationEnvironments::EnvAdded(std::string_view env_type) {
  GlobalCompEnvStats::GetSingleton().EnvAdded(env_type);
}

absl::Status CompilationEnvironments::AddEnvImpl(
    const google::protobuf::Descriptor& descriptor,
    std::unique_ptr<google::protobuf::Message> env) {
  // Check if we already have an environment of env's type
  if (environments_.contains(&descriptor)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Replacing CompilationEnvironment of type %s.",
                        descriptor.full_name()));
  }

  // Process env
  ProcessNewEnvFn process_new_env = GetProcessNewEnvFn(descriptor);
  if (!process_new_env) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Unknown compilation environment type: %s", descriptor.full_name()));
  }
  TF_ASSIGN_OR_RETURN(std::unique_ptr<google::protobuf::Message> processed_env,
                      process_new_env(std::move(env)));

  // Check for unknown fields
  const google::protobuf::UnknownFieldSet& unknown_fields =
      processed_env->GetReflection()->GetUnknownFields(*processed_env);
  std::vector<int> unknown_tags;
  unknown_tags.reserve(unknown_fields.field_count());
  for (int i = 0; i < unknown_fields.field_count(); ++i) {
    const google::protobuf::UnknownField& field = unknown_fields.field(i);
    unknown_tags.push_back(field.number());
  }
  if (!unknown_tags.empty()) {
    LOG(WARNING) << "CompilationEnvironment " << descriptor.full_name()
                 << " contains unknown fields with tag numbers: "
                 << absl::StrJoin(unknown_tags, ", ");
  }

  // Actually add the env
  environments_.insert({&descriptor, std::move(processed_env)});
  EnvAdded(descriptor.full_name());
  return absl::OkStatus();
}

}  // namespace zkx
