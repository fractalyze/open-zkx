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

#include "zkx/debug_options_flags.h"

#include "absl/base/call_once.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/debugging/leak_check.h"
#include "absl/log/check.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_join.h"

#include "zkx/debug_options_parsers.h"
#include "zkx/parse_flags_from_env.h"
#include "zkx/stream_executor/cuda/nvjitlink_support.h"
#include "zkx/stream_executor/cuda/ptx_compiler_support.h"

namespace zkx {

DebugOptions DefaultDebugOptionsIgnoringFlags() {
  DebugOptions opts;
  opts.set_zkx_llvm_enable_invariant_load_metadata(true);
  opts.set_zkx_llvm_disable_expensive_passes(false);
  opts.set_zkx_backend_optimization_level(3);
  opts.set_zkx_gpu_cuda_data_dir("./cuda_sdk_lib");
  opts.set_zkx_gpu_generate_debug_info(false);
  opts.set_zkx_gpu_generate_line_info(false);

  opts.set_zkx_dump_max_hlo_modules(-1);
  opts.set_zkx_dump_module_metadata(false);
  opts.set_zkx_dump_large_constants(false);
  opts.set_zkx_dump_enable_mlir_pretty_form(true);
  opts.set_zkx_annotate_with_emitter_loc(false);
  opts.set_zkx_debug_buffer_assignment_show_max(15);

  opts.set_zkx_gpu_unsafe_fallback_to_driver_on_ptxas_not_found(false);
  opts.set_zkx_multiheap_size_constraint_per_heap(-1);
  opts.set_zkx_enable_dumping(true);

  opts.set_zkx_gpu_temp_buffer_use_separate_color(false);

  opts.set_zkx_gpu_enable_highest_priority_async_stream(true);
  opts.set_zkx_gpu_memory_limit_slop_factor(95);

  opts.set_zkx_llvm_force_inline_before_split(true);

  opts.set_zkx_gpu_filter_kernels_spilling_registers_on_autotuning(true);
  opts.set_zkx_gpu_fail_ptx_compilation_on_register_spilling(false);
  opts.set_zkx_gpu_target_config_filename("");

  opts.set_zkx_gpu_enable_llvm_module_compilation_parallelism(false);
  opts.set_zkx_gpu_enable_libnvptxcompiler(se::IsLibNvPtxCompilerSupported());
  opts.set_zkx_gpu_libnvjitlink_mode(DebugOptions::LIB_NV_JIT_LINK_MODE_AUTO);

  opts.set_zkx_syntax_sugar_async_ops(false);
  return opts;
}

static absl::once_flag flags_init;
static DebugOptions* flag_values;
static std::vector<tsl::Flag>* flag_objects;

// Maps pass -> initial fuel values (parsed when AllocateFlags was run).
static auto* const initial_fuel =
    absl::IgnoreLeak(new absl::flat_hash_map<std::string, int64_t>());

// Maps pass -> whether fuel was ever consumed for that pass.
static auto* const fuel_ever_consumed =
    absl::IgnoreLeak(new absl::node_hash_map<std::string, std::atomic<bool>>());

// Maps pass -> remaining fuel.
//
// All threads start off using this global fuel pool, but ResetThreadLocalFuel()
// switches them to a thread-local fuel pool.
static auto* const global_fuel = absl::IgnoreLeak(
    new absl::node_hash_map<std::string, std::atomic<int64_t>>());

// If we're using thread-local fuel, this stores it.
static thread_local std::unique_ptr<
    absl::node_hash_map<std::string, std::atomic<int64_t>>>
    thread_fuel;  // NOLINT (global variable with nontrivial destructor)

// Logs a warning if a pass's fuel was never consumed, on the theory that this
// may be a typo in the flag value.  Called atexit.
static void WarnIfFuelWasNeverConsumed() {
  CHECK(fuel_ever_consumed != nullptr);
  for (const auto& kv : *fuel_ever_consumed) {
    std::string_view pass = kv.first;
    bool was_consumed = kv.second;
    if (!was_consumed) {
      LOG(ERROR) << absl::StreamFormat(
          "Compiler fuel for \"%s\" was never consumed. This may be a typo in "
          "the --zkx_fuel flag you passed.",
          pass);
    }
  }
}

void MakeDebugOptionsFlags(std::vector<tsl::Flag>* flag_list,
                           DebugOptions* debug_options) {
  // Returns a lambda that calls "member_setter" on "debug_options" with the
  // argument passed in to the lambda.
  auto bool_setter_for =
      [debug_options](void (DebugOptions::*member_setter)(bool)) {
        return [debug_options, member_setter](bool value) {
          (debug_options->*member_setter)(value);
          return true;
        };
      };

  // Returns a lambda that calls "member_setter" on "debug_options" with the
  // argument passed in to the lambda.
  auto int32_setter_for =
      [debug_options](void (DebugOptions::*member_setter)(int32_t)) {
        return [debug_options, member_setter](int32_t value) {
          (debug_options->*member_setter)(value);
          return true;
        };
      };

  auto int64_setter_for =
      [debug_options](void (DebugOptions::*member_setter)(int64_t)) {
        return [debug_options, member_setter](int64_t value) {
          (debug_options->*member_setter)(value);
          return true;
        };
      };

  auto string_setter_for = [debug_options](void (DebugOptions::*member_setter)(
                               const std::string& value)) {
    return [debug_options, member_setter](const std::string& value) {
      (debug_options->*member_setter)(value);
      return true;
    };
  };

  auto make_repeated_string_setter =
      [debug_options](void (DebugOptions::*adder)(const std::string&)) {
        return [debug_options, adder](std::string_view comma_separated_values) {
          for (const std::string_view passname :
               absl::StrSplit(comma_separated_values, ',')) {
            (debug_options->*adder)(std::string(passname));
          }
          return true;
        };
      };

  // Custom "sub-parser" lambda for zkx_disable_hlo_passes.
  auto setter_for_zkx_disable_hlo_passes =
      make_repeated_string_setter(&DebugOptions::add_zkx_disable_hlo_passes);

  // Custom "sub-parser" lambda for zkx_enable_hlo_passes_only.
  auto setter_for_zkx_enable_hlo_passes_only = make_repeated_string_setter(
      &DebugOptions::add_zkx_enable_hlo_passes_only);

  // Custom "sub-parser" lambda for zkx_gpu_ptx_file.
  auto setter_for_zkx_gpu_ptx_file = [debug_options](const std::string& value) {
    debug_options->add_zkx_gpu_ptx_file(value);
    return true;
  };

  // Custom "sub-parser" lambda for zkx_gpu_llvm_ir_file.
  auto setter_for_zkx_gpu_llvm_ir_file =
      [debug_options](const std::string& value) {
        debug_options->add_zkx_gpu_llvm_ir_file(value);
        return true;
      };

  // Custom "sub-parser" lambda for zkx_backend_extra_options.
  auto setter_for_zkx_backend_extra_options =
      [debug_options](std::string_view comma_separated_values) {
        auto* extra_options_map =
            debug_options->mutable_zkx_backend_extra_options();
        parse_zkx_backend_extra_options(extra_options_map,
                                        comma_separated_values);
        return true;
      };

  // Custom "sub-parser" for zkx_fuel.  Note that ConsumeFuel does not do any
  // locking on the fuel global variables.  This means that it's
  // illegal/undefined behavior to modify this flag value while the compiler is
  // running.
  auto setter_for_zkx_fuel = [](std::string zkx_fuel_value) {
    initial_fuel->clear();
    global_fuel->clear();
    fuel_ever_consumed->clear();

    for (const auto& kv : absl::StrSplit(zkx_fuel_value, ',')) {
      std::vector<std::string> pass_and_fuel = absl::StrSplit(kv, '=');
      if (pass_and_fuel.size() != 2) {
        LOG(ERROR) << absl::StreamFormat(
            "Illegal value for --zkx_fuel. Saw %s, but expected token %s to "
            "have format X=INTEGER.",
            zkx_fuel_value, kv);
        return false;
      }
      const auto& pass = pass_and_fuel[0];
      const auto& fuel_str = pass_and_fuel[1];
      int64_t fuel;
      if (!absl::SimpleAtoi(fuel_str, &fuel)) {
        LOG(ERROR) << absl::StreamFormat(
            "Illegal value for --zkx_fuel. Saw %s, but expected token %s to be "
            "an integer.",
            zkx_fuel_value, fuel_str);
        return false;
      }
      initial_fuel->emplace(pass, fuel);
      global_fuel->emplace(pass, fuel);
      fuel_ever_consumed->emplace(pass, false);
    }

    // If --zkx_fuel was specified, register an atexit handler which logs a
    // warning if a pass was specified but never consumed any fuel, on the
    // theory that this is may be a typo.
    if (!initial_fuel->empty()) {
      static absl::once_flag register_atexit_once;
      absl::call_once(
          register_atexit_once,
          +[] { std::atexit(WarnIfFuelWasNeverConsumed); });
    }
    return true;
  };

  // Don't use an initializer list for initializing the vector; this would
  // create a temporary copy, and exceeds the stack space when compiling with
  // certain configurations.
  flag_list->push_back(tsl::Flag(
      "zkx_llvm_enable_invariant_load_metadata",
      bool_setter_for(
          &DebugOptions::set_zkx_llvm_enable_invariant_load_metadata),
      debug_options->zkx_llvm_enable_invariant_load_metadata(),
      "In LLVM-based backends, enable the emission of !invariant.load metadata "
      "in the generated IR."));
  flag_list->push_back(tsl::Flag(
      "zkx_llvm_disable_expensive_passes",
      bool_setter_for(&DebugOptions::set_zkx_llvm_disable_expensive_passes),
      debug_options->zkx_llvm_disable_expensive_passes(),
      "In LLVM-based backends, disable a custom set of expensive optimization "
      "passes."));
  flag_list->push_back(tsl::Flag(
      "zkx_backend_optimization_level",
      int32_setter_for(&DebugOptions::set_zkx_backend_optimization_level),
      debug_options->zkx_backend_optimization_level(),
      "Numerical optimization level for the ZKX compiler backend."));
  flag_list->push_back(tsl::Flag(
      "zkx_disable_hlo_passes", setter_for_zkx_disable_hlo_passes, "",
      "Comma-separated list of hlo passes to be disabled. These names must "
      "exactly match the passes' names; no whitespace around commas."));
  flag_list->push_back(tsl::Flag(
      "zkx_enable_hlo_passes_only", setter_for_zkx_enable_hlo_passes_only, "",
      "Comma-separated list of hlo passes to be enabled. These names must "
      "exactly match the passes' names; no whitespace around commas. The "
      "unspecified passes are all disabled."));
  flag_list->push_back(tsl::Flag(
      "zkx_disable_all_hlo_passes",
      bool_setter_for(&DebugOptions::set_zkx_disable_all_hlo_passes), false,
      "Disables all HLO passes. Notes that some passes are necessary for "
      "correctness and the invariants that must be satisfied by 'fully "
      "optimized' HLO are different for different devices and may change "
      "over time. The only 'guarantee', such as it is, is that if you compile "
      "ZKX and dump the optimized HLO for some graph, you should be able to "
      "run it again on the same device with the same build of ZKX."));
  flag_list->push_back(tsl::Flag(
      "zkx_unsupported_crash_on_hlo_pass_fix_max_iterations",
      bool_setter_for(
          &DebugOptions::
              set_zkx_unsupported_crash_on_hlo_pass_fix_max_iterations),
      debug_options->zkx_unsupported_crash_on_hlo_pass_fix_max_iterations(),
      "Crash if HloPassFix can not converge after a fixed number of "
      "iterations."));
  flag_list->push_back(
      tsl::Flag("zkx_embed_ir_in_executable",
                bool_setter_for(&DebugOptions::set_zkx_embed_ir_in_executable),
                debug_options->zkx_embed_ir_in_executable(),
                "Embed the compiler IR as a string in the executable."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_cuda_data_dir", debug_options->mutable_zkx_gpu_cuda_data_dir(),
      "If non-empty, specifies a local directory containing ptxas and nvvm "
      "libdevice files; otherwise we use those from runfile directories."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_ptx_file", setter_for_zkx_gpu_ptx_file, "",
      "If non-empty, specifies a file containing ptx to use. The filename "
      "prefix must have the same pattern as PTX dumped by ZKX. This allows to "
      "match one specific module. General workflow. Get the generated module "
      "ptx from ZKX, modify it, then pass it back via this option."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_llvm_ir_file", setter_for_zkx_gpu_llvm_ir_file, "",
      "If non-empty, specifies a file containing textual LLVM IR to use. The "
      "filename prefix must have the same pattern as LLVM dumped by ZKX "
      "(i.e. module_0001.ir-no-opt.ll -> module_0001.MY_NEW_FILE.ll). This "
      "allows to match one specific module. General workflow. Get the not "
      "optimized LLVM IR from ZKX, modify it, then pass it back via this "
      "option."));
  flag_list->push_back(tsl::Flag(
      "zkx_hlo_profile", bool_setter_for(&DebugOptions::set_zkx_hlo_profile),
      debug_options->zkx_hlo_profile(),
      "Instrument the computation to collect per-HLO cycle counts"));
  flag_list->push_back(tsl::Flag(
      "zkx_backend_extra_options", setter_for_zkx_backend_extra_options, "",
      "Extra options to pass to a backend; comma-separated list of 'key=val' "
      "strings (=val may be omitted); no whitespace around commas."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_disable_gpuasm_optimizations",
      bool_setter_for(&DebugOptions::set_zkx_gpu_disable_gpuasm_optimizations),
      debug_options->zkx_gpu_disable_gpuasm_optimizations(),
      "In ZKX:GPU run ptxas in -O0 (default is -O3)."));
  flag_list->push_back(
      tsl::Flag("zkx_gpu_generate_debug_info",
                bool_setter_for(&DebugOptions::set_zkx_gpu_generate_debug_info),
                debug_options->zkx_gpu_generate_debug_info(),
                "Generate debug info for codegened CUDA kernels."));
  flag_list->push_back(
      tsl::Flag("zkx_gpu_generate_line_info",
                bool_setter_for(&DebugOptions::set_zkx_gpu_generate_line_info),
                debug_options->zkx_gpu_generate_line_info(),
                "Generate line info for codegened CUDA kernels."));
  flag_list->push_back(tsl::Flag(
      "zkx_fuel", setter_for_zkx_fuel, /*default_value_for_display=*/"",
      "Sets compiler fuel, useful for bisecting bugs in passes. Format "
      "--zkx_fuel=PASS1=NUM1,PASS2=NUM2,..."));
  flag_list->push_back(tsl::Flag(
      "zkx_dump_to", string_setter_for(&DebugOptions::set_zkx_dump_to),
      debug_options->zkx_dump_to(),
      "Directory into which debugging data is written. If not specified but "
      "another dumping flag is passed, data will be written to stdout. To "
      "explicitly write to stdout, set this to \"-\". The values \"sponge\" "
      "and \"test_undeclared_outputs_dir\" have a special meaning: They cause "
      "us to dump into the directory specified by the environment variable "
      "TEST_UNDECLARED_OUTPUTS_DIR."));
  flag_list->push_back(tsl::Flag(
      "zkx_flags_reset", bool_setter_for(&DebugOptions::set_zkx_flags_reset),
      debug_options->zkx_flags_reset(),
      "Whether to reset ZKX_FLAGS next time to parse."));
  flag_list->push_back(tsl::Flag(
      "zkx_annotate_with_emitter_loc",
      bool_setter_for(&DebugOptions::set_zkx_annotate_with_emitter_loc),
      debug_options->zkx_annotate_with_emitter_loc(),
      "Forces emitters that use MLIR to annotate all the created MLIR "
      "instructions with the emitter's C++ source file and line number. The "
      "annotations should appear in the MLIR dumps. The emitters should use "
      "EmitterLocOpBuilder for that."));
  flag_list->push_back(tsl::Flag(
      "zkx_dump_hlo_as_text",
      bool_setter_for(&DebugOptions::set_zkx_dump_hlo_as_text),
      debug_options->zkx_dump_hlo_as_text(),
      "Dumps HLO modules as text before and after optimizations. debug_options "
      "are written to the --zkx_dump_to dir, or, if no dir is specified, to "
      "stdout."));
  flag_list->push_back(
      tsl::Flag("zkx_dump_large_constants",
                bool_setter_for(&DebugOptions::set_zkx_dump_large_constants),
                debug_options->zkx_dump_large_constants(),
                "Dumps HLO modules including large constants before and after "
                "optimizations. debug_options are written to the --zkx_dump_to "
                "dir, or, if no dir is specified, to stdout. Ignored unless "
                "zkx_dump_hlo_as_text is true."));
  flag_list->push_back(tsl::Flag(
      "zkx_dump_hlo_snapshots",
      bool_setter_for(&DebugOptions::set_zkx_dump_hlo_snapshots),
      debug_options->zkx_dump_hlo_snapshots(),
      "Every time an HLO module is run, dumps an HloSnapshot to the directory "
      "specified by --zkx_dump_to."));
  flag_list->push_back(tsl::Flag(
      "zkx_dump_hlo_module_re",
      string_setter_for(&DebugOptions::set_zkx_dump_hlo_module_re),
      debug_options->zkx_dump_hlo_module_re(),
      "Limits dumping only to modules which match this regular expression. "
      "Default is to dump all modules."));
  flag_list->push_back(tsl::Flag(
      "zkx_dump_hlo_pass_re",
      string_setter_for(&DebugOptions::set_zkx_dump_hlo_pass_re),
      debug_options->zkx_dump_hlo_pass_re(),
      "If specified, dumps HLO before and after optimization passes which "
      "match this regular expression, in addition to dumping at the very "
      "beginning and end of compilation."));
  flag_list->push_back(
      tsl::Flag("zkx_dump_max_hlo_modules",
                int32_setter_for(&DebugOptions::set_zkx_dump_max_hlo_modules),
                debug_options->zkx_dump_max_hlo_modules(),
                "Max number of hlo module dumps in a directory. Set to < 0 for "
                "unbounded."));
  flag_list->push_back(tsl::Flag(
      "zkx_dump_module_metadata",
      bool_setter_for(&DebugOptions::set_zkx_dump_module_metadata),
      debug_options->zkx_dump_module_metadata(),
      "Dumps HloModuleMetadata as text protos to the directory specified "
      "by --zkx_dump_to."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_unsafe_fallback_to_driver_on_ptxas_not_found",
      bool_setter_for(
          &DebugOptions::
              set_zkx_gpu_unsafe_fallback_to_driver_on_ptxas_not_found),
      debug_options->zkx_gpu_unsafe_fallback_to_driver_on_ptxas_not_found(),
      "If true, ZKX GPU falls back to the driver if ptxas is not found. Note "
      "that falling back to the driver can have drawbacks like using more "
      "memory and/or other bugs during compilation, so we recommend setting "
      "this flag to false."));
  flag_list->push_back(tsl::Flag(
      "zkx_multiheap_size_constraint_per_heap",
      int32_setter_for(
          &DebugOptions::set_zkx_multiheap_size_constraint_per_heap),
      debug_options->zkx_multiheap_size_constraint_per_heap(),
      "Generates multiple heaps (i.e., temp buffers) with a size "
      "constraint on each heap to avoid Out-of-Memory due to memory "
      "fragmentation. The constraint is soft, so it works with tensors "
      "larger than the given constraint size. -1 corresponds to no "
      "constraints."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_force_compilation_parallelism",
      int32_setter_for(
          &DebugOptions::set_zkx_gpu_force_compilation_parallelism),
      debug_options->zkx_gpu_force_compilation_parallelism(),
      "Overrides normal multi-threaded compilation setting to use this many "
      "threads. Setting to 0 (the default value) means no enforcement."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_enable_llvm_module_compilation_parallelism",
      bool_setter_for(
          &DebugOptions::
              set_zkx_gpu_enable_llvm_module_compilation_parallelism),
      debug_options->zkx_gpu_enable_llvm_module_compilation_parallelism(),
      "Decides whether we can do LLVM module compilation in a parallelised "
      "way. If set to false, then it will be single threaded, otherwise the "
      "number of threads depends on the "
      "--zkx_gpu_force_compilation_parallelism flag and the thread pool "
      "supplied to GpuCompiler."));

  flag_list->push_back(tsl::Flag(
      "zkx_gpu_filter_kernels_spilling_registers_on_autotuning",
      bool_setter_for(
          &DebugOptions::
              set_zkx_gpu_filter_kernels_spilling_registers_on_autotuning),
      debug_options->zkx_gpu_filter_kernels_spilling_registers_on_autotuning(),
      "Filter out kernels that spill registers during autotuning"));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_fail_ptx_compilation_on_register_spilling",
      bool_setter_for(
          &DebugOptions::set_zkx_gpu_fail_ptx_compilation_on_register_spilling),
      debug_options->zkx_gpu_fail_ptx_compilation_on_register_spilling(),
      "Fails the PTX compilation if a kernel spills registers."));
  flag_list->push_back(tsl::Flag(
      "zkx_debug_buffer_assignment_show_max",
      int64_setter_for(&DebugOptions::set_zkx_debug_buffer_assignment_show_max),
      debug_options->zkx_debug_buffer_assignment_show_max(),
      "Number of buffers to display when debugging the buffer assignment"));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_target_config_filename",
      string_setter_for(&DebugOptions::set_zkx_gpu_target_config_filename),
      debug_options->zkx_gpu_target_config_filename(),
      "Filename for GPU TargetConfig. Triggers devicless compilation: attached "
      "device is ignored, and the proto is queried instead"));

  flag_list->push_back(tsl::Flag(
      "zkx_gpu_enable_libnvptxcompiler",
      [debug_options](bool enabled) {
        if (enabled && !se::IsLibNvPtxCompilerSupported()) {
          // This feature can't be enabled when ZKX was built without
          // libnvptxcompiler support.
          return false;
        }
        debug_options->set_zkx_gpu_enable_libnvptxcompiler(enabled);
        return true;
      },
      debug_options->zkx_gpu_enable_libnvptxcompiler(),
      "Use libnvptxcompiler for PTX-to-GPU-assembly compilation instead of "
      "calling ptxas."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_enable_libnvjitlink",
      [debug_options](bool enabled) {
        debug_options->set_zkx_gpu_libnvjitlink_mode(
            enabled ? DebugOptions::LIB_NV_JIT_LINK_MODE_ENABLED
                    : DebugOptions::LIB_NV_JIT_LINK_MODE_DISABLED);
        return true;
      },
      se::IsLibNvJitLinkSupported(),
      "Use libnvjitlink for PTX-to-GPU-assembly compilation instead of "
      "calling ptxas."));

  flag_list->push_back(tsl::Flag(
      "zkx_llvm_force_inline_before_split",
      bool_setter_for(&DebugOptions::set_zkx_llvm_force_inline_before_split),
      debug_options->zkx_llvm_force_inline_before_split(),
      "Decide whether to force inline before llvm module split to get "
      "a more "
      "balanced splits for parallel compilation"));

  flag_list->push_back(
      tsl::Flag("zkx_gpu_dump_llvmir",
                bool_setter_for(&DebugOptions::set_zkx_gpu_dump_llvmir),
                debug_options->zkx_gpu_dump_llvmir(), "Dump LLVM IR."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_dump_hlo_unoptimized_snapshots",
      bool_setter_for(
          &DebugOptions::set_zkx_gpu_dump_hlo_unoptimized_snapshots),
      debug_options->zkx_gpu_dump_hlo_unoptimized_snapshots(),
      "Every time an HLO module is run, dumps an HloUnoptimizedSnapshot to the "
      "directory specified by --zkx_dump_to."));
  flag_list->push_back(
      tsl::Flag("zkx_dump_disable_metadata",
                bool_setter_for(&DebugOptions::set_zkx_dump_disable_metadata),
                debug_options->zkx_dump_disable_metadata(),
                "Disable dumping HLO metadata in HLO dumps."));
  flag_list->push_back(tsl::Flag(
      "zkx_dump_hlo_pipeline_re",
      string_setter_for(&DebugOptions::set_zkx_dump_hlo_pipeline_re),
      debug_options->zkx_dump_hlo_pipeline_re(),
      "If specified, dumps HLO before and after optimization passes in the "
      "pass pipelines that match this regular expression."));
  flag_list->push_back(tsl::Flag(
      "zkx_dump_enable_mlir_pretty_form",
      bool_setter_for(&DebugOptions::set_zkx_dump_enable_mlir_pretty_form),
      debug_options->zkx_dump_enable_mlir_pretty_form(),
      "Enable dumping MLIR using pretty print form. If set to false, the "
      "dumped MLIR will be in the llvm-parsable format and can be processed by "
      "mlir-opt tools. Pretty print form is not legal MLIR."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_temp_buffer_use_separate_color",
      bool_setter_for(
          &DebugOptions::set_zkx_gpu_temp_buffer_use_separate_color),
      debug_options->zkx_gpu_temp_buffer_use_separate_color(),
      "Enables temp User Buffer Registration. Enabling this flag will cause a "
      "separate cuda async memory allocator to allocate a temp buffer to the "
      "fixed address on every iteration"));

  flag_list->push_back(tsl::Flag(
      "zkx_gpu_memory_limit_slop_factor",
      int32_setter_for(&DebugOptions::set_zkx_gpu_memory_limit_slop_factor),
      debug_options->zkx_gpu_memory_limit_slop_factor(),
      "Slop factor for memory limits in ZKX:GPU. This flag serves as a "
      "multiplier applied to the total available memory, creating a threshold "
      "that guides the Latency Hiding Scheduler (LHS) in balancing memory "
      "reduction and latency hiding optimizations. This factor effectively "
      "establishes a memory limit for compiler passes, determining when the "
      "scheduler should prioritize: "
      "  1. Memory reduction: When memory usage approaches or exceeds the "
      "calculated "
      "     threshold. "
      "  2. Latency hiding: When memory usage is below the threshold, allowing "
      "for "
      "     more aggressive optimizations that may temporarily increase memory "
      "usage "
      "     but improve overall performance. "
      "By adjusting this factor, users can fine-tune the trade-off between "
      "memory efficiency and performance optimizations. The default value is "
      "95."));
  flag_list->push_back(tsl::Flag(
      "zkx_gpu_enable_highest_priority_async_stream",
      bool_setter_for(
          &DebugOptions::set_zkx_gpu_enable_highest_priority_async_stream),
      debug_options->zkx_gpu_enable_highest_priority_async_stream(),
      "Enable async stream to have the highest priority."));
  flag_list->push_back(
      tsl::Flag("zkx_syntax_sugar_async_ops",
                bool_setter_for(&DebugOptions::set_zkx_syntax_sugar_async_ops),
                debug_options->zkx_syntax_sugar_async_ops(),
                "Enable syntax sugar for async ops in HLO dumps."));
  flag_list->push_back(
      tsl::Flag("zkx_gpu_kernel_cache_file",
                string_setter_for(&DebugOptions::set_zkx_gpu_kernel_cache_file),
                debug_options->zkx_gpu_kernel_cache_file(),
                "Path to a file to cache compiled kernels. Cached kernels get "
                "reused in further compilations; not yet cached kernels are "
                "compiled as usual and get appended to the cache file whenever "
                "possible."));
}  // NOLINT(readability/fn_size)

// Allocates flag_values and flag_objects; this function must not be called more
// than once - its call done via call_once.
static void AllocateFlags(DebugOptions* defaults) {
  if (defaults == nullptr) {
    defaults =
        absl::IgnoreLeak(new DebugOptions(DefaultDebugOptionsIgnoringFlags()));
  }
  flag_values = defaults;
  flag_objects = absl::IgnoreLeak(new std::vector<tsl::Flag>());
  MakeDebugOptionsFlags(flag_objects, flag_values);
  ParseFlagsFromEnvAndDieIfUnknown("ZKX_FLAGS", *flag_objects);
}

void AppendDebugOptionsFlags(std::vector<tsl::Flag>* flag_list,
                             DebugOptions* debug_options) {
  absl::call_once(flags_init, &AllocateFlags, debug_options);
  flag_list->insert(flag_list->end(), flag_objects->begin(),
                    flag_objects->end());
}

DebugOptions GetDebugOptionsFromFlags() {
  absl::call_once(flags_init, &AllocateFlags, nullptr);
  if (flag_values->zkx_flags_reset()) {
    ParseFlagsFromEnvAndDieIfUnknown("ZKX_FLAGS", *flag_objects,
                                     /*reset_envvar=*/true);
  }
  return *flag_values;
}

void ResetThreadLocalFuel() {
  absl::call_once(flags_init, &AllocateFlags, nullptr);

  thread_fuel = std::make_unique<
      absl::node_hash_map<std::string, std::atomic<int64_t>>>();
  CHECK(initial_fuel != nullptr);
  for (const auto& kv : *initial_fuel) {
    thread_fuel->emplace(kv.first, kv.second);
  }
}

bool ConsumeFuel(std::string_view pass, bool* just_ran_out) {
  absl::call_once(flags_init, &AllocateFlags, nullptr);
  if (just_ran_out != nullptr) {
    *just_ran_out = false;
  }
  auto* fuel_pool = thread_fuel ? thread_fuel.get() : global_fuel;
  if (fuel_pool->empty()) {
    return true;
  }
  auto it = fuel_pool->find(pass);
  if (it == fuel_pool->end()) {
    return true;
  }
  std::atomic<int64_t>& remaining_fuel = it->second;
  std::atomic<bool>& fuel_has_been_consumed = fuel_ever_consumed->at(pass);
  fuel_has_been_consumed = true;

  int64_t remaining = remaining_fuel.fetch_sub(1);
  if (just_ran_out != nullptr) {
    *just_ran_out = remaining == 0;
  }
  return remaining > 0;
}

}  // namespace zkx
