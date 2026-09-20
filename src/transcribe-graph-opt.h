// SPDX-License-Identifier: Apache-2.0
// Adapted from audio.cpp's graph_optimizer.cpp, copyright 2026 ShugoAI LLC.
// See licenses/audio-cpp-Apache-2.0.txt. This subset keeps GGML v0.24 scheduler
// metadata, output tensors and views intact; no downstream GGML APIs are used.
#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "transcribe-env.h"
#include "transcribe-log.h"

#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace transcribe {

struct graph_optimization_report {
    int before  = 0;
    int after   = 0;
    int repeats = 0;
    int copies  = 0;
};

// Only pre-allocation, inference graphs are supported. Explicitly opt in while
// comparing the pass against the unmodified graph on the same backend.
inline graph_optimization_report optimize_inference_graph(ggml_cgraph * graph, bool enabled) {
    graph_optimization_report report;
    report.before = report.after = ggml_graph_n_nodes(graph);
    if (!enabled || report.before < 2) {
        return report;
    }
    auto **                                                       raw = ggml_graph_nodes(graph);
    std::vector<ggml_tensor *>                                    nodes(raw, raw + report.before);
    std::unordered_set<ggml_tensor *>                             view_storage;
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> users;
    for (auto * node : nodes) {
        if (node->view_src != nullptr) {
            view_storage.insert(node->view_src);
            for (auto * src : node->src) {
                if (src != nullptr) {
                    view_storage.insert(src);
                }
            }
        }
        for (auto * src : node->src) {
            if (src != nullptr) {
                users[src].push_back(node);
            }
        }
    }
    std::unordered_set<ggml_tensor *> removed;
    for (int i = 0; i < report.before - 1; ++i) {
        auto * node   = nodes[i];
        auto * source = node->src[0];
        // Copies can intentionally isolate writable storage. Preserve all
        // externally addressable tensors and anything involved in a view.
        if (source == nullptr || (node->flags & ~GGML_TENSOR_FLAG_COMPUTE) != 0 || node->data != nullptr ||
            node->buffer != nullptr || node->view_src != nullptr || view_storage.count(node) ||
            view_storage.count(source) || users[node].empty()) {
            continue;
        }
        bool same = node->type == source->type;
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            same = same && node->ne[dim] == source->ne[dim] && node->nb[dim] == source->nb[dim];
        }
        bool copy   = same && (node->op == GGML_OP_CONT || node->op == GGML_OP_DUP || node->op == GGML_OP_REPEAT) &&
                      source->op != GGML_OP_NONE;
        bool repeat = node->op == GGML_OP_REPEAT;
        for (auto * user : users[node]) {
            copy = copy && user->view_src == nullptr && user->op != GGML_OP_CPY && user->op != GGML_OP_SET &&
                   user->op != GGML_OP_ACC;
            const bool binary = user->op == GGML_OP_ADD || user->op == GGML_OP_SUB || user->op == GGML_OP_MUL ||
                                user->op == GGML_OP_DIV;
            repeat            = repeat && binary && user->src[1] == node && user->src[0] != node &&
                                ggml_can_repeat(source, user->src[0]);
        }
        if (!copy && !repeat) {
            continue;
        }
        // Rewiring does not change the public/output tensor pointers. Retain
        // every other node, including independent state-copy graph roots.
        for (auto * user : users[node]) {
            for (auto *& src : user->src) {
                if (src == node) {
                    src = source;
                }
            }
            users[source].push_back(user);
        }
        removed.insert(node);
        if (copy) {
            ++report.copies;
        } else {
            ++report.repeats;
        }
    }
    if (!removed.empty()) {
        // Rebuild GGML's visited set as well as its schedule. audio.cpp uses a
        // downstream node-count setter that upstream v0.24 does not expose.
        ggml_graph_clear(graph);
        for (auto * node : nodes) {
            if (!removed.count(node)) {
                ggml_build_forward_expand(graph, node);
            }
        }
    }
    report.after = ggml_graph_n_nodes(graph);
    return report;
}

inline graph_optimization_report optimize_inference_graph(ggml_cgraph * graph) {
    return optimize_inference_graph(graph, env::flag("TRANSCRIBE_GRAPH_OPTIMIZER"));
}

inline bool alloc_inference_graph(ggml_backend_sched_t sched, ggml_cgraph * graph) {
    const auto report = optimize_inference_graph(graph);
    if (report.after != report.before) {
        log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "graph optimizer: nodes=%d->%d copies=%d repeats=%d", report.before,
                report.after, report.copies, report.repeats);
    }
    return ggml_backend_sched_alloc_graph(sched, graph);
}

}  // namespace transcribe
