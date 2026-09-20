#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "transcribe-graph-opt.h"

#include <array>
#include <cstdio>

static bool run(bool optimized, std::array<float, 8> & values) {
    ggml_context * ctx = ggml_init({ 1024 * 1024, nullptr, true });
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!ctx || !cpu) {
        return false;
    }
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    auto * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_input(x);
    ggml_set_input(b);
    auto * scaled = ggml_scale(ctx, x, 2.0f);
    auto * copy   = ggml_dup(ctx, scaled);
    auto * repeat = ggml_repeat(ctx, b, copy);
    auto * output = ggml_add(ctx, copy, repeat);
    ggml_set_output(output);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    auto report = transcribe::optimize_inference_graph(graph, optimized);
    std::fprintf(stderr, "optimized=%d nodes=%d->%d copies=%d repeats=%d\n", optimized, report.before, report.after,
                 report.copies, report.repeats);
    bool ok     = optimized ? report.copies == 1 && report.repeats == 1 && report.after == report.before - 2 :
                              report.before == report.after;
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    const std::array<float, 8> input = { -4, -3, -2, -1, 0, 1, 2, 3 };
    const std::array<float, 4> bias  = { 0.25f, 0.5f, 0.75f, 1 };
    ggml_backend_tensor_set(x, input.data(), 0, sizeof(input));
    ggml_backend_tensor_set(b, bias.data(), 0, sizeof(bias));
    ok = (ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS) && ok;
    ggml_backend_tensor_get(output, values.data(), 0, sizeof(values));
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i] != input[i] * 2 + bias[i % 4]) {
            std::fprintf(stderr, "value[%zu]=%g expected=%g\n", i, values[i], input[i] * 2 + bias[i % 4]);
        }
        ok = ok && values[i] == input[i] * 2 + bias[i % 4];
    }
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    return ok;
}

static bool preserve_views_and_outputs() {
    auto * ctx    = ggml_init({ 1024 * 1024, nullptr, true });
    auto * x      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    auto * copy   = ggml_dup(ctx, ggml_scale(ctx, x, 2.0f));
    auto * view   = ggml_view_1d(ctx, copy, 4, 0);
    auto * output = ggml_cont(ctx, view);
    ggml_set_output(output);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    auto       report = transcribe::optimize_inference_graph(graph, true);
    const bool ok     = report.before == report.after && view->src[0] == copy && view->view_src == copy;
    if (!ok) {
        std::fprintf(stderr, "view preservation: nodes=%d->%d source=%d storage=%d\n", report.before, report.after,
                     view->src[0] == copy, view->view_src == copy);
    }
    ggml_free(ctx);
    return ok;
}

int main() {
    std::array<float, 8> baseline{}, optimized{};
    if (!run(false, baseline) || !run(true, optimized) || baseline != optimized || !preserve_views_and_outputs()) {
        std::fprintf(stderr, "graph optimizer numerical or storage-preservation failure\n");
        return 1;
    }
    return 0;
}
