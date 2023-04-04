/*
 * Copyright (c) 2023
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * DNN paddle backend implementation.
 */

#include "dnn_backend_pd.h"
#include "dnn_backend_native.h"
#include "libavutil/avassert.h"
#include "libavutil/avstring.h"
#include "libavutil/cpu.h"
#include "../internal.h"
#include "dnn_io_proc.h"
#include "dnn_backend_common.h"
#include "safe_queue.h"
#include <paddle/pd_inference_api.h>

typedef struct PDOptions {
    float im_height, im_width;
    float scale_factorH, scale_factorW;
    char *input_layout;
    uint8_t async;
    uint32_t nireq;
} PDOptions;

typedef struct PDContext {
    const AVClass *class;
    PDOptions options;
} PDContext;

typedef struct PDModel {
    PDContext ctx;
    DNNModel *model;
    PD_Config *config;
    PD_Predictor *predictor;
    PD_Bool status;
    SafeQueue *request_queue;
    Queue *lltask_queue;
    Queue *task_queue;
} PDModel;
/**
 * Stores execution parameters for single
 * call to the Paddlepaddle C API
 */
typedef struct PDInferRequest {

    PD_OneDimArrayCstr *input_names;
    PD_OneDimArrayCstr *output_names;
    PD_Tensor **output_tensors;
    PD_Tensor *input_tensor;
} PDInferRequest;

typedef struct PDRequestItem {
    PDInferRequest *infer_request;
    LastLevelTaskItem *lltask;
    PD_Bool status;
    DNNAsyncExecModule exec_module;
} PDRequestItem;

#define OFFSET(x) offsetof(PDContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM
static const AVOption dnn_paddle_options[] = {
        {"im_height", "image shape(H,W)", OFFSET(options.im_height), AV_OPT_TYPE_FLOAT, {.dbl = 320}, 0, 10000,
         FLAGS},
        {"im_width", "image shape(H,W)", OFFSET(options.im_width), AV_OPT_TYPE_FLOAT, {.dbl = 320}, 0, 10000,
         FLAGS},
        {"scale_factorH", "scalar factor for height", OFFSET(options.scale_factorH), AV_OPT_TYPE_FLOAT, {.dbl = 1.0},
         0, 10000,
         FLAGS},
        {"scale_factorW", "scalar factor for height", OFFSET(options.scale_factorW), AV_OPT_TYPE_FLOAT, {.dbl = 1.0},
         0, 10000,
         FLAGS},
        {"input_layout", "NHWC or NCHW", OFFSET(options.input_layout), AV_OPT_TYPE_STRING, {.str = "NCHW"}, 0, 0,
         FLAGS},
        DNN_BACKEND_COMMON_OPTIONS
        {NULL}
};

AVFILTER_DEFINE_CLASS(dnn_paddle);

static int execute_model_pd(PDRequestItem *request, Queue *lltask_queue);

static void infer_completion_callback(void *args);

static inline void destroy_request_item(PDRequestItem **arg);


/**
 * Free the contents of Paddle inference request.
 * It does not free the PDInferRequest instance.
 *
 * @param request pointer to PDInferRequest instance.
 * NULL pointer is allowed.
 */
static void pd_free_request(PDInferRequest *request) {
    if (!request)
        return;
    if (request->input_tensor) {
        PD_TensorDestroy(request->input_tensor);
        request->input_tensor = NULL;
    }
    av_freep(&request->input_names);
    av_freep(&request->output_names);
    if (request->output_tensors) {
        int nb_output = sizeof(*request->output_tensors) / sizeof(request->output_tensors[0]);
        for (uint32_t i = 0; i < nb_output; ++i) {
            if (request->output_tensors[i]) {
                PD_TensorDestroy(request->output_tensors[i]);
                request->output_tensors[i] = NULL;
            }
        }
        av_freep(&request->output_tensors);
    }
}

/**
 * Free the PaddkeRequestItem completely.
 *
 * @param arg Address of the PaddleInferRequest instance.
 */
static inline void destroy_request_item(PDRequestItem **arg) {
    PDRequestItem *request;
    if (!arg) {
        return;
    }
    request = *arg;
    pd_free_request(request->infer_request);
    av_freep(&request->infer_request);
    av_freep(&request->lltask);
    ff_dnn_async_module_cleanup(&request->exec_module);
    av_freep(arg);
}

/**
 * Create a Paddle inference request. All properties
 * are initially unallocated and set as NULL.
 *
 * @return pointer to the allocated PDInferRequest instance.
 */
static PDInferRequest *pd_create_inference_request(void) {
    PDInferRequest *infer_request = av_malloc(sizeof(PDInferRequest));
    if (!infer_request) {
        return NULL;
    }
    infer_request->input_names = NULL;
    infer_request->output_names = NULL;
    infer_request->input_tensor = NULL;
    infer_request->output_tensors = NULL;
    return infer_request;
}

static int load_pd_model(PDModel *pd_model, const char *model_filename) {

    PDContext *ctx = &pd_model->ctx;
    char *model_path = (char *) malloc(strlen(model_filename) + strlen(".pdmodel")+1);
    char *params_path = (char *) malloc(strlen(model_filename) + strlen(".pdiparams")+1);
    pd_model->config = PD_ConfigCreate();
    strcpy(model_path, model_filename);
    strcat(model_path, ".pdmodel");
    strcpy(params_path, model_filename);
    strcat(params_path, ".pdiparams");
    PD_ConfigSetModel(pd_model->config, model_path, params_path);
    free(model_path);
    free(params_path);
    pd_model->status = PD_ConfigIsValid(pd_model->config);
    pd_model->predictor = PD_PredictorCreate(pd_model->config);
    if (!pd_model->status) {
        av_log(ctx, AV_LOG_ERROR, "Failed to read model \"%s\" graph\n", model_filename);
        PD_ConfigDestroy(pd_model->config);
        PD_PredictorDestroy(pd_model->predictor);
        return DNN_GENERIC_ERROR;
    }
    return 0;
}

static float *transposeNHWC2NCHW(float *data, const int32_t shape[4]) {
    // the shape layout is NCHW
    int N = shape[0];
    int H = shape[2];
    int W = shape[3];
    int C = shape[1];
    float *transposed = calloc(shape[0] * shape[1] * shape[2] * shape[3], sizeof(float));
    // [N,H,W,C] -> [N,C,H,W]
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int old_index = n * H * W * C + h * W * C + w * C + c;
                    int new_index = n * C * H * W + c * H * W + h * W + w;
                    transposed[new_index] = data[old_index];
                }
            }
        }
    }
    memcpy(data, transposed, shape[0] * shape[1] * shape[2] * shape[3] * sizeof(float));
    free(transposed);
    return data;
}

static float *transposeNCHW2NHWC(float *data, const int32_t shape[4]) {
    // the shape layout is NCHW
    int N = shape[0];
    int C = shape[1];
    int H = shape[2];
    int W = shape[3];
    float *transposed = calloc(shape[0] * shape[1] * shape[2] * shape[3], sizeof(float));
    // [N,C,H,W] -> [N,H,W,C]
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                for (int c = 0; c < C; ++c) {
                    int old_index = n * C * H * W + c * H * W + h * W + w;
                    int new_index = n * H * W * C + h * W * C + w * C + c;
                    transposed[new_index] = data[old_index];
                }
            }
        }
    }
    memcpy(data, transposed, shape[0] * shape[1] * shape[2] * shape[3] * sizeof(float));
    free(transposed);
    return data;
}

static int get_name_index(PDModel *pd_model, TaskItem *task) {
    int name_index = -1;
    PD_OneDimArrayCstr *pd_input_names = PD_PredictorGetInputNames(pd_model->predictor);
    for (int i = 0; i < pd_input_names->size; ++i) {
        if (strcmp(pd_input_names->data[i], task->input_name) == 0) {
            name_index = i;
        }
    }
    PD_OneDimArrayCstrDestroy(pd_input_names);
    if (name_index == -1) {
        av_log(&pd_model->ctx, AV_LOG_ERROR, "Could not find \"%s\" in model\n", task->input_name);
        return AVERROR(EINVAL);
    }
    return name_index;
}

static int pd_start_inference(void *args) {
    DNNData input;
    PDRequestItem *request = args;
    PDInferRequest *infer_request = request->infer_request;
    LastLevelTaskItem *lltask = request->lltask;
    TaskItem *task = lltask->task;
    PDModel *pd_model = task->model;
    // get input data nhwc
    PD_Tensor *input_tensor = infer_request->input_tensor;
    int32_t input_shape[4] = {1, -1, -1, -1};

    for (int i = 0; i < infer_request->input_names->size; ++i) {

        if (strcmp(infer_request->input_names->data[i], "im_shape") == 0) {
            PD_Tensor *im_shape_tensor = PD_PredictorGetInputHandle(pd_model->predictor,
                                                                    infer_request->input_names->data[i]);
            int32_t im_shape_shape[2] = {1, 2};
            float im_shape_data[2] = {pd_model->ctx.options.im_height, pd_model->ctx.options.im_height};
            PD_TensorReshape(im_shape_tensor, 2, im_shape_shape);
            PD_TensorCopyFromCpuFloat(im_shape_tensor, im_shape_data);
        } else if (strcmp(infer_request->input_names->data[i], "scale_factor") == 0) {
            PD_Tensor *scale_factor_tensor = PD_PredictorGetInputHandle(pd_model->predictor,
                                                                        infer_request->input_names->data[i]);
            int32_t scale_factor_shape[2] = {1, 2};
            float scale_factor_data[2] = {pd_model->ctx.options.scale_factorH, pd_model->ctx.options.scale_factorW};
            PD_TensorReshape(scale_factor_tensor, 2, scale_factor_shape);
            PD_TensorCopyFromCpuFloat(scale_factor_tensor, scale_factor_data);
        }
    }

    if (strcmp(pd_model->ctx.options.input_layout, "NCHW") == 0) {
        input_shape[1] = 3;
        input_shape[2] = task->in_frame->height;
        input_shape[3] = task->in_frame->width;
    } else if (strcmp(pd_model->ctx.options.input_layout, "NHWC") == 0) {
        input_shape[1] = task->in_frame->height;
        input_shape[2] = task->in_frame->width;
        input_shape[3] = 3;
    } else {
        av_log(&pd_model->ctx, AV_LOG_ERROR, "The input layout should be NCHW or NHWC\n");
    }
    float *in_data = (float *) calloc(1 * input_shape[1] * input_shape[2] * input_shape[3], sizeof(float));
    PD_TensorCopyToCpuFloat(input_tensor, in_data);
    if (strcmp(pd_model->ctx.options.input_layout, "NCHW") == 0) {
        in_data = transposeNHWC2NCHW(in_data, input_shape);
    }

    PD_TensorReshape(input_tensor, 4, input_shape);
    PD_TensorCopyFromCpuFloat(input_tensor, in_data);
    free(in_data);

    request->status = PD_PredictorRun(pd_model->predictor);

    if (!request->status) {
        av_log(&pd_model->ctx, AV_LOG_ERROR, "%s", "paddlepaddle predictor run fail!");
        pd_free_request(infer_request);
        if (ff_safe_queue_push_back(pd_model->request_queue, request) < 0) {
            destroy_request_item(&request);
        }
        return DNN_GENERIC_ERROR;
    }
    return 0;
}

static void infer_completion_callback(void *args) {
    PDRequestItem *request = args;
    LastLevelTaskItem *lltask = request->lltask;
    TaskItem *task = lltask->task;
    DNNData *outputs;
    PDInferRequest *infer_request = request->infer_request;
    PDModel *pd_model = task->model;
    PDContext *ctx = &pd_model->ctx;

    outputs = av_malloc_array(task->nb_output, sizeof(*outputs));
    if (!outputs) {
        av_log(ctx, AV_LOG_ERROR, "Failed to allocate memory for *outputs\n");
        goto err;
    }

    for (uint32_t i = 0; i < task->nb_output; ++i) {
        const size_t shape_size = PD_TensorGetShape(infer_request->output_tensors[i])->size;
        int32_t length = 1;
        PD_DataType out_dt = PD_TensorGetDataType(infer_request->output_tensors[i]);
        size_t size;
        float *out_data;

        if (strcmp(pd_model->ctx.options.input_layout, "NCHW") == 0) {
            outputs[i].height = PD_TensorGetShape(infer_request->output_tensors[i])->data[2];
            outputs[i].width = PD_TensorGetShape(infer_request->output_tensors[i])->data[3];
            outputs[i].channels = PD_TensorGetShape(infer_request->output_tensors[i])->data[1];
        } else {
            outputs[i].height = PD_TensorGetShape(infer_request->output_tensors[i])->data[1];
            outputs[i].width = PD_TensorGetShape(infer_request->output_tensors[i])->data[2];
            outputs[i].channels = PD_TensorGetShape(infer_request->output_tensors[i])->data[3];
        }

        for (int j = 0; j < shape_size; ++j) {
            length *= PD_TensorGetShape(infer_request->output_tensors[i])->data[j];
        }

        if (out_dt != PD_DATA_FLOAT32){
            av_log(&pd_model->ctx, AV_LOG_ERROR, "The model output datatype has to be float.\n");
        } else {
            outputs[i].dt = DNN_FLOAT;
            size = sizeof(float);
            out_data = (float *) malloc(length * size);
            PD_TensorCopyToCpuFloat(infer_request->output_tensors[i], out_data);
        }

        if (shape_size == 4 && (strcmp(pd_model->ctx.options.input_layout, "NCHW") == 0)) {
            int32_t output_shape[4] = {PD_TensorGetShape(infer_request->output_tensors[i])->data[0],
                                       PD_TensorGetShape(infer_request->output_tensors[i])->data[1],
                                       PD_TensorGetShape(infer_request->output_tensors[i])->data[2],
                                       PD_TensorGetShape(infer_request->output_tensors[i])->data[3]};
            out_data = transposeNCHW2NHWC(out_data, output_shape);
        }

        outputs[i].order = DCO_BGR;
        outputs[i].data = out_data;
    }
    switch (pd_model->model->func_type) {
        case DFT_PROCESS_FRAME:
            //it only support 1 output if it's frame in & frame out
            if (task->do_ioproc) {
                if (pd_model->model->frame_post_proc != NULL) {
                    pd_model->model->frame_post_proc(task->out_frame, outputs, pd_model->model->filter_ctx);
                } else {
                    ff_proc_from_dnn_to_frame(task->out_frame, outputs, ctx);
                }
            } else {
                task->out_frame->width = outputs[0].width;
                task->out_frame->height = outputs[0].height;
            }
            break;
        case DFT_ANALYTICS_DETECT:
            if (!pd_model->model->detect_post_proc) {
                av_log(ctx, AV_LOG_ERROR, "Detect filter needs provide post proc\n");
                return;
            }
            pd_model->model->detect_post_proc(task->in_frame, outputs, task->nb_output, pd_model->model->filter_ctx);
            break;
        default:
            av_log(ctx, AV_LOG_ERROR, "Paddle Inference backend does not support this kind of dnn filter now\n");
            goto err;
    }
    task->inference_done++;
    err:
    pd_free_request(infer_request);
    av_freep(&outputs);
    if (ff_safe_queue_push_back(pd_model->request_queue, request) < 0) {
        destroy_request_item(&request);
        av_log(ctx, AV_LOG_ERROR, "Failed to push back request_queue.\n");
    }
}

static int extract_lltask_from_task(TaskItem *task, Queue *lltask_queue) {
    PDModel *pd_model = task->model;
    PDContext *ctx = &pd_model->ctx;
    LastLevelTaskItem *lltask = av_malloc(sizeof(*lltask));
    if (!lltask) {
        av_log(ctx, AV_LOG_ERROR, "Unable to allocate space for LastLevelTaskItem\n");
        return AVERROR(ENOMEM);
    }
    task->inference_todo = 1;
    task->inference_done = 0;
    lltask->task = task;
    if (ff_queue_push_back(lltask_queue, lltask) < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to push back lltask_queue.\n");
        av_freep(&lltask);
        return AVERROR(ENOMEM);
    }
    return 0;
}

static int get_input_pd(void *model, DNNData *input, const char *input_name) {
    PDModel *pd_model = model;
    PDContext *ctx = &pd_model->ctx;
    int has_name = -1;
    PD_OneDimArrayCstr *pd_input_names = PD_PredictorGetInputNames(pd_model->predictor);
    for (int i = 0; i < pd_input_names->size; ++i) {
        if (strcmp(pd_input_names->data[i], input_name) == 0) {
            has_name = i;
            break;
        }
    }
    PD_OneDimArrayCstrDestroy(pd_input_names);
    if (has_name == -1) {
        av_log(ctx, AV_LOG_ERROR, "Could not find \"%s\" in model\n", input_name);
        return AVERROR(EINVAL);
    }
    input->dt = DNN_FLOAT;
    input->order = DCO_RGB;
    input->height = -1;
    input->width = -1;
    input->channels = 3;
    return 0;
}

static int get_output_pd(void *model, const char *input_name, int input_width, int input_height,
                         const char *output_name, int *output_width, int *output_height) {
    int ret = 0;
    PDModel *pd_model = model;
    PDContext *ctx = &pd_model->ctx;
    TaskItem task;
    PDRequestItem *request;
    DNNExecBaseParams exec_params = {
            .input_name     = input_name,
            .output_names   = &output_name,
            .nb_output      = 1,
            .in_frame       = NULL,
            .out_frame      = NULL,
    };

    ret = ff_dnn_fill_gettingoutput_task(&task, &exec_params, pd_model, input_height, input_width, ctx);
    if (ret != 0) {
        goto err;
    }

    ret = extract_lltask_from_task(&task, pd_model->lltask_queue);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "unable to extract inference from task.\n");
        goto err;
    }

    request = ff_safe_queue_pop_front(pd_model->request_queue);
    if (!request) {
        av_log(ctx, AV_LOG_ERROR, "unable to get infer request.\n");
        ret = AVERROR(EINVAL);
        goto err;
    }

    ret = execute_model_pd(request, pd_model->lltask_queue);
    *output_width = task.out_frame->width;
    *output_height = task.out_frame->height;

    err:
    av_frame_free(&task.out_frame);
    av_frame_free(&task.in_frame);
    return ret;
}

DNNModel *ff_dnn_load_model_pd(const char *model_filename, DNNFunctionType func_type, const char *options,
                               AVFilterContext *filter_ctx) {
    DNNModel *model = NULL;
    PDModel *pd_model = NULL;
    PDRequestItem *item = NULL;
    PDContext *ctx = NULL;

    model = av_mallocz(sizeof(DNNModel));
    if (!model) {
        return NULL;
    }

    pd_model = av_mallocz(sizeof(PDModel));
    if (!pd_model) {
        av_freep(&model);
        return NULL;
    }
    pd_model->model = model;
    ctx = &pd_model->ctx;
    ctx->class = &dnn_paddle_class;

    //parse options
    av_opt_set_defaults(ctx);
    if (av_opt_set_from_string(ctx, options, NULL, "=", "&") < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to parse options \"%s\"\n", options);
        goto err;
    }

    if (load_pd_model(pd_model, model_filename) != 0) {
        goto err;
    }

    if (ctx->options.nireq <= 0) {
        ctx->options.nireq = av_cpu_count() / 2 + 1;
    }

#if !HAVE_PTHREAD_CANCEL
    if (ctx->options.async) {
        ctx->options.async = 0;
        av_log(filter_ctx, AV_LOG_WARNING, "pthread is not supported, roll back to sync.\n");
    }
#endif

    pd_model->request_queue = ff_safe_queue_create();
    if (!pd_model->request_queue) {
        goto err;
    }

    for (int i = 0; i < ctx->options.nireq; i++) {
        PDRequestItem *item = av_mallocz(sizeof(*item));
        if (!item) {
            goto err;
        }
        item->lltask = NULL;
        item->infer_request = pd_create_inference_request();
        if (!item->infer_request) {
            av_log(ctx, AV_LOG_ERROR, "Failed to allocate memory for Paddle inference request\n");
            av_freep(&item);
            goto err;
        }
        item->exec_module.start_inference = &pd_start_inference;
        item->exec_module.callback = &infer_completion_callback;
        item->exec_module.args = item;

        if (ff_safe_queue_push_back(pd_model->request_queue, item) < 0) {
            destroy_request_item(&item);
            goto err;
        }
    }

    pd_model->lltask_queue = ff_queue_create();
    if (!pd_model->lltask_queue) {
        goto err;
    }

    pd_model->task_queue = ff_queue_create();
    if (!pd_model->task_queue) {
        goto err;
    }

    model->model = pd_model;
    model->get_input = &get_input_pd;
    model->get_output = &get_output_pd;
    model->options = options;
    model->filter_ctx = filter_ctx;
    model->func_type = func_type;

    return model;
    err:
    ff_dnn_free_model_pd(&model);
    return NULL;
}

static int fill_model_input_pd(PDModel *pd_model, PDRequestItem *request) {
    DNNData input;
    LastLevelTaskItem *lltask;
    TaskItem *task;
    PDInferRequest *infer_request;
    PDContext *ctx = &pd_model->ctx;
    int ret = 0;
    int32_t input_shape[4] = {1, -1, -1, -1};

    lltask = ff_queue_pop_front(pd_model->lltask_queue);
    av_assert0(lltask);
    task = lltask->task;
    request->lltask = lltask;

    ret = get_input_pd(pd_model, &input, task->input_name);
    if (ret != 0) {
        goto err;
    }

    infer_request = request->infer_request;
    input.height = task->in_frame->height;
    input.width = task->in_frame->width;

    infer_request->input_names = av_malloc(sizeof(PD_OneDimArrayCstr));
    if (!infer_request->input_names) {
        av_log(ctx, AV_LOG_ERROR, "Failed to allocate memory for input tensor\n");
        ret = AVERROR(ENOMEM);
        goto err;
    }

    int name_index = get_name_index(pd_model, task);
    infer_request->input_names = PD_PredictorGetInputNames(pd_model->predictor);
    infer_request->input_tensor = PD_PredictorGetInputHandle(pd_model->predictor,
                                                             infer_request->input_names->data[name_index]);
    if (!infer_request->input_tensor) {
        av_log(ctx, AV_LOG_ERROR, "Failed to allocate memory for input tensor\n");
        ret = AVERROR(ENOMEM);
        goto err;
    }

    if (strcmp(pd_model->ctx.options.input_layout, "NCHW") == 0) {
        input_shape[1] = input.channels;
        input_shape[2] = input.height;
        input_shape[3] = input.width;
    } else if (strcmp(pd_model->ctx.options.input_layout, "NHWC") == 0) {
        input_shape[1] = input.height;
        input_shape[2] = input.width;
        input_shape[3] = input.channels;
    } else {
        av_log(ctx, AV_LOG_ERROR, "The input layout should be NCHW or NHWC\n");
    }
    float *in_data = (float *) calloc(1 * input_shape[1] * input_shape[2] * input_shape[3], sizeof(float));
    PD_TensorReshape(infer_request->input_tensor, 4, input_shape);
    input.data = in_data;
    PD_TensorCopyFromCpuFloat(infer_request->input_tensor, input.data);

    switch (pd_model->model->func_type) {
        case DFT_PROCESS_FRAME:
            if (task->do_ioproc) {
                if (pd_model->model->frame_pre_proc != NULL) {
                    pd_model->model->frame_pre_proc(task->in_frame, &input, pd_model->model->filter_ctx);
                } else {
                    ff_proc_from_frame_to_dnn(task->in_frame, &input, ctx);
                }
                PD_TensorCopyFromCpuFloat(infer_request->input_tensor, input.data);
            }
            break;
        case DFT_ANALYTICS_DETECT:
            ff_proc_from_frame_to_dnn(task->in_frame, &input, ctx);
            PD_TensorCopyFromCpuFloat(infer_request->input_tensor, input.data);
            break;
        default:
            avpriv_report_missing_feature(ctx, "model function type %d", pd_model->model->func_type);
            break;
    }

    infer_request->output_names = PD_PredictorGetOutputNames(pd_model->predictor);;
    if (infer_request->output_names == NULL) {
        av_log(ctx, AV_LOG_ERROR, "Failed to allocate memory for *pd_outputs\n");
        ret = AVERROR(ENOMEM);
        goto err;
    }

    infer_request->output_tensors = av_calloc(task->nb_output, sizeof(*infer_request->output_tensors));
    if (!infer_request->output_tensors) {
        av_log(ctx, AV_LOG_ERROR, "Failed to allocate memory for output tensor\n");
        ret = AVERROR(ENOMEM);
        goto err;
    }


    for (int i = 0; i < task->nb_output; ++i) {
        infer_request->output_tensors[i] = PD_PredictorGetOutputHandle(pd_model->predictor,
                                                                       infer_request->output_names->data[i]);
        if (strcmp(infer_request->output_names->data[i], task->output_names[i]) != 0) {
            av_log(ctx, AV_LOG_ERROR, "Could not find output \"%s\" in model\n", task->output_names[i]);
            ret = DNN_GENERIC_ERROR;
            goto err;
        }
    }
    return 0;
    err:
    pd_free_request(infer_request);
    return ret;
}

static int execute_model_pd(PDRequestItem *request, Queue *lltask_queue) {
    PDModel *pd_model;
    PDContext *ctx;
    LastLevelTaskItem *lltask;
    TaskItem *task;
    int ret;

    if (ff_queue_size(lltask_queue) == 0) {
        destroy_request_item(&request);
        return 0;
    }

    lltask = ff_queue_peek_front(lltask_queue);
    task = lltask->task;
    pd_model = task->model;
    ctx = &pd_model->ctx;

    ret = fill_model_input_pd(pd_model, request);
    if (ret != 0) {
        goto err;
    }

    ret = pd_start_inference(request);
    if (ret != 0) {
        goto err;
    }
    infer_completion_callback(request);
    return (task->inference_done == task->inference_todo) ? 0 : DNN_GENERIC_ERROR;

    err:
    pd_free_request(request->infer_request);
    if (ff_safe_queue_push_back(pd_model->request_queue, request) < 0) {
        destroy_request_item(&request);
    }
    return ret;
}

int ff_dnn_execute_model_pd(const DNNModel *model, DNNExecBaseParams *exec_params) {
    PDModel *pd_model = model->model;
    PDContext *ctx = &pd_model->ctx;
    TaskItem *task;
    PDRequestItem *request;
    int ret = 0;

    ret = ff_check_exec_params(ctx, DNN_PD, model->func_type, exec_params);
    if (ret != 0) {
        return ret;
    }

    task = av_malloc(sizeof(*task));
    if (!task) {
        av_log(ctx, AV_LOG_ERROR, "unable to alloc memory for task item.\n");
        return AVERROR(ENOMEM);
    }

    ret = ff_dnn_fill_task(task, exec_params, pd_model, ctx->options.async, 1);
    if (ret != 0) {
        av_freep(&task);
        return ret;
    }

    if (ff_queue_push_back(pd_model->task_queue, task) < 0) {
        av_freep(&task);
        av_log(ctx, AV_LOG_ERROR, "unable to push back task_queue.\n");
        return AVERROR(ENOMEM);
    }

    ret = extract_lltask_from_task(task, pd_model->lltask_queue);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "unable to extract last level task from task.\n");
        return ret;
    }

    request = ff_safe_queue_pop_front(pd_model->request_queue);
    if (!request) {
        av_log(ctx, AV_LOG_ERROR, "unable to get infer request.\n");
        return AVERROR(EINVAL);
    }
    return execute_model_pd(request, pd_model->lltask_queue);
}

int ff_dnn_flush_pd(const DNNModel *model) {
    PDModel *pd_model = model->model;
    PDContext *ctx = &pd_model->ctx;
    PDRequestItem *request;
    int ret;

    if (ff_queue_size(pd_model->lltask_queue) == 0) {
        // no pending task need to flush
        return 0;
    }

    request = ff_safe_queue_pop_front(pd_model->request_queue);
    if (!request) {
        av_log(ctx, AV_LOG_ERROR, "unable to get infer request.\n");
        return AVERROR(EINVAL);
    }

    ret = fill_model_input_pd(pd_model, request);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to fill model input.\n");
        if (ff_safe_queue_push_back(pd_model->request_queue, request) < 0) {
            destroy_request_item(&request);
        }
        return ret;
    }
    return execute_model_pd(request, pd_model->lltask_queue);
}

void ff_dnn_free_model_pd(DNNModel **model) {
    PDModel *pd_model;

    if (*model) {
        pd_model = (*model)->model;
        while (ff_safe_queue_size(pd_model->request_queue) != 0) {
            PDRequestItem *item = ff_safe_queue_pop_front(pd_model->request_queue);
            destroy_request_item(&item);
        }
        ff_safe_queue_destroy(pd_model->request_queue);

        while (ff_queue_size(pd_model->lltask_queue) != 0) {
            LastLevelTaskItem *item = (LastLevelTaskItem *)ff_queue_pop_front(pd_model->lltask_queue);
            av_freep(&item);
        }
        ff_queue_destroy(pd_model->lltask_queue);

        while (ff_queue_size(pd_model->task_queue) != 0) {
            TaskItem *item = ff_queue_pop_front(pd_model->task_queue);
            av_frame_free(&item->in_frame);
            av_frame_free(&item->out_frame);
            av_freep(&item);
        }
        ff_queue_destroy(pd_model->task_queue);
        av_freep(&pd_model);
        av_freep(model);
    }
}

DNNAsyncStatusType ff_dnn_get_result_pd(const DNNModel *model, AVFrame **in, AVFrame **out) {
    PDModel *pd_model = model->model;
    return ff_dnn_get_result_common(pd_model->task_queue, in, out);
}
