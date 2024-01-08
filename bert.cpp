#include "bert.h"
#include "ggml.h"
#include "ggml-alloc.h"

#ifdef GGML_USE_CUBLAS
#  include "ggml-cuda.h"
#elif defined(GGML_USE_CLBLAST)
#  include "ggml-opencl.h"
#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h> // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#include <alloca.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#include <stdio.h> // for _fseeki64
#endif

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string.h>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <regex>
#include <thread>

static int g_bert_cpu_only = true;
const float EPSILON = 1e-9;

struct gguf_context;

#ifdef __GNUC__
#ifdef __MINGW32__
#define BERT_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define BERT_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define BERT_ATTRIBUTE_FORMAT(...)
#endif

BERT_ATTRIBUTE_FORMAT(1, 2)
static std::string format(const char* fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

// BEGIN FROM GGML.C since these are all internal structures that aren't exposed... :(
#if UINTPTR_MAX == 0xFFFFFFFF
#define GGML_MEM_ALIGN 4
#else
#define GGML_MEM_ALIGN 16
#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
#define GGML_ALIGNED_MALLOC(size)  _aligned_malloc(size, GGML_MEM_ALIGN)
#define GGML_ALIGNED_FREE(ptr)     _aligned_free(ptr)
#else
inline static void* ggml_aligned_malloc(size_t size) {
    void* aligned_memory = NULL;
#ifdef GGML_USE_METAL
    int result = posix_memalign(&aligned_memory, getpagesize(), size);
#else
    int result = posix_memalign(&aligned_memory, GGML_MEM_ALIGN, size);
#endif
    if (result != 0) {
        // Handle allocation failure
        const char* error_desc = "unknown allocation error";
        switch (result) {
        case EINVAL:
            error_desc = "invalid alignment value";
            break;
        case ENOMEM:
            error_desc = "insufficient memory";
            break;
        }
        GGML_PRINT("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, size / (1024.0 * 1024.0));
        return NULL;
    }
    return aligned_memory;
}
#define GGML_ALIGNED_MALLOC(size)  ggml_aligned_malloc(size)
#define GGML_ALIGNED_FREE(ptr)     free(ptr)
#endif

// END FROM GGML.C since these are all internal structures that aren't exposed... :(

void* bert_host_malloc(size_t size) {
    if ( g_bert_cpu_only )
        return malloc(size);

#ifdef GGML_USE_CUBLAS
    return ggml_cuda_host_malloc(size);
#elif GGML_USE_METAL
    return ggml_metal_host_malloc(size);
#elif GGML_USE_CPU_HBM
    return hbw_malloc(size);
#else
    return malloc(size);
#endif
}

void bert_host_free(void* ptr) {
    if (g_bert_cpu_only)
        return free(ptr);

#ifdef GGML_USE_CUBLAS
    return ggml_cuda_host_free(ptr);
#elif GGML_USE_METAL
    return ggml_metal_host_free(ptr);
#elif GGML_USE_CPU_HBM
    if (ptr != NULL)
        hbw_free(ptr);
#else
    return free(ptr);
#endif
}

#if defined(_WIN32)
static std::string bert_format_win_err(DWORD err) {
    LPSTR buf;
    size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&buf, 0, NULL);
    if (!size) {
        return "FormatMessageA failed";
    }
    std::string ret(buf, size);
    LocalFree(buf);
    return ret;
}
#endif

// Replacement for std::vector<uint8_t> that doesn't require zero-initialization.
struct bert_buffer {
    void* data = NULL;
    size_t size = 0;

    // fallback to malloc / free
    // useful in cases where CUDA can try to allocate PINNED memory
    bool fallback = false;

    void resize(size_t n) {
        if (data) {
            if (fallback) { // NOLINT
                free(data);
            }
            else {
                bert_host_free(data);
            }
        }

        data = bert_host_malloc(n);
        if (!data) {
            fallback = true;
            data = malloc(n);
        }
        else {
            fallback = false;
        }

        GGML_ASSERT(data);
        size = n;
    }

    ~bert_buffer() {
        if (data) {
            if (fallback) { // NOLINT
                free(data);
            }
            else {
                bert_host_free(data);
            }
        }

        data = NULL;
    }
};

struct bert_file {
    // use FILE * so we don't have to re-open the file to mmap
    FILE* fp;
    size_t size;

    bert_file(const char* fname, const char* mode) {
        fp = std::fopen(fname, mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
        }
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
#ifdef _WIN32
        __int64 ret = _ftelli64(fp);
#else
        long ret = std::ftell(fp);
#endif
        GGML_ASSERT(ret != -1); // this really shouldn't fail
        return (size_t)ret;
    }

    void seek(size_t offset, int whence) const {
#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64)offset, whence);
#else
        int ret = std::fseek(fp, (long)offset, whence);
#endif
        GGML_ASSERT(ret == 0); // same
    }

    void read_raw(void* ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        std::size_t ret = std::fread(ptr, len, 1, fp);
        if (ferror(fp)) {
            throw std::runtime_error(format("read error: %s", strerror(errno)));
        }
        if (ret != 1) {
            throw std::runtime_error(std::string("unexpectedly reached end of file"));
        }
    }

    uint32_t read_u32() const {
        uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    void write_raw(const void* ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        size_t ret = std::fwrite(ptr, len, 1, fp);
        if (ret != 1) {
            throw std::runtime_error(format("write error: %s", strerror(errno)));
        }
    }

    void write_u32(std::uint32_t val) const {
        write_raw(&val, sizeof(val));
    }

    ~bert_file() {
        if (fp) {
            std::fclose(fp);
        }
    }
};

struct bert_mmap {
    void* addr;
    size_t size;

    bert_mmap(const bert_mmap&) = delete;

#ifdef _POSIX_MAPPED_FILES
    static constexpr bool SUPPORTED = true;

    bert_mmap(struct bert_file* file, size_t prefetch = (size_t)-1 /* -1 = max value */, bool numa = false) {
        size = file->size;
        int fd = fileno(file->fp);
        int flags = MAP_SHARED;
        // prefetch/readahead impairs performance on NUMA systems
        if (numa) { prefetch = 0; }
#ifdef __linux__
        if (prefetch) { flags |= MAP_POPULATE; }
#endif
        addr = mmap(NULL, file->size, PROT_READ, flags, fd, 0);
        if (addr == MAP_FAILED) {
            throw std::runtime_error(format("mmap failed: %s", strerror(errno)));
        }

        if (prefetch > 0) {
            // Advise the kernel to preload the mapped memory
            if (posix_madvise(addr, std::min(file->size, prefetch), POSIX_MADV_WILLNEED)) {
                fprintf(stderr, "warning: posix_madvise(.., POSIX_MADV_WILLNEED) failed: %s\n",
                    strerror(errno));
            }
        }
        if (numa) {
            // advise the kernel not to use readahead
            // (because the next page might not belong on the same node)
            if (posix_madvise(addr, file->size, POSIX_MADV_RANDOM)) {
                fprintf(stderr, "warning: posix_madvise(.., POSIX_MADV_RANDOM) failed: %s\n",
                    strerror(errno));
            }
        }
    }

    ~bert_mmap() {
        munmap(addr, size);
    }
#elif defined(_WIN32)
    static constexpr bool SUPPORTED = true;

    bert_mmap(struct bert_file* file, bool prefetch = true, bool numa = false) {
        (void)numa;

        size = file->size;

        HANDLE hFile = (HANDLE)_get_osfhandle(_fileno(file->fp));

        HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        DWORD error = GetLastError();

        if (hMapping == NULL) {
            throw std::runtime_error(format("CreateFileMappingA failed: %s", bert_format_win_err(error).c_str()));
        }

        addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        error = GetLastError();
        CloseHandle(hMapping);

        if (addr == NULL) {
            throw std::runtime_error(format("MapViewOfFile failed: %s", bert_format_win_err(error).c_str()));
        }

        if (prefetch) {
            // PrefetchVirtualMemory is only present on Windows 8 and above, so we dynamically load it
            BOOL(WINAPI * pPrefetchVirtualMemory) (HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
            HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");

            // may fail on pre-Windows 8 systems
            pPrefetchVirtualMemory = reinterpret_cast<decltype(pPrefetchVirtualMemory)> (GetProcAddress(hKernel32, "PrefetchVirtualMemory"));

            if (pPrefetchVirtualMemory) {
                // advise the kernel to preload the mapped memory
                WIN32_MEMORY_RANGE_ENTRY range;
                range.VirtualAddress = addr;
                range.NumberOfBytes = (SIZE_T)size;
                if (!pPrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
                    fprintf(stderr, "warning: PrefetchVirtualMemory failed: %s\n",
                        bert_format_win_err(GetLastError()).c_str());
                }
            }
        }
    }

    ~bert_mmap() {
        if (!UnmapViewOfFile(addr)) {
            fprintf(stderr, "warning: UnmapViewOfFile failed: %s\n",
                bert_format_win_err(GetLastError()).c_str());
        }
    }
#else
    static constexpr bool SUPPORTED = false;

    bert_mmap(struct bert_file* file, bool prefetch = true, bool numa = false) {
        (void)file;
        (void)prefetch;
        (void)numa;

        throw std::runtime_error(std::string("mmap not supported"));
    }
#endif
};

typedef void (*togpu_func_t)(struct ggml_tensor* tensor);

static void togpu_nop(struct ggml_tensor* tensor) { // don't offload by default
    (void)tensor;
}


// default hparams (all-MiniLM-L6-v2)
struct bert_hparams
{
    int32_t n_vocab = 30522;
    int32_t n_max_tokens = 512;
    int32_t n_embd = 256;
    int32_t n_intermediate = 1536;
    int32_t n_head = 12;
    int32_t n_layer = 6;
    int32_t f16 = 1;
};

struct bert_layer
{
    // normalization
    struct ggml_tensor *ln_att_w;
    struct ggml_tensor *ln_att_b;

    struct ggml_tensor *ln_out_w;
    struct ggml_tensor *ln_out_b;

    // attention
    struct ggml_tensor *q_w;
    struct ggml_tensor *q_b;
    struct ggml_tensor *k_w;
    struct ggml_tensor *k_b;
    struct ggml_tensor *v_w;
    struct ggml_tensor *v_b;

    struct ggml_tensor *o_w;
    struct ggml_tensor *o_b;

    // ff
    struct ggml_tensor *ff_i_w;
    struct ggml_tensor *ff_i_b;

    struct ggml_tensor *ff_o_w;
    struct ggml_tensor *ff_o_b;
};

struct bert_vocab
{
    std::map<std::string, bert_vocab_id> token_to_id;
    std::map<std::string, bert_vocab_id> subword_token_to_id;

    std::map<bert_vocab_id, std::string> _id_to_token;
    std::map<bert_vocab_id, std::string> _id_to_subword_token;
};

struct bert_model
{
    bert_hparams hparams;
    bert_vocab vocab;

    // embeddings weights
    struct ggml_tensor *word_embeddings = {};
    struct ggml_tensor *token_type_embeddings = {};
    struct ggml_tensor* position_embeddings = {};
    struct ggml_tensor* ln_e_w = {};
    struct ggml_tensor* ln_e_b = {};

    std::vector<bert_layer> layers;

    struct ggml_context* ctx = {};
    // spacecowboy:  Use one or the other of these - tensors is the old way, tensors_by_name follows more of a llamacpp usage.
    std::map<std::string, struct ggml_tensor *> tensors;
    std::vector<std::pair<std::string, struct ggml_tensor*>> tensors_by_name;

    ~bert_model() {
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }

        for (size_t i = 0; i < tensors_by_name.size(); ++i) {
            if (tensors_by_name[i].second->backend == GGML_BACKEND_GPU)
            {
#ifdef GGML_USE_CUBLAS
                ggml_cuda_free_data(tensors_by_name[i].second);
#elif defined(GGML_USE_CLBLAST)
                ggml_cl_free_data(tensors_by_name[i].second);
#endif
                // spacecowboy - Confirm that this memory is actually deleted somewhere...
                //else
                //    free(tensors_by_name[i].second);
            }
        }

#ifdef GGML_USE_CUBLAS
        ggml_cuda_free_scratch();
#endif
    }
};

struct bert_ctx
{
    bert_model model;

    size_t mem_per_token;
    int64_t mem_per_input;
    int32_t max_batch_n;
    bert_buffer buf_compute;
    bert_buffer buf_alloc;

    ggml_allocr* alloc = NULL;
};

static std::string bert_format_tensor_shape(const std::vector<int64_t>& ne) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%5" PRId64, ne.at(0));
    for (size_t i = 1; i < ne.size(); i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %5" PRId64, ne.at(i));
    }
    return buf;
}

static std::string bert_format_tensor_shape(const struct ggml_tensor* t) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%5" PRId64, t->ne[0]);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %5" PRId64, t->ne[i]);
    }
    return buf;
}


struct bert_model_loader {
    int n_tensors = 0;
    int n_created = 0;

    bool use_mmap = false;

    bert_file  file;

    std::unique_ptr<bert_mmap> mapping;

    struct gguf_context* ctx_gguf = NULL;
    struct ggml_context* ctx_meta = NULL;

    bert_model_loader(const std::string& fname) : file(fname.c_str(), "rb") {
        if (fname.empty())
            return;

        return;

        //struct gguf_init_params params = {
        //    /*.no_alloc = */ true,
        //    /*.ctx      = */ &ctx_meta,
        //};

        //ctx_gguf = gguf_init_from_file(fname.c_str(), params);


        //struct gguf_context* ctx = (gguf_context *)GGML_ALIGNED_MALLOC(sizeof(struct gguf_context));
        //ctx_gguf = ctx;

        //// spacecowboy
        //// we'll copy this over to the ctx->infos...I'm guessing this is one reason they switched to GGUF, 
        //// .bin has no indication up front of how many tensors you have, it's simple list of named tensors and
        //// data that is read until you run out of file...
        //std::vector<struct gguf_tensor_info> dummy_infos;

        //auto fin = std::ifstream(fname, std::ios::binary);
        //if (!fin)
        //{
        //    fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname);
        //}

        //uint32_t magic;
        //fin.read((char*)&magic, sizeof(magic));
        //if (magic != BERT_FILE_MAGIC_GGSN)
        //{
        //    fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname);
        //    fin.close();
        //    //return false;
        //}

        //// load weights
        //{
        //    int n_tensors = 0;
        //    size_t total_size = 0;

        //    printf("%s: ", __func__);

        //    while (true)
        //    {
        //        dummy_infos.push_back(gguf_tensor_info());
        //        struct gguf_tensor_info& info = dummy_infos.back();

        //        fin.read(reinterpret_cast<char*>(&info.n_dims), sizeof(info.n_dims));
        //        fin.read(reinterpret_cast<char*>(&info.name.n), sizeof(info.name.n));
        //        fin.read(reinterpret_cast<char*>(&info.type), sizeof(info.type));

        //        if (fin.eof())
        //        {
        //            break;
        //        }

        //        int64_t nelements = 1;
        //        // int64_t ne[2] = { 1, 1 };
        //        for (int i = 0; i < info.n_dims; ++i)
        //        {
        //            int32_t ne_cur;
        //            fin.read(reinterpret_cast<char*>(&ne_cur), sizeof(ne_cur));
        //            info.ne[i] = ne_cur;
        //            nelements *= info.ne[i];
        //        }

        //        fin.read(reinterpret_cast<char*>(&info.name.data), info.name.n);

        //        // disabling some safety checking while I'm converting this since I don't have a model in this scope
        //        //if (model.tensors.find(name.data()) == model.tensors.end())
        //        //{
        //        //    fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
        //        //    //return false;
        //        //}

        //        //auto tensor = model.tensors[name.data()];
        //        //if (ggml_nelements(tensor) != nelements)
        //        //{
        //        //    fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
        //        //    //return false;
        //        //}

        //        //if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1])
        //        //{
        //        //    fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%lld, %lld], expected [%lld, %lld]\n",
        //        //        __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
        //        //    //return false;
        //        //}

        //        size_t bpe = 0;

        //        switch (info.type)
        //        {
        //        case GGML_TYPE_F32: // should be 0 for .bin files.
        //            bpe = ggml_type_size(GGML_TYPE_F32);
        //            break;
        //        case GGML_TYPE_F16: // should be 1 for .bin files.
        //            bpe = ggml_type_size(GGML_TYPE_F16);
        //            break;
        //        case GGML_TYPE_Q4_0: // should be 2 for .bin files.
        //            bpe = ggml_type_size(GGML_TYPE_Q4_0);
        //            assert(info.ne[0] % 64 == 0);
        //            break;
        //        case GGML_TYPE_Q4_1: // should be 3 for .bin files.
        //            bpe = ggml_type_size(GGML_TYPE_Q4_1);
        //            assert(info.ne[0] % 64 == 0);
        //            break;
        //        default:
        //        {
        //            fprintf(stderr, "%s: unknown ftype %d in model file\n", __func__, info.type);
        //            //return false;
        //        }
        //        };

        //        //if ((nelements * bpe) / ggml_blck_size(tensor->type) != ggml_nbytes(tensor))
        //        //{
        //        //    fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %llu\n",
        //        //        __func__, name.data(), ggml_nbytes(tensor), nelements * bpe);
        //        //    //return false;
        //        //}
        //        
        //        const size_t mem_size = info.size * bpe;
        //            

        //        struct ggml_init_params pdata = {
        //            .mem_size = mem_size,
        //            .mem_buffer = NULL,
        //            .no_alloc = false,
        //        };

        //        ctx_meta = ggml_init(pdata);
        //        struct ggml_tensor* data = NULL;
        //        data = ggml_new_tensor_1d(ctx_meta, GGML_TYPE_I8, ctx->size);

        //        fin.read(reinterpret_cast<char*>(data->data), mem_size);

        //        // printf("%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);
        //        total_size += mem_size;
        //        if (++n_tensors % 8 == 0)
        //        {
        //            printf(".");
        //            fflush(stdout);
        //        }
        //    }

        //    // spacecowboy: Now take the dummy_infos and properly memaligned ctx->infos

        //    //printf(" done\n");

        //    //// Calculate space requirements for setting up context buffers later
        //    //{
        //    //    bert_vocab_id tokens[] = { 0, 1, 2, 3 };
        //    //    // TODO: We set the initial buffer size to 32MB and hope it's enough. Maybe there is a better way to do this?
        //    //    new_bert->buf_compute.resize(32 * 1024 * 1024);
        //    //    bert_eval(new_bert, 1, tokens, 4, nullptr);
        //    //    new_bert->max_batch_n = 0;

        //    //    // TODO: Max tokens should be a param?
        //    //    int32_t N = new_bert->model.hparams.n_max_tokens;
        //    //    new_bert->mem_per_input = 1.1 * (new_bert->mem_per_token * N); // add 10% to account for ggml object overhead

        //    //}
        //    //printf("%s: mem_per_token %zu KB, mem_per_input %lld MB\n", __func__, new_bert->mem_per_token / (1 << 10), new_bert->mem_per_input / (1 << 20));

        //    //printf("%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size / 1024.0 / 1024.0, n_tensors);
        //}
    }

    ~bert_model_loader() {
        if (ctx_gguf) {
            GGML_ALIGNED_FREE(ctx_gguf);
            //gguf_free(ctx_gguf);
        }
        if (ctx_meta) {
            ggml_free(ctx_meta);
        }
    }

    const char* get_tensor_name(int i) const {
        return gguf_get_tensor_name(ctx_gguf, i);
    }

    struct ggml_tensor* get_tensor_meta(int i) const {
        return ggml_get_tensor(ctx_meta, get_tensor_name(i));
    }

    struct ggml_tensor* create_tensor_for(struct ggml_context* ctx, struct ggml_tensor* meta, ggml_backend_type backend) {
        if (backend != GGML_BACKEND_CPU) {
            ggml_set_no_alloc(ctx, true);
        }

        struct ggml_tensor* tensor = ggml_dup_tensor(ctx, meta);
        tensor->backend = backend; // TODO: ggml_set_backend
        ggml_set_name(tensor, ggml_get_name(meta));

        if (backend != GGML_BACKEND_CPU) {
            ggml_set_no_alloc(ctx, use_mmap);
        }

        n_created++;

        return tensor;
    }

    struct ggml_tensor* create_tensor(struct ggml_context* ctx, const std::string& name, const std::vector<int64_t>& ne, ggml_backend_type backend) {
        struct ggml_tensor* cur = ggml_get_tensor(ctx_meta, name.c_str());

        if (cur == NULL) {
            throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name.c_str()));
        }

        {
            bool is_ok = true;
            for (size_t i = 0; i < ne.size(); ++i) {
                if (ne[i] != cur->ne[i]) {
                    is_ok = false;
                    break;
                }
            }
            if (!is_ok) {
                throw std::runtime_error(
                    format("%s: tensor '%s' has wrong shape; expected %s, got %s",
                        __func__, name.c_str(),
                        bert_format_tensor_shape(ne).c_str(),
                        bert_format_tensor_shape(cur).c_str()));
            }
        }

        return create_tensor_for(ctx, cur, backend);
    }

    void done_getting_tensors() const {
        if (n_created != n_tensors) {
            throw std::runtime_error(format("%s: wrong number of tensors; expected %d, got %d", __func__, n_tensors, n_created));
        }
    }

    size_t file_offset(const char* name) const {
        const int idx = gguf_find_tensor(ctx_gguf, name);

        if (idx < 0) {
            throw std::runtime_error(format("%s: tensor '%s' not found in the file", __func__, name));
        }

        return gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, idx);
    }

    void load_data_for(struct ggml_tensor* cur) const {
        const size_t offs = file_offset(ggml_get_name(cur));

        file.seek(offs, SEEK_SET);
        file.read_raw(cur->data, ggml_nbytes(cur));
    }

    void load_all_data(bert_model& model) {
        size_t size_data = 0;
        size_t size_lock = 0;
        size_t size_pref = 0; // prefetch

        for (auto tsr : model.tensors)
        {
            struct ggml_tensor* cur = (&tsr)->second;

            switch (cur->backend) {
                case GGML_BACKEND_CPU:
                    break;
#ifdef GGML_USE_CUBLAS
                case GGML_BACKEND_GPU:
                case GGML_BACKEND_GPU_SPLIT:
                    // old code:
                    //ggml_cuda_transform_tensor(lt.data, lt.ggml_tensor);

                    // TODO: test if this works !!
                    ggml_cuda_transform_tensor(cur->data, cur);

                    // spacecowboy:  Need to determine where the original host data lives...I think it might be in a memory block of the context.
                    //if (!use_mmap) {
                    //    free(cur->data);
                    //}
                    break;
#elif defined(GGML_USE_CLBLAST)
                case GGML_BACKEND_GPU:
                    ggml_cl_transform_tensor(cur->data, cur);
                    break;
#endif
                default:
                    continue;
            }
        }

//
//        for (int i = 0; i < gguf_get_n_tensors(ctx_gguf); i++) {
//            struct ggml_tensor* cur = ggml_get_tensor(ctx, gguf_get_tensor_name(ctx_gguf, i));
//            size_data += ggml_nbytes(cur);
//            if (cur->backend == GGML_BACKEND_CPU) {
//                size_pref += ggml_nbytes(cur);
//            }
//        }
//
//        size_t done_size = 0;
//        for (int i = 0; i < gguf_get_n_tensors(ctx_gguf); i++) {
//            struct ggml_tensor* cur = ggml_get_tensor(ctx, gguf_get_tensor_name(ctx_gguf, i));
//            GGML_ASSERT(cur); // unused tensors should have been caught by load_data already
//
//            // allocate temp buffer if not using mmap
//            if (!use_mmap && cur->data == NULL) {
//                GGML_ASSERT(cur->backend != GGML_BACKEND_CPU);
//#ifdef GGML_USE_CPU_HBM
//                cur->data = (uint8_t*)hbw_malloc(ggml_nbytes(cur));
//#else
//                cur->data = (uint8_t*)malloc(ggml_nbytes(cur));
//#endif
//            }
//
//            load_data_for(cur);
//
//            switch (cur->backend) {
//            case GGML_BACKEND_CPU:
//                break;
//#ifdef GGML_USE_CUBLAS
//            case GGML_BACKEND_GPU:
//            case GGML_BACKEND_GPU_SPLIT:
//                // old code:
//                //ggml_cuda_transform_tensor(lt.data, lt.ggml_tensor);
//
//                // TODO: test if this works !!
//                ggml_cuda_transform_tensor(cur->data, cur);
//                if (!use_mmap) {
//                    free(cur->data);
//                }
//                break;
//#elif defined(GGML_USE_CLBLAST)
//            case GGML_BACKEND_GPU:
//                ggml_cl_transform_tensor(cur->data, cur);
//                break;
//#endif
//            default:
//                continue;
//            }
//
//            done_size += ggml_nbytes(cur);
//        }
    }
};

int32_t bert_n_embd(bert_ctx * ctx)
{
    return ctx->model.hparams.n_embd;
}

int32_t bert_n_max_tokens(bert_ctx * ctx)
{
    return ctx->model.hparams.n_max_tokens;
}

const char* bert_vocab_id_to_token(bert_ctx * ctx, bert_vocab_id id) {
    bert_vocab & vocab = ctx->model.vocab;
    auto it = vocab._id_to_token.find(id);
    if (it != vocab._id_to_token.end())
    {
        return it->second.c_str();
    }
    it = vocab._id_to_subword_token.find(id);
    if (it != vocab._id_to_subword_token.end())
    {
        return it->second.c_str();
    }
    return "[UNK TOKEN from bert_vocab]";
}

//
// Cli interface
//

void bert_print_usage(char **argv, const bert_params &params)
{
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1)\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -p PROMPT, --prompt PROMPT\n");
    fprintf(stderr, "                        prompt to start generation with (default: random)\n");
    fprintf(stderr, "  --port p     port to bind in server mode (default: %d)\n", params.port);
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model);
    fprintf(stderr, "\n");
}


bool bert_params_parse(int argc, char **argv, bert_params &params)
{
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "-t" || arg == "--threads")
        {
            params.n_threads = std::stoi(argv[++i]);
        }
        else if (arg == "-p" || arg == "--prompt")
        {
            params.prompt = argv[++i];
        }
        else if (arg == "--port")
        {
            params.port = std::stoi(argv[++i]);
        }
        else if (arg == "-m" || arg == "--model")
        {
            params.model = argv[++i];
        }
        else if (arg == "-h" || arg == "--help")
        {
            bert_print_usage(argv, params);
            exit(0);
        }
        else
        {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            bert_print_usage(argv, params);
            exit(0);
        }
    }

    return true;
}

//
// Tokenizing
//

static size_t utf8_len(char src)
{
    const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

std::string stripAccents(const std::string &inputString)
{
    std::string resultString;
    std::map<std::string, char> accentMap = {{"À", 'A'},{"Á", 'A'},
        {"Â", 'A'},{"Ã", 'A'},{"Ä", 'A'},{"Å", 'A'},{"à", 'a'},{"á", 'a'},
        {"â", 'a'},{"ã", 'a'},{"ä", 'a'},{"å", 'a'},{"È", 'E'},{"É", 'E'},
        {"Ê", 'E'},{"Ë", 'E'},{"è", 'e'},{"é", 'e'},{"ê", 'e'},{"ë", 'e'},
        {"Ì", 'I'},{"Í", 'I'},{"Î", 'I'},{"Ï", 'I'},{"ì", 'i'},{"í", 'i'},
        {"î", 'i'},{"ï", 'i'},{"Ò", 'O'},{"Ó", 'O'},{"Ô", 'O'},{"Õ", 'O'},
        {"Ö", 'O'},{"ò", 'o'},{"ó", 'o'},{"ô", 'o'},{"õ", 'o'},{"ö", 'o'},
        {"Ù", 'U'},{"Ú", 'U'},{"Û", 'U'},{"Ü", 'U'},{"ù", 'u'},{"ú", 'u'},
        {"û", 'u'},{"ü", 'u'},{"Ý", 'Y'},{"ý", 'y'},{"Ç", 'C'},{"ç", 'c'},
        {"Ñ", 'N'},{"ñ", 'n'},
    };

    for (size_t i = 0; i < inputString.length();)
    {
        size_t len = utf8_len(inputString[i]);
        std::string curChar = inputString.substr(i, len);
        auto iter = accentMap.find(curChar);
        if (iter != accentMap.end())
        {
            resultString += iter->second;
        }
        else
        {
            resultString += curChar;
        }
        i += len;
    }

    return resultString;
}

std::string bert_normalize_prompt(const std::string &text)
{
    // TODO: handle chinese characters? https://github.com/huggingface/tokenizers/blob/ef5f50605ddf9f8caef1598c0e4853862b9707a7/tokenizers/src/normalizers/bert.rs#L98
    std::string text2 = stripAccents(text);
    for (size_t i = 0; i < text2.size(); i += utf8_len(text2[i]))
    {
        char c = text2[i];
        if (c >= 'A' && c <= 'Z')
            text2[i] = c - 'A' + 'a';
    }
    return text2;
}
void bert_tokenize(
    struct bert_ctx * ctx,
    const char * text,
    bert_vocab_id * tokens,
    int32_t * n_tokens,
    int32_t n_max_tokens)
{
    int cls_tok_id = 101;
    int sep_tok_id = 102;
    const bert_vocab &vocab = ctx->model.vocab;

    std::string str = text;

    std::vector<std::string> words;
    // first split the text into words
    {
        str = bert_normalize_prompt(str);

        std::string pat = R"([[:punct:]]|[[:alpha:]]+|[[:digit:]]+)";

        std::regex re(pat);
        std::smatch m;

        while (std::regex_search(str, m, re))
        {
            for (std::string x : m)
            {
                words.push_back(x);
            }
            str = m.suffix();
        }
    }

    int32_t t = 0;
    tokens[t++] = cls_tok_id;

    // find the longest tokens that form the words:
    for (const auto &word : words)
    {
        if (word.size() == 0)
            continue;

        int i = 0;
        size_t n = word.size();
        auto *token_map = &vocab.token_to_id;
    loop:
        while (i < n)
        {
            if (t >= n_max_tokens - 1)
                break;
            int j = n;
            while (j > i)
            {
                auto it = token_map->find(word.substr(i, j - i));
                if (it != token_map->end())
                {
                    tokens[t++] = it->second;
                    i = j;
                    token_map = &vocab.subword_token_to_id;
                    goto loop;
                }
                --j;
            }
            if (j == i)
            {
                fprintf(stderr, "%s: unknown token '%s'\n", __func__, word.substr(i, 1).data());
                token_map = &vocab.subword_token_to_id;
                ++i;
            }
        }
    }
    tokens[t++] = sep_tok_id;
    *n_tokens = t;
}

static void bert_load_hparams(std::ifstream& fin, bert_model& model)
{
    auto& hparams = model.hparams;

    fin.read((char*)&hparams.n_vocab, sizeof(hparams.n_vocab));
    fin.read((char*)&hparams.n_max_tokens, sizeof(hparams.n_max_tokens));
    fin.read((char*)&hparams.n_embd, sizeof(hparams.n_embd));
    fin.read((char*)&hparams.n_intermediate, sizeof(hparams.n_intermediate));
    fin.read((char*)&hparams.n_head, sizeof(hparams.n_head));
    fin.read((char*)&hparams.n_layer, sizeof(hparams.n_layer));
    fin.read((char*)&hparams.f16, sizeof(hparams.f16));

    printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
    printf("%s: n_max_tokens   = %d\n", __func__, hparams.n_max_tokens);
    printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
    printf("%s: n_intermediate  = %d\n", __func__, hparams.n_intermediate);
    printf("%s: n_head  = %d\n", __func__, hparams.n_head);
    printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
    printf("%s: f16     = %d\n", __func__, hparams.f16);
}

static void bert_load_vocab(std::ifstream& fin, bert_vocab& vocab, int32_t n_vocab)
{
    std::string word;
    for (int i = 0; i < n_vocab; i++)
    {
        uint32_t len;
        fin.read((char*)&len, sizeof(len));

        word.resize(len);
        fin.read((char*)word.data(), len);

        if (word[0] == '#' && word[1] == '#')
        {
            vocab.subword_token_to_id[word.substr(2)] = i;
            vocab._id_to_subword_token[i] = word;
        }

        if (vocab.token_to_id.count(word) == 0)
        {
            vocab.token_to_id[word] = i;
            vocab._id_to_token[i] = word;
        }
    }
}

struct ggml_tensor* bert_new_tensor_1d(
    struct ggml_context* ctx,
    enum   ggml_type      type,
    int64_t ne0,
    bert_model& model,
    std::string name)
{
    struct ggml_tensor* tensor = ggml_new_tensor_1d(ctx, type, ne0);
    ggml_set_name(tensor, name.c_str());
    model.tensors[name] = tensor;
    return tensor;
}

struct ggml_tensor* bert_new_tensor_2d(
    struct ggml_context* ctx,
    enum   ggml_type      type,
    int64_t ne0,
    int64_t ne1,
    bert_model& model,
    std::string name) 
{
    struct ggml_tensor* tensor = ggml_new_tensor_2d(ctx, type, ne0, ne1);
    ggml_set_name(tensor, name.c_str());
    model.tensors[name] = tensor;
    return tensor;
}


static bool bert_load_tensors_original(std::ifstream& fin, bert_ctx* new_bert, int32_t main_gpu)
{
    //std::string dummy_path("");
    //bert_model_loader ml(dummy_path);

    bert_model& model = new_bert->model;
    bert_vocab& vocab = model.vocab;

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = GGML_TYPE_COUNT;
    switch (model.hparams.f16)
    {
        case 0:
            wtype = GGML_TYPE_F32;
            break;
        case 1:
            wtype = GGML_TYPE_F16;
            break;
        case 2:
            wtype = GGML_TYPE_Q4_0;
            break;
        case 3:
            wtype = GGML_TYPE_Q4_1;
            break;
        default:
        {
            fprintf(stderr, "%s: invalid model file' (bad f16 value %d)\n",
                __func__, model.hparams.f16);
            return false;
        }
    }

    auto& ctx = model.ctx;

    size_t model_mem_req = 0;

    {
        const auto& hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_max_tokens = hparams.n_max_tokens;
        const int n_intermediate = hparams.n_intermediate;
        const int n_vocab = hparams.n_vocab;

        // Calculate size requirements

        model_mem_req += n_embd * n_vocab * ggml_type_sizef(wtype); // word_embeddings
        model_mem_req += n_embd * 2 * ggml_type_sizef(wtype); // token_type_embeddings
        model_mem_req += n_embd * n_max_tokens * ggml_type_sizef(wtype); // position_embeddings

        model_mem_req += 2 * n_embd * ggml_type_sizef(GGML_TYPE_F32); // ln_e_*

        model_mem_req += 4 * n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ln_*

        model_mem_req += 4 * n_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // kqvo weights
        model_mem_req += 4 * n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // kqvo bias

        model_mem_req += 2 * n_layer * (n_embd * n_intermediate * ggml_type_sizef(wtype)); // ff_*_w
        model_mem_req += n_layer * (n_intermediate * ggml_type_sizef(GGML_TYPE_F32)); // ff_i_b
        model_mem_req += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ff_o_b

        model_mem_req += (5 + 16 * n_layer) * 512; // object overhead

        printf("%s: ggml ctx size = %6.2f MB\n", __func__, model_mem_req / (1024.0 * 1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            .mem_size = model_mem_req,
            .mem_buffer = NULL,
            .no_alloc = false,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx)
        {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights
    {
        const auto& hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_intermediate = hparams.n_intermediate;
        const int n_max_tokens = hparams.n_max_tokens;
        const int n_vocab = hparams.n_vocab;

        model.layers.resize(n_layer);

        model.word_embeddings = bert_new_tensor_2d(ctx, wtype, n_embd, n_vocab, model, "embeddings.word_embeddings.weight");
        model.token_type_embeddings = bert_new_tensor_2d(ctx, wtype, n_embd, 2, model, "embeddings.token_type_embeddings.weight");
        model.position_embeddings = bert_new_tensor_2d(ctx, wtype, n_embd, n_max_tokens, model, "embeddings.position_embeddings.weight");

        model.ln_e_w = bert_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd, model, "embeddings.LayerNorm.weight"); // cpu
        model.ln_e_b = bert_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd, model, "embeddings.LayerNorm.bias"); // cpu

        for (int i = 0; i < n_layer; ++i)
        {
            auto& layer = model.layers[i];

            layer.ln_att_w = bert_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd, model, std::string("encoder.layer." + std::to_string(i) + ".attention.output.LayerNorm.weight")); // cpu
            layer.ln_att_b = bert_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd, model, std::string("encoder.layer." + std::to_string(i) + ".attention.output.LayerNorm.bias")); // cpu
            layer.ln_out_w = bert_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd, model, std::string("encoder.layer." + std::to_string(i) + ".output.LayerNorm.weight")); //cpu
            layer.ln_out_b = bert_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd, model, std::string("encoder.layer." + std::to_string(i) + ".output.LayerNorm.bias"));  //cpu

            layer.q_w = bert_new_tensor_2d(ctx, wtype, n_embd, n_embd, model, std::string("encoder.layer." + std::to_string(i) + ".attention.self.query.weight")); 
            layer.q_b = bert_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd, model, std::string("encoder.layer." + std::to_string(i) + ".attention.self.query.bias")); // cpu
            layer.k_w = bert_new_tensor_2d(ctx, wtype, n_embd, n_embd, model, std::string("encoder.layer." + std::to_string(i) + ".attention.self.key.weight"));
            layer.k_b = bert_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd, model, std::string("encoder.layer." + std::to_string(i) + ".attention.self.key.bias")); // cpu
            layer.v_w = bert_new_tensor_2d(ctx, wtype, n_embd, n_embd, model, std::string("encoder.layer." + std::to_string(i) + ".attention.self.value.weight"));
            layer.v_b = bert_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd, model, std::string("encoder.layer." + std::to_string(i) + ".attention.self.value.bias")); // cpu
            layer.o_w = bert_new_tensor_2d(ctx, wtype, n_embd, n_embd, model, std::string("encoder.layer." + std::to_string(i) + ".attention.output.dense.weight"));
            layer.o_b = bert_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd, model, std::string("encoder.layer." + std::to_string(i) + ".attention.output.dense.bias"));

            layer.ff_i_w = bert_new_tensor_2d(ctx, wtype, n_embd, n_intermediate, model, std::string("encoder.layer." + std::to_string(i) + ".intermediate.dense.weight"));
            layer.ff_i_b = bert_new_tensor_1d(ctx, GGML_TYPE_F32, n_intermediate, model, std::string("encoder.layer." + std::to_string(i) + ".intermediate.dense.bias")); // cpu

            layer.ff_o_w = bert_new_tensor_2d(ctx, wtype, n_intermediate, n_embd, model, std::string("encoder.layer." + std::to_string(i) + ".output.dense.weight"));
            layer.ff_o_b = bert_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd, model, std::string("encoder.layer." + std::to_string(i) + ".output.dense.bias")); // cpu
        }
    }

    // load weights
    {
        int n_tensors = 0;
        size_t total_size = 0;

        printf("%s: ", __func__);

        (void)main_gpu;
        ggml_backend_type backend_norm = GGML_BACKEND_CPU;

        if (main_gpu > -1)
        {
            g_bert_cpu_only = false;
#ifdef GGML_USE_CUBLAS
            printf("%s: using CUDA for GPU acceleration\n", __func__);
            ggml_cuda_set_main_device(main_gpu);
            backend_norm = GGML_BACKEND_GPU;
#elif defined(GGML_USE_CLBLAST)
            printf("%s: using OpenCL for GPU acceleration\n", __func__);
            backend_norm = GGML_BACKEND_GPU;
#endif
        }

        while (true)
        {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            fin.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char*>(&length), sizeof(length));
            fin.read(reinterpret_cast<char*>(&ftype), sizeof(ftype));

            if (fin.eof())
            {
                break;
            }

            int64_t nelements = 1;
            int64_t ne[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i)
            {
                int32_t ne_cur;
                fin.read(reinterpret_cast<char*>(&ne_cur), sizeof(ne_cur));
                ne[i] = ne_cur;
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            if (model.tensors.find(name.data()) == model.tensors.end())
            {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];

            tensor->backend = backend_norm;

            if (ggml_nelements(tensor) != nelements)
            {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1])
            {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%lld, %lld], expected [%lld, %lld]\n",
                    __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
                return false;
            }

            if (0)
            {
                static const char* ftype_str[] = {
                    "f32",
                    "f16",
                    "q4_0",
                    "q4_1",
                };
                printf("%24s - [%5lld, %5lld], type = %6s, %6.2f MB, %9zu bytes\n", name.data(), ne[0], ne[1], ftype_str[ftype], ggml_nbytes(tensor) / 1024.0 / 1024.0, ggml_nbytes(tensor));
            }

            size_t bpe = 0;

            switch (ftype)
            {
            case 0:
                bpe = ggml_type_size(GGML_TYPE_F32);
                break;
            case 1:
                bpe = ggml_type_size(GGML_TYPE_F16);
                break;
            case 2:
                bpe = ggml_type_size(GGML_TYPE_Q4_0);
                assert(ne[0] % 64 == 0);
                break;
            case 3:
                bpe = ggml_type_size(GGML_TYPE_Q4_1);
                assert(ne[0] % 64 == 0);
                break;
            default:
            {
                fprintf(stderr, "%s: unknown ftype %d in model file\n", __func__, ftype);
                return false;
            }
            };

            if ((nelements * bpe) / ggml_blck_size(tensor->type) != ggml_nbytes(tensor))
            {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %llu\n",
                    __func__, name.data(), ggml_nbytes(tensor), nelements * bpe);
                return false;
            }

            fin.read(reinterpret_cast<char*>(tensor->data), ggml_nbytes(tensor));

            // printf("%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);
            total_size += ggml_nbytes(tensor);
            if (++n_tensors % 8 == 0)
            {
                printf(".");
                fflush(stdout);
            }
        }

        printf(" done\n");

        //// Calculate space requirements for setting up context buffers later
        //{
        //    bert_vocab_id tokens[] = { 0, 1, 2, 3 };
        //    // TODO: We set the initial buffer size to 32MB and hope it's enough. Maybe there is a better way to do this?
        //    new_bert->buf_compute.resize(32 * 1024 * 1024);
        //    bert_eval(new_bert, 1, tokens, 4, nullptr);
        //    new_bert->max_batch_n = 0;

        //    // TODO: Max tokens should be a param?
        //    int32_t N = new_bert->model.hparams.n_max_tokens;
        //    new_bert->mem_per_input = 1.1 * (new_bert->mem_per_token * N); // add 10% to account for ggml object overhead

        //}
        //printf("%s: mem_per_token %zu KB, mem_per_input %lld MB\n", __func__, new_bert->mem_per_token / (1 << 10), new_bert->mem_per_input / (1 << 20));

        printf("%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size / 1024.0 / 1024.0, n_tensors);
    }
}

static bool bert_model_load(const std::string& path_model, bert_ctx* new_bert, int main_gpu)
{
    printf("%s: loading model from '%s' - please wait ...\n", __func__, path_model);
    try {
        auto fin = std::ifstream(path_model, std::ios::binary);
        if (!fin)
        {
            fprintf(stderr, "%s: failed to open '%s'\n", __func__, path_model);
            return false;
        }

        if ( path_model.ends_with(".bin") )
        // verify magic
        {
            uint32_t magic;
            fin.read((char*)&magic, sizeof(magic));
            if (magic != BERT_FILE_MAGIC_GGSN)
            {
                fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, path_model);
                fin.close();
                return false;
            }
        }

        bert_model_loader ml(path_model);
        bert_model& model = new_bert->model;
        bert_vocab& vocab = model.vocab;

        bert_load_hparams(fin, model);
        bert_load_vocab(fin, vocab, model.hparams.n_vocab);
        if (bert_load_tensors_original(fin, new_bert, main_gpu) == false)
        {
            fin.close();
            return false;
        }

        // Reconstruct the ctx_gguf context from the information already populated by bert_load_tensors_original
        // if I do that...then load_all_data *should* have what it needs to do it's job
        ml.ctx_gguf = (struct gguf_context*)GGML_ALIGNED_MALLOC(sizeof(struct gguf_context));
        memcpy(ml.ctx_gguf->header.magic, "GGUF", sizeof(ml.ctx_gguf->header));
        ml.ctx_gguf->header.version = GGUF_VERSION;
        ml.ctx_gguf->header.n_tensors = model.tensors.size();
        ml.ctx_gguf->header.n_kv = 0;
        ml.ctx_gguf->infos = (gguf_tensor_info*)malloc(ml.ctx_gguf->header.n_tensors * sizeof(struct gguf_tensor_info));

        uint32_t i = 0;
        for (auto tsr : model.tensors) {
            struct gguf_tensor_info* info = &ml.ctx_gguf->infos[i];

            for (int j = 0; j < GGML_MAX_DIMS; ++j) {
                info->ne[j] = 1;
            }

            //size_t name_len = strnlen_s((&tsr)->second->name, GGML_MAX_NAME);
            size_t name_true_len = (&tsr)->first.size();
            size_t name_len = strnlen_s((&tsr)->first.c_str(), GGML_MAX_NAME);
            assert(name_true_len < GGML_MAX_NAME && name_true_len == name_len);
            info->name.n = name_len; 
            info->name.data = (char*)calloc(info->name.n + 1, 1);
            strncpy_s(info->name.data, info->name.n + 1, (&tsr)->first.c_str(), name_len);

            info->n_dims = tsr.second->n_dims;
            for (uint32_t j = 0; j < info->n_dims; ++j) {
                info->ne[j] = tsr.second->ne[j];
            }

            info->type = tsr.second->type;
            // we've already loaded all of this...so we're going to load up a ggml_context by
            // "setting" no_alloc to "false"
            info->offset = 0;

            i++;
        }

        // spacecowboy:  Do we need to be aligned since this is _already in memory_?
        // maybe can ditch...unsure.
        ml.ctx_gguf->alignment = GGUF_DEFAULT_ALIGNMENT;

        int alignment_idx = gguf_find_key(ml.ctx_gguf, "general.alignment");
        if (alignment_idx != -1) {
            ml.ctx_gguf->alignment = gguf_get_val_u32(ml.ctx_gguf, alignment_idx);
        }

        //figure out if I need to calculate alignment

        //load the tensor data into a ggml_context - the ctx_meta

        // how do I do this?  Do I create a fake binary blob with offsets to make the 
        // ggml_context happy so that when it tries to reference it, everything else
        // "works"?
        ml.ctx_meta = nullptr;

        // create the tensors
        //for (uint32_t i = 0; i < ml.ctx_gguf->header.n_tensors; ++i) {
        //    const int64_t ne[GGML_MAX_DIMS] = {
        //        ml.ctx_gguf->infos[i].ne[0],
        //        ml.ctx_gguf->infos[i].ne[1],
        //        ml.ctx_gguf->infos[i].ne[2],
        //        ml.ctx_gguf->infos[i].ne[3],
        //    };

        //    if (!model.tensors.contains(ml.ctx_gguf->infos[i].name.data))
        //    {
        //        fprintf(stderr, "%s: Unable to locate tensor %s\n", __func__, ml.ctx_gguf->infos[i].name.data);
        //        return false;
        //    }

        //    struct ggml_tensor* cur = model.tensors[ml.ctx_gguf->infos[i].name.data];

        //    //struct ggml_tensor* cur = ggml_new_tensor(ctx_data, ctx->infos[i].type, ctx->infos[i].n_dims, ne);

        //    //ok = ok && cur != NULL;

        //    ggml_set_name(cur, ml.ctx_gguf->infos[i].name.data);

        //    //if (!ok) {
        //    //    break;
        //    //}

        //    // point the data member to the appropriate location in the binary blob using the tensor infos
        //    //if (!params.no_alloc) {
        //    //    //cur->data = (char *) data->data + ctx->infos[i].offset - ctx->offset; // offset from start of file
        //    //    cur->data = (char*)data->data + ctx->infos[i].offset;               // offset from data
        //    //}
        //}

        ml.load_all_data(model);

        fin.close();

        return new_bert;
    }
    catch (const std::exception& err) {
        printf("error loading model: %s\n", err.what());
        return false;
    }

    return true;
}

//
// Loading and setup
//
//
struct bert_ctx* bert_load_model_from_file(const char* path_model, int32_t gpu)
{
    ggml_time_init();

    bert_ctx* new_bert = new bert_ctx;
    if (bert_model_load(path_model, new_bert, gpu) == false)
    {
        fprintf(stderr, "%s: failed to load '%s'\n", __func__, path_model);
        delete new_bert;
        return nullptr;
    }

    // spacecowboy - At this point llama.cpp llama_load_model_from_file is finished, however, it has a "common.cpp" that 
    // calls "load_model_from_file" then calls "llama_new_context_with_model", so we'll continue that function now

    // Calculate space requirements for setting up context buffers later
    {
        bert_vocab_id tokens[] = { 0, 1, 2, 3 };
        // TODO: We set the initial buffer size to 32MB and hope it's enough. Maybe there is a better way to do this?
        new_bert->buf_compute.resize(32 * 1024 * 1024);

        static const size_t tensor_alignment = 32;

        // spacecowboy - in Llama.cpp, lines, 8905 -> 8938, it's figuring out a worst case memory requirements
        // Currently - for simplicity - I'm going to just put 32MB in, mainly because it simplifies the refactor at the moment.        

        size_t alloc_size = 32 * 1024 * 1024;
        new_bert->buf_alloc.resize(alloc_size);
        new_bert->alloc = ggml_allocr_new(new_bert->buf_alloc.data, new_bert->buf_alloc.size, tensor_alignment);

#ifdef GGML_USE_CUBLAS
        ggml_cuda_set_scratch_size(alloc_size);
        printf("%s: VRAM scratch buffer: %.2f MB\n", __func__, alloc_size / 1024.0 / 1024.0);

        // calculate total VRAM usage
        auto add_tensor = [](const ggml_tensor* t, size_t& size) {
            if (t->backend == GGML_BACKEND_GPU || t->backend == GGML_BACKEND_GPU_SPLIT) {
                size += ggml_nbytes(t);
            }
        };
        size_t model_vram_size = 0;
        for (const auto& kv : new_bert->model.tensors_by_name) {
            add_tensor(kv.second, model_vram_size);
        }

        size_t ctx_vram_size = alloc_size;
        size_t total_vram_size = model_vram_size + ctx_vram_size;

        printf("%s: total VRAM used: %.2f MB (model: %.2f MB, context: %.2f MB)\n", __func__,
            total_vram_size / 1024.0 / 1024.0,
            model_vram_size / 1024.0 / 1024.0,
            ctx_vram_size / 1024.0 / 1024.0);
#endif

        bert_eval(new_bert, 1, tokens, 4, nullptr);
        new_bert->max_batch_n = 0;

        // TODO: Max tokens should be a param?
        int32_t N = new_bert->model.hparams.n_max_tokens;
        new_bert->mem_per_input = 1.1 * (new_bert->mem_per_token * N); // add 10% to account for ggml object overhead
    }
    printf("%s: mem_per_token %zu KB, mem_per_input %lld MB\n", __func__, new_bert->mem_per_token / (1 << 10), new_bert->mem_per_input / (1 << 20));

    // spacecowboy:  Need to continue the work in llama_new_context_with_model starting at line 8905, but continuing
    // as there is cuda scratch space allocated there that will be important later on when we push the
    // compute graph into cuda.

    return new_bert;
}

void bert_resize_ctx(bert_ctx * ctx, int32_t new_size) {    
    int64_t buf_size_new = ctx->mem_per_input * new_size;

    // TODO: Max memory should be a param? Now just 1 GB
    int64_t GB = 1 << 30;
    //printf("%s: requested_buf_size %lldMB\n", __func__, buf_size_new / (1 << 20));
    if (buf_size_new > GB) {
        int32_t adjusted_new_size = GB / ctx->mem_per_input;
        if (adjusted_new_size < 1) adjusted_new_size = 1;
        //printf("%s: requested batch size %d, actual new batch size %d\n", __func__, new_size, adjusted_new_size);
        new_size = adjusted_new_size;
        buf_size_new = ctx->mem_per_input * new_size;
    }
    if (new_size > ctx->max_batch_n) {
        ctx->buf_compute.resize(buf_size_new);
        ctx->max_batch_n = new_size;
    }
}

void bert_free(bert_ctx * ctx) {
    delete ctx;
}

void bert_eval(
    struct bert_ctx *ctx,
    int32_t n_threads,
    bert_vocab_id *tokens,
    int32_t n_tokens,
    float *embeddings)
{
    bert_eval_batch(ctx, n_threads, 1, &tokens, &n_tokens, embeddings ? &embeddings : nullptr);
}

void bert_eval_batch(
    bert_ctx * ctx,
    int32_t n_threads,
    int32_t n_batch_size,
    bert_vocab_id ** batch_tokens,
    int32_t * n_tokens,
    float ** batch_embeddings)
{
    const bert_model& model = ctx->model;
    bool mem_req_mode = !batch_embeddings;
    // batch_embeddings is nullptr for the initial memory requirements run
    if (!mem_req_mode && n_batch_size > ctx->max_batch_n) {
        bert_resize_ctx(ctx, n_batch_size);
        if (n_batch_size > ctx->max_batch_n) {
            fprintf(stderr, "%s: tried to increase buffers to batch size %d but failed\n", __func__, n_batch_size);
            return;
        }
    }

    // TODO: implement real batching
    for (int ba = 0; ba < n_batch_size; ba++)
    {
        const int N = n_tokens[ba];
        const auto &tokens = batch_tokens[ba];

        const auto &hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_max_tokens = hparams.n_max_tokens;
        const int n_head = hparams.n_head;

        const int d_head = n_embd / n_head;

        // offload functions set the tensor output backend to GPU
        // tensors are GPU-accelerated if any input or the output has been offloaded
        togpu_func_t to_gpu_func = togpu_nop; // nr = non-repeating

#ifdef GGML_USE_CUBLAS
        if ( ! g_bert_cpu_only )
            to_gpu_func = ggml_cuda_assign_buffers_no_alloc;
#endif // GGML_USE_CUBLAS

        std::vector<float> result;
        if (N > n_max_tokens)
        {
            fprintf(stderr, "Too many tokens, maximum is %d\n", n_max_tokens);
            return;
        }

        auto & mem_per_token = ctx->mem_per_token;
        auto & buf_compute   = ctx->buf_compute;

        struct ggml_init_params params = {
            .mem_size = buf_compute.size,
            .mem_buffer = buf_compute.data,
            .no_alloc = false,
        };

        struct ggml_context *ctx0 = ggml_init(params);
        struct ggml_cgraph gf = {};

        // Embeddings. word_embeddings + token_type_embeddings + position_embeddings
        struct ggml_tensor *token_layer = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
        ggml_set_name(token_layer, "token_layer");
        memcpy(token_layer->data, tokens, N * ggml_element_size(token_layer));
        
        struct ggml_tensor *token_types = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
        ggml_set_name(token_types, "token_types");
        to_gpu_func(token_types);
        ggml_set_zero(token_types);

        struct ggml_tensor *positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
        ggml_set_name(positions, "positions");
        to_gpu_func(positions);
        for (int i = 0; i < N; i++)
        {
            ggml_set_i32_1d(positions, i, i);
        }

        struct ggml_tensor *inpL = ggml_get_rows(ctx0, model.word_embeddings, token_layer);
        ggml_set_name(inpL, "get_rows_word_embedding");
        to_gpu_func(inpL);

        inpL = ggml_add(ctx0,
                        ggml_get_rows(ctx0, model.token_type_embeddings, token_types),
                        inpL);
        ggml_set_name(inpL, "add_tokentype_wordembed");
        to_gpu_func(inpL);
        ggml_set_name(inpL->src[0], "add_tokentype_wordembed_src0");
        to_gpu_func(inpL->src[0]);
        inpL = ggml_add(ctx0,
                        ggml_get_rows(ctx0, model.position_embeddings, positions),
                        inpL);
        ggml_set_name(inpL, "add_posembed_tokentype");
        to_gpu_func(inpL);
        ggml_set_name(inpL->src[0], "add_posembed_tokentype_src0");
        to_gpu_func(inpL->src[0]);

        // embd norm
        {
            inpL = ggml_norm(ctx0, inpL, EPSILON);
            ggml_set_name(inpL, "norm_posembed_tokentype");
            to_gpu_func(inpL);

            inpL = ggml_add(ctx0,
                            ggml_mul(ctx0,
                                     ggml_repeat(ctx0, model.ln_e_w, inpL),
                                     inpL),
                            ggml_repeat(ctx0, model.ln_e_b, inpL));
            ggml_set_name(inpL, "add_mul_ln_e_w_repeat_ln_e_b");
            to_gpu_func(inpL);
            ggml_set_name(inpL->src[0], "add_mul_ln_e_w_repeat_ln_e_b_src0");
            to_gpu_func(inpL->src[0]);
            // to_gpu_func(inpL->src[0]->src[0]); // ggml_repeat can't be on the gpu
            // to_gpu_func(inpL->src[1]);         // ggml_repeat can't be on the gpu
        }
        // layers
        for (int il = 0; il < n_layer; il++)
        {
            struct ggml_tensor *cur = inpL;
            to_gpu_func(cur);

            // self-attention
            {
                struct ggml_tensor *Qcur = cur;
                Qcur = ggml_reshape_3d(ctx0,
                                       ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].q_b, Qcur),
                                                ggml_mul_mat(ctx0, model.layers[il].q_w, Qcur)),
                                       d_head, n_head, N);
                ggml_format_name(Qcur, "reshape_add_repeat_q_b_mul_q_w_%d", il);
                to_gpu_func(Qcur);
                ggml_format_name(Qcur->src[0], "reshape_add_repeat_q_b_mul_q_w_src0_%d", il);
                to_gpu_func(Qcur->src[0]);
                ggml_format_name(Qcur->src[0]->src[0], "reshape_add_repeat_q_b_mul_q_w_src0_src0_%d", il);
                // to_gpu_func(Qcur->src[0]->src[0]); // ggml_repeat can't be on the gpu
                ggml_format_name(Qcur->src[0]->src[1], "reshape_add_repeat_q_b_mul_q_w_src0_src1_%d", il);
                to_gpu_func(Qcur->src[0]->src[1]);
                struct ggml_tensor *Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
                ggml_format_name(Q, "permute_QCur_%d", il);
                to_gpu_func(Q);

                struct ggml_tensor *Kcur = cur;
                Kcur = ggml_reshape_3d(ctx0,
                                       ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].k_b, Kcur),
                                                ggml_mul_mat(ctx0, model.layers[il].k_w, Kcur)),
                                       d_head, n_head, N);
                ggml_format_name(Kcur, "reshape_add_repeat_k_b_mul_k_w_%d", il);
                to_gpu_func(Kcur);
                ggml_format_name(Kcur->src[0], "reshape_add_repeat_k_b_mul_k_w_src0_%d", il);
                to_gpu_func(Kcur->src[0]);
                ggml_format_name(Kcur->src[0]->src[0], "reshape_add_repeat_k_b_mul_k_w_src0_src0_%d", il);
                // to_gpu_func(Kcur->src[0]->src[0]); // ggml_repeat can't be on the gpu
                ggml_format_name(Kcur->src[0]->src[1], "reshape_add_repeat_k_b_mul_k_w_src0_src1_%d", il);
                to_gpu_func(Kcur->src[0]->src[1]);
                struct ggml_tensor *K = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);
                ggml_format_name(K, "permute_KCur_%d", il);
                to_gpu_func(K);

                struct ggml_tensor *Vcur = cur;
                Vcur = ggml_reshape_3d(ctx0,
                                       ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].v_b, Vcur),
                                                ggml_mul_mat(ctx0, model.layers[il].v_w, Vcur)),
                                       d_head, n_head, N);
                ggml_format_name(Vcur, "reshape_add_repeat_v_b_mul_v_w_%d", il);
                to_gpu_func(Vcur);
                ggml_format_name(Vcur->src[0], "reshape_add_repeat_v_b_mul_v_w_src0_%d", il);
                to_gpu_func(Vcur->src[0]);
                ggml_format_name(Vcur->src[0]->src[0], "reshape_add_repeat_v_b_mul_v_w_src0_src0_%d", il);
                // to_gpu_func(Vcur->src[0]->src[0]); // ggml_repeat can't be on the gpu
                ggml_format_name(Vcur->src[0]->src[1], "reshape_add_repeat_v_b_mul_v_w_src0_src1_%d", il);
                to_gpu_func(Vcur->src[0]->src[1]);
                struct ggml_tensor *V = ggml_permute(ctx0, Vcur, 0, 2, 1, 3);
                ggml_format_name(V, "permute_VCur_%d", il);
                to_gpu_func(V);

                struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K, Q);
                ggml_format_name(KQ, "matmul_K_Q_%d", il);
                to_gpu_func(KQ);
                // KQ = soft_max(KQ / sqrt(head width))
                KQ = ggml_soft_max(ctx0,
                                   ggml_scale(ctx0,
                                              KQ,
                                              ggml_new_f32(ctx0, 1.0f / sqrt((float)d_head))));
                ggml_format_name(KQ, "softmax_scale_%d", il);
                to_gpu_func(KQ);
                ggml_format_name(KQ->src[0], "softmax_scale_src0_%d", il);
                to_gpu_func(KQ->src[0]);
                ggml_format_name(KQ->src[0]->src[1], "softmax_scale_src0_src1_%d", il);
                to_gpu_func(KQ->src[0]->src[1]);

                V = ggml_cont(ctx0, ggml_transpose(ctx0, V));
                ggml_format_name(V, "cont_transpose_V_%d", il);
                to_gpu_func(V);
                struct ggml_tensor *KQV = ggml_mul_mat(ctx0, V, KQ);
                ggml_format_name(KQV, "matmul_KQV_%d", il);
                to_gpu_func(KQV);
                KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
                ggml_format_name(KQV, "permute_KQV_%d", il);
                to_gpu_func(KQV);

                cur = ggml_cpy(ctx0,
                               KQV,
                               ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));
                ggml_format_name(cur, "cpy_KQV_%d", il);
                to_gpu_func(cur);
                to_gpu_func(cur->src[1]);
            }
            // attention output
            cur = ggml_add(ctx0,
                           ggml_repeat(ctx0, model.layers[il].o_b, cur),
                           ggml_mul_mat(ctx0, model.layers[il].o_w, cur));
            ggml_format_name(cur, "add_repeat_o_b_matmul_o_w_%d", il);
            to_gpu_func(cur);
            ggml_format_name(cur->src[0], "add_repeat_o_b_matmul_o_w_src_0_%d", il);
            // to_gpu_func(cur->src[0]); // ggml_repeat can't be on the gpu
            ggml_format_name(cur->src[1], "add_repeat_o_b_matmul_o_w_src_1_%d", il);
            to_gpu_func(cur->src[1]);

            // re-add the layer input
            cur = ggml_add(ctx0, cur, inpL);
            ggml_format_name(cur, "add_cur_inpL_%d", il);
            to_gpu_func(cur);

            // attention norm
            {
                cur = ggml_norm(ctx0, cur, EPSILON);
                ggml_format_name(cur, "attnnorm_norm_cur_%d", il);
                to_gpu_func(cur);

                cur = ggml_add(ctx0,
                               ggml_mul(ctx0,
                                        ggml_repeat(ctx0, model.layers[il].ln_att_w, cur),
                                        cur),
                               ggml_repeat(ctx0, model.layers[il].ln_att_b, cur));
                ggml_format_name(cur, "attnnorm_add_mul_repeat_att_w_repeat_att_b_%d", il);
                to_gpu_func(cur);
                ggml_format_name(cur->src[0], "attnnorm_add_mul_repeat_att_w_repeat_att_b_src0_%d", il);
                to_gpu_func(cur->src[0]);
                ggml_format_name(cur->src[0]->src[0], "attnnorm_add_mul_repeat_att_w_repeat_att_b_src0_src0_%d", il);
                ggml_format_name(cur->src[1], "attnnorm_add_mul_repeat_att_w_repeat_att_b_src1_%d", il);
                // to_gpu_func(cur->src[0]->src[0]); // ggml_repeat can't be on the gpu
                // to_gpu_func(cur->src[1]);         // ggml_repeat can't be on the gpu
            }
            struct ggml_tensor *att_output = cur;
            // intermediate_output = self.intermediate(attention_output)
            cur = ggml_mul_mat(ctx0, model.layers[il].ff_i_w, cur);
            ggml_format_name(cur, "attnout_matmul_ff_i_w_%d", il);
            to_gpu_func(cur);
            cur = ggml_add(ctx0,
                           ggml_repeat(ctx0, model.layers[il].ff_i_b, cur),
                           cur);
            ggml_format_name(cur, "attnout_add_repeat_ff_i_b_%d", il);
            to_gpu_func(cur);
            ggml_format_name(cur->src[0], "attnout_add_repeat_ff_i_b_src0_%d", il);
            // to_gpu_func(cur->src[0]); // ggml_repeat can't be on the gpu
            cur = ggml_gelu(ctx0, cur);
            ggml_format_name(cur, "attnout_gelu_%d", il);
            to_gpu_func(cur);

            // layer_output = self.output(intermediate_output, attention_output)
            cur = ggml_mul_mat(ctx0, model.layers[il].ff_o_w, cur);
            ggml_format_name(cur, "attnout_matmul_ff_o_w_%d", il);
            to_gpu_func(cur);
            cur = ggml_add(ctx0,
                           ggml_repeat(ctx0, model.layers[il].ff_o_b, cur),
                           cur);
            ggml_format_name(cur, "attnout_add_repeat_ff_o_b_%d", il);
            to_gpu_func(cur);
            ggml_format_name(cur->src[0], "attnout_add_repeat_ff_o_b_src0_%d", il);
            // to_gpu_func(cur->src[0]); // ggml_repeat can't be on the gpu
            // attentions bypass the intermediate layer
            cur = ggml_add(ctx0, att_output, cur);
            ggml_format_name(cur, "attnout_add_att_output_%d", il);
            to_gpu_func(cur);

            // output norm
            {
                cur = ggml_norm(ctx0, cur, EPSILON);
                ggml_format_name(cur, "outnorm_norm_cur_%d", il);
                to_gpu_func(cur);

                cur = ggml_add(ctx0,
                               ggml_mul(ctx0,
                                        ggml_repeat(ctx0, model.layers[il].ln_out_w, cur),
                                        cur),
                               ggml_repeat(ctx0, model.layers[il].ln_out_b, cur));
                ggml_format_name(cur, "outnorm_add_matmul_ln_out_w_repeat_ln_out_b_%d", il);
                to_gpu_func(cur);
                ggml_format_name(cur->src[0], "outnorm_add_matmul_ln_out_w_repeat_ln_out_b_src0_%d", il);
                to_gpu_func(cur->src[0]);
                ggml_format_name(cur->src[0]->src[0], "outnorm_add_matmul_ln_out_w_repeat_ln_out_b_src0_src0_%d", il);
                // to_gpu_func(cur->src[0]->src[0]); // ggml_repeat can't be on the gpu
                ggml_format_name(cur->src[1], "outnorm_add_matmul_ln_out_w_repeat_ln_out_b_src1_%d", il);
                // to_gpu_func(cur->src[1]); // ggml_repeat can't be on the gpu
            }
            inpL = cur;
        }
        inpL = ggml_cont(ctx0, ggml_transpose(ctx0, inpL));
        ggml_set_name(inpL, "postloop_cont_transpose_inpL");
        to_gpu_func(inpL);
        ggml_set_name(inpL->src[0], "postloop_cont_transpose_inpL_src0");
        to_gpu_func(inpL->src[0]);
        // pooler
        struct ggml_tensor *sum = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, N, 1);
        ggml_set_name(sum, "postloop_sum_tnsr");
        to_gpu_func(sum);
        ggml_set_f32(sum, 1.0f / N);
        inpL = ggml_mul_mat(ctx0, inpL, sum);
        ggml_set_name(inpL, "postloop_matmul_inpL_sum");
        to_gpu_func(inpL);

        // normalizer
        ggml_tensor *length = ggml_sqrt(ctx0,
                                        ggml_sum(ctx0, ggml_sqr(ctx0, inpL)));
        ggml_set_name(length, "postloop_sqrt_sum_sqr_inpL");
        to_gpu_func(length);
        ggml_set_name(length->src[0], "postloop_sqrt_sum_sqr_inpL_src0");
        to_gpu_func(length->src[0]);
        ggml_set_name(length->src[0]->src[0], "postloop_sqrt_sum_sqr_inpL_src0_src0");
        to_gpu_func(length->src[0]->src[0]);
        inpL = ggml_scale(ctx0, inpL, ggml_div(ctx0, ggml_new_f32(ctx0, 1.0f), length));
        ggml_set_name(inpL, "postloop_scale_div_length");
        to_gpu_func(inpL);
        ggml_set_name(inpL->src[1], "postloop_scale_div_length_src1");
        to_gpu_func(inpL->src[1]);
        ggml_set_name(inpL->src[1]->src[0], "postloop_scale_div_length_src1_src0");
        to_gpu_func(inpL->src[1]->src[0]);

        ggml_tensor *output = inpL;
        // run the computation
        ggml_build_forward_expand(&gf, output);

        // end of graph building phase

        // spacecowboy:  Unclear if I need this yet...
        //ggml_allocr_alloc_graph(lctx.alloc, gf);

        //struct ggml_tensor* res = gf->nodes[gf->n_nodes - 1];
        //struct ggml_tensor* embeddings = gf->nodes[gf->n_nodes - 2];

        //GGML_ASSERT(strcmp(res->name, "result_output") == 0);
        //GGML_ASSERT(strcmp(embeddings->name, "result_norm") == 0);

#ifdef GGML_USE_CUBLAS
        for (int i = 0; i < gf.n_leafs; i++) {
            ggml_tensor* node = gf.leafs[i];
            if (node->backend == GGML_BACKEND_GPU && node->extra == NULL) {
                ggml_cuda_assign_scratch_offset(node, (char*)node->data - (char*)ctx->buf_alloc.data);
                ggml_cuda_copy_to_device(node);
            }
        }

        for (int i = 0; i < gf.n_nodes; i++) {
            ggml_tensor* node = gf.nodes[i];
            if (node->backend == GGML_BACKEND_GPU && node->extra == NULL) {
                ggml_cuda_assign_scratch_offset(node, (char*)node->data - (char*)ctx->buf_alloc.data);
            }
        }

        // ggml_cuda_set_mul_mat_q(cparams.mul_mat_q);

        //// HACK: ggml-alloc may change the tensor backend when reusing a parent, so force output to be on the CPU here if needed
        //if (!lctx.embedding.empty()) {
        //    embeddings->backend = GGML_BACKEND_CPU;
        //}
        //res->backend = GGML_BACKEND_CPU;
#endif

        ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);


        // float *dat = ggml_get_data_f32(output);
        // pretty_print_tensor(dat, output->ne, output->nb, output->n_dims - 1, "");

        #ifdef GGML_PERF
            // print timing information per ggml operation (for debugging purposes)
            // requires GGML_PERF to be defined
            ggml_graph_print(&gf);
        #endif

        if (!mem_req_mode) {
            memcpy(batch_embeddings[ba], (float *)ggml_get_data(output), sizeof(float) * n_embd);
        } else {
            mem_per_token = ggml_used_mem(ctx0) / N;

            // printf("used_mem = %zu KB \n", ggml_used_mem(ctx0) / 1024);
            // printf("mem_per_token = %zu KB \n", mem_per_token / 1024);
        }

        ggml_free(ctx0);
    }
}

void bert_encode(
    struct bert_ctx *ctx,
    int32_t n_threads,
    const char *texts,
    float *embeddings)
{
    bert_encode_batch(ctx, n_threads, 1, 1, &texts, &embeddings);
}

void bert_encode_batch(
    struct bert_ctx *ctx,
    int32_t n_threads,
    int32_t n_batch_size,
    int32_t n_inputs,
    const char ** texts,
    float **embeddings)
{
    // TODO: Disable batching for now
    n_batch_size = 1;
    /*
    if (n_batch_size > n_inputs) {
        n_batch_size = n_inputs;
    }
    if (n_batch_size > ctx->max_batch_n) {
        bert_resize_ctx(ctx, n_batch_size);
        n_batch_size = ctx->max_batch_n;
    }
    */

    int32_t N = bert_n_max_tokens(ctx);

    std::vector<bert_vocab_id> buf_tokens;
    // Most of this buffer will be unused in typical case where inputs are not that long.
    buf_tokens.resize(N * n_inputs);
    std::vector<int32_t> n_tokens = std::vector<int32_t>(n_inputs);
    std::vector<bert_vocab_id*> unsorted_tokens(n_inputs);
    bert_vocab_id* it_tokens = buf_tokens.data();
    for (int i = 0; i < n_inputs; i++) {
        unsorted_tokens[i] = it_tokens;
        bert_tokenize(ctx, texts[i], it_tokens, &n_tokens[i], N);
        it_tokens += n_tokens[i];
    }

    if (n_batch_size == n_inputs) {
        bert_eval_batch(ctx, n_threads, n_batch_size, unsorted_tokens.data(), n_tokens.data(), embeddings);
    } else {
        // sort the inputs by tokenized length, batch and eval

        std::vector<int> indices;
        indices.reserve(n_inputs);
        for (int i = 0; i < n_inputs; i++)
        {
            indices.push_back(i);
        }

        std::vector<int32_t> sorted_n_tokens = std::vector<int32_t>(n_inputs);

        std::vector<bert_vocab_id *> sorted_tokens(n_inputs);

        std::sort(indices.begin(), indices.end(), [&](int a, int b)
                  { return n_tokens[a] < n_tokens[b]; });

        std::vector<float *> sorted_embeddings(n_inputs);
        memcpy(sorted_embeddings.data(), embeddings, n_inputs * sizeof(float *));

        for (int i = 0; i < n_inputs; i++) {
            sorted_embeddings[i] = embeddings[indices[i]];
            sorted_tokens[i] = unsorted_tokens[indices[i]];
            sorted_n_tokens[i] = n_tokens[indices[i]];
        }

        for (int i = 0; i < n_inputs; i += n_batch_size)
        {
            if (i + n_batch_size > n_inputs) {
                n_batch_size = n_inputs - i;
            }
            bert_eval_batch(ctx, n_threads, n_batch_size, &sorted_tokens[i], &sorted_n_tokens[i], &sorted_embeddings[i]);
        }
    }
}
