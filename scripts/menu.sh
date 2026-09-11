#!/usr/bin/env bash
# scripts/menu.sh - Interactive Terminal Menu for transcribe.cpp Build Configuration
#
# Provides a user-friendly, interactive configuration interface to build:
#   1. Modular Plugin Mode (Zero-Model Minimal Core + Downloadable DLL/SO Plugins)
#   2. Monolithic Mode (Single Binary with compiled-in models)
#   3. Model Selection: minimal-multilingual, full, custom, or none
#   4. Compute Backends: CPU, CUDA, Vulkan, Metal

set -e

# ANSI Colors
C_RESET="\033[0m"
C_BOLD="\033[1m"
C_CYAN="\033[36m"
C_GREEN="\033[32m"
C_YELLOW="\033[33m"
C_RED="\033[31m"
C_WHITE="\033[37m"

print_banner() {
    clear 2>/dev/null || true
    echo -e "${C_CYAN}${C_BOLD}========================================================================${C_RESET}"
    echo -e "${C_CYAN}${C_BOLD}          TRANSCRIBE.CPP - MODULAR BUILD CONFIGURATION MENU             ${C_RESET}"
    echo -e "${C_CYAN}${C_BOLD}========================================================================${C_RESET}"
    echo -e "${C_WHITE}  High-performance speech-to-text engine with dynamic plugin architecture${C_RESET}\n"
}

prompt_choice() {
    local title="$1"
    shift
    local default_idx="$1"
    shift
    local options=("$@")

    echo -e "${C_YELLOW}${C_BOLD}[?] ${title}${C_RESET}"
    for i in "${!options[@]}"; do
        local num=$((i + 1))
        local opt="${options[$i]}"
        local def_tag=""
        if [ "$num" -eq "$default_idx" ]; then
            def_tag=" ${C_GREEN}(Default)${C_RESET}"
        fi
        echo -e "  ${C_BOLD}${num})${C_RESET} ${opt}${def_tag}"
    done

    while true; do
        read -rp "  Enter choice [1-${#options[@]}] (default ${default_idx}): " input_val
        if [ -z "$input_val" ]; then
            CHOICE_RESULT="$default_idx"
            return
        fi
        if [[ "$input_val" =~ ^[0-9]+$ ]] && [ "$input_val" -ge 1 ] && [ "$input_val" -le "${#options[@]}" ]; then
            CHOICE_RESULT="$input_val"
            return
        fi
        echo -e "  ${C_RED}Invalid choice. Please choose between 1 and ${#options[@]}.${C_RESET}"
    done
}

# 1. Architecture Mode
print_banner
prompt_choice "Select Build Architecture Mode" 1 \
    "Modular Dynamic Plugins (TRANSCRIBE_ARCH_DL=ON) - Zero-model core + independent plugins [Recommended for Handy]" \
    "Monolithic Static Build (TRANSCRIBE_ARCH_DL=OFF) - All selected models compiled directly into library"
ARCH_DL="ON"
if [ "$CHOICE_RESULT" -eq 2 ]; then
    ARCH_DL="OFF"
fi

# 2. Model Set
print_banner
prompt_choice "Select Model Preset / Composite" 1 \
    "minimal-multilingual [Parakeet TDT 0.6B + Granite Speech 2B + Qwen3-ASR 1.7B] (Fastest, compact, multilingual)" \
    "full [All 18 speech-to-text models + diarizers]" \
    "custom [Select specific model families manually]" \
    "none [Zero models built-in; ideal for dynamic plugin runtime]"

MODEL_SET="minimal-multilingual"
CUSTOM_MODELS=""
case "$CHOICE_RESULT" in
    1) MODEL_SET="minimal-multilingual" ;;
    2) MODEL_SET="full" ;;
    3)
        MODEL_SET="custom"
        echo -e "\n${C_YELLOW}Available families: parakeet, granite, qwen3_asr, whisper, cohere, canary, moss, sortformer, voxtral, voxtral_realtime, canary_qwen, moonshine, moonshine_streaming, sensevoice, funasr_nano, gigaam, granite_nar, medasr${C_RESET}"
        read -rp "  Enter comma-separated list of families (e.g. parakeet,granite): " CUSTOM_MODELS
        if [ -z "$CUSTOM_MODELS" ]; then
            CUSTOM_MODELS="parakeet,granite,qwen3_asr"
        fi
        ;;
    4) MODEL_SET="none" ;;
esac

# 3. Compute Backend
print_banner
GPU_DETECTED=""
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)
    if [ -n "$GPU_NAME" ]; then
        GPU_DETECTED=" (Detected: ${GPU_NAME})"
    fi
fi

prompt_choice "Select Compute Backend" 1 \
    "CPU (Host SIMD AVX2/AVX-512/NEON)" \
    "CUDA (NVIDIA GPU Acceleration)${GPU_DETECTED}" \
    "Metal (Apple Silicon)" \
    "Vulkan (Cross-vendor GPU acceleration)"

ENABLE_CUDA="OFF"
ENABLE_METAL="OFF"
ENABLE_VULKAN="OFF"
CUDA_ARCH="auto"
case "$CHOICE_RESULT" in
    1) ;;
    2)
        ENABLE_CUDA="ON"
        read -rp "  Enter CUDA architectures (default: 'auto' for current GPU): " input_arch
        if [ -n "$input_arch" ]; then
            CUDA_ARCH="$input_arch"
        fi
        ;;
    3) ENABLE_METAL="ON" ;;
    4) ENABLE_VULKAN="ON" ;;
esac

# 4. Target
print_banner
prompt_choice "Select Target to Build" 1 \
    "transcribe-cli (Command-line tool + all architecture plugins)" \
    "transcribe (C ABI Shared Library)" \
    "all (Build all targets, tests, tools, and plugins)"

TARGET="transcribe-cli"
case "$CHOICE_RESULT" in
    1) TARGET="transcribe-cli" ;;
    2) TARGET="transcribe" ;;
    3) TARGET="all" ;;
esac

# Summary
print_banner
echo -e "${C_GREEN}${C_BOLD}Configuration Summary:${C_RESET}"
echo -e "  ${C_BOLD}Plugin Architecture (TRANSCRIBE_ARCH_DL):${C_RESET} ${ARCH_DL}"
echo -e "  ${C_BOLD}Model Composite (TRANSCRIBE_MODEL_SET):${C_RESET}  ${MODEL_SET}"
if [ "$MODEL_SET" = "custom" ]; then
    echo -e "  ${C_BOLD}Custom Models (TRANSCRIBE_MODELS):${C_RESET}        ${CUSTOM_MODELS}"
fi
BACKEND_DESC="CPU"
if [ "$ENABLE_CUDA" = "ON" ]; then BACKEND_DESC="CUDA ($CUDA_ARCH)"; fi
if [ "$ENABLE_METAL" = "ON" ]; then BACKEND_DESC="Metal"; fi
if [ "$ENABLE_VULKAN" = "ON" ]; then BACKEND_DESC="Vulkan"; fi
echo -e "  ${C_BOLD}Backend:${C_RESET}                                 ${BACKEND_DESC}"
echo -e "  ${C_BOLD}Target:${C_RESET}                                  ${TARGET}\n"

prompt_choice "Start Build Now?" 1 "Yes - Start CMake build" "No - Exit"
if [ "$CHOICE_RESULT" -ne 1 ]; then
    echo -e "${C_YELLOW}Build cancelled.${C_RESET}"
    exit 0
fi

BUILD_DIR="build"
if [ "$ARCH_DL" = "ON" ]; then
    BUILD_DIR="build-plugins"
fi

echo -e "\n${C_CYAN}>>> Configuring CMake in ${BUILD_DIR}...${C_RESET}"
CMAKE_ARGS=(
    "-B" "$BUILD_DIR"
    "-DTRANSCRIBE_ARCH_DL=${ARCH_DL}"
    "-DTRANSCRIBE_MODEL_SET=${MODEL_SET}"
)

if [ "$MODEL_SET" = "custom" ]; then
    CMAKE_ARGS+=("-DTRANSCRIBE_MODELS=${CUSTOM_MODELS}")
fi
if [ "$ENABLE_CUDA" = "ON" ]; then
    CMAKE_ARGS+=("-DTRANSCRIBE_CUDA=ON")
    if [ "$CUDA_ARCH" != "auto" ]; then
        CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}")
    fi
elif [ "$ENABLE_METAL" = "ON" ]; then
    CMAKE_ARGS+=("-DTRANSCRIBE_METAL=ON")
elif [ "$ENABLE_VULKAN" = "ON" ]; then
    CMAKE_ARGS+=("-DTRANSCRIBE_VULKAN=ON")
fi

echo "> cmake ${CMAKE_ARGS[*]}"
cmake "${CMAKE_ARGS[@]}"

echo -e "\n${C_CYAN}>>> Building target '${TARGET}' in ${BUILD_DIR}...${C_RESET}"
BUILD_ARGS=("--build" "$BUILD_DIR")
if [ "$TARGET" != "all" ]; then
    BUILD_ARGS+=("--target" "$TARGET")
fi

echo "> cmake ${BUILD_ARGS[*]}"
cmake "${BUILD_ARGS[@]}"

echo -e "\n${C_GREEN}${C_BOLD}========================================================================${C_RESET}"
echo -e "${C_GREEN}${C_BOLD}                     BUILD COMPLETED SUCCESSFULLY!                      ${C_RESET}"
echo -e "${C_GREEN}${C_BOLD}========================================================================${C_RESET}"
echo -e "Artifacts are located in: ${C_CYAN}${BUILD_DIR}/bin/${C_RESET}"
