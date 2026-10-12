# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

#!/bin/bash

cd ProteinMPNN
rm -r ca_model_weights
rm -r vanilla_model_weights

mkdir ca_model_weights
mkdir vanilla_model_weights

cd ca_model_weights
wget -q https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/ca_model_weights/v_48_002.pt
wget -q https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/ca_model_weights/v_48_010.pt
wget -q https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/ca_model_weights/v_48_020.pt
cd ..

cd vanilla_model_weights
wget -q https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/vanilla_model_weights/v_48_002.pt
wget -q https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/vanilla_model_weights/v_48_010.pt
wget -q https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/vanilla_model_weights/v_48_020.pt
wget -q https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/vanilla_model_weights/v_48_030.pt
cd ..
cd ..