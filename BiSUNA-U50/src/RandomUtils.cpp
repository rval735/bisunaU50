//
//  RandomUtils.cpp
//  BiSUNAOpenCL
//
//  Created by RHVT on 31/5/19.
//  Copyright © 2019 R. All rights reserved.
//

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "RandomUtils.hpp"
#include <ctime>
#include <random>

using namespace std;

static mt19937 *randEng = nullptr;

inline mt19937 *RandomUtils::getEngine()
{
    if (randEng == nullptr) {
        const uintT seed = static_cast<uintT>(time(0));
        randEng = new mt19937(seed);
    }
    
    return randEng;
}

void RandomUtils::deleteRand()
{
    if (randEng != nullptr) {
        delete randEng;
    }
}

mt19937 *RandomUtils::changeRandomSeed(const unsigned int seed)
{
    deleteRand();
    randEng = new mt19937(seed);
    return randEng;
}

NFiringRate RandomUtils::randomFiringRate()
{
    int level = randomPositive(frNumberFiringRate - 1);
    switch(level) {
        case 0: return frL1;
        case 1: return frL7;
        default : return frL49;
    }
    
    return frNumberFiringRate;
}

NNeuronType RandomUtils::randomNeuronType()
{
//    int level = randomPositive(ntNumberNeuronType - 2);
    // TODO!! Test here if removing ntControl & ntCPrimer makes
    // a difference
    int level = randomPositive(ntNumberNeuronType - 1);
    return ListedNNeuronType[level];
}

ushortT RandomUtils::randomPositive(ushortT maxVal)
{
    return randomRangeUShort(0, maxVal);
}

template <typename T>
T randomRange(mt19937 *re, T minVal, T maxVal, bool normalDist)
{
    if constexpr (std::is_integral<T>::value) {
        uniform_int_distribution<T> unii(minVal, maxVal);
        return unii(*re);
    }
    else if constexpr (std::is_floating_point<T>::value) {
        if (normalDist == true) {
            normal_distribution<T> ndist(minVal, maxVal);
            return ndist(*re);
        }
        else {
            uniform_real_distribution<T> unif(minVal, maxVal);
            return unif(*re);
        }
    }
    else {
        static_assert(std::is_integral<T>::value, "This function requires integral or floating point");
    }
}

ushortT RandomUtils::randomRangeUShort(ushortT minVal, ushortT maxVal, bool normalDist)
{
    return randomRange<ushortT>(getEngine(), minVal, maxVal, normalDist);
}

floatT RandomUtils::randomRangeFloat(floatT minVal, floatT maxVal, bool normalDist)
{
    return randomRange<floatT>(getEngine(), minVal, maxVal, normalDist);
}

float RandomUtils::randomPositiveFloat(floatT maxVal)
{
    return randomRangeFloat(0.0, maxVal);
}

float RandomUtils::randomNormal(float mean, float stdDev)
{
    return randomRangeFloat(mean, stdDev, true);
}

uintT RandomUtils::randomUInt(uintT maxVal)
{
    return randomRange<uintT>(getEngine(), 0, maxVal, false);
}
