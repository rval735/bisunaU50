//
//  RandomUtils.hpp
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

#ifndef RandomUtils_hpp
#define RandomUtils_hpp

#include "NN/NNetworkStruct.hpp"
#include <random>

class RandomUtils
{
public:
    static ushortT randomPositive(ushortT maxVal);
    static ushort randomRangeUShort(ushort minVal, ushort maxVal, bool normalDist = false);
    static floatT randomRangeFloat(floatT minVal, floatT maxVal, bool normalDist = false);
    static float randomPositiveFloat(floatT maxVal);
    static NNeuronType randomNeuronType();
    static NFiringRate randomFiringRate();
    static float randomNormal(float mean, float stdDev);
    static uintT randomUInt(uintT maxVal);
    // If the caller needs to restart the random function
    // generator or needs to deallocate the object.
    static void deleteRand();
    
    // This function should be used for debugging purposes only
    // to make testing easier with constant seeds
    static mt19937 *changeRandomSeed(const unsigned int seed);
    
    // These two make sure this class is not coppied
    RandomUtils(RandomUtils const &) = delete;
    void operator = (RandomUtils const &) = delete;

private:
    // This makes sure this class is not constructed
    RandomUtils() = delete;
    
    static mt19937 *getEngine();
};

#endif /* RandomUtils_hpp */
