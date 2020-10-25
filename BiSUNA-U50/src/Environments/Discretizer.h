//
//  Discretizer.h
//  BiSUNA
//
//  Created by R on 12/4/19.
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

#ifndef Discretizer_h
#define Discretizer_h

#include "Parameters.hpp"
#include <cmath>

inline float transformFromPT(unsigned short int val, float minBound = -1, float maxBound = 1) {
#ifdef CONTINUOUS_PARAM
    return 0;
#else
    if (val >= MAXIMUM_WEIGHT) {
        return maxBound;
    }
    else if (val == 0) {
        return minBound;
    }
    
    float v = (val - MID_WEIGHT) * maxBound / MID_WEIGHT;
    
    return v;
#endif
}

inline unsigned short int transformToPT(float val, float minBound = -1, float maxBound = 1) {
#ifdef CONTINUOUS_PARAM
    return 0;
#else
    if (val >= maxBound) {
        return MAXIMUM_WEIGHT;
    }
    else if (val <= minBound) {
        return 0;
    }

    float v = abs(round(MID_WEIGHT + (MID_WEIGHT * val) / maxBound));
// TODO!! For some reason to be explored, using HALF_WEIGHT in some environments
// helps to reach a solution faster (ex. MountainCar or FunctionApproximation)
//     float v = abs(round(HALF_WEIGHT + (HALF_WEIGHT * val) / maxBound));
    
    return static_cast<ParameterType>(v);
#endif
}

#endif /* Discretizer_h */
