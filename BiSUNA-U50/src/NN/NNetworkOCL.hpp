//
//  NNetworkOCL.hpp
//  BiSUNAOpenCL
//
//  Created by RH VT on 11/03/20.
//  Copyright © 2020 R. All rights reserved.
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

#ifndef NNetworkOCL_hpp
#define NNetworkOCL_hpp

#include "Configuration/PConfig.hpp"
#include "Environments/ReinforcementEnvironment.hpp"
#include "NN/NNetworkExtra.hpp"
#include "OCL/OCLContainer.hpp"

struct NNFunctionOCL
{
    static float executeOCL(vector<ReinforcementEnvironment *>env, const ushort &episodesPerAgent, OCLContainer *cont, bool profiling);
    static float mainOCLNetwork(PConfig *conf);
};

#endif /* NNetworkOCL_hpp */
