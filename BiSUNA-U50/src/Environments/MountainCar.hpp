//
//  MountainCar.h
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

#ifndef MountainCar_hpp
#define MountainCar_hpp

#include "ReinforcementEnvironment.hpp"

class MountainCar : public ReinforcementEnvironment
{
	public:
		MountainCar(ushortT eID, const char *fileName);
		~MountainCar();
		
    // Global variables:
		float mcarPosition, mcarVelocity;
		int lastChange;
		bool normalizedObservation;
		bool normalizedAction;
	
		float mcarMinPosition;
		float mcarMaxPosition;
		float mcarMaxVelocity;            // the negative of this in the minimum velocity
		float mcarGoalPosition;
			
        uintT mcarTrialsToChange;
        float modMaxVel;
        bool changesMaxVelocity;
        bool isNoisyCar;
        bool isContinuous;
		bool originalValue;
			
		void start(int &numObsVars, int& numActionVars);
		float step(ParameterType *action);
		float restart();
		void print();
		bool set(int feature);
};

#endif
