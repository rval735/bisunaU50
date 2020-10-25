//
//  MountainCar.cpp
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

#include "MountainCar.hpp"
#include "Discretizer.h"
#include "RandomUtils.hpp"

MountainCar::MountainCar(ushortT eID, const char *fileName):
    ReinforcementEnvironment(eID, fileName)
{
	trial = -1;
	lastChange = -1;
	maxSteps = (uintT)ini.GetInteger("MountainCar", "MaxSteps", 1000);
	mcarMinPosition = ini.GetReal("MountainCar", "MinPosition", -1.2);
	mcarMaxPosition = ini.GetReal("MountainCar", "MaxPosition", 0.6);
	mcarMaxVelocity = ini.GetReal("MountainCar", "MaxVelocity", 0.07);
    mcarGoalPosition = ini.GetReal("MountainCar", "GoalPosition", 0.6);
    mcarTrialsToChange = (uintT) ini.GetInteger("MountainCar", "TrialsToChange", 1);
	modMaxVel = ini.GetReal("MountainCar", "ModifiedMaxVelocity", 0.04);
    changesMaxVelocity = ini.GetBoolean("MountainCar", "ModifiedMaxVelocity", false);
    isNoisyCar = ini.GetBoolean("MountainCar", "NoisyEnvironment", false);
    isContinuous = ini.GetBoolean("MountainCar", "ContinuousMountainCar", true);
	mcarVelocity = 0.0;
		
	originalValue = true;
     	
	normalizedObservation = false;
    normalizedAction = false;
    observation = NULL;
}

MountainCar::~MountainCar()
{
    if (observation != NULL) {
        free(observation);
        observation = NULL;
    }
}

void MountainCar::start(int &numObsVars, int &numActionVars)
{
    numObsVars = 2;
    this->observationVars = 2;
    
	observation = (ParameterType *)malloc(numObsVars * sizeof(ParameterType));

#ifdef CONTINUOUS_MountainCar
    numActionVars = 1;
#else
	numActionVars = 3;
#endif
    
	this->actionVars = numActionVars;

	// Initialize state of Car
	restart();
}

float MountainCar::step(ParameterType *action)
{
	// initial reward
	if (action == NULL) {
		return -1;
		//return 1/(x*x + theta*theta + 0.001);             
	}

	float a = 0;
    
    if (isContinuous) {
    #ifdef CONTINUOUS_PARAM
        float act = action[0];
    #else
        float act = transformFromPT(action[0], 0, 1);
    #endif
        
        if (normalizedAction) {
            //supposing that the action's range is [0,1]:
            a = (act - 0.5) * 2.0;
        }
        else {
            a = act;
        }
        
        if (a < -1.0) {
            a = -1.0;
        }
        
        if (a > 1.0) {
            a = 1.0;
        }

        // Take action a, update state of car
        mcarVelocity += a * 0.001 + cos(3.0 * mcarPosition) * (-0.0025);
    }
    else {
        //Discrete Action Mountain Cart
    #ifdef CONTINUOUS_PARAM
        float act0 = action[0];
        float act1 = action[1];
        float act2 = action[2];
    #else
        float act0 = transformFromPT(action[0], 0, 1);
        float act1 = transformFromPT(action[1], 0, 1);
        float act2 = transformFromPT(action[2], 0, 1);
    #endif
        
        //For discrete actions, the range of the action is of no concern (i.e., normalizedAction does not play a role)
        
        if (act0 > act1 && act0 > act2) {
            a = 0.0;
        }
        if (act1 > act0 && act1 > act2) {
            a = 1.0;
        }
        if (act2 > act0 && act2 > act1) {
            a = 2.0;
        }

        // Take action a, update state of car
        mcarVelocity += (a - 1.0) * 0.001 + cos(3.0 * mcarPosition) * (-0.0025);
    }
	

	//limit the car's velocity
	if (mcarVelocity > mcarMaxVelocity) {
		mcarVelocity = mcarMaxVelocity;
	}
	if (mcarVelocity < -mcarMaxVelocity) {
		mcarVelocity = -mcarMaxVelocity;
	}
	
	mcarPosition += mcarVelocity;
	
	//limit the car's position
	if (mcarPosition > mcarMaxPosition) mcarPosition = mcarMaxPosition;
	if (mcarPosition < mcarMinPosition) mcarPosition = mcarMinPosition;

	//stop the car, if it hits the min_position
	if (mcarPosition == mcarMinPosition && mcarVelocity < 0) mcarVelocity = 0;
    
    float obs[observationVars];

    if (isNoisyCar) {
        float gauss0 = RandomUtils::randomNormal(0, 0.06);
        float gauss1 = RandomUtils::randomNormal(0, 0.009);
    
        if (normalizedObservation) {
            //the maximum and minimum position differ, therefore I use the absolute biggest value which is the minimum
            //notice that the noise is ignored in the normalization, therefore, it may go beyond the expected range [-1,1]
            obs[0] = (mcarPosition + gauss0) / fabs(mcarMinPosition);
            obs[1] = (mcarVelocity + gauss1) / mcarMaxVelocity;
        }
        else {
            obs[0] = mcarPosition + gauss0;
            obs[1] = mcarVelocity + gauss1;
        }
    }
    else {
        if (normalizedObservation) {
            //the maximum and minimum position differ, therefore I use the absolute biggest value which is the minimum
            obs[0] = mcarPosition / fabs(mcarMinPosition);
            obs[1] = mcarVelocity / mcarMaxVelocity;
        }
        else {
            obs[0] = mcarPosition;
            obs[1] = mcarVelocity;
        }
    }

	// Is Car within goal region?
	if (mcarPosition >= mcarGoalPosition) {
		restart();
		return 0;
	}
    
#ifdef CONTINUOUS_PARAM
    observation[0] = obs[0];
    observation[1] = obs[1];
#else
    observation[0] = transformToPT(obs[0]);
    observation[1] = transformToPT(obs[1]);
#endif

	return -1;
}

void MountainCar::print()
{
	printf("position %f velocity %f\n",mcarPosition, mcarVelocity);
}

float MountainCar::restart()
{
	trial++;

    if (changesMaxVelocity) {
        if (trial % mcarTrialsToChange == 0 && trial != lastChange)
        {
            lastChange = trial;

            if (originalValue == true) {
                mcarMaxVelocity = modMaxVel;
                originalValue = false;
                //printf("changing to modified\n");
            }
            else {
                mcarMaxVelocity = ini.GetReal("MountainCar", "MaxVelocity", 0.07);
                originalValue = true;
                //printf("changing to original\n");
            }
        }
    }
	//printf("%d %f\n",trial, mcarMaxVelocity);

	// Initialize state of Car
	mcarPosition = -0.5;
	mcarVelocity = 0.0;
    
    float obs[observationVars];
	
    if (isNoisyCar) {
        float gauss0 = RandomUtils::randomNormal(0, 0.06);
        float gauss1 = RandomUtils::randomNormal(0, 0.009);
        
        if (normalizedObservation) {
            //the maximum and minimum position differ, therefore I use the absolute biggest value which is the minimum
            //notice that the noise is ignored in the normalization, therefore, it may go beyond the expected range [-1,1]
            obs[0] = (mcarPosition + gauss0) / fabs(mcarMinPosition);
            obs[1] = (mcarVelocity + gauss1) / mcarMaxVelocity;
        }
        else
        {
            obs[0] = mcarPosition + gauss0;
            obs[1] = mcarVelocity + gauss1;
        }
    }
    else {
        if (normalizedObservation) {
            //the maximum and minimum position differ, therefore I use the absolute biggest value which is the minimum
            obs[0] = mcarPosition / fabs(mcarMinPosition);
            obs[1] = mcarVelocity / mcarMaxVelocity;
        }
        else {
            obs[0] = mcarPosition;
            obs[1] = mcarVelocity;
            //observation[2] = mcarMaxVelocity;
        }
    }
    
#ifdef CONTINUOUS_PARAM
    observation[0] = obs[0];
    observation[1] = obs[1];
#else
    observation[0] = transformToPT(obs[0]);
    observation[1] = transformToPT(obs[1]);
#endif

	return -1;
}

bool MountainCar::set(int feature)
{
	switch(feature)
	{
		case NormalizedObservation: {
			normalizedObservation = true;
			return true;
		}
		break;
		case NormalizedAction: {
			normalizedAction = true;
			return true;
		}
		break;
		default: {
			return false;
		}
	}
}
