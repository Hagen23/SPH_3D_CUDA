#pragma once
#ifndef _SPH_SM_M_KERNEL_H_
#define _SPH_SM_M_KERNEL_H_

#include <m3Vector.h>
#include <m3Bounds.h>
#include <m3Real.h>
#include <m3Matrix.h>
#include <m9Matrix.h>

#include <vector>
#include <map>
#include <chrono>

#include <helper_cuda.h>
#include <helper_math.h>

#define PI 3.141592f
#define INF 1E-12f

/// For density computation
__device__ m3Real Poly6(m3Real Poly6_constant, m3Real r2, m3Real kernel)
{
	m3Real pow_value = kernel * kernel - r2;
	// float constant = pow_value / fabs(pow_value);
	if(r2 >= 0 && r2 <= kernel*kernel)
		return Poly6_constant * pow_value * pow_value * pow_value;
	else 
		return 0.0f;
	// return Poly6_constant * pow_value * pow_value * pow_value; //pow_value * (constant + 1) *0.5;
}

/// For force of pressure computation
__device__ float Spiky(m3Real Spiky_constant, float r, m3Real kernel)
{
	if(r >= 0 && r <= kernel)
		return -Spiky_constant * (kernel - r) * (kernel - r) ;
	else
		return 0.0f;
}

/// For viscosity computation
__device__ float Visco(m3Real Spiky_constant, float r, m3Real kernel)
{
	if(r >= 0 && r <= kernel )
		return Spiky_constant * (kernel - r);
	else
		return 0;
}

__device__ m3Vector Calculate_Cell_Position(m3Vector pos, m3Real Cell_Size)
{
	m3Vector cellpos = pos / Cell_Size;
	cellpos.x = floor(cellpos.x);
	cellpos.y = floor(cellpos.y);
	cellpos.z = floor(cellpos.z);
	return cellpos;
}

__device__ int Calculate_Cell_Hash(m3Vector pos, m3Vector Grid_Size)
{
	// if((pos.x < 0)||(pos.x >= Grid_Size.x)||(pos.y < 0)||(pos.y >= Grid_Size.y)||
	// (pos.z < 0)||(pos.z >= Grid_Size.z))
	// 	return -1;

	int x = (int)pos.x & (int)((Grid_Size.x-1));  // wrap grid, assumes size is power of 2
    int y = (int)pos.y & (int)((Grid_Size.y-1));
    int z = (int)pos.z & (int)((Grid_Size.z-1));

	return  x + (int)Grid_Size.x * (y + (int)Grid_Size.y * z);
}

__device__ void Update_Properties(
	int 		index,
	m3Vector 	*pos_d,
	m3Vector 	*vel_d,
	m3Vector 	*acc_d,
	m3Vector 	World_Size, 
	m3Real 		Time_Delta, 
	int 		Number_Particles, 
	m3Real 		Wall_Hit)
{
	if(index < Number_Particles)
	{
		vel_d[index] = vel_d[index] + (acc_d[index]*Time_Delta);
		pos_d[index] = pos_d[index] + (vel_d[index]*Time_Delta);
		
		if(pos_d[index].x < 0.0f)
		{
			vel_d[index].x = vel_d[index].x * Wall_Hit;
			pos_d[index].x = 0.0f;
		}
		if(pos_d[index].x >= World_Size.x)
		{
			vel_d[index].x = vel_d[index].x * Wall_Hit;
			pos_d[index].x = World_Size.x - 0.0001f;
		}
		if(pos_d[index].y < 0.0f)
		{
			vel_d[index].y = vel_d[index].y * Wall_Hit;
			pos_d[index].y = 0.0f;
		}
		if(pos_d[index].y >= World_Size.y)
		{
			vel_d[index].y = vel_d[index].y * Wall_Hit;
			pos_d[index].y = World_Size.y - 0.0001f;
		}
		if(pos_d[index].z < 0.0f)
		{
			vel_d[index].z = vel_d[index].z * Wall_Hit;
			pos_d[index].z = 0.0f;
		}
		if(pos_d[index].z >= World_Size.z)
		{
			vel_d[index].z = vel_d[index].z * Wall_Hit;
			pos_d[index].z = World_Size.z - 0.0001f;
		}
	}
}

__global__
void calcHashD(uint *gridParticleHash, uint * gridParticleIndex, m3Vector *pos, m3Real Cell_Size, m3Vector Grid_Size, int numberParticles)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index >= numberParticles) return;

	m3Vector p = pos[index];
	int hash = Calculate_Cell_Hash(Calculate_Cell_Position(p, Cell_Size), Grid_Size);

	gridParticleHash[index] = hash;
	gridParticleIndex[index] = index;
}

__global__ 
void reorderDataAndFindCellStartD(
	m3Vector 	*sortedPos_d,
	m3Vector 	*pos_d,
	m3Vector 	*sortedVel_d, 
	m3Vector 	*vel_d,	
	uint 		*cellStart,
	uint 		*cellEnd, 
	uint 		*gridParticleHash, 
	uint 		*gridParticleIndex, 
	uint 		numParticles)
{
	extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    uint hash;

	// handle case when no. of particles not multiple of block size
    if (index < numParticles)
    {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x+1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[index-1];
        }
    }

	__syncthreads();

    if (index < numParticles)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = index;

            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1)
        {
            cellEnd[hash] = index + 1;
            // cellEnd[hash] = index;
        }

        // Now use the sorted index to reorder the pos and vel data
        uint sortedIndex = gridParticleIndex[index];
        
		sortedPos_d[index] = pos_d[sortedIndex];
		sortedVel_d[index] = vel_d[sortedIndex];		
		// sortedAcc_d[index] = acc_d[sortedIndex];
		// sortedMass_d[index] = mass_d[sortedIndex];		
		// sorted_dens_d[index] = dens_d[sortedIndex];
		// sorted_pres_d[index] = pres_d[sortedIndex];
	}
}

__global__ void Compute_Density_SingPressureD(
	m3Vector 	*sortedPos_d,
	m3Real 		*dens_d,
	m3Real 		*pres_d,
	m3Real 		*mass_d,
	uint 		*m_dGridParticleIndex, 
	uint 		*m_dCellStart, 
	uint 		*m_dCellEnd, 
	int 		m_numParticles, 
	int 		m_numGridCells, 
	m3Real		Cell_Size, 
	m3Vector 	Grid_Size, 
	m3Real 		Poly6_constant, 
	m3Real 		kernel,
	m3Real 		K, 
	m3Real 		Stand_Density)
{
	uint id = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;

	m3Vector CellPos, NeighborPos;
	int hash;

	for(int index = id; index< m_numParticles; index+= stride)
	{
		dens_d[index] = 0.f;
		pres_d[index] = 0.f;
		
		CellPos = Calculate_Cell_Position(sortedPos_d[index], Cell_Size);

		float partial_dens = 0.0f;

		for(int k = -1; k <= 1; k++)
		for(int j = -1; j <= 1; j++)
		for(int i = -1; i <= 1; i++)
		{
			NeighborPos = CellPos + m3Vector(i, j, k);
			hash = Calculate_Cell_Hash(NeighborPos, Grid_Size);
			
			uint startIndex = m_dCellStart[hash];
			uint endIndex = m_dCellEnd[hash];

			if (startIndex != 0xffffffff && startIndex != endIndex)
			{
				// uint endIndex = m_dCellEnd[hash];

				// printf("start_index %d end_index %d \n", index, startIndex, endIndex);

				for(uint j = startIndex; j < endIndex; j++)
				{
					if(j != index)
					{
						m3Vector Distance;
						Distance = sortedPos_d[index] - sortedPos_d[j];
						
						m3Real dis2 = Distance.magnitudeSquared();
						// Distance.x * Distance.x + Distance.y * Distance.y + Distance.z * Distance.z;
						// // printf("j %d sortedMass j %f\n", j, sortedMass_d[j]);
						partial_dens += mass_d[j] * Poly6(Poly6_constant, dis2, kernel);
					}
				}
			}
		}

		dens_d[index] = partial_dens;
		pres_d[index] = K * (partial_dens - Stand_Density);
	}
}

__global__ void Compute_ForceD(
	m3Vector 	*pos_d, 
	m3Vector 	*sortedPos_d,
	m3Vector 	*vel_d, 
	m3Vector 	*sortedVel_d,
	m3Vector 	*acc_d,
	m3Real 		*mass_d,
	m3Real 		*dens_d,
	m3Real 		*pres_d, 
	uint 		*m_dGridParticleIndex, 
	uint 		*m_dCellStart, 
	uint 		*m_dCellEnd, 
	int 		m_numParticles, 
	int 		m_numGridCells, 
	m3Real 		Cell_Size, 
	m3Vector 	Grid_Size, 
	m3Real 		Spiky_constant, 
	m3Real 		B_spline_constant, 
	m3Real 		Time_Delta, 
	m3Real 		kernel, 
	m3Vector 	Gravity, 
	m3Real 		mu, 
	m3Vector 	World_Size, 
	m3Real 		Wall_Hit)
{
	uint id = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;

	m3Vector CellPos;
	m3Vector NeighborPos;
	int hash;

	for(int index = id; index< m_numParticles; index+= stride)
	{
		// printf("sorted dens %d -- %f \n", index, sorted_pres_d[index]);
		acc_d[index] = m3Vector(0.0f, 0.0f, 0.0f);

		CellPos = Calculate_Cell_Position(sortedPos_d[index], Cell_Size);

		for(int k = -1; k <= 1; k++)
		for(int j = -1; j <= 1; j++)
		for(int i = -1; i <= 1; i++)
		{
			NeighborPos = CellPos + m3Vector(i, j, k);
			hash = Calculate_Cell_Hash(NeighborPos, Grid_Size);

			uint startIndex = m_dCellStart[hash];

			if (startIndex != 0xffffffff)
			{
				uint endIndex = m_dCellEnd[hash];

				for(uint j = startIndex; j < endIndex; j++)
				{
					if(j != index)
					{
						m3Vector Distance;
						Distance = sortedPos_d[index] - sortedPos_d[j];
						float dis2 = (float)Distance.magnitudeSquared();

						// if(dis2 > INF)
						// {
							float dis = sqrt(dis2);

							/// Calculates the force of pressure, Eq.10
							float Volume = mass_d[j] / dens_d[j];
							// float Force_pressure = Volume * (p->pres+np->pres)/2 * B_spline_1(dis);

							float Force_pressure = Volume * (pres_d[index] + pres_d[j])/2 * Spiky(Spiky_constant, dis, kernel);

							acc_d[index] -= Distance * Force_pressure / dis;

							m3Vector RelativeVel = sortedVel_d[j] - sortedVel_d[index];
							float Force_viscosity = Volume * mu * Visco(Spiky_constant, dis, kernel);
							acc_d[index] += RelativeVel * Force_viscosity;
						// }
					}
				}
			}
		}

		/// Sum of the forces that make up the fluid, Eq.8

		acc_d[index] = acc_d[index] / dens_d[index];

		acc_d[index] += Gravity;

		Update_Properties(index, sortedPos_d, sortedVel_d, acc_d, World_Size, Time_Delta, m_numParticles, Wall_Hit);

		uint originalIndex = m_dGridParticleIndex[index];

		pos_d[originalIndex] = sortedPos_d[index];
		vel_d[originalIndex] = sortedVel_d[index];
	}
}

/// Time integration as in 2016 - Fluid simulation by the SPH Method, a survey.
/// Eq.13 and Eq.14
__global__ void Update_PropertiesD(
	m3Vector *pos_d,
	m3Vector *vel_d,
	m3Vector *acc_d,
	m3Vector World_Size, m3Real Time_Delta, int Number_Particles, m3Real Wall_Hit)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(index < Number_Particles)
	{
		vel_d[index] = vel_d[index] + (acc_d[index]*Time_Delta);
		pos_d[index] = pos_d[index] + (vel_d[index]*Time_Delta);
		
		if(pos_d[index].x < 0.0f)
		{
			vel_d[index].x = vel_d[index].x * Wall_Hit;
			pos_d[index].x = 0.0f;
		}
		if(pos_d[index].x >= World_Size.x)
		{
			vel_d[index].x = vel_d[index].x * Wall_Hit;
			pos_d[index].x = World_Size.x - 0.0001f;
		}
		if(pos_d[index].y < 0.0f)
		{
			vel_d[index].y = vel_d[index].y * Wall_Hit;
			pos_d[index].y = 0.0f;
		}
		if(pos_d[index].y >= World_Size.y)
		{
			vel_d[index].y = vel_d[index].y * Wall_Hit;
			pos_d[index].y = World_Size.y - 0.0001f;
		}
		if(pos_d[index].z < 0.0f)
		{
			vel_d[index].z = vel_d[index].z * Wall_Hit;
			pos_d[index].z = 0.0f;
		}
		if(pos_d[index].z >= World_Size.z)
		{
			vel_d[index].z = vel_d[index].z * Wall_Hit;
			pos_d[index].z = World_Size.z - 0.0001f;
		}
	}
}
// }
#endif