/*
This class implements the SPH [5] and SM [2] algorithms with the needed modifications to correct velocity as presented in [1]. 

@author Octavio Navarro
@version 1.0
*/
#pragma once
#ifndef __SPH_cuda_H__
#define __SPH_cuda_H__

#include <m3Vector.h>
#include <m3Bounds.h>
#include <m3Real.h>
#include <m3Matrix.h>
#include <m9Matrix.h>

#include <vector>
#include <map>
#include <chrono>

#include <helper_cuda.h>

#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#define PI 3.141592f
#define INF 1E-12f

#define GRID_SIZE	64

typedef std::chrono::system_clock::time_point 	tpoint;
typedef std::chrono::duration<double> 			duration_d;

class SPH_cuda
{
	public:
	
		/// Host data
		m3Vector 	*pos;			// Position of the particle
		m3Vector 	*vel;			// Velocity of the particle
		m3Vector 	*acc;			// Acceleration of the particle
		float		*mass;

		float 		*dens;			// density
		float 		*pres;			// pressure

		/// GPU Data
		m3Vector	*pos_d, *sortedPos_d;
		m3Vector 	*vel_d, *sortedVel_d;					// Velocity of the particle
		
		m3Vector 	*acc_d, *sortedAcc_d;					// Acceleration of the particle
		m3Real		*mass_d, *sortedMass_d;

		m3Real 		*dens_d, *sorted_dens_d;			// density
		m3Real 		*pres_d, *sorted_pres_d;			// pressure

		// host grid data for sorting method
        uint  *hGridParticleHash; 		// grid hash value for each particle
        uint  *hGridParticleIndex;		// particle index for each particle
        uint  *hCellStart;        		// index of start of each cell in sorted list
        uint  *hCellEnd;          		// index of end of cell

		// grid data for sorting method
        uint  *dGridParticleHash; 		// grid hash value for each particle
        uint  *dGridParticleIndex;		// particle index for each particle
        uint  *dCellStart;        		// index of start of each cell in sorted list
        uint  *dCellEnd;          		// index of end of cell

		/// Particle system parameters
		m3Real 		kernel;					// kernel or h in kernel function
		int 		Max_Number_Paticles;	// initial array for particles
		int 		Number_Particles;		// paticle number

		m3Vector 	Grid_Size;				// Size of a size of each grid voxel
		m3Vector 	World_Size;				// screen size
		m3Vector	max_vel;
		m3Real 		Cell_Size;				// Size of the divisions in the grid; used to determine the cell position for the has grid; kernel or h
		int 		Number_Cells;			// Number of cells in the hash grid

		m3Vector 	Gravity;
		m3Real 		K;						// ideal pressure formulation k; Stiffness of the fluid
											// The lower the value, the stiffer the fluid
		m3Real 		Stand_Density;			// ideal pressure formulation p0
		m3Real 		Time_Delta;			
		m3Real 		Wall_Hit;				// To manage collisions with the environment.
		m3Real 		mu;						// Viscosity.
		
		m3Real Poly6_constant;
		m3Real Spiky_constant;
		m3Real B_spline_constant;

		SPH_cuda();
		~SPH_cuda();

		void init_particles(std::vector<m3Vector> positions, float Stand_Density, int Number_Particles);

		int total_time_steps;

		void Init_Fluid();	// initialize fluid
		// void Init_Particle(m3Vector pos, m3Vector vel);		// initialize particle system
		
		/// Hashed the particles into a grid
		void calcHash();
		void sortParticles();
		void reorderDataAndFindCellStart();

		/// SPH Methods	
		void Compute_Density_SingPressure();
		void Compute_Force();
		void Update_Properties();					// Updates Position and velocity for SPH, voltage for monodomain

		void compute_SPH_cuda();
		void Animation();

		inline int Get_Particle_Number() { return Number_Particles; }
		inline m3Vector Get_World_Size() { return World_Size; }
		inline m3Real Get_stand_dens()	 { return Stand_Density;}		 

		uint iDivUp(uint a, uint b)
		{
			return (a % b != 0) ? (a / b + 1) : (a / b);
		}

		// compute grid and thread block size for a given number of elements
		void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
		{
			numThreads = min(blockSize, n);
			numBlocks = (n + numThreads - 1) / numThreads;
		}
};


#endif