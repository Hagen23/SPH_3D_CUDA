#ifndef __SPH_cuda_CPP__
#define __SPH_cuda_CPP__

#include <SPH_cuda.h>
#include <SPH_cuda_kernels.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>

using namespace std;

SPH_cuda::SPH_cuda()
{
	kernel = 0.15f;

	Max_Number_Paticles = 50000;
	total_time_steps = 0;
	Number_Particles = 0;
	
	World_Size = m3Vector(1.f, 1.f, 1.f);

	Cell_Size = 0.5;
	Grid_Size = World_Size / Cell_Size;
	Grid_Size.x = (int)Grid_Size.x;
	Grid_Size.y = (int)Grid_Size.y;
	Grid_Size.z = (int)Grid_Size.z;

	Number_Cells = (int)Grid_Size.x * (int)Grid_Size.y * (int)Grid_Size.z;

	Gravity.set(0.0f, -9.8f, 0.0f);
	K = 1.5f;
	Stand_Density = 1000.0f;
	max_vel = m3Vector(3.0f, 3.0f, 3.0f);

	Poly6_constant = 315.0f/(64.0f * m3Pi * pow(kernel, 9));
	Spiky_constant = 45.0f/(m3Pi * pow(kernel, 6));

	/// Time step is calculated as in 2016 - Divergence-Free SPH for Incompressible and Viscous Fluids.
	/// Then we adapt the time step size according to the Courant-Friedrich-Levy (CFL) condition [6] ∆t ≤ 0.4 * d / (||vmax||)
	Time_Delta = 0.4 * kernel / sqrt(max_vel.magnitudeSquared());
	Wall_Hit = -0.05f;
	mu = 80.0f;

	cout<<"SPHSystem"<<endl;
	cout<<"Grid_Size_X : " << Grid_Size.x << endl;
	cout<<"Grid_Size_Y : " << Grid_Size.y << endl;
	cout<<"Grid_Size_Z : " << Grid_Size.y << endl;
	cout<<"Cell Number : "<<Number_Cells<<endl;
	cout<<"Time Delta : "<<Time_Delta<<endl;
}

SPH_cuda::~SPH_cuda()
{
	// delete[] particles;
}

void SPH_cuda::init_particles(std::vector<m3Vector> positions, float Stand_Density, int Number_Particles)
{
	/// Allocate host storagem
    pos = new m3Vector[Number_Particles]();
    vel = new m3Vector[Number_Particles]();
   
    acc = new m3Vector[Number_Particles]();
    mass = new float[Number_Particles]();

    dens = new float[Number_Particles]();
    pres = new float[Number_Particles]();

    for(int i = 0; i < Number_Particles; i++)
    {
        pos[i] = positions[i];
        dens[i] = Stand_Density;
        mass[i] = 0.2f;
    }

    unsigned int memSize = sizeof(m3Vector)*Number_Particles;

    checkCudaErrors(cudaMalloc((void**)&pos_d, memSize));
    checkCudaErrors(cudaMalloc((void**)&sortedPos_d, memSize));
    checkCudaErrors(cudaMalloc((void**)&vel_d, memSize));
    checkCudaErrors(cudaMalloc((void**)&sortedVel_d, memSize));
    
    checkCudaErrors(cudaMalloc((void**)&acc_d, memSize));
    checkCudaErrors(cudaMalloc((void**)&sortedAcc_d, memSize));

    checkCudaErrors(cudaMalloc((void**)&mass_d, sizeof(m3Real)*Number_Particles));
    checkCudaErrors(cudaMalloc((void**)&sortedMass_d, sizeof(m3Real)*Number_Particles));
    
    checkCudaErrors(cudaMalloc((void**)&dens_d, sizeof(m3Real)*Number_Particles));
    checkCudaErrors(cudaMalloc((void**)&sorted_dens_d, sizeof(m3Real)*Number_Particles));
    checkCudaErrors(cudaMalloc((void**)&pres_d, sizeof(m3Real)*Number_Particles));
    checkCudaErrors(cudaMalloc((void**)&sorted_pres_d, sizeof(m3Real)*Number_Particles));
	

	checkCudaErrors(cudaMemcpy(pos_d, pos, memSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(sortedPos_d, 0, memSize));
	checkCudaErrors(cudaMemcpy(vel_d, vel, memSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(sortedVel_d, 0, memSize));
	checkCudaErrors(cudaMemcpy(acc_d, acc, memSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(sortedAcc_d, 0, memSize));

    checkCudaErrors(cudaMemcpy(mass_d, mass, sizeof(m3Real) * Number_Particles, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(sortedMass_d, 0, sizeof(m3Real)*Number_Particles));
    checkCudaErrors(cudaMemcpy(dens_d, dens, sizeof(m3Real) * Number_Particles, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(sorted_dens_d, 0, sizeof(m3Real)*Number_Particles));
	checkCudaErrors(cudaMemcpy(pres_d, pres, sizeof(m3Real) * Number_Particles, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(sorted_pres_d, 0, sizeof(m3Real)*Number_Particles));

    cout << "Initialized particles" << endl;
}

void SPH_cuda::Init_Fluid()
{
	vector<m3Vector> positions;

	for(float k = World_Size.z * 0.3f; k < World_Size.z * 0.7f; k += kernel * 0.6f)
	for(float j = World_Size.y * 0.3f; j < World_Size.y * 0.7f; j += kernel * 0.6f)
	for(float i = World_Size.x * 0.3f; i < World_Size.x * 0.7f; i += kernel * 0.6f)
			positions.push_back(m3Vector(i, j, k));
	
	Number_Particles = positions.size();

	cout <<"Num particles: " <<Number_Particles<< endl;

	init_particles(positions, Stand_Density, Number_Particles);

	hGridParticleHash = new uint[Number_Particles]();
	hGridParticleIndex = new uint[Number_Particles]();
	hCellStart = new uint[Number_Cells]();
	hCellEnd = new uint[Number_Cells]();

	checkCudaErrors(cudaMalloc((void**)&dGridParticleHash, sizeof(uint)*Number_Particles));
	checkCudaErrors(cudaMalloc((void**)&dGridParticleIndex, sizeof(uint)*Number_Particles));
	checkCudaErrors(cudaMalloc((void**)&dCellStart, sizeof(uint)*Number_Cells));
	checkCudaErrors(cudaMalloc((void**)&dCellEnd, sizeof(uint)*Number_Cells));

	checkCudaErrors(cudaMemset(dGridParticleHash, 0, Number_Particles*sizeof(uint)));
	checkCudaErrors(cudaMemset(dGridParticleIndex, 0, Number_Particles*sizeof(uint)));
	
	checkCudaErrors(cudaMemset(dCellStart, 0, Number_Cells*sizeof(uint)));
	checkCudaErrors(cudaMemset(dCellEnd, 0, Number_Cells*sizeof(uint)));
}

/// Calculates the cell position for each particle
void SPH_cuda::calcHash()
{
	uint numThreads, numBlocks;
	computeGridSize(Number_Particles, 256, numBlocks, numThreads);

	calcHashD<<< numBlocks, numThreads >>>(dGridParticleHash, dGridParticleIndex, pos_d, Cell_Size, Grid_Size, Number_Particles);

	// check if kernel invocation generated an error
	cudaDeviceSynchronize();
	getLastCudaError("Kernel execution failed: calc hash");
}

/// Sorts the hashes and indices
void SPH_cuda::sortParticles()
{
	thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash), 
		thrust::device_ptr<uint>(dGridParticleHash + Number_Particles), 
		thrust::device_ptr<uint>(dGridParticleIndex));

	cudaDeviceSynchronize();
	getLastCudaError("Kernel execution failed: SortParticles");
}

/// Reorders the particles based on the hashes and indices. Also finds the start and end cells.
void SPH_cuda::reorderDataAndFindCellStart()
{
	uint numThreads, numBlocks;
	computeGridSize(Number_Particles, 256, numBlocks, numThreads);

	// set all cells to empty
	checkCudaErrors(cudaMemset(dCellStart, 0, Number_Cells*sizeof(uint)));

	uint smemSize = sizeof(uint)*(numThreads+1);
	reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(	
		sortedPos_d, pos_d,
		sortedVel_d, vel_d,
		sortedAcc_d, acc_d,
		sortedMass_d, mass_d,
		sorted_dens_d, dens_d,
		sorted_pres_d, pres_d,
		dCellStart, dCellEnd, dGridParticleHash, dGridParticleIndex, Number_Particles);

	cudaDeviceSynchronize();
	getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");
}

void SPH_cuda::Compute_Density_SingPressure()
{
	uint numThreads, numBlocks;
	computeGridSize(Number_Particles, 256, numBlocks, numThreads);

	Compute_Density_SingPressureD<<<numBlocks, numThreads>>>(	
	sortedPos_d,
	sorted_dens_d,
	sorted_pres_d,
	sortedMass_d,
	dGridParticleIndex, dCellStart, dCellEnd, Number_Particles, Number_Cells, Cell_Size, Grid_Size, Poly6_constant, kernel, K, Stand_Density);

	cudaDeviceSynchronize();
	getLastCudaError("Kernel execution failed: Compute_Density_SingPressureD");
}

void SPH_cuda::Compute_Force()
{
	uint numThreads, numBlocks;
	computeGridSize(Number_Particles, 256, numBlocks, numThreads);

	Compute_ForceD<<<numBlocks, numThreads>>>(	
	pos_d, sortedPos_d,
	vel_d, sortedVel_d,
	acc_d, sortedAcc_d,
	mass_d, sortedMass_d,
	dens_d, sorted_dens_d,
	pres_d, sorted_pres_d,
	dGridParticleIndex, dCellStart, dCellEnd, Number_Particles, Number_Cells, Cell_Size, Grid_Size, Spiky_constant, B_spline_constant, Time_Delta, kernel, Gravity, mu);

	cudaDeviceSynchronize();
	getLastCudaError("Kernel execution failed: Compute_ForceD");
}

void SPH_cuda::Update_Properties()
{
	uint numThreads, numBlocks;
	computeGridSize(Number_Particles, 256, numBlocks, numThreads);

	Update_PropertiesD<<<numBlocks, numThreads>>>(	
		pos_d,
		vel_d,
		acc_d,
		World_Size, Time_Delta, Number_Particles, Wall_Hit);

	cudaDeviceSynchronize();
	getLastCudaError("Kernel execution failed: Update_PropertiesD");
}

void SPH_cuda::compute_SPH_cuda()
{
	calcHash();

	sortParticles();

	reorderDataAndFindCellStart();

	checkCudaErrors(cudaMemcpy(hGridParticleHash, dGridParticleHash, sizeof(uint)*Number_Particles, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hGridParticleIndex, dGridParticleIndex, sizeof(uint)*Number_Particles, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hCellStart, dCellStart, sizeof(uint)*Number_Cells, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hCellEnd, dCellEnd, sizeof(uint)*Number_Cells, cudaMemcpyDeviceToHost));

	for(int i = 0; i<Number_Particles; i++)
	{
		cout << "index: " << i << " hgph " << hGridParticleHash[i] << " hgpi " << hGridParticleIndex[i] << endl;
	}

	for(int i = 0; i<Number_Cells; i++)
	{
		cout << "index: " << i << " hcs: " << hCellStart[i] << " hce: " << hCellEnd[i] << endl;
	}

	// Compute_Density_SingPressure();

	// Compute_Force();

	// Update_Properties();

	// checkCudaErrors(cudaMemcpy(pos, pos_d, sizeof(m3Vector) * Number_Particles, cudaMemcpyDeviceToHost));
	// checkCudaErrors(cudaMemcpy(vel, vel_d, sizeof(m3Vector) * Number_Particles, cudaMemcpyDeviceToHost));
}

void SPH_cuda::Animation()
{
	compute_SPH_cuda();
}

#endif